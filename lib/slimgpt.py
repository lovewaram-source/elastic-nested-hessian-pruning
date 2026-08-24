"""Paper-derived SlimGPT structured pruning primitives.

This module intentionally keeps the implementation independent from the
unstructured SparseGPT and elastic pruning paths.  It physically removes
complete attention heads and FFN intermediate channels.
"""

import math

import torch
import torch.nn as nn


def _index_tensor(indices, device):
    return torch.as_tensor(indices, dtype=torch.long, device=device).flatten()


def _complement(size, indices, device):
    mask = torch.ones(size, dtype=torch.bool, device=device)
    mask[indices] = False
    return torch.arange(size, device=device)[mask]


def _submatrix(matrix, rows, columns):
    return matrix[rows[:, None], columns[None, :]]


def obs_group_loss(weight, hessian_inverse, indices):
    """Return the exact quadratic OBS loss for removing column ``indices``."""
    indices = _index_tensor(indices, hessian_inverse.device)
    block = _submatrix(hessian_inverse, indices, indices)
    selected = weight.index_select(1, indices).float()
    solution = torch.linalg.solve(block.float(), selected.T).T
    return torch.sum(selected * solution)


def remove_obs_group(weight, hessian_inverse, indices):
    """Compensate a removed column group and return survivor coordinates."""
    indices = _index_tensor(indices, hessian_inverse.device)
    survivors = _complement(hessian_inverse.shape[0], indices, hessian_inverse.device)
    block = _submatrix(hessian_inverse, indices, indices).float()
    selected = weight.index_select(1, indices).float()
    coefficients = torch.linalg.solve(block, selected.T).T
    compensated = weight.float().clone()
    compensated.addmm_(
        coefficients,
        hessian_inverse.index_select(0, indices).float(),
        alpha=-1,
    )
    del coefficients, selected
    cross_sg = _submatrix(hessian_inverse, survivors, indices).float()
    cross_gs = _submatrix(hessian_inverse, indices, survivors).float()
    solved_cross = torch.linalg.solve(block, cross_gs)
    survivor_hinv = _submatrix(hessian_inverse, survivors, survivors).float()
    survivor_hinv.addmm_(cross_sg, solved_cross, alpha=-1)
    return (
        compensated.index_select(1, survivors).to(weight.dtype),
        survivor_hinv.to(hessian_inverse.dtype),
        survivors,
    )


def damped_hessian_inverse(hessian, percdamp=0.01):
    """Build a numerically stable inverse Hessian from calibration statistics."""
    hessian = hessian.float().clone()
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    damp = percdamp * torch.mean(torch.diag(hessian))
    hessian[diagonal, diagonal] += damp
    factor = torch.linalg.cholesky(hessian)
    return torch.cholesky_inverse(factor)


def _replace_linear_weight(linear, weight, *, output_sliced=False, kept=None):
    weight = weight.to(device=linear.weight.device, dtype=linear.weight.dtype).contiguous()
    linear.weight = nn.Parameter(weight, requires_grad=linear.weight.requires_grad)
    linear.out_features, linear.in_features = weight.shape
    if output_sliced and linear.bias is not None:
        kept = _index_tensor(kept, linear.bias.device)
        bias = linear.bias.detach().index_select(0, kept).clone()
        linear.bias = nn.Parameter(bias, requires_grad=linear.bias.requires_grad)


def _head_coordinates(heads, head_dim, device):
    heads = _index_tensor(heads, device)
    offsets = torch.arange(head_dim, device=device)
    return (heads[:, None] * head_dim + offsets[None, :]).reshape(-1)


def attention_head_counts(attention):
    query_heads = attention.q_proj.out_features // attention.head_dim
    key_value_heads = attention.k_proj.out_features // attention.head_dim
    return query_heads, key_value_heads


def slice_llama_attention(attention, kept_heads, compensated_o_weight=None):
    """Physically retain complete MHA heads and all coupled projection axes."""
    query_heads, key_value_heads = attention_head_counts(attention)
    if key_value_heads != query_heads:
        raise ValueError("SlimGPT baseline does not support grouped-query attention")
    kept_heads = _index_tensor(kept_heads, attention.q_proj.weight.device)
    if kept_heads.numel() == 0:
        raise ValueError("at least one attention head must survive")
    coordinates = _head_coordinates(kept_heads, attention.head_dim, kept_heads.device)
    for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
        new_weight = projection.weight.detach().index_select(0, coordinates).clone()
        _replace_linear_weight(projection, new_weight, output_sliced=True, kept=coordinates)
    if compensated_o_weight is None:
        new_o_weight = attention.o_proj.weight.detach().index_select(1, coordinates).clone()
    else:
        new_o_weight = compensated_o_weight
    _replace_linear_weight(attention.o_proj, new_o_weight)
    if hasattr(attention, "num_heads"):
        attention.num_heads = int(kept_heads.numel())
    if hasattr(attention, "num_key_value_heads"):
        attention.num_key_value_heads = int(kept_heads.numel())
    attention.num_key_value_groups = 1
    # Transformers <= 4.40 uses this module attribute only when reshaping the
    # concatenated head output before o_proj.  It is not the residual-stream
    # width and does not change q/k/v input features or o_proj output features.
    attention.hidden_size = int(kept_heads.numel()) * attention.head_dim


def slice_llama_mlp(mlp, kept_channels, compensated_down_weight=None):
    """Physically retain coupled gate/up rows and down-projection columns."""
    kept_channels = _index_tensor(kept_channels, mlp.gate_proj.weight.device)
    if kept_channels.numel() == 0:
        raise ValueError("at least one FFN channel must survive")
    for projection in (mlp.gate_proj, mlp.up_proj):
        new_weight = projection.weight.detach().index_select(0, kept_channels).clone()
        _replace_linear_weight(projection, new_weight, output_sliced=True, kept=kept_channels)
    if compensated_down_weight is None:
        new_down_weight = mlp.down_proj.weight.detach().index_select(1, kept_channels).clone()
    else:
        new_down_weight = compensated_down_weight
    _replace_linear_weight(mlp.down_proj, new_down_weight)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(kept_channels.numel())


def _profile_shapes(layer_count, profile):
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if profile == "uniform":
        return [1.0] * layer_count
    if profile != "log_increase":
        raise ValueError(f"unknown SlimGPT ratio profile: {profile}")
    if layer_count == 1:
        return [1.0]
    return [0.5 + math.log(index + 1) / math.log(layer_count) for index in range(layer_count)]


def layer_pruning_ratios(layer_count, target_ratio, profile="log_increase"):
    if not 0 <= target_ratio < 1:
        raise ValueError("target_ratio must be in [0, 1)")
    shapes = _profile_shapes(layer_count, profile)
    mean_shape = sum(shapes) / len(shapes)
    return [min(0.999, target_ratio * shape / mean_shape) for shape in shapes]


def _layer_removed_parameters(layer, ratio):
    attention = layer.self_attn
    mlp = layer.mlp
    num_heads, num_key_value_heads = attention_head_counts(attention)
    if num_key_value_heads != num_heads:
        raise ValueError("SlimGPT baseline does not support grouped-query attention")
    removed_heads = min(num_heads - 1, int(num_heads * ratio))
    head_dim = attention.head_dim
    head_cost = head_dim * (
        attention.q_proj.in_features + attention.k_proj.in_features
        + attention.v_proj.in_features + attention.o_proj.out_features
    )
    head_cost += sum(
        head_dim if projection.bias is not None else 0
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    channels = mlp.gate_proj.out_features
    removed_channels = min(channels - 1, int(channels * ratio))
    channel_cost = (
        mlp.gate_proj.in_features + mlp.up_proj.in_features + mlp.down_proj.out_features
    )
    channel_cost += int(mlp.gate_proj.bias is not None) + int(mlp.up_proj.bias is not None)
    return removed_heads * head_cost + removed_channels * channel_cost


def expected_parameter_reduction(model, ratios):
    layers = model.model.layers
    if len(ratios) != len(layers):
        raise ValueError("one pruning ratio is required per decoder layer")
    removed = sum(_layer_removed_parameters(layer, ratio) for layer, ratio in zip(layers, ratios))
    total = sum(parameter.numel() for parameter in model.parameters())
    return removed / total


def calibrate_structural_ratios(model, target_ratio, profile="log_increase"):
    if not 0 <= target_ratio < 1:
        raise ValueError("target_ratio must be in [0, 1)")
    shapes = _profile_shapes(len(model.model.layers), profile)

    def ratios(scale):
        return [min(0.999, scale * shape) for shape in shapes]

    maximum = expected_parameter_reduction(model, ratios(100.0))
    if target_ratio > maximum:
        raise ValueError(
            f"target parameter reduction {target_ratio:.4f} exceeds structural maximum {maximum:.4f}"
        )
    low, high = 0.0, 1.0
    while expected_parameter_reduction(model, ratios(high)) < target_ratio:
        high *= 2
    best = ratios(high)
    best_error = abs(expected_parameter_reduction(model, best) - target_ratio)
    for _ in range(50):
        middle = (low + high) / 2
        candidate = ratios(middle)
        reduction = expected_parameter_reduction(model, candidate)
        error = abs(reduction - target_ratio)
        if error < best_error:
            best, best_error = candidate, error
        if reduction < target_ratio:
            low = middle
        else:
            high = middle
    return best


def prune_attention_with_obs(attention, hessian_inverse, ratio):
    num_heads, num_key_value_heads = attention_head_counts(attention)
    if num_key_value_heads != num_heads:
        raise ValueError("SlimGPT baseline does not support grouped-query attention")
    current_heads = list(range(num_heads))
    weight = attention.o_proj.weight.detach().float().clone()
    hinv = hessian_inverse.float().clone()
    remove_count = min(num_heads - 1, int(num_heads * ratio))
    if remove_count:
        blocks = torch.stack([
            hinv[
                head * attention.head_dim:(head + 1) * attention.head_dim,
                head * attention.head_dim:(head + 1) * attention.head_dim,
            ]
            for head in range(num_heads)
        ])
        grouped_factors = torch.linalg.cholesky(blocks, upper=True)
        diagonals = torch.diagonal(grouped_factors, dim1=-2, dim2=-1).square()
        head_weights = weight.reshape(weight.shape[0], num_heads, attention.head_dim)
        losses = torch.sum(head_weights.square() / diagonals.unsqueeze(0), dim=(0, 2))
        removed_tensor = torch.topk(losses, remove_count, largest=False).indices.sort().values
        removed = removed_tensor.tolist()
        group = _head_coordinates(removed_tensor, attention.head_dim, hinv.device)
        weight, hinv, _ = remove_obs_group(weight, hinv, group)
        removed_set = set(removed)
        current_heads = [head for head in current_heads if head not in removed_set]
    else:
        removed = []
    kept = torch.tensor(current_heads, device=attention.q_proj.weight.device)
    slice_llama_attention(attention, kept, compensated_o_weight=weight)
    return {"removed_heads": removed, "kept_heads": current_heads}


def prune_mlp_with_obs(mlp, hessian_inverse, ratio, min_group_size=8):
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    original_channels = list(range(mlp.down_proj.in_features))
    current_channels = original_channels.copy()
    weight = mlp.down_proj.weight.detach().float().clone()
    hinv = hessian_inverse.float().clone()
    target = min(len(current_channels) - 1, int(len(current_channels) * ratio))
    removed = []
    while len(removed) < target:
        remaining = target - len(removed)
        batch = min(1024, remaining)
        if remaining > min_group_size:
            power = 2 ** int(math.floor(math.log2(remaining)))
            batch = min(batch, max(min_group_size, power))
        diagonal = torch.diag(hinv).clamp_min(torch.finfo(hinv.dtype).eps)
        losses = torch.sum(weight.float().square(), dim=0) / diagonal
        positions = torch.topk(losses, batch, largest=False).indices.sort().values
        removed.extend(current_channels[int(position)] for position in positions.tolist())
        weight, hinv, survivors = remove_obs_group(weight, hinv, positions)
        current_channels = [current_channels[int(position)] for position in survivors.tolist()]
    kept = torch.tensor(current_channels, device=mlp.gate_proj.weight.device)
    slice_llama_mlp(mlp, kept, compensated_down_weight=weight)
    return {
        "removed_count": len(removed),
        "removed_channels": removed,
        "kept_channels": current_channels,
    }
