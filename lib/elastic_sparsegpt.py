"""Seven-slice, reordered SparseGPT pruning.

Columns are ranked once per linear layer, then processed left-to-right in
that reordered coordinate system. The first slice contains 25% of columns and
the remaining six contain 12.5% each. Every slice is one complete SparseGPT
compensation block and contributes the next global checkpoint from 20% to 80%.
"""

from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT
from .nm import build_nm_prune_mask


MAX_SPARSITY = 0.8
ELASTIC_SPARSITY_STAGES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
ELASTIC_2OF4_SPARSITY_STAGES = (0.2, 0.3, 0.4, 0.5)
ELASTIC_2OF4_BLOCK_FRACTIONS = (0.4, 0.2, 0.2, 0.2)


def prefix_column_count(
    columns: int,
    sparsity: float,
    max_sparsity: float = MAX_SPARSITY,
) -> int:
    """Return the reordered-column prefix active at a target sparsity."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if not 0.0 <= sparsity <= max_sparsity:
        raise ValueError("sparsity must be in [0, max_sparsity]")
    return round(columns * sparsity / max_sparsity)


def prefix_stage_indices(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
) -> tuple[int, int]:
    """Return the newly opened interval in a fixed ordered column prefix."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if not 0.0 <= previous_sparsity < sparsity <= MAX_SPARSITY:
        raise ValueError(
            "stage sparsities must satisfy 0 <= previous < current <= max"
        )
    return round(previous_sparsity * columns), round(sparsity * columns)


def iter_stage_segments(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
    blocksize: int,
    block_fractions: tuple[float, ...] | None = None,
    max_sparsity: float = MAX_SPARSITY,
) -> list[tuple[int, int]]:
    """Return the fixed elastic slices opened between two checkpoints.

    The seven slices end at global sparsities 20%, 30%, ..., 80%. Because
    every slice itself is one SparseGPT compensation block, ``blocksize`` is
    retained only for API compatibility and never subdivides a slice.
    """
    if not 0.0 <= previous_sparsity < sparsity <= max_sparsity:
        raise ValueError("stage sparsities must satisfy 0 <= previous < current <= max")
    if blocksize < 1:
        raise ValueError("blocksize must be positive")

    checkpoints = (0.0,) + ELASTIC_SPARSITY_STAGES

    def checkpoint_index(value: float) -> int:
        for index, checkpoint in enumerate(checkpoints):
            if abs(value - checkpoint) < 1e-8:
                return index
        raise ValueError(
            "elastic stages must use 0.2, 0.3, ..., 0.8 "
            "(previous may be 0.0)"
        )

    previous_index = checkpoint_index(previous_sparsity)
    current_index = checkpoint_index(sparsity)
    if block_fractions is None:
        boundaries = [
            prefix_column_count(columns, checkpoint, max_sparsity)
            for checkpoint in checkpoints
        ]
    else:
        fractions = tuple(float(fraction) for fraction in block_fractions)
        if not fractions or any(fraction <= 0.0 for fraction in fractions):
            raise ValueError("block_fractions must contain positive values")
        if not math.isclose(
            sum(fractions),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("block_fractions must sum to 1.0")
        if current_index > len(fractions):
            raise ValueError(
                "block_fractions does not provide enough blocks for this stage"
            )
        boundaries = [0]
        cumulative = 0.0
        for fraction in fractions:
            cumulative += fraction
            boundaries.append(round(columns * cumulative))
        boundaries[-1] = columns
        if any(
            end <= start for start, end in zip(boundaries, boundaries[1:])
        ):
            raise ValueError(
                "block_fractions produce an empty block for this layer"
            )
    return [
        (boundaries[index - 1], boundaries[index])
        for index in range(previous_index + 1, current_index + 1)
    ]


def allocate_segment_budgets(
    rows: int,
    segments: list[tuple[int, int]],
    total_budget: int,
) -> list[int]:
    """Distribute an exact target count proportionally over active segments."""
    capacities = [rows * (end - start) for start, end in segments]
    total_capacity = sum(capacities)
    if not 0 <= total_budget <= total_capacity:
        raise ValueError("total_budget must fit inside the active prefix")

    budgets = []
    cumulative_capacity = 0
    allocated = 0
    for capacity in capacities:
        cumulative_capacity += capacity
        cumulative_budget = round(total_budget * cumulative_capacity / total_capacity)
        budget = cumulative_budget - allocated
        budgets.append(budget)
        allocated += budget
    return budgets


def iter_2of4_stage_segments(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
    prune_m: int = 4,
    block_fractions: tuple[float, ...] = ELASTIC_2OF4_BLOCK_FRACTIONS,
) -> list[tuple[int, int]]:
    """Return aligned four-column blocks opened by one 2:4 stage.

    The first reordered block occupies 40% of the input columns and the
    remaining three occupy 20% each. Since each active block is pruned at
    2:4 (50% locally), opening these blocks yields global 20/30/40/50%
    checkpoints. Boundaries are rounded to multiples of prune_m so an N:M
    group never straddles two elastic blocks.
    """
    if columns < prune_m or prune_m < 1:
        raise ValueError("columns must be at least prune_m and prune_m positive")
    if not 0.0 <= previous_sparsity < sparsity <= 0.5:
        raise ValueError(
            "2:4 elastic stages must satisfy 0 <= previous < current <= 0.5"
        )
    stages = (0.0,) + ELASTIC_2OF4_SPARSITY_STAGES

    def stage_index(value: float) -> int:
        for index, stage in enumerate(stages):
            if abs(value - stage) < 1e-8:
                return index
        raise ValueError("2:4 elastic stages must use 0.2, 0.3, 0.4, or 0.5")

    previous_index = stage_index(previous_sparsity)
    current_index = stage_index(sparsity)
    fractions = tuple(float(fraction) for fraction in block_fractions)
    if len(fractions) != 4 or any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("2:4 elastic pruning requires four positive block fractions")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("2:4 elastic block fractions must sum to 1.0")

    boundaries = [0]
    cumulative = 0.0
    for index, fraction in enumerate(fractions[:-1]):
        cumulative += fraction
        raw = round(columns * cumulative)
        lower = boundaries[-1] + prune_m
        remaining_blocks = len(fractions) - index - 1
        upper = columns - remaining_blocks * prune_m
        aligned = round(raw / prune_m) * prune_m
        boundaries.append(max(lower, min(upper, aligned)))
    boundaries.append(columns)
    if any(
        end <= start or (end - start) % prune_m
        for start, end in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("2:4 elastic block boundaries must contain complete N:M groups")

    return [
        (boundaries[index - 1], boundaries[index])
        for index in range(previous_index + 1, current_index + 1)
    ]


def iter_flat_2of4_stage_segments(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
    blocksize: int,
    prune_m: int = 4,
) -> list[tuple[int, int]]:
    """Split a checkpoint interval into small blocks without macroblocks.

    Checkpoints truncate the last block to a multiple of four; the next
    stage starts a fresh block using the already compensated suffix.
    """
    if prune_m != 4 or columns < 4 or columns % 4:
        raise ValueError("flat 2:4 requires a positive four-aligned column count")
    if blocksize < prune_m or blocksize % prune_m:
        raise ValueError("blocksize must be a positive multiple of prune_m")
    if not 0.0 <= previous_sparsity < sparsity <= 0.5:
        raise ValueError("2:4 elastic stages must satisfy 0 <= previous < current <= 0.5")
    start = (round(2.0 * previous_sparsity * columns) // prune_m) * prune_m
    end = (round(2.0 * sparsity * columns) // prune_m) * prune_m
    segments = []
    while start < end:
        stop = min(start + blocksize, end)
        stop = max(start + prune_m, (stop // prune_m) * prune_m)
        stop = min(stop, end)
        segments.append((start, stop))
        start = stop
    return segments


_JOINT_2OF4_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_JOINT_2OF4_PAIR_TENSOR_CACHE = {}


def _joint_obs_2of4_mask(
    weights: torch.Tensor,
    inverse_hessian_factor: torch.Tensor,
    prune_n: int = 2,
    prune_m: int = 4,
) -> torch.Tensor:
    """Choose each 2:4 mask by the joint OBS loss of its two pruned weights.

    inverse_hessian_factor is the local upper Cholesky factor U of the
    conditional inverse Hessian (U.T @ U). For every quartet, this evaluates
    all six pairs with 0.5 * w_S.T @ inv((U.T @ U)_SS) @ w_S.
    """
    if (prune_n, prune_m) != (2, 4):
        raise ValueError("joint OBS mask selection currently requires 2:4")
    if weights.ndim != 2 or weights.shape[1] % prune_m:
        raise ValueError("joint OBS mask selection requires complete four-column groups")
    if tuple(inverse_hessian_factor.shape) != (weights.shape[1], weights.shape[1]):
        raise ValueError("joint OBS factor shape must match the selected weight block")

    rows, columns = weights.shape
    group_count = columns // prune_m
    device = weights.device
    if group_count == 1:
        group_factor = inverse_hessian_factor.unsqueeze(0)
    else:
        offsets = torch.arange(prune_m, device=device)
        starts = torch.arange(group_count, device=device) * prune_m
        indices = starts[:, None] + offsets[None, :]
        group_factor = inverse_hessian_factor[
            indices[:, :, None], indices[:, None, :]
        ]
    conditional_inverse_hessian = group_factor.transpose(-1, -2) @ group_factor

    pair_indices = _JOINT_2OF4_PAIR_TENSOR_CACHE.get(device)
    if pair_indices is None:
        pair_indices = torch.tensor(
            _JOINT_2OF4_PAIRS, device=device, dtype=torch.long
        )
        _JOINT_2OF4_PAIR_TENSOR_CACHE[device] = pair_indices
    first, second = pair_indices.unbind(dim=1)
    a = conditional_inverse_hessian[:, first, first]
    b = conditional_inverse_hessian[:, second, second]
    c = conditional_inverse_hessian[:, first, second]
    determinant = (a * b - c.square()).clamp_min(torch.finfo(torch.float32).tiny)

    grouped_weights = weights.float().reshape(rows, group_count, prune_m)
    w_first = grouped_weights[:, :, first]
    w_second = grouped_weights[:, :, second]
    joint_loss = 0.5 * (
        b.unsqueeze(0) * w_first.square()
        - 2.0 * c.unsqueeze(0) * w_first * w_second
        + a.unsqueeze(0) * w_second.square()
    ) / determinant.unsqueeze(0)
    best_pair = pair_indices[joint_loss.argmin(dim=-1)]

    grouped_mask = torch.zeros_like(grouped_weights, dtype=torch.bool)
    grouped_mask.scatter_(2, best_pair[..., :1], True)
    grouped_mask.scatter_(2, best_pair[..., 1:], True)
    return grouped_mask.reshape_as(weights)


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return indices that undo a one-dimensional permutation."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device
    )
    return inverse


def _global_pruned_part_column_order(
    scores: torch.Tensor,
    predicted_global_sparsity: float | None,
) -> torch.Tensor:
    """Rank columns by scores covered by a global predicted mask."""
    if predicted_global_sparsity is None:
        raise ValueError(
            "pruned-part ordering requires predicted_global_sparsity"
        )
    if not 0.0 < predicted_global_sparsity <= 1.0:
        raise ValueError("predicted_global_sparsity must be in (0, 1]")
    flat_scores = scores.reshape(-1)
    pruned_count = round(predicted_global_sparsity * flat_scores.numel())
    if not 1 <= pruned_count <= flat_scores.numel():
        raise ValueError(
            "predicted_global_sparsity selects an invalid number of elements"
        )
    # The predicted mask is global over the layer, not independently per
    # column. Scatter-add its selected element sensitivities by column.
    pruned_indices = torch.topk(
        flat_scores,
        k=pruned_count,
        largest=False,
        sorted=False,
    ).indices
    column_indices = torch.remainder(pruned_indices, scores.shape[1])
    column_costs = torch.zeros(
        scores.shape[1],
        device=scores.device,
        dtype=scores.dtype,
    )
    column_costs.scatter_add_(
        0,
        column_indices,
        flat_scores[pruned_indices],
    )
    return torch.argsort(column_costs, stable=True)


def _half_interleave_order(column_order: torch.Tensor) -> torch.Tensor:
    """Interleave low- and high-sensitivity halves of a sorted order."""
    columns = int(column_order.numel())
    midpoint = (columns + 1) // 2
    low = column_order[:midpoint]
    high = column_order[midpoint:]
    result = torch.empty_like(column_order)
    result[0::2] = low
    if high.numel():
        result[1::2][: high.numel()] = high
    return result


def _wanda_group_order(
    scores: torch.Tensor,
    use_predicted_pruned_part: bool,
    within_group_order: str | None = None,
    prune_n: int = 2,
    prune_m: int = 4,
) -> torch.Tensor:
    """Order complete physical N:M groups by Wanda sensitivity.

    Groups are never split.  The optional pruned-part score predicts the local
    N:M mask first, then sums only the Wanda sensitivity of the elements
    predicted to be zero.  New ablations can order columns inside each group
    by their full-column Wanda score; ``None`` retains the historical order.
    """
    rows, columns = scores.shape
    if (prune_n, prune_m) != (2, 4):
        raise ValueError('Wanda group ordering currently requires 2:4')
    if columns % prune_m:
        raise ValueError('Wanda group ordering requires 4-aligned columns')
    if within_group_order not in {None, "asc", "desc"}:
        raise ValueError("within_group_order must be None, 'asc', or 'desc'")
    group_scores = []
    column_scores = scores.sum(dim=0)
    for start in range(0, columns, prune_m):
        group_scores_tensor = scores[:, start : start + prune_m]
        if use_predicted_pruned_part:
            predicted_mask = build_nm_prune_mask(
                group_scores_tensor,
                prune_n,
                prune_m,
            )
            group_score = (group_scores_tensor * predicted_mask).sum()
        else:
            group_score = group_scores_tensor.sum()
        group_scores.append(group_score)
    group_order = torch.argsort(torch.stack(group_scores), stable=True)
    ordered_groups = []
    for group_index in group_order:
        start = int(group_index) * prune_m
        columns_in_group = torch.arange(
            start,
            start + prune_m,
            device=scores.device,
        )
        if within_group_order is None:
            ordered_groups.append(columns_in_group)
        else:
            local_order = torch.argsort(
                column_scores[columns_in_group],
                descending=within_group_order == "desc",
                stable=True,
            )
            ordered_groups.append(columns_in_group[local_order])
    return torch.cat(
        ordered_groups
    )


def _wanda_group_stage_desc_order(
    scores: torch.Tensor,
    use_predicted_pruned_part: bool,
) -> torch.Tensor:
    """Rank whole 2:4 groups high first inside fixed 40/20/20/20 regions.

    Score either the full group or its predicted-pruned Wanda part. Groups
    cannot move to another checkpoint.
    """
    rows, columns = scores.shape
    if columns % 4:
        raise ValueError("stage-local Wanda group ordering requires 4-aligned columns")
    scored = (
        scores * build_nm_prune_mask(scores, 2, 4)
        if use_predicted_pruned_part
        else scores
    )
    group_scores = scored.reshape(rows, columns // 4, 4)
    group_scores = group_scores.sum(dim=(0, 2))

    ordered_groups = []
    previous = 0.0
    for stage in ELASTIC_2OF4_SPARSITY_STAGES:
        start, end = iter_2of4_stage_segments(
            columns,
            previous,
            stage,
            block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
        )[0]
        first_group, last_group = start // 4, end // 4
        local_order = torch.argsort(
            group_scores[first_group:last_group],
            descending=True,
            stable=True,
        )
        ordered_groups.append(local_order + first_group)
        previous = stage

    group_order = torch.cat(ordered_groups)
    return (
        group_order[:, None] * 4
        + torch.arange(4, device=scores.device)[None, :]
    ).reshape(-1)


def _wanda_group_stage_quartile_interleave_order(
    scores: torch.Tensor,
    block_fractions: tuple[float, ...] = ELASTIC_2OF4_BLOCK_FRACTIONS,
    first_sorted_stage: float = 0.2,
) -> torch.Tensor:
    """Reassign columns across 4-column groups within each fixed macroblock.

    Within each 40/20/20/20 macroblock, rank columns by full-column Wanda
    sensitivity from high to low. Split that ranking into four equal
    sensitivity quartiles, then form each 2:4 group by taking the same rank
    position from every quartile. This is a static P&P-style round-robin
    channel assignment: macroblock membership is unchanged, and every
    checkpoint reuses the same grouping.
    """
    _, columns = scores.shape
    if columns % 4:
        raise ValueError(
            "stage-local quartile interleave requires 4-aligned columns"
        )
    if not any(
        abs(first_sorted_stage - stage) < 1e-8
        for stage in ELASTIC_2OF4_SPARSITY_STAGES
    ):
        raise ValueError("first_sorted_stage must be one of the 2:4 checkpoints")
    column_scores = scores.sum(dim=0)
    ordered_blocks = []
    previous = 0.0
    for stage in ELASTIC_2OF4_SPARSITY_STAGES:
        segments = iter_2of4_stage_segments(
            columns,
            previous,
            stage,
            block_fractions=block_fractions,
        )
        if len(segments) != 1:
            raise ValueError(
                "quartile interleave requires one fixed macroblock per stage"
            )
        start, end = segments[0]
        group_count = (end - start) // 4
        if group_count < 1:
            raise ValueError("each macroblock must contain at least one 2:4 group")
        if stage + 1e-8 < first_sorted_stage:
            ordered_blocks.append(
                torch.arange(start, end, device=scores.device)
            )
            previous = stage
            continue
        ranked_columns = torch.argsort(
            column_scores[start:end],
            descending=True,
            stable=True,
        ) + start
        quartiles = torch.stack(
            [
                ranked_columns[index * group_count : (index + 1) * group_count]
                for index in range(4)
            ],
            dim=1,
        )
        ordered_blocks.append(quartiles.reshape(-1))
        previous = stage
    return torch.cat(ordered_blocks)


@torch.no_grad()
def reorder_next_2of4_stage_groups(
    layer: nn.Module,
    plan: "ElasticSparseGPTPlan",
    previous_sparsity: float,
    sparsity: float,
    block_fractions: tuple[float, ...] = ELASTIC_2OF4_BLOCK_FRACTIONS,
) -> "ElasticSparseGPTPlan":
    """Reorder only the next stage's complete groups using current weights."""
    if plan.activation_norms is None or plan.hessian_inverse is None:
        raise ValueError("dynamic 2:4 ordering requires retained calibration statistics")
    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    segments = iter_2of4_stage_segments(
        columns,
        previous_sparsity,
        sparsity,
        block_fractions=block_fractions,
    )
    if len(segments) != 1:
        raise ValueError("dynamic 2:4 ordering advances one checkpoint at a time")
    start, end = segments[0]
    device = weight.device
    old_permutation = plan.permutation.to(device)
    region_indices = old_permutation[start:end]
    activation_norms = plan.activation_norms.to(device)
    group_scores = (
        weight[:, region_indices].abs()
        * activation_norms[region_indices].reshape(1, -1)
    ).reshape(rows, (end - start) // 4, 4).sum(dim=(0, 2))
    group_order = torch.argsort(group_scores, descending=True, stable=True)
    local_positions = (
        group_order[:, None] * 4
        + torch.arange(4, device=device)[None, :]
    ).reshape(-1)
    new_permutation = old_permutation.clone()
    new_permutation[start:end] = region_indices[local_positions]
    inverse_hessian = plan.hessian_inverse.to(device=device, dtype=torch.float32)
    reordered_inverse = inverse_hessian[new_permutation][:, new_permutation]
    factor = torch.linalg.cholesky(reordered_inverse, upper=True)
    return ElasticSparseGPTPlan(
        permutation=new_permutation.cpu(),
        hessian_factor=factor.float().cpu(),
        dead_columns=plan.dead_columns,
        activation_norms=plan.activation_norms,
        hessian_inverse=plan.hessian_inverse,
        processed_columns=plan.processed_columns,
    )


def column_permutation(
    weight: torch.Tensor,
    hessian_factor: torch.Tensor | None = None,
    ordering: str = "sparsegpt",
    activation_norms: torch.Tensor | None = None,
    predicted_global_sparsity: float | None = None,
) -> torch.Tensor:
    """Rank columns by full-column sensitivity, or preserve their input order."""
    if ordering == "original":
        return torch.arange(weight.shape[1], device=weight.device)
    if ordering == "sparsegpt":
        if hessian_factor is None:
            raise ValueError("SparseGPT ordering requires a Hessian factor")
        denominator = torch.diag(hessian_factor).reshape(1, -1).square()
        scores = weight.float().square() / denominator
    elif ordering == "wanda":
        if activation_norms is None:
            raise ValueError("Wanda ordering requires calibration activation norms")
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return torch.argsort(scores.sum(dim=0), stable=True)
    elif ordering in {"wanda_group_stage_desc", "wanda_group_pruned_part_stage_desc"}:
        if activation_norms is None:
            raise ValueError(
                "stage-local Wanda group ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return _wanda_group_stage_desc_order(
            scores,
            use_predicted_pruned_part=ordering == "wanda_group_pruned_part_stage_desc",
        )
    elif ordering == "wanda_group_stage_quartile_interleave":
        if activation_norms is None:
            raise ValueError(
                "stage-local quartile interleave requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return _wanda_group_stage_quartile_interleave_order(scores)
    elif ordering == "wanda_group_stage_pnp_late_interleave":
        if activation_norms is None:
            raise ValueError(
                "late-stage P&P interleave requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return _wanda_group_stage_quartile_interleave_order(
            scores,
            first_sorted_stage=0.4,
        )
    elif ordering in {
        "wanda_group",
        "wanda_group_pruned_part",
        "wanda_group_pruned_part_col_asc",
        "wanda_group_pruned_part_col_desc",
    }:
        if activation_norms is None:
            raise ValueError(
                "Wanda group ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return _wanda_group_order(
            scores,
            use_predicted_pruned_part=ordering
            in {
                "wanda_group_pruned_part",
                "wanda_group_pruned_part_col_asc",
                "wanda_group_pruned_part_col_desc",
            },
            within_group_order={
                "wanda_group_pruned_part_col_asc": "asc",
                "wanda_group_pruned_part_col_desc": "desc",
            }.get(ordering),
        )
    elif ordering == "wanda_pruned_part":
        if activation_norms is None:
            raise ValueError(
                "Wanda pruned-part ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        return _global_pruned_part_column_order(
            scores,
            predicted_global_sparsity,
        )
    elif ordering == "wanda_pruned_part_interleave":
        if activation_norms is None:
            raise ValueError(
                "Wanda pruned-part ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        base_order = _global_pruned_part_column_order(
            scores,
            predicted_global_sparsity,
        )
        return _half_interleave_order(base_order)
    elif ordering == "wanda_top50_interleave":
        if activation_norms is None:
            raise ValueError(
                "Wanda top-half ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        top_count = max(1, (scores.shape[0] + 1) // 2)
        top_half = torch.topk(
            scores,
            k=top_count,
            dim=0,
            largest=True,
            sorted=False,
        ).values
        base_order = torch.argsort(top_half.sum(dim=0), stable=True)
        return _half_interleave_order(base_order)
    elif ordering == "wanda_bottom50_interleave":
        if activation_norms is None:
            raise ValueError(
                "Wanda bottom-half ordering requires calibration activation norms"
            )
        scores = weight.float().abs() * activation_norms.reshape(1, -1)
        bottom_count = max(1, (scores.shape[0] + 1) // 2)
        bottom_half = torch.topk(
            scores,
            k=bottom_count,
            dim=0,
            largest=False,
            sorted=False,
        ).values
        base_order = torch.argsort(bottom_half.sum(dim=0), stable=True)
        return _half_interleave_order(base_order)
    elif ordering == "sparsegpt_pruned_part":
        if hessian_factor is None:
            raise ValueError(
                "SparseGPT pruned-part ordering requires a Hessian factor"
            )
        denominator = torch.diag(hessian_factor).reshape(1, -1).square()
        scores = weight.float().square() / denominator
        return _global_pruned_part_column_order(
            scores,
            predicted_global_sparsity,
        )
    else:
        raise ValueError(
            "ordering must be 'original', 'sparsegpt', 'wanda', "
            "'wanda_pruned_part', 'wanda_pruned_part_interleave', "
            "'wanda_group', 'wanda_group_pruned_part', "
            "'wanda_group_stage_desc', 'wanda_group_pruned_part_stage_desc', "
            "'wanda_group_pruned_part_col_asc', "
            "'wanda_group_pruned_part_col_desc', "
            "'wanda_top50_interleave', 'wanda_bottom50_interleave', "
            "or 'sparsegpt_pruned_part'"
        )

    column_costs = scores.sum(dim=0)
    return torch.argsort(column_costs, stable=True)


def build_block_mask(
    scores: torch.Tensor,
    sparsity: float | None = None,
    budget: int | None = None,
) -> torch.Tensor:
    """Prune an exact stable prefix of the lowest scores in one segment."""
    if (sparsity is None) == (budget is None):
        raise ValueError("provide exactly one of sparsity or budget")
    flat_scores = scores.reshape(-1)
    if sparsity is not None:
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity must be in [0, 1]")
        budget = round(sparsity * flat_scores.numel())
    if not 0 <= budget <= flat_scores.numel():
        raise ValueError("budget must fit inside the block")

    order = torch.argsort(flat_scores, stable=True)
    flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    flat_mask[order[:budget]] = True
    return flat_mask.reshape_as(scores)


def first_block_local_sparsity(
    target_sparsity: float,
    first_block_fraction: float,
) -> float:
    """Return the local sparsity needed for one prefix block to meet a target."""
    if not 0.0 < first_block_fraction <= 1.0:
        raise ValueError("first_block_fraction must be in (0, 1]")
    if not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity must be in [0, 1]")
    if target_sparsity > first_block_fraction + 1e-8:
        raise ValueError(
            "first_block_fraction must be at least target_sparsity "
            "when one block carries the full target"
        )
    return target_sparsity / first_block_fraction


@dataclass(frozen=True)
class ElasticSparseGPTPlan:
    """Reusable second-order statistics for one reordered linear layer."""

    permutation: torch.Tensor
    hessian_factor: torch.Tensor | None
    dead_columns: torch.Tensor
    activation_norms: torch.Tensor | None = None
    hessian_inverse: torch.Tensor | None = None
    processed_columns: torch.Tensor | None = None

    def validate_for(self, columns: int) -> None:
        if self.permutation.shape != (columns,):
            raise ValueError("plan permutation does not match layer columns")
        if self.hessian_factor is not None and self.hessian_factor.shape != (columns, columns):
            raise ValueError("plan Hessian factor does not match layer columns")
        if self.dead_columns.shape != (columns,):
            raise ValueError("plan dead-column mask does not match layer columns")
        if self.activation_norms is not None and self.activation_norms.shape != (columns,):
            raise ValueError("plan activation norms do not match layer columns")
        if self.hessian_inverse is not None and self.hessian_inverse.shape != (columns, columns):
            raise ValueError("plan Hessian inverse does not match layer columns")
        if self.processed_columns is not None and self.processed_columns.shape != (columns,):
            raise ValueError("plan processed-column mask does not match layer columns")

    def cpu(self) -> "ElasticSparseGPTPlan":
        return ElasticSparseGPTPlan(
            permutation=self.permutation.detach().cpu(),
            hessian_factor=(
                None
                if self.hessian_factor is None
                else self.hessian_factor.detach().float().cpu()
            ),
            dead_columns=self.dead_columns.detach().cpu(),
            activation_norms=(
                None
                if self.activation_norms is None
                else self.activation_norms.detach().float().cpu()
            ),
            hessian_inverse=(
                None
                if self.hessian_inverse is None
                else self.hessian_inverse.detach().float().cpu()
            ),
            processed_columns=(
                None
                if self.processed_columns is None
                else self.processed_columns.detach().cpu()
            ),
        )


def _weight_matrix(layer: nn.Module) -> torch.Tensor:
    weight = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        weight = weight.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    return weight.float()


@torch.no_grad()
def apply_elastic_sparsegpt_stage(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsity: float,
    previous_sparsity: float = 0.0,
    blocksize: int = 128,
    block_fractions: tuple[float, ...] | None = None,
    compensate: bool = True,
    compensation_alpha: float = 1.0,
) -> None:
    """Apply only one new prefix interval using a previously prepared plan."""
    if not 0.0 <= compensation_alpha <= 1.0:
        raise ValueError("compensation_alpha must be in [0, 1]")
    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)

    segments = iter_stage_segments(
        columns,
        previous_sparsity,
        sparsity,
        blocksize,
        block_fractions=block_fractions,
    )
    target_budget = (
        round(sparsity * rows * columns)
        - round(previous_sparsity * rows * columns)
    )
    segment_budgets = allocate_segment_budgets(rows, segments, target_budget)

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    if plan.hessian_factor is None:
        raise ValueError("static elastic stage requires a Hessian factor")
    hessian_factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead_columns = plan.dead_columns.to(device)

    weight[:, dead_columns] = 0
    weight = weight[:, permutation]
    ordered_dead_columns = dead_columns[permutation]
    weight[:, ordered_dead_columns] = 0

    for (block_start, block_end), budget in zip(segments, segment_budgets):
        block_weight = weight[:, block_start:block_end].clone()
        quantized = torch.zeros_like(block_weight)
        errors = torch.zeros_like(block_weight)
        block_factor = hessian_factor[
            block_start:block_end,
            block_start:block_end,
        ]

        scores = block_weight.square() / torch.diag(block_factor).reshape(
            1, -1
        ).square()
        block_mask = build_block_mask(scores, budget=budget)

        for column in range(block_end - block_start):
            current_weight = block_weight[:, column]
            diagonal = block_factor[column, column]

            pruned_weight = current_weight.clone()
            pruned_weight[block_mask[:, column]] = 0
            quantized[:, column] = pruned_weight

            error = (current_weight - pruned_weight) / diagonal
            if compensate:
                error = compensation_alpha * error
                block_weight[:, column:] -= error.unsqueeze(1).matmul(
                    block_factor[column, column:].unsqueeze(0)
                )
            errors[:, column] = error

        weight[:, block_start:block_end] = quantized
        if compensate:
            weight[:, block_end:] -= errors.matmul(
                hessian_factor[block_start:block_end, block_end:]
            )

    # Suffix compensation must never resurrect activation-dead columns.
    # Re-apply the invariant after all blocks have propagated their errors.
    weight[:, ordered_dead_columns] = 0
    weight = weight[:, inverse_permutation]
    if device.type == "cuda":
        torch.cuda.synchronize()
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(
        layer.weight.data.dtype
    )


@torch.no_grad()
def apply_elastic_2of4_stage(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsity: float,
    previous_sparsity: float = 0.0,
    prune_n: int = 2,
    prune_m: int = 4,
    block_fractions: tuple[float, ...] = ELASTIC_2OF4_BLOCK_FRACTIONS,
    obs_blocksize: int | None = None,
    mask_blocksize: int | None = None,
    flat_blocks: bool = False,
    mask_timing: str = "block",
    mask_selector: str = "individual",
    compensate: bool = True,
    compensation_alpha: float = 1.0,
    return_mask: bool = False,
    precomputed_mask: torch.Tensor | None = None,
    group_observer=None,
) -> torch.Tensor | None:
    """Apply one or more nested 2:4 elastic blocks in ordered coordinates.

    The Hessian factor is prepared once for the dense layer. Each newly opened
    block gets a fresh local N:M mask, then receives SparseGPT's sequential
    OBS update. Its quantization error is propagated only into the still-dense
    suffix, so earlier 2:4 masks remain frozen and every checkpoint nests the
    previous one.
    """
    if (prune_n, prune_m) != (2, 4):
        raise ValueError("elastic 2:4 method requires prune_n=2 and prune_m=4")
    if not 0.0 <= compensation_alpha <= 1.0:
        raise ValueError("compensation_alpha must be in [0, 1]")
    if mask_timing not in {"block", "group"}:
        raise ValueError("mask_timing must be block or group")
    if mask_selector not in {"individual", "joint_obs"}:
        raise ValueError("mask_selector must be individual or joint_obs")
    freeze_block_mask = mask_timing == "block" and (
        flat_blocks or mask_blocksize is not None
    )
    if flat_blocks:
        if obs_blocksize is None:
            raise ValueError("flat 2:4 blocks require obs_blocksize")
        segments = iter_flat_2of4_stage_segments(
            plan.permutation.numel(),
            previous_sparsity,
            sparsity,
            blocksize=obs_blocksize,
            prune_m=prune_m,
        )
    else:
        segments = iter_2of4_stage_segments(
            plan.permutation.numel(),
            previous_sparsity,
            sparsity,
            prune_m=prune_m,
            block_fractions=block_fractions,
        )
    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    if plan.hessian_factor is None:
        raise ValueError("elastic 2:4 stage requires a Hessian factor")

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    hessian_factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead_columns = plan.dead_columns.to(device)

    fixed_ordered_mask = None
    if precomputed_mask is not None:
        fixed_mask = precomputed_mask.to(device=device, dtype=torch.bool)
        if isinstance(layer, transformers.Conv1D):
            fixed_mask = fixed_mask.t()
        if tuple(fixed_mask.shape) != tuple(weight.shape):
            raise ValueError(
                "precomputed 2:4 mask shape "
                f"{tuple(fixed_mask.shape)} does not match {tuple(weight.shape)}"
            )
        fixed_ordered_mask = fixed_mask[:, permutation]

    # The plan factor lives in the one-time reordered coordinate system.
    weight[:, dead_columns] = 0
    ordered_weight = weight[:, permutation]
    stage_ordered_mask = (
        torch.zeros_like(ordered_weight, dtype=torch.bool)
        if return_mask
        else None
    )
    ordered_dead_columns = dead_columns[permutation]
    ordered_weight[:, ordered_dead_columns] = 0

    if obs_blocksize is not None and (obs_blocksize < prune_m or obs_blocksize % prune_m):
        raise ValueError("obs_blocksize must be a positive multiple of prune_m")
    if mask_blocksize is not None and (
        mask_blocksize < prune_m or mask_blocksize % prune_m
    ):
        raise ValueError("mask_blocksize must be a positive multiple of prune_m")

    for segment_start, segment_end in segments:
        block_start = segment_start
        while block_start < segment_end:
            block_end = min(
                block_start + (obs_blocksize or (segment_end - segment_start)),
                segment_end,
            )
            block_weight = ordered_weight[:, block_start:block_end].clone()
            count = block_end - block_start
            block_factor = hessian_factor[block_start:block_end, block_start:block_end]
            diagonal = torch.diag(block_factor)
            quantized = torch.zeros_like(block_weight)
            errors = torch.zeros_like(block_weight)
            block_mask = (
                fixed_ordered_mask[:, block_start:block_end].clone()
                if fixed_ordered_mask is not None
                else torch.zeros_like(block_weight, dtype=torch.bool)
            )
            mask_until = 0
            mask_span = mask_blocksize or (obs_blocksize if flat_blocks else count)

            for column in range(count):
                if fixed_ordered_mask is not None:
                    pass
                elif freeze_block_mask and column >= mask_until:
                    mask_end = min(column + mask_span, count)
                    if mask_selector == "joint_obs":
                        block_mask[:, column:mask_end] = _joint_obs_2of4_mask(
                            block_weight[:, column:mask_end],
                            block_factor[column:mask_end, column:mask_end],
                            prune_n,
                            prune_m,
                        )
                    else:
                        for mask_column in range(column, mask_end, prune_m):
                            group_end = min(mask_column + prune_m, count)
                            group_scores = (
                                block_weight[:, mask_column:group_end].square()
                                / diagonal[mask_column:group_end].reshape(1, -1).square()
                            )
                            block_mask[:, mask_column:group_end] = build_nm_prune_mask(
                                group_scores,
                                prune_n,
                                prune_m,
                            )
                    mask_until = mask_end
                elif not freeze_block_mask and column % prune_m == 0:
                    group_end = column + prune_m
                    if mask_selector == "joint_obs":
                        block_mask[:, column:group_end] = _joint_obs_2of4_mask(
                            block_weight[:, column:group_end],
                            block_factor[column:group_end, column:group_end],
                            prune_n,
                            prune_m,
                        )
                    else:
                        group_scores = (
                            block_weight[:, column:group_end].square()
                            / diagonal[column:group_end].reshape(1, -1).square()
                        )
                        block_mask[:, column:group_end] = build_nm_prune_mask(
                            group_scores,
                            prune_n,
                            prune_m,
                        )

                if group_observer is not None and column % prune_m == 0:
                    # Borrowed, read-only views. The observer must not mutate
                    # the trajectory. External suffix updates are still lazy:
                    # errors[:, :column] have not yet reached block_end:.
                    group_observer(
                        ordered_weight=ordered_weight, block_weight=block_weight,
                        quantized=quantized, errors=errors, block_mask=block_mask,
                        hessian_factor=hessian_factor, permutation=permutation,
                        dead_columns=ordered_dead_columns, column=column,
                        block_start=block_start, block_end=block_end,
                        segment_start=segment_start, segment_end=segment_end,
                        group_end=min(column + prune_m, count),
                    )

                current_weight = block_weight[:, column]
                current_diagonal = diagonal[column]
                pruned_weight = current_weight.clone()
                pruned_weight[block_mask[:, column]] = 0
                quantized[:, column] = pruned_weight

                if compensate and compensation_alpha != 0.0:
                    error = (current_weight - pruned_weight) / current_diagonal
                    error = compensation_alpha * error
                    block_weight[:, column:] -= error.unsqueeze(1).matmul(
                        block_factor[column, column:].unsqueeze(0)
                    )
                    errors[:, column] = error

            ordered_weight[:, block_start:block_end] = quantized
            if stage_ordered_mask is not None:
                stage_ordered_mask[:, block_start:block_end] = block_mask
            if compensate and compensation_alpha != 0.0 and block_end < columns:
                ordered_weight[:, block_end:] -= errors.matmul(
                    hessian_factor[block_start:block_end, block_end:]
                )
            block_start = block_end

    # Suffix compensation must never resurrect activation-dead columns.
    ordered_weight[:, ordered_dead_columns] = 0
    weight = ordered_weight[:, inverse_permutation]
    if device.type == "cuda":
        torch.cuda.synchronize()
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(
        layer.weight.data.dtype
    )
    if stage_ordered_mask is None:
        return None
    stage_mask = stage_ordered_mask[:, inverse_permutation]
    if isinstance(layer, transformers.Conv1D):
        stage_mask = stage_mask.t()
    return stage_mask.reshape(layer.weight.shape)

@torch.no_grad()
def apply_wanda_first_block_sparsegpt_stage(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsity: float,
    first_block_fraction: float = 0.25,
) -> None:
    """Prune one Wanda-ordered prefix block to the complete global target."""
    if plan.hessian_factor is None:
        raise ValueError("first-block SparseGPT stage requires a Hessian factor")

    local_sparsity = first_block_local_sparsity(sparsity, first_block_fraction)
    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    # Ceil avoids losing capacity to rounding when fraction == target sparsity.
    block_end = math.ceil(columns * first_block_fraction)
    if block_end < 1 or block_end > columns:
        raise ValueError("first block must contain at least one column")

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    hessian_factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead_columns = plan.dead_columns.to(device)
    ordered_dead_columns = dead_columns[permutation]

    weight[:, dead_columns] = 0
    ordered_weight = weight[:, permutation]
    block_weight = ordered_weight[:, :block_end].clone()
    block_factor = hessian_factor[:block_end, :block_end]
    budget = round(sparsity * rows * columns)
    if budget > rows * block_end:
        raise ValueError(
            "first block is too small to carry the requested global sparsity"
        )

    scores = block_weight.square() / torch.diag(block_factor).reshape(
        1, -1
    ).square()
    block_mask = build_block_mask(scores, budget=budget)
    quantized = torch.zeros_like(block_weight)
    errors = torch.zeros_like(block_weight)

    for column in range(block_end):
        current_weight = block_weight[:, column]
        diagonal = block_factor[column, column]
        pruned_weight = current_weight.clone()
        pruned_weight[block_mask[:, column]] = 0
        quantized[:, column] = pruned_weight

        error = (current_weight - pruned_weight) / diagonal
        block_weight[:, column:] -= error.unsqueeze(1).matmul(
            block_factor[column, column:].unsqueeze(0)
        )
        errors[:, column] = error

    ordered_weight[:, :block_end] = quantized
    if block_end < columns:
        ordered_weight[:, block_end:] -= errors.matmul(
            hessian_factor[:block_end, block_end:]
        )
    ordered_weight[:, ordered_dead_columns] = 0

    weight = ordered_weight[:, inverse_permutation]
    if device.type == "cuda":
        torch.cuda.synchronize()
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(
        layer.weight.data.dtype
    )


@torch.no_grad()
def apply_wanda_prefix_obs_stage(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsity: float,
    previous_sparsity: float = 0.0,
) -> None:
    """Prune the next fixed Wanda-ordered prefix one column at a time."""
    if plan.hessian_factor is None:
        raise ValueError("Wanda prefix OBS stage requires a Hessian factor")

    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    del rows
    start, end = prefix_stage_indices(columns, previous_sparsity, sparsity)
    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    dead_columns = plan.dead_columns.to(device)

    weight[:, dead_columns] = 0
    ordered_weight = weight[:, permutation]
    hessian_factor = plan.hessian_factor.to(
        device=device,
        dtype=torch.float32,
    )

    for column in range(start, end):
        current_weight = ordered_weight[:, column].clone()
        diagonal = hessian_factor[column, column]
        error = current_weight / diagonal
        ordered_weight[:, column:] -= error.unsqueeze(1).matmul(
            hessian_factor[column, column:].unsqueeze(0)
        )
        ordered_weight[:, column] = 0

    weight = ordered_weight[:, inverse_permutation]
    if device.type == "cuda":
        torch.cuda.synchronize()
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(layer.weight.data.dtype)


@torch.no_grad()
def apply_dynamic_wanda_sparsegpt_stage(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsity: float,
    previous_sparsity: float = 0.0,
) -> ElasticSparseGPTPlan:
    """Process one new block after Wanda-reordering only the unprocessed tail."""
    if not 0.0 <= previous_sparsity < sparsity <= MAX_SPARSITY:
        raise ValueError("stage sparsities must satisfy 0 <= previous < current <= max")
    if plan.activation_norms is None or plan.hessian_inverse is None:
        raise ValueError("dynamic Wanda stage requires retained plan statistics")

    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    device = weight.device
    dead_columns = plan.dead_columns.to(device)
    processed_columns = (
        torch.zeros(columns, dtype=torch.bool, device=device)
        if plan.processed_columns is None
        else plan.processed_columns.to(device)
    )
    weight[:, dead_columns] = 0
    segments = iter_stage_segments(
        columns,
        previous_sparsity,
        sparsity,
        blocksize=1,
    )
    if len(segments) != 1:
        raise ValueError("dynamic Wanda stages must advance one checkpoint at a time")
    block_start, block_end = segments[0]
    block_columns = block_end - block_start
    unprocessed_columns = ~processed_columns & ~dead_columns
    if int(unprocessed_columns.sum()) < block_columns:
        raise ValueError("not enough unprocessed columns for the next dynamic block")

    active_indices = torch.where(unprocessed_columns)[0]
    activation_norms = plan.activation_norms.to(device)[active_indices]
    local_permutation = column_permutation(
        weight[:, active_indices],
        ordering="wanda",
        activation_norms=activation_norms,
    )
    ordered_indices = active_indices[local_permutation]
    current_indices = ordered_indices[:block_columns]
    tail_indices = ordered_indices[block_columns:]
    ordered_weight = weight[:, ordered_indices].clone()
    hessian_inverse = plan.hessian_inverse.to(device=device, dtype=torch.float32)
    factor = torch.linalg.cholesky(
        hessian_inverse[ordered_indices][:, ordered_indices],
        upper=True,
    )
    block_weight = ordered_weight[:, :block_columns].clone()
    block_factor = factor[:block_columns, :block_columns]
    target_budget = round(sparsity * rows * columns) - round(
        previous_sparsity * rows * columns
    )
    block_mask = build_block_mask(
        block_weight.square()
        / torch.diag(block_factor).reshape(1, -1).square(),
        budget=target_budget,
    )
    quantized = torch.zeros_like(block_weight)
    errors = torch.zeros_like(block_weight)
    for column in range(block_columns):
        current_weight = block_weight[:, column]
        diagonal = block_factor[column, column]
        pruned_weight = current_weight.clone()
        pruned_weight[block_mask[:, column]] = 0
        quantized[:, column] = pruned_weight
        error = (current_weight - pruned_weight) / diagonal
        block_weight[:, column:] -= error.unsqueeze(1).matmul(
            block_factor[column, column:].unsqueeze(0)
        )
        errors[:, column] = error

    weight[:, current_indices] = quantized
    if tail_indices.numel():
        weight[:, tail_indices] -= errors.matmul(factor[:block_columns, block_columns:])
    processed_columns[current_indices] = True
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(layer.weight.data.dtype)
    return ElasticSparseGPTPlan(
        permutation=plan.permutation,
        hessian_factor=plan.hessian_factor,
        dead_columns=plan.dead_columns,
        activation_norms=plan.activation_norms,
        hessian_inverse=plan.hessian_inverse,
        processed_columns=processed_columns,
    ).cpu()


class ElasticSparseGPT(SparseGPT):
    """SparseGPT with full-layer reordering and staged prefix pruning."""

    def prepare_plan(
        self,
        percdamp: float = 0.01,
        permutation: torch.Tensor | None = None,
        ordering: str = "sparsegpt",
        timings: dict[str, float] | None = None,
        retain_dynamic_wanda_statistics: bool = False,
        predicted_global_sparsity: float | None = None,
        ordering_activation_norms: torch.Tensor | None = None,
    ) -> ElasticSparseGPTPlan:
        """Consume the collected Hessian and return a reusable CPU plan.

        ``ordering_activation_norms`` optionally decouples the statistics used
        for the one-time column ordering from the Hessian used by OBS. This is
        needed by fused elastic calibration: the Hessian is built from the
        equal-weight mixture of all nested branches, while the historical
        ordering remains the dense-branch Wanda ordering.
        """
        def start_timing():
            if timings is not None and self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
            return time.perf_counter()

        def finish_timing(name: str, start: float) -> None:
            if timings is None:
                return
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
            timings[name] = timings.get(name, 0.0) + (
                time.perf_counter() - start
            )

        weight = _weight_matrix(self.layer)
        hessian = self.H
        self.H = None
        if hessian is None:
            raise ValueError("Hessian has already been consumed")

        activation_norms = torch.sqrt(torch.diag(hessian).clamp_min(0))
        if ordering_activation_norms is not None:
            if ordering_activation_norms.ndim != 1:
                raise ValueError("ordering activation norms must be a 1-D tensor")
            if ordering_activation_norms.numel() != self.columns:
                raise ValueError(
                    "ordering activation norms must match the layer input width"
                )
            ordering_norms = ordering_activation_norms.to(
                device=self.dev,
                dtype=activation_norms.dtype,
            )
        else:
            ordering_norms = activation_norms
        dead_columns = torch.diag(hessian) == 0
        hessian[dead_columns, dead_columns] = 1
        weight[:, dead_columns] = 0

        damp = percdamp * torch.mean(torch.diag(hessian))
        diagonal_indices = torch.arange(self.columns, device=self.dev)
        hessian[diagonal_indices, diagonal_indices] += damp

        factorization_start = start_timing()
        hessian_inverse = torch.cholesky_inverse(
            torch.linalg.cholesky(hessian)
        )
        original_factor = None
        if permutation is None and ordering in {"original", "sparsegpt", "sparsegpt_pruned_part"}:
            original_factor = torch.linalg.cholesky(
                hessian_inverse,
                upper=True,
            )
        finish_timing("hessian_factorization_seconds", factorization_start)
        hessian_factor = None
        if retain_dynamic_wanda_statistics:
            permutation = torch.arange(self.columns, device=self.dev)
            if timings is not None:
                timings.setdefault("column_ordering_seconds", 0.0)
                timings.setdefault("reordered_factor_seconds", 0.0)
        elif permutation is None:
            ordering_start = start_timing()
            permutation = column_permutation(
                weight,
                hessian_factor=original_factor,
                ordering=ordering,
                activation_norms=ordering_norms,
                predicted_global_sparsity=predicted_global_sparsity,
            )
            finish_timing("column_ordering_seconds", ordering_start)
        else:
            permutation = permutation.to(self.dev)
            if timings is not None:
                timings.setdefault("column_ordering_seconds", 0.0)

        if not retain_dynamic_wanda_statistics:
            if ordering == "original" and original_factor is not None:
                hessian_factor = original_factor
                if timings is not None:
                    timings.setdefault("reordered_factor_seconds", 0.0)
            else:
                reordered_factor_start = start_timing()
                permuted_inverse = hessian_inverse[permutation][:, permutation]
                hessian_factor = torch.linalg.cholesky(
                    permuted_inverse,
                    upper=True,
                )
                finish_timing("reordered_factor_seconds", reordered_factor_start)
        plan = ElasticSparseGPTPlan(
            permutation=permutation,
            hessian_factor=hessian_factor,
            dead_columns=dead_columns,
            activation_norms=(activation_norms if retain_dynamic_wanda_statistics else None),
            hessian_inverse=(
                hessian_inverse
                if retain_dynamic_wanda_statistics
                else None
            ),
            processed_columns=(
                torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
                if retain_dynamic_wanda_statistics
                else None
            ),
        ).cpu()
        if retain_dynamic_wanda_statistics and timings is not None:
            timings.setdefault("reordered_factor_seconds", 0.0)
        return plan

    def fasterprune(
        self,
        sparsity: float,
        previous_sparsity: float = 0.0,
        permutation: torch.Tensor | None = None,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> torch.Tensor:
        """Compatibility wrapper that prepares once and applies one stage."""
        plan = self.prepare_plan(
            percdamp=percdamp,
            permutation=permutation,
        )
        apply_elastic_sparsegpt_stage(
            self.layer,
            plan,
            sparsity,
            previous_sparsity=previous_sparsity,
            blocksize=blocksize,
        )
        return plan.permutation
@torch.no_grad()
def apply_elastic_2of4_stages_once(
    layer: nn.Module,
    plan: ElasticSparseGPTPlan,
    sparsities: tuple[float, ...] = ELASTIC_2OF4_SPARSITY_STAGES,
    previous_sparsity: float = 0.0,
    prune_n: int = 2,
    prune_m: int = 4,
    block_fractions: tuple[float, ...] = ELASTIC_2OF4_BLOCK_FRACTIONS,
    obs_blocksize: int | None = None,
    mask_blocksize: int | None = None,
    flat_blocks: bool = False,
    mask_timing: str = "block",
    mask_selector: str = "individual",
    compensate: bool = True,
    compensation_alpha: float = 1.0,
) -> dict[float, torch.Tensor]:
    """Run one ordered OBS scan and snapshot every requested 2:4 checkpoint.

    Unlike calling :func:`apply_elastic_2of4_stage` once per checkpoint, this
    function applies the layer permutation, dead-column handling, and Hessian
    factor transfer once. Newly opened groups are still selected only after
    the preceding groups have received their OBS compensation, so the
    returned checkpoints follow the same nested trajectory exactly.
    """
    if (prune_n, prune_m) != (2, 4):
        raise ValueError("elastic 2:4 method requires prune_n=2 and prune_m=4")
    if not 0.0 <= compensation_alpha <= 1.0:
        raise ValueError("compensation_alpha must be in [0, 1]")
    if mask_timing not in {"block", "group"}:
        raise ValueError("mask_timing must be block or group")
    if mask_selector not in {"individual", "joint_obs"}:
        raise ValueError("mask_selector must be individual or joint_obs")
    stages = tuple(float(stage) for stage in sparsities)
    if not stages or any(stage <= previous_sparsity for stage in stages):
        raise ValueError("sparsities must be strictly above previous_sparsity")
    if any(current <= previous for previous, current in zip(stages, stages[1:])):
        raise ValueError("sparsities must be strictly increasing")

    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)
    if plan.hessian_factor is None:
        raise ValueError("elastic 2:4 stage requires a Hessian factor")

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    hessian_factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead_columns = plan.dead_columns.to(device)
    weight[:, dead_columns] = 0
    ordered_weight = weight[:, permutation]
    ordered_dead_columns = dead_columns[permutation]
    ordered_weight[:, ordered_dead_columns] = 0

    if flat_blocks:
        if obs_blocksize is None:
            raise ValueError("flat 2:4 blocks require obs_blocksize")
        if obs_blocksize < prune_m or obs_blocksize % prune_m:
            raise ValueError("obs_blocksize must be a positive multiple of prune_m")
    elif obs_blocksize is not None and (
        obs_blocksize < prune_m or obs_blocksize % prune_m
    ):
        raise ValueError("obs_blocksize must be a positive multiple of prune_m")

    freeze_block_mask = mask_timing == "block" and (
        flat_blocks or mask_blocksize is not None
    )
    if mask_blocksize is not None and (
        mask_blocksize < prune_m or mask_blocksize % prune_m
    ):
        raise ValueError("mask_blocksize must be a positive multiple of prune_m")
    checkpoints = {}
    previous = previous_sparsity

    for stage in stages:
        if flat_blocks:
            segments = iter_flat_2of4_stage_segments(
                columns,
                previous,
                stage,
                blocksize=obs_blocksize,
                prune_m=prune_m,
            )
        else:
            segments = iter_2of4_stage_segments(
                columns,
                previous,
                stage,
                prune_m=prune_m,
                block_fractions=block_fractions,
            )

        for segment_start, segment_end in segments:
            block_start = segment_start
            while block_start < segment_end:
                block_end = min(
                    block_start + (obs_blocksize or (segment_end - segment_start)),
                    segment_end,
                )
                block_weight = ordered_weight[:, block_start:block_end].clone()
                count = block_end - block_start
                block_factor = hessian_factor[block_start:block_end, block_start:block_end]
                diagonal = torch.diag(block_factor)
                quantized = torch.zeros_like(block_weight)
                errors = torch.zeros_like(block_weight)
                block_mask = torch.zeros_like(block_weight, dtype=torch.bool)
                mask_until = 0
                mask_span = mask_blocksize or (obs_blocksize if flat_blocks else count)

                for column in range(count):
                    if freeze_block_mask and column >= mask_until:
                        mask_end = min(column + mask_span, count)
                        if mask_selector == "joint_obs":
                            block_mask[:, column:mask_end] = _joint_obs_2of4_mask(
                                block_weight[:, column:mask_end],
                                block_factor[column:mask_end, column:mask_end],
                                prune_n,
                                prune_m,
                            )
                        else:
                            for mask_column in range(column, mask_end, prune_m):
                                group_end = min(mask_column + prune_m, count)
                                group_scores = (
                                    block_weight[:, mask_column:group_end].square()
                                    / diagonal[mask_column:group_end].reshape(1, -1).square()
                                )
                                block_mask[:, mask_column:group_end] = build_nm_prune_mask(
                                    group_scores,
                                    prune_n,
                                    prune_m,
                                )
                        mask_until = mask_end
                    elif not freeze_block_mask and column % prune_m == 0:
                        group_end = column + prune_m
                        if mask_selector == "joint_obs":
                            block_mask[:, column:group_end] = _joint_obs_2of4_mask(
                                block_weight[:, column:group_end],
                                block_factor[column:group_end, column:group_end],
                                prune_n,
                                prune_m,
                            )
                        else:
                            group_scores = (
                                block_weight[:, column:group_end].square()
                                / diagonal[column:group_end].reshape(1, -1).square()
                            )
                            block_mask[:, column:group_end] = build_nm_prune_mask(
                                group_scores,
                                prune_n,
                                prune_m,
                            )

                    current_weight = block_weight[:, column]
                    current_diagonal = diagonal[column]
                    pruned_weight = current_weight.clone()
                    pruned_weight[block_mask[:, column]] = 0
                    quantized[:, column] = pruned_weight

                    if compensate and compensation_alpha != 0.0:
                        error = (current_weight - pruned_weight) / current_diagonal
                        error = compensation_alpha * error
                        block_weight[:, column:] -= error.unsqueeze(1).matmul(
                            block_factor[column, column:].unsqueeze(0)
                        )
                        errors[:, column] = error

                ordered_weight[:, block_start:block_end] = quantized
                if compensate and compensation_alpha != 0.0 and block_end < columns:
                    ordered_weight[:, block_end:] -= errors.matmul(
                        hessian_factor[block_start:block_end, block_end:]
                    )
                block_start = block_end

        ordered_weight[:, ordered_dead_columns] = 0
        checkpoint = ordered_weight[:, inverse_permutation]
        if isinstance(layer, transformers.Conv1D):
            checkpoint = checkpoint.t()
        checkpoints[stage] = checkpoint.detach().cpu().clone()
        previous = stage

    ordered_weight[:, ordered_dead_columns] = 0
    final_weight = ordered_weight[:, inverse_permutation]
    if isinstance(layer, transformers.Conv1D):
        final_weight = final_weight.t()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    layer.weight.data = final_weight.reshape(layer.weight.shape).to(
        dtype=layer.weight.data.dtype
    )
    return checkpoints
