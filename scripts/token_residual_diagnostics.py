"""Diagnostics only: compare Token selectors without modifying pruning/OBS."""
import math
import time

import torch
import torch.nn.functional as F

from checkpoint_token_isolation_core import RATES, load_block_weights
from run_rate_aligned_nested import _find_layers
from experiment_common import input_groups, synchronize


def top_ids(scores, fraction=.1):
    values = scores.flatten().double().cpu()
    if not torch.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError("expected finite scores and a valid selection fraction")
    return values.argsort(descending=True, stable=True)[:math.ceil(values.numel()*fraction)]


def midranks(values):
    values = values.flatten().double().cpu()
    order = values.argsort(stable=True)
    _, counts = torch.unique_consecutive(values[order], return_counts=True)
    ends = counts.cumsum(0).double()
    ranks = torch.repeat_interleave(ends - (counts.double()+1)/2, counts)
    result = torch.empty_like(values)
    result[order] = ranks
    return result


def compare_selectors(residual, cosine, seed=0, fraction=.1, random_repeats=20):
    """Ranking/coverage diagnostic, not proof of independent PPL benefit."""
    residual, cosine = residual.flatten().double().cpu(), cosine.flatten().double().cpu()
    if residual.shape != cosine.shape or not residual.numel():
        raise ValueError("unaligned or empty score vectors")
    if not torch.isfinite(residual).all() or bool((residual < 0).any()):
        raise ValueError("invalid squared residual scores")
    chosen = top_ids(residual, fraction)
    old = top_ids(cosine, fraction)
    total = float(residual.sum())
    selected = torch.zeros(residual.numel(), dtype=torch.bool)
    selected[chosen] = True
    overlap = int(selected[old].sum()) / chosen.numel()
    a, b = midranks(residual), midranks(cosine)
    a, b = a-a.mean(), b-b.mean()
    denom = float(a.norm()*b.norm())
    generator = torch.Generator().manual_seed(seed)
    random_coverage = [float(residual[torch.randperm(residual.numel(), generator=generator)[:chosen.numel()]].sum())/total
                       for _ in range(random_repeats)] if total > 0 else []
    return dict(total_positions=residual.numel(), selected_positions=chosen.numel(),
                total_residual_replacement_error=total,
                zero_residual_positions=int(residual.eq(0).sum()),
                residual_score_constant=bool(residual.min() == residual.max()),
                cosine_score_constant=bool(cosine.min() == cosine.max()),
                spearman=float(a.dot(b))/denom if denom > 0 else None,
                top_overlap_fraction=overlap,
                coverage_residual=float(residual[chosen].sum())/total if total > 0 else None,
                coverage_cosine=float(residual[old].sum())/total if total > 0 else None,
                coverage_random_mean=sum(random_coverage)/len(random_coverage) if random_coverage else None,
                coverage_random_min=min(random_coverage) if random_coverage else None,
                coverage_random_max=max(random_coverage) if random_coverage else None,
                random_repeats=random_repeats, seed=seed,
                interpretation="self-score coverage favors residual ranking by construction; no PPL claim")


@torch.no_grad()
def block_cosine_scores(paths, device):
    count, length = paths[.2].shape[:2]
    scores = torch.empty(count, length)
    for sample in range(count):
        anchor = paths[.2][sample].to(device).float()
        worst = torch.zeros(length, device=device)
        for rate in RATES[1:]:
            branch = paths[rate][sample].to(device).float()
            worst = torch.maximum(worst, 1-F.cosine_similarity(anchor, branch, dim=-1, eps=1e-8))
        scores[sample] = worst.cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite boundary cosine scores")
    return scores


@torch.no_grad()
def forward_diagnostic_layer(layer, paths, sources, dense_state, attention, positions,
                             device, diagnose=True, chunk_size=512):
    """Actual sparse paths; CPU anchor inputs; bounded FP32 residual projections.

    Only scores are compared across modules: no aggregation of mismatched
    projection scales, no new target weights, no forward replacement.
    """
    modules = _find_layers(layer)
    groups = input_groups(modules)
    group_of = {name: group for group, members in groups.items() for name in members}
    count, length = paths[.2].shape[:2]
    cosine = block_cosine_scores(paths, device) if diagnose else None
    anchors, scores, outputs = {}, {}, {}
    current = dict(rate=None, sample=None, seen={}, residuals={}, deltas={})
    hook_wall = 0.
    kwargs = {}
    if attention is not None:
        kwargs["attention_mask"] = attention.to(device)
    if positions is not None:
        kwargs["position_ids"] = positions.to(device)

    def hook_for(name):
        def hook(_, inputs, output):
            nonlocal hook_wall
            started = time.perf_counter()
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            group = group_of[name]
            identity = (inputs[0].data_ptr(), tuple(inputs[0].shape), inputs[0].stride())
            if name == group:
                current["seen"][group] = identity
            elif current["seen"].get(group) != identity:
                raise AssertionError("projection inputs are not identical")
            sample, rate = current["sample"], current["rate"]
            if rate == .2:
                if name == group:
                    if group not in anchors:
                        anchors[group] = torch.empty(count, length, value.shape[-1], dtype=value.dtype)
                    anchors[group][sample].copy_(value.cpu())
            else:
                destination = scores[rate][name][sample]
                residual = current["residuals"][name]
                # q/k/v and gate/up share actual inputs. Cache their exact
                # FP32 delta once per sample, not a CPU upload per chunk/module.
                if group not in current["deltas"]:
                    current["deltas"][group] = value.float()-anchors[group][sample].to(device).float()
                delta_full = current["deltas"][group]
                energy = torch.empty(length,device=device,dtype=torch.float64)
                for start in range(0, length, chunk_size):
                    end = min(start+chunk_size, length)
                    propagated = F.linear(delta_full[start:end], residual)
                    energy[start:end] = propagated.square().sum(-1,dtype=torch.float64)
                # One synchronization/CPU score transfer per module/sample.
                destination.copy_(energy.cpu())
            hook_wall += time.perf_counter()-started
        return hook

    handles = [module.register_forward_hook(hook_for(name)) for name,module in modules.items()] if diagnose else []
    started = time.perf_counter()
    try:
        for rate in RATES:
            load_block_weights(layer, sources[rate])
            current["rate"] = rate
            if diagnose and rate != .2:
                scores[rate] = {name: torch.empty(count,length,dtype=torch.float64) for name in modules}
                current["residuals"] = {name: dense_state[name].to(device).float()-module.weight.detach().float()
                                        for name,module in modules.items()}
            branch = paths.pop(rate)
            outputs[rate] = torch.empty_like(branch, device="cpu")
            for sample in range(count):
                current["sample"] = sample
                current["seen"].clear()
                current["deltas"].clear()
                output = layer(branch[sample:sample+1].to(device), **kwargs)
                hidden = output[0] if isinstance(output,(tuple,list)) else output
                outputs[rate][sample].copy_(hidden[0].detach().cpu())
            current["residuals"].clear()
            current["deltas"].clear()
            del branch
    finally:
        for handle in handles:
            handle.remove()
    synchronize(device)
    return outputs, scores, cosine, dict(
        forward_counts={str(r):count for r in RATES}, diagnose=diagnose,
        forward_and_diagnostic_seconds=time.perf_counter()-started,
        diagnostic_hook_wall_seconds=hook_wall, chunk_size=chunk_size,
        anchor_storage="CPU actual linear inputs for current selected layer only",
        residual="FP32 original dense W minus compensated sparse reference W_r")
