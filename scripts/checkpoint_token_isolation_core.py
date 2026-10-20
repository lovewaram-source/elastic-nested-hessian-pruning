"""Token isolation from actual, fixed sparse checkpoint forward paths.

One block-boundary position mask is shared by all linear projections. Hooks
consume the real linear inputs produced by each checkpoint's own weights.
"""
import hashlib
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from token_isolation_core import RATES


def select_all_mean_tokens(paths):
    """Every position uses the across-rate mean, including the embedding input."""
    if tuple(paths) != RATES:
        raise ValueError("expected all configured ordered actual reference paths")
    shape = paths[.2].shape
    if len(shape) != 3 or any(value.shape != shape for value in paths.values()):
        raise ValueError("unaligned block-boundary activations")
    mask = torch.ones(shape[:2], dtype=torch.bool)
    total = mask.numel()
    stats = dict(rule="all_mean", private_tokens=total, shared_tokens=0, total_tokens=total,
                 actual_private_fraction=1., actual_shared_fraction=0.,
                 position_sha256=hashlib.sha256(torch.arange(total).numpy().tobytes()).hexdigest())
    return mask, stats


@torch.no_grad()
def select_replacement_error_tokens(paths, device, threshold=.05):
    """True mask means average; only valid worst relative error<=threshold shares20%."""
    from replacement_error_stats import describe_errors
    if tuple(paths) != RATES or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("expected ordered paths and a finite nonnegative threshold")
    shape = paths[.2].shape
    if len(shape) != 3 or any(value.shape != shape for value in paths.values()):
        raise ValueError("unaligned block-boundary activations")
    scores = torch.empty(shape[:2], device=device, dtype=torch.float32)
    valid = torch.empty(shape[:2], device=device, dtype=torch.bool)
    nonfinite = torch.zeros((), device=device, dtype=torch.bool)
    for sample in range(shape[0]):
        anchor = paths[.2][sample].to(device).float()
        nonfinite.logical_or_(~torch.isfinite(anchor).all())
        sample_valid = anchor.norm(dim=-1) > 1e-8
        worst = torch.zeros(shape[1], device=device)
        for rate in RATES[1:]:
            branch = paths[rate][sample].to(device).float()
            nonfinite.logical_or_(~torch.isfinite(branch).all())
            norm = branch.norm(dim=-1)
            sample_valid &= norm > 1e-8
            error = (branch - anchor).norm(dim=-1) / norm.clamp_min(1e-8)
            worst = torch.maximum(worst, error)
        scores[sample], valid[sample] = worst, sample_valid
    if bool(nonfinite) or not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite block-boundary activations or errors")
    scores, valid = scores.cpu(), valid.cpu()
    shared = valid & (scores <= threshold)
    mask = ~shared
    stats = describe_errors(scores, valid)
    k, total = int(mask.sum()), mask.numel()
    ids = mask.reshape(-1).nonzero().flatten()
    stats.update(rule="relative_error", share_error_threshold=threshold,
                 private_tokens=k, shared_tokens=total - k, actual_private_fraction=k / total,
                 actual_shared_fraction=1 - k / total,
                 position_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    return mask, stats, scores, valid


@torch.no_grad()
def select_block_private_tokens(paths, device, fraction=.1):
    if tuple(paths) != RATES or fraction not in (0., .1, .5, 1.):
        raise ValueError("requires configured ordered paths and 0%, 10%, 50% or 100% isolation")
    shape = paths[.2].shape
    if len(shape) != 3 or any(value.shape != shape for value in paths.values()):
        raise ValueError("unaligned block-boundary activations")
    scores = torch.empty(shape[:2], dtype=torch.float32, device=device)
    invalid = torch.zeros((), dtype=torch.bool, device=device)
    for sample in range(shape[0]):
        anchor = paths[.2][sample].to(device).float()
        distances = []
        invalid.logical_or_(~torch.isfinite(anchor).all())
        for rate in RATES[1:]:
            branch = paths[rate][sample].to(device).float()
            invalid.logical_or_(~torch.isfinite(branch).all())
            distances.append(1 - F.cosine_similarity(anchor, branch, dim=-1, eps=1e-8))
        scores[sample] = torch.stack(distances).amax(dim=0)
    if bool(invalid):
        raise ValueError("nonfinite block-boundary activations")
    count = math.prod(shape[:2])
    k = math.ceil(count * fraction)
    # Flatten order is (sample, sequence position); stable ties preserve it.
    ids = scores.flatten().argsort(descending=True, stable=True)[:k].cpu()
    mask = torch.zeros(count, dtype=torch.bool)
    mask[ids] = True
    digest = hashlib.sha256(ids.sort().values.numpy().tobytes()).hexdigest()
    stats = {
        "private_tokens": k, "total_tokens": count,
        "actual_private_fraction": k / count,
        "position_sha256": digest,
        "min_private_cosine_distance": (float(scores.flatten()[ids.to(device)].min()) if k else None),
        "mean_private_cosine_distance": (float(scores.flatten()[ids.to(device)].mean()) if k else None),
    }
    return mask.reshape(shape[:2]), stats


class FixedPositionHessians:
    """Shared anchor statistics plus private statistics at fixed positions."""

    def __init__(self, width, device, expected_samples):
        self.shared = torch.zeros(width, width, device=device, dtype=torch.float32)
        self.private = {rate: torch.zeros_like(self.shared) for rate in RATES}
        self.samples = {rate: 0 for rate in RATES}
        self.expected = expected_samples

    @torch.no_grad()
    def add(self, rate, value, private_ids, shared_ids, include_shared=True):
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("calibration requires one full sequence per forward")
        previous = self.samples[rate]
        self.samples[rate] += 1
        factor = previous / self.samples[rate]
        scale = math.sqrt(2 / self.samples[rate])
        self.private[rate].mul_(factor)
        x = value.reshape(-1, value.shape[-1]).index_select(0, private_ids).float().t()
        x = x * scale
        self.private[rate].add_(x @ x.t())
        if include_shared and rate == .2:
            self.shared.mul_(factor)
            x = value.reshape(-1, value.shape[-1]).index_select(0, shared_ids).float().t()
            x = x * scale
            self.shared.add_(x @ x.t())

    @torch.no_grad()
    def add_mean_shared(self, summed_values, sample):
        if any(n != sample + 1 for n in self.samples.values()):
            raise AssertionError("mean requires all configured actual branch inputs at this sample")
        self.shared.mul_(sample / (sample + 1))
        # Average vectors first: mean(x) mean(x)^T includes cross-branch terms.
        x = (summed_values / len(RATES)).t() * math.sqrt(2 / (sample + 1))
        self.shared.add_(x @ x.t())

    @torch.no_grad()
    def finish(self):
        if any(count != self.expected for count in self.samples.values()):
            raise AssertionError(f"missing checkpoint forwards: {self.samples}")
        return {rate: self.shared + self.private.pop(rate) for rate in RATES}


class MeanHessian:
    """Equal-weight average of all configured actual input Grams in one FP32 buffer."""

    def __init__(self, width, device, expected_samples):
        self.common = torch.zeros(width, width, device=device, dtype=torch.float32)
        self.samples = {rate: 0 for rate in RATES}
        self.expected = expected_samples

    @torch.no_grad()
    def add(self, rate, value, private_ids, shared_ids, include_shared=False):
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("calibration requires one full sequence per forward")
        self.samples[rate] += 1
        x = value.reshape(-1, value.shape[-1]).float().t()
        self.common.addmm_(x, x.t(), beta=1., alpha=2 / (self.expected * len(RATES)))

    def finish(self):
        if any(n != self.expected for n in self.samples.values()):
            raise AssertionError(f"missing checkpoint forwards: {self.samples}")
        return {rate: self.common for rate in RATES}


class IsolatedMeanHessian:
    """One common H: anchor20 outside the selected set; across-rate mean inside."""

    def __init__(self, width, device, expected_samples):
        self.anchor = torch.zeros(width, width, device=device, dtype=torch.float32)
        self.averaged = torch.zeros_like(self.anchor)
        self.samples = {rate: 0 for rate in RATES}
        self.expected = expected_samples

    @torch.no_grad()
    def add(self, rate, value, private_ids, shared_ids, include_shared=False):
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("calibration requires one full sequence per forward")
        self.samples[rate] += 1
        if rate == .2:
            n = self.samples[rate]
            self.anchor.mul_((n - 1) / n)
            if shared_ids.numel() == 0:
                return
            x = value.reshape(-1, value.shape[-1]).index_select(0, shared_ids).float().t()
            x = x * math.sqrt(2 / n)
            self.anchor.add_(x @ x.t())

    @torch.no_grad()
    def add_mean_shared(self, summed_values, sample):
        if any(n != sample + 1 for n in self.samples.values()):
            raise AssertionError("isolated mean requires all configured actual branch inputs")
        self.averaged.mul_(sample / (sample + 1))
        if summed_values.numel() == 0:
            return
        x = (summed_values / len(RATES)).t() * math.sqrt(2 / (sample + 1))
        self.averaged.add_(x @ x.t())

    @torch.no_grad()
    def finish(self):
        if any(n != self.expected for n in self.samples.values()):
            raise AssertionError(f"missing checkpoint forwards: {self.samples}")
        common = self.anchor + self.averaged
        return {rate: common for rate in RATES}


@torch.no_grad()
def load_block_weights(layer, path):
    from run_rate_aligned_nested import _find_layers
    state = path if isinstance(path, Mapping) else torch.load(path, map_location="cpu", weights_only=True)
    modules = _find_layers(layer)
    if set(state) != set(modules):
        raise ValueError(f"checkpoint module mismatch: {path}")
    for name, module in modules.items():
        if state[name].shape != module.weight.shape:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        module.weight.copy_(state[name].to(module.weight.device))


@torch.no_grad()
def collect_checkpoint_hessians(layer, paths, private_mask, checkpoint_files,
                                attention_mask, position_ids, device, shared_fusion="anchor20"):
    from run_rate_aligned_nested import _find_layers
    from experiment_common import input_groups

    if shared_fusion not in ("anchor20", "mean_token", "isolated_mean_anchor20", "mean_hessian"):
        raise ValueError("unknown shared Token fusion")

    modules = _find_layers(layer)
    groups = input_groups(modules)
    count, length = paths[.2].shape[:2]
    if private_mask.shape != (count, length):
        raise ValueError("position mask does not match branch paths")
    if shared_fusion == "mean_hessian" and not bool(private_mask.all()):
        raise ValueError("mean_hessian requires all positions; no Token routing")
    accumulator_type = (MeanHessian if shared_fusion == "mean_hessian" else
                        IsolatedMeanHessian if shared_fusion == "isolated_mean_anchor20" else FixedPositionHessians)
    accumulators = {
        group: accumulator_type(modules[group].weight.shape[1], device, count)
        for group in groups
    }
    # Indices have CPU-known sizes. Build once per sequence and share them
    # across all rates/projections; avoid GPU boolean indexing synchronization.
    indices = [
        (mask.nonzero().flatten().to(device), (~mask).nonzero().flatten().to(device))
        for mask in private_mask
    ]
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(device)
    outputs = {}
    trace = {"forward_counts": {}, "checkpoints": {}, "input_reuse_checked": True,
             "shared_fusion": shared_fusion}
    current = {"rate": None, "sample": None}
    seen = {}

    def hook_for(group, name):
        def hook(_, inputs, output):
            value = inputs[0].detach()
            identity = (value.data_ptr(), tuple(value.shape), value.stride())
            if name != group:
                if seen[group] != identity:
                    raise AssertionError(f"projection inputs differ: {group}/{name}")
                return
            seen[group] = identity
            private, shared = indices[current["sample"]]
            accumulators[group].add(current["rate"], value, private, shared,
                                    include_shared=shared_fusion == "anchor20")
            if shared_fusion not in ("anchor20", "mean_hessian"):
                mean_ids = private if shared_fusion == "isolated_mean_anchor20" else shared
                selected = value.reshape(-1, value.shape[-1]).index_select(0, mean_ids).float()
                if group not in current["shared_sums"]:
                    current["shared_sums"][group] = selected
                else:
                    current["shared_sums"][group].add_(selected)
        return hook

    handles = [modules[name].register_forward_hook(hook_for(group, name))
               for group, names in groups.items() for name in names]

    def record_checkpoint(rate, source):
        state = source if isinstance(source, Mapping) else torch.load(source, map_location="cpu", weights_only=True)
        zero_count = sum(int(value.eq(0).sum()) for value in state.values())
        total = sum(value.numel() for value in state.values())
        trace["checkpoints"][str(rate)] = {
            "file": f"memory:rate={rate:.2f}" if isinstance(source, Mapping) else str(source),
            "actual_zero_fraction": zero_count / total,
        }
        return state

    def forward_sample(rate, sample, branch):
        current["rate"], current["sample"] = rate, sample
        seen.clear()
        output = layer(branch[sample:sample + 1].to(device), **kwargs)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        outputs[rate][sample].copy_(hidden[0].detach().cpu())

    try:
        if shared_fusion not in ("anchor20", "mean_hessian"):
            # Four current-block weight snapshots fit on one GPU. Reinstall
            # them via device-to-device copies, not CPU/disk reload per sample.
            states = {}
            for rate in RATES:
                source = record_checkpoint(rate, checkpoint_files[rate])
                states[rate] = {name: value.to(device, copy=True) for name, value in source.items()}
                outputs[rate] = torch.empty_like(paths[rate], device="cpu")
            trace["gpu_reference_copies"] = len(states)
            trace["mean_buffer_scope"] = "one sample, actual linear-input groups, FP32"
            trace["averaged_position_set"] = "isolated" if shared_fusion == "isolated_mean_anchor20" else "nonisolated"
            trace["all_rates_same_hessian"] = shared_fusion == "isolated_mean_anchor20"
            for sample in range(count):
                current["shared_sums"] = {}
                for rate in RATES:
                    load_block_weights(layer, states[rate])
                    forward_sample(rate, sample, paths[rate])
                for group in groups:
                    accumulators[group].add_mean_shared(current["shared_sums"].pop(group), sample)
            paths.clear()
            for rate in RATES:
                trace["forward_counts"][str(rate)] = count
                print(f"REFERENCE_FORWARD rate={rate:.0%} samples={count} "
                      f"fusion={shared_fusion} checkpoint={trace['checkpoints'][str(rate)]['file']}", flush=True)
            del states, source
        else:
            if shared_fusion == "mean_hessian":
                trace.update(all_rates_same_hessian=True, averaged_position_set="all",
                             aggregation="mean of configured input Grams, not Gram of mean input",
                             gpu_reference_copies=0, hessian_buffers_per_input_group=1)
            for rate in RATES:
                # Install the actual sparse checkpoint before the real forward.
                state = record_checkpoint(rate, checkpoint_files[rate])
                load_block_weights(layer, state)
                del state
                branch = paths.pop(rate)
                outputs[rate] = torch.empty_like(branch, device="cpu")
                for sample in range(count):
                    forward_sample(rate, sample, branch)
                trace["forward_counts"][str(rate)] = count
                print(f"REFERENCE_FORWARD rate={rate:.0%} samples={count} "
                      f"zero_fraction={trace['checkpoints'][str(rate)]['actual_zero_fraction']:.8f} "
                      f"checkpoint={trace['checkpoints'][str(rate)]['file']}", flush=True)
                del branch
    finally:
        for handle in handles:
            handle.remove()
    hessians = {group: accumulators.pop(group).finish() for group in groups}
    return hessians, outputs, trace
