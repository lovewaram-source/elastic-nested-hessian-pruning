"""Read-only probes on the actual stagewise 2:4 OBS trajectory.

All losses are sampled-row, sampled-token LOCAL reconstruction diagnostics,
not PPL. All donors start from the same effective FP32 state, including lazy
suffix compensation. First-group masks are fixed; subsequent window masks
are recomputed from each donor's own updated weights. No trial is committed.
"""
import itertools

import torch

RATES = (.2, .3, .4, .5)


def same_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


@torch.no_grad()
def effective_state(context, row_ids):
    """Materialize pending block errors without changing any borrowed tensor."""
    start, end, column = (context[k] for k in ("block_start", "block_end", "column"))
    state = context["ordered_weight"].index_select(0, row_ids).clone()
    state[:, start:start+column] = context["quantized"].index_select(0, row_ids)[:, :column]
    state[:, start+column:end] = context["block_weight"].index_select(0, row_ids)[:, column:]
    if column and end < state.shape[1]:
        state[:, end:] -= context["errors"].index_select(0, row_ids)[:, :column] @ context[
            "hessian_factor"][start:start+column, end:]
    return state


@torch.no_grad()
def shadow_window(state, factor, first_mask, start, stage_end, alpha, groups):
    """Sequential OBS in the effective state, with lazy suffix GEMM per group.

    This is an immediate-effective-state shadow, not a replacement kernel.
    Regrouping GEMMs can round differently from the driver's larger blocks.
    """
    from rate_aligned_src.nm import build_nm_prune_mask

    value = state.clone()
    masks, before_masks = [], []
    stop = min(stage_end, start + 4*groups)
    for offset in range(start, stop, 4):
        end = offset + 4
        block = value[:, offset:end].clone()
        diagonal = torch.diag(factor)[offset:end]
        mask = (first_mask.clone() if offset == start else
                build_nm_prune_mask(block.square()/diagonal.reshape(1, -1).square(), 2, 4))
        before_masks.append(block.clone())
        masks.append(mask)
        quantized = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        for j in range(4):
            current = block[:, j]
            quantized[:, j] = current.masked_fill(mask[:, j], 0)
            if alpha:
                error = alpha * ((current-quantized[:, j])/diagonal[j])
                block[:, j:] -= error.unsqueeze(1) @ factor[offset+j, offset+j:end].unsqueeze(0)
                errors[:, j] = error
        value[:, offset:end] = quantized
        if alpha and end < value.shape[1]:
            value[:, end:] -= errors @ factor[offset:end, end:]
    return value, torch.cat(masks, 1), before_masks


def _direction(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    na, nb = a.norm(), b.norm()
    return dict(cosine=float((a@b)/(na*nb)) if na > 0 and nb > 0 else None,
                norm_a=float(na), norm_b=float(nb))


class ConflictObserver:
    def __init__(self, plans, probes, dense_weight, emit, rows=64, window_groups=3):
        if rows < 1 or window_groups < 1:
            raise ValueError("positive diagnostic rows and window required")
        self.plans, self.probes, self.emit = plans, probes, emit
        self.window_groups = window_groups
        self.stage = .2
        device = dense_weight.device
        self.ids = torch.linspace(0, dense_weight.shape[0]-1,
                                  min(rows, dense_weight.shape[0]), device=device).long().unique()
        self.dense = dense_weight.index_select(0, self.ids).float().clone()
        self.event_count = 0
        self.wall_seconds = 0.
        for plan in plans.values():
            if not torch.equal(plan.permutation, plans[.2].permutation):
                raise ValueError("conflict diagnosis requires identical ordered coordinates")
            if not torch.equal(plan.dead_columns, plans[.2].dead_columns):
                raise ValueError("branch dead columns differ; fixed-support comparison invalid")
        permutation = plans[.2].permutation.to(device)
        self.dense = self.dense[:, permutation]
        self.features = {split: {r: x.to(device=device, dtype=torch.float32)[permutation]
                                 for r, x in values.items()} for split, values in probes.items()}

    @torch.no_grad()
    def __call__(self, **context):
        import time
        absolute = context["block_start"] + context["column"]
        left, right = context["segment_start"], context["segment_end"]
        total = (right-left)//4
        slots = {0, (total-1)//2, total-1}
        if (absolute-left)//4 not in slots:
            return
        tick = time.perf_counter()
        state = effective_state(context, self.ids)
        if not torch.equal(context["permutation"], self.plans[.2].permutation):
            raise ValueError("driver and donor order differ")
        first_mask = context["block_mask"].index_select(0, self.ids)[
            :, context["column"]:context["group_end"]].clone()
        first_mask_digest = __import__("hashlib").sha256(first_mask.cpu().numpy().tobytes()).hexdigest()
        users = [r for r in RATES if r >= self.stage]
        horizons = []
        for groups in sorted({1, min(self.window_groups, (right-absolute)//4)}):
            candidates, candidate_masks = {}, {}
            no_comp, no_mask, _ = shadow_window(state, context["hessian_factor"], first_mask,
                                                absolute, right, 0., groups)
            candidates["no_compensation"] = no_comp
            candidate_masks["no_compensation"] = no_mask
            for rate, plan in self.plans.items():
                name = f"H{round(rate*100)}"
                candidate, mask, _ = shadow_window(state, plan.hessian_factor, first_mask,
                                                   absolute, right, .25, groups)
                candidate[:, context["dead_columns"]] = 0
                candidates[name], candidate_masks[name] = candidate, mask
            stop = absolute + 4*groups
            for name, value in candidates.items():
                if not torch.equal(value[:, :absolute], state[:, :absolute]):
                    raise AssertionError("shadow changed frozen prefix")
                if not value[:, absolute:stop][candidate_masks[name]].eq(0).all():
                    raise AssertionError("shadow resurrected a pruned weight")
            losses = []
            for split, features in self.features.items():
                for evaluator, x in features.items():
                    target = self.dense @ x
                    baseline = ((no_comp-self.dense) @ x).double().square().mean()
                    for name, value in candidates.items():
                        mse = ((value-self.dense) @ x).double().square().mean()
                        losses.append(dict(split=split, evaluator=evaluator,
                            evaluator_uses_current_slice=evaluator in users,
                            donor=name, donor_uses_current_slice=(name != "no_compensation" and
                                float(name[1:])/100 in users),
                            mse=float(mse), delta_mse_vs_no_compensation=float(mse-baseline),
                            relative_delta_vs_no_compensation=float((mse-baseline)/baseline)
                                if baseline > 0 else None,
                            relative_output_mse=float(mse/target.double().square().mean())
                                if target.double().square().mean() > 0 else None))
            directions = []
            for a, b in itertools.combinations([f"H{round(r*100)}" for r in users], 2):
                # Only still-live, not just-pruned weights can absorb error.
                da, db = (candidates[n]-no_comp for n in (a, b))
                support = torch.ones_like(state, dtype=torch.bool)
                support[:, :absolute] = False
                support[:, absolute:stop] &= ~(candidate_masks[a] | candidate_masks[b])
                support[:, context["dead_columns"]] = False
                directions.append(dict(a=a, b=b, **_direction(da[support], db[support])))
            horizons.append(dict(groups=groups, future_masks_recomputed=groups > 1,
                mask_disagreements_vs_no_compensation={name: int((mask != no_mask).sum())
                    for name, mask in candidate_masks.items()},
                shadow_fp32_effective_state=True, losses=losses, directions=directions))
        self.event_count += 1
        self.emit(dict(stage=self.stage, column=absolute, stage_start=left, stage_end=right,
            position_fraction=(absolute-left)/max(right-left-4, 1), users=users,
            row_ids=self.ids.cpu().tolist(), first_mask_sha256=first_mask_digest,
            horizons=horizons))
        self.wall_seconds += time.perf_counter()-tick
