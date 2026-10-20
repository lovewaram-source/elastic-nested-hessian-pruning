"""Exact global cosine selection with a bounded on-device token reservoir.

Selection is over all calibration positions of one linear-input group. The
20% path is the anchor; the worst cosine distance across the other paths is
ranked. No low-rank approximation or token subsampling is used.
"""
import math

import torch
import torch.nn.functional as F


RATES = (0.2, 0.3, 0.4, 0.5)


class TokenHessianAccumulator:
    def __init__(self, total_tokens, private_fraction):
        if total_tokens < 1 or not 0 <= private_fraction <= 1:
            raise ValueError("invalid token count or private fraction")
        self.total_tokens = total_tokens
        self.fraction = private_fraction
        self.capacity = math.ceil(total_tokens * private_fraction)
        self.samples = 0
        self.seen = 0
        self.size = 0
        self.hessians = {}
        self.values = None
        self.scores = None
        self.ids = None
        self.invalid = None

    @torch.no_grad()
    def add(self, branches):
        if tuple(branches) != RATES:
            raise ValueError("expected paths for all configured ordered rates")
        flat = [branches[r].reshape(-1, branches[r].shape[-1]) for r in RATES]
        if any(x.shape != flat[0].shape for x in flat):
            raise ValueError("unaligned branch activations")
        invalid = torch.stack([~torch.isfinite(x).all() for x in flat]).any()
        self.invalid = invalid if self.invalid is None else self.invalid.logical_or_(invalid)
        tokens, width = flat[0].shape
        if self.seen + tokens > self.total_tokens:
            raise ValueError("more tokens than declared")
        batch = branches[0.2].shape[0] if branches[0.2].ndim == 3 else 1
        previous = self.samples
        self.samples += batch
        for index in (range(len(RATES)) if self.fraction == 1 else (0,)):
            rate = RATES[index]
            if rate not in self.hessians:
                self.hessians[rate] = torch.zeros(
                    width, width, dtype=torch.float32, device=flat[0].device
                )
            hessian = self.hessians[rate]
            hessian.mul_(previous / self.samples)
            x = flat[index].float().t() * math.sqrt(2 / self.samples)
            hessian.add_(x @ x.t())
        if 0 < self.fraction < 1:
            anchor = flat[0].float()
            differences = torch.stack([
                1 - F.cosine_similarity(anchor, x.float(), dim=-1, eps=1e-8)
                for x in flat[1:]
            ]).amax(dim=0)
            if self.values is None:
                self.values = torch.empty(
                    len(RATES), self.capacity, width, device=flat[0].device,
                    dtype=flat[0].dtype,
                )
                self.scores = torch.empty(self.capacity, device=flat[0].device)
                self.ids = torch.empty(self.capacity, device=flat[0].device,
                                       dtype=torch.long)
            candidates = torch.cat((self.scores[:self.size], differences))
            candidate_ids = torch.cat((
                self.ids[:self.size],
                torch.arange(self.seen, self.seen + tokens, device=flat[0].device),
            ))
            # Lexicographic (-distance, position) gives deterministic ties and
            # exactly the same selection as a single full-calibration ranking.
            by_id = candidate_ids.argsort(stable=True)
            ranked = candidates[by_id].argsort(descending=True, stable=True)
            selected = by_id[ranked[:self.capacity]]
            old = selected[selected < self.size]
            new = selected[selected >= self.size] - self.size
            new_size = min(self.capacity, self.size + tokens)
            free = torch.ones(new_size, dtype=torch.bool, device=flat[0].device)
            free[old] = False
            slots = free.nonzero(as_tuple=False).flatten()
            if slots.numel() != new.numel():
                raise AssertionError("reservoir slot accounting failed")
            for index, x in enumerate(flat):
                self.values[index].index_copy_(0, slots, x[new])
            self.scores.index_copy_(0, slots, differences[new])
            self.ids.index_copy_(0, slots, candidate_ids[self.size:][new])
            self.size = new_size
        self.seen += tokens

    @torch.no_grad()
    def finish(self):
        if self.seen != self.total_tokens:
            raise ValueError(f"incomplete calibration: {self.seen}/{self.total_tokens}")
        if bool(self.invalid):
            raise ValueError("nonfinite calibration activations")
        anchor = self.hessians[0.2]
        if self.fraction == 0:
            return {rate: anchor for rate in RATES}, {"private_tokens": 0}
        if self.fraction == 1:
            return self.hessians, {"private_tokens": self.total_tokens}
        if self.size != self.capacity:
            raise AssertionError("incomplete private reservoir")
        private = []
        # Bounded chunks avoid a second full FP32 copy of the reservoir.
        for index in range(len(RATES)):
            hessian = torch.zeros_like(anchor)
            for start in range(0, self.size, 4096):
                x = self.values[index, start:start + 4096].float().t()
                x = x * math.sqrt(2 / self.samples)
                hessian.add_(x @ x.t())
            private.append(hessian)
        shared = anchor - private[0]
        result = {0.2: anchor}
        for index, rate in enumerate(RATES[1:], 1):
            result[rate] = shared + private[index]
        stats = {
            "private_tokens": self.size,
            "total_tokens": self.total_tokens,
            "actual_private_fraction": self.size / self.total_tokens,
            "min_private_cosine_distance": float(self.scores[:self.size].min()),
            "mean_private_cosine_distance": float(self.scores[:self.size].mean()),
            "hessian_relative_to_anchor": {
                str(rate): float((result[rate] - anchor).norm() / anchor.norm().clamp_min(1e-12))
                for rate in RATES[1:]
            },
        }
        self.values = None
        return result, stats


@torch.no_grad()
def prepare_device_plan(hessian, percdamp):
    """Original unsorted SparseGPT factorization, kept on its consumer device."""
    from rate_aligned_src.elastic_sparsegpt import ElasticSparseGPTPlan

    hessian = hessian.clone()
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diagonal, diagonal] += percdamp * torch.diag(hessian).mean()
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    factor = torch.linalg.cholesky(inverse, upper=True)
    return ElasticSparseGPTPlan(diagonal, factor, dead)
