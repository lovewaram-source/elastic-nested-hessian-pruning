"""Prefix-stage, reordered SparseGPT pruning.

Columns are ranked once per linear layer, then processed left-to-right in
that reordered coordinate system. A target global sparsity ``p`` activates
only the first ``p / 0.8`` of columns; every activated compensation segment
selects an approximately 80% SparseGPT mask from its current weights.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


MAX_SPARSITY = 0.8


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


def iter_prefix_segments(
    columns: int,
    sparsity: float,
    blocksize: int,
    max_sparsity: float = MAX_SPARSITY,
) -> list[tuple[int, int]]:
    """Split the active prefix into compensation segments of at most blocksize."""
    if blocksize < 1:
        raise ValueError("blocksize must be positive")
    prefix = prefix_column_count(columns, sparsity, max_sparsity)
    return [
        (start, min(start + blocksize, prefix))
        for start in range(0, prefix, blocksize)
    ]


def iter_stage_segments(
    columns: int,
    previous_sparsity: float,
    sparsity: float,
    blocksize: int,
    max_sparsity: float = MAX_SPARSITY,
) -> list[tuple[int, int]]:
    """Split only the newly activated prefix interval into compensation segments."""
    if not 0.0 <= previous_sparsity < sparsity <= max_sparsity:
        raise ValueError("stage sparsities must satisfy 0 <= previous < current <= max")
    if blocksize < 1:
        raise ValueError("blocksize must be positive")

    start = prefix_column_count(columns, previous_sparsity, max_sparsity)
    end = prefix_column_count(columns, sparsity, max_sparsity)
    return [
        (segment_start, min(segment_start + blocksize, end))
        for segment_start in range(start, end, blocksize)
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


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return indices that undo a one-dimensional permutation."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device
    )
    return inverse


def column_permutation(
    weight: torch.Tensor,
    hessian_factor: torch.Tensor,
) -> torch.Tensor:
    """Rank columns by the predicted cost of pruning their lowest 80%."""
    denominator = torch.diag(hessian_factor).reshape(1, -1).square()
    scores = weight.float().square() / denominator
    prune_count = round(MAX_SPARSITY * scores.shape[0])
    retained_count = scores.shape[0] - prune_count

    column_costs = scores.sum(dim=0)
    if retained_count:
        retained_scores = torch.topk(
            scores,
            retained_count,
            dim=0,
            largest=True,
            sorted=False,
        ).values
        column_costs -= retained_scores.sum(dim=0)
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


@dataclass(frozen=True)
class ElasticSparseGPTPlan:
    """Reusable second-order statistics for one reordered linear layer."""

    permutation: torch.Tensor
    hessian_factor: torch.Tensor
    dead_columns: torch.Tensor

    def validate_for(self, columns: int) -> None:
        if self.permutation.shape != (columns,):
            raise ValueError("plan permutation does not match layer columns")
        if self.hessian_factor.shape != (columns, columns):
            raise ValueError("plan Hessian factor does not match layer columns")
        if self.dead_columns.shape != (columns,):
            raise ValueError("plan dead-column mask does not match layer columns")

    def cpu(self) -> "ElasticSparseGPTPlan":
        return ElasticSparseGPTPlan(
            permutation=self.permutation.detach().cpu(),
            hessian_factor=self.hessian_factor.detach().float().cpu(),
            dead_columns=self.dead_columns.detach().cpu(),
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
) -> None:
    """Apply only one new prefix interval using a previously prepared plan."""
    weight = _weight_matrix(layer)
    rows, columns = weight.shape
    plan.validate_for(columns)

    segments = iter_stage_segments(
        columns,
        previous_sparsity,
        sparsity,
        blocksize,
    )
    target_budget = (
        round(sparsity * rows * columns)
        - round(previous_sparsity * rows * columns)
    )
    segment_budgets = allocate_segment_budgets(rows, segments, target_budget)

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse_permutation = invert_permutation(permutation)
    hessian_factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead_columns = plan.dead_columns.to(device)

    weight[:, dead_columns] = 0
    weight = weight[:, permutation]

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
            block_weight[:, column:] -= error.unsqueeze(1).matmul(
                block_factor[column, column:].unsqueeze(0)
            )
            errors[:, column] = error

        weight[:, block_start:block_end] = quantized
        weight[:, block_end:] -= errors.matmul(
            hessian_factor[block_start:block_end, block_end:]
        )

    weight = weight[:, inverse_permutation]
    if device.type == "cuda":
        torch.cuda.synchronize()
    if isinstance(layer, transformers.Conv1D):
        weight = weight.t()
    layer.weight.data = weight.reshape(layer.weight.shape).to(
        layer.weight.data.dtype
    )


class ElasticSparseGPT(SparseGPT):
    """SparseGPT with full-layer reordering and staged prefix pruning."""

    def prepare_plan(
        self,
        percdamp: float = 0.01,
        permutation: torch.Tensor | None = None,
    ) -> ElasticSparseGPTPlan:
        """Consume the collected Hessian and return a reusable CPU plan."""
        weight = _weight_matrix(self.layer)
        hessian = self.H
        self.H = None
        if hessian is None:
            raise ValueError("Hessian has already been consumed")

        dead_columns = torch.diag(hessian) == 0
        hessian[dead_columns, dead_columns] = 1
        weight[:, dead_columns] = 0

        damp = percdamp * torch.mean(torch.diag(hessian))
        diagonal_indices = torch.arange(self.columns, device=self.dev)
        hessian[diagonal_indices, diagonal_indices] += damp

        hessian_inverse = torch.cholesky_inverse(
            torch.linalg.cholesky(hessian)
        )
        original_factor = torch.linalg.cholesky(
            hessian_inverse,
            upper=True,
        )
        if permutation is None:
            permutation = column_permutation(weight, original_factor)
        else:
            permutation = permutation.to(self.dev)

        permuted_inverse = hessian_inverse[permutation][:, permutation]
        hessian_factor = torch.linalg.cholesky(
            permuted_inverse,
            upper=True,
        )
        return ElasticSparseGPTPlan(
            permutation=permutation,
            hessian_factor=hessian_factor,
            dead_columns=dead_columns,
        ).cpu()

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
