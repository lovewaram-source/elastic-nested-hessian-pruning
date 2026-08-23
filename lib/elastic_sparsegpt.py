"""Seven-slice, reordered SparseGPT pruning.

Columns are ranked once per linear layer, then processed left-to-right in
that reordered coordinate system. The first slice contains 25% of columns and
the remaining six contain 12.5% each. Every slice is one complete SparseGPT
compensation block and contributes the next global checkpoint from 20% to 80%.
"""

from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


MAX_SPARSITY = 0.8
ELASTIC_SPARSITY_STAGES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


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
    boundaries = [
        prefix_column_count(columns, checkpoint, max_sparsity)
        for checkpoint in checkpoints
    ]
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


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return indices that undo a one-dimensional permutation."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device
    )
    return inverse


def column_permutation(
    weight: torch.Tensor,
    hessian_factor: torch.Tensor | None = None,
    ordering: str = "sparsegpt",
    activation_norms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank columns by full-column SparseGPT or Wanda sensitivity."""
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
    else:
        raise ValueError("ordering must be 'sparsegpt' or 'wanda'")

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
    if plan.hessian_factor is None:
        raise ValueError("static elastic stage requires a Hessian factor")
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
    del rows
    plan.validate_for(columns)
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
    ) -> ElasticSparseGPTPlan:
        """Consume the collected Hessian and return a reusable CPU plan."""
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
        if permutation is None and ordering == "sparsegpt":
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
                activation_norms=activation_norms,
            )
            finish_timing("column_ordering_seconds", ordering_start)
        else:
            permutation = permutation.to(self.dev)
            if timings is not None:
                timings.setdefault("column_ordering_seconds", 0.0)

        if not retain_dynamic_wanda_statistics:
            reordered_factor_start = start_timing()
            permuted_inverse = hessian_inverse[permutation][:, permutation]
            hessian_factor = torch.linalg.cholesky(
                permuted_inverse,
                upper=True,
            )
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
        if not retain_dynamic_wanda_statistics:
            finish_timing("reordered_factor_seconds", reordered_factor_start)
        elif timings is not None:
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
