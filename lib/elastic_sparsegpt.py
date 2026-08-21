"""Experimental reordered SparseGPT helpers and pruning implementation."""

import torch


def balanced_group_slices(columns: int, groups: int) -> list[tuple[int, int]]:
    """Split columns into contiguous groups whose widths differ by at most one."""
    if groups < 1:
        raise ValueError("reorder_blocks must be at least 1")
    if groups > columns:
        raise ValueError("reorder_blocks cannot exceed the layer column count")

    base_width, wider_group_count = divmod(columns, groups)
    slices = []
    start = 0
    for group_index in range(groups):
        width = base_width + int(group_index < wider_group_count)
        end = start + width
        slices.append((start, end))
        start = end
    return slices


def allocate_group_budgets(
    rows: int,
    columns: int,
    sparsity: float,
    slices: list[tuple[int, int]],
) -> list[int]:
    """Allocate an exact global weight budget to groups from left to right."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity_ratio must be in [0, 1]")

    remaining = round(sparsity * rows * columns)
    budgets = []
    for start, end in slices:
        capacity = rows * (end - start)
        budget = min(remaining, capacity)
        budgets.append(budget)
        remaining -= budget
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
    """Rank input columns by their mean original-order SparseGPT score."""
    denominator = torch.diag(hessian_factor).reshape(1, -1).square()
    column_scores = (weight.float().square() / denominator).mean(dim=0)
    return torch.argsort(column_scores, stable=True)


def build_grouped_mask(
    scores: torch.Tensor,
    slices: list[tuple[int, int]],
    budgets: list[int],
) -> torch.Tensor:
    """Mask stable score-ranking prefixes within ordered column groups."""
    if len(slices) != len(budgets):
        raise ValueError("each allocation group must have one budget")

    rows = scores.shape[0]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for (start, end), budget in zip(slices, budgets):
        group_scores = scores[:, start:end].reshape(-1)
        if not 0 <= budget <= group_scores.numel():
            raise ValueError("group budget must be within group capacity")
        order = torch.argsort(group_scores, stable=True)
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask[order[:budget]] = True
        mask[:, start:end] = group_mask.reshape(rows, end - start)
    return mask
