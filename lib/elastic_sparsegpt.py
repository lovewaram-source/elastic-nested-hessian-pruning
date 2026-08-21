"""Experimental reordered SparseGPT helpers and pruning implementation."""

import torch
import torch.nn as nn
import transformers

from .sparsegpt import SparseGPT


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


class ElasticSparseGPT(SparseGPT):
    """SparseGPT with an ordered, grouped, fixed unstructured mask."""

    def fasterprune(
        self,
        sparsity: float,
        reorder_blocks: int = 5,
        blocksize: int = 128,
        percdamp: float = 0.01,
    ) -> None:
        slices = balanced_group_slices(self.columns, reorder_blocks)
        budgets = allocate_group_budgets(
            self.rows,
            self.columns,
            sparsity,
            slices,
        )

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        hessian_inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
        original_factor = torch.linalg.cholesky(hessian_inverse, upper=True)
        permutation = column_permutation(W, original_factor)
        inverse_permutation = invert_permutation(permutation)

        W = W[:, permutation]
        hessian_inverse = hessian_inverse[permutation][:, permutation]
        hessian_factor = torch.linalg.cholesky(hessian_inverse, upper=True)

        scores = W.square() / torch.diag(hessian_factor).reshape(1, -1).square()
        mask = build_grouped_mask(scores, slices, budgets)

        for block_start in range(0, self.columns, blocksize):
            block_end = min(block_start + blocksize, self.columns)
            block_columns = block_end - block_start

            block_weight = W[:, block_start:block_end].clone()
            quantized = torch.zeros_like(block_weight)
            errors = torch.zeros_like(block_weight)
            block_factor = hessian_factor[
                block_start:block_end,
                block_start:block_end,
            ]
            block_mask = mask[:, block_start:block_end]

            for column in range(block_columns):
                weight = block_weight[:, column]
                diagonal = block_factor[column, column]

                pruned_weight = weight.clone()
                pruned_weight[block_mask[:, column]] = 0
                quantized[:, column] = pruned_weight

                error = (weight - pruned_weight) / diagonal
                block_weight[:, column:] -= error.unsqueeze(1).matmul(
                    block_factor[column, column:].unsqueeze(0)
                )
                errors[:, column] = error

            W[:, block_start:block_end] = quantized
            W[:, block_end:] -= errors.matmul(
                hessian_factor[block_start:block_end, block_end:]
            )

        W = W[:, inverse_permutation]
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
