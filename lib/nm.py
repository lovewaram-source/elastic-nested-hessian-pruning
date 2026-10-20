import torch


def build_nm_prune_mask(
    metric: torch.Tensor,
    prune_n: int,
    prune_m: int,
) -> torch.Tensor:
    """Return a boolean mask that prunes ``prune_n`` of every ``prune_m``.

    The mask is row-wise over the last dimension. ``True`` marks weights to
    prune. This is the conventional local N:M rule used by the 2:4 baselines.
    """
    if metric.ndim != 2:
        raise ValueError("N:M metric must be two-dimensional")
    if not 0 < prune_n < prune_m:
        raise ValueError("prune_n must be in (0, prune_m)")
    if metric.shape[1] % prune_m != 0:
        raise ValueError("input columns must be divisible by prune_m")

    rows, columns = metric.shape
    grouped_metric = metric.float().reshape(rows, columns // prune_m, prune_m)
    order = torch.argsort(grouped_metric, dim=-1, stable=True)
    grouped_mask = torch.zeros_like(grouped_metric, dtype=torch.bool)
    grouped_mask.scatter_(2, order[:, :, :prune_n], True)
    return grouped_mask.reshape_as(metric)
