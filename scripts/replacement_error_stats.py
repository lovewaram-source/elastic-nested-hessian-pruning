"""Exact CDF/range coverage for worst-branch relative Token replacement error."""
import torch


ERROR_PERCENT_EDGES = tuple(range(5, 101, 5)) + (150, 200, 500, 1000)


def describe_errors(errors, valid):
    errors = errors.detach().reshape(-1).cpu().float()
    valid = valid.detach().reshape(-1).cpu().bool()
    if errors.shape != valid.shape or not errors.numel():
        raise ValueError("error statistics require aligned nonempty positions")
    values = errors[valid]
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("valid relative errors must be finite and nonnegative")
    edges = torch.tensor(ERROR_PERCENT_EDGES, dtype=torch.float32) / 100
    counts = torch.searchsorted(values.sort().values, edges, right=True).tolist()
    total = errors.numel()
    cdf, bins = [], []
    previous_count, previous_edge = 0, 0
    for edge, count in zip(ERROR_PERCENT_EDGES, counts):
        cdf.append(dict(upper_error_percent=edge, count=count, fraction=count / total))
        bins.append(dict(lower_error_percent=previous_edge, upper_error_percent=edge,
                         lower_inclusive=previous_edge == 0, upper_inclusive=True,
                         count=count - previous_count, fraction=(count - previous_count) / total))
        previous_count, previous_edge = count, edge
    bins.append(dict(lower_error_percent=previous_edge, upper_error_percent=None,
                     lower_inclusive=False, upper_inclusive=False,
                     count=values.numel() - previous_count, fraction=(values.numel() - previous_count) / total))
    quantiles = None
    if values.numel():
        labels = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")
        q = torch.tensor([0., .25, .5, .75, .9, .95, .99, 1.])
        quantiles = dict(zip(labels, values.quantile(q).tolist()))
    return dict(total_tokens=total, valid_tokens=values.numel(), invalid_norm_tokens=total - values.numel(),
                cdf=cdf, bins=bins, quantiles_valid_only=quantiles,
                mean_valid_only=float(values.mean()) if values.numel() else None)


def summarize_layers(layer_errors, threshold):
    if not layer_errors:
        raise ValueError("no layer inputs measured")

    def pooled(rows):
        if not rows:
            return None
        errors = torch.cat([x.reshape(-1).cpu() for x, _ in rows])
        valid = torch.cat([v.reshape(-1).cpu() for _, v in rows])
        result = describe_errors(errors, valid)
        shared = int((valid & (errors <= threshold)).sum())
        result.update(anchor_shared_tokens=shared, anchor_share_fraction=shared / errors.numel(),
                      averaged_tokens=errors.numel() - shared,
                      averaged_fraction=1 - shared / errors.numel())
        return result

    return dict(layer_count=len(layer_errors), threshold=threshold,
                denominator="all measured layer-input positions, including invalid-norm positions",
                all_layer_inputs=pooled(layer_errors), excluding_embedding_input=pooled(layer_errors[1:]))
