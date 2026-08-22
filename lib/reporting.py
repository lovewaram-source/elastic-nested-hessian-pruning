def format_result_log(
    model_name,
    method,
    sparsity_ratio,
    perplexity,
    pruning_time_seconds,
):
    return (
        "model\tmethod\tactual_sparsity\tppl_test\tpruning_time_seconds\n"
        f"{model_name}\t{method}\t{sparsity_ratio:.4f}\t"
        f"{perplexity:.4f}\t{pruning_time_seconds:.2f}\n"
    )


def format_elastic_result_log(
    model_name,
    method,
    sparsity_ratio,
    perplexity,
    preparation_time_seconds,
    stage_pruning_time_seconds,
    cumulative_pruning_time_seconds,
    preparation_breakdown=None,
):
    if preparation_breakdown is None:
        return (
            "model\tmethod\tactual_sparsity\tppl_test\tpreparation_time_seconds\t"
            "stage_pruning_time_seconds\tcumulative_pruning_time_seconds\n"
            f"{model_name}\t{method}\t{sparsity_ratio:.4f}\t{perplexity:.4f}\t"
            f"{preparation_time_seconds:.2f}\t{stage_pruning_time_seconds:.2f}\t"
            f"{cumulative_pruning_time_seconds:.2f}\n"
        )
    breakdown_fields = (
        "data_loading_seconds",
        "calibration_forward_seconds",
        "hessian_factorization_seconds",
        "column_ordering_seconds",
        "reordered_factor_seconds",
        "plan_cache_write_seconds",
    )
    breakdown_header = "\t".join(breakdown_fields)
    breakdown_values = "\t".join(
        f"{preparation_breakdown.get(field, 0.0):.2f}"
        for field in breakdown_fields
    )
    return (
        "model\tmethod\tactual_sparsity\tppl_test\tpreparation_time_seconds\t"
        f"{breakdown_header}\tstage_pruning_time_seconds\t"
        "cumulative_pruning_time_seconds\n"
        f"{model_name}\t{method}\t{sparsity_ratio:.4f}\t{perplexity:.4f}\t"
        f"{preparation_time_seconds:.2f}\t{breakdown_values}\t"
        f"{stage_pruning_time_seconds:.2f}\t"
        f"{cumulative_pruning_time_seconds:.2f}\n"
    )
