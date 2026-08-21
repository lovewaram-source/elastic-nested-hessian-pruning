def format_result_log(method, sparsity_ratio, perplexity, pruning_time_seconds):
    return (
        "method\tactual_sparsity\tppl_test\tpruning_time_seconds\n"
        f"{method}\t{sparsity_ratio:.4f}\t{perplexity:.4f}\t{pruning_time_seconds:.2f}\n"
    )


def format_elastic_result_log(
    method,
    sparsity_ratio,
    perplexity,
    preparation_time_seconds,
    stage_pruning_time_seconds,
    cumulative_pruning_time_seconds,
):
    return (
        "method\tactual_sparsity\tppl_test\tpreparation_time_seconds\t"
        "stage_pruning_time_seconds\tcumulative_pruning_time_seconds\n"
        f"{method}\t{sparsity_ratio:.4f}\t{perplexity:.4f}\t"
        f"{preparation_time_seconds:.2f}\t{stage_pruning_time_seconds:.2f}\t"
        f"{cumulative_pruning_time_seconds:.2f}\n"
    )
