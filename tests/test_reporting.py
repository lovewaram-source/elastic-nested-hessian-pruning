import unittest

from lib.reporting import format_result_log


class ResultLogFormattingTests(unittest.TestCase):
    def test_elastic_result_log_separates_shared_and_incremental_time(self):
        from lib.reporting import format_elastic_result_log

        result_log = format_elastic_result_log(
            model_name="llama-2-7b-hf",
            method="elastic_sparsegpt",
            sparsity_ratio=0.2,
            perplexity=5.8,
            preparation_time_seconds=200.0,
            stage_pruning_time_seconds=18.0,
            cumulative_pruning_time_seconds=236.0,
        )

        self.assertEqual(
            result_log,
            "model\tmethod\tactual_sparsity\tppl_test\tpreparation_time_seconds\t"
            "stage_pruning_time_seconds\tcumulative_pruning_time_seconds\n"
            "llama-2-7b-hf\telastic_sparsegpt\t0.2000\t5.8000\t"
            "200.00\t18.00\t236.00\n",
        )

    def test_result_log_includes_pruning_duration(self):
        result_log = format_result_log(
            model_name="llama-2-7b-hf",
            method="wanda",
            sparsity_ratio=0.5,
            perplexity=6.42,
            pruning_time_seconds=183.42,
        )

        self.assertEqual(
            result_log,
            "model\tmethod\tactual_sparsity\tppl_test\tpruning_time_seconds\n"
            "llama-2-7b-hf\twanda\t0.5000\t6.4200\t183.42\n",
        )


if __name__ == "__main__":
    unittest.main()
