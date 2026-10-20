"""Exact mean-Gram semantics, actual checkpoint forwards, and runner coverage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_token_isolation_core import MeanHessian, RATES, collect_checkpoint_hessians, select_all_mean_tokens
from run_checkpoint_token_isolation import parse_args, run
import test_checkpoint_token_isolation as checkpoint_tests


class MeanHessianTests(unittest.TestCase):
    @torch.no_grad()
    def test_average_grams_and_variance_identity(self):
        torch.manual_seed(31)
        values = {r: torch.randn(2, 8, 7) + i for i, r in enumerate(RATES)}
        acc = MeanHessian(7, "cpu", 2)
        ids = torch.arange(8)
        for r in RATES:
            for s in range(2):
                acc.add(r, values[r][s:s+1], ids, ids[:0])
        result = acc.finish()
        expected = sum(x.flatten(0, 1).t() @ x.flatten(0, 1) for x in values.values()) / 4
        mean = sum(values.values()) / 4
        flat = mean.flatten(0, 1)
        variance = sum((x-mean).flatten(0, 1).t() @ (x-mean).flatten(0, 1)
                       for x in values.values()) / 4
        torch.testing.assert_close(result[.2], expected)
        torch.testing.assert_close(result[.2] - flat.t() @ flat, variance)
        self.assertGreater(float(variance.norm()), 1.)
        self.assertTrue(all(result[r] is result[.2] for r in RATES))
        incomplete = MeanHessian(7, "cpu", 2)
        with self.assertRaises(AssertionError):
            incomplete.finish()

    @torch.no_grad()
    def test_collector_matches_individual_hessians_and_actual_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            args, scratch, dense, initial, attention, positions, cache = checkpoint_tests.CheckpointIsolationTests().fixture(folder)
            paths = {r: initial for r in RATES}
            independent_paths = {r: initial for r in RATES}
            for index, layer in enumerate(scratch.model.layers):
                mask, _ = select_all_mean_tokens(paths)
                expected, independent_paths, _ = collect_checkpoint_hessians(
                    copy.deepcopy(dense.model.layers[index]), independent_paths, mask,
                    {r: cache.source(index, r) for r in RATES}, attention, positions,
                    torch.device("cpu"), shared_fusion="anchor20")
                actual, paths, trace = collect_checkpoint_hessians(
                    layer, paths, mask, {r: cache.source(index, r) for r in RATES},
                    attention, positions, torch.device("cpu"), shared_fusion="mean_hessian")
                for group in actual:
                    average = sum(expected[group].values()) / 4
                    for r in RATES:
                        torch.testing.assert_close(actual[group][r], average, rtol=1e-5, atol=1e-5)
                        self.assertIs(actual[group][r], actual[group][.2])
                for r in RATES:
                    torch.testing.assert_close(paths[r], independent_paths[r], rtol=0, atol=0)
                    self.assertEqual(trace["forward_counts"][str(r)], 2)
                self.assertEqual(trace["hessian_buffers_per_input_group"], 1)
                self.assertTrue(trace["all_rates_same_hessian"])

    def test_cli_rejects_partial_routing(self):
        base = ["--mode", "2of4", "--output", "unused", "--cache", "unused-cache",
                "--shared-fusion", "mean_hessian"]
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_args(base)
        self.assertEqual(parse_args(base + ["--selection-rule", "all_mean"]).shared_fusion, "mean_hessian")

    @unittest.skipUnless(torch.cuda.is_available(), "requires one visible CUDA GPU")
    @torch.no_grad()
    def test_full_single_gpu_runner(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from experiment_common import _capture_first_layer_inputs
        for mode in ("2of4", "unstructured"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                model = LlamaForCausalLM(LlamaConfig(
                    vocab_size=64, hidden_size=40, intermediate_size=80, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
                    _attn_implementation="sdpa", use_cache=False)).half().eval()
                model.seqlen = 8
                model.hf_device_map = {"": "cpu"}
                tokens = torch.randint(0, 64, (1, 16))
                args = parse_args(["--mode", mode, "--nsamples", "2", "--seqlen", "8",
                                   "--selection-rule", "all_mean", "--shared-fusion", "mean_hessian",
                                   "--save-final-target", "--output", str(Path(folder)/"results"),
                                   "--cache", str(Path(folder)/"cache")])

                def capture(protocol, loaded, tokenizer, device):
                    return _capture_first_layer_inputs(loaded, [(tokens[:,:8],), (tokens[:,8:],)], 2, 8, device)

                with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                        patch("run_checkpoint_token_isolation._load_model", return_value=model), \
                        patch("run_checkpoint_token_isolation._load_eval_loader", return_value=SimpleNamespace(input_ids=tokens)), \
                        patch("run_checkpoint_token_isolation._load_calibration_inputs", capture):
                    run(args)
                rows = json.loads((args.output/"results.json").read_text())
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(r["averaging_space"] == "hessian" and r["status"] == "completed"
                                    and r["nested_mask_violations"] == 0 for r in rows))
                self.assertTrue(all(r["hessian_policy"] == "true_checkpoint_all_mean_mean_hessian" for r in rows))
                self.assertEqual(list((args.cache/"target").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
