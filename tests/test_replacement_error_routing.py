import json
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_token_isolation_core import RATES, select_replacement_error_tokens, select_all_mean_tokens
from replacement_error_stats import describe_errors, summarize_layers
from run_checkpoint_token_isolation import parse_args, reconstruct, CheckpointCache
import test_checkpoint_token_isolation as checkpoint_tests


class ReplacementRoutingTests(unittest.TestCase):
    def test_pure_mean_routes_every_position_without_any_threshold(self):
        paths = {r: torch.randn(2, 8, 40) for r in RATES}
        mask, stats = select_all_mean_tokens(paths)
        self.assertTrue(bool(mask.all()))
        self.assertEqual(stats["shared_tokens"], 0)
        self.assertEqual(stats["private_tokens"], 16)
        self.assertEqual(stats["rule"], "all_mean")

    def test_threshold_uses_branch_norm_and_worst_of_three(self):
        anchor = torch.tensor([[[1., 0.], [1., 0.], [1., 0.], [1., 0.]]])
        paths = {rate: anchor.clone() for rate in RATES}
        paths[.3][0, 1, 0] = 2.  # direction identical, replacement error50%
        paths[.5][0, 2, 0] = 1.04  # error about3.85%, safe
        paths[.4][0, 3, 0] = 0.  # undefined relative error, do not share
        mask, stats, errors, valid = select_replacement_error_tokens(paths, torch.device("cpu"), .05)
        self.assertEqual(mask.tolist(), [[False, True, False, True]])
        self.assertEqual(stats["shared_tokens"], 2)
        self.assertEqual(stats["invalid_norm_tokens"], 1)
        self.assertAlmostEqual(float(errors[0, 1]), .5)
        self.assertFalse(bool(valid[0, 3]))
        boundary = {r: torch.tensor([[[19., 0.]]]) for r in RATES}
        boundary[.5] = torch.tensor([[[20., 0.]]])
        exact_mask, _, _, _ = select_replacement_error_tokens(boundary, torch.device("cpu"), .05)
        self.assertFalse(bool(exact_mask.any()), "exactly5% must share")
        boundary10 = {r: torch.tensor([[[9., 0.]]]) for r in RATES}
        boundary10[.5] = torch.tensor([[[10., 0.]]])
        strict, _, _, _ = select_replacement_error_tokens(boundary10, torch.device("cpu"), .05)
        relaxed, _, _, _ = select_replacement_error_tokens(boundary10, torch.device("cpu"), .1)
        self.assertTrue(bool(strict.any()))
        self.assertFalse(bool(relaxed.any()), "exactly10% must share at10% threshold")
        with self.assertRaises(ValueError):
            select_replacement_error_tokens(paths, torch.device("cpu"), -.1)

    def test_cdf_bins_quantiles_and_excluding_embedding(self):
        errors = torch.tensor([0., .05, .10, .15, .6, 1.5, 3., 0.])
        valid = torch.tensor([True] * 7 + [False])
        stats = describe_errors(errors, valid)
        cdf = {x["upper_error_percent"]: x["count"] for x in stats["cdf"]}
        self.assertEqual(cdf[5], 2)
        self.assertEqual(cdf[10], 3)
        self.assertEqual(cdf[15], 4)
        self.assertEqual(sum(x["count"] for x in stats["bins"]) + stats["invalid_norm_tokens"], 8)
        self.assertEqual(stats["quantiles_valid_only"]["max"], 3.)
        summary = summarize_layers([(torch.zeros(4), torch.ones(4, dtype=torch.bool)), (errors, valid)], .05)
        self.assertAlmostEqual(summary["all_layer_inputs"]["anchor_share_fraction"], 6 / 12)
        self.assertAlmostEqual(summary["excluding_embedding_input"]["anchor_share_fraction"], 2 / 8)
        summary10 = summarize_layers([(torch.zeros(4), torch.ones(4, dtype=torch.bool)), (errors, valid)], .1)
        self.assertAlmostEqual(summary10["excluding_embedding_input"]["anchor_share_fraction"], 3 / 8)

    @torch.no_grad()
    def test_threshold_runner_writes_actual_coverage_not_fixed_fraction(self):
        for mode, threshold in itertools.product(("unstructured", "2of4"), (.05, .1)):
            with self.subTest(mode=mode, threshold=threshold), tempfile.TemporaryDirectory() as folder:
                fixture = checkpoint_tests.CheckpointIsolationTests()
                args, _, dense, initial, attention, positions, _ = fixture.fixture(folder, mode)
                args.shared_fusion = "isolated_mean_anchor20"
                args.selection_rule = "relative_error"
                args.share_error_threshold = threshold
                args.save_final_target = True
                args.output = Path(folder) / "threshold-results"
                args.output.mkdir()
                cache = CheckpointCache(Path(folder) / "threshold-cache", memory_reference=True)
                counts, total, _ = reconstruct(args, dense, initial, attention, positions, cache,
                                               stream_reference=True, compute_device=torch.device("cpu"))
                summary = json.loads((args.output / "replacement_error_summary.json").read_text())
                self.assertEqual(summary["layer_count"], 2)
                first = torch.load(args.output / "private_positions_layer_000.pt", weights_only=True)
                self.assertEqual(first["statistics"]["shared_tokens"], 16)
                self.assertEqual(first["statistics"]["private_tokens"], 0)
                self.assertFalse(bool(first["mask"].any()))
                self.assertEqual(len(list(cache.target.glob("*_s50.pt"))), 2)
                self.assertTrue((args.output / "replacement_error_layer_001.pt").exists())
                for rate in RATES:
                    self.assertAlmostEqual(counts[rate] / total, rate)
        args = parse_args(["--mode", "2of4", "--output", "unused", "--cache", "unused-cache",
                           "--selection-rule", "relative_error", "--share-error-threshold", ".05",
                           "--shared-fusion", "isolated_mean_anchor20"])
        self.assertEqual(args.selection_rule, "relative_error")

    @unittest.skipUnless(torch.cuda.is_available(), "requires a single visible CUDA GPU")
    @torch.no_grad()
    def test_threshold_full_runner_on_single_gpu(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from run_checkpoint_token_isolation import run
        from experiment_common import _capture_first_layer_inputs
        for mode, threshold in itertools.product(("unstructured", "2of4"), (.05, .1, None)):
            with self.subTest(mode=mode, threshold=threshold), tempfile.TemporaryDirectory() as folder:
                model = LlamaForCausalLM(LlamaConfig(
                    vocab_size=64, hidden_size=40, intermediate_size=80,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                    max_position_embeddings=32, _attn_implementation="sdpa", use_cache=False,
                )).half().eval()
                model.seqlen = 8
                model.hf_device_map = {"": "cpu"}
                tokens = torch.randint(0, 64, (1, 16))
                args = parse_args(["--mode", mode, "--nsamples", "2", "--seqlen", "8",
                                   "--selection-rule", "all_mean" if threshold is None else "relative_error",
                                   "--share-error-threshold", str(threshold if threshold is not None else .05),
                                   "--shared-fusion", "isolated_mean_anchor20", "--save-final-target",
                                   "--output", str(Path(folder) / "results"),
                                   "--cache", str(Path(folder) / "cache")])

                def capture(protocol, loaded, tokenizer, device):
                    return _capture_first_layer_inputs(loaded, [(tokens[:, :8],), (tokens[:, 8:],)], 2, 8, device)

                with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                        patch("run_checkpoint_token_isolation._load_model", return_value=model), \
                        patch("run_checkpoint_token_isolation._load_eval_loader", return_value=SimpleNamespace(input_ids=tokens)), \
                        patch("run_checkpoint_token_isolation._load_calibration_inputs", capture):
                    run(args)
                rows = json.loads((args.output / "results.json").read_text())
                self.assertEqual(len(rows), 4)
                if threshold is None:
                    self.assertTrue(all(r["private_fraction"] == 1. and r["selection_rule"] == "all_mean"
                                        and r["share_error_threshold"] is None
                                        and r["anchor_share_fraction_all_inputs"] == 0 for r in rows))
                    self.assertFalse((args.output / "replacement_error_summary.json").exists())
                    for index in range(2):
                        saved = torch.load(args.output / f"private_positions_layer_{index:03d}.pt", weights_only=True)
                        self.assertTrue(bool(saved["mask"].all()))
                else:
                    self.assertTrue(all(r["private_fraction"] is None and r["selection_rule"] == "relative_error"
                                        and r["share_error_threshold"] == threshold for r in rows))
                    summary = json.loads((args.output / "replacement_error_summary.json").read_text())
                    self.assertEqual(summary["layer_count"], 2)
                self.assertEqual(list((args.cache / "target").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
