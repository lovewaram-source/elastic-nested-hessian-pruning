"""Small-model sorting pipeline: budgets, actual permutations and nested zeros."""
import argparse
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sorting_nested import parse_args, reconstruct_sorted, GROUP_ORDERINGS, COLUMN_ORDERINGS
import run_sorting_nested
from run_checkpoint_token_isolation import CheckpointCache
from checkpoint_token_isolation_core import load_block_weights
from run_rate_aligned_nested import _find_layers


class SortingRunnerTests(unittest.TestCase):
    def test_cli_rejects_incompatible_modes(self):
        base = ["--model", "unused", "--output", "unused", "--cache", "unused-cache"]
        for mode, ordering in (("2of4", "wanda"), ("unstructured", GROUP_ORDERINGS[0])):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + ["--mode", mode, "--ordering", ordering])

    def test_sorted_reconstruction_all_modes_on_tiny_llama(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        original = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=40, intermediate_size=80,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, _attn_implementation="eager")).eval()
        initial = torch.randn(2, 8, 40)
        positions = torch.arange(8).unsqueeze(0)
        attention = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min).triu(1)
        for mode, orderings in (("2of4", ("original",) + GROUP_ORDERINGS),
                                ("unstructured", ("original",) + COLUMN_ORDERINGS)):
            for ordering in orderings:
                with self.subTest(mode=mode, ordering=ordering), tempfile.TemporaryDirectory() as folder:
                    model = copy.deepcopy(original)
                    args = SimpleNamespace(output=Path(folder), mode=mode, ordering=ordering,
                                           percdamp=.01, structured_alpha=.25, unstructured_alpha=1.)
                    cache = CheckpointCache(Path(folder) / "cache")
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    counts, total = reconstruct_sorted(args, model, initial, attention, positions,
                                                       cache, device)
                    previous_zeros = {}
                    # Read checkpoints in ascending order; final layer weights already represent 50%.
                    final = copy.deepcopy(model)
                    for rate in (.2, .3, .4, .5):
                        if rate == .5:
                            model = final
                        else:
                            for i, layer in enumerate(model.model.layers):
                                load_block_weights(layer, cache.path("target", i, rate))
                        self.assertAlmostEqual(counts[rate] / total, rate, places=6)
                        for i, layer in enumerate(model.model.layers):
                            for name, module in _find_layers(layer).items():
                                zeros = module.weight.eq(0)
                                if (i, name) in previous_zeros:
                                    self.assertTrue(torch.all(zeros[previous_zeros[i, name]]))
                                previous_zeros[i, name] = zeros.clone()
                        self.assertTrue((Path(folder) / "permutations_layer_000.pt").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "full runner requires CUDA")
    def test_full_cuda_runner_writes_all_debug_results(self):
        torch.set_num_threads(2)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=40, intermediate_size=80,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, _attn_implementation="eager")).eval()
        model.seqlen = 8
        initial = torch.randn(2, 8, 40)
        attention = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min).triu(1)
        positions = torch.arange(8).unsqueeze(0)
        for mode, ordering in (("2of4", "wanda_group_pruned_part_stage_desc"),
                                ("unstructured", "wanda")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                out, cache = Path(folder) / "output", Path(folder) / "cache"
                args = parse_args(["--model", "unused", "--mode", mode, "--ordering", ordering,
                                   "--output", str(out), "--cache", str(cache),
                                   "--nsamples", "2", "--seqlen", "8", "--eval-samples", "2"])
                evaluation = SimpleNamespace(input_ids=torch.randint(0, 64, (1, 16)))
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                     patch("run_rate_aligned_nested._load_model", return_value=copy.deepcopy(model)), \
                     patch("run_rate_aligned_nested._load_eval_loader", return_value=evaluation), \
                     patch("run_rate_aligned_nested._load_calibration_inputs",
                           return_value=(initial, attention, positions)):
                    run_sorting_nested.run(args)
                rows = json.loads((out / "results.json").read_text())
                self.assertEqual([r["sparsity"] for r in rows], [.5, .4, .3, .2])
                self.assertTrue(all(r["evaluation_scope"] == "truncated_debug" for r in rows))
                self.assertTrue(all(r["status"] == "completed" for r in rows))
                self.assertFalse(list((cache / "target").glob("*.pt")))
                protocol = json.loads((out / "protocol.json").read_text())
                sources = protocol["runtime"]["public_python_source_sha256"]
                self.assertIn("lib/data.py", sources)
                self.assertIn("scripts/run_rate_aligned_nested.py", sources)


if __name__ == "__main__":
    unittest.main()
