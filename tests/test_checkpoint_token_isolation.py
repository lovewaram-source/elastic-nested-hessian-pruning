import copy
import hashlib
import itertools
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_token_isolation_core import (
    RATES, collect_checkpoint_hessians, load_block_weights, select_block_private_tokens, select_all_mean_tokens,
)
import checkpoint_token_isolation_core as checkpoint_isolation_core
import token_isolation_core
import run_checkpoint_token_isolation as checkpoint_isolation_runner
from run_checkpoint_token_isolation import CheckpointCache, parse_args, prepare_references, reconstruct
from run_rate_aligned_nested import _find_layers
from experiment_common import input_groups


class CheckpointIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, folder, mode="2of4", device="cpu"):
        torch.manual_seed(31)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=40, intermediate_size=80,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, _attn_implementation="eager",
        )).eval()
        dense = copy.deepcopy(model)
        initial = torch.randn(2, 8, 40)
        positions = torch.arange(8).unsqueeze(0)
        attention = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min).triu(1)
        output = Path(folder) / "results"
        output.mkdir()
        args = SimpleNamespace(output=output, mode=mode, nsamples=2,
                               unstructured_alpha=1., structured_alpha=.25, percdamp=.01)
        cache = CheckpointCache(Path(folder) / "cache")
        model.to(device)
        prepare_references(args, model, initial, attention, positions, cache)
        return args, model, dense, initial, attention, positions, cache

    def test_global_mask_comes_from_block_representations(self):
        torch.manual_seed(2)
        paths = {rate: torch.randn(3, 8, 12) for rate in RATES}
        mask, stats = select_block_private_tokens(paths, torch.device("cpu"))
        scores = torch.stack([
            1 - F.cosine_similarity(paths[.2], paths[rate], dim=-1)
            for rate in RATES[1:]
        ]).amax(0).flatten()
        selected = scores.argsort(descending=True, stable=True)[:3]
        self.assertEqual(mask.flatten().nonzero().flatten().tolist(), selected.sort().values.tolist())
        self.assertEqual(stats["private_tokens"], 3)
        half_mask, half_stats = select_block_private_tokens(paths, torch.device("cpu"), .5)
        half_ids = scores.argsort(descending=True, stable=True)[:12]
        self.assertEqual(half_mask.flatten().nonzero().flatten().tolist(), half_ids.sort().values.tolist())
        self.assertEqual(half_stats["private_tokens"], 12)
        self.assertTrue(torch.all(half_mask[mask]))

    def test_six_rate_configuration_supports_selection_and_private_hessians(self):
        original_rates = checkpoint_isolation_runner.RATES
        rates = (.2, .3, .4, .5, .6, .7)
        try:
            checkpoint_isolation_runner.configure_rates(rates)
            torch.manual_seed(17)
            paths = {rate: torch.randn(1, 4, 6) + index * .1
                     for index, rate in enumerate(rates)}
            mask, stats = checkpoint_isolation_core.select_block_private_tokens(
                paths, torch.device("cpu"), .5,
            )
            self.assertEqual(tuple(mask.shape), (1, 4))
            self.assertEqual(stats["private_tokens"], 2)
            accumulator = token_isolation_core.TokenHessianAccumulator(4, .5)
            accumulator.add(paths)
            hessians, _ = accumulator.finish()
            self.assertEqual(tuple(hessians), rates)
            self.assertTrue(all(value.shape == (6, 6) for value in hessians.values()))

            with tempfile.TemporaryDirectory() as folder:
                args, model, _, initial, attention, positions, cache = self.fixture(
                    folder, mode="unstructured",
                )
                counts, total, _ = reconstruct(args, model, initial, attention, positions, cache)
                self.assertAlmostEqual(counts[.7] / total, .7, delta=1e-4)
        finally:
            checkpoint_isolation_runner.configure_rates(original_rates)

    def test_cli_accepts_70_percent_target_with_only_final_evaluation(self):
        args = parse_args([
            "--output", "unused-results", "--cache", "unused-cache",
            "--mode", "unstructured", "--rates", "0.2,0.3,0.4,0.5,0.6,0.7",
            "--eval-rates", "0.7", "--private-fraction", "0",
        ])
        self.assertEqual(args.rates, (.2, .3, .4, .5, .6, .7))
        self.assertEqual(args.eval_rates, (.7,))

    @torch.no_grad()
    def test_interleaved_forward_equals_four_independent_checkpoint_models(self):
        with tempfile.TemporaryDirectory() as folder:
            args, scratch, dense, initial, attention, positions, cache = self.fixture(folder)
            expected_outputs = {}
            expected_inputs = {}
            # Independent reference: run each complete sparse checkpoint from
            # first block to last before starting the next checkpoint.
            for rate in RATES:
                reference = copy.deepcopy(dense)
                x = initial.clone()
                for index, block in enumerate(reference.model.layers):
                    load_block_weights(block, cache.path("reference", index, rate))
                    modules = _find_layers(block)
                    captured = {group: [] for group in input_groups(modules)}

                    def hook_for(group):
                        def hook(_, inputs, output):
                            captured[group].append(inputs[0].detach().clone())
                        return hook

                    handles = [modules[group].register_forward_hook(hook_for(group))
                               for group in captured]
                    x = torch.cat([block(sample.unsqueeze(0), attention_mask=attention,
                                         position_ids=positions)[0] for sample in x])
                    for handle in handles:
                        handle.remove()
                    expected_outputs[rate, index] = x.clone()
                    for group, values in captured.items():
                        expected_inputs[rate, index, group] = torch.cat(values)
            paths = {rate: initial for rate in RATES}
            for index, block in enumerate(scratch.model.layers):
                mask, _ = select_block_private_tokens(paths, torch.device("cpu"))
                hessians, paths, trace = collect_checkpoint_hessians(
                    block, paths, mask,
                    {rate: cache.path("reference", index, rate) for rate in RATES},
                    attention, positions, torch.device("cpu"),
                )
                for rate in RATES:
                    self.assertEqual(trace["forward_counts"][str(rate)], 2)
                    self.assertAlmostEqual(trace["checkpoints"][str(rate)]["actual_zero_fraction"], rate)
                    torch.testing.assert_close(paths[rate], expected_outputs[rate, index], rtol=0, atol=0)
                    for group in hessians:
                        merged = expected_inputs[.2, index, group].clone()
                        merged[mask] = expected_inputs[rate, index, group][mask]
                        flat = merged.flatten(0, 1)
                        expected_hessian = (2 / args.nsamples) * flat.t() @ flat
                        torch.testing.assert_close(hessians[group][rate], expected_hessian,
                                                   rtol=1e-5, atol=1e-5)
                self.assertFalse(torch.equal(paths[.2], paths[.5]))

    @torch.no_grad()
    def test_mean_token_hessian_matches_average_activations_not_average_grams(self):
        for fraction in (.1, .5):
            with self.subTest(fraction=fraction), tempfile.TemporaryDirectory() as folder:
                args, scratch, dense, initial, attention, positions, cache = self.fixture(folder)
                paths = {rate: initial for rate in RATES}
                expected_inputs = {}
                expected_outputs = {}
                for index, layer in enumerate(scratch.model.layers):
                    original_paths = {rate: value.clone() for rate, value in paths.items()}
                    mask, _ = select_block_private_tokens(paths, torch.device("cpu"), fraction)
                    for rate in RATES:
                        block = copy.deepcopy(dense.model.layers[index])
                        load_block_weights(block, cache.source(index, rate))
                        captured = {group: [] for group in input_groups(_find_layers(block))}

                        def hook_for(group):
                            def hook(_, inputs, output):
                                captured[group].append(inputs[0].detach().clone())
                            return hook

                        handles = [_find_layers(block)[group].register_forward_hook(hook_for(group))
                                   for group in captured]
                        expected_outputs[rate] = torch.cat([
                            block(value.unsqueeze(0), attention_mask=attention, position_ids=positions)[0]
                            for value in original_paths[rate]
                        ])
                        for handle in handles:
                            handle.remove()
                        for group in captured:
                            expected_inputs[rate, group] = torch.cat(captured[group])
                    hessians, paths, trace = collect_checkpoint_hessians(
                        layer, paths, mask, {r: cache.source(index, r) for r in RATES},
                        attention, positions, torch.device("cpu"), shared_fusion="mean_token",
                    )
                    self.assertEqual(trace["shared_fusion"], "mean_token")
                    for group in hessians:
                        mean = torch.stack([expected_inputs[r, group].float() for r in RATES]).mean(0)
                        for rate in RATES:
                            merged = mean.clone()
                            merged[mask] = expected_inputs[rate, group][mask]
                            flat = merged.flatten(0, 1)
                            expected = (2 / args.nsamples) * flat.t() @ flat
                            torch.testing.assert_close(hessians[group][rate], expected, rtol=1e-5, atol=1e-5)
                    for rate in RATES:
                        self.assertEqual(trace["forward_counts"][str(rate)], args.nsamples)
                        torch.testing.assert_close(paths[rate], expected_outputs[rate], rtol=0, atol=0)
                    if index == 1:
                        group = "self_attn.q_proj"
                        average_gram = sum(
                            expected_inputs[r, group][~mask].t() @ expected_inputs[r, group][~mask]
                            for r in RATES
                        ) / 4
                        mean_gram = torch.stack([
                            expected_inputs[r, group][~mask] for r in RATES
                        ]).mean(0)
                        mean_gram = mean_gram.t() @ mean_gram
                        self.assertGreater(float((average_gram - mean_gram).abs().max()), 1e-8)

    @torch.no_grad()
    def test_isolated_positions_use_mean_and_complement_uses_anchor20(self):
        for fraction in (.1, .5, 1.):
            with self.subTest(fraction=fraction), tempfile.TemporaryDirectory() as folder:
                args, scratch, dense, initial, attention, positions, cache = self.fixture(folder)
                paths = {rate: initial for rate in RATES}
                for index, layer in enumerate(scratch.model.layers):
                    mask, _ = (select_all_mean_tokens(paths) if fraction == 1.
                               else select_block_private_tokens(paths, torch.device("cpu"), fraction))
                    captured = {}
                    expected_outputs = {}
                    for rate in RATES:
                        block = copy.deepcopy(dense.model.layers[index])
                        load_block_weights(block, cache.source(index, rate))
                        buffers = {group: [] for group in input_groups(_find_layers(block))}

                        def hook_for(group):
                            def hook(_, inputs, output):
                                buffers[group].append(inputs[0].detach().clone())
                            return hook

                        handles = [_find_layers(block)[g].register_forward_hook(hook_for(g)) for g in buffers]
                        expected_outputs[rate] = torch.cat([
                            block(x.unsqueeze(0), attention_mask=attention, position_ids=positions)[0]
                            for x in paths[rate]
                        ])
                        for handle in handles:
                            handle.remove()
                        for group in buffers:
                            captured[rate, group] = torch.cat(buffers[group])
                    hessians, paths, trace = collect_checkpoint_hessians(
                        layer, paths, mask, {r: cache.source(index, r) for r in RATES},
                        attention, positions, torch.device("cpu"), shared_fusion="isolated_mean_anchor20",
                    )
                    self.assertEqual(trace["averaged_position_set"], "isolated")
                    self.assertTrue(trace["all_rates_same_hessian"])
                    for group, branches in hessians.items():
                        mean = torch.stack([captured[r, group] for r in RATES]).mean(0)
                        mixed = captured[.2, group].clone()
                        mixed[mask] = mean[mask]
                        flat = mixed.flatten(0, 1)
                        expected = (2 / args.nsamples) * flat.t() @ flat
                        for rate in RATES:
                            self.assertIs(branches[rate], branches[.2])
                            torch.testing.assert_close(branches[rate], expected, rtol=1e-5, atol=1e-5)
                    for rate in RATES:
                        torch.testing.assert_close(paths[rate], expected_outputs[rate], rtol=0, atol=0)

    @torch.no_grad()
    def test_two_structures_reconstruct_shared_prefix_without_mutating_references(self):
        for mode in ("unstructured", "2of4"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                args, model, _, initial, attention, positions, cache = self.fixture(folder, mode)
                before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in cache.reference.iterdir()}
                counts, total, _ = reconstruct(args, model, initial, attention, positions, cache)
                after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in cache.reference.iterdir()}
                self.assertEqual(before, after)
                for rate in RATES:
                    self.assertAlmostEqual(counts[rate] / total, rate)
                for index, block in enumerate(model.model.layers):
                    prior = None
                    prior_rate = 0.
                    for rate in RATES:
                        state = (torch.load(cache.path("target", index, rate), weights_only=True)
                                 if rate != .5 else {name: module.weight.detach()
                                                    for name, module in _find_layers(block).items()})
                        if prior is not None:
                            for name, weight in state.items():
                                frozen = round(weight.shape[1] * prior_rate / .5)
                                torch.testing.assert_close(weight[:, :frozen], prior[name][:, :frozen],
                                                           rtol=0, atol=0)
                        prior, prior_rate = state, rate

    def test_cli_rejects_cross_gpu_execution(self):
        argv = ["--mode", "2of4", "--output", "unused-results", "--cache", "unused-cache"]
        args = parse_args(argv)
        self.assertEqual(args.gpu, "0")
        self.assertEqual(args.device_map, "cpu")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_args(argv + ["--gpu", "0,1"])
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_args(argv + ["--device-map", "balanced"])

    @torch.no_grad()
    def test_streamed_references_match_disk_pipeline_exactly(self):
        devices = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
        for mode, device, fraction, fusion in itertools.product(
                ("unstructured", "2of4"), devices, (.1, .5), ("anchor20", "mean_token", "isolated_mean_anchor20")):
            with self.subTest(mode=mode, device=device, fraction=fraction, fusion=fusion), tempfile.TemporaryDirectory() as folder:
                args, baseline, dense, initial, attention, positions, disk = self.fixture(folder, mode, device)
                args.private_fraction = fraction
                args.shared_fusion = fusion
                baseline_hessians = []
                collector = collect_checkpoint_hessians

                def capture(*values, **kwargs):
                    hessians, outputs, trace = collector(*values, **kwargs)
                    baseline_hessians.append({
                        group: {rate: value.clone() for rate, value in branches.items()}
                        for group, branches in hessians.items()
                    })
                    return hessians, outputs, trace

                with patch("run_checkpoint_token_isolation.collect_checkpoint_hessians", capture):
                    expected_counts, expected_total, _ = reconstruct(
                        args, baseline, initial, attention, positions, disk,
                    )
                streaming_args = copy.copy(args)
                streaming_args.output = Path(folder) / "streaming-results"
                streaming_args.output.mkdir()
                memory = CheckpointCache(Path(folder) / "streaming-cache", memory_reference=True)
                saved = []
                original_save = memory.save

                def checked_save(kind, index, stage, layer):
                    original_save(kind, index, stage, layer)
                    if kind == "reference":
                        expected = torch.load(disk.path(kind, index, stage), weights_only=True)
                        for name, value in memory.source(index, stage).items():
                            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                        saved.append((index, stage))
                        self.assertLessEqual(len(memory.reference_states), 5)

                memory.save = checked_save
                layer_index = 0

                def compare(*values, **kwargs):
                    nonlocal layer_index
                    hessians, outputs, trace = collector(*values, **kwargs)
                    for group, branches in hessians.items():
                        for rate, value in branches.items():
                            torch.testing.assert_close(value, baseline_hessians[layer_index][group][rate],
                                                       rtol=0, atol=0)
                    layer_index += 1
                    return hessians, outputs, trace

                with patch("run_checkpoint_token_isolation.collect_checkpoint_hessians", compare):
                    counts, total, timing = reconstruct(
                        streaming_args, dense, initial, attention, positions, memory,
                        stream_reference=True, compute_device=torch.device(device),
                    )
                self.assertEqual(counts, expected_counts)
                self.assertEqual(total, expected_total)
                self.assertGreater(timing["reference_preparation_seconds"], 0)
                self.assertEqual(len(saved), 10)
                self.assertEqual(memory.reference_states, {})
                self.assertEqual(list(memory.reference.iterdir()), [])
                for index, layer in enumerate(dense.model.layers):
                    before_mask = torch.load(args.output / f"private_positions_layer_{index:03d}.pt",
                                             weights_only=True)
                    after_mask = torch.load(streaming_args.output / f"private_positions_layer_{index:03d}.pt",
                                            weights_only=True)
                    self.assertTrue(torch.equal(before_mask["mask"], after_mask["mask"]))
                    self.assertEqual(before_mask["statistics"], after_mask["statistics"])
                    for rate in RATES:
                        expected = (torch.load(disk.path("target", index, rate), weights_only=True)
                                    if rate != .5 else {name: mod.weight for name, mod in
                                                       _find_layers(baseline.model.layers[index]).items()})
                        actual = (torch.load(memory.path("target", index, rate), weights_only=True)
                                  if rate != .5 else {name: mod.weight for name, mod in _find_layers(layer).items()})
                        for name in actual:
                            torch.testing.assert_close(actual[name].cpu(), expected[name].cpu(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires one visible CUDA GPU")
    @torch.no_grad()
    def test_single_gpu_runner_capture_reconstruction_and_all_ppl(self):
        import json
        from run_checkpoint_token_isolation import run
        from experiment_common import _capture_first_layer_inputs

        self.assertEqual(torch.cuda.device_count(), 1, "run GPU smoke tests with one visible GPU")
        for mode, fraction, fusion in itertools.product(
                ("unstructured", "2of4"), (.1, .5), ("anchor20", "mean_token", "isolated_mean_anchor20")):
            with self.subTest(mode=mode, fraction=fraction, fusion=fusion), tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(71)
                model = LlamaForCausalLM(LlamaConfig(
                    vocab_size=64, hidden_size=40, intermediate_size=80,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                    max_position_embeddings=32, _attn_implementation="sdpa",
                    use_cache=False,
                )).half().eval()
                model.seqlen = 8
                model.hf_device_map = {"": "cpu"}
                failure_model = copy.deepcopy(model)
                tokens = torch.randint(0, 64, (1, 16))
                args = parse_args(["--mode", mode, "--nsamples", "2", "--seqlen", "8",
                                   "--private-fraction", str(fraction),
                                   "--shared-fusion", fusion,
                                   "--output", str(Path(folder) / "results"),
                                   "--cache", str(Path(folder) / "cache")])

                def capture(protocol, loaded, tokenizer, device):
                    self.assertEqual(loaded.model.embed_tokens.weight.device.type, "cuda")
                    self.assertTrue(all(next(layer.parameters()).device.type == "cpu"
                                        for layer in loaded.model.layers))
                    loader = [(tokens[:, :8],), (tokens[:, 8:],)]
                    return _capture_first_layer_inputs(loaded, loader, 2, 8, device)

                with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                        patch("run_checkpoint_token_isolation._load_model", return_value=model), \
                        patch("run_checkpoint_token_isolation._load_eval_loader",
                              return_value=SimpleNamespace(input_ids=tokens)), \
                        patch("run_checkpoint_token_isolation._load_calibration_inputs", capture):
                    run(args)
                rows = json.loads((args.output / "results.json").read_text())
                self.assertTrue(all(row["private_fraction"] == fraction for row in rows))
                self.assertTrue(all(row["shared_fusion"] == fusion for row in rows))
                self.assertTrue(all(row["hessian_policy"].endswith(f"_{round(fraction * 100):03d}") for row in rows))
                self.assertEqual({row["sparsity"] for row in rows}, {"0.20", "0.30", "0.40", "0.50"})
                self.assertTrue(all(float(row["ppl_wikitext2"]) > 0 for row in rows))
                timing = json.loads((args.output / "timings.json").read_text())
                self.assertGreater(timing["total_wall_excluding_ppl_seconds"], 0)
                self.assertEqual(list((args.cache / "reference").iterdir()), [])
                self.assertEqual(list((args.cache / "target").iterdir()), [])

                # A failed PPL evaluation must not discard that rate's snapshots.
                failed_args = copy.copy(args)
                failed_args.output = Path(folder) / "failed-results"
                failed_args.cache = Path(folder) / "failed-cache"
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                        patch("run_checkpoint_token_isolation._load_model", return_value=failure_model), \
                        patch("run_checkpoint_token_isolation._load_eval_loader",
                              return_value=SimpleNamespace(input_ids=tokens)), \
                        patch("run_checkpoint_token_isolation._load_calibration_inputs", capture), \
                        patch("run_checkpoint_token_isolation._eval_ppl",
                              side_effect=[1., RuntimeError("injected evaluation failure")]):
                    with self.assertRaisesRegex(RuntimeError, "injected evaluation failure"):
                        run(failed_args)
                self.assertEqual(len(list((failed_args.cache / "target").glob("*_s20.pt"))), 2)


if __name__ == "__main__":
    unittest.main()
