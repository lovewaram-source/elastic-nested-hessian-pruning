"""Engineering checks, not experimental accuracy evidence."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compensation_conflict import effective_state, shadow_window, ConflictObserver, same_bits
from rate_aligned_src.elastic_sparsegpt import apply_elastic_2of4_stage
from token_isolation_core import prepare_device_plan


class ConflictTests(unittest.TestCase):
    def test_module_selection_and_duplicate_rejection(self):
        import contextlib
        import io
        from run_compensation_conflict import parse_args, requested_groups
        names = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
        self.assertEqual(requested_groups(names, ["self_attn.k_proj", "self_attn.v_proj"]),
                         {"self_attn.q_proj": ("self_attn.k_proj", "self_attn.v_proj")})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--output", "unused", "--cache", "unused-cache", "--modules",
                        "mlp.up_proj", "mlp.up_proj"])

    def test_bitwise_gate_distinguishes_signed_zero(self):
        self.assertFalse(same_bits(torch.tensor([0.]), torch.tensor([-0.])))

    def test_opposed_curvature_causes_cross_branch_harm(self):
        state = torch.zeros(1, 20)
        state[0, 0] = 1
        positive = torch.eye(20)
        negative = torch.eye(20)
        positive[0, 4] = positive[4, 0] = .8
        negative[0, 4] = negative[4, 0] = -.8
        mask = torch.tensor([[True, True, False, False]])
        a = prepare_device_plan(positive, .01)
        b = prepare_device_plan(negative, .01)
        no_comp, _, _ = shadow_window(state, a.hessian_factor, mask, 0, 4, 0., 1)
        own, _, _ = shadow_window(state, a.hessian_factor, mask, 0, 4, .25, 1)
        opposite, _, _ = shadow_window(state, b.hessian_factor, mask, 0, 4, .25, 1)
        x = torch.linalg.cholesky(positive)
        loss = lambda w: ((w-state) @ x).square().sum()
        self.assertLess(loss(own), loss(no_comp))
        self.assertGreater(loss(opposite), loss(no_comp))

    def test_heldout_loader_uses_validation_and_skips_exact_duplicates(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_compensation_conflict import load_inputs
        loader = [(torch.tensor([[1, 2]]), None)]
        validation = SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]))
        args = SimpleNamespace(nsamples=1, seed=0, seqlen=2, heldout_samples=1)
        with patch("rate_aligned_src.data.get_loaders", return_value=(loader, validation)), patch(
                "experiment_common._capture_first_layer_inputs",
                return_value=(None, None, None)) as capture:
            *_, manifest = load_inputs(args, None, None, "cpu")
        self.assertEqual(manifest["heldout_validation_windows"], [1])
        self.assertFalse(set(manifest["train_token_sha256"]) & set(manifest["heldout_token_sha256"]))
        self.assertTrue(torch.equal(capture.call_args.args[1][1][0], torch.tensor([[3, 4]])))

    def test_lazy_suffix_and_frozen_prefix(self):
        w = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        block = w[:, :8].clone() + 5
        errors = torch.ones(2, 8)
        factor = torch.triu(torch.ones(12, 12))
        quantized = block.clone() - 3
        state = effective_state(dict(ordered_weight=w, block_weight=block,
            quantized=quantized, errors=errors, hessian_factor=factor,
            block_start=0, block_end=8, column=4), torch.arange(2))
        torch.testing.assert_close(state[:, :4], quantized[:, :4])
        torch.testing.assert_close(state[:, 4:8], block[:, 4:8])
        torch.testing.assert_close(state[:, 8:], w[:, 8:] - 4)
        result, _, _ = shadow_window(state, factor, torch.ones(2, 4, dtype=torch.bool),
                                      4, 8, .25, 1)
        torch.testing.assert_close(result[:, :4], state[:, :4], rtol=0, atol=0)
        self.assertTrue(result[:, 4:8].eq(0).all())

    def test_shadow_single_block_matches_real_kernel(self):
        torch.manual_seed(2)
        layer = torch.nn.Linear(20, 3, bias=False)
        x = torch.randn(20, 30)
        plan = prepare_device_plan(x @ x.t(), .01)
        initial = layer.weight.detach().float().clone()
        masks = torch.zeros_like(initial, dtype=torch.bool)
        masks[:, 8:10] = True
        shadow, _, _ = shadow_window(initial, plan.hessian_factor, masks[:, 8:12],
                                      8, 12, .25, 1)
        apply_elastic_2of4_stage(layer, plan, .3, previous_sparsity=.2,
            compensation_alpha=.25, precomputed_mask=masks)
        torch.testing.assert_close(shadow, layer.weight, rtol=0, atol=0)

    def test_observer_leaves_complete_trajectory_bitwise_unchanged(self):
        torch.manual_seed(4)
        layer = torch.nn.Linear(40, 5, bias=False)
        control = copy.deepcopy(layer)
        x = torch.randn(40, 64)
        plans = {r: prepare_device_plan(x @ x.t(), .01) for r in (.2, .3, .4, .5)}
        probes = {split: {r: x[:, :16] for r in plans} for split in ("train", "heldout")}
        events = []
        observer = ConflictObserver(plans, probes, layer.weight.detach().clone(),
                                    events.append, rows=5, window_groups=2)
        previous = 0.
        for stage in plans:
            observer.stage = stage
            expected = apply_elastic_2of4_stage(control, plans[stage], stage,
                previous_sparsity=previous, compensation_alpha=.25, return_mask=True)
            actual = apply_elastic_2of4_stage(layer, plans[stage], stage,
                previous_sparsity=previous, compensation_alpha=.25, return_mask=True,
                group_observer=observer)
            self.assertTrue(same_bits(layer.weight, control.weight))
            self.assertTrue(torch.equal(actual, expected))
            previous = stage
        self.assertEqual(len(events), 9)  # three positions in first stage, two in later stages
        for event in events:
            for horizon in event["horizons"]:
                selected = [r for r in horizon["losses"]
                            if r["split"] == "heldout" and r["evaluator"] == .2
                            and r["donor"] != "no_compensation"]
                self.assertEqual(len(selected), 4)
                self.assertTrue(all(abs(r["mse"]-selected[0]["mse"]) < 1e-10 for r in selected))
                self.assertTrue(all(pair["cosine"] is None or abs(pair["cosine"]-1) < 1e-5
                                    for pair in horizon["directions"]))

    @torch.no_grad()
    def test_heldout_never_changes_reference_masks_or_training_hessian(self):
        import tempfile
        from types import SimpleNamespace
        from transformers import LlamaConfig, LlamaForCausalLM
        from run_checkpoint_token_isolation import CheckpointCache, prepare_reference_layer
        from run_compensation_conflict import collect_layer
        from compensation_conflict import RATES
        torch.manual_seed(12)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=40,
            intermediate_size=80, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=32,
            _attn_implementation="eager")).eval()
        initial = torch.randn(3, 8, 40)
        other = initial.clone()
        other[2] = torch.randn_like(other[2])*20
        args = SimpleNamespace(mode="2of4", structured_alpha=.25, unstructured_alpha=1.,
            percdamp=.01, reference_calibration_samples=2, nsamples=2, heldout_samples=1,
            probe_tokens=8, modules=["self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"])
        attention = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min).triu(1)
        positions = torch.arange(8).unsqueeze(0)
        values, states = [], []
        with tempfile.TemporaryDirectory() as folder:
            for i, value in enumerate((initial, other)):
                block = copy.deepcopy(model.model.layers[0])
                cache = CheckpointCache(Path(folder)/str(i), memory_reference=True)
                prepare_reference_layer(args, block, 0, value, attention, positions, cache, torch.device("cpu"))
                sources = {r: cache.source(0, r) for r in RATES}
                states.append(sources)
                _, h, probes, ids = collect_layer(block, {r: value for r in RATES}, sources,
                    attention, positions, torch.device("cpu"), args)
                self.assertFalse(set(ids["train"]) & set(ids["heldout"]))
                values.append((h, probes))
            for rate in RATES:
                for name in states[0][rate]:
                    torch.testing.assert_close(states[0][rate][name], states[1][rate][name], rtol=0, atol=0)
                for group in values[0][0]:
                    torch.testing.assert_close(values[0][0][group][rate], values[1][0][group][rate], rtol=0, atol=0)
            self.assertFalse(torch.equal(values[0][1]["mlp.down_proj"]["heldout"][.2],
                                         values[1][1]["mlp.down_proj"]["heldout"][.2]))


if __name__ == "__main__":
    unittest.main()
