import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn


class DummyAttention(nn.Module):
    def __init__(self, hidden_size=8, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.num_key_value_groups = 1
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class DummyMLP(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=6):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = DummyMLP()


class DummyModel(nn.Module):
    def __init__(self, layer_count=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer() for _ in range(layer_count)])
        self.embed = nn.Embedding(64, 8)
        self.lm_head = nn.Linear(8, 64, bias=False)


class OBSGroupTests(unittest.TestCase):
    def test_in_place_group_removal_preserves_storage_and_obs_result(self):
        from lib.slimgpt import remove_obs_group_in_place

        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 1.0, 0.5]])
        hinv = torch.tensor(
            [[2.0, 0.2, 0.1, 0.0], [0.2, 1.5, 0.3, 0.1],
             [0.1, 0.3, 1.8, 0.2], [0.0, 0.1, 0.2, 1.2]]
        )
        group = torch.tensor([1, 3])
        survivors = torch.tensor([0, 2])
        block = hinv[group][:, group]
        expected_weight = weight - torch.linalg.solve(
            block, weight[:, group].T
        ).T @ hinv[group]
        expected_hinv = hinv - (
            hinv[:, group] @ torch.linalg.solve(block, hinv[group])
        )
        weight_storage = weight.data_ptr()
        hinv_storage = hinv.data_ptr()

        remove_obs_group_in_place(weight, hinv, group)

        self.assertEqual(weight.data_ptr(), weight_storage)
        self.assertEqual(hinv.data_ptr(), hinv_storage)
        self.assertTrue(torch.allclose(weight[:, survivors], expected_weight[:, survivors]))
        self.assertTrue(torch.allclose(hinv[survivors][:, survivors], expected_hinv[survivors][:, survivors]))

    def test_group_loss_and_compensation_match_direct_formula(self):
        from lib.slimgpt import obs_group_loss, remove_obs_group

        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 1.0, 0.5]])
        hinv = torch.tensor(
            [[2.0, 0.2, 0.1, 0.0], [0.2, 1.5, 0.3, 0.1],
             [0.1, 0.3, 1.8, 0.2], [0.0, 0.1, 0.2, 1.2]]
        )
        group = torch.tensor([1, 3])
        block = hinv[group][:, group]
        expected_loss = torch.sum(
            weight[:, group] * torch.linalg.solve(block, weight[:, group].T).T
        )
        self.assertTrue(torch.allclose(obs_group_loss(weight, hinv, group), expected_loss))

        reduced_weight, reduced_hinv, survivors = remove_obs_group(weight, hinv, group)
        expected_full = weight - torch.linalg.solve(block, weight[:, group].T).T @ hinv[group]
        expected_survivors = torch.tensor([0, 2])
        expected_hinv = hinv[expected_survivors][:, expected_survivors] - (
            hinv[expected_survivors][:, group]
            @ torch.linalg.solve(block, hinv[group][:, expected_survivors])
        )
        self.assertTrue(torch.equal(survivors, expected_survivors))
        self.assertTrue(torch.allclose(reduced_weight, expected_full[:, expected_survivors]))
        self.assertTrue(torch.allclose(reduced_hinv, expected_hinv))


class StructuralSlicingTests(unittest.TestCase):
    def test_attention_slicing_keeps_coupled_head_coordinates(self):
        from lib.slimgpt import slice_llama_attention

        attention = DummyAttention()
        original_q = attention.q_proj.weight.detach().clone()
        original_o = attention.o_proj.weight.detach().clone()
        slice_llama_attention(attention, torch.tensor([1]))
        self.assertEqual(attention.num_heads, 1)
        self.assertEqual(attention.num_key_value_heads, 1)
        self.assertEqual(attention.hidden_size, 4)
        self.assertEqual(attention.q_proj.in_features, 8)
        self.assertEqual(attention.o_proj.out_features, 8)
        self.assertEqual(attention.q_proj.out_features, 4)
        self.assertEqual(attention.o_proj.in_features, 4)
        self.assertTrue(torch.equal(attention.q_proj.weight, original_q[4:8]))
        self.assertTrue(torch.equal(attention.o_proj.weight, original_o[:, 4:8]))

    def test_attention_slicing_rejects_gqa(self):
        from lib.slimgpt import slice_llama_attention

        attention = DummyAttention()
        attention.num_key_value_heads = 1
        attention.k_proj = nn.Linear(8, 4, bias=False)
        attention.v_proj = nn.Linear(8, 4, bias=False)
        with self.assertRaisesRegex(ValueError, "grouped-query"):
            slice_llama_attention(attention, torch.tensor([0]))

    def test_mlp_slicing_keeps_coupled_channels(self):
        from lib.slimgpt import slice_llama_mlp

        mlp = DummyMLP()
        gate = mlp.gate_proj.weight.detach().clone()
        down = mlp.down_proj.weight.detach().clone()
        kept = torch.tensor([1, 4, 5])
        slice_llama_mlp(mlp, kept)
        self.assertEqual(mlp.intermediate_size, 3)
        self.assertTrue(torch.equal(mlp.gate_proj.weight, gate[kept]))
        self.assertTrue(torch.equal(mlp.down_proj.weight, down[:, kept]))

    def test_parameter_count_really_decreases(self):
        from lib.slimgpt import slice_llama_attention, slice_llama_mlp

        layer = DummyLayer()
        before = sum(parameter.numel() for parameter in layer.parameters())
        slice_llama_attention(layer.self_attn, torch.tensor([1]))
        slice_llama_mlp(layer.mlp, torch.tensor([0, 2, 4]))
        after = sum(parameter.numel() for parameter in layer.parameters())
        self.assertLess(after, before)

    def test_real_transformers_llama_forward_runs_after_slicing(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from lib.slimgpt import slice_llama_attention, slice_llama_mlp

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        model = LlamaForCausalLM(config)
        layer = model.model.layers[0]
        slice_llama_attention(layer.self_attn, torch.tensor([0, 2]))
        slice_llama_mlp(layer.mlp, torch.arange(0, 32, 2))
        output = model(torch.tensor([[1, 2, 3, 4]])).logits
        self.assertEqual(tuple(output.shape), (1, 4, 32))


class RatioScheduleTests(unittest.TestCase):
    def test_log_profile_increases_with_depth(self):
        from lib.slimgpt import layer_pruning_ratios

        ratios = layer_pruning_ratios(6, 0.2, "log_increase")
        self.assertTrue(all(a <= b for a, b in zip(ratios, ratios[1:])))

    def test_calibration_targets_complete_model_parameter_reduction(self):
        from lib.slimgpt import calibrate_structural_ratios, expected_parameter_reduction

        model = DummyModel(layer_count=8)
        ratios = calibrate_structural_ratios(model, 0.2, "log_increase")
        reduction = expected_parameter_reduction(model, ratios)
        self.assertLess(abs(reduction - 0.2), 0.03)


class StructuredPruningTests(unittest.TestCase):
    def test_attention_pruning_removes_complete_lowest_loss_head(self):
        from lib.slimgpt import prune_attention_with_obs

        attention = DummyAttention()
        attention.o_proj.weight.data[:, :4] = 0.01
        attention.o_proj.weight.data[:, 4:] = 1.0
        metadata = prune_attention_with_obs(attention, torch.eye(8), 0.5)
        self.assertEqual(metadata["removed_heads"], [0])
        self.assertEqual(attention.num_heads, 1)

    def test_mlp_pruning_reaches_exact_channel_count(self):
        from lib.slimgpt import prune_mlp_with_obs

        mlp = DummyMLP(intermediate_size=10)
        mlp.down_proj.weight.data[:, :3] = 0.01
        metadata = prune_mlp_with_obs(mlp, torch.eye(10), 0.3, min_group_size=2)
        self.assertEqual(metadata["removed_count"], 3)
        self.assertEqual(mlp.intermediate_size, 7)

    def test_mlp_fixed_group_size_uses_128_until_final_remainder(self):
        from lib.slimgpt import prune_mlp_with_obs

        mlp = DummyMLP(intermediate_size=300)
        metadata = prune_mlp_with_obs(
            mlp,
            torch.eye(300),
            0.5,
            fixed_group_size=128,
        )
        self.assertEqual(metadata["removed_count"], 150)
        self.assertEqual(metadata["selection_group_sizes"], [128, 22])


class CliExposureTests(unittest.TestCase):
    def test_parser_accepts_slimgpt_options(self):
        from main import build_parser

        args = build_parser().parse_args(
            ["--prune_method", "slimgpt", "--slimgpt_ratio_profile", "uniform",
             "--slimgpt_ffn_min_group_size", "4"]
        )
        self.assertEqual(args.prune_method, "slimgpt")
        self.assertEqual(args.slimgpt_ratio_profile, "uniform")
        self.assertEqual(args.slimgpt_ffn_min_group_size, 4)

    def test_parser_accepts_slimgpt_fixed_ffn_group_size(self):
        from main import build_parser

        args = build_parser().parse_args(
            ["--prune_method", "slimgpt", "--slimgpt_ffn_group_size", "128"]
        )
        self.assertEqual(args.slimgpt_ffn_group_size, 128)


if __name__ == "__main__":
    unittest.main()
