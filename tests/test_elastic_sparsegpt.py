import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class PrefixStageTests(unittest.TestCase):
    def test_seven_slice_boundaries_are_twenty_then_ten_percent_stages(self):
        from lib.elastic_sparsegpt import prefix_column_count

        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.20), 1024)
        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.30), 1536)
        self.assertEqual(prefix_column_count(columns=4096, sparsity=0.80), 4096)

    def test_full_schedule_is_exactly_seven_outer_slices(self):
        from lib.elastic_sparsegpt import iter_stage_segments

        segments = iter_stage_segments(
            columns=4096,
            previous_sparsity=0.0,
            sparsity=0.80,
            blocksize=128,
        )

        self.assertEqual(
            segments,
            [
                (0, 1024),
                (1024, 1536),
                (1536, 2048),
                (2048, 2560),
                (2560, 3072),
                (3072, 3584),
                (3584, 4096),
            ],
        )

    def test_second_stage_is_one_whole_outer_slice(self):
        from lib.elastic_sparsegpt import iter_stage_segments

        self.assertEqual(
            iter_stage_segments(
                columns=4096,
                previous_sparsity=0.20,
                sparsity=0.30,
                blocksize=128,
            ),
            [(1024, 1536)],
        )

class MaskPlanningTests(unittest.TestCase):
    def test_block_mask_prunes_an_exact_stable_eighty_percent_prefix(self):
        from lib.elastic_sparsegpt import build_block_mask

        scores = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        mask = build_block_mask(scores, sparsity=0.8)

        self.assertEqual(int(mask.sum()), 8)
        self.assertTrue(torch.equal(mask, scores < 8))

    def test_columns_are_ordered_from_low_to_high_sparsegpt_score(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
        factor = torch.eye(3)

        self.assertTrue(
            torch.equal(column_permutation(weight, factor), torch.tensor([1, 2, 0]))
        )

    def test_sparsegpt_column_order_uses_the_full_column_sum(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [10.0, 2.0],
            ]
        )
        factor = torch.eye(2)

        self.assertTrue(
            torch.equal(column_permutation(weight, factor), torch.tensor([1, 0]))
        )

    def test_wanda_column_order_uses_calibration_activation_norms(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor(
            [
                [1.0, 5.0],
                [1.0, 5.0],
                [1.0, 5.0],
                [1.0, 5.0],
                [1.0, 5.0],
            ]
        )

    def test_wanda_column_order_sums_the_entire_column(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [10.0, 2.0],
            ]
        )

        self.assertTrue(
            torch.equal(
                column_permutation(
                    weight,
                    ordering="wanda",
                    activation_norms=torch.ones(2),
                ),
                torch.tensor([1, 0]),
            )
        )

        self.assertTrue(
            torch.equal(
                column_permutation(
                    weight,
                    ordering="wanda",
                    activation_norms=torch.tensor([10.0, 1.0]),
                ),
                torch.tensor([1, 0]),
            )
        )

    def test_sparsegpt_ordering_remains_available_for_ablation(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
        factor = torch.eye(3)

        self.assertTrue(
            torch.equal(
                column_permutation(weight, factor, ordering="sparsegpt"),
                torch.tensor([1, 2, 0]),
            )
        )

    def test_inverse_permutation_restores_columns(self):
        from lib.elastic_sparsegpt import invert_permutation

        weight = torch.arange(12).reshape(3, 4)
        permutation = torch.tensor([2, 0, 3, 1])

        restored = weight[:, permutation][:, invert_permutation(permutation)]

        self.assertTrue(torch.equal(restored, weight))

class ElasticCompensationTests(unittest.TestCase):
    def test_plan_preparation_reports_factorization_and_ordering_time(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        collector = ElasticSparseGPT(nn.Linear(4, 2, bias=False))
        collector.H = torch.eye(4)
        timings = {}

        collector.prepare_plan(
            percdamp=0.01,
            ordering="wanda",
            timings=timings,
        )

        self.assertIn("hessian_factorization_seconds", timings)
        self.assertIn("column_ordering_seconds", timings)
        self.assertIn("reordered_factor_seconds", timings)
        self.assertTrue(all(value >= 0.0 for value in timings.values()))

    def test_one_outer_slice_uses_one_global_mask_even_with_small_blocksize(self):
        from torch import nn

        from lib.elastic_sparsegpt import (
            ElasticSparseGPTPlan,
            apply_elastic_sparsegpt_stage,
        )

        layer = nn.Linear(16, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 10.0, 10.0] + [100.0] * 12,
                    [1.0, 1.0, 10.0, 10.0] + [100.0] * 12,
                ]
            )
        )
        plan = ElasticSparseGPTPlan(
            permutation=torch.arange(16),
            hessian_factor=torch.eye(16),
            dead_columns=torch.zeros(16, dtype=torch.bool),
        )

        apply_elastic_sparsegpt_stage(
            layer,
            plan,
            sparsity=0.20,
            previous_sparsity=0.0,
            blocksize=2,
        )

        self.assertEqual(int((layer.weight[:, :2] == 0).sum()), 4)

    def test_wanda_plan_does_not_build_an_unused_original_factor(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT, column_permutation

        layer = nn.Linear(8, 2, bias=False)
        collector = ElasticSparseGPT(layer)
        collector.H = torch.eye(8)

        with patch(
            "lib.elastic_sparsegpt.column_permutation",
            wraps=column_permutation,
        ) as permutation:
            collector.prepare_plan(percdamp=0.01, ordering="wanda")

        self.assertEqual(len(permutation.call_args.args), 1)
        self.assertIsNone(permutation.call_args.kwargs.get("hessian_factor"))

    def test_prepared_plan_can_apply_two_nested_stages_without_a_new_hessian(self):
        from torch import nn

        from lib.elastic_sparsegpt import (
            ElasticSparseGPT,
            apply_elastic_sparsegpt_stage,
        )

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        )
        collector = ElasticSparseGPT(layer)
        collector.H = torch.eye(8)
        plan = collector.prepare_plan(percdamp=0.01)

        apply_elastic_sparsegpt_stage(
            layer, plan, 0.20, 0.0, blocksize=2
        )
        mask_20 = layer.weight == 0
        apply_elastic_sparsegpt_stage(
            layer, plan, 0.30, 0.20, blocksize=2
        )
        mask_30 = layer.weight == 0

        self.assertEqual(int(mask_20.sum()), 3)
        self.assertEqual(int(mask_30.sum()), 5)
        self.assertTrue(torch.all(mask_20 <= mask_30))
        self.assertIsNone(collector.H)

    def test_later_stage_does_not_change_the_frozen_reordered_prefix(self):
        from torch import nn

        from lib.elastic_sparsegpt import (
            ElasticSparseGPT,
            apply_elastic_sparsegpt_stage,
            prefix_column_count,
        )

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        )
        collector = ElasticSparseGPT(layer)
        collector.H = torch.eye(8)
        plan = collector.prepare_plan(percdamp=0.01)

        apply_elastic_sparsegpt_stage(
            layer, plan, 0.20, 0.0, blocksize=2
        )
        reordered_after_20 = layer.weight[:, plan.permutation].clone()
        apply_elastic_sparsegpt_stage(
            layer, plan, 0.30, 0.20, blocksize=2
        )
        reordered_after_30 = layer.weight[:, plan.permutation]
        frozen_end = prefix_column_count(8, 0.20)

        self.assertTrue(
            torch.equal(
                reordered_after_30[:, :frozen_end],
                reordered_after_20[:, :frozen_end],
            )
        )

    def test_second_stage_preserves_existing_zeros_and_adds_only_new_zeros(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                    [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                ]
            )
        )

        first_stage = ElasticSparseGPT(layer)
        first_stage.H = torch.eye(8)
        permutation = first_stage.fasterprune(
            sparsity=0.20,
            previous_sparsity=0.0,
            blocksize=3,
            percdamp=0.01,
        )
        first_mask = layer.weight == 0

        second_stage = ElasticSparseGPT(layer)
        second_stage.H = torch.eye(8)
        second_stage.fasterprune(
            sparsity=0.30,
            previous_sparsity=0.20,
            permutation=permutation,
            blocksize=3,
            percdamp=0.01,
        )
        second_mask = layer.weight == 0

        self.assertEqual(int(first_mask.sum()), 3)
        self.assertEqual(int(second_mask.sum()), 5)
        self.assertTrue(torch.all(first_mask <= second_mask))

    def test_twenty_percent_zero_mask_is_nested_in_thirty_percent_mask(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        weights = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
                [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            ]
        )

        masks = []
        for ratio in (0.20, 0.30):
            layer = nn.Linear(8, 2, bias=False)
            layer.weight.data.copy_(weights)
            pruner = ElasticSparseGPT(layer)
            pruner.H = torch.eye(8)
            pruner.fasterprune(sparsity=ratio, blocksize=3, percdamp=0.01)
            masks.append(layer.weight == 0)

        self.assertEqual(int(masks[0].sum()), 3)
        self.assertEqual(int(masks[1].sum()), 5)
        self.assertTrue(torch.all(masks[0] <= masks[1]))

    def test_target_uses_only_its_prefix_region_not_the_next_group(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(8, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.1, 8.0, 9.0, 10.0, 11.0],
                    [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                ]
            )
        )
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(8)

        pruner.fasterprune(sparsity=0.30, blocksize=3, percdamp=0.01)

        self.assertEqual(int((layer.weight == 0).sum()), 5)
        self.assertTrue(torch.all(layer.weight[:, 3] != 0))

    def test_compensation_is_finite_exact_and_restores_layout(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(4, 2, bias=False)
        layer.weight.data.copy_(
            torch.tensor(
                [[4.0, 1.0, 3.0, 2.0], [8.0, 5.0, 7.0, 6.0]]
            )
        )
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(4)

        pruner.fasterprune(
            sparsity=0.20,
            blocksize=2,
            percdamp=0.01,
        )

        self.assertTrue(torch.isfinite(layer.weight).all())
        self.assertEqual(int((layer.weight == 0).sum()), 2)
        self.assertTrue(
            torch.equal(
                layer.weight,
                torch.tensor(
                    [[4.0, 0.0, 3.0, 2.0], [8.0, 0.0, 7.0, 6.0]]
                ),
            )
        )

    def test_rejects_sparsity_above_the_prefix_schedule_maximum(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(3, 2, bias=False)
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(3)

        with self.assertRaisesRegex(ValueError, "current <= max"):
            pruner.fasterprune(0.9)


class OrchestrationValidationTests(unittest.TestCase):
    def test_preparation_matches_sparsegpt_with_one_hessian_per_module(self):
        from torch import nn

        from lib.elastic_plan_store import MemoryElasticPlanStore
        from lib.prune import prepare_elastic_sparsegpt

        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)
                self.k_proj = nn.Linear(4, 4, bias=False)
                self.v_proj = nn.Linear(4, 4, bias=False)
                self.o_proj = nn.Linear(4, 4, bias=False)

            def forward(self, hidden):
                mixed = (
                    self.q_proj(hidden)
                    + self.k_proj(hidden)
                    + self.v_proj(hidden)
                )
                return self.o_proj(mixed)

        class TinyMlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 4, bias=False)
                self.up_proj = nn.Linear(4, 4, bias=False)
                self.down_proj = nn.Linear(4, 4, bias=False)

            def forward(self, hidden):
                return self.down_proj(
                    self.gate_proj(hidden) + self.up_proj(hidden)
                )

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = TinyAttention()
                self.mlp = TinyMlp()

            def forward(self, hidden, attention_mask=None, position_ids=None):
                return (self.mlp(self.self_attn(hidden)),)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([TinyBlock()])
                self.config = SimpleNamespace(use_cache=True, hidden_size=4)
                self.seqlen = 2
                self.hf_device_map = {}

            def forward(self, token_ids):
                hidden = torch.ones(
                    (token_ids.shape[0], self.seqlen, 4),
                    dtype=next(self.parameters()).dtype,
                    device=token_ids.device,
                )
                attention_mask = torch.ones(
                    (token_ids.shape[0], 1, self.seqlen, self.seqlen),
                    device=token_ids.device,
                )
                position_ids = torch.arange(
                    self.seqlen,
                    device=token_ids.device,
                ).unsqueeze(0)
                return self.model.layers[0](
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        model = TinyModel()
        args = SimpleNamespace(
            nsamples=2,
            seed=0,
            prune_method="elastic_wanda_sparsegpt",
        )
        dataloader = [
            (torch.zeros((1, 2), dtype=torch.long),),
            (torch.zeros((1, 2), dtype=torch.long),),
        ]
        store = MemoryElasticPlanStore()

        with patch("lib.prune.get_loaders", return_value=(dataloader, None)):
            with patch(
                "lib.elastic_sparsegpt.torch.cholesky_inverse",
                wraps=torch.cholesky_inverse,
            ) as inverse:
                prepare_elastic_sparsegpt(
                    args,
                    model,
                    object(),
                    torch.device("cpu"),
                    store,
                )

        self.assertEqual(inverse.call_count, 7)
        self.assertEqual(
            set(store.get_layer(0)),
            {
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            },
        )

    def test_orchestration_synchronizes_before_and_after_each_timed_region(self):
        from lib.prune import run_prepared_elastic_stages

        events = []
        ticks = iter([0.0, 2.0, 2.0, 5.0])

        run_prepared_elastic_stages(
            [0.2],
            lambda: events.append("prepare") or object(),
            lambda _store, _previous, _current: events.append("apply"),
            lambda *_args: None,
            clock=lambda: next(ticks),
            synchronize=lambda: events.append("sync"),
        )

        self.assertEqual(
            events,
            ["sync", "prepare", "sync", "sync", "apply", "sync"],
        )

    def test_multi_stage_orchestration_prepares_once(self):
        from lib.prune import run_prepared_elastic_stages

        calls = {"prepare": 0, "apply": []}
        timings = []

        def prepare():
            calls["prepare"] += 1
            return object()

        def apply(store, previous, current):
            calls["apply"].append((previous, current))

        ticks = iter([0.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0, 14.0])
        run_prepared_elastic_stages(
            [0.2, 0.3, 0.4],
            prepare,
            apply,
            lambda store, previous, current, prep, stage, cumulative: timings.append(
                (prep, stage, cumulative)
            ),
            clock=lambda: next(ticks),
        )

        self.assertEqual(
            calls,
            {
                "prepare": 1,
                "apply": [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4)],
            },
        )
        self.assertEqual(
            timings,
            [(5.0, 2.0, 7.0), (5.0, 3.0, 10.0), (5.0, 4.0, 14.0)],
        )

    def test_dense_preparation_does_not_modify_model_weights(self):
        from torch import nn

        from lib.elastic_plan_store import MemoryElasticPlanStore
        from lib.prune import prepare_elastic_sparsegpt

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)

            def forward(self, hidden, attention_mask=None, position_ids=None):
                return (self.proj(hidden),)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([TinyBlock()])
                self.config = SimpleNamespace(use_cache=True, hidden_size=4)
                self.seqlen = 2
                self.hf_device_map = {}

            def forward(self, token_ids):
                hidden = torch.ones(
                    (token_ids.shape[0], self.seqlen, 4),
                    dtype=next(self.parameters()).dtype,
                    device=token_ids.device,
                )
                attention_mask = torch.ones(
                    (token_ids.shape[0], 1, self.seqlen, self.seqlen),
                    device=token_ids.device,
                )
                position_ids = torch.arange(
                    self.seqlen,
                    device=token_ids.device,
                ).unsqueeze(0)
                return self.model.layers[0](
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        model = TinyModel()
        original = model.model.layers[0].proj.weight.detach().clone()
        args = SimpleNamespace(
            nsamples=2,
            seed=0,
            prune_method="elastic_sparsegpt",
        )
        tokenizer = object()
        dataloader = [
            (torch.zeros((1, 2), dtype=torch.long),),
            (torch.zeros((1, 2), dtype=torch.long),),
        ]
        store = MemoryElasticPlanStore()

        with patch("lib.prune.get_loaders", return_value=(dataloader, None)):
            prepare_elastic_sparsegpt(
                args,
                model,
                tokenizer,
                torch.device("cpu"),
                store,
            )

        self.assertTrue(
            torch.equal(model.model.layers[0].proj.weight, original)
        )
        self.assertIn("proj", store.get_layer(0))
        self.assertTrue(model.config.use_cache)

    def test_model_stage_uses_cached_plans_without_collecting_hessian(self):
        from torch import nn

        from lib.elastic_plan_store import MemoryElasticPlanStore
        from lib.elastic_sparsegpt import ElasticSparseGPTPlan
        from lib.prune import apply_elastic_sparsegpt_stage

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 2, bias=False)

        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([TinyBlock()])
        model.hf_device_map = {}
        model.model.layers[0].proj.weight.data.copy_(
            torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        )

        store = MemoryElasticPlanStore()
        store.put_layer(
            0,
            {
                "proj": ElasticSparseGPTPlan(
                    permutation=torch.arange(8),
                    hessian_factor=torch.eye(8),
                    dead_columns=torch.zeros(8, dtype=torch.bool),
                )
            },
        )

        apply_elastic_sparsegpt_stage(
            model,
            torch.device("cpu"),
            store,
            sparsity=0.20,
            previous_sparsity=0.0,
        )

        self.assertEqual(
            int((model.model.layers[0].proj.weight == 0).sum()),
            3,
        )

    def test_accepts_only_strictly_increasing_elastic_stages(self):
        from lib.prune import validate_elastic_sparsegpt_stages

        self.assertEqual(
            validate_elastic_sparsegpt_stages([0.2, 0.3, 0.4]),
            [0.2, 0.3, 0.4],
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_elastic_sparsegpt_stages([0.2, 0.4, 0.3])

    def test_rejects_stage_outside_the_seven_slice_schedule(self):
        from lib.prune import validate_elastic_sparsegpt_stages

        with self.assertRaisesRegex(ValueError, "0.2, 0.3, ..., 0.8"):
            validate_elastic_sparsegpt_stages([0.1, 0.2])

    def test_rejects_structured_sparsity(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=0.5, reorder_blocks=5)

        with self.assertRaisesRegex(ValueError, "unstructured"):
            validate_elastic_sparsegpt_args(args, prune_n=2, prune_m=4)

    def test_rejects_invalid_ratio(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=0.9)

        with self.assertRaisesRegex(ValueError, "0.2, 0.3, ..., 0.8"):
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)

    def test_accepts_prefix_schedule_ratios(self):
        from lib.prune import validate_elastic_sparsegpt_args

        for ratio in (0.2, 0.3, 0.8):
            args = SimpleNamespace(sparsity_ratio=ratio)
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)


class CliExposureTests(unittest.TestCase):
    def test_low_level_elastic_plan_defaults_to_sparsegpt_ordering(self):
        import inspect

        from lib.elastic_sparsegpt import ElasticSparseGPT

        parameter = inspect.signature(ElasticSparseGPT.prepare_plan).parameters[
            "ordering"
        ]
        self.assertEqual(parameter.default, "sparsegpt")

    def test_elastic_method_names_select_distinct_orderings(self):
        from lib.prune import elastic_ordering_for_method

        self.assertEqual(
            elastic_ordering_for_method("elastic_sparsegpt"),
            "sparsegpt",
        )
        self.assertEqual(
            elastic_ordering_for_method("elastic_wanda_sparsegpt"),
            "wanda",
        )

    def test_rejects_unknown_elastic_method_when_selecting_ordering(self):
        from lib.prune import elastic_ordering_for_method

        with self.assertRaisesRegex(ValueError, "unknown elastic SparseGPT method"):
            elastic_ordering_for_method("elastic_unknown")

    def test_parser_accepts_elastic_disk_cache_options(self):
        from main import build_parser

        args = build_parser().parse_args(
            [
                "--prune_method",
                "elastic_sparsegpt",
                "--elastic_cache_mode",
                "disk",
                "--elastic_cache_dir",
                "/tmp/elastic-cache",
            ]
        )

        self.assertEqual(args.elastic_cache_mode, "disk")
        self.assertEqual(args.elastic_cache_dir, "/tmp/elastic-cache")

    def test_parser_accepts_elastic_preparation_profile(self):
        from main import build_parser

        args = build_parser().parse_args(
            [
                "--prune_method",
                "elastic_wanda_sparsegpt",
                "--elastic_profile_preparation",
            ]
        )

        self.assertTrue(args.elastic_profile_preparation)

    def test_parser_defaults_to_memory_elastic_cache(self):
        from main import build_parser

        args = build_parser().parse_args(
            ["--prune_method", "elastic_sparsegpt"]
        )

        self.assertEqual(args.elastic_cache_mode, "memory")
        self.assertIsNone(args.elastic_cache_dir)

    def test_parser_accepts_wanda_ordered_elastic_method(self):
        from main import build_parser

        args = build_parser().parse_args(
            ["--prune_method", "elastic_wanda_sparsegpt"]
        )

        self.assertEqual(args.prune_method, "elastic_wanda_sparsegpt")

    def test_parser_does_not_expose_ambiguous_elastic_ordering_flag(self):
        from main import build_parser

        args = build_parser().parse_args(
            ["--prune_method", "elastic_sparsegpt"]
        )

        self.assertFalse(hasattr(args, "elastic_ordering"))

    def test_parser_removes_the_obsolete_group_count_option(self):
        from main import build_parser

        args = build_parser().parse_args(["--prune_method", "elastic_sparsegpt"])

        self.assertEqual(args.prune_method, "elastic_sparsegpt")
        self.assertFalse(hasattr(args, "reorder_blocks"))

    def test_parser_accepts_an_ordered_elastic_stage_list(self):
        from main import build_parser

        args = build_parser().parse_args(
            [
                "--prune_method",
                "elastic_sparsegpt",
                "--elastic_sparsities",
                "0.2",
                "0.3",
                "0.4",
            ]
        )

        self.assertEqual(args.elastic_sparsities, [0.2, 0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
