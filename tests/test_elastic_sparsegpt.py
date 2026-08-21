import unittest
from types import SimpleNamespace

import torch


class GroupAllocationTests(unittest.TestCase):
    def test_balances_non_divisible_columns(self):
        from lib.elastic_sparsegpt import balanced_group_slices

        self.assertEqual(
            balanced_group_slices(columns=12, groups=5),
            [(0, 3), (3, 6), (6, 8), (8, 10), (10, 12)],
        )

    def test_five_groups_follow_ten_twenty_thirty_percent_example(self):
        from lib.elastic_sparsegpt import (
            allocate_group_budgets,
            balanced_group_slices,
        )

        slices = balanced_group_slices(columns=10, groups=5)

        self.assertEqual(
            allocate_group_budgets(2, 10, 0.10, slices),
            [2, 0, 0, 0, 0],
        )
        self.assertEqual(
            allocate_group_budgets(2, 10, 0.20, slices),
            [4, 0, 0, 0, 0],
        )
        self.assertEqual(
            allocate_group_budgets(2, 10, 0.30, slices),
            [4, 2, 0, 0, 0],
        )

    def test_ten_groups_are_supported(self):
        from lib.elastic_sparsegpt import (
            allocate_group_budgets,
            balanced_group_slices,
        )

        slices = balanced_group_slices(columns=20, groups=10)

        self.assertEqual(
            allocate_group_budgets(3, 20, 0.10, slices),
            [6] + [0] * 9,
        )


class MaskPlanningTests(unittest.TestCase):
    def test_columns_are_ordered_from_low_to_high_sparsegpt_score(self):
        from lib.elastic_sparsegpt import column_permutation

        weight = torch.tensor([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
        factor = torch.eye(3)

        self.assertTrue(
            torch.equal(column_permutation(weight, factor), torch.tensor([1, 2, 0]))
        )

    def test_inverse_permutation_restores_columns(self):
        from lib.elastic_sparsegpt import invert_permutation

        weight = torch.arange(12).reshape(3, 4)
        permutation = torch.tensor([2, 0, 3, 1])

        restored = weight[:, permutation][:, invert_permutation(permutation)]

        self.assertTrue(torch.equal(restored, weight))

    def test_group_mask_uses_exact_stable_prefixes(self):
        from lib.elastic_sparsegpt import build_grouped_mask

        scores = torch.ones(2, 4)
        mask = build_grouped_mask(scores, [(0, 2), (2, 4)], [3, 0])
        expected = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )

        self.assertTrue(torch.equal(mask, expected))

    def test_masks_are_nested_with_exact_global_counts(self):
        from lib.elastic_sparsegpt import (
            allocate_group_budgets,
            balanced_group_slices,
            build_grouped_mask,
        )

        scores = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        slices = balanced_group_slices(10, 5)
        masks = [
            build_grouped_mask(
                scores,
                slices,
                allocate_group_budgets(2, 10, ratio, slices),
            )
            for ratio in (0.10, 0.20, 0.30)
        ]

        self.assertTrue(torch.all(masks[0] <= masks[1]))
        self.assertTrue(torch.all(masks[1] <= masks[2]))
        self.assertEqual([int(mask.sum()) for mask in masks], [2, 4, 6])


class ElasticCompensationTests(unittest.TestCase):
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
            sparsity=0.25,
            reorder_blocks=2,
            blocksize=2,
            percdamp=0.01,
        )

        self.assertTrue(torch.isfinite(layer.weight).all())
        self.assertEqual(int((layer.weight == 0).sum()), 2)
        self.assertTrue(
            torch.equal(
                layer.weight,
                torch.tensor(
                    [[4.0, 0.0, 3.0, 0.0], [8.0, 5.0, 7.0, 6.0]]
                ),
            )
        )

    def test_rejects_more_groups_than_columns(self):
        from torch import nn

        from lib.elastic_sparsegpt import ElasticSparseGPT

        layer = nn.Linear(3, 2, bias=False)
        pruner = ElasticSparseGPT(layer)
        pruner.H = torch.eye(3)

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            pruner.fasterprune(0.5, reorder_blocks=4)


class OrchestrationValidationTests(unittest.TestCase):
    def test_rejects_structured_sparsity(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=0.5, reorder_blocks=5)

        with self.assertRaisesRegex(ValueError, "unstructured"):
            validate_elastic_sparsegpt_args(args, prune_n=2, prune_m=4)

    def test_rejects_invalid_ratio(self):
        from lib.prune import validate_elastic_sparsegpt_args

        args = SimpleNamespace(sparsity_ratio=1.1, reorder_blocks=5)

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)

    def test_accepts_five_and_ten_groups(self):
        from lib.prune import validate_elastic_sparsegpt_args

        for groups in (5, 10):
            args = SimpleNamespace(sparsity_ratio=0.3, reorder_blocks=groups)
            validate_elastic_sparsegpt_args(args, prune_n=0, prune_m=0)


class CliExposureTests(unittest.TestCase):
    def test_parser_accepts_elastic_method_and_ten_groups(self):
        from main import build_parser

        args = build_parser().parse_args(
            [
                "--prune_method",
                "elastic_sparsegpt",
                "--sparsity_type",
                "unstructured",
                "--reorder_blocks",
                "10",
            ]
        )

        self.assertEqual(args.prune_method, "elastic_sparsegpt")
        self.assertEqual(args.reorder_blocks, 10)


if __name__ == "__main__":
    unittest.main()
