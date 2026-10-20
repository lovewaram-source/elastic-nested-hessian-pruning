import unittest

import torch
from torch import nn

from lib.elastic_sparsegpt import (
    ELASTIC_2OF4_BLOCK_FRACTIONS,
    ElasticSparseGPTPlan,
    apply_elastic_2of4_stage,
    iter_2of4_stage_segments,
    column_permutation,
)


class ElasticTwoOfFourTest(unittest.TestCase):
    def test_stage_boundaries_are_four_column_aligned(self):
        self.assertEqual(iter_2of4_stage_segments(100, 0.0, 0.2), [(0, 40)])
        self.assertEqual(iter_2of4_stage_segments(100, 0.2, 0.3), [(40, 60)])
        self.assertEqual(iter_2of4_stage_segments(100, 0.3, 0.4), [(60, 80)])
        self.assertEqual(iter_2of4_stage_segments(100, 0.4, 0.5), [(80, 100)])

    def test_four_stages_are_nested_and_exact_2of4(self):
        layer = nn.Linear(100, 2, bias=False)
        layer.weight.data.copy_(torch.arange(1, 201, dtype=torch.float32).reshape(2, 100))
        plan = ElasticSparseGPTPlan(
            permutation=torch.arange(100),
            hessian_factor=torch.eye(100),
            dead_columns=torch.zeros(100, dtype=torch.bool),
        )
        previous_mask = torch.zeros_like(layer.weight, dtype=torch.bool)
        for previous, current in zip((0.0, 0.2, 0.3, 0.4), (0.2, 0.3, 0.4, 0.5)):
            apply_elastic_2of4_stage(
                layer,
                plan,
                current,
                previous_sparsity=previous,
                block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
            )
            mask = layer.weight == 0
            self.assertGreaterEqual(int(mask.sum()), int(previous_mask.sum()))
            self.assertTrue(torch.all(mask[previous_mask]))
            self.assertAlmostEqual(float(mask.float().mean()), current, places=6)
            start, end = iter_2of4_stage_segments(100, previous, current)[0]
            active = mask[:, start:end]
            self.assertTrue(torch.all(active.reshape(2, -1, 4).sum(dim=2) == 2))
            previous_mask = mask.clone()

    def test_pnp_quartile_order_keeps_nested_masks_and_local_2of4_groups(self):
        layer = nn.Linear(40, 2, bias=False)
        layer.weight.data.copy_(
            torch.arange(1, 81, dtype=torch.float32).reshape(2, 40)
        )
        permutation = column_permutation(
            layer.weight.data,
            ordering="wanda_group_stage_quartile_interleave",
            activation_norms=torch.ones(40),
        )
        plan = ElasticSparseGPTPlan(
            permutation=permutation,
            hessian_factor=torch.eye(40),
            dead_columns=torch.zeros(40, dtype=torch.bool),
        )
        previous_mask = torch.zeros_like(layer.weight, dtype=torch.bool)
        previous = 0.0
        for current in (0.2, 0.3, 0.4, 0.5):
            apply_elastic_2of4_stage(
                layer,
                plan,
                current,
                previous_sparsity=previous,
                block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
                compensation_alpha=0.25,
            )
            mask = layer.weight == 0
            self.assertTrue(torch.all(mask[previous_mask]))
            self.assertAlmostEqual(float(mask.float().mean()), current, places=6)

            ordered_mask = mask[:, permutation]
            start, end = iter_2of4_stage_segments(40, previous, current)[0]
            active = ordered_mask[:, start:end]
            self.assertTrue(
                torch.all(active.reshape(2, -1, 4).sum(dim=2) == 2)
            )
            previous_mask = mask.clone()
            previous = current

    def test_late_pnp_order_keeps_nested_masks_and_local_2of4_groups(self):
        layer = nn.Linear(40, 2, bias=False)
        layer.weight.data.copy_(
            torch.arange(1, 81, dtype=torch.float32).reshape(2, 40)
        )
        permutation = column_permutation(
            layer.weight.data,
            ordering="wanda_group_stage_pnp_late_interleave",
            activation_norms=torch.ones(40),
        )
        plan = ElasticSparseGPTPlan(
            permutation=permutation,
            hessian_factor=torch.eye(40),
            dead_columns=torch.zeros(40, dtype=torch.bool),
        )
        previous_mask = torch.zeros_like(layer.weight, dtype=torch.bool)
        previous = 0.0
        for current in (0.2, 0.3, 0.4, 0.5):
            apply_elastic_2of4_stage(
                layer,
                plan,
                current,
                previous_sparsity=previous,
                block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
                compensation_alpha=0.25,
            )
            mask = layer.weight == 0
            self.assertTrue(torch.all(mask[previous_mask]))
            self.assertAlmostEqual(float(mask.float().mean()), current, places=6)
            ordered_mask = mask[:, permutation]
            start, end = iter_2of4_stage_segments(40, previous, current)[0]
            active = ordered_mask[:, start:end]
            self.assertTrue(
                torch.all(active.reshape(2, -1, 4).sum(dim=2) == 2)
            )
            previous_mask = mask.clone()
            previous = current


if __name__ == "__main__":
    unittest.main()
