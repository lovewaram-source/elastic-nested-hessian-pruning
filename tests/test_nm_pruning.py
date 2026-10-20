import unittest

import torch

from lib.nm import build_nm_prune_mask


class NMPruningTests(unittest.TestCase):
    def test_two_of_four_prunes_two_lowest_values_in_each_row_group(self):
        metric = torch.tensor(
            [
                [4.0, 1.0, 3.0, 2.0, 10.0, 7.0, 8.0, 9.0],
                [0.5, 0.1, 0.4, 0.2, 6.0, 2.0, 5.0, 1.0],
            ]
        )

        mask = build_nm_prune_mask(metric, prune_n=2, prune_m=4)

        expected = torch.tensor(
            [
                [False, True, False, True, False, True, True, False],
                [False, True, False, True, False, True, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.all(mask.reshape(2, 2, 4).sum(dim=2) == 2))

    def test_rejects_groups_that_cannot_form_complete_nm_blocks(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_nm_prune_mask(torch.ones(2, 5), prune_n=2, prune_m=4)


if __name__ == "__main__":
    unittest.main()
