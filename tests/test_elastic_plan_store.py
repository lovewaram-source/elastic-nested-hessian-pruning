import os
import tempfile
import unittest

import torch

from lib.elastic_sparsegpt import ElasticSparseGPTPlan


def make_plan():
    return ElasticSparseGPTPlan(
        permutation=torch.tensor([1, 0]),
        hessian_factor=torch.tensor([[1.0, 0.25], [0.0, 1.0]]),
        dead_columns=torch.tensor([False, True]),
    )


class MemoryElasticPlanStoreTests(unittest.TestCase):
    def test_round_trips_a_layer_plan(self):
        from lib.elastic_plan_store import MemoryElasticPlanStore

        store = MemoryElasticPlanStore()
        store.put_layer(3, {"self_attn.q_proj": make_plan()})

        loaded = store.get_layer(3)["self_attn.q_proj"]

        self.assertTrue(
            torch.equal(loaded.permutation, torch.tensor([1, 0]))
        )

    def test_rejects_duplicate_layer_index(self):
        from lib.elastic_plan_store import MemoryElasticPlanStore

        store = MemoryElasticPlanStore()
        store.put_layer(3, {"self_attn.q_proj": make_plan()})

        with self.assertRaisesRegex(ValueError, "already exists"):
            store.put_layer(3, {"self_attn.q_proj": make_plan()})


class DiskElasticPlanStoreTests(unittest.TestCase):
    def test_uses_unique_directory_and_cleans_successful_run(self):
        from lib.elastic_plan_store import DiskElasticPlanStore

        with tempfile.TemporaryDirectory() as base:
            store = DiskElasticPlanStore(base)
            run_dir = store.run_dir
            store.put_layer(3, {"self_attn.q_proj": make_plan()})

            self.assertTrue(
                os.path.exists(os.path.join(run_dir, "layer_003.pt"))
            )
            loaded = store.get_layer(3)["self_attn.q_proj"]
            self.assertTrue(
                torch.equal(
                    loaded.dead_columns,
                    torch.tensor([False, True]),
                )
            )

            store.close(success=True)
            self.assertFalse(os.path.exists(run_dir))

    def test_retains_failed_run(self):
        from lib.elastic_plan_store import DiskElasticPlanStore

        with tempfile.TemporaryDirectory() as base:
            store = DiskElasticPlanStore(base)
            run_dir = store.run_dir

            store.close(success=False)

            self.assertTrue(os.path.isdir(run_dir))


class ElasticPlanStoreFactoryTests(unittest.TestCase):
    def test_disk_mode_requires_cache_directory(self):
        from lib.elastic_plan_store import make_elastic_plan_store

        with self.assertRaisesRegex(
            ValueError,
            "elastic_cache_dir is required for disk mode",
        ):
            make_elastic_plan_store("disk", None)


if __name__ == "__main__":
    unittest.main()
