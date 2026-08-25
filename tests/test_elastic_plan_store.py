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
    def test_round_trips_dynamic_wanda_statistics(self):
        from lib.elastic_plan_store import MemoryElasticPlanStore

        plan = ElasticSparseGPTPlan(
            permutation=torch.tensor([0, 1]),
            hessian_factor=torch.eye(2),
            dead_columns=torch.tensor([False, False]),
            activation_norms=torch.tensor([2.0, 3.0]),
            hessian_inverse=torch.tensor([[1.0, 0.1], [0.1, 1.0]]),
            processed_columns=torch.tensor([True, False]),
        )
        store = MemoryElasticPlanStore()
        store.put_layer(3, {"self_attn.q_proj": plan})

        loaded = store.get_layer(3)["self_attn.q_proj"]

        self.assertTrue(torch.equal(loaded.activation_norms, plan.activation_norms))
        self.assertTrue(torch.equal(loaded.hessian_inverse, plan.hessian_inverse))
        self.assertTrue(torch.equal(loaded.processed_columns, plan.processed_columns))

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

    def test_keeps_successful_run_when_explicitly_requested(self):
        from lib.elastic_plan_store import DiskElasticPlanStore

        with tempfile.TemporaryDirectory() as base:
            store = DiskElasticPlanStore(base, keep_on_success=True)
            run_dir = store.run_dir
            store.put_layer(3, {"self_attn.q_proj": make_plan()})

            store.close(success=True)

            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(
                os.path.isfile(os.path.join(run_dir, "layer_003.pt"))
            )


class ElasticPlanStoreFactoryTests(unittest.TestCase):
    def test_disk_mode_without_a_directory_uses_a_temporary_streaming_cache(self):
        from lib.elastic_plan_store import make_elastic_plan_store

        store = make_elastic_plan_store("disk", None)
        run_dir = store.run_dir
        self.assertTrue(os.path.isdir(run_dir))
        store.close(success=True)
        self.assertFalse(os.path.exists(run_dir))


if __name__ == "__main__":
    unittest.main()
