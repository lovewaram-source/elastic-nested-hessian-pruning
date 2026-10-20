"""Data-path configuration tests that do not need PyTorch or datasets installed."""
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class DataPathTests(unittest.TestCase):
    def load_constants(self, overrides):
        fake_datasets = ModuleType("datasets")
        fake_datasets.load_dataset = lambda *a, **k: None
        fake_datasets.load_from_disk = lambda *a, **k: None
        stubs = {"numpy": ModuleType("numpy"), "torch": ModuleType("torch"),
                 "datasets": fake_datasets}
        environment = {k: v for k, v in os.environ.items() if not k.startswith("WANDA_")}
        environment.update(overrides)
        with patch.dict(sys.modules, stubs), patch.dict(os.environ, environment, clear=True):
            return runpy.run_path(str(Path(__file__).resolve().parents[1] / "lib/data.py"))

    def test_original_defaults_remain_unchanged(self):
        constants = self.load_constants({})
        self.assertEqual(constants["LOCAL_C4_TRAIN"], "/data/DataSet/llm_dataset/c4/c4train")
        self.assertEqual(constants["LOCAL_C4_VALIDATION"], "/data/DataSet/llm_dataset/c4/c4validation")
        self.assertEqual(constants["LOCAL_WIKITEXT2_TRAIN"], "/data/DataSet/llm_dataset/wikitext2/wikitext2_train")
        self.assertEqual(constants["LOCAL_WIKITEXT2_TEST"], "/data/DataSet/llm_dataset/wikitext2/wikitext2_test")

    def test_all_four_datasets_can_be_configured_before_import(self):
        overrides = {"WANDA_C4_TRAIN": "/datasets/c4/train", "WANDA_C4_VALIDATION": "/datasets/c4/val",
                     "WANDA_WIKITEXT2_TRAIN": "/datasets/wt/train", "WANDA_WIKITEXT2_TEST": "/datasets/wt/test"}
        constants = self.load_constants(overrides)
        for key, value in overrides.items():
            self.assertEqual(constants[key.replace("WANDA_", "LOCAL_")], value)


if __name__ == "__main__":
    unittest.main()
