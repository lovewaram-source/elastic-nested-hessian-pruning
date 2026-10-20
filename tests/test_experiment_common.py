"""Shared-helper regression checks, without a deep-learning environment."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_common import _append_tsv, _write_json, benchmark_gate, input_groups, synchronize


class CommonHelperTests(unittest.TestCase):
    def test_synchronize_imports_torch_for_cuda_and_is_noop_on_cpu(self):
        with patch.dict(sys.modules, {"torch": None}):
            synchronize(SimpleNamespace(type="cpu"))
        sync = Mock()
        device = SimpleNamespace(type="cuda")
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))}):
            synchronize(device)
        sync.assert_called_once_with(device)

    def test_json_replacement_and_tsv_single_header(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_json(root / "result.json", {"stage": 1})
            _write_json(root / "result.json", {"stage": 2})
            self.assertEqual(json.loads((root / "result.json").read_text()), {"stage": 2})
            _append_tsv(root / "comparison.tsv", {"stage": 1, "ppl": 5.9})
            _append_tsv(root / "comparison.tsv", {"stage": 2, "ppl": 6.5})
            with (root / "comparison.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([r["stage"] for r in rows], ["1", "2"])

    def test_only_standard_llama_projection_groups_are_accepted(self):
        names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        groups = input_groups(dict.fromkeys(names))
        self.assertEqual(len(groups), 4)
        self.assertEqual(set(sum(groups.values(), ())), set(names))
        with self.assertRaises(ValueError):
            input_groups({"unknown": None})

    def test_benchmark_gate_requires_numerical_agreement(self):
        rows = [{"implementation": "direct"}, {"implementation": "cached_direct",
                 "hessian_errors": [{"max_abs": 0}],
                 "target_errors": [{"mask_disagreements": 0, "max_abs": 0}]}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "benchmark.json"
            path.write_text(json.dumps(rows))
            self.assertTrue(benchmark_gate(path)[0])
            rows[1]["target_errors"][0]["mask_disagreements"] = 1
            path.write_text(json.dumps(rows))
            self.assertFalse(benchmark_gate(path)[0])


if __name__ == "__main__":
    unittest.main()
