"""Early validation and evaluation labels without a GPU or model download."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_metadata import evaluation_record, validate_run_arguments


class RunMetadataTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(mode="2of4", rates=(.2, .3, .4, .5), percdamp=.01,
                      unstructured_alpha=1., structured_alpha=.25,
                      eval_samples=None, seqlen=8)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_invalid_options_fail_before_loading(self):
        for update in (dict(eval_samples=0), dict(eval_samples=-1), dict(percdamp=float("nan")),
                       dict(percdamp=-1), dict(structured_alpha=2),
                       dict(unstructured_alpha=float("inf")), dict(rates=(.2, .3, .6)),
                       dict(rates=(.2, .4)), dict(mode="unstructured", rates=(.2, .3, .304))):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_run_arguments(self.args(**update))

    def test_legal_modes_and_prefixes(self):
        validate_run_arguments(self.args(rates=(.2, .3)))
        validate_run_arguments(self.args(mode="unstructured", rates=(.2, .3, .6, .7)))

    def test_debug_label_and_actual_windows(self):
        evaluation = SimpleNamespace(input_ids=SimpleNamespace(numel=lambda: 19))
        debug = evaluation_record(self.args(eval_samples=4), evaluation)
        self.assertEqual(debug["scope"], "truncated_debug")
        self.assertEqual(debug["actual_windows"], 2)
        self.assertEqual(debug["evaluated_next_token_targets"], 14)
        full = evaluation_record(self.args(), evaluation)
        self.assertEqual(full["scope"], "full_available_windows")


if __name__ == "__main__":
    unittest.main()
