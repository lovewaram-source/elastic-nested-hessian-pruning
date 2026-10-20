"""Data and render contracts for the README figures; stdlib only."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_readme_figures import ROOT, METHODS, read_results, result_figure, method_figure


class FigureTests(unittest.TestCase):
    def test_markdown_numbers_and_missing_results_are_preserved(self):
        tables = read_results(ROOT / "docs/RESULTS.md")
        self.assertEqual(tables["2of4"][METHODS[0][0]][-1], 8.72178931)
        self.assertEqual(tables["unstructured"][METHODS[3][0]], [None] * 4)
        self.assertNotIn(METHODS[6][0], tables["unstructured"])

    def test_outputs_are_accessible_vector_graphics(self):
        for svg in (method_figure(), result_figure(read_results(ROOT / "docs/RESULTS.md"))):
            root = ET.fromstring(svg)
            ns = {"s": "http://www.w3.org/2000/svg"}
            self.assertIsNotNone(root.find("s:title", ns))
            self.assertIsNotNone(root.find("s:desc", ns))
            self.assertGreater(len(root.findall(".//s:text", ns)), 10)
            self.assertEqual(root.findall(".//s:image", ns), [])
            self.assertNotIn("\ufffd", svg)

    def test_committed_figures_match_current_source_tables(self):
        expected = {"method-overview.svg": method_figure(),
                    "ppl-comparison.svg": result_figure(read_results(ROOT / "docs/RESULTS.md"))}
        for name, svg in expected.items():
            self.assertEqual((ROOT / "docs/figures" / name).read_text(encoding="utf-8"), svg)


if __name__ == "__main__":
    unittest.main()
