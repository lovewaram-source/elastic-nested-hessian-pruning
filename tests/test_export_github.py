"""Publishing checks; these tests need only the Python standard library."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_github import export_project, select_files


class ExportTests(unittest.TestCase):
    def test_dependency_closure_and_archive_exclude_local_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            root.mkdir()
            (root / "run.py").write_text("import helper\n", encoding="utf-8")
            (root / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
            (root / "private.log").write_text("not for publishing", encoding="utf-8")
            files = select_files(root, ["run.py"])
            self.assertEqual(files, ["helper.py", "run.py"])
            destination = Path(folder) / "public"
            manifest = export_project(root, destination, files, make_zip=True)
            self.assertEqual(manifest["source_file_count"], 2)
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
            with zipfile.ZipFile(destination.with_suffix(".zip")) as archive:
                self.assertEqual(set(archive.namelist()), {"helper.py", "run.py", "SOURCE_MANIFEST.json"})

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "existing"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_project(root, destination, [])
            self.assertEqual(marker.read_text(), "keep")

    def test_credential_aborts_before_any_public_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake_key = "ghp_" + "A" * 36
            (root / "bad.py").write_text(f"KEY = '{fake_key}'\n", encoding="utf-8")
            destination = root / "public"
            with self.assertRaisesRegex(ValueError, "possible credential in bad.py"):
                export_project(root, destination, ["bad.py"])
            self.assertFalse(destination.exists())

    def test_source_outside_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                export_project(root, root / "public", ["../outside.py"])


if __name__ == "__main__":
    unittest.main()
