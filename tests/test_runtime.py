from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pelecpost.runtime import generate_report, run_project
from tests.test_preflight import PreflightTests


class RuntimeTests(unittest.TestCase):
    def make_project(self, root: Path, analyses: list[dict]):
        return PreflightTests().project(root, analyses)

    def test_probe_spectrum_creates_isolated_registered_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "temperature-spectrum", "recipe": "probe_spectrum",
                "variable": "temperature", "frequency_max_hz": 400_000,
                "welch_segment_samples": 256,
            }])
            first = run_project(project, "test")
            second = run_project(project, "test")
            self.assertEqual(first.status, "completed")
            self.assertEqual(second.status, "completed")
            self.assertNotEqual(first.run_dir, second.run_dir)
            artifacts = json.loads((first.run_dir / "artifacts.json").read_text())
            ids = [item["id"] for item in artifacts["artifacts"]]
            paths = [item["path"] for item in artifacts["artifacts"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(paths), len(set(paths)))
            self.assertIn("temperature-spectrum.spectral.psd", ids)
            self.assertTrue((first.run_dir / "report/index.html").is_file())

    def test_independent_failure_preserves_products_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [
                {"id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure"},
                {"id": "modes", "recipe": "modal_screening", "variable": "pressure"},
            ])
            result = run_project(project)
            self.assertEqual(result.status, "failed")
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["workflows"]["analysis.spectrum"]["status"], "completed")
            self.assertEqual(manifest["workflows"]["analysis.modes"]["status"], "unavailable")
            self.assertTrue((result.run_dir / "data/spectrum/stationary_spectrum.npz").is_file())
            report = (result.run_dir / "report/index.html").read_text(encoding="utf-8")
            self.assertIn("unavailable", report)
            self.assertIn("spectrum.spectral.psd", report)

    def test_report_regeneration_needs_no_simulation_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "temperature",
            }])
            result = run_project(project)
            (root / "probes.h5").unlink()
            report = generate_report(result.run_dir)
            self.assertTrue(report.is_file())
            self.assertIn("PeleC post-processing report", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
