from __future__ import annotations

import json
import importlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from pelecpost.analysis.executors import EXECUTORS
from pelecpost.config.loader import load_project
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
            self.assertIn("temperature-spectrum.spectral.coherence", ids)
            self.assertTrue((first.run_dir / "report/index.html").is_file())
            created = {
                str(path.relative_to(first.run_dir))
                for path in first.run_dir.rglob("*") if path.is_file()
            } - {"artifacts.json"}
            self.assertEqual(created, set(paths))
            report_path = first.run_dir / "report/index.html"
            report = report_path.read_text(encoding="utf-8")
            links = set(re.findall(r"(?:href|src)='([^']+)'", report))
            for link in links:
                if link.startswith("#"):
                    anchor = re.escape(link[1:])
                    self.assertRegex(report, rf"id=['\"]{anchor}['\"]")
                else:
                    self.assertTrue((report_path.parent / link).resolve().is_file(), link)
            manifest = json.loads((first.run_dir / "manifest.json").read_text())
            fingerprints = manifest["provenance"]["input_fingerprints"]
            project_fingerprint = next(
                item for item in fingerprints if item["path"].endswith("case.yaml")
            )
            self.assertEqual(project_fingerprint["checksum_algorithm"], "sha256")
            self.assertEqual(len(project_fingerprint["checksum"]), 64)

    def test_independent_failure_preserves_products_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [
                {"id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure"},
                {
                    "id": "packet", "recipe": "transient_wavepacket", "variable": "pressure",
                    "band_min_hz": 1_000, "band_max_hz": 100_000,
                },
            ])
            def fail_packet(_context):
                raise RuntimeError("synthetic independent failure")

            with patch.dict(EXECUTORS, {"transient_wavepacket": fail_packet}):
                result = run_project(project)
            self.assertEqual(result.status, "failed")
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["workflows"]["analysis.spectrum"]["status"], "completed")
            self.assertEqual(manifest["workflows"]["analysis.packet"]["status"], "failed")
            self.assertTrue((result.run_dir / "data/spectrum/stationary_spectrum.npz").is_file())
            report = (result.run_dir / "report/index.html").read_text(encoding="utf-8")
            self.assertIn("synthetic independent failure", report)
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

    def test_single_pulse_executor_uses_distinct_finite_record_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "pulse", "recipe": "single_pulse_response", "variable": "temperature",
                "energy_per_pulse_j_m": 1.0, "pulse_fwhm_s": 1.0e-5,
                "pulse_period_s": 2.0e-4, "start_time_s": 1.0e-4,
                "baseline_end_time_s": 5.0e-5,
            }])
            result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            ids = {item["id"] for item in artifacts["artifacts"]}
            self.assertIn("pulse.pulse.transfer", ids)
            self.assertIn("pulse.pulse.source_spectrum", ids)
            self.assertIn("pulse.pulse.validity", ids)

    def test_directional_executor_registers_wavenumber_and_komega(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "wave", "recipe": "directional_wave", "variable": "pressure",
                "frequency_min_hz": 1_000, "frequency_max_hz": 400_000,
                "minimum_coherence": 0.0,
            }])
            result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            ids = {item["id"] for item in artifacts["artifacts"]}
            self.assertIn("wave.wave.wavenumber", ids)
            self.assertIn("wave.wave.komega", ids)
            self.assertIn("wave.wave.spatial_spectrum", ids)
            self.assertIn("wave.wave.komega_sensitivity", ids)
            sensitivity = json.loads(
                (result.run_dir / "data/wave/komega_sensitivity.json").read_text()
            )
            self.assertIn("window_sensitivity", sensitivity)
            self.assertIn("first_vs_second_half_block_sensitivity", sensitivity)

    def test_modal_executor_registers_products_by_artifact_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "modes", "recipe": "modal_screening", "variable": "pressure",
                "mode_count": 3, "spod_segment_samples": 256,
                "dmd_ranks": [2, 3],
            }])
            result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            ids = {item["id"] for item in artifacts["artifacts"]}
            modal_ids = {item for item in ids if item.startswith("modes.")}
            self.assertEqual(
                modal_ids,
                {"modes.modal.pod", "modes.modal.spod", "modes.modal.dmd", "modes.modal.sensitivity"},
            )

    def test_nonlinear_executor_records_explicit_frequency_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "triads", "recipe": "nonlinear_coupling", "variable": "pressure",
                "segment_samples": 128, "surrogate_count": 19,
                "frequency_max_hz": 100_000, "target_frequencies_hz": [25_000],
            }])
            result = run_project(project)
            self.assertEqual(result.status, "completed")
            summary = json.loads((result.run_dir / "data/triads/triad_summary.json").read_text())
            self.assertIn("explicit target_frequencies_hz", summary["selection"]["method"])

    def test_case_comparison_requires_and_uses_registered_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            source_project = self.make_project(source_root, [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "temperature",
                "welch_segment_samples": 256,
            }])
            baseline = run_project(source_project, "baseline")
            comparison = run_project(source_project, "comparison")
            compare_root = root / "compare-project"
            compare_root.mkdir()
            case = yaml.safe_load((source_project.root / "case.yaml").read_text())
            analyses = {"schema_version": 1, "analyses": [{
                "id": "compare", "recipe": "case_comparison",
                "baseline_run": str(baseline.run_dir),
                "comparison_run": str(comparison.run_dir),
                "artifact_ids": ["spectrum.spectral.psd", "spectrum.spectral.confidence"],
            }]}
            machine = {
                "schema_version": 1,
                "inputs": {"comparison_archives": [str(baseline.run_dir), str(comparison.run_dir)]},
                "outputs": {"root": str(compare_root / "runs")},
            }
            for name, value in (("case.yaml", case), ("analyses.yaml", analyses),
                                ("machine.yaml", machine)):
                (compare_root / name).write_text(yaml.safe_dump(value), encoding="utf-8")
            result = run_project(load_project(compare_root))
            self.assertEqual(result.status, "completed")
            metrics = json.loads((result.run_dir / "data/compare/comparison_metrics.json").read_text())
            self.assertTrue(metrics["metrics"])
            self.assertTrue(any(
                item["artifact_id"] == "spectrum.spectral.confidence"
                for item in metrics["metrics"]
            ))

    def test_interruption_is_atomic_and_reportable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "temperature",
            }])

            def interrupt(_context):
                raise KeyboardInterrupt()

            # Force registration first, then replace the selected executor.
            importlib.import_module("pelecpost.analysis.spectral")
            with patch.dict(EXECUTORS, {"probe_spectrum": interrupt}):
                result = run_project(project)
            self.assertEqual(result.status, "interrupted")
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["workflows"]["analysis.spectrum"]["status"], "interrupted")
            self.assertTrue((result.run_dir / "report/index.html").is_file())


if __name__ == "__main__":
    unittest.main()
