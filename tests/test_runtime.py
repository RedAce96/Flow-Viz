from __future__ import annotations

import json
import importlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from pelecpost.analysis.executors import EXECUTORS
from pelecpost.config.loader import load_project
from pelecpost.runtime import generate_report, run_project
from tests.test_preflight import PreflightTests


class RuntimeTests(unittest.TestCase):
    def make_project(self, root: Path, analyses: list[dict]):
        return PreflightTests().project(root, analyses)

    def test_multi_probe_overlays_and_per_probe_stft_are_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [
                {
                    "id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure",
                    "probe_indices": [0, 2, 5], "frequency_max_hz": 400_000,
                    "welch_segment_samples": 256,
                    "probe_plotting": {"normalization": "per_probe_peak"},
                },
                {
                    "id": "packet", "recipe": "transient_wavepacket", "variable": "pressure",
                    "probe_indices": [0, 2, 5], "band_min_hz": 1_000,
                    "band_max_hz": 100_000, "stft_segment_samples": 256,
                },
            ])
            result = run_project(project, "overlay")
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())["artifacts"]
            ids = {item["id"] for item in artifacts}
            for suffix in (
                "probe.raw_history.figure", "probe.method_ready.figure",
                "spectral.fft_overlay.figure", "spectral.psd_overlay.figure",
            ):
                self.assertIn(f"spectrum.{suffix}", ids)
            for suffix in (
                "probe.raw_history.figure", "probe.method_ready.figure",
                "transient.filtered_overlay.figure", "transient.envelope_overlay.figure",
                "transient.stft_figure",
            ):
                self.assertIn(f"packet.{suffix}", ids)
            raw = next(item for item in artifacts if item["id"] == "spectrum.probe.raw_history.figure")
            self.assertEqual(raw["provenance"]["normalization"], "per_probe_peak")
            self.assertEqual(raw["provenance"]["selected_probe_indices"], [0, 2, 5])
            self.assertEqual(len(raw["provenance"]["normalization_scales"]), 3)
            figure_paths = [result.run_dir / item["path"] for item in artifacts if item["kind"] == "figure"]
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in figure_paths))
            with np.load(result.run_dir / "data/packet/packet_stft.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["complex_stft"].ndim, 2)
                self.assertEqual(arrays["probe_complex_stft"].shape[2], 3)
                self.assertEqual(arrays["representative"].item(), "probe median")

    def test_probe_plotting_panels_mode_omits_overlay_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.make_project(Path(temporary), [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure",
                "probe_plotting": {"mode": "panels"},
                "welch_segment_samples": 256,
            }])
            result = run_project(project, "panels")
            self.assertEqual(result.status, "completed")
            ids = {
                item["id"]
                for item in json.loads((result.run_dir / "artifacts.json").read_text())["artifacts"]
            }
            self.assertIn("spectrum.spectral.probe_figure", ids)
            self.assertNotIn("spectrum.probe.raw_history.figure", ids)
            self.assertNotIn("spectrum.spectral.fft_overlay.figure", ids)

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
            self.assertIn("temperature-spectrum.spectral.figure", ids)
            self.assertIn("temperature-spectrum.spectral.probe_signals", ids)
            self.assertIn("temperature-spectrum.spectral.probe_figure", ids)
            self.assertIn("run.measurement-evidence", ids)
            self.assertTrue((first.run_dir / "report/index.html").is_file())
            self.assertTrue((first.run_dir / "figures/temperature-spectrum/probe_time_fft.png").is_file())
            self.assertTrue((first.run_dir / "data/temperature-spectrum/probe_time_fft.npz").is_file())
            created = {
                str(path.relative_to(first.run_dir))
                for path in first.run_dir.rglob("*") if path.is_file()
            } - {"artifacts.json"}
            self.assertEqual(created, set(paths))
            report_path = first.run_dir / "report/index.html"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Measurement-based classification", report)
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
            psd = next(
                item for item in artifacts["artifacts"]
                if item["id"] == "temperature-spectrum.spectral.psd"
            )
            self.assertEqual(psd["source_inputs"], [str(root / "probes.h5")])
            self.assertEqual(
                psd["provenance"]["preprocessing"]["welch_segment_samples"], 256
            )
            evidence = json.loads(
                (first.run_dir / "data/measurement_evidence.json").read_text()
            )
            self.assertEqual(evidence["excluded_scope"]["LST_PSE"], "not performed or inferred")

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

    def test_executor_cannot_silently_skip_declared_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.make_project(Path(temporary), [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure",
            }])
            importlib.import_module("pelecpost.analysis.spectral")
            with patch.dict(EXECUTORS, {"probe_spectrum": lambda _context: None}):
                result = run_project(project)
            self.assertEqual(result.status, "failed")
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            message = manifest["workflows"]["analysis.spectrum"]["message"]
            self.assertIn("returned without declared artifact", message)
            self.assertIn("spectral.psd", message)

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
            self.assertIn("wave.wave.komega.figure", ids)
            sensitivity = json.loads(
                (result.run_dir / "data/wave/komega_sensitivity.json").read_text()
            )
            self.assertIn("window_sensitivity", sensitivity)
            self.assertIn("first_vs_second_half_block_sensitivity", sensitivity)

    def test_directional_executor_registers_time_localized_wavenumber_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [{
                "id": "wave", "recipe": "directional_wave", "variable": "pressure",
                "frequency_min_hz": 1_000, "frequency_max_hz": 400_000,
                "minimum_coherence": 0.0,
                "temporal_wavenumber": {
                    "enabled": True, "window_duration_s": 128.0e-6,
                    "snapshot_times_s": [256.0e-6, 512.0e-6, 768.0e-6],
                },
            }])
            result = run_project(project, "localized-wave")
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())["artifacts"]
            ids = {item["id"] for item in artifacts}
            for suffix in (
                "wave.temporal_wavenumber", "wave.komega_snapshots",
                "wave.space_time.figure", "wave.temporal_wavenumber.figure",
                "wave.wavenumber_history.figure", "wave.komega_snapshots.figure",
                "wave.dispersion.figure",
            ):
                self.assertIn(f"wave.{suffix}", ids)
            with np.load(result.run_dir / "data/wave/temporal_wavenumber.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["band_power"].ndim, 2)
                self.assertEqual(arrays["time_center_s"].shape, arrays["valid_time_mask"].shape)
                self.assertEqual(arrays["relative_band_power_db"].shape, arrays["band_power"].shape)
            with np.load(result.run_dir / "data/wave/komega_snapshots.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["power"].ndim, 3)
                self.assertEqual(arrays["power"].shape[0], 3)
                self.assertEqual(arrays["power"].shape, arrays["relative_power_db"].shape)

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
                {
                    "modes.modal.pod", "modes.modal.spod", "modes.modal.dmd",
                    "modes.modal.sensitivity", "modes.probe.raw_history.figure",
                    "modes.probe.method_ready.figure",
                    "modes.modal.pod.figure", "modes.modal.spod.figure",
                    "modes.modal.dmd.figure",
                },
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
                "baseline": {"archived_run_id": "baseline", "analysis_id": "spectrum"},
                "comparison": {"archived_run_id": "comparison", "analysis_id": "spectrum"},
                "product_ids": ["spectral.psd", "spectral.confidence"],
            }]}
            machine = {
                "schema_version": 1,
                "inputs": {"archived_runs": {
                    "baseline": str(baseline.run_dir),
                    "comparison": str(comparison.run_dir),
                }},
                "outputs": {"root": str(compare_root / "runs")},
            }
            for name, value in (("case.yaml", case), ("analyses.yaml", analyses),
                                ("machine.yaml", machine)):
                (compare_root / name).write_text(yaml.safe_dump(value), encoding="utf-8")
            result = run_project(load_project(compare_root))
            self.assertEqual(
                result.status, "completed",
                (result.run_dir / "logs/run.log").read_text(encoding="utf-8"),
            )
            metrics = json.loads((result.run_dir / "data/compare/comparison_metrics.json").read_text())
            self.assertTrue(metrics["metrics"])
            self.assertTrue(any(
                item["product_id"] == "spectral.confidence"
                for item in metrics["metrics"]
            ))
            fingerprints = json.loads((result.run_dir / "manifest.json").read_text())[
                "provenance"
            ]["input_fingerprints"]
            self.assertTrue(any(
                item["path"].endswith("stationary_spectrum.npz")
                for item in fingerprints
            ))

    def test_case_comparison_can_use_local_analysis_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.make_project(root, [
                {
                    "id": "asym", "recipe": "probe_spectrum", "variable": "pressure",
                    "welch_segment_samples": 256,
                },
                {
                    "id": "gaus", "recipe": "probe_spectrum", "variable": "pressure",
                    "welch_segment_samples": 256,
                },
                {
                    "id": "compare", "recipe": "case_comparison",
                    "baseline": {"analysis_id": "asym"},
                    "comparison": {"analysis_id": "gaus"},
                    "product_ids": ["spectral.psd"],
                },
            ])
            result = run_project(project)
            self.assertEqual(result.status, "completed")
            metrics = json.loads(
                (result.run_dir / "data/compare/comparison_metrics.json").read_text()
            )
            self.assertEqual(metrics["baseline"], {"analysis_id": "asym", "archived_run_id": None})
            self.assertTrue(metrics["metrics"])

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
