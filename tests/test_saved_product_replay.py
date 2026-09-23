"""Saved-run replay contracts, using a small self-contained archived run."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from pelecpost.analysis.comparison import _Product
from pelecpost.analysis.replay import (
    _output_records,
    _replay_wave_comparison,
    _replay_wave_producer,
    _source_fingerprints,
    replot,
)
from pelecpost.config.models import (
    CaseComparisonAnalysis,
    DirectionalWaveAnalysis,
    PresentationConfig,
)
from replot_saved_kernel_products import main as replay_main


class SavedProductReplayTests(unittest.TestCase):
    def test_replay_config_supplies_cli_defaults_and_records_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root)
            destination = root / "configured-output"
            config = root / "replay.yaml"
            config.write_text(yaml.safe_dump({"replay": {
                "export_root": str(source), "output_root": str(destination),
                "analysis_ids": ["measured"],
            }}))
            with (
                patch.object(sys, "argv", ["replay", "--replay-config", str(config)]),
                redirect_stdout(io.StringIO()),
            ):
                replay_main()
            manifest = json.loads((destination / "replot_manifest.json").read_text())
            self.assertEqual(manifest["selected_analysis_ids"], ["measured"])
            self.assertEqual(manifest["invocation"]["replay_config_path"], str(config.resolve()))
            self.assertIn("CLI options override", manifest["invocation"]["precedence"])

    def test_wave_views_have_semantic_metadata_and_legacy_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            frequency = np.array([1.0e5, 2.0e5, 3.0e5])
            wavenumber = np.array([-1.0, 0.0, 1.0])
            first, second = stage / "a.npz", stage / "b.npz"
            np.savez(first, frequency_hz=frequency, wavenumber_rad_m=wavenumber,
                     power=np.ones((3, 3)))
            np.savez(second, frequency_hz=frequency, wavenumber_rad_m=wavenumber,
                     power=2.0 * np.ones((3, 3)))
            analysis = CaseComparisonAnalysis.model_validate({
                "id": "pair", "recipe": "case_comparison",
                "baseline": {"analysis_id": "a"},
                "comparison": {"analysis_id": "b"},
                "product_ids": ["wave.komega"],
                "fft_ratio_plotting": {
                    "signed_k_ratio": False, "frequency_slices": False,
                },
            })
            project = SimpleNamespace(
                analyses_file=SimpleNamespace(presentation=PresentationConfig())
            )
            provenance = {}
            _replay_wave_comparison(
                analysis,
                (_Product("a.wave.komega", {"units": "Pa²"}, first),
                 _Product("b.wave.komega", {"units": "Pa²"}, second)),
                stage, project, provenance,
            )
            records = _output_records(stage, provenance)
            by_id = {item["id"]: item for item in records}
            for suffix in ("fk_amplitude_ratio", "fk_shape_linear_ratio"):
                figure_id = f"pair.comparison.{suffix}_figure"
                self.assertIn(figure_id, by_id)
                self.assertIn(f"pair.comparison.{suffix}_stats", by_id)
                self.assertEqual(by_id[figure_id]["provenance"]["ratio_direction"], "comparison / baseline")
                self.assertIn("support_rule", by_id[figure_id]["provenance"])
                self.assertEqual(by_id[figure_id]["coordinate_metadata"]["frequency"], "Hz")
                self.assertEqual(
                    by_id[figure_id]["units"],
                    "dB" if suffix == "fk_amplitude_ratio" else "1",
                )
                self.assertEqual(
                    by_id[f"{figure_id}.pdf"]["provenance"]["derived_from"], figure_id
                )
            self.assertEqual(
                by_id["pair.comparison.overlay_figure"]["provenance"]["derived_from"],
                "pair.comparison.fk_amplitude_ratio_figure",
            )
            wave = DirectionalWaveAnalysis(
                id="wave", recipe="directional_wave", probe_set_id="default",
                variable="pressure", frequency_max_hz=3.0e5,
            )
            _replay_wave_producer(
                wave, first, stage, project, provenance, "wave.wave.komega",
            )
            wave_record = next(
                item for item in _output_records(stage, provenance)
                if item["id"] == "wave.wave.komega.figure"
            )
            self.assertFalse(wave_record["provenance"]["transform_recomputed"])
            self.assertTrue((stage / wave_record["path"]).is_file())

    def make_archive(self, root: Path, *, missing_history: bool = False) -> Path:
        source = root / "archive"
        source.mkdir()
        case = yaml.safe_load(
            (Path(__file__).parents[1] / "examples/kernel-spectrum/case.yaml").read_text()
        )
        analyses = {
            "schema_version": 1,
            "analyses": [{
                "id": "measured", "recipe": "probe_spectrum", "probe_set_id": "default",
                "variable": "pressure", "probe_indices": [0, 1],
                "welch_segment_samples": 64, "frequency_max_hz": 20_000.0,
            }],
        }
        (source / "resolved-case.yaml").write_text(yaml.safe_dump(case))
        (source / "resolved-analyses.yaml").write_text(yaml.safe_dump(analyses))
        original = source / "original.bin"
        original.write_bytes(b"original solver input")
        manifest = {
            "schema": "pelecpost.run-manifest", "schema_version": 1,
            "run_id": "test-archive",
            "provenance": {"input_fingerprints": [{
                "path": "original.bin", "input_id": "inputs.probe_sets.default",
                "checksum": hashlib.sha256(original.read_bytes()).hexdigest(),
            }]},
        }
        (source / "manifest.json").write_text(json.dumps(manifest))
        history = source / "data/measured/probe_time_fft.npz"
        history.parent.mkdir(parents=True)
        if not missing_history:
            time = np.arange(256, dtype=float) * 1.0e-5
            values = np.column_stack((
                np.cos(2 * np.pi * 2_000 * time),
                np.cos(2 * np.pi * 2_000 * time - 0.2),
            ))
            np.savez(
                history, raw_time_s=time, raw_values=values,
                probe_indices=np.array([0, 1]), x_m=np.array([0.0, 0.001]),
                signal_unit=np.array("Pa"),
            )
        artifacts = {"schema": "pelecpost.artifacts", "schema_version": 1,
                     "artifacts": [{
                         "id": "measured.spectral.probe_signals",
                         "recipe_instance": "measured", "kind": "array",
                         "path": "data/measured/probe_time_fft.npz",
                         "variable": "pressure", "units": "Pa",
                     }]}
        (source / "artifacts.json").write_text(json.dumps(artifacts))
        return source

    def test_archive_configuration_fingerprints_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root)
            output = root / "replay"
            manifest_path = replot(source, output=output)
            first = json.loads(manifest_path.read_text())
            self.assertEqual(first["schema_version"], 4)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["configuration"]["source"], "archived_resolved_configuration")
            self.assertEqual(first["source_input_fingerprints"][0]["status"], "matched")
            self.assertEqual(first["selected_analysis_ids"], ["measured"])
            saved = output / "data/measured/stationary_spectrum.npz"
            expected = hashlib.sha256(saved.read_bytes()).hexdigest()
            replot(source, output=output)
            resumed = json.loads(manifest_path.read_text())
            self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), expected)
            self.assertTrue(all(
                item["resume_status"] == "kept_verified"
                for item in resumed["output_artifacts"]
            ))
            saved.write_bytes(b"corrupt")
            replot(source, output=output)
            repaired = json.loads(manifest_path.read_text())
            self.assertNotEqual(saved.read_bytes(), b"corrupt")
            self.assertEqual(
                next(item for item in repaired["output_artifacts"]
                     if item["path"] == "data/measured/stationary_spectrum.npz")["resume_status"],
                "regenerated",
            )

    def test_override_selection_mismatch_and_interrupted_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root)
            override = root / "override"
            override.mkdir()
            (override / "case.yaml").write_text((source / "resolved-case.yaml").read_text())
            analyses = yaml.safe_load((source / "resolved-analyses.yaml").read_text())
            analyses["analyses"][0]["window"] = "hamming"
            (override / "analyses.yaml").write_text(yaml.safe_dump(analyses))
            output = root / "replay"
            with (
                patch("pelecpost.analysis.replay._render_selected", side_effect=RuntimeError("interrupted")),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                replot(source, output=output)
            self.assertEqual(json.loads((output / "replot_manifest.json").read_text())["status"], "in_progress")
            replot(source, output=output)
            before = (output / "replot_manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "identity differs"):
                replot(source, project_dir=override, output=output)
            self.assertEqual((output / "replot_manifest.json").read_bytes(), before)
            alternate = json.loads(replot(source, project_dir=override, output=root / "override-replay").read_text())
            self.assertTrue(any(
                item["field"].startswith("analyses.analyses")
                for item in alternate["configuration"]["changed_fields_from_archive"]
            ))

    def test_discovery_selectors_per_analysis_limits_and_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root)
            analyses = yaml.safe_load((source / "resolved-analyses.yaml").read_text())
            analyses["analyses"].append({
                **analyses["analyses"][0], "id": "second",
                "frequency_max_hz": 10_000.0,
            })
            (source / "resolved-analyses.yaml").write_text(yaml.safe_dump(analyses))
            second = source / "data/second/probe_time_fft.npz"
            second.parent.mkdir()
            second.write_bytes((source / "data/measured/probe_time_fft.npz").read_bytes())
            registry_path = source / "artifacts.json"
            registry = json.loads(registry_path.read_text())
            registry["artifacts"].append({
                **registry["artifacts"][0],
                "id": "second.spectral.probe_signals",
                "recipe_instance": "second",
                "path": "data/second/probe_time_fft.npz",
            })
            registry_path.write_text(json.dumps(registry))
            manifest = json.loads(replot(source, output=root / "all").read_text())
            self.assertEqual(manifest["selected_analysis_ids"], ["measured", "second"])
            self.assertEqual(manifest["limits_by_analysis"]["measured"]["frequency_max_hz"], 20_000.0)
            self.assertEqual(manifest["limits_by_analysis"]["second"]["frequency_max_hz"], 10_000.0)
            selected = json.loads(replot(
                source, output=root / "selected", analysis_ids=("second",)
            ).read_text())
            self.assertEqual(selected["selected_analysis_ids"], ["second"])
            self.assertFalse((root / "selected/data/measured").exists())
            legacy_dir = root / "legacy-output"
            legacy_dir.mkdir()
            legacy_path = legacy_dir / "replot_manifest.json"
            legacy_path.write_text(json.dumps({
                "schema": "pelecpost.saved_product_replay", "schema_version": 3,
            }))
            before = legacy_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "older replay manifest"):
                replot(source, output=legacy_dir)
            self.assertEqual(legacy_path.read_bytes(), before)

    def test_missing_artifact_and_checksum_mismatch_are_structured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root, missing_history=True)
            manifest = json.loads(replot(source, output=root / "replay").read_text())
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["unavailable_products"][0]["product_id"], "spectral.probe_signals")
            (source / "original.bin").write_bytes(b"changed")
            self.assertEqual(
                _source_fingerprints(json.loads((source / "manifest.json").read_text()), source)[0]["status"],
                "mismatch",
            )
            (source / "original.bin").unlink()
            self.assertEqual(
                _source_fingerprints(json.loads((source / "manifest.json").read_text()), source)[0]["status"],
                "missing",
            )

    def test_legacy_psd_remains_descriptive_when_confidence_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_archive(root, missing_history=True)
            analyses = yaml.safe_load((source / "resolved-analyses.yaml").read_text())
            analyses["analyses"].append({
                **analyses["analyses"][0], "id": "other",
            })
            analyses["analyses"].append({
                "id": "compare", "recipe": "case_comparison",
                "baseline": {"analysis_id": "measured"},
                "comparison": {"analysis_id": "other"},
                "product_ids": ["spectral.psd"],
            })
            (source / "resolved-analyses.yaml").write_text(yaml.safe_dump(analyses))
            registry_path = source / "artifacts.json"
            registry = json.loads(registry_path.read_text())
            for analysis_id in ("measured", "other"):
                path = source / f"data/{analysis_id}/stationary_spectrum.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path, frequency_hz=np.array([0.0, 1_000.0, 2_000.0]),
                    psd=np.ones((3, 2)), probe_indices=np.array([0, 1]),
                    x_m=np.array([0.0, 0.001]), psd_unit=np.array("Pa²/Hz"),
                )
                registry["artifacts"].append({
                    "id": f"{analysis_id}.spectral.psd",
                    "recipe_instance": analysis_id, "kind": "array",
                    "path": f"data/{analysis_id}/stationary_spectrum.npz",
                    "variable": "pressure", "units": "Pa²/Hz",
                    "provenance": {"preprocessing": {"window": "hann"}},
                })
            registry_path.write_text(json.dumps(registry))
            output = root / "replay"
            manifest = json.loads(replot(source, output=output, analysis_ids=("compare",)).read_text())
            self.assertTrue((output / "figures/compare/comparison_psd.png").is_file())
            metrics = json.loads((output / "data/compare/comparison_metrics.json").read_text())
            self.assertEqual(metrics["status"], "available", manifest["unavailable_products"])
            self.assertEqual(metrics["products"], ["spectral.psd"])
            confidence = json.loads((output / "data/compare/psd_ratio_confidence.json").read_text())
            self.assertEqual(confidence["status"], "unavailable")
            self.assertIn("missing effective Welch", confidence["reason"])
            self.assertEqual(manifest["status"], "partial")


if __name__ == "__main__":
    unittest.main()
