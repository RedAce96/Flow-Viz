from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from typer.testing import CliRunner

from pelecpost.cli import app
from pelecpost.config.loader import load_project, write_project_schema
from pelecpost.errors import ProjectConfigurationError


CASE = {
    "schema_version": 1,
    "case": {
        "id": "test-case",
        "solver": "pelec",
        "dimensionality": 2,
        "solver_units": "cgs",
    },
    "gas": {"gamma": 1.4, "gas_constant_j_kg_k": 287.05},
    "freestream": {
        "source": "explicit",
        "density_kg_m3": 0.02,
        "velocity_m_s": 1000.0,
        "pressure_pa": 1000.0,
        "temperature_k": 175.0,
    },
    "geometry": {"type": "flat_plate"},
}


class ProjectConfigTests(unittest.TestCase):
    def write_project(self, root: Path, analyses=None, machine=None):
        values = {
            "case.yaml": CASE,
            "analyses.yaml": {
                "schema_version": 1,
                "analyses": [] if analyses is None else analyses,
            },
            "machine.yaml": machine or {
                "schema_version": 1,
                "outputs": {"root": str(root / "output")},
            },
        }
        for name, value in values.items():
            (root / name).write_text(yaml.safe_dump(value), encoding="utf-8")

    def test_empty_project_is_valid_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root)
            project = load_project(root)
            self.assertEqual(project.case_file.case.id, "test-case")
            self.assertEqual(project.enabled_analyses, ())

    def test_unknown_key_reports_full_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root)
            case = dict(CASE)
            case["case"] = dict(case["case"], mystery_setting=True)
            (root / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
            with self.assertRaisesRegex(
                ProjectConfigurationError, r"case\.mystery_setting"
            ):
                load_project(root)

    def test_unknown_key_suggests_nearest_valid_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root)
            case = dict(CASE)
            case["case"] = dict(case["case"], dimensionalty=2)
            (root / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
            with self.assertRaisesRegex(
                ProjectConfigurationError, r"nearest valid: dimensionality"
            ):
                load_project(root)

    def test_machine_file_cannot_contain_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine = {
                "schema_version": 1,
                "outputs": {"root": str(root / "output")},
                "freestream": {"velocity_m_s": 1.0},
            }
            self.write_project(root, machine=machine)
            with self.assertRaisesRegex(ProjectConfigurationError, "freestream"):
                load_project(root)

    def test_analysis_union_rejects_unknown_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root, analyses=[{"id": "bad", "recipe": "magic"}])
            with self.assertRaisesRegex(ProjectConfigurationError, "magic"):
                load_project(root)

    def test_probe_indices_are_nonnegative_and_unique_for_every_probe_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root, analyses=[{
                "id": "bad", "recipe": "transient_wavepacket",
                "probe_set_id": "default", "variable": "pressure", "band_min_hz": 1.0,
                "band_max_hz": 2.0, "probe_indices": [1, 1],
            }])
            with self.assertRaisesRegex(ProjectConfigurationError, "probe_indices must be unique"):
                load_project(root)

    def test_probe_plotting_defaults_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root, analyses=[{
                "id": "spectrum", "recipe": "probe_spectrum",
                "probe_set_id": "default", "variable": "pressure",
            }])
            project = load_project(root)
            plotting = project.analyses_file.analyses[0].probe_plotting
            self.assertEqual(plotting.mode, "both")
            self.assertEqual(plotting.normalization, "none")
            self.assertEqual(plotting.label, "index_coordinates")
            bad = yaml.safe_load((root / "analyses.yaml").read_text())
            bad["analyses"][0]["probe_plotting"] = {"normalization": "bad"}
            (root / "analyses.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(ProjectConfigurationError, "normalization"):
                load_project(root)

    def test_temporal_wavenumber_requires_duration_and_rejects_duplicate_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = {
                "id": "wave", "recipe": "directional_wave", "probe_set_id": "default",
                "variable": "pressure", "frequency_max_hz": 100_000,
                "temporal_wavenumber": {"enabled": True},
            }
            self.write_project(root, analyses=[analysis])
            with self.assertRaisesRegex(ProjectConfigurationError, "window_duration_s"):
                load_project(root)
            analysis["temporal_wavenumber"] = {
                "enabled": True, "window_duration_s": 1.0e-4,
                "snapshot_times_s": [1.0, 1.0],
            }
            (root / "analyses.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "analyses": [analysis]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProjectConfigurationError, "snapshot_times_s"):
                load_project(root)

    def test_volume_fraction_smoothing_window_is_odd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root)
            case = dict(CASE)
            case["geometry"] = {
                "type": "volume_fraction", "field": "vfrac", "smoothing_window": 4,
            }
            (root / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
            with self.assertRaisesRegex(ProjectConfigurationError, "smoothing_window must be odd"):
                load_project(root)

    def test_flat_plate_and_polyline_geometry_contracts_fail_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_project(root)
            case = dict(CASE)
            case["geometry"] = {
                "type": "flat_plate", "leading_edge_x_m": 1.0,
                "trailing_edge_x_m": 0.5,
            }
            (root / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
            with self.assertRaisesRegex(ProjectConfigurationError, "trailing_edge"):
                load_project(root)
            case["geometry"] = {
                "type": "polyline", "points_m": [[0.0, 0.0], [1.0, 0.0]],
                "closed": False, "fluid_side": "outside",
            }
            (root / "case.yaml").write_text(yaml.safe_dump(case), encoding="utf-8")
            with self.assertRaisesRegex(ProjectConfigurationError, "left/right"):
                load_project(root)

    def test_missing_machine_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "case.yaml").write_text(yaml.safe_dump(CASE), encoding="utf-8")
            (root / "analyses.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "analyses": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProjectConfigurationError, "machine.example.yaml"):
                load_project(root)

    def test_schema_is_generated_for_all_three_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "schema.json"
            write_project_schema(destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn('"case.yaml"', text)
            self.assertIn('"analyses.yaml"', text)
            self.assertIn('"machine.yaml"', text)
            self.assertIn('"probe_plotting"', text)

    def test_cli_initializes_and_validates_project(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            initialized = runner.invoke(app, ["init", str(root)])
            self.assertEqual(initialized.exit_code, 0, initialized.output)
            self.assertTrue((root / "case.yaml").is_file())
            self.assertTrue((root / "project.schema.json").is_file())
            validated = runner.invoke(app, ["validate", str(root)])
            self.assertEqual(validated.exit_code, 0, validated.output)
            self.assertIn("Valid project", validated.output)


if __name__ == "__main__":
    unittest.main()
