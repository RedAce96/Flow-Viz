from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from pelecpost.config.loader import load_project
from pelecpost.preflight import create_plan
from pelecpost.runtime import run_project


class PlotfileExecutorTests(unittest.TestCase):
    def project(self, root: Path, analysis: dict):
        plot = root / "inputs" / "plt00010"
        plot.mkdir(parents=True)
        fields = ["density", "x_velocity", "y_velocity", "pressure", "temperature"]
        header = [
            "HyperCLaw-V1.1", str(len(fields)), *fields, "2", "1.0e-6", "0",
            "0.0 -3.0", "10.0 3.0",
        ]
        (plot / "Header").write_text("\n".join(header) + "\n", encoding="utf-8")
        case = {
            "schema_version": 1,
            "case": {"id": "plot-case", "solver": "pelec", "dimensionality": 2,
                     "solver_units": "cgs"},
            "gas": {"gamma": 1.4, "gas_constant_j_kg_k": 287.05},
            "freestream": {"source": "explicit", "density_kg_m3": 1.0,
                           "velocity_m_s": 10.0, "pressure_pa": 100.0,
                           "temperature_k": 300.0},
            "geometry": {"type": "flat_plate", "leading_edge_x_m": 0.001,
                         "trailing_edge_x_m": 0.099, "wall_y_m": 0.0,
                         "fluid_side": "above"},
        }
        machine = {
            "schema_version": 1,
            "inputs": {"plotfiles": {"source": str(root / "inputs"), "prefix": "plt"}},
            "outputs": {"root": str(root / "runs")},
            "compute": {"workers": 1, "memory_limit_gb": 8},
        }
        for name, value in (("case.yaml", case),
                            ("analyses.yaml", {"schema_version": 1, "analyses": [analysis]}),
                            ("machine.yaml", machine)):
            (root / name).write_text(yaml.safe_dump(value), encoding="utf-8")
        return load_project(root)

    @staticmethod
    def dataset():
        x = np.linspace(0.001, 0.099, 64)
        y = np.linspace(-0.03, 0.03, 96)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return {
            "source": "synthetic", "plot_label": "plt00010", "time": 1.0e-6,
            "x": x, "y": y, "amr_max_level": 0, "domain_length_x": 0.1,
            "fields": {
                "density": np.ones_like(xx), "x_velocity": 10.0 + 20.0 * yy,
                "y_velocity": np.zeros_like(xx), "pressure": 100.0 + 5.0 * yy,
                "temperature": 300.0 + 10.0 * yy,
            },
        }

    def test_flow_overview_registers_contour_without_global_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(Path(temporary), {
                "id": "overview", "recipe": "flow_overview", "fields": ["temperature"],
            })
            with patch("pp_functions_database.load_pelec_plotfile", return_value=self.dataset()):
                result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            self.assertIn("overview.field.contours.plt00010.temperature",
                          {item["id"] for item in artifacts["artifacts"]})

    def test_preflight_reports_si_domain_and_per_analysis_snapshot_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.project(root, {
                "id": "overview", "recipe": "flow_overview", "fields": ["temperature"],
                "snapshot_start": 20, "snapshot_end": 30, "snapshot_step": 10,
            })
            header = root / "inputs" / "plt00010" / "Header"
            for suffix in (20, 30, 40):
                candidate = root / "inputs" / f"plt{suffix:05d}"
                candidate.mkdir()
                (candidate / "Header").write_text(header.read_text(), encoding="utf-8")
            plan = create_plan(load_project(root))
            self.assertEqual(plan.inventory.plotfiles.domain_bounds_m, ((0.0, 0.1), (-0.03, 0.03)))
            self.assertEqual(
                plan.sampling["selections"]["plotfiles"]["overview"],
                ["plt00020", "plt00030"],
            )

    def test_surface_diagnostics_registers_curve_samples_and_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(Path(temporary), {
                "id": "wall", "recipe": "surface_diagnostics",
                "normal_sample_distance_m": 0.01, "normal_sample_points": 8,
            })
            with patch("pp_functions_database.load_pelec_plotfile", return_value=self.dataset()):
                result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            ids = {item["id"] for item in artifacts["artifacts"]}
            self.assertIn("wall.surface.curve.plt00010.0", ids)
            self.assertIn("wall.surface.samples.plt00010.0", ids)
            self.assertIn("wall.surface.quality.plt00010.0", ids)

    def test_general_wedge_forces_use_validated_designation_and_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.project(root, {
                "id": "loads", "recipe": "aerodynamic_forces",
                "reference_chord_m": 0.05, "reference_span_m": 1.0,
                "dynamic_viscosity_pa_s": 1.0e-5,
                "conductivity_w_m_k": 0.02,
                "normal_sample_distance_m": 0.005, "normal_sample_points": 8,
            })
            case_path = root / "case.yaml"
            case = yaml.safe_load(case_path.read_text())
            case["geometry"] = {
                "type": "wedge", "leading_edge_x_m": 0.01,
                "leading_edge_y_m": 0.0, "length_m": 0.05,
                "half_angle_deg": 10.0, "fluid_side": "outside",
            }
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            project = load_project(root)
            with patch("pp_functions_database.load_pelec_plotfile", return_value=self.dataset()):
                result = run_project(project)
            self.assertEqual(result.status, "completed")
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            by_id = {item["id"]: item for item in artifacts["artifacts"]}
            self.assertIn("loads.forces.history", by_id)
            component = by_id["loads.forces.components.plt00010"]
            self.assertEqual(component["provenance"]["designation"], "validated_2d_eb_v1")
            sensitivity = json.loads((result.run_dir / "data/loads/force_sensitivity.json").read_text())
            self.assertIn("force_delta_n_m", sensitivity["fit_sensitivity"][0])

    def test_volume_fraction_forces_quantify_unsmoothed_geometry_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.project(root, {
                "id": "loads", "recipe": "aerodynamic_forces",
                "reference_chord_m": 0.05, "reference_span_m": 1.0,
                "dynamic_viscosity_pa_s": 1.0e-5,
                "conductivity_w_m_k": 0.02,
                "normal_sample_distance_m": 0.004, "normal_sample_points": 8,
            })
            case_path = root / "case.yaml"
            case = yaml.safe_load(case_path.read_text())
            case["geometry"] = {
                "type": "volume_fraction", "field": "volume_fraction",
                "minimum_component_points": 12, "smoothing_window": 3,
            }
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            header_path = root / "inputs" / "plt00010" / "Header"
            header = header_path.read_text().splitlines()
            field_count = int(header[1])
            header[1] = str(field_count + 1)
            header.insert(2 + field_count, "volume_fraction")
            header_path.write_text("\n".join(header) + "\n", encoding="utf-8")
            dataset = self.dataset()
            xx, yy = np.meshgrid(dataset["x"], dataset["y"], indexing="ij")
            dataset["fields"]["volume_fraction"] = (
                (xx - 0.05) ** 2 + yy**2 >= 0.01**2
            ).astype(float)
            with patch("pp_functions_database.load_pelec_plotfile", return_value=dataset):
                result = run_project(load_project(root))
            self.assertEqual(result.status, "completed")
            sensitivity = json.loads(
                (result.run_dir / "data/loads/force_sensitivity.json").read_text()
            )["fit_sensitivity"][0]
            self.assertEqual(sensitivity["geometry_smoothing_window"], 3)
            self.assertEqual(len(sensitivity["unsmoothed_force_delta_n_m"]), 2)
            with np.load(
                result.run_dir / "data/loads/plt00010_validated_2d_eb_forces.npz"
            ) as arrays:
                self.assertIn("component_000_unsmoothed_coordinates_m", arrays.files)

    def test_static_baseline_is_explicit_and_produces_registered_increment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.project(root, {
                "id": "loads", "recipe": "aerodynamic_forces",
                "reference_chord_m": 0.05, "dynamic_viscosity_pa_s": 1.0e-5,
                "conductivity_w_m_k": 0.02,
            })
            case_path = root / "case.yaml"
            case = yaml.safe_load(case_path.read_text())
            case["geometry"] = {
                "type": "wedge", "leading_edge_x_m": 0.01,
                "leading_edge_y_m": 0.0, "length_m": 0.05,
                "half_angle_deg": 10.0, "fluid_side": "outside",
            }
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            analyses_path = root / "analyses.yaml"
            analyses = yaml.safe_load(analyses_path.read_text())
            analyses["analyses"][0].update({"baseline": "static", "baseline_id": "quiet"})
            analyses_path.write_text(yaml.safe_dump(analyses), encoding="utf-8")
            baseline_plot = root / "baseline" / "pltQuiet"
            baseline_plot.mkdir(parents=True)
            current_header = root / "inputs" / "plt00010" / "Header"
            (baseline_plot / "Header").write_text(current_header.read_text(), encoding="utf-8")
            machine_path = root / "machine.yaml"
            machine = yaml.safe_load(machine_path.read_text())
            machine["inputs"]["baselines"] = {
                "quiet": {"source": str(root / "baseline"), "prefix": "plt"}
            }
            machine_path.write_text(yaml.safe_dump(machine), encoding="utf-8")
            dataset = self.dataset()
            with patch("pp_functions_database.load_pelec_plotfile", return_value=dataset):
                result = run_project(load_project(root))
            self.assertEqual(result.status, "completed")
            with np.load(result.run_dir / "data/loads/plt00010_validated_2d_eb_forces.npz") as data:
                np.testing.assert_allclose(data["force_total_n_m"], 0.0, atol=1.0e-10)
            artifacts = json.loads((result.run_dir / "artifacts.json").read_text())
            item = next(entry for entry in artifacts["artifacts"]
                        if entry["id"] == "loads.forces.components.plt00010")
            self.assertEqual(item["provenance"]["designation"], "validated_2d_eb_v1_increment")


if __name__ == "__main__":
    unittest.main()
