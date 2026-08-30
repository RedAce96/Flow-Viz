from __future__ import annotations

import tempfile
import unittest
import importlib
from pathlib import Path

import h5py
import numpy as np
import yaml

from pelecpost.config.loader import load_project
from pelecpost.preflight import Severity, create_plan
from pelecpost.workflows import (
    INTERNAL_WORKFLOWS,
    RECIPES,
    WORKFLOWS,
    WorkflowProtocol,
    build_workflow_graph,
)
from pelecpost.workflows.graph import WorkflowNode, _topological_order


def write_compact(path: Path, *, samples: int = 1024, spacing_cm: float = 1.0) -> None:
    with h5py.File(path, "w") as archive:
        archive.attrs["schema_name"] = "pelec.compact-probes"
        archive.attrs["schema_version"] = 1
        archive.attrs["quality_approved"] = True
        archive.create_dataset("time", data=np.arange(samples) * 1.0e-6)
        archive.create_dataset("step", data=np.arange(samples))
        probes = archive.create_group("probes")
        probes.create_dataset("requested_x_cm", data=np.arange(8) * spacing_cm)
        probes.create_dataset("requested_y_cm", data=np.zeros(8))
        fields = archive.create_group("fields")
        for field, unit in (("p", "dyne/cm^2"), ("T", "K")):
            dataset = fields.create_dataset(field, shape=(samples, 8), dtype=float)
            dataset.attrs["unit"] = unit
            time = np.arange(samples) * 1.0e-6
            phase = np.arange(8)[None, :] * 0.2
            dataset[:] = np.sin(2.0 * np.pi * 25_000.0 * time[:, None] - phase)
        mapping = archive.create_group("mapping")
        mapping.create_dataset("epoch_start", data=[0])
        source = archive.create_group("source")
        source.create_dataset("files_json", data="[]")


class PreflightTests(unittest.TestCase):
    def project(self, root: Path, analyses: list[dict], *, dimension: int = 2):
        compact = root / "probes.h5"
        write_compact(compact)
        case = {
            "schema_version": 1,
            "case": {
                "id": "preflight-case", "solver": "pelec",
                "dimensionality": dimension, "solver_units": "cgs",
            },
            "gas": {"gamma": 1.4, "gas_constant_j_kg_k": 287.05},
            "freestream": {
                "source": "explicit", "density_kg_m3": 0.02,
                "velocity_m_s": 1000.0, "pressure_pa": 1000.0,
                "temperature_k": 175.0,
            },
            "geometry": {"type": "flat_plate"},
        }
        machine = {
            "schema_version": 1,
            "inputs": {"probes": {"compact_file": str(compact)}},
            "outputs": {"root": str(root / "outputs")},
            "compute": {"workers": 2, "memory_limit_gb": 8},
        }
        for name, value in (
            ("case.yaml", case),
            ("analyses.yaml", {"schema_version": 1, "analyses": analyses}),
            ("machine.yaml", machine),
        ):
            (root / name).write_text(yaml.safe_dump(value), encoding="utf-8")
        return load_project(root)

    def test_catalog_contains_all_public_recipes(self):
        self.assertEqual(len(RECIPES), 11)
        self.assertTrue(all(item.outputs for item in RECIPES.values()))
        self.assertTrue(all(item.limitations for item in RECIPES.values()))
        self.assertEqual(set(RECIPES), set(WORKFLOWS))
        self.assertTrue(all(isinstance(item, WorkflowProtocol) for item in WORKFLOWS.values()))
        self.assertTrue(all(item.artifact_declarations for item in WORKFLOWS.values()))
        self.assertEqual(
            INTERNAL_WORKFLOWS,
            {"input.plotfiles", "input.probes", "input.comparison_archives", "geometry.surface"},
        )

    def test_every_public_workflow_has_a_lazy_executor_registration(self):
        from pelecpost.analysis.executors import EXECUTORS

        for module in ("spectral", "modal", "transient", "nonlinear", "plotfiles", "comparison"):
            importlib.import_module(f"pelecpost.analysis.{module}")
        self.assertEqual(set(EXECUTORS), set(WORKFLOWS))

    def test_probe_only_graph_has_no_plotfile_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(
                Path(temporary),
                [{"id": "spectrum", "recipe": "probe_spectrum", "variable": "temperature"}],
            )
            graph = build_workflow_graph(project)
            self.assertEqual(graph.order, ("input.probes", "analysis.spectrum"))
            plan = create_plan(project)
            self.assertFalse(plan.blockers)
            contract = plan.analysis_contracts["spectrum"]
            self.assertIn("stationary", contract["assumptions"][0].lower())
            self.assertEqual(contract["expected_artifact_ids"], list(RECIPES["probe_spectrum"].outputs))

    def test_force_probe_linkage_adds_probe_input_and_artifact_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self.project(root, [{
                "id": "loads", "recipe": "aerodynamic_forces",
                "reference_chord_m": 0.1,
                "dynamic_viscosity_pa_s": 1.0e-5,
                "conductivity_w_m_k": 0.02,
                "probe_linkage": {
                    "variable": "pressure", "force_component": "x",
                    "forcing_frequency_hz": 25_000,
                },
            }])
            plot = root / "plotfiles" / "plt00010"
            plot.mkdir(parents=True)
            header = [
                "HyperCLaw-V1.1", "5", "density", "x_velocity", "y_velocity",
                "pressure", "temperature", "2", "1.0e-6", "0", "0.0 -3.0",
                "10.0 3.0",
            ]
            (plot / "Header").write_text("\n".join(header) + "\n", encoding="utf-8")
            machine_path = root / "machine.yaml"
            machine = yaml.safe_load(machine_path.read_text())
            machine["inputs"]["plotfiles"] = {
                "source": str(root / "plotfiles"), "prefix": "plt",
            }
            machine_path.write_text(yaml.safe_dump(machine), encoding="utf-8")
            project = load_project(root)
            graph = build_workflow_graph(project)
            self.assertEqual(
                set(graph.node("analysis.loads").dependencies),
                {"input.plotfiles", "input.probes", "geometry.surface"},
            )
            plan = create_plan(project)
            self.assertIn(
                "forces.probe_linkage",
                plan.analysis_contracts["loads"]["expected_artifact_ids"],
            )

    def test_control_volume_is_rejected_for_non_flat_plate_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.project(root, [{
                "id": "loads", "recipe": "aerodynamic_forces",
                "reference_chord_m": 0.1,
                "dynamic_viscosity_pa_s": 1.0e-5,
                "conductivity_w_m_k": 0.02,
                "control_volume": {"x_range_m": [0.0, 0.1], "y_top_m": 0.02},
            }])
            case_path = root / "case.yaml"
            case = yaml.safe_load(case_path.read_text())
            case["geometry"] = {
                "type": "wedge", "leading_edge_x_m": 0.0,
                "leading_edge_y_m": 0.0, "length_m": 0.1,
                "half_angle_deg": 10.0, "fluid_side": "outside",
            }
            case_path.write_text(yaml.safe_dump(case), encoding="utf-8")
            plan = create_plan(load_project(root))
            self.assertIn(
                "CONTROL_VOLUME_FLAT_PLATE_ONLY",
                {item.code for item in plan.blockers},
            )

    def test_above_nyquist_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(
                Path(temporary),
                [{
                    "id": "spectrum", "recipe": "probe_spectrum",
                    "variable": "temperature", "frequency_max_hz": 600_000,
                }],
            )
            plan = create_plan(project)
            self.assertIn("ABOVE_NYQUIST", {item.code for item in plan.blockers})

    def test_finite_record_with_no_resolvable_requested_bin_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(
                Path(temporary),
                [{
                    "id": "spectrum", "recipe": "probe_spectrum",
                    "variable": "temperature", "frequency_max_hz": 500,
                }],
            )
            plan = create_plan(project)
            self.assertIn("NO_RESOLVABLE_FREQUENCY_BIN", {item.code for item in plan.blockers})

    def test_three_dimensional_input_is_rejected_at_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(
                Path(temporary),
                [{"id": "spectrum", "recipe": "probe_spectrum", "variable": "temperature"}],
                dimension=3,
            )
            plan = create_plan(project)
            finding = next(item for item in plan.blockers if item.code == "UNSUPPORTED_DIMENSION")
            self.assertIn("3-D extension interfaces", finding.message)

    def test_plotfile_dimension_must_match_case_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.project(root, [{
                "id": "overview", "recipe": "flow_overview",
                "fields": ["temperature"],
            }])
            plot = root / "plotfiles" / "plt00010"
            plot.mkdir(parents=True)
            header = [
                "HyperCLaw-V1.1", "5", "density", "x_velocity", "y_velocity",
                "pressure", "temperature", "3", "1.0e-6", "0",
                "0.0 -3.0 -2.0", "10.0 3.0 2.0",
            ]
            (plot / "Header").write_text("\n".join(header) + "\n", encoding="utf-8")
            machine_path = root / "machine.yaml"
            machine = yaml.safe_load(machine_path.read_text())
            machine["inputs"]["plotfiles"] = {
                "source": str(root / "plotfiles"), "prefix": "plt",
            }
            machine_path.write_text(yaml.safe_dump(machine), encoding="utf-8")
            plan = create_plan(load_project(root))
            finding = next(
                item for item in plan.blockers
                if item.code == "PLOTFILE_DIMENSION_MISMATCH"
            )
            self.assertIn("case.yaml declares 2-D", finding.message)

    def test_directional_wave_reports_spatial_aliasing(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(
                Path(temporary),
                [{
                    "id": "wave", "recipe": "directional_wave", "variable": "pressure",
                    "frequency_max_hz": 100_000, "expected_speed_min_m_s": 100,
                }],
            )
            plan = create_plan(project)
            self.assertIn("SPATIAL_ALIASING", {item.code for item in plan.blockers})

    def test_cycle_is_rejected(self):
        nodes = {
            "a": WorkflowNode("a", "a", None, True, ("b",)),
            "b": WorkflowNode("b", "b", None, True, ("a",)),
        }
        with self.assertRaisesRegex(Exception, "cycle"):
            _topological_order(nodes)

    def test_plan_serializes_severity_as_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.project(Path(temporary), [])
            plan = create_plan(project)
            self.assertEqual(plan.findings[0].severity, Severity.INFO)
            self.assertEqual(plan.as_dict()["findings"][0]["severity"], Severity.INFO)


if __name__ == "__main__":
    unittest.main()
