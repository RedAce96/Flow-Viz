from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

from pelecpost.config.loader import load_project
from pelecpost.preflight import Severity, create_plan
from pelecpost.workflows import RECIPES, build_workflow_graph
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
