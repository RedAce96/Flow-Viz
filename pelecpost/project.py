"""Project initialization and template management."""

from __future__ import annotations

from pathlib import Path

from pelecpost.config.loader import dump_yaml, load_project, write_project_schema
from pelecpost.errors import ProjectConfigurationError


CASE_TEMPLATE = {
    "schema_version": 1,
    "case": {
        "id": "new-pelec-case",
        "solver": "pelec",
        "dimensionality": 2,
        "solver_units": "cgs",
        "description": "Describe the physical case and analysis purpose.",
    },
    "gas": {"gamma": 1.4, "gas_constant_j_kg_k": 287.05},
    "freestream": {
        "source": "explicit",
        "density_kg_m3": 0.02,
        "velocity_m_s": 1000.0,
        "pressure_pa": 1000.0,
        "temperature_k": 175.0,
    },
    "geometry": {
        "type": "flat_plate",
        "leading_edge_x_m": 0.0,
        "wall_y_m": 0.0,
        "fluid_side": "above",
    },
}

ANALYSES_TEMPLATE = {"schema_version": 1, "analyses": []}

MACHINE_TEMPLATE = {
    "schema_version": 1,
    "inputs": {},
    "outputs": {"root": "runs"},
    "compute": {"workers": 1, "memory_limit_gb": 8.0, "fft_batch_size": 32},
}


def initialize_project(project_dir: str | Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ProjectConfigurationError(
            f"Refusing to initialize non-empty directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    dump_yaml(root / "case.yaml", CASE_TEMPLATE)
    dump_yaml(root / "analyses.yaml", ANALYSES_TEMPLATE)
    dump_yaml(root / "machine.yaml", MACHINE_TEMPLATE)
    dump_yaml(root / "machine.example.yaml", MACHINE_TEMPLATE)
    (root / "runs").mkdir(exist_ok=True)
    write_project_schema(root / "project.schema.json")
    load_project(root)
    return root

