"""SI derived-field utilities with explicit freestream provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pelecpost.config.models import ExplicitFreestream, RegionFreestream


@dataclass(frozen=True)
class FreestreamReference:
    density_kg_m3: float
    velocity_m_s: float
    pressure_pa: float
    temperature_k: float
    provenance: dict


def resolve_freestream_reference(dataset: dict, config: ExplicitFreestream | RegionFreestream) -> FreestreamReference:
    if isinstance(config, ExplicitFreestream):
        return FreestreamReference(
            config.density_kg_m3, config.velocity_m_s, config.pressure_pa,
            config.temperature_k, {"source": "explicit", "values_from": "case.yaml"},
        )
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    x0, x1 = config.bounds.x_m
    y0, y1 = config.bounds.y_m
    x_mask = (x >= x0) & (x <= x1)
    y_mask = (y >= y0) & (y <= y1)
    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("configured freestream region contains no grid centers")
    operation = np.nanmedian if config.statistic == "median" else np.nanmean
    fields = dataset["fields"]

    def statistic(name: str) -> float:
        values = np.asarray(fields[name], dtype=float)[np.ix_(x_mask, y_mask)]
        result = float(operation(values))
        if not np.isfinite(result):
            raise ValueError(f"freestream region produced no finite {name} reference")
        return result

    reference = FreestreamReference(
        statistic("density"), statistic("x_velocity"), statistic("pressure"),
        statistic("temperature"),
        {
            "source": "region", "bounds_x_m": [x0, x1], "bounds_y_m": [y0, y1],
            "statistic": config.statistic,
            "selected_cell_count": int(np.count_nonzero(x_mask) * np.count_nonzero(y_mask)),
        },
    )
    if min(reference.density_kg_m3, reference.pressure_pa, reference.temperature_k) <= 0:
        raise ValueError("freestream density, pressure, and temperature must be positive")
    if abs(reference.velocity_m_s) == 0:
        raise ValueError("freestream velocity magnitude must be nonzero")
    return reference


def add_normalized_fields(dataset: dict, reference: FreestreamReference, gamma: float = 1.4) -> dict:
    fields = dataset["fields"]
    fields["rho/rhoinf"] = np.asarray(fields["density"]) / reference.density_kg_m3
    fields["U/Uinf"] = np.asarray(fields["x_velocity"]) / reference.velocity_m_s
    fields["P/Pinf"] = np.asarray(fields["pressure"]) / reference.pressure_pa
    dynamic_pressure = 0.5 * reference.density_kg_m3 * reference.velocity_m_s**2
    fields["P/P_dyn"] = np.asarray(fields["pressure"]) / dynamic_pressure
    dataset["freestream_reference"] = {
        **reference.provenance,
        "density_kg_m3": reference.density_kg_m3,
        "velocity_m_s": reference.velocity_m_s,
        "pressure_pa": reference.pressure_pa,
        "temperature_k": reference.temperature_k,
        "gamma": gamma,
    }
    return dataset
