"""General two-dimensional wall fits and force/moment integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pelecpost.geometry import SurfaceCurve2D


@dataclass(frozen=True)
class WallFit2D:
    pressure_pa: np.ndarray
    tangential_velocity_gradient_s: np.ndarray
    temperature_gradient_k_m: np.ndarray
    pressure_residual_pa: np.ndarray
    velocity_residual_m_s: np.ndarray
    temperature_residual_k: np.ndarray
    point_count: np.ndarray


@dataclass(frozen=True)
class SurfaceLoad2D:
    pressure_traction_n_m2: np.ndarray
    viscous_traction_n_m2: np.ndarray
    heat_flux_w_m2: np.ndarray
    force_pressure_n_m: np.ndarray
    force_viscous_n_m: np.ndarray
    force_total_n_m: np.ndarray
    moment_pressure_n: float
    moment_viscous_n: float
    moment_total_n: float
    designation: str


def _linear_fit(distance: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(distance) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("at least two finite normal samples are required")
    coefficients, residuals, _, _, _ = np.polyfit(distance[valid], values[valid], 1, full=True)
    fitted = np.polyval(coefficients, distance[valid])
    rms = float(np.sqrt(np.mean((values[valid] - fitted) ** 2)))
    return float(coefficients[0]), float(coefficients[1]), rms


def fit_wall_quantities(
    surface: SurfaceCurve2D,
    normal_distance_m: np.ndarray,
    pressure_pa: np.ndarray,
    velocity_m_s: np.ndarray,
    temperature_k: np.ndarray,
) -> WallFit2D:
    distance = np.asarray(normal_distance_m, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    count = len(surface.coordinates_m)
    if pressure.shape != (count, len(distance)) or temperature.shape != pressure.shape:
        raise ValueError("scalar samples must have shape (surface points, normal samples)")
    if velocity.shape != (count, len(distance), 2):
        raise ValueError("velocity samples must have shape (surface points, normal samples, 2)")
    wall_pressure = np.empty(count)
    velocity_gradient = np.empty(count)
    temperature_gradient = np.empty(count)
    pressure_residual = np.empty(count)
    velocity_residual = np.empty(count)
    temperature_residual = np.empty(count)
    point_count = np.empty(count, dtype=int)
    for index in range(count):
        tangential = velocity[index] @ surface.tangent[index]
        _, wall_pressure[index], pressure_residual[index] = _linear_fit(distance, pressure[index])
        velocity_gradient[index], _, velocity_residual[index] = _linear_fit(distance, tangential)
        temperature_gradient[index], _, temperature_residual[index] = _linear_fit(distance, temperature[index])
        point_count[index] = int(np.count_nonzero(
            np.isfinite(pressure[index]) & np.isfinite(tangential) & np.isfinite(temperature[index])
        ))
    return WallFit2D(
        wall_pressure, velocity_gradient, temperature_gradient,
        pressure_residual, velocity_residual, temperature_residual, point_count,
    )


def integrate_surface_loads(
    surface: SurfaceCurve2D,
    pressure_pa: np.ndarray,
    tangential_velocity_gradient_s: np.ndarray,
    temperature_gradient_k_m: np.ndarray,
    *,
    dynamic_viscosity_pa_s: float,
    conductivity_w_m_k: float,
    moment_origin_m: tuple[float, float] = (0.0, 0.0),
    pressure_reference_pa: float = 0.0,
    designation: str | None = None,
) -> SurfaceLoad2D:
    points = surface.coordinates_m
    count = len(points)
    pressure = np.asarray(pressure_pa, dtype=float)
    gradient = np.asarray(tangential_velocity_gradient_s, dtype=float)
    thermal = np.asarray(temperature_gradient_k_m, dtype=float)
    if pressure.shape != (count,) or gradient.shape != (count,) or thermal.shape != (count,):
        raise ValueError("wall arrays must have one value per surface point")
    if dynamic_viscosity_pa_s <= 0 or conductivity_w_m_k <= 0:
        raise ValueError("transport properties must be positive")
    if designation is None:
        designation = (
            "stationary_one_sided_flat_plate"
            if surface.source == "flat_plate" else "validated_2d_eb_v1"
        )
    next_indices = np.roll(np.arange(count), -1) if surface.closed else np.arange(1, count)
    first_indices = np.arange(count) if surface.closed else np.arange(count - 1)
    pressure_mid = 0.5 * (pressure[first_indices] + pressure[next_indices]) - pressure_reference_pa
    shear_mid = dynamic_viscosity_pa_s * 0.5 * (gradient[first_indices] + gradient[next_indices])
    thermal_mid = 0.5 * (thermal[first_indices] + thermal[next_indices])
    pressure_traction = -pressure_mid[:, None] * surface.segment_fluid_normal
    viscous_traction = shear_mid[:, None] * surface.segment_tangent
    segment_pressure_force = pressure_traction * surface.segment_length_m[:, None]
    segment_viscous_force = viscous_traction * surface.segment_length_m[:, None]
    pressure_force = np.sum(segment_pressure_force, axis=0)
    viscous_force = np.sum(segment_viscous_force, axis=0)
    midpoint = 0.5 * (points[first_indices] + points[next_indices])
    arm = midpoint - np.asarray(moment_origin_m, dtype=float)
    pressure_moment = float(np.sum(arm[:, 0] * segment_pressure_force[:, 1] - arm[:, 1] * segment_pressure_force[:, 0]))
    viscous_moment = float(np.sum(arm[:, 0] * segment_viscous_force[:, 1] - arm[:, 1] * segment_viscous_force[:, 0]))
    return SurfaceLoad2D(
        pressure_traction_n_m2=pressure_traction,
        viscous_traction_n_m2=viscous_traction,
        heat_flux_w_m2=-conductivity_w_m_k * thermal_mid,
        force_pressure_n_m=pressure_force,
        force_viscous_n_m=viscous_force,
        force_total_n_m=pressure_force + viscous_force,
        moment_pressure_n=pressure_moment,
        moment_viscous_n=viscous_moment,
        moment_total_n=pressure_moment + viscous_moment,
        designation=designation,
    )
