"""Plotfile recipe executors using reviewed PeleC/AMReX adapters."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pp_functions_database as fields_api
import pp_plotting_database as plotting_api
from pelecpost.config.models import ExplicitFreestream
from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.geometry import (
    flat_plate_surface,
    polyline_surface,
    volume_fraction_surfaces,
    wedge_surfaces,
)
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .fields import add_normalized_fields, resolve_freestream_reference
from .surface_forces import fit_wall_quantities, integrate_surface_loads


def _source(context: WorkflowContext) -> tuple[Path, str]:
    config = context.project.machine_file.inputs.plotfiles
    if config is None:
        raise FileNotFoundError("plotfiles are not configured")
    source = config.source if config.source.is_absolute() else context.project.root / config.source
    return source.resolve(), config.prefix


def _paths(context: WorkflowContext) -> list[str]:
    source, prefix = _source(context)
    analysis = context.analysis
    return fields_api.discover_plotfile_paths(
        source, plot_prefix=prefix,
        start=getattr(analysis, "snapshot_start", None),
        end=getattr(analysis, "snapshot_end", None),
        step=getattr(analysis, "snapshot_step", 1),
    )


def _region(analysis) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if getattr(analysis, "x_limits_m", None) is None or getattr(analysis, "y_limits_m", None) is None:
        return None
    return analysis.x_limits_m, analysis.y_limits_m


def _load(context: WorkflowContext, path: str, requested: set[str], region=None) -> dict:
    derived = {"mach_number", "schlieren", "vorticity", "vorticity_magnitude"}
    base = set(requested) - derived
    if requested & {"mach_number"}:
        base.update(("density", "pressure", "x_velocity", "y_velocity"))
    if requested & {"schlieren"}:
        base.add("density")
    if requested & {"vorticity", "vorticity_magnitude"}:
        base.update(("x_velocity", "y_velocity"))
    dataset = fields_api.load_pelec_plotfile(
        path, field_names=sorted(base), alias_map=context.project.case_file.field_aliases,
        convert_to_mks=True, derive_native_vorticity=bool(requested & {"vorticity", "vorticity_magnitude"}),
        region_bounds_m=region,
    )
    fields_api.compute_derived_fields(
        dataset, gamma=context.project.case_file.gas.gamma,
        R=context.project.case_file.gas.gas_constant_j_kg_k,
    )
    for stale in ("rho/rhoinf", "U/Uinf", "P/Pinf", "P/P_dyn"):
        dataset["fields"].pop(stale, None)
    needed_reference = {"density", "x_velocity", "pressure", "temperature"}
    if needed_reference.issubset(dataset["fields"]):
        reference = resolve_freestream_reference(dataset, context.project.case_file.freestream)
        add_normalized_fields(dataset, reference, gamma=context.project.case_file.gas.gamma)
    return dataset


def _surfaces(context: WorkflowContext, dataset: dict):
    geometry = context.project.case_file.geometry
    if geometry.type == "flat_plate":
        end = geometry.trailing_edge_x_m or float(np.max(dataset["x"]))
        return (flat_plate_surface(
            geometry.leading_edge_x_m, end, geometry.wall_y_m, geometry.fluid_side,
            points=max(32, min(1024, len(dataset["x"]))),
        ),)
    if geometry.type == "wedge":
        return wedge_surfaces(
            (geometry.leading_edge_x_m, geometry.leading_edge_y_m),
            geometry.length_m, geometry.half_angle_deg,
        )
    if geometry.type == "polyline":
        return (polyline_surface(
            np.asarray(geometry.points_m), closed=geometry.closed,
            fluid_side=geometry.fluid_side,
        ),)
    return volume_fraction_surfaces(
        dataset["x"], dataset["y"], dataset["fields"][geometry.field],
        iso_value=geometry.iso_value, fluid_value=geometry.fluid_value,
        minimum_component_points=geometry.minimum_component_points,
    )


def _sample_wall(dataset: dict, surface, maximum_distance_m: float, points: int):
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    first = 0.5 * min(float(np.median(np.diff(x))), float(np.median(np.diff(y))))
    if maximum_distance_m <= first:
        raise ValueError("normal sample distance must exceed half the local grid spacing")
    distance = np.linspace(first, maximum_distance_m, points)
    coordinates = (
        surface.coordinates_m[:, None, :]
        + distance[None, :, None] * surface.fluid_normal[:, None, :]
    )
    flat = coordinates.reshape(-1, 2)

    def sample(name: str) -> np.ndarray:
        interpolator = RegularGridInterpolator(
            (x, y), np.asarray(dataset["fields"][name]),
            bounds_error=False, fill_value=np.nan,
        )
        return interpolator(flat).reshape(len(surface.coordinates_m), len(distance))

    pressure = sample("pressure")
    temperature = sample("temperature")
    velocity = np.stack((sample("x_velocity"), sample("y_velocity")), axis=2)
    return distance, pressure, velocity, temperature, fit_wall_quantities(
        surface, distance, pressure, velocity, temperature
    )


@executor("flow_overview")
def run_flow_overview(context: WorkflowContext) -> None:
    analysis = context.analysis
    output_fields = {item.value for item in analysis.fields}
    requested = set(output_fields)
    if analysis.streamlines:
        requested.update(("x_velocity", "y_velocity"))
    for plotfile in _paths(context):
        dataset = _load(context, plotfile, requested, _region(analysis))
        label = dataset["plot_label"]
        for field in sorted(output_fields):
            path = context.figure_dir / f"{label}_{field}.png"
            plotting_api.plot_contour(
                dataset, field, output_path=path, xlim=analysis.x_limits_m,
                ylim=analysis.y_limits_m,
                time_annotation=plotting_api.format_dataset_time(dataset),
            )
            context.register(
                artifact_id=f"field.contours.{label}.{field}", path=path, kind="figure",
                variable=field, units="SI; see field label", coordinate_metadata={"x": "m", "y": "m"},
                interpretation="Descriptive two-dimensional field view at one registered plotfile time.",
                provenance={"plotfile": plotfile, "freestream_reference": dataset.get("freestream_reference")},
            )
        for station in analysis.line_stations_x_m:
            profiles = []
            for field in sorted(output_fields):
                if field in dataset["fields"]:
                    profile = fields_api.extract_line(dataset, station, field)
                    profile["label"] = field
                    profiles.append(profile)
            if profiles:
                path = context.figure_dir / f"{label}_line_x_{station:.6g}.png"
                plotting_api.plot_line_profiles(profiles, output_path=path)
                context.register(
                    artifact_id=f"field.lines.{label}.x-{station:.9g}", path=path,
                    kind="figure", variable=None, units="SI", coordinate_metadata={"y": "m"},
                    interpretation="Nearest-grid line samples for the configured streamwise station.",
                )
        if analysis.streamlines:
            streamline = fields_api.extract_streamline_field(dataset, color_key="velocity_magnitude")
            path = context.figure_dir / f"{label}_streamlines.png"
            plotting_api.plot_streamlines([streamline], output_path=path)
            context.register(
                artifact_id=f"field.streamlines.{label}", path=path, kind="figure",
                variable="velocity", units="m/s", coordinate_metadata={"x": "m", "y": "m"},
                interpretation="Steady streamlines of one instantaneous velocity field.",
            )


@executor("boundary_layer_reference")
def run_boundary_layer_reference(context: WorkflowContext) -> None:
    analysis = context.analysis
    for plotfile in _paths(context):
        profiles = fields_api.extract_native_flat_plate_boundary_layers(
            plotfile, analysis.stations_x_m, maximum_height_m=analysis.maximum_height_m,
            wall_y_m=context.project.case_file.geometry.wall_y_m,
            wall_temperature_k=analysis.wall_temperature_k,
            viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
            conductivity_w_m_k=analysis.conductivity_w_m_k,
        )
        label = Path(plotfile).name
        arrays: dict[str, np.ndarray] = {"station_count": np.array(len(profiles))}
        for index, profile in enumerate(profiles):
            for key, value in profile.items():
                candidate = np.asarray(value)
                if candidate.dtype != object:
                    arrays[f"station_{index:03d}_{key}"] = candidate
            gpi = fields_api.compute_gpi_criterion(
                profile["wall_distance_m"], profile["u_t_m_s"],
                profile["temperature_k"], profile["rho_kg_m3"],
                R_specific=context.project.case_file.gas.gas_constant_j_kg_k,
            )
            for key, value in gpi.items():
                candidate = np.asarray(value)
                if candidate.dtype != object:
                    arrays[f"station_{index:03d}_gpi_{key}"] = candidate
            freestream = context.project.case_file.freestream
            if isinstance(freestream, ExplicitFreestream):
                similarity = fields_api.compute_compressible_flat_plate_reference_profile(
                    profile["wall_distance_m"], profile["x_station_m"],
                    freestream.velocity_m_s, freestream.temperature_k,
                    freestream.density_kg_m3, T_wall=analysis.wall_temperature_k,
                    gamma=context.project.case_file.gas.gamma,
                    R=context.project.case_file.gas.gas_constant_j_kg_k,
                    mu=analysis.dynamic_viscosity_pa_s,
                    k=analysis.conductivity_w_m_k,
                )
                for key, value in similarity.items():
                    candidate = np.asarray(value)
                    if candidate.dtype != object:
                        arrays[f"station_{index:03d}_reference_{key}"] = candidate
        path = context.data_dir / f"{label}_boundary_layer.npz"
        np.savez_compressed(path, **arrays)
        context.register(
            artifact_id=f"boundary_layer.profiles.{label}", path=path, kind="array",
            variable="boundary_layer", units="SI", coordinate_metadata={"wall_distance": "m"},
            interpretation="Native-AMR flat-plate profiles and integral thicknesses under laminar ZPG assumptions.",
            provenance={"plotfile": plotfile, "assumptions": ["laminar", "zero pressure gradient"]},
        )


@executor("surface_diagnostics")
def run_surface_diagnostics(context: WorkflowContext) -> None:
    analysis = context.analysis
    geometry = context.project.case_file.geometry
    requested = {"pressure", "temperature", "x_velocity", "y_velocity"}
    if geometry.type == "volume_fraction":
        requested.add(geometry.field)
    for plotfile in _paths(context):
        dataset = _load(context, plotfile, requested)
        label = Path(plotfile).name
        for component, surface in enumerate(_surfaces(context, dataset)):
            distance, pressure, velocity, temperature, fit = _sample_wall(
                dataset, surface, analysis.normal_sample_distance_m,
                analysis.normal_sample_points,
            )
            stem = f"{label}_component_{component:03d}"
            curve_path = context.data_dir / f"{stem}_curve.npz"
            np.savez_compressed(
                curve_path, coordinates_m=surface.coordinates_m,
                unsmoothed_coordinates_m=(surface.unsmoothed_coordinates_m
                                          if surface.unsmoothed_coordinates_m is not None
                                          else surface.coordinates_m),
                arc_length_m=surface.arc_length_m, tangent=surface.tangent,
                fluid_normal=surface.fluid_normal,
            )
            context.register(
                artifact_id=f"surface.curve.{label}.{component}", path=curve_path,
                kind="array", variable="geometry", units="m",
                coordinate_metadata={"x": "m", "y": "m", "arc_length": "m"},
                interpretation="Ordered surface geometry with fluid-facing normals.",
                provenance={"source": surface.source, "confidence": surface.confidence},
            )
            samples_path = context.data_dir / f"{stem}_samples.npz"
            np.savez_compressed(
                samples_path, normal_distance_m=distance, pressure_pa=pressure,
                velocity_m_s=velocity, temperature_k=temperature,
                wall_pressure_pa=fit.pressure_pa,
                tangential_velocity_gradient_s=fit.tangential_velocity_gradient_s,
                temperature_gradient_k_m=fit.temperature_gradient_k_m,
            )
            context.register(
                artifact_id=f"surface.samples.{label}.{component}", path=samples_path,
                kind="array", variable="wall_state", units="SI",
                coordinate_metadata={"normal_distance": "m", "arc_length": "m"},
                interpretation="Normal samples and linear wall extrapolations; fit quality is separate.",
            )
            quality = {
                **surface.diagnostics,
                "pressure_residual_rms_pa": float(np.nanmedian(fit.pressure_residual_pa)),
                "velocity_residual_rms_m_s": float(np.nanmedian(fit.velocity_residual_m_s)),
                "temperature_residual_rms_k": float(np.nanmedian(fit.temperature_residual_k)),
                "accepted_point_fraction": float(np.mean(fit.point_count >= 2)),
            }
            quality_path = context.data_dir / f"{stem}_quality.json"
            quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
            context.register(
                artifact_id=f"surface.quality.{label}.{component}", path=quality_path,
                kind="json", variable="wall_state", units=None, coordinate_metadata={},
                interpretation="Geometry resolution, coverage, and wall-fit residual diagnostics.",
            )
@executor("aerodynamic_forces")
def run_aerodynamic_forces(context: WorkflowContext) -> None:
    analysis = context.analysis
    geometry = context.project.case_file.geometry
    freestream = context.project.case_file.freestream
    if not isinstance(freestream, ExplicitFreestream):
        raise UnsupportedCapabilityError(
            "Surface force extraction currently requires an explicit freestream reference"
        )
    reference = {
        "rho_inf": freestream.density_kg_m3, "u_inf": freestream.velocity_m_s,
        "p_inf": freestream.pressure_pa, "chord": analysis.reference_chord_m,
        "moment_origin": analysis.moment_origin_m,
    }
    history: list[dict] = []
    sensitivity: list[dict] = []
    for plotfile in _paths(context):
        label = Path(plotfile).name
        if geometry.type == "flat_plate":
            end = geometry.trailing_edge_x_m or (
                geometry.leading_edge_x_m + analysis.reference_chord_m
            )
            wall = fields_api.extract_native_flat_plate_wall(
                plotfile, x_range_m=(geometry.leading_edge_x_m, end), wall_y_m=geometry.wall_y_m,
                p_inf_pa=freestream.pressure_pa, rho_inf_kg_m3=freestream.density_kg_m3,
                u_inf_m_s=freestream.velocity_m_s,
                viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
            )
            load = fields_api.integrate_flat_plate_wall_forces(wall, reference)
            alternate_wall = {**wall, "p_wall_pa": wall["p_wall_linear_pa"]}
            alternate = fields_api.integrate_flat_plate_wall_forces(alternate_wall, reference)
            arrays = {}
            for prefix, payload in (("wall_", wall), ("load_", load)):
                for key, value in payload.items():
                    candidate = np.asarray(value)
                    if candidate.dtype != object:
                        arrays[prefix + key] = candidate
            path = context.data_dir / f"{label}_certified_flat_plate_forces.npz"
            np.savez_compressed(path, **arrays)
            force = np.array([load["D_total_N_m"], load["N_total_N_m"]])
            moment = load["M_total_N"]
            time_s = load["time"]
            designation = "stationary_one_sided_flat_plate"
            sensitivity.append({
                "plotfile": label, "pressure_fit_orders": [2, 1],
                "force_delta_n_m": [
                    alternate["D_total_N_m"] - force[0],
                    alternate["N_total_N_m"] - force[1],
                ],
                "moment_delta_n": alternate["M_total_N"] - moment,
            })
            interpretation = (
                "Reviewed stationary one-sided flat-plate pressure, viscous force, and moment."
            )
        else:
            requested = {"pressure", "temperature", "x_velocity", "y_velocity"}
            if geometry.type == "volume_fraction":
                requested.add(geometry.field)
            dataset = _load(context, plotfile, requested)
            components = []
            alternate_components = []
            arrays = {}
            for component, surface in enumerate(_surfaces(context, dataset)):
                _, _, _, _, fit = _sample_wall(
                    dataset, surface, analysis.normal_sample_distance_m,
                    analysis.normal_sample_points,
                )
                load_component = integrate_surface_loads(
                    surface, fit.pressure_pa, fit.tangential_velocity_gradient_s,
                    fit.temperature_gradient_k_m,
                    dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                    conductivity_w_m_k=analysis.conductivity_w_m_k,
                    moment_origin_m=analysis.moment_origin_m,
                    pressure_reference_pa=freestream.pressure_pa,
                )
                reduced_points = max(4, analysis.normal_sample_points // 2)
                _, _, _, _, reduced_fit = _sample_wall(
                    dataset, surface, analysis.normal_sample_distance_m, reduced_points,
                )
                alternate_component = integrate_surface_loads(
                    surface, reduced_fit.pressure_pa,
                    reduced_fit.tangential_velocity_gradient_s,
                    reduced_fit.temperature_gradient_k_m,
                    dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                    conductivity_w_m_k=analysis.conductivity_w_m_k,
                    moment_origin_m=analysis.moment_origin_m,
                    pressure_reference_pa=freestream.pressure_pa,
                )
                components.append(load_component)
                alternate_components.append(alternate_component)
                arrays[f"component_{component:03d}_force_pressure_n_m"] = load_component.force_pressure_n_m
                arrays[f"component_{component:03d}_force_viscous_n_m"] = load_component.force_viscous_n_m
                arrays[f"component_{component:03d}_heat_flux_w_m2"] = load_component.heat_flux_w_m2
                arrays[f"component_{component:03d}_moment_total_n"] = np.array(load_component.moment_total_n)
            force = np.sum([item.force_total_n_m for item in components], axis=0)
            alternate_force = np.sum([item.force_total_n_m for item in alternate_components], axis=0)
            moment = float(sum(item.moment_total_n for item in components))
            alternate_moment = float(sum(item.moment_total_n for item in alternate_components))
            arrays["force_total_n_m"] = force
            arrays["moment_total_n"] = np.array(moment)
            path = context.data_dir / f"{label}_validated_2d_eb_forces.npz"
            np.savez_compressed(path, **arrays)
            time_s = dataset["time"]
            designation = "validated_2d_eb_v1"
            sensitivity.append({
                "plotfile": label,
                "normal_fit_point_counts": [analysis.normal_sample_points, reduced_points],
                "force_delta_n_m": (alternate_force - force).tolist(),
                "moment_delta_n": alternate_moment - moment,
            })
            interpretation = (
                "General two-dimensional EB pressure, viscous, thermal, force, and moment result; validated, not certified."
            )
        context.register(
            artifact_id=f"forces.components.{label}", path=path, kind="array",
            variable="surface_load", units="N/m, N, and W/m^2",
            coordinate_metadata={"x": "m", "y": "m"}, interpretation=interpretation,
            provenance={"designation": designation, "plotfile": plotfile,
                        "baseline": analysis.baseline},
        )
        history.append({"plotfile": label, "time_s": time_s, "force_x_n_m": force[0],
                        "force_y_n_m": force[1], "moment_n": moment,
                        "designation": designation})
    sensitivity_path = context.data_dir / "force_sensitivity.json"
    sensitivity_path.write_text(json.dumps({
        "reference_chord_m": analysis.reference_chord_m,
        "reference_span_m": analysis.reference_span_m,
        "moment_origin_m": analysis.moment_origin_m,
        "baseline": analysis.baseline,
        "fit_sensitivity": sensitivity,
    }, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="forces.sensitivity", path=sensitivity_path, kind="json", variable="surface_load",
        units=None, coordinate_metadata={},
        interpretation="Pressure-fit order or normal-fit point-count sensitivity for each snapshot.",
    )
    history_path = context.data_dir / "force_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(history[0]))
        writer.writeheader()
        writer.writerows(history)
    context.register(
        artifact_id="forces.history", path=history_path, kind="table", variable="surface_load",
        units="time_s, N/m, N", coordinate_metadata={"time": "s"},
        interpretation="Chronological integrated force and moment history.",
    )
