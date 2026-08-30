"""Plotfile recipe executors using reviewed PeleC/AMReX adapters."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pp_functions_database as fields_api
import pp_plotting_database as plotting_api
from pelecpost.config.models import (
    AerodynamicForcesAnalysis,
    BoundaryLayerAnalysis,
    ExplicitFreestream,
    FlatPlateGeometry,
    FlowOverviewAnalysis,
    SurfaceDiagnosticsAnalysis,
)
from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.geometry import (
    SurfaceCurve2D,
    flat_plate_surface,
    polyline_surface,
    volume_fraction_surfaces,
    wedge_surfaces,
)
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .fields import add_normalized_fields, resolve_freestream_reference
from .spectral import load_compact_variable
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
        smoothing_window=geometry.smoothing_window,
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
    analysis = cast(FlowOverviewAnalysis, context.analysis)
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
    analysis = cast(BoundaryLayerAnalysis, context.analysis)
    geometry = cast(FlatPlateGeometry, context.project.case_file.geometry)
    for plotfile in _paths(context):
        profiles = fields_api.extract_native_flat_plate_boundary_layers(
            plotfile, analysis.stations_x_m, maximum_height_m=analysis.maximum_height_m,
            wall_y_m=geometry.wall_y_m,
            wall_temperature_k=analysis.wall_temperature_k,
            viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
            conductivity_w_m_k=analysis.conductivity_w_m_k,
        )
        label = Path(plotfile).name
        arrays: dict[str, np.ndarray] = {"station_count": np.array(len(profiles))}
        thickness_rows: list[dict] = []
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
            thickness_rows.append({
                "x_station_m": profile["x_station_m"],
                "delta_99_m": profile["delta_99_m"],
                "delta_star_m": profile["delta_star_m"],
                "theta_m": profile["theta_m"],
                "shape_factor": profile["H"],
                "re_theta": profile["Re_theta"],
                "wall_shear_pa": profile["tau_wall_pa"],
                "skin_friction_coefficient": profile["C_f"],
                "wall_heat_flux_w_m2": profile["q_wall_w_m2"],
            })
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
        thickness_path = context.data_dir / f"{label}_boundary_layer_thickness.csv"
        with thickness_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(thickness_rows[0]))
            writer.writeheader()
            writer.writerows(thickness_rows)
        context.register(
            artifact_id=f"boundary_layer.thickness.{label}", path=thickness_path,
            kind="table", variable="boundary_layer", units="SI in unit-bearing columns",
            coordinate_metadata={"x_station": "m"},
            interpretation=(
                "Compressible integral thickness, wall transport, and Reynolds-number summary "
                "for each configured flat-plate station."
            ),
            provenance={"plotfile": plotfile},
        )


@executor("surface_diagnostics")
def run_surface_diagnostics(context: WorkflowContext) -> None:
    analysis = cast(SurfaceDiagnosticsAnalysis, context.analysis)
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
    analysis = cast(AerodynamicForcesAnalysis, context.analysis)
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
    baseline_paths: list[str] = []
    if analysis.baseline != "none":
        assert analysis.baseline_id is not None
        baseline_config = context.project.machine_file.inputs.baselines[analysis.baseline_id]
        baseline_source = (
            baseline_config.source if baseline_config.source.is_absolute()
            else context.project.root / baseline_config.source
        )
        baseline_paths = fields_api.discover_plotfile_paths(
            baseline_source, plot_prefix=baseline_config.prefix
        )

    def baseline_for(current_path: str) -> str | None:
        if analysis.baseline == "none":
            return None
        if analysis.baseline == "static":
            return baseline_paths[0]
        matches = [path for path in baseline_paths if Path(path).name == Path(current_path).name]
        if len(matches) != 1:
            raise ValueError(
                f"paired baseline requires exactly one plotfile named {Path(current_path).name}; found {len(matches)}"
            )
        return matches[0]
    history: list[dict] = []
    sensitivity: list[dict] = []
    control_volume_rows: list[dict] = []
    for plotfile in _paths(context):
        label = Path(plotfile).name
        baseline_plotfile = baseline_for(plotfile)
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
            alternate_wall = {**wall, "p_wall_pa": wall["p_wall_linear_pa"]}
            wall_for_load = wall
            alternate_for_load = alternate_wall
            integration_reference = reference
            if baseline_plotfile is not None:
                baseline_wall = fields_api.extract_native_flat_plate_wall(
                    baseline_plotfile, x_range_m=(geometry.leading_edge_x_m, end),
                    wall_y_m=geometry.wall_y_m, p_inf_pa=freestream.pressure_pa,
                    rho_inf_kg_m3=freestream.density_kg_m3,
                    u_inf_m_s=freestream.velocity_m_s,
                    viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                )
                baseline_alternate = {
                    **baseline_wall, "p_wall_pa": baseline_wall["p_wall_linear_pa"]
                }
                wall_for_load = fields_api.difference_flat_plate_wall_surfaces(wall, baseline_wall)
                alternate_for_load = fields_api.difference_flat_plate_wall_surfaces(
                    alternate_wall, baseline_alternate
                )
                integration_reference = {**reference, "p_inf": 0.0}
            load = fields_api.integrate_flat_plate_wall_forces(
                wall_for_load, integration_reference
            )
            alternate = fields_api.integrate_flat_plate_wall_forces(
                alternate_for_load, integration_reference
            )
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
            if baseline_plotfile is not None:
                designation += "_increment"
            sensitivity_item = {
                "plotfile": label, "pressure_fit_orders": [2, 1],
                "force_delta_n_m": [
                    alternate["D_total_N_m"] - force[0],
                    alternate["N_total_N_m"] - force[1],
                ],
                "moment_delta_n": alternate["M_total_N"] - moment,
            }
            if analysis.control_volume is not None:
                control = analysis.control_volume
                control_dataset = _load(
                    context, plotfile,
                    {"density", "pressure", "x_velocity", "y_velocity"},
                )
                control_result = fields_api.compute_flat_plate_control_volume_force(
                    control_dataset, control.x_range_m, control.y_top_m,
                    analysis.dynamic_viscosity_pa_s,
                    bulk_viscosity_pa_s=control.bulk_viscosity_pa_s,
                )
                if baseline_plotfile is not None:
                    baseline_control_dataset = _load(
                        context, baseline_plotfile,
                        {"density", "pressure", "x_velocity", "y_velocity"},
                    )
                    baseline_control = fields_api.compute_flat_plate_control_volume_force(
                        baseline_control_dataset, control.x_range_m, control.y_top_m,
                        analysis.dynamic_viscosity_pa_s,
                        bulk_viscosity_pa_s=control.bulk_viscosity_pa_s,
                    )
                    for key in (
                        "D_control_volume_N_m", "N_control_volume_N_m",
                        "momentum_flux_x_N_m", "momentum_flux_y_N_m",
                        "other_boundary_stress_x_N_m", "other_boundary_stress_y_N_m",
                    ):
                        control_result[key] -= baseline_control[key]
                    control_result["baseline_plotfile"] = baseline_plotfile
                control_result.update({"plotfile": label, "time_s": float(time_s)})
                control_volume_rows.append(control_result)
                for key, value in control_result.items():
                    candidate = np.asarray(value)
                    if candidate.dtype.kind in "biufc":
                        arrays[f"control_volume_{key}"] = candidate
                sensitivity_item.update({
                    "surface_minus_control_volume_force_n_m": [
                        force[0] - control_result["D_control_volume_N_m"],
                        force[1] - control_result["N_control_volume_N_m"],
                    ],
                    "control_volume_assumption": control_result["assumption"],
                })
                # Re-save after adding the optional control-volume arrays.
                np.savez_compressed(path, **arrays)
            sensitivity.append(sensitivity_item)
            interpretation = (
                "Reviewed stationary one-sided flat-plate pressure, viscous force, and moment."
            )
        else:
            requested = {"pressure", "temperature", "x_velocity", "y_velocity"}
            if geometry.type == "volume_fraction":
                requested.add(geometry.field)
            dataset = _load(context, plotfile, requested)
            baseline_dataset = (
                _load(context, baseline_plotfile, requested)
                if baseline_plotfile is not None else None
            )
            components = []
            alternate_components = []
            unsmoothed_components = []
            arrays = {}
            for component, surface in enumerate(_surfaces(context, dataset)):
                _, _, _, _, fit = _sample_wall(
                    dataset, surface, analysis.normal_sample_distance_m,
                    analysis.normal_sample_points,
                )
                if baseline_dataset is not None:
                    _, _, _, _, baseline_fit = _sample_wall(
                        baseline_dataset, surface, analysis.normal_sample_distance_m,
                        analysis.normal_sample_points,
                    )
                    fit = type(fit)(
                        pressure_pa=fit.pressure_pa - baseline_fit.pressure_pa,
                        tangential_velocity_gradient_s=(
                            fit.tangential_velocity_gradient_s
                            - baseline_fit.tangential_velocity_gradient_s
                        ),
                        temperature_gradient_k_m=(
                            fit.temperature_gradient_k_m
                            - baseline_fit.temperature_gradient_k_m
                        ),
                        pressure_residual_pa=fit.pressure_residual_pa,
                        velocity_residual_m_s=fit.velocity_residual_m_s,
                        temperature_residual_k=fit.temperature_residual_k,
                        point_count=np.minimum(fit.point_count, baseline_fit.point_count),
                    )
                load_component = integrate_surface_loads(
                    surface, fit.pressure_pa, fit.tangential_velocity_gradient_s,
                    fit.temperature_gradient_k_m,
                    dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                    conductivity_w_m_k=analysis.conductivity_w_m_k,
                    moment_origin_m=analysis.moment_origin_m,
                    pressure_reference_pa=(
                        0.0 if baseline_dataset is not None else freestream.pressure_pa
                    ),
                )
                reduced_points = max(4, analysis.normal_sample_points // 2)
                _, _, _, _, reduced_fit = _sample_wall(
                    dataset, surface, analysis.normal_sample_distance_m, reduced_points,
                )
                if baseline_dataset is not None:
                    _, _, _, _, baseline_reduced_fit = _sample_wall(
                        baseline_dataset, surface, analysis.normal_sample_distance_m,
                        reduced_points,
                    )
                    reduced_fit = type(reduced_fit)(
                        pressure_pa=reduced_fit.pressure_pa - baseline_reduced_fit.pressure_pa,
                        tangential_velocity_gradient_s=(
                            reduced_fit.tangential_velocity_gradient_s
                            - baseline_reduced_fit.tangential_velocity_gradient_s
                        ),
                        temperature_gradient_k_m=(
                            reduced_fit.temperature_gradient_k_m
                            - baseline_reduced_fit.temperature_gradient_k_m
                        ),
                        pressure_residual_pa=reduced_fit.pressure_residual_pa,
                        velocity_residual_m_s=reduced_fit.velocity_residual_m_s,
                        temperature_residual_k=reduced_fit.temperature_residual_k,
                        point_count=np.minimum(
                            reduced_fit.point_count, baseline_reduced_fit.point_count
                        ),
                    )
                alternate_component = integrate_surface_loads(
                    surface, reduced_fit.pressure_pa,
                    reduced_fit.tangential_velocity_gradient_s,
                    reduced_fit.temperature_gradient_k_m,
                    dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                    conductivity_w_m_k=analysis.conductivity_w_m_k,
                    moment_origin_m=analysis.moment_origin_m,
                    pressure_reference_pa=(
                        0.0 if baseline_dataset is not None else freestream.pressure_pa
                    ),
                )
                components.append(load_component)
                alternate_components.append(alternate_component)
                if geometry.type == "volume_fraction":
                    if surface.unsmoothed_coordinates_m is None:
                        raise ValueError("volume-fraction surface omitted unsmoothed coordinates")
                    unsmoothed_surface = SurfaceCurve2D.from_points(
                        surface.unsmoothed_coordinates_m,
                        closed=surface.closed,
                        fluid_side=surface.fluid_side,
                        component_id=surface.component_id,
                        side_id=surface.side_id,
                        source="volume_fraction_unsmoothed_sensitivity",
                        confidence=surface.confidence,
                    )
                    _, _, _, _, unsmoothed_fit = _sample_wall(
                        dataset, unsmoothed_surface, analysis.normal_sample_distance_m,
                        analysis.normal_sample_points,
                    )
                    if baseline_dataset is not None:
                        _, _, _, _, baseline_unsmoothed_fit = _sample_wall(
                            baseline_dataset, unsmoothed_surface,
                            analysis.normal_sample_distance_m,
                            analysis.normal_sample_points,
                        )
                        unsmoothed_fit = type(unsmoothed_fit)(
                            pressure_pa=(
                                unsmoothed_fit.pressure_pa
                                - baseline_unsmoothed_fit.pressure_pa
                            ),
                            tangential_velocity_gradient_s=(
                                unsmoothed_fit.tangential_velocity_gradient_s
                                - baseline_unsmoothed_fit.tangential_velocity_gradient_s
                            ),
                            temperature_gradient_k_m=(
                                unsmoothed_fit.temperature_gradient_k_m
                                - baseline_unsmoothed_fit.temperature_gradient_k_m
                            ),
                            pressure_residual_pa=unsmoothed_fit.pressure_residual_pa,
                            velocity_residual_m_s=unsmoothed_fit.velocity_residual_m_s,
                            temperature_residual_k=unsmoothed_fit.temperature_residual_k,
                            point_count=np.minimum(
                                unsmoothed_fit.point_count,
                                baseline_unsmoothed_fit.point_count,
                            ),
                        )
                    unsmoothed_components.append(integrate_surface_loads(
                        unsmoothed_surface, unsmoothed_fit.pressure_pa,
                        unsmoothed_fit.tangential_velocity_gradient_s,
                        unsmoothed_fit.temperature_gradient_k_m,
                        dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                        conductivity_w_m_k=analysis.conductivity_w_m_k,
                        moment_origin_m=analysis.moment_origin_m,
                        pressure_reference_pa=(
                            0.0 if baseline_dataset is not None else freestream.pressure_pa
                        ),
                    ))
                    arrays[f"component_{component:03d}_coordinates_m"] = surface.coordinates_m
                    arrays[f"component_{component:03d}_unsmoothed_coordinates_m"] = (
                        surface.unsmoothed_coordinates_m
                    )
                arrays[f"component_{component:03d}_force_pressure_n_m"] = load_component.force_pressure_n_m
                arrays[f"component_{component:03d}_force_viscous_n_m"] = load_component.force_viscous_n_m
                arrays[f"component_{component:03d}_heat_flux_w_m2"] = load_component.heat_flux_w_m2
                arrays[f"component_{component:03d}_moment_total_n"] = np.array(load_component.moment_total_n)
            force = np.sum([item.force_total_n_m for item in components], axis=0)
            alternate_force = np.sum([item.force_total_n_m for item in alternate_components], axis=0)
            moment = float(sum(item.moment_total_n for item in components))
            alternate_moment = float(sum(item.moment_total_n for item in alternate_components))
            unsmoothed_force = (
                np.sum([item.force_total_n_m for item in unsmoothed_components], axis=0)
                if unsmoothed_components else None
            )
            unsmoothed_moment = (
                float(sum(item.moment_total_n for item in unsmoothed_components))
                if unsmoothed_components else None
            )
            arrays["force_total_n_m"] = force
            arrays["moment_total_n"] = np.array(moment)
            path = context.data_dir / f"{label}_validated_2d_eb_forces.npz"
            np.savez_compressed(path, **arrays)
            time_s = dataset["time"]
            designation = "validated_2d_eb_v1"
            if baseline_plotfile is not None:
                designation += "_increment"
            sensitivity_item = {
                "plotfile": label,
                "normal_fit_point_counts": [analysis.normal_sample_points, reduced_points],
                "force_delta_n_m": (alternate_force - force).tolist(),
                "moment_delta_n": alternate_moment - moment,
            }
            if unsmoothed_force is not None and unsmoothed_moment is not None:
                sensitivity_item.update({
                    "geometry_smoothing_window": getattr(geometry, "smoothing_window"),
                    "unsmoothed_force_delta_n_m": (unsmoothed_force - force).tolist(),
                    "unsmoothed_moment_delta_n": unsmoothed_moment - moment,
                })
            sensitivity.append(sensitivity_item)
            interpretation = (
                "General two-dimensional EB pressure, viscous, thermal, force, and moment result; validated, not certified."
            )
        context.register(
            artifact_id=f"forces.components.{label}", path=path, kind="array",
            variable="surface_load", units="N/m, N, and W/m^2",
            coordinate_metadata={"x": "m", "y": "m"}, interpretation=interpretation,
            provenance={"designation": designation, "plotfile": plotfile,
                        "baseline": analysis.baseline,
                        "baseline_plotfile": baseline_plotfile},
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
        interpretation=(
            "Pressure-fit order, normal-fit point-count, and configured EB geometry-smoothing "
            "sensitivity for each snapshot."
        ),
    )
    if control_volume_rows:
        control_path = context.data_dir / "control_volume_balance.json"
        control_path.write_text(json.dumps({
            "schema_version": 1,
            "balances": control_volume_rows,
            "interpretation": (
                "Steady rectangular momentum balances are independent flat-plate force diagnostics; "
                "unsteady momentum storage is omitted."
            ),
        }, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="forces.control_volume", path=control_path, kind="json",
            variable="surface_load", units="N/m", coordinate_metadata={"x": "m", "y": "m"},
            interpretation=(
                "Steady rectangular control-volume force balance and discrepancy from wall integration."
            ),
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
    if analysis.probe_linkage is not None:
        linkage = analysis.probe_linkage
        ordered_history = sorted(history, key=lambda item: float(item["time_s"]))
        force_time = np.asarray([item["time_s"] for item in ordered_history], dtype=float)
        force_key = {
            "x": "force_x_n_m", "y": "force_y_n_m", "moment": "moment_n",
        }[linkage.force_component]
        force_signal = np.asarray([item[force_key] for item in ordered_history], dtype=float)
        variable, probe_unit, probe_time, probe_x, probe_values, selected = load_compact_variable(
            context, linkage.variable.value, linkage.probe_indices,
        )
        linkage_result = fields_api.compute_probe_force_linkage(
            force_time, force_signal, probe_time, probe_values, probe_x,
            linkage.forcing_frequency_hz,
            minimum_forcing_periods=linkage.minimum_forcing_periods,
            nperseg=linkage.welch_segment_samples,
            noverlap=linkage.overlap_fraction,
            minimum_segments=linkage.minimum_segments,
        )
        linkage_arrays = {
            "probe_indices": selected,
            "force_component": np.array(linkage.force_component),
            "probe_variable": np.array(variable),
            "probe_unit": np.array(probe_unit),
        }
        for key, value in linkage_result.items():
            if value is None:
                continue
            candidate = np.asarray(value)
            if candidate.dtype != object:
                linkage_arrays[key] = candidate
        linkage_path = context.data_dir / "force_probe_linkage.npz"
        np.savez_compressed(linkage_path, **linkage_arrays)
        context.register(
            artifact_id="forces.probe_linkage", path=linkage_path, kind="array",
            variable=f"{linkage.force_component}_force_vs_{variable}", units="mixed; see arrays",
            coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
            interpretation=(
                "Synchronized lag, coherence, phase, and H1 linkage between measured probes and "
                "integrated force; association does not establish causality."
            ),
            provenance={
                "force_component": linkage.force_component,
                "forcing_frequency_hz": linkage.forcing_frequency_hz,
                "spectral_status": linkage_result["spectral_status"],
                "anti_alias_filter": linkage_result["anti_alias_filter"],
            },
        )
