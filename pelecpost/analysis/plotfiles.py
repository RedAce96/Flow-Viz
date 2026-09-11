"""Plotfile recipe executors using reviewed PeleC/AMReX adapters."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import pp_functions_database as fields_api
import pp_plotting_database as plotting_api
import pelecpost.visualization as visualization
from pelecpost.config.models import (
    AerodynamicForcesAnalysis,
    BoundaryLayerAnalysis,
    ContourStyle,
    ExplicitFreestream,
    FlatPlateGeometry,
    FlowOverviewAnalysis,
    SurfaceDiagnosticsAnalysis,
    Variable,
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
from .spectral import load_probe_variable, prepare_probe_time_grid
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
            geometry.length_m, geometry.half_angle_deg, geometry.fluid_side,
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


def _coarsen_flat_plate_wall(wall: dict) -> dict:
    """Merge adjacent wall faces for an integration-grid sensitivity estimate."""
    left = np.asarray(wall["x_left_m"], dtype=float)
    right = np.asarray(wall["x_right_m"], dtype=float)
    if len(left) < 4:
        raise ValueError("flat-plate grid sensitivity requires at least four wall faces")
    groups = [np.arange(start, min(start + 2, len(left))) for start in range(0, len(left), 2)]
    result = dict(wall)
    result["x_left_m"] = np.asarray([left[group[0]] for group in groups])
    result["x_right_m"] = np.asarray([right[group[-1]] for group in groups])
    for key in ("p_wall_pa", "p_wall_linear_pa", "tau_wall_pa", "wall_y_m"):
        if key not in wall:
            continue
        values = np.asarray(wall[key], dtype=float)
        if values.ndim == 0:
            continue
        result[key] = np.asarray([
            np.average(values[group], weights=right[group] - left[group])
            for group in groups
        ])
    if "valid" in wall:
        valid = np.asarray(wall["valid"], dtype=bool)
        result["valid"] = np.asarray([np.all(valid[group]) for group in groups])
    return result


def _coarsened_surface_load(
    surface: SurfaceCurve2D,
    fit,
    analysis: AerodynamicForcesAnalysis,
    pressure_reference_pa: float,
):
    """Reintegrate every second surface point when topology permits."""
    minimum = 6 if surface.closed else 5
    if len(surface.coordinates_m) < minimum:
        return None
    indices = np.arange(0, len(surface.coordinates_m), 2)
    if not surface.closed and indices[-1] != len(surface.coordinates_m) - 1:
        indices = np.append(indices, len(surface.coordinates_m) - 1)
    coarse = SurfaceCurve2D.from_points(
        surface.coordinates_m[indices], closed=surface.closed,
        fluid_side=surface.fluid_side, component_id=surface.component_id,
        side_id=surface.side_id, source=f"{surface.source}_coarsened",
        confidence=surface.confidence,
    )
    return integrate_surface_loads(
        coarse, fit.pressure_pa[indices], fit.tangential_velocity_gradient_s[indices],
        fit.temperature_gradient_k_m[indices],
        dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
        conductivity_w_m_k=analysis.conductivity_w_m_k,
        moment_origin_m=analysis.moment_origin_m,
        pressure_reference_pa=pressure_reference_pa,
    )


def _analysis_presentation(context: WorkflowContext, analysis):
    return visualization.resolve_presentation(
        context.project.analyses_file.presentation, analysis.presentation,
    )


def _field_contour_style(context: WorkflowContext, analysis: FlowOverviewAnalysis, field: str):
    default = _analysis_presentation(context, analysis).contour_defaults
    override = analysis.contours.fields.get(Variable(field))
    if field == "vorticity" and override is None:
        default = default.model_copy(update={
            "colormap": "RdBu_r", "symmetric_about_zero": True,
        })
    return visualization.resolve_contour_style(default, override)


def _shared_contour_ranges(
    context: WorkflowContext,
    analysis: FlowOverviewAnalysis,
    paths: list[str],
    requested: set[str],
    styles: dict[str, ContourStyle],
) -> dict[str, tuple[float, float]]:
    selected = {
        field: style for field, style in styles.items()
        if style.range.mode in {"selected_snapshots_minmax", "selected_snapshots_percentile"}
    }
    if not selected:
        return {}
    extrema = {field: [np.inf, -np.inf] for field in selected}
    samples: dict[str, np.ndarray] = {field: np.empty(0) for field in selected}
    maximum_samples = 500_000
    total = len(paths)
    for index, plotfile in enumerate(paths, start=1):
        with context.timed_phase(
            "shared-range-plotfile-load", plotfile=plotfile,
            plotfile_index=index, plotfile_count=total,
        ):
            dataset = _load(context, plotfile, requested, _region(analysis))
        context.progress(
            f"scanning shared contour ranges for plotfile {index}/{total}",
            event="contour-range-scan", plotfile=plotfile,
            plotfile_index=index, plotfile_count=total,
            fields=sorted(selected),
        )
        for field, style in selected.items():
            values = np.asarray(dataset["fields"][field], dtype=float).ravel()
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            extrema[field][0] = min(extrema[field][0], float(np.min(values)))
            extrema[field][1] = max(extrema[field][1], float(np.max(values)))
            if style.range.mode == "selected_snapshots_percentile":
                stride = max(1, int(np.ceil(len(values) / 100_000)))
                combined = np.concatenate((samples[field], values[::stride]))
                if len(combined) > maximum_samples:
                    reduction = int(np.ceil(len(combined) / maximum_samples))
                    combined = combined[::reduction]
                samples[field] = combined
    resolved = {}
    for field, style in selected.items():
        if not np.isfinite(extrema[field]).all():
            raise ValueError(f"cannot determine a finite shared contour range for {field}")
        if style.range.mode == "selected_snapshots_minmax":
            limits = tuple(extrema[field])
        else:
            limits = tuple(np.percentile(
                samples[field],
                [style.range.lower_percentile, style.range.upper_percentile],
            ))
        resolved[field] = (float(limits[0]), float(limits[1]))
        context.progress(
            f"resolved shared contour range for {field}",
            event="contour-range-resolved", field=field,
            minimum=resolved[field][0], maximum=resolved[field][1],
            range_mode=style.range.mode,
        )
    return resolved


def _contour_limits(values: np.ndarray, style, shared=None) -> tuple[float, float]:
    if shared is not None:
        limits = shared
    elif style.range.mode == "fixed":
        limits = (style.range.minimum, style.range.maximum)
    elif style.range.mode == "per_snapshot_percentile":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("contour field contains no finite values")
        limits = tuple(np.percentile(
            finite, [style.range.lower_percentile, style.range.upper_percentile],
        ))
    else:
        raise ValueError(f"shared range {style.range.mode!r} was not resolved")
    minimum, maximum = map(float, limits)
    if not maximum > minimum:
        raise ValueError("contour range must have positive width")
    return minimum, maximum


def _register_figure_variants(
    context: WorkflowContext,
    *,
    artifact_id: str,
    paths: tuple[Path, ...],
    variable: str | None,
    units: str | None,
    coordinate_metadata: dict,
    interpretation: str,
    provenance: dict | None = None,
) -> None:
    for index, path in enumerate(paths):
        identifier = artifact_id if index == 0 else f"{artifact_id}.{path.suffix.lstrip('.')}"
        context.register(
            artifact_id=identifier, path=path, kind="figure", variable=variable,
            units=units, coordinate_metadata=coordinate_metadata,
            interpretation=interpretation,
            provenance={**(provenance or {}), "figure_format": path.suffix.lstrip(".")},
        )


def _resample_cartesian_profile(profile: dict, config) -> dict:
    coordinate = np.asarray(profile["y"], dtype=float)
    values = np.asarray(profile["values"], dtype=float)
    if config.coordinate_range_m is not None:
        lower, upper = config.coordinate_range_m
        mask = (coordinate >= lower) & (coordinate <= upper)
        coordinate, values = coordinate[mask], values[mask]
    if not len(coordinate):
        raise ValueError("line-profile coordinate range contains no samples")
    if config.sample_points is not None:
        target = np.linspace(float(coordinate[0]), float(coordinate[-1]), config.sample_points)
        if config.interpolation == "nearest":
            indices = np.searchsorted(coordinate, target, side="left")
            indices = np.clip(indices, 0, len(coordinate) - 1)
            left = np.maximum(indices - 1, 0)
            choose_left = np.abs(target - coordinate[left]) < np.abs(target - coordinate[indices])
            indices[choose_left] = left[choose_left]
            values = values[indices]
        else:
            values = np.interp(target, coordinate, values)
        coordinate = target
    return {**profile, "y": coordinate, "values": values}


def _line_style(context: WorkflowContext, analysis: FlowOverviewAnalysis, field: str):
    config = analysis.line_profiles
    default = _analysis_presentation(context, analysis).line_defaults
    updates = {
        key: value for key, value in {
            "coordinate_scale": config.coordinate_scale,
            "value_scale": config.value_scale,
            "grid": config.grid,
            "legend_position": config.legend_position,
        }.items() if value is not None
    }
    default = default.model_copy(update=updates)
    return visualization.resolve_line_style(default, config.fields.get(Variable(field)))


def _normalise_profile_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    if not upper > lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def _render_cartesian_profiles(
    context: WorkflowContext,
    analysis: FlowOverviewAnalysis,
    dataset: dict,
    label: str,
    station: float,
    profiles: list[dict],
    time_text: str,
) -> None:
    presentation = _analysis_presentation(context, analysis)
    groups = [[profile] for profile in profiles]
    if analysis.line_profiles.layout == "combined":
        groups = [profiles]
    for group_index, group in enumerate(groups):
        figure, axis, _, _ = visualization.line_figure(presentation, time_text)
        for profile in group:
            field = profile["field_key"]
            style = _line_style(context, analysis, field)
            values = np.asarray(profile["values"], dtype=float)
            if analysis.line_profiles.normalize_values:
                values = _normalise_profile_values(values)
            visualization.plot_profile(
                axis, profile["y"], values, plotting_api.field_title(field), style,
            )
        representative = _line_style(context, analysis, group[0]["field_key"])
        visualization.style_line_axis(axis, representative)
        axis.set_xlabel(r"$y$ [m]")
        axis.set_ylabel(
            "Normalized value" if analysis.line_profiles.normalize_values
            else plotting_api.field_label(group[0]["field_key"])
        )
        axis.set_title(
            rf"Cartesian profile at requested $x={station:.6g}$ m "
            rf"(sampled $x={float(group[0]['x_sampled']):.6g}$ m)"
        )
        if analysis.line_profiles.coordinate_limits is not None:
            axis.set_xlim(analysis.line_profiles.coordinate_limits)
        if analysis.line_profiles.value_limits is not None:
            axis.set_ylim(analysis.line_profiles.value_limits)
        if len(group) > 1:
            axis.legend(loc=visualization.legend_location(representative))
        field_suffix = "combined" if len(group) > 1 else group[0]["field_key"]
        figure_paths = visualization.save_figure_variants(
            figure, context.figure_dir / f"{label}_line_x_{station:.6g}_{field_suffix}",
            presentation.figure,
        )
        base_id = f"field.lines.{label}.x-{station:.9g}"
        if group_index > 0 or len(groups) > 1:
            base_id = f"{base_id}.{field_suffix}"
        _register_figure_variants(
            context, artifact_id=base_id, paths=figure_paths,
            variable=None if len(group) > 1 else group[0]["field_key"],
            units="dimensionless" if analysis.line_profiles.normalize_values else "SI",
            coordinate_metadata={
                "coordinate": "y_m", "requested_x_m": float(station),
                "sampled_x_m": float(group[0]["x_sampled"]),
            },
            interpretation="Styled Cartesian line profile at one registered plotfile time.",
            provenance={
                "plotfile": dataset.get("source"), "layout": analysis.line_profiles.layout,
                "normalized": analysis.line_profiles.normalize_values,
            },
        )


def _surface_point_at_arc(surface: SurfaceCurve2D, distance_m: float):
    total = float(np.sum(surface.segment_length_m))
    if distance_m < 0.0 or distance_m > total:
        raise ValueError(
            f"arc-length station {distance_m:g} m lies outside [0, {total:g}] m"
        )
    starts = surface.arc_length_m
    index = int(np.searchsorted(starts, distance_m, side="right") - 1)
    index = min(max(index, 0), len(surface.segment_length_m) - 1)
    start = surface.coordinates_m[index]
    end_index = (index + 1) % len(surface.coordinates_m)
    fraction = (distance_m - float(starts[index])) / float(surface.segment_length_m[index])
    point = start + fraction * (surface.coordinates_m[end_index] - start)
    return point, surface.segment_tangent[index], surface.segment_fluid_normal[index], distance_m


def _surface_x_intersections(surface: SurfaceCurve2D, x_m: float):
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for index in range(len(surface.segment_length_m)):
        start = surface.coordinates_m[index]
        end = surface.coordinates_m[(index + 1) % len(surface.coordinates_m)]
        lower, upper = sorted((float(start[0]), float(end[0])))
        tolerance = max(1.0, abs(x_m)) * 1.0e-12
        if x_m < lower - tolerance or x_m > upper + tolerance:
            continue
        delta_x = float(end[0] - start[0])
        if abs(delta_x) <= tolerance:
            if abs(x_m - float(start[0])) <= tolerance:
                raise ValueError(
                    f"x={x_m:g} m overlaps a vertical surface segment; use arc_length"
                )
            continue
        fraction = float(np.clip((x_m - float(start[0])) / delta_x, 0.0, 1.0))
        point = start + fraction * (end - start)
        arc = float(surface.arc_length_m[index] + fraction * surface.segment_length_m[index])
        candidate = (point, surface.segment_tangent[index], surface.segment_fluid_normal[index], arc)
        if not any(np.linalg.norm(point - existing[0]) <= tolerance for existing in results):
            results.append(candidate)
    return results


def _resolve_normal_station(surfaces, station):
    candidates = []
    for index, surface in enumerate(surfaces):
        if station.component_id is not None:
            requested = str(station.component_id)
            if requested not in {str(index), str(surface.component_id)}:
                continue
        if station.side_id is not None and station.side_id != surface.side_id:
            continue
        if station.location.type == "arc_length":
            candidates.append((surface, *_surface_point_at_arc(surface, station.location.value_m)))
        else:
            candidates.extend(
                (surface, *item)
                for item in _surface_x_intersections(surface, station.location.value_m)
            )
    if not candidates:
        raise ValueError(
            f"surface-normal station {station.id!r} does not intersect the selected geometry"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"surface-normal station {station.id!r} is ambiguous; select component_id/side_id "
            "and use arc_length for multi-valued geometry"
        )
    return candidates[0]


def _sample_selected_normal(dataset: dict, point, normal, distance_m, points, config):
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    first = 0.5 * min(float(np.median(np.diff(x))), float(np.median(np.diff(y))))
    if distance_m <= first:
        raise ValueError("normal profile distance must exceed half the local grid spacing")
    fraction = np.linspace(0.0, 1.0, points)
    if config.spacing == "wall_clustered":
        fraction = fraction ** config.clustering_exponent
    distances = first + (distance_m - first) * fraction
    coordinates = np.asarray(point)[None, :] + distances[:, None] * np.asarray(normal)[None, :]
    values = {}
    for variable in config.fields:
        field = variable.value
        interpolator = RegularGridInterpolator(
            (x, y), np.asarray(dataset["fields"][field]),
            method=config.interpolation, bounds_error=False, fill_value=np.nan,
        )
        sampled = interpolator(coordinates)
        if not np.any(np.isfinite(sampled)):
            raise ValueError(f"surface-normal station has no finite {field} samples")
        values[field] = sampled
    return distances, coordinates, values


def _wall_extrapolation(distances: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(distances) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    count = min(4, int(np.count_nonzero(finite)))
    return float(np.polyfit(distances[finite][:count], values[finite][:count], 1)[1])


def _surface_profile_style(context: WorkflowContext, profiles):
    analysis = cast(SurfaceDiagnosticsAnalysis, context.analysis)
    default = _analysis_presentation(context, analysis).line_defaults
    return default.model_copy(update={
        "coordinate_scale": profiles.figure.coordinate_scale,
        "value_scale": profiles.figure.value_scale,
        "grid": profiles.figure.grid,
    })


def _render_surface_profile_figures(
    context: WorkflowContext,
    label: str,
    station,
    distances: np.ndarray,
    values: dict[str, np.ndarray],
    time_text: str,
    profiles,
) -> None:
    analysis = cast(SurfaceDiagnosticsAnalysis, context.analysis)
    presentation = _analysis_presentation(context, analysis)
    groups = [[field] for field in values]
    if profiles.figure.layout == "combined":
        groups = [list(values)]
    for group_index, fields in enumerate(groups):
        figure, axis, _, _ = visualization.line_figure(presentation, time_text)
        style = _surface_profile_style(context, profiles)
        for field in fields:
            plotted = values[field]
            if profiles.figure.normalize_values:
                plotted = _normalise_profile_values(plotted)
            visualization.plot_profile(
                axis, distances, plotted, plotting_api.field_title(field), style,
            )
        visualization.style_line_axis(axis, style)
        axis.set_xlabel("Surface-normal distance [m]")
        axis.set_ylabel(
            "Normalized value" if profiles.figure.normalize_values
            else plotting_api.field_label(fields[0])
        )
        axis.set_title(f"Surface-normal profile: {station.id}")
        if len(fields) > 1:
            axis.legend(loc=visualization.legend_location(style))
        suffix = "combined" if len(fields) > 1 else fields[0]
        figure_paths = visualization.save_figure_variants(
            figure, context.figure_dir / f"{label}_{station.id}_{suffix}", presentation.figure,
        )
        artifact_id = f"surface.normal_profile.figure.{label}.{station.id}"
        if group_index > 0 or len(groups) > 1:
            artifact_id = f"{artifact_id}.{suffix}"
        _register_figure_variants(
            context, artifact_id=artifact_id, paths=figure_paths,
            variable=None if len(fields) > 1 else fields[0],
            units="dimensionless" if profiles.figure.normalize_values else "SI",
            coordinate_metadata={"normal_distance": "m"},
            interpretation="Selected geometry-aware surface-normal profile.",
        )


@executor("flow_overview")
def run_flow_overview(context: WorkflowContext) -> None:
    analysis = cast(FlowOverviewAnalysis, context.analysis)
    presentation = _analysis_presentation(context, analysis)
    output_fields = {item.value for item in analysis.fields}
    requested = set(output_fields)
    if analysis.streamlines:
        requested.update(("x_velocity", "y_velocity"))
    with context.timed_phase("plotfile-discovery"):
        paths = _paths(context)
    context.progress(
        f"discovered {len(paths)} selected plotfile(s)",
        event="plotfiles-discovered", plotfile_count=len(paths),
        first_plotfile=paths[0] if paths else None,
        last_plotfile=paths[-1] if paths else None,
    )
    styles = {
        field: _field_contour_style(context, analysis, field)
        for field in sorted(output_fields)
    }
    with context.timed_phase(
        "shared-contour-range-scan", plotfile_count=len(paths),
        fields=sorted(output_fields),
    ):
        shared_ranges = _shared_contour_ranges(context, analysis, paths, requested, styles)
    with visualization.presentation_context(presentation):
        for plotfile_index, plotfile in enumerate(paths, start=1):
            with context.timed_phase(
                "render-pass-plotfile-load", plotfile=plotfile,
                plotfile_index=plotfile_index, plotfile_count=len(paths),
            ):
                dataset = _load(context, plotfile, requested, _region(analysis))
            label = dataset["plot_label"]
            time_text = plotting_api.format_dataset_time(
                dataset, precision=presentation.time_annotation.precision,
            )
            for field in sorted(output_fields):
                with context.timed_phase(
                    "contour-render-and-save", field=field, plotfile=plotfile,
                    plotfile_label=label, plotfile_index=plotfile_index,
                    plotfile_count=len(paths),
                ):
                    style = styles[field]
                    limits = _contour_limits(
                        dataset["fields"][field], style, shared_ranges.get(field),
                    )
                    figure, contour_axis, colorbar_axis, _, time_artist, resolved_limits = (
                        visualization.render_contour(
                            dataset, field, presentation, style, limits,
                            x_limits_m=analysis.x_limits_m, y_limits_m=analysis.y_limits_m,
                            time_text=time_text,
                        )
                    )
                    if time_artist is not None and visualization.artists_overlap(
                        figure, time_artist, colorbar_axis,
                    ):
                        plt.close(figure)
                        raise RuntimeError("time annotation overlaps the contour colorbar")
                    if time_artist is not None and visualization.artists_overlap(
                        figure, time_artist, contour_axis,
                    ):
                        plt.close(figure)
                        raise RuntimeError("time annotation overlaps the contour data axes")
                    if visualization.artists_overlap(figure, colorbar_axis, contour_axis):
                        plt.close(figure)
                        raise RuntimeError("contour colorbar or label overlaps the data axes")
                    figure_paths = visualization.save_figure_variants(
                        figure, context.figure_dir / f"{label}_{field}", presentation.figure,
                    )
                    _register_figure_variants(
                        context, artifact_id=f"field.contours.{label}.{field}",
                        paths=figure_paths, variable=field, units="SI; see field label",
                        coordinate_metadata={"x": "m", "y": "m"},
                        interpretation="Descriptive two-dimensional field view at one registered plotfile time.",
                        provenance={
                            "plotfile": plotfile,
                            "freestream_reference": dataset.get("freestream_reference"),
                            "contour_style": style.model_dump(mode="json"),
                            "resolved_color_range": list(resolved_limits),
                        },
                    )
            for station in analysis.line_stations_x_m:
                profiles = []
                for field in sorted(output_fields):
                    if field not in dataset["fields"]:
                        continue
                    profile = _resample_cartesian_profile(
                        fields_api.extract_line(dataset, station, field), analysis.line_profiles,
                    )
                    profile["label"] = field
                    profiles.append(profile)
                    table_path = context.data_dir / f"{label}_line_x_{station:.6g}_{field}.csv"
                    with table_path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("y_m", field))
                        writer.writerows(zip(profile["y"], profile["values"]))
                    context.register(
                        artifact_id=f"field.lines.data.{label}.x-{station:.9g}.{field}",
                        path=table_path, kind="table", variable=field,
                        units=f"y: m; value: {plotting_api.field_label(field)}",
                        coordinate_metadata={
                            "coordinate": "y_m", "requested_x_m": float(station),
                            "sampled_x_m": float(profile["x_sampled"]),
                        },
                        interpretation="Configured Cartesian line extraction with explicit sampled coordinate.",
                        provenance={
                            "plotfile": plotfile,
                            "interpolation": analysis.line_profiles.interpolation,
                        },
                    )
                if profiles:
                    _render_cartesian_profiles(
                        context, analysis, dataset, label, station, profiles, time_text,
                    )
            if analysis.streamlines:
                streamline = fields_api.extract_streamline_field(
                    dataset, color_key="velocity_magnitude",
                )
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
        gip_rows: list[dict] = []
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
            gip_rows.append({
                "x_station_m": float(profile["x_station_m"]),
                "profile_valid": bool(gpi.get("profile_valid", False)),
                "gip_present": bool(gpi.get("gip_present", False)),
                "crossing_count": int(gpi.get("crossing_count", 0)),
                "gip_locations_m": np.asarray(
                    gpi.get("gip_locations", ()), dtype=float
                ).tolist(),
                "gip_locations_over_delta99": np.asarray(
                    gpi.get("gip_locations_over_delta99", ()), dtype=float
                ).tolist(),
                "rejected_crossing_locations_m": np.asarray(
                    gpi.get("rejected_crossing_locations", ()), dtype=float
                ).tolist(),
                "rejected_crossing_reasons": np.asarray(
                    gpi.get("rejected_crossing_reasons", ()), dtype=str
                ).tolist(),
                "rejection_reason": str(gpi.get("rejection_reason", "not reported")),
            })
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
        gip_path = context.data_dir / f"{label}_gip_screening.json"
        gip_path.write_text(
            json.dumps({
                "schema_version": 1,
                "stations": gip_rows,
                "interpretation": (
                    "A credible generalized inflection point is a necessary screening "
                    "condition only; it is not an LST/PSE mode or instability proof."
                ),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        context.register(
            artifact_id=f"boundary_layer.gip.{label}", path=gip_path, kind="json",
            variable="generalized_inflection_point", units="SI",
            coordinate_metadata={"x_station": "m", "wall_distance": "m"},
            interpretation=(
                "Derivative-quality-gated generalized-inflection-point screening; a passing "
                "screen is necessary evidence only and does not establish instability."
            ),
            provenance={"plotfile": plotfile},
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
    presentation = _analysis_presentation(context, analysis)
    geometry = context.project.case_file.geometry
    requested = {"pressure", "temperature", "x_velocity", "y_velocity"}
    if analysis.normal_profiles is not None:
        requested.update(variable.value for variable in analysis.normal_profiles.fields)
    if geometry.type == "volume_fraction":
        requested.add(geometry.field)
    for plotfile in _paths(context):
        dataset = _load(context, plotfile, requested)
        label = Path(plotfile).name
        surfaces = _surfaces(context, dataset)
        time_text = plotting_api.format_dataset_time(
            dataset, precision=presentation.time_annotation.precision,
        )
        for component, surface in enumerate(surfaces):
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
            with visualization.presentation_context(presentation):
                figure, axis, _, _ = visualization.line_figure(presentation, time_text)
                axis.plot(
                    surface.coordinates_m[:, 0], surface.coordinates_m[:, 1],
                    color=analysis.geometry_figure.surface_color,
                    linewidth=1.5, label="surface",
                )
                stride = max(1, int(np.ceil(
                    len(surface.coordinates_m)
                    / analysis.geometry_figure.maximum_normal_arrows
                )))
                configured_length = analysis.geometry_figure.normal_arrow_length
                normal_scale = (
                    analysis.normal_sample_distance_m
                    if configured_length == "sample_distance" else float(configured_length)
                )
                points = surface.coordinates_m[::stride]
                normals = surface.fluid_normal[::stride]
                axis.quiver(
                    points[:, 0], points[:, 1], normals[:, 0], normals[:, 1],
                    angles="xy", scale_units="xy", scale=1.0 / normal_scale,
                    color=analysis.geometry_figure.normal_color,
                    width=0.003, label="fluid-facing normals",
                )
                axis.set_xlabel("x [m]")
                axis.set_ylabel("y [m]")
                axis.set_aspect("equal", adjustable="datalim")
                axis.grid(True, alpha=0.25)
                axis.legend()
                axis.set_title(f"{label}: component {component} surface reconstruction")
                figure_paths = visualization.save_figure_variants(
                    figure, context.figure_dir / f"{stem}_geometry_normals",
                    presentation.figure,
                )
            _register_figure_variants(
                context, artifact_id=f"surface.figure.{label}.{component}",
                paths=figure_paths, variable="geometry", units="m",
                coordinate_metadata={"x": "m", "y": "m"},
                interpretation=(
                    "Surface reconstruction and decimated fluid-facing normals; numerical "
                    "coordinates and quality gates are registered separately."
                ),
                provenance={
                    "maximum_normal_arrows": analysis.geometry_figure.maximum_normal_arrows,
                    "normal_arrow_length_m": normal_scale,
                },
            )
        if analysis.normal_profiles is not None:
            for station in analysis.normal_profiles.stations:
                surface, point, tangent, normal, arc_length = _resolve_normal_station(
                    surfaces, station,
                )
                maximum_distance = station.distance_m or analysis.normal_sample_distance_m
                sample_points = station.sample_points or analysis.normal_sample_points
                distances, coordinates, sampled = _sample_selected_normal(
                    dataset, point, normal, maximum_distance, sample_points,
                    analysis.normal_profiles,
                )
                extrapolated = np.zeros(len(distances), dtype=bool)
                output_distances, output_coordinates, output_values = distances, coordinates, sampled
                if analysis.normal_profiles.include_wall_extrapolation:
                    output_distances = np.concatenate(([0.0], distances))
                    output_coordinates = np.vstack((point, coordinates))
                    extrapolated = np.concatenate(([True], extrapolated))
                    output_values = {
                        field: np.concatenate(([_wall_extrapolation(distances, values)], values))
                        for field, values in sampled.items()
                    }
                stem = f"{label}_normal_{station.id}"
                array_path = context.data_dir / f"{stem}.npz"
                np.savez_compressed(
                    array_path, normal_distance_m=output_distances,
                    sample_coordinates_m=output_coordinates,
                    surface_point_m=np.asarray(point), tangent=np.asarray(tangent),
                    fluid_normal=np.asarray(normal), arc_length_m=float(arc_length),
                    is_wall_extrapolation=extrapolated,
                    **{field: values for field, values in output_values.items()},
                )
                profile_provenance = {
                    "plotfile": plotfile,
                    "station": station.model_dump(mode="json", exclude_none=True),
                    "surface_component": surface.component_id,
                    "surface_side": surface.side_id,
                    "interpolation": analysis.normal_profiles.interpolation,
                    "spacing": analysis.normal_profiles.spacing,
                }
                context.register(
                    artifact_id=f"surface.normal_profile.array.{label}.{station.id}",
                    path=array_path, kind="array", variable="wall_normal_state", units="SI",
                    coordinate_metadata={
                        "normal_distance": "m", "sample_coordinates": "m", "arc_length": "m",
                    },
                    interpretation=(
                        "Selected geometry-aware surface-normal samples; zero-distance values "
                        "are explicitly flagged wall extrapolations when enabled."
                    ),
                    provenance=profile_provenance,
                )
                table_path = context.data_dir / f"{stem}.csv"
                fields = list(output_values)
                with table_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow((
                        "normal_distance_m", "x_m", "y_m", "is_wall_extrapolation", *fields,
                    ))
                    for index in range(len(output_distances)):
                        writer.writerow((
                            output_distances[index], output_coordinates[index, 0],
                            output_coordinates[index, 1], bool(extrapolated[index]),
                            *(output_values[field][index] for field in fields),
                        ))
                context.register(
                    artifact_id=f"surface.normal_profile.table.{label}.{station.id}",
                    path=table_path, kind="table", variable="wall_normal_state",
                    units="SI in unit-bearing columns and artifact field metadata",
                    coordinate_metadata={"normal_distance": "m", "x": "m", "y": "m"},
                    interpretation="Tabular form of the selected surface-normal profile.",
                    provenance=profile_provenance,
                )
                with visualization.presentation_context(presentation):
                    _render_surface_profile_figures(
                        context, label, station, output_distances, output_values,
                        time_text, analysis.normal_profiles,
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
            raw_load = fields_api.integrate_flat_plate_wall_forces(wall, reference)
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
            coarsened = fields_api.integrate_flat_plate_wall_forces(
                _coarsen_flat_plate_wall(wall_for_load), integration_reference
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
                "integration_grid_coarsening_factor": 2,
                "grid_coarsened_force_delta_n_m": [
                    coarsened["D_total_N_m"] - force[0],
                    coarsened["N_total_N_m"] - force[1],
                ],
                "grid_coarsened_moment_delta_n": coarsened["M_total_N"] - moment,
                "baseline_force_contribution_n_m": [
                    force[0] - raw_load["D_total_N_m"],
                    force[1] - raw_load["N_total_N_m"],
                ],
                "baseline_moment_contribution_n": moment - raw_load["M_total_N"],
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
            raw_components = []
            coarsened_components = []
            coarsened_unavailable: list[str] = []
            arrays = {}
            for component, surface in enumerate(_surfaces(context, dataset)):
                _, _, _, _, fit = _sample_wall(
                    dataset, surface, analysis.normal_sample_distance_m,
                    analysis.normal_sample_points,
                )
                raw_fit = fit
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
                raw_component = integrate_surface_loads(
                    surface, raw_fit.pressure_pa,
                    raw_fit.tangential_velocity_gradient_s,
                    raw_fit.temperature_gradient_k_m,
                    dynamic_viscosity_pa_s=analysis.dynamic_viscosity_pa_s,
                    conductivity_w_m_k=analysis.conductivity_w_m_k,
                    moment_origin_m=analysis.moment_origin_m,
                    pressure_reference_pa=freestream.pressure_pa,
                )
                coarse_component = _coarsened_surface_load(
                    surface, fit, analysis,
                    0.0 if baseline_dataset is not None else freestream.pressure_pa,
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
                raw_components.append(raw_component)
                alternate_components.append(alternate_component)
                if coarse_component is None:
                    coarsened_unavailable.append(
                        f"component {component} has too few points for factor-two coarsening"
                    )
                else:
                    coarsened_components.append(coarse_component)
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
            raw_force = np.sum([item.force_total_n_m for item in raw_components], axis=0)
            raw_moment = float(sum(item.moment_total_n for item in raw_components))
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
                "baseline_force_contribution_n_m": (force - raw_force).tolist(),
                "baseline_moment_contribution_n": moment - raw_moment,
            }
            if not coarsened_unavailable:
                coarsened_force = np.sum(
                    [item.force_total_n_m for item in coarsened_components], axis=0
                )
                coarsened_moment = float(
                    sum(item.moment_total_n for item in coarsened_components)
                )
                sensitivity_item.update({
                    "integration_grid_coarsening_factor": 2,
                    "grid_coarsened_force_delta_n_m": (
                        coarsened_force - force
                    ).tolist(),
                    "grid_coarsened_moment_delta_n": coarsened_moment - moment,
                })
            else:
                sensitivity_item["grid_sensitivity_unavailable"] = coarsened_unavailable
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
            "Pressure/normal-fit, factor-two integration-grid, baseline contribution, and "
            "configured EB geometry-smoothing sensitivity for each snapshot."
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
        variable, probe_unit, probe_time, probe_x, probe_values, selected = load_probe_variable(
            context, linkage.variable.value, linkage.probe_indices,
            probe_set_id=linkage.probe_set_id,
        )
        probe_time, probe_values, _probe_dt, probe_resampled, probe_cleanup = (
            prepare_probe_time_grid(
                context, probe_time, probe_values, linkage.time_grid_policy,
            )
        )
        if probe_cleanup is not None:
            context.add_cleanup(probe_cleanup)
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
                "time_grid_policy": linkage.time_grid_policy,
                "probe_resampled": probe_resampled,
            },
        )
