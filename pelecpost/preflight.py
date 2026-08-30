"""Scientific, input, and resource preflight without bulk data allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import pi
import re
from typing import Any

import numpy as np

from pelecpost.config.models import ResolvedProject
from pelecpost.io import InputInventory, inspect_project
from pelecpost.workflows import build_workflow_graph, recipe_for, workflow_for


class Severity(StrEnum):
    BLOCKER = "BLOCKER"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    analysis_id: str | None = None


@dataclass(frozen=True)
class AnalysisEstimate:
    analysis_id: str
    recipe: str
    estimated_peak_gb: float
    memory_components_gb: dict[str, float]
    expected_artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreflightPlan:
    case_id: str
    inventory: InputInventory
    workflow_order: tuple[str, ...]
    findings: tuple[Finding, ...]
    estimates: tuple[AnalysisEstimate, ...]
    sampling: dict[str, Any]
    analysis_contracts: dict[str, dict[str, Any]]
    output_root: str

    @property
    def blockers(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == Severity.BLOCKER)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.severity == Severity.WARNING)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROBE_FIELD_ALIASES = {
    "density": ("density", "rho"),
    "x_velocity": ("x_velocity", "u", "xvel"),
    "y_velocity": ("y_velocity", "v", "yvel"),
    "pressure": ("pressure", "p"),
    "temperature": ("temperature", "T", "temp"),
}


def _has_probe_field(variable: str, available: tuple[str, ...]) -> bool:
    names = set(available)
    return any(alias in names for alias in PROBE_FIELD_ALIASES.get(variable, (variable,)))


def _segment_samples(analysis: Any, sample_count: int) -> int | None:
    for name in ("welch_segment_samples", "stft_segment_samples", "segment_samples"):
        value = getattr(analysis, name, None)
        if value is not None:
            return min(int(value), sample_count)
    if analysis.recipe == "modal_screening":
        return min(int(analysis.spod_segment_samples), sample_count)
    return None


def _sampling_summary(project: ResolvedProject, inventory: InputInventory) -> dict[str, Any]:
    probes = inventory.probes
    if probes is None or probes.median_timestep_s is None or probes.median_timestep_s <= 0:
        return {}
    dt = probes.median_timestep_s
    duration = (
        probes.time_max_s - probes.time_min_s
        if probes.time_max_s is not None and probes.time_min_s is not None
        else None
    )
    result: dict[str, Any] = {
        "sample_count": probes.sample_count,
        "probe_count": probes.probe_count,
        "median_timestep_s": dt,
        "sampling_frequency_hz": 1.0 / dt,
        "nyquist_frequency_hz": 0.5 / dt,
        "record_duration_s": duration,
        "native_frequency_resolution_hz": 1.0 / duration if duration and duration > 0 else None,
    }
    if len(probes.requested_x_m) > 1:
        unique = np.unique(np.asarray(probes.requested_x_m, dtype=float))
        spacing = np.diff(unique)
        if spacing.size:
            median_spacing = float(np.median(spacing))
            aperture = float(unique[-1] - unique[0])
            result.update({
                "probe_spacing_m": median_spacing,
                "probe_spacing_relative_std": float(np.std(spacing) / median_spacing),
                "probe_aperture_m": aperture,
                "spatial_nyquist_rad_m": pi / median_spacing,
                "native_wavenumber_resolution_rad_m": 2.0 * pi / aperture if aperture else None,
            })
    segments: dict[str, Any] = {}
    for analysis in project.enabled_analyses:
        segment = _segment_samples(analysis, probes.sample_count)
        if segment:
            overlap = float(getattr(analysis, "overlap_fraction", 0.5))
            step = max(1, int(round(segment * (1.0 - overlap))))
            count = 1 + max(0, (probes.sample_count - segment) // step)
            segments[analysis.id] = {
                "segment_samples": segment,
                "segment_count": count,
                "effective_frequency_resolution_hz": 1.0 / (segment * dt),
                "approximate_degrees_of_freedom": 2 * count,
            }
    result["segmented_estimators"] = segments
    return result


def _selected_plotfile_names(analysis: Any, names: tuple[str, ...]) -> list[str]:
    start = getattr(analysis, "snapshot_start", None)
    end = getattr(analysis, "snapshot_end", None)
    step = int(getattr(analysis, "snapshot_step", 1))
    if start is None and end is None and step == 1:
        return list(names)
    suffixes = []
    for name in names:
        match = re.search(r"(\d+)$", name)
        suffixes.append(int(match.group(1)) if match else None)
    if names and all(value is not None for value in suffixes):
        numeric = [int(value) for value in suffixes if value is not None]
        selected_start = start if start is not None else min(numeric)
        selected_end = end if end is not None else max(numeric)
        return [
            name for name, value in zip(names, numeric)
            if selected_start <= value <= selected_end and (value - selected_start) % step == 0
        ]
    first = start if start is not None else 0
    last = end if end is not None else len(names)
    return list(names[first:last:step])


def _geometry_points(project: ResolvedProject) -> np.ndarray | None:
    geometry = project.case_file.geometry
    if geometry.type == "flat_plate":
        if geometry.trailing_edge_x_m is None:
            return None
        return np.array([
            [geometry.leading_edge_x_m, geometry.wall_y_m],
            [geometry.trailing_edge_x_m, geometry.wall_y_m],
        ])
    if geometry.type == "wedge":
        angle = np.deg2rad(geometry.half_angle_deg)
        leading = np.array([geometry.leading_edge_x_m, geometry.leading_edge_y_m])
        return np.vstack((
            leading,
            leading + geometry.length_m * np.array([np.cos(angle), np.sin(angle)]),
            leading + geometry.length_m * np.array([np.cos(angle), -np.sin(angle)]),
        ))
    if geometry.type == "polyline":
        return np.asarray(geometry.points_m, dtype=float)
    return None


def create_plan(project: ResolvedProject, inventory: InputInventory | None = None) -> PreflightPlan:
    inventory = inventory or inspect_project(project)
    graph = build_workflow_graph(project)
    findings: list[Finding] = []
    sampling = _sampling_summary(project, inventory)
    case = project.case_file.case
    geometry = project.case_file.geometry.type
    analysis_contracts: dict[str, dict[str, Any]] = {}

    if not project.enabled_analyses:
        findings.append(Finding(Severity.INFO, "NO_ANALYSES", "No analyses are enabled."))

    for analysis in project.enabled_analyses:
        workflow = workflow_for(analysis.recipe)
        analysis_contracts[analysis.id] = {
            "recipe": analysis.recipe,
            "physical_question": workflow.metadata.question,
            "assumptions": list(workflow.metadata.assumptions),
            "limitations": list(workflow.metadata.limitations),
            "required_inputs": list(workflow.required_inputs_for(analysis)),
            "required_fields": list(workflow.metadata.required_fields),
            "expected_artifact_ids": list(workflow.artifact_declarations_for(analysis)),
        }
        for validation in workflow.validate(project, analysis, inventory):
            findings.append(Finding(
                Severity(validation.level), validation.code, validation.message, analysis.id,
            ))
        variable = getattr(analysis, "variable", None)
        if variable is not None and inventory.probes is not None:
            value = str(variable.value if hasattr(variable, "value") else variable)
            if not _has_probe_field(value, inventory.probes.fields):
                findings.append(Finding(
                    Severity.BLOCKER, "MISSING_PROBE_FIELD",
                    f"Probe variable {value!r} is absent; available: {inventory.probes.fields}.",
                    analysis.id,
                ))
        linkage = getattr(analysis, "probe_linkage", None)
        if linkage is not None and inventory.probes is not None:
            value = linkage.variable.value
            if not _has_probe_field(value, inventory.probes.fields):
                findings.append(Finding(
                    Severity.BLOCKER, "MISSING_LINKAGE_PROBE_FIELD",
                    f"Force–probe linkage variable {value!r} is absent from the archive.",
                    analysis.id,
                ))
            selected_indices = linkage.probe_indices
            if selected_indices and (
                len(set(selected_indices)) != len(selected_indices)
                or min(selected_indices) < 0
                or max(selected_indices) >= inventory.probes.probe_count
            ):
                findings.append(Finding(
                    Severity.BLOCKER, "INVALID_LINKAGE_PROBES",
                    "Force–probe linkage probe_indices must be unique archive indices.",
                    analysis.id,
                ))

        nyquist = sampling.get("nyquist_frequency_hz")
        maximum = getattr(analysis, "frequency_max_hz", None)
        maximum = maximum or getattr(analysis, "band_max_hz", None)
        if maximum is not None and nyquist is not None and maximum > nyquist:
            findings.append(Finding(
                Severity.BLOCKER, "ABOVE_NYQUIST",
                f"Requested maximum {maximum:.6g} Hz exceeds Nyquist {nyquist:.6g} Hz.",
                analysis.id,
            ))
        native_resolution = sampling.get("native_frequency_resolution_hz")
        if maximum is not None and native_resolution is not None and maximum < native_resolution:
            findings.append(Finding(
                Severity.BLOCKER, "NO_RESOLVABLE_FREQUENCY_BIN",
                f"Requested maximum {maximum:.6g} Hz is below the finite-record "
                f"resolution {native_resolution:.6g} Hz.", analysis.id,
            ))
        segment = sampling.get("segmented_estimators", {}).get(analysis.id)
        if segment and segment["segment_count"] < 4:
            findings.append(Finding(
                Severity.WARNING, "FEW_SEGMENTS",
                f"Only {segment['segment_count']} segments are available; spectral confidence is weak.",
                analysis.id,
            ))
        if segment:
            lower = getattr(analysis, "frequency_min_hz", None)
            lower = lower if lower is not None else getattr(analysis, "band_min_hz", None)
            if lower is not None and maximum is not None:
                width = maximum - lower
                if width < segment["effective_frequency_resolution_hz"]:
                    findings.append(Finding(
                        Severity.WARNING, "UNDER_RESOLVED_FREQUENCY_BAND",
                        f"Selected bandwidth {width:.6g} Hz is narrower than the segmented "
                        f"resolution {segment['effective_frequency_resolution_hz']:.6g} Hz.",
                        analysis.id,
                    ))
        if analysis.recipe == "single_pulse_response" and inventory.probes is not None:
            end = analysis.baseline_end_time_s
            if end is None:
                end = analysis.start_time_s
            start = inventory.probes.time_min_s
            dt = inventory.probes.median_timestep_s
            count = int(max(0, (end - start) / dt)) if end is not None and start is not None and dt else 0
            sampling.setdefault("quiescent_baselines", {})[analysis.id] = {
                "baseline_end_time_s": end,
                "estimated_sample_count": count,
                "minimum_sample_count": analysis.minimum_baseline_samples,
            }
            if count < analysis.minimum_baseline_samples:
                findings.append(Finding(
                    Severity.BLOCKER, "INSUFFICIENT_QUIESCENT_BASELINE",
                    f"Estimated baseline has {count} samples; {analysis.minimum_baseline_samples} required.",
                    analysis.id,
                ))
        if analysis.recipe == "aerodynamic_forces" and analysis.baseline != "none":
            available_baseline = inventory.baselines.get(analysis.baseline_id or "", ())
            if not available_baseline:
                findings.append(Finding(
                    Severity.BLOCKER, "BASELINE_INPUT_REQUIRED",
                    f"Baseline {analysis.baseline_id!r} has no discoverable plotfiles in machine.yaml.",
                    analysis.id,
                ))
            elif analysis.baseline == "static" and len(available_baseline) != 1:
                findings.append(Finding(
                    Severity.BLOCKER, "STATIC_BASELINE_COUNT",
                    f"Static baseline must contain exactly one plotfile; found {len(available_baseline)}.",
                    analysis.id,
                ))
            elif analysis.baseline == "paired" and inventory.plotfiles is not None:
                missing_pairs = sorted(set(inventory.plotfiles.names) - set(available_baseline))
                if missing_pairs:
                    findings.append(Finding(
                        Severity.BLOCKER, "PAIRED_BASELINE_MISMATCH",
                        f"Paired baseline is missing {len(missing_pairs)} current plotfile name(s): "
                        f"{', '.join(missing_pairs[:5])}.", analysis.id,
                    ))
        if analysis.recipe == "aerodynamic_forces":
            control_volume = analysis.control_volume
            if control_volume is not None:
                if geometry != "flat_plate":
                    findings.append(Finding(
                        Severity.BLOCKER, "CONTROL_VOLUME_FLAT_PLATE_ONLY",
                        "The reviewed steady rectangular control-volume balance is limited to flat plates.",
                        analysis.id,
                    ))
                if inventory.plotfiles is not None and "density" not in inventory.plotfiles.canonical_fields:
                    findings.append(Finding(
                        Severity.BLOCKER, "CONTROL_VOLUME_DENSITY_REQUIRED",
                        "Control-volume momentum balance requires density.", analysis.id,
                    ))
            if linkage is not None:
                selected_names = _selected_plotfile_names(
                    analysis, inventory.plotfiles.names if inventory.plotfiles else ()
                )
                if len(selected_names) < 4:
                    findings.append(Finding(
                        Severity.BLOCKER, "INSUFFICIENT_FORCE_HISTORY",
                        "Force–probe linkage requires at least four selected force snapshots.",
                        analysis.id,
                    ))
                if inventory.plotfiles and inventory.probes:
                    time_by_name = dict(zip(
                        inventory.plotfiles.names, inventory.plotfiles.times_s
                    ))
                    selected_times = [
                        float(time_value) for name in selected_names
                        if (time_value := time_by_name.get(name)) is not None
                    ]
                    if len(selected_times) != len(selected_names):
                        findings.append(Finding(
                            Severity.BLOCKER, "MISSING_FORCE_TIMES",
                            "Every force snapshot needs a readable physical time for probe linkage.",
                            analysis.id,
                        ))
                    elif any(
                        right <= left
                        for left, right in zip(selected_times, selected_times[1:])
                    ):
                        findings.append(Finding(
                            Severity.BLOCKER, "NONMONOTONIC_FORCE_TIMES",
                            "Selected force snapshot times must be strictly increasing for synchronization.",
                            analysis.id,
                        ))
                    start = max(
                        inventory.plotfiles.time_min_s or 0.0,
                        inventory.probes.time_min_s or 0.0,
                    )
                    stop = min(
                        inventory.plotfiles.time_max_s or 0.0,
                        inventory.probes.time_max_s or 0.0,
                    )
                    periods = max(0.0, stop - start) * linkage.forcing_frequency_hz
                    sampling.setdefault("force_probe_linkage", {})[analysis.id] = {
                        "common_time_start_s": start,
                        "common_time_stop_s": stop,
                        "estimated_forcing_periods": periods,
                        "minimum_forcing_periods": linkage.minimum_forcing_periods,
                    }
                    if periods < linkage.minimum_forcing_periods:
                        findings.append(Finding(
                            Severity.WARNING, "SHORT_FORCE_PROBE_RECORD",
                            f"Only {periods:.3g} forcing periods are estimated in the common record; "
                            "lag products remain available but spectral linkage will be unavailable.",
                            analysis.id,
                        ))

        if analysis.recipe in {"surface_diagnostics", "aerodynamic_forces"}:
            normal_distance = float(getattr(analysis, "normal_sample_distance_m"))
            normal_points = int(getattr(analysis, "normal_sample_points"))
            geometry_field = getattr(project.case_file.geometry, "field", None)
            sampling.setdefault("geometry_requirements", {})[analysis.id] = {
                "geometry_type": geometry,
                "normal_sample_distance_m": normal_distance,
                "normal_sample_points": normal_points,
                "domain_bounds_m": (
                    inventory.plotfiles.domain_bounds_m if inventory.plotfiles else None
                ),
                "volume_fraction_field": (
                    geometry_field if geometry == "volume_fraction" else None
                ),
            }
            if geometry == "volume_fraction" and inventory.plotfiles is not None:
                assert geometry_field is not None
                field = geometry_field
                if field not in inventory.plotfiles.fields and field not in inventory.plotfiles.canonical_fields:
                    findings.append(Finding(
                        Severity.BLOCKER, "MISSING_GEOMETRY_FIELD",
                        f"Configured volume-fraction field {field!r} is absent from plotfiles.",
                        analysis.id,
                    ))
            bounds = inventory.plotfiles.domain_bounds_m if inventory.plotfiles else None
            points = _geometry_points(project)
            if bounds is None:
                findings.append(Finding(
                    Severity.INFO, "GEOMETRY_COVERAGE_RUNTIME_GATE",
                    "Plotfile Header does not expose domain bounds; geometry and normal-sampling "
                    "coverage will be gated when the selected field slab is loaded.", analysis.id,
                ))
            elif points is not None:
                inside = all(
                    bounds[axis][0] <= coordinate[axis] <= bounds[axis][1]
                    for coordinate in points
                    for axis in range(2)
                )
                if not inside:
                    findings.append(Finding(
                        Severity.BLOCKER, "GEOMETRY_OUTSIDE_DOMAIN",
                        "Configured surface coordinates extend outside the plotfile domain.",
                        analysis.id,
                    ))
        if analysis.recipe == "directional_wave":
            if inventory.probes is not None and inventory.probes.probe_count < 5:
                findings.append(Finding(
                    Severity.BLOCKER, "INSUFFICIENT_SPATIAL_PROBES",
                    "Directional complex-wavenumber fitting requires at least five probes.",
                    analysis.id,
                ))
            relative_std = sampling.get("probe_spacing_relative_std")
            if relative_std is None:
                findings.append(Finding(
                    Severity.BLOCKER, "NO_SPATIAL_APERTURE",
                    "At least two distinct streamwise probe coordinates are required.", analysis.id,
                ))
            elif relative_std > 0.2:
                findings.append(Finding(
                    Severity.BLOCKER, "NONUNIFORM_PROBE_GRID",
                    f"Probe-spacing relative standard deviation is {relative_std:.1%}.", analysis.id,
                ))
            elif relative_std > 0.02:
                findings.append(Finding(
                    Severity.WARNING, "APPROXIMATE_PROBE_GRID",
                    f"Probe spacing varies by {relative_std:.1%}; resampling sensitivity is required.",
                    analysis.id,
                ))
            spacing = sampling.get("probe_spacing_m")
            if spacing and analysis.expected_speed_min_m_s:
                wavelength = analysis.expected_speed_min_m_s / analysis.frequency_max_hz
                samples = wavelength / spacing
                sampling.setdefault("directional_wave", {})[analysis.id] = {
                    "minimum_expected_wavelength_m": wavelength,
                    "samples_per_minimum_wavelength": samples,
                }
                if samples < 2:
                    findings.append(Finding(
                        Severity.BLOCKER, "SPATIAL_ALIASING",
                        f"Only {samples:.2f} samples per shortest expected wavelength.", analysis.id,
                    ))
                elif samples < 6:
                    findings.append(Finding(
                        Severity.WARNING, "LOW_SPATIAL_RESOLUTION",
                        f"Only {samples:.2f} samples per shortest expected wavelength.", analysis.id,
                    ))

    for label, item in (("plotfile", inventory.plotfiles), ("probe", inventory.probes)):
        if item is not None:
            for message in item.errors:
                findings.append(Finding(Severity.BLOCKER, f"{label.upper()}_INSPECTION", message))
    if inventory.probes is not None:
        if inventory.probes.nonfinite_time_count:
            findings.append(Finding(Severity.BLOCKER, "NONFINITE_TIME", "Probe time contains nonfinite values."))
        missing_total = sum(inventory.probes.missing_value_count.values())
        if missing_total:
            details = ", ".join(
                f"{name}={count}" for name, count in inventory.probes.missing_value_count.items()
                if count
            )
            findings.append(Finding(
                Severity.BLOCKER, "MISSING_PROBE_VALUES",
                f"Probe fields contain {missing_total} nonfinite value(s): {details}.",
            ))
        if inventory.probes.restart_overlap_count:
            findings.append(Finding(
                Severity.WARNING, "RESTART_OVERLAP",
                f"Probe inventory reports {inventory.probes.restart_overlap_count} restart overlap(s).",
            ))
        dt = inventory.probes.median_timestep_s
        if dt and inventory.probes.timestep_std_s and inventory.probes.timestep_std_s / dt > 1e-3:
            findings.append(Finding(
                Severity.WARNING, "NONUNIFORM_TIME",
                "Probe sampling is nonuniform; FFT recipes require an explicit coordinate policy.",
            ))

    estimate_items = []
    for analysis in project.enabled_analyses:
        workflow = workflow_for(analysis.recipe)
        components = workflow.estimate_resources(analysis, inventory)
        estimate_items.append(AnalysisEstimate(
            analysis.id, analysis.recipe, sum(components.values()), components,
            workflow.artifact_declarations_for(analysis),
        ))
    estimates = tuple(estimate_items)
    limit = float(project.machine_file.compute.memory_limit_gb)
    workers = int(project.machine_file.compute.workers)
    largest = max((item.estimated_peak_gb for item in estimates), default=0.0)
    concurrent = largest * min(workers, max(1, len(estimates)))
    sampling["resource_estimate"] = {
        "workers": workers,
        "largest_workflow_peak_gb": largest,
        "estimated_concurrent_peak_gb": concurrent,
        "memory_limit_gb": limit,
    }
    if concurrent > limit:
        findings.append(Finding(
            Severity.BLOCKER, "MEMORY_LIMIT",
            f"Estimated concurrent peak {concurrent:.2f} GB exceeds configured {limit:.2f} GB.",
        ))
    elif concurrent > 0.8 * limit:
        findings.append(Finding(
            Severity.WARNING, "MEMORY_MARGIN",
            f"Estimated concurrent peak {concurrent:.2f} GB is close to configured {limit:.2f} GB.",
        ))
    output = project.machine_file.outputs.root
    if not output.is_absolute():
        output = (project.root / output).resolve()
    plotfile_names = inventory.plotfiles.names if inventory.plotfiles else ()
    sampling["selections"] = {
        "plotfiles": {
            analysis.id: _selected_plotfile_names(analysis, plotfile_names)
            for analysis in project.enabled_analyses
            if "plotfiles" in workflow_for(analysis.recipe).required_inputs_for(analysis)
        },
        "probes": {
            analysis.id: list(getattr(analysis, "probe_indices", ())) or "all"
            for analysis in project.enabled_analyses
            if "probes" in workflow_for(analysis.recipe).required_inputs_for(analysis)
        },
        "expected_output_root": str(output),
    }
    return PreflightPlan(
        case_id=case.id,
        inventory=inventory,
        workflow_order=graph.order,
        findings=tuple(findings),
        estimates=estimates,
        sampling=sampling,
        analysis_contracts=analysis_contracts,
        output_root=str(output),
    )
