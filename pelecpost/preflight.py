"""Scientific, input, and resource preflight without bulk data allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil, pi
from typing import Any

import numpy as np

from pelecpost.config.models import ResolvedProject
from pelecpost.io import InputInventory, inspect_project
from pelecpost.workflows import build_workflow_graph, recipe_for


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
    expected_artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreflightPlan:
    case_id: str
    inventory: InputInventory
    workflow_order: tuple[str, ...]
    findings: tuple[Finding, ...]
    estimates: tuple[AnalysisEstimate, ...]
    sampling: dict[str, Any]
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


def _estimate_memory_gb(analysis: Any, inventory: InputInventory) -> float:
    probes = inventory.probes
    if analysis.recipe in {
        "probe_spectrum", "single_pulse_response", "directional_wave",
        "transient_wavepacket", "nonlinear_coupling", "modal_screening",
    } and probes is not None:
        matrix = probes.sample_count * probes.probe_count * 8
        multiplier = {
            "probe_spectrum": 3.0,
            "single_pulse_response": 4.0,
            "directional_wave": 6.0,
            "transient_wavepacket": 5.0,
            "nonlinear_coupling": 8.0,
            "modal_screening": 7.0,
        }[analysis.recipe]
        return matrix * multiplier / 1024**3
    if analysis.recipe == "case_comparison":
        return 0.25
    # Plotfile arrays are read one selected region/snapshot at a time. Without
    # cell extents in Header metadata, report a conservative bounded allowance.
    return 2.0


def create_plan(project: ResolvedProject, inventory: InputInventory | None = None) -> PreflightPlan:
    inventory = inventory or inspect_project(project)
    graph = build_workflow_graph(project)
    findings: list[Finding] = []
    sampling = _sampling_summary(project, inventory)
    case = project.case_file.case
    geometry = project.case_file.geometry.type

    if not project.enabled_analyses:
        findings.append(Finding(Severity.INFO, "NO_ANALYSES", "No analyses are enabled."))

    for analysis in project.enabled_analyses:
        definition = recipe_for(analysis.recipe)
        if case.dimensionality not in definition.supported_dimensions:
            findings.append(Finding(
                Severity.BLOCKER,
                "UNSUPPORTED_DIMENSION",
                f"{analysis.recipe} supports dimensions {definition.supported_dimensions}; "
                "3-D extension interfaces are documented but no 3-D algorithm is implemented.",
                analysis.id,
            ))
        if geometry not in definition.supported_geometries:
            findings.append(Finding(
                Severity.BLOCKER, "UNSUPPORTED_GEOMETRY",
                f"{analysis.recipe} does not support geometry {geometry!r}.", analysis.id,
            ))
        for required in definition.required_inputs:
            available = {
                "plotfiles": inventory.plotfiles is not None and inventory.plotfiles.count > 0,
                "probes": inventory.probes is not None and inventory.probes.sample_count > 0,
                "comparison_archives": bool(inventory.comparison_archives),
            }[required]
            if not available:
                findings.append(Finding(
                    Severity.BLOCKER, "MISSING_INPUT",
                    f"Recipe {analysis.recipe} requires configured {required} input.", analysis.id,
                ))
        if definition.required_fields and inventory.plotfiles is not None:
            for field in definition.required_fields:
                if field not in inventory.plotfiles.canonical_fields:
                    findings.append(Finding(
                        Severity.BLOCKER, "MISSING_PLOTFILE_FIELD",
                        f"Required canonical field {field!r} was not mapped from the plotfile.",
                        analysis.id,
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

        nyquist = sampling.get("nyquist_frequency_hz")
        maximum = getattr(analysis, "frequency_max_hz", None)
        maximum = maximum or getattr(analysis, "band_max_hz", None)
        if maximum is not None and nyquist is not None and maximum > nyquist:
            findings.append(Finding(
                Severity.BLOCKER, "ABOVE_NYQUIST",
                f"Requested maximum {maximum:.6g} Hz exceeds Nyquist {nyquist:.6g} Hz.",
                analysis.id,
            ))
        segment = sampling.get("segmented_estimators", {}).get(analysis.id)
        if segment and segment["segment_count"] < 4:
            findings.append(Finding(
                Severity.WARNING, "FEW_SEGMENTS",
                f"Only {segment['segment_count']} segments are available; spectral confidence is weak.",
                analysis.id,
            ))
        if analysis.recipe == "single_pulse_response" and inventory.probes is not None:
            end = analysis.baseline_end_time_s
            if end is None:
                end = analysis.start_time_s
            start = inventory.probes.time_min_s
            dt = inventory.probes.median_timestep_s
            count = int(max(0, (end - start) / dt)) if end is not None and start is not None and dt else 0
            if count < analysis.minimum_baseline_samples:
                findings.append(Finding(
                    Severity.BLOCKER, "INSUFFICIENT_QUIESCENT_BASELINE",
                    f"Estimated baseline has {count} samples; {analysis.minimum_baseline_samples} required.",
                    analysis.id,
                ))
        if analysis.recipe == "directional_wave":
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

    estimates = tuple(
        AnalysisEstimate(
            analysis.id,
            analysis.recipe,
            _estimate_memory_gb(analysis, inventory),
            recipe_for(analysis.recipe).outputs,
        )
        for analysis in project.enabled_analyses
    )
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
    return PreflightPlan(
        case_id=case.id,
        inventory=inventory,
        workflow_order=graph.order,
        findings=tuple(findings),
        estimates=estimates,
        sampling=sampling,
        output_root=str(output),
    )
