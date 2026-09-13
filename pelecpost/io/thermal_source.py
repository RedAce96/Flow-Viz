"""Reader and canonicalizer for PeleC measured thermal-source histories.

The producer writes one row for every accepted AMR level update.  This module
keeps the file format deliberately boring: metadata is a JSON comment followed
by ordinary CSV.  All restart and energy-conservation rules live here so the
analysis executors do not need to know how a run was restarted.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "pelec.thermal-source-history"
SCHEMA_VERSION = 1
REQUIRED_COLUMNS = (
    "coarse_step", "level", "level_step", "subcycle_iteration",
    "time_start_s", "time_end_s", "dt_s", "power_start", "power_end",
    "deposited_energy", "moment_x", "moment_y", "moment_z", "moment_xx",
    "moment_yy", "moment_zz", "moment_xy", "moment_xz", "moment_yz",
)
MOMENT_COLUMNS = REQUIRED_COLUMNS[9:]


@dataclass(frozen=True)
class ThermalSourceSegment:
    path: Path
    segment_index: int
    metadata: dict[str, Any]
    rows: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class ThermalSourceHistory:
    segments: tuple[ThermalSourceSegment, ...]
    rows: tuple[dict[str, float | int], ...]
    metadata: dict[str, Any]
    discarded_incomplete_rows: int

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(segment.path for segment in self.segments)


def _number(value: str, name: str) -> float | int:
    if name in {"coarse_step", "level", "level_step", "subcycle_iteration"}:
        result = int(value)
    else:
        result = float(value)
    if not math.isfinite(float(result)):
        raise ValueError(f"thermal source column {name!r} contains a non-finite value")
    return result


def _metadata_line(stream: Iterable[str], path: Path) -> tuple[dict[str, Any], list[str]]:
    iterator = iter(stream)
    first = next(iterator, "")
    if not first.startswith("#"):
        raise ValueError(f"thermal source segment {path} is missing its JSON metadata comment")
    try:
        payload = json.loads(first[1:].strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"thermal source segment {path} has invalid JSON metadata: {exc}") from exc
    return payload, list(iterator)


def read_segment(path: Path) -> ThermalSourceSegment:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as stream:
        metadata, remaining = _metadata_line(stream, path)
    if metadata.get("schema") != SCHEMA or metadata.get("version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported thermal source history schema in {path}")
    try:
        segment_index = int(metadata["segment_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"thermal source segment {path} has no valid segment_index") from exc
    reader = csv.DictReader(remaining)
    if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
        raise ValueError(
            f"thermal source segment {path} columns do not match the version-1 contract"
        )
    rows: list[dict[str, float | int]] = []
    for row_number, raw in enumerate(reader, start=3):
        if None in raw:
            raise ValueError(f"thermal source segment {path} row {row_number} has extra columns")
        parsed = {name: _number(raw[name], name) for name in REQUIRED_COLUMNS}
        start = float(parsed["time_start_s"])
        end = float(parsed["time_end_s"])
        dt = float(parsed["dt_s"])
        if end <= start or dt <= 0.0 or not math.isclose(dt, end - start, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError(f"thermal source segment {path} row {row_number} has inconsistent time interval")
        rows.append(parsed)
    previous: dict[int, tuple[float, float]] = {}
    for row in rows:
        level = int(row["level"])
        interval = (float(row["time_start_s"]), float(row["time_end_s"]))
        old = previous.get(level)
        if old is not None and interval[0] < old[0] - 1e-14:
            raise ValueError(f"thermal source segment {path} has non-monotonic level intervals")
        previous[level] = interval
    return ThermalSourceSegment(path, segment_index, metadata, tuple(rows))


def _metadata_signature(metadata: dict[str, Any]) -> tuple[Any, ...]:
    keys = (
        "dimensions", "coordinate_system", "units", "source_shape",
        "deposition_model", "requested_pulse_energy", "source_scale",
        "normalization_integrals", "frequency_hz", "period_s", "fwhm_s",
        "sigma_s", "cutoff_sigma", "start_time_s", "laser_duration_s",
        "source_center", "axis", "gaussian_radius_cm", "wang_length_cm",
        "wang_alpha", "wang_beta", "wang_R1_cm", "wang_R2_cm", "domain_bounds_cm",
        "eb_present", "source_implementation_sha256", "measurement_implementation_sha256",
    )
    return tuple(json.dumps(metadata.get(key), sort_keys=True, default=str) for key in keys)


def _complete_groups(rows: tuple[dict[str, float | int], ...]) -> tuple[tuple[dict[str, float | int], ...], int]:
    groups: dict[int, list[dict[str, float | int]]] = {}
    for row in rows:
        groups.setdefault(int(row["coarse_step"]), []).append(row)
    retained: list[dict[str, float | int]] = []
    discarded = 0
    for group in groups.values():
        if any(int(row["level"]) == 0 for row in group):
            retained.extend(group)
        else:
            discarded += len(group)
    return tuple(retained), discarded


def canonicalize_segments(paths: Iterable[Path]) -> ThermalSourceHistory:
    segments = tuple(sorted((read_segment(path) for path in paths), key=lambda item: (item.segment_index, item.path.name)))
    if not segments:
        raise FileNotFoundError("no thermal source history segments were found")
    signature = _metadata_signature(segments[0].metadata)
    if any(_metadata_signature(segment.metadata) != signature for segment in segments[1:]):
        raise ValueError("thermal source segments disagree on source configuration or implementation revision")
    selected: list[dict[str, float | int]] = []
    discarded = 0
    for index, segment in enumerate(segments):
        complete, dropped = _complete_groups(segment.rows)
        discarded += dropped
        if index:
            if complete:
                branch = min(float(row["time_start_s"]) for row in complete if int(row["level"]) == 0)
                selected = [row for row in selected if float(row["time_end_s"]) <= branch + 1e-14]
        selected.extend(complete)
    selected.sort(key=lambda row: (float(row["time_start_s"]), int(row["level"]), int(row["coarse_step"]), int(row["subcycle_iteration"])))
    seen: set[tuple[int, float, float]] = set()
    for row in selected:
        key = (int(row["level"]), float(row["time_start_s"]), float(row["time_end_s"]))
        if key in seen:
            raise ValueError("thermal source history contains duplicate retained intervals")
        seen.add(key)
    return ThermalSourceHistory(segments, tuple(selected), segments[0].metadata, discarded)


def discover_source_segments(source: Path, prefix: str = "thermal-source.segment") -> tuple[Path, ...]:
    source = source.expanduser().resolve()
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"thermal source history directory does not exist: {source}")
    paths = tuple(sorted(path for path in source.glob(f"{prefix}*.csv") if path.is_file()))
    if not paths:
        raise FileNotFoundError(f"no thermal source history segments match {prefix!r} in {source}")
    return paths


def validate_history_configuration(history: ThermalSourceHistory, analysis: Any) -> None:
    """Check the producer metadata against the independently configured target."""
    metadata = history.metadata
    if int(metadata.get("dimensions", -1)) != 2:
        raise ValueError("Flow Viz single-pulse measured-source support requires a 2-D history")
    if not isinstance(metadata.get("source_implementation_sha256"), str) or len(metadata["source_implementation_sha256"]) != 64:
        raise ValueError("thermal source metadata has no known source implementation revision")
    if not isinstance(metadata.get("measurement_implementation_sha256"), str) or len(metadata["measurement_implementation_sha256"]) != 64:
        raise ValueError("thermal source metadata has no known measurement implementation revision")
    energy_factor = 1.0e-5
    checks = (
        ("requested_pulse_energy", float(analysis.energy_per_pulse_j_m) / energy_factor, 1.0e-8),
        ("fwhm_s", float(analysis.pulse_fwhm_s), 1.0e-8),
        ("period_s", float(analysis.pulse_period_s), 1.0e-8),
        ("start_time_s", float(analysis.start_time_s), 1.0e-8),
        ("cutoff_sigma", float(analysis.cutoff_sigma), 1.0e-8),
    )
    for key, expected, tolerance in checks:
        actual = metadata.get(key)
        if actual is None or not math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=1.0e-15):
            raise ValueError(f"thermal source metadata {key!r} does not match the analysis configuration")


def count_intersecting_pulses(history: ThermalSourceHistory, time_start_s: float, time_end_s: float) -> int:
    frequency = float(history.metadata.get("frequency_hz", 0.0))
    period = float(history.metadata.get("period_s", 0.0))
    start = float(history.metadata.get("start_time_s", 0.0))
    sigma = float(history.metadata.get("sigma_s", 0.0))
    cutoff = float(history.metadata.get("cutoff_sigma", 4.0))
    if frequency <= 0.0 or period <= 0.0 or sigma <= 0.0:
        return 1
    half_width = cutoff * sigma
    first = int(math.floor((time_start_s - start - 0.5 * period - half_width) / period)) - 1
    last = int(math.ceil((time_end_s - start - 0.5 * period + half_width) / period)) + 1
    return sum(
        1 for index in range(first, last + 1)
        if start + index * period + 0.5 * period + half_width >= time_start_s
        and start + index * period + 0.5 * period - half_width <= time_end_s
    )


def _linear_integral(row: dict[str, float | int], start: float, end: float) -> float:
    a = float(row["time_start_s"])
    b = float(row["time_end_s"])
    left = max(start, a)
    right = min(end, b)
    if right <= left:
        return 0.0
    duration = b - a
    p0 = float(row["power_start"])
    slope = (float(row["power_end"]) - p0) / duration
    return p0 * (right - left) + 0.5 * slope * ((right - a) ** 2 - (left - a) ** 2)


def rebin_history(history: ThermalSourceHistory, time_s: np.ndarray) -> dict[str, Any]:
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(~np.isfinite(time)):
        raise ValueError("source rebinning requires at least two finite timestamps")
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0 or not np.allclose(np.diff(time), dt, rtol=1e-8, atol=1e-15):
        raise ValueError("source rebinning requires a uniform prepared probe time grid")
    edges = np.concatenate(([time[0] - 0.5 * dt], time[:-1] + 0.5 * dt, [time[-1] + 0.5 * dt]))
    power = np.zeros(time.size, dtype=float)
    for index in range(time.size):
        power[index] = sum(_linear_integral(row, edges[index], edges[index + 1]) for row in history.rows) / dt
    retained_energy = float(sum(float(row["deposited_energy"]) for row in history.rows))
    rebinned_energy = float(np.sum(power) * dt)
    if not math.isclose(rebinned_energy, retained_energy, rel_tol=2e-8, abs_tol=1e-14 * max(1.0, abs(retained_energy))):
        raise ValueError("prepared probe record does not cover the retained measured source history")
    moments = {name: float(sum(float(row[name]) for row in history.rows)) for name in MOMENT_COLUMNS}
    return {"time_s": time, "power_w_m": power, "dt_s": dt, "energy_j_m": retained_energy, "moments": moments}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def build_source_audit(
    history: ThermalSourceHistory,
    rebinned: dict[str, Any],
    *,
    requested_energy_j_m: float,
    pulse_fwhm_s: float,
    cutoff_sigma: float,
    maximum_relative_error: float,
    start_time_s: float,
) -> dict[str, Any]:
    metadata = history.metadata
    measured = float(rebinned["energy_j_m"])
    analytic_retained = requested_energy_j_m * math.erf(cutoff_sigma / math.sqrt(2.0))
    signed_error = _safe_ratio(measured - analytic_retained, analytic_retained)
    power = np.asarray(rebinned["power_w_m"], dtype=float)
    time = np.asarray(rebinned["time_s"], dtype=float)
    positive = np.maximum(power, 0.0)
    peak_index = int(np.argmax(positive))
    centroid_t = _safe_ratio(float(np.sum(time * positive)), float(np.sum(positive)))
    requested_center = start_time_s + 0.5 * float(metadata.get("period_s", pulse_fwhm_s))
    moments = rebinned["moments"]
    centroid_x = _safe_ratio(moments["moment_x"], measured)
    centroid_y = _safe_ratio(moments["moment_y"], measured)
    centroid_z = _safe_ratio(moments["moment_z"], measured)
    cxx = _safe_ratio(moments["moment_xx"], measured)
    cyy = _safe_ratio(moments["moment_yy"], measured)
    cxy = _safe_ratio(moments["moment_xy"], measured)
    cov_xx = cxx - centroid_x * centroid_x if cxx is not None and centroid_x is not None else None
    cov_yy = cyy - centroid_y * centroid_y if cyy is not None and centroid_y is not None else None
    cov_xy = cxy - centroid_x * centroid_y if cxy is not None and centroid_x is not None and centroid_y is not None else None
    if cov_xx is not None and cov_yy is not None and cov_xy is not None:
        eigenvalues, eigenvectors = np.linalg.eigh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))
        principal_widths = [float(math.sqrt(max(0.0, value))) for value in eigenvalues[::-1]]
        angle = float(math.atan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    else:
        principal_widths, angle = [None, None], None
    complete = history.discarded_incomplete_rows == 0
    valid_moments = all(math.isfinite(value) for value in moments.values()) and measured > 0.0
    within_tolerance = signed_error is not None and abs(signed_error) <= maximum_relative_error
    one_percent_gate = signed_error is not None and abs(signed_error) <= 0.01
    if not complete or not valid_moments:
        status = "invalid_measurement"
    elif not within_tolerance:
        status = "outside_energy_tolerance"
    else:
        status = "supported"
    return {
        "schema": "pelecpost.pulse-source-audit", "schema_version": 1,
        "status": status, "source_basis": "measured", "complete": complete,
        "configured_requested_energy_j_m": requested_energy_j_m,
        "analytic_retained_energy_j_m": analytic_retained,
        "measured_deposited_energy_j_m": measured,
        "signed_relative_energy_error": signed_error,
        "absolute_relative_energy_error": abs(signed_error) if signed_error is not None else None,
        "capture_ratio": _safe_ratio(measured, requested_energy_j_m),
        "one_percent_gate": one_percent_gate,
        "configured_maximum_relative_error": maximum_relative_error,
        "resolved_thresholds": {"maximum_source_energy_relative_error": maximum_relative_error},
        "measured_peak_time_s": float(time[peak_index]),
        "measured_energy_weighted_time_centroid_s": centroid_t,
        "requested_pulse_center_time_s": requested_center,
        "measured_peak_time_offset_s": float(time[peak_index] - requested_center),
        "measured_centroid_time_offset_s": (
            centroid_t - requested_center if centroid_t is not None else None
        ),
        "temporal_cutoff_sigma": cutoff_sigma,
        "spatial_centroid_m": {"x": centroid_x, "y": centroid_y, "z": centroid_z},
        "covariance_m2": {"xx": cov_xx, "yy": cov_yy, "xy": cov_xy},
        "rms_widths_m": {"x": math.sqrt(max(0.0, cov_xx)) if cov_xx is not None else None, "y": math.sqrt(max(0.0, cov_yy)) if cov_yy is not None else None},
        "principal_widths_m": principal_widths,
        "principal_axis_angle_rad": angle,
        "source_shape": metadata.get("source_shape"),
        "kernel_geometry": {key: metadata.get(key) for key in ("source_center", "axis", "gaussian_radius_cm", "wang_length_cm", "wang_alpha", "wang_beta", "wang_R1_cm", "wang_R2_cm")},
        "segment_inventory": [str(path) for path in history.paths],
        "discarded_incomplete_row_count": history.discarded_incomplete_rows,
        "source_implementation_sha256": metadata.get("source_implementation_sha256"),
        "measurement_implementation_sha256": metadata.get("measurement_implementation_sha256"),
    }


def measured_source_spectrum(
    rebinned: dict[str, Any], *, modeled: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the same finite-record source products used by the pulse recipe."""
    time = np.asarray(rebinned["time_s"], dtype=float)
    power = np.asarray(rebinned["power_w_m"], dtype=float)
    dt = float(rebinned["dt_s"])
    processed = power - np.mean(power)
    frequency = np.fft.rfftfreq(time.size, dt)
    physical_complex = np.fft.rfft(power) * dt
    processed_complex = np.fft.rfft(processed) / time.size
    if modeled is not None:
        ideal = np.asarray(modeled["ideal_spectrum"], dtype=float)
        sigma = float(modeled["sigma_s"])
        center = float(modeled["center_s"])
    else:
        ideal = np.full(frequency.shape, np.nan, dtype=float)
        sigma = float("nan")
        center = float("nan")
    return {
        "time_s": time, "power": power, "processed_power": processed,
        "frequency_hz": frequency, "physical_complex": physical_complex,
        "physical_spectrum": np.abs(physical_complex), "ideal_spectrum": ideal,
        "processed_complex": processed_complex, "sigma_s": sigma, "center_s": center,
    }
