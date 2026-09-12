"""Transient packet filtering, arrival, and propagation uncertainty."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.signal import hilbert, stft
from scipy.stats import linregress, t

import pp_functions_database as reviewed_transient
from pelecpost.config.models import TransientWavepacketAnalysis
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .probe_plotting import (
    register_probe_line_overlay,
    register_probe_stft_figure,
    register_probe_trace_figures,
)
from .spectral import load_probe_signal, prepare_probe_time_grid


@dataclass(frozen=True)
class PacketFitEstimate:
    """Numerical packet-fit result, independent of workflow and artifact I/O."""

    slope_s_m: float | None
    velocity_m_s: float | None
    slope_interval_s_m: tuple[float, float] | None
    velocity_interval_m_s: tuple[float, float] | None
    degrees_of_freedom: int
    t_multiplier: float | None
    r_squared: float | None
    per_probe_snr_db: tuple[float, ...]
    minimum_snr_db: float | None
    resolved_edge_margin_s: float
    baseline_sample_count: int
    monotonic_arrivals: bool
    gates: dict[str, bool]
    fit_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "slope_s_m": self.slope_s_m,
            "velocity_m_s": self.velocity_m_s,
            "slope_confidence_interval_95_s_m": (
                list(self.slope_interval_s_m) if self.slope_interval_s_m is not None else None
            ),
            "confidence_interval_95_m_s": (
                list(self.velocity_interval_m_s)
                if self.velocity_interval_m_s is not None else None
            ),
            "degrees_of_freedom": self.degrees_of_freedom,
            "t_multiplier_95": self.t_multiplier,
            "arrival_time_regression_r_squared": self.r_squared,
            "per_probe_snr_db": list(self.per_probe_snr_db),
            "minimum_snr_db": self.minimum_snr_db,
            "resolved_edge_margin_s": self.resolved_edge_margin_s,
            "baseline_sample_count": self.baseline_sample_count,
            "monotonic_arrivals": self.monotonic_arrivals,
            "gates": self.gates,
            "fit_status": self.fit_status,
        }


def estimate_packet_fit(
    arrival_time_s: np.ndarray,
    x_m: np.ndarray,
    envelope: np.ndarray,
    baseline_signal: np.ndarray | None,
    *,
    dt_s: float,
    band_min_hz: float,
    band_max_hz: float,
    minimum_baseline_samples: int,
    minimum_packet_snr_db: float,
    minimum_arrival_r_squared: float,
    arrival_edge_margin_s: float | None,
    require_monotonic_arrivals: bool,
    time_start_s: float,
    time_end_s: float,
) -> PacketFitEstimate:
    """Estimate packet kinematics and all configured quality gates."""

    x = np.asarray(x_m, dtype=float)
    arrivals = np.asarray(arrival_time_s, dtype=float)
    envelope_array = np.asarray(envelope, dtype=float)
    if x.ndim != 1 or arrivals.ndim != 1 or len(x) != len(arrivals):
        raise ValueError("packet fit requires one arrival time for each probe position")
    if envelope_array.ndim != 2 or envelope_array.shape[1] != len(x):
        raise ValueError("packet envelope must have shape (time, probe)")
    if len(x) < 3:
        raise ValueError("packet fit requires at least three probe stations")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(arrivals)):
        raise ValueError("packet fit coordinates and arrivals must be finite")

    regression = linregress(x, arrivals)
    degrees_of_freedom = len(x) - 2
    t_multiplier = float(t.ppf(0.975, degrees_of_freedom))
    slope = float(regression.slope)
    stderr = float(regression.stderr) if np.isfinite(regression.stderr) else float("nan")
    slope_low = slope - t_multiplier * stderr
    slope_high = slope + t_multiplier * stderr
    slope_interval = (float(slope_low), float(slope_high))
    crosses_zero = not (np.isfinite(slope_low) and np.isfinite(slope_high)) or (
        slope_low <= 0.0 <= slope_high
    )
    velocity = float(1.0 / slope) if slope != 0.0 else None
    velocity_interval = (
        (float(1.0 / slope_high), float(1.0 / slope_low))
        if not crosses_zero and slope_low != 0.0 and slope_high != 0.0
        else None
    )

    baseline_count = 0 if baseline_signal is None else int(baseline_signal.shape[0])
    per_probe_snr: list[float] = []
    if baseline_signal is not None and baseline_signal.ndim == 2:
        if baseline_signal.shape[1] != len(x):
            raise ValueError("baseline signal probe count does not match packet envelope")
        for probe in range(len(x)):
            baseline_probe = baseline_signal[:, probe]
            median = float(np.median(baseline_probe))
            noise = 1.4826 * float(np.median(np.abs(baseline_probe - median)))
            peak = float(np.max(envelope_array[:, probe]))
            per_probe_snr.append(
                float(20.0 * np.log10(peak / noise)) if noise > 0.0 else float("inf")
            )
    minimum_snr = min(per_probe_snr) if per_probe_snr else None
    resolved_margin = (
        float(arrival_edge_margin_s)
        if arrival_edge_margin_s is not None
        else max(4.0 * float(dt_s), 1.0 / (float(band_max_hz) - float(band_min_hz)))
    )
    edge_passed = bool(np.all(
        (arrivals >= time_start_s + resolved_margin)
        & (arrivals <= time_end_s - resolved_margin)
    ))
    monotonic = bool(np.all(np.diff(arrivals) >= -abs(float(dt_s))))
    baseline_gate = baseline_count >= int(minimum_baseline_samples)
    snr_gate = bool(minimum_snr is not None and minimum_snr >= float(minimum_packet_snr_db))
    r_squared = float(regression.rvalue**2)
    r_squared_gate = bool(np.isfinite(r_squared) and r_squared >= float(minimum_arrival_r_squared))
    direction_gate = bool(slope > 0.0 and not crosses_zero)
    monotonic_gate = monotonic if require_monotonic_arrivals else True
    gates = {
        "baseline_samples": baseline_gate,
        "minimum_snr": snr_gate,
        "arrival_r_squared": r_squared_gate,
        "arrival_edge_margin": edge_passed,
        "monotonic_arrivals": monotonic_gate,
        "resolved_downstream_direction": direction_gate,
        "confidence_interval_identifiable": not crosses_zero,
    }
    if crosses_zero:
        status = "unresolved_direction"
    elif slope < 0.0:
        status = "upstream_out_of_scope"
    elif not all(gates.values()):
        status = "quality_gate_failed"
    else:
        status = "supported"
    return PacketFitEstimate(
        slope_s_m=slope,
        velocity_m_s=velocity,
        slope_interval_s_m=slope_interval,
        velocity_interval_m_s=velocity_interval,
        degrees_of_freedom=degrees_of_freedom,
        t_multiplier=t_multiplier,
        r_squared=r_squared,
        per_probe_snr_db=tuple(per_probe_snr),
        minimum_snr_db=minimum_snr,
        resolved_edge_margin_s=resolved_margin,
        baseline_sample_count=baseline_count,
        monotonic_arrivals=monotonic,
        gates=gates,
        fit_status=status,
    )


@executor("transient_wavepacket")
def run_transient_wavepacket(context: WorkflowContext) -> None:
    analysis = cast(TransientWavepacketAnalysis, context.analysis)
    variable, unit, raw_time, x_m, raw_values, selected = load_probe_signal(context)
    time = raw_time
    values = raw_values
    time, values, dt, resampled, cleanup = prepare_probe_time_grid(context, time, values)
    if cleanup is not None:
        context.add_cleanup(cleanup)
    baseline_values: np.ndarray | None = None
    if analysis.baseline_end_time_s is not None:
        baseline = reviewed_transient.subtract_quiescent_probe_baseline(
            time, values, analysis.baseline_end_time_s,
            minimum_samples=analysis.minimum_baseline_samples,
        )
        disturbance = baseline["disturbance"]
        baseline_count = baseline["baseline_sample_count"]
        baseline_values = disturbance[np.asarray(baseline["baseline_mask"], dtype=bool)]
    else:
        disturbance = values - np.mean(values, axis=0, keepdims=True)
        baseline_count = 0
    register_probe_trace_figures(
        context, raw_time=raw_time, raw_values=raw_values,
        prepared_time=time, prepared_values=disturbance, x_m=x_m, selected=selected,
        variable=variable, units=unit,
        preprocessing={
            "time_grid_policy": analysis.time_grid_policy,
            "baseline_end_time_s": analysis.baseline_end_time_s,
            "baseline_sample_count": baseline_count,
            "mean_subtraction": analysis.baseline_end_time_s is None,
            "resampled": resampled,
        },
    )
    filtered, _ = reviewed_transient.reconstruct_from_band(
        disturbance, dt, analysis.band_min_hz, analysis.band_max_hz
    )
    envelope = np.abs(hilbert(filtered, axis=0))
    register_probe_line_overlay(
        context, artifact_id="transient.filtered_overlay.figure",
        filename="filtered_overlay", x=time, values=filtered,
        x_label="Time [s]", y_label=f"Filtered signal [{unit}]",
        variable=variable, units=unit, selected=selected, x_m=x_m,
        interpretation="Band-pass reconstructed signal overlaid for every selected probe.",
        provenance={"band_hz": [analysis.band_min_hz, analysis.band_max_hz]},
    )
    register_probe_line_overlay(
        context, artifact_id="transient.envelope_overlay.figure",
        filename="envelope_overlay", x=time, values=envelope,
        x_label="Time [s]", y_label=f"Envelope [{unit}]",
        variable=variable, units=unit, selected=selected, x_m=x_m,
        interpretation="Analytic envelope overlaid for every selected probe.",
        provenance={"band_hz": [analysis.band_min_hz, analysis.band_max_hz]},
    )
    arrival_index = np.argmax(envelope, axis=0)
    arrival_time = time[arrival_index]
    baseline_sample_count = int(baseline_count)
    fit = estimate_packet_fit(
        arrival_time, x_m, envelope,
        baseline_values,
        dt_s=dt,
        band_min_hz=analysis.band_min_hz,
        band_max_hz=analysis.band_max_hz,
        minimum_baseline_samples=analysis.minimum_baseline_samples,
        minimum_packet_snr_db=analysis.minimum_packet_snr_db,
        minimum_arrival_r_squared=analysis.minimum_arrival_r_squared,
        arrival_edge_margin_s=analysis.arrival_edge_margin_s,
        require_monotonic_arrivals=analysis.require_monotonic_arrivals,
        time_start_s=float(time[0]),
        time_end_s=float(time[-1]),
    )
    envelope_path = context.data_dir / "packet_envelope.npz"
    np.savez_compressed(
        envelope_path, time_s=time, probe_indices=selected, x_m=x_m,
        filtered_signal=filtered, envelope=envelope, arrival_time_s=arrival_time,
        signal_unit=np.array(unit),
    )
    context.register(
        artifact_id="transient.envelope", path=envelope_path, kind="array",
        variable=variable, units=unit, coordinate_metadata={"time": "s", "probe_x": "m"},
        interpretation="Analytic envelope of the explicitly configured band-pass reconstruction.",
        provenance={"band_hz": [analysis.band_min_hz, analysis.band_max_hz],
                    "baseline_sample_count": baseline_count,
                    "time_grid_policy": analysis.time_grid_policy,
                    "resampled": resampled},
    )
    segment = min(int(analysis.stft_segment_samples), len(time))
    overlap = int(round(segment * analysis.overlap_fraction))
    frequency, stft_time, coefficients = stft(
        np.median(disturbance, axis=1), fs=1.0 / dt, window="hann",
        nperseg=segment, noverlap=overlap, boundary=None, padded=False,
    )
    # SciPy already supports an arbitrary signal axis.  Transform every
    # selected probe in one vectorized call, then restore the established
    # frequency × time × probe product contract.
    _probe_frequency, _probe_time, probe_coefficients = stft(
        disturbance.T, fs=1.0 / dt, window="hann", nperseg=segment,
        noverlap=overlap, boundary=None, padded=False, axis=-1,
    )
    if not np.allclose(_probe_frequency, frequency) or not np.allclose(_probe_time, stft_time):
        raise RuntimeError("vectorized per-probe STFT axes disagree with the representative STFT")
    if probe_coefficients.ndim != 3:
        raise RuntimeError("vectorized per-probe STFT did not return probe, frequency, time axes")
    per_probe_coefficients = np.transpose(probe_coefficients, (1, 2, 0))
    register_probe_stft_figure(
        context, artifact_id="transient.stft_figure", filename="stft_overlay",
        frequency_hz=frequency, time_s=stft_time, coefficients=per_probe_coefficients,
        selected=selected, x_m=x_m, variable=variable, units=unit,
        interpretation="Shared-scale STFT panels, one panel for each selected probe.",
        provenance={"segment_samples": segment, "overlap_samples": overlap,
                    "band_hz": [analysis.band_min_hz, analysis.band_max_hz]},
        time_half_width_s=segment * dt / 2.0,
    )
    stft_path = context.data_dir / "packet_stft.npz"
    np.savez_compressed(
        stft_path, frequency_hz=frequency, relative_time_s=stft_time,
        complex_stft=coefficients, representative=np.array("probe median"),
        probe_complex_stft=per_probe_coefficients,
        probe_indices=selected, x_m=x_m,
    )
    context.register(
        artifact_id="transient.stft", path=stft_path, kind="array", variable=variable,
        units=unit, coordinate_metadata={"frequency": "Hz", "relative_time": "s"},
        interpretation="Hann-window STFT of the probe-median disturbance; segment resolution is recorded.",
        provenance={"segment_samples": segment, "overlap_samples": overlap,
                    "time_grid_policy": analysis.time_grid_policy,
                    "resampled": resampled},
    )
    result = {
        "group_velocity_m_s": fit.velocity_m_s,
        "confidence_interval_95_m_s": (
            list(fit.velocity_interval_m_s) if fit.velocity_interval_m_s is not None else None
        ),
        "slope_confidence_interval_95_s_m": (
            list(fit.slope_interval_s_m) if fit.slope_interval_s_m is not None else None
        ),
        "confidence_interval_identifiable": fit.gates["confidence_interval_identifiable"],
        "arrival_time_regression_r_squared": fit.r_squared,
        "slope_s_m": fit.slope_s_m,
        "slope_standard_error_s_m": float(linregress(x_m, arrival_time).stderr),
        "probe_count": len(x_m),
        **fit.as_dict(),
        # Compatibility aliases retained inside the JSON diagnostic record;
        # the versioned product contract exposes the newer gate structure.
        "snr_db": fit.minimum_snr_db,
        "snr_gate_passed": fit.gates["minimum_snr"],
        "arrival_edge_gate_passed": fit.gates["arrival_edge_margin"],
        "packet_consistency_gate_passed": fit.gates["monotonic_arrivals"],
        "resolved_thresholds": {
            "minimum_baseline_samples": analysis.minimum_baseline_samples,
            "minimum_packet_snr_db": analysis.minimum_packet_snr_db,
            "minimum_arrival_r_squared": analysis.minimum_arrival_r_squared,
            "arrival_edge_margin_s": fit.resolved_edge_margin_s,
            "require_monotonic_arrivals": analysis.require_monotonic_arrivals,
            "confidence_level": 0.95,
        },
        "interpretation": "Kinematic packet-envelope regression; not a causality claim.",
    }
    path = context.data_dir / "group_velocity.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    interpretation = str(result["interpretation"])
    context.register(
        artifact_id="transient.group_velocity", path=path, kind="json", variable=variable,
        units="m/s", coordinate_metadata={}, interpretation=interpretation,
    )
