"""Transient packet filtering, arrival, and propagation uncertainty."""

from __future__ import annotations

import json

import numpy as np
from scipy.signal import hilbert, stft
from scipy.stats import linregress

import pp_functions_database as reviewed_transient
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .spectral import load_compact_signal


@executor("transient_wavepacket")
def run_transient_wavepacket(context: WorkflowContext) -> None:
    analysis = context.analysis
    variable, unit, time, x_m, values, selected = load_compact_signal(context)
    if analysis.baseline_end_time_s is not None:
        baseline = reviewed_transient.subtract_quiescent_probe_baseline(
            time, values, analysis.baseline_end_time_s, minimum_samples=8
        )
        disturbance = baseline["disturbance"]
        baseline_count = baseline["baseline_sample_count"]
    else:
        disturbance = values - np.mean(values, axis=0, keepdims=True)
        baseline_count = 0
    dt = float(np.median(np.diff(time)))
    filtered, _ = reviewed_transient.reconstruct_from_band(
        disturbance, dt, analysis.band_min_hz, analysis.band_max_hz
    )
    envelope = np.abs(hilbert(filtered, axis=0))
    arrival_index = np.argmax(envelope, axis=0)
    arrival_time = time[arrival_index]
    regression = linregress(x_m, arrival_time)
    if regression.slope <= 0:
        raise ValueError("arrival-time regression does not indicate downstream propagation")
    group_velocity = 1.0 / regression.slope
    slope_low = regression.slope - 1.96 * regression.stderr
    slope_high = regression.slope + 1.96 * regression.stderr
    velocity_interval = (
        np.array([1.0 / slope_high, 1.0 / slope_low])
        if slope_low > 0 else np.array([1.0 / slope_high, np.inf])
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
                    "baseline_sample_count": baseline_count},
    )
    segment = min(int(analysis.stft_segment_samples), len(time))
    overlap = int(round(segment * analysis.overlap_fraction))
    frequency, stft_time, coefficients = stft(
        np.median(disturbance, axis=1), fs=1.0 / dt, window="hann",
        nperseg=segment, noverlap=overlap, boundary=None, padded=False,
    )
    stft_path = context.data_dir / "packet_stft.npz"
    np.savez_compressed(
        stft_path, frequency_hz=frequency, relative_time_s=stft_time,
        complex_stft=coefficients, representative=np.array("probe median"),
    )
    context.register(
        artifact_id="transient.stft", path=stft_path, kind="array", variable=variable,
        units=unit, coordinate_metadata={"frequency": "Hz", "relative_time": "s"},
        interpretation="Hann-window STFT of the probe-median disturbance; segment resolution is recorded.",
        provenance={"segment_samples": segment, "overlap_samples": overlap},
    )
    result = {
        "group_velocity_m_s": group_velocity,
        "confidence_interval_95_m_s": velocity_interval.tolist(),
        "arrival_time_regression_r_squared": float(regression.rvalue**2),
        "slope_s_m": float(regression.slope),
        "slope_standard_error_s_m": float(regression.stderr),
        "probe_count": len(x_m),
        "interpretation": "Kinematic packet-envelope regression; not a causality claim.",
    }
    path = context.data_dir / "group_velocity.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="transient.group_velocity", path=path, kind="json", variable=variable,
        units="m/s", coordinate_metadata={}, interpretation=result["interpretation"],
    )
