"""Surrogate/FDR-controlled nonlinear triad screening."""

from __future__ import annotations

import json

import numpy as np
from scipy.signal import find_peaks, welch

import pp_functions_database as reviewed_nonlinear
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .spectral import load_compact_signal


def _automatic_targets(signal: np.ndarray, fs: float, segment: int, maximum: float) -> tuple[np.ndarray, dict]:
    frequency, power = welch(signal, fs=fs, nperseg=segment, noverlap=segment // 2)
    eligible = (frequency > 0) & (frequency <= maximum)
    peaks, properties = find_peaks(power, prominence=max(float(np.max(power)) * 1.0e-6, 0.0))
    peaks = peaks[eligible[peaks]]
    if not peaks.size:
        peaks = np.flatnonzero(eligible)
    order = peaks[np.argsort(power[peaks])[::-1]][:3]
    targets = np.sort(frequency[order])
    if not targets.size:
        raise ValueError("automatic frequency selection found no non-DC bins")
    return targets, {
        "method": "three strongest local Welch-PSD peaks",
        "candidate_count": int(len(peaks)),
        "selected_frequency_hz": targets.tolist(),
        "native_resolution_hz": float(frequency[1] - frequency[0]),
    }


@executor("nonlinear_coupling")
def run_nonlinear_coupling(context: WorkflowContext) -> None:
    analysis = context.analysis
    variable, unit, time, _x_m, values, _ = load_compact_signal(context)
    signal = np.median(values, axis=1)
    fs = 1.0 / float(np.median(np.diff(time)))
    segment = min(int(analysis.segment_samples), len(time) // 2)
    segment = max(8, segment)
    if analysis.automatic_frequency_selection:
        targets, selection = _automatic_targets(signal, fs, segment, analysis.frequency_max_hz)
    else:
        targets = np.asarray(analysis.target_frequencies_hz, dtype=float)
        selection = {"method": "explicit target_frequencies_hz from analyses.yaml",
                     "selected_frequency_hz": targets.tolist()}
    significance = reviewed_nonlinear.compute_surrogate_triad_significance(
        signal, fs, targets, nperseg=segment,
        noverlap=int(round(segment * analysis.overlap_fraction)),
        n_surrogates=analysis.surrogate_count, fdr_alpha=analysis.fdr_alpha,
        minimum_independent_segments=2,
    )
    path = context.data_dir / "triad_significance.npz"
    np.savez_compressed(path, **significance)
    context.register(
        artifact_id="nonlinear.bicoherence", path=path, kind="array", variable=variable,
        units="dimensionless squared bicoherence", coordinate_metadata={"frequency": "Hz"},
        interpretation="Surrogate-tested squared bicoherence; association does not prove causal transfer.",
        provenance={"selection": selection, "fdr_alpha": analysis.fdr_alpha,
                    "surrogate_count": analysis.surrogate_count, "signal_scope": "probe median"},
    )
    summary = {
        "selection": selection,
        "triad_labels": significance["triad_labels"].tolist(),
        "significant_fdr": significance["significant_fdr"].tolist(),
        "fdr_adjusted_p_value": significance["fdr_adjusted_p_value"].tolist(),
        "interpretation": "Only FDR-significant entries are retained as measured coupling candidates.",
    }
    summary_path = context.data_dir / "triad_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="nonlinear.triads", path=summary_path, kind="json", variable=variable,
        units=None, coordinate_metadata={"frequency": "Hz"}, interpretation=summary["interpretation"],
    )
