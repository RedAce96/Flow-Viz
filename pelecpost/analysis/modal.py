"""Bounded probe POD/SPOD/DMD screening using reviewed kernels."""

from __future__ import annotations

import numpy as np

import pp_modal_database as reviewed_modal
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .spectral import load_compact_signal


def _register_npz(context: WorkflowContext, name: str, arrays: dict, interpretation: str) -> None:
    path = context.data_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    context.register(
        artifact_id=f"modal.{name}", path=path, kind="array",
        variable=context.analysis.variable.value, units="variable-dependent",
        coordinate_metadata={"probe_x": "m", "frequency": "Hz", "time": "s"},
        interpretation=interpretation,
    )


@executor("modal_screening")
def run_modal_screening(context: WorkflowContext) -> None:
    analysis = context.analysis
    variable, _unit, time, x_m, values, _ = load_compact_signal(context)
    stride = int(analysis.probe_stride)
    x_m = x_m[::stride]
    values = values[:, ::stride]
    weights = reviewed_modal.trapezoidal_spatial_weights(x_m)
    weights /= np.mean(weights)
    dataset = reviewed_modal.SnapshotMatrix(
        time, values, x_m, variable=variable, weights=weights
    )
    mode_count = min(analysis.mode_count, values.shape[1], values.shape[0])
    pod = reviewed_modal.compute_pod(dataset, n_modes=mode_count)
    _register_npz(context, "pod", {
        "modes": pod["modes"], "temporal_coefficients": pod["temporal_coefficients"],
        "singular_values": pod["singular_values"], "energy_fraction": pod["energy_fraction"],
        "mean": pod["mean"], "coordinates_m": pod["coordinates"],
    }, "Weighted descriptive POD of the probe measure; not a stability eigenmode.")
    segment = min(int(analysis.spod_segment_samples), len(time) // 2)
    segment = max(8, segment)
    spod = reviewed_modal.compute_spod(
        dataset, nperseg=segment, noverlap=segment // 2,
        n_modes=min(mode_count, 3), frequency_stride=max(1, segment // 256),
    )
    _register_npz(context, "spod", {
        "frequency_hz": spod["frequency_hz"], "eigenvalues": spod["eigenvalues"],
        "modes": spod["modes"], "coordinates_m": spod["coordinates"],
        "block_count": np.array(spod["n_blocks"]),
    }, "Welch-block SPOD with explicit block count and probe quadrature weights.")
    dmd_rank = min(max(analysis.dmd_ranks), values.shape[1], values.shape[0] - 1)
    dmd = reviewed_modal.compute_dmd(dataset, n_modes=dmd_rank)
    _register_npz(context, "dmd", {
        "modes": dmd["modes"], "eigenvalues": dmd["eigenvalues"],
        "frequency_hz": dmd["frequency_hz"], "growth_rate_per_s": dmd["growth_rate_per_s"],
        "amplitudes": dmd["amplitudes"],
        "retained_condition_number": np.array(dmd["retained_condition_number"]),
        "coordinates_m": dmd["coordinates"],
    }, "Exact DMD of mean-subtracted probe snapshots; growth is descriptive and conditioning-limited.")
    sensitivity = reviewed_modal.compute_modal_sensitivity(
        dataset, pod_modes=min(2, mode_count), dmd_ranks=analysis.dmd_ranks,
        window_fractions=analysis.sensitivity_windows,
    )
    _register_npz(context, "sensitivity", {
        key: value for key, value in sensitivity.items() if key != "interpretation"
    }, sensitivity["interpretation"])
