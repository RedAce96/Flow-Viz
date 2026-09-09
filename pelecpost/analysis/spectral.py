"""Bounded stationary spectral analysis with an explicit SI boundary."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.signal import coherence, csd, detrend as signal_detrend, get_window, welch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pelecpost-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pelecpost-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pp_functions_database as reviewed_spectral
from pelecpost.config.models import (
    DirectionalWaveAnalysis,
    ProbeSpectrumAnalysis,
    SinglePulseAnalysis,
)
from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.io.signals import (
    open_compact_signal_workspace,
    open_probe_v2_signal_workspace,
)
from pelecpost.runtime.context import WorkflowContext

from .executors import executor


FIELD_NAMES = {
    "density": ("density", "rho"),
    "x_velocity": ("x_velocity", "u", "xvel"),
    "y_velocity": ("y_velocity", "v", "yvel"),
    "pressure": ("pressure", "p"),
    "temperature": ("temperature", "T", "temp"),
}
SI = {
    "density": (1000.0, "kg/m^3"),
    "x_velocity": (0.01, "m/s"),
    "y_velocity": (0.01, "m/s"),
    "pressure": (0.1, "Pa"),
    "temperature": (1.0, "K"),
}
MAX_PROBES_PER_FIGURE = 8


def _field(variable: str, fields: tuple[str, ...]) -> str:
    for candidate in FIELD_NAMES[variable]:
        if candidate in fields:
            return candidate
    raise KeyError(f"No storage field maps to {variable!r}")


def load_compact_signal(
    context: WorkflowContext,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one configured probe variable and convert it to SI."""
    analysis = context.analysis
    configured_variable = getattr(analysis, "variable", None)
    if configured_variable is None:
        raise TypeError(f"Recipe {analysis.recipe!r} has no probe variable")
    return load_compact_variable(
        context, configured_variable.value, getattr(analysis, "probe_indices", ())
    )


def load_compact_variable(
    context: WorkflowContext,
    variable: str,
    probe_indices: tuple[int, ...] = (),
    *,
    probe_input: Any | None = None,
    probe_inventory: Any | None = None,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an explicitly selected compact-probe variable through the SI boundary."""
    analysis = context.analysis
    if probe_input is None:
        probe_input = context.project.machine_file.inputs.probes
    if probe_input is None:
        raise UnsupportedCapabilityError(
            f"The {analysis.recipe} executor requires a configured probe source"
        )
    inventory = probe_inventory if probe_inventory is not None else context.plan.inventory.probes
    assert inventory is not None
    storage_field = _field(variable, inventory.fields)
    selected = np.asarray(
        probe_indices if probe_indices else range(inventory.probe_count), dtype=int
    )
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= inventory.probe_count):
        raise ValueError("probe_indices contain no valid probes")
    factor, unit = SI[variable]
    if context.project.case_file.case.solver_units.value == "si":
        factor = 1.0
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    memory_limit = float(context.project.machine_file.compute.memory_limit_gb)
    if probe_input.compact_file is not None:
        source = probe_input.compact_file
        if not source.is_absolute():
            source = (context.project.root / source).resolve()
        workspace = open_compact_signal_workspace(
            source, storage_field, selected, si_factor=factor,
            memory_limit_gb=memory_limit, scratch_directory=scratch,
        )
    else:
        patterns = []
        for pattern in probe_input.binary_files:
            configured = Path(pattern)
            if any(character in pattern for character in "*?["):
                parent = configured.parent
                if not parent.is_absolute():
                    parent = (context.project.root / parent).resolve()
                patterns.append(str(parent / configured.name))
            else:
                if not configured.is_absolute():
                    configured = (context.project.root / configured).resolve()
                patterns.append(str(configured))
        workspace = open_probe_v2_signal_workspace(
            tuple(patterns), storage_field, selected, si_factor=factor,
            memory_limit_gb=memory_limit, scratch_directory=scratch,
        )
    context.add_cleanup(workspace.close)
    assert context.resource_metadata is not None
    context.resource_metadata["compact_signal"] = {
        "storage": workspace.storage,
        "source_matrix_bytes": workspace.source_matrix_bytes,
        "resident_bound_bytes": workspace.resident_bound_bytes,
        "read_block_rows": workspace.read_block_rows,
    }
    return (
        variable, unit, workspace.time_s, workspace.x_m, workspace.values,
        workspace.selected_probe_indices,
    )


def _prepare_fft_grid(
    time: np.ndarray,
    values: np.ndarray,
    time_grid_policy: str,
    scratch_directory: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, Callable[[], None] | None]:
    """Return a uniform FFT grid or enforce that the native grid is uniform."""
    if len(time) < 2:
        raise ValueError("probe signal requires at least two time samples")
    differences = np.diff(time)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise ValueError("probe time must be finite and strictly increasing")
    dt = float(np.median(differences))
    uniform = np.allclose(
        differences, dt, rtol=1.0e-9, atol=max(abs(dt) * 1.0e-9, 1.0e-15)
    )
    if uniform:
        return time, values, dt, False, None
    if time_grid_policy == "require_uniform":
        raise ValueError(
            "probe sampling is nonuniform; choose time_grid_policy=resample_uniform "
            "or provide uniformly sampled probes"
        )
    sample_count = int(np.floor((time[-1] - time[0]) / dt)) + 1
    uniform_time = time[0] + np.arange(sample_count, dtype=float) * dt
    spill_path: Path | None = None
    if scratch_directory is None:
        uniform_values: np.ndarray = np.empty(
            (sample_count, values.shape[1]), dtype=np.float64
        )
        cleanup: Callable[[], None] | None = None
    else:
        scratch = Path(scratch_directory)
        scratch.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="pelec-post-fft-grid-", suffix=".mmap", dir=str(scratch)
        )
        os.close(descriptor)
        spill_path = Path(temporary_name)
        uniform_values = np.memmap(
            spill_path, mode="w+", dtype=np.float64,
            shape=(sample_count, values.shape[1]),
        )

        def cleanup() -> None:
            uniform_values.flush()
            mmap = getattr(uniform_values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            if spill_path is not None:
                try:
                    spill_path.unlink()
                except FileNotFoundError:
                    pass

    try:
        for column in range(values.shape[1]):
            uniform_values[:, column] = np.interp(
                uniform_time, time, np.asarray(values[:, column], dtype=float)
            )
    except BaseException:
        if cleanup is not None:
            cleanup()
        raise
    return uniform_time, uniform_values, dt, True, cleanup


def _processed_probe_signal(
    values: np.ndarray,
    detrend: str,
    window: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply configured detrending and the FFT window to each probe."""
    if detrend == "mean":
        centered = values - np.mean(values, axis=0, keepdims=True)
    elif detrend == "linear":
        centered = signal_detrend(values, axis=0, type="linear")
    else:
        centered = np.asarray(values, dtype=float)
    window_name = "boxcar" if window == "rectangular" else window
    weights = get_window(window_name, len(values), fftbins=True)
    return centered * weights[:, None], weights


def _single_sided_amplitude(
    values: np.ndarray,
    dt: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coherent-gain-corrected single-sided FFT amplitudes."""
    frequency = np.fft.rfftfreq(len(values), d=dt)
    amplitude = np.abs(np.fft.rfft(values, axis=0)) / np.sum(weights)
    if len(values) % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    return frequency, amplitude


def spectrum_from_signal(
    time: np.ndarray,
    values: np.ndarray,
    *,
    end_time_s: float | None,
    window: str,
    detrend: str,
    welch_segment_samples: int | None,
    overlap_fraction: float,
    time_grid_policy: str,
    frequency_max_hz: float | None,
    scratch_directory: Path | None,
) -> dict[str, Any]:
    """Compute the probe_spectrum pipeline products for an in-memory signal.

    Shared by run_probe_spectrum and the direct-probe comparison executor so
    both sides use identical preprocessing and grids.
    """
    if end_time_s is not None:
        if len(time) == 0:
            raise ValueError("probe signal requires at least two time samples")
        raw_mask = time <= float(end_time_s)
        raw_indices = np.flatnonzero(raw_mask)
        if len(raw_indices) < 2:
            raise ValueError("end_time_s leaves fewer than two time samples")
        if np.max(np.diff(raw_indices)) > 1:
            raise ValueError("end_time_s window must be contiguous from the first sample")
        time = time[raw_indices]
        values = values[raw_indices, :]
    fft_time, fft_values, dt, resampled, fft_cleanup = _prepare_fft_grid(
        time, values, time_grid_policy, scratch_directory
    )
    segment = welch_segment_samples or min(4096, len(fft_time))
    segment = min(segment, len(fft_time))
    overlap = round(segment * overlap_fraction)
    overlap = min(overlap, segment - 1)
    window_name = "boxcar" if window == "rectangular" else window
    scipy_detrend = {"mean": "constant", "linear": "linear", "none": False}[detrend]
    frequency, psd = welch(
        fft_values, fs=1.0 / dt, window=window_name, nperseg=segment,
        noverlap=overlap, detrend=scipy_detrend, axis=0, scaling="density",
        return_onesided=True,
    )
    if frequency_max_hz is not None:
        mask = frequency <= frequency_max_hz
        frequency, psd = frequency[mask], psd[mask]
    processed_values, weights = _processed_probe_signal(fft_values, detrend, window)
    fft_frequency, fft_amplitude = _single_sided_amplitude(processed_values, dt, weights)
    fft_keep = np.ones_like(fft_frequency, dtype=bool)
    if frequency_max_hz is not None:
        fft_keep &= fft_frequency <= frequency_max_hz
    return {
        "raw_time": time,
        "raw_values": values,
        "fft_time": fft_time,
        "fft_values": fft_values,
        "processed_values": processed_values,
        "fft_frequency": fft_frequency[fft_keep],
        "fft_amplitude": fft_amplitude[fft_keep],
        "welch_frequency": frequency,
        "psd": psd,
        "dt": dt,
        "resampled": resampled,
        "segment": segment,
        "overlap": overlap,
        "cleanup": fft_cleanup,
    }


def _plot_probe_time_fft(
    path: Path,
    raw_time: np.ndarray,
    raw_values: np.ndarray,
    time: np.ndarray,
    processed_values: np.ndarray,
    frequency: np.ndarray,
    amplitude: np.ndarray,
    x_m: np.ndarray,
    selected: np.ndarray,
    unit: str,
    detrend: str,
    window: str,
) -> tuple[Path, ...]:
    """Write one row of raw history, processed history, and FFT per probe."""
    paths: list[Path] = []
    page_count = (len(selected) + MAX_PROBES_PER_FIGURE - 1) // MAX_PROBES_PER_FIGURE
    for page, start in enumerate(range(0, len(selected), MAX_PROBES_PER_FIGURE), start=1):
        stop = min(start + MAX_PROBES_PER_FIGURE, len(selected))
        rows = stop - start
        page_path = (
            path if page_count == 1
            else path.with_name(f"{path.stem}_{page:03d}{path.suffix}")
        )
        fig, axes = plt.subplots(
            rows, 3, figsize=(15.0, max(3.2, 3.0 * rows)), squeeze=False
        )
        for row, (probe_index, coordinate) in enumerate(
            zip(selected[start:stop], x_m[start:stop])
        ):
            raw_axis, processed_axis, fft_axis = axes[row]
            column = start + row
            label = f"Probe {probe_index} - x={coordinate * 100.0:.3f} cm"
            raw_axis.plot(raw_time, raw_values[:, column], color="tab:blue", linewidth=1.0)
            raw_axis.set_title(f"{label} - Raw")
            raw_axis.set_xlabel("Time [s]")
            raw_axis.set_ylabel(f"Signal [{unit}]")
            raw_axis.grid(True, alpha=0.25)

            processed_axis.plot(
                time, processed_values[:, column], color="tab:orange", linewidth=1.0
            )
            processed_axis.set_title(f"{label} - Processed ({detrend}, {window})")
            processed_axis.set_xlabel("Time [s]")
            processed_axis.set_ylabel(f"Signal [{unit}]")
            processed_axis.grid(True, alpha=0.25)

            keep = (frequency > 0.0) & (amplitude[:, column] > 0.0)
            if np.any(keep):
                fft_axis.loglog(
                    frequency[keep], amplitude[keep, column],
                    color="tab:green", linewidth=1.0,
                )
            else:
                fft_axis.text(
                    0.5, 0.5, "No positive spectral amplitude",
                    transform=fft_axis.transAxes, ha="center", va="center",
                )
            fft_axis.set_title(f"{label} - FFT")
            fft_axis.set_xlabel("Frequency [Hz]")
            fft_axis.set_ylabel(f"|Amplitude| [{unit}]")
            fft_axis.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(page_path, dpi=180)
        plt.close(fig)
        paths.append(page_path)
    return tuple(paths)


@executor("probe_spectrum")
def run_probe_spectrum(context: WorkflowContext) -> None:
    analysis = cast(ProbeSpectrumAnalysis, context.analysis)
    variable, unit, time, x_m, values, selected = load_compact_signal(context)
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    products = spectrum_from_signal(
        time, values,
        end_time_s=analysis.end_time_s,
        window=analysis.window, detrend=analysis.detrend,
        welch_segment_samples=analysis.welch_segment_samples,
        overlap_fraction=analysis.overlap_fraction,
        time_grid_policy=analysis.time_grid_policy,
        frequency_max_hz=analysis.frequency_max_hz,
        scratch_directory=scratch,
    )
    if products["cleanup"] is not None:
        context.add_cleanup(products["cleanup"])
    time = products["raw_time"]
    values = products["raw_values"]
    fft_time = products["fft_time"]
    processed_preview = products["processed_values"]
    fft_frequency = products["fft_frequency"]
    fft_amplitude = products["fft_amplitude"]
    frequency = products["welch_frequency"]
    psd = products["psd"]
    dt = products["dt"]
    resampled = products["resampled"]
    segment = products["segment"]
    overlap = products["overlap"]
    _ = processed_preview
    data_path = context.data_dir / "stationary_spectrum.npz"
    np.savez_compressed(
        data_path,
        frequency_hz=frequency,
        psd=psd,
        probe_indices=selected,
        x_m=x_m,
        time_s=fft_time,
        signal_unit=np.array(unit),
        psd_unit=np.array(f"({unit})^2/Hz"),
        convention=np.array("one-sided Welch density"),
        time_grid_policy=np.array(analysis.time_grid_policy),
        resampled=np.array(resampled),
    )
    context.register(
        artifact_id="spectral.psd", path=data_path, kind="array", variable=variable,
        units=f"({unit})^2/Hz",
        coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation="One-sided Welch power spectral density; peaks are descriptive stationary content.",
        provenance={
            "si_boundary": f"compact {context.project.case_file.case.solver_units.value} to public SI",
            "window": analysis.window,
            "detrend": analysis.detrend, "segment_samples": segment,
            "overlap_samples": overlap,
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
            "signal_workspace": (context.resource_metadata or {})["compact_signal"],
        },
    )
    step = max(1, segment - overlap)
    segment_count = 1 + max(0, (len(fft_time) - segment) // step)
    confidence = {
        "schema_version": 1,
        "segment_count": segment_count,
        "approximate_degrees_of_freedom": 2 * segment_count,
        "frequency_resolution_hz": float(1.0 / (segment * dt)),
        "record_duration_s": float(fft_time[-1] - fft_time[0]),
        "interpretation": "Degrees of freedom are approximate because overlapped windowed segments are correlated.",
    }
    confidence_path = context.data_dir / "spectral_confidence.json"
    confidence_path.write_text(json.dumps(confidence, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="spectral.confidence", path=confidence_path, kind="json",
        variable=variable, units=None,
        interpretation="Degrees of freedom are approximate because overlapped windowed segments are correlated.",
    )
    order = np.argsort(x_m)
    pair_columns = np.column_stack((order[:-1], order[1:])) if len(order) > 1 else np.empty((0, 2), dtype=int)
    coherence_values = []
    phase_values = []
    window_name = "boxcar" if analysis.window == "rectangular" else analysis.window
    scipy_detrend = {"mean": "constant", "linear": "linear", "none": False}[analysis.detrend]
    coherence_frequency = np.fft.rfftfreq(segment, dt)
    for first, second in pair_columns:
        pair_frequency, pair_coherence = coherence(
            products["fft_values"][:, first], products["fft_values"][:, second], fs=1.0 / dt, window=window_name,
            nperseg=segment, noverlap=overlap, detrend=scipy_detrend,
        )
        _, pair_cross = csd(
            products["fft_values"][:, first], products["fft_values"][:, second], fs=1.0 / dt, window=window_name,
            nperseg=segment, noverlap=overlap, detrend=scipy_detrend, scaling="density",
        )
        coherence_frequency = pair_frequency
        coherence_values.append(pair_coherence)
        phase_values.append(np.angle(pair_cross))
    coherence_matrix = (
        np.asarray(coherence_values).T
        if coherence_values else np.empty((len(coherence_frequency), 0))
    )
    phase_matrix = (
        np.asarray(phase_values).T
        if phase_values else np.empty((len(coherence_frequency), 0))
    )
    coherence_keep = np.ones(len(coherence_frequency), dtype=bool)
    if analysis.frequency_max_hz is not None:
        coherence_keep &= coherence_frequency <= analysis.frequency_max_hz
    coherence_path = context.data_dir / "adjacent_probe_coherence.npz"
    np.savez_compressed(
        coherence_path,
        frequency_hz=coherence_frequency[coherence_keep],
        coherence_squared=coherence_matrix[coherence_keep],
        cross_phase_rad=phase_matrix[coherence_keep],
        upstream_probe_indices=selected[pair_columns[:, 0]] if len(pair_columns) else np.array([], dtype=int),
        downstream_probe_indices=selected[pair_columns[:, 1]] if len(pair_columns) else np.array([], dtype=int),
        upstream_x_m=x_m[pair_columns[:, 0]] if len(pair_columns) else np.array([]),
        downstream_x_m=x_m[pair_columns[:, 1]] if len(pair_columns) else np.array([]),
        convention=np.array("conj(upstream) * downstream"),
    )
    context.register(
        artifact_id="spectral.coherence", path=coherence_path, kind="array",
        variable=variable, units="dimensionless", coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Magnitude-squared coherence and downstream-minus-upstream cross phase for "
            "adjacent probes ordered by physical x coordinate."
        ),
        provenance={"segment_samples": segment, "overlap_samples": overlap,
                    "pair_count": len(pair_columns),
                    "time_grid_policy": analysis.time_grid_policy,
                    "resampled": resampled},
    )
    processed_values = products["processed_values"]
    fft_frequency = products["fft_frequency"]
    fft_amplitude = products["fft_amplitude"]
    signal_path = context.data_dir / "probe_time_fft.npz"
    np.savez_compressed(
        signal_path,
        raw_time_s=time,
        raw_values=values,
        time_s=fft_time,
        processed_values=processed_values,
        frequency_hz=fft_frequency,
        amplitude=fft_amplitude,
        probe_indices=selected,
        x_m=x_m,
        signal_unit=np.array(unit),
        time_grid_policy=np.array(analysis.time_grid_policy),
        resampled=np.array(resampled),
        convention=np.array("single-sided coherent-gain-corrected FFT amplitude"),
    )
    context.register(
        artifact_id="spectral.probe_signals", path=signal_path, kind="array",
        variable=variable, units=unit,
        coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Selected probe histories and configured-detrend/window single-sided FFT amplitudes."
        ),
        provenance={"time_grid_policy": analysis.time_grid_policy, "resampled": resampled,
                    "window": analysis.window, "detrend": analysis.detrend},
    )
    probe_figure_path = context.figure_dir / "probe_time_fft.png"
    probe_figure_paths = _plot_probe_time_fft(
        probe_figure_path, time, values, fft_time, processed_values,
        fft_frequency, fft_amplitude, x_m, selected, unit,
        analysis.detrend, analysis.window,
    )
    for page, path in enumerate(probe_figure_paths, start=1):
        context.register(
            artifact_id=(
                "spectral.probe_figure"
                if len(probe_figure_paths) == 1
                else f"spectral.probe_figure.{page:03d}"
            ),
            path=path, kind="figure", variable=variable, units=unit,
            coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
            interpretation=(
                "Per-probe raw history, processed history, and single-sided FFT amplitude."
            ),
            provenance={"time_grid_policy": analysis.time_grid_policy, "resampled": resampled,
                        "window": analysis.window, "detrend": analysis.detrend,
                        "page": page, "page_count": len(probe_figure_paths)},
        )
    figure_path = context.figure_dir / "stationary_spectrum.png"
    fig, axis = plt.subplots(figsize=(9, 5))
    positive = frequency > 0
    median = np.nanmedian(psd, axis=1)
    lower, upper = np.nanpercentile(psd, [10, 90], axis=1)
    if np.any(median[positive] > 0):
        axis.loglog(frequency[positive], median[positive], label="probe median")
    else:
        axis.plot(frequency[positive], median[positive], label="probe median")
        axis.text(0.5, 0.5, "No positive spectral power", transform=axis.transAxes, ha="center")
    axis.fill_between(frequency[positive], lower[positive], upper[positive], alpha=0.25, label="10–90%")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"PSD [({unit})²/Hz]")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="spectral.figure", path=figure_path, kind="figure", variable=variable,
        units=f"({unit})^2/Hz", coordinate_metadata={"frequency": "Hz"},
        interpretation="Median stationary spectrum with the 10th–90th percentile probe envelope.",
    )


@executor("single_pulse_response")
def run_single_pulse_response(context: WorkflowContext) -> None:
    analysis = cast(SinglePulseAnalysis, context.analysis)
    variable, unit, time, x_m, values, _ = load_compact_signal(context)
    baseline_end = analysis.baseline_end_time_s
    if baseline_end is None:
        baseline_end = analysis.start_time_s
    baseline = reviewed_spectral.subtract_quiescent_probe_baseline(
        time, values, baseline_end, minimum_samples=analysis.minimum_baseline_samples
    )
    response = baseline["disturbance"]
    response -= np.mean(response, axis=0, keepdims=True)
    response_complex = np.fft.rfft(response, axis=0) / len(time)
    source = reviewed_spectral.compute_single_pulse_source_spectrum(
        time,
        energy_per_pulse=analysis.energy_per_pulse_j_m,
        pulse_fwhm_s=analysis.pulse_fwhm_s,
        pulse_period_s=analysis.pulse_period_s,
        start_time_s=analysis.start_time_s,
        cutoff_sigma=analysis.cutoff_sigma,
        mean_subtraction="mean",
        window="none",
    )
    transfer = reviewed_spectral.compute_single_pulse_transfer_function(
        source["processed_complex"], response_complex,
        minimum_relative_source_amplitude=analysis.minimum_relative_source_amplitude,
    )
    frequency = source["frequency_hz"]
    keep = np.ones_like(frequency, dtype=bool)
    if analysis.frequency_max_hz is not None:
        keep &= frequency <= analysis.frequency_max_hz
    source_path = context.data_dir / "single_pulse_source_spectrum.npz"
    np.savez_compressed(
        source_path, time_s=source["time_s"], source_power_w_m=source["power"],
        frequency_hz=frequency[keep], physical_complex_j_m=source["physical_complex"][keep],
        physical_spectrum_j_m=source["physical_spectrum"][keep],
        ideal_spectrum_j_m=source["ideal_spectrum"][keep],
        processed_complex_w_m=source["processed_complex"][keep],
        sigma_s=np.array(source["sigma_s"]), center_s=np.array(source["center_s"]),
    )
    context.register(
        artifact_id="pulse.source_spectrum", path=source_path, kind="array",
        variable="source_power", units="J/m", coordinate_metadata={"time": "s", "frequency": "Hz"},
        interpretation=(
            "Code-matched finite Gaussian pulse history, physical continuous-time spectrum, "
            "and processed spectrum used for deconvolution."
        ),
        provenance={"energy_per_pulse_j_m": analysis.energy_per_pulse_j_m,
                    "pulse_fwhm_s": analysis.pulse_fwhm_s,
                    "pulse_period_s": analysis.pulse_period_s},
    )
    path = context.data_dir / "single_pulse_response.npz"
    np.savez_compressed(
        path, frequency_hz=frequency[keep], source_spectrum_j_m=source["physical_spectrum"][keep],
        transfer=transfer["transfer"][keep], transfer_magnitude=transfer["magnitude"][keep],
        transfer_phase_rad=transfer["phase_rad"][keep], valid_frequency=transfer["valid_frequency"][keep],
        baseline=baseline["baseline"], baseline_sample_count=np.array(baseline["baseline_sample_count"]),
        probe_x_m=x_m, response_unit=np.array(unit),
    )
    context.register(
        artifact_id="pulse.transfer", path=path, kind="array", variable=variable,
        units=f"{unit}/(J/m)", coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation="Finite-record single-pulse response/source ratio; invalid low-source bins are masked.",
        provenance={"estimator": "finite_record_single_pulse", "baseline_end_time_s": baseline_end},
    )
    quality = {
        "baseline_sample_count": baseline["baseline_sample_count"],
        "valid_frequency_fraction": float(np.mean(transfer["valid_frequency"])),
        "minimum_relative_source_amplitude": analysis.minimum_relative_source_amplitude,
    }
    quality_path = context.data_dir / "pulse_validity.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="pulse.validity", path=quality_path, kind="json", variable=variable,
        units=None, interpretation="Baseline and source-amplitude quality gates for pulse deconvolution.",
    )


def _komega_ridge(spectrum: dict, minimum_hz: float, maximum_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(spectrum["frequency_hz"], dtype=float)
    wavenumber = np.asarray(spectrum["wavenumber_rad_per_m"], dtype=float)
    power = np.asarray(spectrum["power"], dtype=float)
    selected = (frequency >= minimum_hz) & (frequency <= maximum_hz)
    if not np.any(selected):
        raise ValueError("k-omega sensitivity band contains no temporal FFT bins")
    band_power = power[selected]
    ridge = wavenumber[np.argmax(band_power, axis=1)]
    return frequency[selected], ridge


def _ridge_difference(reference: tuple[np.ndarray, np.ndarray], candidate: tuple[np.ndarray, np.ndarray]) -> dict:
    frequency, ridge = reference
    candidate_frequency, candidate_ridge = candidate
    common = (
        (frequency >= candidate_frequency[0])
        & (frequency <= candidate_frequency[-1])
    )
    if not np.any(common):
        raise ValueError("k-omega sensitivity spectra have no common frequency bins")
    interpolated = np.interp(frequency[common], candidate_frequency, candidate_ridge)
    difference = np.abs(ridge[common] - interpolated)
    return {
        "compared_frequency_count": int(np.count_nonzero(common)),
        "median_absolute_ridge_difference_rad_m": float(np.median(difference)),
        "percentile_90_absolute_ridge_difference_rad_m": float(np.percentile(difference, 90.0)),
        "maximum_absolute_ridge_difference_rad_m": float(np.max(difference)),
    }


def _komega_sensitivity(
    values: np.ndarray,
    time: np.ndarray,
    x_m: np.ndarray,
    analysis: DirectionalWaveAnalysis,
    full: dict,
) -> dict:
    reference = _komega_ridge(full, analysis.frequency_min_hz, analysis.frequency_max_hz)
    alternate_window = "hann" if analysis.temporal_window == "rectangular" else "rectangular"
    alternate = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values, time, x_m, temporal_window=alternate_window,
        spatial_window=analysis.spatial_window, temporal_mean_subtraction="mean",
    )
    half = len(time) // 2
    if half < 8:
        raise ValueError("k-omega block sensitivity requires at least 16 temporal samples")
    first = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values[:half], time[:half], x_m, temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window, temporal_mean_subtraction="mean",
    )
    second = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values[-half:], time[-half:], x_m, temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window, temporal_mean_subtraction="mean",
    )
    return {
        "schema_version": 1,
        "frequency_band_hz": [analysis.frequency_min_hz, analysis.frequency_max_hz],
        "configured_temporal_window": analysis.temporal_window,
        "alternate_temporal_window": alternate_window,
        "full_record_sample_count": len(time),
        "block_sample_count": half,
        "window_sensitivity": _ridge_difference(
            reference,
            _komega_ridge(alternate, analysis.frequency_min_hz, analysis.frequency_max_hz),
        ),
        "first_vs_second_half_block_sensitivity": _ridge_difference(
            _komega_ridge(first, analysis.frequency_min_hz, analysis.frequency_max_hz),
            _komega_ridge(second, analysis.frequency_min_hz, analysis.frequency_max_hz),
        ),
        "interpretation": (
            "Dominant-ridge changes under temporal-window and half-record perturbations; "
            "large differences limit directional-wave interpretation."
        ),
    }


@executor("directional_wave")
def run_directional_wave(context: WorkflowContext) -> None:
    analysis = cast(DirectionalWaveAnalysis, context.analysis)
    variable, unit, time, x_m, values, _ = load_compact_signal(context)
    speed_bounds = None
    if analysis.expected_speed_min_m_s is not None and analysis.expected_speed_max_m_s is not None:
        speed_bounds = (analysis.expected_speed_min_m_s, analysis.expected_speed_max_m_s)
    segment = min(16384, max(8, len(time) // 2))
    local = reviewed_spectral.compute_frequency_resolved_wavenumber(
        values, x_m, 1.0 / np.median(np.diff(time)),
        (analysis.frequency_min_hz, analysis.frequency_max_hz),
        nperseg=segment, noverlap=segment // 2,
        spatial_window_size=min(101, len(x_m) if len(x_m) % 2 else len(x_m) - 1),
        spatial_step=max(1, len(x_m) // 20),
        fft_batch_size=context.project.machine_file.compute.fft_batch_size,
        min_coherence=analysis.minimum_coherence,
        phase_speed_bounds=speed_bounds,
    )
    komega = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values, time, x_m, temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window, temporal_mean_subtraction="mean",
    )
    path = context.data_dir / "complex_wavenumber.npz"
    arrays = {
        "frequency_hz": local["frequency_hz"], "x_center_m": local["x_center_m"],
        "alpha_real_rad_m": local["alpha_real_rad_per_m"],
        "alpha_imag_rad_m": local["alpha_imag_rad_per_m"],
        "alpha_real_ci95_rad_m": local["alpha_real_ci95_rad_per_m"],
        "alpha_imag_ci95_rad_m": local["alpha_imag_ci95_rad_per_m"],
        "amplification_rate_per_m": local["amplification_rate_per_m"],
        "phase_speed_m_s": local["phase_speed_m_per_s"],
        "phase_valid_mask": local["phase_valid_mask"], "growth_valid_mask": local["growth_valid_mask"],
        "coherence_squared": local["mean_coherence_squared"],
        "phase_fit_r_squared": local["phase_fit_r_squared"],
        "amplitude_fit_r_squared": local["amplitude_fit_r_squared"],
        "spatial_alias_margin": local["spatial_alias_margin"],
    }
    np.savez_compressed(path, **arrays)
    accepted = float(np.mean(local["phase_valid_mask"]))
    growth = float(np.mean(local["growth_valid_mask"]))
    context.register(
        artifact_id="wave.wavenumber", path=path, kind="array", variable=variable,
        units="rad/m", coordinate_metadata={"frequency": "Hz", "x": "m", "wavenumber": "rad/m"},
        interpretation="Coherence-gated dominant-wave estimate; it is not an LST/PSE eigensolution.",
        provenance={"phase_convention": local["phase_convention"], "accepted_fraction": accepted,
                    "growth_accepted_fraction": growth},
    )
    spectrum_path = context.data_dir / "local_spatial_spectrum.npz"
    np.savez_compressed(
        spectrum_path, frequency_hz=local["frequency_hz"],
        x_center_m=local["x_center_m"], spectral_power=local["spectral_power"],
        relative_spectral_power_db=local["relative_spectral_power_db"],
        adjacent_coherence_squared=local["adjacent_coherence_squared"],
        probe_x_m=local["probe_x_sorted_m"],
    )
    context.register(
        artifact_id="wave.spatial_spectrum", path=spectrum_path, kind="array",
        variable=variable, units=f"({unit})^2", coordinate_metadata={"frequency": "Hz", "x": "m"},
        interpretation="Welch-averaged local spatial spectral power and adjacent-probe coherence.",
    )
    komega_path = context.data_dir / "komega_spectrum.npz"
    np.savez_compressed(
        komega_path, frequency_hz=komega["frequency_hz"],
        wavenumber_rad_m=komega["wavenumber_rad_per_m"], power=komega["power"],
        amplitude=komega["amplitude"],
        native_frequency_resolution_hz=np.array(komega["native_frequency_resolution_hz"]),
        native_wavenumber_resolution_rad_m=np.array(komega["native_wavenumber_resolution_rad_per_m"]),
    )
    context.register(
        artifact_id="wave.komega", path=komega_path, kind="array", variable=variable,
        units=f"({unit})^2", coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
        interpretation="Signed numerical k–omega map; positive k denotes downstream cos(omega t - k x).",
        provenance={"temporal_window": analysis.temporal_window,
                    "spatial_window": analysis.spatial_window,
                    "direction_convention": komega["direction_convention"]},
    )
    sensitivity = _komega_sensitivity(values, time, x_m, analysis, komega)
    sensitivity_path = context.data_dir / "komega_sensitivity.json"
    sensitivity_path.write_text(json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="wave.komega_sensitivity", path=sensitivity_path, kind="json",
        variable=variable, units="rad/m", coordinate_metadata={"frequency": "Hz"},
        interpretation=sensitivity["interpretation"],
    )
    figure_path = context.figure_dir / "komega.png"
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.pcolormesh(
        komega["wavenumber_rad_per_m"], komega["frequency_hz"],
        10.0 * np.log10(komega["power"] / max(float(np.nanmax(komega["power"])), 1e-300) + 1e-300),
        shading="auto", vmin=-60, vmax=0,
    )
    axis.set_xlabel("Wavenumber [rad/m]")
    axis.set_ylabel("Frequency [Hz]")
    fig.colorbar(image, ax=axis, label="Relative power [dB]")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="wave.komega.figure", path=figure_path, kind="figure", variable=variable,
        units="relative dB", coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
        interpretation="Signed k–omega map; positive k denotes downstream cos(omega t - k x).",
    )
