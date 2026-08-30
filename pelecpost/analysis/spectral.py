"""Bounded stationary spectral analysis with an explicit SI boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import welch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pelecpost-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pelecpost-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pp_functions_database as reviewed_spectral
from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.io.signals import open_compact_signal_workspace
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
    probe_input = context.project.machine_file.inputs.probes
    if probe_input is None or probe_input.compact_file is None:
        raise UnsupportedCapabilityError(
            f"The migrated {analysis.recipe} executor currently requires a compact HDF5 probe archive"
        )
    source = probe_input.compact_file
    if not source.is_absolute():
        source = (context.project.root / source).resolve()
    inventory = context.plan.inventory.probes
    assert inventory is not None
    variable = analysis.variable.value
    storage_field = _field(variable, inventory.fields)
    configured = getattr(analysis, "probe_indices", ())
    selected = np.asarray(configured if configured else range(inventory.probe_count), dtype=int)
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= inventory.probe_count):
        raise ValueError("probe_indices contain no valid probes")
    factor, unit = SI[variable]
    if context.project.case_file.case.solver_units.value == "si":
        factor = 1.0
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    workspace = open_compact_signal_workspace(
        source, storage_field, selected, si_factor=factor,
        memory_limit_gb=float(context.project.machine_file.compute.memory_limit_gb),
        scratch_directory=scratch,
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


@executor("probe_spectrum")
def run_probe_spectrum(context: WorkflowContext) -> None:
    analysis = context.analysis
    variable, unit, time, x_m, values, selected = load_compact_signal(context)
    dt = float(np.median(np.diff(time)))
    segment = analysis.welch_segment_samples or min(4096, len(time))
    segment = min(segment, len(time))
    overlap = int(round(segment * analysis.overlap_fraction))
    window = "boxcar" if analysis.window == "rectangular" else analysis.window
    detrend = {"mean": "constant", "linear": "linear", "none": False}[analysis.detrend]
    frequency, psd = welch(
        values, fs=1.0 / dt, window=window, nperseg=segment,
        noverlap=overlap, detrend=detrend, axis=0, scaling="density",
        return_onesided=True,
    )
    if analysis.frequency_max_hz is not None:
        mask = frequency <= analysis.frequency_max_hz
        frequency, psd = frequency[mask], psd[mask]
    data_path = context.data_dir / "stationary_spectrum.npz"
    np.savez_compressed(
        data_path,
        frequency_hz=frequency,
        psd=psd,
        probe_indices=selected,
        x_m=x_m,
        signal_unit=np.array(unit),
        psd_unit=np.array(f"({unit})^2/Hz"),
        convention=np.array("one-sided Welch density"),
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
            "signal_workspace": context.resource_metadata["compact_signal"],
        },
    )
    step = max(1, segment - overlap)
    segment_count = 1 + max(0, (len(time) - segment) // step)
    confidence = {
        "schema_version": 1,
        "segment_count": segment_count,
        "approximate_degrees_of_freedom": 2 * segment_count,
        "frequency_resolution_hz": float(1.0 / (segment * dt)),
        "record_duration_s": float(time[-1] - time[0]),
        "interpretation": "Degrees of freedom are approximate because overlapped windowed segments are correlated.",
    }
    confidence_path = context.data_dir / "spectral_confidence.json"
    confidence_path.write_text(json.dumps(confidence, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="spectral.confidence", path=confidence_path, kind="json",
        variable=variable, units=None,
        interpretation=confidence["interpretation"],
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
    analysis = context.analysis
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


@executor("directional_wave")
def run_directional_wave(context: WorkflowContext) -> None:
    analysis = context.analysis
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
    path = context.data_dir / "directional_wave.npz"
    arrays = {
        "frequency_hz": local["frequency_hz"], "x_center_m": local["x_center_m"],
        "alpha_real_rad_m": local["alpha_real_rad_per_m"],
        "alpha_imag_rad_m": local["alpha_imag_rad_per_m"],
        "amplification_rate_per_m": local["amplification_rate_per_m"],
        "phase_speed_m_s": local["phase_speed_m_per_s"],
        "phase_valid_mask": local["phase_valid_mask"], "growth_valid_mask": local["growth_valid_mask"],
        "coherence_squared": local["mean_coherence_squared"],
        "komega_frequency_hz": komega["frequency_hz"],
        "komega_wavenumber_rad_m": komega["wavenumber_rad_per_m"],
        "komega_power": komega["power"],
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
        artifact_id="wave.komega", path=figure_path, kind="figure", variable=variable,
        units="relative dB", coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
        interpretation="Signed k–omega map; positive k denotes downstream cos(omega t - k x).",
    )
