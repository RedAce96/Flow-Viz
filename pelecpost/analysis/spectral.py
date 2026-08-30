"""Bounded stationary spectral analysis with an explicit SI boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pelecpost-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pelecpost-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _read_compact(path: Path, field: str, probe_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as archive:
        time = np.asarray(archive["time"], dtype=float)
        x_m = np.asarray(archive["probes/requested_x_cm"], dtype=float)[probe_ids] * 0.01
        # h5py requires increasing unique fancy indices.
        order = np.argsort(probe_ids)
        sorted_ids = probe_ids[order]
        sorted_values = np.asarray(archive[f"fields/{field}"][:, sorted_ids], dtype=float)
        restore = np.argsort(order)
        return time, x_m, sorted_values[:, restore]


@executor("probe_spectrum")
def run_probe_spectrum(context: WorkflowContext) -> None:
    analysis = context.analysis
    probe_input = context.project.machine_file.inputs.probes
    if probe_input is None or probe_input.compact_file is None:
        raise UnsupportedCapabilityError(
            "The migrated probe_spectrum executor currently requires a compact HDF5 probe archive"
        )
    source = probe_input.compact_file
    if not source.is_absolute():
        source = (context.project.root / source).resolve()
    inventory = context.plan.inventory.probes
    assert inventory is not None
    variable = analysis.variable.value
    storage_field = _field(variable, inventory.fields)
    selected = np.asarray(
        analysis.probe_indices if analysis.probe_indices else range(inventory.probe_count),
        dtype=int,
    )
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= inventory.probe_count):
        raise ValueError("probe_indices contain no valid probes")
    time, x_m, values = _read_compact(source, storage_field, selected)
    factor, unit = SI[variable]
    values *= factor
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
            "storage_field": storage_field, "si_factor": factor, "window": analysis.window,
            "detrend": analysis.detrend, "segment_samples": segment,
            "overlap_samples": overlap,
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
