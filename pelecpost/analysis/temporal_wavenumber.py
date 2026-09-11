"""Bounded, time-localized signed wavenumber analysis for probe apertures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pp_functions_database as reviewed_spectral
from pelecpost.config.models import (
    DirectionalWaveAnalysis,
    TemporalWavenumberEnabled,
)
from pelecpost.runtime.context import WorkflowContext
from pelecpost.runtime.parallel import (
    ParallelTask,
    plan_parallel_stage,
    stage_readonly_array,
)
from pelecpost.visualization import resolve_presentation, save_figure_variants


def _windows(time_s: np.ndarray, config: TemporalWavenumberEnabled) -> tuple[int, int, np.ndarray]:
    dt_values = np.diff(time_s)
    dt = float(np.median(dt_values))
    if dt <= 0.0 or not np.allclose(
        dt_values, dt, rtol=1.0e-8, atol=max(abs(dt) * 1.0e-10, 1.0e-15)
    ):
        raise ValueError("time-localized wavenumber analysis requires uniform time samples")
    n_window = int(round(config.window_duration_s / dt))
    if n_window < 8:
        raise ValueError(
            "temporal_wavenumber.window_duration_s must span at least 8 samples"
        )
    if n_window > len(time_s):
        raise ValueError(
            "temporal_wavenumber.window_duration_s exceeds the selected record duration"
        )
    hop = max(1, int(round(n_window * (1.0 - config.overlap_fraction))))
    starts = np.arange(0, len(time_s) - n_window + 1, hop, dtype=int)
    if len(starts) < 2:
        raise ValueError(
            "time-localized wavenumber analysis requires at least two complete windows"
        )
    return n_window, hop, starts


def _check_spatial_grid(x_m: np.ndarray) -> None:
    spacing = np.diff(x_m)
    if len(spacing) < 3 or np.any(spacing <= 0.0) or not np.allclose(
        spacing, float(np.median(spacing)), rtol=1.0e-5, atol=1.0e-12
    ):
        raise ValueError(
            "time-localized k-omega analysis requires strictly increasing, uniformly spaced probes"
        )


def _window_spectrum(
    values: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    start: int,
    n_window: int,
    analysis: DirectionalWaveAnalysis,
    config: TemporalWavenumberEnabled,
) -> dict[str, np.ndarray]:
    return reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values[start:start + n_window], time_s[start:start + n_window], x_m,
        temporal_window=config.window,
        spatial_window=analysis.spatial_window,
        temporal_mean_subtraction="mean",
    )


def _temporal_window_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute one localized wavenumber summary from read-only staged arrays."""
    values = np.load(payload["values_path"], mmap_mode="r")
    time_s = np.load(payload["time_path"], mmap_mode="r")
    x_m = np.load(payload["x_path"], mmap_mode="r")
    analysis = DirectionalWaveAnalysis.model_validate(payload["analysis"])
    spectrum = _window_spectrum(
        values, time_s, x_m, int(payload["start"]), int(payload["n_window"]),
        analysis, analysis.temporal_wavenumber,
    )
    indices = _frequency_selection(np.asarray(spectrum["frequency_hz"]), analysis)
    power = np.asarray(spectrum["power"], dtype=float)[indices]
    band_power = np.sum(power, axis=0)
    ridge = _ridge(spectrum, indices, analysis)
    return {
        "index": int(payload["index"]),
        "band_power": band_power,
        "energy": float(np.sum(band_power)),
        "ridge": ridge,
    }


def _temporal_snapshot_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute one selected localized f-k snapshot."""
    values = np.load(payload["values_path"], mmap_mode="r")
    time_s = np.load(payload["time_path"], mmap_mode="r")
    x_m = np.load(payload["x_path"], mmap_mode="r")
    analysis = DirectionalWaveAnalysis.model_validate(payload["analysis"])
    spectrum = _window_spectrum(
        values, time_s, x_m, int(payload["start"]), int(payload["n_window"]),
        analysis, analysis.temporal_wavenumber,
    )
    indices = _frequency_selection(np.asarray(spectrum["frequency_hz"]), analysis)
    return {
        "index": int(payload["index"]),
        "power": np.asarray(spectrum["power"], dtype=float)[indices],
    }


def _frequency_selection(
    frequency_hz: np.ndarray, analysis: DirectionalWaveAnalysis,
) -> np.ndarray:
    selected = np.flatnonzero(
        (frequency_hz > 0.0)
        & (frequency_hz >= analysis.frequency_min_hz)
        & (frequency_hz <= analysis.frequency_max_hz)
    )
    if not len(selected):
        raise ValueError(
            "temporal wavenumber frequency band contains no positive FFT bins; "
            "increase the window duration or adjust frequency_min_hz/frequency_max_hz"
        )
    return selected


def _ridge(
    spectrum: dict[str, np.ndarray],
    frequency_indices: np.ndarray,
    analysis: DirectionalWaveAnalysis,
) -> tuple[float, float, float, float, bool, bool]:
    frequency = np.asarray(spectrum["frequency_hz"], dtype=float)[frequency_indices]
    wavenumber = np.asarray(spectrum["wavenumber_rad_per_m"], dtype=float)
    power = np.asarray(spectrum["power"], dtype=float)[frequency_indices]
    band_power = np.sum(power, axis=0)
    if not np.any(np.isfinite(band_power)) or float(np.nanmax(band_power)) <= 0.0:
        return np.nan, np.nan, np.nan, np.nan, False, False
    wavenumber_ok = np.isfinite(wavenumber) & (np.abs(wavenumber) > 1.0e-12)
    if not np.any(wavenumber_ok):
        return np.nan, np.nan, np.nan, np.nan, False, False
    candidate_power = np.where(wavenumber_ok, band_power, -np.inf)
    k_index = int(np.argmax(candidate_power))
    k_value = float(wavenumber[k_index])
    f_index = int(np.argmax(power[:, k_index]))
    f_value = float(frequency[f_index])
    phase_speed = float(2.0 * np.pi * f_value / k_value)
    wavelength = float(2.0 * np.pi / abs(k_value))
    speed_valid = (
        analysis.expected_speed_min_m_s is None
        or analysis.expected_speed_max_m_s is None
        or analysis.expected_speed_min_m_s <= phase_speed <= analysis.expected_speed_max_m_s
    )
    nyquist_k = float(np.max(np.abs(wavenumber)))
    alias_valid = abs(k_value) < max(nyquist_k * (1.0 - 1.0 / max(len(wavenumber), 2)), 1.0e-12)
    return k_value, f_value, phase_speed, wavelength, bool(speed_valid), bool(alias_valid)


def compute_temporal_wavenumber(
    values: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    analysis: DirectionalWaveAnalysis,
    window_runner: Any | None = None,
) -> dict[str, Any]:
    """Compute bounded sliding-window f-k summaries and snapshot selection."""
    config = analysis.temporal_wavenumber
    if not isinstance(config, TemporalWavenumberEnabled):
        raise ValueError("temporal wavenumber processing is disabled")
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float).ravel()
    x_m = np.asarray(x_m, dtype=float).ravel()
    if values.ndim != 2 or values.shape != (len(time_s), len(x_m)):
        raise ValueError("values must have shape (len(time_s), len(x_m))")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(time_s)):
        raise ValueError("time-localized wavenumber inputs must be finite")
    order = np.argsort(x_m)
    if not np.array_equal(order, np.arange(len(order))):
        x_m = x_m[order]
        values = values[:, order]
    _check_spatial_grid(x_m)
    n_window, hop, starts = _windows(time_s, config)
    first = _window_spectrum(values, time_s, x_m, int(starts[0]), n_window, analysis, config)
    frequency_indices = _frequency_selection(np.asarray(first["frequency_hz"]), analysis)
    frequency = np.asarray(first["frequency_hz"])[frequency_indices]
    wavenumber = np.asarray(first["wavenumber_rad_per_m"])
    band_power_rows: list[np.ndarray] = []
    energies: list[float] = []
    ridge_rows: list[tuple[float, float, float, float, bool, bool]] = []
    if window_runner is None:
        for start in starts:
            spectrum = first if int(start) == int(starts[0]) else _window_spectrum(
                values, time_s, x_m, int(start), n_window, analysis, config
            )
            power = np.asarray(spectrum["power"], dtype=float)[frequency_indices]
            band_power = np.sum(power, axis=0)
            band_power_rows.append(band_power)
            energies.append(float(np.sum(band_power)))
            ridge_rows.append(_ridge(spectrum, frequency_indices, analysis))
    else:
        window_results = window_runner(starts, n_window)
        for result in sorted(window_results, key=lambda item: int(item["index"])):
            band_power_rows.append(np.asarray(result["band_power"], dtype=float))
            energies.append(float(result["energy"]))
            ridge_rows.append(tuple(result["ridge"]))
    band_power = np.asarray(band_power_rows)
    energy = np.asarray(energies)
    centers = np.asarray([time_s[int(start) + n_window // 2] for start in starts])
    peak_energy = float(np.nanmax(energy)) if len(energy) else 0.0
    relative_energy_db = 10.0 * np.log10(
        np.maximum(energy, 1.0e-300) / max(peak_energy, 1.0e-300)
    )
    active = np.isfinite(relative_energy_db) & (
        relative_energy_db >= config.minimum_relative_energy_db
    ) & (energy > 0.0)
    ridge_array = np.asarray(ridge_rows, dtype=float)
    k_valid = np.asarray([row[4] and row[5] for row in ridge_rows], dtype=bool)
    valid = active & k_valid & np.isfinite(ridge_array[:, 0])
    relative_power_db = 10.0 * np.log10(
        np.maximum(band_power, 1.0e-300) / max(float(np.nanmax(band_power)), 1.0e-300)
    )
    snapshot_indices, snapshot_roles, requested = _select_snapshots(
        centers, active, energy, config.snapshot_times_s
    )
    return {
        "n_window": n_window,
        "hop": hop,
        "window_starts": starts,
        "time_center_s": centers,
        "frequency_hz": frequency,
        "wavenumber_rad_m": wavenumber,
        "probe_x_m": x_m,
        "band_power": band_power,
        "relative_band_power_db": relative_power_db,
        "band_energy": energy,
        "relative_energy_db": relative_energy_db,
        "dominant_wavenumber_rad_m": ridge_array[:, 0],
        "dominant_frequency_hz": ridge_array[:, 1],
        "phase_speed_m_s": ridge_array[:, 2],
        "wavelength_m": ridge_array[:, 3],
        "active_mask": active,
        "valid_time_mask": valid,
        "snapshot_indices": snapshot_indices,
        "snapshot_roles": snapshot_roles,
        "snapshot_requested_time_s": requested,
        "preprocessing": {
            "window_duration_s": float(n_window * np.median(np.diff(time_s))),
            "overlap_fraction": config.overlap_fraction,
            "window": config.window,
            "frequency_band_hz": [analysis.frequency_min_hz, analysis.frequency_max_hz],
            "minimum_relative_energy_db": config.minimum_relative_energy_db,
        },
    }


def _select_snapshots(
    centers: np.ndarray,
    active: np.ndarray,
    energy: np.ndarray,
    requested_times: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if requested_times:
        if any(value < centers[0] or value > centers[-1] for value in requested_times):
            raise ValueError(
                "snapshot_times_s must fall within the centers of complete temporal windows"
            )
        indices = np.asarray([int(np.argmin(abs(centers - value))) for value in requested_times])
        return indices, np.asarray(["requested"] * len(indices)), np.asarray(requested_times, dtype=float)
    active_indices = np.flatnonzero(active)
    if not len(active_indices):
        active_indices = np.asarray([int(np.argmax(energy))])
    candidates: list[tuple[int, str]] = [
        (int(active_indices[0]), "first_active"),
        (int(active_indices[np.argmax(energy[active_indices])]), "peak_energy"),
        (int(active_indices[-1]), "last_active"),
    ]
    post = int(active_indices[-1]) + 1
    if post < len(centers):
        candidates.append((post, "post_active"))
    unique: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index, role in candidates:
        if index not in seen:
            unique.append((index, role))
            seen.add(index)
    return (
        np.asarray([item[0] for item in unique], dtype=int),
        np.asarray([item[1] for item in unique]),
        np.full(len(unique), np.nan),
    )


def compute_temporal_snapshots(
    values: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    analysis: DirectionalWaveAnalysis,
    summary: dict[str, Any],
    snapshot_runner: Any | None = None,
) -> dict[str, np.ndarray]:
    config = analysis.temporal_wavenumber
    assert isinstance(config, TemporalWavenumberEnabled)
    order = np.argsort(x_m)
    if not np.array_equal(order, np.arange(len(order))):
        x_m = np.asarray(x_m)[order]
        values = np.asarray(values)[:, order]
    if snapshot_runner is None:
        spectra = []
        for index in summary["snapshot_indices"]:
            spectra.append(_window_spectrum(
                values, time_s, x_m, int(summary["window_starts"][index]),
                int(summary["n_window"]), analysis, config,
            ))
        power = np.asarray([
            np.asarray(item["power"])[_frequency_selection(
                np.asarray(item["frequency_hz"]), analysis
            )]
            for item in spectra
        ])
    else:
        results = snapshot_runner(summary["snapshot_indices"], summary["window_starts"], summary["n_window"])
        power = np.asarray([
            np.asarray(item["power"], dtype=float)
            for item in sorted(results, key=lambda item: int(item["index"]))
        ])
    relative_power_db = 10.0 * np.log10(
        np.maximum(power, 1.0e-300) / max(float(np.nanmax(summary["band_power"])), 1.0e-300)
    )
    return {
        "snapshot_time_s": summary["time_center_s"][summary["snapshot_indices"]],
        "snapshot_requested_time_s": summary["snapshot_requested_time_s"],
        "frequency_hz": summary["frequency_hz"],
        "wavenumber_rad_m": summary["wavenumber_rad_m"],
        "power": power,
        "relative_power_db": relative_power_db,
        "valid_snapshot_mask": summary["valid_time_mask"][summary["snapshot_indices"]],
    }


def compute_temporal_wavenumber_parallel(
    context: WorkflowContext,
    values: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    analysis: DirectionalWaveAnalysis,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute localized summaries and snapshots with bounded worker processes."""
    config = analysis.temporal_wavenumber
    if not isinstance(config, TemporalWavenumberEnabled):
        raise ValueError("temporal wavenumber processing is disabled")
    if int(context.project.machine_file.compute.workers) <= 1:
        summary = compute_temporal_wavenumber(values, time_s, x_m, analysis)
        return summary, compute_temporal_snapshots(values, time_s, x_m, analysis, summary)

    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float).ravel()
    x_m = np.asarray(x_m, dtype=float).ravel()
    order = np.argsort(x_m)
    if not np.array_equal(order, np.arange(len(order))):
        x_m = x_m[order]
        values = values[:, order]
    n_window, _hop, starts = _windows(time_s, config)
    if len(starts) < 2:
        summary = compute_temporal_wavenumber(values, time_s, x_m, analysis)
        return summary, compute_temporal_snapshots(values, time_s, x_m, analysis, summary)

    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is None:
        scratch = context.run_dir / "scratch" / context.analysis.id
    elif not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    value_spec, value_cleanup = stage_readonly_array(
        values, scratch, prefix="temporal-wavenumber-values-"
    )
    time_spec, time_cleanup = stage_readonly_array(
        time_s, scratch, prefix="temporal-wavenumber-time-"
    )
    x_spec, x_cleanup = stage_readonly_array(
        x_m, scratch, prefix="temporal-wavenumber-x-"
    )
    context.add_cleanup(value_cleanup)
    context.add_cleanup(time_cleanup)
    context.add_cleanup(x_cleanup)
    analysis_dump = analysis.model_dump(mode="json")
    parent_gb = 0.1
    metadata = context.resource_metadata or {}
    signal_metadata = metadata.get("probe_signal", {})
    if signal_metadata.get("resident_bound_bytes"):
        parent_gb += float(signal_metadata["resident_bound_bytes"]) / 1024**3
    frequency_count = max(
        1, int(np.ceil((analysis.frequency_max_hz - analysis.frequency_min_hz) * n_window * np.median(np.diff(time_s))))
    )
    per_worker_gb = max(
        0.05,
        1.25 * (
            n_window * len(x_m) * 16 + frequency_count * len(x_m) * 16
        ) / 1024**3,
    )

    def make_payload(index: int, start: int) -> dict[str, Any]:
        return {
            "index": int(index), "start": int(start), "n_window": int(n_window),
            "values_path": value_spec.path, "time_path": time_spec.path,
            "x_path": x_spec.path, "analysis": analysis_dump,
        }

    window_plan = plan_parallel_stage(
        "temporal-wavenumber-windows",
        requested_workers=context.project.machine_file.compute.workers,
        task_count=len(starts), memory_limit_gb=context.project.machine_file.compute.memory_limit_gb,
        parent_resident_gb=parent_gb, per_worker_peak_gb=per_worker_gb,
    )

    def run_windows(window_starts: np.ndarray, _window_size: int) -> list[dict[str, Any]]:
        tasks = [
            ParallelTask(
                index, f"window-{index:06d}", make_payload(index, int(start)),
                {"window_index": index, "window_start": int(start)},
            )
            for index, start in enumerate(window_starts)
        ]
        results = context.run_parallel_stage(
            "temporal-wavenumber-windows", tasks, _temporal_window_worker,
            window_plan, recycle_after_tasks=1,
        )
        return [result.value for result in results]

    def run_snapshots(
        snapshot_indices: np.ndarray,
        window_starts: np.ndarray,
        window_size: int,
    ) -> list[dict[str, Any]]:
        tasks = [
            ParallelTask(
                position, f"snapshot-{position:03d}",
                make_payload(position, int(window_starts[int(index)])),
                {"snapshot_index": int(index), "snapshot_position": position},
            )
            for position, index in enumerate(snapshot_indices)
        ]
        snapshot_plan = plan_parallel_stage(
            "temporal-wavenumber-snapshots",
            requested_workers=context.project.machine_file.compute.workers,
            task_count=len(tasks), memory_limit_gb=context.project.machine_file.compute.memory_limit_gb,
            parent_resident_gb=parent_gb, per_worker_peak_gb=per_worker_gb,
        )
        results = context.run_parallel_stage(
            "temporal-wavenumber-snapshots", tasks, _temporal_snapshot_worker,
            snapshot_plan, recycle_after_tasks=1,
        )
        return [result.value for result in results]

    try:
        summary = compute_temporal_wavenumber(
            values, time_s, x_m, analysis, window_runner=run_windows,
        )
        snapshots = compute_temporal_snapshots(
            values, time_s, x_m, analysis, summary,
            snapshot_runner=run_snapshots,
        )
        return summary, snapshots
    except BaseException:
        value_cleanup()
        time_cleanup()
        x_cleanup()
        raise


def _register_figure(
    context: WorkflowContext,
    figure: Any,
    artifact_id: str,
    filename: str,
    variable: str,
    units: str,
    interpretation: str,
    provenance: dict[str, Any],
) -> None:
    presentation = resolve_presentation(
        context.project.analyses_file.presentation, context.analysis.presentation
    )
    for index, path in enumerate(save_figure_variants(
        figure, context.figure_dir / filename, presentation.figure
    )):
        suffix = "" if index == 0 else f".{path.suffix.lstrip('.') }"
        context.register(
            artifact_id=f"{artifact_id}{suffix}", path=path, kind="figure",
            variable=variable, units=units,
            coordinate_metadata={"time": "s", "wavenumber": "rad/m", "frequency": "Hz"},
            interpretation=interpretation,
            provenance=provenance | {"figure_format": path.suffix.lstrip(".")},
        )


def register_temporal_wavenumber_figures(
    context: WorkflowContext,
    *,
    values: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    variable: str,
    units: str,
    summary: dict[str, Any],
    snapshots: dict[str, np.ndarray],
) -> None:
    config = context.analysis.temporal_wavenumber
    assert isinstance(config, TemporalWavenumberEnabled)
    common = {
        "window_duration_s": summary["preprocessing"]["window_duration_s"],
        "snapshot_roles": summary["snapshot_roles"].tolist(),
        "snapshot_time_s": snapshots["snapshot_time_s"].tolist(),
        "snapshot_valid_mask": snapshots["valid_snapshot_mask"].tolist(),
        "interpretation": "Time-localized signed spatial spectrum; relative dB uses one global record maximum.",
    }
    presentation = resolve_presentation(
        context.project.analyses_file.presentation, context.analysis.presentation
    )
    figure, axis = plt.subplots(figsize=(presentation.figure.width_in, max(4.5, presentation.figure.height_in)))
    image = axis.pcolormesh(
        summary["time_center_s"], summary["wavenumber_rad_m"],
        summary["relative_band_power_db"].T, shading="auto",
        vmin=config.display_floor_db, vmax=0.0,
    )
    valid = np.where(summary["valid_time_mask"], summary["dominant_wavenumber_rad_m"], np.nan)
    axis.plot(summary["time_center_s"], valid, color="white", linewidth=1.5, label="dominant ridge")
    for time_value in snapshots["snapshot_time_s"]:
        axis.axvline(float(time_value), color="black", linestyle="--", alpha=0.5)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Wavenumber [rad/m]")
    figure.colorbar(image, ax=axis, label="Band power relative to record maximum [dB]")
    axis.legend(loc="best")
    figure.tight_layout()
    _register_figure(
        context, figure, "wave.temporal_wavenumber.figure", "temporal_wavenumber",
        variable, "relative dB", "Globally scaled band-integrated signed wavenumber evolution.",
        common,
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(presentation.figure.width_in, max(6.0, presentation.figure.height_in + 1.5)), sharex=True)
    series = (
        (summary["dominant_wavenumber_rad_m"], "Wavenumber [rad/m]"),
        (summary["dominant_frequency_hz"], "Frequency [Hz]"),
        (summary["wavelength_m"], "Wavelength [m]"),
        (summary["phase_speed_m_s"], "Phase speed [m/s]"),
    )
    for axis, (data, label) in zip(axes.flat, series):
        axis.plot(summary["time_center_s"], np.where(summary["valid_time_mask"], data, np.nan))
        axis.set_ylabel(label)
        axis.grid(False)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    figure.suptitle("Dominant time-localized spectral ridge")
    figure.tight_layout()
    _register_figure(
        context, figure, "wave.wavenumber_history.figure", "wavenumber_history",
        variable, "variable-dependent", "Dominant signed wavenumber and derived quantities; invalid windows are omitted.",
        common | {"valid_time_mask": summary["valid_time_mask"].tolist()},
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(presentation.figure.width_in, max(4.5, presentation.figure.height_in)))
    order = np.argsort(x_m)
    image = axis.pcolormesh(time_s, np.asarray(x_m)[order], np.asarray(values)[:, order].T, shading="auto", cmap="RdBu_r")
    for time_value in snapshots["snapshot_time_s"]:
        axis.axvline(float(time_value), color="black", linestyle="--", alpha=0.5)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Probe coordinate [m]")
    figure.colorbar(image, ax=axis, label=f"{variable} [{units}]")
    figure.tight_layout()
    _register_figure(
        context, figure, "wave.space_time.figure", "space_time",
        variable, units, "Method-ready SI probe signal over the selected spatial aperture and time.",
        common,
    )
    plt.close(figure)

    rows = len(snapshots["snapshot_time_s"])
    figure, axes = plt.subplots(rows, 1, squeeze=False, figsize=(presentation.figure.width_in, max(4.5, 3.0 * rows)), constrained_layout=True)
    image = None
    for row, time_value in enumerate(snapshots["snapshot_time_s"]):
        axis = axes[row, 0]
        image = axis.pcolormesh(
            snapshots["wavenumber_rad_m"], snapshots["frequency_hz"],
            snapshots["relative_power_db"][row], shading="auto",
            vmin=config.display_floor_db, vmax=0.0,
        )
        axis.set_ylabel("Frequency [Hz]")
        axis.set_title(f"t={float(time_value):.6g} s ({summary['snapshot_roles'][row]})")
    axes[-1, 0].set_xlabel("Wavenumber [rad/m]")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0].tolist(), label="Power relative to record maximum [dB]")
    _register_figure(
        context, figure, "wave.komega_snapshots.figure", "komega_snapshots",
        variable, "relative dB", "Shared-scale time-localized signed f-k snapshots.", common,
    )
    plt.close(figure)


def register_dispersion_figure(
    context: WorkflowContext,
    *,
    variable: str,
    units: str,
    frequency_hz: np.ndarray,
    phase_speed_m_s: np.ndarray,
    wavelength_m: np.ndarray,
    phase_valid_mask: np.ndarray,
    x_center_m: np.ndarray,
) -> None:
    """Register compact dispersion summaries of the full-record estimator."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    valid = np.asarray(phase_valid_mask, dtype=bool)
    phase_speed = np.asarray(phase_speed_m_s, dtype=float)
    wavelength = np.asarray(wavelength_m, dtype=float)

    def bands(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        median = np.full(len(frequency_hz), np.nan)
        lower = np.full(len(frequency_hz), np.nan)
        upper = np.full(len(frequency_hz), np.nan)
        for row in range(len(frequency_hz)):
            samples = values[row][valid[row] & np.isfinite(values[row])]
            if len(samples):
                median[row] = float(np.median(samples))
                lower[row] = float(np.percentile(samples, 16.0))
                upper[row] = float(np.percentile(samples, 84.0))
        return median, lower, upper

    speed_median, speed_lower, speed_upper = bands(phase_speed)
    wavelength_median, wavelength_lower, wavelength_upper = bands(wavelength)
    presentation = resolve_presentation(
        context.project.analyses_file.presentation, context.analysis.presentation
    )
    figure, axes = plt.subplots(1, 2, figsize=(presentation.figure.width_in, max(4.5, presentation.figure.height_in)))
    for axis, median, lower, upper, ylabel in (
        (axes[0], speed_median, speed_lower, speed_upper, "Phase speed [m/s]"),
        (axes[1], wavelength_median, wavelength_lower, wavelength_upper, "Wavelength [m]"),
    ):
        finite = np.isfinite(median)
        axis.plot(frequency_hz[finite], median[finite], color="C0", label="median over valid x")
        axis.fill_between(frequency_hz[finite], lower[finite], upper[finite], color="C0", alpha=0.2, label="16–84% over x")
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel(ylabel)
        axis.grid(False)
        axis.legend(loc="best")
    figure.suptitle("Full-record wavenumber dispersion")
    figure.tight_layout()
    _register_figure(
        context, figure, "wave.dispersion.figure", "dispersion",
        variable, "variable-dependent", "Median and spatial spread of the coherence-gated full-record dispersion estimates.",
        {"valid_frequency_fraction": float(np.mean(valid)), "x_center_m": np.asarray(x_center_m).tolist()},
    )
    plt.close(figure)
