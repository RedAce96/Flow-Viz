"""Bounded stationary spectral analysis with an explicit SI boundary."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.signal import coherence, csd, find_peaks, get_window, welch
from scipy.signal import detrend as signal_detrend

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
    TemporalWavenumberEnabled,
)
from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.io.signals import open_probe_signal_workspace
from pelecpost.io.thermal_source import (
    build_source_audit,
    canonicalize_segments,
    count_intersecting_pulses,
    discover_source_segments,
    measured_source_spectrum,
    rebin_history,
    validate_history_configuration,
)
from pelecpost.runtime.context import WorkflowContext
from pelecpost.runtime.parallel import ParallelTask, plan_parallel_stage, stage_readonly_array

from .executors import executor
from .probe_plotting import register_probe_line_overlay, register_probe_trace_figures
from .products import product_contract
from .temporal_wavenumber import (
    register_dispersion_figure,
    register_temporal_wavenumber_figures,
)

FIELD_NAMES = {
    "density": ("density", "rho"),
    "x_velocity": ("x_velocity", "u", "xvel"),
    "y_velocity": ("y_velocity", "v", "yvel"),
    "pressure": ("pressure", "p"),
    "temperature": ("temperature", "T", "temp"),
    "vorticity": ("vorticity", "omega_z"),
}
SI = {
    "density": (1000.0, "kg/m^3"),
    "x_velocity": (0.01, "m/s"),
    "y_velocity": (0.01, "m/s"),
    "pressure": (0.1, "Pa"),
    "temperature": (1.0, "K"),
    "vorticity": (1.0, "1/s"),
}
MAX_PROBES_PER_FIGURE = 8


def _coherence_batch_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute coherence/CSD for a contiguous batch of physical probe pairs."""
    matrix = np.load(payload["fft_values_path"], mmap_mode="r")
    window_name = payload["window"]
    detrend_value = payload["detrend"]
    segment = int(payload["segment"])
    overlap = int(payload["overlap"])
    rows = []
    phases = []
    frequency = None
    for pair_index, first, second in payload["pairs"]:
        pair_frequency, pair_coherence = coherence(
            matrix[:, int(first)],
            matrix[:, int(second)],
            fs=float(payload["fs"]),
            window=window_name,
            nperseg=segment,
            noverlap=overlap,
            detrend=detrend_value,
        )
        _, pair_cross = csd(
            matrix[:, int(first)],
            matrix[:, int(second)],
            fs=float(payload["fs"]),
            window=window_name,
            nperseg=segment,
            noverlap=overlap,
            detrend=detrend_value,
            scaling="density",
        )
        frequency = pair_frequency
        rows.append((int(pair_index), pair_coherence))
        phases.append((int(pair_index), np.angle(pair_cross)))
    return {
        "frequency": np.asarray(frequency) if frequency is not None else np.empty(0),
        "coherence": rows,
        "phase": phases,
    }


def _field(variable: str, fields: tuple[str, ...]) -> str:
    for candidate in FIELD_NAMES[variable]:
        if candidate in fields:
            return candidate
    raise KeyError(f"No storage field maps to {variable!r}")


def _si_conversion(variable: str, source_unit: str | None, solver_units: str) -> tuple[float, str]:
    """Return a conversion from the inspected source unit into public SI."""
    unit = (source_unit or "").strip().lower().replace(" ", "")
    si_units = {
        "pressure": {"pa", "pascal", "pascals"},
        "density": {"kg/m^3", "kg/m3", "kg·m^-3"},
        "x_velocity": {"m/s", "m·s^-1"},
        "y_velocity": {"m/s", "m·s^-1"},
        "temperature": {"k", "kelvin"},
        "vorticity": {"1/s", "s^-1", "s-1", "hz"},
    }
    public_units = {
        "pressure": "Pa",
        "density": "kg/m^3",
        "x_velocity": "m/s",
        "y_velocity": "m/s",
        "temperature": "K",
        "vorticity": "1/s",
    }
    if variable not in si_units:
        raise KeyError(f"No SI conversion is defined for {variable!r}")
    public_unit = public_units[variable]
    if unit in si_units[variable] or unit == "si":
        return 1.0, public_unit
    cgs = {
        "pressure": {"dyne/cm^2", "dyn/cm^2", "barye", "ba"},
        "density": {"g/cm^3"},
        "x_velocity": {"cm/s"},
        "y_velocity": {"cm/s"},
        "temperature": set(),
        "vorticity": set(),
    }
    if unit in cgs[variable] or unit == "cgs":
        return SI[variable][0], public_unit
    if unit in {"", "unknown"}:
        return (SI[variable][0] if solver_units == "cgs" else 1.0), public_unit
    raise ValueError(
        f"Unsupported source unit {source_unit!r} for variable {variable!r}; "
        f"expected SI or recognized CGS units"
    )


def load_probe_signal(
    context: WorkflowContext,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one named probe source and convert its inspected field to SI."""
    analysis = context.analysis
    configured_variable = getattr(analysis, "variable", None)
    if configured_variable is None:
        raise TypeError(f"Recipe {analysis.recipe!r} has no probe variable")
    return load_probe_variable(
        context,
        configured_variable.value,
        getattr(analysis, "probe_indices", ()),
        probe_set_id=analysis.probe_set_id,
        start_time_s=getattr(analysis, "record_start_time_s", None),
        end_time_s=getattr(analysis, "end_time_s", None),
    )


def load_probe_variable(
    context: WorkflowContext,
    variable: str,
    probe_indices: tuple[int, ...] = (),
    *,
    probe_set_id: str | None = None,
    probe_input: Any | None = None,
    probe_inventory: Any | None = None,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an explicitly selected probe variable through the SI boundary."""
    analysis = context.analysis
    if probe_input is None:
        source_id = probe_set_id or getattr(analysis, "probe_set_id", None)
        if source_id is None:
            raise ValueError("probe recipe requires probe_set_id")
        try:
            probe_input = context.project.machine_file.inputs.probe_sets[source_id]
        except KeyError as exc:
            raise ValueError(f"unknown probe_set_id {source_id!r}") from exc
    if probe_input is None:
        raise UnsupportedCapabilityError(
            f"The {analysis.recipe} executor requires a configured probe source"
        )
    if probe_inventory is None:
        source_id = probe_set_id or getattr(analysis, "probe_set_id", None)
        probe_inventory = context.plan.inventory.probe_sets.get(source_id or "")
    inventory = probe_inventory
    if inventory is None:
        raise UnsupportedCapabilityError("probe source was not inspected")
    storage_field = _field(variable, inventory.fields)
    selected = np.asarray(
        probe_indices if probe_indices else range(inventory.probe_count), dtype=int
    )
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= inventory.probe_count):
        raise ValueError("probe_indices contain no valid probes")
    factor, unit = _si_conversion(
        variable,
        inventory.field_units.get(storage_field),
        context.project.case_file.case.solver_units.value,
    )
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    memory_limit = float(context.project.machine_file.compute.memory_limit_gb)
    if probe_input.compact_file is not None:
        source = probe_input.compact_file.expanduser()
        if not source.is_absolute():
            source = (context.project.root / source).resolve()
        source_spec: Path | tuple[str, ...] = source
    else:
        patterns = []
        for pattern in probe_input.binary_files:
            configured = Path(pattern).expanduser()
            if any(character in pattern for character in "*?["):
                parent = configured.parent
                if not parent.is_absolute():
                    parent = (context.project.root / parent).resolve()
                patterns.append(str(parent / configured.name))
            else:
                if not configured.is_absolute():
                    configured = (context.project.root / configured).resolve()
                patterns.append(str(configured))
        source_spec = tuple(patterns)
    workspace = open_probe_signal_workspace(
        source_spec,
        storage_field,
        selected,
        si_factor=factor,
        memory_limit_gb=memory_limit,
        scratch_directory=scratch,
    )
    context.add_cleanup(workspace.close)
    time = workspace.time_s
    values = workspace.values
    if start_time_s is not None or end_time_s is not None:
        lower = -np.inf if start_time_s is None else float(start_time_s)
        upper = np.inf if end_time_s is None else float(end_time_s)
        mask = (time >= lower) & (time <= upper)
        selected_times = np.flatnonzero(mask)
        if len(selected_times) < 2:
            raise ValueError("time selection leaves fewer than two probe samples")
        if np.max(np.diff(selected_times)) > 1:
            raise ValueError("time selection must be a contiguous probe interval")
        time = time[selected_times]
        values = values[selected_times, :]
    assert context.resource_metadata is not None
    context.resource_metadata["probe_signal"] = {
        "storage": workspace.storage,
        "source_matrix_bytes": workspace.source_matrix_bytes,
        "resident_bound_bytes": workspace.resident_bound_bytes,
        "read_block_rows": workspace.read_block_rows,
    }
    return (
        variable,
        unit,
        time,
        workspace.x_m,
        values,
        workspace.selected_probe_indices,
    )


# Internal aliases retained for plotfile linkage during the staged migration.
load_compact_signal = load_probe_signal
load_compact_variable = load_probe_variable


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
    uniform = np.allclose(differences, dt, rtol=1.0e-9, atol=max(abs(dt) * 1.0e-9, 1.0e-15))
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
    uniform_values: np.ndarray | None = None
    cleanup: Callable[[], None] | None = None
    try:
        if scratch_directory is None:
            uniform_values = np.empty((sample_count, values.shape[1]), dtype=np.float64)
        else:
            scratch = Path(scratch_directory)
            scratch.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="pelec-post-fft-grid-", suffix=".mmap", dir=str(scratch)
            )
            os.close(descriptor)
            spill_path = Path(temporary_name)
            uniform_values = np.memmap(
                spill_path,
                mode="w+",
                dtype=np.float64,
                shape=(sample_count, values.shape[1]),
            )

            closed = False

            def cleanup() -> None:
                nonlocal closed, spill_path
                if closed:
                    return
                closed = True
                if isinstance(uniform_values, np.memmap):
                    uniform_values.flush()
                    mmap = getattr(uniform_values, "_mmap", None)
                    if mmap is not None:
                        mmap.close()
                if spill_path is not None:
                    try:
                        spill_path.unlink()
                    except FileNotFoundError:
                        pass
                    spill_path = None

        assert uniform_values is not None
        for column in range(values.shape[1]):
            uniform_values[:, column] = np.interp(
                uniform_time, time, np.asarray(values[:, column], dtype=float)
            )
    except BaseException:
        if cleanup is not None:
            cleanup()
        elif spill_path is not None:
            if isinstance(uniform_values, np.memmap):
                mmap = getattr(uniform_values, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            try:
                spill_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return uniform_time, uniform_values, dt, True, cleanup


def prepare_probe_time_grid(
    context: WorkflowContext,
    time: np.ndarray,
    values: np.ndarray,
    policy: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, Callable[[], None] | None]:
    """Apply the analysis-wide time-grid policy to a loaded probe signal."""
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    return _prepare_fft_grid(
        time,
        values,
        policy or getattr(context.analysis, "time_grid_policy", "resample_uniform"),
        scratch,
    )


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


def _finite_record_transform(
    values: np.ndarray, dt: float, weights: np.ndarray, time_origin_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite-record transform integral without amplitude scaling.

    This diagnostic is deliberately separate from the legacy single-sided
    amplitude.  Its magnitude has signal-units seconds and is useful for
    separating record-duration normalization from retained pulse content.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    frequency = np.fft.rfftfreq(len(values), d=dt)
    coefficient = np.fft.rfft(values * weights[:, None], axis=0) * float(dt)
    if time_origin_s:
        coefficient *= np.exp(-2j * np.pi * frequency[:, None] * float(time_origin_s))
    return frequency, coefficient


def _match_native_frequency_bins(
    reference_frequency_hz: np.ndarray,
    candidate_frequency_hz: np.ndarray,
) -> tuple[list[tuple[int, int, float]], str | None]:
    """Match native frequency bins one-to-one using the project tolerance."""
    reference_frequency_hz = np.asarray(reference_frequency_hz, dtype=float).ravel()
    candidate_frequency_hz = np.asarray(candidate_frequency_hz, dtype=float).ravel()
    used: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for reference_index, value in enumerate(reference_frequency_hz):
        tolerance = max(1.0e-6, 1.0e-9 * abs(float(value)))
        candidates = [
            index for index, candidate_value in enumerate(candidate_frequency_hz)
            if index not in used and abs(float(candidate_value) - float(value)) <= tolerance
        ]
        if len(candidates) > 1:
            return [], f"ambiguous native frequency match at reference index {reference_index}"
        if candidates:
            candidate_index = candidates[0]
            used.add(candidate_index)
            matched.append((reference_index, candidate_index, float(candidate_frequency_hz[candidate_index])))
    return matched, None


def probe_phase_and_symmetry_diagnostic(
    baseline_values: np.ndarray,
    comparison_values: np.ndarray,
    time_s: np.ndarray,
    probe_indices: np.ndarray,
    probe_x_m: np.ndarray,
    mirrored_groups: tuple[tuple[int, int], ...],
    *,
    variable: str,
    minimum_relative_amplitude: float = 0.01,
    reflection_center_m: float = 0.025,
    scalar_reflection_parity: str = "even",
    reference_frequency_band_hz: tuple[float, float] | None = None,
    values_are_processed: bool = False,
    time_origin_s: float | None = None,
    phase_delay_enabled: bool = True,
    symmetry_enabled: bool = True,
) -> dict[str, Any]:
    """Prepare phase, delay, and signed mirrored-residual diagnostics.

    The baseline and comparison arrays must use the same physical timestamps.
    Gaps in the amplitude support are retained as NaNs and are never unwrapped
    or fit across.
    """
    time_s = np.asarray(time_s, dtype=float).ravel()
    baseline_values = np.asarray(baseline_values, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)
    probe_indices = np.asarray(probe_indices, dtype=int).ravel()
    probe_x_m = np.asarray(probe_x_m, dtype=float).ravel()
    if scalar_reflection_parity not in {"even", "odd"}:
        raise ValueError("scalar_reflection_parity must be 'even' or 'odd'")
    if baseline_values.shape != comparison_values.shape or baseline_values.shape != (len(time_s), len(probe_indices)):
        raise ValueError("phase/symmetry inputs must share time and probe dimensions")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("phase/symmetry timestamps must be finite")
    frequency = np.array([], dtype=float)
    baseline_coeff = comparison_coeff = np.empty((0, len(probe_indices)), dtype=complex)
    if phase_delay_enabled:
        dt_values = np.diff(time_s)
        dt = float(np.median(dt_values))
        if len(time_s) < 2 or dt <= 0.0 or not np.allclose(dt_values, dt, rtol=1e-8, atol=max(dt * 1e-10, 1e-15)):
            raise ValueError("phase/delay diagnostics require a uniform time grid")
        frequency = np.fft.rfftfreq(len(time_s), dt)
        baseline_input = baseline_values if values_are_processed else baseline_values - np.mean(baseline_values, axis=0)
        comparison_input = comparison_values if values_are_processed else comparison_values - np.mean(comparison_values, axis=0)
        baseline_coeff = np.fft.rfft(baseline_input, axis=0) / len(time_s)
        comparison_coeff = np.fft.rfft(comparison_input, axis=0) / len(time_s)
        # One-sided amplitude convention: omit doubling of DC and Nyquist (even N).
        if len(time_s) % 2 == 0:
            baseline_coeff[1:-1] *= 2.0
            comparison_coeff[1:-1] *= 2.0
        else:
            baseline_coeff[1:] *= 2.0
            comparison_coeff[1:] *= 2.0
        if time_origin_s is not None:
            phase_origin = np.exp(-2j * np.pi * frequency * float(time_origin_s))[:, None]
            baseline_coeff *= phase_origin
            comparison_coeff *= phase_origin
    rows: list[dict[str, Any]] = []
    for column, probe_index in enumerate(probe_indices if phase_delay_enabled else ()):
        a, b = np.abs(baseline_coeff[:, column]), np.abs(comparison_coeff[:, column])
        band_peak_a, band_peak_b = float(np.max(a[1:], initial=0.0)), float(np.max(b[1:], initial=0.0))
        support = (frequency > 0.0) & (a >= band_peak_a * minimum_relative_amplitude) & (b >= band_peak_b * minimum_relative_amplitude)
        if reference_frequency_band_hz is not None:
            support &= (frequency >= reference_frequency_band_hz[0]) & (frequency <= reference_frequency_band_hz[1])
        phase = np.full(len(frequency), np.nan)
        phase[support] = np.angle(comparison_coeff[support, column] * np.conj(baseline_coeff[support, column]))
        # Unwrap only contiguous supported runs.
        supported_indices = np.flatnonzero(support)
        for run in np.split(supported_indices, np.flatnonzero(np.diff(supported_indices) > 1) + 1):
            if len(run):
                phase[run] = np.unwrap(phase[run])
        delay = None
        delay_r2 = None
        delay_fits: list[dict[str, Any]] = []
        finite = np.flatnonzero(np.isfinite(phase))
        for run in np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1):
            if len(run) < 5:
                continue
            slope, intercept = np.polyfit(frequency[run], phase[run], 1)
            fitted = slope * frequency[run] + intercept
            ss_res = float(np.sum((phase[run] - fitted) ** 2))
            ss_tot = float(np.sum((phase[run] - np.mean(phase[run])) ** 2))
            current_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            fit = {
                "frequency_min_hz": float(frequency[run[0]]),
                "frequency_max_hz": float(frequency[run[-1]]),
                "bin_count": len(run),
                "slope_rad_per_hz": float(slope),
                "delay_s": float(-slope / (2.0 * np.pi)),
                "residual_sum_squares": ss_res,
                "r_squared": current_r2,
            }
            delay_fits.append(fit)
            if current_r2 is not None and (delay_r2 is None or current_r2 > delay_r2):
                delay = fit["delay_s"]
                delay_r2 = float(current_r2)
        if np.allclose(comparison_coeff[:, column], baseline_coeff[:, column], rtol=1e-10, atol=1e-14):
            phase[:] = 0.0
            delay = 0.0
            delay_r2 = 1.0
        rows.append({
            "probe_index": int(probe_index),
            "x_m": float(probe_x_m[column]),
            "frequency_hz": frequency.tolist(),
            "phase_difference_rad": phase.tolist(),
            "support_fraction": float(np.mean(support[1:])) if len(support) > 1 else 0.0,
            "delay_s": delay if delay_r2 is not None and delay_r2 >= 0.95 else None,
            "delay_fit_r_squared": delay_r2,
            "delay_fits": delay_fits,
            "delay_interpretation": "positive means comparison/Gaussian arrives later" if delay is not None else "no supported single-delay fit",
        })
    symmetry: list[dict[str, Any]] = []
    index_to_column = {int(index): column for column, index in enumerate(probe_indices)}
    for left, right in mirrored_groups if symmetry_enabled else ():
        if left not in index_to_column or right not in index_to_column:
            symmetry.append({"probe_indices": [left, right], "status": "unavailable_missing_probe"})
            continue
        l, r = index_to_column[left], index_to_column[right]
        mirror_error = abs((probe_x_m[l] + probe_x_m[r]) / 2.0 - reflection_center_m)
        even_a = 0.5 * (baseline_values[:, l] + baseline_values[:, r])
        odd_a = 0.5 * (baseline_values[:, l] - baseline_values[:, r])
        even_b = 0.5 * (comparison_values[:, l] + comparison_values[:, r])
        odd_b = 0.5 * (comparison_values[:, l] - comparison_values[:, r])
        even_norm_a = float(np.linalg.norm(even_a))
        odd_norm_a = float(np.linalg.norm(odd_a))
        even_norm_b = float(np.linalg.norm(even_b))
        odd_norm_b = float(np.linalg.norm(odd_b))
        if scalar_reflection_parity == "even":
            expected_a, residual_a = even_norm_a, odd_norm_a
            expected_b, residual_b = even_norm_b, odd_norm_b
        else:
            expected_a, residual_a = odd_norm_a, even_norm_a
            expected_b, residual_b = odd_norm_b, even_norm_b
        symmetry.append({
            "probe_indices": [left, right],
            "reflection_center_m": reflection_center_m,
            "mirror_coordinate_error_m": float(mirror_error),
            "baseline_even_l2": even_norm_a,
            "baseline_odd_l2": odd_norm_a,
            "comparison_even_l2": even_norm_b,
            "comparison_odd_l2": odd_norm_b,
            "baseline_odd_fraction": odd_norm_a / even_norm_a if even_norm_a > 0 else None,
            "comparison_odd_fraction": odd_norm_b / even_norm_b if even_norm_b > 0 else None,
            "expected_parity": scalar_reflection_parity,
            "baseline_parity_residual_fraction": (
                residual_a / expected_a if expected_a > 0 else None
            ),
            "comparison_parity_residual_fraction": (
                residual_b / expected_b if expected_b > 0 else None
            ),
            "parity_residual_status": (
                "available" if expected_a > 0 and expected_b > 0
                else "unavailable_zero_expected_component"
            ),
            "difference_order": "left minus right",
        })
    return {
        "schema": "pelecpost.probe_phase_symmetry",
        "schema_version": 3,
        "variable": variable,
        "frequency_hz": frequency.tolist(),
        "minimum_relative_amplitude": minimum_relative_amplitude,
        "scalar_reflection_parity": scalar_reflection_parity,
        "phase_rows": rows,
        "symmetry": symmetry,
        "preprocessing": {"detrend": "already_processed" if values_are_processed else "mean", "window": "rectangular"},
        "reference_frequency_band_hz": list(reference_frequency_band_hz) if reference_frequency_band_hz is not None else None,
        "time_origin_s": float(time_origin_s) if time_origin_s is not None else 0.0,
    }


def _pressure_rise_diagnostic(time: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    """Report measured samples across the first baseline-to-peak rise."""
    result: list[dict[str, Any]] = []
    for column in range(values.shape[1]):
        signal = np.asarray(values[:, column], dtype=float)
        finite = np.isfinite(signal) & np.isfinite(time)
        if not np.any(finite):
            result.append({"probe_column": column, "status": "unavailable"})
            continue
        indices = np.flatnonzero(finite)
        start = int(indices[0])
        # ``indices`` may skip non-finite samples.  Index the original signal
        # with the selected finite index; adding the offset again shifts the
        # peak and all subsequent crossing times.
        peak = int(indices[int(np.argmax(signal[indices]))])
        baseline = float(signal[start])
        peak_value = float(signal[peak])
        rise = peak_value - baseline
        entry: dict[str, Any] = {
            "probe_column": column,
            "baseline_value": baseline,
            "peak_value": peak_value,
            "peak_time_s": float(time[peak]),
            "baseline_definition": "first finite recorded sample",
        }
        if rise <= 0.0:
            entry["status"] = "no_positive_rise"
            result.append(entry)
            continue
        crossings: dict[str, int] = {}
        for fraction in (0.1, 0.9):
            target = baseline + fraction * rise
            candidates = np.flatnonzero(signal[start:peak + 1] >= target)
            if candidates.size:
                crossings[str(int(fraction * 100))] = start + int(candidates[0])
        if len(crossings) != 2:
            entry["status"] = "crossing_unavailable"
            result.append(entry)
            continue
        first, last = crossings["10"], crossings["90"]
        interval_finite = bool(
            np.all(np.isfinite(signal[first:last + 1]))
            and np.all(np.isfinite(np.asarray(time[first:last + 1], dtype=float)))
        )
        if not interval_finite:
            entry.update({
                "status": "crossing_interval_missing_samples",
                "crossing_10_index": int(first),
                "crossing_90_index": int(last),
            })
            result.append(entry)
            continue
        interpolated: dict[str, float | None] = {}
        for fraction, key in ((0.1, "10"), (0.9, "90")):
            target = baseline + fraction * rise
            crossing_index = crossings[key]
            if crossing_index <= start or not np.isfinite(signal[crossing_index - 1]):
                interpolated[key] = None
                continue
            previous = signal[crossing_index - 1]
            current = signal[crossing_index]
            if current == previous:
                interpolated[key] = float(time[crossing_index])
            else:
                fraction_between = (target - previous) / (current - previous)
                interpolated[key] = float(
                    time[crossing_index - 1]
                    + fraction_between * (time[crossing_index] - time[crossing_index - 1])
                )
        entry.update({
            "status": "ok",
            "crossing_10_time_s": float(time[first]),
            "crossing_90_time_s": float(time[last]),
            "rise_time_10_to_90_s": float(time[last] - time[first]),
            "sample_intervals_10_to_90": int(last - first),
            "samples_inclusive_10_to_90": int(last - first + 1),
            "crossing_10_index": int(first),
            "crossing_90_index": int(last),
            "interpolated_crossing_time_s": {
                "10": interpolated["10"], "90": interpolated["90"]
            },
            "interpolated_crossing_is_estimate": True,
        })
        result.append(entry)
    return result


def _spectral_sampling_diagnostic(
    raw_time: np.ndarray,
    raw_values: np.ndarray,
    fft_time: np.ndarray,
    fft_values: np.ndarray,
    processed: np.ndarray,
    weights: np.ndarray,
    dt: float,
    analysis: ProbeSpectrumAnalysis,
) -> dict[str, Any]:
    raw_dt = np.diff(raw_time)
    median_dt = float(np.median(raw_dt)) if raw_dt.size else float(dt)
    deviations = np.flatnonzero(np.abs(raw_dt - median_dt) > max(abs(median_dt) * 0.01, 1e-18))
    frequency, baseline_amplitude = _single_sided_amplitude(processed, dt, weights)
    baseline_detrended, _ = _processed_probe_signal(fft_values, analysis.detrend, "rectangular")
    pulse_frequency, pulse_transform = _finite_record_transform(
        baseline_detrended, dt, weights,
        time_origin_s=float(fft_time[0]) if len(fft_time) else 0.0,
    )
    baseline_peak = np.maximum(np.max(baseline_amplitude, axis=0), 1e-300)
    endpoint_sensitivity: list[dict[str, Any]] = []
    for fraction in analysis.diagnostic_record_end_fractions:
        count = max(2, min(len(fft_values), round(len(fft_values) * fraction)))
        local = fft_values[:count]
        local_processed, local_weights = _processed_probe_signal(local, analysis.detrend, analysis.window)
        local_frequency, local_amplitude = _single_sided_amplitude(local_processed, dt, local_weights)
        local_detrended, _ = _processed_probe_signal(local, analysis.detrend, "rectangular")
        _, local_pulse_transform = _finite_record_transform(
            local_detrended, dt, local_weights, time_origin_s=float(fft_time[0])
        )
        endpoint_sensitivity.append({
            "record_end_fraction": float(fraction),
            "sample_count": count,
            "frequency_resolution_hz": float(1.0 / (count * dt)),
            "peak_amplitude_by_probe": np.max(local_amplitude, axis=0).tolist(),
            "relative_peak_to_baseline_by_probe": (
                np.max(local_amplitude, axis=0) / baseline_peak
            ).tolist(),
            "native_frequency_bin_count": len(local_frequency),
            "frequency_hz": local_frequency.tolist(),
            "amplitude": local_amplitude.tolist(),
            "pulse_transform_magnitude_by_probe": np.abs(local_pulse_transform).tolist(),
        })
    taper_sensitivity: list[dict[str, Any]] = []
    # End taper sensitivity changes only the final samples.  Apply the same
    # configured detrend as the baseline, with a rectangular preprocessing
    # weight before the diagnostic taper so the taper is the only perturbation.
    detrended, _ = _processed_probe_signal(fft_values, analysis.detrend, "rectangular")
    for fraction in analysis.diagnostic_end_taper_fractions:
        width = max(2, round(len(detrended) * fraction))
        taper = np.ones(len(detrended), dtype=float)
        taper[-width:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, width)))
        taper[-1] = 0.0
        local_frequency, local_amplitude = _single_sided_amplitude(detrended * taper[:, None], dt, taper)
        taper_sensitivity.append({
            "end_taper_fraction": float(fraction),
            "sample_count": len(detrended),
            "taper_sample_count": int(width),
            "coherent_gain": float(np.mean(taper)),
            "weights": taper.tolist(),
            "peak_amplitude_by_probe": np.max(local_amplitude, axis=0).tolist(),
            "relative_peak_to_baseline_by_probe": (
                np.max(local_amplitude, axis=0) / baseline_peak
            ).tolist(),
            "native_frequency_bin_count": len(local_frequency),
            "frequency_hz": local_frequency.tolist(),
            "amplitude": local_amplitude.tolist(),
        })
    timing_sensitivity: dict[str, Any]
    if raw_dt.size:
        timing_processed, timing_weights = _processed_probe_signal(
            raw_values, analysis.detrend, analysis.window
        )
        timing_frequency, timing_amplitude = _single_sided_amplitude(
            timing_processed, median_dt, timing_weights
        )
        matched, match_reason = _match_native_frequency_bins(frequency, timing_frequency)
        if match_reason is not None:
            timing_sensitivity = {
                "status": "unavailable",
                "label": "timing-perturbation diagnostic",
                "reason": match_reason,
            }
        else:
            reference_indices = np.asarray([item[0] for item in matched], dtype=int)
            candidate_indices = np.asarray([item[1] for item in matched], dtype=int)
            timing_difference = np.abs(
                timing_amplitude[candidate_indices] - baseline_amplitude[reference_indices]
            )
            timing_sensitivity = {
                "status": "available",
                "label": "timing-perturbation diagnostic",
                "nominal_timestamp_dt_s": median_dt,
                "matched_frequency_bin_count": len(matched),
                "matched_reference_indices": reference_indices.tolist(),
                "matched_candidate_indices": candidate_indices.tolist(),
                "matched_frequency_hz": [item[2] for item in matched],
                "median_absolute_difference_by_probe": np.median(timing_difference, axis=0).tolist(),
                "maximum_absolute_difference_by_probe": np.max(timing_difference, axis=0).tolist(),
                "frequency_hz": timing_frequency.tolist(),
                "baseline_amplitude": baseline_amplitude.tolist(),
                "timing_amplitude": timing_amplitude.tolist(),
                "interpretation": "Same values assigned nominal median-spacing timestamps; not a more accurate spectrum or an error estimate.",
            }
    else:
        timing_sensitivity = {"status": "unavailable", "reason": "fewer than two raw timestamps"}
    return {
        "schema_version": 2,
        "sample_count_raw": len(raw_time),
        "sample_count_fft": len(fft_time),
        "raw_dt_min_s": float(np.min(raw_dt)) if raw_dt.size else None,
        "raw_dt_median_s": median_dt,
        "raw_dt_max_s": float(np.max(raw_dt)) if raw_dt.size else None,
        "raw_irregular_interval_indices": deviations.tolist(),
        "resampled_dt_s": float(dt),
        "resampled": bool(not np.allclose(raw_dt, dt, rtol=0.0, atol=max(abs(dt) * 1e-12, 1e-18))) if raw_dt.size else False,
        "record_duration_s": float(fft_time[-1] - fft_time[0]),
        "fft_period_s": float(len(fft_time) * dt),
        "frequency_resolution_hz": float(frequency[1] - frequency[0]) if len(frequency) > 1 else None,
        "pulse_transform_frequency_hz": pulse_frequency.tolist(),
        "pulse_transform_magnitude": np.abs(pulse_transform).tolist(),
        "pulse_transform_units": "signal_units*s",
        "nominal_nyquist_hz": float(0.5 / dt),
        "initial_value": np.asarray(raw_values[0], dtype=float).tolist() if len(raw_values) else [],
        "final_value": np.asarray(raw_values[-1], dtype=float).tolist() if len(raw_values) else [],
        "endpoint_difference": (np.asarray(raw_values[-1] - raw_values[0], dtype=float).tolist()
                                if len(raw_values) else []),
        "pressure_rise": _pressure_rise_diagnostic(raw_time, raw_values)
        if str(getattr(analysis, "variable", "pressure")) == "pressure" else None,
        "endpoint_sensitivity": endpoint_sensitivity,
        "end_taper_sensitivity": taper_sensitivity,
        "timing_perturbation": timing_sensitivity,
        "interpretation": (
            "Display and window sensitivity diagnostics. They do not establish a converged "
            "physical bandwidth or an uncertainty estimate."
        ),
    }


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
    register_cleanup: Callable[[Callable[[], None]], None] | None = None,
) -> dict[str, Any]:
    """Compute the probe_spectrum pipeline products for an in-memory signal.

    Shared by the stationary spectrum executor and any future product
    producers that need the same preprocessing and time-grid behavior.
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
    cleanup_returned = fft_cleanup
    if fft_cleanup is not None and register_cleanup is not None:
        try:
            register_cleanup(fft_cleanup)
        except BaseException:
            fft_cleanup()
            raise
        cleanup_returned = None
    try:
        segment = welch_segment_samples or min(4096, len(fft_time))
        segment = min(segment, len(fft_time))
        overlap = round(segment * overlap_fraction)
        overlap = min(overlap, segment - 1)
        window_name = "boxcar" if window == "rectangular" else window
        scipy_detrend = {"mean": "constant", "linear": "linear", "none": False}[detrend]
        frequency, psd = welch(
            fft_values,
            fs=1.0 / dt,
            window=window_name,
            nperseg=segment,
            noverlap=overlap,
            detrend=scipy_detrend,
            axis=0,
            scaling="density",
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
            "weights": weights,
            "fft_frequency": fft_frequency[fft_keep],
            "fft_amplitude": fft_amplitude[fft_keep],
            "welch_frequency": frequency,
            "psd": psd,
            "dt": dt,
            "resampled": resampled,
            "segment": segment,
            "overlap": overlap,
            "welch_confidence": _welch_confidence_metadata(
                sample_count=len(fft_time),
                segment_samples=segment,
                overlap_samples=overlap,
                window=window_name,
                dt=dt,
            ),
            "cleanup": cleanup_returned,
        }
    except BaseException:
        if fft_cleanup is not None and register_cleanup is None:
            fft_cleanup()
        raise


def _welch_confidence_metadata(
    *,
    sample_count: int,
    segment_samples: int,
    overlap_samples: int,
    window: str,
    dt: float,
) -> dict[str, Any]:
    """Return overlap-corrected equivalent DOF for a Welch estimate.

    The correction uses the squared, normalized window autocorrelation at
    each segment displacement. It describes pointwise estimator uncertainty
    under the usual approximately stationary Gaussian-process assumptions.
    """
    hop = segment_samples - overlap_samples
    if segment_samples < 1 or hop < 1 or sample_count < segment_samples:
        raise ValueError("invalid Welch segment geometry")
    segment_count = 1 + (sample_count - segment_samples) // hop
    weights = np.asarray(get_window(window, segment_samples, fftbins=True), dtype=float)
    energy = float(np.dot(weights, weights))
    correlation_sum = 0.0
    if energy > 0.0:
        for offset in range(1, segment_count):
            shift = offset * hop
            if shift >= segment_samples:
                break
            rho = float(np.dot(weights[:-shift], weights[shift:]) / energy)
            correlation_sum += (1.0 - offset / segment_count) * rho * rho
    correction = 1.0 + 2.0 * correlation_sum
    effective_dof = 2.0 * segment_count / correction
    return {
        "schema_version": 3,
        "segment_samples": int(segment_samples),
        "overlap_samples": int(overlap_samples),
        "hop_samples": int(hop),
        "segment_count": int(segment_count),
        "approximate_degrees_of_freedom": int(2 * segment_count),
        "effective_degrees_of_freedom": float(effective_dof),
        "correlation_correction": float(correction),
        "window": window,
        "frequency_resolution_hz": float(1.0 / (segment_samples * dt)),
        "formula": "welch_window_overlap_autocorrelation",
        "assumptions": [
            "approximately stationary Gaussian process",
            "pointwise frequency-bin interval",
            "window-overlap correlation correction; detrending effects are not modeled",
        ],
        "interpretation": (
            "Equivalent degrees of freedom correct the nominal 2K value for "
            "correlation between overlapping windowed segments."
        ),
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
    dpi: int = 180,
) -> tuple[Path, ...]:
    """Write one row of raw history, processed history, and FFT per probe."""
    paths: list[Path] = []
    page_count = (len(selected) + MAX_PROBES_PER_FIGURE - 1) // MAX_PROBES_PER_FIGURE
    for page, start in enumerate(range(0, len(selected), MAX_PROBES_PER_FIGURE), start=1):
        stop = min(start + MAX_PROBES_PER_FIGURE, len(selected))
        rows = stop - start
        page_path = (
            path if page_count == 1 else path.with_name(f"{path.stem}_{page:03d}{path.suffix}")
        )
        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(15.0, max(3.5, 2.5 * rows)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"Probe histories and FFT · {detrend} detrend · {window} window")
        for row, (probe_index, coordinate) in enumerate(zip(selected[start:stop], x_m[start:stop])):
            raw_axis, processed_axis, fft_axis = axes[row]
            column = start + row
            label = f"Probe {probe_index} · x={coordinate * 100.0:.3f} cm"
            raw_axis.plot(raw_time * 1e6, raw_values[:, column], color="tab:blue", linewidth=1.2)
            rise = (_pressure_rise_diagnostic(raw_time, raw_values)[column]
                    if unit.lower() in {"pa", "pascal", "pascals"} else {})
            if rise.get("status") == "ok":
                margin = 5.0 * (float(np.median(np.diff(raw_time))) if len(raw_time) > 1 else 0.0)
                sample_mask = (
                    (raw_time >= rise["crossing_10_time_s"] - margin)
                    & (raw_time <= rise["crossing_90_time_s"] + margin)
                )
                raw_axis.plot(
                    raw_time[sample_mask] * 1e6,
                    raw_values[sample_mask, column],
                    linestyle="none",
                    marker="o",
                    markersize=2.2,
                    color="tab:red",
                    label="rise samples",
                )
            raw_axis.set_title(label, fontsize=10)
            if row == rows - 1:
                raw_axis.set_xlabel("Time [µs]")
            raw_axis.set_ylabel(f"{unit} history [{unit}]")
            raw_axis.grid(False)

            processed_axis.plot(
                time * 1e6, processed_values[:, column], color="tab:orange", linewidth=1.2
            )
            processed_axis.set_title("Processed", fontsize=10)
            if row == rows - 1:
                processed_axis.set_xlabel("Time [µs]")
            processed_axis.set_ylabel(f"Record-mean-subtracted {unit} [{unit}]")
            processed_axis.grid(False)

            keep = (frequency > 0.0) & (amplitude[:, column] > 0.0)
            if np.any(keep):
                fft_axis.loglog(
                    frequency[keep],
                    amplitude[keep, column],
                    color="tab:green",
                    linewidth=1.0,
                )
            else:
                fft_axis.text(
                    0.5,
                    0.5,
                    "No positive spectral amplitude",
                    transform=fft_axis.transAxes,
                    ha="center",
                    va="center",
                )
            fft_axis.set_title("FFT amplitude", fontsize=10)
            if row == rows - 1:
                fft_axis.set_xlabel("Frequency [Hz]")
            fft_axis.set_ylabel(f"Amplitude [{unit}]")
            fft_axis.grid(False)
        fig.savefig(page_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(page_path)
    return tuple(paths)


@executor("probe_spectrum")
def run_probe_spectrum(context: WorkflowContext) -> None:
    analysis = cast(ProbeSpectrumAnalysis, context.analysis)
    variable, unit, time, x_m, values, selected = load_probe_signal(context)
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    products = spectrum_from_signal(
        time,
        values,
        end_time_s=analysis.end_time_s,
        window=analysis.window,
        detrend=analysis.detrend,
        welch_segment_samples=analysis.welch_segment_samples,
        overlap_fraction=analysis.overlap_fraction,
        time_grid_policy=analysis.time_grid_policy,
        frequency_max_hz=analysis.frequency_max_hz,
        scratch_directory=scratch,
        register_cleanup=context.add_cleanup,
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
    welch_confidence = products["welch_confidence"]
    _ = processed_preview
    if analysis.diagnostics_enabled:
        sampling = _spectral_sampling_diagnostic(
            time,
            values,
            fft_time,
            products["fft_values"],
            processed_preview,
            products["weights"],
            dt,
            analysis,
        )
        diagnostics_path = context.data_dir / "spectral_sampling_diagnostic.json"
        diagnostics_path.write_text(json.dumps(sampling, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="spectral.sampling_diagnostic",
            path=diagnostics_path,
            kind="json",
            variable=variable,
            units=unit,
            interpretation=sampling["interpretation"],
            provenance={"window": analysis.window, "detrend": analysis.detrend},
        )
    register_probe_trace_figures(
        context,
        raw_time=time,
        raw_values=values,
        prepared_time=fft_time,
        prepared_values=processed_preview,
        x_m=x_m,
        selected=selected,
        variable=variable,
        units=unit,
        preprocessing={
            "time_grid_policy": analysis.time_grid_policy,
            "window": analysis.window,
            "detrend": analysis.detrend,
            "processed_definition": (
                "record-mean-subtracted" if analysis.detrend == "mean" else analysis.detrend
            ),
            "resampled": resampled,
        },
    )
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
        artifact_id="spectral.psd",
        path=data_path,
        kind="array",
        variable=variable,
        units=f"({unit})^2/Hz",
        coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Welch PSD of the selected transient record; values are descriptive "
            "squared-variable density, not a stationarity or mode claim."
        ),
        provenance={
            "si_boundary": (
                f"inspected probe field units ({context.project.case_file.case.solver_units.value} "
                "fallback) to public SI"
            ),
            "window": analysis.window,
            "detrend": analysis.detrend,
            "segment_samples": segment,
            "overlap_samples": overlap,
            "welch_confidence": welch_confidence,
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
            "signal_workspace": (context.resource_metadata or {})["probe_signal"],
        },
    )
    confidence = {
        **welch_confidence,
        "record_duration_s": float(fft_time[-1] - fft_time[0]),
        "approximate_degrees_of_freedom_status": "superseded_by_effective_degrees_of_freedom",
    }
    confidence_path = context.data_dir / "spectral_confidence.json"
    confidence_path.write_text(json.dumps(confidence, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="spectral.confidence",
        path=confidence_path,
        kind="json",
        variable=variable,
        units=None,
        interpretation=welch_confidence["interpretation"],
        provenance={
            "formula": welch_confidence["formula"],
            "assumptions": welch_confidence["assumptions"],
            "product_contract": product_contract("spectral.confidence", confidence_path),
        },
    )
    order = np.argsort(x_m)
    pair_columns = (
        np.column_stack((order[:-1], order[1:])) if len(order) > 1 else np.empty((0, 2), dtype=int)
    )
    window_name = "boxcar" if analysis.window == "rectangular" else analysis.window
    scipy_detrend = {"mean": "constant", "linear": "linear", "none": False}[analysis.detrend]
    coherence_frequency = np.fft.rfftfreq(segment, dt)
    if int(context.project.machine_file.compute.workers) <= 1 or len(pair_columns) < 2:
        coherence_values = []
        phase_values = []
        for first, second in pair_columns:
            pair_frequency, pair_coherence = coherence(
                products["fft_values"][:, first],
                products["fft_values"][:, second],
                fs=1.0 / dt,
                window=window_name,
                nperseg=segment,
                noverlap=overlap,
                detrend=scipy_detrend,
            )
            _, pair_cross = csd(
                products["fft_values"][:, first],
                products["fft_values"][:, second],
                fs=1.0 / dt,
                window=window_name,
                nperseg=segment,
                noverlap=overlap,
                detrend=scipy_detrend,
                scaling="density",
            )
            coherence_frequency = pair_frequency
            coherence_values.append(pair_coherence)
            phase_values.append(np.angle(pair_cross))
        coherence_matrix = (
            np.asarray(coherence_values).T
            if coherence_values
            else np.empty((len(coherence_frequency), 0))
        )
        phase_matrix = (
            np.asarray(phase_values).T if phase_values else np.empty((len(coherence_frequency), 0))
        )
    else:
        scratch_dir = context.project.machine_file.compute.scratch_directory
        if scratch_dir is None:
            scratch_dir = context.run_dir / "scratch" / context.analysis.id
        elif not scratch_dir.is_absolute():
            scratch_dir = (context.project.root / scratch_dir).resolve()
        matrix_spec, matrix_cleanup = stage_readonly_array(
            products["fft_values"],
            scratch_dir,
            prefix="coherence-fft-values-",
        )
        context.add_cleanup(matrix_cleanup)
        pair_count = len(pair_columns)
        worker_count = max(1, int(context.project.machine_file.compute.workers))
        batch_count = min(pair_count, max(1, worker_count * 4))
        pair_indices = np.array_split(np.arange(pair_count, dtype=int), batch_count)
        tasks = []
        for index, batch in enumerate(pair_indices):
            if not len(batch):
                continue
            payload_pairs = [
                (
                    int(pair_index),
                    int(pair_columns[pair_index, 0]),
                    int(pair_columns[pair_index, 1]),
                )
                for pair_index in batch
            ]
            tasks.append(
                ParallelTask(
                    index,
                    f"coherence-batch-{index:03d}",
                    {
                        "fft_values_path": matrix_spec.path,
                        "pairs": payload_pairs,
                        "fs": 1.0 / dt,
                        "window": window_name,
                        "detrend": scipy_detrend,
                        "segment": segment,
                        "overlap": overlap,
                    },
                    {"pair_indices": [int(item) for item in batch]},
                )
            )
        parent_gb = float((context.resource_metadata or {}).get("parent_resident_gb", 0.25))
        per_worker_gb = max(0.05, 0.15 + products["fft_values"].nbytes / 1024**3)
        stage_plan = plan_parallel_stage(
            "probe-coherence-batches",
            requested_workers=worker_count,
            task_count=len(tasks),
            memory_limit_gb=float(context.project.machine_file.compute.memory_limit_gb),
            parent_resident_gb=parent_gb,
            per_worker_peak_gb=per_worker_gb,
        )
        batch_results = context.run_parallel_stage(
            "probe-coherence-batches",
            tasks,
            _coherence_batch_worker,
            stage_plan,
        )
        coherence_values_by_pair: dict[int, np.ndarray] = {}
        phase_values_by_pair: dict[int, np.ndarray] = {}
        for result in batch_results:
            payload = result.value
            if len(payload["frequency"]):
                coherence_frequency = payload["frequency"]
            coherence_values_by_pair.update(payload["coherence"])
            phase_values_by_pair.update(payload["phase"])
        coherence_matrix = (
            np.column_stack([coherence_values_by_pair[index] for index in range(pair_count)])
            if pair_count
            else np.empty((len(coherence_frequency), 0))
        )
        phase_matrix = (
            np.column_stack([phase_values_by_pair[index] for index in range(pair_count)])
            if pair_count
            else np.empty((len(coherence_frequency), 0))
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
        upstream_probe_indices=selected[pair_columns[:, 0]]
        if len(pair_columns)
        else np.array([], dtype=int),
        downstream_probe_indices=selected[pair_columns[:, 1]]
        if len(pair_columns)
        else np.array([], dtype=int),
        upstream_x_m=x_m[pair_columns[:, 0]] if len(pair_columns) else np.array([]),
        downstream_x_m=x_m[pair_columns[:, 1]] if len(pair_columns) else np.array([]),
        convention=np.array("conj(upstream) * downstream"),
    )
    context.register(
        artifact_id="spectral.coherence",
        path=coherence_path,
        kind="array",
        variable=variable,
        units="dimensionless",
        coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Magnitude-squared coherence and downstream-minus-upstream cross phase for "
            "adjacent probes ordered by physical x coordinate."
        ),
        provenance={
            "segment_samples": segment,
            "overlap_samples": overlap,
            "pair_count": len(pair_columns),
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
        },
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
        artifact_id="spectral.probe_signals",
        path=signal_path,
        kind="array",
        variable=variable,
        units=unit,
        coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Selected probe histories and configured-detrend/window single-sided FFT amplitudes."
        ),
        provenance={
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
            "window": analysis.window,
            "detrend": analysis.detrend,
        },
    )
    positive_fft = fft_frequency > 0.0
    register_probe_line_overlay(
        context,
        artifact_id="spectral.fft_overlay.figure",
        filename="fft_overlay",
        x=fft_frequency[positive_fft],
        values=fft_amplitude[positive_fft],
        x_label="Frequency [Hz]",
        y_label=f"Amplitude [{unit}]",
        variable=variable,
        units=unit,
        selected=selected,
        x_m=x_m,
        interpretation="Single-sided FFT amplitude overlaid for every selected probe.",
        provenance={
            "time_grid_policy": analysis.time_grid_policy,
            "window": analysis.window,
            "detrend": analysis.detrend,
        },
        log_x=True,
        log_y=True,
    )
    positive_psd = frequency > 0.0
    register_probe_line_overlay(
        context,
        artifact_id="spectral.psd_overlay.figure",
        filename="psd_overlay",
        x=frequency[positive_psd],
        values=psd[positive_psd],
        x_label="Frequency [Hz]",
        y_label=f"PSD [({unit})²/Hz]",
        variable=variable,
        units=f"({unit})^2/Hz",
        selected=selected,
        x_m=x_m,
        interpretation="One-sided Welch PSD overlaid for every selected probe.",
        provenance={
            "time_grid_policy": analysis.time_grid_policy,
            "window": analysis.window,
            "detrend": analysis.detrend,
        },
        log_x=True,
        log_y=True,
    )
    probe_figure_path = context.figure_dir / "probe_time_fft.png"
    if analysis.probe_plotting.mode in {"panels", "both"}:
        probe_figure_paths = _plot_probe_time_fft(
            probe_figure_path,
            time,
            values,
            fft_time,
            processed_values,
            fft_frequency,
            fft_amplitude,
            x_m,
            selected,
            unit,
            analysis.detrend,
            analysis.window,
            dpi=int(context.project.analyses_file.presentation.figure.dpi),
        )
        for page, path in enumerate(probe_figure_paths, start=1):
            context.register(
                artifact_id=(
                    "spectral.probe_figure"
                    if len(probe_figure_paths) == 1
                    else f"spectral.probe_figure.{page:03d}"
                ),
                path=path,
                kind="figure",
                variable=variable,
                units=unit,
                coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
                interpretation=(
                    "Per-probe raw history, processed history, and single-sided FFT amplitude."
                ),
                provenance={
                    "time_grid_policy": analysis.time_grid_policy,
                    "resampled": resampled,
                    "window": analysis.window,
                    "detrend": analysis.detrend,
                    "page": page,
                    "page_count": len(probe_figure_paths),
                },
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
    axis.fill_between(
        frequency[positive], lower[positive], upper[positive], alpha=0.25, label="10–90% probe spread"
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"PSD [({unit})²/Hz]")
    axis.set_title(f"{variable.capitalize()}: Welch PSD of selected transient record")
    axis.grid(False)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        figure_path,
        dpi=int(context.project.analyses_file.presentation.figure.dpi),
        bbox_inches="tight",
    )
    if figure_path.suffix.lower() == ".png":
        fig.savefig(
            figure_path.with_suffix(".pdf"),
            dpi=int(context.project.analyses_file.presentation.figure.dpi),
            bbox_inches="tight",
        )
    plt.close(fig)
    context.register(
        artifact_id="spectral.figure",
        path=figure_path,
        kind="figure",
        variable=variable,
        units=f"({unit})^2/Hz",
        coordinate_metadata={"frequency": "Hz"},
        interpretation=(
            "Median and 10–90% spread across selected probes for the Welch PSD of "
            "the selected transient record; the spread is not a confidence interval."
        ),
    )
    pdf_path = figure_path.with_suffix(".pdf")
    if pdf_path.is_file():
        context.register(
            artifact_id="spectral.figure.pdf",
            path=pdf_path,
            kind="figure",
            variable=variable,
            units=f"({unit})^2/Hz",
            coordinate_metadata={"frequency": "Hz"},
            interpretation="PDF companion export of the probe-ensemble Welch PSD figure.",
            provenance={"figure_format": "pdf"},
        )


@executor("single_pulse_response")
def run_single_pulse_response(context: WorkflowContext) -> None:
    analysis = cast(SinglePulseAnalysis, context.analysis)
    variable, unit, raw_time, x_m, raw_values, selected = load_probe_signal(context)
    time = raw_time
    values = raw_values
    time, values, _dt, resampled, cleanup = prepare_probe_time_grid(context, time, values)
    if cleanup is not None:
        context.add_cleanup(cleanup)
    baseline_end = analysis.baseline_end_time_s
    if baseline_end is None:
        baseline_end = analysis.start_time_s
    baseline = reviewed_spectral.subtract_quiescent_probe_baseline(
        time, values, baseline_end, minimum_samples=analysis.minimum_baseline_samples
    )
    response = baseline["disturbance"]
    response -= np.mean(response, axis=0, keepdims=True)
    register_probe_trace_figures(
        context,
        raw_time=raw_time,
        raw_values=raw_values,
        prepared_time=time,
        prepared_values=response,
        x_m=x_m,
        selected=selected,
        variable=variable,
        units=unit,
        preprocessing={
            "time_grid_policy": analysis.time_grid_policy,
            "baseline_end_time_s": baseline_end,
            "mean_subtraction": True,
            "resampled": resampled,
        },
    )
    response_complex = np.fft.rfft(response, axis=0) / len(time)
    modeled_source = reviewed_spectral.compute_single_pulse_source_spectrum(
        time,
        energy_per_pulse=analysis.energy_per_pulse_j_m,
        pulse_fwhm_s=analysis.pulse_fwhm_s,
        pulse_period_s=analysis.pulse_period_s,
        start_time_s=analysis.start_time_s,
        cutoff_sigma=analysis.cutoff_sigma,
        mean_subtraction="mean",
        window="none",
    )
    source_basis = "modeled"
    source = modeled_source
    source_audit: dict[str, Any] = {
        "schema": "pelecpost.pulse-source-audit",
        "schema_version": 1,
        "status": "modeled_only",
        "source_basis": "modeled",
        "complete": False,
        "configured_requested_energy_j_m": analysis.energy_per_pulse_j_m,
        "analytic_retained_energy_j_m": analysis.energy_per_pulse_j_m
        * float(math.erf(analysis.cutoff_sigma / np.sqrt(2.0))),
        "measured_deposited_energy_j_m": None,
        "signed_relative_energy_error": None,
        "absolute_relative_energy_error": None,
        "capture_ratio": None,
        "one_percent_gate": False,
        "configured_maximum_relative_error": analysis.maximum_source_energy_relative_error,
        "resolved_thresholds": {
            "maximum_source_energy_relative_error": analysis.maximum_source_energy_relative_error,
        },
        "source_history": None,
        "measured_peak_time_s": None,
        "measured_energy_weighted_time_centroid_s": None,
        "requested_pulse_center_time_s": analysis.start_time_s + 0.5 * analysis.pulse_period_s,
        "measured_peak_time_offset_s": None,
        "measured_centroid_time_offset_s": None,
        "spatial_centroid_m": {"x": None, "y": None},
        "rms_widths_m": {"x": None, "y": None},
        "interpretation": "No measured thermal-source history was configured; this is a modeled-source-only result.",
    }
    history_id = analysis.source_history_id
    if history_id is not None:
        try:
            history_config = context.project.machine_file.inputs.source_histories[history_id]
        except KeyError as exc:
            raise ValueError(f"unknown source_history_id {history_id!r}") from exc
        history_source = history_config.source.expanduser()
        if not history_source.is_absolute():
            history_source = (context.project.root / history_source).resolve()
        history = canonicalize_segments(
            discover_source_segments(history_source, history_config.prefix)
        )
        validate_history_configuration(history, analysis)
        if count_intersecting_pulses(history, float(time[0]), float(time[-1])) > 1:
            raise ValueError(
                "measured single-pulse processing found more than one pulse in the selected record; "
                "pulse-train association remains out of scope"
            )
        rebinned = rebin_history(history, time)
        source_audit = build_source_audit(
            history,
            rebinned,
            requested_energy_j_m=analysis.energy_per_pulse_j_m,
            pulse_fwhm_s=analysis.pulse_fwhm_s,
            cutoff_sigma=analysis.cutoff_sigma,
            maximum_relative_error=analysis.maximum_source_energy_relative_error,
            start_time_s=analysis.start_time_s,
        )
        source_audit["source_history_id"] = history_id
        source_basis = "measured"
        source = measured_source_spectrum(rebinned, modeled=modeled_source)
    source_path = context.data_dir / "single_pulse_source_spectrum.npz"
    transfer = reviewed_spectral.compute_single_pulse_transfer_function(
        source["processed_complex"],
        response_complex,
        minimum_relative_source_amplitude=analysis.minimum_relative_source_amplitude,
    )
    frequency = source["frequency_hz"]
    keep = np.ones_like(frequency, dtype=bool)
    if analysis.frequency_max_hz is not None:
        keep &= frequency <= analysis.frequency_max_hz
    np.savez_compressed(
        source_path,
        time_s=source["time_s"],
        source_power_w_m=source["power"],
        modeled_source_power_w_m=modeled_source["power"],
        frequency_hz=frequency[keep],
        physical_complex_j_m=source["physical_complex"][keep],
        physical_spectrum_j_m=source["physical_spectrum"][keep],
        ideal_spectrum_j_m=source["ideal_spectrum"][keep],
        processed_complex_w_m=source["processed_complex"][keep],
        sigma_s=np.array(source["sigma_s"]),
        center_s=np.array(source["center_s"]),
        source_basis=np.array(source_basis),
    )
    context.register(
        artifact_id="pulse.source_spectrum",
        path=source_path,
        kind="array",
        variable="source_power",
        units="J/m",
        coordinate_metadata={"time": "s", "frequency": "Hz"},
        interpretation=(
            "Code-matched finite Gaussian pulse history, physical continuous-time spectrum, "
            "and processed spectrum used for deconvolution."
        ),
        provenance={
            "energy_per_pulse_j_m": analysis.energy_per_pulse_j_m,
            "pulse_fwhm_s": analysis.pulse_fwhm_s,
            "pulse_period_s": analysis.pulse_period_s,
            "source_basis": source_basis,
            "source_history_id": history_id,
        },
    )
    path = context.data_dir / "single_pulse_response.npz"
    np.savez_compressed(
        path,
        frequency_hz=frequency[keep],
        source_power_spectrum_w_m=np.abs(source["processed_complex"][keep]),
        transfer=transfer["transfer"][keep],
        transfer_magnitude=transfer["magnitude"][keep],
        transfer_phase_rad=transfer["phase_rad"][keep],
        valid_frequency=transfer["valid_frequency"][keep],
        baseline=baseline["baseline"],
        baseline_sample_count=np.array(baseline["baseline_sample_count"]),
        probe_x_m=x_m,
        response_unit=np.array(unit),
    )
    context.register(
        artifact_id="pulse.transfer",
        path=path,
        kind="array",
        variable=variable,
        units=f"{unit}/(W/m)",
        coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
        interpretation=(
            "Finite-record single-pulse response/source-power ratio; invalid low-source "
            "bins are masked. The separately registered physical source transform retains J/m units."
        ),
        provenance={
            "estimator": "finite_record_single_pulse",
            "baseline_end_time_s": baseline_end,
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
            "source_basis": source_basis,
            "source_history_id": history_id,
        },
    )
    audit_path = context.data_dir / "pulse_source_audit.json"
    audit_path.write_text(json.dumps(source_audit, indent=2, default=str) + "\n", encoding="utf-8")
    context.register(
        artifact_id="pulse.source_audit",
        path=audit_path,
        kind="json",
        variable="source_power",
        units="J/m",
        coordinate_metadata={"time": "s", "space": "m"},
        interpretation=(
            "Measured discrete thermal-source deposition compared with the configured analytical target; "
            "modeled-only records are explicitly downgraded."
        ),
        provenance={"source_basis": source_basis, "source_history_id": history_id},
    )
    audit_figure_path = context.figure_dir / "source_audit.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, source["power"], label=f"{source_basis} source power")
    axes[0].plot(time, modeled_source["power"], "--", label="modeled source power")
    axes[0].set_ylabel("Power [W/m]")
    axes[0].legend()
    axes[0].grid(False)
    measured_cumulative = np.cumsum(source["power"]) * float(np.median(np.diff(time)))
    modeled_cumulative = np.cumsum(modeled_source["power"]) * float(np.median(np.diff(time)))
    axes[1].plot(time, measured_cumulative, label="used-source cumulative energy")
    axes[1].plot(time, modeled_cumulative, "--", label="modeled cumulative energy")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Cumulative energy [J/m]")
    axes[1].legend()
    axes[1].grid(False)
    fig.suptitle(
        f"Source audit: {source_audit.get('status')} | "
        f"deposited={source_audit.get('measured_deposited_energy_j_m')} J/m"
    )
    fig.tight_layout()
    fig.savefig(audit_figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="pulse.source_audit.figure",
        path=audit_figure_path,
        kind="figure",
        variable="source_power",
        units="W/m",
        coordinate_metadata={"time": "s"},
        interpretation="Measured and modeled thermal-source histories and cumulative energy.",
        provenance={"source_basis": source_basis, "source_history_id": history_id},
    )
    register_probe_line_overlay(
        context,
        artifact_id="pulse.transfer_magnitude.figure",
        filename="transfer_magnitude_overlay",
        x=frequency[keep],
        values=np.abs(transfer["magnitude"][keep]),
        x_label="Frequency [Hz]",
        y_label=f"Transfer magnitude [{unit}/(W/m)]",
        variable=variable,
        units=f"{unit}/(W/m)",
        selected=selected,
        x_m=x_m,
        interpretation="Finite-record transfer magnitude overlaid for every selected probe.",
        provenance={"valid_frequency_mask": "invalid source bins omitted"},
        log_x=True,
        log_y=True,
    )
    register_probe_line_overlay(
        context,
        artifact_id="pulse.transfer_phase.figure",
        filename="transfer_phase_overlay",
        x=frequency[keep],
        values=transfer["phase_rad"][keep],
        x_label="Frequency [Hz]",
        y_label="Transfer phase [rad]",
        variable=variable,
        units="rad",
        selected=selected,
        x_m=x_m,
        interpretation="Finite-record transfer phase overlaid for every selected probe.",
        provenance={"valid_frequency_mask": "invalid source bins omitted"},
        log_x=True,
    )
    quality = {
        "baseline_sample_count": baseline["baseline_sample_count"],
        "valid_frequency_fraction": float(np.mean(transfer["valid_frequency"])),
        "minimum_relative_source_amplitude": analysis.minimum_relative_source_amplitude,
    }
    quality_path = context.data_dir / "pulse_validity.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="pulse.validity",
        path=quality_path,
        kind="json",
        variable=variable,
        units=None,
        interpretation="Baseline and source-amplitude quality gates for pulse deconvolution.",
    )


def _komega_ridge(
    spectrum: dict, minimum_hz: float, maximum_hz: float
) -> dict[str, Any]:
    frequency = np.asarray(spectrum["frequency_hz"], dtype=float)
    wavenumber = np.asarray(spectrum["wavenumber_rad_per_m"], dtype=float)
    power = np.asarray(spectrum["power"], dtype=float)
    selected = (frequency >= minimum_hz) & (frequency <= maximum_hz)
    if not np.any(selected):
        raise ValueError("k-omega sensitivity band contains no temporal FFT bins")
    band_power = power[selected]
    selected_frequency = frequency[selected]
    global_max = float(np.nanmax(band_power)) if np.any(np.isfinite(band_power)) else 0.0
    result: dict[str, Any] = {
        "frequency_hz": selected_frequency,
        "negative_k": np.full(len(selected_frequency), np.nan),
        "positive_k": np.full(len(selected_frequency), np.nan),
        "zero_k_power": np.full(len(selected_frequency), np.nan),
        "negative_supported": np.zeros(len(selected_frequency), dtype=bool),
        "positive_supported": np.zeros(len(selected_frequency), dtype=bool),
        "negative_edge_limited": np.zeros(len(selected_frequency), dtype=bool),
        "positive_edge_limited": np.zeros(len(selected_frequency), dtype=bool),
        "negative_ambiguous": np.zeros(len(selected_frequency), dtype=bool),
        "positive_ambiguous": np.zeros(len(selected_frequency), dtype=bool),
        "support_fraction": 1.0e-4,
        "ambiguity_fraction": 0.5,
    }
    for row, row_power in enumerate(band_power):
        for name, mask, edge_index in (
            ("negative", wavenumber < 0.0, 0),
            ("positive", wavenumber > 0.0, len(wavenumber) - 1),
        ):
            branch_indices = np.flatnonzero(mask)
            if not len(branch_indices):
                continue
            branch_values = np.asarray(row_power[branch_indices], dtype=float)
            peak_position = int(np.nanargmax(branch_values))
            peak_value = float(branch_values[peak_position])
            if not np.isfinite(peak_value) or peak_value < global_max * 1.0e-4:
                continue
            # Distinct local peaks require separation by at least two native
            # bins; a broad plateau counts as one peak.
            peak_indices, _ = find_peaks(
                branch_values, height=peak_value * 0.5, plateau_size=1, distance=2
            )
            if peak_position == 0 or peak_position == len(branch_values) - 1:
                peak_indices = np.unique(np.r_[peak_indices, peak_position])
            ambiguous = bool(len(peak_indices) > 1)
            index = int(branch_indices[peak_position])
            result[name + "_k"][row] = float(wavenumber[index])
            result[name + "_supported"][row] = True
            result[name + "_edge_limited"][row] = index == edge_index
            result[name + "_ambiguous"][row] = ambiguous
        zero_indices = np.flatnonzero(wavenumber == 0.0)
        if len(zero_indices):
            result["zero_k_power"][row] = float(row_power[zero_indices[0]])
    return result


def _ridge_difference(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict:
    frequency = np.asarray(reference["frequency_hz"])
    candidate_frequency = np.asarray(candidate["frequency_hz"])
    result: dict[str, Any] = {"schema_version": 3, "branches": {}}
    matched, match_reason = _match_native_frequency_bins(frequency, candidate_frequency)
    if match_reason is not None:
        return {
            "schema_version": 3, "branches": {},
            "compared_frequency_count": 0,
            "reason": match_reason,
        }
    if not matched:
        return {"schema_version": 3, "branches": {}, "compared_frequency_count": 0,
                "reason": "no matching native frequency bins"}
    for branch in ("negative", "positive"):
        shifts: list[float] = []
        excluded: dict[str, int] = {}
        for index, candidate_index, _actual_frequency in matched:
            if not (reference[f"{branch}_supported"][index] and candidate[f"{branch}_supported"][candidate_index]):
                excluded["unsupported"] = excluded.get("unsupported", 0) + 1
                continue
            if reference[f"{branch}_edge_limited"][index] or candidate[f"{branch}_edge_limited"][candidate_index]:
                excluded["edge_limited"] = excluded.get("edge_limited", 0) + 1
                continue
            if reference[f"{branch}_ambiguous"][index] or candidate[f"{branch}_ambiguous"][candidate_index]:
                excluded["ambiguous"] = excluded.get("ambiguous", 0) + 1
                continue
            shifts.append(float(candidate[f"{branch}_k"][candidate_index] - reference[f"{branch}_k"][index]))
        result["branches"][branch] = {
            "compared_frequency_count": len(shifts),
            "excluded_by_reason": excluded,
            "median_shift_rad_m": float(np.median(shifts)) if shifts else None,
            "median_absolute_shift_rad_m": float(np.median(np.abs(shifts))) if shifts else None,
            "percentile_90_absolute_shift_rad_m": float(np.percentile(np.abs(shifts), 90.0)) if shifts else None,
            "shifts_rad_m": shifts,
        }
    result["compared_frequency_count"] = len(matched)
    result["matched_frequency_hz"] = [item[2] for item in matched]
    return result


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
        values,
        time,
        x_m,
        temporal_window=alternate_window,
        spatial_window=analysis.spatial_window,
        temporal_mean_subtraction="mean",
    )
    half = len(time) // 2
    if half < 8:
        raise ValueError("k-omega block sensitivity requires at least 16 temporal samples")
    first = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values[:half],
        time[:half],
        x_m,
        temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window,
        temporal_mean_subtraction="mean",
    )
    second = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values[-half:],
        time[-half:],
        x_m,
        temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window,
        temporal_mean_subtraction="mean",
    )
    return {
        "schema_version": 2,
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


def _spatial_aperture_sensitivity(
    values: np.ndarray,
    time: np.ndarray,
    x_m: np.ndarray,
    analysis: DirectionalWaveAnalysis,
) -> dict[str, Any]:
    """Recompute native f-k maps for centered apertures and spatial windows."""
    values = np.asarray(values, dtype=float)
    x_m = np.asarray(x_m, dtype=float)
    full_count = len(x_m)
    results: list[dict[str, Any]] = []
    for fraction in analysis.spatial_aperture_fractions:
        count = max(5, round(full_count * fraction))
        if count % 2 == 0:
            count -= 1
        count = min(count, full_count if full_count % 2 else full_count - 1)
        start = (full_count - count) // 2
        indices = np.arange(start, start + count, dtype=int)
        for spatial_window in ("rectangular", "hann"):
            spectrum = reviewed_spectral.compute_wavenumber_frequency_spectrum(
                values[:, indices], time, x_m[indices],
                temporal_window=analysis.temporal_window,
                spatial_window=spatial_window,
                temporal_mean_subtraction="mean",
            )
            ridge = _komega_ridge(spectrum, analysis.frequency_min_hz, analysis.frequency_max_hz)
            results.append({
                "aperture_fraction": float(fraction),
                "spatial_window": spatial_window,
                "probe_count": int(count),
                "probe_indices": indices.tolist(),
                "x_bounds_m": [float(x_m[indices[0]]), float(x_m[indices[-1]])],
                "delta_k_rad_per_m": float(spectrum["native_wavenumber_resolution_rad_per_m"]),
                "negative_supported_count": int(np.count_nonzero(ridge["negative_supported"])),
                "positive_supported_count": int(np.count_nonzero(ridge["positive_supported"])),
                "negative_edge_limited_count": int(np.count_nonzero(ridge["negative_edge_limited"])),
                "positive_edge_limited_count": int(np.count_nonzero(ridge["positive_edge_limited"])),
            })
    return {
        "schema": "pelecpost.spatial_aperture_sensitivity",
        "schema_version": 1,
        "frequency_band_hz": [analysis.frequency_min_hz, analysis.frequency_max_hz],
        "results": results,
        "interpretation": "Native-grid aperture and spatial-window sensitivity; no zero-padding or ridge interpolation is used.",
    }


@executor("directional_wave")
def run_directional_wave(context: WorkflowContext) -> None:
    analysis = cast(DirectionalWaveAnalysis, context.analysis)
    variable, unit, raw_time, x_m, raw_values, selected = load_probe_signal(context)
    time = raw_time
    values = raw_values
    time, values, _dt, resampled, cleanup = prepare_probe_time_grid(context, time, values)
    if cleanup is not None:
        context.add_cleanup(cleanup)
    register_probe_trace_figures(
        context,
        raw_time=raw_time,
        raw_values=raw_values,
        prepared_time=time,
        prepared_values=values,
        x_m=x_m,
        selected=selected,
        variable=variable,
        units=unit,
        preprocessing={
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
            "processed_definition": "raw probe history; temporal mean subtraction is applied inside the k-omega transform",
        },
    )
    speed_bounds = None
    if analysis.expected_speed_min_m_s is not None and analysis.expected_speed_max_m_s is not None:
        speed_bounds = (analysis.expected_speed_min_m_s, analysis.expected_speed_max_m_s)
    segment = min(16384, max(8, len(time) // 2))
    local = reviewed_spectral.compute_frequency_resolved_wavenumber(
        values,
        x_m,
        1.0 / np.median(np.diff(time)),
        (analysis.frequency_min_hz, analysis.frequency_max_hz),
        nperseg=segment,
        noverlap=segment // 2,
        spatial_window_size=min(101, len(x_m) if len(x_m) % 2 else len(x_m) - 1),
        spatial_step=max(1, len(x_m) // 20),
        fft_batch_size=context.project.machine_file.compute.fft_batch_size,
        min_coherence=analysis.minimum_coherence,
        phase_speed_bounds=speed_bounds,
        direction=analysis.direction,
    )
    komega = reviewed_spectral.compute_wavenumber_frequency_spectrum(
        values,
        time,
        x_m,
        temporal_window=analysis.temporal_window,
        spatial_window=analysis.spatial_window,
        temporal_mean_subtraction="mean",
    )
    if analysis.spatial_sensitivity_enabled:
        aperture_payload = _spatial_aperture_sensitivity(values, time, x_m, analysis)
        aperture_path = context.data_dir / "spatial_aperture_sensitivity.json"
        aperture_path.write_text(json.dumps(aperture_payload, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="wave.spatial_aperture_sensitivity",
            path=aperture_path,
            kind="json",
            variable=variable,
            units="rad/m",
            coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
            interpretation=aperture_payload["interpretation"],
            provenance={"aperture_fractions": list(analysis.spatial_aperture_fractions)},
        )
    path = context.data_dir / "complex_wavenumber.npz"
    arrays = {
        "frequency_hz": local["frequency_hz"],
        "x_center_m": local["x_center_m"],
        "alpha_real_rad_m": local["alpha_real_rad_per_m"],
        "alpha_imag_rad_m": local["alpha_imag_rad_per_m"],
        "alpha_real_ci95_rad_m": local["alpha_real_ci95_rad_per_m"],
        "alpha_imag_ci95_rad_m": local["alpha_imag_ci95_rad_per_m"],
        "amplification_rate_per_m": local["amplification_rate_per_m"],
        "phase_speed_m_s": local["phase_speed_m_per_s"],
        "phase_valid_mask": local["phase_valid_mask"],
        "growth_valid_mask": local["growth_valid_mask"],
        "coherence_squared": local["mean_coherence_squared"],
        "phase_fit_r_squared": local["phase_fit_r_squared"],
        "amplitude_fit_r_squared": local["amplitude_fit_r_squared"],
        "spatial_alias_margin": local["spatial_alias_margin"],
    }
    np.savez_compressed(path, **arrays)
    accepted = float(np.mean(local["phase_valid_mask"]))
    growth = float(np.mean(local["growth_valid_mask"]))
    context.register(
        artifact_id="wave.wavenumber",
        path=path,
        kind="array",
        variable=variable,
        units="rad/m",
        coordinate_metadata={"frequency": "Hz", "x": "m", "wavenumber": "rad/m"},
        interpretation="Coherence-gated dominant-wave estimate; it is not an LST/PSE eigensolution.",
        provenance={
            "phase_convention": local["phase_convention"],
            "direction_selection": analysis.direction,
            "accepted_fraction": accepted,
            "growth_accepted_fraction": growth,
            "spatial_interval_method": local.get("spatial_interval_method"),
            "spatial_interval_independence_assumption": local.get(
                "spatial_interval_independence_assumption"
            ),
            "spatial_interval_degrees_of_freedom": local.get("spatial_interval_degrees_of_freedom"),
            "spatial_interval_t_multiplier": local.get("spatial_interval_t_multiplier"),
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
        },
    )
    spectrum_path = context.data_dir / "local_spatial_spectrum.npz"
    np.savez_compressed(
        spectrum_path,
        frequency_hz=local["frequency_hz"],
        x_center_m=local["x_center_m"],
        spectral_power=local["spectral_power"],
        relative_spectral_power_db=local["relative_spectral_power_db"],
        adjacent_coherence_squared=local["adjacent_coherence_squared"],
        probe_x_m=local["probe_x_sorted_m"],
        probe_indices=selected,
    )
    context.register(
        artifact_id="wave.spatial_spectrum",
        path=spectrum_path,
        kind="array",
        variable=variable,
        units=f"({unit})^2",
        coordinate_metadata={"frequency": "Hz", "x": "m"},
        interpretation="Welch-averaged local spatial spectral power and adjacent-probe coherence.",
    )
    komega_path = context.data_dir / "komega_spectrum.npz"
    np.savez_compressed(
        komega_path,
        frequency_hz=komega["frequency_hz"],
        wavenumber_rad_m=komega["wavenumber_rad_per_m"],
        power=komega["power"],
        amplitude=komega["amplitude"],
        native_frequency_resolution_hz=np.array(komega["native_frequency_resolution_hz"]),
        native_wavenumber_resolution_rad_m=np.array(
            komega["native_wavenumber_resolution_rad_per_m"]
        ),
    )
    context.register(
        artifact_id="wave.komega",
        path=komega_path,
        kind="array",
        variable=variable,
        units=f"({unit})^2",
        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
        interpretation="Signed numerical k–omega map; positive k denotes downstream cos(omega t - k x).",
        provenance={
            "temporal_window": analysis.temporal_window,
            "spatial_window": analysis.spatial_window,
            "direction_convention": komega["direction_convention"],
            "time_grid_policy": analysis.time_grid_policy,
            "resampled": resampled,
        },
    )
    if isinstance(analysis.temporal_wavenumber, TemporalWavenumberEnabled):
        from .temporal_wavenumber import compute_temporal_wavenumber_parallel

        summary, snapshots = compute_temporal_wavenumber_parallel(
            context,
            values,
            time,
            x_m,
            analysis,
        )
        temporal_path = context.data_dir / "temporal_wavenumber.npz"
        np.savez_compressed(
            temporal_path,
            time_center_s=summary["time_center_s"],
            wavenumber_rad_m=summary["wavenumber_rad_m"],
            band_power=summary["band_power"],
            relative_band_power_db=summary["relative_band_power_db"],
            band_energy=summary["band_energy"],
            relative_energy_db=summary["relative_energy_db"],
            dominant_wavenumber_rad_m=summary["dominant_wavenumber_rad_m"],
            dominant_frequency_hz=summary["dominant_frequency_hz"],
            phase_speed_m_s=summary["phase_speed_m_s"],
            wavelength_m=summary["wavelength_m"],
            active_mask=summary["active_mask"],
            valid_time_mask=summary["valid_time_mask"],
            probe_x_m=summary["probe_x_m"],
        )
        temporal_provenance = {
            "temporal_wavenumber": summary["preprocessing"],
            "snapshot_roles": summary["snapshot_roles"].tolist(),
            "snapshot_requested_time_s": summary["snapshot_requested_time_s"].tolist(),
            "snapshot_time_s": snapshots["snapshot_time_s"].tolist(),
            "valid_time_count": int(np.count_nonzero(summary["valid_time_mask"])),
        }
        context.register(
            artifact_id="wave.temporal_wavenumber",
            path=temporal_path,
            kind="array",
            variable=variable,
            units=f"({unit})^2",
            coordinate_metadata={"time": "s", "wavenumber": "rad/m", "frequency": "Hz"},
            interpretation="Sliding-window band-integrated signed wavenumber ridge for a transient probe aperture.",
            provenance=temporal_provenance,
        )
        snapshot_path = context.data_dir / "komega_snapshots.npz"
        np.savez_compressed(snapshot_path, **snapshots)
        context.register(
            artifact_id="wave.komega_snapshots",
            path=snapshot_path,
            kind="array",
            variable=variable,
            units=f"({unit})^2",
            coordinate_metadata={"snapshot_time": "s", "frequency": "Hz", "wavenumber": "rad/m"},
            interpretation="Selected shared-scale time-localized signed f-k spectra.",
            provenance=temporal_provenance,
        )
        register_temporal_wavenumber_figures(
            context,
            values=values,
            time_s=time,
            x_m=x_m,
            variable=variable,
            units=unit,
            summary=summary,
            snapshots=snapshots,
        )
        register_dispersion_figure(
            context,
            variable=variable,
            units=unit,
            frequency_hz=local["frequency_hz"],
            phase_speed_m_s=local["phase_speed_m_per_s"],
            wavelength_m=local["wavelength_m"],
            phase_valid_mask=local["phase_valid_mask"],
            x_center_m=local["x_center_m"],
        )
    sensitivity = _komega_sensitivity(values, time, x_m, analysis, komega)
    sensitivity_path = context.data_dir / "komega_sensitivity.json"
    sensitivity_path.write_text(json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="wave.komega_sensitivity",
        path=sensitivity_path,
        kind="json",
        variable=variable,
        units="rad/m",
        coordinate_metadata={"frequency": "Hz"},
        interpretation=sensitivity["interpretation"],
    )
    if analysis.diagnostics_enabled:
        raw_dt = np.diff(raw_time)
        sampling = {
            "schema_version": 1,
            "sample_count_raw": len(raw_time),
            "sample_count_fft": len(time),
            "raw_dt_min_s": float(np.min(raw_dt)) if len(raw_dt) else None,
            "raw_dt_median_s": float(np.median(raw_dt)) if len(raw_dt) else None,
            "raw_dt_max_s": float(np.max(raw_dt)) if len(raw_dt) else None,
            "resampled_dt_s": float(np.median(np.diff(time))) if len(time) > 1 else None,
            "record_duration_s": float(time[-1] - time[0]) if len(time) else None,
            "nominal_nyquist_hz": float(0.5 / np.median(np.diff(time))) if len(time) > 1 else None,
            "pressure_rise": _pressure_rise_diagnostic(raw_time, raw_values),
            "interpretation": (
                "Directional-wave sampling and rise audit; sensitivity values are diagnostic "
                "flags and do not establish convergence."
            ),
        }
        sampling_path = context.data_dir / "wave_sampling_diagnostic.json"
        sampling_path.write_text(json.dumps(sampling, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="wave.sampling_diagnostic",
            path=sampling_path,
            kind="json",
            variable=variable,
            units=unit,
            interpretation=sampling["interpretation"],
        )
    figure_path = context.figure_dir / "komega.png"
    display = (
        (komega["frequency_hz"] > 0.0)
        & (komega["frequency_hz"] >= analysis.frequency_min_hz)
        & (komega["frequency_hz"] <= analysis.frequency_max_hz)
    )
    display_frequency = komega["frequency_hz"][display]
    display_power = komega["power"][display]
    display_amplitude = komega["amplitude"][display]
    frequency_edges = np.concatenate((
        [display_frequency[0] - 0.5 * (display_frequency[1] - display_frequency[0])],
        0.5 * (display_frequency[:-1] + display_frequency[1:]),
        [display_frequency[-1] + 0.5 * (display_frequency[-1] - display_frequency[-2])],
    ))
    wavenumber = komega["wavenumber_rad_per_m"]
    wavenumber_edges = np.concatenate((
        [wavenumber[0] - 0.5 * (wavenumber[1] - wavenumber[0])],
        0.5 * (wavenumber[:-1] + wavenumber[1:]),
        [wavenumber[-1] + 0.5 * (wavenumber[-1] - wavenumber[-2])],
    ))
    figure_title = f"{variable.capitalize()}: signed f–k spectrum · {analysis.frequency_min_hz / 1e6:.3g}–{analysis.frequency_max_hz / 1e6:.3g} MHz"
    fig, axis = plt.subplots(figsize=(10, 6.5))
    if analysis.spectral_display == "amplitude":
        amplitude_max = float(np.nanmax(display_amplitude))
        image = axis.pcolormesh(
            wavenumber_edges / 1e3,
            frequency_edges / 1e6,
            display_amplitude,
            shading="auto",
            vmin=0.0,
            vmax=amplitude_max if amplitude_max > 0.0 else 1.0,
        )
        colorbar_label = f"{variable.capitalize()} spectral amplitude [{unit}]"
        figure_units = unit
    else:
        image = axis.pcolormesh(
            wavenumber_edges / 1e3,
            frequency_edges / 1e6,
            10.0 * np.log10(
                display_power / max(float(np.nanmax(display_power)), 1e-300) + 1e-300
            ),
            shading="auto",
            vmin=-60,
            vmax=0,
        )
        colorbar_label = "Power relative to map maximum [dB]"
        figure_units = "relative dB"
    axis.set_xlabel("Signed wavenumber kₓ [10³ rad m⁻¹]")
    axis.set_ylabel("Frequency [MHz]")
    axis.set_xscale("linear")
    axis.set_yscale("log")
    axis.set_xticks(np.linspace(wavenumber_edges[0] / 1e3, wavenumber_edges[-1] / 1e3, 5))
    axis.set_yticks([value / 1e6 for value in (1e5, 2e5, 5e5, 1e6, 2e6, 3e6)
                     if analysis.frequency_min_hz <= value <= analysis.frequency_max_hz])
    axis.set_yticklabels([f"{value:g}" for value in axis.get_yticks()])
    axis.set_title(figure_title)
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=int(context.project.analyses_file.presentation.figure.dpi), bbox_inches="tight")
    if figure_path.suffix.lower() == ".png":
        fig.savefig(
            figure_path.with_suffix(".pdf"),
            dpi=int(context.project.analyses_file.presentation.figure.dpi),
            bbox_inches="tight",
        )
    plt.close(fig)
    context.register(
        artifact_id="wave.komega.figure",
        path=figure_path,
        kind="figure",
        variable=variable,
        units=figure_units,
        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
        interpretation="Signed k–omega map; positive k denotes downstream cos(omega t - k x).",
    )
