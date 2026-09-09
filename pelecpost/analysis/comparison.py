"""Strict comparisons of registered artifacts from isolated runs."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, cast

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pelecpost.config.models import CaseComparisonAnalysis
from pelecpost.io.signals import (
    open_compact_signal_workspace,
    open_probe_v2_signal_workspace,
)
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .spectral import (
    FIELD_NAMES,
    SI,
    _prepare_fft_grid,
    _processed_probe_signal,
    _single_sided_amplitude,
)


def _run_path(context: WorkflowContext, archive_id: str) -> Path:
    configured = context.project.machine_file.inputs.comparison_archives[archive_id]
    return configured if configured.is_absolute() else (context.project.root / configured).resolve()


def _comparison_signal(
    context: WorkflowContext,
    archive_id: str,
    variable: str,
    probe_indices: tuple[int, ...],
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one comparison probe set (direct binary mode) through SI."""
    probe_config = context.project.machine_file.inputs.comparison_probe_sets[archive_id]
    inventory = context.plan.inventory.comparison_probe_sets[archive_id]
    storage_field = None
    for candidate in FIELD_NAMES[variable]:
        if candidate in inventory.fields:
            storage_field = candidate
            break
    if storage_field is None:
        raise KeyError(f"No storage field maps to {variable!r}")
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
    if probe_config.compact_file is not None:
        source = probe_config.compact_file
        if not source.is_absolute():
            source = (context.project.root / source).resolve()
        workspace = open_compact_signal_workspace(
            source, storage_field, selected, si_factor=factor,
            memory_limit_gb=memory_limit, scratch_directory=scratch,
        )
    else:
        patterns = []
        for pattern in probe_config.binary_files:
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
    return (
        variable, unit, workspace.time_s, workspace.x_m, workspace.values,
        workspace.selected_probe_indices,
    )


def _comparison_spectrum(
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
    scratch: Path | None,
) -> dict[str, Any]:
    """Shared Welch/FFT pipeline for one direct-probe comparison side."""
    from scipy.signal import welch

    if end_time_s is not None:
        raw_mask = time <= float(end_time_s)
        raw_indices = np.flatnonzero(raw_mask)
        if len(raw_indices) < 2:
            raise ValueError("end_time_s leaves fewer than two time samples")
        if np.max(np.diff(raw_indices)) > 1:
            raise ValueError("end_time_s window must be contiguous from the first sample")
        time = time[raw_indices]
        values = values[raw_indices, :]
    fft_time, fft_values, dt, resampled, fft_cleanup = _prepare_fft_grid(
        time, values, time_grid_policy, scratch
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
        "fluctuation_values": _temperature_fluctuation(fft_values, detrend),
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


def _temperature_fluctuation(values: np.ndarray, detrend: str) -> np.ndarray:
    """Return the unwindowed signal used to show physical fluctuations."""
    if detrend == "mean":
        return values - np.mean(values, axis=0, keepdims=True)
    if detrend == "linear":
        from scipy.signal import detrend as signal_detrend

        return signal_detrend(values, axis=0, type="linear")
    return np.asarray(values, dtype=float)


def _artifacts(run: Path) -> dict[str, dict]:
    payload = json.loads((run / "artifacts.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["artifacts"]}


def _compatible(first: dict, second: dict) -> None:
    for key in ("schema_version", "variable", "units", "coordinate_metadata", "kind"):
        if first.get(key) != second.get(key):
            raise ValueError(
                f"artifact compatibility mismatch for {key}: {first.get(key)!r} != {second.get(key)!r}"
            )
    first_preprocessing = first.get("provenance", {}).get("preprocessing")
    second_preprocessing = second.get("provenance", {}).get("preprocessing")
    if first_preprocessing is None or second_preprocessing is None:
        raise ValueError("compared artifacts must declare preprocessing provenance")
    if first_preprocessing != second_preprocessing:
        raise ValueError("artifact preprocessing provenance differs")


def _npz_metrics(first: Path, second: Path) -> list[dict]:
    metrics = []
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        common = sorted(set(left.files) & set(right.files))
        for key in common:
            a, b = np.asarray(left[key]), np.asarray(right[key])
            if a.shape != b.shape:
                raise ValueError(f"array {key!r} has incompatible shapes {a.shape} and {b.shape}")
            if a.dtype.kind not in "iufc" or b.dtype.kind not in "iufc":
                continue
            difference = np.asarray(a - b)
            finite = np.isfinite(difference)
            if not np.any(finite):
                continue
            metrics.append({
                "quantity": key,
                "shape": list(a.shape),
                "l2_difference": float(np.linalg.norm(difference[finite])),
                "linf_difference": float(np.max(np.abs(difference[finite]))),
                "reference_l2": float(np.linalg.norm(a[np.isfinite(a)])),
            })
    if not metrics:
        raise ValueError("compatible NPZ artifacts share no finite numeric arrays")
    return metrics


def _numeric_metric(name: str, first: np.ndarray, second: np.ndarray) -> dict | None:
    if first.shape != second.shape:
        raise ValueError(f"quantity {name!r} has incompatible shapes {first.shape} and {second.shape}")
    if first.dtype.kind not in "iufc" or second.dtype.kind not in "iufc":
        return None
    difference = np.asarray(first - second)
    finite = np.isfinite(difference)
    if not np.any(finite):
        return None
    return {
        "quantity": name,
        "shape": list(first.shape),
        "l2_difference": float(np.linalg.norm(difference[finite])),
        "linf_difference": float(np.max(np.abs(difference[finite]))),
        "reference_l2": float(np.linalg.norm(first[np.isfinite(first)])),
    }


def _flatten_json(value, prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_json(item, child))
    elif isinstance(value, (int, float, bool)) and not isinstance(value, str):
        result[prefix] = np.asarray(value)
    elif isinstance(value, list):
        candidate = np.asarray(value)
        if candidate.dtype.kind in "biufc":
            result[prefix] = candidate
    return result


def _json_metrics(first: Path, second: Path) -> list[dict]:
    left = _flatten_json(json.loads(first.read_text(encoding="utf-8")))
    right = _flatten_json(json.loads(second.read_text(encoding="utf-8")))
    if set(left) != set(right):
        raise ValueError("JSON numerical quantity paths differ")
    metrics = [
        metric for key in sorted(left)
        if (metric := _numeric_metric(key, left[key], right[key])) is not None
    ]
    if not metrics:
        raise ValueError("compatible JSON artifacts contain no finite numeric quantities")
    return metrics


def _csv_metrics(first: Path, second: Path) -> list[dict]:
    def read(path: Path) -> tuple[list[str], list[list[str]]]:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            rows = list(reader)
        if not rows:
            raise ValueError(f"empty CSV artifact: {path}")
        return rows[0], rows[1:]

    left_columns, left_rows = read(first)
    right_columns, right_rows = read(second)
    if left_columns != right_columns or len(left_rows) != len(right_rows):
        raise ValueError("CSV columns or row counts differ")
    metrics = []
    for index, name in enumerate(left_columns):
        try:
            left = np.asarray([float(row[index]) for row in left_rows])
            right = np.asarray([float(row[index]) for row in right_rows])
        except (ValueError, IndexError):
            continue
        metric = _numeric_metric(name, left, right)
        if metric is not None:
            metrics.append(metric)
    if not metrics:
        raise ValueError("compatible CSV artifacts contain no finite numeric columns")
    return metrics


def _hdf5_metrics(first: Path, second: Path) -> list[dict]:
    def arrays(path: Path) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as archive:
            def visit(name: str, item) -> None:
                if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc":
                    result[name] = np.asarray(item)
            archive.visititems(visit)
        return result

    left, right = arrays(first), arrays(second)
    if set(left) != set(right):
        raise ValueError("HDF5 numerical dataset paths differ")
    metrics = [
        metric for key in sorted(left)
        if (metric := _numeric_metric(key, left[key], right[key])) is not None
    ]
    if not metrics:
        raise ValueError("compatible HDF5 artifacts contain no finite numeric datasets")
    return metrics


def _artifact_metrics(first: Path, second: Path) -> list[dict]:
    if first.suffix.lower() != second.suffix.lower():
        raise ValueError("artifact file formats differ")
    suffix = first.suffix.lower()
    if suffix == ".npz":
        return _npz_metrics(first, second)
    if suffix == ".json":
        return _json_metrics(first, second)
    if suffix == ".csv":
        return _csv_metrics(first, second)
    if suffix in {".h5", ".hdf5"}:
        return _hdf5_metrics(first, second)
    raise ValueError(f"artifact format {suffix!r} is not a comparable numerical product")


def _probe_overlay_figure(
    left: Any, right: Any,
    left_indices: tuple[int, ...], right_indices: tuple[int, ...],
    probes_to_render: tuple[int, ...],
    baseline_label: str, comparison_label: str,
    figure_path: Path,
) -> tuple[Path, ...]:
    unit = str(np.asarray(left["signal_unit"])) if "signal_unit" in left.files else "unit"
    left_map = {int(v): i for i, v in enumerate(left_indices)}
    right_map = {int(v): i for i, v in enumerate(right_indices)}
    baseline_style = dict(color="tab:blue", linestyle="-", linewidth=1.0)
    comparison_style = dict(color="tab:orange", linestyle="--", linewidth=1.0)
    if unit == "K":
        raw_ylabel, fluctuation_ylabel, fft_ylabel = "T [K]", "$\\Delta T$ [K]", "|FFT| [K]"
        raw_title, fluctuation_title = "Raw temperature", "Temperature fluctuation"
    else:
        raw_ylabel = f"Signal [{unit}]"
        fluctuation_ylabel = f"$\\Delta$ signal [{unit}]"
        fft_ylabel = f"|FFT| [{unit}]"
        raw_title, fluctuation_title = "Raw signal", "Signal fluctuation"
    pair_groups = tuple(
        tuple(probes_to_render[index:index + 2])
        for index in range(0, len(probes_to_render), 2)
    )
    output_paths: list[Path] = []
    for pair_number, pair in enumerate(pair_groups):
        rows = len(pair)
        pair_suffix = "_".join(str(probe_index) for probe_index in pair)
        pair_path = (
            figure_path if len(pair_groups) == 1
            else figure_path.with_name(
                f"{figure_path.stem}_probes_{pair_suffix}{figure_path.suffix}"
            )
        )
        fig, axes = plt.subplots(
            rows, 3, figsize=(15.5, 4.8 + 2.65 * (rows - 1)), squeeze=False,
            sharex="col",
        )
        for row, probe_index in enumerate(pair):
            li = left_map[int(probe_index)]
            ri = right_map[int(probe_index)]
            xm = float(left["x_m"][li]) if "x_m" in left.files else float(right["x_m"][ri])
            raw_axis, fluctuation_axis, fft_axis = axes[row]
            # Column 0: raw time history overlay (fallback to processed if raw unavailable).
            if "raw_time_s" in left.files and "raw_time_s" in right.files and "raw_values" in left.files:
                raw_axis.plot(left["raw_time_s"] * 1.0e6, left["raw_values"][:, li], **baseline_style)
                raw_axis.plot(right["raw_time_s"] * 1.0e6, right["raw_values"][:, ri], **comparison_style)
                raw_axis.set_title(f"{raw_title}\nProbe {probe_index}, x = {xm * 100:.3f} cm", fontsize=11)
                raw_axis.set_ylabel(raw_ylabel)
                raw_axis.grid(True, alpha=0.25)
            elif "time_s" in left.files and "processed_values" in left.files:
                raw_axis.plot(left["time_s"] * 1.0e6, left["processed_values"][:, li], **baseline_style)
                raw_axis.plot(right["time_s"] * 1.0e6, right["processed_values"][:, ri], **comparison_style)
                raw_axis.set_title(f"FFT input\nProbe {probe_index}, x = {xm * 100:.3f} cm", fontsize=11)
                raw_axis.set_ylabel(f"Signal [{unit}]")
                raw_axis.grid(True, alpha=0.25)
            # Column 1: unwindowed fluctuation, which preserves its physical scale.
            if "time_s" in left.files and "fluctuation_values" in left.files:
                fluctuation_axis.plot(left["time_s"] * 1.0e6, left["fluctuation_values"][:, li], **baseline_style)
                fluctuation_axis.plot(right["time_s"] * 1.0e6, right["fluctuation_values"][:, ri], **comparison_style)
                if row == 0:
                    fluctuation_axis.set_title(
                        f"{fluctuation_title}\n$T - \\overline{{T}}$ (unwindowed)",
                        fontsize=12,
                    )
                fluctuation_axis.set_ylabel(fluctuation_ylabel)
                fluctuation_axis.grid(True, alpha=0.25)
            elif "time_s" in left.files and "processed_values" in left.files:
                fluctuation_axis.plot(left["time_s"] * 1.0e6, left["processed_values"][:, li], **baseline_style)
                fluctuation_axis.plot(right["time_s"] * 1.0e6, right["processed_values"][:, ri], **comparison_style)
                if row == 0:
                    fluctuation_axis.set_title("FFT-prepared signal", fontsize=12)
                fluctuation_axis.set_ylabel(f"Processed [{unit}]")
                fluctuation_axis.grid(True, alpha=0.25)
            # Column 2: FFT amplitude overlay (log-log).
            if "frequency_hz" in left.files and "amplitude" in left.files:
                mag = left["amplitude"][:, li] if li < left["amplitude"].shape[1] else None
                rmag = right["amplitude"][:, ri] if ri < right["amplitude"].shape[1] else None
                if mag is not None and rmag is not None:
                    mask = (left["frequency_hz"] > 0) & (mag > 0)
                    rmask = (right["frequency_hz"] > 0) & (rmag > 0)
                    if np.any(mask):
                        fft_axis.loglog(left["frequency_hz"][mask], mag[mask], **baseline_style)
                    if np.any(rmask):
                        fft_axis.loglog(right["frequency_hz"][rmask], rmag[rmask], **comparison_style)
                    if row == 0:
                        fft_axis.set_title("Single-sided FFT amplitude", fontsize=12)
                    fft_axis.set_ylabel(fft_ylabel)
                    fft_axis.grid(True, which="both", alpha=0.25)
            if row == rows - 1:
                raw_axis.set_xlabel("Time [$\\mu$s]")
                fluctuation_axis.set_xlabel("Time [$\\mu$s]")
                fft_axis.set_xlabel("Frequency [Hz]")
        handles = [
            plt.Line2D([], [], **baseline_style, label=baseline_label),
            plt.Line2D([], [], **comparison_style, label=comparison_label),
        ]
        fig.suptitle(
            f"Probe pair: {pair[0]} and {pair[-1]}",
            fontsize=16, fontweight="bold", y=0.99,
        )
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
            ncol=2, fontsize=10, frameon=False,
        )
        fig.subplots_adjust(top=0.83, hspace=0.42, wspace=0.28)
        fig.savefig(pair_path, dpi=180)
        plt.close(fig)
        output_paths.append(pair_path)
    return tuple(output_paths)


def _psd_overlay_figure(
    left: Any, right: Any,
    probes_to_plot: tuple[int, ...],
    baseline_label: str, comparison_label: str,
    figure_path: Path,
) -> None:
    left_idx = {int(v): i for i, v in enumerate(np.asarray(left["probe_indices"]).ravel())}
    right_idx = {int(v): i for i, v in enumerate(np.asarray(right["probe_indices"]).ravel())}
    unit = str(np.asarray(left["psd_unit"])) if "psd_unit" in left.files else "unit^2/Hz"
    fig, axes = plt.subplots(
        max(len(probes_to_plot), 1), 1,
        figsize=(9, max(3.2, 3.0 * len(probes_to_plot))), squeeze=False,
    )
    for row, probe_index in enumerate(probes_to_plot):
        li, ri = left_idx[int(probe_index)], right_idx[int(probe_index)]
        xm_left = float(left["x_m"][li]) if "x_m" in left.files else float("nan")
        xm_right = float(right["x_m"][ri]) if "x_m" in right.files else float("nan")
        xcm = xm_left if np.isfinite(xm_left) else xm_right
        axis = axes[row, 0]
        axis.loglog(left["frequency_hz"], left["psd"][:, li], color="tab:blue", linestyle="-", linewidth=1.0, label=baseline_label)
        axis.loglog(right["frequency_hz"], right["psd"][:, ri], color="tab:orange", linestyle="--", linewidth=1.0, label=comparison_label)
        axis.set_title(f"Probe {probe_index} x={xcm * 100:.3f} cm PSD", fontsize=9)
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel(f"PSD [{unit}]")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _render_overlay(context: WorkflowContext, analysis: CaseComparisonAnalysis) -> None:
    overlay_requested = tuple(analysis.overlay_probes)
    baseline = _run_path(context, analysis.baseline_id)
    comparison = _run_path(context, analysis.comparison_id)
    baseline_artifacts = _artifacts(baseline)
    comparison_artifacts = _artifacts(comparison)
    baseline_label = analysis.baseline_id
    comparison_label = analysis.comparison_id
    registered_any = False
    for artifact_id in analysis.artifact_ids:
        if artifact_id not in baseline_artifacts or artifact_id not in comparison_artifacts:
            continue
        left_meta = baseline_artifacts[artifact_id]
        right_meta = comparison_artifacts[artifact_id]
        if left_meta.get("kind") != "array":
            continue
        first_path = baseline / left_meta["path"]
        second_path = comparison / right_meta["path"]
        if first_path.suffix.lower() != ".npz" or second_path.suffix.lower() != ".npz":
            continue
        if artifact_id.endswith("spectral.probe_signals"):
            with np.load(first_path, allow_pickle=False) as left_probe, np.load(second_path, allow_pickle=False) as right_probe:
                if "probe_indices" not in left_probe.files or "probe_indices" not in right_probe.files:
                    continue
                left_indices = tuple(int(v) for v in np.asarray(left_probe["probe_indices"]).ravel())
                right_indices = tuple(int(v) for v in np.asarray(right_probe["probe_indices"]).ravel())
                common = tuple(v for v in left_indices if v in right_indices)
                probes_to_render: tuple[int, ...]
                if overlay_requested == ():
                    probes_to_render = common
                else:
                    probes_to_render = tuple(v for v in overlay_requested if v in set(common))
                if not probes_to_render:
                    continue
                safe = artifact_id.replace(".", "_")
                figure_path = context.figure_dir / f"overlay_{safe}_vs_{comparison_label}.png"
                figure_paths = _probe_overlay_figure(
                    left_probe, right_probe, left_indices, right_indices,
                    probes_to_render, baseline_label, comparison_label, figure_path,
                )
                for pair_number, pair_figure_path in enumerate(figure_paths):
                    suffix = f"{safe}.pair_{pair_number}"
                    context.register(
                        artifact_id=f"comparison.overlay_figure.{suffix}",
                        path=pair_figure_path, kind="figure", variable=None,
                        units="artifact-dependent", coordinate_metadata={},
                        interpretation="Overlay of one baseline/comparison probe pair with time histories and FFTs.",
                    )
                    registered_any = True
            continue
        # Generic PSD / array overlay: look for psd + frequency_hz
        with np.load(first_path, allow_pickle=False) as left, np.load(second_path, allow_pickle=False) as right:
            if "psd" not in left.files or "frequency_hz" not in left.files:
                continue
            if "probe_indices" not in left.files or "probe_indices" not in right.files:
                continue
            common_set = set(int(v) for v in np.asarray(left["probe_indices"]).ravel()) & set(int(v) for v in np.asarray(right["probe_indices"]).ravel())
            probes_to_plot = tuple(sorted(common_set)) if overlay_requested == () else tuple(v for v in sorted(overlay_requested) if v in common_set)
            if not probes_to_plot:
                continue
            safe = artifact_id.replace(".", "_")
            figure_path = context.figure_dir / f"overlay_{safe}_psd.png"
            _psd_overlay_figure(left, right, probes_to_plot, baseline_label, comparison_label, figure_path)
            context.register(
                artifact_id="comparison.overlay_figure" if not registered_any else f"comparison.overlay_figure.{safe}",
                path=figure_path, kind="figure", variable=None,
                units="artifact-dependent", coordinate_metadata={},
                interpretation="Per-probe Welch PSD overlay of baseline vs comparison.",
            )
            registered_any = True
    if not registered_any:
        # Always satisfy the declared comparison.overlay_figure output.
        placeholder = context.figure_dir / "overlay_placeholder.png"
        fig, axis = plt.subplots(figsize=(7, 3))
        axis.text(0.5, 0.5, "No overlay-compatible array artifacts\nin this comparison.",
                  ha="center", va="center", fontsize=10, transform=axis.transAxes)
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(placeholder, dpi=180)
        plt.close(fig)
        context.register(
            artifact_id="comparison.overlay_figure",
            path=placeholder, kind="figure", variable=None,
            units="artifact-dependent", coordinate_metadata={},
            interpretation="Placeholder overlay; no compatible array artifacts matched the overlay criteria.",
        )


def _run_direct_probe_comparison(context: WorkflowContext, analysis: CaseComparisonAnalysis) -> None:
    """Direct probe mode: both sides from comparison_probe_sets binaries."""
    variable = analysis.variable.value if hasattr(analysis.variable, "value") else str(analysis.variable)
    selected = tuple(analysis.probe_indices) or tuple(analysis.overlay_probes)
    if not selected:
        raise ValueError("direct probe comparison requires probe_indices or overlay_probes")
    overlays = tuple(analysis.overlay_probes) or selected
    window = analysis.window or "hann"
    detrend = analysis.detrend or "mean"
    overlap_fraction = 0.5 if analysis.overlap_fraction is None else float(analysis.overlap_fraction)
    time_grid_policy = analysis.time_grid_policy or "resample_uniform"
    scratch = context.project.machine_file.compute.scratch_directory
    if scratch is not None and not scratch.is_absolute():
        scratch = (context.project.root / scratch).resolve()
    sides: dict[str, dict[str, Any]] = {}
    for archive_id in (analysis.baseline_id, analysis.comparison_id):
        _, unit, time, x_m, values, probed = _comparison_signal(
            context, archive_id, variable, selected
        )
        products = _comparison_spectrum(
            time, values,
            end_time_s=analysis.end_time_s,
            window=window, detrend=detrend,
            welch_segment_samples=analysis.welch_segment_samples,
            overlap_fraction=overlap_fraction,
            time_grid_policy=time_grid_policy,
            frequency_max_hz=analysis.frequency_max_hz,
            scratch=scratch,
        )
        if products["cleanup"] is not None:
            context.add_cleanup(products["cleanup"])
        sides[archive_id] = {
            "unit": unit, "x_m": x_m, "selected": probed, **products,
        }
    left, right = sides[analysis.baseline_id], sides[analysis.comparison_id]
    if left["psd"].shape != right["psd"].shape:
        raise ValueError(
            f"direct comparison PSD shapes differ {left['psd'].shape} vs "
            f"{right['psd'].shape}; set end_time_s and identical Welch settings"
        )
    if not np.array_equal(left["welch_frequency"], right["welch_frequency"]):
        raise ValueError("direct comparison frequency grids differ; check end_time_s")
    # Persist raw arrays so metrics/overlays are reproducible in the run dir.
    signal_path = context.data_dir / "direct_probe_comparison.npz"
    np.savez_compressed(
        signal_path,
        baseline_raw_time_s=left["raw_time"], baseline_raw_values=left["raw_values"],
        comparison_raw_time_s=right["raw_time"], comparison_raw_values=right["raw_values"],
        baseline_time_s=left["fft_time"], baseline_processed_values=left["processed_values"],
        comparison_time_s=right["fft_time"], comparison_processed_values=right["processed_values"],
        baseline_fluctuation_values=left["fluctuation_values"],
        comparison_fluctuation_values=right["fluctuation_values"],
        baseline_frequency_hz=left["fft_frequency"], baseline_amplitude=left["fft_amplitude"],
        comparison_frequency_hz=right["fft_frequency"], comparison_amplitude=right["fft_amplitude"],
        welch_frequency_hz=left["welch_frequency"],
        baseline_psd=left["psd"], comparison_psd=right["psd"],
        probe_indices=np.asarray(selected),
        baseline_x_m=left["x_m"], comparison_x_m=right["x_m"],
        signal_unit=np.array(left["unit"]),
    )
    common = tuple(v for v in selected if v in set(selected))
    probes_to_render = (
        tuple(v for v in overlays if v in set(common)) if overlays else common
    )
    # Metrics: per-array L2/Linf between the two sides.
    all_metrics = []
    arrays = {
        "processed_values": (left["processed_values"], right["processed_values"]),
        "amplitude": (left["fft_amplitude"], right["fft_amplitude"]),
        "psd": (left["psd"], right["psd"]),
    }
    for name, (first_array, second_array) in arrays.items():
        metric = _numeric_metric(name, np.asarray(first_array), np.asarray(second_array))
        if metric is not None:
            all_metrics.append({"artifact_id": "direct_probe_comparison", **metric})
    payload: dict[str, Any] = {
        "schema_version": 1,
        "baseline_run": f"probe_set:{analysis.baseline_id}",
        "comparison_run": f"probe_set:{analysis.comparison_id}",
        "metrics": all_metrics,
        "interpretation": "Direct probe-binary comparison with shared preprocessing.",
    }
    metrics_path = context.data_dir / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="comparison.metrics", path=metrics_path, kind="json",
        variable=variable, units=None, coordinate_metadata={},
        interpretation=str(payload["interpretation"]),
        provenance={
            "mode": "direct_probe_sets",
            "baseline_id": analysis.baseline_id,
            "comparison_id": analysis.comparison_id,
            "end_time_s": analysis.end_time_s,
            "window": window, "detrend": detrend,
            "time_grid_policy": time_grid_policy,
        },
    )
    context.register(
        artifact_id="comparison.probe_signals", path=signal_path, kind="array",
        variable=variable, units=left["unit"],
        coordinate_metadata={"time": "s", "frequency": "Hz", "probe_x": "m"},
        interpretation="Direct two-sided probe histories, FFT amplitudes, and Welch PSDs.",
        provenance={
            "mode": "direct_probe_sets",
            "baseline_id": analysis.baseline_id,
            "comparison_id": analysis.comparison_id,
        },
    )
    # Overlay figures: reuse the shared helpers on in-memory dict views.
    baseline_label, comparison_label = analysis.baseline_id, analysis.comparison_id
    unit = left["unit"]

    class _View:
        def __init__(self, mapping: dict[str, Any], prefix: str):
            self._m = mapping
            self._p = prefix

        @property
        def files(self) -> list[str]:
            keys = []
            for key in (
                "raw_time_s", "raw_values", "time_s", "processed_values", "fluctuation_values",
                "frequency_hz", "amplitude", "probe_indices", "x_m",
                "signal_unit", "psd", "psd_unit",
            ):
                candidate = f"{self._p}_{key}" if self._p else key
                # Map to underlying storage names
                mapping = {
                    "raw_time_s": "raw_time", "raw_values": "raw_values",
                    "time_s": "fft_time", "processed_values": "processed_values",
                    "fluctuation_values": "fluctuation_values",
                    "frequency_hz": "fft_frequency", "amplitude": "fft_amplitude",
                    "probe_indices": "probe_indices_global",
                    "x_m": "x_m", "signal_unit": "signal_unit",
                    "psd": "psd", "psd_unit": "psd_unit",
                }
                if mapping[key] in self._m:
                    keys.append(key)
            return keys

        def __contains__(self, key: str) -> bool:
            return key in self.files

        def __getitem__(self, key: str):
            mapping = {
                "raw_time_s": "raw_time", "raw_values": "raw_values",
                "time_s": "fft_time", "processed_values": "processed_values",
                "fluctuation_values": "fluctuation_values",
                "frequency_hz": "fft_frequency", "amplitude": "fft_amplitude",
                "probe_indices": "probe_indices_global",
                "x_m": "x_m", "signal_unit": "signal_unit",
                "psd": "psd", "psd_unit": "psd_unit",
            }
            return self._m[mapping[key]]

        def get(self, key: str, default: Any = None):
            return self[key] if key in self else default

    left_view = _View({
        "raw_time": left["raw_time"], "raw_values": left["raw_values"],
        "fft_time": left["fft_time"], "processed_values": left["processed_values"],
        "fluctuation_values": left["fluctuation_values"],
        "fft_frequency": left["fft_frequency"], "fft_amplitude": left["fft_amplitude"],
        "probe_indices_global": np.asarray(selected), "x_m": left["x_m"],
        "signal_unit": np.array(unit), "psd": left["psd"],
        "psd_unit": np.array(f"({unit})^2/Hz"),
    }, "")
    right_view = _View({
        "raw_time": right["raw_time"], "raw_values": right["raw_values"],
        "fft_time": right["fft_time"], "processed_values": right["processed_values"],
        "fluctuation_values": right["fluctuation_values"],
        "fft_frequency": right["fft_frequency"], "fft_amplitude": right["fft_amplitude"],
        "probe_indices_global": np.asarray(selected), "x_m": right["x_m"],
        "signal_unit": np.array(unit), "psd": right["psd"],
        "psd_unit": np.array(f"({unit})^2/Hz"),
    }, "")
    # time/FFT overlay
    time_figure = context.figure_dir / "overlay_direct_probe_time_fft.png"
    time_figure_paths = _probe_overlay_figure(
        left_view, right_view, selected, selected, probes_to_render,
        baseline_label, comparison_label, time_figure,
    )
    for pair_number, pair_figure_path in enumerate(time_figure_paths):
        context.register(
            artifact_id=f"comparison.overlay_figure.probe_signals.pair_{pair_number}",
            path=pair_figure_path, kind="figure", variable=variable,
            units="artifact-dependent", coordinate_metadata={},
            interpretation="Overlay of one baseline/comparison probe pair with time histories and FFTs.",
        )
    # PSD overlay uses the Welch frequency grid. Build views whose
    # frequency_hz/amplitude keys alias the Welch arrays.
    def _psd_view(side: dict[str, Any]) -> _View:
        view = _View({
            "raw_time": side["raw_time"], "raw_values": side["raw_values"],
            "fft_time": side["fft_time"], "processed_values": side["processed_values"],
            "fft_frequency": side["welch_frequency"], "fft_amplitude": side["psd"],
            "probe_indices_global": np.asarray(selected), "x_m": side["x_m"],
            "signal_unit": np.array(unit), "psd": side["psd"],
            "psd_unit": np.array(f"({unit})^2/Hz"),
        }, "")
        view._m["frequency_hz_alias"] = side["welch_frequency"]
        view._m["amplitude_alias"] = side["psd"]
        return view

    class _PsdView(_View):
        def __getitem__(self, key: str):
            if key == "frequency_hz":
                return self._m["frequency_hz_alias"]
            if key == "amplitude":
                return self._m["amplitude_alias"]
            return super().__getitem__(key)

        def __contains__(self, key: str) -> bool:
            if key in ("frequency_hz", "amplitude"):
                return True
            return super().__contains__(key)

        @property
        def files(self) -> list[str]:
            base = super().files
            for extra in ("frequency_hz", "amplitude"):
                if extra not in base:
                    base = [*base, extra]
            return base

    psd_figure = context.figure_dir / "overlay_direct_probe_psd.png"
    _psd_overlay_figure(
        _PsdView(_psd_view(left)._m, ""), _PsdView(_psd_view(right)._m, ""),
        probes_to_render,
        baseline_label, comparison_label, psd_figure,
    )
    context.register(
        artifact_id="comparison.overlay_figure.psd",
        path=psd_figure, kind="figure", variable=variable,
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Per-probe Welch PSD overlay of baseline vs comparison.",
    )
    # Linf bar chart for parity with archive mode
    figure_path = context.figure_dir / "comparison_linf.png"
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * max(len(all_metrics), 1)), 5))
    labels = [f"direct:{item['quantity']}" for item in all_metrics]
    values_bar = [item["linf_difference"] for item in all_metrics]
    axis.bar(np.arange(len(all_metrics)), values_bar if values_bar else [0.0])
    axis.set_xticks(np.arange(len(labels)) if labels else [0], labels if labels else [""], rotation=60, ha="right")
    axis.set_ylabel("Maximum absolute difference")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="comparison.figures", path=figure_path, kind="figure",
        variable=None, units="artifact-dependent", coordinate_metadata={},
        interpretation="Maximum absolute differences for direct probe comparison arrays.",
    )


@executor("case_comparison")
def run_case_comparison(context: WorkflowContext) -> None:
    analysis = cast(CaseComparisonAnalysis, context.analysis)
    if getattr(analysis, "variable", None) is not None:
        _run_direct_probe_comparison(context, analysis)
        return
    baseline = _run_path(context, analysis.baseline_id)
    comparison = _run_path(context, analysis.comparison_id)
    baseline_artifacts = _artifacts(baseline)
    comparison_artifacts = _artifacts(comparison)
    all_metrics = []
    for artifact_id in analysis.artifact_ids:
        if artifact_id not in baseline_artifacts or artifact_id not in comparison_artifacts:
            raise ValueError(f"artifact {artifact_id!r} is not registered in both runs")
        first = baseline_artifacts[artifact_id]
        second = comparison_artifacts[artifact_id]
        _compatible(first, second)
        first_path = baseline / first["path"]
        second_path = comparison / second["path"]
        for metric in _artifact_metrics(first_path, second_path):
            all_metrics.append({"artifact_id": artifact_id, **metric})
    payload: dict[str, Any] = {
        "schema_version": 1, "baseline_run": str(baseline),
        "comparison_run": str(comparison), "metrics": all_metrics,
        "interpretation": "Differences are reported only after strict metadata and shape compatibility checks.",
    }
    path = context.data_dir / "comparison_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="comparison.metrics", path=path, kind="json", variable=None,
        units=None, coordinate_metadata={}, interpretation=str(payload["interpretation"]),
        provenance={"baseline_run": str(baseline), "comparison_run": str(comparison)},
    )
    figure_path = context.figure_dir / "comparison_linf.png"
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * len(all_metrics)), 5))
    labels = [f"{item['artifact_id']}:{item['quantity']}" for item in all_metrics]
    axis.bar(np.arange(len(all_metrics)), [item["linf_difference"] for item in all_metrics])
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylabel("Maximum absolute difference")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="comparison.figures", path=figure_path, kind="figure", variable=None,
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Maximum absolute differences for compatible numerical arrays.",
    )
    _render_overlay(context, analysis)
