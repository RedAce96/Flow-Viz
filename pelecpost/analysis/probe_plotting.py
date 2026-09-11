"""Shared, provenance-aware figures for selected probe traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pelecpost.config.models import PresentationConfig
from pelecpost.runtime.context import WorkflowContext
from pelecpost.runtime.parallel import ParallelTask, plan_parallel_stage, stage_readonly_array
from pelecpost.visualization import (
    legend_location,
    plot_profile,
    resolve_line_style,
    resolve_presentation,
    save_figure_variants,
)


def _selected_labels(
    context: WorkflowContext, selected: np.ndarray, x_m: np.ndarray,
) -> list[str]:
    config = context.analysis.probe_plotting
    inventory = context.plan.inventory.probe_sets.get(context.analysis.probe_set_id)
    y_all = inventory.requested_y_m if inventory is not None else ()
    labels: list[str] = []
    for column, (index, x_value) in enumerate(zip(selected, x_m)):
        y_value = float(y_all[int(index)]) if int(index) < len(y_all) else 0.0
        if config.label == "index":
            labels.append(f"Probe {int(index)}")
        elif config.label == "coordinates":
            labels.append(f"x={float(x_value):.6g} m, y={y_value:.6g} m")
        else:
            labels.append(
                f"Probe {int(index)} · x={float(x_value):.6g} m, y={y_value:.6g} m"
            )
    return labels


def _display_values(values: np.ndarray, normalization: str) -> tuple[np.ndarray, list[float]]:
    data = np.asarray(values, dtype=float)
    if normalization == "none":
        return data, [1.0] * data.shape[1]
    displayed = np.array(data, dtype=float, copy=True)
    scales: list[float] = []
    for column in range(data.shape[1]):
        finite = np.isfinite(data[:, column])
        scale = float(np.max(np.abs(data[finite, column]))) if np.any(finite) else 0.0
        scales.append(scale)
        if scale > 0.0:
            displayed[:, column] /= scale
        else:
            displayed[:, column] = np.nan
    return displayed, scales


def _figure_config(context: WorkflowContext):
    return resolve_presentation(
        context.project.analyses_file.presentation, context.analysis.presentation
    )


def _register_figure_variants(
    context: WorkflowContext,
    *,
    figure: Any,
    artifact_id: str,
    stem: Path,
    variable: str,
    units: str,
    interpretation: str,
    provenance: dict[str, Any],
) -> tuple[Path, ...]:
    paths = save_figure_variants(figure, stem, _figure_config(context).figure)
    for index, path in enumerate(paths):
        suffix = "" if index == 0 else f".{path.suffix.lstrip('.') }"
        context.register(
            artifact_id=f"{artifact_id}{suffix}", path=path, kind="figure",
            variable=variable, units=units,
            coordinate_metadata={"time": "s", "probe_x": "m", "probe_y": "m"},
            interpretation=interpretation,
            provenance=provenance | {"figure_format": path.suffix.lstrip(".")},
        )
    return paths


def _save_line_figure(
    context: WorkflowContext,
    *,
    artifact_id: str,
    filename: str,
    x: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    x_label: str,
    y_label: str,
    variable: str,
    units: str,
    interpretation: str,
    provenance: dict[str, Any],
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Path, ...]:
    presentation = _figure_config(context)
    figure, axis = plt.subplots(
        figsize=(presentation.figure.width_in, max(4.5, presentation.figure.height_in))
    )
    style = resolve_line_style(presentation.line_defaults, None)
    for column, label in enumerate(labels):
        line_style = style.model_copy(update={
            "color": plt.get_cmap("tab10")(column % 10),
        })
        plot_profile(axis, x, values[:, column], label, line_style)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
    axis.legend(loc=legend_location(style), ncol=max(1, (len(labels) + 7) // 8))
    if style.grid:
        axis.grid(True, alpha=0.25)
    else:
        axis.grid(False)
    figure.tight_layout()
    return _register_figure_variants(
        context, figure=figure, artifact_id=artifact_id,
        stem=context.figure_dir / filename,
        variable=variable, units=units, interpretation=interpretation,
        provenance=provenance,
    )


def _probe_line_figure_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Render one probe overlay from read-only staged arrays."""
    created: list[str] = []
    try:
        presentation = PresentationConfig.model_validate(payload["presentation"])
        x = np.load(payload["x_path"], mmap_mode="r")
        values = np.load(payload["values_path"], mmap_mode="r")
        labels = list(payload["labels"])
        figure, axis = plt.subplots(
            figsize=(presentation.figure.width_in, max(4.5, presentation.figure.height_in))
        )
        style = resolve_line_style(presentation.line_defaults, None)
        for column, label in enumerate(labels):
            line_style = style.model_copy(update={
                "color": plt.get_cmap("tab10")(column % 10),
            })
            plot_profile(axis, x, values[:, column], label, line_style)
        axis.set_xlabel(payload["x_label"])
        axis.set_ylabel(payload["y_label"])
        if payload.get("log_x"):
            axis.set_xscale("log")
        if payload.get("log_y"):
            axis.set_yscale("log")
        axis.legend(
            loc=legend_location(style),
            ncol=max(1, (len(labels) + 7) // 8),
        )
        # Avoid Matplotlib's warning-producing ``grid(False, properties...)``
        # call while retaining the configured no-grid default.
        if style.grid:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        paths = save_figure_variants(
            figure, Path(payload["stem"]), presentation.figure,
        )
        created.extend(str(path) for path in paths)
        return {"paths": created}
    except BaseException:
        for path in created:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
        plt.close("all")
        raise


def _register_saved_line_paths(
    context: WorkflowContext,
    *,
    result: dict[str, Any],
    artifact_id: str,
    variable: str,
    units: str,
    coordinate_metadata: dict[str, Any],
    interpretation: str,
    provenance: dict[str, Any],
) -> None:
    paths = tuple(Path(path) for path in result["paths"])
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"parallel figure worker did not create {path}")
        suffix = "" if index == 0 else f".{path.suffix.lstrip('.') }"
        context.register(
            artifact_id=f"{artifact_id}{suffix}", path=path, kind="figure",
            variable=variable, units=units,
            coordinate_metadata=coordinate_metadata,
            interpretation=interpretation,
            provenance=provenance | {"figure_format": path.suffix.lstrip(".")},
        )


def _probe_parallel_plan(context: WorkflowContext, task_count: int, matrix_bytes: int):
    metadata = context.resource_metadata or {}
    parent_gb = float(metadata.get("parent_resident_gb", 0.25))
    per_worker = max(0.05, 0.15 + float(matrix_bytes) / 1024**3)
    return plan_parallel_stage(
        "probe-figure-render",
        requested_workers=int(context.project.machine_file.compute.workers),
        task_count=task_count,
        memory_limit_gb=float(context.project.machine_file.compute.memory_limit_gb),
        parent_resident_gb=parent_gb,
        per_worker_peak_gb=per_worker,
    )


def register_probe_trace_figures(
    context: WorkflowContext,
    *,
    raw_time: np.ndarray,
    raw_values: np.ndarray,
    prepared_time: np.ndarray,
    prepared_values: np.ndarray,
    x_m: np.ndarray,
    selected: np.ndarray,
    variable: str,
    units: str,
    preprocessing: dict[str, Any],
) -> None:
    """Register shared raw and method-ready figures for all selected probes."""
    config = context.analysis.probe_plotting
    if config.mode == "panels":
        return
    labels = _selected_labels(context, selected, x_m)
    raw_display, raw_scales = _display_values(raw_values, config.normalization)
    prepared_display, prepared_scales = _display_values(
        prepared_values, config.normalization
    )
    normalization_units = "dimensionless" if config.normalization != "none" else units
    common = {
        "selected_probe_indices": [int(item) for item in selected],
        "probe_labels": labels,
        "probe_x_m": [float(item) for item in x_m],
        "normalization": config.normalization,
        "normalization_scales": raw_scales,
        "preprocessing": preprocessing,
    }
    if int(context.project.machine_file.compute.workers) <= 1:
        _save_line_figure(
            context, artifact_id="probe.raw_history.figure", filename="raw_history_overlay",
            x=np.asarray(raw_time), values=raw_display, labels=labels,
            x_label="Time [s]", y_label=f"Signal [{normalization_units}]",
            variable=variable, units=normalization_units,
            interpretation="Overlay of the selected probe histories before recipe preprocessing.",
            provenance=common | {"stage": "raw"},
        )
        _save_line_figure(
            context, artifact_id="probe.method_ready.figure", filename="method_ready_overlay",
            x=np.asarray(prepared_time), values=prepared_display, labels=labels,
            x_label="Time [s]", y_label=f"Signal [{normalization_units}]",
            variable=variable, units=normalization_units,
            interpretation="Overlay of the selected probe signals supplied to the recipe method.",
            provenance=common | {"stage": "method_ready"},
        )
        return

    common["normalization_scales"] = prepared_scales
    scratch = context.project.machine_file.compute.scratch_directory
    scratch_dir = scratch if scratch is not None else context.run_dir / "scratch"
    if not scratch_dir.is_absolute():
        scratch_dir = (context.project.root / scratch_dir).resolve()
    staged: list[tuple[Any, Any]] = []
    try:
        for prefix, array in (
            ("probe-raw-time-", np.asarray(raw_time)),
            ("probe-raw-values-", raw_display),
            ("probe-prepared-time-", np.asarray(prepared_time)),
            ("probe-prepared-values-", prepared_display),
        ):
            spec, cleanup = stage_readonly_array(array, scratch_dir, prefix=prefix)
            staged.append((spec, cleanup))
            context.add_cleanup(cleanup)
        raw_time_spec, raw_values_spec, prepared_time_spec, prepared_values_spec = (
            item[0] for item in staged
        )
        payloads = (
            {
                "x_path": raw_time_spec.path,
                "values_path": raw_values_spec.path,
                "labels": labels,
                "presentation": _figure_config(context).model_dump(mode="json"),
                "x_label": "Time [s]",
                "y_label": f"Signal [{normalization_units}]",
                "stem": str(context.figure_dir / "raw_history_overlay"),
            },
            {
                "x_path": prepared_time_spec.path,
                "values_path": prepared_values_spec.path,
                "labels": labels,
                "presentation": _figure_config(context).model_dump(mode="json"),
                "x_label": "Time [s]",
                "y_label": f"Signal [{normalization_units}]",
                "stem": str(context.figure_dir / "method_ready_overlay"),
            },
        )
        tasks = tuple(
            ParallelTask(
                index, name, payload,
                {"figure": name, "probe_indices": [int(item) for item in selected]},
            )
            for index, (name, payload) in enumerate(
                zip(("raw-history", "method-ready"), payloads)
            )
        )
        plan = _probe_parallel_plan(context, len(tasks), raw_display.nbytes + prepared_display.nbytes)
        results = context.run_parallel_stage(
            "probe-figure-render", tasks, _probe_line_figure_worker, plan,
        )
        raw_common = common | {"stage": "raw"}
        prepared_common = common | {
            "stage": "method_ready", "normalization_scales": prepared_scales,
        }
        _register_saved_line_paths(
            context, result=results[0].value, artifact_id="probe.raw_history.figure",
            variable=variable, units=normalization_units,
            coordinate_metadata={"time": "s", "probe_x": "m", "probe_y": "m"},
            interpretation="Overlay of the selected probe histories before recipe preprocessing.",
            provenance=raw_common,
        )
        _register_saved_line_paths(
            context, result=results[1].value, artifact_id="probe.method_ready.figure",
            variable=variable, units=normalization_units,
            coordinate_metadata={"time": "s", "probe_x": "m", "probe_y": "m"},
            interpretation="Overlay of the selected probe signals supplied to the recipe method.",
            provenance=prepared_common,
        )
    except BaseException:
        # Context cleanups run on workflow exit; eagerly remove any staged
        # arrays if setup fails before the context has a chance to unwind.
        presentation = _figure_config(context)
        for stem_name in ("raw_history_overlay", "method_ready_overlay"):
            for output_format in presentation.figure.formats:
                try:
                    (context.figure_dir / stem_name).with_suffix(f".{output_format}").unlink()
                except FileNotFoundError:
                    pass
        for _, cleanup in staged:
            cleanup()
        raise


def register_probe_line_overlay(
    context: WorkflowContext,
    *,
    artifact_id: str,
    filename: str,
    x: np.ndarray,
    values: np.ndarray,
    x_label: str,
    y_label: str,
    variable: str,
    units: str,
    selected: np.ndarray,
    x_m: np.ndarray,
    interpretation: str,
    provenance: dict[str, Any] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Register one line-valued result with one series per selected probe."""
    config = context.analysis.probe_plotting
    if config.mode == "panels":
        return
    labels = _selected_labels(context, selected, x_m)
    displayed, scales = _display_values(values, config.normalization)
    units = "dimensionless" if config.normalization != "none" else units
    metadata = {
        "selected_probe_indices": [int(item) for item in selected],
        "probe_labels": labels,
        "probe_x_m": [float(item) for item in x_m],
        "normalization": config.normalization,
        "normalization_scales": scales,
    }
    if provenance:
        metadata.update(provenance)
    _save_line_figure(
        context, artifact_id=artifact_id, filename=filename, x=np.asarray(x),
        values=displayed, labels=labels, x_label=x_label, y_label=y_label,
        variable=variable, units=units, interpretation=interpretation,
        provenance=metadata, log_x=log_x, log_y=log_y,
    )


def register_named_line_figure(
    context: WorkflowContext,
    *,
    artifact_id: str,
    filename: str,
    x: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    x_label: str,
    y_label: str,
    variable: str,
    units: str,
    interpretation: str,
    provenance: dict[str, Any] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Register a line figure whose series are named results, not probes."""
    metadata = {"series_labels": labels}
    if provenance:
        metadata.update(provenance)
    _save_line_figure(
        context, artifact_id=artifact_id, filename=filename, x=np.asarray(x),
        values=np.asarray(values), labels=labels, x_label=x_label, y_label=y_label,
        variable=variable, units=units, interpretation=interpretation,
        provenance=metadata, log_x=log_x, log_y=log_y,
    )


def register_probe_stft_figure(
    context: WorkflowContext,
    *,
    artifact_id: str,
    filename: str,
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
    coefficients: np.ndarray,
    selected: np.ndarray,
    x_m: np.ndarray,
    variable: str,
    units: str,
    interpretation: str,
    provenance: dict[str, Any] | None = None,
    time_half_width_s: float | None = None,
) -> None:
    """Register one shared-scale STFT panel for each selected probe."""
    if context.analysis.probe_plotting.mode == "panels":
        return
    if coefficients.ndim != 3 or coefficients.shape[2] != len(selected):
        raise ValueError("per-probe STFT must have frequency, time, and probe dimensions")
    presentation = _figure_config(context)
    labels = _selected_labels(context, selected, x_m)
    magnitude = np.abs(coefficients)
    scale = max(float(np.nanmax(magnitude)), 1.0e-300)
    power_db = 20.0 * np.log10(np.maximum(magnitude, 1.0e-300) / scale)
    rows = len(selected)
    figure, axes = plt.subplots(
        rows, 1, squeeze=False,
        figsize=(presentation.figure.width_in, max(4.5, 2.8 * rows)),
        constrained_layout=True,
    )
    for row, label in enumerate(labels):
        axis = axes[row, 0]
        if len(time_s) < 2:
            half_width = 0.5 if time_half_width_s is None else max(time_half_width_s, 1.0e-15)
            image = axis.imshow(
                power_db[:, :, row], origin="lower", aspect="auto",
                extent=(float(time_s[0] - half_width), float(time_s[0] + half_width),
                        float(frequency_hz[0]), float(frequency_hz[-1] or 1.0)),
                vmin=-60.0, vmax=0.0,
            )
        else:
            image = axis.pcolormesh(
                time_s, frequency_hz, power_db[:, :, row], shading="auto",
                vmin=-60.0, vmax=0.0,
            )
        axis.set_ylabel("Frequency [Hz]")
        axis.set_title(label)
        axis.grid(False)
    axes[-1, 0].set_xlabel("Time [s]")
    figure.colorbar(image, ax=axes[:, 0].tolist(), label="Relative STFT magnitude [dB]")
    metadata = {
        "selected_probe_indices": [int(item) for item in selected],
        "probe_labels": labels,
        "probe_x_m": [float(item) for item in x_m],
        "shared_color_scale_db": [-60.0, 0.0],
    }
    if provenance:
        metadata.update(provenance)
    _register_figure_variants(
        context, figure=figure, artifact_id=artifact_id,
        stem=context.figure_dir / filename,
        variable=variable, units=units, interpretation=interpretation,
        provenance=metadata,
    )
