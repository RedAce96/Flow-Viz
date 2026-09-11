"""Bounded probe POD/SPOD/DMD screening using reviewed kernels."""

from __future__ import annotations

from typing import cast

import numpy as np

import pp_modal_database as reviewed_modal
from pelecpost.config.models import ModalScreeningAnalysis
from pelecpost.runtime.context import WorkflowContext
from pelecpost.runtime.parallel import ParallelTask, plan_parallel_stage, stage_readonly_array

from .executors import executor
from .probe_plotting import register_named_line_figure, register_probe_trace_figures
from .spectral import load_probe_signal, prepare_probe_time_grid


def _spod_frequency_batch_worker(payload: dict) -> dict:
    blocks = np.load(payload["blocks_path"], mmap_mode="r")
    return reviewed_modal.compute_spod_frequency_batch(
        blocks, np.asarray(payload["weights"], dtype=float), payload["mode_count"],
        payload["start_index"], payload["stop_index"],
    )


def _register_npz(context: WorkflowContext, name: str, arrays: dict, interpretation: str) -> None:
    analysis = cast(ModalScreeningAnalysis, context.analysis)
    path = context.data_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    context.register(
        artifact_id=f"modal.{name}", path=path, kind="array",
        variable=analysis.variable.value, units="variable-dependent",
        coordinate_metadata={"probe_x": "m", "frequency": "Hz", "time": "s"},
        interpretation=interpretation,
    )


@executor("modal_screening")
def run_modal_screening(context: WorkflowContext) -> None:
    analysis = cast(ModalScreeningAnalysis, context.analysis)
    variable, unit, raw_time, x_m, raw_values, selected = load_probe_signal(context)
    time = raw_time
    values = raw_values
    time, values, _dt, _resampled, cleanup = prepare_probe_time_grid(context, time, values)
    if cleanup is not None:
        context.add_cleanup(cleanup)
    register_probe_trace_figures(
        context, raw_time=raw_time, raw_values=raw_values,
        prepared_time=time, prepared_values=values, x_m=x_m, selected=selected,
        variable=variable, units=unit,
        preprocessing={"time_grid_policy": analysis.time_grid_policy,
                       "resampled": _resampled},
    )
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
    pod_coordinates = np.asarray(pod["coordinates"]).reshape(-1)
    register_named_line_figure(
        context, artifact_id="modal.pod.figure", filename="pod_modes",
        x=pod_coordinates, values=np.asarray(pod["modes"]),
        labels=[f"POD mode {index + 1}" for index in range(mode_count)],
        x_label="Probe coordinate [m]", y_label=f"Mode amplitude [{unit}]",
        variable=variable, units=unit,
        interpretation="Weighted POD mode profiles over the selected probe coordinates.",
        provenance={"mode_count": mode_count},
    )
    _register_npz(context, "pod", {
        "modes": pod["modes"], "temporal_coefficients": pod["temporal_coefficients"],
        "singular_values": pod["singular_values"], "energy_fraction": pod["energy_fraction"],
        "mean": pod["mean"], "coordinates_m": pod["coordinates"],
    }, "Weighted descriptive POD of the probe measure; not a stability eigenmode.")
    segment = min(int(analysis.spod_segment_samples), len(time) // 2)
    segment = max(8, segment)
    frequency_stride = max(1, segment // 256)
    if int(context.project.machine_file.compute.workers) <= 1:
        spod = reviewed_modal.compute_spod(
            dataset, nperseg=segment, noverlap=segment // 2,
            n_modes=min(mode_count, 3), frequency_stride=frequency_stride,
        )
    else:
        spod_problem = reviewed_modal.prepare_spod_blocks(
            dataset, nperseg=segment, noverlap=segment // 2,
            n_modes=min(mode_count, 3), frequency_stride=frequency_stride,
        )
        scratch = context.project.machine_file.compute.scratch_directory
        if scratch is None:
            scratch = context.run_dir / "scratch" / context.analysis.id
        elif not scratch.is_absolute():
            scratch = (context.project.root / scratch).resolve()
        blocks_spec, blocks_cleanup = stage_readonly_array(
            spod_problem["blocks"], scratch, prefix="modal-spod-blocks-",
        )
        context.add_cleanup(blocks_cleanup)
        frequency_count = len(spod_problem["frequency_indices"])
        worker_count = max(1, int(context.project.machine_file.compute.workers))
        batch_count = min(frequency_count, max(1, worker_count * 4))
        boundaries = np.linspace(0, frequency_count, batch_count + 1, dtype=int)
        tasks = []
        for index in range(batch_count):
            start_index, stop_index = int(boundaries[index]), int(boundaries[index + 1])
            if stop_index <= start_index:
                continue
            tasks.append(ParallelTask(
                index, f"spod-frequencies-{start_index:04d}-{stop_index:04d}", {
                    "blocks_path": blocks_spec.path,
                    "weights": np.asarray(spod_problem["weights"]).tolist(),
                    "mode_count": int(spod_problem["mode_count"]),
                    "start_index": start_index, "stop_index": stop_index,
                }, {"frequency_start": start_index, "frequency_stop": stop_index},
            ))
        parent_gb = float((context.resource_metadata or {}).get("parent_resident_gb", 0.25))
        per_worker_gb = max(0.05, 0.15 + spod_problem["blocks"].nbytes / 1024**3)
        stage_plan = plan_parallel_stage(
            "modal-spod-frequency-batches", requested_workers=worker_count,
            task_count=len(tasks), memory_limit_gb=float(context.project.machine_file.compute.memory_limit_gb),
            parent_resident_gb=parent_gb, per_worker_peak_gb=per_worker_gb,
        )
        batch_results = context.run_parallel_stage(
            "modal-spod-frequency-batches", tasks, _spod_frequency_batch_worker, stage_plan,
        )
        eigenvalues = np.zeros((frequency_count, int(spod_problem["mode_count"])), dtype=float)
        modes = np.zeros((frequency_count, int(spod_problem["mode_count"]), values.shape[1]), dtype=complex)
        for result in batch_results:
            start_index = int(result.value["start_index"])
            stop_index = start_index + len(result.value["eigenvalues"])
            eigenvalues[start_index:stop_index] = result.value["eigenvalues"]
            modes[start_index:stop_index] = result.value["modes"]
        spod = {
            **{key: spod_problem[key] for key in (
                "frequency_hz", "frequency_indices", "n_blocks", "nperseg",
                "noverlap", "coordinates", "variable",
            )},
            "eigenvalues": eigenvalues, "modes": modes,
        }
    register_named_line_figure(
        context, artifact_id="modal.spod.figure", filename="spod_eigenvalues",
        x=np.asarray(spod["frequency_hz"]), values=np.asarray(spod["eigenvalues"]),
        labels=[f"SPOD mode {index + 1}" for index in range(spod["eigenvalues"].shape[1])],
        x_label="Frequency [Hz]", y_label="SPOD eigenvalue",
        variable=variable, units="variable-dependent",
        interpretation="SPOD eigenvalue spectra for the selected probe aperture.",
        provenance={"mode_count": int(spod["eigenvalues"].shape[1])},
        log_x=True, log_y=True,
    )
    _register_npz(context, "spod", {
        "frequency_hz": spod["frequency_hz"], "eigenvalues": spod["eigenvalues"],
        "modes": spod["modes"], "coordinates_m": spod["coordinates"],
        "block_count": np.array(spod["n_blocks"]),
    }, "Welch-block SPOD with explicit block count and probe quadrature weights.")
    dmd_rank = min(max(analysis.dmd_ranks), values.shape[1], values.shape[0] - 1)
    dmd = reviewed_modal.compute_dmd(dataset, n_modes=dmd_rank)
    register_named_line_figure(
        context, artifact_id="modal.dmd.figure", filename="dmd_amplitudes",
        x=np.asarray(dmd["frequency_hz"]),
        values=np.abs(np.asarray(dmd["amplitudes"])).reshape(-1, 1),
        labels=["DMD amplitude"], x_label="Frequency [Hz]", y_label="Amplitude",
        variable=variable, units="variable-dependent",
        interpretation="DMD modal amplitudes indexed by descriptive frequency.",
        provenance={"rank": dmd_rank}, log_x=True, log_y=True,
    )
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
