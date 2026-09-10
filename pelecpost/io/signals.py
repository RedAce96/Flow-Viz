"""Bounded compact-HDF5 and probe-v2 signal workspaces.

The compact HDF5 archive remains the authoritative source.  Selected matrices
that do not fit the configured resident budget are staged into a temporary
``numpy.memmap`` in bounded row blocks.  The workspace owns and removes that
temporary file; no spill file is part of a scientific run or artifact set.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from pp_probe_store import ProbeV2Collection


DEFAULT_MAX_RESIDENT_BYTES = 256 * 1024**2
DEFAULT_READ_BLOCK_BYTES = 16 * 1024**2


@dataclass
class ProbeSignalWorkspace:
    """One SI-valued probe matrix with explicit storage and cleanup metadata."""

    time_s: np.ndarray
    x_m: np.ndarray
    values: np.ndarray
    selected_probe_indices: np.ndarray
    storage: Literal["memory", "disk"]
    source_matrix_bytes: int
    resident_bound_bytes: int
    read_block_rows: int
    spill_path: Path | None = None

    def close(self) -> None:
        """Flush and unlink a disk workspace; closing twice is harmless."""
        if self.storage == "disk" and self.spill_path is None:
            return
        if isinstance(self.values, np.memmap):
            self.values.flush()
            mmap = getattr(self.values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if self.spill_path is not None:
            try:
                self.spill_path.unlink()
            except FileNotFoundError:
                pass
            self.spill_path = None


def _resident_budget(memory_limit_gb: float) -> int:
    # Reserve most configured memory for the estimator and its output.  The
    # selected source signal uses at most one eighth, capped at 256 MiB.
    configured = max(1, int(memory_limit_gb * 1024**3 / 8.0))
    return min(DEFAULT_MAX_RESIDENT_BYTES, configured)


def open_compact_signal_workspace(
    path: Path,
    field: str,
    probe_indices: np.ndarray,
    *,
    si_factor: float,
    memory_limit_gb: float,
    spill_threshold_bytes: int | None = None,
    read_block_bytes: int = DEFAULT_READ_BLOCK_BYTES,
    scratch_directory: Path | None = None,
) -> ProbeSignalWorkspace:
    """Load selected probes with a bounded resident allocation.

    ``spill_threshold_bytes`` is exposed for deterministic allocation tests.
    Production callers derive the threshold from ``memory_limit_gb``.
    """
    selected = np.asarray(probe_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("at least one probe index is required")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("probe indices must be unique")
    if read_block_bytes <= 0:
        raise ValueError("read_block_bytes must be positive")

    path = Path(path)
    spill_path: Path | None = None
    values: np.ndarray | None = None
    try:
        with h5py.File(path, "r") as archive:
            dataset = archive[f"fields/{field}"]
            if dataset.ndim != 2:
                raise ValueError(f"fields/{field} must be a two-dimensional sample/probe matrix")
            sample_count, probe_count = dataset.shape
            if np.any(selected < 0) or np.any(selected >= probe_count):
                raise ValueError(f"probe indices must lie in [0, {probe_count})")
            time_s = np.asarray(archive["time"], dtype=np.float64)
            if len(time_s) != sample_count:
                raise ValueError("time and probe-field sample counts differ")
            x_all_cm = np.asarray(archive["probes/requested_x_cm"], dtype=np.float64)
            if len(x_all_cm) != probe_count:
                raise ValueError("probe-coordinate and field probe counts differ")
            x_m = x_all_cm[selected] * 0.01

            source_bytes = int(sample_count * selected.size * np.dtype(np.float64).itemsize)
            resident_bound = (
                _resident_budget(memory_limit_gb)
                if spill_threshold_bytes is None
                else int(spill_threshold_bytes)
            )
            if resident_bound <= 0:
                raise ValueError("spill_threshold_bytes must be positive")
            storage: Literal["memory", "disk"] = (
                "memory" if source_bytes <= resident_bound else "disk"
            )
            shape = (sample_count, selected.size)
            if storage == "memory":
                values = np.empty(shape, dtype=np.float64)
            else:
                scratch = Path(scratch_directory) if scratch_directory is not None else None
                if scratch is not None:
                    scratch.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix="pelec-post-signal-", suffix=".mmap",
                    dir=str(scratch) if scratch is not None else None,
                )
                os.close(descriptor)
                spill_path = Path(temporary_name)
                values = np.memmap(spill_path, mode="w+", dtype=np.float64, shape=shape)

            # h5py requires monotonically increasing fancy indices.  Restore
            # the configured order after each bounded row read.
            order = np.argsort(selected)
            sorted_indices = selected[order]
            restore = np.argsort(order)
            bytes_per_row = max(1, selected.size * np.dtype(np.float64).itemsize)
            block_rows = max(1, read_block_bytes // bytes_per_row)
            for start in range(0, sample_count, block_rows):
                stop = min(sample_count, start + block_rows)
                block = np.asarray(dataset[start:stop, sorted_indices], dtype=np.float64)
                np.multiply(block, si_factor, out=block)
                values[start:stop] = block[:, restore]
            if isinstance(values, np.memmap):
                values.flush()
        assert values is not None
        return ProbeSignalWorkspace(
            time_s=time_s,
            x_m=x_m,
            values=values,
            selected_probe_indices=selected,
            storage=storage,
            source_matrix_bytes=source_bytes,
            resident_bound_bytes=resident_bound,
            read_block_rows=block_rows,
            spill_path=spill_path,
        )
    except BaseException:
        if isinstance(values, np.memmap):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if spill_path is not None:
            try:
                spill_path.unlink()
            except FileNotFoundError:
                pass
        raise


def open_probe_v2_signal_workspace(
    paths: tuple[str, ...],
    field: str,
    probe_indices: np.ndarray,
    *,
    si_factor: float,
    memory_limit_gb: float,
    spill_threshold_bytes: int | None = None,
    read_block_bytes: int = DEFAULT_READ_BLOCK_BYTES,
    scratch_directory: Path | None = None,
) -> ProbeSignalWorkspace:
    """Stage selected raw probe-v2 data with the compact-reader memory contract."""
    selected = np.asarray(probe_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("at least one probe index is required")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("probe indices must be unique")
    if read_block_bytes <= 0:
        raise ValueError("read_block_bytes must be positive")

    spill_path: Path | None = None
    values: np.ndarray | None = None
    try:
        with ProbeV2Collection(paths) as collection:
            if field not in collection.field_names:
                raise KeyError(f"Unknown probe field {field!r}; have {collection.field_names}")
            if np.any(selected < 0) or np.any(selected >= collection.n_probes):
                raise ValueError(f"probe indices must lie in [0, {collection.n_probes})")
            sample_count = len(collection.time)
            time_s = np.asarray(collection.time, dtype=np.float64)
            x_m = np.asarray(collection.header["requested_x"], dtype=np.float64)[selected] * 0.01
            source_bytes = int(sample_count * selected.size * np.dtype(np.float64).itemsize)
            resident_bound = (
                _resident_budget(memory_limit_gb)
                if spill_threshold_bytes is None
                else int(spill_threshold_bytes)
            )
            if resident_bound <= 0:
                raise ValueError("spill_threshold_bytes must be positive")
            storage: Literal["memory", "disk"] = (
                "memory" if source_bytes <= resident_bound else "disk"
            )
            shape = (sample_count, selected.size)
            if storage == "memory":
                values = np.empty(shape, dtype=np.float64)
            else:
                scratch = Path(scratch_directory) if scratch_directory is not None else None
                if scratch is not None:
                    scratch.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix="pelec-post-signal-", suffix=".mmap",
                    dir=str(scratch) if scratch is not None else None,
                )
                os.close(descriptor)
                spill_path = Path(temporary_name)
                values = np.memmap(spill_path, mode="w+", dtype=np.float64, shape=shape)
            bytes_per_row = max(1, selected.size * np.dtype(np.float64).itemsize)
            block_rows = max(1, read_block_bytes // bytes_per_row)
            for start in range(0, sample_count, block_rows):
                stop = min(sample_count, start + block_rows)
                block = collection.read_field(field, start, stop, probes=selected)
                np.multiply(block, si_factor, out=block)
                values[start:stop] = block
            if isinstance(values, np.memmap):
                values.flush()
        assert values is not None
        return ProbeSignalWorkspace(
            time_s=time_s,
            x_m=x_m,
            values=values,
            selected_probe_indices=selected,
            storage=storage,
            source_matrix_bytes=source_bytes,
            resident_bound_bytes=resident_bound,
            read_block_rows=block_rows,
            spill_path=spill_path,
        )
    except BaseException:
        if isinstance(values, np.memmap):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if spill_path is not None:
            try:
                spill_path.unlink()
            except FileNotFoundError:
                pass
        raise


def open_probe_signal_workspace(
    source: Path | tuple[str, ...],
    field: str,
    probe_indices: np.ndarray,
    *,
    si_factor: float,
    memory_limit_gb: float,
    spill_threshold_bytes: int | None = None,
    read_block_bytes: int = DEFAULT_READ_BLOCK_BYTES,
    scratch_directory: Path | None = None,
) -> ProbeSignalWorkspace:
    """Open either supported probe format under one bounded workspace API."""
    if isinstance(source, Path):
        return open_compact_signal_workspace(
            source, field, probe_indices, si_factor=si_factor,
            memory_limit_gb=memory_limit_gb,
            spill_threshold_bytes=spill_threshold_bytes,
            read_block_bytes=read_block_bytes,
            scratch_directory=scratch_directory,
        )
    return open_probe_v2_signal_workspace(
        tuple(source), field, probe_indices, si_factor=si_factor,
        memory_limit_gb=memory_limit_gb,
        spill_threshold_bytes=spill_threshold_bytes,
        read_block_bytes=read_block_bytes,
        scratch_directory=scratch_directory,
    )


# Compatibility names for low-level callers; recipe code uses the generic API.
CompactSignalWorkspace = ProbeSignalWorkspace
