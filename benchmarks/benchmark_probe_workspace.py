#!/usr/bin/env python3
"""Reference benchmark for bounded compact-probe staging; not a CI pass/fail test."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from pelecpost.io.signals import open_compact_signal_workspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--selected-probes", type=int, default=96)
    parser.add_argument("--force-spill", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.selected_probes <= arguments.probes:
        parser.error("selected-probes must lie in [1, probes]")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_path = root / "benchmark-probes.h5"
        with h5py.File(archive_path, "w") as archive:
            archive.create_dataset("time", data=np.arange(arguments.samples) * 1.0e-6)
            probes = archive.create_group("probes")
            probes.create_dataset("requested_x_cm", data=np.arange(arguments.probes))
            fields = archive.create_group("fields")
            dataset = fields.create_dataset(
                "p", shape=(arguments.samples, arguments.probes), dtype="f8",
                chunks=(min(1024, arguments.samples), arguments.probes),
            )
            row = np.arange(arguments.probes, dtype=float)[None, :]
            for start in range(0, arguments.samples, 1024):
                stop = min(arguments.samples, start + 1024)
                time_axis = np.arange(start, stop, dtype=float)[:, None] * 1.0e-6
                dataset[start:stop] = np.sin(2.0 * np.pi * 25_000.0 * time_axis) + row
        selected = np.arange(arguments.selected_probes, dtype=int)
        started = time.perf_counter()
        workspace = open_compact_signal_workspace(
            archive_path, "p", selected, si_factor=0.1, memory_limit_gb=1.0,
            spill_threshold_bytes=(1 if arguments.force_spill else None),
            scratch_directory=root,
        )
        checksum = float(np.sum(workspace.values[::4096]))
        elapsed = time.perf_counter() - started
        result = {
            "samples": arguments.samples,
            "source_probes": arguments.probes,
            "selected_probes": arguments.selected_probes,
            "selected_matrix_mib": workspace.source_matrix_bytes / 1024**2,
            "storage": workspace.storage,
            "read_block_rows": workspace.read_block_rows,
            "elapsed_s": elapsed,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "sampled_checksum": checksum,
        }
        workspace.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
