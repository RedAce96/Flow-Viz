import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pelecpost.io.thermal_source import (
    build_source_audit,
    canonicalize_segments,
    discover_source_segments,
    rebin_history,
)


COLUMNS = [
    "coarse_step", "level", "level_step", "subcycle_iteration",
    "time_start_s", "time_end_s", "dt_s", "power_start", "power_end",
    "deposited_energy", "moment_x", "moment_y", "moment_z", "moment_xx",
    "moment_yy", "moment_zz", "moment_xy", "moment_xz", "moment_yz",
]


def _metadata(index: int = 0) -> dict:
    return {
        "schema": "pelec.thermal-source-history", "version": 1,
        "segment_index": index, "dimensions": 2, "coordinate_system": "cartesian",
        "units": {}, "source_shape": "gaussian", "deposition_model": "fixed_total_volume_weighted",
        "requested_pulse_energy": 1.0, "source_scale": 1.0, "normalization_integrals": {},
        "frequency_hz": 1.0, "period_s": 1.0, "fwhm_s": 1.0, "sigma_s": 1.0,
        "cutoff_sigma": 4.0, "start_time_s": 0.0, "laser_duration_s": 1.0,
        "source_center": {}, "axis": {}, "gaussian_radius": 1.0, "wang_length": 0.0,
        "wang_alpha": 1.0, "wang_beta": 1.0, "wang_R1": 0.0, "wang_R2": 0.0,
        "domain_bounds": {}, "eb_present": False,
        "source_implementation_sha256": "a" * 64,
        "measurement_implementation_sha256": "b" * 64,
    }


def _write(path: Path, rows: list[list[float | int]], index: int = 0) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# " + json.dumps(_metadata(index)) + "\n")
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def test_exact_rebin_and_audit(tmp_path: Path) -> None:
    _write(tmp_path / "thermal-source.segment0000.csv", [
        [1, 0, 1, 0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    history = canonicalize_segments(discover_source_segments(tmp_path))
    rebinned = rebin_history(history, np.array([0.25, 0.75]))
    assert np.allclose(rebinned["power_w_m"], [0.5, 1.5])
    audit = build_source_audit(
        history, rebinned, requested_energy_j_m=1.0, pulse_fwhm_s=1.0,
        cutoff_sigma=4.0, maximum_relative_error=0.01, start_time_s=0.0,
    )
    assert audit["status"] == "supported"
    assert audit["spatial_centroid_m"]["x"] == pytest.approx(0.5)


def test_incomplete_fine_group_is_discarded(tmp_path: Path) -> None:
    _write(tmp_path / "thermal-source.segment0000.csv", [
        [1, 1, 1, 0, 0.0, 0.5, 0.5, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    history = canonicalize_segments(discover_source_segments(tmp_path))
    assert history.rows == ()
    assert history.discarded_incomplete_rows == 1


def test_newer_segment_supersedes_old_tail(tmp_path: Path) -> None:
    _write(tmp_path / "thermal-source.segment0000.csv", [
        [1, 0, 1, 0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2, 0, 2, 0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    _write(tmp_path / "thermal-source.segment0001.csv", [
        [2, 0, 2, 0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], index=1)
    history = canonicalize_segments(discover_source_segments(tmp_path))
    assert [row["deposited_energy"] for row in history.rows] == [1.0, 2.0]
