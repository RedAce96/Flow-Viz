"""Rebuild the tracked tiny 2-D AMReX plotfile used by yt adapter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FIELDS = (
    "density",
    "x_velocity",
    "y_velocity",
    "pressure",
    "temperature",
    "volume_fraction",
)
NX = 8
NY = 6


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    level = destination / "Level_0"
    level.mkdir(exist_ok=True)
    root_header = [
        "HyperCLaw-V1.1",
        str(len(FIELDS)),
        *FIELDS,
        "2",
        "1.25e-6",
        "0",
        "0.0 0.0",
        f"{NX}.0 {NY}.0",
        "2",
        f"((0,0) ({NX - 1},{NY - 1}) (0,0))",
        "0",
        "1.0 1.0",
        "0",
        "0",
        "0 1 1.25e-6",
        "0",
        f"0.0 {NX}.0",
        f"0.0 {NY}.0",
        "Level_0/Cell",
    ]
    (destination / "Header").write_text("\n".join(root_header) + "\n", encoding="ascii")
    (destination / "job_info").write_text(
        "Synthetic PeleC AMReX integration fixture\namrex.plot_file = true\n",
        encoding="ascii",
    )
    cell_header = [
        "1",
        "1",
        str(len(FIELDS)),
        "0",
        "(1 0",
        f"((0,0) ({NX - 1},{NY - 1}) (0,0))",
        ")",
        "1",
        "FabOnDisk: Cell_D_00000 0",
    ]
    (level / "Cell_H").write_text("\n".join(cell_header) + "\n", encoding="ascii")

    i, j = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    arrays = (
        1.0e-3 + 1.0e-5 * i + 2.0e-5 * j,
        1000.0 + 10.0 * i,
        -50.0 + 5.0 * j,
        1.0e6 + 1000.0 * i + 2000.0 * j,
        300.0 + i + 2.0 * j,
        ((i - 3.5) ** 2 + (j - 2.5) ** 2 >= 2.0).astype(float),
    )
    fab_header = (
        "FAB ((8, (64 11 52 0 1 12 0 1023)),"
        f"(8, (8 7 6 5 4 3 2 1)))((0,0) ({NX - 1},{NY - 1}) (0,0)) {len(FIELDS)}\n"
    ).encode("ascii")
    with (level / "Cell_D_00000").open("wb") as stream:
        stream.write(fab_header)
        for values in arrays:
            stream.write(np.asarray(values, dtype="<f8").ravel(order="F").tobytes())


if __name__ == "__main__":
    generate(Path(__file__).with_name("pelec_2d_plt00000"))
