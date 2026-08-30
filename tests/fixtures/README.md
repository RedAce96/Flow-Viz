# Tracked scientific fixtures

`pelec_2d_plt00000/` is a complete, single-level 2-D AMReX/BoxLib plotfile.
It contains one 8 × 6 cell-centered FAB with PeleC-style CGS density,
velocity, pressure, temperature, and volume-fraction fields. Tests load its
actual `Header`, `Cell_H`, and binary `Cell_D_00000` through yt before calling
the public SI adapter; it is not a stream-dataset substitute or mock.

Rebuild it deterministically from the repository root with:

```bash
python tests/fixtures/generate_amrex_fixture.py
```

The generator records the array formulas and AMReX layout in reviewable code.
