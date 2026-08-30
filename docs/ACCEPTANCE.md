# Release acceptance record

This record covers `refactor/postprocessor-usability`, based on reviewed commit
`cf286e0`. Development occurred only in the separate
`/lustre/isaac24/scratch/sbrollia/PeleC-Python-usability` worktree. The original
worktree remains at `cf286e0` with its pre-existing local files, and
`git diff --name-only cf286e0 -- '*slurm*' '*SLURM*' '*.sbatch'` is empty.

## Verification performed

- `python -m pytest -q tests`: 119 passed.
- `python -m pytest -q test_engineering.py`: 102 passed, 1 skipped because the
  optional standalone `compressible_similarity` module is not branch input.
- `python -m mypy pelecpost`: no issues in 37 package modules.
- `python -m compileall -q pelecpost tools benchmarks`: passed.
- `git diff --check`: passed.
- Nonlegacy Pyflakes inspection: no findings. The preserved legacy driver remains
  excluded from new-package linting; CI runs Ruff and mypy on Python 3.11 and 3.12.
- A no-isolation wheel was built, installed without dependencies into a fresh
  system-site-packages virtual environment, and its installed `pelec-post` recipe
  command was executed outside the source tree.

## Contract audit

- Strict, immutable Pydantic models own three nonoverlapping YAML files; project
  schemas and the complete configuration reference are generated from those
  models.
- All 11 public recipes are registry-backed, DAG-planned, dimension-aware, and
  have declared artifacts. Probe-only graphs have no plotfile dependency.
- Inspection is metadata/bounded-read only and inventories plotfiles, compact or
  raw probes, baselines, and registered comparison products.
- Preflight reports temporal/spatial resolution, selections, assumptions,
  geometry coverage, resource estimates, expected artifacts, and severity-coded
  findings. Incompatible comparison products are rejected before execution.
- Runs are timestamp-isolated and preserve atomic manifests, resolved YAML,
  preflight, logs, stable artifact metadata, conservative evidence, partial
  results, and regenerable relative-link HTML reports.
- Existing 2-D numerical contracts pass the engineering suite. Freestream
  references are explicit or region-derived with provenance; stationary,
  single-pulse, transient, nonlinear, modal, and force estimators remain distinct.
- General 2-D EB geometry and loads carry the `validated_2d_eb_v1` designation;
  reviewed flat-plate products retain their certified designation. Manufactured
  pressure, shear, heat-flux, wedge, orientation, and resolution cases pass.
- Three-dimensional metadata can be inspected, but every numerical recipe blocks
  at preflight with the documented extension interfaces rather than producing an
  unvalidated result.
- Legacy JSON and workflow flags are rejected with migration guidance. Obsolete
  scripts/configuration examples are archived, while the root launchers remain
  thin HPC-compatible forwarding interfaces.

## Deliberate limits

No 3-D numerical algorithm is implemented. General EB results are validated but
not externally certified. Probe-derived modes, GIP screening, wave diagnostics,
bicoherence, and force–probe linkage are measurement evidence and do not establish
LST/PSE attribution or causality.
