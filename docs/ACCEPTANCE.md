# Release acceptance record

This record covers `refactor/postprocessor-usability`, based on reviewed commit
`c37fabb0e66f1ea36238741a6184c23ed3562945` and the current working tree. The
record is maintained with the source checkout so each claim can be reproduced
against an explicit commit, environment, and test command.

## Verification performed

- Reviewed base: `c37fabb0e66f1ea36238741a6184c23ed3562945`.
- Implementation commit: not recorded because this acceptance record is being
  prepared in the uncommitted working tree.
- Test environment: local `.venv`, CPython 3.14.7 on macOS arm64; CI remains
  authoritative for the supported Python 3.11 and 3.12 environments.
- `./.venv/bin/python -m pytest -q tests test_engineering.py`: **276 passed,
  1 skipped, 58 subtests passed** in 103.06 seconds. The optional standalone
  `compressible_similarity` module remains the only expected skip.
- `./.venv/bin/python -m pytest -q tests/test_category1.py`: passed, including
  packet Student-t uncertainty, unresolved reciprocal intervals, payload-aware
  identity manifests, contract migration rejection, and task timeout coverage.
- `./.venv/bin/python -m compileall -q pelecpost tests pp_functions_database.py`:
  passed.
- `./.venv/bin/ruff check --select F401 pelecpost tests`: passed after removing
  only confirmed unused imports. Broader Ruff modernization remains outside this
  release.
- `./.venv/bin/mypy pelecpost`: the repository has pre-existing typing errors
  under the local mypy/numpy combination; CI pins and runs the supported
  Python 3.11/3.12 job as the typing gate.
- `git diff --check`: passed.
- `./.venv/bin/python -m build --wheel --no-isolation --outdir /tmp/flow-viz-dist`:
  passed. A clean environment outside the checkout installed the wheel and
  passed `pelec-post --help`, `init`, `validate`, and `plan`. The isolated build
  variant could not download build dependencies in the restricted local network.
- The repository-local `tests` package is explicit, so test collection does not
  depend on whether another installed distribution provides a top-level package
  named `tests`.

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
- Flow-overview contours and profiles have strict YAML presentation contracts,
  external collision-tested time/colorbar layouts, stable multi-format figures,
  and registered CSV line data. Surface diagnostics add selected x/arc-length
  normal profiles without changing full-surface wall fitting or force integration.
- General 2-D EB geometry and loads carry the `validated_2d_eb_v1` designation;
  reviewed flat-plate products retain their certified designation. Manufactured
  pressure, shear, heat-flux, wedge, orientation, and resolution cases pass.
- Three-dimensional metadata can be inspected, but every numerical recipe blocks
  at preflight with the documented extension interfaces rather than producing an
  unvalidated result.
- Legacy JSON and workflow flags are rejected with migration guidance. Obsolete
  scripts/configuration examples are archived, while the root launchers remain
  thin HPC-compatible forwarding interfaces.

## Category 1 scope and evidence links

This release supports PeleC, two-dimensional numerical recipes. It does not
claim support for historical MFC scripts; those are compatibility context only
and are not the current PeleC API. Internal validation and certification labels
must point to their supporting tests: runtime containment and finalization are
covered by `tests/test_parallel.py` and `tests/test_runtime.py`, typed product
comparison by `tests/test_unified_probe_comparison.py` and
`tests/test_category1.py`, and engineering acceptance by `test_engineering.py`.
The remaining limitations and uncompleted review IDs are R05, R07, R09–R14,
and R16–R17.

## Deliberate limits

No 3-D numerical algorithm is implemented. General EB results are validated but
not externally certified. Probe-derived modes, GIP screening, wave diagnostics,
bicoherence, and force–probe linkage are measurement evidence and do not establish
LST/PSE attribution or causality. Pulse-transfer products are response per
source-power (W/m); the physical continuous-time source transform is reported
separately in J/m.
