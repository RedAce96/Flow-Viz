# PeleC post-processing refactor guide

This guide documents the current `PeleC-Python-usability` refactor: how to
install it, create or move a project to a server, configure analyses, inspect
inputs, validate configuration, perform preflight, run workflows, and review
the outputs.

The refactor is a clean break from the former global-switch and JSON-overlay
interface. A current project is a directory containing three strict YAML files:

```text
PROJECT/
├── case.yaml             # portable case physics and geometry
├── analyses.yaml         # selected recipes and scientific choices
├── machine.yaml          # server-local inputs, outputs, and resources
├── machine.example.yaml  # template for another server
├── project.schema.json    # generated editor/schema information
└── runs/                  # default run output directory
```

The installed command is `pelec-post`. From a source checkout, the equivalent
launcher is `python pelec_post.py ...`.

## 1. Basic lifecycle

The normal lifecycle is:

```text
install source → init project → edit YAML → inspect → validate → plan → run
```

The short command sequence is:

```bash
python -m pip install -e .
pelec-post init my-project

# Edit my-project/case.yaml, machine.yaml, and analyses.yaml.
pelec-post inspect my-project
pelec-post validate my-project
pelec-post plan my-project
pelec-post run my-project --name first-pass
```

`run` performs validation and preflight again before computation. Running
`validate` and `plan` separately is recommended because it exposes problems
before a long job is started.

## 2. Installation and server setup

### 2.1 Clone beside the existing post-processor

If the existing server checkout is `/path/to/projects/PeleC-Python`, clone the
refactor from its parent directory, not from inside it:

```bash
cd /path/to/projects
git clone --branch refactor/postprocessor-usability \
  <repository-url> PeleC-Python-usability
cd PeleC-Python-usability
```

The result is:

```text
/path/to/projects/
├── PeleC-Python/
└── PeleC-Python-usability/
```

If the branch exists only locally on the development machine, publish it
first:

```bash
git push -u origin refactor/postprocessor-usability
```

On the server, verify the checkout:

```bash
git status --short --branch
git log -1 --oneline
```

### 2.2 Create an isolated Python environment

The package requires Python 3.11 or newer. A virtual environment keeps this
refactor independent from the existing post-processor environment:

```bash
cd /path/to/projects/PeleC-Python-usability
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and verification tools:

```bash
python -m pip install -e '.[dev]'
```

Confirm that the command comes from this checkout:

```bash
which python
which pelec-post
pelec-post --help
```

The `-e` install imports the working tree directly, so a normal source pull
does not require reinstalling. Activate the correct cluster module/Conda
environment before installing and running.

The plotting code is intended for headless/HPC use. Do not accidentally mix
the old repository's Python environment with the new checkout; check the
output of `which python` and `which pelec-post` before submitting a job.

## 3. Creating a project

Create a new project in an empty directory:

```bash
pelec-post init /path/to/my-project
```

`init` writes `case.yaml`, `analyses.yaml`, `machine.yaml`,
`machine.example.yaml`, `project.schema.json`, and an empty `runs/` directory.
It refuses to initialize a non-empty directory to prevent accidental
overwrites.

If `machine.yaml` is missing later:

```bash
cp machine.example.yaml machine.yaml
```

Complete examples are under `examples/`. They are portable starting points;
replace their machine-local input paths before use.

## 4. Configuration files

### 4.1 `case.yaml`: portable physics

This file describes the case, not the server. It contains:

- case identity, solver, dimensionality, solver units, and description;
- gas properties such as `gamma` and `gas_constant_j_kg_k`;
- a freestream reference, either explicit or region-derived;
- geometry and fluid-side orientation;
- optional canonical-to-native field aliases.

Public dimensional values use SI and unit-bearing names, such as
`velocity_m_s`, `pressure_pa`, and `temperature_k`. PeleC/AMReX source values
are converted at the I/O boundary according to `solver_units`.

Example:

```yaml
schema_version: 1
case:
  id: flat-plate-study
  solver: pelec
  dimensionality: 2
  solver_units: cgs
  description: Mach-number and boundary-layer review
gas:
  gamma: 1.4
  gas_constant_j_kg_k: 287.05
freestream:
  source: explicit
  density_kg_m3: 0.02118
  velocity_m_s: 1726.0
  pressure_pa: 760.0
  temperature_k: 125.0
geometry:
  type: flat_plate
  leading_edge_x_m: 0.0
  trailing_edge_x_m: 0.4
  wall_y_m: 0.0
  fluid_side: above
```

Supported geometry types are `flat_plate`, `wedge`, `polyline`, and
`volume_fraction`. Numerical recipes are currently 2-D; 3-D metadata can be
inspected, but numerical recipes block at preflight.

### 4.2 `machine.yaml`: server-local inputs and resources

This file is customized on each server. It contains:

- plotfile locations and named probe sources;
- optional baseline plotfile locations;
- optional named archived analysis runs;
- output root;
- workers, memory limit, FFT batch size, and optional scratch directory.

Example:

```yaml
schema_version: 1
inputs:
  plotfiles:
    source: /scratch/myuser/case/plt
    prefix: plt
outputs:
  root: runs
compute:
  workers: 1
  memory_limit_gb: 16.0
  fft_batch_size: 32
  scratch_directory: /local/scratch/myuser/pelecpost
```

Relative paths resolve relative to the project directory. Thus
`outputs.root: runs` writes below `PROJECT/runs`; an absolute path writes to
that server location.

For probes, configure named sources. Each source uses exactly one compact
archive or collection of binary probe files:

```yaml
inputs:
  probe_sets:
    asym:
      binary_files: [/scratch/myuser/probes/asym.segment*.pbin]
    gaus:
      compact_file: /scratch/myuser/probes/gaus.h5
  archived_runs:
    previous_asym: /results/asym/run
```

Compact HDF5 is preferable for repeated production analysis because it is
self-contained, verified, and read-optimized. Recipes select `probe_set_id`
and never depend on the storage format.

### 4.3 `analyses.yaml`: recipes and scientific choices

This file selects one or more enabled analysis instances. Each instance needs
a unique `id` and a `recipe`; recipe-specific parameters are placed beside
those fields.

Minimal flow-overview example:

```yaml
schema_version: 1
analyses:
  - id: overview
    recipe: flow_overview
    fields: [pressure, temperature, mach_number]
    x_limits_m: [0.0, 0.2]
    y_limits_m: [0.0, 0.02]
    streamlines: false
```

An empty `analyses` list is valid, but produces no analysis products.

## 5. Available recipes

List recipes and required inputs:

```bash
pelec-post recipes list
```

Inspect assumptions, outputs, and limitations for one recipe:

```bash
pelec-post recipes show flow_overview
```

The current recipes are:

| Recipe | Question | Input |
| --- | --- | --- |
| `flow_overview` | What does the resolved 2-D flow field look like? | plotfiles |
| `boundary_layer_reference` | How does a flat-plate boundary layer compare with a laminar reference? | plotfiles |
| `surface_diagnostics` | Is reconstructed wall geometry and near-wall sampling trustworthy? | plotfiles |
| `aerodynamic_forces` | What pressure, viscous, thermal, force, and moment loads act on the body? | plotfiles |
| `probe_spectrum` | What stationary frequency content is present? | probe_sets |
| `single_pulse_response` | What response follows a finite laser pulse? | probe_sets |
| `directional_wave` | What direction, wavelength, phase speed, and amplification are measured? | probe_sets |
| `transient_wavepacket` | How does a transient packet arrive and propagate? | probe_sets |
| `nonlinear_coupling` | Are significant quadratic frequency interactions measured? | probe_sets |
| `modal_screening` | Which coherent low-rank structures are visible? | probe_sets |
| `case_comparison` | How do two registered runs differ? | local products and/or archived_runs |

The registry is authoritative; see [`RECIPES.md`](RECIPES.md) for the full
recipe contract.

## 6. Plot presentation and contour ranges

Presentation defaults belong under top-level `presentation` in
`analyses.yaml`. An analysis may override them, and individual contour fields
may override the analysis.

For an exact fixed range, use:

```yaml
contours:
  fields:
    mach_number:
      range:
        mode: fixed
        minimum: 0.0
        maximum: 8.0
```

Fixed mode requires both limits, and `maximum` must exceed `minimum`. The
range modes are:

- `fixed`: explicit limits for every snapshot; best for cross-case and
  animation comparisons when a physical range is known.
- `per_snapshot_percentile`: computes limits independently per snapshot;
  useful for exploration, but colors do not represent fixed values over time.
- `selected_snapshots_minmax`: one shared minimum and maximum over all selected
  snapshots; consistent, but sensitive to outliers.
- `selected_snapshots_percentile`: one shared percentile range over all
  selected snapshots; often a good animation compromise.

Example shared percentile range:

```yaml
range:
  mode: selected_snapshots_percentile
  lower_percentile: 1.0
  upper_percentile: 99.0
```

Other presentation controls include:

```yaml
presentation:
  contour_defaults:
    colormap: viridis
    normalization: linear       # linear, log, or symlog
    colorbar:
      position: top             # top, bottom, left, or right
      length_fraction: 0.33
      thickness_fraction: 0.30
      include_endpoints: true
      tick_count: 4
  time_annotation:
    enabled: true
    position: above_axes_left # or top_left, top_center, bottom_right, etc.
    boxed: true
    precision: 4
  typography:
    base_size: 12
    axes_label_size: 14
    tick_label_size: 12
    colorbar_label_size: 12
    colorbar_tick_label_size: 10
```

Top/bottom colorbars are horizontal; left/right colorbars are vertical. For
a wide, shallow field, reduce `length_fraction` to reduce header space.
`include_endpoints: true` helps a short colorbar display its resolved limits.

Use `viridis` or another perceptually reliable map for scalar fields. Signed
fields such as vorticity normally need a diverging map and symmetric limits.
Log normalization requires positive values and positive limits.

The full generated configuration reference is in
[`CONFIGURATION.md`](CONFIGURATION.md), and plot-specific guidance is in
[`PRESENTATION.md`](PRESENTATION.md).

## 7. Inspecting inputs

`inspect` inventories inputs without running an analysis or allocating the
full data set:

```bash
pelec-post inspect PROJECT
pelec-post inspect PROJECT --json > input-inventory.json
```

It reports plotfile count/names, AMR levels, time range, fields, canonical
mapping, domain bounds, and solver units. For probes it reports every named
source's format, sample/probe counts, fields, coordinates, timing, restart
overlaps, missing values, mapping epochs, quality approval, and provenance. It
also reports baselines and named archived analysis runs.

Inspection is the first useful check after moving a project to a new server.

## 8. Validation

Run strict configuration and workflow-graph validation:

```bash
pelec-post validate PROJECT
```

Validation checks all three YAML files, rejects unknown keys, checks types and
cross-field rules, builds the workflow graph, and regenerates
`PROJECT/project.schema.json` for editor completion.

Typical failures include missing files, misspelled keys, missing required
fields, invalid fixed ranges, invalid percentiles, incompatible geometry or
dimensionality, duplicate analysis IDs, dependency errors, and recipe
conflicts.

## 9. Configuration wizard

After machine-local inputs are configured and inspectable:

```bash
pelec-post configure PROJECT
```

The wizard inventories inputs, presents compatible physical questions, asks for
recipe-specific values, writes `analyses.yaml`, and validates it. It is a
starting point and does not expose every advanced option, so review the YAML
afterward and run `validate` and `plan`.

## 10. Scientific and resource planning

Run preflight without computation:

```bash
pelec-post plan PROJECT
pelec-post plan PROJECT --json > plan.json
```

Planning reports workflow order and dependencies, blocker/warning/info
findings, temporal and spatial sampling, Nyquist and frequency resolution,
selected snapshots, geometry coverage, recipe assumptions, estimated memory,
declared artifacts, and output root.

Resolve every `BLOCKER` before running. Warnings require scientific judgment;
passing Nyquist alone does not guarantee useful frequency resolution, spatial
resolution, or enough independent segments for statistical inference.

## 11. Running analyses

Run all enabled recipes:

```bash
pelec-post run PROJECT
pelec-post run PROJECT --name mach-fixed-range
```

Each run creates a new UTC timestamped directory below `outputs.root`:

```text
runs/2026-09-01T123456Z_flat-plate-study_mach-fixed-range/
├── data/                 # numerical data products
├── figures/              # figures
├── logs/run.log          # full log and tracebacks
├── report/index.html     # portable report
├── artifacts.json        # product index
├── manifest.json         # workflow state and provenance
├── plan.json             # preflight record
├── resolved-case.yaml
├── resolved-analyses.yaml
└── resolved-machine.yaml
```

Runs are never reused. The resolved YAML files record exactly what was used;
`artifacts.json` is the authoritative product index.

Workflows are independent where possible. If one fails, sibling products may
still complete and the run retains its log, manifest, report, and completed
artifacts. Check the final status and failed-workflow list.

## 12. Reports, artifacts, and provenance

Regenerate a report without rereading source simulation data:

```bash
pelec-post report PROJECT RUN_ID
```

Artifact records include IDs, schema versions, recipe instance, variables,
units, coordinate metadata, source inputs, interpretation, and provenance.
Arrays use compressed NPZ/HDF5, scalar records use JSON, tables use unit-
bearing CSV, and figures use PNG or vector formats.

Case comparison checks typed product schemas, variables, units, coordinates,
masks, and preprocessing provenance before comparing products. Time, frequency,
and space use explicit alignment policies. Probe indices and coordinate arrays
are never numerical difference metrics. Every run also includes conservative
measurement evidence; it does not claim LST/PSE attribution or causality.

## 13. Probe archive workflow

Create a compact HDF5 archive from probe-v2 sources:

```bash
pelec-post probes compact \
  --input '/path/to/probes*.pbin' \
  --output /path/to/probes.h5
```

Verify it against acquisition files:

```bash
pelec-post probes verify /path/to/probes.h5 \
  --manifest /path/to/probes.verify.json
```

Preview source files eligible for removal:

```bash
pelec-post probes prune /path/to/probes.verify.json
```

Delete only after reviewing the preview:

```bash
pelec-post probes prune /path/to/probes.verify.json --confirm-delete
```

Verification must happen before deletion, and `--confirm-delete` is required.
Selected probe matrices can spill to `compute.scratch_directory`; temporary
spill files are removed after each workflow and are not run artifacts.

## 14. SLURM execution

Run the new project command from the refactor checkout:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --job-name=pelec-post

set -euo pipefail
cd /path/to/projects/PeleC-Python-usability
source .venv/bin/activate
pelec-post run /path/to/my-project --name slurm-pass
```

It is usually best to run `inspect`, `validate`, and `plan` interactively,
then submit only `run`. If retaining an existing wrapper, it must forward the
project path and new CLI arguments; it must not invoke the old no-argument
JSON/global-switch interface.

Run provenance records host, Python/package versions, relevant SLURM
environment, Git commit/dirty state, configuration fingerprints, and source
input fingerprints.

## 15. Exit codes and troubleshooting

- exit `0`: command succeeded;
- exit `2`: configuration or preflight rejection;
- exit `1`: at least one runtime workflow failed.

For exit `2`:

```bash
pelec-post validate PROJECT
pelec-post inspect PROJECT
pelec-post plan PROJECT
```

For exit `1`:

```bash
RUN_DIR=/path/to/project/runs/RUN_ID
less "$RUN_DIR/logs/run.log"
cat "$RUN_DIR/manifest.json"
```

Common remedies:

- Missing `machine.yaml`: copy `machine.example.yaml` and set local paths.
- Missing fields: inspect canonical field mapping and configure
  `case.field_aliases` if native names differ.
- Spectrum rejection: compare the requested band with Nyquist and native or
  effective resolution.
- Directional-wave rejection: review station count, spacing, aperture,
  shortest wavelength, and sample count.
- Memory blocker: reduce workers, probes, snapshots, region size, or FFT batch
  size; do not bypass the configured limit.
- Unexpected plot contrast: check whether the range is per-snapshot,
  shared-percentile, shared-min/max, or fixed.
- Stale products: use a new timestamped run directory rather than mixing old
  files into a later run.

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the short reference.

## 16. Testing and development checks

From the repository root:

```bash
python -m pytest -q tests
python -m pytest -q test_engineering.py
python -m mypy pelecpost
python -m compileall -q pelecpost tools benchmarks
python -m tools.generate_reference_docs
git diff --check
```

If models or recipe metadata change, regenerate the configuration/recipe
references and example schemas, then rerun the tests. The benchmark is:

```bash
python -m benchmarks.benchmark_probe_workspace --force-spill
```

Performance varies by hardware, filesystem, data dimensions, and environment;
the benchmark is a reference measurement, not a pass/fail limit.

## 17. Legacy migration

The former JSON and global workflow-switch interface is archived under
`legacy/` and is not accepted by the current CLI. Migration is:

1. Run `pelec-post init PROJECT`.
2. Move portable physics and geometry into `case.yaml`.
3. Move server paths and resources into `machine.yaml`.
4. Select physical-question recipes in `analyses.yaml`.
5. Run `inspect`, `validate`, and `plan`.
6. Run with `pelec-post run PROJECT`.
7. Use the report and artifact ledger instead of legacy filenames.

The root `pelec_post.py` is only a thin HPC-compatible launcher; it does not
restore the retired interface.

Do not commit generated `*.egg-info/` directories or cache directories. Keep
source code, YAML projects, run artifacts, and reports separate according to
the project's reproducibility and storage policy.
