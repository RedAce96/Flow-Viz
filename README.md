# PeleC Post-Processor

`pelec-post` is a recipe-driven, SI-facing post-processor for two-dimensional PeleC data. It replaces interdependent Boolean workflow switches with physical questions, strict YAML, scientific preflight, isolated runs, stable artifacts, and portable HTML reports.

## Five-minute start

```bash
python -m pip install -e .
pelec-post init my-analysis
```

Edit `my-analysis/case.yaml` for portable physics and `my-analysis/machine.yaml` for this server's paths. Then:

```bash
pelec-post inspect my-analysis
pelec-post configure my-analysis
pelec-post validate my-analysis
pelec-post plan my-analysis
pelec-post run my-analysis --name first-pass
```

The report is written to the new timestamped run's `report/index.html`. A failed independent workflow leaves its log, completed sibling products, manifest, artifact ledger, and readable report intact.

The clean-break interface does not accept legacy JSON or old workflow flags. The root `pelec_post.py` is only a launcher, so existing SLURM wrappers can continue forwarding `$@`; pass new CLI arguments such as `run /path/to/project`.

## Replay saved products

Run `python replot_saved_kernel_products.py --export /path/to/archived-run` to
redraw supported spectra and comparisons from one run's registered artifacts.
Replay uses that run's resolved case and analysis settings. Use repeatable
`--analysis-id ID` options to narrow the selection, `--project DIR` to
explicitly override those settings, and `--output DIR` to choose a destination.
`--replay-config LOCAL_REPLAY_CONFIG.yaml` supplies optional command defaults;
explicit command options take precedence. An existing destination resumes only
when source, selected settings, and processing build match exactly. The replay
manifest records missing source inputs and unsupported products. Saved f-k power
can be redrawn, but its original transform cannot be recalculated without the
full probe histories.

## Configuration ownership

- `case.yaml`: case identity, gas, freestream reference, geometry, and solver units.
- `analyses.yaml`: enabled recipe instances and their scientific choices.
- `machine.yaml`: server-local inputs, output root, workers, and memory limit.

Unknown keys and cross-file responsibility violations are rejected. Public dimensional values use SI and unit-bearing key names. PeleC CGS values are converted at the I/O boundary.

See [configuration](docs/CONFIGURATION.md), [figure presentation and profile extraction](docs/PRESENTATION.md), [recipe reference](docs/RECIPES.md), [method selection](docs/METHOD_SELECTION.md), [scientific assumptions](docs/SCIENTIFIC_ASSUMPTIONS.md), [probe acquisition](docs/PROBES.md), [artifact contracts](docs/ARTIFACTS.md), [2-D EB validation](docs/EB_VALIDATION.md), [performance references](docs/PERFORMANCE.md), [SLURM troubleshooting](docs/TROUBLESHOOTING.md), and the [future 3-D design](docs/ARCHITECTURE_3D.md).

The tested release evidence and explicit scope limits are recorded in the [acceptance record](docs/ACCEPTANCE.md).
