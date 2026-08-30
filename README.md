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

## Configuration ownership

- `case.yaml`: case identity, gas, freestream reference, geometry, and solver units.
- `analyses.yaml`: enabled recipe instances and their scientific choices.
- `machine.yaml`: server-local inputs, output root, workers, and memory limit.

Unknown keys and cross-file responsibility violations are rejected. Public dimensional values use SI and unit-bearing key names. PeleC CGS values are converted at the I/O boundary.

See [configuration](docs/CONFIGURATION.md), [recipe reference](docs/RECIPES.md), [method selection](docs/METHOD_SELECTION.md), [scientific assumptions](docs/SCIENTIFIC_ASSUMPTIONS.md), [probe acquisition](docs/PROBES.md), [artifact contracts](docs/ARTIFACTS.md), [2-D EB validation](docs/EB_VALIDATION.md), [performance references](docs/PERFORMANCE.md), [SLURM troubleshooting](docs/TROUBLESHOOTING.md), and the [future 3-D design](docs/ARCHITECTURE_3D.md).

The tested release evidence and explicit scope limits are recorded in the [acceptance record](docs/ACCEPTANCE.md).
