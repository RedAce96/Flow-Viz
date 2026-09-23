# Performance references

Performance measurements are kept separate from pass/fail CI because parallel
filesystem, CPU, HDF5, and scheduler behavior vary substantially between
machines. Run the tracked benchmark from the repository root:

```bash
python -m benchmarks.benchmark_probe_workspace --force-spill
```

Reference measurement on 2026-08-30 on the development Linux compute node,
using Python 3.11, NumPy/HDF5 from the active ISAAC environment:

- Input matrix: 32,768 samples × 128 probes.
- Selected matrix: 32,768 samples × 96 probes (24 MiB float64).
- Storage policy: forced disk-backed workspace.
- Bounded read block: 21,845 rows × 96 probes.
- Elapsed staging/checksum time: 0.064 s.
- Process peak RSS: 121.9 MiB (includes the Python/NumPy/HDF5 process, archive
  construction, and benchmark workspace).
- Sampled checksum: 3644.512595865574.

The benchmark reports a sampled checksum to ensure the staged matrix is read.
It does not set a performance threshold. Compare results only when dataset
dimensions, selected probe count, storage policy, software environment, and
filesystem class match.

## High-AMR flow overview memory check

The process-recycling change must also be measured on representative server
plotfiles. Run one high-AMR snapshot, then at least ten consecutive snapshots
with one worker and recycling enabled. Record dataset identity and dimensions,
worker count, process IDs, peak resident memory for each snapshot, the parent
process peak, and the configured memory limit. The single-snapshot peak tests
whether one task fits; the sequence tests whether memory accumulates across
tasks. Keep the observed measurements in the acceptance record. If the server
dataset is unavailable locally, leave this check open rather than claiming an
OOM resolution from unit tests alone.

On Linux, wrap the single-snapshot `pelec-post run` command with
`/usr/bin/time -v -o single-time.txt`. After each run, use
`python -m tools.flow_overview_memory_report RUN_DIR --time-log single-time.txt`
for the single snapshot and
`python -m tools.flow_overview_memory_report RUN_DIR` for the sequence. The
report reads task process IDs and per-process high-water marks from `run.log`;
the external time log supplies the single-run process peak. Compare these
measurements with the original failed-run log and the Slurm allocation.
