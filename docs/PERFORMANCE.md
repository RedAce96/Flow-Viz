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
