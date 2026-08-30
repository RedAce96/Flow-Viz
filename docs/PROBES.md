# Probe acquisition, compaction, verification, and pruning

PeleC probe-v2 files retain restart-safe steps, chunk mappings, requested coordinates, sampled coordinates, levels, validity, field names, and units. Recipes can read probe-v2 globs or compact HDF5 directly through the same bounded SI workspace. Compact HDF5 remains preferable for repeated production analysis because it is self-contained, verified, and read-optimized.

```yaml
inputs:
  probes:
    binary_files: [/path/probes.segment*.pbin]
# Or use exactly one compact source:
#   compact_file: /path/probes.h5
```

```bash
pelec-post probes compact --input '/path/probes*.pbin' --output probes.h5
pelec-post probes verify probes.h5 --manifest probes.verify.json
pelec-post probes prune probes.verify.json
pelec-post probes prune probes.verify.json --confirm-delete
```

Compaction resolves restart overlap under an explicit policy, preserves mapping epochs, detects disturbance-window quality, records source fingerprints, and writes diagnostics. Verification compares the archive with sources before a prune manifest can authorize deletion. Prune previews by default; `--confirm-delete` is deliberately required.

`inspect` reports fields, units, coordinates, epochs, timing, overlaps, provenance, quality approval, and nonfinite value counts using bounded reads. Probe-only recipes never require plotfile discovery.
