# Probe acquisition, compaction, verification, and pruning

PeleC probe-v2 files retain restart-safe steps, chunk mappings, requested coordinates, sampled coordinates, levels, validity, field names, and units. Recipes can read probe-v2 globs or compact HDF5 directly through the same bounded SI workspace. Compact HDF5 remains preferable for repeated production analysis because it is self-contained, verified, and read-optimized.

Probe data is selected by name in `machine.yaml`; each named source must use
exactly one storage format. The recipe only refers to the name, so changing
FFT to STFT, pulse, nonlinear, modal, or wavenumber analysis does not change
the input contract:

```yaml
inputs:
  probe_sets:
    asym:
      binary_files: [/path/probes.segment*.pbin]
    gaus:
      compact_file: /path/probes.h5
  archived_runs:
    previous_asym: /results/asym/run
```

```yaml
analyses:
  - id: asym-stft
    recipe: transient_wavepacket
    probe_set_id: asym
    variable: pressure
    time_grid_policy: resample_uniform
```

```bash
pelec-post probes compact --input '/path/probes*.pbin' --output probes.h5
pelec-post probes verify probes.h5 --manifest probes.verify.json
pelec-post probes prune probes.verify.json
pelec-post probes prune probes.verify.json --confirm-delete
```

Compaction resolves restart overlap under an explicit policy, preserves mapping epochs, detects disturbance-window quality, records source fingerprints, and writes diagnostics. Verification compares the archive with sources before a prune manifest can authorize deletion. Prune previews by default; `--confirm-delete` is deliberately required.

`inspect` reports fields, units, coordinates, epochs, timing, overlaps,
provenance, quality approval, and nonfinite value counts for every named source
using bounded reads. Probe-only recipes never require plotfile discovery.

Every probe recipe can render all selected probes together. The default is to
keep the existing per-probe diagnostic panels and add shared raw, method-ready,
and applicable result figures. Configure the optional behavior per analysis:

```yaml
probe_indices: [0, 2, 5]
probe_plotting:
  mode: both                 # overlay, panels, or both
  normalization: none        # none or per_probe_peak
  label: index_coordinates   # index_coordinates, coordinates, or index
```

Raw and method-ready overlays use SI values unless per-probe normalization is
explicitly selected. Directional, nonlinear, and modal products retain their
aggregate scientific meaning; their figures do not imply independent
per-probe estimates where the numerical method combines the probe aperture.

Analysis products carry versioned contracts identifying their physical axes,
value arrays, units, validity masks, complex-value semantics, and supported
comparison/plotting operations. Comparisons reference local analysis products
or archived products by `analysis_id`, and align time, frequency, and space
with an explicit policy (`strict`, `intersection`, or interpolation to one
side's grid). Probe indices remain labels; numerical differences use values
matched by physical coordinates.

## Time-localized wavenumber figures

The `directional_wave` recipe can optionally analyze a transient pulse in
bounded sliding temporal windows. The full-record signed k–omega and
coherence-gated complex-wavenumber products remain available:

```yaml
temporal_wavenumber:
  enabled: true
  window_duration_s: 1.0e-6
  overlap_fraction: 0.75
  window: hann
  snapshot_times_s: []       # empty selects active/peak/post-pulse landmarks
  minimum_relative_energy_db: -30.0
  display_floor_db: -60.0
```

This registers a globally scaled k-versus-time map, a dominant-ridge history,
space-time signal context, selected f-k snapshots, and phase-speed/wavelength
dispersion figures. Power is processed one window at a time; the full
time-frequency-wavenumber cube is not retained. Low-energy windows are
explicitly masked, so post-pulse noise is not promoted by per-window
normalization. The localized products support time, frequency, and wavenumber
comparison alignment.
