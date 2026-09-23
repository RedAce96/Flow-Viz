# Run and artifact contracts

Every execution creates a UTC timestamped directory and never reuses it. `manifest.json` records run/workflow state and provenance; `plan.json` records preflight; resolved YAML records the exact configuration; `artifacts.json` is the only product index.

Artifact IDs are independent of directory spelling. Each record includes schema version, recipe instance, variable, units, coordinate metadata, source inputs, provenance, and interpretation. Arrays are compressed NPZ/HDF5 without pickle, scalar records are JSON, tables are CSV with unit metadata, and figures are PNG or vector formats.

Comparison must reject incompatible schema, variable, unit, coordinate, and preprocessing provenance. Reports regenerate from the run directory without simulation inputs.

`spectral.confidence` uses schema version 3 and records overlap-corrected
equivalent Welch degrees of freedom. f-k ratio statistics use schema version 2
and always record support counts and masked fraction, including a typed
`unavailable_no_supported_bins` result. Semantic f-k figure IDs carry their
normalization, scale, support, limits, units, coordinate, and alignment
provenance; legacy overlays identify the semantic product they copy.

Saved-product replay writes a separate schema-version-4 manifest. Original
source-input fingerprints and hashes of saved derived artifacts are reported
separately. An exact identity match permits resume and replacement of damaged
replay-owned files. Missing original inputs remain explicitly missing even when
their saved derived products can be redrawn.

The single-pulse recipe registers `pulse.source_audit` alongside the measured
or modeled source spectrum and transfer products. A configured measured
thermal-source history is reported in 2-D J/m terms, with its deposited energy,
moments, geometry, implementation revisions, and completeness status. Without
that input the audit is retained as `modeled_only` and cannot claim measured
source-deposition support.

Every run also registers `run.measurement-evidence`. It is derived only from
the artifact ledger, records explicit decision thresholds, and conservatively
classifies measured propagation, packet kinematics, FDR-controlled nonlinear
association, modal robustness, and available loads. Empty domains remain empty;
the classifier never fills missing evidence by inference and explicitly
excludes LST/PSE attribution and causality.
