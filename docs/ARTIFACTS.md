# Run and artifact contracts

Every execution creates a UTC timestamped directory and never reuses it. `manifest.json` records run/workflow state and provenance; `plan.json` records preflight; resolved YAML records the exact configuration; `artifacts.json` is the only product index.

Artifact IDs are independent of directory spelling. Each record includes schema version, recipe instance, variable, units, coordinate metadata, source inputs, provenance, and interpretation. Arrays are compressed NPZ/HDF5 without pickle, scalar records are JSON, tables are CSV with unit metadata, and figures are PNG or vector formats.

Comparison must reject incompatible schema, variable, unit, coordinate, and preprocessing provenance. Reports regenerate from the run directory without simulation inputs.
