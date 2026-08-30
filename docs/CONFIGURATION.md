# Configuration reference

Every project requires `case.yaml`, `analyses.yaml`, and `machine.yaml`, each with `schema_version: 1`. `pelec-post init` writes the authoritative `project.schema.json` generated from the Pydantic models. `validate` regenerates it after upgrades.

`case.yaml` owns portable physics: case ID, PeleC dimensionality and solver units, gas constants, explicit or robust-region freestream, field aliases, and flat-plate/wedge/polyline/volume-fraction geometry. `analyses.yaml` owns recipe instances. `machine.yaml` owns plotfile/probe/baseline/comparison locations, output root, workers, memory limit, and FFT batch size.

Unknown keys are errors. A key in the wrong file is an error. Public dimensional values use suffixes such as `_m`, `_s`, `_hz`, `_pa`, `_kg_m3`, `_m_s`, `_w_m_k`, or `_pa_s`; raw PeleC CGS values are converted at the reader boundary. Relative paths resolve from the project directory.

Use the generated JSON Schema for editor completion. Direct YAML editing and wizard output pass through the same strict loader.
