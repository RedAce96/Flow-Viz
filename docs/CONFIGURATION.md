# Configuration reference

Every project requires `case.yaml`, `analyses.yaml`, and `machine.yaml`, each with `schema_version: 1`. `pelec-post init` writes the authoritative `project.schema.json` generated from the Pydantic models. `validate` regenerates it after upgrades.

`case.yaml` owns portable physics: case ID, PeleC dimensionality and solver units, gas constants, explicit or robust-region freestream, field aliases, and flat-plate/wedge/polyline/volume-fraction geometry. `analyses.yaml` owns recipe instances. `machine.yaml` owns plotfile/probe/baseline/comparison locations, output root, workers, memory limit, FFT batch size, and an optional node-local `scratch_directory`.

Unknown keys are errors. A key in the wrong file is an error. Public dimensional values use suffixes such as `_m`, `_s`, `_hz`, `_pa`, `_kg_m3`, `_m_s`, `_w_m_k`, or `_pa_s`; raw PeleC CGS values are converted at the reader boundary. Relative paths resolve from the project directory.

Use the generated JSON Schema for editor completion. Direct YAML editing and wizard output pass through the same strict loader.

`aerodynamic_forces` can optionally add a flat-plate steady rectangular
`control_volume` diagnostic and a `probe_linkage` section. Probe linkage adds
the compact-probe archive as an explicit DAG dependency and reports lag,
coherence, phase, and H1 association without making a causality claim. These
sections are optional, so their artifacts appear in the plan only when they
are configured. See the generated schema for all unit-bearing fields.

Large selected probe matrices spill to a bounded temporary memory-mapped workspace. Set `compute.scratch_directory` to node-local storage on an HPC system; relative paths resolve from the project directory. Spill files are removed after each workflow and never become run artifacts.

Repository maintainers regenerate the committed recipe reference and example
schemas with `python tools/generate_reference_docs.py`.
