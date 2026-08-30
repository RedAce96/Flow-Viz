# Configuration reference

This file is generated from the strict Pydantic project models. Every project requires
`case.yaml`, `analyses.yaml`, and `machine.yaml`, each with `schema_version: 1`.
`pelec-post init` and `validate` also write `project.schema.json` for editor completion.

Unknown keys and keys placed in the wrong file are rejected. Public dimensional values
use SI and unit-bearing names; PeleC CGS values are converted at the I/O boundary. Relative
paths resolve from the project directory. The wizard and direct YAML editing use these same
models. An empty `analyses` array is valid and runs nothing.

Large selected probe matrices spill to a bounded temporary memory-mapped workspace. Set
`compute.scratch_directory` to node-local storage when available; spill files are removed
after each workflow and are not run artifacts.

## `case.yaml`

### `CaseFile`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `schema_version` | no | `1` | `1` | — |
| `case` | yes | `CaseIdentity` | — | — |
| `gas` | no | `GasConfig` | `{"gamma": 1.4, "gas_constant_j_kg_k": 287.05}` | — |
| `freestream` | yes | `ExplicitFreestream \| RegionFreestream` | — | discriminator: `source` |
| `geometry` | yes | `FlatPlateGeometry \| WedgeGeometry \| PolylineGeometry \| VolumeFractionGeometry` | — | discriminator: `type` |
| `field_aliases` | no | `mapping[string, string]` | — | — |

### `CaseIdentity`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `solver` | no | `'pelec'` | `"pelec"` | — |
| `dimensionality` | no | `2 \| 3` | `2` | — |
| `solver_units` | no | `SolverUnits` | `"cgs"` | — |
| `description` | no | `string \| null` | `null` | — |

### `ExplicitFreestream`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `source` | no | `'explicit'` | `"explicit"` | — |
| `density_kg_m3` | yes | `number` | — | greater than: `0.0` |
| `velocity_m_s` | yes | `number` | — | greater than: `0.0` |
| `pressure_pa` | yes | `number` | — | greater than: `0.0` |
| `temperature_k` | yes | `number` | — | greater than: `0.0` |

### `FlatPlateGeometry`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'flat_plate'` | `"flat_plate"` | — |
| `leading_edge_x_m` | no | `number` | `0.0` | — |
| `trailing_edge_x_m` | no | `number \| null` | `null` | — |
| `wall_y_m` | no | `number` | `0.0` | — |
| `fluid_side` | no | `'above' \| 'below'` | `"above"` | — |

### `GasConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `gamma` | no | `number` | `1.4` | greater than: `1.0` |
| `gas_constant_j_kg_k` | no | `number` | `287.05` | greater than: `0.0` |

### `PolylineGeometry`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'polyline'` | `"polyline"` | — |
| `points_m` | yes | `array[tuple[number, number]]` | — | — |
| `closed` | no | `boolean` | `false` | — |
| `fluid_side` | yes | `'left' \| 'right' \| 'outside' \| 'inside'` | — | — |

### `RegionBounds`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `x_m` | yes | `tuple[number, number]` | — | minimum items: `2`; maximum items: `2` |
| `y_m` | yes | `tuple[number, number]` | — | minimum items: `2`; maximum items: `2` |

### `RegionFreestream`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `source` | no | `'region'` | `"region"` | — |
| `bounds` | yes | `RegionBounds` | — | — |
| `statistic` | no | `'median' \| 'mean'` | `"median"` | — |

### `SolverUnits`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |

### `VolumeFractionGeometry`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'volume_fraction'` | `"volume_fraction"` | — |
| `field` | no | `string` | `"volume_fraction"` | — |
| `fluid_value` | no | `0 \| 1` | `1` | — |
| `iso_value` | no | `number` | `0.5` | greater than: `0.0`; less than: `1.0` |
| `minimum_component_points` | no | `integer` | `8` | greater than: `0` |
| `smoothing_window` | no | `integer` | `1` | greater than: `0` |

### `WedgeGeometry`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'wedge'` | `"wedge"` | — |
| `leading_edge_x_m` | yes | `number` | — | — |
| `leading_edge_y_m` | yes | `number` | — | — |
| `length_m` | yes | `number` | — | greater than: `0.0` |
| `half_angle_deg` | yes | `number` | — | greater than: `0.0`; less than: `90.0` |
| `fluid_side` | no | `'outside' \| 'inside'` | `"outside"` | — |

## `analyses.yaml`

### `AnalysesFile`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `schema_version` | no | `1` | `1` | — |
| `analyses` | no | `array[FlowOverviewAnalysis \| BoundaryLayerAnalysis \| SurfaceDiagnosticsAnalysis \| AerodynamicForcesAnalysis \| ProbeSpectrumAnalysis \| SinglePulseAnalysis \| DirectionalWaveAnalysis \| TransientWavepacketAnalysis \| NonlinearCouplingAnalysis \| ModalScreeningAnalysis \| CaseComparisonAnalysis]` | `[]` | — |

### `AerodynamicForcesAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `recipe` | yes | `'aerodynamic_forces'` | — | — |
| `reference_chord_m` | yes | `number` | — | greater than: `0.0` |
| `reference_span_m` | no | `number` | `1.0` | greater than: `0.0` |
| `moment_origin_m` | no | `tuple[number, number]` | `[0.0, 0.0]` | minimum items: `2`; maximum items: `2` |
| `dynamic_viscosity_pa_s` | yes | `number` | — | greater than: `0.0` |
| `conductivity_w_m_k` | yes | `number` | — | greater than: `0.0` |
| `wall_temperature_k` | no | `number \| null` | `null` | — |
| `normal_sample_distance_m` | no | `number` | `0.001` | greater than: `0.0` |
| `normal_sample_points` | no | `integer` | `8` | greater than: `0` |
| `baseline` | no | `'none' \| 'static' \| 'paired'` | `"none"` | — |
| `baseline_id` | no | `string \| null` | `null` | — |
| `control_volume` | no | `ControlVolumeConfig \| null` | `null` | — |
| `probe_linkage` | no | `ForceProbeLinkageConfig \| null` | `null` | — |

### `BoundaryLayerAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `recipe` | yes | `'boundary_layer_reference'` | — | — |
| `stations_x_m` | yes | `array[number]` | — | — |
| `maximum_height_m` | yes | `number` | — | greater than: `0.0` |
| `wall_temperature_k` | yes | `number` | — | greater than: `0.0` |
| `dynamic_viscosity_pa_s` | yes | `number` | — | greater than: `0.0` |
| `conductivity_w_m_k` | yes | `number` | — | greater than: `0.0` |
| `zero_pressure_gradient` | no | `True` | `true` | — |
| `laminar_reference` | no | `True` | `true` | — |

### `CaseComparisonAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `recipe` | yes | `'case_comparison'` | — | — |
| `baseline_id` | yes | `string` | — | — |
| `comparison_id` | yes | `string` | — | — |
| `artifact_ids` | yes | `array[string]` | — | minimum items: `1` |

### `ControlVolumeConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `x_range_m` | yes | `tuple[number, number]` | — | minimum items: `2`; maximum items: `2` |
| `y_top_m` | yes | `number` | — | greater than: `0.0` |
| `bulk_viscosity_pa_s` | no | `number` | `0.0` | minimum: `0.0` |

### `DirectionalWaveAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'directional_wave'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `frequency_min_hz` | no | `number` | `0.0` | minimum: `0.0` |
| `frequency_max_hz` | yes | `number` | — | greater than: `0.0` |
| `expected_speed_min_m_s` | no | `number \| null` | `null` | — |
| `expected_speed_max_m_s` | no | `number \| null` | `null` | — |
| `minimum_coherence` | no | `number` | `0.8` | minimum: `0.0`; maximum: `1.0` |
| `spatial_window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"hann"` | — |
| `temporal_window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"rectangular"` | — |

### `FlowOverviewAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `recipe` | yes | `'flow_overview'` | — | — |
| `fields` | no | `array[Variable]` | `["temperature", "pressure"]` | minimum items: `1` |
| `snapshot_start` | no | `integer \| null` | `null` | — |
| `snapshot_end` | no | `integer \| null` | `null` | — |
| `snapshot_step` | no | `integer` | `1` | greater than: `0` |
| `x_limits_m` | no | `tuple[number, number] \| null` | `null` | — |
| `y_limits_m` | no | `tuple[number, number] \| null` | `null` | — |
| `line_stations_x_m` | no | `array[number]` | `[]` | — |
| `streamlines` | no | `boolean` | `false` | — |

### `ForceProbeLinkageConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `variable` | no | `Variable` | `"pressure"` | — |
| `force_component` | no | `'x' \| 'y' \| 'moment'` | `"y"` | — |
| `forcing_frequency_hz` | yes | `number` | — | greater than: `0.0` |
| `minimum_forcing_periods` | no | `number` | `10.0` | greater than: `0.0` |
| `welch_segment_samples` | no | `integer` | `16384` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.5` | minimum: `0.0`; less than: `1.0` |
| `minimum_segments` | no | `integer` | `8` | greater than: `0` |
| `probe_indices` | no | `array[integer]` | `[]` | — |

### `ModalScreeningAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'modal_screening'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `mode_count` | no | `integer` | `4` | greater than: `0` |
| `probe_stride` | no | `integer` | `1` | greater than: `0` |
| `spod_segment_samples` | no | `integer` | `2048` | greater than: `0` |
| `dmd_ranks` | no | `array[integer]` | `[2, 4, 8]` | — |
| `sensitivity_windows` | no | `array[tuple[number, number]]` | `[[0.0, 0.5], [0.5, 1.0]]` | — |

### `NonlinearCouplingAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'nonlinear_coupling'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `segment_samples` | no | `integer` | `8192` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.5` | minimum: `0.0`; less than: `1.0` |
| `surrogate_count` | no | `integer` | `499` | minimum: `19` |
| `fdr_alpha` | no | `number` | `0.05` | greater than: `0.0`; less than: `1.0` |
| `frequency_max_hz` | yes | `number` | — | greater than: `0.0` |
| `target_frequencies_hz` | no | `array[number]` | `[]` | — |
| `automatic_frequency_selection` | no | `boolean` | `false` | — |

### `ProbeSpectrumAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'probe_spectrum'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `frequency_max_hz` | no | `number \| null` | `null` | — |
| `window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"hann"` | — |
| `detrend` | no | `'mean' \| 'linear' \| 'none'` | `"mean"` | — |
| `welch_segment_samples` | no | `integer \| null` | `null` | — |
| `overlap_fraction` | no | `number` | `0.5` | minimum: `0.0`; less than: `1.0` |

### `SinglePulseAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'single_pulse_response'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `energy_per_pulse_j_m` | yes | `number` | — | greater than: `0.0` |
| `pulse_fwhm_s` | yes | `number` | — | greater than: `0.0` |
| `pulse_period_s` | yes | `number` | — | greater than: `0.0` |
| `start_time_s` | no | `number` | `0.0` | — |
| `cutoff_sigma` | no | `number` | `4.0` | greater than: `0.0` |
| `baseline_end_time_s` | no | `number \| null` | `null` | — |
| `minimum_baseline_samples` | no | `integer` | `8` | greater than: `0` |
| `minimum_relative_source_amplitude` | no | `number` | `0.001` | greater than: `0.0`; less than: `1.0` |
| `frequency_max_hz` | no | `number \| null` | `null` | — |

### `SurfaceDiagnosticsAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `recipe` | yes | `'surface_diagnostics'` | — | — |
| `snapshot_start` | no | `integer \| null` | `null` | — |
| `snapshot_end` | no | `integer \| null` | `null` | — |
| `normal_sample_distance_m` | no | `number` | `0.001` | greater than: `0.0` |
| `normal_sample_points` | no | `integer` | `8` | greater than: `0` |

### `TransientWavepacketAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `recipe` | yes | `'transient_wavepacket'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `band_min_hz` | yes | `number` | — | minimum: `0.0` |
| `band_max_hz` | yes | `number` | — | greater than: `0.0` |
| `baseline_end_time_s` | no | `number \| null` | `null` | — |
| `stft_segment_samples` | no | `integer` | `2048` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.75` | minimum: `0.0`; less than: `1.0` |

### `Variable`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |

## `machine.yaml`

### `MachineFile`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `schema_version` | no | `1` | `1` | — |
| `inputs` | no | `InputConfig` | `{"baselines": {}, "comparison_archives": {}, "plotfiles": null, "probes": null}` | — |
| `outputs` | yes | `OutputConfig` | — | — |
| `compute` | no | `ComputeConfig` | `{"fft_batch_size": 32, "memory_limit_gb": 8.0, "scratch_directory": null, "workers": 1}` | — |

### `ComputeConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `workers` | no | `integer` | `1` | greater than: `0` |
| `memory_limit_gb` | no | `number` | `8.0` | greater than: `0.0` |
| `fft_batch_size` | no | `integer` | `32` | greater than: `0` |
| `scratch_directory` | no | `string \| null` | `null` | — |

### `InputConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `plotfiles` | no | `PlotfileInput \| null` | `null` | — |
| `probes` | no | `ProbeInput \| null` | `null` | — |
| `comparison_archives` | no | `mapping[string, string]` | — | — |
| `baselines` | no | `mapping[string, PlotfileInput]` | — | — |

### `OutputConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `root` | yes | `string` | — | — |

### `PlotfileInput`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `source` | yes | `string` | — | — |
| `prefix` | no | `string` | `"plt"` | — |

### `ProbeInput`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `compact_file` | no | `string \| null` | `null` | — |
| `binary_files` | no | `array[string]` | `[]` | — |

Repository maintainers regenerate this reference, the recipe reference, and example
schemas with `python -m tools.generate_reference_docs`.
