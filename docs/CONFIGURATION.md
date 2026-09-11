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
| `density_kg_m3` | yes | `number` | — | greater than: `0` |
| `velocity_m_s` | yes | `number` | — | greater than: `0` |
| `pressure_pa` | yes | `number` | — | greater than: `0` |
| `temperature_k` | yes | `number` | — | greater than: `0` |

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
| `gas_constant_j_kg_k` | no | `number` | `287.05` | greater than: `0` |

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
| `length_m` | yes | `number` | — | greater than: `0` |
| `half_angle_deg` | yes | `number` | — | greater than: `0.0`; less than: `90.0` |
| `fluid_side` | no | `'outside' \| 'inside'` | `"outside"` | — |

## `analyses.yaml`

### `AnalysesFile`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `schema_version` | no | `1` | `1` | — |
| `presentation` | no | `PresentationConfig` | `{"contour_axes": {"x_tick_format": "auto", "y_tick_format": "auto"}, "contour_defaults": {"colorbar": {"include_endpoints": false, "label": "auto", "length_fraction": 0.43, "position": "top", "thickness_fraction": null, "tick_count": 4, "tick_format": "auto"}, "colormap": "viridis", "normalization": "linear", "range": {"lower_percentile": 1.0, "maximum": null, "minimum": null, "mode": "per_snapshot_percentile", "upper_percentile": 99.0}, "rendering": {"levels": null, "mode": "continuous"}, "symlog_linear_threshold": null, "symmetric_about_zero": false}, "figure": {"dpi": 300, "formats": ["png"], "height_in": 4.5, "transparent": false, "width_in": 14.0}, "line_defaults": {"color": null, "coordinate_scale": "linear", "grid": true, "legend_position": "best", "linestyle": "solid", "linewidth": 2.0, "marker": "none", "value_scale": "linear"}, "preset": "publication", "time_annotation": {"boxed": true, "enabled": true, "position": "top_left", "precision": 4}, "typography": {"axes_label_size": 16.0, "base_size": 14.0, "colorbar_label_pad": 2.0, "colorbar_label_size": 13.0, "colorbar_tick_label_size": 11.0, "colorbar_tick_pad": 2.0, "font_family": "DejaVu Sans", "legend_size": 13.0, "tick_label_size": 14.0}}` | — |
| `analyses` | no | `array[FlowOverviewAnalysis \| BoundaryLayerAnalysis \| SurfaceDiagnosticsAnalysis \| AerodynamicForcesAnalysis \| ProbeSpectrumAnalysis \| SinglePulseAnalysis \| DirectionalWaveAnalysis \| TransientWavepacketAnalysis \| NonlinearCouplingAnalysis \| ModalScreeningAnalysis \| CaseComparisonAnalysis]` | `[]` | — |

### `AerodynamicForcesAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `recipe` | yes | `'aerodynamic_forces'` | — | — |
| `reference_chord_m` | yes | `number` | — | greater than: `0` |
| `reference_span_m` | no | `number` | `1.0` | greater than: `0` |
| `moment_origin_m` | no | `tuple[number, number]` | `[0.0, 0.0]` | minimum items: `2`; maximum items: `2` |
| `dynamic_viscosity_pa_s` | yes | `number` | — | greater than: `0` |
| `conductivity_w_m_k` | yes | `number` | — | greater than: `0` |
| `wall_temperature_k` | no | `number \| null` | `null` | — |
| `normal_sample_distance_m` | no | `number` | `0.001` | greater than: `0` |
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
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `recipe` | yes | `'boundary_layer_reference'` | — | — |
| `stations_x_m` | yes | `array[number]` | — | — |
| `maximum_height_m` | yes | `number` | — | greater than: `0` |
| `wall_temperature_k` | yes | `number` | — | greater than: `0` |
| `dynamic_viscosity_pa_s` | yes | `number` | — | greater than: `0` |
| `conductivity_w_m_k` | yes | `number` | — | greater than: `0` |
| `zero_pressure_gradient` | no | `True` | `true` | — |
| `laminar_reference` | no | `True` | `true` | — |

### `CaseComparisonAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `recipe` | yes | `'case_comparison'` | — | — |
| `baseline` | yes | `ComparisonReference` | — | — |
| `comparison` | yes | `ComparisonReference` | — | — |
| `product_ids` | yes | `array[string]` | — | minimum items: `1` |
| `alignment` | no | `ComparisonAlignment` | — | — |

### `ColorbarOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `position` | no | `'top' \| 'bottom' \| 'left' \| 'right' \| null` | `null` | — |
| `length_fraction` | no | `number \| null` | `null` | — |
| `thickness_fraction` | no | `number \| null` | `null` | — |
| `include_endpoints` | no | `boolean \| null` | `null` | — |
| `tick_count` | no | `integer \| null` | `null` | — |
| `tick_format` | no | `string \| null` | `null` | — |
| `label` | no | `string \| null` | `null` | — |

### `ColorbarPresentation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `position` | no | `'top' \| 'bottom' \| 'left' \| 'right'` | `"top"` | — |
| `length_fraction` | no | `number` | `0.43` | minimum: `0.2`; maximum: `0.75` |
| `thickness_fraction` | no | `number \| null` | `null` | — |
| `include_endpoints` | no | `boolean` | `false` | — |
| `tick_count` | no | `integer` | `4` | minimum: `2`; maximum: `12` |
| `tick_format` | no | `string` | `"auto"` | — |
| `label` | no | `string` | `"auto"` | — |

### `ComparisonAlignment`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `time` | no | `'strict' \| 'intersection' \| 'interpolate_to_baseline' \| 'interpolate_to_comparison'` | `"strict"` | — |
| `frequency` | no | `'strict' \| 'intersection' \| 'interpolate_to_baseline' \| 'interpolate_to_comparison'` | `"strict"` | — |
| `space` | no | `'strict' \| 'intersection' \| 'interpolate_to_baseline' \| 'interpolate_to_comparison'` | `"strict"` | — |
| `wavenumber` | no | `'strict' \| 'intersection' \| 'interpolate_to_baseline' \| 'interpolate_to_comparison'` | `"strict"` | — |
| `time_tolerance_s` | no | `number` | `1e-15` | minimum: `0.0` |
| `frequency_tolerance_hz` | no | `number` | `1e-09` | minimum: `0.0` |
| `space_tolerance_m` | no | `number` | `1e-12` | minimum: `0.0` |
| `wavenumber_tolerance_rad_m` | no | `number` | `1e-09` | minimum: `0.0` |

### `ComparisonReference`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `analysis_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `archived_run_id` | no | `string \| null` | `null` | — |

### `ContourAxesPresentation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `x_tick_format` | no | `string` | `"auto"` | — |
| `y_tick_format` | no | `string` | `"auto"` | — |

### `ContourAxesPresentationOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `x_tick_format` | no | `string \| null` | `null` | — |
| `y_tick_format` | no | `string \| null` | `null` | — |

### `ContourConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `fields` | no | `mapping[string, ContourStyleOverride]` | — | — |

### `ContourRange`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `mode` | no | `'fixed' \| 'per_snapshot_percentile' \| 'selected_snapshots_minmax' \| 'selected_snapshots_percentile'` | `"per_snapshot_percentile"` | — |
| `minimum` | no | `number \| null` | `null` | — |
| `maximum` | no | `number \| null` | `null` | — |
| `lower_percentile` | no | `number` | `1.0` | minimum: `0.0`; maximum: `100.0` |
| `upper_percentile` | no | `number` | `99.0` | minimum: `0.0`; maximum: `100.0` |

### `ContourRendering`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `mode` | no | `'continuous' \| 'discrete'` | `"continuous"` | — |
| `levels` | no | `integer \| array[number] \| null` | `null` | — |

### `ContourRenderingOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `mode` | no | `'continuous' \| 'discrete' \| null` | `null` | — |
| `levels` | no | `integer \| array[number] \| null` | `null` | — |

### `ContourStyle`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `colormap` | no | `string` | `"viridis"` | — |
| `normalization` | no | `'linear' \| 'log' \| 'symlog'` | `"linear"` | — |
| `range` | no | `ContourRange` | `{"lower_percentile": 1.0, "maximum": null, "minimum": null, "mode": "per_snapshot_percentile", "upper_percentile": 99.0}` | — |
| `colorbar` | no | `ColorbarPresentation` | `{"include_endpoints": false, "label": "auto", "length_fraction": 0.43, "position": "top", "thickness_fraction": null, "tick_count": 4, "tick_format": "auto"}` | — |
| `rendering` | no | `ContourRendering` | `{"levels": null, "mode": "continuous"}` | — |
| `symmetric_about_zero` | no | `boolean` | `false` | — |
| `symlog_linear_threshold` | no | `number \| null` | `null` | — |

### `ContourStyleOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `colormap` | no | `string \| null` | `null` | — |
| `normalization` | no | `'linear' \| 'log' \| 'symlog' \| null` | `null` | — |
| `range` | no | `ContourRange \| null` | `null` | — |
| `colorbar` | no | `ColorbarOverride \| null` | `null` | — |
| `rendering` | no | `ContourRenderingOverride \| null` | `null` | — |
| `symmetric_about_zero` | no | `boolean \| null` | `null` | — |
| `symlog_linear_threshold` | no | `number \| null` | `null` | — |

### `ControlVolumeConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `x_range_m` | yes | `tuple[number, number]` | — | minimum items: `2`; maximum items: `2` |
| `y_top_m` | yes | `number` | — | greater than: `0` |
| `bulk_viscosity_pa_s` | no | `number` | `0.0` | minimum: `0.0` |

### `DirectionalWaveAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
| `recipe` | yes | `'directional_wave'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `frequency_min_hz` | no | `number` | `0.0` | minimum: `0.0` |
| `frequency_max_hz` | yes | `number` | — | greater than: `0` |
| `expected_speed_min_m_s` | no | `number \| null` | `null` | — |
| `expected_speed_max_m_s` | no | `number \| null` | `null` | — |
| `minimum_coherence` | no | `number` | `0.8` | minimum: `0.0`; maximum: `1.0` |
| `spatial_window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"hann"` | — |
| `temporal_window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"rectangular"` | — |
| `temporal_wavenumber` | no | `TemporalWavenumberDisabled \| TemporalWavenumberEnabled` | — | discriminator: `enabled` |

### `FigurePresentation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `formats` | no | `array['png' \| 'pdf' \| 'svg']` | `["png"]` | — |
| `dpi` | no | `integer` | `300` | greater than: `0` |
| `width_in` | no | `number` | `14.0` | greater than: `0` |
| `height_in` | no | `number` | `4.5` | greater than: `0` |
| `transparent` | no | `boolean` | `false` | — |

### `FigurePresentationOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `formats` | no | `array['png' \| 'pdf' \| 'svg'] \| null` | `null` | — |
| `dpi` | no | `integer \| null` | `null` | — |
| `width_in` | no | `number \| null` | `null` | — |
| `height_in` | no | `number \| null` | `null` | — |
| `transparent` | no | `boolean \| null` | `null` | — |

### `FlowOverviewAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `recipe` | yes | `'flow_overview'` | — | — |
| `fields` | no | `array[Variable]` | `["temperature", "pressure"]` | minimum items: `1` |
| `snapshot_start` | no | `integer \| null` | `null` | — |
| `snapshot_end` | no | `integer \| null` | `null` | — |
| `snapshot_step` | no | `integer` | `1` | greater than: `0` |
| `x_limits_m` | no | `tuple[number, number] \| null` | `null` | — |
| `y_limits_m` | no | `tuple[number, number] \| null` | `null` | — |
| `line_stations_x_m` | no | `array[number]` | `[]` | — |
| `contours` | no | `ContourConfig` | `{"fields": {}}` | — |
| `line_profiles` | no | `LineProfileConfig` | `{"coordinate_limits": null, "coordinate_range_m": null, "coordinate_scale": null, "fields": {}, "grid": null, "interpolation": "linear", "layout": "separate_fields", "legend_position": null, "normalize_values": false, "sample_points": null, "value_limits": null, "value_scale": null}` | — |
| `streamlines` | no | `boolean` | `false` | — |

### `ForceProbeLinkageConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
| `variable` | no | `Variable` | `"pressure"` | — |
| `force_component` | no | `'x' \| 'y' \| 'moment'` | `"y"` | — |
| `forcing_frequency_hz` | yes | `number` | — | greater than: `0` |
| `minimum_forcing_periods` | no | `number` | `10.0` | greater than: `0` |
| `welch_segment_samples` | no | `integer` | `16384` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.5` | minimum: `0.0`; less than: `1.0` |
| `minimum_segments` | no | `integer` | `8` | greater than: `0` |
| `probe_indices` | no | `array[integer]` | `[]` | — |

### `LineProfileConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `coordinate_range_m` | no | `tuple[number, number] \| null` | `null` | — |
| `interpolation` | no | `'linear' \| 'nearest'` | `"linear"` | — |
| `sample_points` | no | `integer \| null` | `null` | — |
| `layout` | no | `'separate_fields' \| 'combined'` | `"separate_fields"` | — |
| `normalize_values` | no | `boolean` | `false` | — |
| `coordinate_limits` | no | `tuple[number, number] \| null` | `null` | — |
| `value_limits` | no | `tuple[number, number] \| null` | `null` | — |
| `coordinate_scale` | no | `'linear' \| 'log' \| 'symlog' \| null` | `null` | — |
| `value_scale` | no | `'linear' \| 'log' \| 'symlog' \| null` | `null` | — |
| `grid` | no | `boolean \| null` | `null` | — |
| `legend_position` | no | `'best' \| 'upper_left' \| 'upper_right' \| 'lower_left' \| 'lower_right' \| null` | `null` | — |
| `fields` | no | `mapping[string, LineStyleOverride]` | — | — |

### `LineStyle`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `linewidth` | no | `number` | `2.0` | greater than: `0` |
| `linestyle` | no | `'solid' \| 'dashed' \| 'dashdot' \| 'dotted'` | `"solid"` | — |
| `marker` | no | `'none' \| 'circle' \| 'square' \| 'triangle' \| 'diamond'` | `"none"` | — |
| `color` | no | `string \| null` | `null` | — |
| `grid` | no | `boolean` | `true` | — |
| `legend_position` | no | `'best' \| 'upper_left' \| 'upper_right' \| 'lower_left' \| 'lower_right'` | `"best"` | — |
| `coordinate_scale` | no | `'linear' \| 'log' \| 'symlog'` | `"linear"` | — |
| `value_scale` | no | `'linear' \| 'log' \| 'symlog'` | `"linear"` | — |

### `LineStyleOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `linewidth` | no | `number \| null` | `null` | — |
| `linestyle` | no | `'solid' \| 'dashed' \| 'dashdot' \| 'dotted' \| null` | `null` | — |
| `marker` | no | `'none' \| 'circle' \| 'square' \| 'triangle' \| 'diamond' \| null` | `null` | — |
| `color` | no | `string \| null` | `null` | — |
| `grid` | no | `boolean \| null` | `null` | — |
| `legend_position` | no | `'best' \| 'upper_left' \| 'upper_right' \| 'lower_left' \| 'lower_right' \| null` | `null` | — |
| `coordinate_scale` | no | `'linear' \| 'log' \| 'symlog' \| null` | `null` | — |
| `value_scale` | no | `'linear' \| 'log' \| 'symlog' \| null` | `null` | — |

### `ModalScreeningAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
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
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
| `recipe` | yes | `'nonlinear_coupling'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `segment_samples` | no | `integer` | `8192` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.5` | minimum: `0.0`; less than: `1.0` |
| `surrogate_count` | no | `integer` | `499` | minimum: `19` |
| `fdr_alpha` | no | `number` | `0.05` | greater than: `0.0`; less than: `1.0` |
| `frequency_max_hz` | yes | `number` | — | greater than: `0` |
| `target_frequencies_hz` | no | `array[number]` | `[]` | — |
| `automatic_frequency_selection` | no | `boolean` | `false` | — |

### `PresentationConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `preset` | no | `'publication'` | `"publication"` | — |
| `figure` | no | `FigurePresentation` | `{"dpi": 300, "formats": ["png"], "height_in": 4.5, "transparent": false, "width_in": 14.0}` | — |
| `typography` | no | `TypographyPresentation` | `{"axes_label_size": 16.0, "base_size": 14.0, "colorbar_label_pad": 2.0, "colorbar_label_size": 13.0, "colorbar_tick_label_size": 11.0, "colorbar_tick_pad": 2.0, "font_family": "DejaVu Sans", "legend_size": 13.0, "tick_label_size": 14.0}` | — |
| `contour_axes` | no | `ContourAxesPresentation` | `{"x_tick_format": "auto", "y_tick_format": "auto"}` | — |
| `time_annotation` | no | `TimeAnnotationPresentation` | `{"boxed": true, "enabled": true, "position": "top_left", "precision": 4}` | — |
| `contour_defaults` | no | `ContourStyle` | `{"colorbar": {"include_endpoints": false, "label": "auto", "length_fraction": 0.43, "position": "top", "thickness_fraction": null, "tick_count": 4, "tick_format": "auto"}, "colormap": "viridis", "normalization": "linear", "range": {"lower_percentile": 1.0, "maximum": null, "minimum": null, "mode": "per_snapshot_percentile", "upper_percentile": 99.0}, "rendering": {"levels": null, "mode": "continuous"}, "symlog_linear_threshold": null, "symmetric_about_zero": false}` | — |
| `line_defaults` | no | `LineStyle` | `{"color": null, "coordinate_scale": "linear", "grid": true, "legend_position": "best", "linestyle": "solid", "linewidth": 2.0, "marker": "none", "value_scale": "linear"}` | — |

### `PresentationOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `figure` | no | `FigurePresentationOverride \| null` | `null` | — |
| `typography` | no | `TypographyPresentationOverride \| null` | `null` | — |
| `contour_axes` | no | `ContourAxesPresentationOverride \| null` | `null` | — |
| `time_annotation` | no | `TimeAnnotationOverride \| null` | `null` | — |
| `contour_defaults` | no | `ContourStyleOverride \| null` | `null` | — |
| `line_defaults` | no | `LineStyleOverride \| null` | `null` | — |

### `ProbePlottingConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `mode` | no | `'overlay' \| 'panels' \| 'both'` | `"both"` | — |
| `normalization` | no | `'none' \| 'per_probe_peak'` | `"none"` | — |
| `label` | no | `'index_coordinates' \| 'coordinates' \| 'index'` | `"index_coordinates"` | — |

### `ProbeSpectrumAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
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
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
| `recipe` | yes | `'single_pulse_response'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `energy_per_pulse_j_m` | yes | `number` | — | greater than: `0` |
| `pulse_fwhm_s` | yes | `number` | — | greater than: `0` |
| `pulse_period_s` | yes | `number` | — | greater than: `0` |
| `start_time_s` | no | `number` | `0.0` | — |
| `cutoff_sigma` | no | `number` | `4.0` | greater than: `0` |
| `baseline_end_time_s` | no | `number \| null` | `null` | — |
| `minimum_baseline_samples` | no | `integer` | `8` | greater than: `0` |
| `minimum_relative_source_amplitude` | no | `number` | `0.001` | greater than: `0.0`; less than: `1.0` |
| `frequency_max_hz` | no | `number \| null` | `null` | — |

### `SurfaceArcLocation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'arc_length'` | `"arc_length"` | — |
| `value_m` | yes | `number` | — | minimum: `0.0` |

### `SurfaceDiagnosticsAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `recipe` | yes | `'surface_diagnostics'` | — | — |
| `snapshot_start` | no | `integer \| null` | `null` | — |
| `snapshot_end` | no | `integer \| null` | `null` | — |
| `normal_sample_distance_m` | no | `number` | `0.001` | greater than: `0` |
| `normal_sample_points` | no | `integer` | `8` | greater than: `0` |
| `normal_profiles` | no | `SurfaceNormalProfiles \| null` | `null` | — |
| `geometry_figure` | no | `SurfaceGeometryFigure` | `{"maximum_normal_arrows": 40, "normal_arrow_length": "sample_distance", "normal_color": "tab:orange", "surface_color": "black"}` | — |

### `SurfaceGeometryFigure`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `maximum_normal_arrows` | no | `integer` | `40` | greater than: `0` |
| `normal_arrow_length` | no | `'sample_distance' \| number` | `"sample_distance"` | — |
| `normal_color` | no | `string` | `"tab:orange"` | — |
| `surface_color` | no | `string` | `"black"` | — |

### `SurfaceNormalProfiles`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `fields` | yes | `array[Variable]` | — | minimum items: `1` |
| `interpolation` | no | `'linear' \| 'nearest'` | `"linear"` | — |
| `spacing` | no | `'uniform' \| 'wall_clustered'` | `"uniform"` | — |
| `clustering_exponent` | no | `number` | `2.0` | greater than: `0` |
| `include_wall_extrapolation` | no | `boolean` | `true` | — |
| `stations` | yes | `array[SurfaceNormalStation]` | — | minimum items: `1` |
| `figure` | no | `SurfaceProfileFigure` | `{"coordinate_scale": "linear", "grid": true, "layout": "separate_fields", "normalize_values": false, "value_scale": "linear"}` | — |

### `SurfaceNormalStation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `location` | yes | `SurfaceXLocation \| SurfaceArcLocation` | — | discriminator: `type` |
| `component_id` | no | `string \| integer \| null` | `null` | — |
| `side_id` | no | `string \| null` | `null` | — |
| `distance_m` | no | `number \| null` | `null` | — |
| `sample_points` | no | `integer \| null` | `null` | — |

### `SurfaceProfileFigure`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `layout` | no | `'separate_fields' \| 'combined'` | `"separate_fields"` | — |
| `normalize_values` | no | `boolean` | `false` | — |
| `coordinate_scale` | no | `'linear' \| 'log' \| 'symlog'` | `"linear"` | — |
| `value_scale` | no | `'linear' \| 'log' \| 'symlog'` | `"linear"` | — |
| `grid` | no | `boolean` | `true` | — |

### `SurfaceXLocation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `type` | no | `'x'` | `"x"` | — |
| `value_m` | yes | `number` | — | — |

### `TemporalWavenumberDisabled`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `enabled` | no | `False` | `false` | — |

### `TemporalWavenumberEnabled`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `enabled` | no | `True` | `true` | — |
| `window_duration_s` | yes | `number` | — | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.75` | minimum: `0.0`; less than: `1.0` |
| `window` | no | `'hann' \| 'hamming' \| 'blackman' \| 'rectangular'` | `"hann"` | — |
| `snapshot_times_s` | no | `array[number]` | `[]` | — |
| `minimum_relative_energy_db` | no | `number` | `-30.0` | maximum: `0.0` |
| `display_floor_db` | no | `number` | `-60.0` | maximum: `0.0` |

### `TimeAnnotationOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `enabled` | no | `boolean \| null` | `null` | — |
| `position` | no | `'top_left' \| 'top_center' \| 'top_right' \| 'bottom_left' \| 'bottom_center' \| 'bottom_right' \| 'above_axes_left' \| 'above_axes_center' \| 'above_axes_right' \| null` | `null` | — |
| `boxed` | no | `boolean \| null` | `null` | — |
| `precision` | no | `integer \| null` | `null` | — |

### `TimeAnnotationPresentation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `enabled` | no | `boolean` | `true` | — |
| `position` | no | `'top_left' \| 'top_center' \| 'top_right' \| 'bottom_left' \| 'bottom_center' \| 'bottom_right' \| 'above_axes_left' \| 'above_axes_center' \| 'above_axes_right'` | `"top_left"` | — |
| `boxed` | no | `boolean` | `true` | — |
| `precision` | no | `integer` | `4` | minimum: `1`; maximum: `12` |

### `TransientWavepacketAnalysis`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `id` | yes | `string` | — | pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `enabled` | no | `boolean` | `true` | — |
| `presentation` | no | `PresentationOverride \| null` | `null` | — |
| `probe_set_id` | yes | `string` | — | minimum length: `1`; pattern: `^[A-Za-z0-9][A-Za-z0-9_.-]*$` |
| `probe_indices` | no | `array[integer]` | `[]` | — |
| `probe_plotting` | no | `ProbePlottingConfig` | — | — |
| `record_start_time_s` | no | `number \| null` | `null` | — |
| `end_time_s` | no | `number \| null` | `null` | — |
| `time_grid_policy` | no | `'resample_uniform' \| 'require_uniform'` | `"resample_uniform"` | — |
| `recipe` | yes | `'transient_wavepacket'` | — | — |
| `variable` | yes | `Variable` | — | — |
| `band_min_hz` | yes | `number` | — | minimum: `0.0` |
| `band_max_hz` | yes | `number` | — | greater than: `0` |
| `baseline_end_time_s` | no | `number \| null` | `null` | — |
| `stft_segment_samples` | no | `integer` | `2048` | greater than: `0` |
| `overlap_fraction` | no | `number` | `0.75` | minimum: `0.0`; less than: `1.0` |

### `TypographyPresentation`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `font_family` | no | `string` | `"DejaVu Sans"` | — |
| `base_size` | no | `number` | `14.0` | greater than: `0` |
| `axes_label_size` | no | `number` | `16.0` | greater than: `0` |
| `tick_label_size` | no | `number` | `14.0` | greater than: `0` |
| `legend_size` | no | `number` | `13.0` | greater than: `0` |
| `colorbar_label_size` | no | `number` | `13.0` | greater than: `0` |
| `colorbar_tick_label_size` | no | `number` | `11.0` | greater than: `0` |
| `colorbar_label_pad` | no | `number` | `2.0` | minimum: `0.0`; maximum: `24.0` |
| `colorbar_tick_pad` | no | `number` | `2.0` | minimum: `0.0`; maximum: `24.0` |

### `TypographyPresentationOverride`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `font_family` | no | `string \| null` | `null` | — |
| `base_size` | no | `number \| null` | `null` | — |
| `axes_label_size` | no | `number \| null` | `null` | — |
| `tick_label_size` | no | `number \| null` | `null` | — |
| `legend_size` | no | `number \| null` | `null` | — |
| `colorbar_label_size` | no | `number \| null` | `null` | — |
| `colorbar_tick_label_size` | no | `number \| null` | `null` | — |
| `colorbar_label_pad` | no | `number \| null` | `null` | — |
| `colorbar_tick_pad` | no | `number \| null` | `null` | — |

### `Variable`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |

## `machine.yaml`

### `MachineFile`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `schema_version` | no | `1` | `1` | — |
| `inputs` | no | `InputConfig` | `{"archived_runs": {}, "baselines": {}, "plotfiles": null, "probe_sets": {}}` | — |
| `outputs` | yes | `OutputConfig` | — | — |
| `compute` | no | `ComputeConfig` | `{"fft_batch_size": 32, "memory_limit_gb": 8.0, "scratch_directory": null, "workers": 1}` | — |

### `ComputeConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `workers` | no | `integer` | `1` | greater than: `0` |
| `memory_limit_gb` | no | `number` | `8.0` | greater than: `0` |
| `fft_batch_size` | no | `integer` | `32` | greater than: `0` |
| `scratch_directory` | no | `string \| null` | `null` | — |

### `InputConfig`

| Field | Required | Type | Default | Rules |
| --- | --- | --- | --- | --- |
| `plotfiles` | no | `PlotfileInput \| null` | `null` | — |
| `probe_sets` | no | `mapping[string, ProbeInput]` | — | — |
| `archived_runs` | no | `mapping[string, string]` | — | — |
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
