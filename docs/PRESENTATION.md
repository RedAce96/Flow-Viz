# Figure presentation and profile extraction

Presentation settings live in `analyses.yaml`. Top-level `presentation` values
provide reusable project defaults; an analysis may include its own
`presentation` object, and contour/line field entries are the final override.
All keys are strict and are checked by `pelec-post validate`.

## Contours

`flow_overview.contours.fields` accepts one style per canonical field. A style
may choose any installed Matplotlib colormap, `linear`, `log`, or `symlog`
normalization, fixed or percentile limits, symmetric signed limits, and
continuous or discrete rendering.

Use a fixed range when exact visual comparison between cases is required. Use
`selected_snapshots_minmax` or `selected_snapshots_percentile` to share a range
across every selected plotfile in one analysis. Shared percentile calculation
uses a deterministic bounded sample and records the resolved limits in artifact
provenance. `per_snapshot_percentile` is useful for exploration but can make
amplitude changes between frames visually misleading.

Logarithmic contours require all selected values and both limits to be strictly
positive. Signed fields such as vorticity should normally use a diverging map,
limits symmetric about zero, and either linear or `symlog` normalization.

The time annotation always occupies an external header or footer row. Its six
positions are `top_left`, `top_center`, `top_right`, `bottom_left`,
`bottom_center`, and `bottom_right`. Colorbars use separate top, bottom, left,
or right axes. A top or bottom colorbar is horizontal; a left or right colorbar
is vertical. `colorbar.length_fraction` is a visual scale from `0.20` to
`0.75`: it controls both the long-axis length and colored-strip thickness,
without changing tick or label typography. Use smaller values for very wide,
shallow contours; `0.33` is a useful compact starting point. Set optional
`colorbar.thickness_fraction` only when you deliberately want to decouple the
two dimensions. This layout keeps the time box, colorbar, labels, and contour
data from overlapping. Set `include_endpoints: true` with `tick_count: 4` to
show a compact, evenly spaced tick set that retains the resolved minimum and
maximum rather than letting a short bar display only interior values.

`presentation.contour_axes.x_tick_format` and `y_tick_format` accept standard
Matplotlib numeric format strings. For example, `".3g"` writes the wall as
`0` rather than `0.000` while retaining `0.005` m where that precision matters.

## Cartesian line profiles

Set stations with `line_stations_x_m` and extraction/presentation options with
`line_profiles`. Coordinate cropping occurs before optional linear or nearest
resampling. The numerical samples are saved as CSV with requested and actual
nearest-grid x coordinates in artifact metadata.

The default `separate_fields` layout creates one figure per variable, avoiding
axes that mix incompatible units. `combined` is accepted only for compatible
units or when `normalize_values: true` explicitly maps each curve to `[0, 1]`.
Per-field color, linestyle, width, marker, and axis presentation are supported.

## Surface-normal profiles

`surface_diagnostics.normal_profiles` adds selected geometry-aware profiles
without changing the full-surface samples used for wall fitting and forces.
Flat/simple surfaces can use an x location. Wedges, polylines, disconnected EB
components, or any geometry with multiple x intersections should use component,
side, and arc length. Ambiguous x locations are rejected.

Raw samples begin half a local grid spacing into the fluid. Uniform and
wall-clustered spacing are available. If `include_wall_extrapolation` is true,
the output also contains a fitted zero-distance value explicitly marked as an
extrapolation. Every station produces CSV and compressed NPZ data plus styled
figures. The geometry figure independently controls surface color, normal color,
maximum arrow count, and arrow length.

See `examples/flow-overview` and `examples/wedge-eb-forces` for complete YAML.
