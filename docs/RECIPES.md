# Recipe reference

This file is generated from the central workflow registry. `pelec-post recipes show NAME` presents the same contract.

## `flow_overview`

Physical question: What does the resolved two-dimensional flow field look like?

Contours, derived fields, line samples, and optional streamlines.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: plotfiles
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: field.contours, field.lines, field.streamlines

Assumptions:

- Requested fields are represented by the plotfile data.

Interpretation limits:

- Images are descriptive and do not establish causality.

## `boundary_layer_reference`

Physical question: How does a flat-plate boundary layer compare with a laminar reference?

Boundary-layer integral quantities and compressible similarity profiles.

- Dimensions: 2-D
- Geometries: flat_plate
- Required inputs: plotfiles
- Required plotfile fields: density, x_velocity, temperature
- Dependencies: none
- Conflicts: none
- Artifact IDs: boundary_layer.profiles, boundary_layer.thickness, boundary_layer.gip

Assumptions:

- Laminar flow.
- Zero streamwise pressure gradient.

Interpretation limits:

- The similarity curve is a reference, not an LST or PSE result.

## `surface_diagnostics`

Physical question: Is the reconstructed wall geometry and near-wall sampling trustworthy?

Surface coordinates, normals, wall samples, and fit-quality diagnostics.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: plotfiles
- Required plotfile fields: pressure, temperature, x_velocity, y_velocity
- Dependencies: geometry.surface
- Conflicts: none
- Artifact IDs: surface.curve, surface.samples, surface.quality, surface.figure, surface.normal_profile

Assumptions:

- A unique fluid-facing surface normal can be established.

Interpretation limits:

- Resolution and normal-fit quality constrain wall quantities.

## `aerodynamic_forces`

Physical question: What pressure, viscous, thermal, force, and moment loads act on the body?

Wall reconstruction, component integration, baselines, and sensitivity.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: plotfiles
- Required plotfile fields: pressure, temperature, x_velocity, y_velocity
- Dependencies: geometry.surface
- Conflicts: none
- Artifact IDs: forces.history, forces.components, forces.sensitivity, forces.control_volume, forces.probe_linkage

Assumptions:

- Newtonian stress and configured transport properties apply.

Interpretation limits:

- General EB output is validated_2d_eb_v1, not externally certified.

## `probe_spectrum`

Physical question: What stationary frequency content is present in the probe measurements?

One-sided FFT/Welch spectra, coherence, and confidence summaries.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: spectral.psd, spectral.coherence, spectral.confidence, spectral.figure, spectral.probe_signals, spectral.probe_figure, spectral.fft_overlay.figure, spectral.psd_overlay.figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- The selected record is approximately stationary.

Interpretation limits:

- Finite records and windowing limit frequency discrimination.

## `single_pulse_response`

Physical question: What response follows a known finite laser pulse?

Quiescent-baseline and finite-record source/response deconvolution.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: pulse.source_audit, pulse.source_audit.figure, pulse.source_spectrum, pulse.transfer, pulse.validity, pulse.transfer_magnitude.figure, pulse.transfer_phase.figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- A configured measured source history is complete and represents the discrete thermal deposition;
- without one, results are explicitly modeled-source-only.

Interpretation limits:

- Small source amplitudes are masked rather than inverted.

## `directional_wave`

Physical question: What direction, wavelength, phase speed, and amplification are measured?

Spatial FFT, coherence-gated complex wavenumber, amplification, k-omega, and optional time-localized pulse ridges.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: wave.spatial_spectrum, wave.wavenumber, wave.komega, wave.komega_sensitivity, wave.komega.figure, wave.temporal_wavenumber, wave.komega_snapshots, wave.space_time.figure, wave.temporal_wavenumber.figure, wave.wavenumber_history.figure, wave.komega_snapshots.figure, wave.dispersion.figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- Probe coordinates form a suitable approximately uniform aperture.

Interpretation limits:

- Measurements alone do not constitute LST/PSE or causal evidence.

## `transient_wavepacket`

Physical question: How does a transient packet arrive and propagate?

STFT, filtered envelopes, arrival times, group velocity, and uncertainty.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: transient.stft, transient.envelope, transient.group_velocity, transient.filtered_overlay.figure, transient.envelope_overlay.figure, transient.stft_figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- A localized packet exists in the selected time-frequency band.

Interpretation limits:

- Arrival detection depends on bandwidth and signal-to-noise ratio.

## `nonlinear_coupling`

Physical question: Are statistically significant quadratic frequency interactions measured?

Surrogate and FDR-controlled bicoherence and triad screening.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: nonlinear.bicoherence, nonlinear.triads, nonlinear.bicoherence.figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- Enough independent segments exist for surrogate inference.

Interpretation limits:

- Bicoherence is association, not proof of causal energy transfer.

## `modal_screening`

Physical question: Which coherent low-rank structures are visible in the measurements?

Weighted POD, SPOD, DMD, and rank/window conditioning sensitivity.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probe_sets
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: modal.pod, modal.spod, modal.dmd, modal.sensitivity, modal.pod.figure, modal.spod.figure, modal.dmd.figure, probe.raw_history.figure, probe.method_ready.figure

Assumptions:

- The probe measure and weights are meaningful for the question.

Interpretation limits:

- Probe modes are measurement modes, not full-domain eigenmodes.

## `case_comparison`

Physical question: How do compatible products differ between two registered runs?

Schema- and provenance-checked comparison of existing artifacts.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: local analysis dependencies and/or archived_runs
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: comparison.metrics, comparison.figures, comparison.overlay_figure

When the selected products include `spectral.probe_signals`, the comparison
also writes paired raw histories, a 2 µs history zoom, per-probe FFT amplitude
overlays, separately normalized FFT shapes, and a gated normalized-shape ratio
in dB. Probe panels omit locations that have zero disturbance in both cases.
`spectral.psd` adds paired Welch PSD curves. Put `wave.komega` first to
make the main comparison figure
a shared-scale signed frequency–wavenumber map with a logarithmic positive-
frequency axis and power-ratio panel; it
also adds absolute and normalized band-integrated signed-k curves. Ratio-map
pixels below 40 dB of the shared peak in either case are masked. The summary
chart uses relative L2 differences so quantities with different physical
units are not placed on a common absolute-difference axis.

Assumptions:

- Compared artifacts use compatible coordinates and preprocessing.

Interpretation limits:

- Incompatible artifacts are rejected rather than interpolated silently.

All current numerical recipes support 2-D only. Inspection recognizes 3-D datasets,
but planning blocks unsupported algorithms before expensive loading.

## Probe histories with separate panel scales

`probe_spectrum`, `single_pulse`, and other probe recipes can save named raw
history views alongside the regular overlay. Set `probe_plotting.trace_views`
to choose a time range and probe groups. Each group gets its own vertical axis;
`value_limits` is optional when a fixed axis is needed. For one two-probe group,
`pair_difference: true` adds a difference panel. The view reads only the record
selected by the analysis's `record_start_time_s` and `end_time_s`.

```yaml
probe_plotting:
  mode: both
  normalization: none
  label: index
  trace_views:
    - id: pulse-symmetry
      title: Symmetric near-source probe response
      time_end_s: 1.25e-7
      groups:
        - title: Near-source
          probe_indices: [150, 170]
      pair_difference: true
    - id: symmetric-pairs
      title: Symmetric probe pairs on separate vertical scales
      time_end_s: 1.0e-5
      groups:
        - title: Near-source
          probe_indices: [150, 170]
        - title: Outer
          probe_indices: [120, 200]
```

The resulting figures are `trace_pulse-symmetry.png` and
`trace_symmetric-pairs.png` in the analysis figure directory. These settings
are already present in the focused `Flow-Viz/Post-Processing/analyses.yaml`
recipe for both kernel cases and both pressure and temperature.
