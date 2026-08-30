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
- Artifact IDs: surface.curve, surface.samples, surface.quality, surface.figure

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
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: spectral.psd, spectral.coherence, spectral.confidence, spectral.figure

Assumptions:

- The selected record is approximately stationary.

Interpretation limits:

- Finite records and windowing limit frequency discrimination.

## `single_pulse_response`

Physical question: What response follows a known finite laser pulse?

Quiescent-baseline and finite-record source/response deconvolution.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: pulse.source_spectrum, pulse.transfer, pulse.validity

Assumptions:

- The configured pulse model represents the source history.

Interpretation limits:

- Small source amplitudes are masked rather than inverted.

## `directional_wave`

Physical question: What direction, wavelength, phase speed, and amplification are measured?

Spatial FFT, coherence-gated complex wavenumber, amplification, and k-omega.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: wave.spatial_spectrum, wave.wavenumber, wave.komega, wave.komega_sensitivity, wave.komega.figure

Assumptions:

- Probe coordinates form a suitable approximately uniform aperture.

Interpretation limits:

- Measurements alone do not constitute LST/PSE or causal evidence.

## `transient_wavepacket`

Physical question: How does a transient packet arrive and propagate?

STFT, filtered envelopes, arrival times, group velocity, and uncertainty.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: transient.stft, transient.envelope, transient.group_velocity

Assumptions:

- A localized packet exists in the selected time-frequency band.

Interpretation limits:

- Arrival detection depends on bandwidth and signal-to-noise ratio.

## `nonlinear_coupling`

Physical question: Are statistically significant quadratic frequency interactions measured?

Surrogate and FDR-controlled bicoherence and triad screening.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: nonlinear.bicoherence, nonlinear.triads

Assumptions:

- Enough independent segments exist for surrogate inference.

Interpretation limits:

- Bicoherence is association, not proof of causal energy transfer.

## `modal_screening`

Physical question: Which coherent low-rank structures are visible in the measurements?

Weighted POD, SPOD, DMD, and rank/window conditioning sensitivity.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: probes
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: modal.pod, modal.spod, modal.dmd, modal.sensitivity

Assumptions:

- The probe measure and weights are meaningful for the question.

Interpretation limits:

- Probe modes are measurement modes, not full-domain eigenmodes.

## `case_comparison`

Physical question: How do compatible products differ between two registered runs?

Schema- and provenance-checked comparison of existing artifacts.

- Dimensions: 2-D
- Geometries: flat_plate, wedge, polyline, volume_fraction
- Required inputs: comparison_archives
- Required plotfile fields: none
- Dependencies: none
- Conflicts: none
- Artifact IDs: comparison.metrics, comparison.figures

Assumptions:

- Compared artifacts use compatible coordinates and preprocessing.

Interpretation limits:

- Incompatible artifacts are rejected rather than interpolated silently.

All current numerical recipes support 2-D only. Inspection recognizes 3-D datasets,
but planning blocks unsupported algorithms before expensive loading.
