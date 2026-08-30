# Question-driven method selection

Use `pelec-post recipes list` to list questions and `pelec-post recipes show NAME` to read each method's assumptions and limits.

- Use `flow_overview` for descriptive fields, not mechanism claims.
- Use `boundary_layer_reference` only for a laminar, zero-pressure-gradient flat plate.
- Use `surface_diagnostics` before trusting reconstructed wall loads.
- Use `aerodynamic_forces` for pressure, shear, heat flux, force, and moment. Flat-plate output retains its reviewed certified contract; general geometry is `validated_2d_eb_v1`.
- Use `probe_spectrum` for approximately stationary records.
- Use `single_pulse_response` for the quiescent pre-event box and one finite source pulse; do not substitute Welch transfer estimates.
- Use `directional_wave` when spacing, aperture, coherence, and expected wavelength pass preflight.
- Use `transient_wavepacket` for localized arrivals rather than stationary PSD interpretation.
- Use `nonlinear_coupling` only with surrogate/FDR gates; bicoherence is association.
- Use `modal_screening` as descriptive measurement decomposition, not LST/PSE.
- Use `case_comparison` only for registered artifacts with matching schemas, units, coordinates, and preprocessing provenance; incompatible products are rejected rather than silently aligned.

Planning is part of analysis, not an optional dry run. Nyquist, record resolution, segment count, spatial aliasing, baseline size, memory, geometry coverage, and capability findings remain in the report.
