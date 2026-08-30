# Recipe reference

This reference mirrors the central registry. `pelec-post recipes show NAME` is authoritative for required inputs, fields, assumptions, products, dimensions, geometries, and interpretation limits.

- `flow_overview`: contours, selected lines, and optional instantaneous streamlines from plotfiles.
- `boundary_layer_reference`: native-AMR flat-plate profiles, integral thicknesses, compressible similarity, and conservative GPI screening under laminar ZPG assumptions.
- `surface_diagnostics`: ordered curves, fluid normals, normal samples, wall fits, coverage, and residual quality.
- `aerodynamic_forces`: pressure/viscous/thermal loads, force, moment, history, fit sensitivity, and explicit none/static/paired baseline contracts.
- `probe_spectrum`: stationary one-sided Welch PSD and confidence metadata.
- `single_pulse_response`: quiescent baseline and finite-record source/response deconvolution.
- `directional_wave`: coherence-gated complex wavenumber, amplification, and signed k–omega.
- `transient_wavepacket`: STFT, filtered envelope, arrivals, group velocity, and regression confidence.
- `nonlinear_coupling`: explicit or explained automatic targets with surrogate/FDR triad screening.
- `modal_screening`: weighted POD, Welch-block SPOD, exact DMD, and rank/window conditioning sensitivity.
- `case_comparison`: strict schema/unit/coordinate/preprocessing compatibility followed by numerical differences.

All current numerical recipes support dimension 2. Inspection recognizes dimension 3, but planning blocks unsupported analysis before expensive loading.
