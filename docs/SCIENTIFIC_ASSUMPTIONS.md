# Scientific assumptions and interpretation

Freestream normalization is never inferred from a corner cell. It comes from explicit SI case values or a configured region statistic, and its provenance accompanies normalized artifacts.

Stationary Welch, one-pulse finite-record transfer, transient STFT/envelope, and harmonic reconstruction are distinct estimators. Nyquist compliance does not imply adequate resolution: segment count, effective degrees of freedom, aperture, native wavenumber spacing, coherence, and accepted-fit fractions must also be reviewed.

Positive directional wavenumber follows `cos(omega*t-k*x)`. Complex spatial wavenumber uses the documented `exp(i*(alpha*x-omega*t))` convention, so `-alpha_i` is amplification. Probe-derived waves and modes are measurement descriptions; they are not LST/PSE eigenmodes and do not establish causality.

Bicoherence is retained only with surrogate and false-discovery controls and still indicates association. Packet speed is a kinematic envelope regression with a confidence interval. Force conclusions inherit geometry coverage, normal-fit residual, grid/fit sensitivity, transport, baseline, reference area/span, and moment-origin assumptions.

The optional flat-plate control-volume result is a steady momentum balance and
omits unsteady storage. It is an independent discrepancy diagnostic, not a
replacement for wall integration. Force–probe linkage synchronizes the common
record and reports correlation/coherence/transfer measurements; association
does not establish that the probe fluctuation caused the force response.
