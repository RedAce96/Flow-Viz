# Kernel spectral comparison server study

This study supplies evidence that Flow Viz cannot obtain from the transferred
products. It is a specification only; it does not change the CFD solver or
claim that the current run is converged.

## Required evidence before rerunning

Record the solver revision, Flow Viz revision, probe interpolation revision,
mesh hierarchy and timestep, remeshing cadence, source center and geometry,
requested and measured deposited energy, source temporal history, and the
probe-line coordinates for both kernels. Transfer matching 321-probe raw
histories for asymmetric pressure and temperature. Keep checksums for every
binary and source file. Resolve the quiescent-case versus flat-plate/freestream
metadata discrepancy against the actual solver inputs.

## Resolution matrix

Run the same physical domain, source parameters, probe coordinates, record
duration, and output cadence for both kernels. Let `h0` and `dt0` be the
resolved reference mesh and timestep. Save probes at every solver step.

| run | mesh | timestep |
|---|---:|---:|
| reference | h0 | dt0 |
| time-1 | h0 | dt0/2 |
| time-2 | h0 | dt0/4 |
| space-1 | h0/2 | dt0/4 |
| space-2 | h0/4 | dt0/4 |

If stability requires a smaller timestep on the finest mesh, use that timestep
for every member of the space sweep and document the added coarse-mesh run.
For AMR, record actual source-region and propagation-region resolution and
refinement coverage; do not infer a single `h` from a nominal level number.

## Record-length and forcing checks

From a common long history, produce 10, 20, and 40 microsecond records. Record
predicted and observed boundary-return times and analyze reflection-free and
post-return intervals separately. Audit deposited source energy, timing, and
spatial moments for each kernel; equal integrated energy is not equal spatial
forcing.

Repeat each kernel at 0.5 and 0.25 of nominal source strength with unchanged
kernel geometry. Compare peak, arrival, pulse-transform magnitude, phase,
symmetry residual, and f-k branch results on common domains. Record changes in
the background thermal state. Non-proportional scaling rejects a simple linear
transfer interpretation but does not identify the cause by itself.

## Required reported quantities

For every run, retain raw timestamps and field units, then report pressure
peaks and 10–90% sampled rises, temperature peaks and final values, absolute
FFT and finite-record-transform magnitudes, supported phase differences,
mirrored even/odd residuals, signed-k branch positions, and band norms. Use
matching native frequency bins and state the valid support count. Keep display
limits separate from validated bandwidth claims.

The resulting report must classify each observation as stable under tested
preprocessing, stable under tested numerical refinement, sensitive to source
forcing, or unresolved. It must not emit a blanket scientific-validation flag.

Before examining the refinement results, register a numerical acceptance table
for each reported quantity: comparison domain and support rule, absolute and
relative change tolerances, measurement uncertainty, and the refinement pairs
to be checked. Report the measured changes, valid support counts, and the
configured tolerances side by side. Assign a stability classification only
when every relevant registered criterion is met; otherwise report unresolved
or sensitive with the failing criteria. If the tolerances have not been
approved, report the measurements and classify stability as unresolved.
