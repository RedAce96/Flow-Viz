# Reviewed baseline

The usability redesign starts from commit `cf286e0`, which contains the
`40749bc` Isaac V3/kernel-spectrum merge and its reviewed numerical fixes.

## Verification

Run on 2026-08-30 with Python 3.11:

```text
python3 -m unittest test_engineering
Ran 102 tests in 12.039s
OK (skipped=1)
```

The skip is an optional environment-dependent integration check. No test
failed.

## Existing contracts to preserve numerically

- PeleC/AMReX field loading and CGS-to-SI conversion.
- Native-patch AMR vorticity in two dimensions.
- Restart-aware binary probes and compact HDF5 probe reads.
- Certified one-sided flat-plate force extraction.
- Probe FFT, coherence, pulse deconvolution, spatial FFT, and k-omega results.
- Transient, nonlinear, POD/SPOD/DMD, and evidence-screening calculations.

The redesign is a clean break for configuration, commands, and artifact paths;
it is not permission to silently change these reviewed numerical conventions.

## Known interface defects to eliminate

1. Probe-only execution still discovers and requires an AMReX plotfile.
2. Modal products are written below `FFT-Probes/ModalAnalysis`, while the
   evidence reader searches `FFTProbes/ModalAnalysis`.
3. Child analysis toggles can be enabled while their parent workflow is off,
   causing them to be silently skipped.
4. Reused output directories allow stale products to appear in a later run's
   manifest.
5. Output filenames are an undocumented compatibility surface.

An untracked workspace test expected the former unsuffixed
`spectral_summary.npz` name after variable-specific names were introduced.
That file is not part of this branch, but it is evidence that artifact schemas
must be explicit and versioned.
