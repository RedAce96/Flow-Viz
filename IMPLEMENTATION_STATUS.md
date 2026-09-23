# Flow Viz scientific comparison implementation status

This status accompanies the replay at
`Delta_Outputs/FlowViz_corrected/20260920T-scientific-final7/`. It separates
software implementation, real-data replay, and scientific evidence. The
transferred run remains unchanged.

| Register IDs | Implementation | Real-data verification | Scientific evidence |
|---|---|---|---|
| A01–A05, A08–A12 | Shared band/coordinate preparation, fixed-band norms, explicit masks, native-bin matching, and manifest-based replay selection are implemented. | 198 maintained tests pass and final7 records the selected run and Gaussian binary hashes. | These checks establish calculation provenance only; they do not establish physical validity. |
| A06, B03–B04 | Signed-k branches, k=0 handling, edge/ambiguity flags, tolerant native matching, direction selection, and aperture candidates are implemented. | Gaussian full-line inputs are checksum-verified; asymmetric full-line sensitivity is explicitly unavailable. | Matching asymmetric histories are required before both-kernel branch sensitivity can be evaluated. |
| A07, S01–S03 | Sampling audit, corrected pressure-rise indexing, endpoint/taper/timing candidates, and finite-record transform metadata are implemented. | Four selected-probe histories replay for pressure and temperature. | These are observed processing sensitivities, not mesh/timestep convergence or a validated bandwidth. |
| B01–B02, B06 | Record-length and end-taper candidates use the configured preprocessing and native grids. | Candidate arrays and metadata are present in replay diagnostics. | No safe-frequency claim is made. |
| B05, B07–B08 | Processed-signal phase/delay preparation, contiguous support intervals, and signed even/odd diagnostics are implemented. | Pressure and temperature phase/symmetry and threshold-sensitivity JSON are generated in final7. | Delay interpretation still requires common forcing and solver provenance. |
| F01–F06, N01 | Coordinate-correct f-k rendering, shared masks, gray unsupported pixels, linear ratio colors, clipping metadata, and detail/slice preparation are implemented. | Existing f-k views regenerate; the role checklist marks equivalent mappings explicitly. | New asymmetric wave-window sensitivity requires raw full-line histories. |
| K01 | Signed-k band-sum and fraction definitions and the optional norm-ratio path are implemented. | Numerical signed-k tests pass. | This is a band norm, not a PSD density or average of frequency ratios. |
| M01 | Typed band metrics, validity masks, zero-baseline unavailable states, and norm metadata are implemented. | Maintained comparison tests pass. | Values remain unsigned response summaries, not physical error estimates. |
| W01–W02 | Transient-record Welch labels/units/metadata and probe-spread conventions are implemented. | Product metadata and replay artifacts are regenerated. | Probe spread is not repeated-run uncertainty. |
| P01–P02 | Source-run/build identity, hash records, and explicit forcing/provenance dependency records are implemented. | Manifest records source run, build digest, binary verification, and unavailable wave inputs. | Solver revision, matching source histories, and complete deposition data remain unknown. |
| G01–G04, H01–H03 | Shared export/style use, 300-dpi PNG/PDF registration, mirrored ordering/scales, full histories, and rise markers are implemented. | Final7 contains a 60-role checklist: 42 direct replay views and 18 explicitly marked equivalent mappings. | The equivalent mappings need native production inputs for exact role-by-role re-rendering. |

## Replay evidence

The final7 manifest identifies source run
`2026-09-18T082531Z_QB-Study_slurm-3170807`, records the content-digest build
identity, and verifies both transferred Gaussian probe segments against their
archived SHA-256 values. `data/wave_sensitivity_unavailable.json` identifies
`asym-pressure-wave` and `asym-temperature-wave` as requiring matching
asymmetric full-line histories; magnitude-only f-k arrays are not used to
reconstruct them.

## Scientific status

The software calculations and locally available replay products are verified.
This does not establish mesh or timestep convergence, physical bandwidth,
equal spatial forcing, or a source-independent transfer function. Those claims
require matching solver inputs, source histories, and the resolution/forcing
study in `SERVER_STUDY_SPEC.md`.
