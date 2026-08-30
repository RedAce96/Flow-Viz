# Local and SLURM troubleshooting

- Exit `2`: configuration or preflight rejection. Run `validate`, `inspect`, then `plan` and resolve every blocker.
- Exit `1`: at least one runtime workflow failed. Read `logs/run.log`; completed independent products and the report remain valid within their own contracts.
- Missing `machine.yaml`: copy `machine.example.yaml` and set only server-local paths/resources.
- Spectrum rejected: compare the requested band with reported Nyquist and native/effective resolution.
- Directional analysis rejected: inspect distinct station count, spacing variation, aperture, and samples per shortest wavelength.
- Memory rejected: reduce workers, selected probes/snapshots/region, or batch size rather than bypassing the limit.

Do not edit server-specific SLURM scripts for project physics. Forward the CLI arguments through the existing wrapper, for example `run /server/path/project --name case-a`.
