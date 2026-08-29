"""Generate a checked single-pulse source-spectrum example."""

import argparse
from pathlib import Path

import numpy as np

import pp_functions_database as functions
import pp_plotting_database as plotting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="source_spectrum_single_pulse.png",
        help="destination image path",
    )
    args = parser.parse_args(argv)

    energy_per_pulse = 100.0e2  # erg/cm in the two-dimensional case
    pulse_fwhm_s = 10.0e-9
    dt = 2.0e-10
    time = np.arange(0.0, 10.0e-6, dt)
    result = functions.compute_single_pulse_source_spectrum(
        time,
        energy_per_pulse=energy_per_pulse,
        pulse_fwhm_s=pulse_fwhm_s,
        pulse_period_s=1.0e-7,
        start_time_s=0.0,
        cutoff_sigma=4.0,
        mean_subtraction="none",
        window="none",
    )
    plotting.plot_single_pulse_source_spectrum(
        result, output_path=str(Path(args.output)), fmax=200.0e6,
        energy_unit="erg/cm",
    )
    bandwidth_3db = (
        np.sqrt(np.log(2.0)) / (2.0 * np.pi * result["sigma_s"])
    )
    print(f"Pulse energy:       {energy_per_pulse:.2e} erg/cm")
    print(f"Pulse FWHM:         {pulse_fwhm_s:.2e} s")
    print(f"Amplitude -3 dB:    {bandwidth_3db * 1.0e-6:.2f} MHz")
    print(f"Sampled pulse area: {result['physical_spectrum'][0]:.8e} erg/cm")
    print(f"Saved:              {args.output}")


if __name__ == "__main__":
    main()
