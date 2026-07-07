#!/usr/bin/env python3
"""Focused test: probe conversion + FFT + stability diagnostics only."""
import subprocess
import sys
from pathlib import Path

import numpy as np

# Import pelec_post to get CONFIG and the analysis functions.
sys.path.insert(0, str(Path(__file__).parent))
import pelec_post as pp


def main():
    config = pp.CONFIG

    # Ensure we use the binary direct-read path for this test.
    config["fft_use_binary"] = True
    config["make_probe_plots"] = False
    config["make_fft_probes"] = True
    config["make_stability_diagnostics"] = True
    config["fft_max_probes"] = None
    config["fft_plot_contour"] = False

    # Quick sanity check on binary time span
    bin_files = config.get("probe_bin_files", [])
    print(f"Using binary probe files: {bin_files}")

    # --- FFT probes -------------------------------------------------------
    if config.get("make_fft_probes", False):
        print("\nRunning FFT probe analysis ...")
        r = pp._process_fft_probes(config)
        print(f"  FFT result: {r}")

    # --- Stability diagnostics --------------------------------------------
    if config.get("make_stability_diagnostics", False):
        print("\nRunning stability diagnostics ...")
        r = pp._process_stability_diagnostics(config)
        print(f"  Stability result: {r}")


if __name__ == "__main__":
    main()
