#!/usr/bin/env python3
# =============================================================================
#  PeleC Post-Processing: Main Execution Script
# =============================================================================
#  User-friendly entry point for running the PeleC post-processing pipeline.
#
#  For interns / new GRAs:
#    - Edit the CONFIG dict below to control what gets processed.
#    - Run with:  python pelec_post.py
#    - You should NOT need to edit the functions in the *_database.py files.
# =============================================================================

import glob
import os
import re
import struct
import subprocess
import sys
import time
import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import our modular databases
import pp_functions_database as fdb
import pp_plotting_database as pdb


# ---------------------------------------------------------------------------
#  LOGGING UTILITY
# ---------------------------------------------------------------------------

_t_start_global = time.time()


def _ts(msg: str, end: str = "\n") -> None:
    """Print a timestamped log message relative to script start."""
    elapsed = time.time() - _t_start_global
    sys.stdout.write(f"[{elapsed:8.1f}s] {msg}{end}")
    sys.stdout.flush()


def _hr(title: str = "", char: str = "=", width: int = 70) -> None:
    """Print a horizontal rule with optional centred title."""
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char * side}  {title}  {char * side}")
    else:
        print(char * width)


def _section(n: int, total: int, label: str) -> None:
    """Print a section header like  [1/4]  Discovering plotfiles ..."""
    _ts(f"[{n}/{total}] {label} {'─' * max(1, 50 - len(label))}")

# ---------------------------------------------------------------------------
#  USER CONFIGURATION
# ---------------------------------------------------------------------------
#  Edit the values below to control what gets processed.
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Data source ---
    "data_source": "../TS-Driver/FP-Extended-Domain/pltFile",   # Directory with plotfiles
    "plot_prefix": "pltFlatPlatePost",                         # Plotfile directory prefix
    "output_dir": "../TS-Driver/FP-Extended-Domain/1-Plot-Outputs/Contours/Pres",                    # Where results go

    # --- Snapshot range ---
    # Set to None to process all discovered plotfiles.
    "snapshot_start": 212000,
    "snapshot_end": 213000,
    "snapshot_step": 1000,

    # --- Field aliases ---
    # Map solver raw names -> canonical names.  If omitted, default PeleC
    # aliases are used.  Only add / override if your case uses non-standard
    # naming.
    "field_aliases": None,

    # --- Workflow toggles ---
    "make_contour_plots": False,
    "make_line_profiles": False,
    "make_streamlines": False,
    "make_surface_analysis": False,
    "make_group_plots": False,

    "make_probe_plots": False,   # set True only if you need time-history plots (requires ASCII conversion)
    "make_fft_probes": True,      # set True to run FFT / stability analysis on probe data
    "make_pprime_contour": False,
    "make_stability_diagnostics": False,


    # --- Contour plot settings ---
    # List of canonical field names to plot.  Any field present in the
    # dataset (including derived fields) can be used.
    "contour_fields": [
        "density",
    ],
    "contour_cmap": "turbo",             # Colormap for all contour plots
    "contour_norm": "linear",               # Color scaling: "linear", "log", "symlog", or a matplotlib Normalize object
    "contour_vlims": {                       # Per-field color limits [vmin, vmax]
        "density": [0, 1.0],
    },
    "contour_xlim": [-0.001,0.2],                    # [xmin, xmax] or None for full domain
    "contour_ylim": None,                    # [ymin, ymax] or None for full domain

    # --- Line profile settings ---
    "line_x_stations": [0.05, 0.3],                  # x-locations to extract profiles [m]
    "line_fields": ["x_velocity", "temperature"],
    "line_ylim": [0, 0.005],                              # [ymin, ymax] or None for full domain height

    # --- Blasius reference overlay (for line profiles) ---
    "line_blasius_overlay": True,
    "line_blasius_fields": ["x_velocity", "temperature"],
    "line_blasius": {
        "u_inf": 1726.0,        # freestream velocity [m/s]
        "T_inf": 125.0,         # freestream temperature [K]
        "rho_inf": 0.0267,      # freestream density [kg/m^3]
        "T_wall": 293,         # wall temp [K]; None = adiabatic
        "recovery_factor": None,# None = sqrt(Pr_eff) (laminar)
        "const_transport": False,  # True = constant mu and k; False = Sutherland and mu*Cp/Pr
        "mu": 8.65e-6,         # constant viscosity [Pa.s]; used when const_transport=True
        "k": 0.012415,         # constant thermal cond. [W/(m.K)]; used to recompute Pr_eff
    },
    "line_blasius_compressible_overlay": True,
    "line_compare_overlay": True,
    
    "line_compare_plotfile": "/lustre/isaac24/scratch/sbrollia/TS-Driver/FP-ED-Refined-1/pltFile/pltFlatPlateFlow260631",   # Optional external plotfile directory to overlay against current simulation
    "line_compare_label": "Refined-ED",
    "line_compare_color": "C6",
    "line_compare_linestyle": "--",

    # --- Streamline settings ---
    "streamline_color_key": "velocity_magnitude",  # Color streamlines by this field
    "streamline_mask_key": None,                   # Mask streamlines by this field
    "streamline_mask_threshold": None,

    # --- Surface analysis settings ---
    "surface_velocity_threshold": 10.0,
    "surface_max_x": 0.2,
    "surface_geometry_method": "auto",   # 'auto', 'vfrac', 'ib_markers', 'velocity'
    "geometry_type": "flat_plate",              # 'auto', 'flat_plate' (analytical), 'wedge', 'custom'
    "plate_leading_edge": 0.0,            # [m] for geometry_type='flat_plate'
    "surface_rho_inf": 0.0267,
    "surface_u_inf": 1726.0,
    "surface_T_inf": None,
    "surface_mu": 8.65e-6,                  # constant viscosity [Pa.s]; None -> Sutherland
    "surface_k": 0.012415,                   # constant thermal cond. [W/(m.K)]; None -> mu*Cp/Pr
    "surface_wall_temperature": 293,    # constant wall temp [K]; None -> use cell-adjacent T
    "surface_Pr": 0.71,                  # Prandtl number (only if surface_k is None)
    "surface_Cp": 1004.0,                # specific heat [J/(kg.K)]

    # --- Force analysis settings ---
    "make_force_analysis": True,
    "make_spacetime_plots": True,
    "make_force_animation": False,
    "reference_area": None,              # auto = plate length * 1 m span

    # --- Laser annotation (for time-series / animation) ---
    "laser_start_time": 0.005,            # [s] time when laser turns on

    # --- Multiprocessing ---
    "use_multiprocessing": True,
    "num_processes": 10,

    # --- Probe processing settings ---
    "probe_bin_files": [
        "../TS-Driver/FP-Extended-Domain/probes/flow-probe.bin",
        "../TS-Driver/FP-Extended-Domain/probes/flow-probe2.bin",
        "../TS-Driver/FP-Extended-Domain/probes/flow-probe3.bin",
    ],
    "probe_output_dir": "../TS-Driver/FP-Extended-Domain/probes/flow-probe",
    "probe_max": 2000,
    "probe_fields": ["rho", "u", "p", "T"],
    "probe_convert_to_mks": False,

    # --- FFT probe analysis settings ---
    "fft_use_binary": True,        # read probes directly from binary for FFT/stability
    "probe_dir": "../TS-Driver/FP-Extended-Domain/probes/flow-probe",
    "probe_prefix": "flow_probe",
    "fft_var_col": 3,
    "fft_nt_skip": 0,
    "fft_max_probes": None,
    "probe_dedup_tol": 1e-12,
    "fft_resample": False,
    "fft_target_dt": None,
    "fft_mean_subtraction": "mean",   # "mean" | "linear" | "none" — remove DC before FFT
    "fft_window": "hann",            # "hann" | "hamming" | "blackman" | "rect" | "none"
    "fft_window_compensation": True,  # Scale FFT amplitudes to preserve magnitude
    "fft_plot_last_probe": False,
    "fft_plot_probe_indices": [0, 249, 499, 749, 999, 1249],
    "fft_plot_contour": True,  # can OOM with 2000 probes; use True with fft_max_probes <= ~200
    "fft_contour_normalize": True,  # True can create bright artifacts where the reference probe has a node
    "fft_contour_ref_probe": 499,
    "fft_contour_scale": "linear",  # "linear" gives 0-to-max amplitude; "db" gives the legacy dB plot
    "fft_contour_vmax": 400,  # None -> use the maximum plotted amplitude

    # --- FFT advanced analysis ---
    "fft_harmonic_freq": 10.0e6,
    "fft_num_harmonics": 5,
    "fft_plot_harmonics": True,
    "fft_slope_fmin": 1.0e6,
    "fft_slope_fmax": 2.0e8,
    "fft_plot_spectral_slope": True,
    "fft_growth_freqs": [5.0e6, 10.0e6, 30.0e6, 100.0e6, 200.0e6],
    "fft_plot_growth_curves": True,

    # --- Pressure perturbation (p') contour ---
    "pprime_baseline_plotfile": "pltFlatPlateFlow210000",
    "pprime_field": "pressure",
    "pprime_cmap": "RdBu_r",
    "pprime_vlims": [-500, 500],

    # --- Stability diagnostics (2nd Mack mode) ---
    "stability_baseline_plotfile": "pltFlatPlatePost210000",
    "stability_target_freq": None,
    "stability_freq_band": [100e3, 1.0e6],
    "stability_growth_window_size": None,
    "stability_num_gpi_profiles": 5,

    # --- Phase 1: Disturbance signal reconstruction ---
    # Reconstruct time-domain disturbance from FFT harmonics, a frequency
    # band, or the top-N amplitude frequency peaks.
    # Requires make_fft_probes=True.
    "make_disturbance_reconstruction": True,
    # None reuses fft_plot_probe_indices.  This keeps the expensive window
    # detection and reconstruction aligned with the probes selected for plots.
    # Set an explicit list only when reconstruction should use a different set.
    "reconstruction_probe_indices": None,
    "reconstruction_method": "top_frequencies",  # "harmonics" | "band" | "top_frequencies"
    "reconstruction_band": [1.0e6, 50.0e6],        # for method="band" [Hz]
    "reconstruction_num_harmonics": 5,             # for method="harmonics"
    "reconstruction_n_peaks": 15,                  # for method="top_frequencies"
    "reconstruction_peak_min_distance_hz": None,   # None -> auto (max(3*df, 1 kHz))
    "reconstruction_peak_freq_range": None,         # [f_min, f_max] or None
    # --- Disturbance window detection (Phase 1b) ---
    # Restrict reconstruction to the active disturbance interval
    # (shock front + oscillatory tail) instead of the full probe record.
    #   "common_detected" : detect per-probe, then form one common window
    #   "per_probe"       : each probe uses its own window
    #   "fixed"           : use reconstruction_window_start/end seconds
    #   "full"            : legacy — no windowing
    "reconstruction_window_mode": "per_probe",
    "reconstruction_window_detector": "energy",
    # Select the strongest detected packet in each probe rather than the
    # first threshold crossing; important once the packet arrival is delayed.
    "reconstruction_packet_selection": "dominant_peak",  # "dominant_peak" | "first_threshold"
    "reconstruction_window_start": None,       # for mode="fixed" [s]
    "reconstruction_window_end": None,         # for mode="fixed" [s]
    "reconstruction_baseline_margin": 0.001,   # pre-event baseline end [s]
    "reconstruction_pre_event_fraction": 0.02, # fallback baseline fraction
    "reconstruction_onset_sigma": 5.0,         # onset threshold multiplier
    # A 10% entry level keeps separated amplitude lobes of one packet
    # connected while rejecting the baseline/noise floor.
    "reconstruction_onset_peak_fraction": 0.10, # onset level relative to that probe's peak energy
    "reconstruction_offset_sigma": 2.0,        # offset threshold (hysteresis)
    # A per-probe pulse-tail cutoff relative to that probe's peak detector
    # energy.  None uses only the baseline-noise offset threshold.
    "reconstruction_offset_peak_fraction": 1.0e-3,
    "reconstruction_min_active_duration": 50e-6,   # min time above onset [s]
    "reconstruction_min_quiet_duration": 5e-6,     # min time below offset [s]
    "reconstruction_window_pad_before": 20e-6,     # padding before onset [s]
    "reconstruction_window_pad_after": 50e-6,      # padding after offset [s]
    "reconstruction_common_onset_percentile": 10.0,  # robust common start [%]
    "reconstruction_common_offset_percentile": 90.0, # robust common end [%]
    "reconstruction_min_samples": 256,          # min samples in window
    "reconstruction_max_samples": 100000,       # max samples in window
    "reconstruction_save_window_diagnostics": True,

    # --- Phase 2: Transient analysis (STFT / Hilbert envelope) ---
    # Time-frequency analysis for laser-pulse wavepacket tracking.
    "make_transient_analysis": True,
    "transient_band": [5.0e6, 50.0e6],      # band for envelope extraction [Hz]
    "transient_stft_nperseg": 256,
    "transient_stft_noverlap": 0.75,
    "transient_plot_probe_indices": [0, 49, 99, 249, 499],

    # --- Phase 3: Nonlinear interaction diagnostics (bispectrum) ---
    # Quadratic phase-coupling detection via bicoherence.
    "make_nonlinear_diagnostics": True,
    "nonlinear_nperseg": 256,               # segment length for bispectrum
    "nonlinear_noverlap": 0.5,              # overlap fraction
    "nonlinear_plot_probe_indices": [0, 49, 99, 249, 499],
    "nonlinear_significance_level": 0.95,

    # --- Logging ---
    "debug_mode": True,
}


# ---------------------------------------------------------------------------
#  INTERNAL WORKER FUNCTIONS
# ---------------------------------------------------------------------------

def _process_single_contour(args):
    """Worker: plot contours for one snapshot."""
    dataset, config = args
    try:
        output_dir = Path(config["output_dir"]) / "Contours"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        for field_key in config["contour_fields"]:
            if field_key not in dataset["fields"]:
                continue
            out = output_dir / f"{label}_{field_key}.png"
            vlims = config.get("contour_vlims", {}).get(field_key, [None, None])
            pdb.plot_contour(
                dataset, field_key,
                output_path=str(out),
                title=f"{field_key} — {label}",
                cmap=pdb.resolve_cmap(config.get("contour_cmap", "viridis")),
                norm=config.get("contour_norm", "linear"),
                vmin=vlims[0],
                vmax=vlims[1],
                xlim=config.get("contour_xlim"),
                ylim=config.get("contour_ylim"),
            )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_single_line(args):
    """Worker: extract and plot line profiles for one snapshot."""
    dataset, config = args
    try:
        output_dir = Path(config["output_dir"]) / "LineProfiles"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")
        compare_plotfile = config.get("line_compare_plotfile")
        compare_ds = None
        if compare_plotfile:
            try:
                compare_ds = fdb.load_pelec_plotfile(
                    compare_plotfile,
                    field_names=None,
                    alias_map=config.get("field_aliases"),
                    convert_to_mks=True,
                )
                fdb.compute_derived_fields(compare_ds)
            except Exception as exc:
                fdb._log_error(f"Comparison plotfile load failed: {compare_plotfile}", exc)
                compare_ds = None

        # Precompute boundary-layer height (delta_99) at each x-station so it
        # can be marked on every line profile.  Use the x_velocity profile and
        # the user-supplied u_inf if available; otherwise fall back to the
        # maximum velocity in the (possibly cropped) profile.
        bl_heights = {}
        line_ylim = config.get("line_ylim", None)
        bl_cfg = config.get("line_blasius", {})
        if "x_velocity" in dataset["fields"]:
            for x_loc in config["line_x_stations"]:
                try:
                    u_prof = fdb.extract_line(dataset, x_loc, "x_velocity")
                    if line_ylim is not None and len(u_prof["y"]) > 1:
                        y_min, y_max = line_ylim
                        mask = (u_prof["y"] >= y_min) & (u_prof["y"] <= y_max)
                        u_prof["y"] = u_prof["y"][mask]
                        u_prof["values"] = u_prof["values"][mask]
                    u_inf_ref = bl_cfg.get("u_inf")
                    if u_inf_ref is None:
                        u_inf_ref = float(np.nanmax(np.abs(u_prof["values"])))
                    U_Uinf = np.clip(
                        np.abs(u_prof["values"]) / float(u_inf_ref), 0.0, 1.5
                    )
                    d99, _, _, _ = fdb.calculate_BL_thicknesses(
                        u_prof["y"], U_Uinf
                    )
                    bl_heights[x_loc] = float(d99)
                except Exception as exc:
                    fdb._log_error(
                        f"Boundary-layer height at x={x_loc:.3f}", exc
                    )
                    bl_heights[x_loc] = None

        for field_key in config["line_fields"]:
            if field_key not in dataset["fields"]:
                continue
            profiles = []
            compressible_profiles = []
            compare_profiles = []
            for x_loc in config["line_x_stations"]:
                try:
                    prof = fdb.extract_line(dataset, x_loc, field_key)
                    # Apply y-range cropping
                    if line_ylim is not None and len(prof["y"]) > 1:
                        y_min, y_max = line_ylim
                        mask = (prof["y"] >= y_min) & (prof["y"] <= y_max)
                        prof["y"] = prof["y"][mask]
                        prof["values"] = prof["values"][mask]
                    prof["label"] = f"x={x_loc:.3f}"
                    prof["boundary_layer_height"] = bl_heights.get(x_loc)
                    profiles.append(prof)

                    # Blasius reference overlay
                    if (config.get("line_blasius_overlay", False)
                            and field_key in config.get("line_blasius_fields", [])):
                        bl_cfg = config.get("line_blasius", {})
                        if all(k in bl_cfg for k in ("u_inf", "T_inf", "rho_inf")):
                            ref = fdb.compute_blasius_reference_profile(
                                y=prof["y"], x_loc=x_loc,
                                u_inf=bl_cfg["u_inf"],
                                T_inf=bl_cfg["T_inf"],
                                rho_inf=bl_cfg["rho_inf"],
                                T_wall=bl_cfg.get("T_wall"),
                                recovery_factor=bl_cfg.get("recovery_factor"),
                                const_transport=bl_cfg.get("const_transport"),
                                mu=bl_cfg.get("mu"),
                                k=bl_cfg.get("k"),
                            )
                            if field_key in ref:
                                if field_key == "x_velocity":
                                    try:
                                        u_inf_ref_bl = float(bl_cfg["u_inf"])
                                        U_Uinf_bl = np.clip(
                                            np.abs(ref[field_key]["values"]) / u_inf_ref_bl,
                                            0.0, 1.5,
                                        )
                                        d99_bl, _, _, _ = fdb.calculate_BL_thicknesses(
                                            ref[field_key]["y"], U_Uinf_bl
                                        )
                                        ref[field_key]["boundary_layer_height"] = float(d99_bl)
                                    except Exception as exc:
                                        fdb._log_error(
                                            f"Blasius boundary-layer height at x={x_loc:.3f}", exc
                                        )
                                profiles.append(ref[field_key])

                    # Compressible Blasius-style transform overlay for x_velocity.
                    if (field_key == "x_velocity"
                            and config.get("line_blasius_compressible_overlay", False)
                            and all(k in bl_cfg for k in ("u_inf", "rho_inf"))):
                        try:
                            rho_prof = fdb.extract_line(dataset, x_loc, "density")
                            if line_ylim is not None and len(rho_prof["y"]) > 1:
                                y_min, y_max = line_ylim
                                mask = (rho_prof["y"] >= y_min) & (rho_prof["y"] <= y_max)
                                rho_prof["y"] = rho_prof["y"][mask]
                                rho_prof["values"] = rho_prof["values"][mask]

                            T_prof = None
                            if "temperature" in dataset["fields"]:
                                T_prof = fdb.extract_line(dataset, x_loc, "temperature")
                                if line_ylim is not None and len(T_prof["y"]) > 1:
                                    y_min, y_max = line_ylim
                                    mask = (T_prof["y"] >= y_min) & (T_prof["y"] <= y_max)
                                    T_prof["y"] = T_prof["y"][mask]
                                    T_prof["values"] = T_prof["values"][mask]

                            comp = fdb.compute_compressible_blasius_reference_profile(
                                y=prof["y"],
                                u_profile=prof["values"],
                                rho_profile=rho_prof["values"],
                                x_loc=x_loc,
                                u_inf=bl_cfg["u_inf"],
                                T_profile=None if T_prof is None else T_prof["values"],
                                T_inf=bl_cfg.get("T_inf"),
                                const_transport=bl_cfg.get("const_transport", False),
                                mu=bl_cfg.get("mu"),
                            )
                            compressible_profiles.extend([comp["sim"], comp["blasius"]])
                        except Exception as exc:
                            fdb._log_error(
                                f"Compressible Blasius overlay at x={x_loc:.3f}", exc
                            )

                    # Optional external comparison run from another folder.
                    if (config.get("line_compare_overlay", False)
                            and compare_ds is not None
                            and field_key in config.get("line_blasius_fields", [])):
                        try:
                            cprof = fdb.extract_line(compare_ds, x_loc, field_key)
                            if line_ylim is not None and len(cprof["y"]) > 1:
                                y_min, y_max = line_ylim
                                mask = (cprof["y"] >= y_min) & (cprof["y"] <= y_max)
                                cprof["y"] = cprof["y"][mask]
                                cprof["values"] = cprof["values"][mask]
                            cprof["label"] = config.get("line_compare_label", "Comparison")
                            cprof["color"] = config.get("line_compare_color", "C2")
                            cprof["linestyle"] = config.get("line_compare_linestyle", ":")
                            cprof["boundary_layer_height"] = bl_heights.get(x_loc)
                            compare_profiles.append(cprof)
                        except Exception as exc:
                            fdb._log_error(f"Comparison line extraction at x={x_loc:.3f}", exc)
                except Exception as exc:
                    fdb._log_error(f"Line extraction at x={x_loc:.3f}", exc)

            if compare_profiles:
                profiles.extend(compare_profiles)

            if profiles:
                out = output_dir / f"{label}_{field_key}_profiles.png"
                pdb.plot_line_profiles(
                    profiles,
                    output_path=str(out),
                    title=f"{field_key} — {label}",
                    swap_axes=True,
                )

            if compressible_profiles:
                out = output_dir / f"{label}_{field_key}_compressible_blasius.png"
                pdb.plot_line_profiles(
                    compressible_profiles,
                    field_label=r"U/U_e",
                    xlabel=r"Transformed coordinate $\eta_{vd}$",
                    output_path=str(out),
                    title=f"{field_key} — compressible Blasius transform — {label}",
                    x_key="eta",
                    swap_axes=False,
                )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_single_streamline(args):
    """Worker: plot streamlines for one snapshot."""
    dataset, config = args
    try:
        output_dir = Path(config["output_dir"]) / "Streamlines"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        sl = fdb.extract_streamline_field(
            dataset,
            color_key=config.get("streamline_color_key"),
            mask_key=config.get("streamline_mask_key"),
            mask_threshold=config.get("streamline_mask_threshold"),
        )
        out = output_dir / f"{label}_streamlines.png"
        pdb.plot_streamlines(
            [sl],
            output_path=str(out),
            title=f"Streamlines — {label}",
        )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_single_surface(args):
    """Worker: run surface analysis for one snapshot."""
    dataset, config = args
    try:
        output_dir = Path(config["output_dir"]) / "SurfaceAnalysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        # Detect geometry
        surfaces = fdb.define_geometry_surface(
            dataset,
            method=config["surface_geometry_method"],
            velocity_threshold=config["surface_velocity_threshold"],
            max_x=config["surface_max_x"],
            geometry_type=config.get("geometry_type", "auto"),
            plate_leading_edge=config.get("plate_leading_edge", 0.0),
        )

        # Plot geometry
        if len(surfaces["upper"].get("x", [])) > 0 or \
           len(surfaces["lower"].get("x", [])) > 0:
            geo_out = output_dir / f"geometry_{label}.png"
            pdb.plot_surface_geometry(
                surfaces, dataset,
                snapshot=label,
                output_path=str(geo_out),
            )

        # Compute normals
        surfaces = fdb.compute_surface_normals(surfaces)

        # Extract surface properties
        surface_data = fdb.extract_surface_properties(
            dataset, surfaces,
            rho_inf=config["surface_rho_inf"],
            u_inf=config["surface_u_inf"],
            T_inf=config["surface_T_inf"],
            mu=config.get("surface_mu"),
            k_w=config.get("surface_k"),
            T_wall=config.get("surface_wall_temperature"),
            Pr=config.get("surface_Pr", 0.71),
            Cp=config.get("surface_Cp", 1004.0),
        )

        # Plot surface properties
        prop_out = output_dir / f"properties_{label}.png"
        pdb.plot_surface_properties(
            surface_data,
            snapshot=label,
            output_path=str(prop_out),
        )

        # ---- Force analysis ----
        force_result = None
        if config.get("make_force_analysis", False):
            try:
                chord = config.get("reference_area", None)
                if chord is not None:
                    # reference_area is the area; derive chord as area / 1m
                    chord = chord / 1.0
                force_result = fdb.compute_integrated_forces(
                    surface_data,
                    rho_inf=config["surface_rho_inf"],
                    u_inf=config["surface_u_inf"],
                    chord_length=chord,
                )
                force_result["time"] = dataset.get("time", 0.0)
            except Exception as fexc:
                fdb._log_error(f"Force calculation failed for {label}", fexc)

        return (label, True, {"surface": surface_data, "forces": force_result})
    except Exception as exc:
        return (label, False, str(exc))


# ---------------------------------------------------------------------------
#  DELTA WORKER FUNCTIONS — p' contour, FFT probes, stability diagnostics
# ---------------------------------------------------------------------------


def _process_pprime_contour(args):
    """Worker: plot p' = p_post - p_baseline contour for a single snapshot."""
    dataset, config, baseline_fields = args
    try:
        output_dir = Path(config["output_dir"]) / "PPrime"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        field_key = config.get("pprime_field", "pressure")
        if field_key not in dataset["fields"]:
            return (label, False, f"Field '{field_key}' not in dataset")
        if field_key not in baseline_fields:
            return (label, False, f"Field '{field_key}' not in baseline")

        p_post = dataset["fields"][field_key]
        p_base = baseline_fields[field_key]

        if p_post.shape != p_base.shape:
            return (label, False,
                    f"Shape mismatch: post {p_post.shape} vs baseline {p_base.shape}")

        p_prime = p_post - p_base
        pprime_ds = dict(dataset)
        pprime_ds["fields"] = {field_key: p_prime}

        cmap = config.get("pprime_cmap", "RdBu_r")
        vlims = config.get("pprime_vlims", [None, None])
        vmin, vmax = vlims[0], vlims[1]

        out = output_dir / f"{label}_pprime.png"
        pdb.plot_contour(
            pprime_ds, field_key,
            output_path=str(out),
            title=f"p' (pressure perturbation) — {label}",
            cmap=pdb.resolve_cmap(cmap),
            norm="linear",
            vmin=vmin,
            vmax=vmax,
            colorbar_label=f"Delta{field_key} [Pa]",
        )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _parse_probe_header(filepath):
    """Return (x_cm, y_cm) from the header of a probe file."""
    with open(filepath, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"x=([\d\.eE+\-]+)\s*cm.*y=([\d\.eE+\-]+)\s*cm", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


def _parse_probe_actual_position(filepath):
    """Return the actual sampled (x_cm, y_cm) from the first data row.

    The ASCII header stores the *requested* probe location, but in immersed /
    cut-cell simulations the sampled cell centre can differ (sometimes
    substantially).  Column 5 of the data block is x_sample and column 6 is
    y_sample; these are the coordinates that should be used for FFT-vs-position
    and stability analysis.
    """
    with open(filepath, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    return float(parts[5]), float(parts[6])
                except ValueError:
                    break
            break
    return None, None


def _read_probe_binary_header(f):
    """Read and validate a probe binary header; return (endian, n_probes, n_fields, probe_x, probe_y)."""
    magic = struct.unpack('<q', f.read(8))[0]
    expected = 0x005345424F525050  # "PROBES\0\0"
    if magic != expected:
        f.seek(-8, os.SEEK_CUR)
        magic = struct.unpack('>q', f.read(8))[0]
        if magic != expected:
            raise ValueError(f"Bad magic number: 0x{magic:016x}")
        endian = '>'
    else:
        endian = '<'
    n_probes = struct.unpack(f'{endian}q', f.read(8))[0]
    n_fields = struct.unpack(f'{endian}q', f.read(8))[0]
    if n_fields != 7:
        raise ValueError(f"Expected 7 fields, got {n_fields}")
    probe_x = list(struct.unpack(f'{endian}{n_probes}d', f.read(8 * n_probes)))
    probe_y = list(struct.unpack(f'{endian}{n_probes}d', f.read(8 * n_probes)))
    return endian, n_probes, n_fields, probe_x, probe_y


def _build_probe_binary_index(bin_files, dedup_tol=1e-12):
    """Build a sorted, deduplicated index of all timesteps across binaries.

    Returns (file_handles, endians, n_probes, n_fields, probe_x, probe_y, index).
    Caller is responsible for closing file_handles.
    """
    file_handles = []
    file_endians = []
    ref_header = None
    n_probes = None
    n_fields = None
    probe_x = None
    probe_y = None

    for idx, path in enumerate(bin_files):
        f = open(path, 'rb')
        endian, npb, nf, px, py = _read_probe_binary_header(f)
        if idx == 0:
            ref_header = (npb, nf, px, py)
            n_probes, n_fields, probe_x, probe_y = npb, nf, px, py
        else:
            if (npb, nf, px, py) != ref_header:
                raise ValueError(f"Header mismatch in {path}")
        file_handles.append(f)
        file_endians.append(endian)

    index = []
    record_size = 8 * (1 + n_probes * n_fields)
    for file_idx, f in enumerate(file_handles):
        f.seek(0, os.SEEK_SET)
        _read_probe_binary_header(f)
        offset = f.tell()
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        while offset + record_size <= file_size:
            f.seek(offset, os.SEEK_SET)
            raw = f.read(8)
            if len(raw) < 8:
                break
            time = struct.unpack(f'{file_endians[file_idx]}d', raw)[0]
            index.append((time, file_idx, offset))
            offset += record_size

    index.sort(key=lambda r: r[0])

    if dedup_tol >= 0 and index:
        deduped = [index[0]]
        for rec in index[1:]:
            if rec[0] - deduped[-1][0] >= dedup_tol:
                deduped.append(rec)
        index = deduped

    return file_handles, file_endians, n_probes, n_fields, probe_x, probe_y, index


def _load_probe_data_from_binary(config, var_col, nt_skip=0, max_probes=None):
    """Load probe time series directly from binary files.

    var_col follows the ASCII .dat convention: 0=time, 1=rho, 2=u, 3=p, 4=T,
    5=x_sample, 6=y_sample, 7=level.  The binary payload stores fields 0-6
    (rho, u, p, T, x_sample, y_sample, level), so we map var_col -> field index.

    Returns a list of dicts with keys: x, y, x_req, y_req, time, signal, filename.
    Positions are returned in centimetres to match the ASCII header convention.
    """
    bin_files = config.get("probe_bin_files", [])
    if not bin_files:
        return []

    bin_files = [str(Path(b).resolve()) for b in bin_files]
    file_handles, file_endians, n_probes, n_fields, probe_x_req, probe_y_req, index = \
        _build_probe_binary_index(bin_files, dedup_tol=config.get("probe_dedup_tol", 1e-12))

    n_timesteps = len(index)
    if n_timesteps == 0:
        for f in file_handles:
            f.close()
        return []

    if var_col == 0:
        raise ValueError("var_col=0 (time) is not a valid signal column")
    field_idx = var_col - 1
    if not (0 <= field_idx < n_fields):
        raise ValueError(f"var_col={var_col} maps to invalid binary field {field_idx}")

    x_field = 4
    y_field = 5

    probe_ids = list(range(n_probes))
    if max_probes is not None:
        probe_ids = probe_ids[:max_probes]

    n_load = len(probe_ids)
    time_buf = np.empty(n_timesteps, dtype=np.float64)
    signal_buf = np.empty((n_timesteps, n_load), dtype=np.float64)
    x_sample = np.empty(n_load, dtype=np.float64)
    y_sample = np.empty(n_load, dtype=np.float64)
    x_sample_set = np.zeros(n_load, dtype=bool)

    record_payload = 8 * n_probes * n_fields
    for it, (time, file_idx, offset) in enumerate(index):
        f = file_handles[file_idx]
        endian = file_endians[file_idx]
        f.seek(offset + 8, os.SEEK_SET)
        raw = f.read(record_payload)
        data = struct.unpack(f'{endian}{n_probes * n_fields}d', raw)
        time_buf[it] = time
        for j, ip in enumerate(probe_ids):
            base = ip * n_fields
            signal_buf[it, j] = data[base + field_idx]
            if not x_sample_set[j]:
                x_sample[j] = data[base + x_field]
                y_sample[j] = data[base + y_field]
                x_sample_set[j] = True

    for f in file_handles:
        f.close()

    if nt_skip > 0:
        time_buf = time_buf[nt_skip:]
        signal_buf = signal_buf[nt_skip:, :]

    probe_data = []
    for j, ip in enumerate(probe_ids):
        probe_data.append({
            "x": float(x_sample[j]),
            "y": float(y_sample[j]),
            "x_req": float(probe_x_req[ip]),
            "y_req": float(probe_y_req[ip]),
            "time": time_buf.copy(),
            "signal": signal_buf[:, j].copy(),
            "dt": float(time_buf[1] - time_buf[0]) if time_buf.size >= 2 else 0.0,
            "filename": f"flow_probe_{ip:03d}.dat",
        })

    return probe_data


def _load_probe_timeseries(config, var_col, nt_skip=0, max_probes=None):
    """Load probe time series from .dat files or directly from binary.

    If fft_use_binary is True (default) and probe_bin_files are provided, read
    directly from the binary files.  This avoids the slow and memory-heavy
    ASCII conversion for runs that only need FFT/stability diagnostics.
    Otherwise fall back to existing per-probe .dat files.
    """
    use_binary = config.get("fft_use_binary", True)
    bin_files = config.get("probe_bin_files", [])

    if use_binary and bin_files:
        try:
            return _load_probe_data_from_binary(config, var_col, nt_skip=nt_skip, max_probes=max_probes)
        except Exception as exc:
            _ts(f"  [W] Binary probe read failed ({exc}); falling back to .dat files")

    data_source = config["data_source"]
    probe_dir = config.get("probe_dir", "probes")
    probe_prefix = config.get("probe_prefix", "flow_probe")
    probe_path = os.path.join(data_source, probe_dir)
    pattern = os.path.join(probe_path, f"{probe_prefix}_*.dat")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    if max_probes is not None:
        files = files[:max_probes]

    probe_data = []
    for fpath in files:
        x, y = _parse_probe_actual_position(fpath)
        x_req, y_req = _parse_probe_header(fpath)
        if x is None:
            x, y = x_req, y_req
        if x is None:
            continue
        data = np.loadtxt(fpath, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time = data[nt_skip:, 0]
        signal = data[nt_skip:, var_col]
        dt_actual = time[1] - time[0] if len(time) >= 2 else 0.0
        probe_data.append({
            "x": x, "y": y,
            "x_req": x_req, "y_req": y_req,
            "time": time, "signal": signal,
            "dt": dt_actual, "filename": os.path.basename(fpath),
        })
    return probe_data


def _find_nearest_freq_bin(freq, target):
    """Return (bin_index, freq_value) for the frequency bin closest to target."""
    freq = np.asarray(freq)
    idx = int(np.argmin(np.abs(freq - target)))
    return idx, freq[idx]


def _fit_loglog_slope(freq, P1, fmin, fmax):
    """Fit log10(P1) = slope * log10(freq) + intercept within [fmin, fmax]."""
    freq = np.asarray(freq)
    P1 = np.asarray(P1)
    mask = (freq >= fmin) & (freq <= fmax) & (P1 > 0) & np.isfinite(P1)
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan
    x = np.log10(freq[mask])
    y = np.log10(P1[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r_squared


def _preprocess_probe_signal(signal_matrix, config, time_uniform=None):
    """Apply per-probe mean subtraction and windowing to the signal matrix.

    Operates in-place on *signal_matrix* and returns the window amplitude
    compensation factor ``scale = L / sum(window)`` (or 1.0 if windowing is
    disabled / compensation is off).

    Parameters
    ----------
    signal_matrix : ndarray, shape (L, n_probes)
        Uniformly-resampled time series, one column per probe.
    config : dict
        Must contain keys ``fft_mean_subtraction``, ``fft_window``,
        ``fft_window_compensation``.
    time_uniform : ndarray, optional
        Time array matching *signal_matrix* row axis. Required when
        ``fft_mean_subtraction == "linear"``.

    Returns
    -------
    signal_matrix : ndarray
        Modified in-place.
    scale : float
        Amplitude scaling factor (1.0 if no compensation).
    raw_matrix : ndarray or None
        Copy of the input before any modification, for plotting.
    """
    L, n_probes = signal_matrix.shape
    mean_mode = config.get("fft_mean_subtraction", "mean")
    window_type = config.get("fft_window", "hann")
    do_comp = config.get("fft_window_compensation", True)

    # Save a copy of the raw (unprocessed) signal for dual-row time-trace plots
    raw_matrix = signal_matrix.copy()

    # --- Per-probe mean subtraction / detrend ---
    if mean_mode == "none":
        pass
    elif mean_mode == "mean":
        for ip in range(n_probes):
            col = signal_matrix[:, ip]
            signal_matrix[:, ip] = col - np.nanmean(col)
    elif mean_mode == "linear":
        if time_uniform is None:
            time_uniform = np.arange(L, dtype=float)
        for ip in range(n_probes):
            col = signal_matrix[:, ip]
            valid = np.isfinite(col)
            if np.sum(valid) < 3:
                continue
            p = np.polyfit(time_uniform[valid], col[valid], 1)
            signal_matrix[:, ip] = col - np.polyval(p, time_uniform)
    else:
        _ts(f"  [W] Unknown fft_mean_subtraction='{mean_mode}'; using 'none'")

    # --- Windowing ---
    window_type = window_type.lower()
    if window_type in ("none", "rect", "rectangular"):
        window = np.ones(L)
    elif window_type == "hann":
        window = np.hanning(L)
    elif window_type == "hamming":
        window = np.hamming(L)
    elif window_type == "blackman":
        window = np.blackman(L)
    else:
        _ts(f"  [W] Unknown fft_window='{window_type}'; using Hann")
        window = np.hanning(L)

    signal_matrix *= window[:, None]

    # --- Amplitude compensation ---
    win_sum = np.sum(window)
    if do_comp and win_sum > 0:
        scale = float(L) / win_sum
    else:
        scale = 1.0

    if mean_mode != "none" or window_type not in ("none", "rect", "rectangular"):
        _ts(f"  Preprocess: mean='{mean_mode}'  window='{window_type}'")

    return signal_matrix, scale, raw_matrix


def _process_fft_probes(config):
    """Worker: perform 1D FFT analysis on point-probe time series."""
    try:
        output_dir = Path(config["output_dir"]) / "FFT-Probes"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = "fft_probes"

        probe_dir = config.get("probe_dir", "probes")
        probe_prefix = config.get("probe_prefix", "flow_probe")
        var_col = config.get("fft_var_col", 4)
        nt_skip = config.get("fft_nt_skip", 0)
        max_probes = config.get("fft_max_probes", None)
        resample = config.get("fft_resample", True)
        target_dt = config.get("fft_target_dt", None)
        probe_indices = config.get("fft_plot_probe_indices", None)
        plot_contour_enabled = config.get("fft_plot_contour", True)
        contour_normalize = config.get("fft_contour_normalize", False)
        contour_ref = config.get("fft_contour_ref_probe", "first")
        harmonic_freq = config.get("fft_harmonic_freq", 10.0e6)
        num_harmonics = config.get("fft_num_harmonics", 5)
        plot_harmonics = config.get("fft_plot_harmonics", True)
        slope_fmin = config.get("fft_slope_fmin", 1.0e5)
        slope_fmax = config.get("fft_slope_fmax", 1.0e9)
        plot_spectral_slope = config.get("fft_plot_spectral_slope", True)
        growth_freqs = config.get("fft_growth_freqs", [10.0e6, 20.0e6, 30.0e6])
        plot_growth = config.get("fft_plot_growth_curves", True)

        # Load probe time series.  Prefer direct binary read (fast, low
        # memory) when fft_use_binary is enabled; otherwise fall back to
        # per-probe ASCII .dat files.
        probe_data = _load_probe_timeseries(
            config, var_col, nt_skip=nt_skip, max_probes=max_probes
        )

        if not probe_data:
            return (label, False, "No probe data could be loaded")

        probe_x = [d["x"] for d in probe_data]
        probe_y = [d["y"] for d in probe_data]
        n_valid = len(probe_data)
        _ts(f"  Successfully loaded {n_valid} probes")

        if resample:
            t_min = max(d["time"][0] for d in probe_data)
            t_max = min(d["time"][-1] for d in probe_data)
            dt_vals = [d["dt"] for d in probe_data if d["dt"] > 0]
            dt_use = target_dt if target_dt else (np.median(dt_vals) if dt_vals else 1.0)
            time_uniform = np.arange(t_min, t_max, dt_use)
            L = len(time_uniform)
            signal_matrix = np.zeros((L, n_valid))
            for ip, d in enumerate(probe_data):
                signal_matrix[:, ip] = np.interp(
                    time_uniform, d["time"], d["signal"],
                    left=d["signal"][0], right=d["signal"][-1],
                )
            dt_actual = dt_use
        else:
            lengths = [d["time"].size for d in probe_data]
            L = int(np.median(lengths))
            signal_matrix = np.zeros((L, n_valid))
            dt_actual = np.median([d["dt"] for d in probe_data if d["dt"] > 0])
            if dt_actual == 0:
                dt_actual = 1.0
            for ip, d in enumerate(probe_data):
                n = min(d["signal"].size, L)
                signal_matrix[:n, ip] = d["signal"][:n]

        # Apply mean subtraction and windowing
        time_array_for_detrend = time_uniform if resample else None
        signal_matrix, win_scale, raw_matrix = _preprocess_probe_signal(
            signal_matrix, config, time_uniform=time_array_for_detrend,
        )

        Fs = 1.0 / dt_actual
        freq = Fs * np.arange(0, L // 2 + 1) / L
        n_freq = len(freq)
        P1 = np.zeros((n_freq, n_valid))
        for ip in range(n_valid):
            sig = signal_matrix[:, ip]
            Y = np.fft.fft(sig)
            P2 = np.abs(Y / L)
            P1[:, ip] = P2[:n_freq] * win_scale
            P1[1:-1, ip] = 2 * P1[1:-1, ip]
        _ts(f"  FFT complete — {n_freq} bins x {n_valid} probes")

        # Auto-detect dominant frequency from FFT spectrum (exclude DC)
        auto_detect = config.get("fft_auto_detect_harmonic_freq", True)
        if auto_detect:
            # Exclude first few bins (DC + very low freq) to avoid drift
            dc_skip = max(1, int(5e3 * L / Fs))  # skip ~5 kHz worth of bins
            dc_skip = min(dc_skip, n_freq // 4)
            P1_mean = np.mean(P1[dc_skip:, :], axis=1)
            peak_idx = int(np.argmax(P1_mean)) + dc_skip
            detected_f = float(freq[peak_idx])
            if detected_f > 0:
                old_f = harmonic_freq
                harmonic_freq = detected_f
                _ts(f"  Auto-detected dominant frequency: {detected_f:.3e} Hz"
                    f" (was {old_f:.3e} Hz)")

        # Human-readable label for the signal column
        var_labels = {
            1: "Density [g/cm^3]",
            2: "u [cm/s]",
            3: "Pressure [dyne/cm^2]",
            4: "Temperature [K]",
        }
        ylabel_sig = var_labels.get(var_col, f"Signal (col {var_col})")

        # Probe time-trace + FFT plots
        plot_indices = []
        if probe_indices is not None and len(probe_indices) > 0:
            n = n_valid
            resolved = [max(0, min(idx if idx >= 0 else n + idx, n - 1)) for idx in probe_indices]
            plot_indices = sorted(set(resolved))
        elif config.get("fft_plot_last_probe", False):
            plot_indices = [n_valid - 1]

        # Reconstruction can be restricted independently, but by default it
        # follows the probes selected for the FFT time-history plots.  The
        # previous implementation reconstructed all n_valid probes even
        # though it only wrote figures for plot_indices.
        reconstruction_indices_cfg = config.get("reconstruction_probe_indices")
        if reconstruction_indices_cfg is None:
            reconstruction_indices_cfg = probe_indices
        if reconstruction_indices_cfg:
            recon_probe_indices = sorted(set(
                max(0, min(idx if idx >= 0 else n_valid + idx, n_valid - 1))
                for idx in reconstruction_indices_cfg
            ))
        else:
            recon_probe_indices = []

        if len(plot_indices) > 0:
            pos_indices = sorted(set(
                max(0, min(idx if idx >= 0 else n_valid + idx, n_valid - 1))
                for idx in plot_indices
            ))
            plot_time = time_uniform if resample else probe_data[0]["time"][:L]
            n_rows = len(pos_indices)
            # 3 columns: raw time-trace | processed time-trace | FFT spectrum
            fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, 3)
            for row, idx in enumerate(pos_indices):
                p = probe_data[idx]
                raw_sig = raw_matrix[:, idx]
                proc_sig = signal_matrix[:, idx]
                # Raw time-trace
                axes[row, 0].plot(plot_time, raw_sig, lw=0.8)
                axes[row, 0].set_xlabel("Time [s]", fontsize=10)
                axes[row, 0].set_ylabel(ylabel_sig, fontsize=10)
                axes[row, 0].set_title(f"Probe {idx} — x={p['x']:.3f} cm — Raw", fontsize=11)
                axes[row, 0].grid(True, alpha=0.4)
                # Processed time-trace
                axes[row, 1].plot(plot_time, proc_sig, lw=0.8, color="C1")
                axes[row, 1].set_xlabel("Time [s]", fontsize=10)
                axes[row, 1].set_ylabel(ylabel_sig, fontsize=10)
                axes[row, 1].set_title(f"Probe {idx} — x={p['x']:.3f} cm — Processed (mean-sub, Hann)", fontsize=11)
                axes[row, 1].grid(True, alpha=0.4)
                # FFT spectrum
                axes[row, 2].loglog(freq, P1[:, idx], lw=0.8)
                axes[row, 2].set_xlabel("Frequency [Hz]", fontsize=10)
                axes[row, 2].set_ylabel("|Amplitude|", fontsize=10)
                axes[row, 2].grid(True, which="both", ls=":", alpha=0.3)
            fig.tight_layout()
            out = output_dir / f"{probe_prefix}_fft_probes.png"
            fig.savefig(str(out), dpi=150)
            plt.close(fig)
            _ts(f"  Saved {len(pos_indices)} probe FFT plots: {out}")

        # Freq-vs-position contour
        if plot_contour_enabled and n_valid > 1:
            try:
                idx_sorted = np.argsort(probe_x)
                probe_x_sorted = np.array(probe_x)[idx_sorted]
                P1_sorted = P1[:, idx_sorted]
                eps = 1e-20
                contour_scale = config.get("fft_contour_scale", "linear").lower()
                if contour_scale == "linear":
                    # Use the physical nonnegative FFT amplitude directly.
                    # This gives the requested color range [0, max].
                    P1_plot = np.maximum(P1_sorted, 0.0)
                    cbar_label = "Amplitude"
                    plot_vmin = 0.0
                    plot_vmax = config.get("fft_contour_vmax")
                else:
                    if contour_normalize:
                        ref_idx = 0 if contour_ref == "first" else (
                            -1 if contour_ref == "last" else int(contour_ref))
                        P1_ref = P1_sorted[:, ref_idx]
                        P1_ref_safe = np.where(P1_ref < eps, eps, P1_ref)
                        P1_plot = 20.0 * np.log10(
                            P1_sorted / P1_ref_safe[:, np.newaxis] + eps)
                        cbar_label = "Amplitude factor [dB]"
                    else:
                        P1_plot = 10.0 * np.log10(P1_sorted + eps)
                        cbar_label = "Amplitude [dB]"
                    plot_vmin = None
                    plot_vmax = None
                first_signal = 1
                idx_f = slice(first_signal, n_freq)
                freq_sub = freq[idx_f]
                P1_plot_sub = P1_plot[idx_f, :]
                if len(freq_sub) > 1:
                    if contour_scale == "linear" and plot_vmax is None:
                        plot_vmax = float(np.nanmax(P1_plot_sub))
                    if contour_scale == "linear" and (
                            not np.isfinite(plot_vmax) or plot_vmax <= 0.0):
                        plot_vmax = 1.0
                    fig, ax = plt.subplots(figsize=(12, 6))
                    XX, YY = np.meshgrid(probe_x_sorted, freq_sub)
                    ax.pcolormesh(
                        XX, YY, P1_plot_sub,
                        shading="auto", cmap="inferno", rasterized=True,
                        vmin=plot_vmin, vmax=plot_vmax,
                    )
                    cb = fig.colorbar(ax.collections[0], ax=ax, pad=0.02, shrink=0.6)
                    cb.set_label(cbar_label, fontsize=11)
                    ax.set_xlabel("Probe X position [cm]", fontsize=12)
                    ax.set_ylabel("Frequency [Hz]", fontsize=12)
                    ax.set_yscale("log")
                    fig.tight_layout()
                    out = output_dir / f"{probe_prefix}_freq_vs_x_contour.png"
                    fig.savefig(str(out), dpi=150)
                    plt.close(fig)
                    _ts(f"  Saved freq-vs-position contour: {out}")
            except Exception as exc:
                _ts(f"  [W] Freq-vs-position contour failed: {exc}")

        # Harmonic amplitudes vs probe x
        if plot_harmonics and n_valid > 0:
            try:
                harmonic_list = [i * harmonic_freq for i in range(1, num_harmonics + 1)]
                harmonic_labels = [f"{i}xf0" for i in range(1, num_harmonics + 1)]
                idx_sorted = np.argsort(probe_x)
                probe_x_sorted = np.array(probe_x)[idx_sorted]
                P1_sorted = P1[:, idx_sorted]
                harmonic_amp = np.zeros((len(harmonic_list), len(idx_sorted)))
                for ih, hf in enumerate(harmonic_list):
                    for ip_sorted in range(len(idx_sorted)):
                        fbin, _ = _find_nearest_freq_bin(freq, hf)
                        harmonic_amp[ih, ip_sorted] = P1_sorted[fbin, ip_sorted]
                fig, ax = plt.subplots(figsize=(12, 6))
                for ih, label in enumerate(harmonic_labels):
                    ax.semilogy(probe_x_sorted, harmonic_amp[ih, :],
                                marker="o", markersize=3, lw=0.8, label=label)
                ax.set_xlabel("Probe X position [cm]", fontsize=12)
                ax.set_ylabel("Harmonic Amplitude", fontsize=12)
                ax.set_title(f"Harmonic amplitudes vs position (f0 = {harmonic_freq:.3e} Hz)", fontsize=13)
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_harmonics_vs_x.png"
                fig.savefig(str(out), dpi=150)
                plt.close(fig)
                _ts(f"  Saved harmonic amplitudes plot: {out}")
            except Exception as exc:
                _ts(f"  [W] Harmonic amplitudes plot failed: {exc}")

        # Spectral slope vs probe x
        if plot_spectral_slope and n_valid > 0:
            try:
                idx_sorted = np.argsort(probe_x)
                probe_x_sorted = np.array(probe_x)[idx_sorted]
                P1_sorted = P1[:, idx_sorted]
                slopes = np.full(len(idx_sorted), np.nan)
                r2s = np.full(len(idx_sorted), np.nan)
                for ip_sorted in range(len(idx_sorted)):
                    slopes[ip_sorted], _, r2s[ip_sorted] = _fit_loglog_slope(
                        freq, P1_sorted[:, ip_sorted], slope_fmin, slope_fmax)
                fig, ax = plt.subplots(figsize=(12, 5))
                valid = np.isfinite(slopes)
                ax.plot(probe_x_sorted[valid], slopes[valid], "o-", markersize=4, lw=1.0)
                ax.set_xlabel("Probe X position [cm]", fontsize=12)
                ax.set_ylabel("Spectral slope (log-log fit)", fontsize=12)
                ax.set_title(f"Spectral slope vs probe position", fontsize=13)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_spectral_slope_vs_x.png"
                fig.savefig(str(out), dpi=150)
                plt.close(fig)
                _ts(f"  Saved spectral slope plot: {out}")
            except Exception as exc:
                _ts(f"  [W] Spectral slope plot failed: {exc}")

        # Growth curves
        if plot_growth and n_valid > 0 and growth_freqs:
            try:
                idx_sorted = np.argsort(probe_x)
                probe_x_sorted = np.array(probe_x)[idx_sorted]
                P1_sorted = P1[:, idx_sorted]
                fig, ax = plt.subplots(figsize=(12, 6))
                for gf in growth_freqs:
                    fbin, actual_f = _find_nearest_freq_bin(freq, gf)
                    amp = P1_sorted[fbin, :]
                    ax.semilogy(probe_x_sorted, amp, "o-", markersize=3, lw=0.8,
                                label=f"f = {actual_f:.3e} Hz")
                ax.set_xlabel("Probe X position [cm]", fontsize=12)
                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_title("Amplitude growth curves vs probe position", fontsize=13)
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_growth_curves.png"
                fig.savefig(str(out), dpi=150)
                plt.close(fig)
                _ts(f"  Saved growth curves plot: {out}")
            except Exception as exc:
                _ts(f"  [W] Growth curves plot failed: {exc}")

        # ------------------------------------------------------------------
        # Phase 1: Disturbance signal reconstruction (windowed)
        # ------------------------------------------------------------------
        if config.get("make_disturbance_reconstruction", False) and not recon_probe_indices:
            _ts("  [W] Disturbance reconstruction skipped: no reconstruction probes selected")
        elif config.get("make_disturbance_reconstruction", False):
            _ts("  Running disturbance signal reconstruction ...")
            try:
                recon_method = config.get("reconstruction_method", "harmonics")
                num_harm = config.get("reconstruction_num_harmonics", 5)
                recon_freq = harmonic_freq
                recon_band = config.get("reconstruction_band", [1.0e6, 50.0e6])
                recon_output_dir = output_dir / "Reconstruction"
                recon_output_dir.mkdir(parents=True, exist_ok=True)

                raw = raw_matrix  # full-record raw signal
                dt_use_local = dt_actual
                plot_time = time_uniform if resample else probe_data[0]["time"][:L]

                # ============================================================
                # Stage 1: Disturbance window detection
                # ============================================================
                window_mode = config.get("reconstruction_window_mode", "per_probe")
                do_window = window_mode not in (None, "full")
                window_info_by_probe = {}
                common_start_idx = 0
                common_end_idx = L - 1

                if do_window:
                    laser_t = config.get("laser_start_time", None)
                    baseline_margin = config.get("reconstruction_baseline_margin", 0.001)
                    pre_frac = config.get("reconstruction_pre_event_fraction", 0.02)
                    onset_sig = config.get("reconstruction_onset_sigma", 5.0)
                    onset_peak_frac = config.get("reconstruction_onset_peak_fraction")
                    offset_sig = config.get("reconstruction_offset_sigma", 2.0)
                    offset_peak_frac = config.get("reconstruction_offset_peak_fraction")
                    packet_selection = config.get("reconstruction_packet_selection", "dominant_peak")
                    min_active = config.get("reconstruction_min_active_duration", 50e-6)
                    min_quiet = config.get("reconstruction_min_quiet_duration", 100e-6)
                    pad_bef = config.get("reconstruction_window_pad_before", 20e-6)
                    pad_aft = config.get("reconstruction_window_pad_after", 50e-6)
                    min_samp = config.get("reconstruction_min_samples", 256)
                    max_samp = config.get("reconstruction_max_samples", 50000)
                    det_mode = config.get("reconstruction_window_detector", "energy")

                    for ip in recon_probe_indices:
                        sig_raw = raw[:, ip]
                        win = fdb.detect_disturbance_window(
                            sig_raw, plot_time,
                            laser_start_time=laser_t,
                            baseline_margin=baseline_margin,
                            pre_event_fraction=pre_frac,
                            onset_sigma=onset_sig,
                            onset_peak_fraction=onset_peak_frac,
                            offset_sigma=offset_sig,
                            offset_peak_fraction=offset_peak_frac,
                            packet_selection=packet_selection,
                            min_active_duration=min_active,
                            min_quiet_duration=min_quiet,
                            pad_before=pad_bef,
                            pad_after=pad_aft,
                            min_samples=min_samp,
                            max_samples=max_samp,
                            detector_mode=det_mode,
                        )
                        window_info_by_probe[ip] = win

                    # Aggregate for common_detected mode
                    if window_mode == "common_detected":
                        valid_onsets = [w["onset_time"] for w in window_info_by_probe.values()
                                        if w["status"] != "invalid" and w["onset_time"] is not None]
                        valid_offsets = [w["offset_time"] for w in window_info_by_probe.values()
                                         if w["status"] != "invalid" and w["offset_time"] is not None]
                        if valid_onsets and valid_offsets:
                            onset_pct = config.get("reconstruction_common_onset_percentile", 10.0)
                            offset_pct = config.get("reconstruction_common_offset_percentile", 90.0)
                            common_start_time = float(np.percentile(valid_onsets, onset_pct))
                            common_end_time = float(np.percentile(valid_offsets, offset_pct))
                        else:
                            common_start_time = plot_time[0]
                            common_end_time = plot_time[-1]
                        common_start_idx = max(0, int(np.searchsorted(plot_time, common_start_time)))
                        common_end_idx = min(L - 1, int(np.searchsorted(plot_time, common_end_time)))
                        if common_end_idx <= common_start_idx:
                            common_end_idx = min(L - 1, common_start_idx + min_samp)
                    elif window_mode == "fixed":
                        common_start_time = config.get("reconstruction_window_start", plot_time[0])
                        common_end_time = config.get("reconstruction_window_end", plot_time[-1])
                        common_start_idx = max(0, int(np.searchsorted(plot_time, common_start_time)))
                        common_end_idx = min(L - 1, int(np.searchsorted(plot_time, common_end_time)))
                        if common_end_idx <= common_start_idx:
                            common_end_idx = min(L - 1, common_start_idx + min_samp)

                    # Save window diagnostics
                    if config.get("reconstruction_save_window_diagnostics", True):
                        try:
                            diag_out = recon_output_dir / "disturbance_windows.csv"
                            with open(str(diag_out), "w") as fh:
                                fh.write("probe_idx,x_cm,status,onset_time,offset_time,"
                                         "window_start,window_end,duration,n_samples,"
                                         "baseline,noise_scale,peak_time,baseline_source,"
                                         "fallback_reason\n")
                                for ip in recon_probe_indices:
                                    p = probe_data[ip]
                                    w = window_info_by_probe[ip]
                                    fh.write(f"{ip},{p['x']:.6f},{w['status']},"
                                             f"{w['onset_time']},{w['offset_time']},"
                                             f"{w['start_time']},{w['end_time']},"
                                             f"{w['duration']},{w['n_samples']},"
                                             f"{w['baseline']:.6e},{w['noise_scale']:.6e},"
                                             f"{w.get('pulse_peak_time')},{w.get('baseline_source')},"
                                             f"{w['fallback_reason'] or 'None'}\n")
                            _ts(f"  Saved window diagnostics: {diag_out}")
                        except Exception as exc:
                            _ts(f"  [W] Window diagnostics save failed: {exc}")
                else:
                    _ts("  Window mode='full' — using complete probe record")

                # ============================================================
                # Stage 2: Reconstruct on (possibly windowed) signals
                # ============================================================
                energy_budgets = []
                residual_stats = []
                recon_probe_x = []
                recon_window_info = {}

                for ip in recon_probe_indices:
                    sig_raw = raw[:, ip]
                    p = probe_data[ip]
                    recon_probe_x.append(p["x"])

                    # Extract windowed signal
                    if window_mode == "common_detected":
                        ws, wt, nw = fdb.extract_probe_window(
                            sig_raw, plot_time, common_start_idx, common_end_idx)
                        w_start_t = wt[0] if len(wt) > 0 else None
                        w_end_t = wt[-1] if len(wt) > 0 else None
                    elif window_mode == "per_probe":
                        if ip in window_info_by_probe:
                            wi = window_info_by_probe[ip]
                            ws, wt, nw = fdb.extract_probe_window(
                                sig_raw, plot_time, wi["start_idx"], wi["end_idx"])
                            w_start_t = wt[0] if len(wt) > 0 else None
                            w_end_t = wt[-1] if len(wt) > 0 else None
                        else:
                            ws, wt, nw = sig_raw.copy(), plot_time.copy(), L
                            w_start_t, w_end_t = None, None
                    elif window_mode == "fixed":
                        ws, wt, nw = fdb.extract_probe_window(
                            sig_raw, plot_time, common_start_idx, common_end_idx)
                        w_start_t = wt[0] if len(wt) > 0 else None
                        w_end_t = wt[-1] if len(wt) > 0 else None
                    else:  # "full" mode
                        ws, wt, nw = sig_raw.copy(), plot_time.copy(), L
                        w_start_t, w_end_t = None, None

                    # Fallback if window is too short
                    min_samp = config.get("reconstruction_min_samples", 256)
                    if nw < min_samp:
                        _ts(f"  [W] Probe {ip}: window too short ({nw} < {min_samp} samples), "
                            f"using full record")
                        ws, wt, nw = sig_raw.copy(), plot_time.copy(), L
                        w_start_t, w_end_t = None, None

                    recon_window_info[ip] = {
                        "start_time": w_start_t, "end_time": w_end_t,
                        "n_samples": nw, "windowed_signal": ws.ravel(),
                        "window_time": wt.ravel(),
                    }

                    # Mean-subtract the windowed signal
                    sig_mean = float(np.nanmean(ws))
                    sig = ws - sig_mean

                    if recon_method == "harmonics":
                        recon_total, comp_sigs, used_bins = fdb.reconstruct_from_harmonics(
                            sig, dt_use_local, recon_freq, num_harm,
                        )
                    elif recon_method == "band":
                        recon_total, used_bins = fdb.reconstruct_from_band(
                            sig, dt_use_local, recon_band[0], recon_band[1],
                        )
                        comp_sigs = {"band": recon_total.ravel()}
                    elif recon_method == "top_frequencies":
                        recon_total, comp_sigs, used_bins = fdb.reconstruct_from_top_frequencies(
                            sig, dt_use_local,
                            n_peaks=config.get("reconstruction_n_peaks", 15),
                            freq_range=config.get("reconstruction_peak_freq_range"),
                            min_peak_distance_hz=config.get("reconstruction_peak_min_distance_hz"),
                        )
                    else:
                        raise ValueError(f"Unknown reconstruction_method: {recon_method}")

                    recon_total = recon_total.ravel()
                    for key in comp_sigs:
                        comp_sigs[key] = comp_sigs[key].ravel()

                    residual = sig.ravel() - recon_total
                    eb = fdb.compute_energy_budget(sig.ravel(), recon_total, comp_sigs)
                    rs = fdb.compute_residual_stats(sig.ravel(), recon_total)
                    eb["mean_background"] = sig_mean
                    eb["reconstruction_method"] = recon_method
                    eb["fundamental_freq"] = recon_freq
                    eb["used_harmonic_bins"] = used_bins if recon_method != "top_frequencies" else []
                    energy_budgets.append(eb)
                    residual_stats.append(rs)

                    label = f"Probe {ip} — x={p['x']:.3f} cm"
                    out_path = recon_output_dir / f"{probe_prefix}_recon_probe{ip:03d}.png"
                    wi = recon_window_info[ip]
                    pdb.plot_harmonic_reconstruction(
                        plot_time,
                        sig_raw.ravel(),            # full signal (with mean)
                        recon_total + sig_mean,     # reconstruction shifted to match
                        residual,                   # residual on zero-mean basis
                        comp_sigs,
                        probe_label=label,
                        output_path=str(out_path),
                        mean_background=sig_mean,
                        recon_method=recon_method,
                        recon_freq=recon_freq,
                        window_start=wi["start_time"],
                        window_end=wi["end_time"],
                        windowed_signal=wi["windowed_signal"],
                        window_time=wi["window_time"],
                    )

                # ------------------------------------------------------------------
                # Window diagnostic plots (for selected probes)
                # ------------------------------------------------------------------
                if do_window and config.get("reconstruction_save_window_diagnostics", True):
                    try:
                        diag_probes = recon_probe_indices
                        diag_dir = recon_output_dir / "WindowDiagnostics"
                        diag_dir.mkdir(parents=True, exist_ok=True)
                        for ip in diag_probes:
                            w = window_info_by_probe.get(ip)
                            if w is None:
                                continue
                            if w.get("metric") is None:
                                continue
                            d_out = diag_dir / f"{probe_prefix}_window_diag_probe{ip:03d}.png"
                            p = probe_data[ip]
                            pdb.plot_disturbance_window_diagnostics(
                                plot_time,
                                raw[:, ip].ravel(),
                                w["metric"],
                                baseline=w["baseline"],
                                noise_scale=w["noise_scale"],
                                onset_threshold=w["onset_threshold"],
                                offset_threshold=w["offset_threshold"],
                                metric_baseline=w.get("metric_baseline"),
                                metric_noise_scale=w.get("metric_noise_scale"),
                                onset_time=w["onset_time"],
                                offset_time=w["offset_time"],
                                peak_time=w.get("pulse_peak_time"),
                                start_time=w["start_time"],
                                end_time=w["end_time"],
                                probe_label=f"Probe {ip} (x={p['x']:.3f} cm)",
                                output_path=str(d_out),
                            )
                        _ts(f"  Saved window diagnostic plots ({len(diag_probes)} probes)")
                    except Exception as exc:
                        _ts(f"  [W] Window diagnostic plots failed: {exc}")

                # Energy budget vs x
                pdb.plot_energy_budget_vs_x(
                    recon_probe_x, energy_budgets,
                    output_path=str(recon_output_dir / f"{probe_prefix}_energy_budget_vs_x.png"),
                )
                # Residual stats vs x
                pdb.plot_residual_vs_x(
                    recon_probe_x, residual_stats,
                    output_path=str(recon_output_dir / f"{probe_prefix}_residual_stats_vs_x.png"),
                )
                _ts(f"  Disturbance reconstruction complete — {len(recon_probe_indices)} probes processed")
            except Exception as exc:
                _ts(f"  [W] Disturbance reconstruction failed: {exc}")

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_stability_diagnostics(config):
    """Worker: run second Mack mode stability diagnostics."""
    try:
        output_dir = Path(config["output_dir"]) / "StabilityDiagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_name = config.get("stability_baseline_plotfile", "")
        baseline_path = os.path.join(config["data_source"], baseline_name)
        target_freq = config.get("stability_target_freq", None)
        freq_band = config.get("stability_freq_band", [100e3, 1.0e6])
        growth_win = config.get("stability_growth_window_size", None)
        num_gpi = config.get("stability_num_gpi_profiles", 5)

        _ts("  [SD] Loading baseline plotfile for BL profiles ...")
        baseline_ds = fdb.load_pelec_plotfile(
            baseline_path,
            field_names=None,
            alias_map=config.get("field_aliases"),
            convert_to_mks=True,
        )
        fdb.compute_derived_fields(baseline_ds)

        if "x_velocity" in baseline_ds["fields"] and "y_velocity" in baseline_ds["fields"]:
            vx = baseline_ds["fields"]["x_velocity"]
            vy = baseline_ds["fields"]["y_velocity"]
            baseline_ds["fields"]["velocity_magnitude"] = np.sqrt(vx**2 + vy**2)

        _ts("  [SD] Defining geometry surface (velocity method, threshold=50 m/s)...")
        surfaces = fdb.define_geometry_surface(
            baseline_ds,
            method="velocity",
            velocity_threshold=50.0,
            max_x=np.max(baseline_ds["x"]),
        )
        for side_name in ("upper", "lower"):
            d = surfaces.get(side_name, {})
            n_pts = len(d.get("x", []))
            _ts(f"  [SD] Surface '{side_name}': {n_pts} points")
        surfaces = fdb.compute_surface_normals(surfaces)

        upper = surfaces.get("upper", {})
        n_surf = len(upper.get("x", []))
        if n_surf < 3:
            return ("stability", False, "Too few surface points")

        _ts(f"  [SD] Extracting BL profiles at {n_surf} x-stations ...")
        bl_profiles = []
        rho_inf = config.get("surface_rho_inf", 0.0267)
        u_inf = config.get("surface_u_inf", 1726.0)

        for idx in range(n_surf):
            i = int(upper["i"][idx])
            j = int(upper["j"][idx])
            nx = upper.get("nx", [0.0] * n_surf)[idx]
            ny = upper.get("ny", [1.0] * n_surf)[idx]
            try:
                bl = fdb.extract_BL_profile_at_surface(
                    baseline_ds, i, j, nx, ny, u_inf, rho_inf)
                bl["x"] = upper["x"][idx]
                bl_profiles.append(bl)
            except Exception as exc:
                if idx < 5:
                    fdb._log_error(f"  [SD] BL profile at idx={idx}", exc)

        if len(bl_profiles) < 3:
            return ("stability", False, "Too few BL profiles")
        _ts(f"  [SD] Extracted {len(bl_profiles)} BL profiles")

        freq_data = fdb.compute_second_mode_frequency(bl_profiles)
        f_omega = np.nanmedian(freq_data["f_omega_star"])
        f_acou = np.nanmedian(freq_data["f_acoustic"])

        if target_freq is None:
            if np.isfinite(f_omega) and f_omega > 0:
                target_freq = f_omega
            elif np.isfinite(f_acou) and f_acou > 0:
                target_freq = f_acou
            else:
                target_freq = np.mean(freq_band)
            _ts(f"  [SD] Auto target frequency: {target_freq:.3e} Hz")
        else:
            _ts(f"  [SD] User target frequency: {target_freq:.3e} Hz")

        # Probe loading + FFT
        var_col = config.get("fft_var_col", 4)
        nt_skip = config.get("fft_nt_skip", 0)

        probe_data = _load_probe_timeseries(
            config, var_col, nt_skip=nt_skip, max_probes=None
        )
        if not probe_data:
            return ("stability", False, "No probe data")

        probe_x_m = [d["x"] * 1e-2 for d in probe_data]
        for d in probe_data:
            d["x"] = d["x"] * 1e-2
            if d["y"] is not None:
                d["y"] = d["y"] * 1e-2

        n_valid = len(probe_data)
        if n_valid < 3:
            return ("stability", False, "Too few valid probes")

        t_min = max(d["time"][0] for d in probe_data)
        t_max = min(d["time"][-1] for d in probe_data)
        dt_vals = [d["time"][1] - d["time"][0] for d in probe_data if len(d["time"]) >= 2]
        dt_use = np.median(dt_vals) if dt_vals else 1.0
        time_uniform = np.arange(t_min, t_max, dt_use)
        L = len(time_uniform)

        signal_matrix = np.zeros((L, n_valid))
        for ip, d in enumerate(probe_data):
            signal_matrix[:, ip] = np.interp(
                time_uniform, d["time"], d["signal"],
                left=d["signal"][0], right=d["signal"][-1],
            )

        # Apply same mean subtraction and windowing as FFT probe analysis
        signal_matrix, win_scale, _ = _preprocess_probe_signal(
            signal_matrix, config, time_uniform=time_uniform,
        )

        Y_complex = np.fft.fft(signal_matrix, axis=0)
        freq = np.fft.fftfreq(L, dt_use)
        half = L // 2
        freq_ss = freq[:half]
        Y_ss = Y_complex[:half, :] * win_scale
        P1 = np.abs(Y_ss) / L
        P1[1:, :] = 2.0 * P1[1:, :]
        _ts(f"  [SD] FFT complete: {len(freq_ss)} bins x {n_valid} probes")

        # Phase speed
        bl_x = freq_data["x"]
        u_edge_interp = np.interp(probe_x_m, bl_x, freq_data["u_edge"])
        T_edge_interp = None
        if "temperature" in baseline_ds["fields"]:
            T_edge_vals = np.zeros(len(bl_profiles))
            for i, bl in enumerate(bl_profiles):
                Tp = bl.get("T_profile", np.array([300.0]))
                T_edge_vals[i] = Tp[-1] if len(Tp) > 0 else 300.0
            T_edge_interp = np.interp(probe_x_m, bl_x, T_edge_vals)

        phase_data = fdb.compute_phase_speed_from_probes(
            np.array(probe_x_m), freq_ss, Y_ss, target_freq,
            u_edge=u_edge_interp, T_edge=T_edge_interp,
        )

        # Growth rate
        d99_interp = np.interp(probe_x_m, bl_x, freq_data["delta_99"])
        growth_data = fdb.compute_growth_rate_from_probes(
            np.array(probe_x_m), freq_ss, P1, target_freq,
            delta_99_interp=d99_interp, window_size=growth_win,
        )

        # GPI
        n_bl = len(bl_profiles)
        gpi_indices = np.linspace(0, n_bl - 1, min(num_gpi, n_bl)).astype(int)
        gpi_results = []
        for idx in gpi_indices:
            bl = bl_profiles[idx]
            yp = bl.get("y_profile", np.array([]))
            up = bl.get("u_profile", np.array([]))
            Tp = bl.get("T_profile", np.array([]))
            rp = bl.get("rho_profile", None)
            if rp is None or len(rp) == 0 or np.all(np.isnan(rp)):
                continue
            try:
                gpi = fdb.compute_gpi_criterion(yp, up, Tp, rp)
                gpi["x"] = bl["x"]
                gpi_results.append(gpi)
            except Exception:
                pass

        plot_data = {
            "freq_estimate": freq_data,
            "phase_speed": phase_data,
            "growth_rate": growth_data,
            "gpi_profiles": gpi_results,
            "target_freq": target_freq,
            "freq_band": freq_band,
            "bl_profiles": bl_profiles,
        }

        _ts("  [SD] Generating stability summary plot ...")
        out_path = output_dir / "stability_summary.png"
        pdb.plot_stability_summary(plot_data, output_path=str(out_path))

        if len(gpi_results) > 0:
            gpi_out = output_dir / "gpi_profiles.png"
            pdb.plot_gpi_profiles(bl_profiles, gpi_results, output_path=str(gpi_out))
            _ts(f"  [SD] Saved GPI profiles: {gpi_out}")

        _ts(f"  [SD] Saved stability summary: {out_path}")
        return ("stability", True, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return ("stability", False, str(exc))


# ---------------------------------------------------------------------------
#  PHASE 2 WORKER: TRANSIENT ANALYSIS (STFT / Hilbert envelope)
# ---------------------------------------------------------------------------

def _process_transient_analysis(config):
    """Worker: run STFT spectrogram and Hilbert envelope on probe signals.

    Uses the same probe-loading path as _process_fft_probes.  Operates
    on the mean-subtracted raw signal (unwindowed) for transient tracking.
    """
    try:
        output_dir = Path(config["output_dir"]) / "TransientAnalysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = "transient"

        var_col = config.get("fft_var_col", 4)
        nt_skip = config.get("fft_nt_skip", 0)
        max_probes = config.get("fft_max_probes", None)
        band = config.get("transient_band", [5.0e6, 50.0e6])
        nperseg = config.get("transient_stft_nperseg", 256)
        noverlap_frac = config.get("transient_stft_noverlap", 0.75)
        plot_probes = config.get("transient_plot_probe_indices", [0, 49, 99])

        probe_data = _load_probe_timeseries(config, var_col, nt_skip=nt_skip, max_probes=max_probes)
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        n_valid = len(probe_data)
        probe_x = [d["x"] for d in probe_data]
        _ts(f"  [TA] Loaded {n_valid} probes")

        # Resample to common uniform time base
        t_min = max(d["time"][0] for d in probe_data)
        t_max = min(d["time"][-1] for d in probe_data)
        dt_vals = [d["dt"] for d in probe_data if d["dt"] > 0]
        dt_use = np.median(dt_vals) if dt_vals else 1.0
        time_uniform = np.arange(t_min, t_max, dt_use)
        L = len(time_uniform)

        signal_matrix = np.zeros((L, n_valid))
        for ip, d in enumerate(probe_data):
            signal_matrix[:, ip] = np.interp(
                time_uniform, d["time"], d["signal"],
                left=d["signal"][0], right=d["signal"][-1],
            )

        fs = 1.0 / dt_use

        # Spectrograms for selected probes
        for idx in plot_probes:
            if idx >= n_valid:
                continue
            p = probe_data[idx]
            sig = signal_matrix[:, idx]
            # Mean-subtract
            sig_ms = sig - np.mean(sig)

            # STFT spectrogram
            try:
                noverlap = int(noverlap_frac * nperseg)
                out_stft = output_dir / f"spectrogram_probe{idx:03d}.png"
                pdb.plot_spectrogram(
                    time_uniform, sig_ms, fs,
                    output_path=str(out_stft),
                    fmax=band[1] * 2,
                    title=f"STFT spectrogram — Probe {idx} — x={p['x']:.3f} cm",
                )
                _ts(f"  [TA] Saved spectrogram: {out_stft}")
            except Exception as exc:
                _ts(f"  [TA] Spectrogram failed for probe {idx}: {exc}")

            # Hilbert envelope
            try:
                envelope, inst_phase, inst_freq, filtered = fdb.bandpass_hilbert_envelope(
                    sig_ms, fs, band[0], band[1], order=4,
                )
                pstats = fdb.extract_packet_stats(envelope, time_uniform)
                out_env = output_dir / f"envelope_probe{idx:03d}.png"
                pdb.plot_envelope_with_signal(
                    time_uniform, sig_ms, filtered, envelope, pstats,
                    output_path=str(out_env),
                    title=f"Envelope — Probe {idx} — x={p['x']:.3f} cm",
                )
                _ts(f"  [TA] Saved envelope: {out_env}")
            except Exception as exc:
                _ts(f"  [TA] Envelope failed for probe {idx}: {exc}")

        # Batch envelope stats vs x
        packet_stats_list = []
        for ip in range(n_valid):
            sig = signal_matrix[:, ip] - np.mean(signal_matrix[:, ip])
            try:
                envelope, _, _, _ = fdb.bandpass_hilbert_envelope(
                    sig, fs, band[0], band[1], order=4,
                )
                pstats = fdb.extract_packet_stats(envelope, time_uniform)
                packet_stats_list.append(pstats)
            except Exception:
                packet_stats_list.append({
                    "peak_amplitude": np.nan, "peak_time": np.nan,
                    "arrival_time": np.nan, "integrated_energy": np.nan,
                })

        if len(packet_stats_list) > 1:
            try:
                pdb.plot_envelope_growth(
                    probe_x, packet_stats_list,
                    output_path=str(output_dir / "envelope_growth_vs_x.png"),
                )
                _ts("  [TA] Saved envelope growth vs x")
            except Exception as exc:
                _ts(f"  [TA] Envelope growth plot failed: {exc}")

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


# ---------------------------------------------------------------------------
#  PHASE 3 WORKER: NONLINEAR INTERACTION DIAGNOSTICS (bispectrum)
# ---------------------------------------------------------------------------

def _process_nonlinear_diagnostics(config):
    """Worker: compute bicoherence for quadratic phase-coupling detection.

    Uses the same probe-loading path as _process_fft_probes.  Computes
    b²(f1, f2) for selected probes and tracks triad values vs x.
    """
    try:
        output_dir = Path(config["output_dir"]) / "NonlinearDiagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = "nonlinear"

        var_col = config.get("fft_var_col", 4)
        nt_skip = config.get("fft_nt_skip", 0)
        max_probes = config.get("fft_max_probes", None)
        nperseg = config.get("nonlinear_nperseg", 256)
        noverlap_frac = config.get("nonlinear_noverlap", 0.5)
        plot_probes = config.get("nonlinear_plot_probe_indices", [0, 49, 99])
        sig_level = config.get("nonlinear_significance_level", 0.95)

        probe_data = _load_probe_timeseries(config, var_col, nt_skip=nt_skip, max_probes=max_probes)
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        n_valid = len(probe_data)
        probe_x = [d["x"] for d in probe_data]
        _ts(f"  [NL] Loaded {n_valid} probes")

        # Resample to uniform time base
        t_min = max(d["time"][0] for d in probe_data)
        t_max = min(d["time"][-1] for d in probe_data)
        dt_vals = [d["dt"] for d in probe_data if d["dt"] > 0]
        dt_use = np.median(dt_vals) if dt_vals else 1.0
        time_uniform = np.arange(t_min, t_max, dt_use)
        L = len(time_uniform)

        signal_matrix = np.zeros((L, n_valid))
        for ip, d in enumerate(probe_data):
            signal_matrix[:, ip] = np.interp(
                time_uniform, d["time"], d["signal"],
                left=d["signal"][0], right=d["signal"][-1],
            )

        fs = 1.0 / dt_use
        noverlap = int(noverlap_frac * nperseg)

        # Determine fundamental frequency for triad targets
        harmonic_freq = config.get("fft_harmonic_freq", 10.0e6)
        target_freqs = [harmonic_freq * n for n in range(1, 6)]

        # Bicoherence maps for selected probes
        bicoherence_data = []
        for idx in plot_probes:
            if idx >= n_valid:
                continue
            p = probe_data[idx]
            sig = signal_matrix[:, idx]
            sig_ms = sig - np.mean(sig)
            try:
                freq_bic, bicoh = fdb.compute_bicoherence(
                    sig_ms, fs, nperseg=nperseg, noverlap=noverlap,
                )
                out_bic = output_dir / f"bicoherence_probe{idx:03d}.png"
                pdb.plot_bicoherence_map(
                    freq_bic, bicoh,
                    output_path=str(out_bic),
                    fmax=harmonic_freq * 4,
                    title=f"Bicoherence — Probe {idx} — x={p['x']:.3f} cm",
                )
                _ts(f"  [NL] Saved bicoherence map: {out_bic}")

                triad_bic = fdb.extract_triad_bicoherence(freq_bic, bicoh, target_freqs)
                bicoherence_data.append((idx, p["x"], triad_bic))
            except Exception as exc:
                _ts(f"  [NL] Bicoherence failed for probe {idx}: {exc}")
                bicoherence_data.append((idx, p["x"], {}))

        # Triad bicoherence vs x for all probes
        try:
            triad_all = []
            x_all = []
            remaining = [ip for ip in range(n_valid) if ip not in plot_probes]
            for idx in remaining:
                sig = signal_matrix[:, idx] - np.mean(signal_matrix[:, idx])
                try:
                    freq_bic, bicoh = fdb.compute_bicoherence(
                        sig, fs, nperseg=nperseg, noverlap=noverlap,
                    )
                    t_bic = fdb.extract_triad_bicoherence(freq_bic, bicoh, target_freqs)
                except Exception:
                    t_bic = {}
                triad_all.append(t_bic)
                x_all.append(probe_data[idx]["x"])

            # Merge with already computed
            for idx, x_pos, t_bic in bicoherence_data:
                triad_all.append(t_bic)
                x_all.append(x_pos)

            if len(triad_all) > 1:
                # Sort by x
                sort_idx = np.argsort(x_all)
                x_sorted = np.array(x_all)[sort_idx]
                triad_sorted = [triad_all[i] for i in sort_idx]
                pdb.plot_bicoherence_vs_x(
                    x_sorted, triad_sorted,
                    output_path=str(output_dir / "bicoherence_vs_x.png"),
                )
                _ts("  [NL] Saved bicoherence vs x")
        except Exception as exc:
            _ts(f"  [NL] Bicoherence vs x failed: {exc}")

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _run_workflow_mp(datasets, config, worker_func, desc="Processing"):
    """Run a workflow in parallel or sequentially."""
    args = [(ds, config) for ds in datasets]

    if config["use_multiprocessing"] and len(args) > 1:
        print(f"  Starting {desc} with {config['num_processes']} workers "
              f"for {len(args)} snapshots...")
        with Pool(processes=config["num_processes"]) as pool:
            results = pool.map(worker_func, args, chunksize=1)
    else:
        print(f"  {desc} sequentially...")
        results = [worker_func(a) for a in args]

    return results


def _print_results(results, desc):
    """Print a timestamped summary of successes / failures."""
    ok = sum(1 for r in results if r[1])
    fail = len(results) - ok
    _ts(f"{desc}: {ok} successful, {fail} failed")
    for r in results:
        if not r[1]:
            _ts(f"  ✗ [FAIL] {r[0]}: {r[2]}")


# ---------------------------------------------------------------------------
#  MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main(config=None):
    """Run the full post-processing pipeline.

    Parameters
    ----------
    config : dict, optional
        If provided, overrides the global CONFIG dict.
    """
    if config is None:
        config = CONFIG

    # Set debug mode
    fdb.set_debug_mode(config.get("debug_mode", False))

    # Create output directory
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Startup banner ---
    _hr("PeleC Post-Processing", char="=", width=75)
    print()  # blank line below banner title
    _ts(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _ts(f"Host:       {os.uname().nodename}")
    _ts(f"PID:        {os.getpid()}")
    print()
    _ts(f"Data source:       {config['data_source']}")
    _ts(f"Output directory:  {out_dir.resolve()}")
    _ts(f"Plot prefix:       {config['plot_prefix']}")
    _ts(f"Snapshot range:    {config['snapshot_start']} → {config['snapshot_end']}"
         f"  (step={config['snapshot_step']})")
    print()
    # List which workflows are enabled and disabled
    _ts("Workflow status:")
    _ts(f"  Contour plots:    {'ON  ✓' if config['make_contour_plots'] else 'OFF ✗'}")
    _ts(f"  Line profiles:    {'ON  ✓' if config['make_line_profiles'] else 'OFF ✗'}")
    _ts(f"  Streamlines:      {'ON  ✓' if config['make_streamlines'] else 'OFF ✗'}")
    _ts(f"  Surface analysis: {'ON  ✓' if config['make_surface_analysis'] else 'OFF ✗'}")
    _ts(f"  Group plots:      {'ON  ✓' if config['make_group_plots'] else 'OFF ✗'}")
    _ts(f"  Force analysis:   {'ON  ✓' if config.get('make_force_analysis', False) else 'OFF ✗'}")
    _ts(f"  Space-time plots: {'ON  ✓' if config.get('make_spacetime_plots', False) else 'OFF ✗'}")
    _ts(f"  Force animation:  {'ON  ✓' if config.get('make_force_animation', False) else 'OFF ✗'}")
    _ts(f"  Probe plots:      {'ON  ✓' if config.get('make_probe_plots', False) else 'OFF ✗'}")
    _ts(f"  FFT probes:       {'ON  ✓' if config.get('make_fft_probes', False) else 'OFF ✗'}")
    _ts(f"  p' contours:      {'ON  ✓' if config.get('make_pprime_contour', False) else 'OFF ✗'}")
    _ts(f"  Stability diag:   {'ON  ✓' if config.get('make_stability_diagnostics', False) else 'OFF ✗'}")
    _ts(f"  Disturbance recon:{'ON  ✓' if config.get('make_disturbance_reconstruction', False) else 'OFF ✗'}")
    _ts(f"  Transient analysis:{'ON  ✓' if config.get('make_transient_analysis', False) else 'OFF ✗'}")
    _ts(f"  Nonlinear diag:    {'ON  ✓' if config.get('make_nonlinear_diagnostics', False) else 'OFF ✗'}")
    print()
    _ts(f"Multiprocessing:   {config['use_multiprocessing']} "
         f"({config['num_processes']} workers)")
    _hr()

    # ------------------------------------------------------------------
    # 1. Discover plotfiles
    # ------------------------------------------------------------------
    _section(1, 10, "Discovering plotfiles ...")
    t0 = time.time()

    try:
        plotfile_paths = fdb.discover_plotfile_paths(
            config["data_source"],
            plot_prefix=config["plot_prefix"],
            start=config["snapshot_start"],
            end=config["snapshot_end"],
            step=config["snapshot_step"],
        )
    except Exception as exc:
        fdb._log_error("Failed to discover plotfiles", exc)
        return 1

    if len(plotfile_paths) == 0:
        _ts("ERROR: No plotfiles found!  Check data_source and plot_prefix.")
        return 1

    _ts(f"Found {len(plotfile_paths)} plotfiles"
         f" (first: {os.path.basename(plotfile_paths[0])},"
         f" last: {os.path.basename(plotfile_paths[-1])})")
    _ts(f"Mem strategy: ONE AT A TIME (low-memory mode)")

    # --- Load baseline for p' contour if needed ---
    baseline_fields = None
    if config.get("make_pprime_contour", False):
        baseline_name = config.get("pprime_baseline_plotfile", "")
        baseline_path = os.path.join(config["data_source"], baseline_name)
        _ts(f"Loading baseline plotfile for p' subtraction: {baseline_name}")
        try:
            baseline_ds = fdb.load_pelec_plotfile(
                baseline_path,
                field_names=None,
                alias_map=config.get("field_aliases"),
                convert_to_mks=True,
            )
            fdb.compute_derived_fields(baseline_ds)
            baseline_fields = baseline_ds["fields"]
            _ts("Baseline loaded and derived fields computed.")
        except Exception as exc:
            _ts(f"[W] Failed to load baseline for p': {exc}")
            baseline_fields = None

    # ------------------------------------------------------------------
    # 2. Process each plotfile individually
    # ------------------------------------------------------------------
    _section(2, 10, "Running workflows on each snapshot ...")

    contour_results = []
    pprime_results = []
    line_results = []
    streamline_results = []
    surface_results = []
    surface_data_dict = {}
    forces_dict = {}

    # Count enabled workflows for progress reporting
    enabled_workflows = sum([
        1 if config["make_contour_plots"] else 0,
        1 if config["make_line_profiles"] else 0,
        1 if config["make_streamlines"] else 0,
        1 if config["make_surface_analysis"] else 0,
    ])

    for i, pfile in enumerate(plotfile_paths, 1):
        label = os.path.basename(pfile)
        t_snap = time.time()

        _ts(f"┌─ Load {i:>4d}/{len(plotfile_paths)}:  {label}")
        t_load = time.time()

        try:
            dataset = fdb.load_pelec_plotfile(
                pfile,
                field_names=None,
                alias_map=config.get("field_aliases"),
                convert_to_mks=True,
            )
        except Exception as exc:
            fdb._log_error(f"Failed to load {label}", exc)
            _ts(f"├─ ✗ LOAD FAILED — {exc}")
            continue

        _ts(f"├─ Loaded in {time.time() - t_load:.1f} s  "
             f"({len(dataset.get('fields', []))} fields, "
             f"grid {dataset.get('grid_shape', '?')})")

        # Compute derived fields
        try:
            t_der = time.time()
            fdb.compute_derived_fields(dataset)
            _ts(f"├─ Derived fields computed in {time.time() - t_der:.1f} s")
        except Exception as exc:
            fdb._log_error(f"Derived fields failed for {label}", exc)
            _ts(f"├─ ✗ Derived fields FAILED — {exc}")

        # --- Contour plots ---
        if config["make_contour_plots"]:
            t_ci = time.time()
            try:
                r = _process_single_contour((dataset, config))
                contour_results.append(r)
                if r[1]:
                    _ts(f"├─ Contours → OK  "
                         f"({len(config['contour_fields'])} fields "
                         f"in {time.time() - t_ci:.1f} s)")
                else:
                    _ts(f"├─ ✗ Contours → FAIL — {r[2]}")
            except Exception as exc:
                contour_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ Contours → FAIL — {exc}")

        # --- p' (pressure perturbation) contour ---
        if config.get("make_pprime_contour", False) and baseline_fields is not None:
            t_pp = time.time()
            try:
                r = _process_pprime_contour((dataset, config, baseline_fields))
                pprime_results.append(r)
                if r[1]:
                    _ts(f"├─ p' contour -> OK  ({time.time() - t_pp:.1f} s)")
                else:
                    _ts(f"├─ ✗ p' contour -> FAIL — {r[2]}")
            except Exception as exc:
                pprime_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ p' contour -> FAIL — {exc}")

        # --- Line profiles ---
        if config["make_line_profiles"]:
            t_lp = time.time()
            try:
                r = _process_single_line((dataset, config))
                line_results.append(r)
                if r[1]:
                    _ts(f"├─ Line profiles → OK  "
                         f"({len(config['line_fields'])} fields × "
                         f"{len(config['line_x_stations'])} stations, "
                         f"{time.time() - t_lp:.1f} s)")
                else:
                    _ts(f"├─ ✗ Line profiles → FAIL — {r[2]}")
            except Exception as exc:
                line_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ Line profiles → FAIL — {exc}")

        # --- Streamlines ---
        if config["make_streamlines"]:
            t_sl = time.time()
            try:
                r = _process_single_streamline((dataset, config))
                streamline_results.append(r)
                if r[1]:
                    _ts(f"├─ Streamlines → OK  "
                         f"({time.time() - t_sl:.1f} s)")
                else:
                    _ts(f"├─ ✗ Streamlines → FAIL — {r[2]}")
            except Exception as exc:
                streamline_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ Streamlines → FAIL — {exc}")

        # --- Surface analysis ---
        if config["make_surface_analysis"]:
            t_sa = time.time()
            try:
                r = _process_single_surface((dataset, config))
                surface_results.append(r)
                if r[1]:
                    _ts(f"├─ Surface analysis → OK  "
                         f"({time.time() - t_sa:.1f} s)")
                    if isinstance(r[2], dict):
                        payload = r[2]
                        surf_data = payload.get("surface", {})
                        force_data = payload.get("forces", None)
                        surface_data_dict[r[0]] = surf_data
                        if force_data is not None:
                            forces_dict[r[0]] = force_data
                else:
                    _ts(f"├─ ✗ Surface analysis → FAIL — {r[2]}")
            except Exception as exc:
                surface_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ Surface analysis → FAIL — {exc}")

        # Per-snapshot timing summary
        t_snap_elapsed = time.time() - t_snap
        _ts(f"└─ Done {label} — total {t_snap_elapsed:.1f} s")

        # Estimate remaining time
        if i < len(plotfile_paths):
            remaining = (len(plotfile_paths) - i) * t_snap_elapsed
            _ts(f"    ⏳ Est. remaining: ~{remaining/60:.1f} min  "
                 f"({len(plotfile_paths) - i} snapshots left)", end="\n\n")

        # Explicitly drop the dataset to free memory before next iteration
        del dataset

    # ------------------------------------------------------------------
    # 2b. Probe time-history plots (runs before FFT to ensure .dat files exist)
    # ------------------------------------------------------------------
    if config.get("make_probe_plots", False):
        _hr("Probe Processing", char="\u2500", width=75)

        probe_out = Path(config["probe_output_dir"])
        probe_max = config.get("probe_max", None)

        # Check if conversion already done
        dat_files = list(probe_out.glob("flow_probe_*.dat"))

        if len(dat_files) == 0:
            _ts("Converting binary probe files to ASCII .dat ...")
            converter = Path(__file__).parent / "convert_probes.py"
            if not converter.is_file():
                _ts(f"ERROR: convert_probes.py not found at {converter}")
            else:
                bin_files = [str(Path(b).resolve()) for b in config["probe_bin_files"]]
                try:
                    subprocess.run(
                        [sys.executable, str(converter),
                         "--input"] + bin_files +
                        ["--output", str(probe_out.resolve()),
                         "--dedup-tol", "1e-12"],
                        check=True
                    )
                    _ts("Probe conversion complete.")
                except subprocess.CalledProcessError as exc:
                    _ts(f"ERROR: Probe conversion failed (return code {exc.returncode})")
                    _ts("Skipping probe plots.")
                    probe_out = None
        else:
            _ts(f"Probe .dat files already exist ({len(dat_files)} found), skipping conversion.")

        # Load and plot
        if probe_out is not None:
            _ts("Loading probe data ...")
            try:
                probe_data = fdb.load_probe_dat_files(
                    str(probe_out),
                    max_probes=probe_max
                )
                _ts(f"Loaded {len(probe_data)} probes (max={probe_max}).")

                if len(probe_data) > 0:
                    _ts("Generating probe time-history plots ...")
                    pdb.plot_probe_timeseries(
                        probe_data,
                        output_dir=out_dir / "Probes",
                        laser_start_time=config.get("laser_start_time"),
                        convert_to_mks=config.get("probe_convert_to_mks", False),
                    )
                    _ts(f"Probe plots saved to {out_dir / 'Probes'}")
            except Exception as exc:
                fdb._log_error("Probe plotting failed", exc)

        _hr()

    # ------------------------------------------------------------------
    # 3. FFT probe analysis (runs once, not per-snapshot)
    # ------------------------------------------------------------------
    if config.get("make_fft_probes", False):
        _ts("Running FFT probe analysis ...")
        try:
            r = _process_fft_probes(config)
            if r[1]:
                _ts("FFT probe analysis -> OK")
            else:
                _ts(f"✗ FFT probe analysis -> FAIL — {r[2]}")
        except Exception as exc:
            _ts(f"✗ FFT probe analysis -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 4. Stability diagnostics (runs once, not per-snapshot)
    # ------------------------------------------------------------------
    if config.get("make_stability_diagnostics", False):
        _ts("Running stability diagnostics ...")
        try:
            r = _process_stability_diagnostics(config)
            if r[1]:
                _ts("Stability diagnostics -> OK")
            else:
                _ts(f"✗ Stability diagnostics -> FAIL — {r[2]}")
        except Exception as exc:
            _ts(f"✗ Stability diagnostics -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 5. Transient analysis (STFT / Hilbert envelope)
    # ------------------------------------------------------------------
    if config.get("make_transient_analysis", False):
        _ts("Running transient analysis ...")
        try:
            r = _process_transient_analysis(config)
            if r[1]:
                _ts("Transient analysis -> OK")
            else:
                _ts(f"✗ Transient analysis -> FAIL — {r[2]}")
        except Exception as exc:
            _ts(f"✗ Transient analysis -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 6. Nonlinear interaction diagnostics (bispectrum)
    # ------------------------------------------------------------------
    if config.get("make_nonlinear_diagnostics", False):
        _ts("Running nonlinear diagnostics ...")
        try:
            r = _process_nonlinear_diagnostics(config)
            if r[1]:
                _ts("Nonlinear diagnostics -> OK")
            else:
                _ts(f"✗ Nonlinear diagnostics -> FAIL — {r[2]}")
        except Exception as exc:
            _ts(f"✗ Nonlinear diagnostics -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 7. Print summaries
    # ------------------------------------------------------------------
    _section(7, 10, "Workflow results summary ...")

    if config["make_contour_plots"]:
        _print_results(contour_results, "Contour plots")
    if config.get("make_pprime_contour", False):
        _print_results(pprime_results, "p' contours")
    if config["make_line_profiles"]:
        _print_results(line_results, "Line profiles")
    if config["make_streamlines"]:
        _print_results(streamline_results, "Streamlines")
    if config["make_surface_analysis"]:
        _print_results(surface_results, "Surface analysis")

    # ------------------------------------------------------------------
    # 10. Group plots (original)
    # ------------------------------------------------------------------
    _section(8, 10, "Generating group plots ...")

    if config["make_group_plots"] and surface_data_dict:
        try:
            snap_list = sorted(surface_data_dict.keys(), key=fdb.natural_sort_key)
            group_out = out_dir / "SurfaceAnalysis" / "group_properties.png"
            group_out.parent.mkdir(parents=True, exist_ok=True)
            pdb.plot_surface_properties_group(
                surface_data_dict,
                snap_list,
                output_path=str(group_out),
            )
            _ts(f"Group surface plot saved: {group_out}")
        except Exception as exc:
            fdb._log_error("Group plot failed", exc)
    else:
        _ts("No group plots requested or no surface data available.")

    # ------------------------------------------------------------------
    # 11. Force time-series
    # ------------------------------------------------------------------
    _section(9, 10, "Generating force time-series plots ...")

    if config["make_force_analysis"] and forces_dict:
        try:
            force_out = out_dir / "forces_vs_time.png"
            pdb.plot_forces_vs_time(
                forces_dict,
                output_path=str(force_out),
                laser_start_time=config.get("laser_start_time"),
            )
            _ts(f"Force time-series saved: {force_out}")

            # Also save raw data as .npz for external analysis
            npz_out = out_dir / "forces_timeseries.npz"
            snap_labels = sorted(forces_dict.keys())
            n = len(snap_labels)
            f0 = forces_dict[snap_labels[0]]
            # Build arrays
            arrays = {"labels": snap_labels}
            scalar_keys = ["time", "C_D", "C_L", "C_Dp", "C_Dv", "C_Lp", "C_Lv",
                           "D_total", "L_total", "D_p", "D_v", "L_p", "L_v"]
            for k in scalar_keys:
                if k in f0:
                    arrays[k] = np.array([forces_dict[lbl].get(k, np.nan) for lbl in snap_labels])
            np.savez(npz_out, **arrays)
            _ts(f"Force data saved: {npz_out}")
        except Exception as exc:
            fdb._log_error("Force time-series plot failed", exc)
    else:
        _ts("Force analysis not requested or no force data available.")

    # ------------------------------------------------------------------
    # 12. Space-time plots
    # ------------------------------------------------------------------
    _section(10, 10, "Generating space-time contour plots ...")

    if config["make_spacetime_plots"] and surface_data_dict:
        spacetime_dir = out_dir / "SpaceTime"
        spacetime_dir.mkdir(parents=True, exist_ok=True)
        spacetime_fields = ["C_p", "C_f", "tau_w", "delta_99", "H"]

        # Build time-annotated copy for space-time plotting
        annotated_sd = {}
        for lbl, sd in surface_data_dict.items():
            annotated_sd[lbl] = {"upper": sd.get("upper", {}).copy(),
                                 "lower": sd.get("lower", {}).copy()}
            if lbl in forces_dict and "time" in forces_dict[lbl]:
                annotated_sd[lbl]["time"] = forces_dict[lbl]["time"]
            else:
                annotated_sd[lbl]["time"] = 0.0

        for st_key in spacetime_fields:
            try:
                # Check if the key exists in first snapshot
                first_sd = {}
                for lbl, asd in annotated_sd.items():
                    if len(asd.get("upper", {}).get("x", [])) > 0:
                        first_sd = asd
                        break
                if st_key not in first_sd.get("upper", {}):
                    continue
                st_out = spacetime_dir / f"spacetime_{st_key}.png"
                pdb.plot_spacetime(
                    annotated_sd,
                    field_key=st_key,
                    output_path=str(st_out),
                    laser_start_time=config.get("laser_start_time"),
                )
                _ts(f"  Space-time {st_key} saved: {st_out}")
            except Exception as exc:
                fdb._log_error(f"Space-time plot for {st_key} failed", exc)
    else:
        _ts("Space-time plots not requested or no surface data available.")

    # ------------------------------------------------------------------
    # 13. Animation of contour with surface
    # ------------------------------------------------------------------
    # Note: Animation requires all datasets in memory (they were freed per
    # snapshot).  If animation is requested, datasets must be re-loaded.
    # This is expensive; we only do it on explicit demand.
    if config["make_force_animation"] and surface_data_dict:
        _ts("Animation requested — re-loading datasets into memory...")
        anim_dir = out_dir / "Animation"
        anim_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Re-discover plotfiles (same range as before)
            anim_plotfiles = fdb.discover_plotfile_paths(
                config["data_source"],
                plot_prefix=config["plot_prefix"],
                start=config["snapshot_start"],
                end=config["snapshot_end"],
                step=config["snapshot_step"],
            )
            # Reload all datasets and recompute surfaces into series dicts
            anim_datasets = {}
            anim_surfaces = {}
            for pfile in anim_plotfiles:
                lbl = os.path.basename(pfile)
                try:
                    ds = fdb.load_pelec_plotfile(pfile, convert_to_mks=True)
                    fdb.compute_derived_fields(ds)
                    anim_datasets[lbl] = ds
                    if lbl in surface_data_dict:
                        surfs = fdb.define_geometry_surface(
                            ds,
                            method=config["surface_geometry_method"],
                            velocity_threshold=config["surface_velocity_threshold"],
                            max_x=config["surface_max_x"],
                            geometry_type=config.get("geometry_type", "auto"),
                            plate_leading_edge=config.get("plate_leading_edge", 0.0),
                        )
                        anim_surfaces[lbl] = surfs
                except Exception as exc:
                    fdb._log_error(f"Reload failed for {lbl}", exc)

            if anim_datasets:
                anim_field = config.get("contour_fields", ["mach_number"])[0]
                anim_out = str(anim_dir / f"loading_evolution.gif")
                pdb.animate_contour_with_surface(
                    anim_datasets,
                    field_key=anim_field,
                    surfaces_series=anim_surfaces if anim_surfaces else None,
                    output_path=anim_out,
                    xlim=config.get("contour_xlim"),
                    ylim=config.get("contour_ylim"),
                    fps=8,
                    laser_start_time=config.get("laser_start_time"),
                )
                _ts(f"Animation saved: {anim_out}")
            else:
                _ts("No datasets could be re-loaded for animation.")
        except Exception as exc:
            fdb._log_error("Animation generation failed", exc)
    else:
        _ts("Animation not requested or no surface data available.")

    # ------------------------------------------------------------------
    # 14. Done
    # ------------------------------------------------------------------
    total_elapsed = time.time() - _t_start_global
    _hr("Complete", char="=", width=75)
    _ts(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _ts(f"Total wall time: {total_elapsed/60:.1f} min  ({total_elapsed:.1f} s)")
    _ts(f"Results in: {out_dir.resolve()}")
    _ts(f"Snapshots processed: {len(plotfile_paths)}")
    _hr()
    return 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
