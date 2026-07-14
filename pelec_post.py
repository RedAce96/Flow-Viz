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

import argparse
import glob
import gc
import importlib.metadata
import json
import os
import platform
import re
import struct
import subprocess
import sys
import time
import datetime
from multiprocessing import Pool
from pathlib import Path

# Keep plotting/font caches writable for direct login-node validation runs as
# well as Slurm jobs. Respect explicit user settings when they are provided.
_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / (
    f"pelec-post-{os.environ.get('USER', 'user')}"
)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

# Import our modular databases
import pp_functions_database as fdb
import pp_config
import pp_modal_database as mdb
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


def _plot_title(dataset, subject, config):
    """Return a consistent title containing configured time coordinates."""
    return pdb.dataset_title(
        dataset,
        subject,
        reference_time=config.get("plot_time_reference"),
        time_origin=config.get("plot_time_origin", 0.0),
        time_mode=config.get("plot_time_mode", "flow_through"),
        freestream_velocity=config.get(
            "plot_flow_through_u_inf", config.get("surface_u_inf")
        ),
    )


def _plot_time_label(dataset, config):
    """Return the configured time coordinate without a field-name prefix."""
    return pdb.format_dataset_time(
        dataset,
        mode=config.get("plot_time_mode", "flow_through"),
        freestream_velocity=config.get(
            "plot_flow_through_u_inf", config.get("surface_u_inf")
        ),
        reference_time=config.get("plot_time_reference"),
        origin=config.get("plot_time_origin", 0.0),
    )


def _required_plotfile_fields(config):
    """Return the smallest safe base-field set for enabled snapshot plots.

    ``None`` deliberately requests the complete default set when a workflow
    needs derived fields, geometry detection, or arbitrary streamline fields.
    """
    if config.get("make_surface_analysis") or config.get("make_streamlines"):
        return None

    direct_fields = {
        "density", "pressure", "temperature",
        "x_velocity", "y_velocity", "z_velocity",
    }
    requested = set()
    if config.get("make_contour_plots"):
        requested.update(config.get("contour_fields", []))
    if config.get("make_line_profiles"):
        requested.update(config.get("line_fields", []))
        requested.add("x_velocity")  # delta_99 coordinate
        if config.get("line_blasius_compressible_overlay"):
            requested.update(("density", "temperature"))
    if config.get("make_pprime_contour"):
        requested.add(config.get("pprime_field", "pressure"))

    if not requested or not requested.issubset(direct_fields):
        return None
    return sorted(requested)


def _json_safe(value):
    """Convert configuration/provenance values to JSON-safe objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance():
    repo = Path(__file__).resolve().parent
    result = {"commit": None, "dirty": None}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        result["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _write_run_manifest(output_dir, config, status, plotfiles=None,
                        failures=None, started_at=None, command=None,
                        slurm_context=None, run_notes=None):
    """Write an atomic provenance and run-status manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for path in plotfiles or []:
        item = {"path": str(Path(path).resolve())}
        try:
            stat = Path(path).stat()
            item.update({"mtime": stat.st_mtime, "size_bytes": stat.st_size})
        except OSError:
            item["missing"] = True
        inputs.append(item)
    outputs = []
    if status != "running":
        outputs = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "run_manifest.json"
        )
    manifest = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command if command is not None else [sys.executable, *sys.argv],
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "matplotlib", "yt")
        },
        "git": _git_provenance(),
        "slurm": slurm_context if slurm_context is not None else {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST",
                "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            )
        },
        "config": _json_safe(config),
        "inputs": inputs,
        "outputs": outputs,
        "failures": _json_safe(failures or []),
        "run_notes": _json_safe(run_notes or []),
    }
    destination = output_dir / "run_manifest.json"
    temporary = output_dir / ".run_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)
    return destination

# ---------------------------------------------------------------------------
#  USER CONFIGURATION
# ---------------------------------------------------------------------------
#  Edit the values below to control what gets processed.
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Data source ---
    "data_source": "../TS-Driver/FP-Extended-Domain/pltFile",   # Directory with plotfiles
    "plot_prefix": "pltFlatPlatePost",                         # Plotfile directory prefix
    # Dedicated review folder: preserves older output families while showing
    # the corrected figure formats and analysis products from this workflow.
    "output_dir": "../TS-Driver/FP-Extended-Domain/1-Plot-Outputs/Analysis-Review",

    # --- Snapshot range ---
    # Set to None to process all discovered plotfiles.
    "snapshot_start": 350000,
    "snapshot_end": 380000,
    "snapshot_step": 1000,

    # --- Field aliases ---
    # Map solver raw names -> canonical names.  If omitted, default PeleC
    # aliases are used.  Only add / override if your case uses non-standard
    # naming.
    "field_aliases": None,

    # --- Workflow toggles ---
    "make_contour_plots": True,   # flow-through-time contour titles
    "make_line_profiles": True,   # y/delta_99 boundary-layer profiles
    "make_streamlines": False,
    "make_surface_analysis": False,
    "make_group_plots": False,

    "make_probe_plots": False,   # set True only if you need time-history plots (requires ASCII conversion)
    "make_fft_probes": True,      # set True to run FFT / stability analysis on probe data
    "make_pprime_contour": True,       # symmetric perturbation contours
    "make_stability_diagnostics": True, # corrected phase-speed references


    # --- Contour plot settings ---
    # List of canonical field names to plot.  Any field present in the
    # dataset (including derived fields) can be used.
    "contour_fields": [
        "density",
    ],
    "contour_cmap": "viridis",           # Perceptually uniform scalar-field map
    "contour_norm": "linear",               # Color scaling: "linear", "log", "symlog", or a matplotlib Normalize object
    "contour_vlims": {                       # Per-field color limits [vmin, vmax]
        "density": [None, None],             # robust 1st--99th percentile autoscale
    },
    "contour_xlim": [-0.001, 0.4],           # full 0.4 m plate
    "contour_ylim": None,                    # [ymin, ymax] or None for full domain
    # Time shown in figures. Flow-through time is t_FT = L_x / U_inf, where
    # L_x is the full AMReX domain length (independent of contour x-limits).
    "plot_time_mode": "flow_through",       # "flow_through" | "physical" | "reference"
    "plot_flow_through_u_inf": 1726.0,      # U_inf [m/s]
    "plot_time_reference": None,
    "plot_time_origin": 0.0,

    # --- Line profile settings ---
    "line_x_stations": [0.05, 0.3],                  # x-locations to extract profiles [m]
    "line_fields": ["x_velocity", "temperature"],
    "line_ylim": [0, 0.005],                              # [ymin, ymax] or None for full domain height
    "line_normalize_by_delta99": True,                    # plot y/delta_99 instead of y [m]
    "line_normalized_coordinate_limits": [0.0, 2.0],      # focus on BL + near-edge region

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
    # Surface-dependent products remain disabled because this review run does
    # not enable surface analysis.
    "make_force_analysis": False,
    "make_spacetime_plots": False,
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
    "fft_batch_size": 32,             # Probe columns per vectorized FFT batch
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
    # Welch coherence is evaluated on adjacent probe pairs near the selected
    # plotting stations. None builds [(i, i+1), ...] automatically.
    "make_coherence_analysis": True,
    "coherence_probe_pairs": None,
    "coherence_nperseg": 16384,
    "coherence_noverlap": 0.5,
    "coherence_fmax": 50.0e6,
    # Modal screening on selected probe stations. These decompositions are
    # explicitly descriptive and are not substituted for LST eigenmodes.
    "make_modal_analysis": True,
    "modal_probe_indices": None,  # None -> selected FFT/reconstruction probes
    "modal_n_modes": 4,
    "modal_spod_nperseg": 4096,
    "modal_spod_noverlap": 0.5,
    "modal_spod_frequency_stride": 2,
    "modal_spod_fmax": 50.0e6,

    # --- Pressure perturbation (p') contour ---
    "pprime_baseline_plotfile": "pltFlatPlateFlow210000",
    "base_flow_definition": (
        "Independent pre-laser instantaneous plotfile; replace with a "
        "verified steady/Favre/ensemble base before making LST attribution"
    ),
    "pprime_field": "pressure",
    "pprime_cmap": "RdBu_r",
    "pprime_vlims": [-500, 500],

    # --- Stability diagnostics (2nd Mack mode) ---
    # Existing pre-event/base-flow plotfile; use an independently verified
    # steady base here when a more appropriate baseline becomes available.
    "stability_baseline_plotfile": "pltFlatPlateFlow210000",
    "stability_target_freq": None,
    "stability_freq_band": [100e3, 1.0e6],
    "stability_growth_window_size": None,
    "stability_num_gpi_profiles": 5,
    "stability_coherence_nperseg": 16384,
    "stability_coherence_noverlap": 0.5,
    "stability_min_coherence": 0.5,
    # Downstream wave convention: exp(i*(omega*t - alpha*x)).
    "stability_phase_convention": "omega_t_minus_alpha_x",

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
    "reconstruction_peak_min_prominence": None,    # FFT-magnitude prominence; None -> no cutoff
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
    # Shared frequencies are selected on a training interval, then their
    # fitted amplitudes/phases are extrapolated to held-out data. This guards
    # against interpreting same-window Fourier fit as predictive linear modes.
    "reconstruction_common_mode_validation": True,
    "reconstruction_common_n_modes": 10,
    "reconstruction_common_train_fraction": 0.6,
    "reconstruction_common_freq_range": [1.0e5, 50.0e6],

    # --- Phase 2: Transient analysis (STFT / Hilbert envelope) ---
    # Time-frequency analysis for laser-pulse wavepacket tracking.
    "make_transient_analysis": True,
    "transient_band": [1.0e5, 1.0e6],       # includes heuristic/observed low-MHz modes
    "transient_stft_nperseg": 16384,
    "transient_stft_noverlap": 0.75,
    "transient_plot_probe_indices": [0, 49, 99, 249, 499],
    "transient_probe_stride": 4,

    # --- Phase 3: Nonlinear interaction diagnostics (bispectrum) ---
    # Quadratic phase-coupling detection via bicoherence.
    "make_nonlinear_diagnostics": True,
    "nonlinear_nperseg": 16384,             # resolves O(0.1 MHz) content
    "nonlinear_noverlap": 0.5,              # overlap fraction
    "nonlinear_plot_probe_indices": [0, 49, 99, 249, 499],
    "nonlinear_target_freqs": None,          # None -> strongest independent peaks
    "nonlinear_target_band": [1.0e5, 50.0e6],
    "nonlinear_num_target_modes": 5,
    "nonlinear_fmax": 50.0e6,               # limits full-map cost
    "nonlinear_probe_stride": 4,             # spatial sampling for triad trend
    # No significance line is drawn without a validated null/surrogate model.
    "nonlinear_reference_threshold": None,

    # --- Logging ---
    "debug_mode": True,
}


def run_validation_case(output_dir="validation_outputs"):
    """Run a fast synthetic case with known spectral and propagation answers.

    This test does not read AMReX plotfiles or production probe binaries. It
    writes four figures that exercise the corrected phase convention,
    same-window reconstruction metric, absolute-time PSD spectrogram,
    instantaneous frequency, and bicoherence calculation.

    Run from the project directory with::

        python pelec_post.py --validation-case
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _hr("Synthetic analysis validation")
    _ts(f"Writing validation products to: {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # 1. A known downstream wave: exp(i*(omega*t-alpha*x)).
    # ------------------------------------------------------------------
    phase_frequency = 5.0e6
    expected_phase_speed = 800.0
    probe_x = np.arange(8, dtype=float) * 20.0e-6
    alpha = 2.0 * np.pi * phase_frequency / expected_phase_speed
    phase_fft = np.exp(-1j * alpha * probe_x)[None, :]
    phase_result = fdb.compute_phase_speed_from_probes(
        probe_x,
        np.array([phase_frequency]),
        phase_fft,
        phase_frequency,
        u_edge=np.full(len(probe_x), 1000.0),
        T_edge=np.full(len(probe_x), 300.0),
        phase_convention="omega_t_minus_alpha_x",
    )
    measured_phase_speed = float(np.nanmedian(phase_result["c_p"]))
    expected_slow_speed = 1000.0 - np.sqrt(1.4 * 287.05 * 300.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(phase_result["x_mid"] * 1.0e3, phase_result["c_p"], "o-",
            label="Recovered downstream wave")
    ax.axhline(expected_phase_speed, color="k", linestyle="--",
               label="Imposed 800 m/s")
    ax.plot(phase_result["x_mid"] * 1.0e3,
            phase_result["c_p_slow"], color="C2", linestyle=":",
            label=r"Slow acoustic reference $U_e-a_e$")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$c_p$ [m s$^{-1}$]")
    ax.set_title(r"Phase convention validation: $e^{i(\omega t-\alpha x)}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "01_phase_speed_validation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. A finite-duration two-frequency packet for reconstruction/STFT.
    # ------------------------------------------------------------------
    fs = 100.0e6
    n_samples = 8192
    t_relative = np.arange(n_samples, dtype=float) / fs
    time = 5.0e-3 + t_relative
    active = slice(1024, 7168)
    window_time = time[active]
    packet_coordinate = np.linspace(0.0, 1.0, active.stop - active.start)
    packet_envelope = np.sin(np.pi * packet_coordinate) ** 2
    packet = packet_envelope * (
        1.0 * np.cos(2.0 * np.pi * 5.0e6 * t_relative[active] + 0.2)
        + 0.45 * np.cos(2.0 * np.pi * 12.0e6 * t_relative[active] - 0.4)
    )
    measured = np.full(n_samples, 100.0)
    measured[active] += packet
    windowed_signal = measured[active]
    mean_background = float(np.mean(windowed_signal))
    disturbance = windowed_signal - mean_background
    reconstructed, band_mask = fdb.reconstruct_from_band(
        disturbance, 1.0 / fs, 4.0e6, 10.0e6
    )
    reconstructed = reconstructed.ravel()
    residual = disturbance - reconstructed
    residual_stats = fdb.compute_residual_stats(disturbance, reconstructed)
    pdb.plot_harmonic_reconstruction(
        time,
        measured,
        reconstructed + mean_background,
        residual,
        {"4--10 MHz band": reconstructed},
        probe_label="Synthetic finite-duration packet",
        output_path=str(output_dir / "02_reconstruction_validation.png"),
        mean_background=mean_background,
        relative_rms=residual_stats["rms_residual_rel"],
        recon_method="band",
        window_start=float(window_time[0]),
        window_end=float(window_time[-1]),
        windowed_signal=windowed_signal,
        window_time=window_time,
    )

    nperseg = 512
    noverlap = 384
    pdb.plot_spectrogram(
        time,
        measured - np.mean(measured),
        fs,
        output_path=str(output_dir / "03_psd_spectrogram_validation.png"),
        fmax=20.0e6,
        title="PSD spectrogram validation (absolute flow time)",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    envelope, _, inst_freq, filtered = fdb.bandpass_hilbert_envelope(
        measured - np.mean(measured), fs, 4.0e6, 7.0e6, order=4
    )
    packet_stats = fdb.extract_packet_stats(envelope, time)
    pdb.plot_envelope_with_signal(
        time,
        measured - np.mean(measured),
        filtered,
        envelope,
        packet_stats,
        output_path=str(output_dir / "04_envelope_frequency_validation.png"),
        title="Hilbert envelope and instantaneous frequency validation",
        instantaneous_frequency=inst_freq,
        frequency_band=(4.0e6, 7.0e6),
    )

    # ------------------------------------------------------------------
    # 3. A deliberately phase-coupled triad for bicoherence.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(20260713)
    f1, f2 = 5.0e6, 8.0e6
    coupled = (
        np.cos(2.0 * np.pi * f1 * t_relative + 0.2)
        + np.cos(2.0 * np.pi * f2 * t_relative - 0.4)
        + 0.5 * np.cos(2.0 * np.pi * (f1 + f2) * t_relative - 0.2)
        + 0.25 * rng.standard_normal(n_samples)
    )
    bicoh_freq, bicoh = fdb.compute_bicoherence(
        coupled, fs, nperseg=500, noverlap=250
    )
    i1 = int(np.argmin(np.abs(bicoh_freq - f1)))
    i2 = int(np.argmin(np.abs(bicoh_freq - f2)))
    coupled_b2 = float(bicoh[min(i1, i2), max(i1, i2)])
    pdb.plot_bicoherence_map(
        bicoh_freq,
        bicoh,
        output_path=str(output_dir / "05_bicoherence_validation.png"),
        fmax=20.0e6,
        title=(r"Known coupled triad: 5 + 8 = 13 MHz; "
               rf"$b^2={coupled_b2:.3f}$"),
    )

    # ------------------------------------------------------------------
    # 4. Coherence and held-out common-frequency prediction.
    # ------------------------------------------------------------------
    common_x = np.linspace(0.0, 0.03, 6)
    common_f1 = 205.0 * fs / (n_samples // 2)
    common_f2 = 492.0 * fs / (n_samples // 2)
    common_signals = np.column_stack([
        np.cos(2.0 * np.pi * common_f1 * t_relative - 0.25 * station)
        + 0.4 * np.cos(2.0 * np.pi * common_f2 * t_relative + 0.1 * station)
        for station in range(len(common_x))
    ])
    coherence = fdb.compute_pair_coherence(
        common_signals[:, 0], common_signals[:, 1], fs,
        nperseg=1024, noverlap=512,
    )
    pdb.plot_pair_coherence(
        [{
            "indices": (0, 1), "label": "synthetic adjacent pair",
            "result": coherence,
        }],
        output_path=str(output_dir / "06_coherence_validation.png"),
        fmax=20.0e6,
    )
    common_model = fdb.fit_common_frequency_model(
        common_signals, 1.0 / fs, n_modes=2, train_fraction=0.5,
        freq_range=(1.0e6, 20.0e6),
    )
    pdb.plot_common_mode_validation(
        time, common_model, common_x * 100.0,
        output_path=str(output_dir / "07_common_mode_holdout_validation.png"),
    )
    common_validation_rms = float(np.max(
        common_model["validation_relative_rms"]
    ))

    # ------------------------------------------------------------------
    # 5. POD/SPOD/DMD on a known traveling wave.
    # ------------------------------------------------------------------
    modal_x = np.linspace(0.0, 0.04, 8, endpoint=False)
    modal_values = np.cos(
        2.0 * np.pi * 5.0e6 * t_relative[:, None]
        - 2.0 * np.pi * modal_x[None, :] / 0.04
    )
    modal_dataset = mdb.SnapshotMatrix(
        time, modal_values, modal_x[:, None], variable="synthetic pressure"
    )
    pod_validation = mdb.compute_pod(modal_dataset, n_modes=3)
    spod_grid = np.fft.rfftfreq(512, 1.0 / fs)
    spod_selected = np.flatnonzero(spod_grid <= 20.0e6)[::4]
    spod_validation = mdb.compute_spod(
        modal_dataset, nperseg=512, noverlap=256, n_modes=2,
        frequency_indices=spod_selected,
    )
    dmd_validation = mdb.compute_dmd(modal_dataset, n_modes=2)
    pdb.plot_modal_summary(
        pod_validation, spod_validation, dmd_validation,
        output_path=str(output_dir / "08_modal_validation.png"),
    )
    recovered_dmd_frequency = float(np.nanmedian(
        np.abs(dmd_validation["frequency_hz"])
    ))

    # Machine-readable values make the visual case suitable for regression.
    np.savez(
        output_dir / "validation_results.npz",
        expected_phase_speed=expected_phase_speed,
        measured_phase_speed=measured_phase_speed,
        expected_slow_speed=expected_slow_speed,
        measured_slow_speed=np.nanmedian(phase_result["c_p_slow"]),
        reconstruction_relative_rms=residual_stats["rms_residual_rel"],
        bicoherence_coupled_triad=coupled_b2,
        stft_nperseg=nperseg,
        stft_noverlap=noverlap,
        reconstruction_band_mask=band_mask,
        coherence_frequency_hz=coherence["frequency_hz"],
        coherence_squared=coherence["coherence_squared"],
        common_selected_frequency_hz=common_model["selected_frequency_hz"],
        common_validation_relative_rms=common_model["validation_relative_rms"],
        pod_energy_fraction=pod_validation["energy_fraction"],
        recovered_dmd_frequency_hz=recovered_dmd_frequency,
    )

    if not np.isclose(measured_phase_speed, expected_phase_speed, rtol=1.0e-10):
        raise RuntimeError("Phase-speed validation failed")
    if not np.isclose(np.nanmedian(phase_result["c_p_slow"]),
                      expected_slow_speed, rtol=1.0e-10):
        raise RuntimeError("Slow-acoustic-reference validation failed")
    if coupled_b2 < 0.8:
        raise RuntimeError("Coupled-triad bicoherence validation failed")
    if common_validation_rms > 1.0e-8:
        raise RuntimeError("Held-out common-frequency validation failed")
    if not np.isclose(recovered_dmd_frequency, 5.0e6, rtol=1.0e-3):
        raise RuntimeError("DMD frequency validation failed")

    _ts(f"Phase speed: expected {expected_phase_speed:.3f}, "
        f"recovered {measured_phase_speed:.3f} m/s")
    _ts(f"Slow acoustic reference: {expected_slow_speed:.3f} m/s")
    _ts(f"Same-window reconstruction relative RMS: "
        f"{residual_stats['rms_residual_rel']:.4f}")
    _ts(f"Known coupled-triad bicoherence: {coupled_b2:.4f}")
    _ts(f"Common-mode held-out max relative RMS: {common_validation_rms:.3e}")
    _ts(f"DMD frequency: expected 5.000e6, recovered {recovered_dmd_frequency:.6e} Hz")
    _ts("Validation PASSED")
    return 0


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
                title=_plot_title(dataset, field_key, config),
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
                    field_names=_required_plotfile_fields(config),
                    alias_map=config.get("field_aliases"),
                    convert_to_mks=True,
                )
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

        compare_bl_heights = {}
        if compare_ds is not None and "x_velocity" in compare_ds["fields"]:
            for x_loc in config["line_x_stations"]:
                try:
                    u_prof = fdb.extract_line(compare_ds, x_loc, "x_velocity")
                    if line_ylim is not None and len(u_prof["y"]) > 1:
                        y_min, y_max = line_ylim
                        mask = (u_prof["y"] >= y_min) & (u_prof["y"] <= y_max)
                        u_prof["y"] = u_prof["y"][mask]
                        u_prof["values"] = u_prof["values"][mask]
                    u_inf_ref = bl_cfg.get("u_inf")
                    if u_inf_ref is None:
                        u_inf_ref = float(np.nanmax(np.abs(u_prof["values"])))
                    d99, _, _, _ = fdb.calculate_BL_thicknesses(
                        u_prof["y"],
                        np.clip(np.abs(u_prof["values"]) / float(u_inf_ref), 0.0, 1.5),
                    )
                    compare_bl_heights[x_loc] = float(d99)
                except Exception as exc:
                    fdb._log_error(
                        f"Comparison boundary-layer height at x={x_loc:.3f}", exc
                    )
                    compare_bl_heights[x_loc] = None

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
                    prof["label"] = rf"Simulation, $x={x_loc:.3f}$ m"
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
                                ref[field_key]["label"] = rf"Blasius, $x={x_loc:.3f}$ m"
                                ref[field_key].setdefault("boundary_layer_height", bl_heights.get(x_loc))
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
                            cprof["label"] = (
                                rf"{config.get('line_compare_label', 'Comparison')}, "
                                rf"$x={x_loc:.3f}$ m"
                            )
                            cprof["color"] = config.get("line_compare_color", "C2")
                            cprof["linestyle"] = config.get("line_compare_linestyle", ":")
                            cprof["boundary_layer_height"] = compare_bl_heights.get(x_loc)
                            compare_profiles.append(cprof)
                        except Exception as exc:
                            fdb._log_error(f"Comparison line extraction at x={x_loc:.3f}", exc)
                except Exception as exc:
                    fdb._log_error(f"Line extraction at x={x_loc:.3f}", exc)

            if compare_profiles:
                profiles.extend(compare_profiles)

            if profiles:
                out = output_dir / f"{label}_{field_key}_profiles.png"
                normalize_y = config.get("line_normalize_by_delta99", False)
                pdb.plot_line_profiles(
                    profiles,
                    output_path=str(out),
                    title=_plot_title(dataset, field_key, config),
                    swap_axes=True,
                    xlabel=(r"$y/\delta_{99}$" if normalize_y else r"$y$ [m]"),
                    coordinate_normalization=(
                        "boundary_layer_height" if normalize_y else None
                    ),
                    annotate_boundary_layer=not normalize_y,
                    coordinate_limits=(
                        config.get("line_normalized_coordinate_limits")
                        if normalize_y else None
                    ),
                )

            if compressible_profiles:
                out = output_dir / f"{label}_{field_key}_compressible_blasius.png"
                pdb.plot_line_profiles(
                    compressible_profiles,
                    field_label=r"U/U_e",
                    xlabel=r"Transformed coordinate $\eta_{vd}$",
                    output_path=str(out),
                    title=(
                        f"Compressible Blasius transform — "
                        + _plot_time_label(dataset, config)
                    ),
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
            title=(
                "Streamlines — "
                + _plot_time_label(dataset, config)
            ),
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
                snapshot=_plot_time_label(dataset, config),
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
            snapshot=_plot_time_label(dataset, config),
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
    dataset, config, baseline_dataset = args
    try:
        output_dir = Path(config["output_dir"]) / "PPrime"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        field_key = config.get("pprime_field", "pressure")
        if field_key not in dataset["fields"]:
            return (label, False, f"Field '{field_key}' not in dataset")
        if field_key not in baseline_dataset["fields"]:
            return (label, False, f"Field '{field_key}' not in baseline")

        p_post = dataset["fields"][field_key]
        p_base = baseline_dataset["fields"][field_key]
        p_prime = fdb.subtract_rectilinear_baseline(
            p_post, dataset["x"], dataset["y"],
            p_base, baseline_dataset["x"], baseline_dataset["y"],
            method=config.get("pprime_interpolation", "linear"),
            chunk_size=config.get("pprime_interpolation_chunk_size", 128),
        )
        pprime_ds = dict(dataset)
        pprime_ds["fields"] = {field_key: p_prime}

        cmap = config.get("pprime_cmap", "RdBu_r")
        vlims = config.get("pprime_vlims", [None, None])
        vmin, vmax = vlims[0], vlims[1]

        out = output_dir / f"{label}_pprime.png"
        pdb.plot_contour(
            pprime_ds, field_key,
            output_path=str(out),
            title=(
                "Pressure perturbation — "
                + _plot_time_label(dataset, config)
            ),
            cmap=pdb.resolve_cmap(cmap),
            norm="linear",
            vmin=vmin,
            vmax=vmax,
            colorbar_label=r"$p'$ [Pa]" if field_key == "pressure"
            else rf"$\Delta$ {pdb.field_label(field_key)}",
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

    n_load = n_probes if max_probes is None else min(int(max_probes), n_probes)
    probe_ids = range(n_load)

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
        if len(raw) != record_payload:
            raise EOFError(
                f"Short probe record in {bin_files[file_idx]} at byte {offset}"
            )
        # Interpret the complete probe-by-field record in compiled NumPy code.
        # The previous nested Python loop executed once per timestep *and*
        # probe (hundreds of millions of iterations for production cases).
        data = np.frombuffer(
            raw,
            dtype=np.dtype(f"{endian}f8"),
            count=n_probes * n_fields,
        ).reshape(n_probes, n_fields)
        time_buf[it] = time
        signal_buf[it, :] = data[:n_load, field_idx]
        if it == 0:
            x_sample[:] = data[:n_load, x_field]
            y_sample[:] = data[:n_load, y_field]
            x_sample_set[:] = True

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
            # All probes share the same immutable time base; signal columns
            # remain views backed by signal_buf. This avoids duplicating the
            # multi-gigabyte production arrays once per returned probe.
            "time": time_buf,
            "signal": signal_buf[:, j],
            "dt": float(time_buf[1] - time_buf[0]) if time_buf.size >= 2 else 0.0,
            "filename": f"flow_probe_{ip:03d}.dat",
        })

    if probe_data:
        # Keep one explicit reference to the contiguous backing array.  This
        # lets downstream workflows operate on column views rather than
        # rebuilding another multi-gigabyte matrix from the individual dicts.
        probe_data[0]["_shared_signal_matrix"] = signal_buf
        probe_data[0]["_shared_uniform_time"] = True

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


def _probe_matrix_on_common_time(probe_data, resample=True, target_dt=None):
    """Return a common time base and matrix with the fewest possible copies.

    Binary probe records already share a uniform contiguous matrix.  That
    matrix is returned directly when no time-step change is requested.  ASCII
    or genuinely asynchronous records are interpolated once.
    """
    if not probe_data:
        raise ValueError("probe_data is empty")

    first = probe_data[0]
    native_time = np.asarray(first["time"], dtype=float)
    shared = first.get("_shared_signal_matrix")
    is_shared_uniform = bool(first.get("_shared_uniform_time", False))
    if native_time.size < 2:
        raise ValueError("At least two probe samples are required")
    native_dt = float(np.median(np.diff(native_time)))

    if shared is not None and is_shared_uniform:
        shared = np.asarray(shared)
        uniform = np.allclose(
            np.diff(native_time), native_dt, rtol=1.0e-8,
            atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
        )
        same_dt = target_dt is None or np.isclose(
            float(target_dt), native_dt, rtol=1.0e-8,
            atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
        )
        if not resample or (uniform and same_dt):
            return native_time, shared[:, :len(probe_data)], native_dt, True

    n_valid = len(probe_data)
    if resample:
        t_min = max(d["time"][0] for d in probe_data)
        t_max = min(d["time"][-1] for d in probe_data)
        dt_vals = [d["dt"] for d in probe_data if d.get("dt", 0.0) > 0]
        dt_use = float(target_dt) if target_dt is not None else (
            float(np.median(dt_vals)) if dt_vals else native_dt
        )
        # Include a common final sample when it lies on the uniform grid.
        count = int(np.floor((t_max - t_min) / dt_use + 1.0e-9)) + 1
        time_uniform = t_min + np.arange(count, dtype=float) * dt_use
        matrix = np.empty((count, n_valid), dtype=float)
        for ip, d in enumerate(probe_data):
            matrix[:, ip] = np.interp(
                time_uniform, d["time"], d["signal"],
                left=d["signal"][0], right=d["signal"][-1],
            )
        return time_uniform, matrix, dt_use, False

    length = min(d["time"].size for d in probe_data)
    matrix = np.empty((length, n_valid), dtype=float)
    for ip, d in enumerate(probe_data):
        matrix[:, ip] = d["signal"][:length]
    return native_time[:length], matrix, native_dt, False


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


def _preprocess_probe_signal(signal_matrix, config, time_uniform=None,
                             copy_raw=True, log_details=True):
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
        Copy of the input before any modification when ``copy_raw`` is true.
    """
    L, n_probes = signal_matrix.shape
    mean_mode = config.get("fft_mean_subtraction", "mean")
    window_type = config.get("fft_window", "hann")
    do_comp = config.get("fft_window_compensation", True)

    # Save a copy of the raw (unprocessed) signal for dual-row time-trace plots
    raw_matrix = signal_matrix.copy() if copy_raw else None

    # --- Per-probe mean subtraction / detrend ---
    if mean_mode == "none":
        pass
    elif mean_mode == "mean":
        signal_matrix -= np.nanmean(signal_matrix, axis=0, keepdims=True)
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

    if log_details and (
            mean_mode != "none"
            or window_type not in ("none", "rect", "rectangular")):
        _ts(f"  Preprocess: mean='{mean_mode}'  window='{window_type}'")

    return signal_matrix, scale, raw_matrix


def _process_fft_probes(config, probe_data=None):
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
        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=max_probes
            )

        if not probe_data:
            return (label, False, "No probe data could be loaded")

        probe_x = [d["x"] for d in probe_data]
        probe_y = [d["y"] for d in probe_data]
        n_valid = len(probe_data)
        _ts(f"  Successfully loaded {n_valid} probes")

        time_uniform, raw_matrix, dt_actual, reused_shared = \
            _probe_matrix_on_common_time(
                probe_data, resample=resample, target_dt=target_dt
            )
        L = len(time_uniform)
        # Preserve the unprocessed record for time traces/reconstruction and
        # create only the one working copy required by the FFT.
        signal_matrix = raw_matrix.copy()
        if reused_shared:
            _ts("  Reusing contiguous binary probe matrix (no interpolation copy)")

        # Apply mean subtraction and windowing
        signal_matrix, win_scale, _ = _preprocess_probe_signal(
            signal_matrix, config, time_uniform=time_uniform, copy_raw=False,
        )

        Fs = 1.0 / dt_actual
        freq = Fs * np.arange(0, L // 2 + 1) / L
        n_freq = len(freq)
        P1 = np.zeros((n_freq, n_valid))
        fft_batch_size = max(1, int(config.get("fft_batch_size", 32)))
        for first in range(0, n_valid, fft_batch_size):
            last = min(first + fft_batch_size, n_valid)
            Y = np.fft.rfft(signal_matrix[:, first:last], axis=0)
            P1[:, first:last] = np.abs(Y / L) * win_scale
            P1[1:-1, first:last] *= 2.0
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

        # Compact, reusable spectral output.  The full 2,000-probe spectrum
        # can exceed a gigabyte, so retain all spatial amplitudes only at the
        # configured growth frequencies and full spectra for selected probes.
        export_indices = sorted(set(plot_indices + recon_probe_indices))
        growth_bins = np.array([
            _find_nearest_freq_bin(freq, value)[0] for value in growth_freqs
        ], dtype=int) if growth_freqs else np.empty(0, dtype=int)
        growth_actual = freq[growth_bins] if growth_bins.size else np.empty(0)
        selected_spectra = (
            P1[:, export_indices] if export_indices
            else np.empty((len(freq), 0), dtype=float)
        )
        np.savez_compressed(
            output_dir / "spectral_summary.npz",
            frequency_hz=freq,
            probe_x_cm=np.asarray(probe_x, dtype=float),
            probe_y_cm=np.asarray(probe_y, dtype=float),
            mean_amplitude=np.nanmean(P1, axis=1),
            selected_probe_indices=np.asarray(export_indices, dtype=int),
            selected_amplitude=selected_spectra,
            growth_frequency_hz=growth_actual,
            growth_amplitude=P1[growth_bins, :] if growth_bins.size
            else np.empty((0, n_valid), dtype=float),
            dominant_frequency_hz=np.array(harmonic_freq),
            sample_interval_s=np.array(dt_actual),
            window_amplitude_scale=np.array(win_scale),
        )
        _ts(f"  Saved reusable spectral data: {output_dir / 'spectral_summary.npz'}")

        if config.get("make_coherence_analysis", False) and export_indices:
            configured_pairs = config.get("coherence_probe_pairs")
            if configured_pairs is None:
                configured_pairs = [
                    (idx, idx + 1) for idx in export_indices if idx + 1 < n_valid
                ]
            pair_results = []
            nperseg_coh = int(config.get("coherence_nperseg", 16384))
            overlap_coh = int(
                config.get("coherence_noverlap", 0.5) * nperseg_coh
            )
            for first, second in configured_pairs:
                first, second = int(first), int(second)
                if not (0 <= first < n_valid and 0 <= second < n_valid):
                    raise ValueError(
                        f"Invalid coherence probe pair ({first}, {second})"
                    )
                result = fdb.compute_pair_coherence(
                    raw_matrix[:, first], raw_matrix[:, second], Fs,
                    nperseg=nperseg_coh, noverlap=overlap_coh,
                )
                pair_results.append({
                    "indices": (first, second),
                    "label": (
                        f"{first}->{second} "
                        f"({probe_x[first]:.3f}->{probe_x[second]:.3f} cm)"
                    ),
                    "result": result,
                })
            if pair_results:
                coherence_values = np.column_stack([
                    item["result"]["coherence_squared"] for item in pair_results
                ])
                phase_values = np.column_stack([
                    item["result"]["cross_phase_rad"] for item in pair_results
                ])
                np.savez_compressed(
                    output_dir / "pair_coherence.npz",
                    frequency_hz=pair_results[0]["result"]["frequency_hz"],
                    probe_pairs=np.asarray([
                        item["indices"] for item in pair_results
                    ], dtype=int),
                    coherence_squared=coherence_values,
                    cross_phase_rad=phase_values,
                    nperseg=np.array(pair_results[0]["result"]["nperseg"]),
                    noverlap=np.array(pair_results[0]["result"]["noverlap"]),
                )
                pdb.plot_pair_coherence(
                    pair_results,
                    output_path=str(output_dir / "pair_coherence.png"),
                    fmax=config.get("coherence_fmax"),
                )
                _ts(f"  Saved Welch coherence for {len(pair_results)} probe pairs")

        if config.get("make_modal_analysis", False):
            modal_indices_cfg = config.get("modal_probe_indices")
            modal_indices = export_indices if modal_indices_cfg is None else sorted(set(
                int(value) for value in modal_indices_cfg
            ))
            if len(modal_indices) < 2:
                raise ValueError("Modal analysis requires at least two probe stations")
            if modal_indices[0] < 0 or modal_indices[-1] >= n_valid:
                raise ValueError("modal_probe_indices contains an invalid index")
            coordinates = np.column_stack((
                np.asarray(probe_x)[modal_indices],
                np.asarray(probe_y)[modal_indices],
            ))
            # All Fourier stages use the median dt when resampling is disabled.
            # Give modal algorithms that explicit uniform coordinate rather
            # than pretending the ppm-level native clock jitter is exact.
            modal_time = time_uniform[0] + np.arange(L, dtype=float) * dt_actual
            snapshots = mdb.SnapshotMatrix(
                modal_time,
                raw_matrix[:, modal_indices],
                coordinates,
                variable=var_labels.get(var_col, f"column_{var_col}"),
            )
            modal_count = min(
                int(config.get("modal_n_modes", 4)), len(modal_indices)
            )
            pod_result = mdb.compute_pod(snapshots, n_modes=modal_count)
            spod_nperseg = min(
                int(config.get("modal_spod_nperseg", 4096)), L
            )
            spod_frequency = np.fft.rfftfreq(spod_nperseg, dt_actual)
            spod_stride = max(
                1, int(config.get("modal_spod_frequency_stride", 2))
            )
            spod_fmax = config.get("modal_spod_fmax")
            spod_indices = np.arange(0, len(spod_frequency), spod_stride)
            if spod_fmax is not None:
                spod_indices = spod_indices[
                    spod_frequency[spod_indices] <= float(spod_fmax)
                ]
            spod_result = mdb.compute_spod(
                snapshots,
                nperseg=spod_nperseg,
                noverlap=int(
                    config.get("modal_spod_noverlap", 0.5) * spod_nperseg
                ),
                n_modes=min(modal_count, 3),
                frequency_indices=spod_indices,
            )
            dmd_result = mdb.compute_dmd(snapshots, n_modes=modal_count)
            modal_dir = output_dir / "ModalAnalysis"
            modal_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                modal_dir / "modal_results.npz",
                probe_indices=np.asarray(modal_indices, dtype=int),
                coordinates_cm=coordinates,
                pod_modes=pod_result["modes"],
                pod_energy_fraction=pod_result["energy_fraction"],
                pod_temporal_coefficients=pod_result["temporal_coefficients"],
                spod_frequency_hz=spod_result["frequency_hz"],
                spod_eigenvalues=spod_result["eigenvalues"],
                spod_modes=spod_result["modes"],
                dmd_eigenvalues=dmd_result["eigenvalues"],
                dmd_frequency_hz=dmd_result["frequency_hz"],
                dmd_growth_rate_per_s=dmd_result["growth_rate_per_s"],
                dmd_modes=dmd_result["modes"],
            )
            pdb.plot_modal_summary(
                pod_result, spod_result, dmd_result,
                output_path=str(modal_dir / "modal_summary.png"),
            )
            _ts(
                f"  Saved POD/SPOD/DMD screening for {len(modal_indices)} probes"
            )

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
                identity_order = np.array_equal(idx_sorted, np.arange(n_valid))
                P1_sorted = P1 if identity_order else P1[:, idx_sorted]
                eps = 1e-20
                contour_scale = config.get("fft_contour_scale", "linear").lower()
                if contour_scale == "linear":
                    # Use the physical nonnegative FFT amplitude directly.
                    # This gives the requested color range [0, max].
                    P1_plot = P1_sorted
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
                    ax.pcolormesh(
                        probe_x_sorted, freq_sub, P1_plot_sub,
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
                plot_time = time_uniform

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
                            min_peak_prominence=config.get(
                                "reconstruction_peak_min_prominence"
                            ),
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
                    eb["used_frequency_bins"] = used_bins
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
                        relative_rms=rs["rms_residual_rel"],
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

                if config.get("reconstruction_common_mode_validation", True):
                    common_signals = raw[:, recon_probe_indices]
                    common_model = fdb.fit_common_frequency_model(
                        common_signals,
                        dt_use_local,
                        n_modes=config.get("reconstruction_common_n_modes", 10),
                        train_fraction=config.get(
                            "reconstruction_common_train_fraction", 0.6
                        ),
                        freq_range=config.get(
                            "reconstruction_common_freq_range"
                        ),
                        min_peak_distance_hz=config.get(
                            "reconstruction_peak_min_distance_hz"
                        ),
                    )
                    common_path = recon_output_dir / "common_mode_validation.npz"
                    np.savez_compressed(
                        common_path,
                        probe_indices=np.asarray(recon_probe_indices, dtype=int),
                        probe_x_cm=np.asarray(recon_probe_x, dtype=float),
                        selected_frequency_hz=common_model[
                            "selected_frequency_hz"
                        ],
                        spectral_score=common_model["spectral_score"],
                        n_train=np.array(common_model["n_train"]),
                        train_relative_rms=common_model["train_relative_rms"],
                        validation_relative_rms=common_model[
                            "validation_relative_rms"
                        ],
                        validation_observed=common_model[
                            "validation_observed"
                        ],
                        validation_prediction=common_model[
                            "validation_prediction"
                        ],
                    )
                    pdb.plot_common_mode_validation(
                        plot_time, common_model, recon_probe_x,
                        output_path=str(
                            recon_output_dir / "common_mode_validation.png"
                        ),
                    )
                    _ts(
                        "  Common-mode held-out median relative RMS: "
                        f"{np.nanmedian(common_model['validation_relative_rms']):.3f}"
                    )
                _ts(f"  Disturbance reconstruction complete — {len(recon_probe_indices)} probes processed")
            except Exception as exc:
                _ts(f"  [W] Disturbance reconstruction failed: {exc}")
                return (label, False, f"Disturbance reconstruction failed: {exc}")

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_stability_diagnostics(config, probe_data=None):
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

        # Extract all baseline-dependent edge quantities before releasing the
        # large AMReX covering-grid fields.  Probe spectral work below needs
        # the compact profiles only.
        T_edge_vals = None
        if "temperature" in baseline_ds["fields"]:
            T_edge_vals = np.array([
                bl.get("T_profile", np.array([300.0]))[-1]
                if len(bl.get("T_profile", [])) > 0 else 300.0
                for bl in bl_profiles
            ], dtype=float)
        del baseline_ds, surfaces

        # Probe loading + FFT
        var_col = config.get("fft_var_col", 4)
        nt_skip = config.get("fft_nt_skip", 0)

        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=None
            )
        if not probe_data:
            return ("stability", False, "No probe data")

        probe_x_m = [d["x"] * 1e-2 for d in probe_data]

        n_valid = len(probe_data)
        if n_valid < 3:
            return ("stability", False, "Too few valid probes")

        time_uniform, signal_matrix, dt_use, reused_shared = \
            _probe_matrix_on_common_time(
                probe_data, resample=config.get("fft_resample", False),
                target_dt=config.get("fft_target_dt"),
            )
        L = len(time_uniform)
        freq_full = np.fft.rfftfreq(L, dt_use)
        target_idx, actual_target_freq = _find_nearest_freq_bin(
            freq_full, target_freq
        )
        Y_target = np.empty(n_valid, dtype=complex)
        P_target = np.empty(n_valid, dtype=float)
        fft_batch_size = max(1, int(config.get("fft_batch_size", 32)))
        for first in range(0, n_valid, fft_batch_size):
            last = min(first + fft_batch_size, n_valid)
            work = signal_matrix[:, first:last].copy()
            work, win_scale, _ = _preprocess_probe_signal(
                work, config, time_uniform=time_uniform, copy_raw=False,
                log_details=(first == 0),
            )
            spectrum = np.fft.rfft(work, axis=0)
            coeff = spectrum[target_idx, :] * win_scale
            Y_target[first:last] = coeff
            P_target[first:last] = np.abs(coeff) / L
        if target_idx not in (0, len(freq_full) - 1):
            P_target *= 2.0
        freq_ss = np.array([actual_target_freq])
        Y_ss = Y_target[None, :]
        P1 = P_target[None, :]
        _ts(
            f"  [SD] Target-bin FFT complete: {actual_target_freq:.6e} Hz "
            f"x {n_valid} probes"
        )

        # Phase speed
        bl_x = freq_data["x"]
        u_edge_interp = np.interp(probe_x_m, bl_x, freq_data["u_edge"])
        T_edge_interp = None if T_edge_vals is None else np.interp(
            probe_x_m, bl_x, T_edge_vals
        )

        phase_data = fdb.compute_phase_speed_from_probes(
            np.array(probe_x_m), freq_ss, Y_ss, target_freq,
            u_edge=u_edge_interp, T_edge=T_edge_interp,
            phase_convention=config.get(
                "stability_phase_convention", "omega_t_minus_alpha_x"
            ),
        )
        probe_order = np.argsort(probe_x_m)
        coherence_nperseg = int(config.get(
            "stability_coherence_nperseg", 16384
        ))
        target_coherence = fdb.compute_adjacent_target_coherence(
            signal_matrix, 1.0 / dt_use, target_freq,
            nperseg=coherence_nperseg,
            noverlap=int(
                config.get("stability_coherence_noverlap", 0.5)
                * coherence_nperseg
            ),
            column_order=probe_order,
        )
        coherence_squared = target_coherence["coherence_squared"]
        min_coherence = float(config.get("stability_min_coherence", 0.5))
        phase_data["c_p_raw"] = phase_data["c_p"].copy()
        phase_data["coherence_squared"] = coherence_squared
        phase_data["coherence_threshold"] = min_coherence
        phase_data["c_p"] = np.where(
            coherence_squared >= min_coherence,
            phase_data["c_p"], np.nan,
        )
        _ts(
            f"  [SD] Coherence gate retained "
            f"{np.sum(coherence_squared >= min_coherence)}/{len(coherence_squared)} "
            f"adjacent pairs at gamma^2 >= {min_coherence:.2f}"
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

        np.savez_compressed(
            output_dir / "stability_summary.npz",
            target_frequency_hz=np.array(target_freq),
            actual_fft_frequency_hz=np.array(actual_target_freq),
            estimate_x_m=np.asarray(freq_data.get("x", [])),
            f_omega_star_hz=np.asarray(freq_data.get("f_omega_star", [])),
            f_acoustic_hz=np.asarray(freq_data.get("f_acoustic", [])),
            delta_99_m=np.asarray(freq_data.get("delta_99", [])),
            phase_x_mid_m=np.asarray(phase_data.get("x_mid", [])),
            phase_speed_m_per_s=np.asarray(phase_data.get("c_p", [])),
            phase_speed_raw_m_per_s=np.asarray(phase_data.get("c_p_raw", [])),
            phase_coherence_squared=np.asarray(
                phase_data.get("coherence_squared", [])
            ),
            phase_coherence_threshold=np.array(min_coherence),
            slow_acoustic_speed_m_per_s=np.asarray(
                phase_data.get("c_p_slow", [])
            ),
            fast_acoustic_speed_m_per_s=np.asarray(
                phase_data.get("c_p_fast", [])
            ),
            growth_x_m=np.asarray(growth_data.get("x", [])),
            spatial_growth_per_m=np.asarray(growth_data.get("alpha_i", [])),
            spatial_growth_delta99=np.asarray(
                growth_data.get("alpha_i_delta", [])
            ),
            spatial_growth_ci95_per_m=np.asarray(
                growth_data.get("alpha_i_ci95", [])
            ),
            growth_fit_r_squared=np.asarray(
                growth_data.get("r_squared", [])
            ),
        )

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

def _process_transient_analysis(config, probe_data=None):
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

        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=max_probes
            )
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        n_valid = len(probe_data)
        probe_x = [d["x"] for d in probe_data]
        _ts(f"  [TA] Loaded {n_valid} probes")

        time_uniform, signal_matrix, dt_use, reused_shared = \
            _probe_matrix_on_common_time(
                probe_data, resample=config.get("fft_resample", False),
                target_dt=config.get("fft_target_dt"),
            )
        if reused_shared:
            _ts("  [TA] Reusing contiguous binary probe matrix")

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
                    nperseg=nperseg,
                    noverlap=noverlap,
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
                    instantaneous_frequency=inst_freq,
                    frequency_band=band,
                )
                _ts(f"  [TA] Saved envelope: {out_env}")
            except Exception as exc:
                _ts(f"  [TA] Envelope failed for probe {idx}: {exc}")

        # Batch envelope stats vs x
        packet_stats_list = []
        packet_stride = max(1, int(config.get("transient_probe_stride", 1)))
        packet_indices = sorted(
            set(range(0, n_valid, packet_stride)) | {
                int(value) for value in plot_probes if int(value) < n_valid
            }
        )
        packet_x = [probe_x[ip] for ip in packet_indices]
        for ip in packet_indices:
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
                    packet_x, packet_stats_list,
                    output_path=str(output_dir / "envelope_growth_vs_x.png"),
                )
                _ts("  [TA] Saved envelope growth vs x")
            except Exception as exc:
                _ts(f"  [TA] Envelope growth plot failed: {exc}")

        stat_keys = (
            "peak_amplitude", "peak_time", "arrival_time",
            "half_width_half_max", "integrated_energy",
        )
        np.savez_compressed(
            output_dir / "packet_statistics.npz",
            probe_indices=np.asarray(packet_indices, dtype=int),
            probe_x_cm=np.asarray(packet_x, dtype=float),
            **{
                key: np.asarray([
                    np.nan if item.get(key) is None else item.get(key, np.nan)
                    for item in packet_stats_list
                ], dtype=float)
                for key in stat_keys
            },
        )
        _ts(f"  [TA] Saved packet statistics: {output_dir / 'packet_statistics.npz'}")

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


# ---------------------------------------------------------------------------
#  PHASE 3 WORKER: NONLINEAR INTERACTION DIAGNOSTICS (bispectrum)
# ---------------------------------------------------------------------------

def _process_nonlinear_diagnostics(config, probe_data=None):
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
        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=max_probes
            )
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        n_valid = len(probe_data)
        probe_x = [d["x"] for d in probe_data]
        _ts(f"  [NL] Loaded {n_valid} probes")

        time_uniform, signal_matrix, dt_use, reused_shared = \
            _probe_matrix_on_common_time(
                probe_data, resample=config.get("fft_resample", False),
                target_dt=config.get("fft_target_dt"),
            )
        if reused_shared:
            _ts("  [NL] Reusing contiguous binary probe matrix")

        fs = 1.0 / dt_use
        noverlap = int(noverlap_frac * nperseg)

        # Select independent spectral modes rather than assuming integer
        # harmonics. Explicit targets win; otherwise use the strongest local
        # peaks of a representative probe within the configured band.
        target_freqs = config.get("nonlinear_target_freqs")
        nonlinear_fmax = float(config.get("nonlinear_fmax", 50.0e6))
        if target_freqs is None:
            from scipy.signal import find_peaks
            representative = min(
                config.get("fft_contour_ref_probe", 0), n_valid - 1
            )
            reference_signal = signal_matrix[:, representative]
            reference_spectrum = np.abs(np.fft.rfft(
                reference_signal - np.mean(reference_signal)
            ))
            reference_frequency = np.fft.rfftfreq(
                len(reference_signal), dt_use
            )
            target_band = config.get(
                "nonlinear_target_band", [1.0e5, nonlinear_fmax]
            )
            valid_band = (
                (reference_frequency >= target_band[0])
                & (reference_frequency <= min(target_band[1], nonlinear_fmax))
            )
            search = reference_spectrum.copy()
            search[~valid_band] = -np.inf
            peaks, _ = find_peaks(search, distance=3)
            peaks = peaks[valid_band[peaks]]
            count = int(config.get("nonlinear_num_target_modes", 5))
            strongest = peaks[np.argsort(reference_spectrum[peaks])[::-1][:count]]
            target_freqs = np.sort(reference_frequency[strongest]).tolist()
            if not target_freqs:
                return (label, False, "No nonlinear target spectral peaks found")
        target_freqs = [float(value) for value in target_freqs]
        _ts(
            "  [NL] Independent target modes: "
            + ", ".join(f"{value:.3e} Hz" for value in target_freqs)
        )

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
                    fmax=nonlinear_fmax,
                )
                out_bic = output_dir / f"bicoherence_probe{idx:03d}.png"
                pdb.plot_bicoherence_map(
                    freq_bic, bicoh,
                    output_path=str(out_bic),
                    fmax=nonlinear_fmax,
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
            stride = max(1, int(config.get("nonlinear_probe_stride", 1)))
            sampled = set(range(0, n_valid, stride)) | set(plot_probes)
            remaining = [
                ip for ip in sorted(sampled)
                if ip < n_valid and ip not in plot_probes
            ]
            for idx in remaining:
                sig = signal_matrix[:, idx] - np.mean(signal_matrix[:, idx])
                try:
                    t_bic = fdb.compute_triad_bicoherence(
                        sig, fs, target_freqs,
                        nperseg=nperseg, noverlap=noverlap,
                    )
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
                    reference_threshold=config.get(
                        "nonlinear_reference_threshold"
                    ),
                )
                triad_labels = sorted({
                    key for item in triad_sorted for key in item
                })
                triad_values = np.full(
                    (len(triad_sorted), len(triad_labels)), np.nan, dtype=float
                )
                for row, item in enumerate(triad_sorted):
                    for col, key in enumerate(triad_labels):
                        if key in item:
                            triad_values[row, col] = item[key]
                np.savez_compressed(
                    output_dir / "triad_bicoherence.npz",
                    probe_x_cm=x_sorted,
                    triad_labels=np.asarray(triad_labels),
                    bicoherence_squared=triad_values,
                    nperseg=np.array(nperseg),
                    noverlap=np.array(noverlap),
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
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_run_manifest(out_dir, config, "running", started_at=started_at)

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
        _write_run_manifest(
            out_dir, config, "failed", failures=[str(exc)],
            started_at=started_at,
        )
        return 1

    if len(plotfile_paths) == 0:
        _ts("ERROR: No plotfiles found!  Check data_source and plot_prefix.")
        _write_run_manifest(
            out_dir, config, "failed", failures=["No plotfiles found"],
            started_at=started_at,
        )
        return 1

    _ts(f"Found {len(plotfile_paths)} plotfiles"
         f" (first: {os.path.basename(plotfile_paths[0])},"
         f" last: {os.path.basename(plotfile_paths[-1])})")
    _ts(f"Mem strategy: ONE AT A TIME (low-memory mode)")
    plotfile_fields = _required_plotfile_fields(config)
    if plotfile_fields is not None:
        _ts(f"Field selection:    {', '.join(plotfile_fields)}")

    # --- Load baseline for p' contour if needed ---
    baseline_dataset = None
    baseline_error = None
    if config.get("make_pprime_contour", False):
        baseline_name = config.get("pprime_baseline_plotfile", "")
        baseline_path = os.path.join(config["data_source"], baseline_name)
        _ts(f"Loading baseline plotfile for p' subtraction: {baseline_name}")
        try:
            baseline_ds = fdb.load_pelec_plotfile(
                baseline_path,
                field_names=[config.get("pprime_field", "pressure")],
                alias_map=config.get("field_aliases"),
                convert_to_mks=True,
            )
            baseline_dataset = baseline_ds
            _ts("Baseline field loaded.")
        except Exception as exc:
            _ts(f"[W] Failed to load baseline for p': {exc}")
            baseline_dataset = None
            baseline_error = str(exc)

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
    analysis_results = []
    if baseline_error is not None:
        pprime_results.append(("pprime_baseline", False, baseline_error))

    # Count enabled workflows for progress reporting
    enabled_workflows = sum([
        1 if config["make_contour_plots"] else 0,
        1 if config["make_line_profiles"] else 0,
        1 if config["make_streamlines"] else 0,
        1 if config["make_surface_analysis"] else 0,
        1 if config.get("make_pprime_contour", False) else 0,
    ])

    snapshot_paths = plotfile_paths if enabled_workflows else []
    if not snapshot_paths:
        _ts("No snapshot-based workflows enabled; skipping AMReX plotfile loading.")

    for i, pfile in enumerate(snapshot_paths, 1):
        label = os.path.basename(pfile)
        t_snap = time.time()

        _ts(f"┌─ Load {i:>4d}/{len(snapshot_paths)}:  {label}")
        t_load = time.time()

        try:
            dataset = fdb.load_pelec_plotfile(
                pfile,
                field_names=plotfile_fields,
                alias_map=config.get("field_aliases"),
                convert_to_mks=True,
            )
        except Exception as exc:
            fdb._log_error(f"Failed to load {label}", exc)
            _ts(f"├─ ✗ LOAD FAILED — {exc}")
            failed = (label, False, f"Snapshot load failed: {exc}")
            if config["make_contour_plots"]:
                contour_results.append(failed)
            if config.get("make_pprime_contour", False):
                pprime_results.append(failed)
            if config["make_line_profiles"]:
                line_results.append(failed)
            if config["make_streamlines"]:
                streamline_results.append(failed)
            if config["make_surface_analysis"]:
                surface_results.append(failed)
            continue

        _ts(f"├─ Loaded in {time.time() - t_load:.1f} s  "
             f"({len(dataset.get('fields', []))} fields, "
             f"grid {dataset.get('grid_shape', '?')})")

        # Derived fields are needed only for complete/derived-field loads.
        if plotfile_fields is None:
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
        if config.get("make_pprime_contour", False) and baseline_dataset is not None:
            t_pp = time.time()
            try:
                r = _process_pprime_contour((dataset, config, baseline_dataset))
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
        if i < len(snapshot_paths):
            remaining = (len(snapshot_paths) - i) * t_snap_elapsed
            _ts(f"    ⏳ Est. remaining: ~{remaining/60:.1f} min  "
                 f"({len(snapshot_paths) - i} snapshots left)", end="\n\n")

        # Explicitly drop the dataset to free memory before next iteration
        del dataset

    # The p' baseline is no longer needed. Release it before loading the much
    # larger probe record and ask Python to promptly collect yt-owned objects.
    baseline_dataset = None
    gc.collect()

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
    shared_probe_data = None
    probe_workflows_enabled = any((
        config.get("make_fft_probes", False),
        config.get("make_stability_diagnostics", False),
        config.get("make_transient_analysis", False),
        config.get("make_nonlinear_diagnostics", False),
    ))
    if probe_workflows_enabled:
        _ts("Loading shared probe dataset for enabled analysis workflows ...")
        try:
            shared_max_probes = (
                None if config.get("make_stability_diagnostics", False)
                else config.get("fft_max_probes")
            )
            shared_probe_data = _load_probe_timeseries(
                config,
                config.get("fft_var_col", 4),
                nt_skip=config.get("fft_nt_skip", 0),
                max_probes=shared_max_probes,
            )
            if not shared_probe_data:
                raise ValueError("No probe data could be loaded")
            _ts(
                f"Shared probe dataset ready: {len(shared_probe_data)} probes; "
                "subsequent workflows will not rescan the binaries"
            )
        except Exception as exc:
            analysis_results.append(("probe_data", False, str(exc)))
            _ts(f"✗ Shared probe-data load failed — {exc}")

    if config.get("make_fft_probes", False):
        _ts("Running FFT probe analysis ...")
        try:
            r = _process_fft_probes(config, shared_probe_data)
            analysis_results.append(r)
            if r[1]:
                _ts("FFT probe analysis -> OK")
            else:
                _ts(f"✗ FFT probe analysis -> FAIL — {r[2]}")
        except Exception as exc:
            analysis_results.append(("fft_probes", False, str(exc)))
            _ts(f"✗ FFT probe analysis -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 4. Stability diagnostics (runs once, not per-snapshot)
    # ------------------------------------------------------------------
    if config.get("make_stability_diagnostics", False):
        _ts("Running stability diagnostics ...")
        try:
            r = _process_stability_diagnostics(config, shared_probe_data)
            analysis_results.append(r)
            if r[1]:
                _ts("Stability diagnostics -> OK")
            else:
                _ts(f"✗ Stability diagnostics -> FAIL — {r[2]}")
        except Exception as exc:
            analysis_results.append(("stability", False, str(exc)))
            _ts(f"✗ Stability diagnostics -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 5. Transient analysis (STFT / Hilbert envelope)
    # ------------------------------------------------------------------
    if config.get("make_transient_analysis", False):
        _ts("Running transient analysis ...")
        try:
            r = _process_transient_analysis(config, shared_probe_data)
            analysis_results.append(r)
            if r[1]:
                _ts("Transient analysis -> OK")
            else:
                _ts(f"✗ Transient analysis -> FAIL — {r[2]}")
        except Exception as exc:
            analysis_results.append(("transient", False, str(exc)))
            _ts(f"✗ Transient analysis -> FAIL — {exc}")

    # ------------------------------------------------------------------
    # 6. Nonlinear interaction diagnostics (bispectrum)
    # ------------------------------------------------------------------
    if config.get("make_nonlinear_diagnostics", False):
        _ts("Running nonlinear diagnostics ...")
        try:
            r = _process_nonlinear_diagnostics(config, shared_probe_data)
            analysis_results.append(r)
            if r[1]:
                _ts("Nonlinear diagnostics -> OK")
            else:
                _ts(f"✗ Nonlinear diagnostics -> FAIL — {r[2]}")
        except Exception as exc:
            analysis_results.append(("nonlinear", False, str(exc)))
            _ts(f"✗ Nonlinear diagnostics -> FAIL — {exc}")

    shared_probe_data = None
    gc.collect()

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
    if analysis_results:
        _print_results(analysis_results, "Probe/stability analysis")

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
    result_groups = [
        contour_results, pprime_results, line_results,
        streamline_results, surface_results, analysis_results,
    ]
    failed_results = [r for group in result_groups for r in group if not r[1]]
    completion_title = "Complete" if not failed_results else "Completed with failures"
    _hr(completion_title, char="=", width=75)
    _ts(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _ts(f"Total wall time: {total_elapsed/60:.1f} min  ({total_elapsed:.1f} s)")
    _ts(f"Results in: {out_dir.resolve()}")
    _ts(f"Snapshots processed: {len(plotfile_paths)}")
    if failed_results:
        _ts(f"ERROR: {len(failed_results)} workflow result(s) failed; returning exit status 1")
        for name, _, reason in failed_results:
            _ts(f"  - {name}: {reason}")
    _write_run_manifest(
        out_dir, config,
        "failed" if failed_results else "completed",
        plotfiles=plotfile_paths,
        failures=[{"workflow": r[0], "reason": r[2]} for r in failed_results],
        started_at=started_at,
    )
    _hr()
    return 1 if failed_results else 0


def _parse_command_line(argv=None):
    """Parse the small command-line surface while preserving the default run."""
    parser = argparse.ArgumentParser(
        description="PeleC CFD post-processing and analysis validation"
    )
    parser.add_argument(
        "--validation-case",
        action="store_true",
        help="run the fast synthetic analysis-validation case instead of production",
    )
    parser.add_argument(
        "--validation-output",
        default="validation_outputs",
        help="output directory used with --validation-case",
    )
    parser.add_argument(
        "--config",
        help="partial JSON configuration overlay (unknown keys are rejected)",
    )
    parser.add_argument("--output-dir", help="override the configured output directory")
    parser.add_argument("--snapshot-start", type=int)
    parser.add_argument("--snapshot-end", type=int)
    parser.add_argument(
        "--validate-config", action="store_true",
        help="validate the effective configuration and exit without loading data",
    )
    parser.add_argument(
        "--write-effective-config",
        help="write the merged, validated configuration JSON before running",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = _parse_command_line()
    if cli_args.validation_case:
        sys.exit(run_validation_case(cli_args.validation_output))
    try:
        effective_config = pp_config.build_config(
            CONFIG,
            json_path=cli_args.config,
            command_line_overrides={
                "output_dir": cli_args.output_dir,
                "snapshot_start": cli_args.snapshot_start,
                "snapshot_end": cli_args.snapshot_end,
            },
        )
        if cli_args.write_effective_config:
            pp_config.write_config(effective_config, cli_args.write_effective_config)
        if cli_args.validate_config:
            print("Configuration is valid.")
            sys.exit(0)
        sys.exit(main(effective_config))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
