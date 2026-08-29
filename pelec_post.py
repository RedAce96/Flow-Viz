#!/usr/bin/env python3
# =============================================================================
#  PeleC Post-Processing: Main Execution Script
# =============================================================================
#  User-friendly entry point for running the PeleC post-processing pipeline.
#
#  For interns / new GRAs:
#    - Keep scientific case settings in a committed configs/*.json profile.
#    - Keep server paths in an ignored configs/local/*.json overlay.
#    - Run both with repeated --config arguments; later files take precedence.
#    - You should not edit the functions in the *_database.py files.
# =============================================================================

import argparse
import csv
import glob
import gc
import hashlib
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
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
try:
    from pp_probe_store import HDF5ProbeStore
    HDF5_PROBE_STORE_AVAILABLE = True
except ImportError:
    class HDF5ProbeStore:
        pass

    HDF5_PROBE_STORE_AVAILABLE = False


# ---------------------------------------------------------------------------
#  PROBE VARIABLE METADATA
# ---------------------------------------------------------------------------

# ASCII .dat column convention: 0=time, 1=rho, 2=u, 3=p, 4=T, 5=x_sample,
# 6=y_sample, 7=level.  The selectable FFT variables are columns 1-4.
_PROBE_VARIABLES = {
    1: {
        "field": "rho",
        "slug": "density",
        "name": "Density",
        "unit_cgs": "g/cm^3",
        "unit_si": "kg/m^3",
        "symbol": r"$\rho$",
        "ms_symbol": r"$\rho-\overline{\rho}$",
        "modal_weight": "rho",
    },
    2: {
        "field": "u",
        "slug": "velocity",
        "name": "Streamwise velocity",
        "unit_cgs": "cm/s",
        "unit_si": "m/s",
        "symbol": r"$u$",
        "ms_symbol": r"$u-\overline{u}$",
        "modal_weight": "u",
    },
    3: {
        "field": "p",
        "slug": "pressure",
        "name": "Pressure",
        "unit_cgs": "dyne/cm^2",
        "unit_si": "Pa",
        "symbol": r"$p$",
        "ms_symbol": r"$p-\overline{p}$",
        "modal_weight": "pressure",
    },
    4: {
        "field": "T",
        "slug": "temperature",
        "name": "Temperature",
        "unit_cgs": "K",
        "unit_si": "K",
        "symbol": r"$T$",
        "ms_symbol": r"$T-\overline{T}$",
        "modal_weight": "temperature",
    },
}


def _fft_variable_meta(config_or_col):
    """Return metadata for the configured FFT probe variable.

    Parameters
    ----------
    config_or_col : dict or int
        Either the full CONFIG dict (reads ``fft_var_col``) or an explicit
        integer column index.

    Returns
    -------
    dict
        Metadata for the selected variable, including field key, display
        name, units, and modal-analysis weight name.
    """
    if isinstance(config_or_col, dict):
        var_col = int(config_or_col.get("fft_var_col", 3))
    else:
        var_col = int(config_or_col)
    if var_col not in _PROBE_VARIABLES:
        raise ValueError(
            f"fft_var_col={var_col} is not supported; "
            f"choose one of {sorted(_PROBE_VARIABLES)} "
            "(1=density, 2=velocity, 3=pressure, 4=temperature)"
        )
    return _PROBE_VARIABLES[var_col]


def _fft_var_slug(config_or_col):
    """Return the short filename-safe slug for the selected variable."""
    return _fft_variable_meta(config_or_col)["slug"]


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
    """Return an optional legacy title; presentation plots default to none."""
    if not config.get("plot_show_titles", False):
        return None
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


def _configure_plot_style(config):
    """Apply configurable minimum presentation font sizes."""
    pdb.configure_plot_style({
        "font.size": float(config.get("plot_font_size", 14)),
        "axes.labelsize": float(config.get("plot_axes_label_size", 16)),
        "xtick.labelsize": float(config.get("plot_tick_label_size", 14)),
        "ytick.labelsize": float(config.get("plot_tick_label_size", 14)),
        "legend.fontsize": float(config.get("plot_legend_size", 14)),
    })


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
        # The loader handles these explicitly: ``magvort`` is preserved for
        # magnitude, while signed curl is evaluated on native AMR patches.
        "vorticity", "vorticity_magnitude",
    }
    requested = set()
    if config.get("make_contour_plots"):
        requested.update(config.get("contour_fields", []))
    if config.get("make_line_profiles"):
        requested.update(config.get("line_fields", []))
        requested.add("x_velocity")  # delta_99 coordinate
        if config.get("line_reference_overlay") and config.get(
                "line_reference_model") == "compressible_similarity":
            requested.update(("density", "temperature"))
    if config.get("make_pprime_contour"):
        requested.add(config.get("pprime_field", "pressure"))

    if not requested and config.get("make_force_analysis", False):
        # The certified force path reads native AMR wall grids directly.
        # An empty field list keeps this orchestration dataset metadata-only.
        return []
    if not requested or not requested.issubset(direct_fields):
        return None
    return sorted(requested)


def _config_needs_native_vorticity(config):
    """Whether an enabled snapshot workflow requests signed vorticity."""
    requested = set()
    if config.get("make_contour_plots"):
        requested.update(config.get("contour_fields", []))
    if config.get("make_line_profiles"):
        requested.update(config.get("line_fields", []))
    if config.get("make_streamlines"):
        requested.add(config.get("streamline_color_key"))
        requested.add(config.get("streamline_mask_key"))
    return "vorticity" in requested


def _json_safe(value):
    """Convert configuration/provenance values to JSON-safe objects."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
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


def _write_analysis_evidence_report(
        config, analysis_results, force_report=None):
    """Assemble current-run measurement evidence without making LST claims."""
    output_root = Path(config["output_dir"])
    evidence_dir = output_root / "AnalysisEvidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    successful = {
        str(label): bool(status)
        for label, status, _ in analysis_results
    }
    sources = {}

    def read_json(label, relative_path):
        path = output_root / relative_path
        if not successful.get(label, False) or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        sources[label] = str(path.relative_to(output_root))
        return value

    var_slug = _fft_var_slug(config)
    wave_report = read_json(
        "stability", f"StabilityDiagnostics/wave_analysis_report_{var_slug}.json"
    )
    packet_report = read_json(
        "transient", f"TransientAnalysis/packet_propagation_{var_slug}.json"
    )
    nonlinear_report = read_json(
        "nonlinear", f"NonlinearDiagnostics/nonlinear_significance_{var_slug}.json"
    )

    modal_summary = None
    modal_path = (
        output_root / "FFTProbes" / "ModalAnalysis"
        / f"modal_results_{_fft_var_slug(config)}.npz"
    )
    if successful.get("fft_probes", False) and modal_path.is_file():
        with np.load(modal_path, allow_pickle=False) as archive:
            modal_summary = {
                "pod_subspace_cosines": archive[
                    "sensitivity_pod_subspace_min_cosine_to_first_window"
                ],
                "dmd_dominant_frequency_hz": archive[
                    "sensitivity_dmd_dominant_frequency_hz"
                ],
                "dmd_condition_number": archive[
                    "sensitivity_dmd_retained_condition_number"
                ],
                "norm_definition": str(
                    archive["modal_norm_definition"].item()
                ),
                "norm_scope": str(archive["modal_norm_scope"].item()),
            }
        sources["modal"] = str(modal_path.relative_to(output_root))

    report = fdb.classify_measured_dynamics(
        wave_report=wave_report,
        packet_report=packet_report,
        nonlinear_report=nonlinear_report,
        modal_summary=modal_summary,
        force_report=force_report,
    )
    report.update({
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "code_provenance": _git_provenance(),
        "source_products": sources,
        "enabled_workflows": {
            "coherent_wave": bool(config.get(
                "make_stability_diagnostics", False
            )),
            "transient_packet": bool(config.get(
                "make_transient_analysis", False
            )),
            "quadratic_nonlinearity": bool(config.get(
                "make_nonlinear_diagnostics", False
            )),
            "modal": bool(
                config.get("make_fft_probes", False)
                and config.get("make_modal_analysis", False)
            ),
            "aerodynamic_loads": bool(config.get(
                "make_force_analysis", False
            )),
        },
    })
    report_path = evidence_dir / "analysis_evidence.json"
    temporary = evidence_dir / ".analysis_evidence.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, report_path)

    matrix_path = evidence_dir / "analysis_evidence_matrix.csv"
    temporary_matrix = evidence_dir / ".analysis_evidence_matrix.csv.tmp"
    with temporary_matrix.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "evidence_domain", "status", "meaning",
            "LST_or_PSE_attribution",
        ])
        for domain, item in report["evidence"].items():
            writer.writerow([
                domain,
                item.get("status", "not_available"),
                item.get("meaning", ""),
                "not performed; outside project scope",
            ])
    os.replace(temporary_matrix, matrix_path)
    return report

# ---------------------------------------------------------------------------
#  USER CONFIGURATION
# ---------------------------------------------------------------------------
#  Edit the values below to control what gets processed.
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Data source ---
    # Portable fallbacks only. Real data/output locations belong in a
    # configs/local/*.json server overlay.
    "data_source": "../data",
    "plot_prefix": "plt",
    "output_dir": "post_processing_output",

    # --- Snapshot range ---
    # Set to None to process all discovered plotfiles.
    "snapshot_start": None,
    "snapshot_end": None,
    "snapshot_step": 1,

    # --- Field aliases ---
    # Map solver raw names -> canonical names.  If omitted, default PeleC
    # aliases are used.  Only add / override if your case uses non-standard
    # naming.
    "field_aliases": None,

    # --- Workflow toggles ---
    # Probe-only demonstration: avoid loading every selected AMReX snapshot.
    "make_contour_plots": False,
    "make_line_profiles": False,   # eta for similarity plots; otherwise y/delta_99
    "make_streamlines": False,
    "make_surface_analysis": False,
    "make_group_plots": False,

    "make_probe_plots": True,
    "make_fft_probes": True,
    "make_pprime_contour": False,       # symmetric perturbation contours
    # Legacy workflow name; the enabled result is a measurement-first,
    # coherence-gated wave analysis and does not perform LST/PSE.
    "make_stability_diagnostics": True,


    # --- Contour plot settings ---
    # List of canonical field names to plot.  Any field present in the
    # dataset (including derived fields) can be used.
    "contour_fields": ["temperature", "pressure", "vorticity"],
    "contour_cmap": "turbo",           # Perceptually uniform scalar-field map
    "contour_norm": "linear",               # Color scaling: "linear", "log", "symlog", or a matplotlib Normalize object
    # "auto" shortens the colorbar as the displayed streamwise span grows.
    # A numeric value in (0, 1] overrides the automatic calculation.
    "contour_colorbar_shrink": "auto",
    "contour_colorbar_reference_span": 0.25,  # [m], gives the maximum shrink
    # Contour text scales gently below the global presentation sizes for wide
    # x-ranges. A 0.4 m view is about 84% of the 0.2 m reference typography.
    "contour_font_scale": "auto",
    "contour_font_reference_span": 0.2,
    "contour_colorbar_font_scale": 0.85,
    # Field-specific choices override the global fallbacks above. Signed
    # vorticity needs a zero-centred diverging map; magnitude is non-negative
    # and spans several orders of magnitude.
    "contour_cmaps": {
        "temperature": "magma",
        "vorticity": "RdBu_r",
        "pressure": "turbo",
    },
    "contour_norms": {
        "vorticity": "linear",
        "temperature": "log",
        "pressure": "linear",
    },
    "contour_vlims": {
        "vorticity": [-1.0e5, 1.0e5],
        "temperature": [0, 4.0e3],
        "pressure": [0, 5.0e3],
    },
    "contour_xlim": None,
    "contour_ylim": None,
    # Time shown in figures. Flow-through time is t_FT = L_x / U_inf, where
    # L_x is the full AMReX domain length (independent of contour x-limits).
    "plot_time_mode": "physical",       # "flow_through" | "physical" | "reference"
    "plot_flow_through_u_inf": 1726.0,      # U_inf [m/s]
    "plot_time_reference": None,
    "plot_time_origin": 0.0,
    # Presentation defaults: no external title; time is a note inside the
    # axes. Font sizes are intended to remain readable on projected slides.
    "plot_show_titles": False,
    "plot_show_time_annotation": True,
    "plot_time_annotation_location": "upper left",
    "plot_font_size": 14,
    "plot_axes_label_size": 16,
    "plot_tick_label_size": 14,
    "plot_legend_size": 14,

    # --- Line profile settings ---
    "line_x_stations": [0.05, 0.3],                  # x-locations to extract profiles [m]
    "line_fields": ["x_velocity", "temperature", "density"],
    "line_ylim": [0, 0.005],                              # [ymin, ymax] or None for full domain height
    "line_normalize_by_delta99": True,                    # plot y/delta_99 instead of y [m]
    "line_normalized_coordinate_limits": [0.0, 2.0],      # focus on BL + near-edge region
    "line_similarity_eta_limits": [0.0, 10.0],            # eta range for compressible-similarity comparisons

    # --- Laminar flat-plate reference overlay (for line profiles) ---
    # The default is an independently solved, compressible similarity profile.
    # It is a steady base-flow reference, not a model of the laser disturbance.
    "line_reference_overlay": True,
    "line_reference_model": "compressible_similarity",
    "line_reference_fields": ["x_velocity", "temperature", "density"],
    "line_reference": {
        "u_inf": 1726.0,       # [m/s]
        "T_inf": 125.0,        # [K]
        # rho_inf = p_inf / (R T_inf) = 760 / (287.05 * 125) [kg/m^3]
        "rho_inf": 0.021180978923532486,
        "T_wall": 293.0,       # [K]; None selects an adiabatic wall
        "leading_edge_x": 0.0, # [m]
        "zero_pressure_gradient": True,
        "gamma": 1.4,
        "R": 287.05,           # [J/(kg K)]
        "Cp": 1004.0,          # [J/(kg K)]
        # Match FP-Extended-Domain/PeleC constant transport by default.
        "transport_model": "constant",  # "constant" or "sutherland"
        "mu": 8.65e-6,         # [Pa s], required for constant transport
        "k": 0.012415,         # [W/(m K)], required for constant transport
        "Pr": 0.71,            # used with the Sutherland option
    },

    # Kept only to permit an explicit legacy comparison if requested.
    "line_blasius_overlay": False,
    "line_blasius_fields": ["x_velocity", "temperature"],
    "line_blasius": {
        "u_inf": 1726.0,        # freestream velocity [m/s]
        "T_inf": 125.0,         # freestream temperature [K]
        "rho_inf": 0.021180978923532486,  # freestream density [kg/m^3]
        "T_wall": 293,         # wall temp [K]; None = adiabatic
        "recovery_factor": None,# None = sqrt(Pr_eff) (laminar)
        "const_transport": False,  # True = constant mu and k; False = Sutherland and mu*Cp/Pr
        "mu": 8.65e-6,         # constant viscosity [Pa.s]; used when const_transport=True
        "k": 0.012415,         # constant thermal cond. [W/(m.K)]; used to recompute Pr_eff
    },
    # Deprecated: the prior post-hoc transformed-coordinate overlay has been
    # removed because it is not an independent compressible reference.
    "line_blasius_compressible_overlay": False,
    "line_compare_overlay": False,
    
    "line_compare_plotfile": None,
    "line_compare_label": "Refined-ED",
    "line_compare_color": "C6",
    "line_compare_linestyle": "--",

    # --- Streamline settings ---
    "streamline_color_key": "velocity_magnitude",  # Color streamlines by this field
    "streamline_mask_key": None,                   # Mask streamlines by this field
    "streamline_mask_threshold": None,

    # --- Surface analysis settings ---
    "surface_velocity_threshold": 10.0,
    "surface_max_x": 0.4,
    "surface_geometry_method": "auto",   # 'auto', 'vfrac', 'ib_markers', 'velocity'
    "geometry_type": "flat_plate",              # 'auto', 'flat_plate' (analytical), 'wedge', 'custom'
    "plate_leading_edge": 0.0,            # [m] for geometry_type='flat_plate'
    "surface_rho_inf": 0.021180978923532486,
    "surface_u_inf": 1726.0,
    "surface_T_inf": 125.0,
    "surface_mu": 8.65e-6,                  # constant viscosity [Pa.s]; None -> Sutherland
    "surface_k": 0.012415,                   # constant thermal cond. [W/(m.K)]; None -> mu*Cp/Pr
    "surface_wall_temperature": 293,    # constant wall temp [K]; None -> use cell-adjacent T
    "surface_Pr": 0.71,                  # Prandtl number (only if surface_k is None)
    "surface_Cp": 1004.0,                # specific heat [J/(kg.K)]

    # --- Force analysis settings ---
    # Keep certified force extraction off in this probe-focused demonstration
    # so the run does not traverse the selected plotfile series. The evidence
    # matrix will correctly mark aerodynamic-load evidence as not available;
    # use the separately certified priorities 1--5 setup for force products.
    "make_force_analysis": False,
    "make_spacetime_plots": False,
    "make_force_animation": False,       # synchronized contour + surface-value animation
    "animation_field": "temperature",     # visual contour in the synchronized animation
    "animation_surface_key": "C_p",       # surface values advanced at the same time
    "animation_fps": 8,
    "reference_area": None,              # auto = plate length * 1 m span

    # Certified one-sided flat-plate force analysis.  The legacy ``surface_*``
    # keys above remain available for general surface visualisation, but force
    # integration uses this explicit physical contract.
    "force": {
        "geometry": {
            "type": "flat_plate_one_sided",
            "x_range_m": [0.0, 0.4],
            "wall_y_m": 0.0,
        },
        "reference": {
            "rho_inf": 0.021180978923532486,
            "u_inf": 1726.0,
            "p_inf": 760.0,
            "T_inf": 125.0,
            "R": 287.05,
            "chord_m": 0.4,
            "span_m": 1.0,
            "moment_origin_m": [0.1, 0.0],
        },
        "transport": {
            "model": "constant",
            "mu_pa_s": 8.65e-6,
            "k_w_m_k": 0.012415,
            "wall_temperature_k": 293.0,
        },
        "wall_fit": {
            "pressure_order": 2,
            "velocity_order": 2,
            "fluid_points": 4,
            "viscosity_relative_tolerance": 1.0e-4,
        },
        "baseline": {
            "mode": "static",  # "static", "paired", or "none"
            "plotfile": "pltFlatPlateFlow210729",
            "data_source": None,  # None -> top-level data_source
            "paired_data_source": None,
            "paired_plot_prefix": None,
            "time_tolerance_s": 1.0e-12,
            "drift_data_source": None,
            "drift_plot_prefix": "pltFlatPlatePost",
            "drift_sample_count": 10,
        },
        "boundary_layer": {
            "enabled": True,
            "stations_m": [0.05, 0.10, 0.20, 0.30],
            "maximum_height_m": 0.01,
        },
        "quality": {
            "minimum_coverage": 0.995,
            "maximum_gap_widths": 2.0,
            "minimum_forcing_periods": 10.0,
            "minimum_welch_segments": 8,
        },
        "laser": {
            "start_time_s": 0.000,
            "frequency_hz": 10.0e6,
            "pulse_fwhm_s": 10.0e-9,
            "energy_per_pulse_j": 1.0e-3,
            "duration_s": 3.0e-4,
            "impulse_interval_s": None,
        },
        "history": {
            "path": None,
            # Probe/force linkage is opt-in because production probe files can
            # be very large. Coordinates in PeleC probe headers are converted
            # from centimetres to metres in the saved linkage product.
            "probe_bin_files": [],
            "probe_pressure_var_col": 3,
            "probe_max_probes": 2000,
            "spectral_nperseg": 16384,
            "spectral_noverlap": 0.5,
        },
        "validation": {
            "fit_sensitivity": True,
            "integration_start_locations_m": [0.0, 0.001, 0.005, 0.01],
            "grid_comparison_plotfile": None,
            "control_volume": {
                "enabled": False,
                "plotfile": None,
                "x_range_m": [0.05, 0.30],
                "top_y_m": 0.01,
                "maximum_level": 1,
            },
        },
    },

    # --- Laser annotation (for time-series / animation) ---
    "laser_start_time": 0.0,
    # Individual probe station plots show the first 2 microseconds after
    # laser turn-on; the all-probe overview remains full-record.
    "probe_station_time_window_s": 0.2e-5,

    # --- Multiprocessing ---
    "use_multiprocessing": True,
    "num_processes": 2,

    # --- Probe processing settings ---
    "probe_bin_files": [],
    # Optional self-contained archive produced by compact_probes.py. When
    # configured, this takes precedence over .pbin and ASCII sources.
    "probe_compact_file": None,
    "probe_output_dir": "probes",
    "probe_max": 50,
    "probe_fields": ["rho", "u", "p", "T"],
    "probe_convert_to_mks": False,

    # --- FFT probe analysis settings ---
    "fft_use_binary": True,        # read probes directly from binary for FFT/stability
    "probe_dir": "probes",
    "probe_prefix": "kernel-probe",
    # Probe column for FFT/stability/transient/modal/nonlinear analysis:
    # 1=density, 2=streamwise velocity, 3=pressure, 4=temperature.
    # Output filenames include a short slug for the selected variable.
    "fft_var_col": 4,
    "fft_nt_skip": 0,
    "fft_max_probes": 50,
    "probe_dedup_tol": 1e-12,
    # Restart segments can repeat their final/initial sample.  If a repeated
    # step was recomputed after restart, prefer the newer segment and retain
    # an audit summary; conflicting repeats within one segment remain errors.
    "probe_overlap_policy": "latest_segment",  # "latest_segment" | "error"
    # AMR can move the cell centre sampled for a fixed requested location.
    # "strict" rejects such changes, "nominal" retains the complete record
    # and its compact mapping history while labelling probes by their fixed
    # requested positions, and "longest_epoch" keeps only the longest
    # contiguous interval with one stationary sampling map.
    "probe_coordinate_policy": "nominal",
    # The stopped segment contains one shortened final step.  Resampling onto
    # the native 2 ns cadence prevents that endpoint from biasing the FFT.
    "fft_resample": True,
    "fft_target_dt": None,
    "fft_mean_subtraction": "mean",   # "mean" | "linear" | "none" — remove DC before FFT
    "fft_window": "hann",            # "hann" | "hamming" | "blackman" | "rect" | "none"
    "fft_window_compensation": True,  # Scale FFT amplitudes to preserve magnitude
    "fft_batch_size": 32,             # Probe columns per vectorized FFT batch
    "fft_plot_last_probe": False,
    # Five stations spanning the 50-probe source line.
    "fft_plot_probe_indices": [0, 10, 20, 30, 40, 49],
    "fft_plot_contour": True,
    # Bound the in-memory raster even when the full spectrum is disk-backed.
    "fft_contour_max_frequency_points": 4096,
    "fft_contour_max_probe_points": 1000,
    "fft_contour_normalize": False,  # True can create bright artifacts where the reference probe has a node
    "fft_contour_ref_probe": 5,
    "fft_contour_scale": "linear",  # "linear" gives 0-to-max amplitude; "db" gives the legacy dB plot
    "fft_contour_vmax": 500,  # None -> use the maximum plotted amplitude

    # --- Single-pulse source spectrum and source-to-response transfer ---
    # The source is the domain-integrated temporal deposition model. It is
    # analytically exact for fixed-volume deposition and the uniform-density
    # calibration reference for fixed-specific-energy deposition.
    "make_source_response_analysis": False,
    "source_response": {
        "model": "single_gaussian",
        "spatial_shape_label": "gaussian_kernel",
        "energy_per_pulse": 100.0e2,       # [erg/cm] in this 2-D case
        "energy_unit": "erg/cm",
        "pulse_fwhm_s": 10.0e-9,
        "pulse_period_s": 1.0e-7,
        "start_time_s": 0.0,
        "cutoff_sigma": 4.0,
        # A single impulse has nonzero integral, so preserve the physical
        # source history for transfer ratios. Probe mean subtraction removes
        # the ambient-state baseline and remains controlled by fft_* above.
        "transfer_source_mean_subtraction": "none",
        # A single pulse is analyzed as a finite record. A rectangular window
        # avoids weighting the source and delayed response by different Hann
        # coefficients; the response baseline comes only from quiescent data.
        "transfer_window": "none",
        "response_baseline_end_time_s": None,
        "minimum_baseline_samples": 8,
        # Direct ratios are masked when the processed source is this far below
        # its peak, preventing division by its negligible high-frequency tail.
        "minimum_relative_source_amplitude": 1.0e-3,
        "plot_fmax_hz": 200.0e6,
        "plot_probe_indices": [0, 10, 20, 30, 40, 49],
    },
    # --- Spatial FFT along the fixed y=2.5 cm probe aperture ---
    # The transform is descriptive: it measures spatial energy and a
    # response/source ratio, but does not identify an LST eigenmode.
    "make_spatial_fft": False,
    "spatial_fft": {
        "model": "wang_kernel",
        "center_x_cm": 2.5,
        "center_y_cm": 2.5,
        "radius_cm": 0.0384843,
        "wang_length_cm": 0.184,
        "wang_aspect_ratio": 3.23,
        "wang_asymmetry": 1.24,
        # A quiescent pre-event interval defines q0(x); spectra use q-q0.
        # Keep spatial DC by default because net heating is physical.
        "baseline_mode": "pre_event_mean",  # "pre_event_mean" | "none"
        "baseline_end_time_s": None,  # None -> start of truncated source pulse
        "minimum_baseline_samples": 8,
        "mean_subtraction": "none",
        "window": "hann",
        "window_compensation": True,
        "zero_padding": 0,
        "minimum_relative_source_amplitude": 1.0e-3,
        "snapshot_indices": None,
        "snapshot_times_s": None,
        # None analyzes every available resampled probe time. Set an integer
        # only when a reduced, evenly spaced snapshot set is desired.
        "max_snapshots": None,
        "wavenumber_colorbar_vmax": 50000.0,
        "make_k_omega": True,
        "k_omega_temporal_window": "none",
        "k_omega_spatial_window": "hann",
        "k_omega_temporal_mean_subtraction": "none",
        "k_omega_spatial_mean_subtraction": "none",
        "k_omega_temporal_zero_padding": 0,
        "k_omega_spatial_zero_padding": 0,
        "k_omega_frequency_max_hz": 200.0e6,
        "k_omega_db_floor": -80.0,
        "k_omega_trustworthy_nyquist_fraction": 0.8,
    },
    "make_spatial_case_comparison": False,
    "spatial_case_comparison": {
        "baseline_label": "Gaus_JW",
        "baseline_archive": None,
        "comparison_label": "Asym_JW",
        "comparison_archive": None,
    },
    # --- Direct Gaussian-versus-Asym spectral comparison ---
    # Gaus_JW is deliberately the reference in both the figure titles and
    # Asym_JW/Gaus_JW ratio; station IDs are checked against physical x values.
    # FFT filenames include the selected variable slug (e.g. _pressure below).
    "make_case_spectrum_comparison": False,
    "case_spectrum_comparison": {
        "baseline_label": "Gaus_JW",
        "baseline_source_response_archive": None,
        "baseline_spectral_summary_archive": None,
        "comparison_label": "Asym_JW",
        "comparison_source_response_archive": None,
        "comparison_spectral_summary_archive": None,
        "probe_indices": [0, 10, 20, 30, 40, 49],
        "plot_fmax_hz": 200.0e6,
    },

    # --- FFT advanced analysis ---
    "fft_harmonic_freq": 10.0e6,
    "fft_num_harmonics": 5,
    "fft_plot_harmonics": True,
    "fft_slope_fmin": 1.0e6,
    "fft_slope_fmax": 2.0e8,
    "fft_plot_spectral_slope": True,
    "fft_growth_freqs": [5.0e6, 10.0e6, 30.0e6, 100.0e6, 200.0e6],
    # Stationary full-record growth curves are superseded here by the
    # coherence-gated local wavenumber fit and convecting-packet energy fit.
    "fft_plot_growth_curves": True,
    # Welch coherence is evaluated on adjacent probe pairs near the selected
    # plotting stations. None builds [(i, i+1), ...] automatically.
    "make_coherence_analysis": True,
    "coherence_probe_pairs": None,
    "coherence_nperseg": 4096,
    "coherence_noverlap": 0.5,
    "coherence_fmax": 50.0e6,
    # Modal screening on selected probe stations. These decompositions are
    # explicitly descriptive and are not substituted for LST eigenmodes.
    "make_modal_analysis": False,
    "modal_probe_indices": None,  # None -> full ordered line at modal_probe_stride
    "modal_probe_stride": 4,
    "modal_n_modes": 4,
    "modal_spod_nperseg": 2048,
    "modal_spod_noverlap": 0.5,
    "modal_spod_frequency_stride": 2,
    "modal_spod_fmax": 50.0e6,
    "modal_use_compressible_energy_weights": True,
    "modal_base_rho_kg_m3": 0.021180978923532486,
    "modal_base_temperature_k": 125.0,
    "modal_gamma": 1.4,
    "modal_gas_constant_j_kg_k": 287.05,
    "modal_sensitivity_dmd_ranks": [2, 4, 8],
    "modal_sensitivity_windows": [[0.0, 0.5], [0.5, 1.0]],

    # --- Pressure perturbation (p') contour ---
    "pprime_baseline_plotfile": "pltFlatPlateFlow210000",
    "base_flow_definition": (
        "Final pre-laser flat-plate flow plotfile pltFlatPlateFlow210729; "
        "instantaneous converged base used only for non-LST GIP screening"
    ),
    "pprime_field": "pressure",
    "pprime_cmap": "RdBu_r",
    "pprime_vlims": [-1, 1],

    # --- Stability diagnostics (2nd Mack mode) ---
    # Existing pre-event/base-flow plotfile; use an independently verified
    # steady base here when a more appropriate baseline becomes available.
    "stability_baseline_plotfile": "pltFlatPlateFlow210729",
    # Base-flow-only mode: produce GIP diagnostics and return before loading
    # probes or evaluating phase speed, growth, or N_probe.
    "stability_gip_only": True,
    "stability_target_freq": None,
    "stability_freq_band": [100e3, 1.0e6],
    "stability_growth_window_size": None,
    "stability_num_gpi_profiles": 5,
    "stability_gip_derivative_window": 11,
    "stability_gip_derivative_order": 3,
    "stability_gip_exclusion_fraction": 0.02,
    "stability_gip_residual_multiplier": 3.0,
    "stability_coherence_nperseg": 16384,
    "stability_coherence_noverlap": 0.5,
    "stability_min_coherence": 0.5,
    # Downstream wave convention: exp(i*(omega*t - alpha*x)).
    "stability_phase_convention": "omega_t_minus_alpha_x",
    # Frequency-resolved, measurement-first complex-wavenumber analysis.
    # This estimates the dominant coherent wave in each local (x, f) window;
    # it does not assign definitive LST F/S eigenmode labels.
    "stability_make_wavenumber_analysis": False,
    "stability_analysis_time_window": None,  # [start_s, end_s] or None
    "stability_wavenumber_nperseg": 16384,
    "stability_wavenumber_noverlap": 0.5,
    "stability_frequency_stride": 1,
    "stability_spatial_window_size": 101,
    "stability_spatial_step": 20,
    "stability_wavenumber_min_coherence": 0.8,
    "stability_wavenumber_min_coherent_fraction": 0.8,
    "stability_wavenumber_min_phase_r2": 0.8,
    "stability_wavenumber_min_amplitude_r2": 0.5,
    "stability_wavenumber_min_relative_power_db": -40.0,
    "stability_wavenumber_max_edge_phase_rad": 2.827433388230814,
    # Non-LST logarithmic amplification measured from the accepted probe data.
    "stability_make_probe_amplification": False,
    "stability_amplification_plot_frequencies": [],
    "stability_amplification_max_consistency_error": 1.0,
    "stability_amplification_min_contiguous_centres": 3,
    "stability_phase_speed_bounds": [500.0, 2500.0],
    # Relative distance to the U_e +/- a_e reference, measured as a fraction
    # of their separation, for cautious "fast-like"/"slow-like" candidates.
    "stability_acoustic_reference_tolerance": 0.25,

    # --- Phase 1: Disturbance signal reconstruction ---
    # Reconstruct time-domain disturbance from FFT harmonics, a frequency
    # band, or the top-N amplitude frequency peaks.
    # Requires make_fft_probes=True.
    # The older harmonic/top-frequency reconstruction is not part of this
    # focused recommendations 6--10 run.
    "make_disturbance_reconstruction": False,
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
    "make_transient_analysis": False,
    "transient_band": [1.0e5, 1.0e6],       # includes heuristic/observed low-MHz modes
    "transient_stft_nperseg": 16384,
    "transient_stft_noverlap": 0.75,
    # Nine probes spanning the complete line; these match the FFT figures.
    "transient_plot_probe_indices": [
        0, 249, 499, 749, 999, 1249, 1499, 1749, 1999,
    ],
    "transient_make_fft_stft_comparison": False,
    "transient_spectrogram_scale": "linear",
    # Linear PSD is nonnegative, matching the zero-based FFT-vs-x contour
    # convention. None selects the maximum of each plotted spectrogram.
    "transient_spectrogram_vmin": 0.0,
    "transient_spectrogram_vmax": None,
    "transient_spectrogram_fmax_hz": 2.0e6,
    "transient_probe_stride": 4,
    "transient_baseline_end_time_s": 0.005,
    "transient_arrival_noise_sigma": 6.0,
    "transient_arrival_peak_fraction": 0.05,
    "transient_arrival_persistent_samples": 4,
    "transient_filter_edge_fraction": 0.01,

    # --- Phase 3: Nonlinear interaction diagnostics (bispectrum) ---
    # Quadratic phase-coupling detection via bicoherence.
    "make_nonlinear_diagnostics": False,
    # 8192 samples give about 61 kHz bins at the current 500 MHz sampling
    # rate and 14 conservative non-overlapping blocks in the pilot record.
    "nonlinear_nperseg": 8192,
    "nonlinear_noverlap": 0.5,              # overlap fraction
    "nonlinear_plot_probe_indices": [0, 49, 99, 249, 499],
    "nonlinear_target_freqs": None,          # None -> strongest independent peaks
    "nonlinear_target_band": [1.0e5, 50.0e6],
    "nonlinear_num_target_modes": 5,
    "nonlinear_fmax": 50.0e6,               # limits full-map cost
    "nonlinear_probe_stride": 4,             # spatial sampling for triad trend
    # No significance line is drawn without a validated null/surrogate model.
    "nonlinear_reference_threshold": None,
    "nonlinear_surrogate_validation": True,
    # 499 surrogates resolve p=0.002, sufficient for the first BH threshold
    # with the 15 unique unordered triads from five target frequencies.
    "nonlinear_surrogate_count": 499,
    "nonlinear_surrogate_alpha": 0.05,
    "nonlinear_minimum_independent_segments": 8,
    "nonlinear_surrogate_seed": 271828,
    "nonlinear_laser_frequency_hz": 10.0e6,

    # Unified evidence report. This classifies measured behavior without
    # attempting LST/PSE or instability-eigenmode identification.
    "make_evidence_classification": True,

    # --- Logging ---
    "debug_mode": True,
}


def run_validation_case(output_dir="validation_outputs"):
    """Run a fast synthetic case with known spectral and propagation answers.

    This test does not read AMReX plotfiles or production probe binaries. It
    exercises phase/coherence gating, packet kinematics, surrogate
    bicoherence significance, weighted modal sensitivity, and the
    measurement-only evidence classifier in addition to the core plots.

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
    ax.grid(False)
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
        scale="linear",
        vmin=0.0,
    )
    pdb.plot_fft_stft_comparison(
        time, measured, fs,
        output_path=str(
            output_dir / "03_fft_stft_comparison_validation.png"
        ),
        fmax=20.0e6,
        title="Synthetic same-probe FFT/STFT comparison",
        signal_label="Mean-subtracted synthetic pressure",
        nperseg=nperseg,
        noverlap=noverlap,
        spectrogram_scale="linear",
        spectrogram_vmin=0.0,
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
        time, modal_values, modal_x[:, None], variable="synthetic pressure",
        weights=mdb.scalar_compressible_energy_weights(
            modal_x, "pressure", rho_base=0.02, temperature_base=125.0
        )["weights"],
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
    modal_sensitivity = mdb.compute_modal_sensitivity(
        modal_dataset, pod_modes=2, dmd_ranks=[2],
        window_fractions=[[0.0, 0.5], [0.5, 1.0]],
    )

    # ------------------------------------------------------------------
    # 6. Robust packet propagation and nonlinear-significance gates.
    # ------------------------------------------------------------------
    packet_x = np.linspace(0.0, 0.04, 9)
    imposed_group_velocity = 800.0
    packet_centre = (
        time[0] + 10.0e-6 + packet_x / imposed_group_velocity
    )
    packet_envelopes = np.exp(0.5 * 8.0 * packet_x)[None, :] * np.exp(
        -0.5 * (
            (time[:, None] - packet_centre[None, :]) / 3.0e-6
        ) ** 2
    )
    packet_propagation = fdb.estimate_packet_propagation(
        packet_x, time, packet_envelopes,
        baseline_end_time_s=time[0] + 3.0e-6,
        peak_fraction=0.05,
    )
    pdb.plot_packet_propagation(
        packet_propagation,
        output_path=str(output_dir / "09_packet_propagation_validation.png"),
    )
    nonlinear_significance = fdb.compute_surrogate_triad_significance(
        coupled, fs, [f1, f2], nperseg=500, noverlap=250,
        n_surrogates=99, minimum_independent_segments=8,
        random_seed=20260724, laser_frequency_hz=10.0e6,
    )
    evidence_validation = fdb.classify_measured_dynamics(
        wave_report={
            "phase_fit_accepted_fraction": 1.0,
            "complex_wavenumber_accepted_fraction": 1.0,
        },
        packet_report=packet_propagation,
        nonlinear_report={
            "probe_results": [{
                "status": "complete",
                "significant_nonlaser_triad_count": int(np.sum(
                    nonlinear_significance["significant_fdr"]
                    & ~nonlinear_significance["laser_harmonic_related"]
                )),
                "significant_laser_related_triad_count": int(np.sum(
                    nonlinear_significance["significant_fdr"]
                    & nonlinear_significance["laser_harmonic_related"]
                )),
            }]
        },
        modal_summary={
            "pod_subspace_cosines": modal_sensitivity[
                "pod_subspace_min_cosine_to_first_window"
            ],
            "dmd_dominant_frequency_hz": modal_sensitivity[
                "dmd_dominant_frequency_hz"
            ],
            "dmd_condition_number": modal_sensitivity[
                "dmd_retained_condition_number"
            ],
        },
    )
    with (output_dir / "10_evidence_validation.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(_json_safe(evidence_validation), stream, indent=2)
        stream.write("\n")

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
        recovered_group_velocity_m_s=packet_propagation[
            "group_velocity_m_s"
        ],
        packet_arrival_fit_r_squared=packet_propagation[
            "arrival_fit_r_squared"
        ],
        nonlinear_empirical_p_value=nonlinear_significance[
            "empirical_p_value"
        ],
        nonlinear_significant_fdr=nonlinear_significance[
            "significant_fdr"
        ],
        modal_pod_subspace_cosine=modal_sensitivity[
            "pod_subspace_min_cosine_to_first_window"
        ],
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
    if not np.isclose(
            packet_propagation["group_velocity_m_s"],
            imposed_group_velocity, rtol=0.01):
        raise RuntimeError("Packet group-velocity validation failed")
    if not np.any(nonlinear_significance["significant_fdr"]):
        raise RuntimeError("Surrogate bicoherence significance validation failed")

    _ts(f"Phase speed: expected {expected_phase_speed:.3f}, "
        f"recovered {measured_phase_speed:.3f} m/s")
    _ts(f"Slow acoustic reference: {expected_slow_speed:.3f} m/s")
    _ts(f"Same-window reconstruction relative RMS: "
        f"{residual_stats['rms_residual_rel']:.4f}")
    _ts(f"Known coupled-triad bicoherence: {coupled_b2:.4f}")
    _ts(f"Common-mode held-out max relative RMS: {common_validation_rms:.3e}")
    _ts(f"DMD frequency: expected 5.000e6, recovered {recovered_dmd_frequency:.6e} Hz")
    _ts(
        f"Packet speed: expected {imposed_group_velocity:.3f}, recovered "
        f"{packet_propagation['group_velocity_m_s']:.3f} m/s"
    )
    _ts(
        "Surrogate-significant coupled triads: "
        f"{np.count_nonzero(nonlinear_significance['significant_fdr'])}"
    )
    _ts("Validation PASSED")
    return 0


# ---------------------------------------------------------------------------
#  INTERNAL WORKER FUNCTIONS
# ---------------------------------------------------------------------------

def _process_single_contour(args):
    """Worker: plot contours for one snapshot."""
    dataset, config = args
    try:
        _configure_plot_style(config)
        output_dir = Path(config["output_dir"]) / "Contours"
        output_dir.mkdir(parents=True, exist_ok=True)
        label = dataset.get("plot_label", "snapshot")

        for field_key in config["contour_fields"]:
            if field_key not in dataset["fields"]:
                continue
            out = output_dir / f"{label}_{field_key}.png"
            vlims = config.get("contour_vlims", {}).get(field_key, [None, None])
            cmap = config.get("contour_cmaps", {}).get(
                field_key,
                config.get("contour_cmap", "viridis"),
            )
            norm = config.get("contour_norms", {}).get(
                field_key,
                config.get("contour_norm", "linear"),
            )
            pdb.plot_contour(
                dataset, field_key,
                output_path=str(out),
                title=_plot_title(dataset, field_key, config),
                time_annotation=(
                    _plot_time_label(dataset, config)
                    if config.get("plot_show_time_annotation", False)
                    else None
                ),
                time_annotation_location=config.get(
                    "plot_time_annotation_location", "upper left"
                ),
                cmap=pdb.resolve_cmap(cmap),
                norm=norm,
                vmin=vlims[0],
                vmax=vlims[1],
                xlim=config.get("contour_xlim"),
                ylim=config.get("contour_ylim"),
                colorbar_shrink=config.get(
                    "contour_colorbar_shrink", "auto"
                ),
                colorbar_reference_span=config.get(
                    "contour_colorbar_reference_span", 0.25
                ),
                font_scale=config.get("contour_font_scale", "auto"),
                font_reference_span=config.get(
                    "contour_font_reference_span", 0.2
                ),
                colorbar_font_scale=config.get(
                    "contour_colorbar_font_scale", 0.85
                ),
            )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_single_line(args):
    """Worker: extract and plot line profiles for one snapshot."""
    dataset, config = args
    try:
        _configure_plot_style(config)
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
                    derive_native_vorticity=(
                        _config_needs_native_vorticity(config)
                    ),
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
        reference_model = config.get("line_reference_model", "none")
        reference_cfg = config.get("line_reference", {})
        legacy_bl_cfg = config.get("line_blasius", {})
        reference_enabled = (
            config.get("line_reference_overlay", False)
            and reference_model != "none"
        )
        if (reference_enabled and reference_model == "compressible_similarity"
                and (config.get("geometry_type") != "flat_plate"
                     or not reference_cfg.get("zero_pressure_gradient", False))):
            _ts(
                "  [Line] Compressible similarity reference disabled: it is valid "
                "only for a zero-pressure-gradient flat plate."
            )
            reference_enabled = False
        if reference_model == "compressible_similarity":
            bl_cfg = reference_cfg
        else:
            bl_cfg = legacy_bl_cfg
        reference_fields = config.get(
            "line_reference_fields", config.get("line_blasius_fields", [])
        )
        reference_cache = {}

        similarity_labels = {
            "x_velocity": r"$u/U_\infty$",
            "temperature": r"$\theta=T/T_\infty$",
            "density": r"$\rho/\rho_\infty$",
        }
        similarity_scales = {
            "x_velocity": reference_cfg.get("u_inf"),
            "temperature": reference_cfg.get("T_inf"),
            "density": reference_cfg.get("rho_inf"),
        }
        similarity_eta_cache = {}

        def _similarity_eta(ds, x_loc, profile_y):
            """Return the CFD density-weighted eta coordinate at one station."""
            cache_key = (id(ds), float(x_loc))
            cached = similarity_eta_cache.get(cache_key)
            if cached is None:
                rho_profile = fdb.extract_line(ds, x_loc, "density")
                if line_ylim is not None and len(rho_profile["y"]) > 1:
                    y_min, y_max = line_ylim
                    mask = ((rho_profile["y"] >= y_min)
                            & (rho_profile["y"] <= y_max))
                    rho_profile["y"] = rho_profile["y"][mask]
                    rho_profile["values"] = rho_profile["values"][mask]
                if reference_cfg.get("transport_model", "constant") == "constant":
                    mu_inf = reference_cfg["mu"]
                else:
                    mu_inf = fdb.sutherland_viscosity(reference_cfg["T_inf"])
                eta = fdb.compute_compressible_similarity_coordinate(
                    rho_profile["y"], rho_profile["values"], x_loc,
                    u_inf=reference_cfg["u_inf"],
                    rho_inf=reference_cfg["rho_inf"],
                    mu_inf=mu_inf,
                    leading_edge_x=reference_cfg.get("leading_edge_x", 0.0),
                )
                cached = (np.asarray(rho_profile["y"], dtype=float), eta)
                similarity_eta_cache[cache_key] = cached
            return np.interp(profile_y, cached[0], cached[1])

        def _use_similarity_coordinates(profile, ds, x_loc, field_key):
            """Convert one CFD profile to eta and a dimensionless field."""
            scale = similarity_scales.get(field_key)
            if scale is None or float(scale) <= 0.0:
                raise ValueError(f"No positive similarity scale for {field_key}")
            profile["eta"] = _similarity_eta(
                ds, x_loc, np.asarray(profile["y"], dtype=float)
            )
            profile["values"] = np.asarray(profile["values"], dtype=float) / float(scale)
            profile.pop("boundary_layer_height", None)

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
            similarity_plot = (
                reference_enabled
                and reference_model == "compressible_similarity"
                and field_key in reference_fields
            )
            profiles = []
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
                    if similarity_plot:
                        _use_similarity_coordinates(
                            prof, dataset, x_loc, field_key
                        )
                    profiles.append(prof)

                    # Independently solved compressible laminar base-flow
                    # reference.  This deliberately does not use the current
                    # CFD profile, so it remains meaningful for transient data.
                    if (reference_enabled
                            and reference_model == "compressible_similarity"
                            and field_key in reference_fields):
                        try:
                            cache_key = (
                                float(x_loc), tuple(np.asarray(prof["y"], dtype=float))
                            )
                            reference = reference_cache.get(cache_key)
                            if reference is None:
                                reference = fdb.compute_compressible_flat_plate_reference_profile(
                                    y=prof["y"], x_loc=x_loc,
                                    u_inf=reference_cfg["u_inf"],
                                    T_inf=reference_cfg["T_inf"],
                                    rho_inf=reference_cfg["rho_inf"],
                                    T_wall=reference_cfg.get("T_wall"),
                                    gamma=reference_cfg.get("gamma", 1.4),
                                    R=reference_cfg.get("R", 287.05),
                                    Cp=reference_cfg.get("Cp", 1004.0),
                                    leading_edge_x=reference_cfg.get("leading_edge_x", 0.0),
                                    transport_model=reference_cfg.get("transport_model", "constant"),
                                    mu=reference_cfg.get("mu", 8.65e-6),
                                    k=reference_cfg.get("k", 0.012415),
                                    Pr=reference_cfg.get("Pr", 0.71),
                                )
                                reference_cache[cache_key] = reference
                            reference_values = {
                                "x_velocity": reference["u_ratio"],
                                "temperature": reference["theta"],
                                "density": (
                                    reference["density_similarity"]
                                    / float(reference_cfg["rho_inf"])
                                ),
                            }
                            ref_profile = {
                                "eta": reference["eta"],
                                "values": reference_values[field_key],
                                "label": rf"Compressible similarity, $x={x_loc:.3f}$ m",
                                "field_key": field_key,
                                "linestyle": "--",
                                "reference_model": "compressible_flat_plate_similarity",
                            }
                            profiles.append(ref_profile)
                        except Exception as exc:
                            fdb._log_error(
                                f"Compressible similarity reference at x={x_loc:.3f}", exc
                            )

                    # Retained only for deliberate legacy comparisons.
                    if (reference_enabled
                            and reference_model == "incompressible_blasius_legacy"
                            and field_key in config.get("line_blasius_fields", [])):
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

                    # Optional external comparison run from another folder.
                    if (config.get("line_compare_overlay", False)
                            and compare_ds is not None
                            and field_key in reference_fields):
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
                            if similarity_plot:
                                _use_similarity_coordinates(
                                    cprof, compare_ds, x_loc, field_key
                                )
                            compare_profiles.append(cprof)
                        except Exception as exc:
                            fdb._log_error(f"Comparison line extraction at x={x_loc:.3f}", exc)
                except Exception as exc:
                    fdb._log_error(f"Line extraction at x={x_loc:.3f}", exc)

            if compare_profiles:
                profiles.extend(compare_profiles)

            if profiles:
                out = output_dir / f"{label}_{field_key}_profiles.png"
                normalize_y = (
                    config.get("line_normalize_by_delta99", False)
                    and not similarity_plot
                )
                pdb.plot_line_profiles(
                    profiles,
                    field_label=(
                        similarity_labels.get(field_key) if similarity_plot else None
                    ),
                    output_path=str(out),
                    title=_plot_title(dataset, field_key, config),
                    time_annotation=(
                        _plot_time_label(dataset, config)
                        if config.get("plot_show_time_annotation", False)
                        else None
                    ),
                    time_annotation_location=config.get(
                        "plot_time_annotation_location", "upper left"
                    ),
                    swap_axes=True,
                    xlabel=(
                        r"$\eta$" if similarity_plot else
                        (r"$y/\delta_{99}$" if normalize_y else r"$y$ [m]")
                    ),
                    x_key="eta" if similarity_plot else "y",
                    coordinate_normalization=(
                        "boundary_layer_height" if normalize_y else None
                    ),
                    annotate_boundary_layer=(not normalize_y and not similarity_plot),
                    coordinate_limits=(
                        config.get("line_similarity_eta_limits")
                        if similarity_plot else
                        (config.get("line_normalized_coordinate_limits")
                         if normalize_y else None)
                    ),
                )

        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_single_streamline(args):
    """Worker: plot streamlines for one snapshot."""
    dataset, config = args
    try:
        _configure_plot_style(config)
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
                "Streamlines — " + _plot_time_label(dataset, config)
                if config.get("plot_show_titles", False) else None
            ),
            time_annotation=(
                _plot_time_label(dataset, config)
                if config.get("plot_show_time_annotation", False) else None
            ),
            time_annotation_location=config.get(
                "plot_time_annotation_location", "upper left"
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
            p_inf=config.get("force", {}).get("reference", {}).get("p_inf"),
        )

        # Plot surface properties
        prop_out = output_dir / f"properties_{label}.png"
        pdb.plot_surface_properties(
            surface_data,
            snapshot=_plot_time_label(dataset, config),
            output_path=str(prop_out),
        )

        # Certified force integration is deliberately decoupled from this
        # legacy general-surface visualisation path.
        return (label, True, {"surface": surface_data, "forces": None})
    except Exception as exc:
        return (label, False, str(exc))


def _certified_force_reference(config, pressure_increment=False):
    """Translate public force configuration into the numerical API."""
    force = config["force"]
    reference = force["reference"]
    return {
        "rho_inf": float(reference["rho_inf"]),
        "u_inf": float(reference["u_inf"]),
        "p_inf": 0.0 if pressure_increment else float(reference["p_inf"]),
        "chord": float(reference["chord_m"]),
        "span": float(reference.get("span_m", 1.0)),
        "moment_origin": np.asarray(reference["moment_origin_m"], dtype=float),
        "x_range_m": np.asarray(force["geometry"]["x_range_m"], dtype=float),
        "viscosity_pa_s": float(force["transport"]["mu_pa_s"]),
    }


def _force_output_metadata(config):
    """Metadata contract shared by every certified force NPZ product."""
    force = config["force"]
    reference = force["reference"]
    transport = force["transport"]
    return {
        "schema_version": np.array(1),
        "certified_scope": np.array("stationary_one_sided_flat_plate"),
        "load_designation": np.array("one_sided_per_unit_span"),
        "pressure_convention": np.array("p_wall_minus_explicit_p_inf"),
        "geometry_type": np.array(force["geometry"]["type"]),
        "x_range_m": np.asarray(force["geometry"]["x_range_m"], dtype=float),
        "wall_y_m": np.array(force["geometry"]["wall_y_m"]),
        "rho_inf_kg_m3": np.array(reference["rho_inf"]),
        "u_inf_m_s": np.array(reference["u_inf"]),
        "p_inf_pa": np.array(reference["p_inf"]),
        "T_inf_k": np.array(reference["T_inf"]),
        "chord_m": np.array(reference["chord_m"]),
        "span_m": np.array(reference["span_m"]),
        "moment_origin_m": np.asarray(
            reference["moment_origin_m"], dtype=float
        ),
        "transport_model": np.array(transport["model"]),
        "mu_pa_s": np.array(transport["mu_pa_s"]),
        "k_w_m_k": np.array(transport["k_w_m_k"]),
        "wall_temperature_k": np.array(transport["wall_temperature_k"]),
    }


def _save_force_wall_npz(path, wall, config, increment_wall=None):
    """Persist one wall snapshot without object/pickle arrays."""
    arrays = _force_output_metadata(config)
    for key, value in wall.items():
        if isinstance(value, (str, os.PathLike)):
            arrays[key] = np.array(str(value))
        elif value is None:
            continue
        else:
            candidate = np.asarray(value)
            if candidate.dtype != object:
                arrays[key] = candidate
    if increment_wall is not None:
        for key, value in increment_wall.items():
            if isinstance(value, (str, os.PathLike)):
                arrays[f"delta_{key}"] = np.array(str(value))
            elif value is not None:
                candidate = np.asarray(value)
                if candidate.dtype != object:
                    arrays[f"delta_{key}"] = candidate
    np.savez_compressed(path, **arrays)


def _save_boundary_layer_npz(path, profiles, config):
    arrays = {
        **_force_output_metadata(config),
        "station_count": np.array(len(profiles), dtype=int),
    }
    for index, profile in enumerate(profiles):
        prefix = f"station_{index:03d}_"
        for key, value in profile.items():
            if isinstance(value, (str, os.PathLike)):
                arrays[prefix + key] = np.array(str(value))
            elif value is not None:
                candidate = np.asarray(value)
                if candidate.dtype != object:
                    arrays[prefix + key] = candidate
    np.savez_compressed(path, **arrays)


def _process_single_certified_force(args):
    """Extract and integrate one certified flat-plate force snapshot."""
    plotfile_path, config, baseline_wall, paired_baseline_path = args
    label = os.path.basename(os.fspath(plotfile_path))
    force_cfg = config["force"]
    geometry = force_cfg["geometry"]
    reference = force_cfg["reference"]
    transport = force_cfg["transport"]
    wall_fit = force_cfg["wall_fit"]
    quality = force_cfg["quality"]
    try:
        wall = fdb.extract_native_flat_plate_wall(
            plotfile_path,
            x_range_m=geometry["x_range_m"],
            wall_y_m=geometry["wall_y_m"],
            p_inf_pa=reference["p_inf"],
            rho_inf_kg_m3=reference["rho_inf"],
            u_inf_m_s=reference["u_inf"],
            viscosity_pa_s=transport["mu_pa_s"],
            pressure_order=wall_fit["pressure_order"],
            velocity_order=wall_fit["velocity_order"],
            fluid_points=wall_fit["fluid_points"],
            viscosity_relative_tolerance=wall_fit[
                "viscosity_relative_tolerance"
            ],
        )
        force_result = fdb.integrate_flat_plate_wall_forces(
            wall,
            _certified_force_reference(config),
            minimum_coverage=quality["minimum_coverage"],
            maximum_gap_widths=quality["maximum_gap_widths"],
        )
        time_value = float(wall["time_s"])
        laser_start = float(force_cfg["laser"]["start_time_s"])
        force_result["time"] = time_value
        force_result["time_relative_s"] = time_value - laser_start
        force_result["analysis_classification"] = (
            "short_transient_validation"
        )
        force_result.update({
            "load_designation": "one_sided_per_unit_span",
            "pressure_convention": "p_wall_minus_explicit_p_inf",
            "transport_model": transport["model"],
            "mu_pa_s": float(transport["mu_pa_s"]),
            "k_w_m_k": float(transport["k_w_m_k"]),
            "wall_temperature_k": float(transport["wall_temperature_k"]),
            "moment_origin_x_m": float(reference["moment_origin_m"][0]),
            "moment_origin_y_m": float(reference["moment_origin_m"][1]),
        })

        comparison_wall = baseline_wall
        if paired_baseline_path is not None:
            comparison_wall = fdb.extract_native_flat_plate_wall(
                paired_baseline_path,
                x_range_m=geometry["x_range_m"],
                wall_y_m=geometry["wall_y_m"],
                p_inf_pa=reference["p_inf"],
                rho_inf_kg_m3=reference["rho_inf"],
                u_inf_m_s=reference["u_inf"],
                viscosity_pa_s=transport["mu_pa_s"],
                pressure_order=wall_fit["pressure_order"],
                velocity_order=wall_fit["velocity_order"],
                fluid_points=wall_fit["fluid_points"],
                viscosity_relative_tolerance=wall_fit[
                    "viscosity_relative_tolerance"
                ],
            )

        increment_wall = None
        increment_force = None
        if comparison_wall is not None:
            increment_wall = fdb.difference_flat_plate_wall_surfaces(
                wall, comparison_wall
            )
            increment_force = fdb.integrate_flat_plate_wall_forces(
                increment_wall,
                _certified_force_reference(config, pressure_increment=True),
                minimum_coverage=quality["minimum_coverage"],
                maximum_gap_widths=quality["maximum_gap_widths"],
            )
            increment_mapping = {
                "D_pressure_N_m": "delta_D_pressure_N_m",
                "D_viscous_N_m": "delta_D_viscous_N_m",
                "D_total_N_m": "delta_D_total_N_m",
                "N_pressure_N_m": "delta_N_pressure_N_m",
                "N_viscous_N_m": "delta_N_viscous_N_m",
                "N_total_N_m": "delta_N_total_N_m",
                "M_pressure_N": "delta_M_pressure_N",
                "M_viscous_N": "delta_M_viscous_N",
                "M_total_N": "delta_M_total_N",
                "C_D_pressure": "delta_C_D_pressure",
                "C_D_viscous": "delta_C_D_viscous",
                "C_D": "delta_C_D",
                "C_N_pressure_one_sided":
                    "delta_C_N_pressure_one_sided",
                "C_N_viscous_one_sided":
                    "delta_C_N_viscous_one_sided",
                "C_N_one_sided": "delta_C_N_one_sided",
                "C_L_one_sided": "delta_C_L_one_sided",
                "C_M_pressure": "delta_C_M_pressure",
                "C_M_viscous": "delta_C_M_viscous",
                "C_M": "delta_C_M",
            }
            for source_key, destination_key in increment_mapping.items():
                force_result[destination_key] = increment_force[source_key]
            comparison_force = fdb.integrate_flat_plate_wall_forces(
                comparison_wall,
                _certified_force_reference(config),
                minimum_coverage=quality["minimum_coverage"],
                maximum_gap_widths=quality["maximum_gap_widths"],
            )
            for key in (
                    "D_pressure_N_m", "D_viscous_N_m", "D_total_N_m",
                    "N_pressure_N_m", "N_viscous_N_m", "N_total_N_m",
                    "M_pressure_N", "M_viscous_N", "M_total_N",
                    "C_D_pressure", "C_D_viscous", "C_D",
                    "C_N_pressure_one_sided", "C_N_viscous_one_sided",
                    "C_N_one_sided", "C_L_one_sided",
                    "C_M_pressure", "C_M_viscous", "C_M"):
                force_result[f"baseline_{key}"] = comparison_force[key]
            direct_delta = (
                force_result["D_total_N_m"]
                - comparison_force["D_total_N_m"]
            )
            force_result["delta_drag_consistency_error_N_m"] = (
                increment_force["D_total_N_m"] - direct_delta
            )
            force_result["baseline_source"] = comparison_wall.get("source")
            force_result["baseline_amr_max_level"] = int(
                comparison_wall.get("amr_max_level", -1)
            )
            force_result["current_amr_max_level"] = int(
                wall.get("amr_max_level", -1)
            )
            force_result["baseline_amr_level_match"] = bool(
                force_result["baseline_amr_max_level"]
                == force_result["current_amr_max_level"]
            )

        profiles = []
        bl_cfg = force_cfg["boundary_layer"]
        if bl_cfg.get("enabled", False):
            profiles = fdb.extract_native_flat_plate_boundary_layers(
                plotfile_path,
                stations_m=bl_cfg["stations_m"],
                maximum_height_m=bl_cfg["maximum_height_m"],
                wall_y_m=geometry["wall_y_m"],
                wall_temperature_k=transport["wall_temperature_k"],
                viscosity_pa_s=transport["mu_pa_s"],
                conductivity_w_m_k=transport["k_w_m_k"],
                fluid_points=wall_fit["fluid_points"],
            )

        output_root = Path(config["output_dir"]) / "ForceAnalysis"
        wall_dir = output_root / "WallSurfaces"
        bl_dir = output_root / "BoundaryLayers"
        wall_dir.mkdir(parents=True, exist_ok=True)
        _save_force_wall_npz(
            wall_dir / f"wall_surface_{label}.npz",
            wall, config, increment_wall=increment_wall,
        )
        if profiles:
            bl_dir.mkdir(parents=True, exist_ok=True)
            _save_boundary_layer_npz(
                bl_dir / f"boundary_layer_{label}.npz", profiles, config
            )
        return (
            label, True,
            {
                "wall": wall,
                "force": force_result,
                "increment_wall": increment_wall,
                "increment_force": increment_force,
                "boundary_layers": profiles,
            },
        )
    except Exception as exc:
        return (label, False, str(exc))


def _piecewise_wall_values(wall, key, centers):
    starts = np.asarray(wall["x_left_m"], dtype=float)
    ends = np.asarray(wall["x_right_m"], dtype=float)
    values = np.asarray(wall[key], dtype=float)
    centers = np.asarray(centers, dtype=float)
    indices = np.searchsorted(starts, centers, side="right") - 1
    valid = (
        (indices >= 0)
        & (indices < len(starts))
        & (centers < ends[np.clip(indices, 0, len(ends) - 1)] + 1.0e-12)
    )
    output = np.full(centers.shape, np.nan)
    output[valid] = values[indices[valid]]
    return output


def _run_force_core_self_validation():
    """Fast manufactured checks included in every force validation report."""
    edges = np.linspace(0.0, 1.0, 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    distance = np.array([0.01, 0.02, 0.03, 0.04])
    pressure = 120.0 + 3.0 * distance + 2.0 * distance ** 2
    velocity = 50.0 * distance - 4.0 * distance ** 2
    wall = fdb.reconstruct_flat_plate_wall_stencils(
        edges[:-1], edges[1:],
        np.tile(distance, (len(centers), 1)),
        np.tile(pressure, (len(centers), 1)),
        np.tile(velocity, (len(centers), 1)),
        viscosity_pa_s=2.0,
        p_inf_pa=100.0, rho_inf_kg_m3=2.0, u_inf_m_s=10.0,
    )
    wall["requested_x_range_m"] = np.array([0.0, 1.0])
    result = fdb.integrate_flat_plate_wall_forces(
        wall,
        {
            "rho_inf": 2.0, "u_inf": 10.0, "p_inf": 100.0,
            "chord": 1.0, "moment_origin": [0.25, 0.0],
        },
    )
    checks = {
        "wall_pressure_reconstruction": bool(np.allclose(
            wall["p_wall_pa"], 120.0, rtol=0.0, atol=1.0e-10
        )),
        "wall_shear_reconstruction": bool(np.allclose(
            wall["tau_wall_pa"], 100.0, rtol=0.0, atol=1.0e-10
        )),
        "constant_shear_integration": bool(np.isclose(
            result["D_total_N_m"], 100.0, rtol=0.0, atol=1.0e-10
        )),
        "constant_pressure_sign": bool(np.isclose(
            result["N_total_N_m"], -20.0, rtol=0.0, atol=1.0e-10
        )),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
    }


def _evaluate_force_baseline_drift(config, output_dir):
    baseline_cfg = config["force"]["baseline"]
    sample_count = int(baseline_cfg.get("drift_sample_count", 0))
    if sample_count <= 0:
        return {"status": "not_requested"}
    source = baseline_cfg.get("drift_data_source") or config["data_source"]
    paths = fdb.discover_plotfile_paths(
        source, plot_prefix=baseline_cfg["drift_plot_prefix"]
    )
    if not paths:
        return {"status": "unavailable", "reason": "no drift plotfiles found"}
    indices = np.unique(np.linspace(
        0, len(paths) - 1, min(sample_count, len(paths)), dtype=int
    ))
    force_cfg = config["force"]
    geometry = force_cfg["geometry"]
    reference = force_cfg["reference"]
    transport = force_cfg["transport"]
    fit = force_cfg["wall_fit"]
    quality = force_cfg["quality"]
    records = []
    for index in indices:
        wall = fdb.extract_native_flat_plate_wall(
            paths[int(index)],
            x_range_m=geometry["x_range_m"],
            wall_y_m=geometry["wall_y_m"],
            p_inf_pa=reference["p_inf"],
            rho_inf_kg_m3=reference["rho_inf"],
            u_inf_m_s=reference["u_inf"],
            viscosity_pa_s=transport["mu_pa_s"],
            pressure_order=fit["pressure_order"],
            velocity_order=fit["velocity_order"],
            fluid_points=fit["fluid_points"],
            viscosity_relative_tolerance=fit[
                "viscosity_relative_tolerance"
            ],
        )
        force = fdb.integrate_flat_plate_wall_forces(
            wall, _certified_force_reference(config),
            minimum_coverage=quality["minimum_coverage"],
            maximum_gap_widths=quality["maximum_gap_widths"],
        )
        records.append(force)
    times = np.asarray([item["time"] for item in records])
    drag = np.asarray([item["D_total_N_m"] for item in records])
    normal = np.asarray([item["N_total_N_m"] for item in records])
    moment = np.asarray([item["M_total_N"] for item in records])
    centered_time = times - times[0]

    def statistics(values):
        slope = (
            float(np.polyfit(centered_time, values, 1)[0])
            if len(values) >= 2 and np.ptp(centered_time) > 0.0 else np.nan
        )
        return {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1))
            if len(values) > 1 else 0.0,
            "peak_to_peak": float(np.ptp(values)),
            "slope_per_s": slope,
        }

    np.savez_compressed(
        Path(output_dir) / "baseline_drift.npz",
        **_force_output_metadata(config),
        source_paths=np.asarray([paths[int(index)] for index in indices]),
        time_s=times, D_total_N_m=drag, N_total_N_m=normal, M_total_N=moment,
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for axis, values, ylabel in zip(
            axes, (drag, normal, moment),
            (r"$D'$ [N/m]", r"$N'_{\mathrm{1s}}$ [N/m]", r"$M'_z$ [N]")):
        axis.plot(centered_time * 1.0e6, values, "o-")
        axis.set_ylabel(ylabel)
        axis.grid(False)
    axes[-1].set_xlabel("Laser-off continuation time [µs]")
    axes[0].set_title("Laser-off baseline drift")
    figure.tight_layout()
    figure.savefig(
        Path(output_dir) / "baseline_drift.png", dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)
    return {
        "status": "complete",
        "sample_count": len(records),
        "time_span_s": float(np.ptp(times)),
        "drag": statistics(drag),
        "normal": statistics(normal),
        "moment": statistics(moment),
    }


def _validate_baseline_against_similarity(config, baseline_wall, profiles):
    if baseline_wall is None or not profiles:
        return {"status": "not_available"}
    force_cfg = config["force"]
    reference = force_cfg["reference"]
    transport = force_cfg["transport"]
    comparisons = []
    for profile in profiles:
        x_station = float(profile["x_station_m"])
        similarity = fdb.compute_compressible_flat_plate_reference_profile(
            y=np.asarray(profile["wall_distance_m"], dtype=float),
            x_loc=x_station,
            u_inf=reference["u_inf"],
            T_inf=reference["T_inf"],
            rho_inf=reference["rho_inf"],
            T_wall=transport["wall_temperature_k"],
            gamma=1.4,
            R=reference["R"],
            Cp=1004.0,
            transport_model="constant",
            mu=transport["mu_pa_s"],
            k=transport["k_w_m_k"],
        )
        scalars = similarity["scalars"]
        comparisons.append({
            "x_station_m": x_station,
            "simulation_C_f": float(profile["C_f"]),
            "reference_C_f": float(scalars["C_f"]),
            "C_f_relative_error": float(
                abs(profile["C_f"] - scalars["C_f"])
                / max(abs(scalars["C_f"]), np.finfo(float).tiny)
            ),
            "simulation_delta_99_m": float(profile["delta_99_m"]),
            "reference_delta_99_m": float(scalars["delta_99"]),
            "simulation_theta_m": float(profile["theta_m"]),
            "reference_theta_m": float(scalars["theta"]),
        })

    x_start = max(
        0.05, float(force_cfg["geometry"]["x_range_m"][0])
    )
    x_end = float(force_cfg["geometry"]["x_range_m"][1])
    x_left = np.asarray(baseline_wall["x_left_m"])
    x_right = np.asarray(baseline_wall["x_right_m"])
    widths = np.maximum(
        np.minimum(x_right, x_end) - np.maximum(x_left, x_start), 0.0
    )
    simulated_drag = float(np.sum(
        np.asarray(baseline_wall["tau_wall_pa"]) * widths
    ))
    similarity_constant = float(np.mean([
        item["reference_C_f"] * np.sqrt(item["x_station_m"])
        for item in comparisons
    ]))
    q_inf = (
        0.5 * float(reference["rho_inf"]) * float(reference["u_inf"]) ** 2
    )
    reference_drag = float(
        q_inf * 2.0 * similarity_constant
        * (np.sqrt(x_end) - np.sqrt(x_start))
    )
    maximum_cf_error = max(item["C_f_relative_error"] for item in comparisons)
    drag_error = abs(simulated_drag - reference_drag) / max(
        abs(reference_drag), np.finfo(float).tiny
    )
    return {
        "status": (
            "pass" if maximum_cf_error <= 0.10 and drag_error <= 0.10
            else "outside_10_percent_target"
        ),
        "stations": comparisons,
        "maximum_C_f_relative_error": maximum_cf_error,
        "downstream_interval_m": [x_start, x_end],
        "simulation_viscous_drag_N_m": simulated_drag,
        "reference_viscous_drag_N_m": reference_drag,
        "viscous_drag_relative_error": drag_error,
    }


def _clip_wall_to_interval(wall, x_range_m):
    """Return a wall-face view clipped to one closed integration interval."""
    left_limit, right_limit = map(float, x_range_m)
    left = np.maximum(np.asarray(wall["x_left_m"], dtype=float), left_limit)
    right = np.minimum(np.asarray(wall["x_right_m"], dtype=float), right_limit)
    keep = right > left
    clipped = {}
    face_count = len(keep)
    for key, value in wall.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == face_count:
            clipped[key] = array[keep].copy()
        else:
            clipped[key] = value
    clipped["x_left_m"] = left[keep]
    clipped["x_right_m"] = right[keep]
    clipped["x_center_m"] = 0.5 * (left[keep] + right[keep])
    clipped["requested_x_range_m"] = np.array([left_limit, right_limit])
    return clipped


def _evaluate_force_grid_and_fit_sensitivity(config, baseline_wall):
    validation = config["force"].get("validation", {})
    if baseline_wall is None:
        return {"status": "not_available", "reason": "baseline wall is absent"}
    force_cfg = config["force"]
    quality = force_cfg["quality"]
    reference = _certified_force_reference(config)
    report = {"status": "complete"}

    first_order = dict(baseline_wall)
    first_order["p_wall_pa"] = np.asarray(
        baseline_wall["p_wall_linear_pa"], dtype=float
    )
    first_order["valid"] = (
        np.asarray(baseline_wall["valid"], dtype=bool)
        & np.isfinite(first_order["p_wall_pa"])
    )
    first_force = fdb.integrate_flat_plate_wall_forces(
        first_order, reference,
        minimum_coverage=quality["minimum_coverage"],
        maximum_gap_widths=quality["maximum_gap_widths"],
    )
    second_force = fdb.integrate_flat_plate_wall_forces(
        baseline_wall, reference,
        minimum_coverage=quality["minimum_coverage"],
        maximum_gap_widths=quality["maximum_gap_widths"],
    )
    report["pressure_fit"] = {
        "first_order_N_pressure_N_m": first_force["N_pressure_N_m"],
        "second_order_N_pressure_N_m": second_force["N_pressure_N_m"],
        "absolute_difference_N_m": abs(
            first_force["N_pressure_N_m"]
            - second_force["N_pressure_N_m"]
        ),
    }

    starts = validation.get("integration_start_locations_m", [])
    x_end = float(force_cfg["geometry"]["x_range_m"][1])
    report["integration_start_sensitivity"] = []
    for start in starts:
        clipped = _clip_wall_to_interval(
            baseline_wall, [float(start), x_end]
        )
        result = fdb.integrate_flat_plate_wall_forces(
            clipped, reference,
            minimum_coverage=quality["minimum_coverage"],
            maximum_gap_widths=quality["maximum_gap_widths"],
        )
        report["integration_start_sensitivity"].append({
            "start_x_m": float(start),
            "D_viscous_N_m": result["D_viscous_N_m"],
            "N_pressure_N_m": result["N_pressure_N_m"],
        })

    report["wall_stencil_sensitivity"] = {"status": "not_requested"}
    if validation.get("fit_sensitivity", False):
        fit = force_cfg["wall_fit"]
        geometry = force_cfg["geometry"]
        freestream = force_cfg["reference"]
        transport = force_cfg["transport"]
        stencils = []
        for points in (2, 3):
            wall = fdb.extract_native_flat_plate_wall(
                baseline_wall["source"],
                x_range_m=geometry["x_range_m"],
                wall_y_m=geometry["wall_y_m"],
                p_inf_pa=freestream["p_inf"],
                rho_inf_kg_m3=freestream["rho_inf"],
                u_inf_m_s=freestream["u_inf"],
                viscosity_pa_s=transport["mu_pa_s"],
                pressure_order=min(fit["pressure_order"], points - 1),
                velocity_order=fit["velocity_order"],
                fluid_points=points,
                viscosity_relative_tolerance=fit[
                    "viscosity_relative_tolerance"
                ],
            )
            stencils.append((points, wall))
        stencils.append((int(fit["fluid_points"]), baseline_wall))
        stencil_results = []
        downstream_drag = []
        for points, wall in stencils:
            clipped = _clip_wall_to_interval(wall, [0.05, x_end])
            force = fdb.integrate_flat_plate_wall_forces(
                clipped, reference,
                minimum_coverage=quality["minimum_coverage"],
                maximum_gap_widths=quality["maximum_gap_widths"],
            )
            downstream_drag.append(force["D_viscous_N_m"])
            stencil_results.append({
                "fluid_points": points,
                "downstream_D_viscous_N_m": force["D_viscous_N_m"],
            })
        finest_change = abs(downstream_drag[-1] - downstream_drag[-2]) / max(
            abs(downstream_drag[-1]), np.finfo(float).tiny
        )
        report["wall_stencil_sensitivity"] = {
            "status": (
                "pass" if finest_change < 0.05
                else "outside_5_percent_target"
            ),
            "treatments": stencil_results,
            "two_finest_relative_change": finest_change,
        }

    refined_path = validation.get("grid_comparison_plotfile")
    report["grid_comparison"] = {"status": "not_requested"}
    if refined_path:
        geometry = force_cfg["geometry"]
        freestream = force_cfg["reference"]
        transport = force_cfg["transport"]
        fit = force_cfg["wall_fit"]
        refined_wall = fdb.extract_native_flat_plate_wall(
            refined_path,
            x_range_m=geometry["x_range_m"],
            wall_y_m=geometry["wall_y_m"],
            p_inf_pa=freestream["p_inf"],
            rho_inf_kg_m3=freestream["rho_inf"],
            u_inf_m_s=freestream["u_inf"],
            viscosity_pa_s=transport["mu_pa_s"],
            pressure_order=fit["pressure_order"],
            velocity_order=fit["velocity_order"],
            fluid_points=fit["fluid_points"],
            viscosity_relative_tolerance=fit[
                "viscosity_relative_tolerance"
            ],
        )
        current = fdb.integrate_flat_plate_wall_forces(
            _clip_wall_to_interval(baseline_wall, [0.05, x_end]), reference
        )
        refined = fdb.integrate_flat_plate_wall_forces(
            _clip_wall_to_interval(refined_wall, [0.05, x_end]), reference
        )
        change = abs(
            refined["D_viscous_N_m"] - current["D_viscous_N_m"]
        ) / max(abs(refined["D_viscous_N_m"]), np.finfo(float).tiny)
        report["grid_comparison"] = {
            "status": (
                "pass" if change < 0.05 else "outside_5_percent_target"
            ),
            "current_source": baseline_wall["source"],
            "refined_source": os.fspath(refined_path),
            "current_downstream_D_viscous_N_m": current["D_viscous_N_m"],
            "refined_downstream_D_viscous_N_m": refined["D_viscous_N_m"],
            "relative_change": change,
        }
    sensitivity_states = (
        report["wall_stencil_sensitivity"]["status"],
        report["grid_comparison"]["status"],
    )
    report["status"] = (
        "pass" if all(
            state in ("pass", "not_requested") for state in sensitivity_states
        ) else "requires_attention"
    )
    return report


def _evaluate_force_control_volume(config, baseline_wall):
    validation = config["force"].get("validation", {})
    cv_cfg = validation.get("control_volume", {})
    if not cv_cfg.get("enabled", False):
        return {"status": "not_requested"}
    if baseline_wall is None:
        return {"status": "not_available", "reason": "baseline wall is absent"}
    plotfile = cv_cfg.get("plotfile") or baseline_wall.get("source")
    dataset = fdb.load_pelec_plotfile(
        plotfile,
        field_names=[
            "density", "pressure", "x_velocity", "y_velocity"
        ],
        convert_to_mks=True,
        maximum_level=cv_cfg.get("maximum_level"),
    )
    balance = fdb.compute_flat_plate_control_volume_force(
        dataset,
        x_range_m=cv_cfg["x_range_m"],
        y_top_m=cv_cfg["top_y_m"],
        viscosity_pa_s=config["force"]["transport"]["mu_pa_s"],
    )
    wall = _clip_wall_to_interval(baseline_wall, cv_cfg["x_range_m"])
    wall_force = fdb.integrate_flat_plate_wall_forces(
        wall, _certified_force_reference(config)
    )
    residual_vector = np.array([
        wall_force["D_total_N_m"] - balance["D_control_volume_N_m"],
        wall_force["N_total_N_m"] - balance["N_control_volume_N_m"],
    ])
    wall_vector = np.array([
        wall_force["D_total_N_m"], wall_force["N_total_N_m"]
    ])
    reference = config["force"]["reference"]
    q_inf = 0.5 * reference["rho_inf"] * reference["u_inf"] ** 2
    cv_chord = float(np.diff(np.asarray(cv_cfg["x_range_m"]))[0])
    closure = float(
        np.linalg.norm(residual_vector)
        / max(np.linalg.norm(wall_vector), q_inf * cv_chord)
    )
    return {
        "status": "pass" if closure <= 0.05 else "outside_5_percent_target",
        "normalized_closure_residual": closure,
        "target": 0.05,
        "wall_force": {
            "D_N_m": wall_force["D_total_N_m"],
            "N_N_m": wall_force["N_total_N_m"],
        },
        "momentum_balance": balance,
        "covering_grid_level": dataset["amr_max_level"],
        "source": os.fspath(plotfile),
    }


def _write_certified_force_products(
        config, forces_dict, wall_dict, increment_wall_dict,
        boundary_layer_results, baseline_wall=None, baseline_profiles=None):
    """Write force histories, distributed loads, linkage, and quality report."""
    force_dir = Path(config["output_dir"]) / "ForceAnalysis"
    force_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(
        forces_dict,
        key=lambda label: float(forces_dict[label].get("time", np.inf)),
    )
    entries = [forces_dict[label] for label in labels]
    times = np.asarray([item["time"] for item in entries], dtype=float)
    force_cfg = config["force"]
    laser_cfg = force_cfg["laser"]
    time_relative = times - float(laser_cfg["start_time_s"])

    pdb.plot_certified_force_timeseries(
        forces_dict,
        output_path=str(force_dir / "forces_vs_time.png"),
        laser_start_time=laser_cfg["start_time_s"],
    )

    scalar_keys = sorted({
        key for item in entries for key, value in item.items()
        if np.asarray(value).ndim == 0
        and not isinstance(value, (dict, list, tuple, np.ndarray))
        and key != "plot_label"
    })
    # Explicitly retain numeric and short string metadata; array-valued wall
    # distributions are stored separately.
    columns = ["plot_label", *scalar_keys]
    with (force_dir / "force_timeseries.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, item in zip(labels, entries):
            row = {"plot_label": label}
            for key in scalar_keys:
                value = item.get(key, "")
                if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
                    row[key] = value
            writer.writerow(row)

    arrays = {
        **_force_output_metadata(config),
        "labels": np.asarray(labels),
        "time_s": times,
        "time_relative_s": time_relative,
    }
    for key in scalar_keys:
        values = [item.get(key, np.nan) for item in entries]
        if all(isinstance(value, (int, float, np.integer, np.floating, bool))
               for value in values):
            arrays[key] = np.asarray(values)

    impulse_summary = {}
    peak_summary = {}
    for key, output_name in (
        ("delta_D_total_N_m", "drag"),
        ("delta_N_total_N_m", "normal"),
        ("delta_M_total_N", "moment"),
    ):
        values = np.asarray([item.get(key, np.nan) for item in entries])
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            peak_index = int(finite[np.argmax(np.abs(values[finite]))])
            peak_summary[output_name] = {
                "signed_peak": float(values[peak_index]),
                "absolute_peak": float(abs(values[peak_index])),
                "absolute_time_s": float(times[peak_index]),
                "laser_relative_time_s": float(time_relative[peak_index]),
            }
            arrays[f"{output_name}_signed_peak"] = np.array(
                values[peak_index]
            )
            arrays[f"{output_name}_peak_time_s"] = np.array(
                times[peak_index]
            )
            arrays[f"{output_name}_peak_delay_s"] = np.array(
                time_relative[peak_index]
            )
    impulse_interval = laser_cfg.get("impulse_interval_s")
    if impulse_interval is None:
        impulse_mask = np.ones(len(times), dtype=bool)
    else:
        impulse_mask = (
            (times >= float(impulse_interval[0]))
            & (times <= float(impulse_interval[1]))
        )
    if np.count_nonzero(impulse_mask) >= 2:
        for key, output_key in (
            ("delta_D_total_N_m", "drag_impulse_N_s_m"),
            ("delta_N_total_N_m", "normal_impulse_N_s_m"),
            ("delta_M_total_N", "moment_impulse_N_s"),
        ):
            values = np.asarray([item.get(key, np.nan) for item in entries])
            selected = values[impulse_mask]
            selected_time = times[impulse_mask]
            if np.all(np.isfinite(selected)):
                impulse_summary[output_key] = float(
                    fdb.trapezoidal_integral(selected, selected_time)
                )
                arrays[output_key] = np.array(impulse_summary[output_key])
        impulse_summary["interval_s"] = [
            float(times[impulse_mask][0]), float(times[impulse_mask][-1])
        ]
    np.savez_compressed(force_dir / "force_timeseries.npz", **arrays)

    surface_history = None
    if increment_wall_dict:
        geometry = force_cfg["geometry"]
        edges = np.linspace(
            float(geometry["x_range_m"][0]),
            float(geometry["x_range_m"][1]),
            2001,
        )
        x_centers = 0.5 * (edges[:-1] + edges[1:])
        selected_labels = [
            label for label in labels if label in increment_wall_dict
        ]
        selected_times = np.asarray([
            forces_dict[label]["time"] for label in selected_labels
        ])
        delta_pressure = np.vstack([
            _piecewise_wall_values(
                increment_wall_dict[label], "p_wall_pa", x_centers
            )
            for label in selected_labels
        ])
        delta_shear = np.vstack([
            _piecewise_wall_values(
                increment_wall_dict[label], "tau_wall_pa", x_centers
            )
            for label in selected_labels
        ])
        dx = np.diff(edges)
        delta_drag_density = delta_shear
        delta_normal_density = -delta_pressure
        cumulative_drag = np.cumsum(delta_drag_density * dx[None, :], axis=1)
        cumulative_normal = np.cumsum(
            delta_normal_density * dx[None, :], axis=1
        )
        surface_history = {
            **_force_output_metadata(config),
            "labels": np.asarray(selected_labels),
            "time_s": selected_times,
            "time_relative_s": (
                selected_times - float(laser_cfg["start_time_s"])
            ),
            "x_m": x_centers,
            "delta_pressure_pa": delta_pressure,
            "delta_shear_pa": delta_shear,
            "delta_dD_dx_N_m2": delta_drag_density,
            "delta_dN_dx_N_m2": delta_normal_density,
            "delta_D_cumulative_N_m": cumulative_drag,
            "delta_N_cumulative_N_m": cumulative_normal,
        }
        np.savez_compressed(
            force_dir / "force_surface_history.npz", **surface_history
        )
        pdb.plot_force_surface_history(
            surface_history["time_relative_s"], x_centers,
            delta_drag_density, r"$\Delta(dD'/dx)$ [N/m$^2$]",
            output_path=str(force_dir / "distributed_drag_history.png"),
        )
        pdb.plot_force_surface_history(
            surface_history["time_relative_s"], x_centers,
            delta_normal_density, r"$\Delta(dN'/dx)$ [N/m$^2$]",
            output_path=str(force_dir / "distributed_normal_history.png"),
        )

    if boundary_layer_results:
        bl_labels = [label for label in labels if label in boundary_layer_results]
        station_values = np.asarray(
            force_cfg["boundary_layer"]["stations_m"], dtype=float
        )
        bl_arrays = {
            **_force_output_metadata(config),
            "labels": np.asarray(bl_labels),
            "time_s": np.asarray([forces_dict[label]["time"] for label in bl_labels]),
            "stations_m": station_values,
        }
        for profile_key, output_key in (
            ("delta_99_m", "delta_99_m"),
            ("delta_star_m", "delta_star_m"),
            ("theta_m", "theta_m"),
            ("H", "H"),
            ("tau_wall_pa", "tau_wall_pa"),
            ("C_f", "C_f"),
            ("Re_theta", "Re_theta"),
            ("q_wall_w_m2", "q_wall_w_m2"),
            ("edge_velocity_m_s", "edge_velocity_m_s"),
            ("edge_density_kg_m3", "edge_density_kg_m3"),
            ("fit_condition_velocity", "fit_condition_velocity"),
            ("fit_condition_pressure", "fit_condition_pressure"),
        ):
            bl_arrays[output_key] = np.asarray([
                [profile.get(profile_key, np.nan) for profile in
                 boundary_layer_results[label]]
                for label in bl_labels
            ])
        all_profiles = [
            profile for label in bl_labels
            for profile in boundary_layer_results[label]
        ]
        maximum_points = max(
            (len(profile["wall_distance_m"]) for profile in all_profiles),
            default=0,
        )
        profile_shape = (
            len(bl_labels), len(station_values), maximum_points
        )
        for profile_key, output_key in (
            ("wall_distance_m", "wall_distance_profile_m"),
            ("u_t_m_s", "u_t_profile_m_s"),
            ("rho_kg_m3", "rho_profile_kg_m3"),
            ("temperature_k", "temperature_profile_k"),
            ("pressure_pa", "pressure_profile_pa"),
            ("mu_pa_s", "mu_profile_pa_s"),
            ("amr_level", "amr_level_profile"),
        ):
            fill = -1 if profile_key == "amr_level" else np.nan
            dtype = int if profile_key == "amr_level" else float
            values = np.full(profile_shape, fill, dtype=dtype)
            for time_index, label in enumerate(bl_labels):
                for station_index, profile in enumerate(
                        boundary_layer_results[label]):
                    source = np.asarray(profile[profile_key], dtype=dtype)
                    values[
                        time_index, station_index, :len(source)
                    ] = source
            bl_arrays[output_key] = values
        bl_arrays["profile_point_count"] = np.asarray([
            [len(profile["wall_distance_m"]) for profile in
             boundary_layer_results[label]]
            for label in bl_labels
        ], dtype=int)
        np.savez_compressed(
            force_dir / "boundary_layer_profiles.npz", **bl_arrays
        )

    linkage_time = times
    linkage_responses = {}
    if all("delta_D_total_N_m" in item for item in entries):
        linkage_responses = {
            "drag": np.asarray([
                item["delta_D_total_N_m"] for item in entries
            ]),
            "normal": np.asarray([
                item["delta_N_total_N_m"] for item in entries
            ]),
            "moment": np.asarray([
                item["delta_M_total_N"] for item in entries
            ]),
        }
    compact_agreement = {"status": "not_available"}
    compact_path = force_cfg["history"].get("path")
    if compact_path:
        compact = fdb.load_compact_wall_force_history(
            compact_path, _certified_force_reference(config)
        )
        baseline_force = (
            fdb.integrate_flat_plate_wall_forces(
                baseline_wall, _certified_force_reference(config),
                minimum_coverage=force_cfg["quality"]["minimum_coverage"],
                maximum_gap_widths=force_cfg["quality"][
                    "maximum_gap_widths"
                ],
            ) if baseline_wall is not None else None
        )
        linkage_time = np.asarray(compact["time_s"], dtype=float)
        linkage_responses = {
            "drag": np.asarray(compact["D_total_N_m"], dtype=float),
            "normal": np.asarray(compact["N_total_N_m"], dtype=float),
            "moment": np.asarray(compact["M_total_N"], dtype=float),
        }
        if baseline_force is not None:
            linkage_responses["drag"] -= baseline_force["D_total_N_m"]
            linkage_responses["normal"] -= baseline_force["N_total_N_m"]
            linkage_responses["moment"] -= baseline_force["M_total_N"]
        compact_arrays = _force_output_metadata(config)
        compact_arrays.update({
            key: value for key, value in compact.items()
            if not isinstance(value, (str, dict))
            and np.asarray(value).dtype != object
        })
        np.savez_compressed(
            force_dir / "compact_force_history.npz", **compact_arrays
        )
        matching = (
            (times >= linkage_time[0]) & (times <= linkage_time[-1])
        )
        if np.any(matching):
            relative_errors = {}
            offline_keys = {
                "drag": "D_total_N_m",
                "normal": "N_total_N_m",
                "moment": "M_total_N",
            }
            for name, offline_key in offline_keys.items():
                compact_at_plot = np.interp(
                    times[matching], linkage_time,
                    (
                        linkage_responses[name]
                        + (
                            0.0 if baseline_force is None else
                            baseline_force[offline_key]
                        )
                    ),
                )
                offline = np.asarray([
                    item[offline_key] for item in entries
                ])[matching]
                scale = np.maximum(
                    np.abs(offline),
                    1.0e-12 * float(force_cfg["reference"]["p_inf"])
                    * float(force_cfg["reference"]["chord_m"]),
                )
                relative_errors[name] = float(np.max(
                    np.abs(compact_at_plot - offline) / scale
                ))
            compact_agreement = {
                "status": (
                    "pass" if max(relative_errors.values()) <= 0.02
                    else "outside_2_percent_target"
                ),
                "maximum_relative_errors": relative_errors,
                "matching_plotfile_count": int(np.count_nonzero(matching)),
            }

    duration_s = (
        float(np.ptp(linkage_time)) if len(linkage_time) > 1 else 0.0
    )
    forcing_period_s = 1.0 / float(laser_cfg["frequency_hz"])
    forcing_periods = duration_s / forcing_period_s
    minimum_periods = float(
        force_cfg["quality"]["minimum_forcing_periods"]
    )
    linkage = {
        "status": "insufficient_data",
        "reason": (
            f"record contains {forcing_periods:.3f} forcing periods; "
            f"{minimum_periods:g} required"
        ),
        "duration_s": duration_s,
        "forcing_periods": forcing_periods,
    }
    if len(linkage_time) >= 3 and linkage_responses:
        uniform_time = np.linspace(
            linkage_time[0], linkage_time[-1], len(linkage_time)
        )
        pulse = fdb.gaussian_pulse_train(
            uniform_time,
            start_time_s=laser_cfg["start_time_s"],
            frequency_hz=laser_cfg["frequency_hz"],
            pulse_fwhm_s=laser_cfg["pulse_fwhm_s"],
            duration_s=laser_cfg["duration_s"],
        )
        linkage_arrays = {
            **_force_output_metadata(config),
            "time_s": uniform_time,
            "laser_input_normalized": pulse["signal"],
        }
        for short_name in ("drag", "normal", "moment"):
            response = np.interp(
                uniform_time, linkage_time, linkage_responses[short_name],
            )
            correlation = fdb.normalized_cross_correlation(
                pulse["signal"], response,
                uniform_time[1] - uniform_time[0],
            )
            linkage_arrays[f"{short_name}_response"] = response
            linkage_arrays[f"{short_name}_lag_s"] = correlation["lag_s"]
            linkage_arrays[f"{short_name}_correlation"] = correlation[
                "correlation"
            ]
            linkage[f"{short_name}_peak_lag_s"] = correlation["peak_lag_s"]
            linkage[f"{short_name}_peak_correlation"] = correlation[
                "peak_correlation"
            ]

        if forcing_periods >= minimum_periods:
            dt = float(uniform_time[1] - uniform_time[0])
            history_cfg = force_cfg["history"]
            spectral_outputs = {}
            try:
                for short_name in ("drag", "normal", "moment"):
                    spectral = fdb.compute_input_output_spectra(
                        pulse["signal"],
                        linkage_arrays[f"{short_name}_response"],
                        sample_rate_hz=1.0 / dt,
                        nperseg=history_cfg["spectral_nperseg"],
                        noverlap=history_cfg["spectral_noverlap"],
                        minimum_segments=force_cfg["quality"][
                            "minimum_welch_segments"
                        ],
                    )
                    for key, value in spectral.items():
                        spectral_outputs[f"{short_name}_{key}"] = value
                linkage_arrays.update(spectral_outputs)
                linkage["status"] = "spectral_analysis_complete"
                linkage["reason"] = None
            except ValueError as exc:
                linkage["reason"] = str(exc)
        np.savez_compressed(
            force_dir / "force_linkage.npz", **linkage_arrays
        )

    probe_force_linkage = {"status": "not_requested"}
    probe_paths = force_cfg["history"].get("probe_bin_files") or []
    if probe_paths and linkage_responses:
        try:
            probe_config = dict(config)
            probe_config["probe_bin_files"] = list(probe_paths)
            probe_data = _load_probe_timeseries(
                probe_config,
                force_cfg["history"].get("probe_pressure_var_col", 3),
                nt_skip=0,
                max_probes=force_cfg["history"].get("probe_max_probes"),
            )
            probe_time, probe_matrix, _, _ = _probe_matrix_on_common_time(
                probe_data, resample=False
            )
            # PeleC probe pressure and coordinates are stored in cgs units.
            pressure_pa = np.asarray(probe_matrix, dtype=float) * 0.1
            pressure_pa -= np.mean(pressure_pa, axis=0, keepdims=True)
            probe_x_m = np.asarray(
                [item.get("x", item.get("x_req")) for item in probe_data],
                dtype=float,
            ) * 0.01
            linkage_output = {
                **_force_output_metadata(config),
                "probe_x_m": probe_x_m,
                "pressure_convention": np.array(
                    "pressure fluctuation about common-interval mean"
                ),
            }
            linkage_status = []
            for response_name, response_values in linkage_responses.items():
                result = fdb.compute_probe_force_linkage(
                    linkage_time, response_values,
                    probe_time, pressure_pa, probe_x_m,
                    forcing_frequency_hz=laser_cfg["frequency_hz"],
                    minimum_forcing_periods=minimum_periods,
                    nperseg=force_cfg["history"]["spectral_nperseg"],
                    noverlap=force_cfg["history"]["spectral_noverlap"],
                    minimum_segments=force_cfg["quality"][
                        "minimum_welch_segments"
                    ],
                )
                for key, value in result.items():
                    if isinstance(value, str):
                        linkage_output[f"{response_name}_{key}"] = np.array(value)
                    elif value is not None:
                        linkage_output[f"{response_name}_{key}"] = value
                linkage_status.append(result["spectral_status"])
            np.savez_compressed(
                force_dir / "pressure_force_linkage.npz", **linkage_output
            )
            probe_force_linkage = {
                "status": (
                    "spectral_analysis_complete"
                    if all(value == "complete" for value in linkage_status)
                    else "time_domain_only"
                ),
                "probe_count": len(probe_data),
                "source_files": [os.fspath(path) for path in probe_paths],
                "pressure_units": "Pa",
                "coordinate_units": "m",
            }
        except Exception as exc:
            probe_force_linkage = {
                "status": "failed",
                "reason": str(exc),
                "source_files": [os.fspath(path) for path in probe_paths],
            }

    pressure_fit_difference = []
    coverage_values = []
    consistency_values = []
    baseline_level_matches = []
    for label in labels:
        wall = wall_dict[label]
        finite = np.isfinite(wall["p_wall_linear_pa"])
        if np.any(finite):
            pressure_fit_difference.append(float(np.nanmax(np.abs(
                wall["p_wall_pa"][finite] - wall["p_wall_linear_pa"][finite]
            ))))
        coverage_values.append(float(forces_dict[label]["coverage_fraction"]))
        if "delta_drag_consistency_error_N_m" in forces_dict[label]:
            consistency_values.append(abs(float(
                forces_dict[label]["delta_drag_consistency_error_N_m"]
            )))
        if "baseline_amr_level_match" in forces_dict[label]:
            baseline_level_matches.append(bool(
                forces_dict[label]["baseline_amr_level_match"]
            ))
    coverage_pass = (
        min(coverage_values) >= force_cfg["quality"]["minimum_coverage"]
    )
    baseline_level_pass = (
        all(baseline_level_matches) if baseline_level_matches else True
    )
    similarity_validation = _validate_baseline_against_similarity(
        config, baseline_wall, baseline_profiles or []
    )
    try:
        baseline_drift = _evaluate_force_baseline_drift(config, force_dir)
    except Exception as exc:
        baseline_drift = {"status": "failed", "reason": str(exc)}
    try:
        grid_and_fit = _evaluate_force_grid_and_fit_sensitivity(
            config, baseline_wall
        )
    except Exception as exc:
        grid_and_fit = {"status": "failed", "reason": str(exc)}
    try:
        control_volume = _evaluate_force_control_volume(
            config, baseline_wall
        )
    except Exception as exc:
        control_volume = {"status": "failed", "reason": str(exc)}
    accepted_auxiliary_states = {
        "pass", "complete", "not_requested", "not_available"
    }
    numerical_layers_pass = (
        coverage_pass
        and baseline_level_pass
        and similarity_validation.get("status") in accepted_auxiliary_states
        and grid_and_fit.get("status") in accepted_auxiliary_states
        and control_volume.get("status") in accepted_auxiliary_states
        and compact_agreement.get("status") in accepted_auxiliary_states
        and baseline_drift.get("status") in accepted_auxiliary_states
    )
    analysis_classification = (
        "production_second_order_force_analysis"
        if linkage["status"] == "spectral_analysis_complete"
        else "short_transient_validation"
    )
    report = {
        "schema_version": 1,
        "certified_scope": "stationary_one_sided_flat_plate",
        "code_provenance": _git_provenance(),
        "source_plotfiles": [
            os.fspath(item["source"]) for item in entries
            if item.get("source") is not None
        ],
        "baseline_source": (
            None if baseline_wall is None
            else os.fspath(baseline_wall.get("source"))
        ),
        "code_correctness": _run_force_core_self_validation(),
        "numerical_quality": {
            "status": (
                "pass" if numerical_layers_pass
                else "requires_attention"
            ),
            "minimum_wall_coverage": min(coverage_values),
            "baseline_amr_level_match": baseline_level_pass,
            "baseline_amr_warning": (
                None if baseline_level_pass else
                "laser and baseline plotfiles have different maximum AMR "
                "levels; viscous force increments include grid-change bias"
            ),
            "maximum_pressure_fit_difference_pa": max(
                pressure_fit_difference, default=np.nan
            ),
            "maximum_delta_integration_consistency_error_N_m": max(
                consistency_values, default=np.nan
            ),
            "compressible_similarity": similarity_validation,
            "compact_history_agreement": compact_agreement,
            "grid_and_fit_sensitivity": grid_and_fit,
            "control_volume_closure": control_volume,
            "baseline_drift": baseline_drift,
        },
        "scientific_adequacy": {
            "classification": analysis_classification,
            "duration_s": duration_s,
            "forcing_periods": forcing_periods,
            "spectral_inference_allowed": bool(
                linkage["status"] == "spectral_analysis_complete"
            ),
            "linkage": linkage,
            "probe_force_linkage": probe_force_linkage,
        },
        "impulses": impulse_summary,
        "peak_response": peak_summary,
        "reference": _json_safe(force_cfg["reference"]),
        "transport": _json_safe(force_cfg["transport"]),
        "geometry": _json_safe(force_cfg["geometry"]),
    }
    report_path = force_dir / "force_validation.json"
    temporary = force_dir / ".force_validation.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, report_path)
    return report


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
                if config.get("plot_show_titles", False) else None
            ),
            time_annotation=(
                _plot_time_label(dataset, config)
                if config.get("plot_show_time_annotation", False) else None
            ),
            time_annotation_location=config.get(
                "plot_time_annotation_location", "upper left"
            ),
            cmap=pdb.resolve_cmap(cmap),
            norm="linear",
            vmin=vmin,
            vmax=vmax,
            colorbar_label=r"$p'$ [Pa]" if field_key == "pressure"
            else rf"$\Delta$ {pdb.field_label(field_key)}",
            xlim=config.get("contour_xlim"),
            ylim=config.get("contour_ylim"),
            colorbar_shrink=config.get(
                "contour_colorbar_shrink", "auto"
            ),
            colorbar_reference_span=config.get(
                "contour_colorbar_reference_span", 0.25
            ),
            font_scale=config.get("contour_font_scale", "auto"),
            font_reference_span=config.get(
                "contour_font_reference_span", 0.2
            ),
            colorbar_font_scale=config.get(
                "contour_colorbar_font_scale", 0.85
            ),
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


_PROBE_V2_FILE_MAGIC = b"PROBES2\0"
_PROBE_V2_CHUNK_MAGIC = b"PRBCHNK2"
_PROBE_V2_FOOTER_MAGIC = b"PRBEND2\0"
_PROBE_V2_ENDIAN_MARKER = 0x0102030405060708


def _expand_probe_binary_files(paths):
    """Resolve literal paths and glob patterns for restart-segment inputs."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    expanded = []
    for value in paths:
        pattern = os.path.expanduser(str(value))
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(str(Path(path).resolve()) for path in matches)
        else:
            expanded.append(str(Path(pattern).resolve()))
    # Preserve segment order while avoiding repeated paths from overlapping
    # glob patterns.
    return list(dict.fromkeys(expanded))


def _probe_binary_version(path):
    with open(path, "rb") as stream:
        magic = stream.read(8)
    if magic == _PROBE_V2_FILE_MAGIC:
        return 2
    try:
        if struct.unpack("<q", magic)[0] == 0x005345424F525050:
            return 1
        if struct.unpack(">q", magic)[0] == 0x005345424F525050:
            return 1
    except struct.error:
        pass
    raise ValueError(f"Unrecognized probe binary format: {path}")


def _read_probe_v2_header(stream):
    """Read a chunked probe-v2 file header at the start of ``stream``."""
    stream.seek(0, os.SEEK_SET)
    if stream.read(8) != _PROBE_V2_FILE_MAGIC:
        raise ValueError("Not a chunked probe-v2 file")
    raw = stream.read(8 * 8)
    if len(raw) != 8 * 8:
        raise EOFError("Incomplete probe-v2 file header")
    values = struct.unpack("<8q", raw)
    if values[1] == _PROBE_V2_ENDIAN_MARKER:
        endian = "<"
    else:
        values = struct.unpack(">8q", raw)
        if values[1] != _PROBE_V2_ENDIAN_MARKER:
            raise ValueError("Probe-v2 endian marker is invalid")
        endian = ">"
    (version, _, n_probes, n_fields, chunk_capacity, probe_int,
     name_width, unit_width) = values
    if version != 2 or n_probes < 1 or n_fields < 1:
        raise ValueError("Invalid probe-v2 header values")

    def read_fixed_strings(count, width):
        result = []
        for _ in range(count):
            raw_value = stream.read(width)
            if len(raw_value) != width:
                raise EOFError("Incomplete probe-v2 string metadata")
            result.append(raw_value.split(b"\0", 1)[0].decode("ascii"))
        return result

    field_names = read_fixed_strings(n_fields, name_width)
    field_units = read_fixed_strings(n_fields, unit_width)
    float_dtype = np.dtype(f"{endian}f8")
    raw_x = stream.read(8 * n_probes)
    raw_y = stream.read(8 * n_probes)
    if len(raw_x) != 8 * n_probes or len(raw_y) != 8 * n_probes:
        raise EOFError("Incomplete probe-v2 coordinate metadata")
    requested_x = np.frombuffer(raw_x, dtype=float_dtype).astype(float)
    requested_y = np.frombuffer(raw_y, dtype=float_dtype).astype(float)
    return {
        "endian": endian,
        "n_probes": n_probes,
        "n_fields": n_fields,
        "chunk_capacity": chunk_capacity,
        "probe_int": probe_int,
        "field_names": field_names,
        "field_units": field_units,
        "requested_x": requested_x,
        "requested_y": requested_y,
        "data_offset": stream.tell(),
    }


def _scan_probe_v2_chunks(stream, header, path):
    """Return metadata for each complete chunk, ignoring a partial tail."""
    endian = header["endian"]
    chunks = []
    stream.seek(header["data_offset"], os.SEEK_SET)
    while True:
        chunk_offset = stream.tell()
        magic = stream.read(8)
        if not magic:
            break
        if len(magic) < 8:
            break
        if magic != _PROBE_V2_CHUNK_MAGIC:
            raise ValueError(
                f"Bad probe-v2 chunk marker in {path} at byte {chunk_offset}"
            )
        raw_header = stream.read(7 * 8 + 2 * 8)
        if len(raw_header) != 7 * 8 + 2 * 8:
            break
        (chunk_index, n_samples, n_probes, n_fields, payload_bytes,
         first_step, last_step, first_time, last_time) = struct.unpack(
            f"{endian}7q2d", raw_header
        )
        if (n_samples < 1 or n_probes != header["n_probes"]
                or n_fields != header["n_fields"] or payload_bytes < 0):
            raise ValueError(f"Invalid probe-v2 chunk header in {path}")
        payload_offset = stream.tell()
        footer_offset = payload_offset + payload_bytes
        stream.seek(footer_offset, os.SEEK_SET)
        raw_footer = stream.read(32)
        if len(raw_footer) != 32:
            break
        footer_magic, footer_index, checksum, footer_bytes = struct.unpack(
            f"{endian}8sqQq", raw_footer
        )
        if (footer_magic != _PROBE_V2_FOOTER_MAGIC
                or footer_index != chunk_index
                or footer_bytes != payload_bytes):
            raise ValueError(f"Invalid probe-v2 chunk footer in {path}")
        chunks.append({
            "index": chunk_index,
            "n_samples": n_samples,
            "payload_offset": payload_offset,
            "payload_bytes": payload_bytes,
            "first_step": first_step,
            "last_step": last_step,
            "first_time": first_time,
            "last_time": last_time,
            "checksum": checksum,
        })
        stream.seek(footer_offset + 32, os.SEEK_SET)
    return chunks


def _probe_mapping_digest(sample_x, sample_y, level, valid):
    """Return a stable digest for one selected probe-to-cell mapping."""
    digest = hashlib.blake2b(digest_size=16)
    for values in (sample_x, sample_y, level, valid):
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(struct.pack("<q", array.size))
        digest.update(array.tobytes())
    return digest.digest()


def _same_probe_mapping(first, second):
    """Return whether two mappings are exactly identical."""
    return (
        np.array_equal(first["sample_x"], second["sample_x"], equal_nan=True)
        and np.array_equal(
            first["sample_y"], second["sample_y"], equal_nan=True
        )
        and np.array_equal(first["level"], second["level"])
        and np.array_equal(first["valid"], second["valid"])
    )


def _mapping_epoch_runs(mapping_id, steps, times, mappings):
    """Compress a per-sample mapping identifier into contiguous epochs."""
    if mapping_id.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(mapping_id) != 0) + 1]
    stops = np.r_[starts[1:], mapping_id.size]
    epochs = []
    for epoch_index, (start, stop) in enumerate(zip(starts, stops)):
        registry_id = int(mapping_id[start])
        mapping = mappings[registry_id]
        epochs.append({
            "epoch": epoch_index,
            "mapping_id": registry_id,
            "start_index": int(start),
            "stop_index": int(stop),
            "sample_count": int(stop - start),
            "first_step": int(steps[start]),
            "last_step": int(steps[stop - 1]),
            "first_time": float(times[start]),
            "last_time": float(times[stop - 1]),
            "sample_x": mapping["sample_x"],
            "sample_y": mapping["sample_y"],
            "level": mapping["level"],
            "valid": mapping["valid"],
        })
    return epochs


def _probe_mapping_report(epochs, requested_x, requested_y):
    """Summarize coordinate motion and validity without expanding by time."""
    n_probes = requested_x.size
    maximum_shift = np.zeros(n_probes, dtype=float)
    invalid_samples = np.zeros(n_probes, dtype=np.int64)
    total_samples = sum(epoch["sample_count"] for epoch in epochs)
    levels = [set() for _ in range(n_probes)]
    for epoch in epochs:
        shift = np.hypot(
            epoch["sample_x"] - requested_x,
            epoch["sample_y"] - requested_y,
        )
        maximum_shift = np.maximum(maximum_shift, shift)
        invalid_samples += epoch["sample_count"] * (~epoch["valid"])
        for probe_id, level in enumerate(epoch["level"]):
            levels[probe_id].add(int(level))
    invalid_fraction = (
        invalid_samples.astype(float) / total_samples
        if total_samples else np.zeros(n_probes, dtype=float)
    )
    return {
        "epoch_count": len(epochs),
        "unique_mapping_count": len({
            epoch["mapping_id"] for epoch in epochs
        }),
        "transition_count": max(0, len(epochs) - 1),
        "sample_count": int(total_samples),
        "maximum_shift_cm": maximum_shift,
        "maximum_shift_overall_cm": (
            float(np.max(maximum_shift)) if n_probes else 0.0
        ),
        "invalid_fraction": invalid_fraction,
        "maximum_invalid_fraction": (
            float(np.max(invalid_fraction)) if n_probes else 0.0
        ),
        "sampled_levels": [tuple(sorted(values)) for values in levels],
    }


def _load_probe_data_from_chunked_binary(
        bin_files, var_col, dedup_tol=1e-12, nt_skip=0, max_probes=None,
        coordinate_policy="strict", overlap_policy="latest_segment"):
    """Load one field while preserving compact AMR mapping epochs.

    ``coordinate_policy='strict'`` rejects a mapping transition. ``nominal``
    retains the full record and uses the fixed requested position as the
    nominal spatial coordinate. ``longest_epoch`` selects the longest
    contiguous interval with a stationary coordinate/level/validity map.
    """
    coordinate_policy = str(coordinate_policy).lower()
    if coordinate_policy not in ("strict", "nominal", "longest_epoch"):
        raise ValueError(
            "coordinate_policy must be 'strict', 'nominal', or "
            "'longest_epoch'"
        )
    overlap_policy = str(overlap_policy).lower()
    if overlap_policy not in ("latest_segment", "error"):
        raise ValueError(
            "overlap_policy must be 'latest_segment' or 'error'"
        )
    requested_name = _fft_variable_meta(var_col)["field"]

    file_info = []
    reference = None
    for path in bin_files:
        stream = open(path, "rb")
        try:
            header = _read_probe_v2_header(stream)
            chunks = _scan_probe_v2_chunks(stream, header, path)
        except Exception:
            stream.close()
            raise
        signature = (
            header["n_probes"], tuple(header["field_names"]),
            tuple(header["field_units"]),
            tuple(header["requested_x"]), tuple(header["requested_y"]),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            stream.close()
            for item in file_info:
                item["stream"].close()
            raise ValueError(f"Probe-v2 segment layout mismatch in {path}")
        if requested_name not in header["field_names"]:
            stream.close()
            raise ValueError(f"Field {requested_name!r} is absent from {path}")
        file_info.append({
            "path": path,
            "stream": stream,
            "header": header,
            "chunks": chunks,
            "field_index": header["field_names"].index(requested_name),
        })

    if not file_info:
        return []
    n_probes = file_info[0]["header"]["n_probes"]
    n_load = n_probes if max_probes is None else min(int(max_probes), n_probes)
    times_parts, steps_parts, signal_parts, mapping_parts = [], [], [], []
    source_parts = []
    mappings = []
    mapping_ids_by_digest = {}

    try:
        for source_index, item in enumerate(file_info):
            stream = item["stream"]
            endian = item["header"]["endian"]
            for chunk in item["chunks"]:
                n_samples = chunk["n_samples"]
                payload = chunk["payload_offset"]
                steps_bytes = 8 * n_samples
                times_bytes = 8 * n_samples
                coordinates_bytes = 8 * n_probes
                levels_bytes = 4 * n_probes
                validity_bytes = n_probes

                stream.seek(payload, os.SEEK_SET)
                steps = np.frombuffer(
                    stream.read(steps_bytes), dtype=f"{endian}i8"
                ).astype(np.int64)
                times = np.frombuffer(
                    stream.read(times_bytes), dtype=f"{endian}f8"
                ).astype(float)
                chunk_x = np.frombuffer(
                    stream.read(coordinates_bytes), dtype=f"{endian}f8"
                ).astype(float)
                chunk_y = np.frombuffer(
                    stream.read(coordinates_bytes), dtype=f"{endian}f8"
                ).astype(float)
                level = np.frombuffer(
                    stream.read(levels_bytes), dtype=f"{endian}i4"
                ).astype(np.int32)
                valid = np.frombuffer(
                    stream.read(validity_bytes), dtype=np.uint8
                ).astype(bool)

                mapping = {
                    "sample_x": chunk_x[:n_load].copy(),
                    "sample_y": chunk_y[:n_load].copy(),
                    "level": level[:n_load].copy(),
                    "valid": valid[:n_load].copy(),
                }
                digest = _probe_mapping_digest(**mapping)
                mapping_id = None
                for candidate in mapping_ids_by_digest.get(digest, []):
                    if _same_probe_mapping(mapping, mappings[candidate]):
                        mapping_id = candidate
                        break
                if mapping_id is None:
                    mapping_id = len(mappings)
                    mappings.append(mapping)
                    mapping_ids_by_digest.setdefault(digest, []).append(
                        mapping_id
                    )

                field_base = (
                    payload + steps_bytes + times_bytes
                    + 2 * coordinates_bytes + levels_bytes + validity_bytes
                )
                field_bytes = 8 * n_samples * n_probes
                stream.seek(
                    field_base + item["field_index"] * field_bytes,
                    os.SEEK_SET,
                )
                raw_field = stream.read(field_bytes)
                if len(raw_field) != field_bytes:
                    raise EOFError(f"Short probe-v2 field block in {item['path']}")
                values = np.frombuffer(
                    raw_field, dtype=f"{endian}f8"
                ).reshape(n_samples, n_probes)[:, :n_load].astype(float)
                if not np.all(valid[:n_load]):
                    values[:, ~valid[:n_load]] = np.nan
                steps_parts.append(steps)
                times_parts.append(times)
                signal_parts.append(values)
                mapping_parts.append(np.full(
                    n_samples, mapping_id, dtype=np.int32
                ))
                source_parts.append(np.full(
                    n_samples, source_index, dtype=np.int32
                ))
    finally:
        for item in file_info:
            item["stream"].close()

    if not times_parts:
        return []
    steps = np.concatenate(steps_parts)
    time_buf = np.concatenate(times_parts)
    signal_buf = np.vstack(signal_parts)
    mapping_id = np.concatenate(mapping_parts)
    source_id = np.concatenate(source_parts)
    order = np.lexsort((source_id, steps, time_buf))
    steps = steps[order]
    time_buf = time_buf[order]
    signal_buf = signal_buf[order, :]
    mapping_id = mapping_id[order]
    source_id = source_id[order]

    overlap_report = {
        "policy": overlap_policy,
        "field": requested_name,
        "field_unit": file_info[0]["header"]["field_units"][
            file_info[0]["field_index"]
        ],
        "duplicate_sample_count": 0,
        "conflicting_sample_count": 0,
        "conflicting_mapping_count": 0,
        "replaced_by_later_segment_count": 0,
        "maximum_absolute_signal_difference": 0.0,
        "conflict_examples": [],
    }
    if dedup_tol >= 0 and time_buf.size:
        keep = [0]
        for idx in range(1, time_buf.size):
            previous = keep[-1]
            duplicate = (
                steps[idx] == steps[previous]
                or abs(time_buf[idx] - time_buf[previous]) < dedup_tol
            )
            if duplicate:
                same_mapping = mapping_id[idx] == mapping_id[previous]
                same_signal = np.allclose(
                        signal_buf[idx], signal_buf[previous],
                        rtol=1.0e-12, atol=0.0, equal_nan=True)
                same_source = source_id[idx] == source_id[previous]
                if same_source and (not same_mapping or not same_signal):
                    raise ValueError(
                        "Conflicting duplicate probe samples were found "
                        "within one segment at "
                        f"step={steps[idx]}, time={time_buf[idx]:.17g}, "
                        f"segment={file_info[int(source_id[idx])]['path']}"
                    )
                overlap_report["duplicate_sample_count"] += 1
                if not same_mapping:
                    overlap_report["conflicting_mapping_count"] += 1
                sample_difference = 0.0
                if not same_signal:
                    overlap_report["conflicting_sample_count"] += 1
                    difference = np.abs(
                        signal_buf[idx] - signal_buf[previous]
                    )
                    finite = difference[np.isfinite(difference)]
                    if finite.size:
                        sample_difference = float(np.max(finite))
                        overlap_report[
                            "maximum_absolute_signal_difference"
                        ] = max(
                            overlap_report[
                                "maximum_absolute_signal_difference"
                            ],
                            sample_difference,
                        )
                if ((not same_mapping or not same_signal)
                        and not same_source
                        and len(overlap_report["conflict_examples"]) < 10):
                    earlier_source = min(
                        int(source_id[previous]), int(source_id[idx])
                    )
                    later_source = max(
                        int(source_id[previous]), int(source_id[idx])
                    )
                    overlap_report["conflict_examples"].append({
                        "step": int(steps[idx]),
                        "time": float(time_buf[idx]),
                        "earlier_segment": file_info[earlier_source]["path"],
                        "later_segment": file_info[later_source]["path"],
                        "mapping_differs": bool(not same_mapping),
                        "maximum_absolute_signal_difference": (
                            sample_difference
                        ),
                    })
                if (overlap_policy == "error"
                        and (not same_mapping or not same_signal)):
                    raise ValueError(
                        "Conflicting duplicate probe samples were found "
                        "across restart segments at "
                        f"step={steps[idx]}, time={time_buf[idx]:.17g}"
                    )
                if (overlap_policy == "latest_segment"
                        and source_id[idx] > source_id[previous]):
                    keep[-1] = idx
                    overlap_report[
                        "replaced_by_later_segment_count"
                    ] += 1
                continue
            keep.append(idx)
        keep = np.asarray(keep, dtype=int)
        steps = steps[keep]
        time_buf = time_buf[keep]
        signal_buf = signal_buf[keep, :]
        mapping_id = mapping_id[keep]
        source_id = source_id[keep]

    if overlap_report["duplicate_sample_count"]:
        _ts(
            "  [W] Resolved "
            f"{overlap_report['duplicate_sample_count']} restart-overlap "
            "sample(s); "
            f"{overlap_report['conflicting_sample_count']} had differing "
            f"{requested_name} values. Policy={overlap_policy!r}; maximum "
            "absolute difference="
            f"{overlap_report['maximum_absolute_signal_difference']:.6g} "
            f"{overlap_report['field_unit']}"
        )

    if nt_skip > 0:
        steps = steps[nt_skip:]
        time_buf = time_buf[nt_skip:]
        signal_buf = signal_buf[nt_skip:, :]
        mapping_id = mapping_id[nt_skip:]
        source_id = source_id[nt_skip:]

    all_epochs = _mapping_epoch_runs(mapping_id, steps, time_buf, mappings)
    analysis_epochs = all_epochs
    if coordinate_policy == "strict" and len(all_epochs) > 1:
        raise ValueError(
            f"Probe sampling map changed {len(all_epochs) - 1} time(s); use "
            "probe_coordinate_policy='nominal' to retain the mapping history "
            "or 'longest_epoch' for a stationary interval"
        )
    if coordinate_policy == "longest_epoch" and len(all_epochs) > 1:
        selected_epoch = max(
            all_epochs, key=lambda epoch: epoch["sample_count"]
        )
        selected = slice(
            selected_epoch["start_index"], selected_epoch["stop_index"]
        )
        steps = steps[selected]
        time_buf = time_buf[selected]
        signal_buf = signal_buf[selected, :]
        mapping_id = mapping_id[selected]
        analysis_epochs = _mapping_epoch_runs(
            mapping_id, steps, time_buf, mappings
        )

    if time_buf.size == 0:
        return []

    header = file_info[0]["header"]
    requested_x = header["requested_x"][:n_load]
    requested_y = header["requested_y"][:n_load]
    report = _probe_mapping_report(all_epochs, requested_x, requested_y)
    report["restart_overlap"] = overlap_report
    report["analysis_epoch_count"] = len(analysis_epochs)
    report["analysis_sample_count"] = int(time_buf.size)
    if coordinate_policy == "nominal" and report["transition_count"]:
        _ts(
            "  [W] Retaining probe samples across "
            f"{report['epoch_count']} AMR mapping epochs; maximum sampled-"
            f"location shift is {report['maximum_shift_overall_cm']:.6g} cm"
        )
    stationary_mapping = mappings[int(mapping_id[0])]
    if coordinate_policy == "nominal":
        x_output, y_output = requested_x, requested_y
    else:
        x_output = stationary_mapping["sample_x"]
        y_output = stationary_mapping["sample_y"]
    probe_data = []
    for probe_id in range(n_load):
        probe_data.append({
            "x": float(x_output[probe_id]),
            "y": float(y_output[probe_id]),
            "x_req": float(requested_x[probe_id]),
            "y_req": float(requested_y[probe_id]),
            "mapping_max_shift_cm": float(
                report["maximum_shift_cm"][probe_id]
            ),
            "invalid_fraction": float(report["invalid_fraction"][probe_id]),
            "sampled_levels": report["sampled_levels"][probe_id],
            "time": time_buf,
            "step": steps,
            "signal": signal_buf[:, probe_id],
            "dt": float(time_buf[1] - time_buf[0]) if time_buf.size >= 2 else 0.0,
            "filename": f"chunked_probe_{probe_id:04d}",
        })
    if probe_data:
        probe_data[0]["_shared_signal_matrix"] = signal_buf
        probe_data[0]["_shared_uniform_time"] = True
        probe_data[0]["_mapping_epoch_id"] = mapping_id
        probe_data[0]["_mapping_epochs"] = all_epochs
        probe_data[0]["_analysis_mapping_epochs"] = analysis_epochs
        probe_data[0]["_mapping_report"] = report
        probe_data[0]["_coordinate_policy"] = coordinate_policy
        probe_data[0]["_restart_overlap_report"] = overlap_report
    return probe_data


def _write_probe_mapping_products(probe_data, output_root):
    """Persist compact AMR mapping provenance and a readable quality report."""
    # Compact HDF5 stores already contain the complete mapping epochs. Avoid
    # materializing probe zero merely to rediscover that provenance.
    if isinstance(probe_data, HDF5ProbeStore):
        return None
    if not probe_data or "_mapping_report" not in probe_data[0]:
        return None
    first = probe_data[0]
    report = first["_mapping_report"]
    epochs = first["_mapping_epochs"]
    mapping_dir = Path(output_root) / "ProbeMapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    maximum_shift = np.asarray(report["maximum_shift_cm"], dtype=float)
    invalid_fraction = np.asarray(report["invalid_fraction"], dtype=float)
    summary = {
        "coordinate_policy": first["_coordinate_policy"],
        "probe_count": len(probe_data),
        "sample_count": report["sample_count"],
        "analysis_sample_count": report["analysis_sample_count"],
        "epoch_count": report["epoch_count"],
        "analysis_epoch_count": report["analysis_epoch_count"],
        "unique_mapping_count": report["unique_mapping_count"],
        "transition_count": report["transition_count"],
        "maximum_shift_overall_cm": report["maximum_shift_overall_cm"],
        "shift_percentiles_cm": {
            str(percentile): float(np.percentile(maximum_shift, percentile))
            for percentile in (50, 90, 95, 99, 100)
        },
        "maximum_invalid_fraction": report["maximum_invalid_fraction"],
        "restart_overlap": report.get("restart_overlap", {}),
        "probes_with_invalid_samples": int(np.count_nonzero(
            invalid_fraction > 0.0
        )),
        "epoch_intervals": [{
            key: epoch[key] for key in (
                "epoch", "mapping_id", "start_index", "stop_index",
                "sample_count", "first_step", "last_step", "first_time",
                "last_time",
            )
        } for epoch in epochs],
    }
    summary_path = mapping_dir / "mapping_report.json"
    temporary_summary = mapping_dir / ".mapping_report.json.tmp"
    with temporary_summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_summary, summary_path)

    unique = {}
    for epoch in epochs:
        unique.setdefault(epoch["mapping_id"], epoch)
    mapping_ids = sorted(unique)
    mapping_row = {mapping_id: row for row, mapping_id in enumerate(mapping_ids)}
    detail_path = mapping_dir / "mapping_epochs.npz"
    temporary_detail = mapping_dir / ".mapping_epochs.npz.tmp"
    with temporary_detail.open("wb") as stream:
        np.savez_compressed(
            stream,
            requested_x_cm=np.asarray([item["x_req"] for item in probe_data]),
            requested_y_cm=np.asarray([item["y_req"] for item in probe_data]),
            maximum_shift_cm=maximum_shift,
            invalid_fraction=invalid_fraction,
            epoch_start=np.asarray([
                epoch["start_index"] for epoch in epochs
            ], dtype=np.int64),
            epoch_stop=np.asarray([
                epoch["stop_index"] for epoch in epochs
            ], dtype=np.int64),
            epoch_mapping_row=np.asarray([
                mapping_row[epoch["mapping_id"]] for epoch in epochs
            ], dtype=np.int32),
            mapping_id=np.asarray(mapping_ids, dtype=np.int32),
            sample_x_cm=np.vstack([
                unique[mapping_id]["sample_x"] for mapping_id in mapping_ids
            ]),
            sample_y_cm=np.vstack([
                unique[mapping_id]["sample_y"] for mapping_id in mapping_ids
            ]),
            level=np.vstack([
                unique[mapping_id]["level"] for mapping_id in mapping_ids
            ]),
            valid=np.vstack([
                unique[mapping_id]["valid"] for mapping_id in mapping_ids
            ]),
        )
    os.replace(temporary_detail, detail_path)
    return summary_path, detail_path


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

    bin_files = _expand_probe_binary_files(bin_files)
    versions = {_probe_binary_version(path) for path in bin_files}
    if versions == {2}:
        return _load_probe_data_from_chunked_binary(
            bin_files, var_col,
            dedup_tol=config.get("probe_dedup_tol", 1e-12),
            nt_skip=nt_skip, max_probes=max_probes,
            coordinate_policy=config.get(
                "probe_coordinate_policy", "strict"
            ),
            overlap_policy=config.get(
                "probe_overlap_policy", "latest_segment"
            ),
        )
    if versions != {1}:
        raise ValueError("Legacy and chunked probe files cannot be mixed")
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
    compact_file = config.get("probe_compact_file")
    if compact_file:
        if not HDF5_PROBE_STORE_AVAILABLE:
            raise ImportError(
                "Compact HDF5 probe input requires the optional h5py package"
            )
        compact_path = Path(compact_file).expanduser()
        if not compact_path.is_absolute():
            compact_path = Path(config.get("data_source", ".")) / compact_path
        if not compact_path.is_file():
            raise FileNotFoundError(
                f"Configured compact probe archive is missing: {compact_path}"
            )
        field = _fft_variable_meta(var_col)["field"]
        return HDF5ProbeStore(
            compact_path, field=field, nt_skip=nt_skip,
            max_probes=max_probes,
        )

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

    if isinstance(probe_data, HDF5ProbeStore):
        native_time = probe_data.time
        if native_time.size < 2:
            raise ValueError("At least two probe samples are required")
        native_dt = float(np.median(np.diff(native_time)))
        uniform = np.allclose(
            np.diff(native_time), native_dt, rtol=1.0e-8,
            atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
        )
        dt_use = float(target_dt) if target_dt is not None else native_dt
        if not resample or (
                uniform and np.isclose(
                    dt_use, native_dt, rtol=1.0e-8,
                    atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
                )):
            return native_time, probe_data.field_view(), native_dt, False
        count = int(np.floor(
            (native_time[-1] - native_time[0]) / dt_use + 1.0e-9
        )) + 1
        time_uniform = native_time[0] + np.arange(count) * dt_use
        matrix = np.empty((count, len(probe_data)), dtype=float)
        batch_size = 32
        for first, last, values in probe_data.iter_probe_batches(batch_size):
            for column in range(last - first):
                matrix[:, first + column] = np.interp(
                    time_uniform, native_time, values[:, column]
                )
        return time_uniform, matrix, dt_use, False

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
        var_col = int(config.get("fft_var_col", 3))
        var_meta = _fft_variable_meta(var_col)
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

        compact_store = isinstance(probe_data, HDF5ProbeStore)
        if compact_store:
            probe_x = probe_data.x.tolist()
            probe_y = probe_data.y.tolist()
        else:
            probe_x = [d["x"] for d in probe_data]
            probe_y = [d["y"] for d in probe_data]
        n_valid = len(probe_data)
        _ts(f"  Successfully loaded {n_valid} probes")

        fft_batch_size = max(1, int(config.get("fft_batch_size", 32)))
        stream_workspace = None
        if compact_store:
            native_time = probe_data.time
            native_dt = float(np.median(np.diff(native_time)))
            uniform = np.allclose(
                np.diff(native_time), native_dt, rtol=1.0e-8,
                atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
            )
            dt_actual = (
                float(target_dt) if target_dt is not None else native_dt
            )
            needs_resampling = bool(
                resample and (
                    not uniform or not np.isclose(
                        dt_actual, native_dt, rtol=1.0e-8,
                        atol=max(abs(native_dt) * 1.0e-10, 1.0e-15),
                    )
                )
            )
            if needs_resampling:
                count = int(np.floor(
                    (native_time[-1] - native_time[0]) / dt_actual + 1.0e-9
                )) + 1
                time_uniform = native_time[0] + np.arange(count) * dt_actual
            else:
                time_uniform = native_time
                dt_actual = native_dt
            reused_shared = False
        else:
            time_uniform, raw_matrix, dt_actual, reused_shared = \
                _probe_matrix_on_common_time(
                    probe_data, resample=resample, target_dt=target_dt
                )

        # A probe that is absent from the finest AMR level is explicitly
        # stored as NaN by the chunked probe writer.  Fourier/coherence/modal
        # calculations require complete records, so retain only columns that
        # are finite over the selected time interval.  Keep the source index
        # mapping so plots and reusable products still identify the original
        # diagonal probe numbers.
        source_probe_indices = (
            probe_data.probe_ids.copy() if compact_store
            else np.arange(n_valid, dtype=int)
        )
        complete = (
            probe_data.complete_probe_mask(fft_batch_size)
            if compact_store else np.all(np.isfinite(raw_matrix), axis=0)
        )
        if not np.any(complete):
            raise ValueError(
                "No probe has a complete finite record for FFT analysis"
            )
        if not np.all(complete):
            dropped = int(np.count_nonzero(~complete))
            source_probe_indices = source_probe_indices[complete]
            if compact_store:
                probe_data = probe_data.select_probes(np.flatnonzero(complete))
                probe_x = probe_data.x.tolist()
                probe_y = probe_data.y.tolist()
            else:
                raw_matrix = raw_matrix[:, complete]
                probe_data = [
                    item for item, keep in zip(probe_data, complete) if keep
                ]
                probe_x = [item["x"] for item in probe_data]
                probe_y = [item["y"] for item in probe_data]
            n_valid = len(probe_data)
            reused_shared = False
            _ts(
                f"  [W] Excluded {dropped} incomplete AMR probes; "
                f"retaining {n_valid} complete probes "
                f"(source indices {source_probe_indices.tolist()})"
            )
        L = len(time_uniform)
        Fs = 1.0 / dt_actual
        freq = Fs * np.arange(0, L // 2 + 1) / L
        n_freq = len(freq)

        if compact_store:
            # Keep full compatibility matrices disk-backed while only one
            # probe batch is resident in RAM. Later reconstruction and plot
            # code can continue indexing columns without materializing the
            # complete archive.
            stream_workspace = tempfile.TemporaryDirectory(
                prefix="fft-probe-work-", dir=output_dir
            )
            work_root = Path(stream_workspace.name)
            signal_matrix = np.memmap(
                work_root / "preprocessed.f64", mode="w+", dtype=np.float64,
                shape=(L, n_valid),
            )
            P1 = np.memmap(
                work_root / "amplitude.f64", mode="w+", dtype=np.float64,
                shape=(n_freq, n_valid),
            )
            if needs_resampling:
                raw_matrix = np.memmap(
                    work_root / "resampled.f64", mode="w+", dtype=np.float64,
                    shape=(L, n_valid),
                )
            else:
                raw_matrix = probe_data.field_view()
            win_scale = 1.0
            for first, last, native_batch in probe_data.iter_probe_batches(
                    fft_batch_size):
                if needs_resampling:
                    raw_batch = np.empty((L, last - first), dtype=float)
                    for column in range(last - first):
                        raw_batch[:, column] = np.interp(
                            time_uniform, native_time, native_batch[:, column]
                        )
                    raw_matrix[:, first:last] = raw_batch
                else:
                    raw_batch = native_batch
                work = raw_batch.copy()
                work, win_scale, _ = _preprocess_probe_signal(
                    work, config, time_uniform=time_uniform, copy_raw=False,
                    log_details=(first == 0),
                )
                signal_matrix[:, first:last] = work
                Y = np.fft.rfft(work, axis=0)
                amplitudes = np.abs(Y / L) * win_scale
                amplitudes[1:-1, :] *= 2.0
                P1[:, first:last] = amplitudes
            signal_matrix.flush()
            P1.flush()
            if needs_resampling:
                raw_matrix.flush()
            _ts(
                "  Streamed compact HDF5 in probe batches; full working "
                "matrices are disk-backed"
            )
        else:
            # Preserve the unprocessed record for time traces/reconstruction
            # and create only the one working copy required by the FFT.
            signal_matrix = raw_matrix.copy()
            if reused_shared:
                _ts("  Reusing contiguous binary probe matrix (no interpolation copy)")
            signal_matrix, win_scale, _ = _preprocess_probe_signal(
                signal_matrix, config, time_uniform=time_uniform,
                copy_raw=False,
            )
            P1 = np.zeros((n_freq, n_valid))
            for first in range(0, n_valid, fft_batch_size):
                last = min(first + fft_batch_size, n_valid)
                Y = np.fft.rfft(signal_matrix[:, first:last], axis=0)
                P1[:, first:last] = np.abs(Y / L) * win_scale
                P1[1:-1, first:last] *= 2.0
        _ts(f"  FFT complete — {n_freq} bins x {n_valid} probes")

        source_response_result = None
        if config.get("make_source_response_analysis", False):
            source_cfg = config.get("source_response", {})
            model = str(source_cfg.get("model", "single_gaussian")).lower()
            if model != "single_gaussian":
                raise ValueError(
                    "source_response.model currently supports only "
                    "'single_gaussian'"
                )
            source_result = fdb.compute_single_pulse_source_spectrum(
                time_uniform,
                energy_per_pulse=source_cfg["energy_per_pulse"],
                pulse_fwhm_s=source_cfg["pulse_fwhm_s"],
                pulse_period_s=source_cfg["pulse_period_s"],
                start_time_s=source_cfg.get("start_time_s", 0.0),
                cutoff_sigma=source_cfg.get("cutoff_sigma", 4.0),
                mean_subtraction=source_cfg.get(
                    "transfer_source_mean_subtraction", "none"
                ),
                window=source_cfg.get("transfer_window", "none"),
                window_compensation=True,
            )
            if not np.allclose(source_result["frequency_hz"], freq):
                raise RuntimeError(
                    "Source and probe FFT frequency grids do not match"
                )
            baseline_end = source_cfg.get("response_baseline_end_time_s")
            if baseline_end is None:
                baseline_end = (
                    source_result["center_s"]
                    - float(source_cfg.get("cutoff_sigma", 4.0))
                    * source_result["sigma_s"]
                )
            baseline_mask = time_uniform < float(baseline_end)
            minimum_baseline_samples = int(source_cfg.get(
                "minimum_baseline_samples", 8
            ))
            if np.count_nonzero(baseline_mask) < minimum_baseline_samples:
                raise ValueError(
                    "Single-pulse response baseline contains fewer than "
                    f"{minimum_baseline_samples} quiescent samples"
                )
            transfer_window, transfer_gain = fdb._fft_window(
                L, source_cfg.get("transfer_window", "none")
            )
            response_complex = np.empty((n_freq, n_valid), dtype=complex)
            response_baseline = np.empty(n_valid, dtype=float)
            for first in range(0, n_valid, fft_batch_size):
                last = min(first + fft_batch_size, n_valid)
                raw_batch = np.asarray(raw_matrix[:, first:last], dtype=float)
                if not np.all(np.isfinite(raw_batch)):
                    raise ValueError(
                        "Probe response contains NaN or infinite values"
                    )
                batch_baseline = np.mean(
                    raw_batch[baseline_mask, :], axis=0
                )
                response_baseline[first:last] = batch_baseline
                response_work = raw_batch - batch_baseline[None, :]
                response_work *= transfer_window[:, None]
                response_complex[:, first:last] = (
                    np.fft.rfft(response_work, axis=0)
                    * transfer_gain / L
                )
            transfer_result = fdb.compute_single_pulse_transfer_function(
                source_result["processed_complex"], response_complex,
                minimum_relative_source_amplitude=source_cfg.get(
                    "minimum_relative_source_amplitude", 1.0e-3
                ),
            )
            response_amplitude = np.abs(response_complex)
            if L % 2 == 0 and response_amplitude.shape[0] > 2:
                response_amplitude[1:-1, :] *= 2.0
            elif L % 2 == 1 and response_amplitude.shape[0] > 1:
                response_amplitude[1:, :] *= 2.0
            source_response_result = {
                "source": source_result,
                "transfer": transfer_result,
                "config": source_cfg,
                "response_baseline": response_baseline,
                "response_amplitude": response_amplitude,
                "baseline_start_time_s": float(time_uniform[baseline_mask][0]),
                "baseline_end_time_s": float(time_uniform[baseline_mask][-1]),
                "baseline_sample_count": int(np.count_nonzero(baseline_mask)),
            }
            _ts(
                "  Computed single-pulse source spectrum and quiescent-"
                "baseline finite-record deconvolution"
            )

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
        ylabel_sig = f"{var_meta['name']} [{var_meta['unit_cgs']}]"

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
        var_slug = var_meta["slug"]
        np.savez_compressed(
            output_dir / f"spectral_summary_{var_slug}.npz",
            frequency_hz=freq,
            probe_x_cm=np.asarray(probe_x, dtype=float),
            probe_y_cm=np.asarray(probe_y, dtype=float),
            source_probe_indices=source_probe_indices,
            mean_amplitude=np.nanmean(P1, axis=1),
            selected_probe_indices=np.asarray(export_indices, dtype=int),
            selected_source_probe_indices=(
                source_probe_indices[export_indices]
                if export_indices else np.empty(0, dtype=int)
            ),
            selected_amplitude=selected_spectra,
            growth_frequency_hz=growth_actual,
            growth_amplitude=P1[growth_bins, :] if growth_bins.size
            else np.empty((0, n_valid), dtype=float),
            dominant_frequency_hz=np.array(harmonic_freq),
            sample_interval_s=np.array(dt_actual),
            window_amplitude_scale=np.array(win_scale),
            signal_name=np.array(var_meta["name"]),
            signal_unit_cgs=np.array(var_meta["unit_cgs"]),
        )
        _ts(f"  Saved reusable spectral data: "
            f"{output_dir / f'spectral_summary_{var_slug}.npz'}")

        if source_response_result is not None:
            source_result = source_response_result["source"]
            transfer_result = source_response_result["transfer"]
            source_cfg = source_response_result["config"]
            requested = source_cfg.get("plot_probe_indices", export_indices)
            transfer_indices = sorted(set(
                max(0, min(int(index), n_valid - 1)) for index in requested
            ))
            transfer_labels = [
                f"Probe {source_probe_indices[index]} "
                f"(x={probe_x[index]:.4f} cm)"
                for index in transfer_indices
            ]
            np.savez_compressed(
                output_dir / f"source_response_spectrum_{var_slug}.npz",
                time_s=source_result["time_s"],
                source_power=source_result["power"],
                source_processed_power=source_result["processed_power"],
                frequency_hz=freq,
                source_physical_spectrum=source_result["physical_spectrum"],
                source_ideal_spectrum=source_result["ideal_spectrum"],
                source_processed_amplitude=source_result["processed_amplitude"],
                response_processed_amplitude=source_response_result[
                    "response_amplitude"
                ],
                source_sigma_s=np.array(source_result["sigma_s"]),
                source_center_s=np.array(source_result["center_s"]),
                source_energy_per_pulse=np.array(
                    source_cfg["energy_per_pulse"]
                ),
                source_energy_unit=np.array(
                    source_cfg.get("energy_unit", "energy/depth")
                ),
                source_spatial_shape_label=np.array(
                    source_cfg.get("spatial_shape_label", "unspecified")
                ),
                transfer_complex=transfer_result["transfer"],
                transfer_magnitude=transfer_result["magnitude"],
                transfer_phase_rad=transfer_result["phase_rad"],
                valid_transfer_frequency=transfer_result["valid_frequency"],
                transfer_probe_indices=np.asarray(transfer_indices, dtype=int),
                transfer_source_probe_indices=source_probe_indices[
                    transfer_indices
                ],
                transfer_probe_x_cm=np.asarray(probe_x)[transfer_indices],
                transfer_minimum_source_amplitude=np.array(
                    transfer_result["minimum_source_amplitude"]
                ),
                response_quiescent_baseline=source_response_result[
                    "response_baseline"
                ],
                response_baseline_start_time_s=np.array(
                    source_response_result["baseline_start_time_s"]
                ),
                response_baseline_end_time_s=np.array(
                    source_response_result["baseline_end_time_s"]
                ),
                response_baseline_sample_count=np.array(
                    source_response_result["baseline_sample_count"]
                ),
                transfer_estimator=np.array(
                    "single-pulse finite-record deconvolution"
                ),
                transfer_window=np.array(
                    source_cfg.get("transfer_window", "none")
                ),
                signal_name=np.array(var_meta["name"]),
                signal_unit_cgs=np.array(var_meta["unit_cgs"]),
            )
            source_fmax = source_cfg.get("plot_fmax_hz")
            pdb.plot_single_pulse_source_spectrum(
                source_result,
                output_path=str(
                    output_dir / f"single_pulse_source_spectrum_{var_slug}.png"
                ),
                fmax=source_fmax,
                energy_unit=source_cfg.get("energy_unit", "energy/depth"),
            )
            pdb.plot_normalized_source_response_spectra(
                freq, source_result["processed_amplitude"],
                source_response_result["response_amplitude"][:, transfer_indices],
                transfer_labels,
                output_path=str(
                    output_dir
                    / f"normalized_source_response_spectra_{var_slug}.png"
                ),
                fmax=source_fmax,
            )
            pdb.plot_single_pulse_transfer_functions(
                freq, transfer_result["magnitude"][:, transfer_indices],
                transfer_result["phase_rad"][:, transfer_indices],
                transfer_result["valid_frequency"], transfer_labels,
                output_path=str(
                    output_dir
                    / f"single_pulse_transfer_function_{var_slug}.png"
                ),
                fmax=source_fmax,
            )
            _ts(
                "  Saved source spectrum and source-to-response transfer "
                "products"
            )

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
                        f"{source_probe_indices[first]}->"
                        f"{source_probe_indices[second]} "
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
                    output_dir / f"pair_coherence_{var_slug}.npz",
                    frequency_hz=pair_results[0]["result"]["frequency_hz"],
                    probe_pairs=np.asarray([
                        item["indices"] for item in pair_results
                    ], dtype=int),
                    coherence_squared=coherence_values,
                    cross_phase_rad=phase_values,
                    nperseg=np.array(pair_results[0]["result"]["nperseg"]),
                    noverlap=np.array(pair_results[0]["result"]["noverlap"]),
                    signal_name=np.array(var_meta["name"]),
                    signal_unit_cgs=np.array(var_meta["unit_cgs"]),
                )
                pdb.plot_pair_coherence(
                    pair_results,
                    output_path=str(
                        output_dir / f"pair_coherence_{var_slug}.png"
                    ),
                    fmax=config.get("coherence_fmax"),
                )
                _ts(f"  Saved Welch coherence for {len(pair_results)} probe pairs")

        if config.get("make_modal_analysis", False):
            modal_indices_cfg = config.get("modal_probe_indices")
            modal_indices = (
                list(range(
                    0, n_valid, int(config.get("modal_probe_stride", 1))
                ))
                if modal_indices_cfg is None
                else sorted(set(int(value) for value in modal_indices_cfg))
            )
            if len(modal_indices) < 2:
                raise ValueError("Modal analysis requires at least two probe stations")
            if modal_indices[0] < 0 or modal_indices[-1] >= n_valid:
                raise ValueError("modal_probe_indices contains an invalid index")
            coordinates_m = 0.01 * np.column_stack((
                np.asarray(probe_x)[modal_indices],
                np.asarray(probe_y)[modal_indices],
            ))
            # All Fourier stages use the median dt when resampling is disabled.
            # Give modal algorithms that explicit uniform coordinate rather
            # than pretending the ppm-level native clock jitter is exact.
            modal_time = time_uniform[0] + np.arange(L, dtype=float) * dt_actual
            modal_variable = var_meta["modal_weight"]
            modal_weight_info = {
                "weights": mdb.trapezoidal_spatial_weights(coordinates_m),
                "quadrature_weights": mdb.trapezoidal_spatial_weights(
                    coordinates_m
                ),
                "norm_definition": "spatial L2 quadrature",
                "scope": "single measured scalar",
            }
            if config.get("modal_use_compressible_energy_weights", True):
                modal_weight_info = mdb.scalar_compressible_energy_weights(
                    coordinates_m, modal_variable,
                    rho_base=config["modal_base_rho_kg_m3"],
                    temperature_base=config["modal_base_temperature_k"],
                    gamma=config["modal_gamma"],
                    gas_constant=config["modal_gas_constant_j_kg_k"],
                )
            snapshots = mdb.SnapshotMatrix(
                modal_time,
                raw_matrix[:, modal_indices],
                coordinates_m,
                variable=modal_variable,
                weights=modal_weight_info["weights"],
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
            modal_sensitivity = mdb.compute_modal_sensitivity(
                snapshots,
                pod_modes=min(2, modal_count),
                dmd_ranks=config.get(
                    "modal_sensitivity_dmd_ranks", [2, 4, 8]
                ),
                window_fractions=config.get(
                    "modal_sensitivity_windows",
                    [[0.0, 0.5], [0.5, 1.0]],
                ),
            )
            modal_dir = output_dir / "ModalAnalysis"
            modal_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                modal_dir / f"modal_results_{var_slug}.npz",
                probe_indices=np.asarray(modal_indices, dtype=int),
                source_probe_indices=source_probe_indices[modal_indices],
                coordinates_m=coordinates_m,
                spatial_quadrature_weights=modal_weight_info[
                    "quadrature_weights"
                ],
                modal_energy_weights=modal_weight_info["weights"],
                modal_norm_definition=np.array(
                    modal_weight_info["norm_definition"]
                ),
                modal_norm_scope=np.array(modal_weight_info["scope"]),
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
                dmd_retained_singular_values=dmd_result[
                    "retained_singular_values"
                ],
                dmd_retained_condition_number=np.array(
                    dmd_result["retained_condition_number"]
                ),
                **{
                    f"sensitivity_{key}": value
                    for key, value in modal_sensitivity.items()
                    if not isinstance(value, str)
                },
                sensitivity_interpretation=np.array(
                    modal_sensitivity["interpretation"]
                ),
            )
            pdb.plot_modal_summary(
                pod_result, spod_result, dmd_result,
                output_path=str(modal_dir / f"modal_summary_{var_slug}.png"),
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
                source_idx = int(source_probe_indices[idx])
                axes[row, 0].set_title(f"Probe {source_idx} — x={p['x']:.3f} cm — Raw", fontsize=11)
                axes[row, 0].grid(False)
                # Processed time-trace
                axes[row, 1].plot(plot_time, proc_sig, lw=0.8, color="C1")
                axes[row, 1].set_xlabel("Time [s]", fontsize=10)
                axes[row, 1].set_ylabel(ylabel_sig, fontsize=10)
                axes[row, 1].set_title(f"Probe {source_idx} — x={p['x']:.3f} cm — Processed (mean-sub, Hann)", fontsize=11)
                axes[row, 1].grid(False)
                # FFT spectrum
                axes[row, 2].loglog(freq, P1[:, idx], lw=0.8)
                axes[row, 2].set_xlabel("Frequency [Hz]", fontsize=10)
                axes[row, 2].set_ylabel("|Amplitude|", fontsize=10)
                axes[row, 2].grid(False)
            fig.tight_layout()
            out = output_dir / f"{probe_prefix}_{var_slug}_fft_probes.png"
            fig.savefig(str(out), dpi=150)
            plt.close(fig)
            _ts(f"  Saved {len(pos_indices)} probe FFT plots: {out}")

        # Freq-vs-position contour
        if plot_contour_enabled and n_valid > 1:
            try:
                idx_sorted = np.argsort(probe_x)
                max_probe_points = max(2, int(config.get(
                    "fft_contour_max_probe_points", 1000
                )))
                probe_stride = max(1, int(np.ceil(n_valid / max_probe_points)))
                contour_probe_ids = idx_sorted[::probe_stride]
                probe_x_sorted = np.array(probe_x)[contour_probe_ids]
                max_frequency_points = max(2, int(config.get(
                    "fft_contour_max_frequency_points", 4096
                )))
                frequency_stride = max(
                    1, int(np.ceil(max(1, n_freq - 1) / max_frequency_points))
                )
                contour_frequency_ids = np.arange(
                    1, n_freq, frequency_stride, dtype=int
                )
                # np.ix_ materializes only the bounded plot raster from the
                # disk-backed full spectrum.
                amplitude_plot = np.asarray(P1[np.ix_(
                    contour_frequency_ids, contour_probe_ids
                )])
                eps = 1e-20
                contour_scale = config.get("fft_contour_scale", "linear").lower()
                if contour_scale == "linear":
                    # Use the physical nonnegative FFT amplitude directly.
                    # This gives the requested color range [0, max].
                    P1_plot_sub = amplitude_plot
                    cbar_label = "Amplitude"
                    plot_vmin = 0.0
                    plot_vmax = config.get("fft_contour_vmax")
                else:
                    if contour_normalize:
                        ref_idx = 0 if contour_ref == "first" else (
                            n_valid - 1 if contour_ref == "last"
                            else int(contour_ref))
                        ref_idx = max(0, min(ref_idx, n_valid - 1))
                        P1_ref = np.asarray(P1[contour_frequency_ids, ref_idx])
                        P1_ref_safe = np.where(P1_ref < eps, eps, P1_ref)
                        P1_plot_sub = 20.0 * np.log10(
                            amplitude_plot / P1_ref_safe[:, np.newaxis] + eps)
                        cbar_label = "Amplitude factor [dB]"
                    else:
                        P1_plot_sub = 10.0 * np.log10(amplitude_plot + eps)
                        cbar_label = "Amplitude [dB]"
                    plot_vmin = None
                    plot_vmax = None
                freq_sub = freq[contour_frequency_ids]
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
                    out = output_dir / f"{probe_prefix}_{var_slug}_freq_vs_x_contour.png"
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
                for ih, harmonic_label in enumerate(harmonic_labels):
                    ax.semilogy(probe_x_sorted, harmonic_amp[ih, :],
                                marker="o", markersize=3, lw=0.8,
                                label=harmonic_label)
                ax.set_xlabel("Probe X position [cm]", fontsize=12)
                ax.set_ylabel("Harmonic Amplitude", fontsize=12)
                ax.set_title(f"Harmonic amplitudes vs position (f0 = {harmonic_freq:.3e} Hz)", fontsize=13)
                ax.legend(loc="best", fontsize=9)
                ax.grid(False)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_{var_slug}_harmonics_vs_x.png"
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
                ax.grid(False)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_{var_slug}_spectral_slope_vs_x.png"
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
                ax.grid(False)
                fig.tight_layout()
                out = output_dir / f"{probe_prefix}_{var_slug}_growth_curves.png"
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
                if stream_workspace is not None:
                    stream_workspace.cleanup()
                return (label, False, f"Disturbance reconstruction failed: {exc}")

        if stream_workspace is not None:
            stream_workspace.cleanup()
        return (label, True, None)
    except Exception as exc:
        workspace = locals().get("stream_workspace")
        if workspace is not None:
            workspace.cleanup()
        return (label, False, str(exc))


def _write_gip_only_products(config, bl_profiles, output_dir):
    """Evaluate and save base-flow-only Lees-Lin GIP diagnostics."""
    results = []
    for profile in bl_profiles:
        result = fdb.compute_gpi_criterion(
            profile.get("y_profile", np.array([])),
            profile.get("u_profile", np.array([])),
            profile.get("T_profile", np.array([])),
            profile.get("rho_profile", np.array([])),
            derivative_window=int(config.get(
                "stability_gip_derivative_window", 11
            )),
            derivative_order=int(config.get(
                "stability_gip_derivative_order", 3
            )),
            exclusion_fraction=float(config.get(
                "stability_gip_exclusion_fraction", 0.02
            )),
            residual_multiplier=float(config.get(
                "stability_gip_residual_multiplier", 3.0
            )),
        )
        result["x"] = profile.get("x", np.nan)
        results.append(result)

    point_count = max(
        (len(item.get("y_profile", [])) for item in results), default=0
    )
    crossing_count = max(
        (int(item.get("crossing_count", 0)) for item in results), default=0
    )
    y_profiles = np.full((len(results), point_count), np.nan)
    criteria = np.full_like(y_profiles, np.nan)
    crossings = np.full((len(results), crossing_count), np.nan)
    crossings_normalized = np.full_like(crossings, np.nan)
    for row, item in enumerate(results):
        count = len(item.get("y_profile", []))
        y_profiles[row, :count] = item.get("y_profile", [])
        criteria[row, :count] = item.get("F", [])
        locations = np.asarray(item.get("gip_locations", []), dtype=float)
        normalized = np.asarray(
            item.get("gip_locations_over_delta99", []), dtype=float
        )
        crossings[row, :locations.size] = locations
        crossings_normalized[row, :normalized.size] = normalized

    np.savez_compressed(
        output_dir / "gip_summary.npz",
        baseline_plotfile=np.asarray(config["stability_baseline_plotfile"]),
        x_m=np.asarray([item.get("x", np.nan) for item in results]),
        y_profile_m=y_profiles,
        criterion=criteria,
        delta_99_m=np.asarray([
            item.get("delta_99_est", np.nan) for item in results
        ]),
        crossing_locations_m=crossings,
        crossing_locations_over_delta99=crossings_normalized,
        crossing_count=np.asarray([
            item.get("crossing_count", 0) for item in results
        ], dtype=int),
        gip_present=np.asarray([
            item.get("gip_present", False) for item in results
        ], dtype=bool),
        profile_valid=np.asarray([
            item.get("profile_valid", False) for item in results
        ], dtype=bool),
        rejection_reason=np.asarray([
            item.get("rejection_reason", "") for item in results
        ], dtype="U128"),
        derivative_residual_scale=np.asarray([
            item.get("derivative_residual_scale", np.nan) for item in results
        ]),
        points_inside_accepted_layer=np.asarray([
            np.count_nonzero(
                (np.asarray(item.get("y_profile", []), dtype=float)
                 / float(item.get("delta_99_est", np.nan)) > 0.02)
                & (np.asarray(item.get("y_profile", []), dtype=float)
                   / float(item.get("delta_99_est", np.nan)) < 0.98)
            ) if np.isfinite(item.get("delta_99_est", np.nan))
            and item.get("delta_99_est", np.nan) > 0.0 else 0
            for item in results
        ], dtype=int),
    )

    rejection_counts = {}
    candidate_rejection_counts = {}
    for item in results:
        reason = str(item.get("rejection_reason", ""))
        if reason:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        for candidate_reason in np.asarray(
                item.get("rejected_crossing_reasons", []), dtype=str):
            candidate_rejection_counts[candidate_reason] = (
                candidate_rejection_counts.get(candidate_reason, 0) + 1
            )
    points_inside = []
    for item in results:
        delta = float(item.get("delta_99_est", np.nan))
        y = np.asarray(item.get("y_profile", []), dtype=float)
        points_inside.append(int(np.count_nonzero(
            (y / delta > 0.02) & (y / delta < 0.98)
        )) if np.isfinite(delta) and delta > 0.0 else 0)
    report = {
        "analysis": "Lees-Lin generalized inflection point screening",
        "baseline_plotfile": config["stability_baseline_plotfile"],
        "base_flow_definition": config.get("base_flow_definition", "unspecified"),
        "criterion": "G(y)=d(rho_bar*dU_bar/dy)/dy",
        "interpretation": (
            "A credible interior GIP is a necessary inviscid criterion; it "
            "does not establish modal instability, growth, or transition."
        ),
        "derivative_method": "nonuniform-grid local polynomial least squares",
        "derivative_window": int(config.get("stability_gip_derivative_window", 11)),
        "derivative_order": int(config.get("stability_gip_derivative_order", 3)),
        "exclusion_fraction": float(config.get("stability_gip_exclusion_fraction", 0.02)),
        "residual_multiplier": float(config.get("stability_gip_residual_multiplier", 3.0)),
        "profile_count": len(results),
        "valid_profile_count": int(sum(
            bool(item.get("profile_valid", False)) for item in results
        )),
        "profiles_with_gip": int(sum(
            bool(item.get("gip_present", False)) for item in results
        )),
        "profile_rejection_counts": rejection_counts,
        "candidate_rejection_counts": candidate_rejection_counts,
        "boundary_layer_point_count": {
            "minimum": int(np.min(points_inside)) if points_inside else 0,
            "median": float(np.median(points_inside)) if points_inside else 0.0,
            "maximum": int(np.max(points_inside)) if points_inside else 0,
            "stations_below_seven_points": int(np.sum(
                np.asarray(points_inside) < 7
            )),
        },
        "probe_or_amplification_analysis_performed": False,
    }
    report_path = output_dir / "gip_report.json"
    temporary_path = output_dir / ".gip_report.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, report_path)

    representative_count = min(
        int(config.get("stability_num_gpi_profiles", 5)), len(results)
    )
    representative_indices = np.linspace(
        0, len(results) - 1, representative_count
    ).astype(int) if results else np.array([], dtype=int)
    representatives = [results[index] for index in representative_indices]
    plot_path = output_dir / "gip_diagnostics.png"
    pdb.plot_gip_diagnostics(
        bl_profiles, representatives, output_path=str(plot_path),
        streamwise_results=results,
    )
    return results


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
        rho_inf = config.get("surface_rho_inf", 0.021180978923532486)
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

        if config.get("stability_gip_only", False):
            _ts("  [SD] GIP-only mode: skipping probes, FFT, and amplification")
            _write_gip_only_products(config, bl_profiles, output_dir)
            _ts(f"  [SD] Saved GIP-only products in {output_dir}")
            return ("stability", True, None)

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
        var_col = int(config.get("fft_var_col", 3))
        var_meta = _fft_variable_meta(var_col)
        var_slug = var_meta["slug"]
        signal_name = var_meta["name"].lower()
        nt_skip = config.get("fft_nt_skip", 0)

        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=None
            )
        if not probe_data:
            return ("stability", False, "No probe data")

        probe_x_m = (
            (probe_data.x * 1.0e-2).tolist()
            if isinstance(probe_data, HDF5ProbeStore)
            else [d["x"] * 1e-2 for d in probe_data]
        )

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

        # Frequency-resolved dominant-wave analysis. Unlike the legacy
        # adjacent-pair target-frequency product, this uses Welch-averaged
        # cross spectra and local phase/amplitude regression over the complete
        # configured band. It remains a measurement, not an F/S eigensolution.
        wave_data = {}
        if config.get("stability_make_wavenumber_analysis", True):
            wave_signals = signal_matrix
            wave_time = time_uniform
            analysis_window = config.get(
                "stability_analysis_time_window", None
            )
            if analysis_window is not None:
                time_mask = (
                    (wave_time >= float(analysis_window[0]))
                    & (wave_time <= float(analysis_window[1]))
                )
                if np.sum(time_mask) < 8:
                    raise ValueError(
                        "stability_analysis_time_window contains fewer than "
                        "8 samples"
                    )
                selected_rows = np.flatnonzero(time_mask)
                first_row = int(selected_rows[0])
                stop_row = int(selected_rows[-1]) + 1
                wave_signals = wave_signals[first_row:stop_row, :]
                wave_time = wave_time[first_row:stop_row]
            overlap_fraction = float(config.get(
                "stability_wavenumber_noverlap", 0.5
            ))
            # Preserve at least two statistically independent Welch blocks
            # when an explicit transient window is shorter than the requested
            # segment length.
            largest_two_block_segment = int(
                len(wave_time) / max(2.0 - overlap_fraction, 1.0)
            )
            wave_nperseg = min(
                int(config.get("stability_wavenumber_nperseg", 16384)),
                largest_two_block_segment,
            )
            if wave_nperseg < 8:
                raise ValueError(
                    "Stability analysis window is too short for two Welch "
                    "blocks of at least 8 samples"
                )
            wave_overlap = int(
                overlap_fraction * wave_nperseg
            )
            _ts(
                "  [SD] Computing frequency-resolved complex wavenumber "
                f"over {freq_band[0]:.3e}--{freq_band[1]:.3e} Hz ..."
            )
            wave_data = fdb.compute_frequency_resolved_wavenumber(
                wave_signals,
                np.asarray(probe_x_m),
                1.0 / dt_use,
                freq_band,
                nperseg=wave_nperseg,
                noverlap=wave_overlap,
                frequency_stride=int(config.get(
                    "stability_frequency_stride", 1
                )),
                spatial_window_size=int(config.get(
                    "stability_spatial_window_size", 101
                )),
                spatial_step=int(config.get(
                    "stability_spatial_step", 20
                )),
                fft_batch_size=fft_batch_size,
                min_coherence=float(config.get(
                    "stability_wavenumber_min_coherence", 0.8
                )),
                min_coherent_fraction=float(config.get(
                    "stability_wavenumber_min_coherent_fraction", 0.8
                )),
                min_phase_r_squared=float(config.get(
                    "stability_wavenumber_min_phase_r2", 0.8
                )),
                min_amplitude_r_squared=float(config.get(
                    "stability_wavenumber_min_amplitude_r2", 0.5
                )),
                min_relative_power_db=float(config.get(
                    "stability_wavenumber_min_relative_power_db", -40.0
                )),
                max_edge_phase_rad=float(config.get(
                    "stability_wavenumber_max_edge_phase_rad",
                    0.9 * np.pi,
                )),
                phase_speed_bounds=config.get(
                    "stability_phase_speed_bounds", None
                ),
                phase_convention=config.get(
                    "stability_phase_convention",
                    "omega_t_minus_alpha_x",
                ),
            )
            wave_data["signal_name"] = var_meta["name"]
            wave_data["signal_unit_cgs"] = var_meta["unit_cgs"]
            wave_x = wave_data["x_center_m"]
            wave_u_edge = np.interp(
                wave_x, bl_x, freq_data["u_edge"]
            )
            if T_edge_vals is not None:
                wave_T_edge = np.interp(wave_x, bl_x, T_edge_vals)
                wave_a_edge = np.sqrt(1.4 * 287.05 * wave_T_edge)
                wave_slow = wave_u_edge - wave_a_edge
                wave_fast = wave_u_edge + wave_a_edge
            else:
                wave_slow = np.full_like(wave_x, np.nan)
                wave_fast = np.full_like(wave_x, np.nan)
            wave_data["slow_acoustic_speed_m_per_s"] = wave_slow
            wave_data["fast_acoustic_speed_m_per_s"] = wave_fast
            wave_data["target_frequency_hz"] = float(target_freq)
            wave_data["analysis_time_start_s"] = float(wave_time[0])
            wave_data["analysis_time_end_s"] = float(wave_time[-1])

            # Candidate labels are deliberately reference proximity only.
            phase_speed_grid = wave_data["phase_speed_m_per_s"]
            slow_distance = np.abs(
                phase_speed_grid - wave_slow[None, :]
            )
            fast_distance = np.abs(
                phase_speed_grid - wave_fast[None, :]
            )
            reference_gap = np.abs(wave_fast - wave_slow)[None, :]
            tolerance = float(config.get(
                "stability_acoustic_reference_tolerance", 0.25
            ))
            nearest_distance = np.minimum(slow_distance, fast_distance)
            candidate = np.where(
                slow_distance <= fast_distance, 1, 2
            ).astype(np.int8)
            candidate[
                (~wave_data["phase_valid_mask"])
                | (nearest_distance > tolerance * reference_gap)
                | (~np.isfinite(nearest_distance))
            ] = 0
            wave_data["acoustic_candidate"] = candidate
            wave_data["acoustic_candidate_meaning"] = (
                "0=unclassified, 1=slow-like, 2=fast-like; "
                "reference proximity only, not LST"
            )
            _ts(
                "  [SD] Wavenumber analysis retained "
                f"{np.mean(wave_data['phase_valid_mask']) * 100.0:.1f}% "
                "of phase fits and "
                f"{np.mean(wave_data['growth_valid_mask']) * 100.0:.1f}% "
                "of complex-wavenumber fits"
            )

        amplification_data = {}
        if wave_data and config.get("stability_make_probe_amplification", True):
            amplification_data = fdb.compute_probe_amplification(
                wave_data,
                min_contiguous_centres=int(config.get(
                    "stability_amplification_min_contiguous_centres", 3
                )),
                max_consistency_error=float(config.get(
                    "stability_amplification_max_consistency_error", 1.0
                )),
            )
            _ts(
                "  [SD] Probe-derived amplification: "
                f"{amplification_data['segment_count']} accepted segments; non-LST"
            )

        # GIP is evaluated at every extracted base-flow station. A smaller,
        # representative subset is used for profile overlays only.
        n_bl = len(bl_profiles)
        all_gip_results = []
        for bl in bl_profiles:
            yp = bl.get("y_profile", np.array([]))
            up = bl.get("u_profile", np.array([]))
            Tp = bl.get("T_profile", np.array([]))
            rp = bl.get("rho_profile", np.array([]))
            try:
                gip = fdb.compute_gpi_criterion(
                    yp, up, Tp, rp,
                    derivative_window=int(config.get(
                        "stability_gip_derivative_window", 11
                    )),
                    derivative_order=int(config.get(
                        "stability_gip_derivative_order", 3
                    )),
                    exclusion_fraction=float(config.get(
                        "stability_gip_exclusion_fraction", 0.02
                    )),
                    residual_multiplier=float(config.get(
                        "stability_gip_residual_multiplier", 3.0
                    )),
                )
                gip["x"] = bl["x"]
                all_gip_results.append(gip)
            except Exception as exc:
                fdb._log_error(f"GIP profile at x={bl.get('x', np.nan)}", exc)
        representative_indices = np.linspace(
            0, len(all_gip_results) - 1,
            min(num_gpi, len(all_gip_results)),
        ).astype(int) if all_gip_results else np.array([], dtype=int)
        gpi_results = [all_gip_results[index] for index in representative_indices]

        max_gip_points = max(
            (len(item.get("y_profile", [])) for item in all_gip_results),
            default=0,
        )
        gip_y = np.full((len(all_gip_results), max_gip_points), np.nan)
        gip_criterion = np.full_like(gip_y, np.nan)
        max_crossings = max(
            (int(item.get("crossing_count", 0)) for item in all_gip_results),
            default=0,
        )
        gip_crossings = np.full((len(all_gip_results), max_crossings), np.nan)
        for row, item in enumerate(all_gip_results):
            point_count = len(item.get("y_profile", []))
            gip_y[row, :point_count] = item.get("y_profile", [])
            gip_criterion[row, :point_count] = item.get("F", [])
            crossings = np.asarray(item.get("gip_locations", []), dtype=float)
            gip_crossings[row, :len(crossings)] = crossings

        rejection_counts = {}
        candidate_rejection_counts = {}
        points_inside = []
        for item in all_gip_results:
            reason = str(item.get("rejection_reason", ""))
            if reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            for candidate_reason in np.asarray(
                    item.get("rejected_crossing_reasons", []), dtype=str):
                candidate_rejection_counts[candidate_reason] = (
                    candidate_rejection_counts.get(candidate_reason, 0) + 1
                )
            delta = float(item.get("delta_99_est", np.nan))
            y = np.asarray(item.get("y_profile", []), dtype=float)
            points_inside.append(int(np.count_nonzero(
                (y / delta > 0.02) & (y / delta < 0.98)
            )) if np.isfinite(delta) and delta > 0.0 else 0)
        gip_report = {
            "analysis": "Lees-Lin generalized inflection point screening",
            "criterion": "G(y)=d(rho_bar*dU_bar/dy)/dy",
            "interpretation": (
                "A credible interior GIP is a necessary inviscid criterion; "
                "it does not establish modal instability or growth."
            ),
            "derivative_method": "nonuniform-grid local polynomial least squares",
            "derivative_window": int(config.get("stability_gip_derivative_window", 11)),
            "derivative_order": int(config.get("stability_gip_derivative_order", 3)),
            "exclusion_fraction": float(config.get("stability_gip_exclusion_fraction", 0.02)),
            "residual_multiplier": float(config.get("stability_gip_residual_multiplier", 3.0)),
            "profile_count": len(all_gip_results),
            "valid_profile_count": int(sum(
                bool(item.get("profile_valid", False)) for item in all_gip_results
            )),
            "profiles_with_gip": int(sum(
                bool(item.get("gip_present", False)) for item in all_gip_results
            )),
            "profile_rejection_counts": rejection_counts,
            "candidate_rejection_counts": candidate_rejection_counts,
            "boundary_layer_point_count": {
                "minimum": int(np.min(points_inside)) if points_inside else 0,
                "median": float(np.median(points_inside)) if points_inside else 0.0,
                "maximum": int(np.max(points_inside)) if points_inside else 0,
                "stations_below_seven_points": int(np.sum(
                    np.asarray(points_inside) < 7
                )),
            },
        }
        gip_report_path = output_dir / "gip_report.json"
        gip_report_temporary = output_dir / ".gip_report.json.tmp"
        with gip_report_temporary.open("w", encoding="utf-8") as stream:
            json.dump(gip_report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(gip_report_temporary, gip_report_path)

        plot_data = {
            "freq_estimate": freq_data,
            "phase_speed": phase_data,
            "growth_rate": growth_data,
            "gpi_profiles": gpi_results,
            "target_freq": target_freq,
            "freq_band": freq_band,
            "bl_profiles": bl_profiles,
            "wavenumber": wave_data,
            "probe_amplification": amplification_data,
        }

        np.savez_compressed(
            output_dir / f"stability_summary_{var_slug}.npz",
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
            spatial_amplification_per_m=np.asarray(
                growth_data.get("amplification_rate", [])
            ),
            spatial_growth_delta99=np.asarray(
                growth_data.get("alpha_i_delta", [])
            ),
            spatial_growth_ci95_per_m=np.asarray(
                growth_data.get("alpha_i_ci95", [])
            ),
            growth_fit_r_squared=np.asarray(
                growth_data.get("r_squared", [])
            ),
            wave_frequency_hz=np.asarray(
                wave_data.get("frequency_hz", [])
            ),
            wave_x_center_m=np.asarray(
                wave_data.get("x_center_m", [])
            ),
            wave_alpha_real_rad_per_m=np.asarray(
                wave_data.get("alpha_real_rad_per_m", [])
            ),
            wave_alpha_imag_rad_per_m=np.asarray(
                wave_data.get("alpha_imag_rad_per_m", [])
            ),
            wave_amplification_rate_per_m=np.asarray(
                wave_data.get("amplification_rate_per_m", [])
            ),
            wave_phase_speed_m_per_s=np.asarray(
                wave_data.get("phase_speed_m_per_s", [])
            ),
            wave_wavelength_m=np.asarray(
                wave_data.get("wavelength_m", [])
            ),
            wave_alpha_real_ci95_rad_per_m=np.asarray(
                wave_data.get("alpha_real_ci95_rad_per_m", [])
            ),
            wave_alpha_imag_ci95_rad_per_m=np.asarray(
                wave_data.get("alpha_imag_ci95_rad_per_m", [])
            ),
            wave_phase_fit_r_squared=np.asarray(
                wave_data.get("phase_fit_r_squared", [])
            ),
            wave_amplitude_fit_r_squared=np.asarray(
                wave_data.get("amplitude_fit_r_squared", [])
            ),
            wave_mean_coherence_squared=np.asarray(
                wave_data.get("mean_coherence_squared", [])
            ),
            wave_coherent_pair_fraction=np.asarray(
                wave_data.get("coherent_pair_fraction", [])
            ),
            wave_spatial_alias_margin=np.asarray(
                wave_data.get("spatial_alias_margin", [])
            ),
            wave_spectral_power=np.asarray(
                wave_data.get("spectral_power", [])
            ),
            wave_relative_spectral_power_db=np.asarray(
                wave_data.get("relative_spectral_power_db", [])
            ),
            wave_spectral_valid_mask=np.asarray(
                wave_data.get("spectral_valid_mask", []), dtype=bool
            ),
            wave_phase_valid_mask=np.asarray(
                wave_data.get("phase_valid_mask", []), dtype=bool
            ),
            wave_growth_valid_mask=np.asarray(
                wave_data.get("growth_valid_mask", []), dtype=bool
            ),
            wave_acoustic_candidate=np.asarray(
                wave_data.get("acoustic_candidate", []), dtype=np.int8
            ),
            wave_slow_acoustic_speed_m_per_s=np.asarray(
                wave_data.get("slow_acoustic_speed_m_per_s", [])
            ),
            wave_fast_acoustic_speed_m_per_s=np.asarray(
                wave_data.get("fast_acoustic_speed_m_per_s", [])
            ),
            wave_analysis_time_start_s=np.asarray(
                wave_data.get("analysis_time_start_s", np.nan)
            ),
            wave_analysis_time_end_s=np.asarray(
                wave_data.get("analysis_time_end_s", np.nan)
            ),
            wave_alpha_convention=np.asarray(
                wave_data.get(
                    "alpha_convention", "not computed"
                )
            ),
            wave_phase_convention=np.asarray(
                wave_data.get(
                    "phase_convention", "not computed"
                )
            ),
            gip_x_m=np.asarray([
                item.get("x", np.nan) for item in all_gip_results
            ]),
            gip_y_profile_m=gip_y,
            gip_criterion=gip_criterion,
            gip_crossing_locations_m=gip_crossings,
            gip_crossing_count=np.asarray([
                item.get("crossing_count", 0) for item in all_gip_results
            ], dtype=int),
            gip_present=np.asarray([
                item.get("gip_present", False) for item in all_gip_results
            ], dtype=bool),
            gip_profile_valid=np.asarray([
                item.get("profile_valid", False) for item in all_gip_results
            ], dtype=bool),
            gip_rejection_reason=np.asarray([
                item.get("rejection_reason", "") for item in all_gip_results
            ], dtype="U128"),
            probe_amplification_frequency_hz=np.asarray(
                amplification_data.get("frequency_hz", [])
            ),
            probe_amplification_x_center_m=np.asarray(
                amplification_data.get("x_center_m", [])
            ),
            probe_amplification_n_probe=np.asarray(
                amplification_data.get("n_probe", [])
            ),
            probe_amplification_n_direct=np.asarray(
                amplification_data.get("n_direct", [])
            ),
            probe_amplification_lower_diagnostic=np.asarray(
                amplification_data.get("n_probe_lower_diagnostic", [])
            ),
            probe_amplification_upper_diagnostic=np.asarray(
                amplification_data.get("n_probe_upper_diagnostic", [])
            ),
            probe_amplification_consistency_error=np.asarray(
                amplification_data.get("consistency_error", [])
            ),
            probe_amplification_segment_id=np.asarray(
                amplification_data.get("segment_id", []), dtype=int
            ),
            probe_amplification_reference_x_m=np.asarray(
                amplification_data.get("reference_x_m", [])
            ),
            probe_amplification_valid_mask=np.asarray(
                amplification_data.get("integration_valid_mask", []), dtype=bool
            ),
            probe_amplification_reliable_mask=np.asarray(
                amplification_data.get("reliable_mask", []), dtype=bool
            ),
        )

        if wave_data:
            phase_valid = np.asarray(
                wave_data["phase_valid_mask"], dtype=bool
            )
            growth_valid = np.asarray(
                wave_data["growth_valid_mask"], dtype=bool
            )
            candidate = np.asarray(
                wave_data["acoustic_candidate"], dtype=np.int8
            )
            minimum_coherence = float(config.get(
                "stability_wavenumber_min_coherence", 0.8
            ))
            minimum_coherent_fraction = float(config.get(
                "stability_wavenumber_min_coherent_fraction", 0.8
            ))
            minimum_phase_r2 = float(config.get(
                "stability_wavenumber_min_phase_r2", 0.8
            ))
            minimum_amplitude_r2 = float(config.get(
                "stability_wavenumber_min_amplitude_r2", 0.5
            ))
            minimum_power_db = float(config.get(
                "stability_wavenumber_min_relative_power_db", -40.0
            ))
            maximum_edge_phase = float(config.get(
                "stability_wavenumber_max_edge_phase_rad", 0.9 * np.pi
            ))
            speed_bounds = config.get(
                "stability_phase_speed_bounds", None
            )
            phase_speed_grid = np.asarray(
                wave_data["phase_speed_m_per_s"], dtype=float
            )
            speed_rejected = np.zeros_like(phase_valid)
            if speed_bounds is not None:
                speed_rejected = (
                    (phase_speed_grid < float(speed_bounds[0]))
                    | (phase_speed_grid > float(speed_bounds[1]))
                    | (~np.isfinite(phase_speed_grid))
                )
            report = {
                "analysis": "frequency_resolved_complex_wavenumber",
                "interpretation": (
                    f"Dominant coherent wave measured from the wall {signal_name} "
                    "probe line. Acoustic candidate labels are not F/S LST "
                    "eigenmode identifications."
                ),
                "alpha_convention": wave_data["alpha_convention"],
                "phase_convention": wave_data["phase_convention"],
                "amplification_definition": "-alpha_imag",
                "frequency_band_hz": [
                    float(wave_data["frequency_hz"][0]),
                    float(wave_data["frequency_hz"][-1]),
                ],
                "frequency_bins": int(len(wave_data["frequency_hz"])),
                "spatial_centres": int(len(wave_data["x_center_m"])),
                "analysis_time_s": [
                    float(wave_data["analysis_time_start_s"]),
                    float(wave_data["analysis_time_end_s"]),
                ],
                "welch_blocks": int(wave_data["n_blocks"]),
                "phase_fit_accepted_fraction": float(np.mean(phase_valid)),
                "complex_wavenumber_accepted_fraction": float(
                    np.mean(growth_valid)
                ),
                "candidate_counts": {
                    "unclassified": int(np.sum(candidate == 0)),
                    "slow_like": int(np.sum(candidate == 1)),
                    "fast_like": int(np.sum(candidate == 2)),
                },
                "quality_rejection_counts_nonexclusive": {
                    "relative_power": int(np.sum(
                        np.asarray(
                            wave_data["relative_spectral_power_db"]
                        ) < minimum_power_db
                    )),
                    "coherent_pair_fraction": int(np.sum(
                        np.asarray(
                            wave_data["coherent_pair_fraction"]
                        ) < minimum_coherent_fraction
                    )),
                    "phase_fit_r_squared": int(np.sum(
                        np.asarray(
                            wave_data["phase_fit_r_squared"]
                        ) < minimum_phase_r2
                    )),
                    "spatial_alias_margin": int(np.sum(
                        np.asarray(
                            wave_data["spatial_alias_margin"]
                        ) < (np.pi / maximum_edge_phase)
                    )),
                    "phase_speed_bounds": int(np.sum(speed_rejected)),
                    "amplitude_fit_r_squared": int(np.sum(
                        np.asarray(
                            wave_data["amplitude_fit_r_squared"]
                        ) < minimum_amplitude_r2
                    )),
                },
                "quality_thresholds": {
                    "minimum_coherence_squared": minimum_coherence,
                    "minimum_coherent_pair_fraction": (
                        minimum_coherent_fraction
                    ),
                    "minimum_phase_fit_r_squared": minimum_phase_r2,
                    "minimum_amplitude_fit_r_squared": minimum_amplitude_r2,
                    "minimum_relative_spectral_power_db": minimum_power_db,
                    "phase_speed_bounds_m_per_s": speed_bounds,
                },
            }
            report_path = output_dir / f"wave_analysis_report_{var_slug}.json"
            report_temporary = output_dir / f".wave_analysis_report_{var_slug}.json.tmp"
            with report_temporary.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(report_temporary, report_path)
            _ts(f"  [SD] Saved wave-analysis report: {report_path}")

        selected_amplification_indices = []
        if amplification_data:
            amp_frequency = np.asarray(amplification_data["frequency_hz"])
            requested_frequencies = [float(target_freq)] + [
                float(value) for value in config.get(
                    "stability_amplification_plot_frequencies", []
                )
            ]
            selected_amplification_indices.extend(
                int(np.argmin(np.abs(amp_frequency - value)))
                for value in requested_frequencies
            )
            amp_valid = np.asarray(
                amplification_data["integration_valid_mask"], dtype=bool
            )
            amp_power = np.asarray(wave_data["spectral_power"], dtype=float)
            final_power = np.full(amp_frequency.size, -np.inf)
            for frequency_index in range(amp_frequency.size):
                valid_indices = np.flatnonzero(amp_valid[frequency_index])
                if valid_indices.size:
                    final_power[frequency_index] = amp_power[
                        frequency_index, valid_indices[-1]
                    ]
            strongest = np.argsort(final_power)[::-1]
            selected_amplification_indices.extend(
                int(index) for index in strongest[:3]
                if np.isfinite(final_power[index])
            )
            selected_amplification_indices = list(dict.fromkeys(
                selected_amplification_indices
            ))
            amplitude_variable = {
                1: "density disturbance", 2: "streamwise-velocity disturbance",
                3: "pressure disturbance", 4: "temperature disturbance",
            }.get(var_col, f"probe column {var_col} disturbance")
            amplification_report = {
                "analysis": "probe-derived amplification exponent",
                "non_lst_qualification": (
                    "This is measured logarithmic amplification in the probe "
                    "data, not an LST/PSE transition-prediction N-factor."
                ),
                "integrated_definition": amplification_data["definition"],
                "direct_definition": amplification_data["direct_definition"],
                "pressure_reference_clarification": (
                    "A(f,x_ref) is the disturbance amplitude at a reference "
                    "probe, never mean pressure, freestream pressure, or p_infinity."
                ),
                "signal_variable": amplitude_variable,
                "reference_policy": amplification_data["reference_policy"],
                "frequency_band_hz": [
                    float(amp_frequency[0]), float(amp_frequency[-1])
                ],
                "frequency_bins": int(amp_frequency.size),
                "welch_nperseg": int(wave_data["nperseg"]),
                "welch_noverlap": int(wave_data["noverlap"]),
                "welch_blocks": int(wave_data["n_blocks"]),
                "minimum_contiguous_centres": int(
                    amplification_data["min_contiguous_centres"]
                ),
                "maximum_consistency_error": float(
                    amplification_data["max_consistency_error"]
                ),
                "segment_count": int(amplification_data["segment_count"]),
                "integrated_accepted_fraction": float(np.mean(amp_valid)),
                "reliable_fraction_of_integrated": float(
                    np.sum(amplification_data["reliable_mask"])
                    / max(np.sum(amp_valid), 1)
                ),
                "selected_plot_frequencies_hz": [
                    float(amp_frequency[index])
                    for index in selected_amplification_indices
                ],
                "uncertainty_note": amplification_data["uncertainty_note"],
                "rejection_counts": {
                    "growth_quality_gate": int(
                        amp_valid.size - np.sum(amp_valid)
                    ),
                    "consistency_limit": int(np.sum(
                        amp_valid & ~amplification_data["reliable_mask"]
                    )),
                },
            }
            report_path = output_dir / "probe_amplification_report.json"
            temporary_path = output_dir / ".probe_amplification_report.json.tmp"
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(amplification_report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary_path, report_path)

        _ts("  [SD] Generating stability summary plot ...")
        out_path = output_dir / f"stability_summary_{var_slug}.png"
        pdb.plot_stability_summary(plot_data, output_path=str(out_path))

        if wave_data:
            wave_path = output_dir / f"wavenumber_summary_{var_slug}.png"
            pdb.plot_wavenumber_summary(
                wave_data, output_path=str(wave_path)
            )
            dispersion_path = output_dir / f"phase_speed_dispersion_{var_slug}.png"
            pdb.plot_phase_speed_dispersion(
                wave_data, output_path=str(dispersion_path)
            )
            _ts(f"  [SD] Saved wavenumber summary: {wave_path}")
            _ts(f"  [SD] Saved phase-speed dispersion: {dispersion_path}")

        if amplification_data:
            amplification_map = output_dir / "probe_amplification_map.png"
            pdb.plot_probe_amplification_map(
                amplification_data, output_path=str(amplification_map)
            )
            amplification_curves = output_dir / "probe_amplification_curves.png"
            pdb.plot_probe_amplification_curves(
                amplification_data, selected_amplification_indices,
                output_path=str(amplification_curves),
            )
            _ts(f"  [SD] Saved probe-derived amplification: {amplification_map}")

        if len(gpi_results) > 0:
            gip_out = output_dir / f"gip_diagnostics_{var_slug}.png"
            pdb.plot_gip_diagnostics(
                bl_profiles, gpi_results, output_path=str(gip_out),
                streamwise_results=all_gip_results,
            )
            _ts(f"  [SD] Saved GIP diagnostics: {gip_out}")

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

        var_col = int(config.get("fft_var_col", 3))
        var_meta = _fft_variable_meta(var_col)
        var_slug = var_meta["slug"]
        nt_skip = config.get("fft_nt_skip", 0)
        max_probes = config.get("fft_max_probes", None)
        band = config.get("transient_band", [5.0e6, 50.0e6])
        nperseg = config.get("transient_stft_nperseg", 256)
        noverlap_frac = config.get("transient_stft_noverlap", 0.75)
        plot_probes = config.get("transient_plot_probe_indices", [0, 49, 99])
        spectrogram_scale = config.get(
            "transient_spectrogram_scale", "db"
        )
        spectrogram_vmin = config.get(
            "transient_spectrogram_vmin"
        )
        spectrogram_vmax = config.get(
            "transient_spectrogram_vmax"
        )
        spectrogram_fmax = config.get(
            "transient_spectrogram_fmax_hz", 2.0 * float(band[1])
        )
        make_fft_stft_comparison = config.get(
            "transient_make_fft_stft_comparison", False
        )

        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col, nt_skip=nt_skip, max_probes=max_probes
            )
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        n_valid = len(probe_data)
        probe_x = (
            probe_data.x.tolist()
            if isinstance(probe_data, HDF5ProbeStore)
            else [d["x"] for d in probe_data]
        )
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
                out_stft = output_dir / f"spectrogram_{var_slug}_probe{idx:03d}.png"
                pdb.plot_spectrogram(
                    time_uniform, sig_ms, fs,
                    output_path=str(out_stft),
                    fmax=spectrogram_fmax,
                    title=f"STFT spectrogram — Probe {idx} — x={p['x']:.3f} cm",
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scale=spectrogram_scale,
                    vmin=spectrogram_vmin,
                    vmax=spectrogram_vmax,
                )
                _ts(f"  [TA] Saved spectrogram: {out_stft}")
            except Exception as exc:
                _ts(f"  [TA] Spectrogram failed for probe {idx}: {exc}")

            if make_fft_stft_comparison:
                try:
                    comparison_output = (
                        output_dir
                        / f"fft_stft_comparison_{var_slug}_probe{idx:03d}.png"
                    )
                    pdb.plot_fft_stft_comparison(
                        time_uniform, sig, fs,
                        output_path=str(comparison_output),
                        fmax=spectrogram_fmax,
                        title=(
                            f"Probe {idx} — x={p['x']:.3f} cm: "
                            "mean-subtracted history, FFT, and STFT"
                        ),
                        signal_label=var_meta["ms_symbol"],
                        nperseg=nperseg,
                        noverlap=noverlap,
                        spectrogram_scale=spectrogram_scale,
                        spectrogram_vmin=spectrogram_vmin,
                        spectrogram_vmax=spectrogram_vmax,
                        fft_window=config.get("fft_window", "hann"),
                    )
                    _ts(
                        "  [TA] Saved FFT/STFT comparison: "
                        f"{comparison_output}"
                    )
                except Exception as exc:
                    _ts(
                        f"  [TA] FFT/STFT comparison failed for probe "
                        f"{idx}: {exc}"
                    )

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
        packet_envelopes = []
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
                packet_envelopes.append(envelope)
            except Exception:
                packet_stats_list.append({
                    "peak_amplitude": np.nan, "peak_time": np.nan,
                    "arrival_time": np.nan, "integrated_energy": np.nan,
                })
                packet_envelopes.append(
                    np.full(time_uniform.shape, np.nan)
                )

        if len(packet_stats_list) > 1:
            try:
                pdb.plot_envelope_growth(
                    packet_x, packet_stats_list,
                    output_path=str(
                        output_dir / f"envelope_growth_vs_x_{var_slug}.png"
                    ),
                )
                _ts("  [TA] Saved envelope growth vs x")
            except Exception as exc:
                _ts(f"  [TA] Envelope growth plot failed: {exc}")

        stat_keys = (
            "peak_amplitude", "peak_time", "arrival_time",
            "half_width_half_max", "integrated_energy",
        )
        np.savez_compressed(
            output_dir / f"packet_statistics_{var_slug}.npz",
            probe_indices=np.asarray(packet_indices, dtype=int),
            probe_x_cm=np.asarray(packet_x, dtype=float),
            signal_name=np.array(var_meta["name"]),
            signal_unit_cgs=np.array(var_meta["unit_cgs"]),
            **{
                key: np.asarray([
                    np.nan if item.get(key) is None else item.get(key, np.nan)
                    for item in packet_stats_list
                ], dtype=float)
                for key in stat_keys
            },
        )
        _ts(f"  [TA] Saved packet statistics: "
            f"{output_dir / f'packet_statistics_{var_slug}.npz'}")

        propagation = None
        if packet_envelopes:
            if np.all(np.isfinite(packet_envelopes)):
                propagation = fdb.estimate_packet_propagation(
                    np.asarray(packet_x, dtype=float) * 0.01,
                    time_uniform,
                    np.column_stack(packet_envelopes),
                    baseline_end_time_s=config.get(
                        "transient_baseline_end_time_s"
                    ),
                    noise_sigma=config.get(
                        "transient_arrival_noise_sigma", 6.0
                    ),
                    peak_fraction=config.get(
                        "transient_arrival_peak_fraction", 0.05
                    ),
                    persistent_samples=config.get(
                        "transient_arrival_persistent_samples", 4
                    ),
                    filter_edge_fraction=config.get(
                        "transient_filter_edge_fraction", 0.01
                    ),
                )
            else:
                failed_envelopes = int(sum(
                    not np.all(np.isfinite(values))
                    for values in packet_envelopes
                ))
                propagation = {
                    "status": "insufficient_data",
                    "reason": (
                        f"{failed_envelopes} of {len(packet_envelopes)} "
                        "band-limited probe envelopes failed"
                    ),
                    "baseline_source": "not_evaluated",
                }
            np.savez_compressed(
                output_dir / f"packet_propagation_{var_slug}.npz",
                **{
                    key: value for key, value in propagation.items()
                    if not isinstance(value, str) and value is not None
                },
                status=np.array(propagation["status"]),
                baseline_source=np.array(
                    propagation.get("baseline_source", "unknown")
                ),
                band_hz=np.asarray(band, dtype=float),
                filtering=np.array(
                    "fourth-order zero-phase Butterworth bandpass"
                ),
                signal_name=np.array(var_meta["name"]),
                signal_unit_cgs=np.array(var_meta["unit_cgs"]),
            )
            if propagation["status"] == "complete":
                pdb.plot_packet_propagation(
                    propagation,
                    output_path=str(
                        output_dir / f"packet_propagation_{var_slug}.png"
                    ),
                )
            propagation_report = {
                key: _json_safe(value) for key, value in propagation.items()
                if np.asarray(value).ndim == 0
            }
            propagation_report.update({
                "band_hz": list(map(float, band)),
                "interpretation": (
                    "Band-limited packet kinematics; not an LST group-"
                    "velocity calculation"
                ),
                "signal_name": var_meta["name"],
                "signal_unit_cgs": var_meta["unit_cgs"],
            })
            report_path = output_dir / f"packet_propagation_{var_slug}.json"
            temporary = output_dir / f".packet_propagation_{var_slug}.json.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(propagation_report, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, report_path)
            _ts(
                "  [TA] Saved robust packet propagation and group-velocity "
                "analysis"
            )

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

        var_col = int(config.get("fft_var_col", 3))
        var_meta = _fft_variable_meta(var_col)
        var_slug = var_meta["slug"]
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
        probe_x = (
            probe_data.x.tolist()
            if isinstance(probe_data, HDF5ProbeStore)
            else [d["x"] for d in probe_data]
        )
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
        significance_results = []
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
                if config.get("nonlinear_surrogate_validation", True):
                    try:
                        significance = (
                            fdb.compute_surrogate_triad_significance(
                                sig_ms, fs, target_freqs,
                                nperseg=nperseg, noverlap=noverlap,
                                n_surrogates=config.get(
                                    "nonlinear_surrogate_count", 200
                                ),
                                fdr_alpha=config.get(
                                    "nonlinear_surrogate_alpha", 0.05
                                ),
                                minimum_independent_segments=config.get(
                                    "nonlinear_minimum_independent_segments",
                                    8,
                                ),
                                random_seed=(
                                    int(config.get(
                                        "nonlinear_surrogate_seed", 271828
                                    )) + int(idx)
                                ),
                                laser_frequency_hz=config.get(
                                    "nonlinear_laser_frequency_hz"
                                ),
                            )
                        )
                        significance_results.append({
                            "probe_index": int(idx),
                            "probe_x_m": float(p["x"]) * 0.01,
                            "status": "complete",
                            "result": significance,
                        })
                    except ValueError as exc:
                        significance_results.append({
                            "probe_index": int(idx),
                            "probe_x_m": float(p["x"]) * 0.01,
                            "status": "insufficient_data",
                            "reason": str(exc),
                        })
            except Exception as exc:
                _ts(f"  [NL] Bicoherence failed for probe {idx}: {exc}")
                bicoherence_data.append((idx, p["x"], {}))

        if config.get("nonlinear_surrogate_validation", True):
            completed = [
                item for item in significance_results
                if item["status"] == "complete"
            ]
            significance_report = {
                "analysis": "phase_randomized_bicoherence_significance",
                "status": (
                    "complete" if completed else "insufficient_data"
                ),
                "probe_results": [
                    {
                        "probe_index": item["probe_index"],
                        "probe_x_m": item["probe_x_m"],
                        "status": item["status"],
                        "reason": item.get("reason"),
                        "significant_nonlaser_triad_count": (
                            int(np.sum(
                                item["result"]["significant_fdr"]
                                & ~item["result"]["laser_harmonic_related"]
                            )) if item["status"] == "complete" else 0
                        ),
                        "significant_laser_related_triad_count": (
                            int(np.sum(
                                item["result"]["significant_fdr"]
                                & item["result"]["laser_harmonic_related"]
                            )) if item["status"] == "complete" else 0
                        ),
                    }
                    for item in significance_results
                ],
                "interpretation": (
                    "Significant laser-related triads are separated from "
                    "candidate downstream mode-mode coupling."
                ),
            }
            temporary = output_dir / f".nonlinear_significance_{var_slug}.json.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    significance_report, stream, indent=2, sort_keys=True
                )
                stream.write("\n")
            os.replace(
                temporary,
                output_dir / f"nonlinear_significance_{var_slug}.json",
            )
            if completed:
                labels = completed[0]["result"]["triad_labels"]
                np.savez_compressed(
                    output_dir / f"nonlinear_significance_{var_slug}.npz",
                    probe_indices=np.asarray([
                        item["probe_index"] for item in completed
                    ], dtype=int),
                    probe_x_m=np.asarray([
                        item["probe_x_m"] for item in completed
                    ]),
                    triad_labels=labels,
                    observed_bicoherence_squared=np.vstack([
                        item["result"]["observed_bicoherence_squared"]
                        for item in completed
                    ]),
                    surrogate_median_bicoherence_squared=np.vstack([
                        item["result"][
                            "surrogate_median_bicoherence_squared"
                        ] for item in completed
                    ]),
                    empirical_p_value=np.vstack([
                        item["result"]["empirical_p_value"]
                        for item in completed
                    ]),
                    fdr_adjusted_p_value=np.vstack([
                        item["result"]["fdr_adjusted_p_value"]
                        for item in completed
                    ]),
                    significant_fdr=np.vstack([
                        item["result"]["significant_fdr"]
                        for item in completed
                    ]),
                    laser_harmonic_related=np.vstack([
                        item["result"]["laser_harmonic_related"]
                        for item in completed
                    ]),
                    signal_name=np.array(var_meta["name"]),
                    signal_unit_cgs=np.array(var_meta["unit_cgs"]),
                )

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
                    output_path=str(
                        output_dir / f"bicoherence_vs_x_{var_slug}.png"
                    ),
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
                    output_dir / f"triad_bicoherence_{var_slug}.npz",
                    probe_x_cm=x_sorted,
                    triad_labels=np.asarray(triad_labels),
                    bicoherence_squared=triad_values,
                    nperseg=np.array(nperseg),
                    noverlap=np.array(noverlap),
                    signal_name=np.array(var_meta["name"]),
                    signal_unit_cgs=np.array(var_meta["unit_cgs"]),
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


def _process_spatial_fft_probes(config, probe_data=None):
    """Measure spatial response spectra along the fixed probe aperture."""
    label = "spatial_fft"
    try:
        output_dir = Path(config["output_dir"]) / "Spatial-FFT"
        output_dir.mkdir(parents=True, exist_ok=True)
        spatial_cfg = config.get("spatial_fft", {})
        var_col = int(config.get("fft_var_col", 3))
        var_meta = _fft_variable_meta(var_col)
        var_slug = var_meta["slug"]
        signal_name = var_meta["name"].lower()
        if probe_data is None:
            probe_data = _load_probe_timeseries(
                config, var_col,
                nt_skip=config.get("fft_nt_skip", 0),
                max_probes=config.get("fft_max_probes", None),
            )
        if not probe_data:
            return (label, False, "No probe data could be loaded")

        time_s, signal_matrix, dt_use, _ = _probe_matrix_on_common_time(
            probe_data,
            resample=config.get("fft_resample", True),
            target_dt=config.get("fft_target_dt"),
        )
        if isinstance(probe_data, HDF5ProbeStore):
            probe_x_cm = np.asarray(probe_data.x, dtype=float)
            probe_y_cm = np.asarray(probe_data.y, dtype=float)
        else:
            probe_x_cm = np.asarray([item["x"] for item in probe_data], dtype=float)
            probe_y_cm = np.asarray([item["y"] for item in probe_data], dtype=float)
        complete = np.all(np.isfinite(signal_matrix), axis=0)
        if np.count_nonzero(complete) < 4:
            raise ValueError("Spatial FFT requires at least four complete probes")
        signal_matrix = np.asarray(signal_matrix[:, complete], dtype=float)
        probe_x_cm = probe_x_cm[complete]
        probe_y_cm = probe_y_cm[complete]
        order = np.argsort(probe_x_cm)
        signal_matrix = signal_matrix[:, order]
        probe_x_cm = probe_x_cm[order]
        probe_y_cm = probe_y_cm[order]

        configured_indices = spatial_cfg.get("snapshot_indices")
        configured_times = spatial_cfg.get("snapshot_times_s")
        if configured_indices is not None:
            snapshot_indices = np.asarray(configured_indices, dtype=int).ravel()
            snapshot_indices = snapshot_indices[
                (snapshot_indices >= 0) & (snapshot_indices < len(time_s))
            ]
        elif configured_times is not None:
            snapshot_indices = np.unique([
                int(np.argmin(np.abs(time_s - float(value))))
                for value in configured_times
            ])
        else:
            maximum = spatial_cfg.get("max_snapshots")
            if maximum is None:
                snapshot_indices = np.arange(len(time_s), dtype=int)
            else:
                maximum = max(1, int(maximum))
                snapshot_indices = np.unique(np.linspace(
                    0, len(time_s) - 1, min(maximum, len(time_s)), dtype=int
                ))
        if snapshot_indices.size == 0:
            raise ValueError("No valid spatial FFT snapshots were selected")

        x_m = probe_x_cm * 1.0e-2
        y_m = probe_y_cm * 1.0e-2
        baseline_mode = str(
            spatial_cfg.get("baseline_mode", "pre_event_mean")
        ).lower()
        if baseline_mode == "pre_event_mean":
            baseline_end = spatial_cfg.get("baseline_end_time_s")
            if baseline_end is None:
                source_cfg = config.get("source_response", {})
                pulse_fwhm = float(source_cfg["pulse_fwhm_s"])
                sigma = pulse_fwhm / (
                    2.0 * np.sqrt(2.0 * np.log(2.0))
                )
                pulse_center = (
                    float(source_cfg.get("start_time_s", 0.0))
                    + 0.5 * float(source_cfg["pulse_period_s"])
                )
                baseline_end = pulse_center - float(
                    source_cfg.get("cutoff_sigma", 4.0)
                ) * sigma
            baseline_result = fdb.subtract_quiescent_probe_baseline(
                time_s, signal_matrix, float(baseline_end),
                minimum_samples=int(
                    spatial_cfg.get("minimum_baseline_samples", 8)
                ),
            )
            disturbance_matrix = baseline_result["disturbance"]
            baseline = baseline_result["baseline"]
        elif baseline_mode == "none":
            disturbance_matrix = np.asarray(signal_matrix, dtype=float)
            baseline = np.zeros(disturbance_matrix.shape[1], dtype=float)
            baseline_result = {
                "baseline_start_time_s": np.nan,
                "baseline_end_time_s": np.nan,
                "baseline_sample_count": 0,
            }
        else:
            raise ValueError("spatial_fft.baseline_mode must be pre_event_mean or none")
        selected_signals = disturbance_matrix[snapshot_indices, :]
        fft_kwargs = {
            "mean_subtraction": spatial_cfg.get("mean_subtraction", "mean"),
            "window": spatial_cfg.get("window", "hann"),
            "window_compensation": spatial_cfg.get("window_compensation", True),
            "zero_padding": int(spatial_cfg.get("zero_padding", 0)),
        }
        response = fdb.compute_spatial_fft(selected_signals, x_m, **fft_kwargs)
        model = str(spatial_cfg.get("model", "wang_kernel")).lower()
        source_kwargs = {
            "mean_subtraction": fft_kwargs["mean_subtraction"],
            "window": fft_kwargs["window"],
            "window_compensation": fft_kwargs["window_compensation"],
            "zero_padding": fft_kwargs["zero_padding"],
            "center_x_m": 1.0e-2 * float(spatial_cfg.get("center_x_cm", 2.5)),
            "center_y_m": 1.0e-2 * float(spatial_cfg.get("center_y_cm", 2.5)),
        }
        if model == "gaussian":
            source_kwargs["radius_m"] = 1.0e-2 * float(
                spatial_cfg["radius_cm"]
            )
        else:
            source_kwargs["wang_parameters"] = {
                "length_m": 1.0e-2 * float(spatial_cfg["wang_length_cm"]),
                "aspect_ratio": float(spatial_cfg["wang_aspect_ratio"]),
                "asymmetry": float(spatial_cfg["wang_asymmetry"]),
            }
        source = fdb.compute_spatial_source_spectrum(
            x_m, y_m, model=model, **source_kwargs
        )
        if not np.allclose(
                response["wavenumber_rad_per_m"],
                source["wavenumber_rad_per_m"],
                rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError("Spatial source and response grids do not match")
        transfer = fdb.compute_spatial_transfer_function(
            source["complex_spectrum"][0], response["complex_spectrum"],
            minimum_relative_source_amplitude=float(
                spatial_cfg.get("minimum_relative_source_amplitude", 1.0e-3)
            ),
        )
        summary = {
            "wavenumber_rad_per_m": response["wavenumber_rad_per_m"],
            "probe_x_m": response["probe_x_m"],
            "probe_y_m": y_m,
            "source_profile": source["profile"],
            "source_spatial_integral_m2": np.array(
                source["spatial_integral_m2"]
            ),
            "source_profile_units": np.array("m^-2"),
            "source_complex": source["complex_spectrum"][0],
            "source_amplitude": source["amplitude"][0],
            "response_complex": response["complex_spectrum"],
            "response_amplitude": response["amplitude"],
            "response_signal_name": np.array(signal_name),
            "response_signal_unit_cgs": np.array(var_meta["unit_cgs"]),
            "response_definition": np.array(
                "per-probe disturbance q'(x,t)=q(x,t)-q0(x)"
            ),
            "quiescent_baseline": baseline,
            "baseline_mode": np.array(baseline_mode),
            "baseline_start_time_s": np.array(
                baseline_result["baseline_start_time_s"]
            ),
            "baseline_end_time_s": np.array(
                baseline_result["baseline_end_time_s"]
            ),
            "baseline_sample_count": np.array(
                baseline_result["baseline_sample_count"]
            ),
            "event_start_time_s": np.array(
                float(baseline_end) if baseline_mode == "pre_event_mean"
                else np.nan
            ),
            "transfer_magnitude": transfer["magnitude"],
            "transfer_phase_rad": transfer["phase_rad"],
            "valid_wavenumber": transfer["valid_wavenumber"],
            "snapshot_time_s": time_s[snapshot_indices],
            "snapshot_indices": snapshot_indices,
            "sample_interval_s": np.array(dt_use),
            "dx_m": np.array(response["dx_m"]),
            "sample_span_m": np.array(response["sample_span_m"]),
            "physical_aperture_m": np.array(response["physical_aperture_m"]),
            "dft_record_length_m": np.array(response["dft_record_length_m"]),
            "transform_record_length_m": np.array(
                response["transform_record_length_m"]
            ),
            "native_delta_k_rad_per_m": np.array(
                response["native_delta_k_rad_per_m"]
            ),
            "display_delta_k_rad_per_m": np.array(
                response["display_delta_k_rad_per_m"]
            ),
            "nyquist_wavenumber_rad_per_m": np.array(
                response["nyquist_wavenumber_rad_per_m"]
            ),
            "n_spatial_samples": np.array(response["n_spatial_samples"]),
            "transform_length": np.array(response["transform_length"]),
            "zero_padding": np.array(response["zero_padding"]),
            "wavenumber_colorbar_vmax": np.array(
                spatial_cfg.get("wavenumber_colorbar_vmax", np.nan)
            ),
            "source_model": np.array(model),
            "transform_convention": np.array(response["transform_convention"]),
            "source_response_ratio_interpretation": np.array(
                "diagnostic source-shape-normalized response; not a "
                "dimensionless transfer function"
            ),
        }
        np.savez_compressed(
            output_dir / f"spatial_spectral_summary_{var_slug}.npz", **summary
        )
        source_plot_data = dict(source)
        pdb.plot_spatial_source_spectrum(
            source_plot_data,
            output_path=str(
                output_dir / f"spatial_source_spectrum_{var_slug}.png"
            ),
        )
        pdb.plot_spatial_response_spectrum(
            summary,
            output_path=str(
                output_dir / f"spatial_response_spectrum_{var_slug}.png"
            ),
        )
        fft_probe_output_dir = Path(config["output_dir"]) / "FFT-Probes"
        fft_probe_output_dir.mkdir(parents=True, exist_ok=True)
        probe_prefix = config.get("probe_prefix", "kernel-probe")
        pdb.plot_wavenumber_time_contour(
            summary,
            output_path=str(
                fft_probe_output_dir
                / f"{probe_prefix}_{var_slug}_wavenumber_vs_time_contour.png"
            ),
        )
        pdb.plot_spatial_transfer_function(
            summary,
            output_path=str(
                output_dir / f"spatial_transfer_wavenumber_{var_slug}.png"
            ),
        )
        if spatial_cfg.get("make_k_omega", True):
            k_omega = fdb.compute_wavenumber_frequency_spectrum(
                disturbance_matrix, time_s, x_m,
                temporal_window=spatial_cfg.get(
                    "k_omega_temporal_window", "none"
                ),
                spatial_window=spatial_cfg.get(
                    "k_omega_spatial_window", "hann"
                ),
                temporal_mean_subtraction=spatial_cfg.get(
                    "k_omega_temporal_mean_subtraction", "none"
                ),
                spatial_mean_subtraction=spatial_cfg.get(
                    "k_omega_spatial_mean_subtraction", "none"
                ),
                temporal_zero_padding=int(spatial_cfg.get(
                    "k_omega_temporal_zero_padding", 0
                )),
                spatial_zero_padding=int(spatial_cfg.get(
                    "k_omega_spatial_zero_padding", 0
                )),
                window_compensation=bool(spatial_cfg.get(
                    "window_compensation", True
                )),
            )
            k_omega_archive = output_dir / f"k_omega_spectrum_{var_slug}.npz"
            np.savez_compressed(
                k_omega_archive,
                frequency_hz=k_omega["frequency_hz"],
                wavenumber_rad_per_m=k_omega["wavenumber_rad_per_m"],
                power=k_omega["power"],
                amplitude=k_omega["amplitude"],
                native_frequency_resolution_hz=np.array(
                    k_omega["native_frequency_resolution_hz"]
                ),
                display_frequency_spacing_hz=np.array(
                    k_omega["display_frequency_spacing_hz"]
                ),
                native_wavenumber_resolution_rad_per_m=np.array(
                    k_omega["native_wavenumber_resolution_rad_per_m"]
                ),
                display_wavenumber_spacing_rad_per_m=np.array(
                    k_omega["display_wavenumber_spacing_rad_per_m"]
                ),
                nyquist_frequency_hz=np.array(k_omega["nyquist_frequency_hz"]),
                nyquist_wavenumber_rad_per_m=np.array(
                    k_omega["nyquist_wavenumber_rad_per_m"]
                ),
                direction_convention=np.array(k_omega["direction_convention"]),
                response_signal_name=np.array(signal_name),
                response_signal_unit_cgs=np.array(var_meta["unit_cgs"]),
            )
            pdb.plot_wavenumber_frequency_spectrum(
                k_omega,
                output_path=str(
                    output_dir / f"k_omega_spectrum_{var_slug}.png"
                ),
                frequency_max_hz=spatial_cfg.get(
                    "k_omega_frequency_max_hz"
                ),
                db_floor=float(spatial_cfg.get("k_omega_db_floor", -80.0)),
                trustworthy_nyquist_fraction=float(spatial_cfg.get(
                    "k_omega_trustworthy_nyquist_fraction", 0.8
                )),
                signal_name=var_meta["name"],
            )
        _ts(
            f"  Spatial FFT: {len(snapshot_indices)} snapshots x "
            f"{len(probe_x_cm)} probes; aperture={response['physical_aperture_m']:.6e} m"
        )
        return (label, True, None)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (label, False, str(exc))


def _process_spatial_case_comparison(config):
    """Compare saved spatial source/response spectra for two cases."""
    label = "spatial_case_comparison"
    try:
        comparison_cfg = config.get("spatial_case_comparison", {})
        root = Path(__file__).resolve().parent

        def archive_path(key):
            path = Path(comparison_cfg[key]).expanduser()
            return path if path.is_absolute() else root / path

        paths = {
            "baseline": archive_path("baseline_archive"),
            "comparison": archive_path("comparison_archive"),
        }
        loaded = {}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Spatial comparison archive not found: {path}")
            loaded[name] = np.load(path)
        try:
            baseline = loaded["baseline"]
            comparison = loaded["comparison"]
            required = {
                "wavenumber_rad_per_m", "probe_x_m", "source_amplitude",
                "response_amplitude",
            }
            for name, archive in loaded.items():
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"{name} spatial archive missing {sorted(missing)}")
            frequency = np.asarray(baseline["wavenumber_rad_per_m"], dtype=float)
            if (not np.allclose(frequency, comparison["wavenumber_rad_per_m"],
                                rtol=1.0e-10, atol=1.0e-8)
                    or not np.allclose(baseline["probe_x_m"], comparison["probe_x_m"],
                                       rtol=1.0e-10, atol=1.0e-12)):
                raise ValueError("Spatial case coordinates or wavenumber grids do not match")
            baseline_response = np.sqrt(np.nanmean(
                np.asarray(baseline["response_amplitude"]) ** 2, axis=0
            ))
            comparison_response = np.sqrt(np.nanmean(
                np.asarray(comparison["response_amplitude"]) ** 2, axis=0
            ))
            response_signal_name = "selected probe signal"
            if "response_signal_name" in baseline.files:
                response_signal_name = str(
                    np.asarray(baseline["response_signal_name"]).item()
                )
            if ("response_signal_name" in baseline.files
                    and "response_signal_name" in comparison.files
                    and response_signal_name != str(
                        np.asarray(comparison["response_signal_name"]).item()
                    )):
                raise ValueError("Spatial comparison probe variables do not match")
            plot_data = {
                "wavenumber_rad_per_m": frequency,
                "baseline_source_amplitude": np.asarray(baseline["source_amplitude"]),
                "comparison_source_amplitude": np.asarray(comparison["source_amplitude"]),
                "baseline_response_amplitude": baseline_response,
                "comparison_response_amplitude": comparison_response,
                "response_signal_name": response_signal_name,
                "baseline_snapshot_count": np.asarray(
                    baseline["response_amplitude"]
                ).shape[0],
                "comparison_snapshot_count": np.asarray(
                    comparison["response_amplitude"]
                ).shape[0],
                "baseline_label": str(comparison_cfg.get("baseline_label", "baseline")),
                "comparison_label": str(comparison_cfg.get("comparison_label", "comparison")),
            }
        finally:
            for archive in loaded.values():
                archive.close()
        output_dir = Path(config["output_dir"]) / "SpatialCaseComparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        ratio = comparison_response / np.maximum(baseline_response, np.finfo(float).tiny)
        np.savez_compressed(
            output_dir / "spatial_case_comparison.npz",
            **plot_data,
            comparison_to_baseline_response_ratio=ratio,
        )
        pdb.plot_spatial_case_comparison(
            plot_data,
            output_path=str(output_dir / "spatial_source_response_comparison.png"),
        )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


def _process_case_spectrum_comparison(config):
    """Compare stored source/transfer spectra for two named simulation cases."""
    label = "case_spectrum_comparison"
    try:
        comparison_cfg = config.get("case_spectrum_comparison", {})
        root = Path(__file__).resolve().parent

        def archive_path(key):
            path = Path(comparison_cfg[key]).expanduser()
            return path if path.is_absolute() else root / path

        baseline_source_path = archive_path("baseline_source_response_archive")
        baseline_summary_path = archive_path("baseline_spectral_summary_archive")
        comparison_source_path = archive_path(
            "comparison_source_response_archive"
        )
        comparison_summary_path = archive_path(
            "comparison_spectral_summary_archive"
        )
        required_source = {
            "frequency_hz", "source_processed_amplitude",
            "transfer_magnitude", "valid_transfer_frequency",
        }
        required_summary = {"source_probe_indices", "probe_x_cm"}
        for path in (
                baseline_source_path, baseline_summary_path,
                comparison_source_path, comparison_summary_path):
            if not path.is_file():
                raise FileNotFoundError(f"Comparison archive not found: {path}")

        with np.load(baseline_source_path) as baseline_source, \
                np.load(baseline_summary_path) as baseline_summary, \
                np.load(comparison_source_path) as comparison_source, \
                np.load(comparison_summary_path) as comparison_summary:
            for name, archive in (
                    ("baseline source", baseline_source),
                    ("comparison source", comparison_source)):
                missing = required_source - set(archive.files)
                if missing:
                    raise ValueError(f"{name} archive missing keys: {sorted(missing)}")
            for name, archive in (
                    ("baseline summary", baseline_summary),
                    ("comparison summary", comparison_summary)):
                missing = required_summary - set(archive.files)
                if missing:
                    raise ValueError(f"{name} archive missing keys: {sorted(missing)}")

            frequency = np.asarray(baseline_source["frequency_hz"], dtype=float)
            comparison_frequency = np.asarray(
                comparison_source["frequency_hz"], dtype=float
            )
            if not np.allclose(frequency, comparison_frequency, rtol=1.0e-10,
                               atol=max(abs(frequency[1]) * 1.0e-10, 1.0e-15)):
                raise ValueError("Case source frequency grids do not match")

            baseline_ids = np.asarray(
                baseline_summary["source_probe_indices"], dtype=int
            )
            comparison_ids = np.asarray(
                comparison_summary["source_probe_indices"], dtype=int
            )
            baseline_x = np.asarray(baseline_summary["probe_x_cm"], dtype=float)
            comparison_x = np.asarray(
                comparison_summary["probe_x_cm"], dtype=float
            )
            if (not np.array_equal(baseline_ids, comparison_ids)
                    or baseline_x.shape != comparison_x.shape
                    or not np.allclose(baseline_x, comparison_x)):
                raise ValueError(
                    "Case probe IDs or physical x coordinates do not match"
                )

            baseline_transfer = np.asarray(
                baseline_source["transfer_magnitude"], dtype=float
            )
            comparison_transfer = np.asarray(
                comparison_source["transfer_magnitude"], dtype=float
            )
            if (baseline_transfer.shape != comparison_transfer.shape
                    or baseline_transfer.shape != (frequency.size, baseline_ids.size)):
                raise ValueError("Case transfer arrays do not match probe/frequency grids")

            requested_ids = comparison_cfg.get("probe_indices", [])
            if not requested_ids:
                requested_ids = baseline_ids.tolist()
            index_by_id = {int(probe_id): index for index, probe_id in enumerate(baseline_ids)}
            selected_columns = []
            for probe_id in requested_ids:
                probe_id = int(probe_id)
                if probe_id not in index_by_id:
                    raise ValueError(f"Comparison probe ID {probe_id} is unavailable")
                selected_columns.append(index_by_id[probe_id])
            selected_columns = np.asarray(sorted(set(selected_columns)), dtype=int)
            selected_ids = baseline_ids[selected_columns]
            selected_x = baseline_x[selected_columns]
            station_labels = [
                f"Probe {probe_id} (x={x_value:.4f} cm)"
                for probe_id, x_value in zip(selected_ids, selected_x)
            ]
            valid_frequency = (
                np.asarray(baseline_source["valid_transfer_frequency"], dtype=bool)
                & np.asarray(comparison_source["valid_transfer_frequency"], dtype=bool)
            )
            baseline_source_amplitude = np.asarray(
                baseline_source["source_processed_amplitude"], dtype=float
            )
            comparison_source_amplitude = np.asarray(
                comparison_source["source_processed_amplitude"], dtype=float
            )

        baseline_label = str(comparison_cfg.get("baseline_label", "baseline"))
        comparison_label = str(comparison_cfg.get("comparison_label", "comparison"))
        output_dir = Path(config["output_dir"]) / "CaseComparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_fmax = comparison_cfg.get("plot_fmax_hz")
        floor = np.finfo(float).tiny
        ratio = comparison_transfer[:, selected_columns] / np.maximum(
            baseline_transfer[:, selected_columns], floor
        )
        np.savez_compressed(
            output_dir / "gaus_baseline_vs_asym_spectra.npz",
            frequency_hz=frequency,
            baseline_label=np.array(baseline_label),
            comparison_label=np.array(comparison_label),
            baseline_source_processed_amplitude=baseline_source_amplitude,
            comparison_source_processed_amplitude=comparison_source_amplitude,
            selected_source_probe_indices=selected_ids,
            selected_probe_x_cm=selected_x,
            baseline_transfer_magnitude=baseline_transfer[:, selected_columns],
            comparison_transfer_magnitude=comparison_transfer[:, selected_columns],
            comparison_to_baseline_transfer_ratio=ratio,
            valid_transfer_frequency=valid_frequency,
        )
        pdb.plot_case_source_spectrum_comparison(
            frequency, baseline_source_amplitude, comparison_source_amplitude,
            baseline_label, comparison_label,
            output_path=str(output_dir / "source_fft_comparison.png"),
            fmax=plot_fmax,
        )
        pdb.plot_case_transfer_comparison(
            frequency, baseline_transfer[:, selected_columns],
            comparison_transfer[:, selected_columns], valid_frequency,
            station_labels, baseline_label, comparison_label,
            output_path=str(output_dir / "fluid_transfer_comparison.png"),
            fmax=plot_fmax,
        )
        _ts(
            f"  Saved {comparison_label} versus {baseline_label} source and "
            "fluid-response comparison products"
        )
        return (label, True, None)
    except Exception as exc:
        return (label, False, str(exc))


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

    _configure_plot_style(config)

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
    _ts(f"  Pulse deconv:     {'ON  ✓' if config.get('make_source_response_analysis', False) else 'OFF ✗'}")
    _ts(f"  Spatial FFT:      {'ON  ✓' if config.get('make_spatial_fft', False) else 'OFF ✗'}")
    _ts(f"  Spatial compare:  {'ON  ✓' if config.get('make_spatial_case_comparison', False) else 'OFF ✗'}")
    _ts(f"  Spectrum compare: {'ON  ✓' if config.get('make_case_spectrum_comparison', False) else 'OFF ✗'}")
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

    # --- Certified flat-plate force baseline ---
    force_baseline_wall = None
    force_baseline_profiles = []
    force_baseline_error = None
    paired_force_paths = {}
    force_baseline_inputs = []
    if config.get("make_force_analysis", False):
        force_cfg = config["force"]
        baseline_cfg = force_cfg["baseline"]
        baseline_mode = baseline_cfg.get("mode", "none")
        geometry = force_cfg["geometry"]
        reference = force_cfg["reference"]
        transport = force_cfg["transport"]
        wall_fit = force_cfg["wall_fit"]
        try:
            if baseline_mode == "static":
                baseline_source = (
                    baseline_cfg.get("data_source") or config["data_source"]
                )
                baseline_path = os.path.join(
                    baseline_source, baseline_cfg["plotfile"]
                )
                _ts(
                    "Loading certified static force baseline: "
                    f"{baseline_cfg['plotfile']}"
                )
                force_baseline_wall = fdb.extract_native_flat_plate_wall(
                    baseline_path,
                    x_range_m=geometry["x_range_m"],
                    wall_y_m=geometry["wall_y_m"],
                    p_inf_pa=reference["p_inf"],
                    rho_inf_kg_m3=reference["rho_inf"],
                    u_inf_m_s=reference["u_inf"],
                    viscosity_pa_s=transport["mu_pa_s"],
                    pressure_order=wall_fit["pressure_order"],
                    velocity_order=wall_fit["velocity_order"],
                    fluid_points=wall_fit["fluid_points"],
                    viscosity_relative_tolerance=wall_fit[
                        "viscosity_relative_tolerance"
                    ],
                )
                # Fail before the production loop if baseline wall coverage or
                # reference conventions are invalid.
                fdb.integrate_flat_plate_wall_forces(
                    force_baseline_wall,
                    _certified_force_reference(config),
                    minimum_coverage=force_cfg["quality"]["minimum_coverage"],
                    maximum_gap_widths=force_cfg["quality"][
                        "maximum_gap_widths"
                    ],
                )
                force_baseline_inputs.append(baseline_path)
                bl_cfg = force_cfg["boundary_layer"]
                if bl_cfg.get("enabled", False):
                    force_baseline_profiles = (
                        fdb.extract_native_flat_plate_boundary_layers(
                            baseline_path,
                            stations_m=bl_cfg["stations_m"],
                            maximum_height_m=bl_cfg["maximum_height_m"],
                            wall_y_m=geometry["wall_y_m"],
                            wall_temperature_k=transport[
                                "wall_temperature_k"
                            ],
                            viscosity_pa_s=transport["mu_pa_s"],
                            conductivity_w_m_k=transport["k_w_m_k"],
                            fluid_points=wall_fit["fluid_points"],
                        )
                    )
                _ts("Certified static force baseline loaded.")
            elif baseline_mode == "paired":
                paired_paths = fdb.discover_plotfile_paths(
                    baseline_cfg["paired_data_source"],
                    plot_prefix=baseline_cfg["paired_plot_prefix"],
                )
                paired_times = np.asarray([
                    fdb.read_amrex_plotfile_time(path) for path in paired_paths
                ])
                tolerance = float(baseline_cfg["time_tolerance_s"])
                used_paired_indices = set()
                for current_path in plotfile_paths:
                    current_time = fdb.read_amrex_plotfile_time(current_path)
                    candidates = np.flatnonzero(
                        np.abs(paired_times - current_time) <= tolerance
                    )
                    if candidates.size != 1:
                        raise ValueError(
                            f"expected exactly one paired force baseline for "
                            f"{os.path.basename(current_path)} within "
                            f"{tolerance:g} s; found {candidates.size}"
                        )
                    index = int(candidates[0])
                    if index in used_paired_indices:
                        raise ValueError(
                            "paired force baseline mapping is not one-to-one"
                        )
                    used_paired_indices.add(index)
                    paired_force_paths[current_path] = paired_paths[index]
                force_baseline_inputs.extend(paired_paths)
                _ts(
                    f"Matched {len(paired_force_paths)} paired force baselines."
                )
        except Exception as exc:
            force_baseline_error = str(exc)
            fdb._log_error("Certified force baseline failed", exc)

    # ------------------------------------------------------------------
    # 2. Process each plotfile individually
    # ------------------------------------------------------------------
    _section(2, 10, "Running workflows on each snapshot ...")

    contour_results = []
    pprime_results = []
    line_results = []
    streamline_results = []
    surface_results = []
    force_results = []
    force_wall_dict = {}
    force_increment_wall_dict = {}
    boundary_layer_results = {}
    surface_data_dict = {}
    forces_dict = {}
    analysis_results = []
    if baseline_error is not None:
        pprime_results.append(("pprime_baseline", False, baseline_error))
    if force_baseline_error is not None:
        force_results.append((
            "force_baseline", False, force_baseline_error
        ))

    # Count enabled workflows for progress reporting
    enabled_workflows = sum([
        1 if config["make_contour_plots"] else 0,
        1 if config["make_line_profiles"] else 0,
        1 if config["make_streamlines"] else 0,
        1 if config["make_surface_analysis"] else 0,
        1 if config.get("make_force_analysis", False) else 0,
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
                derive_native_vorticity=(
                    _config_needs_native_vorticity(config)
                ),
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
            if config.get("make_force_analysis", False):
                force_results.append(failed)
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

        # --- Certified one-sided flat-plate force analysis ---
        if (
            config.get("make_force_analysis", False)
            and force_baseline_error is None
        ):
            t_force = time.time()
            paired_path = paired_force_paths.get(pfile)
            try:
                r = _process_single_certified_force((
                    pfile, config, force_baseline_wall, paired_path
                ))
                force_results.append(r)
                if r[1]:
                    payload = r[2]
                    forces_dict[label] = payload["force"]
                    force_wall_dict[label] = payload["wall"]
                    if payload.get("increment_wall") is not None:
                        force_increment_wall_dict[label] = payload[
                            "increment_wall"
                        ]
                    if payload.get("boundary_layers"):
                        boundary_layer_results[label] = payload[
                            "boundary_layers"
                        ]
                    _ts(
                        f"├─ Certified forces → OK "
                        f"({time.time() - t_force:.1f} s)"
                    )
                else:
                    _ts(f"├─ ✗ Certified forces → FAIL — {r[2]}")
            except Exception as exc:
                force_results.append((label, False, str(exc)))
                _ts(f"├─ ✗ Certified forces → FAIL — {exc}")

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
        compact_plot_data = None
        compact_file = config.get("probe_compact_file")
        if compact_file:
            if not HDF5_PROBE_STORE_AVAILABLE:
                raise ImportError(
                    "Compact HDF5 probe input requires the optional h5py package"
                )
            compact_path = Path(compact_file).expanduser()
            if not compact_path.is_absolute():
                compact_path = Path(config.get("data_source", ".")) / compact_path
            with HDF5ProbeStore(
                    compact_path, field="p", max_probes=probe_max) as store:
                available = set(store.field_names)
                required = ("rho", "u", "p", "T")
                missing = sorted(set(required) - available)
                if missing:
                    raise ValueError(
                        f"Compact probe plot source lacks fields: {missing}"
                    )
                matrices = {field: store.read(field) for field in required}
                compact_plot_data = {
                    probe: {
                        "probe_id": probe,
                        "probe_x": float(store.x[probe]),
                        "probe_y": float(store.y[probe]),
                        "time": store.time,
                        **{
                            field: matrices[field][:, probe]
                            for field in required
                        },
                    }
                    for probe in range(len(store))
                }

        native_plot_data = None
        if compact_plot_data is None and config.get("probe_bin_files"):
            bin_files = _expand_probe_binary_files(
                config["probe_bin_files"]
            )
            if bin_files and {_probe_binary_version(path) for path in bin_files} == {2}:
                _ts("Loading chunked binary probes for time-history plots.")
                field_columns = {"rho": 1, "u": 2, "p": 3, "T": 4}
                field_data = {
                    name: _load_probe_data_from_binary(
                        config, column, max_probes=probe_max
                    )
                    for name, column in field_columns.items()
                }
                counts = {name: len(data) for name, data in field_data.items()}
                if len(set(counts.values())) != 1:
                    raise ValueError(
                        "Chunked probe field counts differ: "
                        f"{counts}"
                    )
                native_plot_data = {
                    probe_id: {
                        "probe_id": probe_id,
                        "probe_x": field_data["rho"][probe_id]["x"],
                        "probe_y": field_data["rho"][probe_id]["y"],
                        "time": field_data["rho"][probe_id]["time"],
                        **{
                            name: field_data[name][probe_id]["signal"]
                            for name in field_columns
                        },
                    }
                    for probe_id in range(counts["rho"])
                }

        # Check if conversion already done
        dat_files = list(probe_out.glob("flow_probe_*.dat"))

        if compact_plot_data is not None:
            probe_out = None
            _ts("Using compact HDF5 directly for probe-history plots.")
        elif native_plot_data is not None:
            probe_out = None
            _ts("Using chunked binary probes directly for time-history plots.")
        elif len(dat_files) == 0:
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
        if compact_plot_data is not None:
            pdb.plot_probe_timeseries(
                compact_plot_data,
                output_dir=out_dir / "Probes",
                laser_start_time=config.get("laser_start_time"),
                convert_to_mks=config.get("probe_convert_to_mks", False),
                station_time_window_s=config.get("probe_station_time_window_s"),
            )
            _ts(f"Probe plots saved to {out_dir / 'Probes'}")
        elif native_plot_data is not None:
            pdb.plot_probe_timeseries(
                native_plot_data,
                output_dir=out_dir / "Probes",
                laser_start_time=config.get("laser_start_time"),
                convert_to_mks=config.get("probe_convert_to_mks", False),
                station_time_window_s=config.get("probe_station_time_window_s"),
            )
            _ts(f"Probe plots saved to {out_dir / 'Probes'}")
        elif probe_out is not None:
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
                        station_time_window_s=config.get("probe_station_time_window_s"),
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
        config.get("make_spatial_fft", False),
        (
            config.get("make_stability_diagnostics", False)
            and not config.get("stability_gip_only", False)
        ),
        config.get("make_transient_analysis", False),
        config.get("make_nonlinear_diagnostics", False),
    ))
    if probe_workflows_enabled:
        _ts("Loading shared probe dataset for enabled analysis workflows ...")
        try:
            shared_max_probes = (
                None if (
                    config.get("make_stability_diagnostics", False)
                    and not config.get("stability_gip_only", False)
                )
                else config.get("fft_max_probes")
            )
            shared_probe_data = _load_probe_timeseries(
                config,
                int(config.get("fft_var_col", 3)),
                nt_skip=config.get("fft_nt_skip", 0),
                max_probes=shared_max_probes,
            )
            if not shared_probe_data:
                raise ValueError("No probe data could be loaded")
            _ts(
                f"Shared probe dataset ready: {len(shared_probe_data)} probes; "
                "subsequent workflows will not rescan the binaries"
            )
            mapping_products = _write_probe_mapping_products(
                shared_probe_data, out_dir
            )
            if mapping_products is not None:
                _ts(
                    "Probe AMR mapping report saved: "
                    f"{mapping_products[0]}"
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

    if config.get("make_spatial_fft", False):
        _ts("Running spatial FFT probe analysis ...")
        try:
            r = _process_spatial_fft_probes(config, shared_probe_data)
            analysis_results.append(r)
            if r[1]:
                _ts("Spatial FFT probe analysis -> OK")
            else:
                _ts(f"✗ Spatial FFT probe analysis -> FAIL — {r[2]}")
        except Exception as exc:
            analysis_results.append(("spatial_fft", False, str(exc)))
            _ts(f"✗ Spatial FFT probe analysis -> FAIL — {exc}")

    if config.get("make_spatial_case_comparison", False):
        _ts("Running spatial Gaus_JW-baseline case comparison ...")
        r = _process_spatial_case_comparison(config)
        analysis_results.append(r)
        if r[1]:
            _ts("Spatial Gaus_JW-baseline case comparison -> OK")
        else:
            _ts(f"✗ Spatial case comparison -> FAIL — {r[2]}")

    if config.get("make_case_spectrum_comparison", False):
        _ts("Running Gaus_JW-baseline case spectrum comparison ...")
        r = _process_case_spectrum_comparison(config)
        analysis_results.append(r)
        if r[1]:
            _ts("Gaus_JW-baseline case comparison -> OK")
        else:
            _ts(f"✗ Gaus_JW-baseline case comparison -> FAIL — {r[2]}")

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

    if isinstance(shared_probe_data, HDF5ProbeStore):
        shared_probe_data.close()
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
    if config.get("make_force_analysis", False):
        _print_results(force_results, "Certified force analysis")
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

    force_report = None
    if config["make_force_analysis"] and forces_dict:
        try:
            force_report = _write_certified_force_products(
                config, forces_dict, force_wall_dict,
                force_increment_wall_dict, boundary_layer_results,
                baseline_wall=force_baseline_wall,
                baseline_profiles=force_baseline_profiles,
            )
            _ts(
                "Certified force products saved to "
                f"{out_dir / 'ForceAnalysis'}"
            )
            if not force_report["scientific_adequacy"][
                    "spectral_inference_allowed"]:
                _ts(
                    "Force spectra withheld: "
                    + force_report["scientific_adequacy"]["linkage"]["reason"]
                )
        except Exception as exc:
            fdb._log_error("Force time-series plot failed", exc)
            force_results.append(("force_products", False, str(exc)))
    else:
        _ts("Force analysis not requested or no force data available.")

    if config.get("make_evidence_classification", True):
        try:
            evidence_report = _write_analysis_evidence_report(
                config, analysis_results, force_report=force_report
            )
            analysis_results.append(("evidence", True, None))
            _ts(
                "Measurement evidence matrix saved: "
                f"{out_dir / 'AnalysisEvidence'}"
            )
            if not evidence_report["supported_classifications"]:
                _ts(
                    "  Evidence matrix contains no supported classification; "
                    "inspect its unavailable/insufficient-data entries."
                )
        except Exception as exc:
            analysis_results.append(("evidence", False, str(exc)))
            fdb._log_error("Analysis evidence classification failed", exc)

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
            anim_field = config.get(
                "animation_field",
                config.get("contour_fields", ["mach_number"])[0],
            )
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
                    ds = fdb.load_pelec_plotfile(
                        pfile,
                        convert_to_mks=True,
                        derive_native_vorticity=(
                            anim_field == "vorticity"
                        ),
                    )
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
                anim_out = str(anim_dir / f"loading_evolution.gif")
                pdb.animate_contour_with_surface(
                    anim_datasets,
                    field_key=anim_field,
                    surfaces_series=anim_surfaces if anim_surfaces else None,
                    surface_data_series=surface_data_dict,
                    loading_key=config.get("animation_surface_key", "C_p"),
                    output_path=anim_out,
                    xlim=config.get("contour_xlim"),
                    ylim=config.get("contour_ylim"),
                    fps=config.get("animation_fps", 8),
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
        streamline_results, surface_results, force_results, analysis_results,
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
        plotfiles=[*plotfile_paths, *force_baseline_inputs],
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
        action="append",
        help=(
            "partial JSON configuration overlay; repeat in precedence order "
            "(unknown keys are rejected)"
        ),
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
