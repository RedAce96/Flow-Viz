# =============================================================================
#  PeleC Post-Processing: Plotting Database
# =============================================================================
#  Core plotting functions for PeleC / AMReX plotfile post-processing.
#
#  All functions accept the canonical ``StandardDataset`` dicts produced by
#  ``pp_functions_database.py`` so they are solver-agnostic.
#
#  For interns / new GRAs:
#    - You should NOT need to edit this file for normal use.
#    - All user settings go in ``pelec_post.py`` (the execution script).
# =============================================================================

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter

import pp_functions_database as fdb_mod


# ---------------------------------------------------------------------------
# 0.  PUBLICATION STYLE AND LABELS
# ---------------------------------------------------------------------------

_FIELD_TITLES = {
    "density": "Density", "pressure": "Pressure",
    "temperature": "Temperature", "x_velocity": "Streamwise velocity",
    "y_velocity": "Wall-normal velocity",
    "velocity_magnitude": "Velocity magnitude", "mach_number": "Mach number",
    "vorticity": "Vorticity", "vorticity_magnitude": "Vorticity magnitude",
    "schlieren": "Numerical schlieren", "C_p": "Pressure coefficient",
    "C_f": "Skin-friction coefficient",
    "delta_99": r"Boundary-layer thickness $\delta_{99}$",
}

_FIELD_LABELS = {
    "density": r"$\rho$ [kg m$^{-3}$]", "pressure": r"$p$ [Pa]",
    "temperature": r"$T$ [K]", "x_velocity": r"$u$ [m s$^{-1}$]",
    "y_velocity": r"$v$ [m s$^{-1}$]",
    "velocity_magnitude": r"$|\mathbf{u}|$ [m s$^{-1}$]",
    "mach_number": r"$M$", "vorticity": r"$\omega_z$ [s$^{-1}$]",
    "vorticity_magnitude": r"$|\omega|$ [s$^{-1}$]", "C_p": r"$C_p$",
    "C_f": r"$C_f$", "delta_99": r"$\delta_{99}$ [m]",
}


def configure_plot_style(overrides=None):
    """Apply one presentation-readable, publication-oriented figure style."""
    style = {
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 16, "axes.labelsize": 16, "axes.linewidth": 1.0,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 14, "legend.frameon": False,
        "legend.labelcolor": "linecolor",
        "lines.linewidth": 2.0, "savefig.dpi": 300,
        "savefig.bbox": "tight", "figure.dpi": 120,
    }
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def field_title(field_key):
    """Return a human-readable title for a canonical field name."""
    return _FIELD_TITLES.get(field_key, str(field_key).replace("_", " ").title())


def field_label(field_key):
    """Return a symbol-and-unit label for an MKS canonical field."""
    return _FIELD_LABELS.get(field_key, field_title(field_key))


def format_flow_time(time_seconds, reference_time=None, origin=0.0):
    """Format physical flow time, or nondimensional time when a scale is set."""
    if time_seconds is None or not np.isfinite(time_seconds):
        return ""
    elapsed = float(time_seconds) - float(origin)
    if reference_time is not None:
        reference_time = float(reference_time)
        if reference_time <= 0:
            raise ValueError("reference_time must be positive")
        return rf"$t^* = {elapsed / reference_time:.4g}$"
    if abs(elapsed) >= 1.0:
        return rf"$t = {elapsed:.4g}$ s"
    if abs(elapsed) >= 1.0e-3:
        return rf"$t = {elapsed * 1.0e3:.4g}$ ms"
    if abs(elapsed) >= 1.0e-6:
        return rf"$t = {elapsed * 1.0e6:.4g}$ $\mu$s"
    return rf"$t = {elapsed:.4e}$ s"


def streamwise_domain_length(dataset):
    """Return the full streamwise domain length represented by a dataset."""
    explicit = dataset.get("domain_length_x")
    if explicit is not None and np.isfinite(explicit) and explicit > 0:
        return float(explicit)
    x = np.asarray(dataset.get("x", []), dtype=float)
    if x.size < 2:
        raise ValueError("At least two streamwise coordinates are required")
    dx = float(np.nanmedian(np.diff(x)))
    return float(np.nanmax(x) - np.nanmin(x) + abs(dx))


def format_dataset_time(dataset, mode="physical", freestream_velocity=None,
                        reference_time=None, origin=0.0):
    """Format dataset time in physical, reference, or flow-through units."""
    time_seconds = dataset.get("time")
    if mode == "flow_through":
        if freestream_velocity is None or float(freestream_velocity) <= 0:
            raise ValueError("A positive freestream_velocity is required")
        t_flow_through = streamwise_domain_length(dataset) / float(freestream_velocity)
        value = (float(time_seconds) - float(origin)) / t_flow_through
        return rf"$t = {value:.2f}\,L/U_\infty$"
    if mode not in ("physical", "reference"):
        raise ValueError(f"Unknown time display mode: {mode}")
    return format_flow_time(
        time_seconds,
        reference_time=reference_time if mode == "reference" else None,
        origin=origin,
    )


def dataset_title(dataset, subject, reference_time=None, time_origin=0.0,
                  time_mode="physical", freestream_velocity=None):
    """Build a scientific title from a subject and the dataset flow time."""
    time_text = format_dataset_time(
        dataset,
        mode=time_mode,
        freestream_velocity=freestream_velocity,
        reference_time=reference_time,
        origin=time_origin,
    )
    return f"{field_title(subject)} — {time_text}" if time_text else field_title(subject)


def annotate_plot_time(ax, time_text, location="upper left", fontsize=None):
    """Place a time coordinate inside an axes without using a plot title."""
    if not time_text:
        return None
    locations = {
        "upper left": (0.025, 0.965, "left", "top"),
        "upper right": (0.975, 0.965, "right", "top"),
        "lower left": (0.025, 0.035, "left", "bottom"),
        "lower right": (0.975, 0.035, "right", "bottom"),
    }
    if location not in locations:
        raise ValueError(
            "time annotation location must be upper/lower left/right"
        )
    x_pos, y_pos, horizontal, vertical = locations[location]
    return ax.text(
        x_pos, y_pos, time_text,
        transform=ax.transAxes,
        ha=horizontal, va=vertical,
        fontsize=(
            plt.rcParams["font.size"] if fontsize is None else fontsize
        ),
        zorder=20,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.7,
            "alpha": 0.85,
        },
    )


configure_plot_style()


# ---------------------------------------------------------------------------
# 1.  COLOURMAP HELPERS
# ---------------------------------------------------------------------------

def get_custom_colormaps():
    """Return a dict of custom colormaps for post-processing.

    Returns
    -------
    dict
        Keys: ``my_reds``, ``my_blues``, ``my_reds_r``, ``my_blues_r``,
        ``Blue2Red``.
    """
    my_reds = mcolors.LinearSegmentedColormap.from_list(
        "my_reds",
        ["crimson", "red", "orangered", "salmon", "pink", "white", (1, 1, 1, 0.0)],
        N=256,
    )
    my_reds_r = mcolors.LinearSegmentedColormap.from_list(
        "my_reds_r",
        ["crimson", "red", "orangered", "salmon", "pink", "white", (1, 1, 1, 0.0)][::-1],
        N=256,
    )
    my_blues = mcolors.LinearSegmentedColormap.from_list(
        "my_blues",
        ["navy", "blue", "dodgerblue", "skyblue", "lightblue", "white", (1, 1, 1, 0.0)],
        N=256,
    )
    my_blues_r = mcolors.LinearSegmentedColormap.from_list(
        "my_blues_r",
        ["navy", "blue", "dodgerblue", "skyblue", "lightblue", "white", (1, 1, 1, 0.0)][::-1],
        N=256,
    )
    Blue2Red = mcolors.LinearSegmentedColormap.from_list(
        "Blue2Red",
        ["navy", "blue", "dodgerblue", "skyblue", "lightblue",
         (1, 1, 1, 0.0), "pink", "salmon", "orangered", "red", "crimson"],
        N=256,
    )
    return {
        "my_reds": my_reds,
        "my_blues": my_blues,
        "my_reds_r": my_reds_r,
        "my_blues_r": my_blues_r,
        "Blue2Red": Blue2Red,
    }


def resolve_cmap(cmap):
    """Resolve a colormap name to a Colormap object.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        If a string, it is first looked up in the custom colormap dict
        (from ``get_custom_colormaps``), then falls back to matplotlib's
        built-in ``get_cmap``.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if isinstance(cmap, mcolors.Colormap):
        return cmap

    custom = get_custom_colormaps()
    if cmap in custom:
        return custom[cmap]
    # Compatible with Matplotlib >= 3.9 (get_cmap removed from cm module)
    try:
        return plt.get_cmap(cmap)
    except TypeError:
        # Very old fallback for pre-3.5
        return cm.get_cmap(cmap)


# ---------------------------------------------------------------------------
# 1.  BASIC CONTOUR PLOTS
# ---------------------------------------------------------------------------

def _resolve_colorbar_shrink(value, x, xlim=None, reference_span=0.3):
    """Resolve a fixed or streamwise-span-aware colorbar height fraction."""
    if not isinstance(value, str):
        shrink = float(value)
        if not 0.0 < shrink <= 1.0:
            raise ValueError("colorbar_shrink must lie in (0, 1]")
        return shrink
    if value.lower() != "auto":
        raise ValueError("colorbar_shrink must be 'auto' or a number")
    reference_span = float(reference_span)
    if not np.isfinite(reference_span) or reference_span <= 0.0:
        raise ValueError("colorbar_reference_span must be positive")
    limits = np.asarray(xlim if xlim is not None else [np.min(x), np.max(x)])
    displayed_span = abs(float(limits[1]) - float(limits[0]))
    if not np.isfinite(displayed_span) or displayed_span <= 0.0:
        return 0.5
    # A <= reference-span view uses a half-height colorbar. Wider views
    # decrease smoothly, reaching the 0.2 lower bound at 2.5 reference spans.
    return float(np.clip(0.5 * reference_span / displayed_span, 0.2, 0.5))


def _resolve_contour_font_scale(value, x, xlim=None, reference_span=0.2):
    """Scale contour typography gently for unusually wide displayed spans."""
    if not isinstance(value, str):
        scale = float(value)
        if scale <= 0.0:
            raise ValueError("contour font scale must be positive")
        return scale
    if value.lower() != "auto":
        raise ValueError("contour font scale must be 'auto' or a number")
    reference_span = float(reference_span)
    if not np.isfinite(reference_span) or reference_span <= 0.0:
        raise ValueError("contour font reference span must be positive")
    limits = np.asarray(xlim if xlim is not None else [np.min(x), np.max(x)])
    displayed_span = abs(float(limits[1]) - float(limits[0]))
    if not np.isfinite(displayed_span) or displayed_span <= reference_span:
        return 1.0
    # Fourth-root scaling is deliberately gentle: a 0.4 m view referenced to
    # 0.2 m uses 84% fonts, while very wide plots never fall below 75%.
    return float(np.clip(
        (reference_span / displayed_span) ** 0.25, 0.75, 1.0
    ))

def plot_contour(dataset, field_key, output_path=None, figsize=(14, 4),
                 cmap="viridis", vmin=None, vmax=None, title=None,
                 xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", colorbar_label=None,
                 xlim=None, ylim=None, ax=None, rasterized=True,
                 norm="linear", colorbar_shrink="auto",
                 colorbar_reference_span=0.3, time_annotation=None,
                 time_annotation_location="upper left",
                 font_scale="auto", font_reference_span=0.2,
                 colorbar_font_scale=0.85):
    """Plot a single 2-D field as a pcolormesh contour.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    field_key : str
        Canonical field name to plot.
    output_path : str, optional
        If provided, save the figure to this path.
    figsize : tuple, default (10, 6)
    cmap : str or Colormap, default 'viridis'
    vmin, vmax : float, optional
        Color limits.  Auto-computed from percentiles if omitted.
    title : str, optional
        Retained for compatibility. Production plots leave this unset and use
        ``time_annotation`` inside the axes.
    xlabel, ylabel : str
    colorbar_label : str, optional
        Defaults to ``field_key``.
    xlim, ylim : tuple, optional
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw into.
    rasterized : bool, default True
        Rasterize the pcolormesh for smaller file sizes.
    norm : str or Normalize, default "linear"
        Color scaling: "linear", "log" (LogNorm), or "symlog" (SymLogNorm).
    colorbar_shrink : "auto" or float, default "auto"
        Colorbar height fraction. Automatic mode decreases the fraction as
        the displayed x-span grows.
    colorbar_reference_span : float, default 0.3
        X-span receiving the maximum automatic colorbar height.

    Returns
    -------
    matplotlib.axes.Axes
    """
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True

    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    field = np.asarray(dataset["fields"][field_key], dtype=float)
    if field.shape != (x.size, y.size):
        raise ValueError(
            f"Field shape {field.shape} does not match grid ({x.size}, {y.size})"
        )

    if field_key == "vorticity" and vmin is None and vmax is None:
        # A signed curl must use limits symmetric about zero; independent
        # percentile limits can visually bias one rotation direction.
        bound = np.nanpercentile(np.abs(field), 99)
        vmin, vmax = -bound, bound
    else:
        if vmin is None:
            vmin = np.nanpercentile(field, 1)
        if vmax is None:
            vmax = np.nanpercentile(field, 99)

    # Resolve color normalisation
    if isinstance(norm, str):
        if norm == "log":
            if vmin <= 0:
                pos = field[field > 0]
                vmin = np.nanpercentile(pos, 1) if len(pos) else 1e-3 * vmax
            p_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        elif norm == "symlog":
            linthresh = max(abs(vmin), abs(vmax)) / 1000 if vmin != vmax else 1e-3
            p_norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        else:
            p_norm = None
    else:
        p_norm = norm

    p = ax.pcolormesh(
        x, y, field.T,
        cmap=cmap,
        shading="auto",
        norm=p_norm,
        vmin=vmin if p_norm is None else None,
        vmax=vmax if p_norm is None else None,
        rasterized=rasterized,
    )
    resolved_shrink = _resolve_colorbar_shrink(
        colorbar_shrink, x, xlim=xlim,
        reference_span=colorbar_reference_span,
    )
    cb = ax.figure.colorbar(
        p, ax=ax, pad=0.02, shrink=resolved_shrink
    )
    resolved_font_scale = _resolve_contour_font_scale(
        font_scale, x, xlim=xlim, reference_span=font_reference_span
    )
    colorbar_font_scale = float(colorbar_font_scale)
    if colorbar_font_scale <= 0.0:
        raise ValueError("colorbar_font_scale must be positive")
    axes_label_size = float(plt.rcParams["axes.labelsize"])
    tick_label_size = float(plt.rcParams["xtick.labelsize"])
    cb.set_label(
        colorbar_label if colorbar_label is not None else field_label(field_key),
        fontsize=(
            axes_label_size * resolved_font_scale * colorbar_font_scale
        ),
    )
    cb.ax.tick_params(
        labelsize=(
            tick_label_size * resolved_font_scale * colorbar_font_scale
        )
    )

    ax.set_xlabel(xlabel, fontsize=axes_label_size * resolved_font_scale)
    ax.set_ylabel(ylabel, fontsize=axes_label_size * resolved_font_scale)
    ax.tick_params(labelsize=tick_label_size * resolved_font_scale)
    if title:
        ax.set_title(title)
    annotate_plot_time(
        ax, time_annotation, location=time_annotation_location,
        fontsize=float(plt.rcParams["font.size"]) * resolved_font_scale,
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
    elif created_figure:
        ax.figure.tight_layout()

    return ax


# ---------------------------------------------------------------------------
# 2.  LINE / PROFILE PLOTS
# ---------------------------------------------------------------------------

def plot_line_profiles(profiles, field_label=None, xlabel=r"$y$ [m]",
                       title=None, output_path=None, ax=None,
                       x_key="y", linewidth=2.0, linestyle="-",
                       swap_axes=False, coordinate_normalization=None,
                       annotate_boundary_layer=True, coordinate_limits=None,
                       time_annotation=None,
                       time_annotation_location="upper left"):
    """Plot one or more 1-D line profiles on a shared axis.

    Parameters
    ----------
    profiles : list[dict]
        Profile dicts returned by ``extract_line`` or
        ``extract_surface_normal_profile``.
    field_label : str, optional
        Legend label for the field.  Defaults to the first profile's key.
    xlabel : str, default 'y [m]'
        Horizontal-axis label.
    title : str, optional
    output_path : str, optional
    ax : matplotlib.axes.Axes, optional
    x_key : str, default 'y'
        Key to use for the horizontal axis from each profile.
    linewidth : float, default 2.0
    linestyle : str, default '-'
    swap_axes : bool, default False
        If True, plot ``values`` on the x-axis and ``x_key`` on the
        y-axis, and swap the axis labels accordingly.
    coordinate_normalization : str, optional
        Profile key containing a length scale. For example,
        ``"boundary_layer_height"`` plots ``y / delta_99``.
    annotate_boundary_layer : bool, default True
        Mark the supplied boundary-layer height on dimensional profiles.
    coordinate_limits : tuple, optional
        Limits for the profile coordinate (y-axis with ``swap_axes=True``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(profiles) == 0:
        raise ValueError("At least one profile is required.")

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_figure = True

    for profile in profiles:
        label = profile.get("label", profile.get("field_key", "Profile"))
        coordinate = np.asarray(profile[x_key], dtype=float)
        if coordinate_normalization is not None:
            coordinate_scale = profile.get(coordinate_normalization)
            if (coordinate_scale is not None
                    and np.isfinite(coordinate_scale)
                    and coordinate_scale > 0):
                coordinate = coordinate / float(coordinate_scale)
        if swap_axes:
            ax.plot(
                profile["values"],
                coordinate,
                linewidth=linewidth,
                linestyle=profile.get("linestyle", linestyle),
                color=profile.get("color", None),
                label=label,
            )
        else:
            ax.plot(
                coordinate,
                profile["values"],
                linewidth=linewidth,
                linestyle=profile.get("linestyle", linestyle),
                color=profile.get("color", None),
                label=label,
            )

        # Boundary-layer height marker (if supplied by the caller)
        bl_height = profile.get("boundary_layer_height")
        if annotate_boundary_layer and bl_height is not None:
            y_arr = np.asarray(profile[x_key], dtype=float)
            val_arr = np.asarray(profile["values"], dtype=float)
            if len(y_arr) >= 2:
                val_at_d99 = float(np.interp(float(bl_height), y_arr, val_arr))
                if swap_axes:
                    marker_x, marker_y = val_at_d99, float(bl_height)
                else:
                    marker_x, marker_y = float(bl_height), val_at_d99
                ax.plot(
                    marker_x, marker_y,
                    marker="o", color="black", markersize=6,
                    linestyle="None",
                )
                ax.annotate(
                    f"{x_key}={float(bl_height):.6f}",
                    xy=(marker_x, marker_y),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=12, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )

    if field_label is None:
        key = profiles[0].get("field_key", "Field")
        field_label = _FIELD_LABELS.get(key, field_title(key))

    if swap_axes:
        ax.set_xlabel(field_label)
        ax.set_ylabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(field_label)
    if title is not None:
        ax.set_title(title)
    annotate_plot_time(
        ax, time_annotation, location=time_annotation_location
    )

    if coordinate_normalization is not None:
        if swap_axes:
            ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.0,
                       label=r"$y/\delta_{99}=1$")
        else:
            ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0,
                       label=r"$y/\delta_{99}=1$")

    if coordinate_limits is not None:
        if swap_axes:
            ax.set_ylim(coordinate_limits)
        else:
            ax.set_xlim(coordinate_limits)

    if any(p.get("label") for p in profiles):
        ax.legend()

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
    elif created_figure:
        ax.figure.tight_layout()

    return ax


# ---------------------------------------------------------------------------
# 3.  STREAMLINE PLOTS
# ---------------------------------------------------------------------------

def plot_streamlines(streamline_sets, title=None, output_path=None,
                     xlim=None, ylim=None, density=1.5,
                     linewidth=1.0, arrowsize=1.0, color=None,
                     cmap="viridis", start_points=None,
                     seed_count=None, seed_x_position=None,
                     seed_y_min=None, seed_y_max=None,
                     seed_spacing="uniform", seed_power=2.0,
                     broken_streamlines=True, colorbar=None,
                     colorbar_label=None, mask_fill_color="black",
                     mask_fill_alpha=1.0, ax=None, time_annotation=None,
                     time_annotation_location="upper left"):
    """Plot one or more steady streamline fields on a shared axis.

    Parameters
    ----------
    streamline_sets : list[dict]
        Streamline payloads returned by ``extract_streamline_field``.
    title : str, optional
    output_path : str, optional
    xlim, ylim : tuple, optional
    density : float, default 1.5
    linewidth : float, default 1.0
    arrowsize : float, default 1.0
    color : str, optional
        Explicit streamline color.  If omitted, a per-case color cycle is used.
    cmap : str, default 'viridis'
        Colormap when a streamline set provides ``color`` data.
    start_points : array-like, optional
        Streamline seed points, shape ``(n, 2)``.
    seed_count : int, optional
        Auto-generate this many seed points at a fixed x-position.
    seed_x_position : float, optional
    seed_y_min, seed_y_max : float, optional
    seed_spacing : {'uniform', 'power', 'sqrt'}, default 'uniform'
    seed_power : float, default 2.0
    broken_streamlines : bool, default True
    colorbar : bool, optional
        Defaults to True when a single color-mapped set is plotted.
    colorbar_label : str, optional
    mask_fill_color : str, default 'black'
    mask_fill_alpha : float, default 1.0
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if len(streamline_sets) == 0:
        raise ValueError("At least one streamline field is required.")

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_figure = True

    if colorbar is None:
        colorbar = len(streamline_sets) == 1 and streamline_sets[0].get("color") is not None

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    legend_handles = []
    colorbar_artist = None

    for idx, sl in enumerate(streamline_sets):
        x = np.asarray(sl["x"], dtype=float)
        y = np.asarray(sl["y"], dtype=float)
        u = np.asarray(sl["u"], dtype=float)
        v = np.asarray(sl["v"], dtype=float)
        c = sl.get("color")
        m = sl.get("mask")

        if m is not None:
            m = np.asarray(m, dtype=bool)
            u = np.ma.masked_where(m, u)
            v = np.ma.masked_where(m, v)
            if c is not None:
                c = np.ma.masked_where(m, np.asarray(c, dtype=float))

        kwargs = {
            "density": density,
            "linewidth": linewidth,
            "arrowsize": arrowsize,
            "broken_streamlines": broken_streamlines,
        }

        if start_points is not None:
            kwargs["start_points"] = np.asarray(start_points, dtype=float)
        elif seed_count is not None:
            kwargs["start_points"] = _generate_streamline_seeds(
                x, y, seed_count, seed_x_position, seed_y_min, seed_y_max,
                seed_spacing, seed_power,
            )

        if c is not None:
            artist = ax.streamplot(
                x, y, u.T, v.T,
                color=np.asarray(c, dtype=float).T,
                cmap=cmap,
                **kwargs,
            )
            if colorbar and colorbar_artist is None:
                colorbar_artist = artist.lines
        else:
            lc = color if color is not None else color_cycle[idx % len(color_cycle)]
            ax.streamplot(x, y, u.T, v.T, color=lc, **kwargs)
            if sl.get("label"):
                legend_handles.append(
                    Line2D([0], [0], color=lc, linewidth=2.0, label=sl["label"])
                )

        if m is not None and np.any(m):
            masked_region = np.ma.masked_where(~m.T, np.ones(m.T.shape, dtype=float))
            ax.pcolormesh(
                x, y, masked_region,
                shading="auto",
                cmap=ListedColormap([mask_fill_color]),
                alpha=mask_fill_alpha,
                zorder=3,
            )

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title, fontsize=14)
    annotate_plot_time(
        ax, time_annotation, location=time_annotation_location
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    if legend_handles:
        ax.legend(handles=legend_handles)

    if colorbar and colorbar_artist is not None:
        cb = ax.figure.colorbar(colorbar_artist, ax=ax)
        cb.set_label(
            colorbar_label if colorbar_label is not None
            else streamline_sets[0].get("color_key", "Field")
        )

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
    elif created_figure:
        ax.figure.tight_layout()

    return ax


def _generate_streamline_seeds(x_line, y_line, count=40,
                               x_position=None, y_min=None, y_max=None,
                               spacing="uniform", power=2.0):
    """Generate streamline seed points with optional wall-biased spacing."""
    x_arr = np.asarray(x_line, dtype=float)
    y_arr = np.asarray(y_line, dtype=float)

    if count <= 0:
        raise ValueError("count must be positive.")

    if x_position is None:
        x_position = float(x_arr[0])
    if y_min is None:
        y_min = float(np.min(y_arr))
    if y_max is None:
        y_max = float(np.max(y_arr))

    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min.")

    eta = np.linspace(0.0, 1.0, int(count))
    spacing = str(spacing).lower()
    if spacing == "uniform":
        y_eta = eta
    elif spacing == "power":
        y_eta = eta ** float(power)
    elif spacing == "sqrt":
        y_eta = np.sqrt(eta)
    else:
        raise ValueError("spacing must be 'uniform', 'power', or 'sqrt'.")

    y_vals = y_min + (y_max - y_min) * y_eta
    x_vals = np.full_like(y_vals, float(x_position), dtype=float)
    return np.column_stack([x_vals, y_vals])


# ---------------------------------------------------------------------------
# 4.  OVERLAY / COMPOSITE PLOTS
# ---------------------------------------------------------------------------

def plot_overlay(dataset, base_key="schlieren", overlay_key="vorticity",
                 base_cmap="gray", overlay_cmap=None,
                 base_vmin=None, base_vmax=None,
                 overlay_vmin=None, overlay_vmax=None,
                 overlay_signed=None, max_alpha=0.8,
                 overlay_deadband=None, mask_out_of_range=True,
                 xlim=None, ylim=None, title=None,
                 output_path=None, figsize=(12, 8),
                 rasterized=True):
    """Plot a base field (e.g. Schlieren) with a semi-transparent overlay.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    base_key : str, default 'schlieren'
    overlay_key : str, default 'vorticity'
    base_cmap : str, default 'gray'
    overlay_cmap : str, optional
        Auto: ``RdBu_r`` if signed, ``turbo`` if non-negative.
    base_vmin, base_vmax : float, optional
    overlay_vmin, overlay_vmax : float, optional
    overlay_signed : bool, optional
        Auto-detected from data if omitted.
    max_alpha : float, default 0.8
    overlay_deadband : float, optional
        Absolute threshold for signed fields (e.g. 0.1).
    mask_out_of_range : bool, default True
        Hide values outside [vmin, vmax] if True.
    xlim, ylim : tuple, optional
    title : str, optional
    output_path : str, optional
    figsize : tuple, default (12, 8)
    rasterized : bool, default True

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    xx, yy = np.meshgrid(dataset["x"], dataset["y"], indexing="ij")
    base = np.asarray(dataset["fields"][base_key], dtype=float).copy()
    overlay = np.asarray(dataset["fields"][overlay_key], dtype=float).copy()

    # Base limits
    if base_vmin is None:
        base_vmin = np.nanpercentile(base, 1)
    if base_vmax is None:
        base_vmax = np.nanpercentile(base, 99)

    # Overlay limits
    finite = np.isfinite(overlay)
    vals = overlay[finite]
    if vals.size == 0:
        raise ValueError(f"No finite values for overlay field '{overlay_key}'.")

    if overlay_signed is None:
        overlay_signed = np.nanmin(vals) < 0.0
    if overlay_cmap is None:
        overlay_cmap = "RdBu_r" if overlay_signed else "turbo"

    if overlay_vmin is None:
        overlay_vmin = np.nanmin(vals)
    if overlay_vmax is None:
        overlay_vmax = np.nanmax(vals)

    if overlay_vmin > overlay_vmax:
        overlay_vmin, overlay_vmax = overlay_vmax, overlay_vmin

    norm = mcolors.Normalize(vmin=overlay_vmin, vmax=overlay_vmax)

    # Show mask
    if mask_out_of_range:
        show_mask = finite & (overlay >= overlay_vmin) & (overlay <= overlay_vmax)
    else:
        show_mask = finite

    if overlay_deadband is not None:
        if overlay_signed:
            show_mask &= np.abs(overlay) >= float(overlay_deadband)
        else:
            show_mask &= overlay >= float(overlay_deadband)

    # Base layer
    p0 = ax.pcolormesh(
        xx, yy, base,
        cmap=base_cmap,
        shading="auto",
        vmin=base_vmin,
        vmax=base_vmax,
        rasterized=rasterized,
        zorder=1,
    )
    cb0 = fig.colorbar(p0, ax=ax, pad=0.02)
    cb0.set_label(base_key, fontsize=12)

    # Overlay layer
    overlay_masked = np.ma.masked_where(~show_mask, overlay)
    p1 = ax.pcolormesh(
        xx, yy, overlay_masked,
        cmap=overlay_cmap,
        shading="auto",
        vmin=overlay_vmin,
        vmax=overlay_vmax,
        alpha=max_alpha,
        rasterized=rasterized,
        zorder=2,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=overlay_cmap)
    sm.set_array([])
    cb1 = fig.colorbar(sm, ax=ax, pad=0.02)
    cb1.set_label(overlay_key, fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()

    return ax


# ---------------------------------------------------------------------------
# 5.  SURFACE / GEOMETRY PLOTS
# ---------------------------------------------------------------------------

def plot_surface_geometry(surfaces, dataset, snapshot="",
                          output_path=None, figsize=(14, 6),
                          ax=None):
    """Plot the detected surface geometry over the velocity field.

    Parameters
    ----------
    surfaces : dict
        Output from ``define_geometry_surface``.
    dataset : dict
        ``StandardDataset`` (used for axis limits and background).
    snapshot : str
        Label for the plot title.
    output_path : str, optional
    figsize : tuple, default (14, 6)
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True

    # Plot surface lines
    for side, color in (("upper", "red"), ("lower", "blue")):
        if len(surfaces[side].get("x", [])) > 0:
            ax.plot(surfaces[side]["x"], surfaces[side]["y"],
                    "o", color=color, markersize=1, alpha=0.3,
                    label=f"{side} (cells)")
            ax.plot(surfaces[side]["x"], surfaces[side]["y_interp"],
                    "-", color=color, linewidth=2,
                    label=f"{side} (interpolated)")

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(f"Surface Geometry Detection — {snapshot}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Set reasonable limits
    all_x = []
    all_y = []
    for side in ("upper", "lower"):
        if len(surfaces[side].get("x", [])) > 0:
            all_x.extend(surfaces[side]["x"])
            all_y.extend(surfaces[side]["y_interp"])
    if all_x:
        ax.set_xlim(min(all_x) - 0.05, max(all_x) + 0.05)
        ax.set_ylim(min(all_y) - 0.02, max(all_y) + 0.02)

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
    elif created_figure:
        ax.figure.tight_layout()

    return ax


def plot_surface_properties(surface_data, snapshot="",
                            output_path=None, figsize=(16, 12),
                            ax=None):
    """Plot surface properties (Cp, Cf, pressure, temperature) in a 2x2 panel.

    Parameters
    ----------
    surface_data : dict
        Output from ``extract_surface_properties``.
    snapshot : str
        Label for the plot title.
    output_path : str, optional
    figsize : tuple, default (16, 12)
    ax : array-like of matplotlib.axes.Axes, optional
        2x2 array of existing axes.

    Returns
    -------
    list[matplotlib.axes.Axes]
    """
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        created_figure = True
    ax = np.asarray(ax).reshape(2, 2)

    colors = {"upper": "red", "lower": "blue"}
    titles = [
        "Pressure Coefficient",
        "Skin Friction Coefficient",
        "Wall Pressure",
        "Wall Heat Flux",
    ]
    ylabels = [r"$C_p$", r"$C_f$", "Pressure [Pa]", r"$q_w$ [W/m$^2$]"]
    keys = ["C_p", "C_f", "p", "q_w"]

    for idx, (title, ylabel, key) in enumerate(zip(titles, ylabels, keys)):
        row, col = idx // 2, idx % 2
        a = ax[row, col]
        for side in ("upper", "lower"):
            if side not in surface_data:
                continue
            data = surface_data[side]
            if len(data["x"]) == 0:
                continue
            a.plot(data["x"], data[key], label=side,
                   color=colors[side], linewidth=2)
        a.set_title(title, fontsize=12)
        a.set_xlabel("x [m]", fontsize=10)
        a.set_ylabel(ylabel, fontsize=10)
        a.legend()
        a.grid(True, alpha=0.3)

    if created_figure:
        plt.suptitle(f"Surface Properties — {snapshot}", fontsize=14)

    if output_path is not None:
        ax[0, 0].figure.tight_layout()
        ax[0, 0].figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(ax[0, 0].figure)
    elif created_figure:
        ax[0, 0].figure.tight_layout()

    return ax


def plot_surface_properties_group(surface_data_dict, snapshot_list,
                                  output_path=None, figsize=(18, 14)):
    """Plot surface properties for multiple snapshots on shared axes.

    Parameters
    ----------
    surface_data_dict : dict
        Keys are snapshot labels, values are surface_data dicts.
    snapshot_list : list
        Ordered list of snapshot labels to plot.
    output_path : str, optional
    figsize : tuple, default (18, 14)

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    side_colors = {"upper": "red", "lower": "blue"}
    linestyles = ["-", "--", "-.", ":"]
    linewidths = [2.5, 2.0, 2.0, 2.0]

    titles = [
        "Pressure Coefficient",
        "Skin Friction Coefficient",
        "Wall Pressure",
        "Wall Heat Flux",
    ]
    ylabels = [r"$C_p$", r"$C_f$", "Pressure [Pa]", r"$q_w$ [W/m$^2$]"]
    keys = ["C_p", "C_f", "p", "q_w"]

    for sn_idx, sn in enumerate(snapshot_list):
        if sn not in surface_data_dict:
            print(f"  Warning: snapshot {sn} not found, skipping")
            continue

        sd = surface_data_dict[sn]
        ls = linestyles[sn_idx % len(linestyles)]
        lw = linewidths[sn_idx % len(linewidths)]
        sn_label = f"t={sn}"

        for side in ("upper", "lower"):
            if side not in sd:
                continue
            data = sd[side]
            if len(data["x"]) == 0:
                continue
            color = side_colors[side]
            label = f"{side} {sn_label}"

            for idx, (key, ax) in enumerate(zip(keys, axes.flat)):
                ax.plot(data["x"], data[key], ls,
                        color=color, linewidth=lw, label=label)

    for idx, (title, ylabel, ax) in enumerate(zip(titles, ylabels, axes.flat)):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Surface Properties — {', '.join(str(s) for s in snapshot_list)}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return axes


# ---------------------------------------------------------------------------
# 7.  AERODYNAMIC FORCE TIME-SERIES
# ---------------------------------------------------------------------------

def plot_forces_vs_time(forces_dict, output_path=None, figsize=(14, 10),
                         laser_start_time=None):
    """Plot integrated force coefficients vs physical time across snapshots.

    Parameters
    ----------
    forces_dict : dict
        Keys are snapshot labels, values are force-result dicts from
        ``compute_integrated_forces``.  Each entry must have keys
        ``time``, ``C_D``, ``C_L``, ``C_Dp``, ``C_Dv``.
    output_path : str, optional
    figsize : tuple, default (14, 10)
    laser_start_time : float, optional
        If provided, draw a vertical dashed line at this time.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not forces_dict:
        raise ValueError("forces_dict is empty.")

    # Sort by time
    snap_labels = sorted(forces_dict.keys())
    times = []
    C_D_arr = []
    C_L_arr = []
    C_Dp_arr = []
    C_Dv_arr = []

    for label in snap_labels:
        fd = forces_dict[label]
        times.append(fd["time"])
        C_D_arr.append(fd.get("C_D", 0.0))
        C_L_arr.append(fd.get("C_L", 0.0))
        C_Dp_arr.append(fd.get("C_Dp", 0.0))
        C_Dv_arr.append(fd.get("C_Dv", 0.0))

    times = np.asarray(times, dtype=float)
    C_D_arr = np.asarray(C_D_arr, dtype=float)
    C_L_arr = np.asarray(C_L_arr, dtype=float)
    C_Dp_arr = np.asarray(C_Dp_arr, dtype=float)
    C_Dv_arr = np.asarray(C_Dv_arr, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: C_D and C_L
    ax1.plot(times, C_D_arr, "r-o", markersize=3, linewidth=1.5, label=r"$C_D$")
    ax1.plot(times, C_L_arr, "b-s", markersize=3, linewidth=1.5, label=r"$C_L$")
    if laser_start_time is not None:
        ax1.axvline(laser_start_time, color="gray", linestyle="--", alpha=0.7,
                    label=f"Laser ON (t={laser_start_time:.2e} s)")
    ax1.set_ylabel("Force Coefficient", fontsize=12)
    ax1.set_title("Integrated Force Coefficients vs Time", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: pressure vs viscous drag
    ax2.plot(times, C_Dp_arr, "g-o", markersize=3, linewidth=1.5, label=r"$C_{D,p}$ (pressure)")
    ax2.plot(times, C_Dv_arr, "m-o", markersize=3, linewidth=1.5, label=r"$C_{D,v}$ (viscous)")
    if laser_start_time is not None:
        ax2.axvline(laser_start_time, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Drag Coefficient", fontsize=12)
    ax2.set_title("Pressure vs Viscous Drag Contribution", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_certified_force_timeseries(
        forces_dict, output_path=None, laser_start_time=None,
        figsize=(13, 11)):
    """Plot certified one-sided drag, normal load, and moment histories."""
    if not forces_dict:
        raise ValueError("forces_dict is empty")
    entries = sorted(
        forces_dict.values(), key=lambda item: float(item.get("time", 0.0))
    )
    absolute_time = np.asarray([item["time"] for item in entries], dtype=float)
    origin = (
        float(laser_start_time)
        if laser_start_time is not None else float(absolute_time[0])
    )
    time_ns = (absolute_time - origin) * 1.0e9
    has_increment = all("delta_D_total_N_m" in item for item in entries)
    prefix = "delta_" if has_increment else ""

    def values(key):
        return np.asarray([
            item.get(prefix + key, np.nan) for item in entries
        ], dtype=float)

    figure, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[0].plot(time_ns, values("D_total_N_m"), color="C3", label="total")
    axes[0].plot(
        time_ns, values("D_pressure_N_m"), color="C2", linestyle="--",
        label="pressure",
    )
    axes[0].plot(
        time_ns, values("D_viscous_N_m"), color="C4", linestyle=":",
        label="viscous",
    )
    axes[0].set_ylabel(
        r"$\Delta D'$ [N/m]" if has_increment else r"$D'$ [N/m]"
    )
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(time_ns, values("N_total_N_m"), color="C0", label="total")
    axes[1].plot(
        time_ns, values("N_pressure_N_m"), color="C1", linestyle="--",
        label="pressure",
    )
    axes[1].plot(
        time_ns, values("N_viscous_N_m"), color="C5", linestyle=":",
        label="viscous",
    )
    axes[1].set_ylabel(
        r"$\Delta N'_{\mathrm{1s}}$ [N/m]"
        if has_increment else r"$N'_{\mathrm{1s}}$ [N/m]"
    )
    axes[1].legend(frameon=False, ncol=3)

    axes[2].plot(time_ns, values("M_total_N"), color="C6")
    axes[2].set_ylabel(
        r"$\Delta M'_z$ [N]" if has_increment else r"$M'_z$ [N]"
    )
    axes[2].set_xlabel(r"Time from laser start [ns]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    title = (
        "Certified one-sided flat-plate load increments"
        if has_increment else "Certified one-sided flat-plate loads"
    )
    axes[0].set_title(title)
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_force_surface_history(
        time_relative_s, x_m, values, label, output_path=None,
        cmap="RdBu_r", figsize=(13, 6)):
    """Plot a signed distributed force quantity over x and time."""
    time_relative_s = np.asarray(time_relative_s, dtype=float)
    x_m = np.asarray(x_m, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (time_relative_s.size, x_m.size):
        raise ValueError("surface-history values must have shape (time, x)")
    bound = float(np.nanpercentile(np.abs(values), 99.0))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    figure, axis = plt.subplots(figsize=figsize)
    mesh = axis.pcolormesh(
        x_m, time_relative_s * 1.0e9, values,
        shading="auto", cmap=cmap, vmin=-bound, vmax=bound,
    )
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(label)
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel("Time from laser start [ns]")
    axis.set_title("Laser-induced distributed one-sided wall loading")
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_packet_propagation(propagation, output_path=None):
    """Plot robust packet arrival, envelope energy, and peak response."""
    x = np.asarray(propagation["probe_x_m"], dtype=float)
    valid = np.asarray(propagation["valid"], dtype=bool)
    arrival = np.asarray(propagation["arrival_time_s"], dtype=float)
    energy = np.asarray(propagation["envelope_energy"], dtype=float)
    peak = np.asarray(propagation["peak_amplitude"], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(x[valid], arrival[valid] * 1.0e6, "o", label="Detected arrival")
    if np.any(valid) and np.isfinite(
            propagation.get("arrival_fit_slope_s_m", np.nan)):
        fit = (
            float(propagation["arrival_fit_intercept_s"])
            + float(propagation["arrival_fit_slope_s_m"]) * x
        )
        axes[0].plot(
            x, fit * 1.0e6, "-",
            label=(
                f"Theil–Sen fit, $U_g$="
                f"{propagation['group_velocity_m_s']:.1f} m/s"
            ),
        )
    axes[0].set_ylabel("Arrival time [µs]")
    axes[0].legend(frameon=False)
    axes[1].semilogy(
        x, np.maximum(energy, np.finfo(float).tiny), "o-"
    )
    axes[1].set_ylabel("Envelope energy [signal² s]")
    axes[2].plot(x, peak, "o-")
    axes[2].set_ylabel("Peak envelope")
    axes[2].set_xlabel("x [m]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("Band-limited transient packet propagation")
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return figure


# ---------------------------------------------------------------------------
# 8.  SPACE-TIME CONTOUR PLOTS
# ---------------------------------------------------------------------------

def plot_spacetime(surface_data_series, field_key="C_p", output_path=None,
                    figsize=(14, 8), cmap="turbo", vmin=None, vmax=None,
                    xlabel="x [m]", ylabel="Time [s]",
                    colorbar_label=None, laser_start_time=None):
    """Plot a field's evolution along the surface as a space-time contour.

    Parameters
    ----------
    surface_data_series : dict
        Keys are snapshot labels, values are surface_data dicts (from
        ``extract_surface_properties``).  Each surface_data must have
        ``'upper'`` key with arrays for ``x`` and ``field_key``.
    field_key : str, default 'C_p'
        Field to contour (e.g. 'C_p', 'C_f', 'tau_w', 'delta_99', 'H').
    output_path : str, optional
    figsize : tuple, default (14, 8)
    cmap : str, default 'turbo'
    vmin, vmax : float, optional
    xlabel, ylabel : str
    colorbar_label : str, optional
    laser_start_time : float, optional
        Vertical dashed line for laser onset.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not surface_data_series:
        raise ValueError("surface_data_series is empty.")

    # Collect times and surface x-positions
    snap_labels = sorted(surface_data_series.keys())
    times = []

    # Build a common x grid from the first snapshot
    first_sd = surface_data_series[snap_labels[0]].get("upper", {})
    if len(first_sd.get("x", [])) == 0:
        raise ValueError("First snapshot has no upper-surface data.")
    x_ref = first_sd["x"]

    # Interpolate each snapshot's field onto x_ref
    n_time = len(snap_labels)
    n_x = len(x_ref)
    Z = np.full((n_time, n_x), np.nan, dtype=float)

    for ti, label in enumerate(snap_labels):
        sd = surface_data_series[label].get("upper", {})
        if len(sd.get("x", [])) == 0 or field_key not in sd:
            Z[ti, :] = np.nan
            continue
        times.append(surface_data_series[label].get("time", float(ti)))
        # Interpolate to common grid
        x_i = sd["x"]
        f_i = np.asarray(sd[field_key], dtype=float)
        Z[ti, :] = np.interp(x_ref, x_i, f_i, left=np.nan, right=np.nan)

    times = np.asarray(times, dtype=float)

    if vmin is None:
        vmin = np.nanpercentile(Z, 2)
    if vmax is None:
        vmax = np.nanpercentile(Z, 98)
    # Guard against blank plots when all values are identical (e.g. zeros)
    if vmin is None or np.isnan(vmin):
        vmin = 0.0
    if vmax is None or np.isnan(vmax):
        vmax = 1.0
    if abs(vmax - vmin) < 1e-30:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    if colorbar_label is None:
        colorbar_label = field_key

    fig, ax = plt.subplots(figsize=figsize)
    xx, tt = np.meshgrid(x_ref, times)
    p = ax.pcolormesh(xx, tt, Z, cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(p, ax=ax, pad=0.02, shrink=0.4)
    cb.set_label(colorbar_label, fontsize=12)

    if laser_start_time is not None:
        ax.axhline(laser_start_time, color="white", linestyle="--", linewidth=2,
                   alpha=0.8, label=f"Laser ON")
        ax.legend(fontsize=10, loc="upper right")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Space-Time Evolution of {field_key}", fontsize=14)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 9.  CONTOUR WITH SURFACE LOADING OVERLAY
# ---------------------------------------------------------------------------

def plot_contour_with_surface_loading(dataset, surfaces, surface_data,
                                       field_key="temperature",
                                       loading_key="C_p",
                                       output_path=None, figsize=(14, 8),
                                       cmap="turbo", vmin=None, vmax=None,
                                       xlim=None, ylim=None):
    """Plot a flow contour with the detected surface and loading distribution.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    surfaces : dict
        Output from ``define_geometry_surface`` + ``compute_surface_normals``.
    surface_data : dict
        Output from ``extract_surface_properties``.
    field_key : str, default 'temperature'
        Flow field to show in the main contour.
    loading_key : str, default 'C_p'
        Surface loading to plot on the right axis.
    output_path : str, optional
    figsize : tuple, default (14, 8)
    cmap : str, default 'turbo'
    vmin, vmax : float, optional
    xlim, ylim : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax_contour = fig.add_subplot(gs[0])
    ax_loading = fig.add_subplot(gs[1], sharex=ax_contour)

    # ---- Left: flow contour ----
    xx, yy = np.meshgrid(dataset["x"], dataset["y"], indexing="ij")
    field = np.asarray(dataset["fields"][field_key], dtype=float)
    if vmin is None:
        vmin = np.nanpercentile(field, 1)
    if vmax is None:
        vmax = np.nanpercentile(field, 99)

    p = ax_contour.pcolormesh(xx, yy, field, cmap=cmap, shading="auto",
                               vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(p, ax=ax_contour, pad=0.02, shrink=0.4)
    cb.set_label(field_key, fontsize=12)

    # Overlay surface
    for side, color in (("upper", "red"), ("lower", "blue")):
        if len(surfaces[side].get("x", [])) > 0:
            ax_contour.plot(surfaces[side]["x"], surfaces[side]["y_interp"],
                            "-", color=color, linewidth=2,
                            label=f"{side} surface")
    ax_contour.legend(fontsize=9)
    ax_contour.set_xlabel("")
    ax_contour.tick_params(labelbottom=False)
    ax_contour.set_ylabel("y [m]", fontsize=12)
    ax_contour.set_title(f"{field_key} with Surface", fontsize=14)
    if xlim is not None:
        ax_contour.set_xlim(xlim)
    if ylim is not None:
        ax_contour.set_ylim(ylim)
    ax_contour.set_aspect("equal", adjustable="box")

    # ---- Right: surface loading ----
    colors = {"upper": "red", "lower": "blue"}
    for side in ("upper", "lower"):
        if side not in surface_data:
            continue
        sd = surface_data[side]
        if len(sd.get("x", [])) == 0 or loading_key not in sd:
            continue
        ax_loading.plot(sd["x"], sd[loading_key], "-", color=colors[side],
                        linewidth=2, label=f"{side} {loading_key}")
    ax_loading.set_xlabel("x [m]", fontsize=12)
    ax_loading.set_ylabel(loading_key, fontsize=12)
    ax_loading.set_title(f"Surface {loading_key} Distribution", fontsize=14)
    ax_loading.legend(fontsize=9)
    ax_loading.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 10. ANIMATION OF CONTOUR WITH SURFACE OVER TIME
# ---------------------------------------------------------------------------

def animate_contour_with_surface(dataset_series, field_key,
                                  surfaces_series=None,
                                  surface_data_series=None,
                                  loading_key="C_p",
                                  output_path="loading_evolution.gif",
                                  figsize=(14, 8),
                                  cmap="turbo", vmin=None, vmax=None,
                                  xlim=None, ylim=None,
                                  fps=10, dpi=150,
                                  laser_start_time=None):
    """Animate a flow contour across snapshots with optional surface overlay.

    If ``surfaces_series`` is provided, each frame also shows detected surface
    points.  If ``surface_data_series`` is provided, a second panel advances
    the matching surface-value distribution in lockstep with the contour.

    Parameters
    ----------
    dataset_series : dict
        Keys are snapshot labels, values are ``StandardDataset`` dicts.
    field_key : str
        Field to contour.
    surfaces_series : dict, optional
        Keys match ``dataset_series``, values are surface dicts.
    surface_data_series : dict, optional
        Keys match ``dataset_series``, values are surface-property dicts.  The
        requested ``loading_key`` is plotted for the same snapshot as each
        contour frame.
    loading_key : str, default 'C_p'
        Surface property to show in the synchronized right-hand panel.
    output_path : str, default 'loading_evolution.gif'
    figsize : tuple, default (14, 8)
    cmap : str, default 'turbo'
    vmin, vmax : float, optional
    xlim, ylim : tuple, optional
    fps : int, default 10
    dpi : int, default 150
    laser_start_time : float, optional
        Time threshold to highlight laser-on frames.

    Returns
    -------
    str
        Path to the output file.
    """
    snap_labels = sorted(dataset_series.keys(), key=fdb_mod.natural_sort_key)
    if len(snap_labels) == 0:
        raise ValueError("dataset_series is empty.")

    # Determine global limits
    vmin_global = vmin
    vmax_global = vmax
    if vmin_global is None or vmax_global is None:
        all_vals = []
        for label in snap_labels:
            ds = dataset_series[label]
            fld = ds["fields"].get(field_key)
            if fld is not None:
                all_vals.append(np.asarray(fld, dtype=float).ravel())
        if all_vals:
            combined = np.concatenate(all_vals)
            finite = combined[np.isfinite(combined)]
            if vmin_global is None:
                vmin_global = float(np.nanpercentile(finite, 1))
            if vmax_global is None:
                vmax_global = float(np.nanpercentile(finite, 99))
        else:
            vmin_global, vmax_global = 0.0, 1.0

    # Set up a contour-only figure for backwards compatibility, or the paired
    # contour/loading layout requested for synchronized surface diagnostics.
    show_loading = surface_data_series is not None
    if show_loading:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.35)
        ax = fig.add_subplot(grid[0])
        ax_loading = fig.add_subplot(grid[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_loading = None
    ax.set_xlabel("" if show_loading else "x [m]", fontsize=12)
    if show_loading:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel("y [m]", fontsize=12)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    # First frame to initialise pcolormesh
    ds0 = dataset_series[snap_labels[0]]
    xx0, yy0 = np.meshgrid(ds0["x"], ds0["y"], indexing="ij")
    f0 = np.asarray(ds0["fields"].get(field_key, np.zeros_like(xx0)), dtype=float)
    p = ax.pcolormesh(xx0, yy0, f0, cmap=cmap, shading="auto",
                       vmin=vmin_global, vmax=vmax_global, rasterized=True)
    cb = fig.colorbar(p, ax=ax, pad=0.02, shrink=0.4)
    cb.set_label(field_label(field_key), fontsize=12)

    # Surface lines (if available)
    surf_lines = {}
    for side, color in (("upper", "red"), ("lower", "blue")):
        ln, = ax.plot([], [], "-", color=color, linewidth=2, label=f"{side}")
        surf_lines[side] = ln

    # Time annotation
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14,
                         color="white", fontweight="bold",
                         bbox=dict(facecolor="black", alpha=0.5, pad=3))
    laser_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, fontsize=12,
                          color="yellow", fontweight="bold",
                          bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.legend(fontsize=9, loc="upper right")

    loading_lines = {}
    if show_loading:
        loading_values = []
        loading_x_values = []
        for label in snap_labels:
            frame_data = surface_data_series.get(label, {})
            for side in ("upper", "lower"):
                side_data = frame_data.get(side, {})
                if loading_key not in side_data:
                    continue
                x_values = np.asarray(side_data.get("x", []), dtype=float)
                values = np.asarray(side_data[loading_key], dtype=float)
                finite = np.isfinite(x_values) & np.isfinite(values)
                if np.any(finite):
                    loading_x_values.append(x_values[finite])
                    loading_values.append(values[finite])

        for side, color in (("upper", "red"), ("lower", "blue")):
            line, = ax_loading.plot(
                [], [], "-", color=color, linewidth=2,
                label=f"{side} {loading_key}",
            )
            loading_lines[side] = line
        ax_loading.set_xlabel("x [m]", fontsize=12)
        ax_loading.set_ylabel(field_label(loading_key), fontsize=12)
        ax_loading.set_title(f"Surface {field_title(loading_key)}", fontsize=14)
        ax_loading.grid(True, alpha=0.3)
        if xlim is not None:
            ax_loading.set_xlim(xlim)
        elif loading_x_values:
            all_loading_x = np.concatenate(loading_x_values)
            ax_loading.set_xlim(np.nanmin(all_loading_x), np.nanmax(all_loading_x))
        if loading_values:
            all_loading_values = np.concatenate(loading_values)
            load_min = float(np.nanmin(all_loading_values))
            load_max = float(np.nanmax(all_loading_values))
            if np.isclose(load_min, load_max):
                padding = max(0.05 * abs(load_min), 1.0e-12)
            else:
                padding = 0.05 * (load_max - load_min)
            ax_loading.set_ylim(load_min - padding, load_max + padding)
        ax_loading.legend(fontsize=9, loc="best")

    def _init():
        p.set_array(np.ma.array(f0.ravel()))
        for ln in surf_lines.values():
            ln.set_data([], [])
        time_text.set_text("")
        laser_text.set_text("")
        for ln in loading_lines.values():
            ln.set_data([], [])
        return (
            p, *surf_lines.values(), *loading_lines.values(),
            time_text, laser_text,
        )

    def _update(frame_idx):
        label = snap_labels[frame_idx]
        ds = dataset_series[label]
        t = ds.get("time", 0.0)

        # Update contour
        fld = np.asarray(ds["fields"].get(field_key, np.zeros_like(xx0)), dtype=float)
        p.set_array(np.ma.array(fld.ravel()))

        # Update surface lines
        if surfaces_series and label in surfaces_series:
            surfs = surfaces_series[label]
            for side, ln in surf_lines.items():
                if len(surfs[side].get("x", [])) > 0:
                    ln.set_data(surfs[side]["x"], surfs[side]["y_interp"])
                else:
                    ln.set_data([], [])

        # Update the surface-value panel using the exact same snapshot label.
        if show_loading:
            frame_data = surface_data_series.get(label, {})
            for side, ln in loading_lines.items():
                side_data = frame_data.get(side, {})
                if loading_key in side_data and len(side_data.get("x", [])) > 0:
                    ln.set_data(side_data["x"], side_data[loading_key])
                else:
                    ln.set_data([], [])

        # Update annotations
        time_text.set_text(f"t = {t:.2e} s")
        if laser_start_time is not None and t >= laser_start_time:
            laser_text.set_text("⚡ LASER ON")
        else:
            laser_text.set_text("")

        return (
            p, *surf_lines.values(), *loading_lines.values(),
            time_text, laser_text,
        )

    anim = FuncAnimation(fig, _update, frames=len(snap_labels),
                         init_func=_init, blit=True, repeat=False)

    f_ext = os.path.splitext(output_path)[1].lower()
    if f_ext == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)  # fallback; use ffmpeg for mp4

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Probe time-history plots
# ---------------------------------------------------------------------------

def plot_probe_timeseries(probe_data, output_dir,
                          station_probes=None,
                          laser_start_time=None,
                          convert_to_mks=False):
    """Plot probe time-history visualizations.

    Creates:
    1. ``probe_overview.png`` — 4-panel figure (rho/u/p/T) with all probes
       overlaid as thin lines colored by x-position.
    2. ``probe_stations/`` — Individual 2x2 panel plots for selected probes.

    Parameters
    ----------
    probe_data : dict
        Output from ``load_probe_dat_files()``.
    output_dir : str or Path
        Directory where output figures are saved.
    station_probes : list of int, optional
        List of probe indices for individual 2x2 panel plots.
        If None, auto-selects ~9 evenly spaced probes.
    laser_start_time : float, optional
        Vertical dashed line marker for laser turn-on.
    convert_to_mks : bool
        If True, convert CGS to MKS units.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(probe_data.keys(),
                        key=lambda pid: probe_data[pid]["probe_x"])
    n_probes = len(sorted_ids)
    if n_probes == 0:
        return

    # Field definitions: (key, CGS-label, MKS-label, CGS-to-MKS factor)
    fields = [
        ("rho", r"$\rho$ [g/cm³]", r"$\rho$ [kg/m³]", 1000.0),
        ("u",   r"$u$ [cm/s]",     r"$u$ [m/s]",      0.01),
        ("p",   r"$p$ [dyne/cm²]", r"$p$ [Pa]",       0.1),
        ("T",   r"$T$ [K]",        r"$T$ [K]",        1.0),
    ]

    # ---- 1. Overview: 4-panel overlay figure ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    probe_x_vals = np.array([probe_data[pid]["probe_x"] for pid in sorted_ids])
    x_min, x_max = probe_x_vals.min(), probe_x_vals.max()
    norm = plt.Normalize(x_min, x_max)
    cmap_obj = plt.get_cmap("viridis")

    for idx, (fname, clabel, mlabel, cfac) in enumerate(fields):
        ax = axes[idx]
        ylabel = mlabel if convert_to_mks else clabel

        for pid in sorted_ids:
            pd = probe_data[pid]
            ydata = pd[fname] * (cfac if convert_to_mks else 1.0)
            ax.plot(pd["time"], ydata,
                    color=cmap_obj(norm(pd["probe_x"])),
                    lw=0.3, alpha=0.5)

        if laser_start_time is not None:
            ax.axvline(laser_start_time, color="red",
                       linestyle="--", alpha=0.7,
                       label=f"Laser ON (t={laser_start_time:.2e} s)")
            ax.legend(fontsize=9, loc="upper right")

        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(clabel.split("[")[0].strip(), fontsize=13)
        ax.grid(True, alpha=0.3)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        label="Probe x [cm]", pad=0.02)

    fig.suptitle("Probe Time Histories \u2014 All Stations",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "probe_overview.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Individual station plots ----
    if station_probes is None:
        n_stations = min(9, n_probes)
        step = max(1, n_probes // n_stations)
        station_probes = sorted_ids[::step]
        if sorted_ids[-1] not in station_probes:
            station_probes.append(sorted_ids[-1])

    stations_dir = output_dir / "stations"
    stations_dir.mkdir(parents=True, exist_ok=True)

    for pid in station_probes:
        if pid not in probe_data:
            continue
        pd = probe_data[pid]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (fname, clabel, mlabel, cfac) in enumerate(fields):
            ax = axes[idx]
            ylabel = mlabel if convert_to_mks else clabel
            ydata = pd[fname] * (cfac if convert_to_mks else 1.0)

            ax.plot(pd["time"], ydata, "b-", lw=1.0)
            if laser_start_time is not None:
                ax.axvline(laser_start_time, color="red",
                           linestyle="--", alpha=0.7)
            ax.set_xlabel("Time [s]", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(clabel.split("[")[0].strip(), fontsize=13)
            ax.grid(True, alpha=0.3)

        px = pd.get("probe_x", float(pid))
        py = pd.get("probe_y", 0.0)
        fig.suptitle(f"Probe {pid} \u2014 x = {px:.4f} cm, y = {py:.4f} cm",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(str(stations_dir / f"probe_{pid:04d}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 9.  STABILITY DIAGNOSTICS PLOTS  (2nd Mack mode)
# ---------------------------------------------------------------------------

def plot_stability_summary(data, output_path=None, figsize=(18, 14)):
    """4-panel figure summarising heuristic stability-screening diagnostics.

    Panel layout
    ------------
    1. Estimated 2nd mode frequency vs x (omega* approx 0.3 and acoustic formulas)
    2. Phase speed c_p vs x with fast/slow acoustic reference lines
    3. Spatial amplification -alpha_i (raw and delta_99-normalised)
    4. Boundary-layer profiles at selected x-stations with GPI markers

    Parameters
    ----------
    data : dict
        Keys: ``freq_estimate``, ``phase_speed``, ``growth_rate``,
        ``gpi_profiles``, ``target_freq``, ``freq_band``, ``bl_profiles``.
    output_path : str, optional
    figsize : tuple, default (18, 14)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flat

    # -- Panel 1: Frequency estimate
    fd = data.get("freq_estimate", {})
    x_f = fd.get("x", [])
    if len(x_f) > 0:
        f_om = fd.get("f_omega_star", [])
        f_ac = fd.get("f_acoustic", [])
        valid_om = np.isfinite(f_om)
        valid_ac = np.isfinite(f_ac)

        if np.any(valid_om):
            ax1.semilogy(x_f[valid_om], f_om[valid_om], "b-",
                         linewidth=1.5, label=r"$f \; (\omega^*=0.3)$")
        if np.any(valid_ac):
            ax1.semilogy(x_f[valid_ac], f_ac[valid_ac], "r--",
                         linewidth=1.5, label=r"$f \; (\lambda/2 = \delta_{99})$")

        f_band = data.get("freq_band", [100e3, 1.0e6])
        ax1.axhspan(f_band[0], f_band[1], alpha=0.08, color="gray",
                     label=f"Search band [{f_band[0]:.1e}, {f_band[1]:.1e}]")

        ax1.axhline(y=data.get("target_freq", np.nan), color="k",
                     linestyle=":", linewidth=1.0,
                     label=f"Target: {data.get('target_freq', 0):.3e} Hz")

    ax1.set_xlabel("x [m]", fontsize=12)
    ax1.set_ylabel("Frequency [Hz]", fontsize=12)
    ax1.set_title("Panel 1: Heuristic second-mode frequency scales", fontsize=13)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # -- Panel 2: Phase speed
    ps = data.get("phase_speed", {})
    x_cp = ps.get("x_mid", [])
    if len(x_cp) > 0:
        cp = ps.get("c_p", [])
        valid_cp = np.isfinite(cp)
        cp_raw = np.asarray(ps.get("c_p_raw", []), dtype=float)
        if cp_raw.shape == np.asarray(cp).shape:
            invalid_cp = np.isfinite(cp_raw) & ~valid_cp
            if np.any(invalid_cp):
                ax2.scatter(
                    np.asarray(x_cp)[invalid_cp], cp_raw[invalid_cp],
                    color="0.75", s=4, alpha=0.45,
                    label="Rejected by coherence gate",
                )
        if np.any(valid_cp):
            valid_x = np.asarray(x_cp)[valid_cp]
            valid_speed = np.asarray(cp)[valid_cp]
            ax2.scatter(
                valid_x, valid_speed, s=5, alpha=0.30,
                label=r"adjacent-pair $c_p = \omega/\alpha_r$",
            )
            # A binned median exposes the branch-level trend without joining
            # phase-wrap outliers with misleading vertical line segments.
            edges = np.linspace(np.min(valid_x), np.max(valid_x), 81)
            bin_id = np.digitize(valid_x, edges) - 1
            median_x, median_speed = [], []
            for index in range(len(edges) - 1):
                in_bin = bin_id == index
                if np.any(in_bin):
                    median_x.append(np.median(valid_x[in_bin]))
                    median_speed.append(np.median(valid_speed[in_bin]))
            ax2.plot(
                median_x, median_speed, color="C0", linewidth=1.8,
                label="spatial-bin median",
            )

        cpf = ps.get("c_p_fast", [])
        cps = ps.get("c_p_slow", [])
        if len(cpf) > 0:
            ax2.plot(x_cp, cpf, "r--", linewidth=1.0,
                     label=r"$c_{p,fast} = a_e + u_e$")
        if len(cps) > 0:
            ax2.plot(x_cp, cps, "b--", linewidth=1.0,
                     label=r"$c_{p,slow} = u_e - a_e$")

        if not np.any(valid_cp):
            ax2.text(0.5, 0.5, "Need complex FFT data\nfor phase calculation",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=11, color="gray", style="italic")

        reference_values = np.concatenate((
            np.asarray(cpf, dtype=float).ravel(),
            np.asarray(cps, dtype=float).ravel(),
        ))
        reference_values = reference_values[np.isfinite(reference_values)]
        if reference_values.size:
            upper = max(1.25 * np.max(reference_values), 1.0)
            lower = min(0.0, 1.25 * np.min(reference_values))
            ax2.set_ylim(lower, upper)
        coherence = np.asarray(ps.get("coherence_squared", []), dtype=float)
        if coherence.size:
            threshold = ps.get("coherence_threshold", np.nan)
            ax2.text(
                0.02, 0.03,
                rf"retained {np.sum(coherence >= threshold)}/{coherence.size}; "
                rf"$\gamma^2 \geq {threshold:.2f}$",
                transform=ax2.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"),
            )

    ax2.set_xlabel("x [m]", fontsize=12)
    ax2.set_ylabel("Phase speed [m/s]", fontsize=12)
    ax2.set_title("Panel 2: Phase Speed at Target Frequency", fontsize=13)
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(handles, labels, fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    # -- Panel 3: Growth rate
    gd = data.get("growth_rate", {})
    x_g = gd.get("x", [])
    if len(x_g) > 0:
        alpha_i = np.asarray(gd.get("alpha_i", []), dtype=float)
        amplification = -alpha_i
        valid_ai = np.isfinite(amplification)

        if np.any(valid_ai):
            color1 = "tab:blue"
            ax3_twin = ax3.twinx()
            l1 = ax3.plot(
                np.asarray(x_g)[valid_ai], amplification[valid_ai],
                "o-", markersize=4, linewidth=1.2, color=color1,
                label=r"$-\alpha_i$ [m$^{-1}$]",
            )
            ci95 = np.asarray(gd.get("alpha_i_ci95", []), dtype=float)
            if ci95.shape == alpha_i.shape:
                valid_ci = valid_ai & np.isfinite(ci95)
                if np.any(valid_ci):
                    ax3.fill_between(
                        np.asarray(x_g)[valid_ci],
                        amplification[valid_ci] - ci95[valid_ci],
                        amplification[valid_ci] + ci95[valid_ci],
                        color=color1, alpha=0.18, linewidth=0,
                        label="local-fit 95% CI",
                    )
            ax3.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
            ax3.set_ylabel(
                r"Amplification $-\alpha_i$ [m$^{-1}$]",
                fontsize=12, color=color1,
            )
            ax3.tick_params(axis="y", labelcolor=color1)

            aid = -np.asarray(gd.get("alpha_i_delta", []), dtype=float)
            valid_aid = np.isfinite(aid)
            if np.any(valid_aid):
                color2 = "tab:red"
                l2 = ax3_twin.plot(x_g[valid_aid], aid[valid_aid], "s--",
                                   markersize=4, linewidth=1.0, color=color2,
                                   label=r"$-\alpha_i \delta_{99}$")
                ax3_twin.set_ylabel(
                    r"$-\alpha_i \delta_{99}$", fontsize=12, color=color2
                )
                ax3_twin.tick_params(axis="y", labelcolor=color2)

            lines = l1
            labels = [r"$-\alpha_i$ [m$^{-1}$]"]
            if np.any(valid_aid):
                lines = l1 + l2
                labels = [
                    r"$-\alpha_i$ [m$^{-1}$]",
                    r"$-\alpha_i \delta_{99}$",
                ]
            ax3.legend(lines, labels, fontsize=8, loc="best")

    ax3.set_xlabel("x [m]", fontsize=12)
    ax3.set_title(
        "Panel 3: Spatial Amplification (positive means growth)", fontsize=13
    )
    ax3.grid(True, alpha=0.3)

    # -- Panel 4: GPI profiles
    gpi_list = data.get("gpi_profiles", [])
    bl_list = data.get("bl_profiles", [])
    colors = plt.cm.inferno(np.linspace(0.3, 0.9, max(len(gpi_list), 1)))
    plotted_gpi = False

    for i, gpi in enumerate(gpi_list):
        yp = gpi.get("y_profile", [])
        Fp = gpi.get("F", [])
        x_pos = gpi.get("x", np.nan)
        color = colors[i % len(colors)]

        if len(yp) == 0 or len(Fp) == 0:
            continue

        F_norm = Fp / (np.nanmax(np.abs(Fp)) + 1e-20)
        ax4.plot(F_norm, yp, color=color, linewidth=1.2,
                 label=f"x = {x_pos:.3f} m")
        plotted_gpi = True

        y_gpi = gpi.get("y_gpi", np.nan)
        stable = gpi.get("unstable", False)
        if np.isfinite(y_gpi):
            marker = "v" if stable else "o"
            ax4.axhline(y=y_gpi, xmin=0, xmax=0.3, color=color,
                        linestyle="--", linewidth=0.8)
            ax4.plot(0, y_gpi, marker=marker, color=color, markersize=8,
                     label=f"GPI" if i == 0 else "")

    ax4.axvline(x=0, color="gray", linestyle=":", linewidth=0.8)
    ax4.set_xlabel(r"$F(y) = d(\rho \, du/dy)/dy$  (normalized)", fontsize=12)
    ax4.set_ylabel("y [m]", fontsize=12)
    ax4.set_title("Panel 4: GPI Criterion", fontsize=13)
    if not plotted_gpi:
        ax4.text(
            0.5, 0.5, "No valid density profiles for GPI evaluation",
            transform=ax4.transAxes, ha="center", va="center",
            color="0.4", style="italic",
        )
    handles, labels = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend(handles, labels, fontsize=8, loc="best")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Stability screening — heuristic scales, not an LST eigensolution",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def _robust_symmetric_limit(values, percentile=98.0):
    """Return a finite symmetric colour limit for a possibly masked array."""
    values = np.asarray(values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return limit if limit > 0.0 else 1.0


def plot_wavenumber_summary(data, output_path=None, figsize=(18, 13)):
    """Plot measured frequency-resolved wavenumber and phase-speed products.

    These panels show dominant coherent wave estimates from the probe line.
    Fast-like and slow-like markers indicate proximity to acoustic references,
    not definitive F/S eigenmode identification.
    """
    frequency = np.asarray(data["frequency_hz"], dtype=float)
    x = np.asarray(data["x_center_m"], dtype=float)
    phase_valid = np.asarray(data["phase_valid_mask"], dtype=bool)
    growth_valid = np.asarray(data["growth_valid_mask"], dtype=bool)
    power = np.asarray(data["spectral_power"], dtype=float)
    alpha_real = np.asarray(data["alpha_real_rad_per_m"], dtype=float)
    amplification = np.asarray(
        data["amplification_rate_per_m"], dtype=float
    )
    phase_speed = np.asarray(data["phase_speed_m_per_s"], dtype=float)
    candidate = np.asarray(
        data.get("acoustic_candidate", np.zeros_like(phase_speed, dtype=np.int8))
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    ax_power, ax_real, ax_growth, ax_speed = axes.flat

    reference_power = np.nanmax(power)
    power_db = 10.0 * np.log10(
        np.maximum(power, 1.0e-300) / max(reference_power, 1.0e-300)
    )
    mesh = ax_power.pcolormesh(
        x, frequency, np.clip(power_db, -60.0, 0.0),
        shading="auto", cmap="magma", vmin=-60.0, vmax=0.0,
    )
    fig.colorbar(mesh, ax=ax_power, label="Local spectral power [dB re max]")
    ax_power.set_title("Measured pressure energy")

    real_plot = np.where(phase_valid, alpha_real, np.nan)
    real_limit = _robust_symmetric_limit(real_plot)
    mesh = ax_real.pcolormesh(
        x, frequency, real_plot, shading="auto", cmap="viridis",
        vmin=0.0, vmax=real_limit,
    )
    fig.colorbar(mesh, ax=ax_real, label=r"$\alpha_r$ [rad m$^{-1}$]")
    ax_real.set_title(r"Real wavenumber $\alpha_r$ (quality accepted)")

    growth_plot = np.where(growth_valid, amplification, np.nan)
    growth_limit = _robust_symmetric_limit(growth_plot)
    mesh = ax_growth.pcolormesh(
        x, frequency, growth_plot, shading="auto", cmap="RdBu_r",
        vmin=-growth_limit, vmax=growth_limit,
    )
    fig.colorbar(
        mesh, ax=ax_growth,
        label=r"Amplification $-\alpha_i$ [m$^{-1}$]",
    )
    ax_growth.axhline(
        data.get("target_frequency_hz", np.nan), color="0.2",
        linestyle=":", linewidth=0.8, label="Legacy heuristic target",
    )
    ax_growth.set_title(
        r"Imaginary wavenumber as amplification $-\alpha_i$"
    )
    ax_growth.legend(fontsize=8, loc="best")

    speed_plot = np.where(phase_valid, phase_speed, np.nan)
    finite_speed = speed_plot[np.isfinite(speed_plot)]
    if finite_speed.size:
        speed_low, speed_high = np.nanpercentile(
            finite_speed, [2.0, 98.0]
        )
        if speed_high <= speed_low:
            speed_high = speed_low + 1.0
    else:
        speed_low, speed_high = 0.0, 1.0
    mesh = ax_speed.pcolormesh(
        x, frequency, speed_plot, shading="auto", cmap="turbo",
        vmin=speed_low, vmax=speed_high,
    )
    fig.colorbar(mesh, ax=ax_speed, label=r"$c_p$ [m s$^{-1}$]")
    x_grid, frequency_grid = np.meshgrid(x, frequency)
    for code, color, label in (
            (1, "cyan", "slow-like candidate"),
            (2, "red", "fast-like candidate")):
        selected = phase_valid & (candidate == code)
        if np.any(selected):
            ax_speed.plot(
                x_grid[selected], frequency_grid[selected],
                linestyle="none", marker="o", markersize=2.5,
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=0.45, label=label,
            )
    ax_speed.set_title(
        "Phase speed; markers are acoustic-reference candidates"
    )
    if np.any(candidate):
        ax_speed.legend(fontsize=8, loc="best")

    retained_phase = 100.0 * np.mean(phase_valid)
    retained_growth = 100.0 * np.mean(growth_valid)
    mean_coherence = np.asarray(
        data["mean_coherence_squared"], dtype=float
    )
    fig.suptitle(
        "Measurement-first wave analysis — dominant coherent branch, not LST\n"
        f"phase accepted {retained_phase:.1f}%, "
        f"growth accepted {retained_growth:.1f}%, "
        rf"median $\gamma^2={np.nanmedian(mean_coherence):.3f}$",
        fontsize=14, fontweight="bold",
    )
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Frequency [Hz]")
        ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    if output_path is not None:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_phase_speed_dispersion(data, output_path=None, n_stations=5,
                                figsize=(16, 9)):
    """Plot local measured phase-speed dispersion with acoustic references."""
    frequency = np.asarray(data["frequency_hz"], dtype=float)
    x = np.asarray(data["x_center_m"], dtype=float)
    phase_speed = np.asarray(data["phase_speed_m_per_s"], dtype=float)
    valid = np.asarray(data["phase_valid_mask"], dtype=bool)
    slow = np.asarray(data.get("slow_acoustic_speed_m_per_s", []), dtype=float)
    fast = np.asarray(data.get("fast_acoustic_speed_m_per_s", []), dtype=float)
    candidate = np.asarray(
        data.get("acoustic_candidate", np.zeros_like(phase_speed, dtype=np.int8))
    )
    n_stations = max(1, min(int(n_stations), x.size))
    indices = np.unique(np.linspace(
        0, x.size - 1, n_stations, dtype=int
    ))
    n_columns = min(3, len(indices))
    n_rows = int(np.ceil(len(indices) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=figsize, squeeze=False,
        sharex=True, sharey=True,
    )
    for ax, index in zip(axes.flat, indices):
        accepted = valid[:, index]
        unclassified = accepted & (candidate[:, index] == 0)
        slow_like = accepted & (candidate[:, index] == 1)
        fast_like = accepted & (candidate[:, index] == 2)
        if np.any(unclassified):
            ax.plot(
                frequency[unclassified], phase_speed[unclassified, index],
                "o", color="0.35", markersize=3, label="unclassified",
            )
        if np.any(slow_like):
            ax.plot(
                frequency[slow_like], phase_speed[slow_like, index],
                "o", color="C0", markersize=4, label="slow-like",
            )
        if np.any(fast_like):
            ax.plot(
                frequency[fast_like], phase_speed[fast_like, index],
                "o", color="C3", markersize=4, label="fast-like",
            )
        if slow.shape == x.shape and np.isfinite(slow[index]):
            ax.axhline(
                slow[index], color="C0", linestyle="--", linewidth=1.0,
                label=r"$U_e-a_e$",
            )
        if fast.shape == x.shape and np.isfinite(fast[index]):
            ax.axhline(
                fast[index], color="C3", linestyle="--", linewidth=1.0,
                label=r"$U_e+a_e$",
            )
        ax.set_title(f"x = {x[index]:.4f} m")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(indices):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Frequency [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$c_p$ [m s$^{-1}$]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=min(5, len(handles)),
            bbox_to_anchor=(0.5, 0.96),
        )
    fig.suptitle(
        "Local phase-speed dispersion\n"
        "fast-like/slow-like denotes proximity to acoustic references, not "
        "an F/S eigensolution",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if output_path is not None:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pair_coherence(pair_results, output_path=None, fmax=None,
                        figsize=(11, 8)):
    """Plot Welch coherence and cross-spectral phase for probe pairs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for item in pair_results:
        result = item["result"]
        freq = np.asarray(result["frequency_hz"])
        mask = freq > 0
        if fmax is not None:
            mask &= freq <= float(fmax)
        label = item.get("label", "probe pair")
        ax1.semilogx(freq[mask], result["coherence_squared"][mask], label=label)
        ax2.semilogx(freq[mask], result["cross_phase_rad"][mask], label=label)
    ax1.set_ylabel(r"Coherence $\gamma^2$")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_title("Welch magnitude-squared coherence")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Cross-spectral phase [rad]")
    ax2.set_title(r"Phase convention: $\arg\{X_1^*X_2\}$")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    if pair_results:
        ax1.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_common_mode_validation(time, model, probe_x, output_path=None,
                                example_column=0, figsize=(12, 8)):
    """Plot held-out error and one common-frequency model prediction."""
    time = np.asarray(time)
    n_train = int(model["n_train"])
    train_error = np.asarray(model["train_relative_rms"])
    validation_error = np.asarray(model["validation_relative_rms"])
    probe_x = np.asarray(probe_x)
    example_column = int(np.clip(example_column, 0, len(probe_x) - 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax1.plot(probe_x, train_error, "o-", label="Training")
    ax1.plot(probe_x, validation_error, "s--", label="Held-out")
    ax1.set_xlabel("Probe x [cm]")
    ax1.set_ylabel("Relative RMS error")
    ax1.set_title("Shared-frequency model: in-sample vs held-out error")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    validation_time = time[n_train:]
    observed = np.asarray(model["validation_observed"])[:, example_column]
    predicted = np.asarray(model["validation_prediction"])[:, example_column]
    ax2.plot(validation_time, observed, color="0.25", label="Measured disturbance")
    ax2.plot(validation_time, predicted, color="C3", label="Training-fit prediction")
    ax2.set_xlabel("Flow time [s]")
    ax2.set_ylabel("Zero-mean signal")
    ax2.set_title(
        f"Held-out prediction at x={probe_x[example_column]:.3f} cm; "
        f"relative RMS={validation_error[example_column]:.3f}"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_modal_summary(pod_result, spod_result, dmd_result, output_path=None,
                       figsize=(15, 4.8)):
    """Plot compact POD, SPOD, and DMD diagnostics from probe snapshots."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    pod_energy = np.asarray(pod_result["energy_fraction"])
    mode_number = np.arange(1, pod_energy.size + 1)
    axes[0].bar(mode_number, pod_energy, alpha=0.65, label="Per mode")
    axes[0].plot(mode_number, np.cumsum(pod_energy), "ko-", label="Cumulative")
    axes[0].set_xlabel("POD mode")
    axes[0].set_ylabel("Fraction of total fluctuation energy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("POD energy")
    axes[0].legend(fontsize=8)

    frequency = np.asarray(spod_result["frequency_hz"])
    eigenvalues = np.asarray(spod_result["eigenvalues"])
    positive = frequency > 0
    for mode in range(eigenvalues.shape[1]):
        axes[1].semilogy(
            frequency[positive], eigenvalues[positive, mode],
            label=f"Mode {mode + 1}",
        )
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("SPOD eigenvalue")
    axes[1].set_title("SPOD spectrum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    eigen = np.asarray(dmd_result["eigenvalues"])
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    axes[2].plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5,
                 label="Unit circle")
    scatter = axes[2].scatter(
        eigen.real, eigen.imag,
        c=np.abs(np.asarray(dmd_result["frequency_hz"])),
        cmap="viridis", edgecolor="k", linewidth=0.4,
    )
    fig.colorbar(scatter, ax=axes[2], label="|Frequency| [Hz]")
    axes[2].axhline(0, color="0.7", linewidth=0.7)
    axes[2].axvline(0, color="0.7", linewidth=0.7)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"Re$(\lambda)$")
    axes[2].set_ylabel(r"Im$(\lambda)$")
    axes[2].set_title("DMD eigenvalues")
    axes[2].legend(fontsize=8)
    fig.suptitle("Probe-line modal screening (descriptive; not LST)")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_gpi_profiles(bl_profiles, gpi_results, output_path=None,
                      figsize=(14, 10), n_cols=3):
    """Plot individual BL profiles with velocity, temperature, and GPI.

    Parameters
    ----------
    bl_profiles : list[dict]
        BL profile dicts (output from ``extract_BL_profile_at_surface``).
    gpi_results : list[dict]
        GPI results from ``compute_gpi_criterion`` (same length / subset).
    output_path : str, optional
    figsize : tuple, default (14, 10)
    n_cols : int, default 3

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(gpi_results)
    if n == 0:
        return None

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(np.asarray(axes).ravel())

    for i, (gpi, ax) in enumerate(zip(gpi_results, axes.flat)):
        x_pos = gpi.get("x", np.nan)
        yp = gpi.get("y_profile", [])
        Fp = gpi.get("F", [])
        y_gpi = gpi.get("y_gpi", np.nan)
        unstable = gpi.get("unstable", False)
        d99 = gpi.get("delta_99_est", np.nan)

        u_p = np.array([])
        T_p = np.array([])
        for bl in bl_profiles:
            if abs(bl.get("x", np.nan) - x_pos) < 1e-6:
                u_p = bl.get("u_profile", np.array([]))
                T_p = bl.get("T_profile", np.array([]))
                break

        ax_twin = ax.twinx()
        l1 = ax.plot(yp, u_p / (np.nanmax(u_p) + 1e-20), "b-",
                     linewidth=1.5, label=r"$u/u_e$")
        if len(T_p) > 0:
            l2 = ax_twin.plot(yp, T_p, "r--", linewidth=1.2, label="T [K]")
            ax_twin.set_ylabel("Temperature [K]", fontsize=10, color="r")
            ax_twin.tick_params(axis="y", labelcolor="r")

        F_norm = Fp / (np.nanmax(np.abs(Fp)) + 1e-20)
        ax.plot(F_norm, yp, "g:", linewidth=1.0, alpha=0.7, label="F(y)")

        if np.isfinite(y_gpi):
            marker = "v" if unstable else "o"
            ax.axhline(y=y_gpi, color="k", linestyle="--", linewidth=0.8,
                       alpha=0.5)
            ax.plot(0, y_gpi, marker=marker, color="k", markersize=10,
                    label="GPI")

        if np.isfinite(d99):
            ax.axhline(y=d99, color="gray", linestyle=":", linewidth=0.8,
                       alpha=0.5, label=r"$\\delta_{99}$")

        ax.set_xlabel(r"$u/u_e$,  $F(y)$", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        gpi_marker = '\u2713' if unstable else '\u2717'
        ax.set_title(f"x = {x_pos:.4f} m   (GPI: {gpi_marker})",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    for j in range(i + 1, len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.suptitle("GPI Criterion — Boundary Layer Profiles", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


# ===========================================================================
#  PHASE 1: DISTURBANCE RECONSTRUCTION PLOTS
# ===========================================================================

def plot_harmonic_reconstruction(time, measured, reconstructed, residual,
                                  harmonic_signals, probe_label="",
                                  output_path=None, figsize=(14, 10),
                                  mean_background=None,
                                  relative_rms=None,
                                  recon_method=None,
                                  recon_freq=None,
                                  window_start=None,
                                  window_end=None,
                                  windowed_signal=None,
                                  window_time=None):
    """4-panel figure showing full signal (with optional window highlight),
    reconstruction overlay, per-frequency contributions, and residual.

    When *window_start* and *window_end* are provided, panel 1 shades the
    selected disturbance interval and overlays the extracted windowed signal
    for visual validation of the window boundaries.

    Parameters
    ----------
    time : ndarray — time array [s]
    measured : ndarray — measured (full) signal
    reconstructed : ndarray — summed reconstruction (mean-shifted)
    residual : ndarray — zero-mean residual
    harmonic_signals : dict — per-frequency signals
    probe_label : str, default ''
    output_path : str, optional
    figsize : tuple, default (14, 10)
    mean_background : float, optional
    relative_rms : float, optional
        Precomputed residual RMS normalized on the same reconstruction window.
    recon_method : str, optional
    recon_freq : float, optional
    window_start, window_end : float, optional — disturbance window bounds [s]
    windowed_signal : ndarray, optional — the extracted window segment
    window_time : ndarray, optional — time coordinates for the window segment

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Build metadata string
    meta_parts = []
    if recon_method is not None:
        meta_parts.append(f"method={recon_method}")
    if recon_freq is not None:
        meta_parts.append(f"f₀={recon_freq:.3e} Hz")
    if mean_background is not None:
        meta_parts.append(f"mean={mean_background:.2f}")
    has_window = window_start is not None and window_end is not None
    if has_window:
        meta_parts.append(f"window=[{window_start:.3e},{window_end:.3e}]s")
        meta_parts.append(f"Δt={window_end - window_start:.3e}s")
    meta_str = "  |  ".join(meta_parts) if meta_parts else ""

    # ---------------------------------------------------------------
    # Panel 1: Full measured signal with window highlight
    # ---------------------------------------------------------------
    axes[0].plot(time, measured, "k-", linewidth=0.8, label="Measured (full)")
    if mean_background is not None:
        axes[0].axhline(mean_background, color="gray", linestyle=":",
                        linewidth=0.7, label=f"Background mean = {mean_background:.1f}")

    if has_window:
        # Shade the disturbance interval
        axes[0].axvspan(window_start, window_end, alpha=0.12, color="C1",
                        label=f"Window [{window_start:.3e}, {window_end:.3e}] s")

    if has_window and windowed_signal is not None and window_time is not None:
        # Overlay the extracted window segment
        axes[0].plot(time, measured, "k-", linewidth=0.8, alpha=0.3)
        win_t = np.asarray(window_time).ravel()
        win_s = windowed_signal.ravel()
        if len(win_t) == len(win_s):
            axes[0].plot(win_t, win_s, "r-", linewidth=1.2,
                         label="Windowed signal")

    axes[0].set_ylabel("Signal", fontsize=11)
    title0 = f"Measured signal — {probe_label}"
    if meta_str:
        title0 += f"\n{meta_str}"
    axes[0].set_title(title0, fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")

    # ---------------------------------------------------------------
    # Panel 2: Overlay measured + reconstruction
    # ---------------------------------------------------------------
    axes[1].plot(time, measured, "k-", linewidth=0.5, alpha=0.5, label="Measured")

    if has_window:
        # Reconstruction arrays contain only the selected window, so use the
        # exact sliced time coordinates rather than masking the full record.
        recon_time = np.asarray(window_time).ravel() if window_time is not None else np.array([])
        recon_vals = reconstructed.ravel()
        if len(recon_time) == len(recon_vals):
            axes[1].plot(recon_time, recon_vals, "r-", linewidth=1.2,
                         label="Reconstructed")
    else:
        axes[1].plot(time, reconstructed, "r-", linewidth=1.2,
                     label="Reconstructed")

    axes[1].set_ylabel("Signal", fontsize=11)
    rms_err = np.sqrt(np.mean(residual**2))
    if relative_rms is None:
        if has_window and windowed_signal is not None:
            metric_signal = np.asarray(windowed_signal, dtype=float).ravel()
        else:
            metric_signal = np.asarray(measured, dtype=float).ravel()
        metric_signal = metric_signal - np.mean(metric_signal)
        metric_rms = np.sqrt(np.mean(metric_signal**2))
        rel_rms = rms_err / metric_rms if metric_rms > 0 else 0.0
    else:
        rel_rms = float(relative_rms)
    axes[1].set_title(f"Measured vs reconstruction  |  rel. RMS = {rel_rms:.3f}", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper right")

    # ---------------------------------------------------------------
    # Panel 3: Per-frequency contributions (offset for clarity)
    # ---------------------------------------------------------------
    offset = 0.0
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    component_time = (np.asarray(window_time).ravel()
                      if has_window and window_time is not None else time)
    for i, (label, sig) in enumerate(sorted(harmonic_signals.items())):
        sig_off = sig - offset
        if len(component_time) == len(sig_off):
            axes[2].plot(component_time, sig_off, color=colors[i % len(colors)],
                         linewidth=0.8, label=label)
        offset += max(np.abs(sig)) * 1.5
    axes[2].set_ylabel("Signal (offset)", fontsize=11)
    axes[2].set_title("Per-frequency contributions", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    if harmonic_signals:
        axes[2].legend(fontsize=8, loc="upper right", ncol=2)

    # ---------------------------------------------------------------
    # Panel 4: Residual
    # ---------------------------------------------------------------
    residual_time = (np.asarray(window_time).ravel()
                     if has_window and window_time is not None else time)
    if len(residual_time) == len(residual):
        axes[3].plot(residual_time, residual, "b-", linewidth=0.8, label="Residual")
    axes[3].axhline(0, color="gray", linestyle=":", linewidth=0.5)
    axes[3].set_xlabel("Time [s]", fontsize=11)
    axes[3].set_ylabel("Residual", fontsize=11)
    axes[3].set_title(f"Residual (zero-mean basis)  |  RMS = {rms_err:.3f}", fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# -------------------------------------------------------------------
#  Disturbance window diagnostics plot
# -------------------------------------------------------------------

def plot_disturbance_window_diagnostics(time, signal, detection_metric,
                                         baseline, noise_scale,
                                         onset_threshold, offset_threshold,
                                         metric_baseline=None,
                                         metric_noise_scale=None,
                                         onset_time=None, offset_time=None,
                                         peak_time=None,
                                         start_time=None, end_time=None,
                                         probe_label="",
                                         output_path=None, figsize=(12, 8)):
    """Diagnostic figure for the disturbance window detector.

    Three vertically stacked panels:
        1. Raw signal with onset/offset markers and window shading.
        2. Detection metric with baseline, onset/offset thresholds, and
           shaded window.
        3. Zoom-in on the detection onset region.

    Parameters
    ----------
    time : ndarray — time array [s]
    signal : ndarray — probe signal
    detection_metric : ndarray — smoothed energy metric
    baseline : float — pre-event median
    noise_scale : float — pre-event noise scale
    onset_threshold : float — threshold for onset
    offset_threshold : float — threshold for offset (hysteresis)
    metric_baseline, metric_noise_scale : float, optional — pre-event
        statistics in the units of ``detection_metric``
    onset_time, offset_time : float or None — detected boundaries [s]
    peak_time : float or None — selected packet's energy peak [s]
    start_time, end_time : float or None — final window [s]
    probe_label : str
    output_path : str, optional
    figsize : tuple, default (12, 8)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    has_window = start_time is not None and end_time is not None
    has_onset = onset_time is not None
    has_offset = offset_time is not None
    has_peak = peak_time is not None

    metric_baseline = baseline if metric_baseline is None else metric_baseline
    metric_noise_scale = noise_scale if metric_noise_scale is None else metric_noise_scale
    meta = f"metric baseline={metric_baseline:.3e}, noise={metric_noise_scale:.3e}"
    if has_onset:
        meta += f"  onset={onset_threshold:.3e}"
    if has_offset:
        meta += f"  offset={offset_threshold:.3e}"

    # --- Panel 1: Raw signal ---
    axes[0].plot(time, signal, "k-", linewidth=0.8)
    axes[0].axhline(baseline, color="gray", linestyle=":", linewidth=0.7,
                    label=f"Baseline = {baseline:.3e}")

    if has_window:
        axes[0].axvspan(start_time, end_time, alpha=0.12, color="C1",
                        label=f"Window [{start_time:.3e}, {end_time:.3e}] s")
    if has_onset:
        axes[0].axvline(onset_time, color="C1", linestyle="--", linewidth=1.0,
                        label=f"Onset = {onset_time:.3e} s")
    if has_offset:
        axes[0].axvline(offset_time, color="C2", linestyle="--", linewidth=1.0,
                        label=f"Offset = {offset_time:.3e} s")
    if has_peak:
        axes[0].axvline(peak_time, color="C3", linestyle=":", linewidth=1.2,
                        label=f"Selected peak = {peak_time:.3e} s")

    axes[0].set_ylabel("Signal", fontsize=11)
    axes[0].set_title(f"Disturbance window detector — {probe_label}", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")

    # --- Panel 2: Detection metric ---
    axes[1].semilogy(time, detection_metric, "b-", linewidth=0.8,
                     label="Metric")
    axes[1].axhline(metric_baseline, color="gray", linestyle=":", linewidth=0.7,
                    label=f"Metric baseline = {metric_baseline:.3e}")
    axes[1].axhline(onset_threshold, color="C1", linestyle="--", linewidth=0.8,
                    label=f"Onset threshold = {onset_threshold:.3e}")
    axes[1].axhline(offset_threshold, color="C2", linestyle="--", linewidth=0.8,
                    label=f"Offset threshold = {offset_threshold:.3e}")

    if has_window:
        axes[1].axvspan(start_time, end_time, alpha=0.10, color="C1")
    if has_onset:
        axes[1].axvline(onset_time, color="C1", linestyle=":", linewidth=0.8)
    if has_offset:
        axes[1].axvline(offset_time, color="C2", linestyle=":", linewidth=0.8)
    if has_peak:
        axes[1].axvline(peak_time, color="C3", linestyle=":", linewidth=1.0)

    axes[1].set_ylabel("Detection metric", fontsize=11)
    axes[1].set_title(f"Detection metric — {meta}", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")

    # --- Panel 3: Zoom on onset ---
    if has_onset:
        zoom_pad = max(5e-5, (time[-1] - time[0]) * 0.02)
        zoom_start = onset_time - 5 * zoom_pad
        zoom_end = onset_time + 5 * zoom_pad
        zoom_mask = (time >= zoom_start) & (time <= zoom_end)
        if np.any(zoom_mask):
            axes[2].plot(time[zoom_mask], signal[zoom_mask], "k-",
                         linewidth=0.8, label="Signal")
            axes[2].plot(time[zoom_mask], detection_metric[zoom_mask],
                         "b-", linewidth=0.8, alpha=0.7, label="Metric")
            axes[2].axhline(onset_threshold, color="C1", linestyle="--",
                            linewidth=0.8, label=f"Onset = {onset_threshold:.3e}")
            axes[2].axhline(offset_threshold, color="C2", linestyle="--",
                            linewidth=0.8, alpha=0.5)
            axes[2].axvline(onset_time, color="C1", linestyle=":", linewidth=1.0,
                            label=f"t_onset = {onset_time:.3e} s")
            if has_window:
                axes[2].axvspan(start_time, end_time, alpha=0.10, color="C1")
            axes[2].set_xlabel("Time [s]", fontsize=11)
            axes[2].set_ylabel("Signal / Metric", fontsize=11)
            axes[2].set_title("Onset zoom", fontsize=11)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=8, loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "No onset detected",
                     transform=axes[2].transAxes, ha="center", va="center",
                     fontsize=12, color="gray", style="italic")
        axes[2].set_xlabel("Time [s]", fontsize=11)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_energy_budget_vs_x(probe_x, energy_budgets, output_path=None,
                             figsize=(12, 5)):
    """Plot energy fractions (fundamental, harmonic, residual) vs streamwise position.

    Parameters
    ----------
    probe_x : array-like — probe x-positions [cm]
    energy_budgets : list of dict — one per probe, from ``compute_energy_budget``
    output_path : str, optional
    figsize : tuple, default (12, 5)

    Returns
    -------
    matplotlib.figure.Figure
    """
    probe_x = np.asarray(probe_x, dtype=float)
    n = len(energy_budgets)

    E_total = np.array([eb["E_total"] for eb in energy_budgets])
    E_recon = np.array([eb["E_recon_frac"] for eb in energy_budgets])
    E_resid = np.array([eb["E_residual_frac"] for eb in energy_budgets])

    # Sort by x for clean monotonic plots
    sort_idx = np.argsort(probe_x)
    probe_x = probe_x[sort_idx]
    E_total = E_total[sort_idx]
    E_recon = E_recon[sort_idx]
    E_resid = E_resid[sort_idx]

    # Extract metadata if present
    mean_bg = np.array([eb.get("mean_background", np.nan) for eb in energy_budgets])
    method = energy_budgets[0].get("reconstruction_method", "unknown") if energy_budgets else "unknown"
    f0 = energy_budgets[0].get("fundamental_freq", None) if energy_budgets else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: stacked area — fractions
    ax1.fill_between(probe_x, 0, E_recon, color="C0", alpha=0.6,
                     label="Reconstruction")
    ax1.fill_between(probe_x, E_recon, E_recon + E_resid, color="C1", alpha=0.6,
                     label="Residual")
    ax1.set_ylabel("Energy fraction", fontsize=11)
    title1 = "Energy budget vs x — reconstruction vs residual"
    if f0 is not None:
        title1 += f"  |  f₀={f0:.3e} Hz"
    ax1.set_title(title1, fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Bottom: log total energy
    ax2.semilogy(probe_x, E_total, "ko-", markersize=3, linewidth=1.0,
                 label="Total mean-square energy")
    ax2.set_xlabel("x [cm]", fontsize=11)
    ax2.set_ylabel("Mean-square energy", fontsize=11)
    ax2.set_title("Total disturbance energy (zero-mean basis)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Add annotation
    info_text = f"Method: {method}  |  N={n} probes"
    if not np.all(np.isnan(mean_bg)):
        info_text += f"  |  mean_bg ≈ {np.nanmean(mean_bg):.1f}"
    fig.text(0.5, 0.01, info_text, ha="center", fontsize=9,
             style="italic", color="dimgray")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_residual_vs_x(probe_x, residual_stats, output_path=None,
                        figsize=(12, 8)):
    """Plot residual RMS, R², skewness, and kurtosis vs streamwise position.

    Parameters
    ----------
    probe_x : array-like — probe x-positions [cm]
    residual_stats : list of dict — one per probe, from ``compute_residual_stats``
    output_path : str, optional
    figsize : tuple, default (12, 8)

    Returns
    -------
    matplotlib.figure.Figure
    """
    probe_x = np.asarray(probe_x, dtype=float)
    n = len(residual_stats)

    rms_rel = np.array([rs["rms_residual_rel"] for rs in residual_stats])
    R_sq = np.array([rs["R_squared"] for rs in residual_stats])
    skew = np.array([rs["skewness"] for rs in residual_stats])
    kurt = np.array([rs["kurtosis"] for rs in residual_stats])

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Sort by x for clean monotonic plots
    sort_idx = np.argsort(probe_x)
    probe_x = probe_x[sort_idx]
    rms_rel = rms_rel[sort_idx]
    R_sq = R_sq[sort_idx]
    skew = skew[sort_idx]
    kurt = kurt[sort_idx]

    axes[0].plot(probe_x, rms_rel, "bo-", markersize=4, linewidth=1.0)
    axes[0].set_ylabel("RMS residual (rel.)", fontsize=11)
    axes[0].set_title("Residual RMS (rel. to measured signal variance)", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(probe_x, R_sq, "go-", markersize=4, linewidth=1.0)
    axes[1].axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="0.95")
    axes[1].axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="0.99")
    axes[1].set_ylabel("R²", fontsize=11)
    axes[1].set_title("Coefficient of determination (1 = perfect reconstruction)", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(probe_x, skew, "mo-", markersize=4, linewidth=1.0)
    axes[2].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Skewness", fontsize=11)
    axes[2].set_title("Residual skewness (0 = symmetric; |skew| > 2 suggests nonlinearity)", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(probe_x, kurt, "co-", markersize=4, linewidth=1.0)
    axes[3].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[3].set_xlabel("x [cm]", fontsize=11)
    axes[3].set_ylabel("Kurtosis (excess)", fontsize=11)
    axes[3].set_title("Residual excess kurtosis (0 = Gaussian; >> 0 = heavy tails)", fontsize=12)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_harmonic_recon_comparison(time_uniform, signal_matrix, probe_data,
                                    probe_x, energy_budgets, residual_stats_chosen,
                                    n_valid, output_dir, probe_prefix,
                                    probe_indices=None,
                                    harmonic_freq=None, num_harmonics=5):
    """Convenience: loop over selected probes and generate reconstruction plots.

    Parameters
    ----------
    time_uniform : ndarray — common time base
    signal_matrix : ndarray, (L, n_valid) — preprocessed signals
    probe_data : list of dict — probe metadata
    probe_x : list of float — probe x-positions [cm]
    energy_budgets : list of dict — per-probe energy budgets
    residual_stats_chosen : list of dict — per-probe residual stats
    n_valid : int — number of valid probes
    output_dir : Path — output directory for figures
    probe_prefix : str — prefix for filenames
    probe_indices : list of int, optional — which probes to plot (default [0,49,99])
    harmonic_freq : float, optional — fundamental harmonic frequency [Hz]
    num_harmonics : int, default 5 — number of harmonics to reconstruct
    """
    if harmonic_freq is None:
        harmonic_freq = 10.0e6
    if probe_indices is None:
        probe_indices = [0, min(49, n_valid - 1), min(99, n_valid - 1)]

    for idx in probe_indices:
        if idx >= n_valid:
            continue
        sim_signal = signal_matrix[:, idx]
        p = probe_data[idx]
        label = f"Probe {idx} — x={p['x']:.3f} cm"

        # Reconstruct harmonics for this probe
        dt = p.get("dt", 1.0)
        recon_total, harm_sigs, _ = fdb_mod.reconstruct_from_harmonics(
            sim_signal, dt,
            harmonic_freq=harmonic_freq,
            num_harmonics=num_harmonics,
        )
        recon_total = recon_total.ravel()
        for key in harm_sigs:
            harm_sigs[key] = harm_sigs[key].ravel()

        residual = sim_signal - recon_total
        out = output_dir / f"{probe_prefix}_recon_probe{idx:03d}.png"
        plot_harmonic_reconstruction(
            time_uniform, sim_signal, recon_total, residual,
            harm_sigs, probe_label=label, output_path=str(out),
        )


# ===========================================================================
#  PHASE 2: TRANSIENT ANALYSIS PLOTS
# ===========================================================================

def plot_spectrogram(time, signal, fs, output_path=None, fmax=None,
                     title="Spectrogram", figsize=(12, 5), nperseg=256,
                     noverlap=None, scale="db", vmin=None, vmax=None):
    """Plot STFT spectrogram of a probe signal.

    Parameters
    ----------
    time : ndarray — time [s]
    signal : ndarray — signal
    fs : float — sampling frequency [Hz]
    output_path : str, optional
    fmax : float, optional — frequency limit for y-axis
    title : str, default "Spectrogram"
    figsize : tuple, default (12, 5)
    nperseg, noverlap : int, optional
        STFT segment length and overlap passed to the spectral estimator.
    scale : {'linear', 'db'}
        Display linear PSD or PSD in decibels. Linear PSD uses a zero lower
        colour limit unless ``vmin`` is supplied.
    vmin, vmax : float, optional
        Colour limits in the units selected by ``scale``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = np.asarray(signal, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    scale = str(scale).lower()
    f, t_relative, spectral_values = fdb_mod.compute_spectrogram(
        signal, fs, nperseg=nperseg, noverlap=noverlap,
        output_scale=scale,
    )
    t = t_relative + float(time[0])
    if scale == "linear":
        if vmin is None:
            vmin = 0.0
        colorbar_label = r"PSD [signal$^2$/Hz]"
    else:
        colorbar_label = r"PSD [dB re signal$^2$/Hz]"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        t, f, spectral_values, shading="auto", cmap="inferno",
        rasterized=True, vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(colorbar_label, fontsize=10)

    ax.set_xlabel("Flow time [s]", fontsize=11)
    ax.set_ylabel("Frequency [Hz]", fontsize=11)
    ax.set_title(title, fontsize=12)
    if fmax is not None:
        ax.set_ylim(0, fmax)
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_fft_stft_comparison(
        time, signal, fs, output_path=None, fmax=None,
        title="Full-record FFT and STFT comparison",
        signal_label="Mean-subtracted signal", nperseg=256,
        noverlap=None, spectrogram_scale="linear",
        spectrogram_vmin=0.0, spectrogram_vmax=None,
        fft_window="hann", figsize=(12, 11)):
    """Plot one probe's mean-subtracted history, FFT, and STFT together."""
    time = np.asarray(time, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    if time.size != signal.size or time.size < 8:
        raise ValueError(
            "time and signal must have the same length of at least 8"
        )
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("time and signal must be finite")
    signal_mean_subtracted = signal - np.mean(signal)
    window_name = str(fft_window).lower()
    if window_name == "hann":
        window = np.hanning(signal.size)
    elif window_name == "hamming":
        window = np.hamming(signal.size)
    elif window_name == "blackman":
        window = np.blackman(signal.size)
    elif window_name in ("none", "rect", "rectangular"):
        window = np.ones(signal.size)
        window_name = "rectangular"
    else:
        raise ValueError(
            "fft_window must be hann, hamming, blackman, or rectangular"
        )
    amplitude_scale = signal.size / max(np.sum(window), 1.0e-300)
    fft_frequency = np.fft.rfftfreq(signal.size, 1.0 / fs)
    fft_amplitude = (
        np.abs(np.fft.rfft(signal_mean_subtracted * window))
        / signal.size * amplitude_scale
    )
    if signal.size > 2:
        fft_amplitude[1:-1] *= 2.0

    spectrogram_scale = str(spectrogram_scale).lower()
    stft_frequency, stft_time_relative, spectral_values = (
        fdb_mod.compute_spectrogram(
            signal_mean_subtracted, fs, nperseg=nperseg,
            noverlap=noverlap, output_scale=spectrogram_scale,
        )
    )
    if spectrogram_scale == "linear":
        colorbar_label = r"STFT PSD [signal$^2$/Hz]"
    else:
        colorbar_label = r"STFT PSD [dB re signal$^2$/Hz]"

    record_time_us = (time - time[0]) * 1.0e6
    stft_time_us = stft_time_relative * 1.0e6
    figure, axes = plt.subplots(3, 1, figsize=figsize)
    axes[0].plot(record_time_us, signal_mean_subtracted, linewidth=0.75)
    axes[0].set_xlabel(r"Time from record start [$\mu$s]")
    axes[0].set_ylabel(signal_label)
    axes[0].set_title("Mean-subtracted probe history")

    positive = fft_frequency > 0.0
    if fmax is not None:
        positive &= fft_frequency <= float(fmax)
    axes[1].semilogy(
        fft_frequency[positive] * 1.0e-6,
        np.maximum(
            fft_amplitude[positive], np.finfo(float).tiny
        ),
        linewidth=0.8,
    )
    axes[1].set_xlabel("Frequency [MHz]")
    axes[1].set_ylabel("One-sided FFT amplitude")
    axes[1].set_title(
        f"Full-record FFT ({window_name} window, amplitude corrected)"
    )
    if fmax is not None:
        axes[1].set_xlim(0.0, float(fmax) * 1.0e-6)

    mesh = axes[2].pcolormesh(
        stft_time_us, stft_frequency * 1.0e-6, spectral_values,
        shading="auto", cmap="inferno", rasterized=True,
        vmin=spectrogram_vmin, vmax=spectrogram_vmax,
    )
    colorbar = figure.colorbar(mesh, ax=axes[2], pad=0.02)
    colorbar.set_label(colorbar_label)
    axes[2].set_xlabel(r"Time from record start [$\mu$s]")
    axes[2].set_ylabel("Frequency [MHz]")
    axes[2].set_title(
        f"STFT spectrogram ({nperseg} samples per segment)"
    )
    if fmax is not None:
        axes[2].set_ylim(0.0, float(fmax) * 1.0e-6)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"{title}\nAbsolute record start: {time[0]:.9f} s"
    )
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=175, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_envelope_with_signal(time, signal_raw, signal_filtered, envelope,
                               packet_stats, output_path=None,
                               title="Envelope analysis", figsize=(12, 8),
                               instantaneous_frequency=None,
                               frequency_band=None):
    """3-panel: raw+filtered, envelope, instantaneous frequency.

    Parameters
    ----------
    time : ndarray
    signal_raw : ndarray — original signal
    signal_filtered : ndarray — bandpass-filtered signal
    envelope : ndarray — Hilbert envelope
    packet_stats : dict — from ``extract_packet_stats``
    instantaneous_frequency : ndarray, optional
        Hilbert instantaneous frequency [Hz]. Values where the envelope is
        below 5% of its peak are hidden because phase is poorly conditioned.
    frequency_band : tuple, optional
        Expected frequency band [Hz], used to set the third-panel limits.
    output_path : str, optional
    title : str
    figsize : tuple, default (12, 8)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.plot(time, signal_raw, "k-", linewidth=0.5, alpha=0.5, label="Raw")
    ax.plot(time, signal_filtered, "r-", linewidth=1.0, label="Filtered")
    ax.set_ylabel("Signal", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)

    ax = axes[1]
    ax.plot(time, signal_filtered, "r-", linewidth=0.5, alpha=0.3)
    ax.plot(time, envelope, "b-", linewidth=1.5, label="Envelope")
    if packet_stats.get("peak_time") is not None:
        pt = packet_stats["peak_time"]
        pa = packet_stats["peak_amplitude"]
        ax.axvline(pt, color="gray", linestyle=":", alpha=0.7)
        ax.plot(pt, pa, "ko", markersize=6)
    if packet_stats.get("arrival_time") is not None:
        ax.axvline(packet_stats["arrival_time"], color="green",
                   linestyle="--", alpha=0.5, label="Arrival")
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if instantaneous_frequency is not None:
        inst_freq = np.asarray(instantaneous_frequency, dtype=float).ravel()
        if len(inst_freq) != len(time):
            raise ValueError("instantaneous_frequency must match time length")
        reliable = envelope >= 0.05 * np.nanmax(envelope)
        ax.plot(time, np.where(reliable, inst_freq / 1.0e6, np.nan),
                color="C2", linewidth=0.9)
        ax.set_ylabel("Instantaneous\nfrequency [MHz]", fontsize=11)
        if frequency_band is not None:
            ax.set_ylim(float(frequency_band[0]) / 1.0e6,
                        float(frequency_band[1]) / 1.0e6)
    else:
        ax.text(0.5, 0.5, "Instantaneous frequency not supplied",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlabel("Time [s]", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_envelope_growth(probe_x, packet_stats_list, output_path=None,
                          figsize=(12, 8)):
    """Plot envelope metrics vs streamwise position.

    Parameters
    ----------
    probe_x : array-like — x-positions [cm]
    packet_stats_list : list of dict — one per probe from ``extract_packet_stats``
    output_path : str, optional
    figsize : tuple, default (12, 8)

    Returns
    -------
    matplotlib.figure.Figure
    """
    probe_x = np.asarray(probe_x, dtype=float)
    peak_amp = np.array([ps.get("peak_amplitude", np.nan) for ps in packet_stats_list])
    arrival = np.array([ps.get("arrival_time", np.nan) for ps in packet_stats_list])
    energy = np.array([ps.get("integrated_energy", np.nan) for ps in packet_stats_list])

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    ax.semilogy(probe_x, peak_amp, "ro-", markersize=4, linewidth=1.0)
    ax.set_ylabel("Peak envelope amplitude", fontsize=11)
    ax.set_title("Envelope peak growth vs x", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    valid = np.isfinite(arrival)
    if np.any(valid):
        ax.plot(probe_x[valid], arrival[valid], "bs-", markersize=4, linewidth=1.0)
        ax.set_ylabel("Arrival time [s]", fontsize=11)
        ax.set_title("Envelope arrival time vs x", fontsize=12)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(probe_x, energy, "go-", markersize=4, linewidth=1.0)
    ax.set_xlabel("x [cm]", fontsize=11)
    ax.set_ylabel("Integrated energy", fontsize=11)
    ax.set_title("Wavepacket energy vs x", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ===========================================================================
#  PHASE 3: NONLINEAR DIAGNOSTICS PLOTS (BISPECTRUM)
# ===========================================================================

def plot_bicoherence_map(freq, bicoh_matrix, output_path=None, fmax=None,
                         title="Squared bicoherence", figsize=(10, 8)):
    """2-D contour plot of b²(f1, f2) with labeled triads and significance.

    Parameters
    ----------
    freq : ndarray — frequency bins [Hz]
    bicoh_matrix : ndarray, (n_freq, n_freq) — b² values
    output_path : str, optional
    fmax : float, optional — frequency limit
    title : str
    figsize : tuple, default (10, 8)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_freq = len(freq)

    # Upper triangle only
    tri_upper = np.triu(bicoh_matrix)
    tri_upper[tri_upper < 1e-6] = np.nan  # mask near-zero for visual clarity

    im = ax.pcolormesh(freq[:n_freq], freq[:n_freq], tri_upper,
                       cmap="hot", shading="auto", vmin=0, vmax=1,
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label(r"$b^2$", fontsize=12)

    ax.plot([0, freq[-1]], [0, freq[-1]], "w--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(r"$f_1$ [Hz]", fontsize=12)
    ax.set_ylabel(r"$f_2$ [Hz]", fontsize=12)
    ax.set_title(title, fontsize=13)
    if fmax is not None:
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, fmax)
    ax.set_aspect("equal")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_bicoherence_vs_x(probe_x, triad_bicoh_list, output_path=None,
                           figsize=(12, 5), reference_threshold=None,
                           reference_label="Reference threshold"):
    """Plot bicoherence at key triads vs streamwise position.

    Parameters
    ----------
    probe_x : array-like — x-positions [cm]
    triad_bicoh_list : list of dict — one per probe from ``extract_triad_bicoherence``
    output_path : str, optional
    figsize : tuple, default (12, 5)
    reference_threshold : float, optional
        Optional externally justified threshold. It is deliberately not
        labelled as statistical significance by this plotting routine.

    Returns
    -------
    matplotlib.figure.Figure
    """
    probe_x = np.asarray(probe_x, dtype=float)

    # Collect all triad labels
    all_labels = set()
    for tb in triad_bicoh_list:
        all_labels.update(tb.keys())
    all_labels = sorted(all_labels)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_labels)))

    for li, label in enumerate(all_labels):
        vals = np.array([tb.get(label, np.nan) for tb in triad_bicoh_list])
        valid = np.isfinite(vals)
        if np.sum(valid) < 3:
            continue
        ax.plot(probe_x[valid], vals[valid], "o-", markersize=3, linewidth=0.8,
                color=colors[li], label=label)

    if reference_threshold is not None:
        ax.axhline(float(reference_threshold), color="gray", linestyle=":",
                   alpha=0.6, label=reference_label)
    ax.set_xlabel("x [cm]", fontsize=12)
    ax.set_ylabel(r"$b^2$", fontsize=12)
    ax.set_title("Triad bicoherence vs streamwise position", fontsize=13)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
