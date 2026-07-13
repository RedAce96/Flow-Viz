# =============================================================================
#  PeleC Post-Processing: Functions Database
# =============================================================================
#  Core data I/O, derived-field computation, extraction, and geometry
#  functions for PeleC / AMReX plotfile post-processing.
#
#  All data loaders return a ``StandardDataset`` dict with canonical field
#  names so downstream plotting / analysis functions are solver-agnostic.
#
#  For interns / new GRAs:
#    - You should NOT need to edit this file for normal use.
#    - All user settings go in ``pelec_post.py`` (the execution script).
# =============================================================================

import os
import re
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# 0.  LOGGING / ERROR HANDLING
# ---------------------------------------------------------------------------

_DEBUG_MODE = False


def set_debug_mode(enabled: bool):
    """Toggle whether full tracebacks are written to ``error_log.txt``."""
    global _DEBUG_MODE
    _DEBUG_MODE = enabled


def _log_error(msg: str, exception: Exception = None):
    """Print a concise error message.  In debug mode, also write the full
    traceback to ``error_log.txt``."""
    print(f"  [ERROR] {msg}")
    if exception is not None:
        print(f"          ({type(exception).__name__}: {exception})")
    if _DEBUG_MODE and exception is not None:
        with open("error_log.txt", "a") as fh:
            fh.write(f"\n{'='*60}\n")
            fh.write(f"ERROR: {msg}\n")
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=fh)


# ---------------------------------------------------------------------------
# 1.  STANDARD DATA FORMAT
# ---------------------------------------------------------------------------

# Canonical field names used throughout the post-processing pipeline.
# Solver-specific loaders map raw names to these canonical names.
_CANONICAL_DERIVED_FIELDS = {
    "velocity_magnitude",
    "mach_number",
    "vorticity",
    "vorticity_magnitude",
    "schlieren",
    "P/Pinf",
    "U/Uinf",
    "rho/rhoinf",
    "P/P_dyn",
}

# Default aliases for PeleC / AMReX plotfiles.
_PELEC_DEFAULT_ALIASES = {
    "Temp":               "temperature",
    "pressure":           "pressure",
    "density":            "density",
    "x_velocity":         "x_velocity",
    "y_velocity":         "y_velocity",
    "z_velocity":         "z_velocity",
    "vfrac":              "volume_fraction",
    "MachNumber":         "mach_number",
    "magvel":             "velocity_magnitude",
    "magvort":            "vorticity_magnitude",
    "temp":               "temperature",
    "pres":               "pressure",
    "rho":                "density",
    "velx":               "x_velocity",
    "vely":               "y_velocity",
    "velz":               "z_velocity",
}

# Unit conversion factors (CGS -> MKS)
_CGS_TO_MKS = {
    "density":        1.0e3,   # g/cm^3 -> kg/m^3
    "pressure":       1.0e-1,  # dyne/cm^2 -> Pa
    "x_velocity":     1.0e-2,  # cm/s -> m/s
    "y_velocity":     1.0e-2,
    "z_velocity":     1.0e-2,
    "velocity_magnitude": 1.0e-2,
    "viscosity":      1.0e-1,  # g/(cm s) -> Pa s
    "bulk_viscosity": 1.0e-1,
    "conductivity":   1.0e-5,  # erg/(cm s K) -> W/(m K)
}


# ---------------------------------------------------------------------------
# 2.  DATA I/O  (PeleC / AMReX plotfiles)
# ---------------------------------------------------------------------------

def natural_sort_key(name):
    """Split a string into numeric / non-numeric parts for natural sorting.

    Example
    -------
    >>> sorted(["plt10", "plt1", "plt2"], key=natural_sort_key)
    ['plt1', 'plt2', 'plt10']
    """
    parts = re.split(r"(\d+)", str(name))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _resolve_field_aliases(raw_names, alias_map=None):
    """Build a canonical-name -> raw-name(s) mapping from user input.

    Parameters
    ----------
    raw_names : None, dict, or iterable
        *None*  -> use the default PeleC aliases.
        *dict*  -> keys are canonical names, values are raw solver names.
        *iterable* -> each item is treated as both canonical and raw name.
    alias_map : dict, optional
        Additional aliases merged on top of the defaults.

    Returns
    -------
    dict
        ``{canonical_name: [raw_name, ...]}``  (list of candidates)
    """
    aliases = dict(_PELEC_DEFAULT_ALIASES)
    if alias_map is not None:
        aliases.update(alias_map)

    # Collect all raw candidates for each canonical name
    canonical_to_raws = {}
    for raw, canonical in aliases.items():
        canonical_to_raws.setdefault(canonical, []).append(raw)

    if raw_names is None:
        return canonical_to_raws

    if isinstance(raw_names, dict):
        out = {}
        for canonical, raw in raw_names.items():
            # If the user provided a raw name that is itself a canonical name
            # in the default map, expand it to all known candidates.
            if raw in canonical_to_raws:
                out[canonical] = canonical_to_raws[raw]
            else:
                out[canonical] = [raw]
        return out

    # Iterable of names — treat each as canonical = raw
    return {str(name): [str(name)] for name in raw_names}


def _collapse_field(values, dimensionality):
    """Collapse trailing dimensions for lower-dimensional datasets."""
    values = np.asarray(values)
    if dimensionality == 2 and values.ndim == 3:
        return values[:, :, 0]
    if dimensionality == 1 and values.ndim >= 2:
        return values.reshape(values.shape[0])
    return values


def load_pelec_plotfile(plotfile_path, field_names=None, alias_map=None,
                        convert_to_mks=True):
    """Load a single PeleC / AMReX plotfile into a ``StandardDataset``.

    Parameters
    ----------
    plotfile_path : str or path-like
        Path to a single plotfile directory.
    field_names : None, dict, or iterable
        Fields to load.  *None* uses a sensible default set.
    alias_map : dict, optional
        Extra solver-specific aliases merged with defaults.
    convert_to_mks : bool, default True
        Convert CGS coordinates / supported fields to MKS.

    Returns
    -------
    dict
        ``StandardDataset`` with canonical field names.
    """
    try:
        import yt
    except ImportError as exc:
        raise ImportError(
            "yt is required to read PeleC AMReX plotfiles.  "
            "Install it with:  pip install yt"
        ) from exc

    plotfile_path = os.fspath(plotfile_path)
    field_map = _resolve_field_aliases(field_names, alias_map)

    ds = yt.load(plotfile_path)
    finest_level = ds.index.max_level
    refine_factor = ds.refine_by ** finest_level
    dims = ds.domain_dimensions * refine_factor
    if ds.dimensionality < 3:
        dims = np.asarray(dims, dtype=int)
        dims[ds.dimensionality:] = 1

    cover = ds.covering_grid(
        level=finest_level,
        left_edge=ds.domain_left_edge,
        dims=dims,
    )

    left_edge = ds.domain_left_edge.d
    right_edge = ds.domain_right_edge.d
    nx, ny = int(dims[0]), int(dims[1])
    dx = (right_edge[0] - left_edge[0]) / nx
    dy = (right_edge[1] - left_edge[1]) / ny
    x = np.linspace(left_edge[0], right_edge[0], nx, endpoint=False) + 0.5 * dx
    y = np.linspace(left_edge[1], right_edge[1], ny, endpoint=False) + 0.5 * dy

    if convert_to_mks:
        x = x / 100.0
        y = y / 100.0

    available_fields = {name for _, name in ds.field_list}
    # Case-insensitive lookup table for fallback
    available_fields_lower = {name.lower(): name for name in available_fields}

    fields = {}
    raw_field_names = {}

    for canonical, raw_candidates in field_map.items():
        # Try each candidate raw name in order
        matched_raw = None
        for raw in raw_candidates:
            if raw in available_fields:
                matched_raw = raw
                break
            # Fallback: case-insensitive match
            if raw.lower() in available_fields_lower:
                matched_raw = available_fields_lower[raw.lower()]
                break

        if matched_raw is None:
            # When field_names is explicitly provided, raise an error.
            # When using the default alias set, just skip missing fields
            # (e.g. volume_fraction on non-EB grids).
            if field_names is not None:
                raise KeyError(
                    f"Field '{canonical}' not found in plotfile.  "
                    f"Tried candidates: {raw_candidates}.  "
                    f"Available fields include: {sorted(available_fields)[:12]}"
                )
            continue

        values = _collapse_field(cover[("boxlib", matched_raw)], ds.dimensionality)
        values = np.asarray(values).squeeze()
        if convert_to_mks:
            values = values * _CGS_TO_MKS.get(canonical, 1.0)
        fields[canonical] = np.asarray(values, dtype=float)
        raw_field_names[canonical] = matched_raw

    return {
        "source": plotfile_path,
        "case_label": os.path.basename(os.path.dirname(plotfile_path)),
        "plot_label": os.path.basename(plotfile_path),
        "time": float(ds.current_time),
        "x": x,
        "y": y,
        "fields": fields,
        "raw_field_names": raw_field_names,
        "available_fields": sorted(available_fields),
        "units": "MKS" if convert_to_mks else "CGS",
        "dimensionality": int(ds.dimensionality),
    }


def discover_plotfile_paths(plot_source, plot_prefix="plt",
                           start=None, end=None, step=1):
    """Discover plotfile paths without loading data.

    Parameters
    ----------
    plot_source : str or iterable of paths
        Case directory containing plotfiles, or explicit list of paths.
    plot_prefix : str, default 'plt'
        Directory prefix when auto-discovering.
    start, end, step : int, optional
        Filter by the trailing numerical suffix in the plotfile name
        (e.g. ``pltFlatPlateFlow05000`` → suffix 5000).  Falls back to
        0-based index slicing if no number is found in the filenames.

    Returns
    -------
    list[str]
        Absolute paths to plotfile directories, sorted.
    """
    if isinstance(plot_source, (str, os.PathLike)):
        source_path = os.path.abspath(os.fspath(plot_source))

        if os.path.isdir(source_path) and os.path.isfile(os.path.join(source_path, "Header")):
            plotfiles = [source_path]
        elif os.path.isdir(source_path):
            plotfiles = []
            for entry in os.listdir(source_path):
                entry_path = os.path.join(source_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                if not entry.startswith(plot_prefix):
                    continue
                if ".old." in entry:
                    continue
                if not os.path.isfile(os.path.join(entry_path, "Header")):
                    continue
                plotfiles.append(entry_path)
            plotfiles.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
        else:
            raise FileNotFoundError(f"Could not find plotfile source: {plot_source}")
    else:
        plotfiles = [os.path.abspath(os.fspath(p)) for p in plot_source]
        plotfiles.sort(key=lambda p: natural_sort_key(os.path.basename(p)))

    # --- Range filter ---
    # If start/end/step are given, filter by the trailing numerical suffix
    # in each plotfile name (e.g. pltFlatPlateFlow05000 -> suffix 5000).
    # This is what most users intuitively expect.
    # Fall back to 0-based index slicing if no suffix can be extracted.
    if start is not None or end is not None or step != 1:
        # Extract trailing numbers from all basenames
        suffixes = []
        for p in plotfiles:
            m = re.search(r"(\d+)$", os.path.basename(p))
            suffixes.append(int(m.group(1)) if m else None)

        # Use suffix-based filtering only if EVERY file has a parseable suffix
        if all(s is not None for s in suffixes):
            n_start = start if start is not None else min(suffixes)
            n_end   = end   if end   is not None else max(suffixes)
            filtered = []
            for p, s in zip(plotfiles, suffixes):
                if n_start <= s <= n_end and (s - n_start) % step == 0:
                    filtered.append(p)
            plotfiles = filtered
        else:
            # Fallback: 0-based index slicing
            total = len(plotfiles)
            start_idx = start if start is not None else 0
            end_idx   = end   if end   is not None else total
            plotfiles = plotfiles[start_idx:end_idx:step]

    return plotfiles


def load_pelec_plotfile_series(plot_source, plot_prefix="plt",
                               field_names=None, alias_map=None,
                               convert_to_mks=True,
                               start=None, end=None, step=1):
    """Discover and load a series of PeleC plotfiles.

    Parameters
    ----------
    plot_source : str or iterable of paths
        Case directory containing plotfiles, or explicit list of paths.
    plot_prefix : str, default 'plt'
        Directory prefix when auto-discovering.
    start, end, step : int, optional
        Filter by the trailing numerical suffix in the plotfile name
        (e.g. ``pltFlatPlateFlow05000`` → suffix 5000).  Falls back to
        0-based index slicing if no number is found in the filenames.

    Returns
    -------
    list[dict]
        ``StandardDataset`` objects, sorted by time / name.
    """
    plotfiles = discover_plotfile_paths(
        plot_source, plot_prefix=plot_prefix,
        start=start, end=end, step=step,
    )

    return [
        load_pelec_plotfile(p, field_names=field_names, alias_map=alias_map,
                            convert_to_mks=convert_to_mks)
        for p in plotfiles
    ]


# ---------------------------------------------------------------------------
# 3.  DERIVED FIELDS
# ---------------------------------------------------------------------------

def compute_derived_fields(dataset, gamma=1.4, R=287.05, beta=50.0):
    """Add commonly-used derived fields to a ``StandardDataset``.

    Adds the following fields (if base fields are present):
        * velocity_magnitude
        * mach_number
        * vorticity
        * vorticity_magnitude
        * schlieren
        * temperature   (from ideal gas law if ``pressure`` and ``density`` exist)
        * P/Pinf, U/Uinf, rho/rhoinf, P/P_dyn

    Parameters
    ----------
    dataset : dict
        ``StandardDataset`` returned by a loader.
    gamma : float, default 1.4
        Ratio of specific heats.
    R : float, default 287.05
        Specific gas constant (J / kg / K).
    beta : float, default 50.0
        Schlieren contrast factor.

    Returns
    -------
    dict
        The same *dataset* dict (modified in-place).
    """
    f = dataset["fields"]
    x = dataset["x"]
    y = dataset["y"]
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0

    # Velocity magnitude
    if "velocity_magnitude" not in f:
        if "x_velocity" in f and "y_velocity" in f:
            f["velocity_magnitude"] = np.sqrt(f["x_velocity"]**2 + f["y_velocity"]**2)
            if "z_velocity" in f:
                f["velocity_magnitude"] = np.sqrt(
                    f["x_velocity"]**2 + f["y_velocity"]**2 + f["z_velocity"]**2
                )

    # Temperature from ideal gas law
    if "temperature" not in f:
        if "pressure" in f and "density" in f:
            f["temperature"] = f["pressure"] / (f["density"] * R)

    # Mach number
    if "mach_number" not in f:
        if "velocity_magnitude" in f and "temperature" in f:
            a = np.sqrt(gamma * R * f["temperature"])
            a = np.where(a > 0, a, np.nan)
            f["mach_number"] = f["velocity_magnitude"] / a

    # Vorticity (2-D in-plane)
    if "vorticity" not in f:
        if "x_velocity" in f and "y_velocity" in f:
            dv_dx, _ = np.gradient(f["y_velocity"], dx, dy)
            _, du_dy = np.gradient(f["x_velocity"], dx, dy)
            f["vorticity"] = dv_dx - du_dy
            f["vorticity_magnitude"] = np.abs(f["vorticity"])

    # Schlieren
    if "schlieren" not in f:
        if "density" in f:
            drho_dy, drho_dx = np.gradient(f["density"], dy, dx)
            grad_mag = np.sqrt(drho_dx**2 + drho_dy**2)
            gmax = grad_mag.max()
            if gmax == 0:
                f["schlieren"] = np.ones_like(grad_mag)
            else:
                f["schlieren"] = np.exp(-beta * (grad_mag / gmax))

    # Normalisations (freestream taken from corner [0, 0])
    if "density" in f:
        rho_inf = f["density"][0, 0]
        if rho_inf > 0:
            f["rho/rhoinf"] = f["density"] / rho_inf
    if "x_velocity" in f:
        u_inf = f["x_velocity"][0, 0]
        if abs(u_inf) > 0:
            f["U/Uinf"] = f["x_velocity"] / u_inf
    if "pressure" in f:
        p_inf = f["pressure"][0, 0]
        if p_inf > 0:
            f["P/Pinf"] = f["pressure"] / p_inf
    if "mach_number" in f and "pressure" in f:
        p_inf = f["pressure"][0, 0]
        M_inf = f["mach_number"][0, 0]
        if p_inf > 0 and M_inf > 0:
            q_dyn = 0.5 * gamma * p_inf * M_inf**2
            f["P/P_dyn"] = f["pressure"] / q_dyn

    return dataset


# ---------------------------------------------------------------------------
# 4.  FIELD EXTRACTION
# ---------------------------------------------------------------------------

def extract_line(dataset, x_location, field_key):
    """Extract a 1-D profile at the nearest x-location.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    x_location : float
        Target x-coordinate.
    field_key : str
        Canonical field name to extract.

    Returns
    -------
    dict
        Profile with keys ``x``, ``y``, ``values``, ``field_key``.
    """
    x = dataset["x"]
    y = dataset["y"]
    fields = dataset["fields"]

    if field_key not in fields:
        raise KeyError(f"Field '{field_key}' not found in dataset.")

    idx = int(np.argmin(np.abs(x - x_location)))
    x_sampled = float(x[idx])
    values = np.asarray(fields[field_key]).squeeze()

    if values.ndim != 2:
        raise ValueError(f"Field '{field_key}' must be 2-D after squeezing; got {values.shape}")

    return {
        "field_key": field_key,
        "x_requested": float(x_location),
        "x_sampled": x_sampled,
        "x": np.full_like(y, x_sampled, dtype=float),
        "y": np.asarray(y, dtype=float),
        "values": np.asarray(values[idx, :], dtype=float),
        "time": dataset.get("time"),
        "source": dataset.get("source"),
    }


def extract_surface_normal_profile(dataset, field_key, x_location,
                                   surface_y=0.0, surface_offset=0.0,
                                   y_max=None, side="positive",
                                   surface_angle_deg=None,
                                   surface_origin=None,
                                   n_samples=None, interp_method="linear"):
    """Extract a wall-normal line profile from a 2-D dataset.

    Supports both vertical (legacy) and angled surface normals.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    field_key : str
        Canonical field name to extract.
    x_location : float
        Target x-location.
    surface_y : float, default 0.0
        Baseline surface location.
    surface_offset : float, default 0.0
        Extra shift applied to ``surface_y``.
    y_max : float, optional
        Maximum wall-normal distance.
    side : {'positive', 'negative'}, default 'positive'
        Direction from the surface.
    surface_angle_deg : float, optional
        Surface angle in degrees (CCW from +x).  If None, vertical path.
    surface_origin : tuple of 2 floats, optional
        Required when ``surface_angle_deg`` is given.
    n_samples : int, optional
        Number of interpolated points for angled extraction.
    interp_method : {'linear', 'nearest'}, default 'linear'

    Returns
    -------
    dict
        Profile dict with ``distance_from_surface``, ``values``, etc.
    """
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    fields = dataset["fields"]

    if field_key not in fields:
        raise KeyError(f"Field '{field_key}' not found in dataset.")

    values_2d = np.asarray(fields[field_key]).squeeze()
    if values_2d.ndim != 2:
        raise ValueError(f"Field '{field_key}' must be 2-D; got {values_2d.shape}")

    if side.lower() not in {"positive", "negative"}:
        raise ValueError("side must be 'positive' or 'negative'.")

    # ------------------------------------------------------------------
    # Vertical extraction (legacy path)
    # ------------------------------------------------------------------
    if surface_angle_deg is None:
        idx = int(np.argmin(np.abs(x - x_location)))
        x_s = float(x[idx])
        eff_surf = float(surface_y) + float(surface_offset)

        if side.lower() == "positive":
            mask = y >= eff_surf
            if y_max is not None:
                mask &= y <= eff_surf + y_max
            y_prof = y[mask]
            v_prof = values_2d[idx, mask]
            dist = y_prof - eff_surf
        else:
            mask = y <= eff_surf
            if y_max is not None:
                mask &= y >= eff_surf - y_max
            y_prof = y[mask][::-1]
            v_prof = values_2d[idx, mask][::-1]
            dist = eff_surf - y_prof

        normal = np.array([0.0, 1.0 if side.lower() == "positive" else -1.0], dtype=float)
        surf_pt = np.array([x_s, eff_surf], dtype=float)
        return {
            "field_key": field_key,
            "x_requested": float(x_location),
            "x_sampled": x_s,
            "x": np.full_like(y_prof, x_s, dtype=float),
            "y": np.asarray(y_prof, dtype=float),
            "distance_from_surface": np.asarray(dist, dtype=float),
            "values": np.asarray(v_prof, dtype=float),
            "surface_y": eff_surf,
            "normal_direction": normal,
            "surface_point": surf_pt,
            "time": dataset.get("time"),
            "source": dataset.get("source"),
        }

    # ------------------------------------------------------------------
    # Angled extraction
    # ------------------------------------------------------------------
    if surface_origin is None:
        raise ValueError("surface_origin is required when surface_angle_deg is provided.")

    theta = np.deg2rad(float(surface_angle_deg))
    tangent = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    if side.lower() == "negative":
        normal = -normal

    x0, y0 = [float(v) for v in surface_origin]
    surf_pt = np.array([x0, y0], dtype=float) + float(surface_offset) * normal

    # Domain bounds
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max_d = float(y.min()), float(y.max())

    # Exit distances
    tol = 1.0e-14
    exit_d = []
    if abs(normal[0]) > tol:
        t = (x_max - surf_pt[0]) / normal[0] if normal[0] > 0 else (x_min - surf_pt[0]) / normal[0]
        if t >= 0:
            exit_d.append(t)
    if abs(normal[1]) > tol:
        t = (y_max_d - surf_pt[1]) / normal[1] if normal[1] > 0 else (y_min - surf_pt[1]) / normal[1]
        if t >= 0:
            exit_d.append(t)

    if not exit_d:
        raise ValueError("Could not determine a valid in-domain surface-normal segment.")

    max_dist = min(exit_d)
    if y_max is not None:
        max_dist = min(max_dist, float(y_max))
    if max_dist <= 0:
        raise ValueError("Requested wall-normal interval is empty.")

    # Auto n_samples
    if n_samples is None:
        dx_sp = np.diff(x)
        dy_sp = np.diff(y)
        sp = np.concatenate([dx_sp[dx_sp > 0], dy_sp[dy_sp > 0]])
        min_sp = float(np.min(sp)) if sp.size > 0 else max_dist / 200.0
        n_samples = max(2, int(np.ceil(max_dist / min_sp)) + 1)
    else:
        n_samples = max(2, int(n_samples))

    dist = np.linspace(0.0, max_dist, n_samples, dtype=float)
    pts = surf_pt[None, :] + dist[:, None] * normal[None, :]

    interp = regular_grid_interpolator(x, y, values_2d, method=interp_method, fill_value=np.nan)
    v_prof = interp(pts)

    valid = np.isfinite(v_prof)
    if not np.any(valid):
        raise ValueError("Interpolation along surface normal returned no valid points.")
    if not valid[0]:
        raise ValueError("Surface-normal origin is outside dataset bounds.")

    first_bad = np.flatnonzero(~valid)
    if first_bad.size > 0:
        end = int(first_bad[0])
        dist = dist[:end]
        pts = pts[:end]
        v_prof = v_prof[:end]

    if dist.size == 0:
        raise ValueError("No points selected for the requested wall-normal interval.")

    return {
        "field_key": field_key,
        "x_requested": float(x_location),
        "x": np.asarray(pts[:, 0], dtype=float),
        "y": np.asarray(pts[:, 1], dtype=float),
        "distance_from_surface": np.asarray(dist, dtype=float),
        "values": np.asarray(v_prof, dtype=float),
        "surface_y": float(surf_pt[1]),
        "normal_direction": np.asarray(normal, dtype=float),
        "surface_point": np.asarray(surf_pt, dtype=float),
        "surface_angle_deg": float(surface_angle_deg),
        "surface_origin": (x0, y0),
        "time": dataset.get("time"),
        "source": dataset.get("source"),
    }


def extract_streamline_field(dataset, u_key="x_velocity", v_key="y_velocity",
                             color_key=None, mask_key=None,
                             mask_threshold=None, mask_mode="below"):
    """Prepare velocity data for streamline plotting.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    u_key, v_key : str, default ('x_velocity', 'y_velocity')
    color_key : str, optional
        Field used to color streamlines.
    mask_key : str, optional
        Field used to mask cells.
    mask_threshold : float, optional
    mask_mode : {'below', 'above'}, default 'below'

    Returns
    -------
    dict
        Streamline payload with ``x``, ``y``, ``u``, ``v``, ``color``, ``mask``.
    """
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    fields = dataset["fields"]

    for key in (u_key, v_key):
        if key not in fields:
            raise KeyError(f"Velocity field '{key}' not found in dataset.")

    u = np.asarray(fields[u_key], dtype=float).squeeze()
    v = np.asarray(fields[v_key], dtype=float).squeeze()
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"Velocity fields must be 2-D; got {u.shape}, {v.shape}")

    color = None
    if color_key is not None:
        if color_key in fields:
            color = np.asarray(fields[color_key], dtype=float).squeeze()
        elif color_key in {"speed", "velocity_magnitude", "vel_mag"}:
            color = np.sqrt(u**2 + v**2)
            color_key = "velocity_magnitude"
        else:
            raise KeyError(f"Color field '{color_key}' not found.")
        if color.ndim != 2:
            raise ValueError(f"Color field must be 2-D; got {color.shape}")

    mask = None
    if mask_key is not None:
        if mask_key not in fields:
            raise KeyError(f"Mask field '{mask_key}' not found.")
        if mask_threshold is None:
            raise ValueError("mask_threshold is required when mask_key is given.")
        m = np.asarray(fields[mask_key], dtype=float).squeeze()
        if m.ndim != 2:
            raise ValueError(f"Mask field must be 2-D; got {m.shape}")
        mode = str(mask_mode).lower()
        if mode == "below":
            mask = m < float(mask_threshold)
        elif mode == "above":
            mask = m > float(mask_threshold)
        else:
            raise ValueError("mask_mode must be 'below' or 'above'.")

    return {
        "x": x, "y": y,
        "u": u, "v": v,
        "color": color,
        "color_key": color_key,
        "mask": mask,
        "mask_key": mask_key,
        "mask_threshold": mask_threshold,
        "u_key": u_key,
        "v_key": v_key,
        "time": dataset.get("time"),
        "source": dataset.get("source"),
    }


# ---------------------------------------------------------------------------
# 5.  GEOMETRY & SURFACE
# ---------------------------------------------------------------------------

def define_geometry_surface(dataset, method="auto", velocity_threshold=10.0,
                            max_x=0.75, vfrac_threshold=0.9,
                            ib_key="ib_markers", vfrac_key="volume_fraction",
                            geometry_type="auto", plate_leading_edge=0.0):
    """Detect upper and lower body surfaces using multiple strategies.

    Supports PeleC (EB / ``vfrac``), MFC (IBM markers), and a generic
    velocity-based fallback.  ``method='auto'`` tries the most accurate
    available method in order: ``vfrac`` → ``ib_markers`` → ``velocity``.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``.
    method : str, default 'auto'
        Detection strategy:

        * ``'vfrac'``      – Use volume fraction (PeleC Embedded Boundary).
        * ``'ib_markers'`` – Use immersed-boundary markers (MFC).
        * ``'velocity'``   – Use velocity threshold (generic fallback).
        * ``'auto'``       – Try ``vfrac``, then ``ib_markers``, then ``velocity``.
    velocity_threshold : float
        For ``'velocity'`` method: velocity below this indicates wall.
    max_x : float
        Stop searching past this x-coordinate.
    vfrac_threshold : float
        For ``'vfrac'`` method: cells with ``vfrac < vfrac_threshold`` are
        considered near-wall (cut cells or solid).
    ib_key : str
        Field name for IBM markers.
    vfrac_key : str
        Field name for volume fraction.
    geometry_type : str, default 'auto'
        Surface geometry hint: ``'auto'`` (detect from solution),
        ``'flat_plate'`` (analytical y=0 for x >= plate_leading_edge),
        ``'wedge'`` or ``'custom'`` (reserved for future use).
    plate_leading_edge : float, default 0.0
        For ``geometry_type='flat_plate'``: x-location where the plate starts [m].
        Upstream of this, no surface is defined.

    Returns
    -------
    dict
        ``{'upper': {...}, 'lower': {...}}`` with keys ``x``, ``y``, ``y_interp``,
        ``i``, ``j``, ``confidence``.
    """
    x = dataset["x"]
    y = dataset["y"]
    fields = dataset["fields"]

    # ------------------------------------------------------------------
    # Analytical geometry shortcut
    # ------------------------------------------------------------------
    geom = str(geometry_type).lower()
    if geom == "flat_plate":
        return _detect_flat_plate(dataset, plate_leading_edge, max_x)
    elif geom in ("wedge", "custom"):
        # Reserved for future parametric geometry support
        raise NotImplementedError(
            f"geometry_type='{geometry_type}' not yet implemented.  "
            "Use 'auto' or 'flat_plate'."
        )

    # ------------------------------------------------------------------
    # Auto-detect method
    # ------------------------------------------------------------------
    if method == "auto":
        if vfrac_key in fields:
            method = "vfrac"
        elif ib_key in fields:
            method = "ib_markers"
        else:
            method = "velocity"

    # ------------------------------------------------------------------
    # vfrac method (PeleC Embedded Boundary)
    # ------------------------------------------------------------------
    if method == "vfrac":
        if vfrac_key not in fields:
            raise KeyError(
                f"Volume-fraction field '{vfrac_key}' not found in dataset.  "
                f"Available fields: {sorted(fields.keys())[:20]}"
            )

        vfrac = fields[vfrac_key]
        surfaces = {"upper": [], "lower": []}

        for i, xv in enumerate(x):
            if xv > max_x:
                break
            vfrac_col = vfrac[i, :]
            cut_cells = np.where(vfrac_col < vfrac_threshold)[0]
            if len(cut_cells) == 0:
                continue

            j_min = cut_cells.min()
            j_max = cut_cells.max()
            j_center = (j_min + j_max) // 2

            # Upper surface: highest cut cell
            upper = cut_cells[cut_cells >= j_center]
            if len(upper) > 0 and upper[-1] + 1 < len(y):
                j_cut = upper[-1]
                j_fluid = j_cut + 1
                y_exact = _interp_surface(
                    y[j_cut], y[j_fluid],
                    vfrac_col[j_cut], vfrac_col[j_fluid],
                    threshold=vfrac_threshold,
                )
                surfaces["upper"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid}
                )

            # Lower surface: lowest cut cell
            lower = cut_cells[cut_cells <= j_center]
            if len(lower) > 0 and lower[0] > 0:
                j_cut = lower[0]
                j_fluid = j_cut - 1
                y_exact = _interp_surface(
                    y[j_cut], y[j_fluid],
                    vfrac_col[j_cut], vfrac_col[j_fluid],
                    threshold=vfrac_threshold,
                )
                surfaces["lower"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid}
                )

    # ------------------------------------------------------------------
    # ib_markers method (MFC Immersed Boundary)
    # ------------------------------------------------------------------
    elif method == "ib_markers":
        if ib_key not in fields:
            raise KeyError(
                f"IBM field '{ib_key}' not found in dataset.  "
                f"Available fields: {sorted(fields.keys())[:20]}"
            )

        vel_mag = fields.get("velocity_magnitude")
        if vel_mag is None:
            raise KeyError(
                "velocity_magnitude required for geometry detection."
            )

        ib = fields[ib_key]
        surfaces = {"upper": [], "lower": []}

        for i, xv in enumerate(x):
            if xv > max_x:
                break
            vel_col = vel_mag[i, :]
            ib_col = ib[i, :]
            candidates = (vel_col < velocity_threshold) & (ib_col > 0.1)
            if not np.any(candidates):
                continue

            geom_idx = np.where(candidates)[0]
            j_min = geom_idx.min()
            j_max = geom_idx.max()
            j_center = (j_min + j_max) // 2

            # Upper surface
            upper = geom_idx[geom_idx >= j_center]
            if len(upper) > 0 and upper[-1] + 1 < len(y):
                j_geom = upper[-1]
                j_fluid = j_geom + 1
                y_exact = _interp_surface(
                    y[j_geom], y[j_fluid],
                    ib_col[j_geom], ib_col[j_fluid],
                )
                surfaces["upper"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid}
                )

            # Lower surface
            lower = geom_idx[geom_idx <= j_center]
            if len(lower) > 0 and lower[0] > 0:
                j_geom = lower[0]
                j_fluid = j_geom - 1
                y_exact = _interp_surface(
                    y[j_geom], y[j_fluid],
                    ib_col[j_geom], ib_col[j_fluid],
                )
                surfaces["lower"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid}
                )

    # ------------------------------------------------------------------
    # velocity method (generic fallback with multi-indicator enhancement)
    # ------------------------------------------------------------------
    elif method == "velocity":
        vel_mag = fields.get("velocity_magnitude")
        if vel_mag is None:
            raise KeyError(
                "velocity_magnitude required for velocity-based geometry detection."
            )

        # ---- Multi-indicator setup ----
        has_pressure = "pressure" in fields
        has_temperature = "temperature" in fields

        if has_pressure:
            # Normalised vertical pressure gradient
            dp_dy = np.gradient(fields["pressure"], y, axis=1)
            dp_dy_abs = np.abs(dp_dy)
            dp_dy_norm = dp_dy_abs / (dp_dy_abs.max() + 1e-12)

        surfaces = {"upper": [], "lower": []}

        for i, xv in enumerate(x):
            if xv > max_x:
                break
            vel_col = vel_mag[i, :]

            # Primary: velocity below threshold
            vel_cand = vel_col < velocity_threshold

            # Secondary: pressure-gradient cue (high |dp/dy| near walls)
            pg_cand = np.ones_like(vel_cand, dtype=bool)
            if has_pressure:
                pg_col = dp_dy_norm[i, :]
                # Adaptive threshold: top 15 % within this column
                pg_thresh = np.percentile(pg_col[pg_col > 0], 85) if np.any(pg_col > 0) else 0.5
                pg_cand = pg_col > pg_thresh

            # Combined: velocity AND (pressure-gradient OR temperature)
            if has_temperature:
                T_col = fields["temperature"][i, :]
                T_wall = float(T_col[0])  # wall-adjacent cell
                # Look for sharp rise away from wall
                dT_dy = np.abs(np.gradient(T_col, y))
                dT_thresh = np.percentile(dT_dy, 80) if np.any(dT_dy > 0) else 0.0
                temp_cand = dT_dy > dT_thresh
                candidates = vel_cand & (pg_cand | temp_cand)
            else:
                candidates = vel_cand & pg_cand

            # Fallback to pure velocity if combined yields nothing
            if not np.any(candidates):
                candidates = vel_cand

            if not np.any(candidates):
                continue

            geom_idx = np.where(candidates)[0]
            j_min = geom_idx.min()
            j_max = geom_idx.max()
            j_center = (j_min + j_max) // 2

            # Confidence: how far below threshold is the velocity
            vel_near_wall = vel_col[geom_idx]
            conf = np.clip(1.0 - vel_near_wall / (velocity_threshold + 1e-12), 0.0, 1.0)

            # Upper surface
            upper = geom_idx[geom_idx >= j_center]
            if len(upper) > 0 and upper[-1] + 1 < len(y):
                j_geom = upper[-1]
                j_fluid = j_geom + 1
                y_exact = 0.5 * (y[j_geom] + y[j_fluid])
                # Refine with inverse-distance weighting if available
                if has_temperature:
                    T_geom = fields["temperature"][i, j_geom]
                    T_fluid = fields["temperature"][i, j_fluid]
                    if abs(T_fluid - T_geom) > 1.0:  # meaningful gradient
                        frac = np.clip((T_wall - T_geom) / (T_fluid - T_geom), 0, 1)
                        y_exact = y[j_geom] + frac * (y[j_fluid] - y[j_geom])
                surfaces["upper"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid, "confidence": float(conf[upper[-1]])}
                )

            # Lower surface
            lower = geom_idx[geom_idx <= j_center]
            if len(lower) > 0 and lower[0] > 0:
                j_geom = lower[0]
                j_fluid = j_geom - 1
                y_exact = 0.5 * (y[j_geom] + y[j_fluid])
                if has_temperature:
                    T_geom = fields["temperature"][i, j_geom]
                    T_fluid = fields["temperature"][i, j_fluid]
                    if abs(T_fluid - T_geom) > 1.0:
                        frac = np.clip((T_wall - T_geom) / (T_fluid - T_geom), 0, 1)
                        y_exact = y[j_geom] + frac * (y[j_fluid] - y[j_geom])
                surfaces["lower"].append(
                    {"x": xv, "y": y[j_fluid], "y_interp": y_exact,
                     "i": i, "j": j_fluid, "confidence": float(conf[lower[0]])}
                )

    else:
        raise ValueError(
            f"Unknown geometry detection method '{method}'.  "
            f"Choose from: 'auto', 'vfrac', 'ib_markers', 'velocity'."
        )

    # ------------------------------------------------------------------
    # Pack into structured arrays (with confidence & outlier rejection)
    # ------------------------------------------------------------------
    for side in ("upper", "lower"):
        if len(surfaces[side]) > 0:
            # Build arrays
            arr = {
                "x": np.array([p["x"] for p in surfaces[side]], dtype=float),
                "y": np.array([p["y"] for p in surfaces[side]], dtype=float),
                "y_interp": np.array([p["y_interp"] for p in surfaces[side]], dtype=float),
                "i": np.array([p["i"] for p in surfaces[side]], dtype=int),
                "j": np.array([p["j"] for p in surfaces[side]], dtype=int),
                "confidence": np.array([p.get("confidence", 1.0) for p in surfaces[side]], dtype=float),
            }

            # Outlier rejection: reject points where |dy/dx| > 45° for plate-like surfaces
            if len(arr["x"]) > 3:
                dx_arr = np.diff(arr["x"])
                dy_arr = np.diff(arr["y_interp"])
                slope = np.abs(np.divide(dy_arr, dx_arr, out=np.zeros_like(dy_arr), where=dx_arr > 0))
                bad = np.concatenate([[False], slope > 1.0])  # > 45 deg
                if np.any(bad):
                    keep = ~bad
                    for k in arr:
                        arr[k] = arr[k][keep]

            surfaces[side] = arr
        else:
            surfaces[side] = {
                "x": np.array([], dtype=float),
                "y": np.array([], dtype=float),
                "y_interp": np.array([], dtype=float),
                "i": np.array([], dtype=int),
                "j": np.array([], dtype=int),
                "confidence": np.array([], dtype=float),
            }

    return surfaces


# ---------------------------------------------------------------------------
def _detect_flat_plate(dataset, plate_leading_edge=0.0, max_x=0.75):
    """Detect the surface of a flat plate at y = 0 using analytical geometry.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``.
    plate_leading_edge : float, default 0.0
        x-location where the plate starts [m].
    max_x : float
        Stop searching past this x-coordinate [m].

    Returns
    -------
    dict
        Surfaces dict with keys ``upper`` and ``lower``.
    """
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)

    # Only define the plate for x >= plate_leading_edge
    plate_mask = x >= plate_leading_edge
    plate_indices = np.where(plate_mask)[0]

    # Enforce max_x
    if max_x is not None:
        plate_indices = plate_indices[x[plate_indices] <= max_x]

    n_pts = len(plate_indices)
    if n_pts == 0:
        return {
            "upper": {"x": np.array([], dtype=float), "y": np.array([], dtype=float),
                      "y_interp": np.array([], dtype=float), "i": np.array([], dtype=int),
                      "j": np.array([], dtype=int), "confidence": np.array([], dtype=float)},
            "lower": {"x": np.array([], dtype=float), "y": np.array([], dtype=float),
                      "y_interp": np.array([], dtype=float), "i": np.array([], dtype=int),
                      "j": np.array([], dtype=int), "confidence": np.array([], dtype=float)},
        }

    # The wall is at y=0. The first fluid cell centre is at j=0 (y = dy/2).
    # We use j=0 as the fluid-adjacent index and interpolate the surface to y=0.
    j_fluid = np.zeros(n_pts, dtype=int)  # closest fluid cell
    y_exact = np.zeros(n_pts, dtype=float)  # wall at y = 0

    surfaces = {
        "upper": {
            "x": x[plate_indices],
            "y": y[j_fluid],
            "y_interp": y_exact,
            "i": np.asarray(plate_indices, dtype=int),
            "j": j_fluid,
            "confidence": np.ones(n_pts, dtype=float),
        },
        "lower": {
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "y_interp": np.array([], dtype=float),
            "i": np.array([], dtype=int),
            "j": np.array([], dtype=int),
            "confidence": np.array([], dtype=float),
        },
    }
    return surfaces


def _interp_surface(y1, y2, val1, val2, threshold=0.5):
    """Linearly interpolate surface location between two cells.

    Parameters
    ----------
    y1, y2 : float
        Y-coordinates of the two cells.
    val1, val2 : float
        Indicator values (e.g. vfrac or ib_markers) at the two cells.
    threshold : float, default 0.5
        Target value where the surface is located.
    """
    if abs(val2 - val1) > 1e-6:
        frac = (threshold - val1) / (val2 - val1)
        frac = np.clip(frac, 0, 1)
        return y1 + frac * (y2 - y1)
    return 0.5 * (y1 + y2)


def compute_surface_normals(surfaces, smoothing_window=51):
    """Compute smoothed surface normals via Savitzky-Golay filter.

    Returns the *surfaces* dict (modified in-place) with added keys:
    ``nx``, ``ny``, ``theta``, ``tangent_x``, ``tangent_y``, ``y_smooth``.
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        raise ImportError("scipy is required for surface normal smoothing.")

    for side in ("upper", "lower"):
        data = surfaces[side]
        if len(data.get("x", [])) == 0:
            continue

        x = data["x"]
        y = data["y_interp"]
        n = len(x)

        if n > smoothing_window:
            y_s = savgol_filter(y, smoothing_window, 3)
        else:
            y_s = y

        dx = np.gradient(x)
        dy = np.gradient(y_s)
        ds = np.sqrt(dx**2 + dy**2)
        tx = dx / (ds + 1e-10)
        ty = dy / (ds + 1e-10)

        if side == "upper":
            nx = -ty
            ny = tx
        else:
            nx = ty
            ny = -tx

        # Ensure normals point away from body
        if side == "upper":
            s = np.sign(ny)
            s[s == 0] = 1
            nx *= s
            ny *= s
        else:
            s = -np.sign(ny)
            s[s == 0] = -1
            nx *= s
            ny *= s

        data["y_smooth"] = y_s
        data["nx"] = nx
        data["ny"] = ny
        data["theta"] = np.arctan2(dy, dx)
        data["tangent_x"] = tx
        data["tangent_y"] = ty

    return surfaces


# ---------------------------------------------------------------------------
# 6.  BOUNDARY LAYER & SURFACE PROPERTIES
# ---------------------------------------------------------------------------

def sutherland_viscosity(T, T_ref=273.15, mu_ref=1.716e-5, S=110.4):
    """Dynamic viscosity of air via Sutherland's law.

    Parameters
    ----------
    T : float or ndarray
        Temperature (K).
    T_ref : float, default 273.15
        Reference temperature (K).
    mu_ref : float, default 1.716e-5
        Viscosity at T_ref (Pa s).
    S : float, default 110.4
        Sutherland constant (K).

    Returns
    -------
    float or ndarray
        Viscosity (Pa s).
    """
    T = np.asarray(T, dtype=float)
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)


def calculate_BL_thicknesses(y_wall_normal, U_Uinf):
    """Calculate boundary layer thicknesses and shape factor.

    Returns
    -------
    delta_99, delta_star, theta, H
    """
    y = np.asarray(y_wall_normal, dtype=float)
    U = np.asarray(U_Uinf, dtype=float)

    idx = np.where(U >= 0.99)[0]
    delta_99 = y[idx[0]] if len(idx) > 0 else y[-1]

    delta_star = np.trapz(1.0 - U, y)
    theta = np.trapz(U * (1.0 - U), y)
    H = delta_star / theta if theta > 0 else np.nan

    return delta_99, delta_star, theta, H


def extract_BL_profile_at_surface(dataset, i_surf, j_surf, nx, ny,
                                  u_inf, rho_inf,
                                  max_BL_height=0.010, n_points_BL=200,
                                  mu=None, k_w=None, T_wall=None,
                                  Pr=0.71, Cp=1004.0):
    """Extract boundary layer profile along the true surface normal.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    i_surf, j_surf : int
        Surface cell indices.
    nx, ny : float
        Unit normal components (pointing away from surface).
    u_inf, rho_inf : float
        Freestream velocity (m/s) and density (kg/m^3).
    max_BL_height : float, default 0.010
        Maximum distance from wall (m).
    n_points_BL : int, default 200
    mu : float, optional
        Dynamic viscosity [Pa.s]. If None, use Sutherland's law.
    k_w : float, optional
        Thermal conductivity [W/(m.K)]. If None, compute as mu*Cp/Pr.
    T_wall : float, optional
        Wall temperature [K]. If None, use cell-adjacent temperature.
    Pr : float, default 0.71
        Prandtl number (used only when k_w is None).
    Cp : float, default 1004.0
        Specific heat at constant pressure [J/(kg.K)].

    Returns
    -------
    dict
        BL results: ``y_profile``, ``U_Uinf``, ``u_profile``, ``u_edge``,
        ``tau_w``, ``C_f``, ``q_w``, ``delta_99``, ``delta_star``, ``theta``, ``H``,
        ``Re_x``, ``Re_theta``.
    """
    x = dataset["x"]
    y = dataset["y"]
    fields = dataset["fields"]

    x_s = float(x[i_surf])
    y_s = float(y[j_surf])

    # Stretched distribution: finer near wall
    eta = np.linspace(0, 1, n_points_BL)
    stretch = 1.5
    eta_s = np.tanh(stretch * eta) / np.tanh(stretch)
    s = eta_s * max_BL_height

    x_prof = x_s + s * nx
    y_prof_abs = y_s + s * ny

    # Interpolate
    interp_u = regular_grid_interpolator(x, y, fields["x_velocity"],
                                         fill_value=None)
    interp_v = regular_grid_interpolator(x, y, fields["y_velocity"],
                                         fill_value=None)
    interp_T = regular_grid_interpolator(x, y, fields["temperature"],
                                         fill_value=None)
    interp_rho = regular_grid_interpolator(x, y, fields["density"],
                                           fill_value=None)

    pts = np.column_stack([x_prof, y_prof_abs])
    u_x = interp_u(pts)
    u_y = interp_v(pts)
    T_prof = interp_T(pts)
    rho_prof = interp_rho(pts)

    # Tangential velocity
    tx_c = ny
    ty_c = -nx
    if tx_c < 0:
        tx_c = -ny
        ty_c = nx
    u_tan = u_x * tx_c + u_y * ty_c

    # BL edge detection via gradient method
    du = np.gradient(u_tan, s)
    n_s = min(15, len(du) // 4)
    if n_s >= 5 and n_s % 2 == 0:
        n_s -= 1
    if n_s >= 5:
        try:
            from scipy.signal import savgol_filter
            du_s = savgol_filter(du, n_s, 3)
        except ImportError:
            du_s = du
    else:
        du_s = du

    i_start = max(2, n_points_BL // 50)
    du_max = np.max(np.abs(du_s[i_start:]))

    if du_max < 1e-6:
        u_edge = u_tan[-1] if u_tan[-1] > 10.0 else u_inf
        i_edge = len(s) - 1
    else:
        thresh = 0.01 * du_max
        i_max = i_start + np.argmax(np.abs(du_s[i_start:]))
        candidates = np.where(np.abs(du_s[i_max:]) < thresh)[0]
        if len(candidates) > 0:
            i_edge = i_max + candidates[0]
        else:
            c2 = np.where(np.abs(du_s[i_max:]) < 0.05 * du_max)[0]
            i_edge = i_max + c2[0] if len(c2) > 0 else len(s) - 1
        u_edge = u_tan[i_edge]

    if u_edge < 10.0:
        u_edge = np.max(u_tan)
    if u_edge < 10.0:
        u_edge = u_inf

    U_Uinf = np.clip(u_tan / u_edge, 0.0, 1.5)

    # Traditional 0.99 check
    search = min(i_edge + n_points_BL // 10, len(U_Uinf))
    idx99 = np.where(U_Uinf[:search] >= 0.99)[0]
    delta_99 = s[idx99[0]] if len(idx99) > 0 else s[i_edge]
    if delta_99 <= s[2]:
        delta_99 = s[i_edge]

    # Integration within BL
    i_cut = min(i_edge + 5, len(s))
    i_cut = max(i_cut, 10)
    y_bl = s[:i_cut]
    U_bl = np.minimum(U_Uinf[:i_cut], 1.0)
    delta_star = np.trapz(1.0 - U_bl, y_bl)
    theta = np.trapz(U_bl * (1.0 - U_bl), y_bl)
    H = delta_star / theta if theta > 1e-12 else np.nan

    # Wall shear & heat flux
    # Use provided mu or compute from Sutherland's law
    mu_w = sutherland_viscosity(T_prof[0]) if mu is None else float(mu)
    T_wall_use = T_wall if T_wall is not None else T_prof[0]
    # Use provided k_w or compute from mu*Cp/Pr
    k_w_val = mu_w * Cp / Pr if k_w is None else float(k_w)

    # Find the first non-zero tangential velocity in the profile to
    # skip cells that PeleC's EB may have zeroed (cut cells inside the
    # immersed boundary / embedded boundary).
    u_nonzero = np.where(np.abs(u_tan) > 1e-6)[0]
    if len(u_nonzero) > 0:
        i_start_fit = u_nonzero[0]
    else:
        i_start_fit = 0
    # Clamp so we have at least 2 profile points + the wall for a
    # meaningful linear fit.
    n_fit = max(2, min(6, len(s) - i_start_fit))
    i_end_fit = i_start_fit + n_fit

    # Assemble fit arrays.  The wall lies at s = -y_s along the
    # wall-normal direction (the profile s=0 is at the cell centre).
    if y_s > 1e-12:
        s_wall = -y_s
        s_fit = np.concatenate([[s_wall], s[i_start_fit:i_end_fit]])
        u_fit = np.concatenate([[0.0], u_tan[i_start_fit:i_end_fit]])
        T_fit = np.concatenate([[T_wall_use], T_prof[i_start_fit:i_end_fit]])
    else:
        s_fit = s[:n_fit]
        u_fit = u_tan[:n_fit]
        T_fit = T_prof[:n_fit]

    coeffs_u = np.polyfit(s_fit, u_fit, 1)
    du_ds = coeffs_u[0]
    tau_w = mu_w * du_ds

    # Wall heat flux: q_w = -k_w * dT/dn
    # Sign: q_w > 0 means heat flows AWAY from wall (into the fluid, wall cooling);
    #       q_w < 0 means heat flows TOWARD the wall (into the wall, wall heating).
    # n points from the wall into the fluid (outward normal).
    coeffs_T = np.polyfit(s_fit, T_fit, 1)
    dT_dn = coeffs_T[0]
    q_w = -k_w_val * dT_dn

    q_inf = 0.5 * rho_inf * u_inf**2
    C_f = tau_w / q_inf

    # Reynolds numbers
    x_stat = x_s
    T_far = T_prof[-1]
    mu_far = sutherland_viscosity(T_far) if mu is None else float(mu)
    Re_x = rho_inf * u_inf * x_stat / mu_far
    Re_theta = rho_inf * u_inf * theta / mu_far if theta > 0 else 0.0

    return {
        "y_profile": s,
        "U_Uinf": U_Uinf,
        "u_profile": u_tan,
        "u_edge": u_edge,
        "T_profile": T_prof,
        "tau_w": tau_w,
        "C_f": C_f,
        "q_w": q_w,
        "delta_99": delta_99,
        "delta_star": delta_star,
        "theta": theta,
        "H": H,
        "Re_x": Re_x,
        "Re_theta": Re_theta,
    }


def extract_surface_properties(dataset, surfaces, rho_inf=0.0267, u_inf=1011.0,
                                 T_inf=None, mu=None, k_w=None, T_wall=None,
                                 Pr=0.71, Cp=1004.0):
    """Extract surface properties (Cp, Cf, delta_99, etc.) at all surface points.

    Parameters
    ----------
    dataset : dict
        ``StandardDataset``
    surfaces : dict
        Output from ``define_geometry_surface`` + ``compute_surface_normals``.
    rho_inf, u_inf : float
        Freestream density and velocity.
    T_inf : float, optional
        Freestream temperature (K).
    mu : float, optional
        Constant dynamic viscosity [Pa.s]. If None, use Sutherland's law.
    k_w : float, optional
        Constant thermal conductivity [W/(m.K)]. If None, compute from mu*Cp/Pr.
    T_wall : float, optional
        Constant wall temperature [K]. If None, use cell-adjacent temperature.
    Pr : float, default 0.71
        Prandtl number (used only when k_w is None).
    Cp : float, default 1004.0
        Specific heat at constant pressure [J/(kg.K)].

    Returns
    -------
    dict
        ``{'upper': {...}, 'lower': {...}}`` with arrays for each quantity.
    """
    x = dataset["x"]
    y = dataset["y"]
    fields = dataset["fields"]

    surface_data = {}

    for side in ("upper", "lower"):
        if len(surfaces[side].get("x", [])) == 0:
            continue

        n = len(surfaces[side]["x"])
        data = {
            "x": surfaces[side]["x"],
            "y": surfaces[side]["y"],
            "s": np.zeros(n),
            "p": np.zeros(n),
            "temperature": np.zeros(n),
            "rho": np.zeros(n),
            "tau_w": np.zeros(n),
            "C_f": np.zeros(n),
            "C_p": np.zeros(n),
            "q_w": np.zeros(n),
            "Re_x": np.zeros(n),
            "Re_theta": np.zeros(n),
            "delta_99": np.zeros(n),
            "delta_star": np.zeros(n),
            "theta": np.zeros(n),
            "H": np.zeros(n),
        }

        # Copy surface normals and tangents from surfaces dict (computed by
        # compute_surface_normals) so that compute_sectional_forces can use them.
        for key in ("nx", "ny", "tangent_x", "tangent_y"):
            data[key] = surfaces[side].get(key, np.zeros(n))

        # Arc length
        dx = np.diff(data["x"])
        dy = np.diff(data["y"])
        ds = np.sqrt(dx**2 + dy**2)
        data["s"][1:] = np.cumsum(ds)

        has_normals = "nx" in surfaces[side] and "ny" in surfaces[side]

        n_ok = 0
        n_fail = 0

        for idx in range(n):
            i = int(surfaces[side]["i"][idx])
            j = int(surfaces[side]["j"][idx])
            data["p"][idx] = fields["pressure"][i, j]
            data["temperature"][idx] = fields["temperature"][i, j]
            data["rho"][idx] = fields["density"][i, j]

            if has_normals:
                nx = surfaces[side]["nx"][idx]
                ny = surfaces[side]["ny"][idx]
            else:
                nx, ny = (0.0, 1.0) if side == "upper" else (0.0, -1.0)

            try:
                bl = extract_BL_profile_at_surface(
                    dataset, i, j, nx, ny, u_inf, rho_inf,
                    mu=mu, k_w=k_w, T_wall=T_wall, Pr=Pr, Cp=Cp
                )
                data["tau_w"][idx] = bl["tau_w"]
                data["C_f"][idx] = bl["C_f"]
                data["q_w"][idx] = bl["q_w"]
                data["delta_99"][idx] = bl["delta_99"]
                data["delta_star"][idx] = bl["delta_star"]
                data["theta"][idx] = bl["theta"]
                data["H"][idx] = bl["H"]
                data["Re_x"][idx] = bl["Re_x"]
                data["Re_theta"][idx] = bl["Re_theta"]
                n_ok += 1
            except Exception as exc:
                if n_fail < 5:
                    _log_error(f"BL extraction at point {idx} (x={x[i]:.4f})", exc)
                n_fail += 1

        # Smoothing
        try:
            from scipy.signal import savgol_filter
            from scipy.ndimage import median_filter

            arc = data["s"][-1] if data["s"][-1] > 0 else 1.0
            pts_mm = n / (arc * 1000.0)
            med_win = max(3, int(pts_mm * 5))
            sg_win = max(med_win * 3 + 1, 21)
            if med_win % 2 == 0:
                med_win += 1
            if sg_win % 2 == 0:
                sg_win += 1
            med_win = min(med_win, n if n % 2 == 1 else n - 1)
            sg_win = min(sg_win, n if n % 2 == 1 else n - 1)

            def _smooth(arr):
                a = arr.copy().astype(float)
                valid = np.isfinite(a)
                if np.sum(valid) < sg_win:
                    return a
                xi = np.arange(len(a))
                a[~valid] = np.interp(xi[~valid], xi[valid], a[valid])
                a = median_filter(a, size=med_win)
                a = savgol_filter(a, sg_win, polyorder=3)
                a[~valid] = np.nan
                return a

            for key in ("C_f", "tau_w", "q_w", "delta_99", "delta_star", "theta", "H"):
                data[key] = _smooth(data[key])
        except Exception:
            pass  # smoothing is optional

        # Pressure coefficient
        q_inf = 0.5 * rho_inf * u_inf**2
        if T_inf is None:
            T_inf = np.mean(data["temperature"])
        p_inf = rho_inf * 287.05 * T_inf
        data["C_p"] = (data["p"] - p_inf) / q_inf

        surface_data[side] = data

    return surface_data


# ---------------------------------------------------------------------------
# 7.  INTERPOLATION UTILITIES
# ---------------------------------------------------------------------------

def regular_grid_interpolator(x_line, y_line, values_2d,
                              method="linear", fill_value=np.nan):
    """Return an interpolator for a regular 2-D grid.

    If ``scipy`` is available, returns a ``scipy.interpolate.RegularGridInterpolator``.
    Otherwise, falls back to a pure-numpy implementation.

    Parameters
    ----------
    x_line, y_line : 1-D array-like
        Strictly increasing coordinates.
    values_2d : 2-D array
        Data aligned with ``(x_line, y_line)``.
    method : {'linear', 'nearest'}, default 'linear'
    fill_value : float, default np.nan

    Returns
    -------
    callable
        ``interpolator(points)`` where *points* has shape ``(n, 2)``.
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
        return RegularGridInterpolator(
            (x_line, y_line),
            values_2d,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Pure-numpy fallback
    # ------------------------------------------------------------------
    xc = np.asarray(x_line, dtype=float)
    yc = np.asarray(y_line, dtype=float)
    z = np.asarray(values_2d, dtype=float)

    if xc.ndim != 1 or yc.ndim != 1:
        raise ValueError("Grid coordinates must be 1-D.")
    if z.shape != (xc.size, yc.size):
        raise ValueError(f"Values shape {z.shape} does not match grid ({xc.size}, {yc.size}).")
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'.")

    def _interpolate(pts):
        pts = np.asarray(pts, dtype=float)
        scalar = pts.ndim == 1
        if scalar:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("Sample points must have shape (n, 2).")

        out = np.full(pts.shape[0], fill_value, dtype=float)
        px, py = pts[:, 0], pts[:, 1]

        valid = (
            (px >= xc[0]) & (px <= xc[-1])
            & (py >= yc[0]) & (py <= yc[-1])
        )
        if not np.any(valid):
            return out[0] if scalar else out

        vx, vy = px[valid], py[valid]

        if method == "nearest":
            ix = np.searchsorted(xc, vx, side="left")
            ix = np.clip(ix, 1, xc.size - 1)
            ix_l = ix - 1
            choose_x = (vx - xc[ix_l]) >= (xc[ix] - vx)
            ix = np.where(choose_x, ix, ix_l)

            iy = np.searchsorted(yc, vy, side="left")
            iy = np.clip(iy, 1, yc.size - 1)
            iy_l = iy - 1
            choose_y = (vy - yc[iy_l]) >= (yc[iy] - vy)
            iy = np.where(choose_y, iy, iy_l)

            out[valid] = z[ix, iy]
            return out[0] if scalar else out

        # Linear
        ix = np.searchsorted(xc, vx, side="right")
        ix = np.clip(ix, 1, xc.size - 1)
        ix_l = ix - 1

        iy = np.searchsorted(yc, vy, side="right")
        iy = np.clip(iy, 1, yc.size - 1)
        iy_l = iy - 1

        tx = np.divide(vx - xc[ix_l], xc[ix] - xc[ix_l],
                        out=np.zeros_like(vx), where=(xc[ix] != xc[ix_l]))
        ty = np.divide(vy - yc[iy_l], yc[iy] - yc[iy_l],
                        out=np.zeros_like(vy), where=(yc[iy] != yc[iy_l]))

        out[valid] = (
            (1.0 - tx) * (1.0 - ty) * z[ix_l, iy_l]
            + tx * (1.0 - ty) * z[ix, iy_l]
            + (1.0 - tx) * ty * z[ix_l, iy]
            + tx * ty * z[ix, iy]
        )
        return out[0] if scalar else out

    return _interpolate


# ---------------------------------------------------------------------------
# 9.  BLASIUS REFERENCE PROFILES
# ---------------------------------------------------------------------------


def _blasius_ode_solution(eta_max=15.0, n=800):
    """Solve the Blasius ODE f''' + 0.5*f*f'' = 0.

    Returns
    -------
    eta : 1-D ndarray
    fprime : 1-D ndarray
        u/u_inf = f'(eta).
    """
    from scipy.integrate import solve_ivp

    eta = np.linspace(0, eta_max, n)

    def ode(t, Y):
        f, fp, fpp = Y
        return [fp, fpp, -0.5 * f * fpp]

    # f''(0) = 0.332057 for the form f''' + 0.5*f*f'' = 0.
    # (The classical Blasius value 0.469600 corresponds to f''' + f*f'' = 0
    #  with a different similarity-variable scaling.)
    sol = solve_ivp(ode, [0, eta_max], [0.0, 0.0, 0.332057],
                    t_eval=eta, method='RK45', rtol=1e-8, atol=1e-10)

    return sol.t, sol.y[1]  # eta, fprime


def compute_blasius_reference_profile(y, x_loc, u_inf, T_inf, rho_inf,
                                      T_wall=None, gamma=1.4, R=287.05,
                                      Pr=0.71, Cp=1004.0,
                                      recovery_factor=None,
                                      const_transport=False, mu=None, k=None,
                                      eta_max=15.0, n=800):
    """Compute incompressible Blasius + Crocco-Busemann reference profiles.

    Parameters
    ----------
    y : 1-D ndarray
        Wall-normal distances [m] (wall assumed at y=0).
    x_loc : float
        Streamwise station [m] (leading edge assumed at x=0).
    u_inf, T_inf, rho_inf : float
        Freestream velocity [m/s], temperature [K], density [kg/m^3].
    T_wall : float, optional
        Wall temperature [K]. If None, adiabatic wall assumed.
    gamma, R, Pr, Cp : float
        Gas properties (defaults: air).
    recovery_factor : float, optional
        Recovery factor for Crocco-Busemann relation.
        Default = sqrt(Pr_eff) (laminar), where Pr_eff is the effective
        Prandtl number (see ``const_transport``).
    const_transport : bool, default False
        If True, use constant transport properties:
        * ``mu`` is used for the freestream viscosity (falls back to
          Sutherland if ``mu`` is None).
        * If both ``mu`` and ``k`` are given, the effective Prandtl number
          is recomputed as ``Pr_eff = mu * Cp / k`` (overriding ``Pr``)
          so the recovery factor stays consistent with the constant
          transport assumption.
        If False, Sutherland viscosity at ``T_inf`` and the supplied
        ``Pr`` are used.
    mu : float, optional
        Constant dynamic viscosity [Pa.s]. Used only when
        ``const_transport`` is True.
    k : float, optional
        Constant thermal conductivity [W/(m.K)]. Used only when
        ``const_transport`` is True (to recompute ``Pr_eff``).
    eta_max, n : float, int
        ODE integration domain and resolution.

    Returns
    -------
    dict
        Two profile dicts keyed by field name:
        ``{"x_velocity": {...}, "temperature": {...}}``.
        Each is compatible with ``plot_line_profiles``.
    """
    y = np.asarray(y, dtype=float)

    # --- Freestream viscosity ---
    if const_transport and mu is not None:
        mu_inf = float(mu)
    else:
        mu_inf = sutherland_viscosity(T_inf)
    nu_inf = mu_inf / rho_inf

    # --- Effective Prandtl number ---
    if const_transport and mu is not None and k is not None:
        Pr_eff = float(mu) * Cp / float(k)
    else:
        Pr_eff = Pr

    # Similarity variable
    eta = y * np.sqrt(u_inf / (nu_inf * max(x_loc, 1e-12)))

    # ODE solve on a domain large enough for the largest eta value
    eta_needed = float(np.max(eta)) if len(eta) > 0 else 0.0
    eta_solve = max(eta_max, float(np.ceil(eta_needed * 1.2)))
    eta_tab, fprime_tab = _blasius_ode_solution(eta_solve, n)
    u_uinf = np.interp(eta, eta_tab, fprime_tab)

    # Velocity profile dict
    u_profile = {
        "y": y,
        "values": u_uinf * u_inf,
        "label": "Blasius",
        "linestyle": "--",
        "field_key": "x_velocity",
    }

    # Temperature profile (Crocco-Busemann)
    M_inf = u_inf / np.sqrt(gamma * R * T_inf) if T_inf > 0 else 0.0
    r = float(recovery_factor) if recovery_factor is not None else np.sqrt(Pr_eff)
    T_aw = T_inf + r * u_inf**2 / (2.0 * Cp)
    T_w = T_aw if T_wall is None else float(T_wall)

    T = T_w + (T_aw - T_w) * u_uinf - r * (u_uinf * u_inf)**2 / (2.0 * Cp)

    T_profile = {
        "y": y,
        "values": T,
        "label": "Blasius",
        "linestyle": "--",
        "field_key": "temperature",
    }

    return {"x_velocity": u_profile, "temperature": T_profile}


def compute_compressible_blasius_reference_profile(y, u_profile, rho_profile,
                                                   x_loc, u_inf,
                                                   T_profile=None, T_inf=None,
                                                   const_transport=False,
                                                   mu=None,
                                                   eta_max=15.0, n=800):
    """Compute a compressible Blasius-style overlay using a semi-local transform.

    This is a practical van Driest / Illingworth-Stewartson-style extension for
    comparing a compressible laminar boundary layer against the incompressible
    Blasius similarity solution.

    The transformed coordinate is built from local density and viscosity ratios:

        eta_t = sqrt(u_inf / (nu_e * x_loc)) * integral_0^y sqrt((rho*mu)/(rho_e*mu_e)) dy

    where ``rho_e`` and ``mu_e`` are taken from the outermost valid point in the
    supplied profile.

    Parameters
    ----------
    y : 1-D ndarray
        Wall-normal coordinates [m].
    u_profile, rho_profile : 1-D ndarray
        Streamwise velocity [m/s] and density [kg/m^3] sampled at ``y``.
    x_loc : float
        Streamwise station [m].
    u_inf : float
        Freestream velocity [m/s].
    T_profile : 1-D ndarray, optional
        Temperature profile [K]. Used to evaluate Sutherland viscosity when
        ``mu`` is not supplied.
    T_inf : float, optional
        Freestream temperature [K]. Used only as a fallback for viscosity.
    const_transport : bool, default False
        If True, use the supplied constant ``mu``. Otherwise use Sutherland's law
        unless an explicit viscosity profile is inferred from ``T_profile``.
    mu : float, optional
        Constant dynamic viscosity [Pa.s].
    eta_max, n : float, int
        ODE integration domain and resolution for the Blasius reference.

    Returns
    -------
    dict
        ``{"sim": {...}, "blasius": {...}}`` profile dicts that can be plotted
        with ``plot_line_profiles(..., x_key="eta")``.
    """
    y = np.asarray(y, dtype=float)
    u_profile = np.asarray(u_profile, dtype=float)
    rho_profile = np.asarray(rho_profile, dtype=float)

    if y.size != u_profile.size or y.size != rho_profile.size:
        raise ValueError("y, u_profile, and rho_profile must have the same length")

    valid = np.isfinite(y) & np.isfinite(u_profile) & np.isfinite(rho_profile)
    if T_profile is not None:
        T_profile = np.asarray(T_profile, dtype=float)
        if T_profile.size != y.size:
            raise ValueError("T_profile must have the same length as y")
        valid &= np.isfinite(T_profile)

    y = y[valid]
    u_profile = u_profile[valid]
    rho_profile = rho_profile[valid]
    if T_profile is not None:
        T_profile = T_profile[valid]

    if y.size < 2:
        raise ValueError("At least two valid points are required for the transform")

    # Viscosity profile: constant if requested, otherwise Sutherland-based.
    if const_transport and mu is not None:
        mu_profile = np.full_like(y, float(mu), dtype=float)
    elif T_profile is not None:
        mu_profile = sutherland_viscosity(T_profile)
    elif T_inf is not None:
        mu_profile = np.full_like(y, sutherland_viscosity(T_inf), dtype=float)
    else:
        mu_profile = np.full_like(y, 1.716e-5, dtype=float)

    rho_e = float(rho_profile[-1])
    mu_e = float(mu_profile[-1])
    nu_e = mu_e / rho_e
    u_ref = float(u_inf)

    if rho_e <= 0 or mu_e <= 0 or nu_e <= 0:
        raise ValueError("Edge density and viscosity must be positive")

    # Semi-local similarity coordinate: collapses to the Blasius eta when
    # density and viscosity are constant.
    try:
        from scipy.integrate import cumulative_trapezoid
        scale = np.sqrt(u_ref / (nu_e * max(float(x_loc), 1e-12)))
        integrand = np.sqrt(np.maximum(rho_profile * mu_profile, 0.0) / (rho_e * mu_e))
        eta_t = cumulative_trapezoid(integrand, y, initial=0.0) * scale
    except ImportError:
        scale = np.sqrt(u_ref / (nu_e * max(float(x_loc), 1e-12)))
        integrand = np.sqrt(np.maximum(rho_profile * mu_profile, 0.0) / (rho_e * mu_e))
        eta_t = np.zeros_like(y)
        if y.size > 1:
            eta_t[1:] = scale * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y))

    U_Ue = np.clip(u_profile / u_ref, 0.0, 1.5)

    eta_needed = float(np.max(eta_t)) if len(eta_t) > 0 else 0.0
    eta_solve = max(eta_max, float(np.ceil(eta_needed * 1.2)))
    eta_tab, fprime_tab = _blasius_ode_solution(eta_solve, n)
    U_bl = np.interp(eta_t, eta_tab, fprime_tab)

    d99_sim, _, _, _ = calculate_BL_thicknesses(eta_t, U_Ue)
    d99_bl, _, _, _ = calculate_BL_thicknesses(eta_t, U_bl)

    sim_profile = {
        "eta": eta_t,
        "values": U_Ue,
        "label": "Simulation (transformed)",
        "linestyle": "-",
        "field_key": "x_velocity",
        "boundary_layer_height": float(d99_sim),
    }

    bl_profile = {
        "eta": eta_t,
        "values": U_bl,
        "label": "Blasius (transformed)",
        "linestyle": "--",
        "field_key": "x_velocity",
        "boundary_layer_height": float(d99_bl),
    }

    return {"sim": sim_profile, "blasius": bl_profile}


# ---------------------------------------------------------------------------
# 8.  AERODYNAMIC FORCE CALCULATIONS
# ---------------------------------------------------------------------------

def compute_sectional_forces(surface_data, side="upper", rho_inf=1.0):
    """Compute sectional (per-unit-span) pressure and viscous forces.

    Parameters
    ----------
    surface_data : dict
        Output from ``extract_surface_properties`` for **one** side.
    side : str, default 'upper'
        Side label (used only for tangential-direction sign convention;
        normal always points away from the body).
    rho_inf : float, default 1.0
        Freestream density; only used for dynamic-pressure reporting.

    Returns
    -------
    dict
        ``dF_p_x``, ``dF_p_y``, ``dF_v_x``, ``dF_v_y`` (sectional vectors
        per point, shape ``(n,)``), ``D_cum``, ``L_cum`` (cumulative integrals
        along chord), and ``D_total``, ``L_total``, ``D_p``, ``D_v``, ``L_p``, ``L_v``.
    """
    s = np.asarray(surface_data.get("s", []), dtype=float)
    if len(s) < 2:
        raise ValueError(f"Surface data for '{side}' has < 2 points; cannot integrate.")

    # Arc-length segments
    ds = np.diff(s)
    # Segment mid-point normals / tangents (needed for proper integration)
    nx = np.asarray(surface_data.get("nx", np.zeros(len(s))), dtype=float)
    ny = np.asarray(surface_data.get("ny", np.zeros(len(s))), dtype=float)
    tx = np.asarray(surface_data.get("tangent_x", np.zeros(len(s))), dtype=float)
    ty = np.asarray(surface_data.get("tangent_y", np.zeros(len(s))), dtype=float)

    p = np.asarray(surface_data.get("p", np.zeros(len(s))), dtype=float)
    tau_w = np.asarray(surface_data.get("tau_w", np.zeros(len(s))), dtype=float)

    n_pts = len(s)

    # --- Pressure force ---
    # dF_p = -p * n * dA   (negative sign: n points outward, pressure pushes inward)
    # In 2-D, span = 1 m, so dA = ds * 1 m
    dF_p_x = -p * nx * 1.0  # per unit span
    dF_p_y = -p * ny * 1.0

    # --- Viscous force ---
    # dF_v = tau_w * t * dA  (shear acts along the surface tangent,
    #                          in the direction of the flow)
    # The tangent (tx, ty) should point in the streamwise direction.
    # For the upper surface, tangent = (+x direction).
    dF_v_x = tau_w * tx * 1.0
    dF_v_y = tau_w * ty * 1.0

    # --- Segment-level drag & lift contributions ---
    # Drag = x-component of total force
    dD_ds = (dF_p_x + dF_v_x)  # N/m per point
    dL_ds = (dF_p_y + dF_v_y)  # N/m per point

    # Cumulative integration
    D_cum = np.zeros(n_pts, dtype=float)
    L_cum = np.zeros(n_pts, dtype=float)
    for i in range(1, n_pts):
        D_cum[i] = D_cum[i - 1] + 0.5 * (dD_ds[i] + dD_ds[i - 1]) * ds[i - 1]
        L_cum[i] = L_cum[i - 1] + 0.5 * (dL_ds[i] + dL_ds[i - 1]) * ds[i - 1]

    # Total integrated forces
    D_total = float(D_cum[-1])
    L_total = float(L_cum[-1])

    # Pressure vs viscous breakdown
    D_p = float(np.trapz(dF_p_x, s))
    D_v = float(np.trapz(dF_v_x, s))
    L_p = float(np.trapz(dF_p_y, s))
    L_v = float(np.trapz(dF_v_y, s))

    return {
        "side": side,
        "x": np.asarray(surface_data.get("x", [])),
        "s": s,
        "ds": ds,
        # Sectional vectors (N/m per unit span)
        "dF_p_x": dF_p_x,
        "dF_p_y": dF_p_y,
        "dF_v_x": dF_v_x,
        "dF_v_y": dF_v_y,
        "dD_ds": dD_ds,
        "dL_ds": dL_ds,
        # Cumulative
        "D_cum": D_cum,
        "L_cum": L_cum,
        # Totals
        "D_total": D_total,
        "L_total": L_total,
        "D_p": D_p,
        "D_v": D_v,
        "L_p": L_p,
        "L_v": L_v,
    }


def compute_integrated_forces(surface_data_both_sides, rho_inf, u_inf,
                               chord_length=None):
    """Compute integrated drag & lift coefficients for the full body.

    Parameters
    ----------
    surface_data_both_sides : dict
        Full surface-properties dict with keys ``'upper'`` and/or ``'lower'``,
        as returned by ``extract_surface_properties``.
    rho_inf : float
        Freestream density [kg/m^3].
    u_inf : float
        Freestream velocity [m/s].
    chord_length : float, optional
        Reference chord [m].  If None, estimated from max x of the surface.

    Returns
    -------
    dict
        ``D_total``, ``L_total``, ``C_D``, ``C_L``, ``C_Dp``, ``C_Dv``,
        ``C_Lp``, ``C_Lv``, ``A_ref``, and per-side ``sectional`` data.
    """
    q_inf = 0.5 * rho_inf * u_inf ** 2  # dynamic pressure

    # Estimate chord if not given
    if chord_length is None:
        x_max = 0.0
        for side in ("upper", "lower"):
            sd = surface_data_both_sides.get(side)
            if sd is not None and len(sd.get("x", [])) > 0:
                x_max = max(x_max, float(sd["x"].max()))
        chord_length = x_max

    # Reference area (2-D: chord * span = chord * 1 m)
    A_ref = chord_length * 1.0

    # Force accumulation
    D_total = 0.0
    L_total = 0.0
    D_p = 0.0
    D_v = 0.0
    L_p = 0.0
    L_v = 0.0
    sectional = {}

    for side in ("upper", "lower"):
        sd = surface_data_both_sides.get(side)
        if sd is None or len(sd.get("s", [])) < 2:
            continue

        sec = compute_sectional_forces(sd, side=side, rho_inf=rho_inf)
        sectional[side] = sec
        D_total += sec["D_total"]
        L_total += sec["L_total"]
        D_p += sec["D_p"]
        D_v += sec["D_v"]
        L_p += sec["L_p"]
        L_v += sec["L_v"]

    # Non-dimensional coefficients
    C_D = D_total / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
    C_L = L_total / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
    C_Dp = D_p / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
    C_Dv = D_v / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
    C_Lp = L_p / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
    C_Lv = L_v / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0

    return {
        "D_total": D_total,
        "L_total": L_total,
        "D_p": D_p,
        "D_v": D_v,
        "L_p": L_p,
        "L_v": L_v,
        "C_D": C_D,
        "C_L": C_L,
        "C_Dp": C_Dp,
        "C_Dv": C_Dv,
        "C_Lp": C_Lp,
        "C_Lv": C_Lv,
        "A_ref": A_ref,
        "chord_length": chord_length,
        "q_inf": q_inf,
        "rho_inf": rho_inf,
        "u_inf": u_inf,
        "sectional": sectional,
    }


def load_probe_dat_files(probe_dir, max_probes=None):
    """Load converted probe .dat files into a structured dict.

    Parameters
    ----------
    probe_dir : str
        Directory containing flow_probe_*.dat files.
    max_probes : int, optional
        Maximum number of probes to load (starting from probe 0).

    Returns
    -------
    dict
        {probe_id: {"time": ndarray, "rho": ndarray, "u": ndarray,
                    "p": ndarray, "T": ndarray, "x_sample": ndarray,
                    "y_sample": ndarray, "level": ndarray,
                    "probe_x": float, "probe_y": float}}
    """
    from pathlib import Path

    dat_dir = Path(probe_dir)
    dat_files = sorted(dat_dir.glob("flow_probe_*.dat"),
                       key=lambda p: int(p.stem.split("_")[-1]))

    if max_probes is not None:
        dat_files = dat_files[:max_probes]

    probe_data = {}
    for fpath in dat_files:
        probe_id = int(fpath.stem.split("_")[-1])

        # Parse requested probe location from second header line
        probe_x = np.nan
        probe_y = np.nan
        with open(fpath) as fh:
            for _ in range(4):
                line = fh.readline()
                if line.startswith("# Requested location:"):
                    m = re.search(r'x=([\d.eE+\-]+)\s*cm.*y=([\d.eE+\-]+)', line)
                    if m:
                        probe_x = float(m.group(1))
                        probe_y = float(m.group(2))
                    break

        data = np.loadtxt(str(fpath), skiprows=4)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        probe_data[probe_id] = {
            "time": data[:, 0],
            "rho": data[:, 1],
            "u": data[:, 2],
            "p": data[:, 3],
            "T": data[:, 4],
            "x_sample": data[:, 5],
            "y_sample": data[:, 6],
            "level": data[:, 7],
            "probe_x": probe_x,
            "probe_y": probe_y,
        }

    return probe_data


# ---------------------------------------------------------------------------
# 9.  STABILITY DIAGNOSTICS  (2nd Mack mode)
# ---------------------------------------------------------------------------
#  Functions for hypersonic boundary-layer stability analysis based on
#  local mean-flow properties and wall-probe FFT data.
# ---------------------------------------------------------------------------


def compute_second_mode_frequency(bl_profiles, method="both"):
    """Estimate 2nd Mack mode frequency from local boundary-layer thickness.

    Parameters
    ----------
    bl_profiles : list[dict]
        List of BL profile dicts (output from ``extract_BL_profile_at_surface``).
        Each must contain ``delta_99``, ``delta_star``, ``u_edge``.
    method : str, default "both"
        "omega_star" ->  omega* approx 0.3 estimate only.
        "acoustic"   ->  half-wavelength acoustic estimate only.
        "both"       ->  return both.

    Returns
    -------
    dict
        Arrays keyed by ``x``, ``f_omega_star``, ``f_acoustic``,
        ``delta_99``, ``delta_star``, ``u_edge``, ``Re_x``.
    """
    n = len(bl_profiles)
    out = {
        "x": np.zeros(n),
        "delta_99": np.zeros(n),
        "delta_star": np.zeros(n),
        "u_edge": np.zeros(n),
        "Re_x": np.zeros(n),
        "f_omega_star": np.full(n, np.nan),
        "f_acoustic": np.full(n, np.nan),
    }

    for i, bl in enumerate(bl_profiles):
        out["x"][i] = bl.get("x", np.nan)
        out["delta_99"][i] = bl.get("delta_99", np.nan)
        out["delta_star"][i] = bl.get("delta_star", np.nan)
        out["u_edge"][i] = bl.get("u_edge", np.nan)
        out["Re_x"][i] = bl.get("Re_x", np.nan)

        d99 = out["delta_99"][i]
        ds = out["delta_star"][i]
        ue = out["u_edge"][i]

        if method in ("both", "omega_star") and ds > 0 and np.isfinite(ds):
            out["f_omega_star"][i] = 0.3 * ue / (2.0 * np.pi * ds)

        if method in ("both", "acoustic") and d99 > 0 and np.isfinite(d99):
            out["f_acoustic"][i] = ue / (2.0 * d99)

    return out


def compute_gpi_criterion(y, u, T, rho, R_specific=287.05):
    """Compute the Generalized Inflection Point (GPI) criterion.

    F(y) = d(rho * du/dy) / dy

    A GPI within the BL indicates a potential inviscid instability
    (a necessary condition for the 2nd Mack mode).

    Returns
    -------
    dict
        ``y_profile``, ``F``, ``y_gpi``, ``F_gpi``, ``unstable``,
        ``delta_99_est``.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)

    valid = np.isfinite(y) & np.isfinite(u) & np.isfinite(T) & np.isfinite(rho)
    if np.sum(valid) < 5:
        return {
            "y_profile": y,
            "F": np.full_like(y, np.nan),
            "y_gpi": np.nan,
            "F_gpi": np.nan,
            "unstable": False,
            "delta_99_est": np.nan,
        }

    y = y[valid]
    u = u[valid]
    T = T[valid]
    rho = rho[valid]
    sort_idx = np.argsort(y)
    y = y[sort_idx]
    u = u[sort_idx]
    T = T[sort_idx]
    rho = rho[sort_idx]

    dudy = np.gradient(u, y)
    F = np.gradient(rho * dudy, y)

    U = u / np.max(u) if np.max(u) > 0 else u
    idx99 = np.where(U >= 0.99)[0]
    delta_99_est = y[idx99[0]] if len(idx99) > 0 else y[-1]

    y_gpi = np.nan
    F_gpi = np.nan
    unstable = False

    inside = y <= delta_99_est
    if np.sum(inside) >= 3:
        y_in = y[inside]
        F_in = F[inside]
        sign = np.sign(F_in)
        sign[sign == 0] = 1
        zero_cross = np.where(np.diff(sign) != 0)[0]

        if len(zero_cross) > 0:
            iz = zero_cross[0]
            y0, y1 = y_in[iz], y_in[iz + 1]
            F0, F1 = F_in[iz], F_in[iz + 1]
            if abs(F1 - F0) > 1e-20:
                y_gpi = y0 - F0 * (y1 - y0) / (F1 - F0)
                F_gpi = 0.0
                tol = delta_99_est * 0.01
                unstable = (tol < y_gpi < delta_99_est - tol)

    return {
        "y_profile": y,
        "F": F,
        "y_gpi": y_gpi,
        "F_gpi": F_gpi,
        "unstable": unstable,
        "delta_99_est": delta_99_est,
    }


def compute_phase_speed_from_probes(probe_x, freq, Y_complex, target_freq,
                                    u_edge=None, T_edge=None, gamma=1.4,
                                    R_specific=287.05):
    """Compute phase speed from cross-spectral phase between adjacent probes.

    Parameters
    ----------
    probe_x : 1-D array-like
        Probe x-positions [m] (sorted increasing).
    freq : 1-D array-like
        Frequency bins [Hz].
    Y_complex : 2-D array-like
        Complex FFT coefficients (n_freq x n_probes), single-sided.
    target_freq : float
        Target frequency [Hz] at which to evaluate phase speed.
    u_edge : 1-D array-like, optional
        Edge velocity [m/s] at each probe x-position for reference lines.
    T_edge : 1-D array-like, optional
        Edge temperature [K] for reference lines.

    Returns
    -------
    dict
        ``x_mid``, ``c_p``, ``c_p_fast``, ``c_p_slow``, ``alpha_r``,
        ``delta_phi``, ``target_freq_actual``.
    """
    probe_x = np.asarray(probe_x, dtype=float)
    freq = np.asarray(freq, dtype=float)
    Y_complex = np.asarray(Y_complex, dtype=complex)

    sort_idx = np.argsort(probe_x)
    probe_x = probe_x[sort_idx]
    Y_complex = Y_complex[:, sort_idx]

    n_probes = len(probe_x)
    n_pairs = n_probes - 1
    if n_pairs < 1:
        return {
            "x_mid": np.array([]), "c_p": np.array([]),
            "c_p_fast": np.array([]), "c_p_slow": np.array([]),
            "alpha_r": np.array([]), "delta_phi": np.array([]),
            "target_freq_actual": np.nan,
        }

    f_idx = int(np.argmin(np.abs(freq - target_freq)))
    f_actual = freq[f_idx]
    omega = 2.0 * np.pi * f_actual

    phase = np.angle(Y_complex[f_idx, :])

    x_mid = np.zeros(n_pairs)
    c_p = np.full(n_pairs, np.nan)
    alpha_r = np.full(n_pairs, np.nan)
    delta_phi = np.full(n_pairs, np.nan)

    for ip in range(n_pairs):
        dx = probe_x[ip + 1] - probe_x[ip]
        x_mid[ip] = 0.5 * (probe_x[ip] + probe_x[ip + 1])
        if dx <= 0:
            continue
        dphi = phase[ip + 1] - phase[ip]
        dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
        alpha_r[ip] = dphi / dx
        if abs(alpha_r[ip]) > 1e-12:
            c_p[ip] = omega / alpha_r[ip]
        delta_phi[ip] = dphi

    c_p_fast = np.full(n_pairs, np.nan)
    c_p_slow = np.full(n_pairs, np.nan)
    if u_edge is not None and T_edge is not None:
        u_edge_v = np.asarray(u_edge, dtype=float)
        T_edge_v = np.asarray(T_edge, dtype=float)
        a_edge = np.sqrt(gamma * R_specific * T_edge_v)
        c_p_fast = np.interp(x_mid, probe_x, a_edge + u_edge_v)
        c_p_slow = np.interp(x_mid, probe_x, np.maximum(a_edge - u_edge_v, 0.0))

    return {
        "x_mid": x_mid, "c_p": c_p,
        "c_p_fast": c_p_fast, "c_p_slow": c_p_slow,
        "alpha_r": alpha_r, "delta_phi": delta_phi,
        "target_freq_actual": f_actual,
    }


def compute_growth_rate_from_probes(probe_x, freq, P1, target_freq,
                                    delta_99_interp=None,
                                    window_size=None):
    """Compute spatial growth rate alpha_i from FFT amplitude vs probe position.

    Fits ln(A) = -alpha_i * x + const over a sliding window.

    Parameters
    ----------
    probe_x : 1-D array-like
        Probe x-positions [m] (sorted increasing).
    freq : 1-D array-like
        Frequency bins [Hz].
    P1 : 2-D array-like
        Single-sided amplitude spectrum (n_freq x n_probes).
    target_freq : float
        Target frequency [Hz].
    delta_99_interp : 1-D array-like, optional
        BL thickness delta_99 [m] at each probe position.
    window_size : int, optional
        Number of probes per local fit. Default: max(5, n_probes // 10).

    Returns
    -------
    dict
        ``x``, ``alpha_i``, ``alpha_i_delta`` (if delta_99 given),
        ``A_at_f``, ``r_squared``.
    """
    probe_x = np.asarray(probe_x, dtype=float)
    freq = np.asarray(freq, dtype=float)
    P1 = np.asarray(P1, dtype=float)

    sort_idx = np.argsort(probe_x)
    probe_x = probe_x[sort_idx]
    P1 = P1[:, sort_idx]

    n_probes = len(probe_x)
    if n_probes < 3:
        return {"x": probe_x, "alpha_i": np.full(n_probes, np.nan),
                "A_at_f": np.full(n_probes, np.nan),
                "r_squared": np.full(n_probes, np.nan)}

    f_idx = int(np.argmin(np.abs(freq - target_freq)))
    A = P1[f_idx, :].copy()
    eps = 1e-20
    A = np.where(A < eps, eps, A)

    if window_size is None:
        window_size = max(5, n_probes // 10)
    window_size = min(window_size, n_probes)
    if window_size < 3:
        window_size = 3

    alpha_i = np.full(n_probes, np.nan)
    r2 = np.full(n_probes, np.nan)
    half = window_size // 2

    for i in range(n_probes):
        i0 = max(0, i - half)
        i1 = min(n_probes, i0 + window_size)
        i0 = max(0, i1 - window_size)
        x_win = probe_x[i0:i1]
        A_win = A[i0:i1]
        y_win = np.log(A_win)
        if len(x_win) < 3:
            continue
        coeffs = np.polyfit(x_win, y_win, 1)
        alpha_i[i] = -coeffs[0]
        y_fit = np.polyval(coeffs, x_win)
        ss_res = np.sum((y_win - y_fit) ** 2)
        ss_tot = np.sum((y_win - np.mean(y_win)) ** 2)
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan

    out = {
        "x": probe_x, "alpha_i": alpha_i,
        "A_at_f": A, "r_squared": r2,
    }

    if delta_99_interp is not None:
        d99 = np.asarray(delta_99_interp, dtype=float)
        if len(d99) == n_probes:
            out["alpha_i_delta"] = alpha_i * d99[sort_idx]
        else:
            out["alpha_i_delta"] = np.full(n_probes, np.nan)

    return out


# ===========================================================================
#  PHASE 1: DISTURBANCE SIGNAL RECONSTRUCTION
# ===========================================================================
#  Functions for recomposing time-domain disturbances from FFT frequency
#  bins, computing energy budgets, and residual statistics.
# ===========================================================================

def compute_complex_fft(signal_matrix, dt, win_scale=1.0):
    """Compute full complex FFT and single-sided amplitude spectrum.

    Parameters
    ----------
    signal_matrix : ndarray, shape (L, n_signals)
        Uniformly-sampled time series, one column per signal.
    dt : float
        Sampling interval [s].
    win_scale : float, default 1.0
        Amplitude compensation factor from windowing.

    Returns
    -------
    freq : ndarray, shape (n_freq,)
        Single-sided frequency bins [Hz].
    Y_complex : ndarray, shape (n_freq, n_signals)
        Single-sided complex FFT coefficients (NOT multiplied by 2, so DC and
        Nyquist are correctly positioned; use for phase).
    P1 : ndarray, shape (n_freq, n_signals)
        Single-sided amplitude spectrum (DC correct, other bins x2).
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(-1, 1)
    L, n_signals = signal_matrix.shape
    Fs = 1.0 / dt

    Y_full = np.fft.fft(signal_matrix, axis=0)

    n_freq = L // 2 + 1
    freq = Fs * np.arange(0, n_freq) / L
    Y_complex = Y_full[:n_freq, :].copy()

    P1 = np.abs(Y_complex) / L * win_scale
    if L % 2 == 0:
        P1[1:-1, :] = 2.0 * P1[1:-1, :]
    else:
        P1[1:, :] = 2.0 * P1[1:, :]

    return freq, Y_complex, P1


def reconstruct_from_bins(Y_full, bin_mask):
    """Reconstruct time-domain signal from selected frequency bins.

    Parameters
    ----------
    Y_full : ndarray, shape (L, n_signals)
        Full (two-sided) complex FFT coefficients from ``np.fft.fft``.
    bin_mask : ndarray, shape (L,) — bool mask, True for bins to retain.
        Must include both positive and conjugate-negative bins for a real
        reconstructed signal.

    Returns
    -------
    reconstructed : ndarray, shape (L, n_signals)
        Time-domain reconstruction via IFFT of masked spectrum.
    """
    Y_masked = Y_full * bin_mask[:, np.newaxis]
    reconstructed = np.real(np.fft.ifft(Y_masked, axis=0))
    return reconstructed


def reconstruct_from_harmonics(signal_matrix, dt, harmonic_freq, num_harmonics):
    """Reconstruct signal from fundamental and harmonic frequency bins only.

    Parameters
    ----------
    signal_matrix : ndarray, shape (L, n_signals)
        Input time series.
    dt : float
        Sampling interval [s].
    harmonic_freq : float
        Fundamental frequency [Hz].
    num_harmonics : int
        Number of harmonics to include (1 = fundamental only).

    Returns
    -------
    reconstructed_total : ndarray, shape (L, n_signals)
        Sum of all selected harmonic reconstructions.
    harmonic_signals : dict of ndarray
        Keys ``'1xf0'``, ``'2xf0'``, ..., each shape (L, n_signals).
    harmonic_bins : list of int
        Positive-frequency bin indices used.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(-1, 1)
    L, n_signals = signal_matrix.shape
    Fs = 1.0 / dt

    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq = np.fft.fftfreq(L, dt)

    harmonic_bins = []
    for nh in range(1, num_harmonics + 1):
        f_target = nh * harmonic_freq
        idx = int(np.argmin(np.abs(freq - f_target)))
        harmonic_bins.append(idx)

    total_mask = np.zeros(L, dtype=bool)
    for idx in harmonic_bins:
        total_mask[idx] = True
        neg_idx = L - idx
        if neg_idx < L and neg_idx != idx:
            total_mask[neg_idx] = True
    reconstructed_total = reconstruct_from_bins(Y_full, total_mask)

    harmonic_signals = {}
    for nh, idx in enumerate(harmonic_bins, 1):
        mask = np.zeros(L, dtype=bool)
        mask[idx] = True
        neg_idx = L - idx
        if neg_idx < L and neg_idx != idx:
            mask[neg_idx] = True
        harmonic_signals[f"{nh}xf0"] = reconstruct_from_bins(Y_full, mask)

    return reconstructed_total, harmonic_signals, harmonic_bins


def reconstruct_from_band(signal_matrix, dt, f_low, f_high):
    """Reconstruct signal from a frequency band via IFFT.

    Parameters
    ----------
    signal_matrix : ndarray, shape (L, n_signals)
        Input time series.
    dt : float
        Sampling interval [s].
    f_low, f_high : float
        Band-edges [Hz].

    Returns
    -------
    reconstructed : ndarray, shape (L, n_signals)
        Band-limited time-domain reconstruction.
    bin_mask : ndarray, shape (L,)
        Bool mask of retained frequency bins.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(-1, 1)
    L = signal_matrix.shape[0]
    Fs = 1.0 / dt

    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq = np.fft.fftfreq(L, dt)

    bin_mask = np.abs(freq) >= f_low
    bin_mask &= np.abs(freq) <= f_high
    bin_mask[0] = False

    reconstructed = reconstruct_from_bins(Y_full, bin_mask)
    return reconstructed, bin_mask


def reconstruct_from_top_frequencies(signal_matrix, dt, n_peaks=15,
                                     freq_range=None, min_peak_distance_hz=None):
    """Reconstruct signal from the N largest-amplitude frequency bins.

    Picks the top *n_peaks* amplitude peaks from the one-sided FFT spectrum
    for *each* signal column independently (per-probe), then takes the union
    across all columns so that every probe's dominant frequencies are
    included.  The time-domain reconstruction is built from those bins and
    their conjugate-negative counterparts.

    Parameters
    ----------
    signal_matrix : ndarray, shape (L, n_signals)
        Input time series.  If 1-D, reshaped to (L, 1).
    dt : float
        Sampling interval [s].
    n_peaks : int, default 15
        Number of peaks to select *per signal* (fewer if the spectrum has
        fewer bins).
    freq_range : tuple (f_min, f_max) or None, optional
        Frequency range [Hz] to restrict peak search.
    min_peak_distance_hz : float or None, optional
        Minimum separation between selected peaks [Hz].
        ``None`` auto-computes as ``max(3 * Fs/L, 1000.0)``.

    Returns
    -------
    reconstructed_total : ndarray, shape (L, n_signals)
        Sum of all selected peak reconstructions.
    component_signals : dict of ndarray
        Keys like ``"123.4 kHz"``, each shape (L, n_signals).
    peak_bins : list of int
        Positive-frequency bin indices selected.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(-1, 1)
    L, n_signals = signal_matrix.shape
    Fs = 1.0 / dt
    df = Fs / L

    if min_peak_distance_hz is None:
        min_peak_distance_hz = max(3.0 * df, 1000.0)

    # Full two-sided FFT
    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq = np.fft.fftfreq(L, dt)

    # One-sided positive frequencies (exclude DC)
    half = L // 2 + 1
    freq_pos = freq[:half]
    Y_pos = Y_full[:half, :]

    # Build candidate mask (common frequency range restriction)
    mask_pos = np.ones(half, dtype=bool)
    mask_pos[0] = False  # exclude DC
    if freq_range is not None:
        f_min, f_max = freq_range
        mask_pos &= (freq_pos >= f_min) & (freq_pos <= f_max)
    candidate_indices = np.where(mask_pos)[0]

    # --- Per-probe peak selection ---
    # For each column, independently pick the top N peaks, then take the
    # union so that every probe's dominant frequencies are represented.
    selected_set = set()
    for col in range(n_signals):
        amp_col = np.abs(Y_pos[:, col])
        remaining = list(candidate_indices)
        n_local = min(n_peaks, len(remaining))
        for _ in range(n_local):
            if not remaining:
                break
            amps = np.array([amp_col[i] for i in remaining])
            best = remaining[int(np.argmax(amps))]
            selected_set.add(best)
            f_best = freq_pos[best]
            remaining = [i for i in remaining
                         if abs(freq_pos[i] - f_best) >= min_peak_distance_hz]

    selected = sorted(selected_set)

    # Build two-sided mask
    total_mask = np.zeros(L, dtype=bool)
    for idx in selected:
        total_mask[idx] = True
        neg = L - idx if idx > 0 else 0
        if neg < L and neg != idx:
            total_mask[neg] = True

    reconstructed_total = reconstruct_from_bins(Y_full, total_mask)

    # Individual component signals with human-readable labels
    component_signals = {}
    for idx in selected:
        fval = freq_pos[idx]
        if fval >= 1e6:
            lbl = f"{fval / 1e6:.2f} MHz"
        elif fval >= 1e3:
            lbl = f"{fval / 1e3:.2f} kHz"
        else:
            lbl = f"{fval:.2f} Hz"
        mask = np.zeros(L, dtype=bool)
        mask[idx] = True
        neg = L - idx
        if neg < L and neg != idx:
            mask[neg] = True
        component_signals[lbl] = reconstruct_from_bins(Y_full, mask)

    return reconstructed_total, component_signals, selected


def compute_energy_budget(signal_measured, signal_reconstructed, harmonic_signals):
    """Compute energy fractions for fundamental, harmonics, and residual.

    Parameters
    ----------
    signal_measured : ndarray, shape (L,)
        Original (preprocessed) signal for one probe.
    signal_reconstructed : ndarray, shape (L,)
        Summed harmonic reconstruction.
    harmonic_signals : dict of ndarray
        Per-harmonic signals from ``reconstruct_from_harmonics``.

    Returns
    -------
    dict with keys:
        'E_total' : mean square of measured signal
        'E_recon_frac' : fraction of energy explained by reconstruction
        'E_residual_frac' : fraction of energy in residual
        'E_per_harm_frac' : dict of per-harmonic energy fractions
        'recon_to_total_ratio' : same as E_recon_frac
    """
    E_total = float(np.mean(signal_measured**2))
    eps = 1e-30
    if E_total < eps:
        E_total = eps

    E_recon = float(np.mean(signal_reconstructed**2))
    residual = signal_measured - signal_reconstructed
    E_residual = float(np.mean(residual**2))

    E_per_harm = {}
    for label, sig in harmonic_signals.items():
        E_h = float(np.mean(sig**2))
        E_per_harm[label] = E_h / E_total if E_total > eps else 0.0

    return {
        "E_total": E_total,
        "E_recon_frac": E_recon / E_total if E_total > eps else 0.0,
        "E_residual_frac": E_residual / E_total if E_total > eps else 0.0,
        "E_per_harm_frac": E_per_harm,
        "recon_to_total_ratio": E_recon / E_total if E_total > eps else 0.0,
    }


def compute_residual_stats(measured, reconstructed):
    """Compute residual statistics between measured and reconstructed signals.

    Parameters
    ----------
    measured : ndarray, shape (L,)
    reconstructed : ndarray, shape (L,)

    Returns
    -------
    dict with keys: rms_residual, rms_residual_rel, peak_residual,
    R_squared, skewness, kurtosis.
    """
    measured = np.asarray(measured, dtype=float).ravel()
    reconstructed = np.asarray(reconstructed, dtype=float).ravel()
    residual = measured - reconstructed

    rms_res = float(np.sqrt(np.mean(residual**2)))
    rms_meas = float(np.sqrt(np.mean(measured**2)))
    rms_rel = rms_res / rms_meas if rms_meas > 0 else 0.0

    R_sq = 1.0 - np.mean(residual**2) / np.mean(measured**2) if rms_meas > 0 else 0.0

    resid_c = residual - np.mean(residual)
    sigma = float(np.std(residual))
    if sigma > 1e-30:
        skew = float(np.mean(resid_c**3) / sigma**3)
        kurt = float(np.mean(resid_c**4) / sigma**4 - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    return {
        "rms_residual": rms_res,
        "rms_residual_rel": rms_rel,
        "peak_residual": float(np.max(np.abs(residual))),
        "R_squared": float(R_sq),
        "skewness": skew,
        "kurtosis": kurt,
    }


# ===========================================================================
#  DISTURBANCE WINDOW DETECTION (Phase 1b)
# ===========================================================================

def detect_disturbance_window(signal, time, laser_start_time=None,
                               baseline_margin=0.001,
                               pre_event_fraction=0.02,
                               onset_sigma=5.0, offset_sigma=2.0,
                               onset_peak_fraction=None,
                               offset_peak_fraction=None,
                               packet_selection="first_threshold",
                               min_active_duration=50e-6,
                               min_quiet_duration=100e-6,
                               pad_before=20e-6, pad_after=50e-6,
                               min_samples=256, max_samples=50000,
                               detector_mode="energy"):
    """Detect the disturbance time window in a probe signal using a
    broadband energy detector with hysteresis thresholds.

    Designed for laser-pulse / shock-tunnel experiments where a finite
    wavepacket (shock + oscillatory tail) appears after a known trigger
    time.  Returns the onset and offset indices and times, plus diagnostic
    values, so the calling code can reconstruct only the active interval.

    The detector algorithm:

    1. Estimate baseline statistics from a pre-event interval (samples
       before *laser_start_time - baseline_margin*, or the earliest
       *pre_event_fraction* of the record if no laser time is given).
    2. Compute a broadband energy metric: smoothed absolute signal or
       smoothed squared signal.
    3. Estimate baseline statistics on that same metric, then detect *onset*
       when it exceeds ``metric_baseline + onset_sigma * metric_noise_scale`` for at least
       *min_active_duration*.
    4. Detect *offset* when the metric drops below
       ``baseline + offset_sigma * noise_scale`` for at least
       *min_quiet_duration* (hysteresis).
    5. Apply *pad_before* and *pad_after* (clamped to record bounds).
    6. Enforce *min_samples* / *max_samples* limits.

    Parameters
    ----------
    signal : ndarray, shape (L,)
        Raw probe signal (not necessarily zero-mean).
    time : ndarray, shape (L,)
        Time array [s], assumed uniformly spaced and monotonically
        increasing.
    laser_start_time : float or None, optional
        Expected event time [s].  Used only as a lower bound — the
        detector does not search for onset before this time minus
        *baseline_margin*.  If None, the baseline is estimated from
        the initial fraction of the record.
    baseline_margin : float, default 0.001
        Time [s] subtracted from *laser_start_time* to define the
        pre-event baseline window.  Ignored if *laser_start_time* is
        None.
    pre_event_fraction : float, default 0.02
        Fallback fraction of the total record used for baseline
        estimation when *laser_start_time* is None.
    onset_sigma : float, default 5.0
        Threshold multiplier above baseline noise for onset detection.
    onset_peak_fraction : float or None, optional
        When ``packet_selection='dominant_peak'``, onset is additionally
        constrained to this fraction of the selected packet's peak metric
        above its metric baseline.
    offset_sigma : float, default 2.0
        Threshold multiplier above baseline noise for offset detection
        (lower than *onset_sigma* to provide hysteresis).
    offset_peak_fraction : float or None, optional
        When set, the offset threshold is additionally constrained to be at
        least this fraction of the detected pulse's peak metric above its
        metric baseline.  This gives each probe a tail cutoff relative to its
        own pulse strength.  ``None`` preserves baseline-noise-only offset
        detection.
    packet_selection : {'first_threshold', 'dominant_peak'}
        ``'first_threshold'`` preserves the legacy behaviour.
        ``'dominant_peak'`` identifies the strongest packet after the trigger
        and bounds that packet, preventing an earlier weak precursor from
        masking a later, larger arrival.
    min_active_duration : float, default 50e-6
        Minimum time [s] the metric must stay above the onset threshold
        before declaring onset (avoids noise spikes).
    min_quiet_duration : float, default 100e-6
        Minimum time [s] the metric must stay below the offset threshold
        before declaring offset (avoids premature end).
    pad_before : float, default 20e-6
        Time [s] added before the detected onset to include pre-shock
        baseline context.
    pad_after : float, default 50e-6
        Time [s] added after detected offset to capture late decay.
    min_samples : int, default 256
        Minimum number of samples required in the window.  If the
        detected window is shorter, the detector returns a fallback
        status.
    max_samples : int, default 50000
        Maximum number of samples allowed in the window.  If the
        detected window exceeds this, it is truncated.
    detector_mode : str, default 'energy'
        Detection metric.  ``'energy'`` uses the smoothed squared
        signal; ``'abs'`` uses the smoothed absolute value.

    Returns
    -------
    dict with keys:
        status : str
            ``'valid'``, ``'fallback'``, or ``'invalid'``.
        start_idx, end_idx : int
            Sample indices into *time* / *signal*.
        start_time, end_time : float
            Corresponding times [s].
        duration : float
            Window duration ``end_time - start_time`` [s].
        n_samples : int
            Number of samples in the window.
        baseline : float
            Median of the pre-event signal.
        noise_scale : float
            MAD (or RMS) of the pre-event signal.
        onset_threshold : float
            Metric baseline + onset_sigma * metric noise scale.
        offset_threshold : float
            Baseline + offset_sigma * noise_scale.
        onset_time, offset_time : float
            Detected onset / offset times (before padding) [s].
        onset_idx, offset_idx : int
            Detected onset / offset indices (before padding).
        laser_lower_bound : float or None
            Lower bound used for the search [s].
        fallback_reason : str or None
            Reason for fallback/invalid status.
        metric : ndarray
            The detection metric array (useful for diagnostic plots).
        metric_baseline, metric_noise_scale : float
            Pre-event statistics in the units of ``metric``.
        pulse_peak_metric : float or None
            Peak detection metric after the detected onset.
        pulse_peak_time : float or None
            Time of the selected packet's peak metric.
        metric_time : ndarray
            Time array matching *metric* (same as *time*).
    """
    # --- Input validation ---
    signal = np.asarray(signal, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    L = len(signal)
    if len(time) != L:
        raise ValueError(f"signal ({L}) and time ({len(time)}) must have the same length")

    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time must be monotonically increasing with positive dt")

    # Default result (invalid)
    result = {
        "status": "invalid",
        "start_idx": 0, "end_idx": L - 1,
        "start_time": time[0], "end_time": time[-1],
        "duration": time[-1] - time[0],
        "n_samples": L,
        "baseline": 0.0, "noise_scale": 0.0,
        "metric_baseline": 0.0, "metric_noise_scale": 0.0,
        "pulse_peak_metric": None, "pulse_peak_time": None,
        "baseline_source": None,
        "onset_threshold": 0.0, "offset_threshold": 0.0,
        "onset_time": None, "offset_time": None,
        "onset_idx": None, "offset_idx": None,
        "laser_lower_bound": laser_start_time,
        "fallback_reason": None, "metric": None, "metric_time": None,
    }

    # --- Determine pre-event baseline window ---
    pre_event_available = False
    if laser_start_time is not None:
        baseline_end_time = laser_start_time - baseline_margin
        pre_event_available = baseline_end_time > time[0]
    else:
        baseline_end_time = time[0] + pre_event_fraction * (time[-1] - time[0])
        pre_event_available = True

    if pre_event_available:
        baseline_start_idx = 0
        baseline_end_idx = int(np.searchsorted(time, baseline_end_time))
    else:
        # The record can begin after the laser has fired (e.g. the early
        # probes).  In that case its initial samples are not a baseline: use
        # the quietest short block anywhere in the record instead.
        baseline_width = min(L, max(32, int(pre_event_fraction * L)))
        starts = np.arange(0, max(1, L - baseline_width + 1),
                           max(1, baseline_width // 8), dtype=int)
        if starts[-1] != L - baseline_width:
            starts = np.append(starts, L - baseline_width)
        sums = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
        sums_sq = np.concatenate(([0.0], np.cumsum(signal ** 2, dtype=float)))
        means = (sums[starts + baseline_width] - sums[starts]) / baseline_width
        variances = ((sums_sq[starts + baseline_width] - sums_sq[starts]) /
                     baseline_width - means ** 2)
        baseline_start_idx = int(starts[np.argmin(np.maximum(variances, 0.0))])
        baseline_end_idx = baseline_start_idx + baseline_width

    baseline_end_idx = min(baseline_end_idx, L)
    baseline_start_idx = max(0, min(baseline_start_idx, baseline_end_idx - 1))
    baseline_source = "pre_event" if pre_event_available else "quietest_block"
    pre_event = signal[baseline_start_idx:baseline_end_idx]
    if len(pre_event) < 2:
        # Fallback — use the first few samples
        pre_event = signal[:max(2, min(10, L // 10))]
        if len(pre_event) < 2:
            pre_event = signal[:2]

    baseline = float(np.median(pre_event))
    noise_scale = float(np.median(np.abs(pre_event - baseline)))
    if noise_scale < 1e-30:
        noise_scale = float(np.std(pre_event))
    if noise_scale < 1e-30:
        noise_scale = 1e-30

    # --- Compute broadband energy metric ---
    signal_c = signal - baseline  # remove baseline offset

    if detector_mode == "energy":
        metric_raw = signal_c ** 2
    else:  # "abs"
        metric_raw = np.abs(signal_c)

    # Smooth the metric with a moving average.  ``np.convolve`` uses a direct
    # O(L * win_len) algorithm here; at 100,000 samples/probe that made the
    # detector take several hours for 2,000 probes.  The cumulative-sum form
    # below has identical ``mode='same'`` zero-padded semantics in O(L).
    win_len = max(3, int(L * 0.01))
    same_start = (win_len - 1) // 2
    pad_left = win_len - 1 - same_start
    pad_right = same_start
    padded_metric = np.pad(metric_raw, (pad_left, pad_right), mode="constant")
    cumulative = np.concatenate(([0.0], np.cumsum(padded_metric, dtype=float)))
    metric = (cumulative[win_len:] - cumulative[:-win_len]) / win_len
    metric_time = time

    # Thresholds must use statistics in the same units as the detection
    # metric.  The old code compared a squared-signal metric to the raw
    # signal baseline, which made the detector miss clear pulses (notably
    # probe 499) and trigger the arbitrary max-sample fallback window.
    metric_pre_event = metric[baseline_start_idx:baseline_end_idx]
    metric_baseline = float(np.median(metric_pre_event))
    metric_noise_scale = float(np.median(np.abs(metric_pre_event - metric_baseline)))
    if metric_noise_scale < 1e-30:
        metric_noise_scale = float(np.std(metric_pre_event))
    if metric_noise_scale < 1e-30:
        metric_noise_scale = 1e-30

    onset_threshold = metric_baseline + onset_sigma * metric_noise_scale
    offset_threshold = metric_baseline + offset_sigma * metric_noise_scale

    # --- Find the selected packet's onset ---
    # Only search after the laser lower bound.
    if laser_start_time is not None:
        search_start_idx = max(0, int(np.searchsorted(time, laser_start_time - baseline_margin)))
    else:
        search_start_idx = baseline_end_idx

    if packet_selection not in ("first_threshold", "dominant_peak"):
        raise ValueError("packet_selection must be 'first_threshold' or 'dominant_peak'")

    def _run_starts(mask, run_length):
        """Return starts of true runs at least ``run_length`` samples long."""
        if run_length > len(mask):
            return np.empty(0, dtype=int)
        cumulative_mask = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        return np.flatnonzero(
            cumulative_mask[run_length:] - cumulative_mask[:-run_length] == run_length)

    onset_idx_val = None
    peak_idx_val = None
    pulse_peak_metric = None
    n_active_min = max(1, int(min_active_duration / dt))
    n_quiet_min = max(1, int(min_quiet_duration / dt))

    if packet_selection == "dominant_peak":
        peak_idx_val = search_start_idx + int(np.argmax(metric[search_start_idx:]))
        pulse_peak_metric = float(metric[peak_idx_val])
        if onset_peak_fraction is not None:
            if onset_peak_fraction <= 0.0 or onset_peak_fraction >= 1.0:
                raise ValueError("onset_peak_fraction must lie between 0 and 1")
            peak_relative_onset = metric_baseline + onset_peak_fraction * (
                pulse_peak_metric - metric_baseline)
            onset_threshold = max(onset_threshold, peak_relative_onset)

        # Walk backward from the dominant peak to the end of the last quiet
        # interval.  That bounds this packet rather than a preceding weak one.
        quiet_starts = _run_starts(metric < onset_threshold, n_quiet_min)
        quiet_starts = quiet_starts[quiet_starts + n_quiet_min <= peak_idx_val]
        if quiet_starts.size:
            onset_idx_val = int(quiet_starts[-1] + n_quiet_min)
        else:
            onset_idx_val = search_start_idx
    else:
        i = search_start_idx
        while i < L:
            if metric[i] > onset_threshold:
                # Check if it stays above for min_active_duration
                j = i
                while j < L and j - i < n_active_min:
                    if metric[j] <= onset_threshold:
                        i = j + 1
                        break
                    j += 1
                if j - i >= n_active_min:
                    onset_idx_val = i
                    break
            i += 1

    # --- Find offset ---
    offset_idx_val = None

    if onset_idx_val is not None:
        if pulse_peak_metric is None:
            peak_idx_val = onset_idx_val + int(np.argmax(metric[onset_idx_val:]))
            pulse_peak_metric = float(metric[peak_idx_val])
        if offset_peak_fraction is not None:
            if offset_peak_fraction <= 0.0:
                raise ValueError("offset_peak_fraction must be positive or None")
            peak_relative_threshold = metric_baseline + offset_peak_fraction * (
                pulse_peak_metric - metric_baseline)
            # Keep the noise-based threshold as a floor so a weak pulse is
            # never tracked below the measured background fluctuation level.
            offset_threshold = max(offset_threshold, peak_relative_threshold)
        i = peak_idx_val + 1 if packet_selection == "dominant_peak" else onset_idx_val + 1
        while i < L:
            if metric[i] < offset_threshold:
                # Check if it stays below for min_quiet_duration
                j = i
                while j < L and j - i < n_quiet_min:
                    if metric[j] >= offset_threshold:
                        i = j + 1
                        break
                    j += 1
                if j - i >= n_quiet_min:
                    offset_idx_val = i
                    break
            i += 1

    # --- Handle fallback for offset ---
    fallback_reason = None
    if onset_idx_val is None:
        fallback_reason = "onset_not_detected"
        onset_idx_val = baseline_end_idx
        offset_idx_val = L - 1
    elif offset_idx_val is None:
        fallback_reason = "offset_not_detected"
        offset_idx_val = L - 1

    # --- Apply padding ---
    pad_before_samples = max(0, int(pad_before / dt))
    pad_after_samples = max(0, int(pad_after / dt))

    start_idx = max(0, onset_idx_val - pad_before_samples)
    end_idx = min(L - 1, offset_idx_val + pad_after_samples)

    # --- Enforce min / max samples ---
    n_samples = end_idx - start_idx + 1
    if n_samples < min_samples:
        if start_idx > 0:
            extra = min_samples - n_samples
            start_idx = max(0, start_idx - extra)
        if end_idx - start_idx + 1 < min_samples and end_idx < L - 1:
            extra = min_samples - (end_idx - start_idx + 1)
            end_idx = min(L - 1, end_idx + extra)
        n_samples = end_idx - start_idx + 1
        if n_samples < min_samples:
            fallback_reason = "window_too_short"

    if n_samples > max_samples:
        # Truncate symmetrically
        excess = n_samples - max_samples
        start_idx += excess // 2
        end_idx -= (excess - excess // 2)
        if start_idx >= end_idx:
            start_idx = max(0, end_idx - max_samples)
        n_samples = max_samples

    # --- Determine final status ---
    status = "valid"
    if fallback_reason is not None:
        if onset_idx_val == baseline_end_idx and offset_idx_val == L - 1:
            status = "fallback"
        else:
            status = "valid (partial fallback)"

    # Build result
    onset_time_val = float(time[onset_idx_val]) if onset_idx_val is not None else None
    offset_time_val = float(time[offset_idx_val]) if offset_idx_val is not None else None

    result.update({
        "status": status,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": float(time[start_idx]),
        "end_time": float(time[end_idx]),
        "duration": float(time[end_idx] - time[start_idx]),
        "n_samples": n_samples,
        "baseline": baseline,
        "noise_scale": noise_scale,
        "metric_baseline": metric_baseline,
        "metric_noise_scale": metric_noise_scale,
        "pulse_peak_metric": pulse_peak_metric,
        "pulse_peak_time": float(time[peak_idx_val]) if peak_idx_val is not None else None,
        "baseline_source": baseline_source,
        "onset_threshold": onset_threshold,
        "offset_threshold": offset_threshold,
        "onset_time": onset_time_val,
        "offset_time": offset_time_val,
        "onset_idx": onset_idx_val,
        "offset_idx": offset_idx_val,
        "laser_lower_bound": laser_start_time,
        "fallback_reason": fallback_reason,
        "metric": metric,
        "metric_time": metric_time,
    })
    return result


def extract_probe_window(signal, time, start_idx, end_idx):
    """Extract a contiguous window from a signal and time array.

    Parameters
    ----------
    signal : ndarray, shape (L,)
    time : ndarray, shape (L,)
    start_idx : int
        Starting index (inclusive).
    end_idx : int
        Ending index (inclusive).

    Returns
    -------
    window_signal : ndarray
    window_time : ndarray
    n_samples : int
    """
    signal = np.asarray(signal, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    L = len(signal)

    start_idx = max(0, min(start_idx, L - 1))
    end_idx = max(start_idx, min(end_idx, L - 1))

    n_samples = end_idx - start_idx + 1
    window_signal = signal[start_idx:end_idx + 1].copy()
    window_time = time[start_idx:end_idx + 1].copy()
    return window_signal, window_time, n_samples


# ===========================================================================
#  PHASE 2: TRANSIENT ANALYSIS (STFT / HILBERT ENVELOPE)
# ===========================================================================

def compute_spectrogram(signal, fs, nperseg=256, noverlap=None, window="hann"):
    """Compute STFT spectrogram in dB.

    Parameters
    ----------
    signal : ndarray, shape (L,)
    fs : float
        Sampling frequency [Hz].
    nperseg : int, default 256
        Segment length.
    noverlap : int, optional
        Overlap in samples. Defaults to 75% of nperseg.
    window : str, default 'hann'

    Returns
    -------
    f : ndarray — frequency bins [Hz]
    t : ndarray — time bins [s]
    Sxx_dB : ndarray — spectrogram magnitude in dB
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise ImportError("scipy.signal is required for compute_spectrogram")

    signal = np.asarray(signal, dtype=float).ravel()
    if noverlap is None:
        noverlap = int(0.75 * nperseg)
    f, t, Sxx = scipy_signal.spectrogram(
        signal, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, mode="magnitude",
    )
    eps = 1e-20
    Sxx_dB = 10.0 * np.log10(np.maximum(Sxx, eps))
    return f, t, Sxx_dB


def bandpass_hilbert_envelope(signal, fs, f_low, f_high, order=4):
    """Apply bandpass filter and extract analytic signal envelope.

    Parameters
    ----------
    signal : ndarray, shape (L,)
    fs : float
        Sampling frequency [Hz].
    f_low, f_high : float
        Passband edges [Hz].
    order : int, default 4
        Butterworth filter order.

    Returns
    -------
    envelope : ndarray — instantaneous amplitude
    inst_phase : ndarray — unwrapped instantaneous phase [rad]
    inst_freq : ndarray — instantaneous frequency [Hz]
    filtered : ndarray — bandpass-filtered signal
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise ImportError("scipy.signal is required for bandpass_hilbert_envelope")

    signal = np.asarray(signal, dtype=float).ravel()
    nyquist = fs / 2.0
    Wn = [f_low / nyquist, f_high / nyquist]

    sos = scipy_signal.butter(order, Wn, btype="band", output="sos")
    filtered = scipy_signal.sosfiltfilt(sos, signal)

    analytic = scipy_signal.hilbert(filtered)
    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(inst_phase) * fs / (2.0 * np.pi)

    return envelope, inst_phase, inst_freq, filtered


def extract_packet_stats(envelope, time):
    """Extract wavepacket statistics from envelope.

    Parameters
    ----------
    envelope : ndarray — instantaneous amplitude
    time : ndarray — time array [s]

    Returns
    -------
    dict with keys: peak_amplitude, peak_time, arrival_time,
    half_width_half_max, integrated_energy.
    """
    envelope = np.asarray(envelope, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    peak_idx = int(np.argmax(envelope))
    peak_amp = float(envelope[peak_idx])
    peak_time = float(time[peak_idx])

    threshold = 0.1 * peak_amp
    leading = np.where(envelope[:peak_idx] >= threshold)[0]
    if len(leading) > 0:
        arrival_time = float(time[leading[0]])
    else:
        arrival_time = float(time[0])

    half_threshold = 0.5 * peak_amp
    half_leading = np.where(envelope[:peak_idx] >= half_threshold)[0]
    hwhm = float(peak_time - time[half_leading[0]]) if len(half_leading) > 0 else None

    energy = float(np.trapz(envelope**2, time))

    return {
        "peak_amplitude": peak_amp,
        "peak_time": peak_time,
        "arrival_time": arrival_time,
        "half_width_half_max": hwhm,
        "integrated_energy": energy,
    }


# ===========================================================================
#  PHASE 3: NONLINEAR INTERACTION DIAGNOSTICS (BISPECTRUM)
# ===========================================================================

def compute_bicoherence(signal, fs, nperseg=256, noverlap=None):
    """Compute squared bicoherence b²(f1, f2) ∈ [0, 1].

    The bicoherence measures quadratic phase coupling between frequency
    triads (f1, f2, f1+f2).  Values near 1 indicate strong nonlinear
    coupling; values near 0 indicate independent modes.

    Parameters
    ----------
    signal : ndarray, shape (L,)
    fs : float
        Sampling frequency [Hz].
    nperseg : int, default 256
        Segment length for averaging.
    noverlap : int, optional
        Overlap in samples. Defaults to nperseg // 2.

    Returns
    -------
    freq : ndarray, shape (n_freq,) — frequency bins [Hz]
    bicoh : ndarray, shape (n_freq, n_freq) — upper-triangular b² matrix
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise ImportError("scipy.signal is required for compute_bicoherence")

    signal = np.asarray(signal, dtype=float).ravel()
    if noverlap is None:
        noverlap = nperseg // 2

    nstep = nperseg - noverlap
    if nstep <= 0:
        nstep = 1

    window = np.hanning(nperseg)

    n_segments = max(1, (len(signal) - nperseg) // nstep + 1)
    n_freq = nperseg // 2 + 1

    X = np.zeros((n_segments, nperseg), dtype=complex)
    for i in range(n_segments):
        start = i * nstep
        if start + nperseg > len(signal):
            break
        seg = signal[start:start + nperseg] * window
        X[i] = np.fft.fft(seg)
    n_segments = np.sum(np.any(X != 0, axis=1))

    X = X[:n_segments, :n_freq]

    bicoh = np.zeros((n_freq, n_freq), dtype=float)
    eps = 1e-30

    for f1 in range(1, n_freq - 1):
        for f2 in range(f1, n_freq):
            f3 = f1 + f2
            if f3 >= n_freq:
                continue
            B = np.mean(X[:, f1] * X[:, f2] * np.conj(X[:, f3]), axis=0)
            denom = np.mean(np.abs(X[:, f1] * X[:, f2]) ** 2, axis=0) * \
                    np.mean(np.abs(X[:, f3]) ** 2, axis=0)
            bicoh[f1, f2] = np.abs(B) ** 2 / (denom + eps)

    freq = np.fft.fftfreq(nperseg, 1.0 / fs)[:n_freq]
    return freq, bicoh


def extract_triad_bicoherence(freq, bicoh_matrix, target_freqs):
    """Extract bicoherence values at specific frequency triads.

    Parameters
    ----------
    freq : ndarray — frequency bins [Hz]
    bicoh_matrix : ndarray, (n_freq, n_freq) — b²(f1, f2)
    target_freqs : list of float
        Frequencies to evaluate [Hz].  For each pair (fi, fj) with
        fi+fj <= freq.max(), the triad bicoherence is reported.

    Returns
    -------
    dict — keys like 'b²(1.00e+07, 1.00e+07)', values are b² ∈ [0,1].
    """
    result = {}
    for fi in target_freqs:
        fi_idx = int(np.argmin(np.abs(freq - fi)))
        for fj in target_freqs:
            fj_idx = int(np.argmin(np.abs(freq - fj)))
            fk = fi + fj
            if fk > freq[-1]:
                continue
            if fj_idx >= fi_idx:
                b_val = float(bicoh_matrix[fi_idx, fj_idx])
            else:
                b_val = float(bicoh_matrix[fj_idx, fi_idx])
            label = f"b²({fi:.3e}, {fj:.3e})"
            result[label] = b_val
    return result
