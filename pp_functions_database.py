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
import struct
import traceback
import warnings
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

    # Iterable of names — accept canonical names as documented and expand
    # them to their solver-specific raw candidates. Unknown names are still
    # treated as canonical = raw for custom fields.
    out = {}
    for name in raw_names:
        canonical = str(name)
        out[canonical] = canonical_to_raws.get(canonical, [canonical])
    return out


def _collapse_field(values, dimensionality):
    """Collapse trailing dimensions for lower-dimensional datasets."""
    values = np.asarray(values)
    if dimensionality == 2 and values.ndim == 3:
        return values[:, :, 0]
    if dimensionality == 1 and values.ndim >= 2:
        return values.reshape(values.shape[0])
    return values


def _compute_native_amr_vorticity(ds):
    """Composite signed 2-D vorticity after differentiating native AMR grids.

    Differentiating a finest-level ``covering_grid`` is invalid in regions
    represented only by coarser AMR levels: yt repeats each coarse value into
    a block of fine cells, so a finite difference produces zero block
    interiors and impulses at block edges.  This routine reverses that order:

    1. obtain one interpolated ghost cell around each native grid,
    2. evaluate ``dv/dx - du/dy`` at that grid's own resolution, and
    3. composite coarse-to-fine, allowing finer grids to overwrite covered
       coarse cells.

    The returned array follows the package convention ``(nx, ny)`` and uses
    inverse seconds for PeleC's CGS velocity/coordinate fields.
    """
    if int(ds.dimensionality) != 2:
        raise ValueError(
            "Native AMR vorticity currently supports 2-D PeleC datasets only"
        )

    max_level = int(ds.index.max_level)
    refine_by = int(ds.refine_by)
    finest_dims = (
        np.asarray(ds.domain_dimensions, dtype=int)
        * refine_by ** max_level
    )
    composite = np.full(
        (int(finest_dims[0]), int(finest_dims[1])),
        np.nan,
        dtype=float,
    )
    domain_dims = np.asarray(ds.domain_dimensions, dtype=int)
    velocity_fields = [
        ("boxlib", "x_velocity"),
        ("boxlib", "y_velocity"),
    ]

    # yt requires symmetric ghost zones, including the collapsed z direction.
    # Temporarily enabling periodic lookup supplies those cells.  Values at
    # the real x/y domain boundaries are replaced below with second-order
    # one-sided differences, so periodic data never enters the final curl.
    periodicity_was_forced = bool(getattr(ds, "_force_periodicity", False))
    ds.force_periodicity(True)
    try:
        for grid in sorted(ds.index.grids, key=lambda item: item.Level):
            level = int(grid.Level)
            # yt 4.4 emits a benign weak-proxy RuntimeWarning while building
            # this smoothed ghost cube.  The requested raw velocity fields are
            # still populated correctly; suppress only that exact warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Something went wrong during field computation.*"
                        "weakref.ProxyType"
                    ),
                    category=RuntimeWarning,
                )
                ghosted = grid.retrieve_ghost_zones(
                    1,
                    velocity_fields,
                    smoothed=True,
                )
                u = np.asarray(ghosted[velocity_fields[0]], dtype=float)
                v = np.asarray(ghosted[velocity_fields[1]], dtype=float)
            dx = float(grid.dds[0].d)
            dy = float(grid.dds[1].d)

            dv_dx = (
                v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]
            ) / (2.0 * dx)
            du_dy = (
                u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]
            ) / (2.0 * dy)
            curl = dv_dx - du_dy

            start = np.asarray(grid.get_global_startindex(), dtype=int)
            active = np.asarray(grid.ActiveDimensions, dtype=int)
            level_dims = domain_dims * refine_by ** level

            # Replace the periodic ghost contribution at physical boundaries
            # with second-order one-sided derivatives.
            if start[0] == 0:
                dv_dx_left = (
                    -3.0 * v[1, 1:-1, 1:-1]
                    + 4.0 * v[2, 1:-1, 1:-1]
                    - v[3, 1:-1, 1:-1]
                ) / (2.0 * dx)
                curl[0, :, :] = dv_dx_left - du_dy[0, :, :]
            if start[0] + active[0] == level_dims[0]:
                dv_dx_right = (
                    3.0 * v[-2, 1:-1, 1:-1]
                    - 4.0 * v[-3, 1:-1, 1:-1]
                    + v[-4, 1:-1, 1:-1]
                ) / (2.0 * dx)
                curl[-1, :, :] = dv_dx_right - du_dy[-1, :, :]
            if start[1] == 0:
                du_dy_bottom = (
                    -3.0 * u[1:-1, 1, 1:-1]
                    + 4.0 * u[1:-1, 2, 1:-1]
                    - u[1:-1, 3, 1:-1]
                ) / (2.0 * dy)
                curl[:, 0, :] = dv_dx[:, 0, :] - du_dy_bottom
            if start[1] + active[1] == level_dims[1]:
                du_dy_top = (
                    3.0 * u[1:-1, -2, 1:-1]
                    - 4.0 * u[1:-1, -3, 1:-1]
                    + u[1:-1, -4, 1:-1]
                ) / (2.0 * dy)
                curl[:, -1, :] = dv_dx[:, -1, :] - du_dy_top

            curl = curl[:, :, 0]
            prolongation = refine_by ** (max_level - level)
            if prolongation > 1:
                curl = np.repeat(curl, prolongation, axis=0)
                curl = np.repeat(curl, prolongation, axis=1)

            finest_start = start[:2] * prolongation
            finest_end = finest_start + np.asarray(curl.shape, dtype=int)
            composite[
                finest_start[0]:finest_end[0],
                finest_start[1]:finest_end[1],
            ] = curl
    finally:
        ds.force_periodicity(periodicity_was_forced)

    missing = int(np.count_nonzero(~np.isfinite(composite)))
    if missing:
        raise RuntimeError(
            f"Native AMR vorticity composite left {missing} cells unfilled"
        )
    return composite


def load_pelec_plotfile(plotfile_path, field_names=None, alias_map=None,
                        convert_to_mks=True, derive_native_vorticity=False,
                        maximum_level=None):
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
    derive_native_vorticity : bool, default False
        Compute signed ``dv/dx - du/dy`` on native AMR patches even when
        ``field_names=None`` requests the general default field set.
    maximum_level : int, optional
        Cap the covering-grid level. This is primarily intended for bounded
        control-volume validation where a full finest-level array is
        unnecessarily expensive.

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
    finest_level = int(ds.index.max_level)
    if maximum_level is not None:
        finest_level = min(finest_level, int(maximum_level))
    refine_factor = ds.refine_by ** finest_level
    dims = ds.domain_dimensions * refine_factor
    if ds.dimensionality < 3:
        dims = np.asarray(dims, dtype=int)
        dims[ds.dimensionality:] = 1

    cover = None
    if field_map or derive_native_vorticity:
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
    native_vorticity_needed = bool(derive_native_vorticity)
    derive_native_magnitude = False

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
            if canonical == "vorticity":
                native_vorticity_needed = True
                continue
            if canonical == "vorticity_magnitude":
                native_vorticity_needed = True
                derive_native_magnitude = True
                continue
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

    if native_vorticity_needed and "vorticity" not in fields:
        fields["vorticity"] = _compute_native_amr_vorticity(ds)
        raw_field_names["vorticity"] = (
            "native_amr_dv_dx_minus_du_dy"
        )
        if derive_native_magnitude:
            fields["vorticity_magnitude"] = np.abs(fields["vorticity"])
            raw_field_names["vorticity_magnitude"] = (
                "abs(native_amr_curl)"
            )

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
        "amr_max_level": int(finest_level),
        "amr_refine_by": int(ds.refine_by),
        "grid_shape": (nx, ny),
        "domain_length_x": float(
            (right_edge[0] - left_edge[0]) / (100.0 if convert_to_mks else 1.0)
        ),
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


def read_amrex_plotfile_time(plotfile_path):
    """Read physical time from an AMReX plotfile Header without loading yt."""
    header = os.path.join(os.fspath(plotfile_path), "Header")
    with open(header, encoding="utf-8") as stream:
        stream.readline()  # version string
        n_fields_line = stream.readline()
        if not n_fields_line:
            raise ValueError(f"truncated AMReX Header: {header}")
        n_fields = int(n_fields_line.strip())
        for _ in range(n_fields):
            if not stream.readline():
                raise ValueError(f"truncated AMReX Header: {header}")
        dimensionality = stream.readline()
        time_line = stream.readline()
    if not dimensionality or not time_line:
        raise ValueError(f"truncated AMReX Header: {header}")
    return float(time_line.strip())


def load_pelec_plotfile_series(plot_source, plot_prefix="plt",
                               field_names=None, alias_map=None,
                               convert_to_mks=True,
                               start=None, end=None, step=1,
                               derive_native_vorticity=False):
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
                            convert_to_mks=convert_to_mks,
                            derive_native_vorticity=derive_native_vorticity)
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

    # Vorticity (2-D in-plane).  AMR plotfiles must provide a curl computed
    # on native patches; differentiating their finest-level covering grid
    # creates coarse-cell edge impulses.  Uniform datasets remain safe to
    # derive here.
    if "vorticity" not in f:
        if (
            int(dataset.get("amr_max_level", 0)) == 0
            and "x_velocity" in f
            and "y_velocity" in f
        ):
            dv_dx, _ = np.gradient(f["y_velocity"], dx, dy)
            _, du_dy = np.gradient(f["x_velocity"], dx, dy)
            f["vorticity"] = dv_dx - du_dy

    # Never replace a solver-provided ``magvort`` field, which the loader
    # canonicalizes as ``vorticity_magnitude``.
    if "vorticity_magnitude" not in f and "vorticity" in f:
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


def calculate_compressible_BL_thicknesses(
        wall_distance, tangential_velocity, density,
        edge_velocity=None, edge_density=None):
    """Return density-weighted compressible boundary-layer thicknesses.

    The displacement and momentum thickness definitions are

    ``delta_star = integral(1 - rho*u/(rho_e*U_e)) dn`` and
    ``theta = integral(rho*u/(rho_e*U_e) * (1-u/U_e)) dn``.

    Parameters are dimensional.  The caller is responsible for supplying a
    profile that starts at the physical wall and ends in a valid edge region.
    """
    n = np.asarray(wall_distance, dtype=float).ravel()
    u = np.asarray(tangential_velocity, dtype=float).ravel()
    rho = np.asarray(density, dtype=float).ravel()
    if not (n.size == u.size == rho.size) or n.size < 3:
        raise ValueError("wall distance, velocity, and density need equal length >= 3")
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(u)) or not np.all(
            np.isfinite(rho)):
        raise ValueError("compressible boundary-layer profiles must be finite")
    if np.any(np.diff(n) <= 0.0):
        raise ValueError("wall distance must be strictly increasing")
    if np.any(rho <= 0.0):
        raise ValueError("density must remain positive")

    U_e = float(u[-1] if edge_velocity is None else edge_velocity)
    rho_e = float(rho[-1] if edge_density is None else edge_density)
    if not np.isfinite(U_e) or U_e <= 0.0:
        raise ValueError("edge velocity must be positive and finite")
    if not np.isfinite(rho_e) or rho_e <= 0.0:
        raise ValueError("edge density must be positive and finite")

    u_ratio = u / U_e
    mass_flux_ratio = rho * u / (rho_e * U_e)
    delta_star = float(np.trapz(1.0 - mass_flux_ratio, n))
    theta = float(np.trapz(
        mass_flux_ratio * (1.0 - u_ratio), n
    ))
    shape_factor = delta_star / theta if theta > 0.0 else np.nan

    crossing = np.flatnonzero(u_ratio >= 0.99)
    if crossing.size:
        idx = int(crossing[0])
        if idx == 0:
            delta_99 = float(n[0])
        else:
            u0, u1 = u_ratio[idx - 1], u_ratio[idx]
            fraction = (
                (0.99 - u0) / (u1 - u0)
                if abs(u1 - u0) > np.finfo(float).eps else 1.0
            )
            delta_99 = float(n[idx - 1] + fraction * (n[idx] - n[idx - 1]))
    else:
        delta_99 = np.nan

    return {
        "delta_99": delta_99,
        "delta_star": delta_star,
        "theta": theta,
        "H": shape_factor,
        "edge_velocity": U_e,
        "edge_density": rho_e,
        "u_ratio": u_ratio,
        "mass_flux_ratio": mass_flux_ratio,
    }


def extract_BL_profile_at_surface(dataset, i_surf, j_surf, nx, ny,
                                  u_inf, rho_inf,
                                  max_BL_height=0.010, n_points_BL=200,
                                  mu=None, k_w=None, T_wall=None,
                                  Pr=0.71, Cp=1004.0,
                                  wall_distance_to_cell=None):
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
    s_from_cell = eta_s * max_BL_height
    if wall_distance_to_cell is None:
        # Legacy callers do not provide the detected wall point.  This is
        # exact for the certified y=0 flat plate and only an approximation
        # for arbitrary geometry.
        wall_distance_to_cell = abs(y_s)
    wall_distance_to_cell = float(wall_distance_to_cell)
    if wall_distance_to_cell < 0.0:
        raise ValueError("wall_distance_to_cell must be non-negative")
    s = wall_distance_to_cell + s_from_cell

    x_prof = x_s + s_from_cell * nx
    y_prof_abs = y_s + s_from_cell * ny

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

    # Integration within the detected BL.  Add the physical no-slip wall
    # point so thicknesses are never integrated from a cell centre.
    i_cut = min(i_edge + 5, len(s))
    i_cut = max(i_cut, 10)
    y_bl = np.concatenate([[0.0], s[:i_cut]])
    u_bl = np.concatenate([[0.0], u_tan[:i_cut]])
    rho_wall = float(rho_prof[0])
    rho_bl = np.concatenate([[rho_wall], rho_prof[:i_cut]])
    rho_edge = float(rho_prof[min(i_edge, len(rho_prof) - 1)])
    thickness = calculate_compressible_BL_thicknesses(
        y_bl, u_bl, rho_bl,
        edge_velocity=u_edge, edge_density=rho_edge,
    )
    delta_star = thickness["delta_star"]
    theta = thickness["theta"]
    H = thickness["H"]
    if np.isfinite(thickness["delta_99"]):
        delta_99 = thickness["delta_99"]

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

    # Assemble fit arrays in a true wall-distance coordinate.
    if wall_distance_to_cell > 1e-12:
        s_fit = np.concatenate([[0.0], s[i_start_fit:i_end_fit]])
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
    Re_theta = rho_edge * u_edge * theta / mu_far if theta > 0 else 0.0

    return {
        "y_profile": s,
        "U_Uinf": U_Uinf,
        "u_profile": u_tan,
        "u_edge": u_edge,
        "T_profile": T_prof,
        "rho_profile": rho_prof,
        "rho_edge": rho_edge,
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
                                 Pr=0.71, Cp=1004.0, p_inf=None):
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
            "p": np.full(n, np.nan),
            "temperature": np.full(n, np.nan),
            "rho": np.full(n, np.nan),
            "tau_w": np.full(n, np.nan),
            "C_f": np.full(n, np.nan),
            "C_p": np.full(n, np.nan),
            "q_w": np.full(n, np.nan),
            "Re_x": np.full(n, np.nan),
            "Re_theta": np.full(n, np.nan),
            "delta_99": np.full(n, np.nan),
            "delta_star": np.full(n, np.nan),
            "theta": np.full(n, np.nan),
            "H": np.full(n, np.nan),
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
                    mu=mu, k_w=k_w, T_wall=T_wall, Pr=Pr, Cp=Cp,
                    wall_distance_to_cell=abs(
                        float(surfaces[side]["y"][idx])
                        - float(surfaces[side]["y_interp"][idx])
                    ),
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
        if p_inf is None:
            if T_inf is None:
                raise ValueError(
                    "Explicit p_inf or T_inf is required; wall-adjacent "
                    "temperature is not a freestream reference"
                )
            p_ref = rho_inf * 287.05 * float(T_inf)
        else:
            p_ref = float(p_inf)
        data["C_p"] = (data["p"] - p_ref) / q_inf

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

        # scipy uses fill_value=None to request linear extrapolation.  Match
        # that behaviour in the NumPy fallback rather than silently failing
        # while constructing an array filled with ``None``.
        extrapolate = fill_value is None
        out = np.empty(pts.shape[0], dtype=float)
        if not extrapolate:
            out.fill(fill_value)
        px, py = pts[:, 0], pts[:, 1]

        valid = np.ones(pts.shape[0], dtype=bool) if extrapolate else (
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


def subtract_rectilinear_baseline(target_field, target_x, target_y,
                                  baseline_field, baseline_x, baseline_y,
                                  method="linear", chunk_size=128):
    """Subtract a baseline field after mapping it to the target grid.

    The fields use the framework's canonical ``(x, y)`` storage order.  A
    direct subtraction is used for matching grids; otherwise the baseline is
    interpolated in streamwise chunks so a second target-sized 2-D field is
    never allocated.  Linear extrapolation is limited to the half-cell offset
    encountered when two cell-centred grids cover the same physical domain.

    Returns
    -------
    ndarray
        ``target_field - baseline_on_target_grid`` with the target shape.
    """
    tx = np.asarray(target_x, dtype=float)
    ty = np.asarray(target_y, dtype=float)
    bx = np.asarray(baseline_x, dtype=float)
    by = np.asarray(baseline_y, dtype=float)
    target = np.asarray(target_field, dtype=float)
    baseline = np.asarray(baseline_field, dtype=float)

    for name, coord in (("target_x", tx), ("target_y", ty),
                        ("baseline_x", bx), ("baseline_y", by)):
        if coord.ndim != 1 or coord.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array with at least two points")
        if not np.all(np.isfinite(coord)):
            raise ValueError(f"{name} contains non-finite coordinates")

    if target.shape != (tx.size, ty.size):
        raise ValueError(
            f"Target field shape {target.shape} does not match grid "
            f"({tx.size}, {ty.size})"
        )
    if baseline.shape != (bx.size, by.size):
        raise ValueError(
            f"Baseline field shape {baseline.shape} does not match grid "
            f"({bx.size}, {by.size})"
        )
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")

    # Accept descending loader output while giving the interpolator strictly
    # increasing coordinates.
    if np.any(np.diff(bx) == 0) or np.any(np.diff(by) == 0):
        raise ValueError("Baseline coordinates must be unique")
    if np.any(np.diff(tx) == 0) or np.any(np.diff(ty) == 0):
        raise ValueError("Target coordinates must be unique")
    bx_order = np.argsort(bx)
    by_order = np.argsort(by)
    bx = bx[bx_order]
    by = by[by_order]
    baseline = baseline[np.ix_(bx_order, by_order)]

    same_grid = (
        target.shape == baseline.shape
        and np.allclose(tx, bx, rtol=1.0e-12, atol=1.0e-14)
        and np.allclose(ty, by, rtol=1.0e-12, atol=1.0e-14)
    )
    if same_grid:
        return target - baseline

    interpolator = regular_grid_interpolator(
        bx, by, baseline, method=method, fill_value=None
    )
    result = np.empty_like(target, dtype=float)
    chunk_size = max(1, int(chunk_size))
    for first in range(0, tx.size, chunk_size):
        last = min(first + chunk_size, tx.size)
        xx, yy = np.meshgrid(tx[first:last], ty, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        baseline_chunk = interpolator(points).reshape(last - first, ty.size)
        result[first:last, :] = target[first:last, :] - baseline_chunk
    return result


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


# Cache only the dimensionless BVP solution.  Mapping it onto y at a particular
# x station remains inexpensive and makes a steady reference reusable across
# multiple transient snapshots in one worker process.
_COMPRESSIBLE_SIMILARITY_CACHE = {}


def compute_compressible_similarity_coordinate(
        y, density, x_loc, u_inf, rho_inf, mu_inf, leading_edge_x=0.0):
    r"""Transform a physical CFD profile to the density-weighted coordinate.

    The Howarth--Dorodnitsyn coordinate used by the compressible flat-plate
    reference is

    .. math::

        \eta = \sqrt{\frac{U_e}{2\rho_e\mu_e x}}
               \int_0^y \rho\,\mathrm{d}y.

    ``y`` may start at the first cell centre rather than at the wall.  In that
    case the first density value is extended to ``y=0`` for the short missing
    interval.  Results are returned in the caller's original point order.
    """
    y = np.asarray(y, dtype=float)
    density = np.asarray(density, dtype=float)
    if y.ndim != 1 or density.ndim != 1 or y.size != density.size or y.size == 0:
        raise ValueError("y and density must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(density)):
        raise ValueError("y and density must contain only finite values")
    if np.any(y < 0.0) or np.any(density <= 0.0):
        raise ValueError("y must be nonnegative and density must be positive")

    x_relative = float(x_loc) - float(leading_edge_x)
    for name, value in (("x_relative", x_relative), ("u_inf", u_inf),
                        ("rho_inf", rho_inf), ("mu_inf", mu_inf)):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")

    order = np.argsort(y)
    y_sorted = y[order]
    rho_sorted = density[order]
    if np.any(np.diff(y_sorted) <= 0.0):
        raise ValueError("y must contain unique coordinates")

    mass_integral = np.empty_like(y_sorted)
    mass_integral[0] = rho_sorted[0] * y_sorted[0]
    if y_sorted.size > 1:
        mass_integral[1:] = mass_integral[0] + np.cumsum(
            0.5 * (rho_sorted[1:] + rho_sorted[:-1]) * np.diff(y_sorted)
        )
    eta_sorted = np.sqrt(
        float(u_inf) / (2.0 * float(rho_inf) * float(mu_inf) * x_relative)
    ) * mass_integral
    eta = np.empty_like(eta_sorted)
    eta[order] = eta_sorted
    return eta


def _compressible_similarity_core(M_inf, T_wall_ratio, gamma, Pr,
                                  transport_model, T_inf, mu_inf,
                                  eta_max, n_nodes):
    """Solve the dimensionless Howarth--Dorodnitsyn flat-plate BVP.

    The dependent variables are ``F, F', C F'', theta, C theta'/Pr`` with
    ``theta = T/T_inf`` and ``C = (rho mu)/(rho_inf mu_inf)``.  For a
    calorically perfect gas at zero pressure gradient, ``rho/rho_inf=1/theta``.
    """
    key = (float(M_inf), None if T_wall_ratio is None else float(T_wall_ratio),
           float(gamma), float(Pr), str(transport_model), float(T_inf),
           float(mu_inf), float(eta_max), int(n_nodes))
    cached = _COMPRESSIBLE_SIMILARITY_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from scipy.integrate import solve_bvp
    except ImportError as exc:
        raise ImportError(
            "scipy is required for the compressible similarity reference"
        ) from exc

    eta = np.linspace(0.0, float(eta_max), int(n_nodes))
    recovery_temperature = 1.0 + np.sqrt(Pr) * 0.5 * (gamma - 1.0) * M_inf**2
    theta_wall_guess = (float(T_wall_ratio) if T_wall_ratio is not None
                        else recovery_temperature)

    def coefficient(theta):
        theta_safe = np.maximum(theta, 1.0e-8)
        if transport_model == "constant":
            mu_ratio = np.ones_like(theta_safe)
        else:
            mu_ratio = sutherland_viscosity(theta_safe * T_inf) / mu_inf
        return mu_ratio / theta_safe

    # Smooth initial state satisfying the wall values approximately.  The BVP
    # solver adjusts the unknown wall shear and heat flux to meet edge values.
    fp_guess = 1.0 - np.exp(-eta)
    F_guess = eta - 1.0 + np.exp(-eta)
    theta_guess = 1.0 + (theta_wall_guess - 1.0) * np.exp(-eta)
    C_guess = coefficient(theta_guess)
    fp_prime_guess = np.exp(-eta)
    theta_prime_guess = -(theta_wall_guess - 1.0) * np.exp(-eta)
    initial = np.vstack((
        F_guess,
        fp_guess,
        C_guess * fp_prime_guess,
        theta_guess,
        C_guess * theta_prime_guess / Pr,
    ))

    def ode(_eta, state):
        F, fp, momentum_flux, theta, heat_flux = state
        C = coefficient(theta)
        fpp = momentum_flux / C
        theta_prime = Pr * heat_flux / C
        return np.vstack((
            fp,
            fpp,
            -F * fpp,
            theta_prime,
            -F * theta_prime - (gamma - 1.0) * M_inf**2 * C * fpp**2,
        ))

    if T_wall_ratio is None:
        def boundary_conditions(wall, edge):
            return np.array([
                wall[0], wall[1], wall[4], edge[1] - 1.0, edge[3] - 1.0,
            ])
    else:
        def boundary_conditions(wall, edge):
            return np.array([
                wall[0], wall[1], wall[3] - T_wall_ratio,
                edge[1] - 1.0, edge[3] - 1.0,
            ])

    result = solve_bvp(
        ode, boundary_conditions, eta, initial, tol=2.0e-5,
        max_nodes=max(10000, 20 * int(n_nodes)), verbose=0,
    )
    if not result.success:
        raise RuntimeError(
            "compressible flat-plate similarity solve failed: " + result.message
        )

    # Evaluate on an evenly spaced coordinate for stable plotting/integration.
    eta_out = np.linspace(0.0, float(eta_max), max(800, int(n_nodes)))
    state = result.sol(eta_out)
    theta_out = state[3]
    if np.any(theta_out <= 0) or not np.all(np.isfinite(state)):
        raise RuntimeError("compressible similarity solution has nonphysical state")
    C_out = coefficient(theta_out)
    solution = {
        "eta": eta_out,
        "F": state[0],
        "u_ratio": state[1],
        "momentum_flux": state[2],
        "theta": theta_out,
        "heat_flux": state[4],
        "C": C_out,
    }
    _COMPRESSIBLE_SIMILARITY_CACHE[key] = solution
    return solution


def compute_compressible_flat_plate_reference_profile(
        y, x_loc, u_inf, T_inf, rho_inf, T_wall=293.0, gamma=1.4,
        R=287.05, Cp=1004.0, leading_edge_x=0.0,
        transport_model="constant", mu=8.65e-6, k=0.012415, Pr=0.71,
        eta_max=30.0, n=500):
    """Return an independent compressible laminar flat-plate reference.

    This solves the coupled zero-pressure-gradient Howarth--Dorodnitsyn
    similarity equations for a calorically perfect gas.  Unlike a velocity
    transformation of CFD data, the result is an independent base-flow
    prediction suitable for comparing laminar velocity, temperature, and
    density profiles.

    ``transport_model='constant'`` uses the supplied ``mu`` and ``k`` and
    derives ``Pr = mu*Cp/k``.  ``transport_model='sutherland'`` uses
    Sutherland viscosity and the supplied constant ``Pr`` with
    ``k = mu*Cp/Pr``.  ``T_wall=None`` applies an adiabatic wall condition.

    Returns
    -------
    dict
        ``profiles`` contains plot-compatible ``x_velocity``, ``temperature``
        and ``density`` profiles.  ``scalars`` contains reference boundary
        layer and wall quantities; ``y_similarity`` and ``eta`` expose the
        complete similarity grid for further analysis.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty one-dimensional coordinate")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("y must contain finite wall-normal distances >= 0")

    x_rel = float(x_loc) - float(leading_edge_x)
    if x_rel <= 0:
        raise ValueError("x_loc must lie downstream of leading_edge_x")
    for name, value in (("u_inf", u_inf), ("T_inf", T_inf),
                        ("rho_inf", rho_inf), ("gamma", gamma),
                        ("R", R), ("Cp", Cp)):
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if transport_model not in ("constant", "sutherland"):
        raise ValueError("transport_model must be 'constant' or 'sutherland'")

    if transport_model == "constant":
        if float(mu) <= 0 or float(k) <= 0:
            raise ValueError("constant transport requires positive mu and k")
        mu_inf = float(mu)
        Pr_eff = float(mu) * float(Cp) / float(k)
    else:
        if float(Pr) <= 0:
            raise ValueError("Sutherland transport requires positive Pr")
        mu_inf = float(sutherland_viscosity(float(T_inf)))
        Pr_eff = float(Pr)

    M_inf = float(u_inf) / np.sqrt(float(gamma) * float(R) * float(T_inf))
    wall_ratio = None if T_wall is None else float(T_wall) / float(T_inf)
    if wall_ratio is not None and wall_ratio <= 0:
        raise ValueError("T_wall must be positive when specified")

    core = _compressible_similarity_core(
        M_inf, wall_ratio, float(gamma), Pr_eff, transport_model,
        float(T_inf), mu_inf, float(eta_max), int(n),
    )
    eta = core["eta"]
    theta = core["theta"]
    u_ratio = core["u_ratio"]
    C = core["C"]
    fpp = core["momentum_flux"] / C
    theta_prime = Pr_eff * core["heat_flux"] / C

    # eta = sqrt(Ue/(2 rho_e mu_e x)) int(rho dy).  With p=p_e,
    # rho/rho_e=1/theta, so dy/deta = sqrt(2 mu_e x/(rho_e Ue))*theta.
    y_scale = np.sqrt(2.0 * mu_inf * x_rel / (float(rho_inf) * float(u_inf)))
    y_similarity = np.zeros_like(eta)
    y_similarity[1:] = y_scale * np.cumsum(
        0.5 * (theta[1:] + theta[:-1]) * np.diff(eta)
    )
    temperature = theta * float(T_inf)
    density = float(rho_inf) / theta
    if transport_model == "constant":
        mu_similarity = np.full_like(theta, float(mu))
        k_similarity = np.full_like(theta, float(k))
    else:
        mu_similarity = sutherland_viscosity(temperature)
        k_similarity = mu_similarity * float(Cp) / Pr_eff

    # Include the wall explicitly even when the CFD profile starts at a cell
    # centre, then map the reference onto the requested physical y coordinate.
    y_plot = np.unique(np.concatenate(([0.0], np.sort(y))))
    u_plot = np.interp(y_plot, y_similarity, u_ratio * float(u_inf),
                       left=0.0, right=float(u_inf))
    T_plot = np.interp(y_plot, y_similarity, temperature,
                       left=temperature[0], right=float(T_inf))
    rho_plot = np.interp(y_plot, y_similarity, density,
                         left=density[0], right=float(rho_inf))

    idx99 = np.flatnonzero(u_ratio >= 0.99)
    delta_99 = (float(np.interp(0.99, u_ratio, y_similarity)) if idx99.size
                else float(y_similarity[-1]))
    mass_velocity_ratio = density * u_ratio / float(rho_inf)
    delta_star = float(np.trapz(1.0 - mass_velocity_ratio, y_similarity))
    theta_momentum = float(np.trapz(
        mass_velocity_ratio * (1.0 - u_ratio), y_similarity
    ))
    shape_factor = delta_star / theta_momentum if theta_momentum > 0 else np.nan

    deta_dy_wall = density[0] * np.sqrt(
        float(u_inf) / (2.0 * float(rho_inf) * mu_inf * x_rel)
    )
    tau_w = float(mu_similarity[0] * float(u_inf) * fpp[0] * deta_dy_wall)
    dT_dy_wall = float(T_inf) * theta_prime[0] * deta_dy_wall
    # q_w > 0 is heat from the wall into the fluid; q_w < 0 is wall heating.
    q_w = float(-k_similarity[0] * dT_dy_wall)
    C_f = tau_w / (0.5 * float(rho_inf) * float(u_inf)**2)

    shared = {
        "linestyle": "--",
        "boundary_layer_height": delta_99,
        "reference_model": "compressible_flat_plate_similarity",
    }
    profiles = {
        "x_velocity": {
            "y": y_plot, "values": u_plot,
            "label": "Compressible laminar similarity",
            "field_key": "x_velocity", **shared,
        },
        "temperature": {
            "y": y_plot, "values": T_plot,
            "label": "Compressible laminar similarity",
            "field_key": "temperature", **shared,
        },
        "density": {
            "y": y_plot, "values": rho_plot,
            "label": "Compressible laminar similarity",
            "field_key": "density", **shared,
        },
    }
    return {
        "profiles": profiles,
        "eta": eta,
        "theta": theta,
        "y_similarity": y_similarity,
        "u_ratio": u_ratio,
        "temperature_similarity": temperature,
        "density_similarity": density,
        "mu_similarity": mu_similarity,
        "scalars": {
            "x_relative": x_rel,
            "M_inf": M_inf,
            "Pr": Pr_eff,
            "transport_model": transport_model,
            "delta_99": delta_99,
            "delta_star": delta_star,
            "theta": theta_momentum,
            "H": shape_factor,
            "tau_w": tau_w,
            "C_f": C_f,
            "q_w": q_w,
        },
    }


# ---------------------------------------------------------------------------
# 8.  CERTIFIED FLAT-PLATE WALL AND FORCE CALCULATIONS
# ---------------------------------------------------------------------------

_FLAT_PLATE_FORCE_SCHEMA_VERSION = 1
_WALL_FORCE_HISTORY_MAGIC = b"WFORCE1\0"
_WALL_FORCE_HISTORY_FIELDS = (
    "time_s",
    "D_pressure_cgs",
    "D_viscous_cgs",
    "N_pressure_cgs",
    "N_viscous_cgs",
    "M_pressure_cgs",
    "M_viscous_cgs",
    "coverage_fraction",
    "amr_max_level",
)


def _fit_wall_polynomial(distance, values, order=2, wall_value=None):
    """Fit a scaled wall-normal polynomial and return wall value/gradient."""
    distance = np.asarray(distance, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if distance.size != values.size:
        raise ValueError("distance and values must have equal length")
    if wall_value is not None:
        distance = np.concatenate([[0.0], distance])
        values = np.concatenate([[float(wall_value)], values])
    finite = np.isfinite(distance) & np.isfinite(values)
    distance = distance[finite]
    values = values[finite]
    if distance.size < 2 or np.any(distance < 0.0):
        raise ValueError("wall fit requires at least two non-negative samples")
    if np.any(np.diff(distance) <= 0.0):
        raise ValueError("wall-fit distances must be strictly increasing")
    fit_order = min(int(order), distance.size - 1)
    if fit_order < 1:
        raise ValueError("wall polynomial order must be at least one")
    scale = float(np.max(distance))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("wall-fit distance scale must be positive")
    coordinate = distance / scale
    matrix = np.vander(coordinate, N=fit_order + 1, increasing=True)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    fitted = matrix @ coefficients
    residual_rms = float(np.sqrt(np.mean((values - fitted) ** 2)))
    return {
        "wall_value": float(coefficients[0]),
        "wall_gradient": float(coefficients[1] / scale),
        "order": fit_order,
        "residual_rms": residual_rms,
        "condition_number": float(np.linalg.cond(matrix)),
        "coefficients_scaled": coefficients,
        "distance_scale": scale,
    }


def reconstruct_flat_plate_wall_stencils(
        x_left_m, x_right_m, wall_distance_m, pressure_pa,
        tangential_velocity_m_s, viscosity_pa_s,
        p_inf_pa, rho_inf_kg_m3, u_inf_m_s,
        pressure_order=2, velocity_order=2, amr_level=None,
        source_cell=None):
    """Reconstruct flat-plate wall pressure and shear from native stencils.

    This pure-numpy layer is intentionally independent of yt so manufactured
    tests exercise the same reconstruction used by production plotfiles.
    """
    x_left = np.asarray(x_left_m, dtype=float).ravel()
    x_right = np.asarray(x_right_m, dtype=float).ravel()
    distance = np.asarray(wall_distance_m, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    velocity = np.asarray(tangential_velocity_m_s, dtype=float)
    n_faces = x_left.size
    if x_right.size != n_faces:
        raise ValueError("x_left and x_right must have equal length")
    if distance.ndim == 1:
        distance = np.broadcast_to(distance[None, :], (n_faces, distance.size))
    if pressure.shape != distance.shape or velocity.shape != distance.shape:
        raise ValueError("wall stencil arrays must share shape (n_face, n_point)")
    viscosity = np.asarray(viscosity_pa_s, dtype=float)
    if viscosity.ndim == 0:
        viscosity = np.full(n_faces, float(viscosity))
    elif viscosity.ndim == 2:
        viscosity = viscosity[:, 0]
    viscosity = viscosity.ravel()
    if viscosity.size != n_faces:
        raise ValueError("viscosity must be scalar or one value per wall face")
    if np.any(x_right <= x_left):
        raise ValueError("wall-face intervals must have positive width")

    p_wall = np.full(n_faces, np.nan)
    p_wall_linear = np.full(n_faces, np.nan)
    tau_wall = np.full(n_faces, np.nan)
    p_residual = np.full(n_faces, np.nan)
    u_residual = np.full(n_faces, np.nan)
    p_condition = np.full(n_faces, np.nan)
    u_condition = np.full(n_faces, np.nan)
    p_order_used = np.zeros(n_faces, dtype=int)
    u_order_used = np.zeros(n_faces, dtype=int)
    valid = np.ones(n_faces, dtype=bool)
    failure_code = np.zeros(n_faces, dtype=np.int16)
    failure_reason = np.full(n_faces, "", dtype="<U160")

    for index in range(n_faces):
        try:
            p_fit = _fit_wall_polynomial(
                distance[index], pressure[index], order=pressure_order
            )
            p_linear = _fit_wall_polynomial(
                distance[index], pressure[index], order=1
            )
            u_fit = _fit_wall_polynomial(
                distance[index], velocity[index],
                order=velocity_order, wall_value=0.0,
            )
            if not np.isfinite(viscosity[index]) or viscosity[index] <= 0.0:
                raise ValueError("wall viscosity is not positive and finite")
            p_wall[index] = p_fit["wall_value"]
            p_wall_linear[index] = p_linear["wall_value"]
            tau_wall[index] = viscosity[index] * u_fit["wall_gradient"]
            p_residual[index] = p_fit["residual_rms"]
            u_residual[index] = u_fit["residual_rms"]
            p_condition[index] = p_fit["condition_number"]
            u_condition[index] = u_fit["condition_number"]
            p_order_used[index] = p_fit["order"]
            u_order_used[index] = u_fit["order"]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            valid[index] = False
            failure_code[index] = 1
            failure_reason[index] = str(exc)

    q_inf = 0.5 * float(rho_inf_kg_m3) * float(u_inf_m_s) ** 2
    if not np.isfinite(q_inf) or q_inf <= 0.0:
        raise ValueError("freestream dynamic pressure must be positive")
    if amr_level is None:
        amr_level = np.zeros(n_faces, dtype=int)
    if source_cell is None:
        source_cell = np.arange(n_faces, dtype=np.int64)
    return {
        "schema_version": _FLAT_PLATE_FORCE_SCHEMA_VERSION,
        "x_left_m": x_left,
        "x_right_m": x_right,
        "x_center_m": 0.5 * (x_left + x_right),
        "wall_y_m": np.zeros(n_faces),
        "p_wall_pa": p_wall,
        "p_wall_linear_pa": p_wall_linear,
        "p_gauge_pa": p_wall - float(p_inf_pa),
        "tau_wall_pa": tau_wall,
        "viscosity_wall_pa_s": viscosity,
        "C_p": (p_wall - float(p_inf_pa)) / q_inf,
        "C_f": tau_wall / q_inf,
        "amr_level": np.asarray(amr_level, dtype=int),
        "source_cell": np.asarray(source_cell, dtype=np.int64),
        "pressure_fit_order": p_order_used,
        "velocity_fit_order": u_order_used,
        "pressure_fit_residual_pa": p_residual,
        "velocity_fit_residual_m_s": u_residual,
        "pressure_fit_condition": p_condition,
        "velocity_fit_condition": u_condition,
        "valid": valid,
        "failure_code": failure_code,
        "failure_reason": failure_reason,
        "p_inf_pa": float(p_inf_pa),
        "q_inf_pa": q_inf,
        "certified_scope": "stationary_one_sided_flat_plate",
    }


def _find_boxlib_field(ds, canonical, required=True):
    """Resolve one canonical field against a yt BoxLib/AMReX dataset."""
    available = {name for _, name in ds.field_list}
    available_lower = {name.lower(): name for name in available}
    candidates = _resolve_field_aliases([canonical]).get(canonical, [canonical])
    if canonical not in candidates:
        candidates = [canonical, *candidates]
    for candidate in candidates:
        if candidate in available:
            return ("boxlib", candidate)
        match = available_lower.get(str(candidate).lower())
        if match is not None:
            return ("boxlib", match)
    if required:
        raise KeyError(
            f"Required field '{canonical}' not found; "
            f"available fields include {sorted(available)[:20]}"
        )
    return None


def extract_native_flat_plate_wall(
        plotfile_path, x_range_m=(0.0, 0.4), wall_y_m=0.0,
        p_inf_pa=760.0, rho_inf_kg_m3=0.021180978923532486,
        u_inf_m_s=1726.0, viscosity_pa_s=8.65e-6,
        pressure_order=2, velocity_order=2, fluid_points=4,
        viscosity_relative_tolerance=1.0e-4):
    """Extract a certified wall-only snapshot directly from native AMR grids.

    Coarse wall cells covered by finer AMR levels are replaced by the fine
    cells.  Wall-normal fits always use values at the cell's native level;
    the finest-level covering-grid repetition is never differentiated.
    """
    try:
        import yt
    except ImportError as exc:
        raise ImportError("yt is required for native AMR wall extraction") from exc

    fluid_points = int(fluid_points)
    if fluid_points < 2:
        raise ValueError("fluid_points must be at least two")
    x_start, x_end = [float(value) for value in x_range_m]
    if not x_end > x_start:
        raise ValueError("x_range_m must have increasing endpoints")
    ds = yt.load(os.fspath(plotfile_path))
    if int(ds.dimensionality) != 2:
        raise ValueError("certified flat-plate force extraction requires 2-D data")

    unit_scale = 0.01  # PeleC case coordinates are centimetres.
    domain_left = np.asarray(ds.domain_left_edge.d, dtype=float)
    domain_right = np.asarray(ds.domain_right_edge.d, dtype=float)
    domain_wall_m = float(domain_left[1] * unit_scale)
    finest_level = int(ds.index.max_level)
    refine_by = int(ds.refine_by)
    finest_dims = (
        np.asarray(ds.domain_dimensions, dtype=int)
        * refine_by ** finest_level
    )
    finest_dx_m = (
        float(domain_right[0] - domain_left[0])
        / float(finest_dims[0]) * unit_scale
    )
    if abs(domain_wall_m - float(wall_y_m)) > max(1.0e-12, finest_dx_m):
        raise ValueError(
            "certified flat plate must coincide with the lower domain boundary"
        )
    if x_start < domain_left[0] * unit_scale - 1.0e-12 or (
            x_end > domain_right[0] * unit_scale + 1.0e-12):
        raise ValueError("requested force interval lies outside the plotfile domain")

    pressure_field = _find_boxlib_field(ds, "pressure")
    velocity_field = _find_boxlib_field(ds, "x_velocity")
    transverse_field = _find_boxlib_field(ds, "y_velocity", required=False)
    viscosity_field = _find_boxlib_field(ds, "viscosity", required=False)

    # Finest-index slots are overwritten from coarse to fine.  Each value
    # points back to one native source cell, which is consolidated below.
    finest_slots = {}
    source_records = {}
    source_counter = 0
    grids = sorted(ds.index.grids, key=lambda grid: int(grid.Level))
    for grid in grids:
        level = int(grid.Level)
        start = np.asarray(grid.get_global_startindex(), dtype=int)
        active = np.asarray(grid.ActiveDimensions, dtype=int)
        if start[1] != 0 or active[1] < fluid_points:
            continue
        geometry = grid
        dx_cgs = np.asarray(grid.dds.d, dtype=float)
        refinement_to_finest = refine_by ** (finest_level - level)

        pressure_values = np.asarray(grid[pressure_field], dtype=float).squeeze()
        velocity_values = np.asarray(grid[velocity_field], dtype=float).squeeze()
        if pressure_values.ndim != 2 or velocity_values.ndim != 2:
            raise ValueError("native AMR wall fields must be two-dimensional")
        transverse_values = (
            None if transverse_field is None
            else np.asarray(grid[transverse_field], dtype=float).squeeze()
        )
        viscosity_values = (
            None if viscosity_field is None
            else np.asarray(grid[viscosity_field], dtype=float).squeeze()
        )
        if pressure_values.shape[1] < fluid_points:
            continue

        distance_m = (
            (np.arange(fluid_points, dtype=float) + 0.5)
            * dx_cgs[1] * unit_scale
        )
        for local_i in range(int(active[0])):
            level_i = int(start[0] + local_i)
            native_left_m = (
                domain_left[0] + level_i * dx_cgs[0]
            ) * unit_scale
            native_right_m = native_left_m + dx_cgs[0] * unit_scale
            if native_right_m <= x_start or native_left_m >= x_end:
                continue

            pressure_stencil = (
                pressure_values[local_i, :fluid_points] * 0.1
            )
            velocity_stencil = (
                velocity_values[local_i, :fluid_points] * 0.01
            )
            if transverse_values is not None:
                transverse_stencil = (
                    transverse_values[local_i, :fluid_points] * 0.01
                )
                if not np.all(np.isfinite(transverse_stencil)):
                    continue
            if viscosity_values is None:
                mu_wall = float(viscosity_pa_s)
            else:
                mu_stencil = (
                    np.asarray(viscosity_values[local_i, :fluid_points],
                               dtype=float) * 0.1
                )
                mu_wall = float(np.mean(mu_stencil))
                relative_error = abs(mu_wall - float(viscosity_pa_s)) / max(
                    abs(float(viscosity_pa_s)), np.finfo(float).tiny
                )
                if relative_error > float(viscosity_relative_tolerance):
                    raise ValueError(
                        f"plotfile viscosity {mu_wall:.9g} Pa.s differs from "
                        f"configured constant {float(viscosity_pa_s):.9g} Pa.s"
                    )

            source_counter += 1
            source_records[source_counter] = {
                "distance_m": distance_m.copy(),
                "pressure_pa": pressure_stencil,
                "velocity_m_s": velocity_stencil,
                "viscosity_pa_s": mu_wall,
                "level": level,
                "native_i": level_i,
            }
            finest_start = level_i * refinement_to_finest
            for offset in range(refinement_to_finest):
                finest_slots[finest_start + offset] = source_counter

    selected = []
    for finest_i, source_id in sorted(finest_slots.items()):
        left = (
            domain_left[0] * unit_scale + finest_i * finest_dx_m
        )
        right = left + finest_dx_m
        if right > x_start and left < x_end:
            selected.append((
                finest_i, source_id, max(left, x_start), min(right, x_end)
            ))
    if not selected:
        raise ValueError("no native AMR wall cells cover the requested plate interval")

    consolidated = []
    for finest_i, source_id, left, right in selected:
        if (
            consolidated
            and source_id == consolidated[-1]["source_id"]
            and finest_i == consolidated[-1]["last_finest_i"] + 1
            and abs(left - consolidated[-1]["right"]) <= 1.0e-12
        ):
            consolidated[-1]["right"] = right
            consolidated[-1]["last_finest_i"] = finest_i
        else:
            consolidated.append({
                "source_id": source_id,
                "first_finest_i": finest_i,
                "last_finest_i": finest_i,
                "left": left,
                "right": right,
            })

    x_left = np.array([item["left"] for item in consolidated], dtype=float)
    x_right = np.array([item["right"] for item in consolidated], dtype=float)
    records = [source_records[item["source_id"]] for item in consolidated]
    distance = np.vstack([item["distance_m"] for item in records])
    pressure = np.vstack([item["pressure_pa"] for item in records])
    velocity = np.vstack([item["velocity_m_s"] for item in records])
    viscosity = np.array([item["viscosity_pa_s"] for item in records])
    levels = np.array([item["level"] for item in records], dtype=int)
    source_cells = np.array(
        [item["native_i"] for item in records], dtype=np.int64
    )
    wall = reconstruct_flat_plate_wall_stencils(
        x_left, x_right, distance, pressure, velocity, viscosity,
        p_inf_pa=p_inf_pa,
        rho_inf_kg_m3=rho_inf_kg_m3,
        u_inf_m_s=u_inf_m_s,
        pressure_order=pressure_order,
        velocity_order=velocity_order,
        amr_level=levels,
        source_cell=source_cells,
    )
    wall.update({
        "source": os.fspath(plotfile_path),
        "plot_label": os.path.basename(os.fspath(plotfile_path)),
        "time_s": float(ds.current_time),
        "wall_y_m": np.full(len(x_left), float(wall_y_m)),
        "requested_x_range_m": np.array([x_start, x_end], dtype=float),
        "amr_max_level": finest_level,
        "amr_refine_by": refine_by,
        "finest_dx_m": finest_dx_m,
    })
    return wall


def extract_native_flat_plate_boundary_layers(
        plotfile_path, stations_m, maximum_height_m=0.01,
        wall_y_m=0.0, wall_temperature_k=293.0,
        viscosity_pa_s=8.65e-6, conductivity_w_m_k=0.012415,
        fluid_points=4):
    """Extract density-weighted BL profiles from native AMR cells.

    Each requested x station uses the finest available AMR cell for every
    wall-normal interval.  Coarse cells covered by finer patches are removed.
    """
    try:
        import yt
    except ImportError as exc:
        raise ImportError("yt is required for native AMR BL extraction") from exc

    ds = yt.load(os.fspath(plotfile_path))
    if int(ds.dimensionality) != 2:
        raise ValueError("flat-plate BL extraction requires a 2-D plotfile")
    stations = np.asarray(stations_m, dtype=float).ravel()
    if stations.size == 0:
        return []
    maximum_height_m = float(maximum_height_m)
    if maximum_height_m <= 0.0:
        raise ValueError("maximum_height_m must be positive")
    fluid_points = max(2, int(fluid_points))

    field_specs = {
        "u": (_find_boxlib_field(ds, "x_velocity"), 0.01),
        "rho": (_find_boxlib_field(ds, "density"), 1.0e3),
        "temperature": (_find_boxlib_field(ds, "temperature"), 1.0),
        "pressure": (_find_boxlib_field(ds, "pressure"), 0.1),
    }
    viscosity_field = _find_boxlib_field(ds, "viscosity", required=False)
    domain_left = np.asarray(ds.domain_left_edge.d, dtype=float)
    domain_right = np.asarray(ds.domain_right_edge.d, dtype=float)
    unit_scale = 0.01
    finest_level = int(ds.index.max_level)
    refine_by = int(ds.refine_by)
    finest_dims = (
        np.asarray(ds.domain_dimensions, dtype=int)
        * refine_by ** finest_level
    )
    finest_dy_m = (
        (domain_right[1] - domain_left[1]) / finest_dims[1] * unit_scale
    )
    if abs(domain_left[1] * unit_scale - float(wall_y_m)) > max(
            1.0e-12, finest_dy_m):
        raise ValueError("wall_y_m does not match the lower domain boundary")

    grids = sorted(ds.index.grids, key=lambda grid: int(grid.Level))
    results = []
    for station_m in stations:
        station_cgs = station_m / unit_scale
        finest_slots = {}
        records = {}
        source_counter = 0
        for grid in grids:
            level = int(grid.Level)
            start = np.asarray(grid.get_global_startindex(), dtype=int)
            active = np.asarray(grid.ActiveDimensions, dtype=int)
            dx_cgs = np.asarray(grid.dds.d, dtype=float)
            global_i = int(np.floor(
                (station_cgs - domain_left[0]) / dx_cgs[0]
            ))
            if global_i < start[0] or global_i >= start[0] + active[0]:
                continue
            local_i = global_i - int(start[0])
            refinement_to_finest = refine_by ** (finest_level - level)
            arrays = {
                key: np.asarray(grid[field], dtype=float).squeeze() * factor
                for key, (field, factor) in field_specs.items()
            }
            mu_array = (
                None if viscosity_field is None
                else np.asarray(grid[viscosity_field], dtype=float).squeeze() * 0.1
            )
            for local_j in range(int(active[1])):
                global_j = int(start[1] + local_j)
                lower_m = (
                    domain_left[1] + global_j * dx_cgs[1]
                ) * unit_scale
                upper_m = lower_m + dx_cgs[1] * unit_scale
                distance_lower = lower_m - float(wall_y_m)
                distance_upper = upper_m - float(wall_y_m)
                if distance_upper <= 0.0 or distance_lower >= maximum_height_m:
                    continue
                source_counter += 1
                records[source_counter] = {
                    "lower_m": max(distance_lower, 0.0),
                    "upper_m": min(distance_upper, maximum_height_m),
                    "level": level,
                    "global_j": global_j,
                    **{
                        key: float(value[local_i, local_j])
                        for key, value in arrays.items()
                    },
                    "mu": (
                        float(viscosity_pa_s) if mu_array is None
                        else float(mu_array[local_i, local_j])
                    ),
                }
                finest_start = global_j * refinement_to_finest
                for offset in range(refinement_to_finest):
                    finest_slots[finest_start + offset] = source_counter

        selected = []
        for finest_j, source_id in sorted(finest_slots.items()):
            lower = finest_j * finest_dy_m
            upper = lower + finest_dy_m
            if upper > 0.0 and lower < maximum_height_m:
                selected.append((finest_j, source_id))
        consolidated = []
        for finest_j, source_id in selected:
            if (
                consolidated
                and consolidated[-1]["source_id"] == source_id
                and consolidated[-1]["last"] + 1 == finest_j
            ):
                consolidated[-1]["last"] = finest_j
            else:
                consolidated.append({
                    "source_id": source_id,
                    "first": finest_j,
                    "last": finest_j,
                })
        native = [records[item["source_id"]] for item in consolidated]
        if len(native) < max(8, fluid_points):
            raise ValueError(
                f"too few native BL cells at x={station_m:.6g} m"
            )
        distance = np.array([
            0.5 * (item["lower_m"] + item["upper_m"]) for item in native
        ])
        u = np.array([item["u"] for item in native])
        rho = np.array([item["rho"] for item in native])
        temperature = np.array([item["temperature"] for item in native])
        pressure = np.array([item["pressure"] for item in native])
        mu = np.array([item["mu"] for item in native])
        levels = np.array([item["level"] for item in native], dtype=int)

        if np.any(mu <= 0.0) or not np.all(np.isfinite(mu)):
            raise ValueError(f"invalid viscosity profile at x={station_m:.6g} m")
        configured_mu = float(viscosity_pa_s)
        if np.max(np.abs(mu - configured_mu)) > 1.0e-4 * configured_mu:
            raise ValueError(
                f"BL viscosity at x={station_m:.6g} m disagrees with "
                "configured constant transport"
            )

        tail_count = max(5, int(np.ceil(0.1 * len(u))))
        edge_velocity = float(np.median(u[-tail_count:]))
        edge_density = float(np.median(rho[-tail_count:]))
        if edge_velocity <= 0.0 or edge_density <= 0.0:
            raise ValueError(f"invalid BL edge state at x={station_m:.6g} m")
        velocity_ratio = u / edge_velocity
        persistent = min(5, max(2, len(u) // 20))
        edge_index = None
        for index in range(len(u) - persistent + 1):
            if np.all(velocity_ratio[index:index + persistent] >= 0.985):
                edge_index = index
                break
        if edge_index is None or not np.any(velocity_ratio >= 0.99):
            raise ValueError(
                f"BL edge not found before {maximum_height_m:g} m "
                f"at x={station_m:.6g} m"
            )

        rho_wall_fit = _fit_wall_polynomial(
            distance[:fluid_points], rho[:fluid_points], order=2
        )
        pressure_wall_fit = _fit_wall_polynomial(
            distance[:fluid_points], pressure[:fluid_points], order=2
        )
        u_wall_fit = _fit_wall_polynomial(
            distance[:fluid_points], u[:fluid_points],
            order=2, wall_value=0.0,
        )
        temperature_wall_fit = _fit_wall_polynomial(
            distance[:fluid_points], temperature[:fluid_points],
            order=2, wall_value=float(wall_temperature_k),
        )
        wall_density = max(rho_wall_fit["wall_value"], np.finfo(float).tiny)
        n_profile = np.concatenate([[0.0], distance])
        u_profile = np.concatenate([[0.0], u])
        rho_profile = np.concatenate([[wall_density], rho])
        temperature_profile = np.concatenate([
            [float(wall_temperature_k)], temperature
        ])
        pressure_profile = np.concatenate([
            [pressure_wall_fit["wall_value"]], pressure
        ])
        thickness = calculate_compressible_BL_thicknesses(
            n_profile, u_profile, rho_profile,
            edge_velocity=edge_velocity, edge_density=edge_density,
        )
        tau_wall = configured_mu * u_wall_fit["wall_gradient"]
        q_wall = -float(conductivity_w_m_k) * temperature_wall_fit[
            "wall_gradient"
        ]
        mu_edge = float(np.median(mu[-tail_count:]))
        Re_theta = (
            edge_density * edge_velocity * thickness["theta"] / mu_edge
        )
        results.append({
            "schema_version": _FLAT_PLATE_FORCE_SCHEMA_VERSION,
            "source": os.fspath(plotfile_path),
            "time_s": float(ds.current_time),
            "x_station_m": float(station_m),
            "wall_distance_m": n_profile,
            "u_t_m_s": u_profile,
            "rho_kg_m3": rho_profile,
            "temperature_k": temperature_profile,
            "pressure_pa": pressure_profile,
            "mu_pa_s": np.concatenate([[configured_mu], mu]),
            "amr_level": np.concatenate([[-1], levels]),
            "edge_index": int(edge_index + 1),
            "edge_velocity_m_s": edge_velocity,
            "edge_density_kg_m3": edge_density,
            "delta_99_m": thickness["delta_99"],
            "delta_star_m": thickness["delta_star"],
            "theta_m": thickness["theta"],
            "H": thickness["H"],
            "tau_wall_pa": tau_wall,
            "C_f": tau_wall / (
                0.5 * edge_density * edge_velocity ** 2
            ),
            "q_wall_w_m2": q_wall,
            "Re_theta": Re_theta,
            "fit_condition_velocity": u_wall_fit["condition_number"],
            "fit_condition_pressure": pressure_wall_fit["condition_number"],
            "valid": True,
        })
    return results


def integrate_flat_plate_wall_forces(
        wall_data, reference, minimum_coverage=0.995,
        maximum_gap_widths=2.0):
    """Integrate certified one-sided flat-plate loads per unit span."""
    x_left = np.asarray(wall_data["x_left_m"], dtype=float)
    x_right = np.asarray(wall_data["x_right_m"], dtype=float)
    order = np.argsort(x_left)
    x_left, x_right = x_left[order], x_right[order]
    p_wall = np.asarray(wall_data["p_wall_pa"], dtype=float)[order]
    tau_wall = np.asarray(wall_data["tau_wall_pa"], dtype=float)[order]
    valid = np.asarray(
        wall_data.get("valid", np.ones(len(order), dtype=bool)), dtype=bool
    )[order]
    x_range = np.asarray(
        wall_data.get(
            "requested_x_range_m", [float(x_left[0]), float(x_right[-1])]
        ),
        dtype=float,
    )
    expected_length = float(x_range[1] - x_range[0])
    widths = x_right - x_left
    tolerance = max(1.0e-12, expected_length * 1.0e-12)
    if np.any(widths <= 0.0):
        raise ValueError("wall data contain non-positive face widths")
    overlap = x_right[:-1] - x_left[1:]
    if np.any(overlap > tolerance):
        raise ValueError("wall-face intervals overlap")
    gaps = np.maximum(x_left[1:] - x_right[:-1], 0.0)
    boundary_gaps = np.array([
        max(x_left[0] - x_range[0], 0.0),
        max(x_range[1] - x_right[-1], 0.0),
    ])
    total_gap = float(np.sum(gaps) + np.sum(boundary_gaps))
    coverage = float((np.sum(widths) - np.sum(np.maximum(overlap, 0.0)))
                     / expected_length)
    local_width = float(np.median(widths))
    max_gap = float(max(
        np.max(gaps) if gaps.size else 0.0,
        np.max(boundary_gaps),
    ))
    if coverage < float(minimum_coverage):
        raise ValueError(
            f"wall coverage {coverage:.6f} is below {minimum_coverage:.6f}"
        )
    if max_gap > float(maximum_gap_widths) * local_width + tolerance:
        raise ValueError("wall data contain a gap wider than the allowed limit")
    if not np.all(valid) or not np.all(np.isfinite(p_wall)) or not np.all(
            np.isfinite(tau_wall)):
        raise ValueError("wall data contain invalid pressure or shear values")

    rho_inf = float(reference["rho_inf"])
    u_inf = float(reference["u_inf"])
    p_inf = float(reference["p_inf"])
    chord = float(reference["chord"])
    moment_origin = np.asarray(reference.get(
        "moment_origin", [0.25 * chord, 0.0]
    ), dtype=float)
    if moment_origin.shape != (2,):
        raise ValueError("moment_origin must contain [x, y]")
    q_inf = 0.5 * rho_inf * u_inf ** 2
    if q_inf <= 0.0 or chord <= 0.0:
        raise ValueError("reference dynamic pressure and chord must be positive")

    x_center = 0.5 * (x_left + x_right)
    wall_y = np.asarray(
        wall_data.get("wall_y_m", np.zeros(len(order))), dtype=float
    )
    if wall_y.ndim == 0:
        wall_y = np.full(len(order), float(wall_y))
    else:
        wall_y = wall_y[order]
    pressure_gauge = p_wall - p_inf

    # Flat-plate body-to-fluid normal=(0,+1), tangent=(+1,0).
    dD_p_dx = np.zeros_like(pressure_gauge)
    dD_v_dx = tau_wall
    dN_p_dx = -pressure_gauge
    dN_v_dx = np.zeros_like(tau_wall)
    dM_p_dx = (
        (x_center - moment_origin[0]) * dN_p_dx
        - (wall_y - moment_origin[1]) * dD_p_dx
    )
    dM_v_dx = (
        (x_center - moment_origin[0]) * dN_v_dx
        - (wall_y - moment_origin[1]) * dD_v_dx
    )

    def integrate(density):
        return float(np.sum(density * widths))

    D_p = integrate(dD_p_dx)
    D_v = integrate(dD_v_dx)
    N_p = integrate(dN_p_dx)
    N_v = integrate(dN_v_dx)
    M_p = integrate(dM_p_dx)
    M_v = integrate(dM_v_dx)
    D_total, N_total, M_total = D_p + D_v, N_p + N_v, M_p + M_v
    denominator = q_inf * chord
    moment_denominator = q_inf * chord ** 2

    face_D = (dD_p_dx + dD_v_dx) * widths
    face_N = (dN_p_dx + dN_v_dx) * widths
    face_M = (dM_p_dx + dM_v_dx) * widths
    result = {
        "schema_version": _FLAT_PLATE_FORCE_SCHEMA_VERSION,
        "certified_scope": "stationary_one_sided_flat_plate",
        "one_sided_load": True,
        "source": wall_data.get("source"),
        "plot_label": wall_data.get("plot_label"),
        "time": float(wall_data.get("time_s", np.nan)),
        "x_left_m": x_left,
        "x_right_m": x_right,
        "x_center_m": x_center,
        "dD_pressure_dx_N_m2": dD_p_dx,
        "dD_viscous_dx_N_m2": dD_v_dx,
        "dD_total_dx_N_m2": dD_p_dx + dD_v_dx,
        "dN_pressure_dx_N_m2": dN_p_dx,
        "dN_viscous_dx_N_m2": dN_v_dx,
        "dN_total_dx_N_m2": dN_p_dx + dN_v_dx,
        "dM_pressure_dx_N_m": dM_p_dx,
        "dM_viscous_dx_N_m": dM_v_dx,
        "dM_total_dx_N_m": dM_p_dx + dM_v_dx,
        "D_pressure_N_m": D_p,
        "D_viscous_N_m": D_v,
        "D_total_N_m": D_total,
        "N_pressure_N_m": N_p,
        "N_viscous_N_m": N_v,
        "N_total_N_m": N_total,
        "M_pressure_N": M_p,
        "M_viscous_N": M_v,
        "M_total_N": M_total,
        "C_D_pressure": D_p / denominator,
        "C_D_viscous": D_v / denominator,
        "C_D": D_total / denominator,
        "C_N_pressure_one_sided": N_p / denominator,
        "C_N_viscous_one_sided": N_v / denominator,
        "C_N_one_sided": N_total / denominator,
        "C_L_one_sided": N_total / denominator,
        "C_M_pressure": M_p / moment_denominator,
        "C_M_viscous": M_v / moment_denominator,
        "C_M": M_total / moment_denominator,
        "D_cumulative_N_m": np.cumsum(face_D),
        "N_cumulative_N_m": np.cumsum(face_N),
        "M_cumulative_N": np.cumsum(face_M),
        "coverage_fraction": coverage,
        "maximum_gap_m": max_gap,
        "total_gap_m": total_gap,
        "q_inf_pa": q_inf,
        "rho_inf_kg_m3": rho_inf,
        "u_inf_m_s": u_inf,
        "p_inf_pa": p_inf,
        "chord_m": chord,
        "moment_origin_m": moment_origin,
        # Compatibility keys for existing plotting/export code.
        "D_p": D_p,
        "D_v": D_v,
        "D_total": D_total,
        "L_p": N_p,
        "L_v": N_v,
        "L_total": N_total,
        "C_Dp": D_p / denominator,
        "C_Dv": D_v / denominator,
        "C_Lp": N_p / denominator,
        "C_Lv": N_v / denominator,
        "C_L": N_total / denominator,
        "A_ref": chord,
        "chord_length": chord,
        "q_inf": q_inf,
        "rho_inf": rho_inf,
        "u_inf": u_inf,
    }
    if not np.isclose(result["D_total_N_m"], D_p + D_v, rtol=1e-12, atol=1e-12):
        raise RuntimeError("drag component consistency check failed")
    if not np.isclose(result["N_total_N_m"], N_p + N_v, rtol=1e-12, atol=1e-12):
        raise RuntimeError("normal-force component consistency check failed")
    if not np.isclose(result["M_total_N"], M_p + M_v, rtol=1e-12, atol=1e-12):
        raise RuntimeError("moment component consistency check failed")
    for total_key, pressure_key, viscous_key in (
            ("C_D", "C_D_pressure", "C_D_viscous"),
            (
                "C_N_one_sided", "C_N_pressure_one_sided",
                "C_N_viscous_one_sided",
            ),
            ("C_M", "C_M_pressure", "C_M_viscous")):
        if not np.isclose(
                result[total_key],
                result[pressure_key] + result[viscous_key],
                rtol=1.0e-12, atol=1.0e-14):
            raise RuntimeError(
                f"{total_key} component consistency check failed"
            )
    return result


def difference_flat_plate_wall_surfaces(current, baseline):
    """Conservatively subtract two piecewise-constant wall distributions."""
    current_left = np.asarray(current["x_left_m"], dtype=float)
    current_right = np.asarray(current["x_right_m"], dtype=float)
    baseline_left = np.asarray(baseline["x_left_m"], dtype=float)
    baseline_right = np.asarray(baseline["x_right_m"], dtype=float)
    left = max(float(current_left[0]), float(baseline_left[0]))
    right = min(float(current_right[-1]), float(baseline_right[-1]))
    if right <= left:
        raise ValueError("current and baseline wall intervals do not overlap")

    raw_edges = np.concatenate([
        current_left, current_right, baseline_left, baseline_right, [left, right]
    ])
    raw_edges = np.sort(raw_edges[(raw_edges >= left) & (raw_edges <= right)])
    merged = []
    tolerance = max(1.0e-12, (right - left) * 1.0e-12)
    for edge in raw_edges:
        if not merged or edge - merged[-1] > tolerance:
            merged.append(float(edge))
        else:
            merged[-1] = 0.5 * (merged[-1] + float(edge))
    edges = np.asarray(merged, dtype=float)
    interval_left, interval_right = edges[:-1], edges[1:]
    keep = interval_right - interval_left > tolerance
    interval_left, interval_right = interval_left[keep], interval_right[keep]
    centers = 0.5 * (interval_left + interval_right)

    def sample_piecewise(surface, key):
        starts = np.asarray(surface["x_left_m"], dtype=float)
        ends = np.asarray(surface["x_right_m"], dtype=float)
        values = np.asarray(surface[key], dtype=float)
        indices = np.searchsorted(starts, centers, side="right") - 1
        if np.any(indices < 0) or np.any(centers >= ends[indices] + tolerance):
            raise ValueError(f"wall distribution '{key}' has an uncovered interval")
        return values[indices], indices

    p_current, current_indices = sample_piecewise(current, "p_wall_pa")
    p_baseline, baseline_indices = sample_piecewise(baseline, "p_wall_pa")
    tau_current, _ = sample_piecewise(current, "tau_wall_pa")
    tau_baseline, _ = sample_piecewise(baseline, "tau_wall_pa")
    delta_p = p_current - p_baseline
    delta_tau = tau_current - tau_baseline
    q_inf = float(current.get("q_inf_pa", baseline.get("q_inf_pa", 1.0)))
    result = {
        "schema_version": _FLAT_PLATE_FORCE_SCHEMA_VERSION,
        "certified_scope": "stationary_one_sided_flat_plate_increment",
        "is_increment": True,
        "source": current.get("source"),
        "baseline_source": baseline.get("source"),
        "plot_label": current.get("plot_label"),
        "time_s": float(current.get("time_s", np.nan)),
        "x_left_m": interval_left,
        "x_right_m": interval_right,
        "x_center_m": centers,
        "wall_y_m": np.zeros_like(centers),
        # For integration with p_inf=0, the "wall pressure" is delta p.
        "p_wall_pa": delta_p,
        "p_wall_linear_pa": np.full_like(delta_p, np.nan),
        "p_gauge_pa": delta_p,
        "tau_wall_pa": delta_tau,
        "C_p": delta_p / q_inf,
        "C_f": delta_tau / q_inf,
        "valid": np.isfinite(delta_p) & np.isfinite(delta_tau),
        "failure_code": np.zeros(len(centers), dtype=np.int16),
        "failure_reason": np.full(len(centers), "", dtype="<U160"),
        "amr_level": np.maximum(
            np.asarray(current["amr_level"])[current_indices],
            np.asarray(baseline["amr_level"])[baseline_indices],
        ),
        "requested_x_range_m": np.array([left, right]),
        "p_inf_pa": 0.0,
        "q_inf_pa": q_inf,
    }
    return result


def compute_flat_plate_control_volume_force(
        dataset, x_range_m, y_top_m, viscosity_pa_s,
        bulk_viscosity_pa_s=0.0):
    """Estimate steady wall force from a rectangular momentum balance.

    The control volume spans the lower wall to ``y_top_m``.  Returned force is
    the force exerted by the fluid on the wall per unit span.  This is an
    independent integral diagnostic and assumes the selected snapshot is
    statistically steady; no unsteady momentum-storage term is included.
    """
    x = np.asarray(dataset["x"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float)
    fields = dataset["fields"]
    for key in ("density", "pressure", "x_velocity", "y_velocity"):
        if key not in fields:
            raise KeyError(f"control-volume balance requires '{key}'")
    rho = np.asarray(fields["density"], dtype=float)
    pressure = np.asarray(fields["pressure"], dtype=float)
    u = np.asarray(fields["x_velocity"], dtype=float)
    v = np.asarray(fields["y_velocity"], dtype=float)
    if any(array.shape != (x.size, y.size)
           for array in (rho, pressure, u, v)):
        raise ValueError("control-volume fields must follow canonical (x, y) order")
    x_start, x_end = [float(value) for value in x_range_m]
    i_left = int(np.argmin(np.abs(x - x_start)))
    i_right = int(np.argmin(np.abs(x - x_end)))
    j_top = int(np.argmin(np.abs(y - float(y_top_m))))
    if i_right <= i_left or j_top < 2:
        raise ValueError("control volume requires increasing x and a resolved top")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    du_dx, du_dy = np.gradient(u, dx, dy, edge_order=2)
    dv_dx, dv_dy = np.gradient(v, dx, dy, edge_order=2)
    divergence = du_dx + dv_dy
    mu = float(viscosity_pa_s)
    lambda_viscous = float(bulk_viscosity_pa_s) - (2.0 / 3.0) * mu
    tau_xx = 2.0 * mu * du_dx + lambda_viscous * divergence
    tau_yy = 2.0 * mu * dv_dy + lambda_viscous * divergence
    tau_xy = mu * (du_dy + dv_dx)

    y_slice = slice(0, j_top + 1)
    x_slice = slice(i_left, i_right + 1)
    y_values = y[y_slice]
    x_values = x[x_slice]

    momentum_x = (
        np.trapz(-rho[i_left, y_slice] * u[i_left, y_slice] ** 2, y_values)
        + np.trapz(rho[i_right, y_slice] * u[i_right, y_slice] ** 2, y_values)
        + np.trapz(
            rho[x_slice, j_top] * u[x_slice, j_top] * v[x_slice, j_top],
            x_values,
        )
    )
    momentum_y = (
        np.trapz(
            -rho[i_left, y_slice] * u[i_left, y_slice] * v[i_left, y_slice],
            y_values,
        )
        + np.trapz(
            rho[i_right, y_slice] * u[i_right, y_slice] * v[i_right, y_slice],
            y_values,
        )
        + np.trapz(
            rho[x_slice, j_top] * v[x_slice, j_top] ** 2,
            x_values,
        )
    )
    other_stress_x = (
        np.trapz(
            pressure[i_left, y_slice] - tau_xx[i_left, y_slice], y_values
        )
        + np.trapz(
            -pressure[i_right, y_slice] + tau_xx[i_right, y_slice], y_values
        )
        + np.trapz(tau_xy[x_slice, j_top], x_values)
    )
    other_stress_y = (
        np.trapz(-tau_xy[i_left, y_slice], y_values)
        + np.trapz(tau_xy[i_right, y_slice], y_values)
        + np.trapz(
            -pressure[x_slice, j_top] + tau_yy[x_slice, j_top],
            x_values,
        )
    )
    # Force on the body is the opposite of wall-on-fluid traction.
    body_force_x = other_stress_x - momentum_x
    body_force_y = other_stress_y - momentum_y
    return {
        "D_control_volume_N_m": float(body_force_x),
        "N_control_volume_N_m": float(body_force_y),
        "momentum_flux_x_N_m": float(momentum_x),
        "momentum_flux_y_N_m": float(momentum_y),
        "other_boundary_stress_x_N_m": float(other_stress_x),
        "other_boundary_stress_y_N_m": float(other_stress_y),
        "x_sampled_m": [float(x[i_left]), float(x[i_right])],
        "y_top_sampled_m": float(y[j_top]),
        "assumption": "steady rectangular control volume",
    }


def load_compact_wall_force_history(
        path, reference, duplicate_time_tolerance_s=1.0e-15):
    """Load the versioned case-local PeleC compact wall-force history."""
    path = os.fspath(path)
    header_format = "<8sII6d"
    header_size = struct.calcsize(header_format)
    with open(path, "rb") as stream:
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError("compact force-history header is truncated")
        (
            magic, version, field_count, x_start_cm, x_end_cm,
            p_inf_cgs, mu_cgs, moment_x_cm, moment_y_cm,
        ) = struct.unpack(header_format, header)
        payload = stream.read()
    if magic != _WALL_FORCE_HISTORY_MAGIC:
        raise ValueError("compact force-history magic is invalid")
    if version != 1:
        raise ValueError(f"unsupported compact force-history version {version}")
    if field_count != len(_WALL_FORCE_HISTORY_FIELDS):
        raise ValueError(
            f"force-history field count {field_count} does not match "
            f"schema {len(_WALL_FORCE_HISTORY_FIELDS)}"
        )
    record_bytes = field_count * 8
    complete_bytes = len(payload) - len(payload) % record_bytes
    truncated_tail_bytes = len(payload) - complete_bytes
    raw = np.frombuffer(payload[:complete_bytes], dtype="<f8")
    if raw.size == 0:
        raise ValueError("compact force history contains no complete records")
    records = raw.reshape(-1, field_count)
    times = records[:, 0]
    keep = []
    tolerance = float(duplicate_time_tolerance_s)
    for index, value in enumerate(times):
        if keep and value < times[keep[-1]] - tolerance:
            raise ValueError(
                "compact force history contains a non-duplicate time reversal"
            )
        if keep and abs(value - times[keep[-1]]) <= tolerance:
            previous = keep[-1]
            if not np.array_equal(
                    records[index, 1:], records[previous, 1:]):
                raise ValueError(
                    "compact force history contains conflicting duplicate "
                    "restart records"
                )
            keep[-1] = index
        else:
            keep.append(index)
    records = records[np.asarray(keep, dtype=int)]
    result = {
        name: records[:, index].copy()
        for index, name in enumerate(_WALL_FORCE_HISTORY_FIELDS)
    }
    # dyne/cm -> N/m for two-dimensional force per unit span.
    force_conversion = 1.0e-3
    # (dyne/cm)*cm -> dyne = 1e-5 N = N*m per metre span.
    moment_conversion = 1.0e-5
    result.update({
        "D_pressure_N_m": result["D_pressure_cgs"] * force_conversion,
        "D_viscous_N_m": result["D_viscous_cgs"] * force_conversion,
        "N_pressure_N_m": result["N_pressure_cgs"] * force_conversion,
        "N_viscous_N_m": result["N_viscous_cgs"] * force_conversion,
        "M_pressure_N": result["M_pressure_cgs"] * moment_conversion,
        "M_viscous_N": result["M_viscous_cgs"] * moment_conversion,
        "x_range_m": np.array([x_start_cm, x_end_cm]) * 0.01,
        "p_inf_pa": float(p_inf_cgs * 0.1),
        "viscosity_pa_s": float(mu_cgs * 0.1),
        "moment_origin_m": np.array([moment_x_cm, moment_y_cm]) * 0.01,
        "schema_version": int(version),
        "truncated_tail_bytes": int(truncated_tail_bytes),
        "source": path,
    })
    if "x_range_m" in reference and not np.allclose(
            result["x_range_m"], np.asarray(reference["x_range_m"], dtype=float),
            rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            "compact force-history x interval does not match the configured "
            "certified interval"
        )
    if "p_inf" in reference and not np.isclose(
            result["p_inf_pa"], float(reference["p_inf"]),
            rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(
            "compact force-history p_inf does not match the configured reference"
        )
    if "viscosity_pa_s" in reference and not np.isclose(
            result["viscosity_pa_s"], float(reference["viscosity_pa_s"]),
            rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(
            "compact force-history viscosity does not match configured transport"
        )
    if "moment_origin" in reference and not np.allclose(
            result["moment_origin_m"],
            np.asarray(reference["moment_origin"], dtype=float),
            rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            "compact force-history moment origin does not match the configured "
            "reference"
        )
    for total, pressure_key, viscous_key in (
        ("D_total_N_m", "D_pressure_N_m", "D_viscous_N_m"),
        ("N_total_N_m", "N_pressure_N_m", "N_viscous_N_m"),
        ("M_total_N", "M_pressure_N", "M_viscous_N"),
    ):
        result[total] = result[pressure_key] + result[viscous_key]
    q_inf = 0.5 * float(reference["rho_inf"]) * float(reference["u_inf"]) ** 2
    chord = float(reference["chord"])
    result["C_D"] = result["D_total_N_m"] / (q_inf * chord)
    result["C_N_one_sided"] = result["N_total_N_m"] / (q_inf * chord)
    result["C_L_one_sided"] = result["C_N_one_sided"]
    result["C_M"] = result["M_total_N"] / (q_inf * chord ** 2)
    return result


# ---------------------------------------------------------------------------
# 8b.  LEGACY GENERAL-GEOMETRY FORCE CALCULATIONS (EXPERIMENTAL)
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
                                    R_specific=287.05,
                                    phase_convention="omega_t_minus_alpha_x"):
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
    phase_convention : str, default "omega_t_minus_alpha_x"
        Travelling-wave convention. The default interprets a downstream wave
        as ``exp(i*(omega*t - alpha*x))``. ``"omega_t_plus_alpha_x"`` is
        available for data defined with the opposite spatial sign.

    Returns
    -------
    dict
        ``x_mid``, ``c_p``, ``c_p_fast``, ``c_p_slow``, ``alpha_r``,
        ``delta_phi``, ``target_freq_actual``.
    """
    probe_x = np.asarray(probe_x, dtype=float)
    freq = np.asarray(freq, dtype=float)
    Y_complex = np.asarray(Y_complex, dtype=complex)

    if phase_convention not in (
            "omega_t_minus_alpha_x", "omega_t_plus_alpha_x"):
        raise ValueError(f"Unknown phase convention: {phase_convention}")
    if Y_complex.ndim != 2 or Y_complex.shape[0] != len(freq):
        raise ValueError("Y_complex must have shape (n_freq, n_probes)")
    if Y_complex.shape[1] != len(probe_x):
        raise ValueError("Y_complex probe dimension must match probe_x")
    if not np.all(np.isfinite(probe_x)) or not np.all(np.isfinite(freq)):
        raise ValueError("probe_x and freq must contain only finite values")
    if not np.isfinite(target_freq) or target_freq <= 0:
        raise ValueError("target_freq must be positive and finite")

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
        # For exp(i*(omega*t-alpha*x)), FFT phase decreases downstream:
        # dphi/dx = -alpha. Preserve the alternative convention explicitly
        # instead of hiding the sign choice in a plot.
        phase_sign = -1.0 if phase_convention == "omega_t_minus_alpha_x" else 1.0
        alpha_r[ip] = phase_sign * dphi / dx
        if abs(alpha_r[ip]) > 1e-12:
            c_p[ip] = omega / alpha_r[ip]
        delta_phi[ip] = dphi

    c_p_fast = np.full(n_pairs, np.nan)
    c_p_slow = np.full(n_pairs, np.nan)
    if u_edge is not None and T_edge is not None:
        u_edge_v = np.asarray(u_edge, dtype=float)
        T_edge_v = np.asarray(T_edge, dtype=float)
        if len(u_edge_v) != n_probes or len(T_edge_v) != n_probes:
            raise ValueError("u_edge and T_edge must match probe_x length")
        u_edge_v = u_edge_v[sort_idx]
        T_edge_v = T_edge_v[sort_idx]
        a_edge = np.sqrt(gamma * R_specific * T_edge_v)
        c_p_fast = np.interp(x_mid, probe_x, a_edge + u_edge_v)
        c_p_slow = np.interp(x_mid, probe_x, u_edge_v - a_edge)

    return {
        "x_mid": x_mid, "c_p": c_p,
        "c_p_fast": c_p_fast, "c_p_slow": c_p_slow,
        "alpha_r": alpha_r, "delta_phi": delta_phi,
        "target_freq_actual": f_actual,
        "phase_convention": phase_convention,
    }


def compute_growth_rate_from_probes(probe_x, freq, P1, target_freq,
                                    delta_99_interp=None,
                                    window_size=None):
    """Compute spatial alpha_i and amplification from FFT amplitude.

    Fits ln(A) = -alpha_i * x + const over a sliding window.
    Under the standard spatial-LST convention, ``-alpha_i`` is therefore the
    amplification rate: positive values grow downstream.

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
        ``x``, ``alpha_i``, ``amplification_rate``, ``alpha_i_delta``
        (if delta_99 given), ``A_at_f``, ``r_squared``.
    """
    probe_x = np.asarray(probe_x, dtype=float)
    freq = np.asarray(freq, dtype=float)
    P1 = np.asarray(P1, dtype=float)

    sort_idx = np.argsort(probe_x)
    probe_x = probe_x[sort_idx]
    P1 = P1[:, sort_idx]

    n_probes = len(probe_x)
    if n_probes < 3:
        missing = np.full(n_probes, np.nan)
        return {
            "x": probe_x, "alpha_i": missing.copy(),
            "amplification_rate": missing.copy(),
            "A_at_f": missing.copy(), "r_squared": missing.copy(),
        }

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
    alpha_i_ci95 = np.full(n_probes, np.nan)
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
        sxx = np.sum((x_win - np.mean(x_win)) ** 2)
        if len(x_win) > 2 and sxx > 0:
            slope_se = np.sqrt((ss_res / (len(x_win) - 2)) / sxx)
            alpha_i_ci95[i] = 1.96 * slope_se

    out = {
        "x": probe_x, "alpha_i": alpha_i,
        "amplification_rate": -alpha_i,
        "A_at_f": A, "r_squared": r2,
        "alpha_i_ci95": alpha_i_ci95,
        "window_size": int(window_size),
    }

    if delta_99_interp is not None:
        d99 = np.asarray(delta_99_interp, dtype=float)
        if len(d99) == n_probes:
            out["alpha_i_delta"] = alpha_i * d99[sort_idx]
            out["amplification_rate_delta"] = (
                -alpha_i * d99[sort_idx]
            )
        else:
            out["alpha_i_delta"] = np.full(n_probes, np.nan)
            out["amplification_rate_delta"] = np.full(n_probes, np.nan)

    return out


def compute_pair_coherence(signal_a, signal_b, fs, nperseg=16384,
                           noverlap=None):
    """Estimate magnitude-squared coherence and cross-spectral phase.

    The cross spectrum follows SciPy's ``conj(X_a) * X_b`` convention, so
    its phase is the downstream-minus-upstream phase for an ordered pair.
    """
    try:
        from scipy.signal import coherence, csd
    except ImportError as exc:
        raise ImportError("scipy.signal is required for coherence analysis") from exc
    a = np.asarray(signal_a, dtype=float).ravel()
    b = np.asarray(signal_b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 8:
        raise ValueError("Signals must have equal shape and at least 8 samples")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Signals contain NaN or infinite values")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    nperseg = min(int(nperseg), a.size)
    if nperseg < 8:
        raise ValueError("nperseg must be at least 8")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    freq, gamma2 = coherence(
        a, b, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, detrend="constant",
    )
    freq_csd, cross = csd(
        a, b, fs=fs, window="hann", nperseg=nperseg,
        noverlap=noverlap, detrend="constant", scaling="density",
    )
    if not np.array_equal(freq, freq_csd):
        raise RuntimeError("Coherence and CSD frequency grids differ")
    return {
        "frequency_hz": freq,
        "coherence_squared": np.clip(gamma2, 0.0, 1.0),
        "cross_phase_rad": np.angle(cross),
        "nperseg": nperseg,
        "noverlap": noverlap,
    }


def gaussian_pulse_train(
        time_s, start_time_s, frequency_hz, pulse_fwhm_s,
        duration_s, amplitude=1.0):
    """Evaluate a Gaussian pulse train using FWHM as the pulse duration."""
    time_values = np.asarray(time_s, dtype=float)
    frequency_hz = float(frequency_hz)
    pulse_fwhm_s = float(pulse_fwhm_s)
    duration_s = float(duration_s)
    if frequency_hz <= 0.0 or pulse_fwhm_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("pulse frequency, FWHM, and duration must be positive")
    sigma = pulse_fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    period = 1.0 / frequency_hz
    first = float(start_time_s)
    final = first + duration_s
    # Only pulses within five standard deviations of the requested record can
    # contribute measurably.  This keeps long records computationally bounded.
    lower = float(np.min(time_values)) - 5.0 * sigma
    upper = float(np.max(time_values)) + 5.0 * sigma
    first_index = max(0, int(np.floor((lower - first) / period)))
    last_index = min(
        int(np.floor(duration_s / period)),
        int(np.ceil((upper - first) / period)),
    )
    signal = np.zeros_like(time_values, dtype=float)
    if last_index >= first_index:
        centers = first + np.arange(first_index, last_index + 1) * period
        for center in centers:
            if center <= final + np.finfo(float).eps:
                signal += float(amplitude) * np.exp(
                    -0.5 * ((time_values - center) / sigma) ** 2
                )
    return {
        "signal": signal,
        "sigma_s": sigma,
        "period_s": period,
        "pulse_count_evaluated": max(0, last_index - first_index + 1),
    }


def normalized_cross_correlation(signal_a, signal_b, sample_interval_s):
    """Return normalized full cross-correlation and physical lag."""
    a = np.asarray(signal_a, dtype=float).ravel()
    b = np.asarray(signal_b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("cross-correlation signals need equal length >= 2")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("cross-correlation signals must be finite")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= np.finfo(float).tiny:
        correlation = np.zeros(2 * a.size - 1)
    else:
        try:
            from scipy.signal import correlate
            correlation = correlate(b, a, mode="full", method="auto") / denominator
        except ImportError:
            correlation = np.correlate(b, a, mode="full") / denominator
    lags = np.arange(-a.size + 1, a.size) * float(sample_interval_s)
    peak = int(np.argmax(np.abs(correlation)))
    return {
        "lag_s": lags,
        "correlation": correlation,
        "peak_lag_s": float(lags[peak]),
        "peak_correlation": float(correlation[peak]),
        "convention": "positive lag means signal_b follows signal_a",
    }


def synchronize_force_and_probe_signals(
        force_time_s, force_signal, probe_time_s, probe_matrix):
    """Synchronize force and probe signals without extrapolation.

    The coarser native sample rate is retained.  A zero-phase low-pass filter
    is applied to whichever input is downsampled before interpolation onto the
    common physical-time grid.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError as exc:
        raise ImportError("scipy.signal is required for signal synchronization") from exc

    force_time = np.asarray(force_time_s, dtype=float).ravel()
    force_values = np.asarray(force_signal, dtype=float).ravel()
    probe_time = np.asarray(probe_time_s, dtype=float).ravel()
    probes = np.asarray(probe_matrix, dtype=float)
    if probes.ndim == 1:
        probes = probes[:, None]
    if force_time.size != force_values.size or probe_time.size != probes.shape[0]:
        raise ValueError("time and signal dimensions are inconsistent")
    if force_time.size < 4 or probe_time.size < 4:
        raise ValueError("at least four force and probe samples are required")
    if (not np.all(np.isfinite(force_time))
            or not np.all(np.isfinite(force_values))
            or not np.all(np.isfinite(probe_time))
            or not np.all(np.isfinite(probes))):
        raise ValueError("force/probe synchronization requires finite values")
    if np.any(np.diff(force_time) <= 0.0) or np.any(np.diff(probe_time) <= 0.0):
        raise ValueError("force and probe times must be strictly increasing")

    force_dt = float(np.median(np.diff(force_time)))
    probe_dt = float(np.median(np.diff(probe_time)))
    for name, values, dt in (
            ("force", force_time, force_dt), ("probe", probe_time, probe_dt)):
        if not np.allclose(
                np.diff(values), dt, rtol=1.0e-6,
                atol=max(1.0e-15, 1.0e-8 * abs(dt))):
            raise ValueError(f"{name} history is not uniformly sampled")
    target_dt = max(force_dt, probe_dt)
    target_rate = 1.0 / target_dt

    def filtered(values, source_dt):
        source_rate = 1.0 / source_dt
        if source_rate <= target_rate * (1.0 + 1.0e-8):
            return values
        normalized_cutoff = 0.9 * target_rate / source_rate
        sos = butter(6, normalized_cutoff, btype="low", output="sos")
        pad_requirement = 3 * (2 * len(sos) + 1)
        if values.shape[0] <= pad_requirement:
            raise ValueError(
                "too few samples for anti-alias filtering before decimation"
            )
        return sosfiltfilt(sos, values, axis=0)

    force_filtered = filtered(force_values, force_dt)
    probes_filtered = filtered(probes, probe_dt)
    start = max(float(force_time[0]), float(probe_time[0]))
    stop = min(float(force_time[-1]), float(probe_time[-1]))
    if stop <= start:
        raise ValueError("force and probe histories have no common time interval")
    count = int(np.floor((stop - start) / target_dt + 1.0e-9)) + 1
    if count < 4:
        raise ValueError("common force/probe interval contains fewer than four samples")
    common_time = start + np.arange(count, dtype=float) * target_dt
    common_time = common_time[common_time <= stop + 1.0e-12 * target_dt]
    synchronized_force = np.interp(common_time, force_time, force_filtered)
    synchronized_probes = np.column_stack([
        np.interp(common_time, probe_time, probes_filtered[:, index])
        for index in range(probes.shape[1])
    ])
    return {
        "time_s": common_time,
        "force": synchronized_force,
        "probe_matrix": synchronized_probes,
        "force_sample_rate_original_hz": 1.0 / force_dt,
        "probe_sample_rate_original_hz": 1.0 / probe_dt,
        "sample_rate_resampled_hz": target_rate,
        "anti_alias_filter": "sixth-order zero-phase Butterworth at 0.45 target Fs",
    }


def compute_probe_force_linkage(
        force_time_s, force_signal, probe_time_s, probe_matrix, probe_x_m,
        forcing_frequency_hz, minimum_forcing_periods=10.0,
        nperseg=16384, noverlap=0.5, minimum_segments=8):
    """Relate a pressure-probe line to one aerodynamic-force response."""
    synchronized = synchronize_force_and_probe_signals(
        force_time_s, force_signal, probe_time_s, probe_matrix
    )
    pressure = synchronized["probe_matrix"]
    force = synchronized["force"]
    probe_x = np.asarray(probe_x_m, dtype=float).ravel()
    if pressure.shape[1] != probe_x.size:
        raise ValueError("probe_x_m must contain one coordinate per probe")
    dt = 1.0 / synchronized["sample_rate_resampled_hz"]
    peak_lag = np.empty(probe_x.size)
    peak_correlation = np.empty(probe_x.size)
    for index in range(probe_x.size):
        correlation = normalized_cross_correlation(
            pressure[:, index], force, dt
        )
        peak_lag[index] = correlation["peak_lag_s"]
        peak_correlation[index] = correlation["peak_correlation"]

    duration = float(np.ptp(synchronized["time_s"]))
    forcing_periods = duration * float(forcing_frequency_hz)
    result = {
        **synchronized,
        "probe_x_m": probe_x,
        "peak_lag_s": peak_lag,
        "peak_correlation": peak_correlation,
        "forcing_periods": forcing_periods,
        "spectral_status": "insufficient_data",
    }
    if forcing_periods < float(minimum_forcing_periods):
        result["spectral_reason"] = (
            f"common interval contains {forcing_periods:.3f} forcing periods; "
            f"{float(minimum_forcing_periods):g} required"
        )
        return result

    spectral_results = [
        compute_input_output_spectra(
            pressure[:, index], force,
            synchronized["sample_rate_resampled_hz"],
            nperseg=nperseg, noverlap=noverlap,
            minimum_segments=minimum_segments,
        )
        for index in range(probe_x.size)
    ]
    result["frequency_hz"] = spectral_results[0]["frequency_hz"]
    for key in (
            "cross_spectrum", "coherence_squared", "cross_phase_rad",
            "H1", "valid_transfer"):
        result[key] = np.asarray([item[key] for item in spectral_results])
    result["spectral_status"] = "complete"
    result["spectral_reason"] = None
    return result


def compute_input_output_spectra(
        input_signal, output_signal, sample_rate_hz,
        nperseg=16384, noverlap=0.5, minimum_segments=8,
        minimum_input_power_fraction=1.0e-10):
    """Compute Welch PSD/CSD, coherence, phase, and the H1 transfer estimate."""
    try:
        from scipy.signal import coherence, csd, welch
    except ImportError as exc:
        raise ImportError("scipy.signal is required for force spectra") from exc
    x = np.asarray(input_signal, dtype=float).ravel()
    response = np.asarray(output_signal, dtype=float).ravel()
    if x.shape != response.shape or x.size < 16:
        raise ValueError("input and output need equal length >= 16")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(response)):
        raise ValueError("input and output signals must be finite")
    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    overlap_fraction = float(noverlap)
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("noverlap must be a fraction in [0, 1)")
    requested = min(int(nperseg), x.size)
    overlap_samples = int(round(overlap_fraction * requested))
    step = requested - overlap_samples
    segment_count = (
        1 + (x.size - requested) // step if step > 0 else 0
    )
    if segment_count < int(minimum_segments):
        raise ValueError(
            f"only {segment_count} Welch segments are available; "
            f"{int(minimum_segments)} required"
        )
    frequency, input_psd = welch(
        x, fs=sample_rate_hz, window="hann", nperseg=requested,
        noverlap=overlap_samples, detrend="constant", scaling="density",
    )
    _, output_psd = welch(
        response, fs=sample_rate_hz, window="hann", nperseg=requested,
        noverlap=overlap_samples, detrend="constant", scaling="density",
    )
    _, cross = csd(
        x, response, fs=sample_rate_hz, window="hann",
        nperseg=requested, noverlap=overlap_samples,
        detrend="constant", scaling="density",
    )
    _, coherence_squared = coherence(
        x, response, fs=sample_rate_hz, window="hann",
        nperseg=requested, noverlap=overlap_samples,
        detrend="constant",
    )
    power_threshold = (
        float(minimum_input_power_fraction) * float(np.max(input_psd))
    )
    valid_transfer = input_psd > power_threshold
    transfer = np.full_like(cross, np.nan + 1j * np.nan)
    transfer[valid_transfer] = cross[valid_transfer] / input_psd[valid_transfer]
    return {
        "frequency_hz": frequency,
        "input_psd": input_psd,
        "output_psd": output_psd,
        "cross_spectrum": cross,
        "coherence_squared": np.clip(coherence_squared, 0.0, 1.0),
        "cross_phase_rad": np.angle(cross),
        "H1": transfer,
        "valid_transfer": valid_transfer,
        "nperseg": requested,
        "noverlap": overlap_samples,
        "segment_count": int(segment_count),
    }


def compute_adjacent_target_coherence(signal_matrix, fs, target_freq,
                                      nperseg=16384, noverlap=None,
                                      column_order=None):
    """Welch coherence at one frequency for every adjacent spatial pair.

    Only one complex Fourier coefficient per block and probe is retained, so
    this provides a coherence gate for large probe lines without allocating a
    frequency-by-probe tensor.
    """
    signals = np.asarray(signal_matrix, dtype=float)
    if signals.ndim != 2 or signals.shape[0] < 8 or signals.shape[1] < 2:
        raise ValueError("signal_matrix must have shape (n_time>=8, n_probe>=2)")
    if not np.all(np.isfinite(signals)):
        raise ValueError("signal_matrix contains NaN or infinite values")
    fs = float(fs)
    nperseg = min(int(nperseg), signals.shape[0])
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    frequency = np.fft.rfftfreq(nperseg, 1.0 / fs)
    target_index = int(np.argmin(np.abs(frequency - float(target_freq))))
    actual_frequency = float(frequency[target_index])
    sample = np.arange(nperseg, dtype=float)
    window = np.hanning(nperseg)
    kernel = window * np.exp(
        -2j * np.pi * target_index * sample / nperseg
    )
    kernel_sum = np.sum(kernel)
    step = nperseg - noverlap
    starts = np.arange(0, signals.shape[0] - nperseg + 1, step, dtype=int)
    if starts.size < 2:
        raise ValueError("Coherence requires at least two complete blocks")
    order = (
        np.arange(signals.shape[1]) if column_order is None
        else np.asarray(column_order, dtype=int)
    )
    if sorted(order.tolist()) != list(range(signals.shape[1])):
        raise ValueError("column_order must be a permutation of probe columns")
    auto_sum = np.zeros(signals.shape[1], dtype=float)
    cross_sum = np.zeros(signals.shape[1] - 1, dtype=complex)
    for start in starts:
        segment = signals[start:start + nperseg, :]
        coefficient = kernel @ segment - np.mean(segment, axis=0) * kernel_sum
        coefficient = coefficient[order]
        auto_sum += np.abs(coefficient) ** 2
        cross_sum += np.conj(coefficient[:-1]) * coefficient[1:]
    auto_mean = auto_sum / starts.size
    cross_mean = cross_sum / starts.size
    denominator = auto_mean[:-1] * auto_mean[1:]
    coherence_squared = np.abs(cross_mean) ** 2 / np.maximum(
        denominator, 1.0e-30
    )
    return {
        "target_frequency_actual_hz": actual_frequency,
        "coherence_squared": np.clip(coherence_squared, 0.0, 1.0),
        "cross_phase_rad": np.angle(cross_mean),
        "n_blocks": int(starts.size),
        "nperseg": nperseg,
        "noverlap": noverlap,
    }


def _weighted_linear_fit(x, y, weights):
    """Return slope, intercept, weighted R-squared, and slope standard error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
        & (weights > 0.0)
    )
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x = x[valid]
    y = y[valid]
    weights = weights[valid]
    weights = weights / np.mean(weights)
    design = np.column_stack((x, np.ones_like(x)))
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance_base = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    coefficients = covariance_base @ (
        design.T @ (weights * y)
    )
    slope, intercept = coefficients
    fitted = design @ coefficients
    mean = np.average(y, weights=weights)
    residual_sum = np.sum(weights * (y - fitted) ** 2)
    total_sum = np.sum(weights * (y - mean) ** 2)
    r_squared = (
        1.0 - residual_sum / total_sum
        if total_sum > 1.0e-30 else np.nan
    )
    degrees_freedom = len(y) - 2
    if degrees_freedom > 0:
        residual_variance = residual_sum / degrees_freedom
        slope_standard_error = np.sqrt(
            max(0.0, residual_variance * covariance_base[0, 0])
        )
    else:
        slope_standard_error = np.nan
    return slope, intercept, r_squared, slope_standard_error


def compute_frequency_resolved_wavenumber(
        signal_matrix, probe_x, fs, frequency_band,
        nperseg=16384, noverlap=None, frequency_stride=1,
        spatial_window_size=101, spatial_step=20, fft_batch_size=32,
        min_coherence=0.8, min_coherent_fraction=0.8,
        min_phase_r_squared=0.8, min_amplitude_r_squared=0.5,
        min_relative_power_db=-40.0,
        max_edge_phase_rad=0.9 * np.pi, phase_speed_bounds=None,
        phase_convention="omega_t_minus_alpha_x"):
    """Estimate frequency-resolved complex streamwise wavenumber.

    The routine first forms Welch-averaged auto spectra and adjacent-probe
    cross spectra.  Within overlapping spatial windows it then fits

    ``phase(x) = constant - alpha_r*x``

    and

    ``log(amplitude(x)) = constant - alpha_i*x``.

    The latter follows the standard spatial-LST convention
    ``q' = Re(qhat exp(i*(alpha*x - omega*t)))``.  Consequently
    ``-alpha_i`` is the spatial amplification rate.

    This is a dominant-wave estimator.  It deliberately does not label the
    result as an F or S eigenmode: simultaneous waves at one frequency can
    produce a composite cross-spectral phase and require an eigensolution or
    a separately validated multi-exponential decomposition.

    Parameters
    ----------
    signal_matrix : array-like, shape (n_time, n_probe)
        Uniformly sampled probe signals.
    probe_x : 1-D array-like
        Probe positions [m]. They may be irregular but must be distinct.
    fs : float
        Temporal sample rate [Hz].
    frequency_band : pair of float
        Inclusive frequency range [Hz].
    nperseg, noverlap : int
        Welch block length and overlap.
    frequency_stride : int
        Retain every Nth FFT bin within ``frequency_band``.
    spatial_window_size, spatial_step : int
        Number of probes per local fit and distance between fit centres.
    fft_batch_size : int
        Probe columns transformed together to bound peak memory.
    min_coherence, min_coherent_fraction : float
        Adjacent-pair coherence requirements for a valid phase fit.
    min_phase_r_squared, min_amplitude_r_squared : float
        Fit-quality gates. Phase and growth masks are returned separately.
    min_relative_power_db : float
        Minimum local spectral power relative to the strongest frequency at
        the same spatial centre. This is a leakage/noise guard, not a
        calibrated signal-to-noise ratio.
    max_edge_phase_rad : float
        Phase-alias guard applied to adjacent cross-spectral phase.
    phase_speed_bounds : pair of float, optional
        Accepted positive phase-speed interval [m/s].
    phase_convention : str
        Convention used to interpret the positive-frequency FFT phase.

    Returns
    -------
    dict
        Frequency/x grids, raw ``alpha_real`` and ``alpha_imag``, explicit
        ``amplification_rate=-alpha_imag``, phase speed, wavelength, spectral
        power, uncertainty and quality metrics, and separate validity masks.
    """
    signals = np.asarray(signal_matrix, dtype=float)
    x = np.asarray(probe_x, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != x.size:
        raise ValueError(
            "signal_matrix must have shape (n_time, len(probe_x))"
        )
    if signals.shape[0] < 8 or x.size < 5:
        raise ValueError("At least 8 time samples and 5 probes are required")
    if not np.all(np.isfinite(signals)) or not np.all(np.isfinite(x)):
        raise ValueError("Signals and probe positions must be finite")
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be positive and finite")
    if phase_convention not in (
            "omega_t_minus_alpha_x", "omega_t_plus_alpha_x"):
        raise ValueError(f"Unknown phase convention: {phase_convention}")

    band = np.asarray(frequency_band, dtype=float)
    if band.shape != (2,) or not 0.0 <= band[0] < band[1] <= 0.5 * fs:
        raise ValueError("frequency_band must lie within [0, Nyquist]")
    frequency_stride = int(frequency_stride)
    spatial_window_size = int(spatial_window_size)
    spatial_step = int(spatial_step)
    fft_batch_size = int(fft_batch_size)
    if frequency_stride < 1 or spatial_step < 1 or fft_batch_size < 1:
        raise ValueError(
            "frequency_stride, spatial_step, and fft_batch_size must be positive"
        )
    if spatial_window_size < 5:
        raise ValueError("spatial_window_size must be at least 5")
    if spatial_window_size > x.size:
        spatial_window_size = x.size
    if spatial_window_size % 2 == 0:
        spatial_window_size -= 1
    if not 0.0 <= min_coherence <= 1.0:
        raise ValueError("min_coherence must lie within [0, 1]")
    if not 0.0 <= min_coherent_fraction <= 1.0:
        raise ValueError("min_coherent_fraction must lie within [0, 1]")
    min_relative_power_db = float(min_relative_power_db)
    if not np.isfinite(min_relative_power_db) or min_relative_power_db > 0.0:
        raise ValueError("min_relative_power_db must be finite and <= 0")
    if not 0.0 < max_edge_phase_rad <= np.pi:
        raise ValueError("max_edge_phase_rad must lie within (0, pi]")
    if phase_speed_bounds is not None:
        phase_speed_bounds = np.asarray(phase_speed_bounds, dtype=float)
        if (phase_speed_bounds.shape != (2,)
                or phase_speed_bounds[0] < 0.0
                or phase_speed_bounds[1] <= phase_speed_bounds[0]):
            raise ValueError(
                "phase_speed_bounds must be [positive_min, larger_max]"
            )

    order = np.argsort(x)
    x = x[order]
    signals = signals[:, order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("probe_x positions must be distinct")

    nperseg = min(int(nperseg), signals.shape[0])
    if nperseg < 8:
        raise ValueError("nperseg must be at least 8")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    step = nperseg - noverlap
    starts = np.arange(
        0, signals.shape[0] - nperseg + 1, step, dtype=int
    )
    if starts.size < 2:
        raise ValueError(
            "Frequency-resolved analysis requires at least two Welch blocks"
        )

    all_frequency = np.fft.rfftfreq(nperseg, 1.0 / fs)
    selected = np.flatnonzero(
        (all_frequency >= band[0]) & (all_frequency <= band[1])
    )[::frequency_stride]
    if selected.size == 0:
        raise ValueError("No FFT bins fall inside frequency_band")
    frequency = all_frequency[selected]
    n_frequency = frequency.size
    n_probe = x.size
    auto_sum = np.zeros((n_frequency, n_probe), dtype=float)
    cross_sum = np.zeros((n_frequency, n_probe - 1), dtype=complex)
    window = np.hanning(nperseg)

    for start in starts:
        coefficient = np.empty((n_frequency, n_probe), dtype=complex)
        for first in range(0, n_probe, fft_batch_size):
            last = min(first + fft_batch_size, n_probe)
            work = signals[start:start + nperseg, first:last]
            work = work - np.mean(work, axis=0, keepdims=True)
            spectrum = np.fft.rfft(
                work * window[:, None], axis=0
            )
            coefficient[:, first:last] = spectrum[selected, :]
        auto_sum += np.abs(coefficient) ** 2
        cross_sum += np.conj(coefficient[:, :-1]) * coefficient[:, 1:]

    auto_power = auto_sum / starts.size
    cross_power = cross_sum / starts.size
    denominator = auto_power[:, :-1] * auto_power[:, 1:]
    coherence = np.clip(
        np.abs(cross_power) ** 2 / np.maximum(denominator, 1.0e-300),
        0.0, 1.0,
    )
    edge_phase = np.angle(cross_power)
    amplitude = np.sqrt(np.maximum(auto_power, 1.0e-300))

    half_window = spatial_window_size // 2
    centre_indices = np.arange(
        half_window, n_probe - half_window, spatial_step, dtype=int
    )
    if centre_indices.size == 0:
        centre_indices = np.array([n_probe // 2], dtype=int)
    n_centre = centre_indices.size
    shape = (n_frequency, n_centre)
    alpha_real = np.full(shape, np.nan)
    alpha_imag = np.full(shape, np.nan)
    alpha_real_ci95 = np.full(shape, np.nan)
    alpha_imag_ci95 = np.full(shape, np.nan)
    phase_r_squared = np.full(shape, np.nan)
    amplitude_r_squared = np.full(shape, np.nan)
    mean_coherence = np.full(shape, np.nan)
    coherent_fraction = np.full(shape, np.nan)
    alias_margin = np.full(shape, np.nan)
    local_power = np.full(shape, np.nan)

    phase_sign = (
        -1.0 if phase_convention == "omega_t_minus_alpha_x" else 1.0
    )
    for frequency_index in range(n_frequency):
        for centre_number, centre in enumerate(centre_indices):
            first = centre - half_window
            last = centre + half_window + 1
            x_window = x[first:last]
            edge_coherence = coherence[
                frequency_index, first:last - 1
            ]
            phase_increment = edge_phase[
                frequency_index, first:last - 1
            ]
            local_phase = np.concatenate((
                [0.0], np.cumsum(phase_increment)
            ))
            probe_weights = np.empty(spatial_window_size, dtype=float)
            probe_weights[0] = edge_coherence[0]
            probe_weights[-1] = edge_coherence[-1]
            probe_weights[1:-1] = np.minimum(
                edge_coherence[:-1], edge_coherence[1:]
            )
            phase_slope, _, phase_r2, phase_se = _weighted_linear_fit(
                x_window, local_phase, probe_weights
            )
            log_amplitude = np.log(
                amplitude[frequency_index, first:last]
            )
            amplitude_slope, _, amplitude_r2, amplitude_se = (
                _weighted_linear_fit(
                    x_window, log_amplitude, probe_weights
                )
            )
            alpha_real[frequency_index, centre_number] = (
                phase_sign * phase_slope
            )
            alpha_imag[frequency_index, centre_number] = -amplitude_slope
            alpha_real_ci95[frequency_index, centre_number] = 1.96 * phase_se
            alpha_imag_ci95[frequency_index, centre_number] = (
                1.96 * amplitude_se
            )
            phase_r_squared[frequency_index, centre_number] = phase_r2
            amplitude_r_squared[frequency_index, centre_number] = amplitude_r2
            mean_coherence[frequency_index, centre_number] = np.mean(
                edge_coherence
            )
            coherent_fraction[frequency_index, centre_number] = np.mean(
                edge_coherence >= min_coherence
            )
            largest_phase = np.max(np.abs(phase_increment))
            alias_margin[frequency_index, centre_number] = (
                np.inf if largest_phase == 0.0
                else np.pi / largest_phase
            )
            local_power[frequency_index, centre_number] = np.mean(
                auto_power[frequency_index, first:last]
            )

    omega = 2.0 * np.pi * frequency[:, None]
    phase_speed = np.divide(
        omega, alpha_real,
        out=np.full_like(alpha_real, np.nan),
        where=np.abs(alpha_real) > 1.0e-12,
    )
    wavelength = np.divide(
        2.0 * np.pi, alpha_real,
        out=np.full_like(alpha_real, np.nan),
        where=np.abs(alpha_real) > 1.0e-12,
    )
    amplification_rate = -alpha_imag
    centre_peak_power = np.nanmax(local_power, axis=0, keepdims=True)
    relative_power_db = 10.0 * np.log10(
        np.maximum(local_power, 1.0e-300)
        / np.maximum(centre_peak_power, 1.0e-300)
    )
    spectral_valid = relative_power_db >= min_relative_power_db
    phase_valid = (
        np.isfinite(alpha_real)
        & (alpha_real > 0.0)
        & np.isfinite(phase_r_squared)
        & (phase_r_squared >= min_phase_r_squared)
        & (coherent_fraction >= min_coherent_fraction)
        & spectral_valid
    )
    # The edge phase is wrapped to [-pi, pi]. Values too close to that limit
    # cannot distinguish the physical wavenumber from a spatial alias.
    phase_valid &= alias_margin >= (np.pi / max_edge_phase_rad)
    if phase_speed_bounds is not None:
        phase_valid &= (
            (phase_speed >= phase_speed_bounds[0])
            & (phase_speed <= phase_speed_bounds[1])
        )
    growth_valid = (
        phase_valid
        & np.isfinite(alpha_imag)
        & np.isfinite(amplitude_r_squared)
        & (amplitude_r_squared >= min_amplitude_r_squared)
    )

    return {
        "frequency_hz": frequency,
        "x_center_m": x[centre_indices],
        "alpha_real_rad_per_m": alpha_real,
        "alpha_imag_rad_per_m": alpha_imag,
        "amplification_rate_per_m": amplification_rate,
        "phase_speed_m_per_s": phase_speed,
        "wavelength_m": wavelength,
        "alpha_real_ci95_rad_per_m": alpha_real_ci95,
        "alpha_imag_ci95_rad_per_m": alpha_imag_ci95,
        "phase_fit_r_squared": phase_r_squared,
        "amplitude_fit_r_squared": amplitude_r_squared,
        "mean_coherence_squared": mean_coherence,
        "coherent_pair_fraction": coherent_fraction,
        "spatial_alias_margin": alias_margin,
        "spectral_power": local_power,
        "relative_spectral_power_db": relative_power_db,
        "spectral_valid_mask": spectral_valid,
        "phase_valid_mask": phase_valid,
        "growth_valid_mask": growth_valid,
        "adjacent_coherence_squared": coherence,
        "adjacent_cross_phase_rad": edge_phase,
        "probe_x_sorted_m": x,
        "probe_sort_order": order,
        "n_blocks": int(starts.size),
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "spatial_window_size": int(spatial_window_size),
        "spatial_step": int(spatial_step),
        "phase_convention": phase_convention,
        "alpha_convention": "exp(i*(alpha*x-omega*t))",
    }


def fit_common_frequency_model(signal_matrix, dt, n_modes=10,
                               train_fraction=0.6, freq_range=None,
                               min_peak_distance_hz=None):
    """Fit shared spectral frequencies and evaluate them out of sample.

    Frequencies are selected from aggregate, per-probe-normalized training
    spectra.  Sinusoidal coefficients are fit on the training interval only
    and extrapolated into the held-out interval.  Validation error therefore
    measures stationary coherent-mode predictability instead of same-window
    Fourier reconstruction fidelity.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy.signal is required for mode selection") from exc
    signals, length, fs, nyquist = _validate_reconstruction_sampling(
        signal_matrix, dt
    )
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if not 0.2 <= float(train_fraction) <= 0.8:
        raise ValueError("train_fraction must lie between 0.2 and 0.8")
    n_train = int(round(length * float(train_fraction)))
    if n_train < 16 or length - n_train < 8:
        raise ValueError("Training and validation intervals are too short")

    means = np.mean(signals[:n_train, :], axis=0, keepdims=True)
    centered = signals - means
    train = centered[:n_train, :]
    validation = centered[n_train:, :]
    spectrum = np.fft.rfft(train, axis=0)
    freq = np.fft.rfftfreq(n_train, dt)
    power = np.abs(spectrum) ** 2
    normalization = np.sum(power, axis=0, keepdims=True)
    score = np.mean(power / np.maximum(normalization, 1.0e-30), axis=1)
    mask = np.ones(freq.size, dtype=bool)
    mask[0] = False
    if freq_range is not None:
        f_low, f_high = map(float, freq_range)
        if not 0 <= f_low < f_high <= nyquist:
            raise ValueError("freq_range must lie within [0, Nyquist]")
        mask &= (freq >= f_low) & (freq <= f_high)
    df = fs / n_train
    if min_peak_distance_hz is None:
        min_peak_distance_hz = 3.0 * df
    distance = max(1, int(np.ceil(float(min_peak_distance_hz) / df)))
    search = score.copy()
    search[~mask] = -np.inf
    peaks, _ = find_peaks(search, distance=distance)
    peaks = peaks[mask[peaks]]
    if peaks.size == 0:
        raise ValueError("No common spectral peaks found in the training interval")
    selected = peaks[np.argsort(score[peaks])[::-1][:n_modes]]
    selected = np.sort(selected)
    selected_freq = freq[selected]

    time = np.arange(length, dtype=float) * dt
    columns = [np.ones(length)]
    for value in selected_freq:
        columns.extend((
            np.cos(2.0 * np.pi * value * time),
            np.sin(2.0 * np.pi * value * time),
        ))
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(
        design[:n_train, :], train, rcond=None
    )
    predicted = design @ coefficients
    train_prediction = predicted[:n_train, :]
    validation_prediction = predicted[n_train:, :]

    def _relative_rms(observed, estimate):
        numerator = np.sqrt(np.mean((observed - estimate) ** 2, axis=0))
        denominator = np.sqrt(np.mean(observed ** 2, axis=0))
        return numerator / np.maximum(denominator, 1.0e-30)

    return {
        "selected_frequency_hz": selected_freq,
        "selected_bins": selected,
        "spectral_score": score[selected],
        "n_train": n_train,
        "train_observed": train,
        "train_prediction": train_prediction,
        "validation_observed": validation,
        "validation_prediction": validation_prediction,
        "train_relative_rms": _relative_rms(train, train_prediction),
        "validation_relative_rms": _relative_rms(
            validation, validation_prediction
        ),
        "training_mean": means.ravel(),
    }


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


def _validate_reconstruction_sampling(signal_matrix, dt):
    """Validate reconstruction inputs and return ``(array, L, Fs, Nyquist)``."""
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(-1, 1)
    if signal_matrix.ndim != 2 or signal_matrix.shape[0] < 2:
        raise ValueError("signal_matrix must contain at least two time samples")
    if not np.all(np.isfinite(signal_matrix)):
        raise ValueError("signal_matrix contains NaN or infinite values")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite sampling interval")
    Fs = 1.0 / dt
    return signal_matrix, signal_matrix.shape[0], Fs, 0.5 * Fs


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
    signal_matrix, L, Fs, nyquist = _validate_reconstruction_sampling(
        signal_matrix, dt
    )
    harmonic_freq = float(harmonic_freq)
    num_harmonics = int(num_harmonics)
    if not np.isfinite(harmonic_freq) or harmonic_freq <= 0:
        raise ValueError("harmonic_freq must be positive and finite")
    if num_harmonics < 1:
        raise ValueError("num_harmonics must be at least 1")
    highest = num_harmonics * harmonic_freq
    if highest > nyquist * (1.0 + 10.0 * np.finfo(float).eps):
        raise ValueError(
            f"Requested harmonic {highest:.6g} Hz exceeds Nyquist "
            f"frequency {nyquist:.6g} Hz"
        )

    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq_pos = np.fft.rfftfreq(L, dt)

    harmonic_bins = []
    for nh in range(1, num_harmonics + 1):
        f_target = nh * harmonic_freq
        idx = int(np.argmin(np.abs(freq_pos - f_target)))
        harmonic_bins.append(idx)
    if len(set(harmonic_bins)) != len(harmonic_bins):
        raise ValueError(
            "The requested harmonics are not distinguishable at the current "
            "frequency resolution"
        )

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
    signal_matrix, L, Fs, nyquist = _validate_reconstruction_sampling(
        signal_matrix, dt
    )
    f_low = float(f_low)
    f_high = float(f_high)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        raise ValueError("Band edges must be finite")
    if f_low < 0 or f_high <= f_low:
        raise ValueError("Band edges must satisfy 0 <= f_low < f_high")
    if f_high > nyquist * (1.0 + 10.0 * np.finfo(float).eps):
        raise ValueError(
            f"Band upper edge {f_high:.6g} Hz exceeds Nyquist "
            f"frequency {nyquist:.6g} Hz"
        )

    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq = np.fft.fftfreq(L, dt)

    bin_mask = np.abs(freq) >= f_low
    bin_mask &= np.abs(freq) <= f_high
    bin_mask[0] = False

    reconstructed = reconstruct_from_bins(Y_full, bin_mask)
    return reconstructed, bin_mask


def reconstruct_from_top_frequencies(signal_matrix, dt, n_peaks=15,
                                     freq_range=None, min_peak_distance_hz=None,
                                     min_peak_prominence=None):
    """Reconstruct signal from the N strongest local spectral peaks.

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
    min_peak_prominence : float or None, optional
        Minimum FFT-magnitude prominence passed to ``scipy.signal.find_peaks``.

    Returns
    -------
    reconstructed_total : ndarray, shape (L, n_signals)
        Sum of all selected peak reconstructions.
    component_signals : dict of ndarray
        Keys like ``"123.4 kHz"``, each shape (L, n_signals).
    peak_bins : list of int
        Positive-frequency bin indices selected.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError(
            "scipy.signal is required for spectral peak reconstruction"
        ) from exc

    signal_matrix, L, Fs, nyquist = _validate_reconstruction_sampling(
        signal_matrix, dt
    )
    n_signals = signal_matrix.shape[1]
    n_peaks = int(n_peaks)
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")
    df = Fs / L

    if min_peak_distance_hz is None:
        min_peak_distance_hz = max(3.0 * df, 1000.0)

    # Full two-sided FFT
    Y_full = np.fft.fft(signal_matrix, axis=0)
    freq_pos = np.fft.rfftfreq(L, dt)

    # One-sided nonnegative frequencies. rfftfreq represents the even-length
    # Nyquist bin as +Fs/2 instead of the signed -Fs/2 returned by fftfreq.
    half = len(freq_pos)
    Y_pos = Y_full[:half, :]

    # Build candidate mask (common frequency range restriction)
    mask_pos = np.ones(half, dtype=bool)
    mask_pos[0] = False  # exclude DC
    if freq_range is not None:
        f_min, f_max = freq_range
        f_min = float(f_min)
        f_max = float(f_max)
        if not (0 <= f_min < f_max <= nyquist):
            raise ValueError(
                "freq_range must satisfy 0 <= f_min < f_max <= Nyquist"
            )
        mask_pos &= (freq_pos >= f_min) & (freq_pos <= f_max)
    min_distance_bins = max(1, int(np.ceil(min_peak_distance_hz / df)))

    # --- Per-probe peak selection ---
    # For each column, independently pick the top N peaks, then take the
    # union so that every probe's dominant frequencies are represented.
    selected_set = set()
    for col in range(n_signals):
        amp_col = np.abs(Y_pos[:, col])
        search_amp = amp_col.copy()
        search_amp[~mask_pos] = -np.inf
        peak_idx, _ = find_peaks(
            search_amp,
            distance=min_distance_bins,
            prominence=min_peak_prominence,
        )
        peak_idx = peak_idx[mask_pos[peak_idx]]
        if len(peak_idx) == 0:
            continue
        order = np.argsort(amp_col[peak_idx])[::-1]
        selected_set.update(peak_idx[order[:n_peaks]].tolist())

    selected = sorted(selected_set)
    if not selected:
        raise ValueError("No local spectral peaks satisfied the selection criteria")

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

def compute_spectrogram(
        signal, fs, nperseg=256, noverlap=None, window="hann",
        output_scale="db"):
    """Compute a power spectral density spectrogram.

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
    Sxx : ndarray
        PSD in signal-unit squared per hertz when
        ``output_scale='linear'`` or dB re that unit when
        ``output_scale='db'``.
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        raise ImportError("scipy.signal is required for compute_spectrogram")

    signal = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    nperseg = int(nperseg)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if nperseg < 8:
        raise ValueError("nperseg must be at least 8")
    if len(signal) < nperseg:
        raise ValueError(
            f"signal length ({len(signal)}) must be at least nperseg ({nperseg})"
        )
    if noverlap is None:
        noverlap = int(0.75 * nperseg)
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    output_scale = str(output_scale).lower()
    if output_scale not in ("linear", "db"):
        raise ValueError("output_scale must be 'linear' or 'db'")
    f, t, Sxx = scipy_signal.spectrogram(
        signal, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, detrend="constant", scaling="density", mode="psd",
    )
    if output_scale == "linear":
        return f, t, Sxx
    eps = 1e-20
    return f, t, 10.0 * np.log10(np.maximum(Sxx, eps))


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
    fs = float(fs)
    f_low = float(f_low)
    f_high = float(f_high)
    order = int(order)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if order < 1:
        raise ValueError("order must be at least 1")
    nyquist = fs / 2.0
    if not (0.0 < f_low < f_high < nyquist):
        raise ValueError("Band must satisfy 0 < f_low < f_high < Nyquist")
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


def estimate_packet_propagation(
        probe_x_m, time_s, envelope_matrix, baseline_end_time_s=None,
        baseline_fraction=0.15, noise_sigma=6.0, peak_fraction=0.05,
        persistent_samples=4, filter_edge_fraction=0.01):
    """Estimate band-limited packet arrival, group velocity, and energy growth.

    Arrival is the first persistent crossing of a threshold that exceeds both
    a robust pre-event noise floor and a small fraction of the local packet
    peak. Group velocity is obtained from a Theil-Sen fit of arrival time
    versus physical probe position.
    """
    try:
        from scipy.stats import theilslopes
    except ImportError as exc:
        raise ImportError("scipy.stats is required for packet propagation") from exc
    x = np.asarray(probe_x_m, dtype=float).ravel()
    time = np.asarray(time_s, dtype=float).ravel()
    envelope = np.asarray(envelope_matrix, dtype=float)
    if envelope.ndim == 1:
        envelope = envelope[:, None]
    if envelope.shape != (time.size, x.size):
        raise ValueError("envelope_matrix must have shape (n_time, n_probe)")
    if time.size < 16 or x.size < 3:
        raise ValueError("packet propagation requires >=16 samples and >=3 probes")
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(time))
            or not np.all(np.isfinite(envelope))):
        raise ValueError("packet propagation inputs must be finite")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("packet time must be strictly increasing")
    order = np.argsort(x)
    x = x[order]
    envelope = envelope[:, order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("packet probe positions must be distinct")
    persistent_samples = int(persistent_samples)
    if persistent_samples < 1:
        raise ValueError("persistent_samples must be positive")
    dt = float(np.median(np.diff(time)))
    edge_count = max(1, int(round(float(filter_edge_fraction) * time.size)))
    usable = np.zeros(time.size, dtype=bool)
    usable[edge_count:time.size - edge_count] = True
    if baseline_end_time_s is not None:
        baseline_mask = (time < float(baseline_end_time_s)) & usable
        baseline_source = "configured_pre_event"
    else:
        baseline_mask = np.zeros(time.size, dtype=bool)
        baseline_source = "leading_record_fallback"
    if np.count_nonzero(baseline_mask) < 8:
        baseline_count = max(
            8, int(round(float(baseline_fraction) * time.size))
        )
        baseline_mask = np.zeros(time.size, dtype=bool)
        baseline_mask[edge_count:min(baseline_count, time.size - edge_count)] = True
        baseline_source = "leading_record_fallback"
    if np.count_nonzero(baseline_mask) < 8:
        raise ValueError("fewer than eight usable pre-event samples are available")

    arrival = np.full(x.size, np.nan)
    peak_time = np.full(x.size, np.nan)
    peak_amplitude = np.full(x.size, np.nan)
    threshold = np.full(x.size, np.nan)
    noise_floor = np.full(x.size, np.nan)
    noise_scale = np.full(x.size, np.nan)
    energy = np.full(x.size, np.nan)
    valid = np.zeros(x.size, dtype=bool)
    kernel = np.ones(persistent_samples, dtype=int)
    for index in range(x.size):
        values = envelope[:, index]
        baseline = values[baseline_mask]
        median = float(np.median(baseline))
        sigma = float(
            1.4826 * np.median(np.abs(baseline - median))
        )
        sigma = max(sigma, np.finfo(float).eps * max(1.0, abs(median)))
        local_peak_index = int(np.argmax(np.where(usable, values, -np.inf)))
        local_peak = float(values[local_peak_index])
        local_threshold = max(
            median + float(noise_sigma) * sigma,
            median + float(peak_fraction) * max(local_peak - median, 0.0),
        )
        crossings = usable & (values >= local_threshold)
        persistent = np.convolve(
            crossings.astype(int), kernel, mode="valid"
        )
        candidate = np.flatnonzero(
            (persistent >= persistent_samples)
            & (
                np.arange(persistent.size) + persistent_samples - 1
                <= local_peak_index
            )
        )
        noise_floor[index] = median
        noise_scale[index] = sigma
        threshold[index] = local_threshold
        peak_time[index] = time[local_peak_index]
        peak_amplitude[index] = local_peak
        if candidate.size:
            arrival[index] = time[int(candidate[0])]
            valid[index] = True

    valid_count = int(np.count_nonzero(valid))
    if valid_count < 3:
        return {
            "status": "insufficient_data",
            "reason": f"only {valid_count} valid packet arrivals",
            "probe_x_m": x,
            "arrival_time_s": arrival,
            "peak_time_s": peak_time,
            "peak_amplitude": peak_amplitude,
            "envelope_energy": energy,
            "arrival_threshold": threshold,
            "noise_floor": noise_floor,
            "noise_scale": noise_scale,
            "valid": valid,
            "baseline_source": baseline_source,
        }

    # Integrate every valid probe over the same arrival-relative duration.
    # This follows the convecting packet and avoids comparing arbitrary
    # stationary full-record windows at different streamwise stations.
    peak_delay = peak_time[valid] - arrival[valid]
    positive_delay = peak_delay[
        np.isfinite(peak_delay) & (peak_delay > 0.0)
    ]
    nominal_duration = (
        2.0 * float(np.median(positive_delay))
        if positive_delay.size else persistent_samples * dt
    )
    remaining_duration = np.min(time[-1] - arrival[valid])
    packet_duration = min(
        max(nominal_duration, persistent_samples * dt),
        remaining_duration,
    )
    if packet_duration < persistent_samples * dt:
        return {
            "status": "insufficient_data",
            "reason": "common arrival-relative packet window is too short",
            "probe_x_m": x,
            "arrival_time_s": arrival,
            "peak_time_s": peak_time,
            "peak_amplitude": peak_amplitude,
            "envelope_energy": energy,
            "arrival_threshold": threshold,
            "noise_floor": noise_floor,
            "noise_scale": noise_scale,
            "valid": valid,
            "baseline_source": baseline_source,
        }
    for index in np.flatnonzero(valid):
        packet_mask = (
            (time >= arrival[index])
            & (time <= arrival[index] + packet_duration)
        )
        energy[index] = float(np.trapz(
            np.maximum(
                envelope[packet_mask, index] - noise_floor[index], 0.0
            ) ** 2,
            time[packet_mask],
        ))

    slope, intercept, slope_low, slope_high = theilslopes(
        arrival[valid], x[valid], alpha=0.95
    )
    fitted = intercept + slope * x[valid]
    residual = arrival[valid] - fitted
    total = arrival[valid] - np.mean(arrival[valid])
    r_squared = (
        1.0 - np.sum(residual ** 2) / np.sum(total ** 2)
        if np.sum(total ** 2) > 0.0 else np.nan
    )
    group_velocity = (
        1.0 / slope if slope > 0.0 else np.nan
    )
    velocity_ci = np.array([
        1.0 / slope_high if slope_high > 0.0 else np.nan,
        1.0 / slope_low if slope_low > 0.0 else np.nan,
    ])
    positive_energy = valid & (energy > 0.0)
    if np.count_nonzero(positive_energy) >= 3:
        energy_slope, energy_intercept, energy_low, energy_high = theilslopes(
            np.log(energy[positive_energy]), x[positive_energy], alpha=0.95
        )
    else:
        energy_slope = energy_intercept = energy_low = energy_high = np.nan
    return {
        "status": "complete",
        "probe_x_m": x,
        "arrival_time_s": arrival,
        "arrival_uncertainty_s": np.full(x.size, dt),
        "peak_time_s": peak_time,
        "peak_amplitude": peak_amplitude,
        "envelope_energy": energy,
        "arrival_threshold": threshold,
        "noise_floor": noise_floor,
        "noise_scale": noise_scale,
        "valid": valid,
        "baseline_source": baseline_source,
        "arrival_fit_slope_s_m": float(slope),
        "arrival_fit_intercept_s": float(intercept),
        "arrival_fit_r_squared": float(r_squared),
        "group_velocity_m_s": float(group_velocity),
        "group_velocity_ci95_m_s": velocity_ci,
        "log_energy_growth_per_m": float(energy_slope),
        "log_energy_growth_ci95_per_m": np.array(
            [energy_low, energy_high], dtype=float
        ),
        "log_energy_fit_intercept": float(energy_intercept),
        "packet_energy_window_duration_s": float(packet_duration),
        "valid_arrival_count": valid_count,
    }


# ===========================================================================
#  PHASE 3: NONLINEAR INTERACTION DIAGNOSTICS (BISPECTRUM)
# ===========================================================================

def compute_bicoherence(signal, fs, nperseg=256, noverlap=None, fmax=None):
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
    signal = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    nperseg = int(nperseg)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if nperseg < 8:
        raise ValueError("nperseg must be at least 8")
    if len(signal) < nperseg:
        raise ValueError(
            f"signal length ({len(signal)}) must be at least nperseg ({nperseg})"
        )
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    nstep = nperseg - noverlap
    window = np.hanning(nperseg)

    n_segments = (len(signal) - nperseg) // nstep + 1
    if n_segments < 2:
        raise ValueError(
            "Bicoherence requires at least two complete segments; increase "
            "the record length or reduce nperseg"
        )
    frequency_full = np.fft.rfftfreq(nperseg, 1.0 / fs)
    if fmax is None:
        n_freq = frequency_full.size
    else:
        fmax = float(fmax)
        if not 0 < fmax <= 0.5 * fs:
            raise ValueError("fmax must lie in (0, Nyquist]")
        n_freq = int(np.searchsorted(frequency_full, fmax, side="right"))
        n_freq = max(2, n_freq)

    X = np.empty((n_segments, n_freq), dtype=complex)
    for i in range(n_segments):
        start = i * nstep
        seg = signal[start:start + nperseg]
        seg = (seg - np.mean(seg)) * window
        X[i] = np.fft.rfft(seg)[:n_freq]

    # Entries outside the valid f1+f2 <= Nyquist triangle are undefined, not
    # zero-coupling measurements, so retain them as NaN.
    bicoh = np.full((n_freq, n_freq), np.nan, dtype=float)
    eps = 1e-30

    for f1 in range(1, n_freq - 1):
        for f2 in range(f1, n_freq):
            f3 = f1 + f2
            if f3 >= n_freq:
                continue
            B = np.mean(X[:, f1] * X[:, f2] * np.conj(X[:, f3]), axis=0)
            denom = np.mean(np.abs(X[:, f1] * X[:, f2]) ** 2, axis=0) * \
                    np.mean(np.abs(X[:, f3]) ** 2, axis=0)
            bicoh[f1, f2] = np.clip(np.abs(B) ** 2 / (denom + eps), 0.0, 1.0)

    freq = frequency_full[:n_freq]
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


def compute_triad_bicoherence(signal, fs, target_freqs, nperseg=256,
                              noverlap=None):
    """Compute only requested squared-bicoherence triads.

    This is numerically identical to extracting the same bins from
    :func:`compute_bicoherence`, but avoids constructing and traversing the
    full frequency-by-frequency map for every spatial probe.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    nperseg = int(nperseg)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if nperseg < 8 or len(signal) < nperseg:
        raise ValueError("signal must contain at least nperseg >= 8 samples")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    step = nperseg - noverlap
    starts = np.arange(0, len(signal) - nperseg + 1, step, dtype=int)
    if starts.size < 2:
        raise ValueError("Bicoherence requires at least two complete segments")
    window = np.hanning(nperseg)
    spectra = np.empty((starts.size, nperseg // 2 + 1), dtype=complex)
    for row, start in enumerate(starts):
        segment = signal[start:start + nperseg]
        spectra[row] = np.fft.rfft((segment - np.mean(segment)) * window)

    freq = np.fft.rfftfreq(nperseg, 1.0 / fs)
    bins = {float(value): int(np.argmin(np.abs(freq - value)))
            for value in target_freqs}
    eps = 1.0e-30
    result = {}
    for fi in target_freqs:
        i = bins[float(fi)]
        for fj in target_freqs:
            j = bins[float(fj)]
            if fi + fj > freq[-1] or i + j >= len(freq):
                continue
            i0, j0 = sorted((i, j))
            product = spectra[:, i0] * spectra[:, j0]
            summed = spectra[:, i0 + j0]
            bispectrum = np.mean(product * np.conj(summed))
            denominator = np.mean(np.abs(product) ** 2) * np.mean(
                np.abs(summed) ** 2
            )
            value = np.clip(np.abs(bispectrum) ** 2 / (denominator + eps), 0.0, 1.0)
            result[f"b²({fi:.3e}, {fj:.3e})"] = float(value)
    return result


def compute_surrogate_triad_significance(
        signal, fs, target_freqs, nperseg=256, noverlap=None,
        n_surrogates=200, fdr_alpha=0.05,
        minimum_independent_segments=8, random_seed=0,
        laser_frequency_hz=None):
    """Test requested bicoherence triads against phase-randomized surrogates.

    The surrogates retain the observed Fourier magnitudes but randomize phase.
    Empirical one-sided p-values are corrected across the requested triad
    family with Benjamini-Hochberg false-discovery-rate control.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    nperseg = int(nperseg)
    n_surrogates = int(n_surrogates)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be positive and finite")
    if nperseg < 8 or signal.size < nperseg:
        raise ValueError("signal must contain at least nperseg >= 8 samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values")
    if n_surrogates < 19:
        raise ValueError("at least 19 surrogates are required")
    fdr_alpha = float(fdr_alpha)
    if not 0.0 < fdr_alpha < 1.0:
        raise ValueError("fdr_alpha must lie in (0, 1)")
    minimum_independent_segments = int(minimum_independent_segments)
    if minimum_independent_segments < 2:
        raise ValueError("minimum_independent_segments must be at least 2")
    independent_segments = signal.size // nperseg
    if independent_segments < minimum_independent_segments:
        raise ValueError(
            f"only {independent_segments} non-overlapping segments are "
            f"available; {minimum_independent_segments} required"
        )
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    observed = compute_triad_bicoherence(
        signal, fs, target_freqs, nperseg=nperseg, noverlap=noverlap
    )
    # Bicoherence is symmetric in (f1, f2).  Test each physical triad only
    # once so the multiple-comparison family is not inflated by mirrored
    # duplicates.
    targets = [float(value) for value in target_freqs]
    labels = []
    for first_index, first in enumerate(targets):
        for second in targets[first_index:]:
            label = f"b²({first:.3e}, {second:.3e})"
            if label in observed:
                labels.append(label)
    labels = sorted(set(labels))
    if not labels:
        raise ValueError("no requested triads lie below Nyquist")
    minimum_surrogates = int(np.ceil(
        len(labels) / float(fdr_alpha)
    ) - 1)
    if n_surrogates < minimum_surrogates:
        raise ValueError(
            f"{n_surrogates} surrogates cannot resolve the first "
            f"Benjamini-Hochberg threshold for {len(labels)} unique triads "
            f"at alpha={float(fdr_alpha):g}; at least "
            f"{minimum_surrogates} are required"
        )
    observed_values = np.asarray([observed[label] for label in labels])
    # Form exactly the same windowed ensemble used by the observed estimate.
    # Randomizing one phase per frequency for the *whole* record would leave
    # phase locking between Welch segments intact and therefore provide a
    # misleading null for deterministic tones.  Instead, independently
    # randomize each segment-frequency phase while retaining every segment's
    # observed spectral magnitude.
    step = nperseg - noverlap
    starts = np.arange(0, signal.size - nperseg + 1, step, dtype=int)
    window = np.hanning(nperseg)
    spectra = np.empty(
        (starts.size, nperseg // 2 + 1), dtype=complex
    )
    for row, start in enumerate(starts):
        segment = signal[start:start + nperseg]
        spectra[row] = np.fft.rfft(
            (segment - np.mean(segment)) * window
        )
    magnitudes = np.abs(spectra)
    frequency_grid = np.fft.rfftfreq(nperseg, 1.0 / fs)
    bins = {
        float(value): int(np.argmin(np.abs(frequency_grid - value)))
        for value in target_freqs
    }
    triad_bins = {}
    for first in target_freqs:
        first_bin = bins[float(first)]
        for second in target_freqs:
            second_bin = bins[float(second)]
            if (
                    first + second > frequency_grid[-1]
                    or first_bin + second_bin >= frequency_grid.size):
                continue
            label = f"b²({first:.3e}, {second:.3e})"
            first_sorted, second_sorted = sorted(
                (first_bin, second_bin)
            )
            triad_bins[label] = (
                first_sorted, second_sorted,
                first_sorted + second_sorted,
            )
    rng = np.random.default_rng(random_seed)
    eps = 1.0e-30
    surrogate_values = np.empty((n_surrogates, len(labels)), dtype=float)
    for surrogate_index in range(n_surrogates):
        phase = rng.uniform(0.0, 2.0 * np.pi, spectra.shape)
        phase[:, 0] = 0.0
        randomized = magnitudes * np.exp(1j * phase)
        randomized[:, 0] = 0.0
        values = []
        for label in labels:
            first_bin, second_bin, summed_bin = triad_bins[label]
            product = (
                randomized[:, first_bin]
                * randomized[:, second_bin]
            )
            summed = randomized[:, summed_bin]
            bispectrum = np.mean(product * np.conj(summed))
            denominator = (
                np.mean(np.abs(product) ** 2)
                * np.mean(np.abs(summed) ** 2)
            )
            values.append(float(np.clip(
                np.abs(bispectrum) ** 2 / (denominator + eps),
                0.0, 1.0,
            )))
        surrogate_values[surrogate_index] = values
    p_value = (
        1.0 + np.sum(surrogate_values >= observed_values[None, :], axis=0)
    ) / (n_surrogates + 1.0)
    order = np.argsort(p_value)
    ranked = p_value[order]
    thresholds = (
        float(fdr_alpha)
        * np.arange(1, len(labels) + 1) / len(labels)
    )
    accepted_rank = np.flatnonzero(ranked <= thresholds)
    significant = np.zeros(len(labels), dtype=bool)
    if accepted_rank.size:
        cutoff = ranked[int(accepted_rank[-1])]
        significant = p_value <= cutoff
    adjusted = np.empty_like(p_value)
    monotone = np.minimum.accumulate(
        (ranked * len(labels) / np.arange(1, len(labels) + 1))[::-1]
    )[::-1]
    adjusted[order] = np.minimum(monotone, 1.0)

    laser_related = np.zeros(len(labels), dtype=bool)
    if laser_frequency_hz is not None:
        tolerance = (
            frequency_grid[1] - frequency_grid[0]
            if frequency_grid.size > 1 else 0.0
        )
        label_laser = {}
        for first in targets:
            for second in targets:
                if first + second > frequency_grid[-1]:
                    continue
                label = f"b²({first:.3e}, {second:.3e})"
                values = (first, second, first + second)
                label_laser[label] = any(
                    abs(value / float(laser_frequency_hz)
                        - round(value / float(laser_frequency_hz)))
                    * float(laser_frequency_hz) <= tolerance
                    for value in values
                )
        laser_related = np.asarray([
            label_laser.get(label, False) for label in labels
        ], dtype=bool)
    return {
        "triad_labels": np.asarray(labels),
        "observed_bicoherence_squared": observed_values,
        "surrogate_median_bicoherence_squared": np.median(
            surrogate_values, axis=0
        ),
        "surrogate_95_bicoherence_squared": np.quantile(
            surrogate_values, 0.95, axis=0
        ),
        "empirical_p_value": p_value,
        "fdr_adjusted_p_value": adjusted,
        "significant_fdr": significant,
        "laser_harmonic_related": laser_related,
        "n_surrogates": n_surrogates,
        "unique_triad_hypothesis_count": len(labels),
        "minimum_empirical_p_value": 1.0 / (n_surrogates + 1.0),
        "independent_segment_count": independent_segments,
        "fdr_alpha": float(fdr_alpha),
        "null_model": (
            "independent segment-frequency phase randomization with each "
            "windowed segment magnitude retained"
        ),
    }


def classify_measured_dynamics(
        wave_report=None, packet_report=None, nonlinear_report=None,
        modal_summary=None, force_report=None):
    """Build a conservative evidence matrix for measured flow/load behavior.

    This is deliberately not an LST classifier.  It records whether the
    available measurements support coherent propagation, a convecting
    transient packet, statistically significant quadratic coupling, robust
    descriptive modal structure, and coupling to aerodynamic loads.
    """
    thresholds = {
        "minimum_wave_domain_accepted_fraction": 0.10,
        "minimum_packet_arrival_fit_r_squared": 0.80,
        "minimum_pod_subspace_cosine": 0.90,
        "maximum_dmd_dominant_frequency_relative_range": 0.10,
        "maximum_dmd_retained_condition_number": 1.0e8,
    }

    wave_report = wave_report or {}
    phase_fraction = float(
        wave_report.get("phase_fit_accepted_fraction", np.nan)
    )
    growth_fraction = float(
        wave_report.get(
            "complex_wavenumber_accepted_fraction", np.nan
        )
    )
    coherent_supported = (
        np.isfinite(phase_fraction)
        and phase_fraction >= thresholds[
            "minimum_wave_domain_accepted_fraction"
        ]
    )
    amplification_supported = (
        np.isfinite(growth_fraction)
        and growth_fraction >= thresholds[
            "minimum_wave_domain_accepted_fraction"
        ]
    )
    wave_evidence = {
        "status": (
            "supported" if coherent_supported else
            "not_supported" if wave_report else "not_available"
        ),
        "phase_fit_accepted_fraction": phase_fraction,
        "spatial_amplification_status": (
            "supported" if amplification_supported else
            "not_supported" if wave_report else "not_available"
        ),
        "complex_wavenumber_accepted_fraction": growth_fraction,
        "meaning": (
            "frequency-resolved coherent propagation measured on the probe "
            "line; this does not identify an instability eigenmode"
        ),
    }

    packet_report = packet_report or {}
    packet_r2 = float(
        packet_report.get("arrival_fit_r_squared", np.nan)
    )
    packet_speed = float(
        packet_report.get("group_velocity_m_s", np.nan)
    )
    packet_supported = (
        packet_report.get("status") == "complete"
        and np.isfinite(packet_speed) and packet_speed > 0.0
        and np.isfinite(packet_r2)
        and packet_r2 >= thresholds[
            "minimum_packet_arrival_fit_r_squared"
        ]
    )
    packet_evidence = {
        "status": (
            "supported" if packet_supported else
            "not_supported" if packet_report else "not_available"
        ),
        "analysis_status": packet_report.get("status", "not_available"),
        "group_velocity_m_s": packet_speed,
        "arrival_fit_r_squared": packet_r2,
        "baseline_source": packet_report.get(
            "baseline_source", "not_available"
        ),
        "meaning": (
            "band-limited packet kinematics from arrival-time regression"
        ),
    }

    nonlinear_report = nonlinear_report or {}
    probe_results = nonlinear_report.get("probe_results", [])
    complete_nonlinear = [
        item for item in probe_results if item.get("status") == "complete"
    ]
    nonlaser_count = int(sum(
        item.get("significant_nonlaser_triad_count", 0)
        for item in complete_nonlinear
    ))
    laser_count = int(sum(
        item.get("significant_laser_related_triad_count", 0)
        for item in complete_nonlinear
    ))
    nonlinear_evidence = {
        "status": (
            "supported" if nonlaser_count > 0 else
            "not_detected" if complete_nonlinear else
            "insufficient_data" if nonlinear_report else "not_available"
        ),
        "significant_nonlaser_triad_count": nonlaser_count,
        "laser_harmonic_status": (
            "supported" if laser_count > 0 else
            "not_detected" if complete_nonlinear else
            "insufficient_data" if nonlinear_report else "not_available"
        ),
        "significant_laser_related_triad_count": laser_count,
        "probes_with_complete_significance_tests": len(
            complete_nonlinear
        ),
        "meaning": (
            "FDR-controlled surrogate significance; laser-related triads "
            "are not counted as downstream mode-mode coupling"
        ),
    }

    modal_summary = modal_summary or {}
    pod_cosines = np.asarray(
        modal_summary.get("pod_subspace_cosines", []), dtype=float
    )
    dmd_frequency = np.asarray(
        modal_summary.get("dmd_dominant_frequency_hz", []), dtype=float
    )
    dmd_condition = np.asarray(
        modal_summary.get("dmd_condition_number", []), dtype=float
    )
    finite_frequency = dmd_frequency[
        np.isfinite(dmd_frequency) & (dmd_frequency > 0.0)
    ]
    if finite_frequency.size:
        frequency_relative_range = float(
            np.ptp(finite_frequency)
            / max(np.median(finite_frequency), 1.0e-300)
        )
    else:
        frequency_relative_range = np.nan
    minimum_pod_cosine = (
        float(np.min(pod_cosines[np.isfinite(pod_cosines)]))
        if np.any(np.isfinite(pod_cosines)) else np.nan
    )
    maximum_condition = (
        float(np.max(dmd_condition[np.isfinite(dmd_condition)]))
        if np.any(np.isfinite(dmd_condition)) else np.nan
    )
    modal_available = bool(modal_summary)
    modal_robust = (
        modal_available
        and np.isfinite(minimum_pod_cosine)
        and minimum_pod_cosine >= thresholds[
            "minimum_pod_subspace_cosine"
        ]
        and np.isfinite(frequency_relative_range)
        and frequency_relative_range <= thresholds[
            "maximum_dmd_dominant_frequency_relative_range"
        ]
        and np.isfinite(maximum_condition)
        and maximum_condition <= thresholds[
            "maximum_dmd_retained_condition_number"
        ]
    )
    modal_evidence = {
        "status": (
            "robust_descriptive_structure" if modal_robust else
            "sensitivity_not_passed" if modal_available else
            "not_available"
        ),
        "minimum_pod_subspace_cosine": minimum_pod_cosine,
        "dmd_dominant_frequency_relative_range": (
            frequency_relative_range
        ),
        "maximum_dmd_retained_condition_number": maximum_condition,
        "meaning": (
            "POD/SPOD/DMD robustness screening only; no eigenmode "
            "attribution"
        ),
    }

    force_report = force_report or {}
    adequacy = force_report.get("scientific_adequacy", {})
    linkage = adequacy.get("linkage", {})
    probe_force = adequacy.get("probe_force_linkage", {})
    load_status = (
        "supported_second_order"
        if linkage.get("status") == "spectral_analysis_complete"
        else "supported_transient"
        if (
            force_report
            and adequacy.get("classification")
            == "short_transient_validation"
        )
        else "not_available"
    )
    load_evidence = {
        "status": load_status,
        "force_classification": adequacy.get(
            "classification", "not_available"
        ),
        "probe_force_linkage_status": probe_force.get(
            "status", "not_available"
        ),
        "meaning": (
            "one-sided flat-plate force response under the certified wall-"
            "traction convention"
        ),
    }

    classifications = []
    if coherent_supported:
        classifications.append("coherent_wave_propagation")
    if amplification_supported:
        classifications.append("measured_spatial_amplification")
    if packet_supported:
        classifications.append("convecting_transient_wavepacket")
    if nonlaser_count > 0:
        classifications.append(
            "statistically_supported_quadratic_flow_coupling"
        )
    if laser_count > 0:
        classifications.append("laser_harmonic_generation")
    if modal_robust:
        classifications.append("window_robust_descriptive_modal_structure")
    if load_status != "not_available":
        classifications.append("aerodynamic_load_response")

    limitations = []
    if not wave_report:
        limitations.append("coherent propagation report unavailable")
    if packet_report.get("baseline_source") == "leading_record_fallback":
        limitations.append(
            "packet arrival used a leading-record noise estimate because "
            "no adequate pre-trigger interval was available"
        )
    if nonlinear_report and not complete_nonlinear:
        limitations.append(
            "nonlinear significance was withheld because the independent-"
            "segment requirement was not met"
        )
    if modal_available and not modal_robust:
        limitations.append(
            "descriptive modal results did not pass the configured window/"
            "rank robustness screen"
        )

    return {
        "schema_version": 1,
        "scope": (
            "measurement-based transient, coherent, nonlinear, modal, and "
            "aerodynamic-load evidence"
        ),
        "excluded_scope": {
            "LST_PSE": (
                "not performed and not inferred; eigenfunction-based "
                "instability attribution is outside the project scope"
            )
        },
        "decision_thresholds": thresholds,
        "evidence": {
            "coherent_wave": wave_evidence,
            "transient_packet": packet_evidence,
            "quadratic_nonlinearity": nonlinear_evidence,
            "modal_robustness": modal_evidence,
            "aerodynamic_loads": load_evidence,
        },
        "supported_classifications": classifications,
        "limitations": limitations,
        "interpretation": (
            "Classifications are non-exclusive because a laser-forced flow "
            "can simultaneously contain coherent propagation, transient "
            "evolution, nonlinear coupling, and load response."
        ),
    }
