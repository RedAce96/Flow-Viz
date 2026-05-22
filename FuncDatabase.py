# This file is to serve as a function database for the MFC post-processing

import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap 
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================================================================= #
#                       PELEC PLOTFILE READ FUNCTIONS                       #
# ========================================================================= #

_PELEC_DEFAULT_FIELDS = {
    'Temperature': 'Temp',
    'Pressure': 'pressure',
    'Density': 'density',
    'u': 'x_velocity',
    'v': 'y_velocity',
}

_PELEC_FIELD_ALIASES = {
    'Temperature': 'Temp',
    'Pressure': 'pressure',
    'Density': 'density',
    'u': 'x_velocity',
    'v': 'y_velocity',
    'VelocityX': 'x_velocity',
    'VelocityY': 'y_velocity',
    'Temp': 'Temp',
    'pressure': 'pressure',
    'density': 'density',
    'vfrac': 'vfrac',
    'volume_fraction': 'vfrac',
    'x_velocity': 'x_velocity',
    'y_velocity': 'y_velocity',
    'MachNumber': 'MachNumber',
}

_PELEC_MKS_SCALES = {
    'density': 1.0e3,          # g/cm^3 -> kg/m^3
    'pressure': 1.0e-1,        # dyne/cm^2 -> Pa
    'x_velocity': 1.0e-2,      # cm/s -> m/s
    'y_velocity': 1.0e-2,      # cm/s -> m/s
    'z_velocity': 1.0e-2,      # cm/s -> m/s
    'viscosity': 1.0e-1,       # g/(cm s) -> Pa s
    'bulk_viscosity': 1.0e-1,  # g/(cm s) -> Pa s
    'conductivity': 1.0e-5,    # erg/(cm s K) -> W/(m K)
}


def _natural_sort_key(name):
    parts = re.split(r'(\d+)', str(name))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _resolve_pelec_field_map(field_names):
    if field_names is None:
        return dict(_PELEC_DEFAULT_FIELDS)

    if isinstance(field_names, dict):
        return {
            out_name: _PELEC_FIELD_ALIASES.get(raw_name, raw_name)
            for out_name, raw_name in field_names.items()
        }

    return {
        str(name): _PELEC_FIELD_ALIASES.get(str(name), str(name))
        for name in field_names
    }


def _pelec_to_mks(field_name, values):
    scale = _PELEC_MKS_SCALES.get(field_name, 1.0)
    return values * scale


def _collapse_pelec_field(values, dimensionality):
    values = np.asarray(values)

    if dimensionality == 2 and values.ndim == 3:
        return values[:, :, 0]

    if dimensionality == 1 and values.ndim >= 2:
        return np.asarray(values).reshape(values.shape[0])

    return values


def _structured_1d_coordinates(x_data, y_data):
    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)

    if x_arr.ndim == 2:
        x_arr = x_arr[:, 0]
    if y_arr.ndim == 2:
        y_arr = y_arr[0, :]

    return np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float)


def _unpack_structured_dataset(data):
    if isinstance(data, tuple) and len(data) == 3:
        xx, yy, fields = data
        x_line, y_line = _structured_1d_coordinates(xx, yy)
        return x_line, y_line, fields, {}

    if isinstance(data, dict):
        if {'x', 'y', 'fields'} <= set(data.keys()):
            return (
                np.asarray(data['x'], dtype=float),
                np.asarray(data['y'], dtype=float),
                data['fields'],
                data,
            )

        if {'xx', 'yy', 'fields'} <= set(data.keys()):
            x_line, y_line = _structured_1d_coordinates(data['xx'], data['yy'])
            return x_line, y_line, data['fields'], data

    raise TypeError(
        'Structured data must be either (xx, yy, fields_data) or a dict '
        "with keys 'x', 'y', and 'fields'."
    )


def load_pelec_plotfile(plotfile_path, field_names=None, convert_to_mks=True):
    """
    Load a single PeleC/AMReX plotfile into structured 2-D numpy arrays.

    Parameters
    ----------
    plotfile_path : str or path-like
        Path to a single PeleC plotfile directory.
    field_names : iterable or dict, optional
        Fields to load. If omitted, a comparison-friendly default set is used.
        A dict maps output names to raw PeleC field names.
    convert_to_mks : bool, default=True
        Convert PeleC CGS outputs to MKS units for easier comparison with MFC.

    Returns
    -------
    dict
        Structured dataset with 1-D x/y coordinates, 2-D field arrays, and
        plotfile metadata.
    """
    try:
        import yt
    except ImportError as exc:
        raise ImportError(
            "yt is required to read PeleC AMReX plotfiles. Install it with 'pip install yt'."
        ) from exc

    plotfile_path = os.fspath(plotfile_path)
    field_map = _resolve_pelec_field_map(field_names)

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
    fields = {}
    raw_field_names = {}
    for output_name, raw_name in field_map.items():
        if raw_name not in available_fields:
            raise KeyError(
                f"Field '{raw_name}' was not found in plotfile {plotfile_path}. "
                f"Available fields include: {sorted(available_fields)[:12]}"
            )

        values = _collapse_pelec_field(cover[('boxlib', raw_name)], ds.dimensionality)
        values = np.asarray(values).squeeze()
        if convert_to_mks:
            values = _pelec_to_mks(raw_name, values)
        fields[output_name] = np.asarray(values, dtype=float)
        raw_field_names[output_name] = raw_name

    return {
        'plotfile': plotfile_path,
        'case_label': os.path.basename(os.path.dirname(plotfile_path)),
        'plot_label': os.path.basename(plotfile_path),
        'time': float(ds.current_time),
        'x': x,
        'y': y,
        'fields': fields,
        'raw_field_names': raw_field_names,
        'available_fields': sorted(available_fields),
        'units': 'MKS' if convert_to_mks else 'CGS',
    }


def load_pelec_plotfile_series(plot_source, plot_prefix='plt', field_names=None, convert_to_mks=True):
    """
    Discover and load a series of PeleC plotfiles from a case directory.

    Parameters
    ----------
    plot_source : str, path-like, or iterable of paths
        Case directory containing plotfiles, a single plotfile directory, or an
        explicit iterable of plotfile paths.
    plot_prefix : str, default='plt'
        Directory prefix used when discovering plotfiles inside a case folder.
    field_names : iterable or dict, optional
        Forwarded to ``load_pelec_plotfile``.
    convert_to_mks : bool, default=True
        Convert output coordinates and supported fields into MKS units.

    Returns
    -------
    list[dict]
        One structured dataset per plotfile, sorted by time/name.
    """
    if isinstance(plot_source, (str, os.PathLike)):
        source_path = os.path.abspath(os.fspath(plot_source))

        if os.path.isdir(source_path) and os.path.isfile(os.path.join(source_path, 'Header')):
            plotfiles = [source_path]
        elif os.path.isdir(source_path):
            plotfiles = []
            for entry in os.listdir(source_path):
                entry_path = os.path.join(source_path, entry)
                if not os.path.isdir(entry_path):
                    continue
                if not entry.startswith(plot_prefix):
                    continue
                if '.old.' in entry:
                    continue
                if not os.path.isfile(os.path.join(entry_path, 'Header')):
                    continue
                plotfiles.append(entry_path)
            plotfiles.sort(key=lambda path: _natural_sort_key(os.path.basename(path)))
        else:
            raise FileNotFoundError(f'Could not find plotfile source: {plot_source}')
    else:
        plotfiles = [os.path.abspath(os.fspath(path)) for path in plot_source]
        plotfiles.sort(key=lambda path: _natural_sort_key(os.path.basename(path)))

    return [
        load_pelec_plotfile(
            plotfile_path,
            field_names=field_names,
            convert_to_mks=convert_to_mks,
        )
        for plotfile_path in plotfiles
    ]


def extract_surface_normal_profile(
    data,
    field_key,
    x_location,
    surface_y=0.0,
    y_max=None,
    side='positive',
    label=None,
):
    """
    Extract a wall-normal line profile from structured 2-D data.

    For the flat-plate cases, the surface normal is the vertical ``+y``
    direction, so this function returns ``field(y)`` at the nearest available
    x-location.

    Parameters
    ----------
    data : tuple or dict
        Either ``(xx, yy, fields_data)`` from ``Silo_Read`` or a structured
        dataset returned by ``load_pelec_plotfile``.
    field_key : str
        Name of the field to extract.
    x_location : float
        Target x-location for the profile.
    surface_y : float, default=0.0
        Physical surface location.
    y_max : float, optional
        Optional maximum wall-normal coordinate to keep.
    side : {'positive', 'negative'}, default='positive'
        Direction along the normal from the surface.
    label : str, optional
        Legend label for later plotting.

    Returns
    -------
    dict
        Extracted 1-D profile and metadata.
    """
    x_line, y_line, fields, metadata = _unpack_structured_dataset(data)

    if field_key not in fields:
        raise KeyError(f"Field '{field_key}' was not found in the provided dataset.")

    values_2d = np.asarray(fields[field_key]).squeeze()
    if values_2d.ndim != 2:
        raise ValueError(
            f"Field '{field_key}' must be a 2-D array after squeezing; got shape {values_2d.shape}."
        )

    x_index = int(np.argmin(np.abs(x_line - x_location)))
    sampled_x = float(x_line[x_index])

    if side.lower() not in {'positive', 'negative'}:
        raise ValueError("side must be either 'positive' or 'negative'.")

    if side.lower() == 'positive':
        mask = y_line >= surface_y
        if y_max is not None:
            mask &= y_line <= y_max
        y_profile = y_line[mask]
        field_profile = values_2d[x_index, mask]
        distance = y_profile - surface_y
    else:
        mask = y_line <= surface_y
        if y_max is not None:
            mask &= y_line >= y_max
        y_profile = y_line[mask][::-1]
        field_profile = values_2d[x_index, mask][::-1]
        distance = surface_y - y_profile

    if y_profile.size == 0:
        raise ValueError(
            'No points were selected for the requested wall-normal interval. '
            'Check x_location, surface_y, and y_max.'
        )

    if label is None:
        plot_label = metadata.get('plot_label')
        case_label = metadata.get('case_label')
        time_value = metadata.get('time')
        if plot_label is not None and time_value is not None:
            label = f"{case_label}: {plot_label} (t={time_value:.3e} s)"
        elif time_value is not None:
            label = f"t={time_value:.3e} s"
        else:
            label = field_key

    return {
        'field_key': field_key,
        'label': label,
        'time': metadata.get('time'),
        'x_requested': float(x_location),
        'x_sampled': sampled_x,
        'surface_y': float(surface_y),
        'y': np.asarray(y_profile, dtype=float),
        'distance_from_surface': np.asarray(distance, dtype=float),
        'values': np.asarray(field_profile, dtype=float),
        'source': metadata.get('plotfile'),
        'units': metadata.get('units'),
    }


def plot_surface_normal_profiles(
    profiles,
    field_label=None,
    title=None,
    output_path=None,
    x_key='distance_from_surface',
    xlabel='Distance From Surface [m]',
    ax=None,
):
    """
    Plot one or more wall-normal line profiles on a shared axis.

    Parameters
    ----------
    profiles : sequence of dict
        Profile dictionaries returned by ``extract_surface_normal_profile``.
    field_label : str, optional
        Y-axis label. Defaults to the field key of the first profile.
    title : str, optional
        Figure title.
    output_path : str, optional
        If provided, save the figure to this path.
    x_key : str, default='distance_from_surface'
        Horizontal-axis key to plot from each profile.
    xlabel : str, default='Distance From Surface [m]'
        Horizontal-axis label.
    ax : matplotlib axis, optional
        Existing axis to draw into.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the comparison plot.
    """
    if len(profiles) == 0:
        raise ValueError('At least one profile is required for plotting.')

    created_figure = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        created_figure = True

    for profile in profiles:
        ax.plot(
            profile['values'],
            profile[x_key],
            linewidth=2.0,
            linestyle=profile.get('linestyle', '-'),
            label=profile.get('label'),
        )

    if field_label is None:
        field_label = profiles[0].get('field_key', 'Field')

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(field_label, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)

    if any(profile.get('label') for profile in profiles):
        ax.legend()

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=200)
    elif created_figure:
        ax.figure.tight_layout()

    return ax


def extract_streamline_field(
    data,
    u_key='u',
    v_key='v',
    color_key=None,
    mask_key=None,
    mask_threshold=None,
    mask_mode='below',
    label=None,
):
    """
    Extract in-plane velocity components for steady streamline plotting.

    Parameters
    ----------
    data : tuple or dict
        Structured dataset returned by ``Silo_Read`` or ``load_pelec_plotfile``.
    u_key, v_key : str, default=('u', 'v')
        Field names for the in-plane velocity components.
    color_key : str, optional
        Optional field used to color the streamlines.
    mask_key : str, optional
        Optional scalar field used to suppress streamlines in masked cells.
    mask_threshold : float, optional
        Threshold applied to ``mask_key``.
    mask_mode : {'below', 'above'}, default='below'
        Mask cells where ``mask_key < mask_threshold`` or
        ``mask_key > mask_threshold``.
    label : str, optional
        Optional label for legends or metadata.

    Returns
    -------
    dict
        Structured streamline payload with 1-D coordinates and 2-D fields.
    """
    x_line, y_line, fields, metadata = _unpack_structured_dataset(data)

    missing_keys = [key for key in (u_key, v_key) if key not in fields]
    if missing_keys:
        raise KeyError(f"Velocity field(s) {missing_keys} were not found in the provided dataset.")

    u_values = np.asarray(fields[u_key], dtype=float).squeeze()
    v_values = np.asarray(fields[v_key], dtype=float).squeeze()
    if u_values.ndim != 2 or v_values.ndim != 2:
        raise ValueError(
            f"Velocity fields must be 2-D after squeezing; got {u_values.shape} and {v_values.shape}."
        )

    streamline_color = None
    resolved_color_key = color_key
    if color_key is not None:
        if color_key in fields:
            streamline_color = np.asarray(fields[color_key], dtype=float).squeeze()
        elif color_key in {'speed', 'velocity_magnitude', 'vel_mag'}:
            streamline_color = np.sqrt(u_values**2 + v_values**2)
            resolved_color_key = 'vel_mag'
        else:
            raise KeyError(f"Color field '{color_key}' was not found in the provided dataset.")

        if streamline_color.ndim != 2:
            raise ValueError(
                f"Color field '{color_key}' must be 2-D after squeezing; got shape {streamline_color.shape}."
            )

    streamline_mask = None
    resolved_mask_key = mask_key
    if mask_key is not None:
        if mask_key not in fields:
            raise KeyError(f"Mask field '{mask_key}' was not found in the provided dataset.")
        if mask_threshold is None:
            raise ValueError('mask_threshold is required when mask_key is provided.')

        mask_values = np.asarray(fields[mask_key], dtype=float).squeeze()
        if mask_values.ndim != 2:
            raise ValueError(
                f"Mask field '{mask_key}' must be 2-D after squeezing; got shape {mask_values.shape}."
            )

        normalized_mask_mode = str(mask_mode).lower()
        if normalized_mask_mode == 'below':
            streamline_mask = mask_values < float(mask_threshold)
        elif normalized_mask_mode == 'above':
            streamline_mask = mask_values > float(mask_threshold)
        else:
            raise ValueError("mask_mode must be either 'below' or 'above'.")

    if label is None:
        plot_label = metadata.get('plot_label')
        case_label = metadata.get('case_label')
        time_value = metadata.get('time')
        if plot_label is not None and time_value is not None:
            label = f"{case_label}: {plot_label} (t={time_value:.3e} s)"
        elif plot_label is not None:
            label = plot_label
        elif time_value is not None:
            label = f"t={time_value:.3e} s"
        else:
            label = f"{u_key}/{v_key}"

    return {
        'label': label,
        'time': metadata.get('time'),
        'x': np.asarray(x_line, dtype=float),
        'y': np.asarray(y_line, dtype=float),
        'u': u_values,
        'v': v_values,
        'color': streamline_color,
        'color_key': resolved_color_key,
        'mask': streamline_mask,
        'mask_key': resolved_mask_key,
        'mask_threshold': mask_threshold,
        'u_key': u_key,
        'v_key': v_key,
        'source': metadata.get('plotfile'),
        'units': metadata.get('units'),
    }


def generate_streamline_start_points(
    x_line,
    y_line,
    count=40,
    x_position=None,
    y_min=None,
    y_max=None,
    spacing='uniform',
    power=2.0,
):
    """
    Generate streamline seed points with optional wall-biased y-spacing.

    Parameters
    ----------
    x_line, y_line : array-like
        Structured 1-D coordinates for the streamline field.
    count : int, default=40
        Number of seed points to generate.
    x_position : float, optional
        Streamwise seed location. Defaults to the first x-coordinate.
    y_min, y_max : float, optional
        Vertical range used for seeding. Defaults to the full y-domain.
    spacing : {'uniform', 'power', 'sqrt'}, default='uniform'
        Distribution of seed points in y.
    power : float, default=2.0
        Exponent used when ``spacing='power'``. Values larger than one
        cluster seed points near ``y_min``.

    Returns
    -------
    np.ndarray, shape (count, 2)
        Streamline seed points suitable for ``Axes.streamplot``.
    """
    x_arr = np.asarray(x_line, dtype=float)
    y_arr = np.asarray(y_line, dtype=float)

    if count <= 0:
        raise ValueError('count must be positive when generating streamline seed points.')

    if x_position is None:
        x_position = float(x_arr[0])
    if y_min is None:
        y_min = float(np.min(y_arr))
    if y_max is None:
        y_max = float(np.max(y_arr))

    if y_max <= y_min:
        raise ValueError('y_max must be greater than y_min for streamline seeding.')

    eta = np.linspace(0.0, 1.0, int(count))
    spacing_name = str(spacing).lower()
    if spacing_name == 'uniform':
        y_eta = eta
    elif spacing_name == 'power':
        y_eta = eta ** float(power)
    elif spacing_name == 'sqrt':
        y_eta = np.sqrt(eta)
    else:
        raise ValueError("spacing must be one of 'uniform', 'power', or 'sqrt'.")

    y_values = y_min + (y_max - y_min) * y_eta
    x_values = np.full_like(y_values, float(x_position), dtype=float)
    return np.column_stack([x_values, y_values])


def generate_multiple_streamline_start_points(x_line, y_line, seed_lines):
    """
    Generate streamline seed points from multiple seed-line specifications.

    Parameters
    ----------
    x_line, y_line : array-like
        Structured 1-D coordinates for the streamline field.
    seed_lines : sequence of dict
        Each dict is forwarded to ``generate_streamline_start_points`` and may
        define ``count``, ``x_position``, ``y_min``, ``y_max``, ``spacing``,
        and ``power``.

    Returns
    -------
    np.ndarray, shape (n_points, 2)
        Concatenated streamline seed points.
    """
    if seed_lines is None or len(seed_lines) == 0:
        return np.empty((0, 2), dtype=float)

    point_sets = []
    for seed_line in seed_lines:
        if not isinstance(seed_line, dict):
            raise TypeError('Each seed line must be a dict of streamline seed parameters.')

        point_sets.append(
            generate_streamline_start_points(
                x_line,
                y_line,
                count=seed_line.get('count', 40),
                x_position=seed_line.get('x_position'),
                y_min=seed_line.get('y_min'),
                y_max=seed_line.get('y_max'),
                spacing=seed_line.get('spacing', 'uniform'),
                power=seed_line.get('power', 2.0),
            )
        )

    return np.vstack(point_sets)


def plot_streamlines(
    streamline_sets,
    title=None,
    output_path=None,
    xlim=None,
    ylim=None,
    density=1.5,
    linewidth=1.0,
    arrowsize=1.0,
    color=None,
    cmap='viridis',
    start_points=None,
    seed_lines=None,
    seed_count=None,
    seed_x_position=None,
    seed_y_min=None,
    seed_y_max=None,
    seed_spacing='uniform',
    seed_power=2.0,
    broken_streamlines=True,
    colorbar=None,
    colorbar_label=None,
    mask_fill_color='black',
    mask_fill_alpha=1.0,
    ax=None,
):
    """
    Plot one or more steady streamline fields on a shared axis.

    Parameters
    ----------
    streamline_sets : sequence of dict
        Streamline payloads returned by ``extract_streamline_field``.
    title : str, optional
        Figure title.
    output_path : str, optional
        If provided, save the figure to this path.
    xlim, ylim : tuple[float, float], optional
        Plot limits.
    density : float or tuple, default=1.5
        Forwarded to ``matplotlib.axes.Axes.streamplot``.
    linewidth : float, default=1.0
        Base streamline width when a scalar width is used.
    arrowsize : float, default=1.0
        Streamline arrow scaling.
    color : str or sequence, optional
        Explicit streamline color. If omitted, a per-case color cycle is used
        unless a color field is attached to the streamline set.
    cmap : str, default='viridis'
        Colormap used when a streamline set provides ``color`` data.
    start_points : array-like, optional
        Optional streamline seed points.
    seed_lines : sequence of dict, optional
        Multiple seed-line specifications. Each entry may define ``count``,
        ``x_position``, ``y_min``, ``y_max``, ``spacing``, and ``power``.
    seed_count : int, optional
        If provided and ``start_points`` is omitted, auto-generate this many
        seed points at a fixed x-position.
    seed_x_position : float, optional
        Streamwise location used for auto-generated seed points.
    seed_y_min, seed_y_max : float, optional
        Vertical bounds used for auto-generated seed points.
    seed_spacing : {'uniform', 'power', 'sqrt'}, default='uniform'
        Y-distribution used for auto-generated seed points.
    seed_power : float, default=2.0
        Exponent used when ``seed_spacing='power'``. Larger values cluster
        more seed points near ``seed_y_min``.
    broken_streamlines : bool, default=True
        Forwarded to ``streamplot``.
    colorbar : bool, optional
        Whether to draw a colorbar. Defaults to true only when plotting a
        single color-mapped streamline set.
    colorbar_label : str, optional
        Colorbar label. Defaults to the streamline set color key.
    mask_fill_color : str, default='black'
        Fill color used to cover masked cells when a streamline set defines a mask.
    mask_fill_alpha : float, default=1.0
        Alpha used for the masked-cell fill.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw into.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the streamline plot.
    """
    if len(streamline_sets) == 0:
        raise ValueError('At least one streamline field is required for plotting.')

    created_figure = False
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
        created_figure = True

    if colorbar is None:
        colorbar = len(streamline_sets) == 1 and streamline_sets[0].get('color') is not None

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    legend_handles = []
    colorbar_artist = None

    for index, streamline_set in enumerate(streamline_sets):
        x_line = np.asarray(streamline_set['x'], dtype=float)
        y_line = np.asarray(streamline_set['y'], dtype=float)
        u_values = np.asarray(streamline_set['u'], dtype=float)
        v_values = np.asarray(streamline_set['v'], dtype=float)
        color_values = streamline_set.get('color')
        mask_values = streamline_set.get('mask')

        if mask_values is not None:
            mask_values = np.asarray(mask_values, dtype=bool)
            u_values = np.ma.masked_where(mask_values, u_values)
            v_values = np.ma.masked_where(mask_values, v_values)
            if color_values is not None:
                color_values = np.ma.masked_where(mask_values, np.asarray(color_values, dtype=float))

        streamplot_kwargs = {
            'density': density,
            'linewidth': linewidth,
            'arrowsize': arrowsize,
            'broken_streamlines': broken_streamlines,
        }

        resolved_start_points = []
        if start_points is not None:
            resolved_start_points.append(np.asarray(start_points, dtype=float))
        if seed_lines is not None:
            multi_seed_points = generate_multiple_streamline_start_points(x_line, y_line, seed_lines)
            if len(multi_seed_points) > 0:
                resolved_start_points.append(multi_seed_points)
        elif seed_count is not None:
            resolved_start_points.append(
                generate_streamline_start_points(
                    x_line,
                    y_line,
                    count=seed_count,
                    x_position=seed_x_position,
                    y_min=seed_y_min,
                    y_max=seed_y_max,
                    spacing=seed_spacing,
                    power=seed_power,
                )
            )

        if resolved_start_points:
            streamplot_kwargs['start_points'] = np.vstack(resolved_start_points)

        if color_values is not None:
            color_array = np.asarray(color_values, dtype=float)
            artist = ax.streamplot(
                x_line,
                y_line,
                u_values.T,
                v_values.T,
                color=color_array.T,
                cmap=cmap,
                **streamplot_kwargs,
            )
            if colorbar and colorbar_artist is None:
                colorbar_artist = artist.lines
        else:
            line_color = color if color is not None else color_cycle[index % len(color_cycle)]
            ax.streamplot(
                x_line,
                y_line,
                u_values.T,
                v_values.T,
                color=line_color,
                **streamplot_kwargs,
            )
            if streamline_set.get('label'):
                legend_handles.append(Line2D([0], [0], color=line_color, linewidth=2.0, label=streamline_set['label']))

        if mask_values is not None and np.any(mask_values):
            masked_region = np.ma.masked_where(~mask_values.T, np.ones(mask_values.T.shape, dtype=float))
            ax.pcolormesh(
                x_line,
                y_line,
                masked_region,
                shading='auto',
                cmap=ListedColormap([mask_fill_color]),
                alpha=mask_fill_alpha,
                zorder=3,
            )

    ax.set_xlabel('x [m]', fontsize=14)
    ax.set_ylabel('y [m]', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    if title is not None:
        ax.set_title(title, fontsize=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    if legend_handles:
        ax.legend(handles=legend_handles)

    if colorbar and colorbar_artist is not None:
        cb = ax.figure.colorbar(colorbar_artist, ax=ax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        else:
            cb.set_label(streamline_sets[0].get('color_key', 'Field'))

    if output_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(output_path, dpi=200)
    elif created_figure:
        ax.figure.tight_layout()

    return ax

# ========================================================================= #
#                       SILO DATA FILE READ FUNCTIONS                       #
# ========================================================================= #

def grab_and_reshape(field_id, db, num_x, num_y):
    fid_str = f'0{field_id}' if field_id < 10 else f'{field_id}'
    return np.reshape(
        db[f'/.silo/#0000{fid_str}'][()],
        (num_y, num_x),
        order='C'
    ).T


def _grab_and_reshape_3d(field_id, db, num_x, num_y, num_z):
    fid_str = f'0{field_id}' if field_id < 10 else f'{field_id}'
    return np.reshape(
        db[f'/.silo/#0000{fid_str}'][()],
        (num_z, num_y, num_x),
        order='C'
    ).T  # -> (num_x, num_y, num_z)


def _resolve_silo_data_file(silo_file):
    silo_path = os.path.abspath(os.fspath(silo_file))
    file_name = os.path.basename(silo_path)

    if not file_name.startswith('collection_'):
        return silo_path

    root_dir = os.path.dirname(silo_path)
    silo_root = os.path.dirname(root_dir)
    per_rank_name = file_name.replace('collection_', '', 1)

    rank_dirs = []
    for entry in os.listdir(silo_root):
        entry_path = os.path.join(silo_root, entry)
        if os.path.isdir(entry_path) and re.fullmatch(r'p\d+', entry):
            rank_dirs.append(entry_path)

    candidates = [
        os.path.join(rank_dir, per_rank_name)
        for rank_dir in sorted(rank_dirs)
        if os.path.isfile(os.path.join(rank_dir, per_rank_name))
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ValueError(
            'collection_*.silo is a multi-rank master file. '
            'Pass a specific per-rank silo file instead.'
        )

    return silo_path

def script_path(*parts):
    return os.path.normpath(os.path.join(SCRIPT_DIR, *parts))


def _read_alt_header(source_path):
    with open(source_path, encoding="utf-8") as source_file:
        for line in source_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if not stripped_line.startswith("#"):
                return {}

            header_names = stripped_line.lstrip("#").split()
            return {name: index for index, name in enumerate(header_names)}

    return {}


def _resolve_column_index(column_spec, header_map, role_name):
    if isinstance(column_spec, int):
        return column_spec
    if isinstance(column_spec, str) and column_spec in header_map:
        return header_map[column_spec]

    raise ValueError(f"Could not resolve {role_name} column {column_spec!r}.")


def load_alt_profile(alt_case, x_location=None, unit_reynolds=None):
    loadtxt_kwargs = {"comments": alt_case.get("comments", "#")}
    loadtxt_kwargs.update(alt_case.get("loadtxt_kwargs", {}))

    alt_data = np.atleast_2d(np.loadtxt(alt_case["source"], **loadtxt_kwargs))
    header_map = _read_alt_header(alt_case["source"])
    column_map = alt_case.get("columns", {})

    y_column = _resolve_column_index(column_map.get("y", 0), header_map, "y")
    value_column = _resolve_column_index(column_map.get("value", 1), header_map, "value")

    y_values = alt_data[:, y_column]
    value_values = alt_data[:, value_column]

    y_transform = alt_case.get("y_transform", "identity")
    if y_transform == "boundary_layer_nondim":
        if x_location is None or unit_reynolds is None:
            raise ValueError("x_location and unit_reynolds are required for nondimensional y data.")
        y_values = y_values * np.sqrt(x_location / unit_reynolds)
    elif y_transform != "identity":
        raise ValueError(f"Unsupported y_transform {y_transform!r}.")

    y_values = y_values * alt_case.get("y_scale", 1.0)
    value_values = value_values * alt_case.get("value_scale", 1.0)

    return {
        "label": alt_case.get("label", "Reference"),
        "y": y_values,
        "values": value_values,
    }

def inspect_contents(db):
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'DATASET {obj.name}:: ({obj.shape}) \tmax: {obj[()].max()}, min: {obj[()].min()}')
        elif isinstance(obj, h5py.Group):
            print(f'GROUP {name}')
        elif isinstance(obj, h5py.Datatype):
            print(f'NAMEDTYPE {name}, {obj.dtype}')
    db.visititems(walk)
    return

# Silo file reading and field data processing function
def _coerce_bool_option(option_value, option_name):
    if isinstance(option_value, bool):
        return option_value
    if isinstance(option_value, str):
        normalized_value = option_value.strip().lower()
        if normalized_value in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized_value in {"false", "0", "no", "n", "off"}:
            return False

    raise ValueError(
        f"{option_name} must be a bool or a boolean-like string; got {option_value!r}."
    )


def Silo_Read(silo_file, slice_plane=None, slice_index=None, slice_coord=None, species=True):
    resolved_silo_file = _resolve_silo_data_file(silo_file)
    db = h5py.File(resolved_silo_file, "r")
    species = _coerce_bool_option(species, "species")

    # ------------------------------------------------------------------ #
    #  AUTO-DETECT DIMENSIONALITY                                         #
    #  In a 2D silo file, #000003 is the first field (alpha_rho) and     #
    #  has num_x*num_y elements. In a 3D file, #000003 is the z          #
    #  coordinate array and has only num_z+1 elements. Compare sizes     #
    #  to distinguish the two cases.                                      #
    # ------------------------------------------------------------------ #
    x = db['/.silo/#000001'][:]
    y = db['/.silo/#000002'][:]
    x_cc = 0.5 * (x[:-1] + x[1:])
    y_cc = 0.5 * (y[:-1] + y[1:])

    #inspect_contents(h5py.File(resolved_silo_file, "r"))
    #exit()

    _size_003 = db['/.silo/#000003'].size if '/.silo/#000003' in db else 0
    # A z-coordinate array has O(num_z) elements; a 2D field has num_x*num_y
    is_3d = '/.silo/#000003' in db and _size_003 != len(x_cc) * len(y_cc)

    if is_3d:
        # ---------------------------------------------------------------- #
        #  3-D BRANCH: read full volume, then extract 2-D slice            #
        # ---------------------------------------------------------------- #
        z = db['/.silo/#000003'][:]
        z_cc = 0.5 * (z[:-1] + z[1:])
        num_x, num_y, num_z = len(x_cc), len(y_cc), len(z_cc)

        # 3-D field list — rho_w and w are inserted between rho_v and E
        _fields_3d = [
            'alpha_rho',   # 04
            'rho',         # 05
            'rho_u',       # 06
            'rho_v',       # 07
            'rho_w',       # 08  (not present in 2-D)
            'u',           # 09
            'v',           # 10
            'w',           # 11  (not present in 2-D)
            'E',           # 12
            'p',           # 13
            'alpha',       # 14
            'c',           # 15
            'ib_markers',  # 16
        ]

        fields_data = {}
        for i, f in enumerate(_fields_3d):
            field_id = i + 4  # 3 coordinate arrays -> fields start at #000004
            fields_data[f] = _grab_and_reshape_3d(field_id, db, num_x, num_y, num_z)
        db.close()

        if slice_plane is None:
            raise ValueError(
                "3D silo file detected. "
                "Provide slice_plane='xy'|'xz'|'yz' and either "
                "slice_coord or slice_index."
            )

        _normal_coords = {'xy': z_cc, 'xz': y_cc, 'yz': x_cc}
        if slice_plane not in _normal_coords:
            raise ValueError(f"slice_plane must be 'xy', 'xz', or 'yz'; got '{slice_plane}'")
        norm_cc = _normal_coords[slice_plane]

        if slice_coord is not None:
            k_slice = int(np.argmin(np.abs(norm_cc - slice_coord)))
        elif slice_index is not None:
            k_slice = int(slice_index)
        else:
            k_slice = len(norm_cc) // 2
            print(f"  [Silo_Read] No slice specified — using midplane "
                  f"index {k_slice} ({slice_plane} at {norm_cc[k_slice]:.6f})")

        # Full 3-D velocity magnitude preserves the out-of-plane contribution to Mach
        vel_mag_3d = np.sqrt(
            fields_data['u']**2 + fields_data['v']**2 + fields_data['w']**2
        )

        # Extract 2-D slice; remap in-plane velocity components to u/v so that
        # all downstream code (gradients, vorticity, plots) is unified.
        slice_data = {}
        if slice_plane == 'xy':
            # Normal: z  |  horizontal: x  |  vertical: y
            h_cc, v_cc = x_cc, y_cc
            for f in _fields_3d:
                slice_data[f] = fields_data[f][:, :, k_slice]
            slice_data['vel_mag'] = vel_mag_3d[:, :, k_slice]
        elif slice_plane == 'xz':
            # Normal: y  |  horizontal: x  |  vertical: z (relabelled to v)
            h_cc, v_cc = x_cc, z_cc
            for f in _fields_3d:
                slice_data[f] = fields_data[f][:, k_slice, :]
            slice_data['v'] = slice_data['w']   # z-velocity becomes in-plane lateral
            slice_data['vel_mag'] = vel_mag_3d[:, k_slice, :]
        elif slice_plane == 'yz':
            # Normal: x  |  horizontal: y (-> u)  |  vertical: z (-> v)
            h_cc, v_cc = y_cc, z_cc
            for f in _fields_3d:
                slice_data[f] = fields_data[f][k_slice, :, :]
            slice_data['u'] = slice_data['v']   # y-velocity -> in-plane horizontal
            slice_data['v'] = slice_data['w']   # z-velocity -> in-plane vertical
            slice_data['vel_mag'] = vel_mag_3d[k_slice, :, :]

        slice_data['slice_coord'] = float(norm_cc[k_slice])
        dh = h_cc[1] - h_cc[0]
        dv = v_cc[1] - v_cc[0]
        xx, yy = np.meshgrid(h_cc, v_cc, indexing='ij')
        fields_data = slice_data
        dx, dy = dh, dv  # reuse names for the shared derived-field block
        print(f"  [Silo_Read] 3D -> 2D slice: plane={slice_plane}, "
              f"coord={norm_cc[k_slice]:.6f} (index {k_slice}), "
              f"grid {len(h_cc)} x {len(v_cc)}")

    else:
        # ---------------------------------------------------------------- #
        #                           2-D BRANCH                     
        # ---------------------------------------------------------------- #
        if species:
            _fields_2d = [
                "alpha_rho1",
                "rho",
                "rho_u",
                "rho_v",
                "u",
                "v",
                "Y_O2",
                "Y_N2",
                "E",
                "p",
                "alpha",
                "c",
                "schlieren",
            ]
        else:
            _fields_2d = [
                'alpha_rho',   # 03
                'rho',         # 04
                'rho_u',       # 05
                'rho_v',       # 06
                'u',           # 07
                'v',           # 08
                'E',           # 09
                'p',           # 10
                'alpha',       # 11
                'c',           # 12
                'ib_markers',  # 13
            ]

        num_x = len(x_cc)
        num_y = len(y_cc)
        dx = x_cc[1] - x_cc[0]
        dy = y_cc[1] - y_cc[0]
        xx, yy = np.meshgrid(x_cc, y_cc, indexing='ij')

        fields_data = {}
        for i, f in enumerate(_fields_2d):
            field_id = i + 3
            fields_data[f] = grab_and_reshape(field_id, db, num_x, num_y)
        db.close()

    # ------------------------------------------------------------------ #
    #  DERIVED FIELDS — identical for 2-D and 3-D slices                  #
    # ------------------------------------------------------------------ #
    drho_dy, drho_dx = np.gradient(fields_data['rho'], dy, dx)
    grad_mag = np.sqrt(drho_dx**2 + drho_dy**2)
    beta = 50
    _grad_max = grad_mag.max()
    if _grad_max == 0:
        fields_data['Schlieren'] = np.ones_like(grad_mag)  # uniform field → white image
    else:
        fields_data['Schlieren'] = np.exp(-beta * (grad_mag / _grad_max))

    # Velocity magnitude (3-D slice already holds the full-magnitude value)
    if 'vel_mag' not in fields_data:
        fields_data['vel_mag'] = np.sqrt(fields_data['u']**2 + fields_data['v']**2)

    fields_data['Mach'] = fields_data['vel_mag'] / fields_data['c']

    R = 287.05  # J/(kg·K) for air
    fields_data['Temperature'] = fields_data['p'] / (fields_data['rho'] * R)

    # In-plane vorticity (component normal to the slice plane)
    dv_dx, dv_dy = np.gradient(fields_data['v'], dx, dy)
    du_dx, du_dy = np.gradient(fields_data['u'], dx, dy)
    fields_data['vorticity'] = dv_dx - du_dy
    fields_data['vorticity_mag'] = np.abs(fields_data['vorticity'])

    # Normalisations — freestream taken from corner [0,0] of the slice
    # (valid for xy/xz planes where [0,0] is the upstream inflow corner)
    fields_data['P/P∞'] = fields_data['p']   / fields_data['p'][0, 0]
    fields_data['U/U∞'] = fields_data['u']   / fields_data['u'][0, 0]
    fields_data['ρ/ρ∞'] = fields_data['rho'] / fields_data['rho'][0, 0]
    fields_data['$P/P_{dyn}$'] = fields_data['p'] / (
        0.5 * 1.4 * fields_data['p'][0, 0] * fields_data['Mach'][0, 0]**2
    )
    fields_data['$Vorticity/Vorticity_{char}$'] = (
        fields_data['vorticity'] / (fields_data['u'][0, 0] / 0.2)
    )
    fields_data["$|∇ρ|/(ρ∞/L_{char})$"] = (
        grad_mag / (fields_data['rho'][0, 0] / 0.2)
    )
    return xx, yy, fields_data

# ========================================================================= #
#                             FANCY CONTOUR PLOTS                           #
# ========================================================================= #

def _zoom_limits_from_state(zoom_state, xx=None, yy=None, aspect_ratio=None):
    # x-ranges define the region of interest; y-range is auto-derived from the
    # domain midpoint so it works for any simulation domain size.
    # set_aspect('equal') in the caller enforces undistorted rendering.
    z = str(zoom_state).lower()

    y_mid = 0.5 * (yy.min() + yy.max()) if yy is not None else 0.0

    if z == "base":
        xmin, xmax = (xx.min(), xx.max()) if xx is not None else (-0.1, 1.25)
    elif z == "sec1":
        xmin, xmax = 0.05, 0.15
    elif z == "sec2":
        xmin, xmax = 0.20, 0.60
    else:
        return None, None  # full-domain

    x_half = 0.5 * (xmax - xmin)
    ymin = max(y_mid - x_half, yy.min()) if yy is not None else y_mid - x_half
    ymax = min(y_mid + x_half, yy.max()) if yy is not None else y_mid + x_half
    return (xmin, xmax), (ymin, ymax)


def plot_overlay_from_fields(
    xx, yy, fields_data,
    base_key="Schlieren",
    overlay_key="vorticity",
    ib_key="ib_markers",
    base_cmap="gray",
    overlay_cmap=None,                # auto: RdBu_r if signed, turbo if non-negative
    side="Upper",                   # "Upper" or "Lower" side of CCF
    base_vmin=None,
    base_vmax=None,
    overlay_vmin=None,
    overlay_vmax=None,
    overlay_signed=None,              # None => auto-detect from data
    max_alpha=0.8,
    overlay_deadband=None,            # NEW: abs-threshold for signed fields (e.g., 0.1)
    mask_out_of_range=True,           # NEW: hide values outside [vmin, vmax] if True
    zoom_state="Base",
    xlim=None,
    ylim=None,
    snapshot="test",
    output_dir="Specialized_Plots",
    ib_color="white",      
    ib_alpha=1.0,
    stl_profile=None           
):
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base = fields_data[base_key].astype(float).copy()
    overlay = fields_data[overlay_key].astype(float).copy()

    # Fluid mask
    fluid_mask = None
    solid_mask = None
    if ib_key in fields_data:
        fluid_mask = fields_data[ib_key] < 0.1
        solid_mask = ~fluid_mask
        base[solid_mask] = np.nan

    

    # Colormap default
    if overlay_cmap is None:
        overlay_cmap = "RdBu_r" if overlay_signed else "turbo"

    # Base limits
    if base_vmin is None:
        base_vmin = np.nanpercentile(base, 1)
    if base_vmax is None:
        base_vmax = np.nanpercentile(base, 99)

    finite = np.isfinite(overlay)
    if fluid_mask is not None:
        finite &= fluid_mask
    vals = overlay[finite]
    if vals.size == 0:
        raise ValueError(f"No finite values for overlay field '{overlay_key}'.")

    # Auto colormap only (no forced signed range behavior)
    if overlay_signed is None:
        overlay_signed = (np.nanmin(vals) < 0.0)
    if overlay_cmap is None:
        overlay_cmap = "RdBu_r" if overlay_signed else "turbo"

    # Use user range if provided; otherwise full variable range
    if overlay_vmin is None:
        overlay_vmin = np.nanmin(vals)
    if overlay_vmax is None:
        overlay_vmax = np.nanmax(vals)

    if overlay_vmin > overlay_vmax:
        overlay_vmin, overlay_vmax = overlay_vmax, overlay_vmin

    norm = mcolors.Normalize(vmin=overlay_vmin, vmax=overlay_vmax)

    # Toggle whether out-of-range values are hidden or shown
    if mask_out_of_range:
        show_mask = finite & (overlay >= overlay_vmin) & (overlay <= overlay_vmax)
    else:
        show_mask = finite

    # Remove center band if requested
    if overlay_deadband is not None:
        if overlay_signed:
            show_mask &= (np.abs(overlay) >= float(overlay_deadband))
        else:
            show_mask &= (overlay >= float(overlay_deadband))

    # Debug
    n_total = np.count_nonzero(finite)
    n_show = np.count_nonzero(show_mask)
    print(f"[overlay={overlay_key}] data min/max = {np.nanmin(vals):.4e} / {np.nanmax(vals):.4e}")
    print(f"[overlay={overlay_key}] plot min/max = {overlay_vmin:.4e} / {overlay_vmax:.4e}")
    print(f"[overlay={overlay_key}] mask_out_of_range = {mask_out_of_range}")
    if overlay_deadband is not None:
        print(f"[overlay={overlay_key}] deadband = {overlay_deadband:.4e} "
              f"({'abs' if overlay_signed else 'min'})")
    print(f"[overlay={overlay_key}] shown cells = {n_show}/{n_total} ({100.0*n_show/max(n_total,1):.2f}%)")

    # Figure setup
    x_extent = xx.max() - xx.min()
    y_extent = yy.max() - yy.min()
    aspect_ratio = y_extent / x_extent
    fig, ax = plt.subplots(figsize=(20, 10))

    # Base layer
    p0 = ax.pcolormesh(
        xx, yy, base, cmap=base_cmap, shading="auto",
        vmin=base_vmin, vmax=base_vmax,
        rasterized=True, zorder=1
    )
    cb0 = plt.colorbar(p0, ax=ax, pad=0.065, fraction=0.015)
    cb0.set_label(base_key, fontsize=18)

    # Overlay layer
    overlay_masked = np.ma.masked_where(~show_mask, overlay)

    p1 = ax.pcolormesh(
        xx, yy, overlay_masked,
        cmap=overlay_cmap,
        shading="auto",
        vmin=overlay_vmin,
        vmax=overlay_vmax,
        alpha=max_alpha,
        rasterized=True,
        zorder=2
    )

    # Draw IB region with user-selected color
    if stl_profile is not None:
        print(f"  Building STL geometry mask...")
        geometry_mask = build_stl_geometry_mask(xx, yy, stl_profile)
    else:
        # Original ib_markers mask
        geometry_mask = np.where(np.abs(fields_data['ib_markers']) > 0, 1, np.nan)

    ax.pcolormesh(
        xx, yy, geometry_mask,
        shading="auto",
        cmap=ListedColormap([ib_color]),
        vmin=0.0, vmax=1.0,
        alpha=ib_alpha,
        zorder=3
    )

    # Overlay colorbar 
    sm = cm.ScalarMappable(norm=norm, cmap=overlay_cmap)
    sm.set_array([])
    cb1 = plt.colorbar(sm, ax=ax, pad=0.025, fraction=0.015)
    cb1.set_label(f"{overlay_key}", fontsize=22)

    # Zoom handling 
    if xlim is None or ylim is None:
        zx, zy = _zoom_limits_from_state(zoom_state, xx=xx, yy=yy)
        if zx is not None and zy is not None:
            ax.set_xlim(*zx)
            ax.set_ylim(*zy)
        else:
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
    else:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.set_xlabel("x (m)", fontsize=30)
    ax.set_ylabel("y (m)", fontsize=30)
    ax.tick_params(labelsize=24)
    ax.set_aspect("equal")
    ax.set_title(f"{base_key} + {overlay_key} — snapshot {snapshot}", fontsize=35)

    plt.tight_layout()
    out = f"{output_dir}/{side}-{base_key.split('/')[0].split('$')[-1]}_{overlay_key.split('/')[0].split('$')[-1]}_{snapshot}.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out}")
    return out


def Colormaps():
    # (R, G, B, A)

    my_reds_cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_reds",
        ["crimson", "red", "orangered", "salmon", "pink", "white", (1, 1, 1, 0.0)],
        N=256
    )

    my_reds_cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "my_reds_r",
        ["crimson", "red", "orangered", "salmon", "pink", "white", (1, 1, 1, 0.0)][::-1],
        N=256
    )

    my_blues_cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_blues",
        ["navy", "blue", "dodgerblue", "skyblue", "lightblue", "white", (1, 1, 1, 0.0)],
        N=256
    )

    my_blues_cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "my_blues_r",
        ["navy", "blue", "dodgerblue", "skyblue", "lightblue", "white", (1, 1, 1, 0.0)][::-1],
        N=256
    )

    #my_inferno_cmap = mcolors.LinearSegmentedColormap.from_list(
    #    "my_inferno",
    #    ["", "blue", "dodgerblue", "skyblue", "lightblue", "", (1, 1, 1, 0.0)],
    #    N=256
    #)

    Blue2Red = mcolors.LinearSegmentedColormap.from_list(
        "Blue2Red",
           ["navy", "blue", "dodgerblue", "skyblue", "lightblue", (1, 1, 1, 0.0), "pink", "salmon", "orangered", "red", "crimson"],
        N=256
    )

    return {
        "my_reds": my_reds_cmap,
        "my_blues": my_blues_cmap,
        "my_reds_r": my_reds_cmap_r,
        "my_blues_r": my_blues_cmap_r,
        "Blue2Red": Blue2Red
        }











# ========================================================================= #
#                         LINE EXTRACTION FUNCTIONS                         #
# ========================================================================= #

def BL_extract_line(x_cc,y_cc,data_field,T_field,x_station,y_val):
    '''Boundary Layer Analysis: Velocity wall profile, Wall Shear Stress, '''

    # Extract profile at x_station
    y_wall_normal, u_profile = extract_line(x_cc, y_cc, data_field, x_station, y_val)
    _, T_profile = extract_line(x_cc, y_cc, T_field, x_station, y_val)

    # Find where du/dy becomes very small
    du_dy = np.gradient(u_profile, y_wall_normal)
    threshold = 1e-2  # Define a small threshold for gradient

    # --- Edge Detection Logic ---
    # First index where gradient drops below threshold
    plateau_idx = np.where(np.abs(du_dy) < threshold)[0]
    if len(plateau_idx) > 0:
        u_edge = u_profile[plateau_idx[0]]
    else:
        u_edge = u_profile[-1]  # fallback to last point

    U_Uinf = u_profile / u_edge  # Normalize velocity profile

    # --- Shear Stress Calculation ---
    n_points = min(4, len(y_wall_normal))
    y_near_wall = y_wall_normal[:n_points]
    u_near_wall = u_profile[:n_points]
    T_wall = T_profile[0]  # Temperature at wall
    dyn_vis = sutherland_viscosity(T_wall)

    # Linear fit for robustness
    coeffs = np.polyfit(y_near_wall, u_near_wall, 1)
    du_dy_wall = coeffs[0]

    # Wall Shear Stress
    tau_wall = dyn_vis * du_dy_wall

    # Skin Friction Coefficient
    rho_inf = 0.0267    # kg/m^3
    u_inf = 1011        # m/s
    q_inf = 0.5 * rho_inf * u_inf**2  # Dynamic pressure
    C_f = tau_wall / q_inf

    return y_wall_normal, U_Uinf, tau_wall, C_f

def sutherland_viscosity(T):
    """Dynamic viscosity using Sutherland's law for air"""
    T_ref = 273.15  # K
    mu_ref = 1.716e-5  # Pa·s at T_ref
    S = 110.4  # Sutherland constant for air (K)
    
    return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)


# ========================================================================= #
#                       SURFACE EXTRACTION FUNCTIONS                        #
# ========================================================================= #

def calculate_reynolds_numbers(x_station, u_inf, rho_inf, mu_inf, delta_99=None, theta=None):
    """
    Calculate Reynolds numbers for boundary layer analysis
    """
    # Reynolds number based on distance from leading edge
    Re_x = rho_inf * u_inf * x_station / mu_inf
    
    # Reynolds number based on boundary layer thickness
    Re_delta = rho_inf * u_inf * delta_99 / mu_inf if delta_99 is not None else None
    
    # Reynolds number based on momentum thickness
    Re_theta = rho_inf * u_inf * theta / mu_inf if theta is not None else None
    
    return Re_x, Re_delta, Re_theta

def calculate_BL_thicknesses(y_wall_normal, U_Uinf):
    """Calculate δ99, δ*, θ, and shape factor H"""
    
    # δ99: where u = 0.99 * u_inf
    idx_99 = np.where(U_Uinf >= 0.99)[0]
    delta_99 = y_wall_normal[idx_99[0]] if len(idx_99) > 0 else y_wall_normal[-1]
    
    # Displacement thickness δ*
    integrand_disp = 1 - U_Uinf
    delta_star = np.trapz(integrand_disp, y_wall_normal)
    
    # Momentum thickness θ
    integrand_mom = U_Uinf * (1 - U_Uinf)
    theta = np.trapz(integrand_mom, y_wall_normal)
    
    # Shape factor H
    H = delta_star / theta if theta > 0 else np.nan
    
    return delta_99, delta_star, theta, H

def define_geometry_surface(x_cc, y_cc, fields_data, velocity_threshold=10.0, 
                            max_x=0.75, debug=False):
    """
    Extract upper and lower surface coordinates from geometry.
    
    Uses a hybrid approach:
    1. Velocity magnitude < threshold identifies cells near/in the geometry
    2. ib_markers gradient refines the exact surface location
    3. Stops at max_x to avoid detecting wake as geometry
    
    The ib_markers field ranges 0-5, where:
      - ib_markers = 0: fluid cell (cell center in fluid)
      - ib_markers > 0: cell center is inside IB geometry definition
    
    Parameters:
    -----------
    velocity_threshold : float
        Velocity magnitude below this (m/s) indicates potential geometry
    max_x : float
        Maximum x-coordinate to search for geometry (meters).
        Prevents detecting wake as geometry.
    """
    
    vel_mag = fields_data['vel_mag']
    ib = fields_data['ib_markers']
    
    surfaces = {'upper': [], 'lower': []}
    
    if debug:
        print(f"  [DEBUG] Grid: x from {x_cc.min():.4f} to {x_cc.max():.4f}")
        print(f"  [DEBUG] Grid: y from {y_cc.min():.4f} to {y_cc.max():.4f}")
        print(f"  [DEBUG] ib_markers range: [{ib.min():.2f}, {ib.max():.2f}]")
        print(f"  [DEBUG] Velocity threshold: {velocity_threshold} m/s")
        print(f"  [DEBUG] Max x for geometry: {max_x} m")
    
    # For each x-station (but only up to max_x), find upper and lower surface
    for i, x_val in enumerate(x_cc):
        # Stop searching past the known geometry extent
        if x_val > max_x:
            break
        
        vel_column = vel_mag[i, :]
        ib_column = ib[i, :]
        
        # Require BOTH low velocity AND ib_markers indicating geometry
        # This prevents wake (low vel, ib=0) from being flagged
        geometry_candidates = (vel_column < velocity_threshold) & (ib_column > 0.1)
        
        if not np.any(geometry_candidates):
            continue
        
        geom_indices = np.where(geometry_candidates)[0]
        if len(geom_indices) == 0:
            continue
        
        # Find geometry center
        geom_y_min_idx = geom_indices.min()
        geom_y_max_idx = geom_indices.max()
        geom_y_center_idx = (geom_y_min_idx + geom_y_max_idx) // 2
        
        # --- UPPER SURFACE ---
        upper_candidates = geom_indices[geom_indices >= geom_y_center_idx]
        
        if len(upper_candidates) > 0:
            j_upper_geom = upper_candidates[-1]
            
            if j_upper_geom + 1 < len(y_cc):
                j_fluid = j_upper_geom + 1
                
                y_exact = interpolate_surface_location(
                    y_cc[j_upper_geom], y_cc[j_fluid],
                    fields_data, i, j_upper_geom, j_fluid
                )
                
                surfaces['upper'].append({
                    'x': x_val,
                    'y': y_cc[j_fluid],
                    'y_interp': y_exact,
                    'i': i,
                    'j': j_fluid
                })
        
        # --- LOWER SURFACE ---
        lower_candidates = geom_indices[geom_indices <= geom_y_center_idx]
        
        if len(lower_candidates) > 0:
            j_lower_geom = lower_candidates[0]
            
            if j_lower_geom > 0:
                j_fluid = j_lower_geom - 1
                
                y_exact = interpolate_surface_location(
                    y_cc[j_lower_geom], y_cc[j_fluid],
                    fields_data, i, j_lower_geom, j_fluid
                )
                
                surfaces['lower'].append({
                    'x': x_val,
                    'y': y_cc[j_fluid],
                    'y_interp': y_exact,
                    'i': i,
                    'j': j_fluid
                })
    
    # Convert to structured arrays
    for side in ['upper', 'lower']:
        if len(surfaces[side]) > 0:
            surfaces[side] = {
                'x': np.array([p['x'] for p in surfaces[side]]),
                'y': np.array([p['y'] for p in surfaces[side]]),
                'y_interp': np.array([p['y_interp'] for p in surfaces[side]]),
                'i': np.array([p['i'] for p in surfaces[side]], dtype=int),
                'j': np.array([p['j'] for p in surfaces[side]], dtype=int)
            }
            if debug:
                print(f"  [DEBUG] {side}: {len(surfaces[side]['x'])} points")
                print(f"    x range: [{surfaces[side]['x'].min():.6f}, {surfaces[side]['x'].max():.6f}]")
                print(f"    y range: [{surfaces[side]['y_interp'].min():.6f}, {surfaces[side]['y_interp'].max():.6f}]")
        else:
            surfaces[side] = {
                'x': np.array([]),
                'y': np.array([]),
                'y_interp': np.array([]),
                'i': np.array([], dtype=int),
                'j': np.array([], dtype=int)
            }
            if debug:
                print(f"  [DEBUG] {side}: 0 points")
    
    return surfaces


def interpolate_surface_location(y1, y2, fields_data, i, j1, j2):
    """
    Interpolate the exact surface location between two cells using ib_markers.
    
    The ib_markers field provides a smooth transition:
    - ib = 0: fluid cell center
    - ib > 0: cell center inside geometry (but cell may contain fluid)
    
    We interpolate to find where ib_markers ≈ 0.5, which represents the
    approximate 50% volume fraction (halfway between fluid and solid cell centers).
    
    Parameters:
    -----------
    y1, y2 : float
        Y-coordinates of the two adjacent cells
    fields_data : dict
        Must contain 'ib_markers' field
    i : int
        X-index
    j1, j2 : int
        Y-indices of the two cells (one should have ib~0, other ib>0)
    """
    ib1 = fields_data['ib_markers'][i, j1]
    ib2 = fields_data['ib_markers'][i, j2]
    
    # Target: find where ib_markers crosses a threshold (e.g., 0.5)
    # This represents the approximate surface location
    threshold = 0.5
    
    if abs(ib2 - ib1) > 1e-6:
        # Linear interpolation to find where ib = threshold
        frac = (threshold - ib1) / (ib2 - ib1)
        frac = np.clip(frac, 0, 1)
        return y1 + frac * (y2 - y1)
    
    # Fallback: if ib values are identical, use midpoint
    return 0.5 * (y1 + y2)

def compute_surface_normals(surfaces, smoothing_window=51):
    """
    Compute smoothed surface normals for staircase geometry
    
    For a 2D surface defined by (x, y):
    - Tangent vector: t = (dx/ds, dy/ds)
    - Normal vector: n = (-dy/ds, dx/ds) [rotated 90°]
    
    Parameters:
    -----------
    surfaces : dict
        Output from define_geometry_surface()
    smoothing_window : int
        Window size for Savitzky-Golay smoothing (must be odd)
    
    Returns:
    --------
    surfaces : dict (modified in place)
        Adds 'nx', 'ny', 'theta' (angle from horizontal) to each surface
    """
    from scipy.signal import savgol_filter
    
    for side in ['upper', 'lower']:
        # FIX: Check if it's a dict with data, not an empty list
        if not isinstance(surfaces[side], dict) or len(surfaces[side].get('x', [])) == 0:
            continue
        
        x = surfaces[side]['x']
        y = surfaces[side]['y_interp']  # Use interpolated surface
        n_points = len(x)
        
        # Smooth the surface to remove staircase artifacts
        if n_points > smoothing_window:
            # Savitzky-Golay filter preserves features better than moving average
            y_smooth = savgol_filter(y, smoothing_window, 3)  # 3rd order polynomial
        else:
            y_smooth = y
        
        # Compute derivatives (tangent vector components)
        dx_ds = np.gradient(x)
        dy_ds = np.gradient(y_smooth)
        
        # Normalize to get unit tangent
        ds = np.sqrt(dx_ds**2 + dy_ds**2)
        tx = dx_ds / (ds + 1e-10)
        ty = dy_ds / (ds + 1e-10)
        
        # Normal vector (perpendicular to tangent)
        # For upper surface: normal points away from body (+y direction)
        # For lower surface: normal points away from body (-y direction)
        if side == 'upper':
            nx = -ty  # Rotate tangent 90° counterclockwise
            ny = tx
        else:
            nx = ty   # Rotate tangent 90° clockwise
            ny = -tx
        
        # Ensure normal points away from body
        # (should point in general +y direction for upper, -y for lower)
        if side == 'upper':
            # Normal should have positive y-component for upper surface
            sign_correction = np.sign(ny)
            sign_correction[sign_correction == 0] = 1
            nx *= sign_correction
            ny *= sign_correction
        else:
            # Normal should have negative y-component for lower surface
            sign_correction = -np.sign(ny)
            sign_correction[sign_correction == 0] = -1
            nx *= sign_correction
            ny *= sign_correction
        
        # Angle from horizontal (useful for pressure coefficient calculations)
        theta = np.arctan2(dy_ds, dx_ds)
        
        # Store results
        surfaces[side]['y_smooth'] = y_smooth
        surfaces[side]['nx'] = nx
        surfaces[side]['ny'] = ny
        surfaces[side]['theta'] = theta
        surfaces[side]['tangent_x'] = tx
        surfaces[side]['tangent_y'] = ty
    
    return surfaces


def extract_BL_profile_at_surface(x_cc, y_cc, fields_data, i_surf, j_surf,
                                   nx, ny, u_inf, rho_inf,
                                   max_BL_height=0.010, n_points_BL=200):
    """
    Extract boundary layer profile along TRUE NORMAL direction.
    
    Computes the TANGENTIAL velocity component (parallel to surface)
    and detects the BL edge using the velocity gradient method, which
    is robust in the presence of a shock layer above the BL.
    
    Parameters:
    -----------
    nx, ny : float
        Components of unit normal vector pointing away from surface
    u_inf : float
        Freestream velocity (pre-shock)
    n_points_BL : int
        Number of points to sample along normal direction
    max_BL_height : float
        Maximum distance from wall to sample (meters). Default 10 mm.
    """
    
    # Starting point (surface location)
    x_surf = x_cc[i_surf]
    y_surf = y_cc[j_surf]
    
    # Create points along the normal direction
    # Use a stretched distribution: finer near wall, coarser far from wall
    eta = np.linspace(0, 1, n_points_BL)
    stretch = 1.5  # stretching factor (>1 clusters points near wall)
    eta_stretched = np.tanh(stretch * eta) / np.tanh(stretch)
    s = eta_stretched * max_BL_height  # Distance from wall
    
    x_profile = x_surf + s * nx
    y_profile_abs = y_surf + s * ny
    y_profile = s  # Distance from wall
    
    # Interpolate field values at these points
    from scipy.interpolate import RegularGridInterpolator
    
    u_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['u'], 
                                       bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['v'],
                                       bounds_error=False, fill_value=None)
    T_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['Temperature'],
                                       bounds_error=False, fill_value=None)
    rho_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['rho'],
                                         bounds_error=False, fill_value=None)
    
    # Sample along the profile
    points = np.column_stack([x_profile, y_profile_abs])
    u_x = u_interp(points)   # x-component of velocity
    u_y = v_interp(points)   # y-component of velocity
    T_profile = T_interp(points)
    rho_profile = rho_interp(points)
    
    # --- COMPUTE TANGENTIAL VELOCITY ---
    tx_candidate = ny
    ty_candidate = -nx
    if tx_candidate < 0:
        tx_candidate = -ny
        ty_candidate = nx
    
    u_tangential = u_x * tx_candidate + u_y * ty_candidate
    vel_mag = np.sqrt(u_x**2 + u_y**2)
    
    # --- BL EDGE DETECTION VIA GRADIENT METHOD ---
    # The key insight: inside the BL, du/ds is large (velocity changes rapidly).
    # At the BL edge, the gradient drops sharply as we enter the inviscid region.
    # We find the BL edge as where the gradient first drops below a fraction
    # of the maximum gradient (which occurs near the wall).
    
    du_ds = np.gradient(u_tangential, y_profile)
    
    # Smooth the gradient to avoid noise-triggered false detection
    from scipy.signal import savgol_filter
    n_smooth = min(15, len(du_ds) // 4)
    if n_smooth >= 5 and n_smooth % 2 == 0:
        n_smooth -= 1
    if n_smooth >= 5:
        du_ds_smooth = savgol_filter(du_ds, n_smooth, 3)
    else:
        du_ds_smooth = du_ds
    
    # Maximum gradient (typically near the wall, in the BL)
    # Skip the first few points which may be in the IBM staircase
    i_start = max(2, n_points_BL // 50)
    du_ds_max = np.max(np.abs(du_ds_smooth[i_start:]))
    
    if du_ds_max < 1e-6:
        # Essentially zero gradient everywhere — no BL detected
        u_edge = u_tangential[-1] if u_tangential[-1] > 10.0 else u_inf
        i_edge = len(y_profile) - 1
    else:
        # BL edge: where |du/ds| drops below 1% of the max gradient
        # Search from the location of max gradient outward
        gradient_threshold = 0.01 * du_ds_max
        i_max_grad = i_start + np.argmax(np.abs(du_ds_smooth[i_start:]))
        
        edge_candidates = np.where(
            (np.abs(du_ds_smooth[i_max_grad:]) < gradient_threshold)
        )[0]
        
        if len(edge_candidates) > 0:
            i_edge = i_max_grad + edge_candidates[0]
        else:
            # Gradient never drops sufficiently — use 90% of max as fallback
            edge_candidates_90 = np.where(
                (np.abs(du_ds_smooth[i_max_grad:]) < 0.05 * du_ds_max)
            )[0]
            if len(edge_candidates_90) > 0:
                i_edge = i_max_grad + edge_candidates_90[0]
            else:
                i_edge = len(y_profile) - 1
        
        u_edge = u_tangential[i_edge]
    
    # Sanity: u_edge must be positive and physically reasonable
    if u_edge < 10.0:
        u_edge = np.max(u_tangential)
    if u_edge < 10.0:
        u_edge = u_inf
    
    # Normalize by edge velocity
    U_Uinf = u_tangential / u_edge
    U_Uinf = np.clip(U_Uinf, 0.0, 1.5)
    
    # Wall values
    T_wall = T_profile[0]
    
    # --- δ99 from the gradient-detected edge ---
    delta_99 = y_profile[i_edge]
    
    # Also check the traditional 0.99 criterion, but only up to i_edge region
    # This gives a more precise location within the gradient-detected BL
    search_range = min(i_edge + n_points_BL // 10, len(U_Uinf))
    idx_99 = np.where(U_Uinf[:search_range] >= 0.99)[0]
    if len(idx_99) > 0:
        delta_99 = y_profile[idx_99[0]]
    
    # Ensure δ99 is at least a few profile points from the wall
    if delta_99 <= y_profile[2]:
        delta_99 = y_profile[i_edge]
    
    # --- Integration limits: only within the BL ---
    i_cutoff = min(i_edge + 5, len(y_profile))  # Small buffer past edge
    i_cutoff = max(i_cutoff, 10)
    
    y_bl = y_profile[:i_cutoff]
    U_bl = U_Uinf[:i_cutoff]
    # Ensure profile reaches ~1.0 at the edge for proper integration
    U_bl_capped = np.minimum(U_bl, 1.0)
    
    # Displacement thickness δ*
    integrand_disp = 1.0 - U_bl_capped
    delta_star = np.trapz(integrand_disp, y_bl)
    delta_star = max(delta_star, 0.0)
    
    # Momentum thickness θ
    integrand_mom = U_bl_capped * (1.0 - U_bl_capped)
    theta = np.trapz(integrand_mom, y_bl)
    theta = max(theta, 0.0)
    
    # Shape factor H
    H = delta_star / theta if theta > 1e-12 else np.nan
    
    # --- Wall shear stress ---
    n_fit = min(6, len(y_profile))
    mu_wall = sutherland_viscosity(T_wall)
    coeffs = np.polyfit(y_profile[:n_fit], u_tangential[:n_fit], 1)
    du_ds_wall = coeffs[0]
    tau_w = mu_wall * du_ds_wall
    
    # Skin friction coefficient
    q_inf = 0.5 * rho_inf * u_inf**2
    C_f = tau_w / q_inf
    
    # Reynolds numbers
    x_station = x_cc[i_surf]
    T_far = T_profile[-1]
    mu_far = sutherland_viscosity(T_far)
    
    Re_x = rho_inf * u_inf * x_station / mu_far
    Re_theta = rho_inf * u_inf * theta / mu_far if theta > 0 else 0
    
    results = {
        'y_profile': y_profile,
        'U_Uinf': U_Uinf,
        'u_profile': u_tangential,
        'u_edge': u_edge,
        'T_profile': T_profile,
        'tau_w': tau_w,
        'C_f': C_f,
        'delta_99': delta_99,
        'delta_star': delta_star,
        'theta': theta,
        'H': H,
        'Re_x': Re_x,
        'Re_theta': Re_theta,
    }
    
    return results


def extract_surface_properties(x_cc, y_cc, xx, yy, fields_data, surfaces, 
                                rho_inf=0.0267, u_inf=1011, T_inf=None):
    """
    Extract surface properties at all surface points
    """
    
    surface_data = {}
    
    for side in ['upper', 'lower']:
        if len(surfaces[side]['x']) == 0:
            continue
            
        n_points = len(surfaces[side]['x'])
        print(f"    Processing {side} surface: {n_points} points")
        
        # Initialize storage
        data = {
            'x': surfaces[side]['x'],
            'y': surfaces[side]['y'],
            's': np.zeros(n_points),
            'p': np.zeros(n_points),
            'Temperature': np.zeros(n_points),
            'rho': np.zeros(n_points),
            'tau_w': np.zeros(n_points),
            'C_f': np.zeros(n_points),
            'C_p': np.zeros(n_points),
            'q_w': np.zeros(n_points),
            'Re_x': np.zeros(n_points),
            'Re_theta': np.zeros(n_points),
            'delta_99': np.zeros(n_points),
            'delta_star': np.zeros(n_points),
            'theta': np.zeros(n_points),
            'H': np.zeros(n_points),
        }
        
        # Calculate arc length
        dx_arr = np.diff(surfaces[side]['x'])
        dy_arr = np.diff(surfaces[side]['y'])
        ds_arr = np.sqrt(dx_arr**2 + dy_arr**2)
        data['s'][1:] = np.cumsum(ds_arr)
        
        # Check if normals are available
        has_normals = 'nx' in surfaces[side] and 'ny' in surfaces[side]
        
        n_success = 0
        n_failed = 0
        
        # Extract surface values at each point
        for idx in range(n_points):
            i = int(surfaces[side]['i'][idx])
            j = int(surfaces[side]['j'][idx])
            
            # Surface values (at wall)
            data['p'][idx] = fields_data['p'][i, j]
            data['Temperature'][idx] = fields_data['Temperature'][i, j]
            data['rho'][idx] = fields_data['rho'][i, j]
            
            # Get normal direction
            if has_normals:
                nx = surfaces[side]['nx'][idx]
                ny = surfaces[side]['ny'][idx]
            else:
                nx = 0.0
                ny = 1.0 if side == 'upper' else -1.0
            
            try:
                bl_results = extract_BL_profile_at_surface(
                    x_cc, y_cc, fields_data, 
                    i, j, nx, ny,
                    u_inf, rho_inf
                )
                
                data['tau_w'][idx] = bl_results['tau_w']
                data['C_f'][idx] = bl_results['C_f']
                data['delta_99'][idx] = bl_results['delta_99']
                data['delta_star'][idx] = bl_results['delta_star']
                data['theta'][idx] = bl_results['theta']
                data['H'][idx] = bl_results['H']
                data['Re_x'][idx] = bl_results['Re_x']
                data['Re_theta'][idx] = bl_results['Re_theta']
                n_success += 1
                
            except Exception as e:
                if n_failed < 5:
                    print(f"      Warning at point {idx} (x={surfaces[side]['x'][idx]:.4f}): {e}")
                data['tau_w'][idx] = np.nan
                data['C_f'][idx] = np.nan
                data['delta_99'][idx] = np.nan
                data['delta_star'][idx] = np.nan
                data['theta'][idx] = np.nan
                data['H'][idx] = np.nan
                data['Re_x'][idx] = np.nan
                data['Re_theta'][idx] = np.nan
                n_failed += 1
        
        print(f"      BL extraction: {n_success} successful, {n_failed} failed")
        
        # ------------------------------------------------------------------ #
        #  POST-EXTRACTION SMOOTHING                                          #
        #  Point-by-point BL extraction on IBM staircase grids is inherently  #
        #  noisy. Smooth all derived quantities to recover physical trends.   #
        #  Uses: median filter (kills isolated spikes) then Savitzky-Golay   #
        #  (smooths the trend while preserving real features like the flare   #
        #  reattachment bump).                                                #
        # ------------------------------------------------------------------ #
        from scipy.signal import savgol_filter
        from scipy.ndimage import median_filter

        # How many surface points correspond to ~5 mm of body arc length?
        # Adapt the window to the actual point density to keep it physical.
        arc_length = data['s'][-1] if data['s'][-1] > 0 else 1.0
        pts_per_mm = n_points / (arc_length * 1000.0)
        # Target ~5 mm median spike-kill, ~15 mm Savitzky-Golay smooth
        med_win  = max(3, int(pts_per_mm * 5))
        sg_win   = max(med_win * 3 + 1, 21)
        # Windows must be odd
        if med_win % 2 == 0: med_win  += 1
        if sg_win  % 2 == 0: sg_win   += 1
        # Don't exceed available data
        med_win = min(med_win, n_points if n_points % 2 == 1 else n_points - 1)
        sg_win  = min(sg_win,  n_points if n_points % 2 == 1 else n_points - 1)

        def smooth_field(arr):
            """Median spike-removal then Savitzky-Golay trend smoothing."""
            a = arr.copy().astype(float)
            valid = np.isfinite(a)
            if np.sum(valid) < sg_win:
                return a  # not enough data
            # Fill NaN gaps by linear interpolation so filters don't break
            x_idx = np.arange(len(a))
            a[~valid] = np.interp(x_idx[~valid], x_idx[valid], a[valid])
            a = median_filter(a, size=med_win)          # kill spikes
            a = savgol_filter(a, sg_win, polyorder=3)   # smooth trend
            a[~valid] = np.nan                          # restore NaNs
            return a

        for key in ['C_f', 'tau_w', 'delta_99', 'delta_star', 'theta', 'H']:
            data[key] = smooth_field(data[key])

        print(f"      Smoothing: median window={med_win} pts, "
              f"SG window={sg_win} pts ({sg_win/pts_per_mm:.1f} mm)")

        # Calculate pressure coefficient
        q_inf = 0.5 * rho_inf * u_inf**2
        p_inf = rho_inf * 287.05 * (T_inf if T_inf else np.mean(data['Temperature']))
        data['C_p'] = (data['p'] - p_inf) / q_inf
        
        surface_data[side] = data
    
    return surface_data

def plot_surface_geometry(surfaces, fields_data, xx, yy, snapshot, output_dir='Surface_Analysis'):
    """Plot the detected surface geometry - SIMPLIFIED VERSION"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys
    
    try:
        print(f"  [DEBUG] Creating output directory: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"  [DEBUG] Creating figure...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # SIMPLIFIED: Skip the heavy pcolormesh, just plot surfaces
        print(f"  [DEBUG] Plotting surfaces only (skipping velocity field)...")
        
        for side in ['upper', 'lower']:
            n_points = len(surfaces[side]['x'])
            print(f"    {side}: {n_points} points")
            
            if n_points > 0:
                color = 'red' if side == 'upper' else 'blue'
                # Plot both cell centers and interpolated surface
                ax.plot(surfaces[side]['x'], surfaces[side]['y'], 
                       'o', color=color, markersize=1, alpha=0.3, label=f'{side} (cells)')
                ax.plot(surfaces[side]['x'], surfaces[side]['y_interp'], 
                       '-', color=color, linewidth=2, label=f'{side} (interpolated)')
        
        print(f"  [DEBUG] Formatting plot...")
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        ax.set_title(f'Surface Geometry Detection - Snapshot {snapshot}', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Set reasonable axis limits
        all_x = np.concatenate([surfaces['upper']['x'], surfaces['lower']['x']]) if len(surfaces['lower']['x']) > 0 else surfaces['upper']['x']
        all_y = np.concatenate([surfaces['upper']['y_interp'], surfaces['lower']['y_interp']]) if len(surfaces['lower']['y_interp']) > 0 else surfaces['upper']['y_interp']
        
        if len(all_x) > 0:
            ax.set_xlim(all_x.min() - 0.05, all_x.max() + 0.05)
            ax.set_ylim(all_y.min() - 0.02, all_y.max() + 0.02)
        
        print(f"  [DEBUG] Applying tight_layout...")
        plt.tight_layout()
        
        print(f"  [DEBUG] Saving...")
        output_file = output_path / f'geometry_{snapshot}.png'
        sys.stdout.flush()
        
        plt.savefig(str(output_file), dpi=150, format='png')
        plt.close(fig)
        
        if output_file.exists():
            print(f"  ✓ Saved: {output_file} ({output_file.stat().st_size} bytes)")
        else:
            print(f"  ✗ File not created!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR in plot_surface_geometry:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        try:
            plt.close('all')
        except:
            pass
        
        return False

def plot_surface_properties_simple(surface_data, snapshot, output_dir='Surface_Analysis'):
    """Plot surface properties"""
    import matplotlib
    matplotlib.use('Agg')  # Ensure non-interactive backend
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    try:
        print(f"  [DEBUG] Creating output directory: {output_dir}")
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        print(f"  [DEBUG] Creating figure...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = {'upper': 'red', 'lower': 'blue'}
        
        print(f"  [DEBUG] Plotting surface data...")
        for side in ['upper', 'lower']:
            if side not in surface_data:
                print(f"    {side}: not in surface_data")
                continue
            
            data = surface_data[side]
            n_points = len(data['x'])
            print(f"    {side}: {n_points} points")
            
            if n_points == 0:
                print(f"      Skipping {side} (no data)")
                continue
                
            color = colors[side]
            
            axes[0, 0].plot(data['x'], data['C_p'], label=side, color=color, linewidth=2)
            axes[0, 1].plot(data['x'], data['C_f'], label=side, color=color, linewidth=2)
            axes[1, 0].plot(data['x'], data['p'], label=side, color=color, linewidth=2)
            axes[1, 1].plot(data['x'], data['Temperature'], label=side, color=color, linewidth=2)
        
        print(f"  [DEBUG] Formatting plots...")
        axes[0, 0].set_ylabel(r'$C_p$'); axes[0, 0].legend(); axes[0, 0].grid()
        axes[0, 1].set_ylabel(r'$C_f$'); axes[0, 1].legend(); axes[0, 1].grid()
        axes[1, 0].set_ylabel('Pressure (Pa)'); axes[1, 0].legend(); axes[1, 0].grid()
        axes[1, 1].set_ylabel('Temperature (K)'); axes[1, 1].legend(); axes[1, 1].grid()
        for ax in axes.flat:
            ax.set_xlabel('x (m)')
        
        print(f"  [DEBUG] Saving plot...")
        plt.suptitle(f'Surface Properties - Snapshot {snapshot}')
        plt.tight_layout()
        output_file = f'{output_dir}/properties_{snapshot}.png'
        
        print(f"  [DEBUG] Calling savefig for: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        
        print(f"  [DEBUG] Closing figure...")
        plt.close(fig)
        
        print(f"  ✓ Saved: {output_file}")

    except Exception as e:
        print(f"  ✗ ERROR in plot_surface_properties_simple:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close('all')
        except:
            pass


def plot_surface_properties_group(surface_data_dict, snapshot_list, 
                                   output_dir='Surface_Analysis'):
    """
    Plot surface properties for multiple snapshots on the same axes.
    
    Parameters:
    -----------
    surface_data_dict : dict
        Keys are snapshot numbers, values are surface_data dicts
        e.g. {100000: surface_data_100k, 200000: surface_data_200k, ...}
    snapshot_list : list
        Ordered list of snapshot numbers to plot
    output_dir : str
        Output directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Line styles cycle through snapshots — color encodes side, style encodes time
    side_colors  = {'upper': 'red',  'lower': 'blue'}
    linestyles   = ['-', '--', '-.', ':']          # up to 4 snapshots
    linewidths   = [2.5, 2.0, 2.0, 2.0]

    for sn_idx, sn in enumerate(snapshot_list):
        if sn not in surface_data_dict:
            print(f"    Warning: snapshot {sn} not found in data dict, skipping")
            continue

        surface_data = surface_data_dict[sn]
        ls  = linestyles[sn_idx % len(linestyles)]
        lw  = linewidths[sn_idx % len(linewidths)]
        sn_label = f't={sn:,}'

        for side in ['upper', 'lower']:
            if side not in surface_data:
                continue

            data  = surface_data[side]
            color = side_colors[side]
            label = f'{side} {sn_label}'

            if len(data['x']) == 0:
                continue

            axes[0, 0].plot(data['x'], data['C_p'],
                            ls, color=color, linewidth=lw, label=label)
            axes[0, 1].plot(data['x'], data['C_f'],
                            ls, color=color, linewidth=lw, label=label)
            axes[1, 0].plot(data['x'], data['p'],
                            ls, color=color, linewidth=lw, label=label)
            axes[1, 1].plot(data['x'], data['Temperature'],
                            ls, color=color, linewidth=lw, label=label)

    # Formatting
    axes[0, 0].set_ylabel(r'$C_p$',          fontsize=18)
    axes[0, 1].set_ylabel(r'$C_f$',          fontsize=18)
    axes[1, 0].set_ylabel('Pressure (Pa)',    fontsize=18)
    axes[1, 1].set_ylabel('Temperature (K)', fontsize=18)

    titles = ['Pressure Coefficient', 'Skin Friction Coefficient',
              'Wall Pressure',        'Wall Temperature']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x (m)', fontsize=14)
        ax.legend(fontsize=10, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)

    snap_str = '_'.join(str(s) for s in snapshot_list)
    plt.suptitle(f'Surface Properties — Snapshots {", ".join(str(s) for s in snapshot_list)}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = f'{output_dir}/surface_properties_group_{snap_str}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_file}")

    return output_file


# ============================================================================
#                         GEOMETRY DEFINITION
# ============================================================================

def define_CCF_geometry(x_surface,
                        theta_cone_deg=7.0,
                        x_cone_end=0.2069,
                        r_cylinder=0.0254,
                        x_flare_start=0.3847,
                        theta_flare_deg=10.0,
                        x_nose=0.0):
    """
    Assign analytic surface angle to each surface point based on known
    cone-cylinder-flare geometry.
    
    The geometry has three constant-angle sections:
        1. Cone:     θ = theta_cone   from x_nose to x_cone_end
        2. Cylinder: θ = 0            from x_cone_end to x_flare_start
        3. Flare:    θ = theta_flare  from x_flare_start onward
    
    Transition regions get a smooth blend over ±Δx to avoid discontinuities.
    
    Parameters:
    -----------
    x_surface : array
        Streamwise surface x-coordinates
    theta_cone_deg : float
        Cone half-angle (degrees)
    x_cone_end : float
        x-location where cone meets cylinder
    r_cylinder : float
        Cylinder radius (meters) — also the body radius on the cylinder section
    x_flare_start : float
        x-location where flare begins
    theta_flare_deg : float
        Flare half-angle (degrees)
    x_nose : float
        x-location of nose tip
    
    Returns:
    --------
    theta_local : array
        Surface inclination angle at each point (radians), always >= 0
    section_id : array of str
        Label for each point: 'cone', 'cyl-junc', 'cylinder', 'flare-junc', 'flare'
    geometry_info : dict
        Summary of geometry parameters
    """
    theta_cone = np.radians(theta_cone_deg)
    theta_flare = np.radians(theta_flare_deg)
    
    x = np.asarray(x_surface, dtype=float)
    theta_local = np.zeros_like(x)
    section_id = np.empty(len(x), dtype='U12')
    
    # Transition half-width (smooth blend region)
    dx_blend = 0.005  # 5 mm blend zone on each side of junction
    
    for i, xi in enumerate(x):
        if xi < x_cone_end - dx_blend:
            # Pure cone
            theta_local[i] = theta_cone
            section_id[i] = 'cone'
            
        elif xi < x_cone_end + dx_blend:
            # Cone → Cylinder transition (smooth blend)
            t = (xi - (x_cone_end - dx_blend)) / (2.0 * dx_blend)
            t = np.clip(t, 0, 1)
            # Smooth step (cubic Hermite)
            s = 3 * t**2 - 2 * t**3
            theta_local[i] = theta_cone * (1.0 - s)
            section_id[i] = 'cyl-junc'
            
        elif xi < x_flare_start - dx_blend:
            # Pure cylinder
            theta_local[i] = 0.0
            section_id[i] = 'cylinder'
            
        elif xi < x_flare_start + dx_blend:
            # Cylinder → Flare transition (smooth blend)
            t = (xi - (x_flare_start - dx_blend)) / (2.0 * dx_blend)
            t = np.clip(t, 0, 1)
            s = 3 * t**2 - 2 * t**3
            theta_local[i] = theta_flare * s
            section_id[i] = 'flare-junc'
            
        else:
            # Pure flare
            theta_local[i] = theta_flare
            section_id[i] = 'flare'
    
    geometry_info = {
        'theta_cone_deg': theta_cone_deg,
        'theta_cone_rad': theta_cone,
        'x_cone_end': x_cone_end,
        'r_cylinder': r_cylinder,
        'x_flare_start': x_flare_start,
        'theta_flare_deg': theta_flare_deg,
        'theta_flare_rad': theta_flare,
        'x_nose': x_nose,
    }
    
    # Print section breakdown
    n_cone = np.sum(section_id == 'cone')
    n_cjunc = np.sum(section_id == 'cyl-junc')
    n_cyl = np.sum(section_id == 'cylinder')
    n_fjunc = np.sum(section_id == 'flare-junc')
    n_flare = np.sum(section_id == 'flare')
    print(f"    Geometry breakdown: {n_cone} cone, {n_cjunc} cone-cyl junc, "
          f"{n_cyl} cylinder, {n_fjunc} cyl-flare junc, {n_flare} flare")
    
    return theta_local, section_id, geometry_info


# ============================================================================
#                    THEORETICAL / SEMI-EMPIRICAL METHODS
# ============================================================================

def oblique_shock_Cp(M_inf, gamma, theta_deflection):
    """
    Compute post-oblique-shock Cp for a WEDGE at deflection angle theta.
    Uses the theta-beta-Mach relation (weak shock solution).
    """
    from scipy.optimize import brentq
    
    if theta_deflection <= 0 or M_inf <= 1.0:
        return 0.0, np.nan, M_inf, 1.0
    
    def theta_beta_mach(beta, M, gam):
        M2sin2 = (M * np.sin(beta))**2
        num = 2.0 * (M2sin2 - 1.0) / np.tan(beta)
        den = M**2 * (gam + np.cos(2.0 * beta)) + 2.0
        return np.arctan2(num, den)
    
    beta_min = np.arcsin(1.0 / M_inf) + 1e-8
    beta_max = np.pi / 2.0 - 1e-8
    
    n_scan = 200
    betas = np.linspace(beta_min, beta_max, n_scan)
    thetas = np.array([theta_beta_mach(b, M_inf, gamma) for b in betas])
    theta_max = np.nanmax(thetas)
    
    if theta_deflection > theta_max * 0.98:
        return np.nan, np.nan, np.nan, np.nan
    
    idx_max = np.nanargmax(thetas)
    
    def residual(beta):
        return theta_beta_mach(beta, M_inf, gamma) - theta_deflection
    
    try:
        beta = brentq(residual, beta_min, betas[idx_max], xtol=1e-10)
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan
    
    M1n = M_inf * np.sin(beta)
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1n**2 - 1.0)
    
    M2n_sq = (1.0 + 0.5 * (gamma - 1.0) * M1n**2) / \
             (gamma * M1n**2 - 0.5 * (gamma - 1.0))
    if M2n_sq < 0:
        return np.nan, np.nan, np.nan, np.nan
    M2 = np.sqrt(M2n_sq) / np.sin(beta - theta_deflection)
    
    Cp = (p2_p1 - 1.0) / (0.5 * gamma * M_inf**2)
    return Cp, beta, M2, p2_p1


def taylor_maccoll_cone_Cp(M_inf, gamma, theta_cone):
    """
    Solve the Taylor-Maccoll equation for CONICAL flow.
    Surface pressure on a cone is LOWER than the post-oblique-shock
    pressure for the equivalent wedge.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    
    if theta_cone <= 0 or theta_cone < np.radians(0.1) or M_inf <= 1.0:
        return 0.0, np.nan, M_inf
    
    def taylor_maccoll_ode(theta, V, gamma):
        Vr, Vt = V
        a2 = (gamma - 1.0) / 2.0 * (1.0 - Vr**2 - Vt**2)
        if a2 <= 0 or abs(theta) < 1e-10:
            return [0.0, 0.0]
        numerator = Vt**2 * Vr - a2 * (2.0 * Vr + Vt / np.tan(theta))
        denominator = a2 - Vt**2
        if abs(denominator) < 1e-12:
            return [0.0, 0.0]
        return [Vt, numerator / denominator]
    
    def get_post_shock(beta_guess):
        M1n = M_inf * np.sin(beta_guess)
        if M1n <= 1.0:
            return None
        M2n_sq = (1.0 + 0.5 * (gamma - 1.0) * M1n**2) / \
                 (gamma * M1n**2 - 0.5 * (gamma - 1.0))
        if M2n_sq <= 0:
            return None
        M2n = np.sqrt(M2n_sq)
        tan_delta = 2.0 / np.tan(beta_guess) * \
                    (M1n**2 - 1.0) / (M_inf**2 * (gamma + np.cos(2.0 * beta_guess)) + 2.0)
        delta = np.arctan(tan_delta)
        M2 = M2n / np.sin(beta_guess - delta)
        V_total_norm = M2 / np.sqrt(1.0 + (gamma - 1.0) / 2.0 * M2**2)
        Vr = V_total_norm * np.cos(beta_guess - delta)
        Vt = -V_total_norm * np.sin(beta_guess - delta)
        return Vr, Vt, M1n, delta, M2
    
    def cone_residual(beta_guess):
        if beta_guess <= theta_cone:
            return 1.0
        result = get_post_shock(beta_guess)
        if result is None:
            return 1.0
        Vr, Vt, _, _, _ = result
        sol = solve_ivp(taylor_maccoll_ode, [beta_guess, theta_cone], [Vr, Vt],
                        args=(gamma,), method='RK45', rtol=1e-6, atol=1e-8, max_step=0.05)
        if not sol.success or len(sol.y[1]) == 0:
            return 1.0
        return sol.y[1][-1]
    
    beta_mach = np.arcsin(1.0 / M_inf) + 0.002
    beta_max = min(np.pi / 2.0 - 0.01, theta_cone + np.radians(50))
    
    betas = np.linspace(beta_mach, beta_max * 0.9, 30)
    residuals = np.array([cone_residual(b) for b in betas])
    sign_changes = np.where(np.diff(np.sign(residuals)))[0]
    
    if len(sign_changes) == 0:
        return np.nan, np.nan, np.nan
    
    try:
        beta_shock = brentq(cone_residual, betas[sign_changes[0]],
                           betas[sign_changes[0] + 1], xtol=1e-6)
    except (ValueError, IndexError):
        return np.nan, np.nan, np.nan
    
    result = get_post_shock(beta_shock)
    if result is None:
        return np.nan, np.nan, np.nan
    Vr_0, Vt_0, M1n, delta, M2 = result
    
    sol = solve_ivp(taylor_maccoll_ode, [beta_shock, theta_cone], [Vr_0, Vt_0],
                    args=(gamma,), method='RK45', rtol=1e-6, atol=1e-8, max_step=0.05)
    
    Vr_surface = sol.y[0][-1]
    V_surface = abs(Vr_surface)
    denom = (gamma - 1.0) / 2.0 * (1.0 - V_surface**2)
    if denom <= 0:
        return np.nan, np.nan, np.nan
    M_surface_sq = V_surface**2 / denom
    if M_surface_sq <= 0:
        return np.nan, np.nan, np.nan
    M_surface = np.sqrt(M_surface_sq)
    
    p02_p01 = ((gamma + 1.0) * M1n**2 / ((gamma - 1.0) * M1n**2 + 2.0))**(gamma / (gamma - 1.0)) * \
              ((gamma + 1.0) / (2.0 * gamma * M1n**2 - (gamma - 1.0)))**(1.0 / (gamma - 1.0))
    p01_pinf = (1.0 + (gamma - 1.0) / 2.0 * M_inf**2)**(gamma / (gamma - 1.0))
    p_surf_p02 = (1.0 + (gamma - 1.0) / 2.0 * M_surface**2)**(- gamma / (gamma - 1.0))
    p_ratio = p02_p01 * p01_pinf * p_surf_p02
    Cp_cone = (p_ratio - 1.0) / (0.5 * gamma * M_inf**2)
    
    return Cp_cone, beta_shock, M_surface


def newton_Cp(theta_local, M_inf, gamma=1.4):
    """Newton's method: Cp = 2 sin²(θ). Leeward → Cp_vacuum."""
    theta_local = np.asarray(theta_local, dtype=float)
    Cp_vacuum = -2.0 / (gamma * M_inf**2)
    return np.where(theta_local > 0, 2.0 * np.sin(theta_local)**2, Cp_vacuum)


def modified_newton_Cp(theta_local, M_inf, gamma=1.4):
    """Modified Newton: Cp = Cp_max sin²(θ) with Rayleigh Pitot Cp_max."""
    theta_local = np.asarray(theta_local, dtype=float)
    M2 = M_inf**2
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    term1 = (gp1**2 * M2) / (4.0 * gamma * M2 - 2.0 * gm1)
    term2 = (1.0 - gamma + 2.0 * gamma * M2) / gp1
    p0_p_inf = term1**(gamma / gm1) * term2
    Cp_max = 2.0 / (gamma * M2) * (p0_p_inf - 1.0)
    Cp_vacuum = -2.0 / (gamma * M2)
    Cp = np.where(theta_local > 0, Cp_max * np.sin(theta_local)**2, Cp_vacuum)
    return Cp, Cp_max


def tangent_cone_Cp(theta_local, M_inf, gamma=1.4):
    """
    Tangent-cone Cp via pre-computed Taylor-Maccoll lookup table.
    Only solves ODE for the unique angles present, then interpolates.
    """
    from scipy.interpolate import interp1d
    
    theta_local = np.asarray(theta_local, dtype=float)
    theta_pos = theta_local[theta_local > np.radians(0.1)]
    Cp_vacuum = -2.0 / (gamma * M_inf**2)
    
    if len(theta_pos) == 0:
        return np.full_like(theta_local, Cp_vacuum)
    
    # Build lookup table
    th_min = max(0.2, np.degrees(theta_pos.min()) - 0.5)
    th_max = min(50.0, np.degrees(theta_pos.max()) + 2.0)
    n_table = 50
    thetas_deg = np.linspace(th_min, th_max, n_table)
    thetas_rad = np.radians(thetas_deg)
    Cp_table = np.zeros(n_table)
    
    n_ok, n_fail = 0, 0
    for i, th in enumerate(thetas_rad):
        cp, _, _ = taylor_maccoll_cone_Cp(M_inf, gamma, th)
        if np.isfinite(cp):
            Cp_table[i] = cp
            n_ok += 1
        else:
            Cp_table[i] = np.nan
            n_fail += 1
    
    # Fill NaN with modified Newton
    _, Cp_max = modified_newton_Cp(np.array([np.pi/2]), M_inf, gamma)
    for i in range(n_table):
        if np.isnan(Cp_table[i]):
            Cp_table[i] = Cp_max * np.sin(thetas_rad[i])**2
    
    print(f"    Tangent-cone table: {n_ok}/{n_table} converged, {n_fail} fallback")
    
    interp_func = interp1d(thetas_rad, Cp_table, kind='cubic',
                           bounds_error=False, fill_value=(Cp_table[0], Cp_table[-1]))
    
    Cp = np.where(theta_local > np.radians(0.1), interp_func(theta_local), Cp_vacuum)
    return Cp


# ============================================================================
#  SKIN FRICTION: LAMINAR & TURBULENT WITH MANGLER TRANSFORMATION
# ============================================================================

def mangler_transform(x_surface, y_surface):
    """Mangler transformation for axisymmetric BL."""
    r = np.abs(y_surface)
    r_safe = np.maximum(r, 1e-10)
    dx = np.diff(x_surface)
    dy = np.diff(y_surface)
    ds = np.sqrt(dx**2 + dy**2)
    L = np.sum(ds)
    integrand = (r_safe / L)**2
    x_mangler = np.zeros_like(x_surface)
    x_mangler[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * np.abs(dx))
    return x_mangler, r_safe


def Cf_laminar_flatplate(Re_x):
    """Blasius: Cf = 0.664 / √(Re_x)"""
    Re_x = np.asarray(Re_x, dtype=float)
    return np.where(Re_x > 0, 0.664 / np.sqrt(np.maximum(Re_x, 1.0)), 0.0)


def Cf_turbulent_flatplate(Re_x, method='schultz-grunow'):
    """Turbulent flat-plate Cf correlations."""
    Re_x = np.asarray(Re_x, dtype=float)
    if method == 'schultz-grunow':
        return np.where(Re_x > 10, 0.370 / (np.log10(np.maximum(Re_x, 10.0)))**2.584, 0.0)
    elif method == 'power-law':
        return np.where(Re_x > 0, 0.0592 / np.maximum(Re_x, 1.0)**0.2, 0.0)
    elif method == 'schlichting':
        return np.where(Re_x > 10, 0.455 / (np.log10(np.maximum(Re_x, 10.0)))**2.58, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")


def Cf_compressible_correction(Cf_incomp, M_edge, T_wall, T_inf, gamma=1.4,
                                method='reference-temperature', flow='turbulent'):
    """Compressibility correction for Cf (Eckert ref-temp or Van Driest II)."""
    Cf_incomp = np.asarray(Cf_incomp, dtype=float)
    M_edge = np.asarray(M_edge, dtype=float)
    T_wall = np.asarray(T_wall, dtype=float)
    T_inf = np.asarray(T_inf, dtype=float)
    
    if method == 'reference-temperature':
        r = 0.85 if flow == 'turbulent' else 0.72
        T_star = T_inf * (0.5 * (1.0 + T_wall / T_inf) +
                          0.22 * r * 0.5 * (gamma - 1.0) * M_edge**2)
        exponent = -0.5 if flow == 'laminar' else -0.4
        with np.errstate(divide='ignore', invalid='ignore'):
            correction = np.where(T_inf > 0, (T_star / T_inf)**exponent, 1.0)
            correction = np.where(np.isfinite(correction) & (correction > 0), correction, 1.0)
        return Cf_incomp * correction
    elif method == 'van-driest-II':
        r = 0.89 if flow == 'turbulent' else 0.72
        T_aw = T_inf * (1.0 + r * (gamma - 1.0) / 2.0 * M_edge**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_ratio = T_aw / np.maximum(T_wall, 1.0)
            F_c = np.where(M_edge > 0.3, T_ratio**0.4, 1.0)
            F_c = np.where(np.isfinite(F_c) & (F_c > 0), F_c, 1.0)
        return Cf_incomp / F_c
    else:
        raise ValueError(f"Unknown method: {method}")


def Cf_laminar_mangler(x_surface, y_surface, rho_inf, u_inf, mu_inf,
                        M_edge=None, T_wall=None, T_inf=None, compressible=True):
    """Laminar Cf with Mangler transform. Cone factor = √3."""
    dx = np.diff(x_surface); dy = np.diff(y_surface)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros_like(x_surface); s[1:] = np.cumsum(ds)
    r = np.abs(y_surface); r_safe = np.maximum(r, 1e-10)
    Re_s = rho_inf * u_inf * s / mu_inf
    Cf_flat = Cf_laminar_flatplate(Re_s)
    r2 = r_safe**2
    r2_integral = np.zeros_like(s)
    r2_integral[1:] = np.cumsum(0.5 * (r2[:-1] + r2[1:]) * ds)
    with np.errstate(divide='ignore', invalid='ignore'):
        mf = np.sqrt(r2 * np.maximum(s, 1e-15) / np.maximum(r2_integral, 1e-30))
        mf = np.where(np.isfinite(mf), mf, np.sqrt(3))
    mf[0] = np.sqrt(3)
    Cf = Cf_flat * mf
    if compressible and M_edge is not None and T_wall is not None and T_inf is not None:
        Cf = Cf_compressible_correction(Cf, M_edge, T_wall, T_inf,
                                         method='reference-temperature', flow='laminar')
    return Cf, Re_s


def Cf_turbulent_mangler(x_surface, y_surface, rho_inf, u_inf, mu_inf,
                          M_edge=None, T_wall=None, T_inf=None,
                          compressible=True, turb_method='schultz-grunow'):
    """Turbulent Cf with Mangler transform. Cone factor = 3^(1/5)."""
    dx = np.diff(x_surface); dy = np.diff(y_surface)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros_like(x_surface); s[1:] = np.cumsum(ds)
    r = np.abs(y_surface); r_safe = np.maximum(r, 1e-10)
    Re_s = rho_inf * u_inf * s / mu_inf
    Cf_flat = Cf_turbulent_flatplate(Re_s, method=turb_method)
    r2 = r_safe**2
    r2_integral = np.zeros_like(s)
    r2_integral[1:] = np.cumsum(0.5 * (r2[:-1] + r2[1:]) * ds)
    with np.errstate(divide='ignore', invalid='ignore'):
        mf = (r2 * np.maximum(s, 1e-15) / np.maximum(r2_integral, 1e-30))**(1.0/5.0)
        mf = np.where(np.isfinite(mf), mf, 3**(1.0/5.0))
    mf[0] = 3**(1.0/5.0)
    Cf = Cf_flat * mf
    if compressible and M_edge is not None and T_wall is not None and T_inf is not None:
        Cf = Cf_compressible_correction(Cf, M_edge, T_wall, T_inf,
                                         method='reference-temperature', flow='turbulent')
    return Cf, Re_s


# ============================================================================
#                       MAIN DRIVER & PLOTTING
# ============================================================================

def compute_theoretical_comparison(surfaces, surface_data, M_inf, gamma=1.4,
                                    rho_inf=0.0267, u_inf=1011, T_inf=None,
                                    theta_cone_deg=7.0, x_cone_end=0.2069,
                                    r_cylinder=0.0254, x_flare_start=0.387,
                                    theta_flare_deg=10.0):
    """
    Compute all theoretical Cp and Cf curves for comparison with CFD.
    
    Uses ANALYTIC geometry definition instead of noisy IBM surface angles.
    
    Computes:
    - Newton Cp, Modified Newton Cp, Tangent-cone Cp (Taylor-Maccoll)
    - Laminar & Turbulent Cf (Mangler + compressibility)
    
    Also stores BL thickness data from CFD for plotting.
    """
    R = 287.05
    if T_inf is None:
        T_inf = u_inf**2 / (gamma * R * M_inf**2)
    
    mu_inf = sutherland_viscosity(T_inf)
    
    print(f"  Theoretical comparison parameters:")
    print(f"    M_inf = {M_inf:.2f}, gamma = {gamma}")
    print(f"    T_inf = {T_inf:.1f} K, mu_inf = {mu_inf:.3e} Pa·s")
    print(f"    rho_inf = {rho_inf:.4f} kg/m³, u_inf = {u_inf:.1f} m/s")
    print(f"    Geometry: cone {theta_cone_deg}° → cyl at x={x_cone_end} → flare {theta_flare_deg}° at x={x_flare_start}")
    
    theory = {}
    
    for side in ['upper', 'lower']:
        if side not in surface_data or len(surfaces[side]['x']) == 0:
            continue
        
        x = surfaces[side]['x']
        y = surfaces[side].get('y_smooth', surfaces[side]['y_interp'])
        
        # --- ANALYTIC GEOMETRY ANGLES ---
        theta_local, section_id, geom_info = define_CCF_geometry(
            x, theta_cone_deg=theta_cone_deg, x_cone_end=x_cone_end,
            r_cylinder=r_cylinder, x_flare_start=x_flare_start,
            theta_flare_deg=theta_flare_deg
        )
        
        # --- Cp METHODS ---
        Cp_newton = newton_Cp(theta_local, M_inf, gamma)
        Cp_mod_newton, Cp_max = modified_newton_Cp(theta_local, M_inf, gamma)
        
        print(f"  {side} Cp_max (Mod. Newton) = {Cp_max:.4f}")
        print(f"  Computing tangent-cone Cp (Taylor-Maccoll)...")
        Cp_tangent_cone = tangent_cone_Cp(theta_local, M_inf, gamma)
        
        # --- Cf METHODS ---
        T_wall = surface_data[side]['Temperature'] if side in surface_data else T_inf * np.ones_like(x)
        M_edge = M_inf * np.ones_like(x)
        
        Cf_lam, Re_s = Cf_laminar_mangler(
            x, y, rho_inf, u_inf, mu_inf,
            M_edge=M_edge, T_wall=T_wall, T_inf=T_inf, compressible=True
        )
        Cf_turb, _ = Cf_turbulent_mangler(
            x, y, rho_inf, u_inf, mu_inf,
            M_edge=M_edge, T_wall=T_wall, T_inf=T_inf,
            compressible=True, turb_method='schultz-grunow'
        )
        
        theory[side] = {
            'x': x,
            'theta_local': theta_local,
            'section_id': section_id,
            'Re_s': Re_s,
            'Cp_newton': Cp_newton,
            'Cp_modified_newton': Cp_mod_newton,
            'Cp_max': Cp_max,
            'Cp_tangent_cone': Cp_tangent_cone,
            'Cf_laminar': Cf_lam,
            'Cf_turbulent': Cf_turb,
            'geometry_info': geom_info,
        }
        
        print(f"  {side} Re_s range: [{Re_s[Re_s > 0].min():.0f}, {Re_s.max():.0f}]")
    
    return theory


def plot_theory_comparison(surface_data, theory, snapshot, output_dir='Surface_Data'):
    """
    Plot CFD vs. theoretical predictions.
    
    4-panel figure:
    - Top-left:  Cp (CFD vs Newton, Mod Newton, Tangent-cone)
    - Top-right: Cf (CFD vs Laminar/Turbulent Mangler)
    - Bottom-left:  BL thicknesses (δ99, δ*, θ) from CFD
    - Bottom-right: Shape factor H from CFD
    
    Geometry section boundaries shown as vertical lines on all panels.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    colors = {
        'upper': {'cfd': 'red',  't1': 'darkred',   't2': 'salmon',         't3': 'orange'},
        'lower': {'cfd': 'blue', 't1': 'darkblue',  't2': 'cornflowerblue', 't3': 'cyan'}
    }
    
    # Get geometry boundaries for vertical lines
    geom_info = None
    for side in theory:
        if 'geometry_info' in theory[side]:
            geom_info = theory[side]['geometry_info']
            break
    
    for side in ['upper', 'lower']:
        if side not in theory or side not in surface_data:
            continue
        
        x_cfd = surface_data[side]['x']
        x_th = theory[side]['x']
        c = colors[side]
        
        # --- Cp ---
        ax = axes[0, 0]
        ax.plot(x_cfd, surface_data[side]['C_p'], '-', color=c['cfd'],
                linewidth=2.5, label=f'{side} CFD')
        ax.plot(x_th, theory[side]['Cp_modified_newton'], '--', color=c['t1'],
                linewidth=2, label=f'{side} Mod. Newton')
        ax.plot(x_th, theory[side]['Cp_newton'], ':', color=c['t2'],
                linewidth=2, label=f'{side} Newton')
        ax.plot(x_th, theory[side]['Cp_tangent_cone'], '-.', color=c['t3'],
                linewidth=2, label=f'{side} Tangent Cone')
        
        # --- Cf ---
        ax = axes[0, 1]
        Cf_cfd = np.where(np.isfinite(surface_data[side]['C_f']),
                          surface_data[side]['C_f'], np.nan)
        ax.plot(x_cfd, Cf_cfd, '-', color=c['cfd'],
                linewidth=2.5, label=f'{side} CFD')
        ax.plot(x_th, theory[side]['Cf_laminar'], '--', color=c['t1'],
                linewidth=2, label=f'{side} Laminar (Mangler)')
        ax.plot(x_th, theory[side]['Cf_turbulent'], '-.', color=c['t2'],
                linewidth=2, label=f'{side} Turbulent (Mangler)')
        ax.set_ylim(bottom=np.nanmin(Cf_cfd), top=0.004)
        
        # --- BL Thicknesses ---
        ax = axes[1, 0]
        d99 = np.where(np.isfinite(surface_data[side]['delta_99']),
                       surface_data[side]['delta_99'], np.nan)
        dstar = np.where(np.isfinite(surface_data[side]['delta_star']),
                         surface_data[side]['delta_star'], np.nan)
        theta_bl = np.where(np.isfinite(surface_data[side]['theta']),
                            surface_data[side]['theta'], np.nan)

        n_valid_bl = np.sum(np.isfinite(d99))
        if n_valid_bl == 0:
            ax.text(0.5, 0.5, f'No valid BL data for {side}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='gray', fontstyle='italic')
        else:
            ax.plot(x_cfd, d99 * 1000, '-', color=c['cfd'], linewidth=2.5,
                    label=f'{side} $\\delta_{{99}}$')
            ax.plot(x_cfd, dstar * 1000, '--', color=c['t1'], linewidth=2,
                    label=f'{side} $\\delta^*$')
            ax.plot(x_cfd, theta_bl * 1000, ':', color=c['t2'], linewidth=2,
                    label=f'{side} $\\theta$')

        # --- Shape Factor ---
        ax = axes[1, 1]
        H = np.where(np.isfinite(surface_data[side]['H']),
                     surface_data[side]['H'], np.nan)
        n_valid_H = np.sum(np.isfinite(H))
        if n_valid_H == 0:
            ax.text(0.5, 0.5, f'No valid H data for {side}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='gray', fontstyle='italic')
        else:
            ax.plot(x_cfd, H, '-', color=c['cfd'], linewidth=2, label=f'{side}')
    
    # Add geometry section boundaries to all panels
    if geom_info is not None:
        for ax in axes.flat:
            ax.axvline(x=geom_info['x_cone_end'], color='gray', linewidth=2,
                      linestyle='--', alpha=0.8)
            ax.axvline(x=geom_info['x_flare_start'], color='gray', linewidth=2,
                      linestyle='--', alpha=0.8)
        
        # Add section labels to top-left panel
        ax0 = axes[0, 0]
        y_label = ax0.get_ylim()[1] * 0.92
        x_mid_cone = geom_info['x_cone_end'] / 2
        x_mid_cyl = (geom_info['x_cone_end'] + geom_info['x_flare_start']) / 2
        x_mid_flare = geom_info['x_flare_start'] + 0.1
        ax0.text(x_mid_cone, y_label, f"Cone {geom_info['theta_cone_deg']}°",
                ha='center', fontsize=14, color='gray', fontstyle='italic')
        ax0.text(x_mid_cyl, y_label, "Cylinder",
                ha='center', fontsize=14, color='gray', fontstyle='italic')
        ax0.text(x_mid_flare, y_label, f"Flare {geom_info['theta_flare_deg']}°",
                ha='center', fontsize=14, color='gray', fontstyle='italic')
    
    # Formatting
    axes[0, 0].set_ylabel(r'$C_p$', fontsize=18)
    axes[0, 0].set_title('Pressure Coefficient: Simulation vs Theory', fontsize=14)
    axes[0, 0].legend(fontsize=12, ncol=2, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('x (m)', fontsize=14)
    
    axes[0, 1].set_ylabel(r'$C_f$', fontsize=18)
    axes[0, 1].set_title('Skin Friction: Simulation vs Theory', fontsize=14)
    axes[0, 1].legend(fontsize=12, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('x (m)', fontsize=14)
    
    axes[1, 0].set_ylabel('Thickness (mm)', fontsize=18)
    axes[1, 0].set_title('Boundary Layer Thicknesses (Simulation)', fontsize=14)
    axes[1, 0].legend(fontsize=12, ncol=2, loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('x (m)', fontsize=14)
    
    axes[1, 1].set_ylabel('H', fontsize=18)
    axes[1, 1].set_title('Shape Factor H (Simulation)', fontsize=14)
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('x (m)', fontsize=14)
    # Reference lines for laminar/turbulent H
    axes[1, 1].axhline(y=2.59, color='green', linewidth=2, linestyle=':', alpha=0.9,
                       label='Blasius (H=2.59)')
    axes[1, 1].axhline(y=1.3, color='orange', linewidth=2, linestyle=':', alpha=0.9,
                       label='Turbulent (H≈1.3)')
    axes[1, 1].legend(fontsize=12, loc='best')
    
    plt.suptitle(f'Simulation vs. Theory — Snapshot {snapshot}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = f'{output_dir}/theory_comparison_{snapshot}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_file}")
    
    return output_file

# ...existing code... (imports at top of file)
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap 
from matplotlib.cm import ScalarMappable

# ========================================================================= #
#                        STL GEOMETRY FUNCTIONS                             #
# ========================================================================= #

def read_ascii_stl(stl_path):
    """
    Read an ASCII STL file and return triangle vertices.
    
    Parameters:
    -----------
    stl_path : str or Path
        Path to ASCII .stl file
    
    Returns:
    --------
    triangles : np.ndarray, shape (N, 3, 3)
        Array of N triangles, each with 3 vertices of (x, y, z)
    normals : np.ndarray, shape (N, 3)
        Face normal vectors
    """
    from pathlib import Path
    
    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    triangles = []
    normals = []
    
    with open(stl_path, 'r') as f:
        current_normal = None
        current_vertices = []
        
        for line in f:
            line = line.strip()
            
            if line.startswith('facet normal'):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                current_vertices = []
            
            elif line.startswith('vertex'):
                parts = line.split()
                current_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            elif line.startswith('endfacet'):
                if len(current_vertices) == 3 and current_normal is not None:
                    triangles.append(current_vertices)
                    normals.append(current_normal)
                current_normal = None
                current_vertices = []
    
    if len(triangles) == 0:
        raise ValueError(f"No triangles found in STL file: {stl_path}")
    
    triangles = np.array(triangles, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)
    
    print(f"  STL loaded: {len(triangles)} triangles")
    print(f"    x range: [{triangles[:,:,0].min():.6f}, {triangles[:,:,0].max():.6f}]")
    print(f"    y range: [{triangles[:,:,1].min():.6f}, {triangles[:,:,1].max():.6f}]")
    print(f"    z range: [{triangles[:,:,2].min():.6f}, {triangles[:,:,2].max():.6f}]")
    
    return triangles, normals


def transform_stl(triangles, scale=1.0, translate=(0.0, 0.0, 0.0),
                   scale_xyz=None):
    """
    Apply scaling and translation to STL triangles.
    
    Applies in order: scale, then translate.
    
    Parameters:
    -----------
    triangles : np.ndarray, shape (N, 3, 3)
        Triangle vertices
    scale : float
        Uniform scale factor (applied first). Use this if STL is in mm
        and simulation is in meters (scale=0.001).
    translate : tuple of 3 floats
        (dx, dy, dz) translation applied after scaling
    scale_xyz : tuple of 3 floats or None
        Non-uniform scaling (sx, sy, sz). Overrides `scale` if provided.
    
    Returns:
    --------
    triangles_transformed : np.ndarray, shape (N, 3, 3)
    """
    tri = triangles.copy()
    
    if scale_xyz is not None:
        sx, sy, sz = scale_xyz
        tri[:, :, 0] *= sx
        tri[:, :, 1] *= sy
        tri[:, :, 2] *= sz
    else:
        tri *= scale
    
    tri[:, :, 0] += translate[0]
    tri[:, :, 1] += translate[1]
    tri[:, :, 2] += translate[2]
    
    print(f"  STL transformed:")
    print(f"    scale = {scale_xyz if scale_xyz else scale}")
    print(f"    translate = {translate}")
    print(f"    x range: [{tri[:,:,0].min():.6f}, {tri[:,:,0].max():.6f}]")
    print(f"    y range: [{tri[:,:,1].min():.6f}, {tri[:,:,1].max():.6f}]")
    print(f"    z range: [{tri[:,:,2].min():.6f}, {tri[:,:,2].max():.6f}]")
    
    return tri


def stl_to_2d_profile(triangles, plane='xy', collapse_method='max_radius',
                      angular_tolerance_deg=5.0, analytic_fill=None, axis_offset=None):
    """
    Simplified 2D STL profile extraction.
    Builds upper/lower wall envelopes directly from STL vertices:
        y_upper(x) = max(y at x-bin)
        y_lower(x) = min(y at x-bin)

    Assumes a 2D STL cross-section (z ~ constant).
    """
    verts = triangles.reshape(-1, 3)
    x_all = verts[:, 0]
    y_all = verts[:, 1]

    if len(x_all) == 0:
        raise ValueError("Empty STL vertex list.")

    # Bin x to merge duplicate/near-duplicate stations
    xr = x_all.max() - x_all.min()
    if xr <= 1e-15:
        raise ValueError("Degenerate STL: zero x-range.")
    x_tol = xr / 1_000_000.0

    idx = np.argsort(x_all)
    x_sorted = x_all[idx]
    y_sorted = y_all[idx]

    x_bins = [x_sorted[0]]
    y_bins = [[y_sorted[0]]]
    for k in range(1, len(x_sorted)):
        if abs(x_sorted[k] - x_bins[-1]) < x_tol:
            y_bins[-1].append(y_sorted[k])
        else:
            x_bins.append(x_sorted[k])
            y_bins.append([y_sorted[k]])

    x_prof = np.array(x_bins)
    y_upper = np.array([np.max(b) for b in y_bins])
    y_lower = np.array([np.min(b) for b in y_bins])

    # Unique/monotonic x
    _, uidx = np.unique(x_prof, return_index=True)
    x_prof = x_prof[uidx]
    y_upper = y_upper[uidx]
    y_lower = y_lower[uidx]

    if len(x_prof) < 2:
        raise ValueError("Need at least 2 unique x-stations in STL.")

    # Slopes for normals
    dyu_dx = np.gradient(y_upper, x_prof)
    dyl_dx = np.gradient(y_lower, x_prof)

    print(f"  2D STL profile: {len(x_prof)} x-stations")
    print(f"    x range: [{x_prof.min():.6f}, {x_prof.max():.6f}]")
    print(f"    y_upper range: [{y_upper.min():.6f}, {y_upper.max():.6f}]")
    print(f"    y_lower range: [{y_lower.min():.6f}, {y_lower.max():.6f}]")

    return {
        'x': x_prof,
        'y_upper': y_upper,
        'y_lower': y_lower,
        'dyu_dx': dyu_dx,
        'dyl_dx': dyl_dx,
        'x_min': x_prof.min(),
        'x_max': x_prof.max(),
    }


def stl_profile_at_plane(triangles, plane='xy', coord=0.0):
    """
    Compute the 2D cross-section of a 3D STL at a coordinate plane and
    return a profile dict compatible with build_stl_geometry_mask().

    Finds every triangle edge that crosses the plane, collects the
    intersection points, and builds upper/lower envelopes — identical
    in format to stl_to_2d_profile() so the rest of the pipeline is
    unchanged.

    Parameters
    ----------
    triangles : np.ndarray, shape (N, 3, 3)
        Triangle vertices from read_ascii_stl / transform_stl.
    plane : str
        'xy'  -> z = coord  (in-plane axes: x horizontal, y vertical)
        'xz'  -> y = coord  (in-plane axes: x horizontal, z vertical)
        'yz'  -> x = coord  (in-plane axes: y horizontal, z vertical)
    coord : float
        Value of the constant coordinate (e.g. z=0.0 for plane='xy').

    Returns
    -------
    profile : dict
        Same schema as stl_to_2d_profile():
        {'x', 'y_upper', 'y_lower', 'dyu_dx', 'dyl_dx', 'x_min', 'x_max'}
    """
    _plane_axes = {
        'xy': (2, 0, 1),   # normal=z, horizontal=x, vertical=y
        'xz': (1, 0, 2),   # normal=y, horizontal=x, vertical=z
        'yz': (0, 1, 2),   # normal=x, horizontal=y, vertical=z
    }
    if plane not in _plane_axes:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz'; got '{plane}'")
    norm_ax, h_ax, v_ax = _plane_axes[plane]

    pts_h, pts_v = [], []

    for tri in triangles:
        d = tri[:, norm_ax] - coord   # signed distance from plane, shape (3,)
        for i in range(3):
            j = (i + 1) % 3
            di, dj = d[i], d[j]
            if di * dj < 0:           # edge crosses the plane
                t = di / (di - dj)
                p = tri[i] + t * (tri[j] - tri[i])
                pts_h.append(float(p[h_ax]))
                pts_v.append(float(p[v_ax]))
            elif abs(di) < 1e-14:     # vertex lies exactly on the plane
                pts_h.append(float(tri[i, h_ax]))
                pts_v.append(float(tri[i, v_ax]))

    if len(pts_h) < 2:
        raise ValueError(
            f"No cross-section found at {plane}={coord:.6f}. "
            "Verify the plane intersects the STL geometry."
        )

    pts_h = np.array(pts_h)
    pts_v = np.array(pts_v)

    # Build upper/lower envelope using the same binning as stl_to_2d_profile
    xr = pts_h.max() - pts_h.min()
    if xr <= 1e-15:
        raise ValueError("Degenerate cross-section: zero horizontal extent.")
    x_tol = xr / 1_000_000.0

    idx = np.argsort(pts_h)
    x_sorted, y_sorted = pts_h[idx], pts_v[idx]

    x_bins, y_bins = [x_sorted[0]], [[y_sorted[0]]]
    for k in range(1, len(x_sorted)):
        if abs(x_sorted[k] - x_bins[-1]) < x_tol:
            y_bins[-1].append(y_sorted[k])
        else:
            x_bins.append(x_sorted[k])
            y_bins.append([y_sorted[k]])

    x_prof  = np.array(x_bins)
    y_upper = np.array([np.max(b) for b in y_bins])
    y_lower = np.array([np.min(b) for b in y_bins])

    _, uidx = np.unique(x_prof, return_index=True)
    x_prof, y_upper, y_lower = x_prof[uidx], y_upper[uidx], y_lower[uidx]

    if len(x_prof) < 2:
        raise ValueError("Need at least 2 unique horizontal stations in cross-section.")

    dyu_dx = np.gradient(y_upper, x_prof)
    dyl_dx = np.gradient(y_lower, x_prof)

    print(f"  STL cross-section: {plane}={coord:.6f} -> "
          f"{len(x_prof)} stations, h=[{x_prof.min():.6f}, {x_prof.max():.6f}]")

    return {
        'x':       x_prof,
        'y_upper': y_upper,
        'y_lower': y_lower,
        'dyu_dx':  dyu_dx,
        'dyl_dx':  dyl_dx,
        'x_min':   float(x_prof.min()),
        'x_max':   float(x_prof.max()),
    }


def stl_surface_normal_2d(x_query, profile, side='upper'):
    """Normals from piecewise wall slope."""
    xq = np.asarray(x_query, dtype=float)
    xq = np.clip(xq, profile['x_min'], profile['x_max'])

    if side == 'upper':
        dydx = np.interp(xq, profile['x'], profile['dyu_dx'])
        nx_raw, ny_raw = -dydx, np.ones_like(dydx)
    else:
        dydx = np.interp(xq, profile['x'], profile['dyl_dx'])
        nx_raw, ny_raw = dydx, -np.ones_like(dydx)

    mag = np.sqrt(nx_raw**2 + ny_raw**2)
    mag = np.where(mag > 1e-15, mag, 1.0)
    return nx_raw / mag, ny_raw / mag


def define_geometry_surface_stl(x_cc, y_cc, fields_data, profile, max_x=None, debug=False):
    """
    Simplified STL-based surface pick:
    - exact wall y from STL envelope
    - nearest fluid-side cell to wall in each x-column
    """
    if max_x is None:
        max_x = profile['x_max']

    surfaces = {'upper': [], 'lower': []}

    for i, x_val in enumerate(x_cc):
        if x_val < profile['x_min'] or x_val > max_x:
            continue

        y_wall_u = np.interp(x_val, profile['x'], profile['y_upper'])
        y_wall_l = np.interp(x_val, profile['x'], profile['y_lower'])

        # upper: closest cell at/above wall
        cand_u = np.where(y_cc >= y_wall_u)[0]
        if len(cand_u) > 0:
            j_u = cand_u[0]
            nx, ny = stl_surface_normal_2d(x_val, profile, side='upper')
            surfaces['upper'].append({
                'x': x_val,
                'y': y_cc[j_u],
                'y_interp': y_wall_u,
                'y_exact': y_wall_u,
                'i': i, 'j': int(j_u),
                'nx': float(nx), 'ny': float(ny),
                'dist_to_wall': float(abs(y_cc[j_u] - y_wall_u)),
            })

        # lower: closest cell at/below wall
        cand_l = np.where(y_cc <= y_wall_l)[0]
        if len(cand_l) > 0:
            j_l = cand_l[-1]
            nx, ny = stl_surface_normal_2d(x_val, profile, side='lower')
            surfaces['lower'].append({
                'x': x_val,
                'y': y_cc[j_l],
                'y_interp': y_wall_l,
                'y_exact': y_wall_l,
                'i': i, 'j': int(j_l),
                'nx': float(nx), 'ny': float(ny),
                'dist_to_wall': float(abs(y_cc[j_l] - y_wall_l)),
            })

    # pack arrays
    for side in ['upper', 'lower']:
        if len(surfaces[side]) == 0:
            surfaces[side] = {
                'x': np.array([]), 'y': np.array([]),
                'y_interp': np.array([]), 'y_exact': np.array([]),
                'i': np.array([], dtype=int), 'j': np.array([], dtype=int),
                'nx': np.array([]), 'ny': np.array([]),
                'dist_to_wall': np.array([]), 'y_smooth': np.array([]),
            }
            continue

        rows = surfaces[side]
        out = {}
        for k in ['x', 'y', 'y_interp', 'y_exact', 'nx', 'ny', 'dist_to_wall']:
            out[k] = np.array([r[k] for r in rows], dtype=float)
        out['i'] = np.array([r['i'] for r in rows], dtype=int)
        out['j'] = np.array([r['j'] for r in rows], dtype=int)
        out['y_smooth'] = out['y_exact'].copy()
        surfaces[side] = out

    if debug:
        for side in ['upper', 'lower']:
            n = len(surfaces[side]['x'])
            print(f"  [STL surface] {side}: {n} points")

    return surfaces


def extract_BL_profile_at_surface_stl(x_cc, y_cc, fields_data, profile,
                                       i_surf, j_surf, nx, ny,
                                       u_inf, rho_inf,
                                       max_BL_height=0.010,
                                       n_points_BL=200,
                                       interp_order='cubic',
                                       y_wall=None,
                                       side=None):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.signal import savgol_filter

    x_surf = x_cc[i_surf]

    # Use exact wall location passed from surfaces[side]['y_exact']
    if y_wall is None:
        if side == 'lower':
            y_wall = float(np.interp(x_surf, profile['x'], profile['y_lower']))
        else:
            y_wall = float(np.interp(x_surf, profile['x'], profile['y_upper']))

    eta = np.linspace(0, 1, n_points_BL)
    stretch = 1.5
    eta_stretched = np.tanh(stretch * eta) / np.tanh(stretch)
    s = eta_stretched * max_BL_height

    x_profile = x_surf + s * nx
    y_profile_abs = y_wall + s * ny
    y_profile = s

    # ---------------------------------------------------------- #
    #  BOUNDS CHECK — clip sample points to grid extent           #
    #  Points outside the CFD domain give garbage results.        #
    # ---------------------------------------------------------- #
    x_min, x_max = x_cc[0], x_cc[-1]
    y_min, y_max = y_cc[0], y_cc[-1]

    in_bounds = ((x_profile >= x_min) & (x_profile <= x_max) &
                 (y_profile_abs >= y_min) & (y_profile_abs <= y_max))

    n_valid = np.sum(in_bounds)
    if n_valid < 10:
        raise ValueError(f"Only {n_valid} BL sample points in grid bounds "
                         f"(x=[{x_min:.4f},{x_max:.4f}], y=[{y_min:.4f},{y_max:.4f}])")

    # Trim to the longest contiguous in-bounds segment from the wall
    first_oob = np.where(~in_bounds)[0]
    if len(first_oob) > 0 and first_oob[0] > 0:
        i_trim = first_oob[0]
    else:
        i_trim = n_points_BL

    if i_trim < 10:
        raise ValueError(f"BL profile leaves grid after {i_trim} points "
                         f"(wall at x={x_surf:.4f}, y={y_wall:.4f})")

    x_profile = x_profile[:i_trim]
    y_profile_abs = y_profile_abs[:i_trim]
    y_profile = y_profile[:i_trim]
    s = s[:i_trim]

    # ---------------------------------------------------------- #
    #  INTERPOLATE FIELDS                                         #
    # ---------------------------------------------------------- #
    method = 'cubic' if interp_order == 'cubic' else 'linear'

    u_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['u'],
                                       method=method, bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['v'],
                                       method=method, bounds_error=False, fill_value=np.nan)
    T_interp = RegularGridInterpolator((x_cc, y_cc), fields_data['Temperature'],
                                       method=method, bounds_error=False, fill_value=np.nan)

    points = np.column_stack([x_profile, y_profile_abs])
    u_x = u_interp(points)
    u_y = v_interp(points)
    T_profile_vals = T_interp(points)

    # Check for NaN from out-of-bounds or extrapolation
    valid_interp = np.isfinite(u_x) & np.isfinite(u_y) & np.isfinite(T_profile_vals)
    if np.sum(valid_interp) < 10:
        raise ValueError(f"Interpolation returned {np.sum(~valid_interp)} NaN values "
                         f"out of {len(u_x)} points")

    # Replace isolated NaNs with nearest valid value
    if not np.all(valid_interp):
        idx_valid = np.where(valid_interp)[0]
        idx_all = np.arange(len(u_x))
        u_x = np.interp(idx_all, idx_valid, u_x[valid_interp])
        u_y = np.interp(idx_all, idx_valid, u_y[valid_interp])
        T_profile_vals = np.interp(idx_all, idx_valid, T_profile_vals[valid_interp])

    # ---------------------------------------------------------- #
    #  TANGENTIAL VELOCITY                                        #
    # ---------------------------------------------------------- #
    tx_candidate = ny
    ty_candidate = -nx
    if tx_candidate < 0:
        tx_candidate = -ny
        ty_candidate = nx

    u_tangential = u_x * tx_candidate + u_y * ty_candidate
    u_tangential[0] = 0.0

    # ---------------------------------------------------------- #
    #  BL EDGE DETECTION                                          #
    # ---------------------------------------------------------- #
    du_ds = np.gradient(u_tangential, y_profile)
    n_smooth = min(15, len(du_ds) // 4)
    if n_smooth >= 5 and n_smooth % 2 == 0:
        n_smooth -= 1
    du_ds_smooth = savgol_filter(du_ds, n_smooth, 3) if n_smooth >= 5 else du_ds

    i_start = max(2, len(y_profile) // 50)
    du_ds_max = np.max(np.abs(du_ds_smooth[i_start:]))

    if du_ds_max < 1e-6:
        i_edge = len(y_profile) - 1
        u_edge = u_tangential[-1] if u_tangential[-1] > 10.0 else u_inf
    else:
        i_max_grad = i_start + np.argmax(np.abs(du_ds_smooth[i_start:]))
        thr = 0.01 * du_ds_max
        cand = np.where(np.abs(du_ds_smooth[i_max_grad:]) < thr)[0]
        i_edge = i_max_grad + cand[0] if len(cand) else len(y_profile) - 1
        u_edge = u_tangential[i_edge]

    if u_edge < 10.0:
        u_edge = np.max(u_tangential)
    if u_edge < 10.0:
        u_edge = u_inf

    U_Uinf = np.clip(u_tangential / u_edge, 0.0, 1.5)

    # ---------------------------------------------------------- #
    #  BL THICKNESSES                                             #
    # ---------------------------------------------------------- #
    delta_99 = y_profile[i_edge]
    idx_99 = np.where(U_Uinf[:min(i_edge + len(y_profile) // 10, len(U_Uinf))] >= 0.99)[0]
    if len(idx_99) > 0:
        delta_99 = y_profile[idx_99[0]]

    i_cutoff = max(min(i_edge + 5, len(y_profile)), 10)
    y_bl = y_profile[:i_cutoff]
    U_bl = np.minimum(U_Uinf[:i_cutoff], 1.0)

    delta_star = max(np.trapezoid(1.0 - U_bl, y_bl), 0.0)
    theta = max(np.trapezoid(U_bl * (1.0 - U_bl), y_bl), 0.0)
    H = delta_star / theta if theta > 1e-12 else np.nan

    # ---------------------------------------------------------- #
    #  WALL SHEAR STRESS                                          #
    # ---------------------------------------------------------- #
    T_wall = T_profile_vals[0]
    mu_wall = sutherland_viscosity(T_wall)
    n_fit = min(6, len(y_profile))
    coeffs = np.polyfit(y_profile[:n_fit], u_tangential[:n_fit], 1)
    tau_w = mu_wall * coeffs[0]

    q_inf = 0.5 * rho_inf * u_inf**2
    C_f = tau_w / q_inf

    mu_far = sutherland_viscosity(T_profile_vals[-1])
    Re_x = rho_inf * u_inf * x_surf / mu_far
    Re_theta = rho_inf * u_inf * theta / mu_far if theta > 0 else 0

    return {
        'y_profile': y_profile, 'U_Uinf': U_Uinf, 'u_profile': u_tangential,
        'u_edge': u_edge, 'T_profile': T_profile_vals, 'T_wall': T_wall,
        'tau_w': tau_w, 'C_f': C_f, 'delta_99': delta_99,
        'delta_star': delta_star, 'theta': theta, 'H': H,
        'Re_x': Re_x, 'Re_theta': Re_theta, 'wall_y': y_wall,
    }


# ...existing code...

def extract_surface_properties_stl(x_cc, y_cc, xx, yy, fields_data, surfaces,
                                    profile, rho_inf=0.0267, u_inf=1011,
                                    T_inf=None, interp_order='cubic',
                                    diag_stations=None, snapshot=None):
    """
    Extract surface properties using STL-based geometry.

    Parameters:
    -----------
    diag_stations : list of float or None
        x-locations (m) at which to produce detailed BL diagnostic output
        and interpolation-stencil plots.  e.g. [0.05, 0.15, 0.30, 0.50]
    snapshot : int or str or None
        Snapshot label used in diagnostic filenames.
    """
    from scipy.interpolate import RegularGridInterpolator
    
    surface_data = {}
    
    if diag_stations is None:
        diag_stations = []
    
    # ---------------------------------------------------------- #
    #  BUILD INTERPOLATORS ONCE (expensive for large grids)       #
    # ---------------------------------------------------------- #
    method = 'cubic' if interp_order == 'cubic' else 'linear'
    print(f"    Building {method} interpolators for {fields_data['u'].shape} grid...")

    shared_interps = {
        'u':   RegularGridInterpolator((x_cc, y_cc), fields_data['u'],
                                        method=method, bounds_error=False, fill_value=np.nan),
        'v':   RegularGridInterpolator((x_cc, y_cc), fields_data['v'],
                                        method=method, bounds_error=False, fill_value=np.nan),
        'T':   RegularGridInterpolator((x_cc, y_cc), fields_data['Temperature'],
                                        method=method, bounds_error=False, fill_value=np.nan),
        # Linear for p and rho: cubic stencil spanning the ghost/fluid
        # discontinuity produces Runge-type oscillations in Cp.
        # Linear is monotone across the IB interface.
        'p':   RegularGridInterpolator((x_cc, y_cc), fields_data['p'],
                                        method='linear', bounds_error=False, fill_value=np.nan),
        'rho': RegularGridInterpolator((x_cc, y_cc), fields_data['rho'],
                                        method='linear', bounds_error=False, fill_value=np.nan),
    }
    print(f"    ✓ Interpolators ready (u/v/T: {method}, p/rho: linear)")
    
    for side in ['upper', 'lower']:
        if len(surfaces[side]['x']) == 0:
            continue
        
        n_points = len(surfaces[side]['x'])
        print(f"    Processing {side} surface (STL): {n_points} points")
        
        data = {
            'x': surfaces[side]['x'],
            'y': surfaces[side]['y_exact'],
            's': np.zeros(n_points),
            'p': np.zeros(n_points),
            'Temperature': np.zeros(n_points),
            'rho': np.zeros(n_points),
            'tau_w': np.zeros(n_points),
            'C_f': np.zeros(n_points),
            'C_p': np.zeros(n_points),
            'q_w': np.zeros(n_points),
            'Re_x': np.zeros(n_points),
            'Re_theta': np.zeros(n_points),
            'delta_99': np.zeros(n_points),
            'delta_star': np.zeros(n_points),
            'theta': np.zeros(n_points),
            'H': np.zeros(n_points),
        }
        
        # Arc length along exact STL surface
        dx_arr = np.diff(surfaces[side]['x'])
        dy_arr = np.diff(surfaces[side]['y_exact'])
        ds_arr = np.sqrt(dx_arr**2 + dy_arr**2)
        data['s'][1:] = np.cumsum(ds_arr)
        
        n_success = 0
        n_failed = 0
        
        for idx in range(n_points):
            i = int(surfaces[side]['i'][idx])
            j = int(surfaces[side]['j'][idx])
            
            # Evaluate wall thermodynamic properties at the exact STL surface
            # using linear interpolation (avoids Runge oscillations at the IB).
            x_here  = float(surfaces[side]['x'][idx])
            y_here  = float(surfaces[side]['y_exact'][idx])
            wall_pt = np.array([[x_here, y_here]])

            p_wall   = float(shared_interps['p'](wall_pt)[0])
            T_w_prop = float(shared_interps['T'](wall_pt)[0])
            rho_wall = float(shared_interps['rho'](wall_pt)[0])

            if not np.isfinite(p_wall):
                p_wall = fields_data['p'][i, j]
            if not np.isfinite(T_w_prop):
                T_w_prop = fields_data['Temperature'][i, j]
            if not np.isfinite(rho_wall):
                rho_wall = fields_data['rho'][i, j]

            data['p'][idx]           = p_wall
            data['Temperature'][idx] = T_w_prop
            data['rho'][idx]         = rho_wall
            
            # Normal from STL
            nx = surfaces[side]['nx'][idx]
            ny = surfaces[side]['ny'][idx]
            
            # Check if this point is near a diagnostic station
            x_here = float(surfaces[side]['x'][idx])
            is_diag = any(abs(x_here - xd) < (x_cc[1] - x_cc[0]) * 0.6
                          for xd in diag_stations)
            
            try:
                bl_results = _extract_BL_single_point(
                    x_cc, y_cc, shared_interps,
                    i, nx, ny, u_inf, rho_inf,
                    y_wall=float(surfaces[side]['y_exact'][idx]),
                    side=side,
                    debug=is_diag,
                )
                
                data['tau_w'][idx] = bl_results['tau_w']
                data['C_f'][idx] = bl_results['C_f']
                data['delta_99'][idx] = bl_results['delta_99']
                data['delta_star'][idx] = bl_results['delta_star']
                data['theta'][idx] = bl_results['theta']
                data['H'][idx] = bl_results['H']
                data['Re_x'][idx] = bl_results['Re_x']
                data['Re_theta'][idx] = bl_results['Re_theta']
                n_success += 1
                
                # Generate diagnostic plot if requested
                if is_diag and '_diag' in bl_results:
                    # Phase 4: Direct cell-based Cf for comparison
                    try:
                        direct_cf = _extract_Cf_direct(
                            x_cc, y_cc, fields_data, i,
                            float(surfaces[side]['y_exact'][idx]),
                            nx, ny, u_inf, rho_inf,
                        )
                        if direct_cf is not None:
                            bl_results['_diag']['direct_cf'] = direct_cf
                            print(f"      Direct-cell Cf = {direct_cf['C_f']:.6e}  "
                                  f"(vs interp Cf = {bl_results['C_f']:.6e})")
                    except Exception as dc_err:
                        print(f"      [direct Cf failed: {dc_err}]")

                    surf_row = {k: surfaces[side][k][idx]
                                for k in ['x', 'y_exact', 'nx', 'ny', 'i', 'j']}
                    try:
                        plot_BL_diagnostic(
                            x_cc, y_cc, fields_data, bl_results, surf_row,
                            side, snapshot if snapshot is not None else 'diag',
                        )
                    except Exception as plot_err:
                        print(f"      [diag plot failed: {plot_err}]")
                
            except Exception as e:
                if n_failed < 5:
                    print(f"      Warning at point {idx} "
                          f"(x={surfaces[side]['x'][idx]:.4f}): {e}")
                data['tau_w'][idx] = np.nan
                data['C_f'][idx] = np.nan
                data['delta_99'][idx] = np.nan
                data['delta_star'][idx] = np.nan
                data['theta'][idx] = np.nan
                data['H'][idx] = np.nan
                data['Re_x'][idx] = np.nan
                data['Re_theta'][idx] = np.nan
                n_failed += 1
            
            # Progress indicator every 500 points
            if (idx + 1) % 500 == 0:
                print(f"      ... {idx+1}/{n_points} ({n_success} ok, {n_failed} fail)")
        
        print(f"      BL extraction: {n_success} successful, {n_failed} failed")
        
        # --- POST-EXTRACTION SMOOTHING ---
        from scipy.signal import savgol_filter
        from scipy.ndimage import median_filter
        
        arc_length = data['s'][-1] if data['s'][-1] > 0 else 1.0
        pts_per_mm = n_points / (arc_length * 1000.0)
        med_win = max(3, int(pts_per_mm * 5))
        sg_win = max(med_win * 3 + 1, 21)
        if med_win % 2 == 0: med_win += 1
        if sg_win % 2 == 0: sg_win += 1
        med_win = min(med_win, n_points if n_points % 2 == 1 else n_points - 1)
        sg_win = min(sg_win, n_points if n_points % 2 == 1 else n_points - 1)
        
        def smooth_field(arr):
            a = arr.copy().astype(float)
            valid = np.isfinite(a)
            if np.sum(valid) < sg_win:
                return a
            x_idx = np.arange(len(a))
            a[~valid] = np.interp(x_idx[~valid], x_idx[valid], a[valid])
            a = median_filter(a, size=med_win)
            a = savgol_filter(a, sg_win, polyorder=3)
            a[~valid] = np.nan
            return a
        
        for key in ['C_f', 'tau_w', 'delta_99', 'delta_star', 'theta', 'H']:
            data[key] = smooth_field(data[key])
        
        print(f"      Smoothing: median={med_win} pts, SG={sg_win} pts "
              f"({sg_win / pts_per_mm:.1f} mm)")
        
        # Pressure coefficient
        q_inf = 0.5 * rho_inf * u_inf**2
        R = 287.05
        p_inf = rho_inf * R * (T_inf if T_inf else np.mean(data['Temperature']))
        data['C_p'] = (data['p'] - p_inf) / q_inf
        
        surface_data[side] = data
    
    return surface_data


def _extract_BL_single_point(x_cc, y_cc, interps,
                              i_surf, nx, ny, u_inf, rho_inf,
                              y_wall, side,
                              max_BL_height=0.015, n_points_BL=500,
                              debug=False):
    """
    Single-point BL extraction using PRE-BUILT interpolators.

    Ghost-cell IBM and the correct anchor for du/dn
    ------------------------------------------------
    MFC enforces no-slip by setting:
        u_ghost = −u_fluid × (d_ghost / d_fluid)
    where d_ghost = |y_cc[j_ghost] − y_wall| and d_fluid = |y_cc[j_fluid] − y_wall|.
    This drives u → 0 exactly AT y_wall, regardless of where the surface sits
    within the cell gap.  The surface can be anywhere from barely past the ghost
    cell centre (d_fluid ≈ dy) to nearly at the cell face (d_fluid ≈ dy/2).

    The original code sampled the cubic interpolant starting at y_wall and
    stepping into the fluid.  Inside the ghost/fluid overlap zone the cubic
    stencil spans cells with opposite-sign velocities (ghost = −fluid), producing
    a negative dip in u_tang before it recovers.  An unconstrained polynomial fit
    over that dip extracts a near-zero or negative slope.

    Correct approach
    ----------------
    1. Identify j_fluid: the first cell centre that is genuinely on the fluid
       side of y_wall (ib_markers = 0, physically outside the body).
    2. Query the shared interpolators at those cell-centre coordinates.
       Querying at an exact grid point returns the exact grid value — no
       interpolation across the ghost/fluid boundary happens at all.
    3. Compute s_fit = |y_cc[j] − y_wall| for each fluid cell.
       This is the true wall-normal distance from the no-slip surface.
    4. Constrained least-squares fit:  u_tang = a·s + b·s²  (no intercept).
       Forces u = 0 at s = 0 (y_wall).  du/dn|_wall = a.

    The BL profile for thickness integrals starts at j_fluid and prepends a
    u = 0 point at s = 0 (y_wall) so the integrals are correctly anchored.
    """
    from scipy.signal import savgol_filter

    x_surf  = x_cc[i_surf]
    dy_grid = y_cc[1] - y_cc[0]
    dx_grid = x_cc[1] - x_cc[0]

    # ---------------------------------------------------------- #
    #  STEP 1 — Identify fluid / ghost cell indices              #
    #  ny >= 0 → upper surface: fluid cells are ABOVE y_wall     #
    #  ny <  0 → lower surface: fluid cells are BELOW y_wall     #
    # ---------------------------------------------------------- #
    if ny >= 0:
        j_fluid = int(np.searchsorted(y_cc, y_wall))       # first centre ABOVE wall
        j_fluid = min(j_fluid, len(y_cc) - 1)
        j_ghost = max(j_fluid - 1, 0)
    else:
        j_fluid = int(np.searchsorted(y_cc, y_wall)) - 1   # last centre BELOW wall
        j_fluid = max(j_fluid, 0)
        j_ghost = min(j_fluid + 1, len(y_cc) - 1)

    d_fluid_to_wall = abs(y_cc[j_fluid] - y_wall)   # can be anywhere in (0, dy)
    d_ghost_to_wall = abs(y_cc[j_ghost] - y_wall)   # the complement

    # ---------------------------------------------------------- #
    #  STEP 2 — Tangential direction                             #
    # ---------------------------------------------------------- #
    tx = ny
    ty = -nx
    if tx < 0:
        tx = -ny
        ty =  nx

    # ---------------------------------------------------------- #
    #  STEP 3 — Wall temperature                                 #
    #  MFC ghost cell for temperature:                           #
    #    adiabatic  → T_ghost = T_fluid   → T_wall = T_fluid     #
    #    isothermal → T_ghost = 2·T_w − T_fluid → avg = T_wall   #
    #  Weighted average by distance is correct for both:         #
    #    T_wall ≈ (d_ghost·T_fluid + d_fluid·T_ghost)/(d_ghost+d_fluid)
    # ---------------------------------------------------------- #
    T_ghost_val = float(interps['T'](np.array([[x_surf, y_cc[j_ghost]]]))[0])
    T_fluid_val = float(interps['T'](np.array([[x_surf, y_cc[j_fluid]]]))[0])
    d_sum = d_ghost_to_wall + d_fluid_to_wall
    if np.isfinite(T_ghost_val) and np.isfinite(T_fluid_val) and d_sum > 1e-15:
        T_wall_val = (d_ghost_to_wall * T_fluid_val +
                      d_fluid_to_wall * T_ghost_val) / d_sum
    elif np.isfinite(T_fluid_val):
        T_wall_val = T_fluid_val
    else:
        T_wall_val = 300.0
    mu_wall = sutherland_viscosity(T_wall_val)

    # ---------------------------------------------------------- #
    #  STEP 4 — du/dn: constrained fit on fluid cell centres     #
    #                                                            #
    #  s measured from y_wall (the IBM no-slip surface).         #
    #  Only fluid cells used — no cubic stencil spans the        #
    #  ghost/fluid interface, so no negative-dip artifact.       #
    #                                                            #
    #  Model: u_tang = a·s + b·s²   (u = 0 at s = 0, y_wall)   #
    #  → du/dn|_wall = a                                         #
    # ---------------------------------------------------------- #
    N_FIT_CELLS = 6
    if ny >= 0:
        j_range = np.arange(j_fluid, min(j_fluid + N_FIT_CELLS, len(y_cc)))
    else:
        j_range = np.arange(j_fluid, max(j_fluid - N_FIT_CELLS, -1), -1)

    # True distances from the STL surface — NOT from any cell face
    s_fit = np.abs(y_cc[j_range] - y_wall)

    fit_pts_xy = np.column_stack([np.full(len(j_range), x_surf), y_cc[j_range]])
    u_x_fit = interps['u'](fit_pts_xy)
    u_y_fit = interps['v'](fit_pts_xy)
    u_tang_fit = u_x_fit * tx + u_y_fit * ty

    valid_fit = np.isfinite(u_x_fit) & np.isfinite(u_y_fit)
    if np.sum(valid_fit) < 2:
        raise ValueError(f"Only {np.sum(valid_fit)} valid fit points at x={x_surf:.4f}")

    s_fit_valid       = s_fit[valid_fit]
    u_tang_fit_valid  = u_tang_fit[valid_fit]

    # Constrained least-squares (no intercept → u = 0 at y_wall)
    A_fit = np.column_stack([s_fit_valid, s_fit_valid**2])
    coeffs_c, _, _, _ = np.linalg.lstsq(A_fit, u_tang_fit_valid, rcond=None)
    du_dn_wall = float(coeffs_c[0])
    tau_w      = mu_wall * du_dn_wall
    n_fit      = int(np.sum(valid_fit))
    fit_lo     = float(s_fit_valid[0])
    fit_hi     = float(s_fit_valid[-1])

    q_inf = 0.5 * rho_inf * u_inf**2
    C_f   = tau_w / q_inf

    # ---------------------------------------------------------- #
    #  STEP 5 — BL profile for thicknesses                       #
    #                                                            #
    #  Outer stretched profile starting from j_fluid.           #
    #  Wall point (u = 0, s = 0) prepended for correct integrals.
    # ---------------------------------------------------------- #
    s_start = d_fluid_to_wall   # distance of first fluid cell from y_wall

    eta           = np.linspace(0, 1, n_points_BL)
    stretch       = 1.5
    eta_stretched = np.tanh(stretch * eta) / np.tanh(stretch)
    s_outer       = s_start + eta_stretched * max_BL_height

    x_profile     = x_surf  + s_outer * nx
    y_profile_abs = y_wall   + s_outer * ny
    y_profile_s   = s_outer

    x_min, x_max = x_cc[0], x_cc[-1]
    y_min, y_max = y_cc[0], y_cc[-1]
    in_bounds = ((x_profile >= x_min) & (x_profile <= x_max) &
                 (y_profile_abs >= y_min) & (y_profile_abs <= y_max))
    first_oob = np.where(~in_bounds)[0]
    i_trim = first_oob[0] if (len(first_oob) > 0 and first_oob[0] > 0) else n_points_BL
    if i_trim < 10:
        raise ValueError(f"BL profile leaves grid after {i_trim} pts at "
                         f"x={x_surf:.4f}, y_wall={y_wall:.4f}")

    x_profile     = x_profile[:i_trim]
    y_profile_abs = y_profile_abs[:i_trim]
    y_profile_s   = y_profile_s[:i_trim]

    profile_pts = np.column_stack([x_profile, y_profile_abs])
    u_x = interps['u'](profile_pts)
    u_y = interps['v'](profile_pts)
    T_profile_vals = interps['T'](profile_pts)

    valid_interp = np.isfinite(u_x) & np.isfinite(u_y) & np.isfinite(T_profile_vals)
    if np.sum(valid_interp) < 10:
        raise ValueError(f"Only {np.sum(valid_interp)} valid profile points at "
                         f"x={x_surf:.4f}")
    if not np.all(valid_interp):
        idx_v   = np.where(valid_interp)[0]
        idx_all = np.arange(len(u_x))
        u_x            = np.interp(idx_all, idx_v, u_x[valid_interp])
        u_y            = np.interp(idx_all, idx_v, u_y[valid_interp])
        T_profile_vals = np.interp(idx_all, idx_v, T_profile_vals[valid_interp])

    u_tangential = u_x * tx + u_y * ty

    # Prepend wall point: s=0, u=0
    y_profile      = np.concatenate([[0.0],       y_profile_s])
    u_tangential   = np.concatenate([[0.0],       u_tangential])
    T_profile_vals = np.concatenate([[T_wall_val], T_profile_vals])
    x_profile_full     = np.concatenate([[x_surf], x_profile])
    y_profile_abs_full = np.concatenate([[y_wall],  y_profile_abs])

    # ---------------------------------------------------------- #
    #  STEP 6 — BL edge detection                                #
    # ---------------------------------------------------------- #
    du_ds = np.gradient(u_tangential, y_profile)
    n_smooth = min(15, len(du_ds) // 4)
    if n_smooth >= 5 and n_smooth % 2 == 0:
        n_smooth -= 1
    du_ds_smooth = savgol_filter(du_ds, n_smooth, 3) if n_smooth >= 5 else du_ds

    i_start   = max(2, len(y_profile) // 50)
    du_ds_max = np.max(np.abs(du_ds_smooth[i_start:]))

    if du_ds_max < 1e-6:
        i_edge = len(y_profile) - 1
        u_edge = u_tangential[-1] if u_tangential[-1] > 10.0 else u_inf
    else:
        i_max_grad = i_start + np.argmax(np.abs(du_ds_smooth[i_start:]))
        thr  = 0.01 * du_ds_max
        cand = np.where(np.abs(du_ds_smooth[i_max_grad:]) < thr)[0]
        i_edge = i_max_grad + cand[0] if len(cand) else len(y_profile) - 1
        u_edge = u_tangential[i_edge]

    if u_edge < 10.0:
        u_edge = np.max(u_tangential)
    if u_edge < 10.0:
        u_edge = u_inf

    U_Uinf = np.clip(u_tangential / u_edge, 0.0, 1.5)

    # ---------------------------------------------------------- #
    #  STEP 7 — BL thicknesses                                   #
    # ---------------------------------------------------------- #
    delta_99 = y_profile[i_edge]
    idx_99   = np.where(U_Uinf[:min(i_edge + len(y_profile) // 10,
                                     len(U_Uinf))] >= 0.99)[0]
    if len(idx_99) > 0:
        delta_99 = y_profile[idx_99[0]]

    i_cutoff   = max(min(i_edge + 5, len(y_profile)), 10)
    y_bl       = y_profile[:i_cutoff]
    U_bl       = np.minimum(U_Uinf[:i_cutoff], 1.0)
    delta_star = max(np.trapezoid(1.0 - U_bl, y_bl), 0.0)
    theta      = max(np.trapezoid(U_bl * (1.0 - U_bl), y_bl), 0.0)
    H          = delta_star / theta if theta > 1e-12 else np.nan

    mu_far   = sutherland_viscosity(T_profile_vals[-1])
    Re_x     = rho_inf * u_inf * x_surf / mu_far
    Re_theta = rho_inf * u_inf * theta  / mu_far if theta > 0 else 0.0

    # ---------------------------------------------------------- #
    #  DIAGNOSTIC                                                 #
    # ---------------------------------------------------------- #
    if debug:
        print(f"\n      ── BL DIAGNOSTIC at x={x_surf:.5f} m, side={side} ──")
        print(f"      STL wall:      y_wall     = {y_wall:.6f} m")
        print(f"      Ghost centre:  y_cc[{j_ghost}] = {y_cc[j_ghost]:.6f} m  "
              f"(d_ghost = {d_ghost_to_wall*1e3:.3f} mm)")
        print(f"      Fluid centre:  y_cc[{j_fluid}] = {y_cc[j_fluid]:.6f} m  "
              f"(d_fluid = {d_fluid_to_wall*1e3:.3f} mm,  "
              f"{d_fluid_to_wall/dy_grid*100:.0f}% of dy)")
        print(f"      Normal:  (nx, ny) = ({nx:.4f}, {ny:.4f})")
        print(f"      Tangent: (tx, ty) = ({tx:.4f}, {ty:.4f})")
        print(f"      T_ghost = {T_ghost_val:.2f} K,  T_fluid = {T_fluid_val:.2f} K  "
              f"→  T_wall = {T_wall_val:.2f} K")
        print(f"      mu_wall = {mu_wall:.4e} Pa·s")
        print(f"      Fit: {n_fit} fluid cells, "
              f"s = [{fit_lo*1e3:.3f}, {fit_hi*1e3:.3f}] mm from y_wall")
        print(f"      du/dn = {du_dn_wall:.2f} (m/s)/m  "
              f"[a={coeffs_c[0]:.4f}, b={coeffs_c[1]:.4f}]")
        print(f"      tau_w = {tau_w:.4f} Pa,  q_inf = {q_inf:.2f} Pa,  "
              f"C_f = {C_f:.6e}")
        print(f"      u_edge = {u_edge:.2f} m/s,  delta_99 = {delta_99*1e3:.4f} mm")
        print(f"      Grid: dx = {dx_grid*1e3:.4f} mm,  dy = {dy_grid*1e3:.4f} mm")
        print(f"      Fit-point detail (s measured from y_wall):")
        print(f"        {'j':>5}  {'s/dy':>6}  {'s (mm)':>9}  {'u_tang':>12}")
        for k in range(len(j_range)):
            if valid_fit[k]:
                print(f"        {j_range[k]:5d}  {s_fit[k]/dy_grid:6.2f}  "
                      f"{s_fit[k]*1e3:9.4f}  {u_tang_fit[k]:12.4f}")
        print(f"      ─────────────────────────────────────────────")

    results = {
        'y_profile': y_profile, 'U_Uinf': U_Uinf, 'u_profile': u_tangential,
        'u_edge': u_edge, 'T_profile': T_profile_vals, 'T_wall': T_wall_val,
        'tau_w': tau_w, 'C_f': C_f, 'delta_99': delta_99,
        'delta_star': delta_star, 'theta': theta, 'H': H,
        'Re_x': Re_x, 'Re_theta': Re_theta, 'wall_y': y_wall,
    }

    if debug:
        results['_diag'] = {
            'x_profile':     x_profile_full,
            'y_profile_abs': y_profile_abs_full,
            'u_x':           np.concatenate([[0.0], u_x]),
            'u_y':           np.concatenate([[0.0], u_y]),
            'du_dn_wall':    du_dn_wall,
            'mu_wall':       mu_wall,
            'n_fit':         n_fit,
            'fit_coeffs':    np.array([coeffs_c[1], coeffs_c[0], 0.0]),
            'nx': nx, 'ny': ny,
            'wall_offset':   d_fluid_to_wall,
            'dy_grid':       dy_grid,
            'fit_lo':        fit_lo,
            'fit_hi':        fit_hi,
            's_fit_valid':   s_fit_valid,
            'u_fit_tang':    u_tang_fit_valid,
            'j_fluid':       j_fluid,
            'j_ghost':       j_ghost,
            'y_face':        0.5 * (y_cc[j_ghost] + y_cc[j_fluid]),
        }

    return results


def _extract_Cf_direct(x_cc, y_cc, fields_data, i_surf, y_wall, nx, ny,
                       u_inf, rho_inf):
    """
    Compute Cf directly from the first fluid cell above the IBM wall,
    without interpolation. Uses single-cell du/dn ~ u_tang / dn.

    Returns dict with Cf and supporting data, or None on failure.
    """
    # First fluid cell: nearest cell center in the ny direction from wall
    j_wall = np.searchsorted(y_cc, y_wall)
    if ny >= 0:
        j1 = min(j_wall, len(y_cc) - 1)
        j2 = min(j1 + 1, len(y_cc) - 1)
    else:
        j1 = max(j_wall - 1, 0)
        j2 = max(j1 - 1, 0)

    if j1 == j2:
        return None

    # Tangential direction (same convention as BL extraction)
    tx = ny
    ty = -nx
    if tx < 0:
        tx = -ny
        ty = nx

    u_tang1 = fields_data['u'][i_surf, j1] * tx + fields_data['v'][i_surf, j1] * ty
    u_tang2 = fields_data['u'][i_surf, j2] * tx + fields_data['v'][i_surf, j2] * ty
    T1 = fields_data['Temperature'][i_surf, j1]

    # Wall-normal distance from wall to first cell center
    dn1 = abs(y_cc[j1] - y_wall)
    dn2 = abs(y_cc[j2] - y_wall)

    if dn1 < 1e-12:
        return None

    # du/dn at wall ~ u_tang1 / dn1  (assuming u=0 at wall)
    du_dn_wall = u_tang1 / dn1

    mu_wall = sutherland_viscosity(T1)
    tau_w = mu_wall * du_dn_wall
    q_inf = 0.5 * rho_inf * u_inf**2
    Cf = tau_w / q_inf

    return {
        'C_f': Cf, 'tau_w': tau_w, 'du_dn_wall': du_dn_wall,
        'mu_wall': mu_wall, 'T_cell': T1,
        'j1': j1, 'j2': j2, 'dn1': dn1, 'dn2': dn2,
        'u_tang1': u_tang1, 'u_tang2': u_tang2,
    }


def plot_BL_diagnostic(x_cc, y_cc, fields_data, bl_results, surfaces_side_row,
                       side, snapshot, output_dir='BL_Diagnostics',
                       n_profile_show=40, stencil_every=5):
    """
    Diagnostic plot for a single BL extraction point.

    Left panel:  Physical space — CFD cell centers coloured by ib_markers,
                 STL wall line, BL sample points (every `stencil_every`-th
                 highlighted with the 4×4 cubic interpolation stencil cells).
    Right panel: Near-wall tangential velocity profile with the linear fit
                 used for du/dn.

    Parameters:
    -----------
    bl_results : dict
        Return value of _extract_BL_single_point(debug=True). Must contain
        the '_diag' key.
    surfaces_side_row : dict
        Single-point row from surfaces[side] with keys x, y_exact, nx, ny, i, j.
    side : str
        'upper' or 'lower'
    n_profile_show : int
        Number of profile points to show (from the wall outward).
    stencil_every : int
        Show interpolation stencil cells for every N-th profile point.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    diag = bl_results['_diag']
    x_profile = diag['x_profile']
    y_profile_abs = diag['y_profile_abs']
    u_tang = bl_results['u_profile']
    y_profile = bl_results['y_profile']
    n_fit = diag['n_fit']
    du_dn = diag['du_dn_wall']
    nx_vec, ny_vec = diag['nx'], diag['ny']

    x_wall = float(surfaces_side_row['x'])
    y_wall = float(surfaces_side_row['y_exact'])
    i_cell = int(surfaces_side_row['i'])

    dx = x_cc[1] - x_cc[0]
    dy = y_cc[1] - y_cc[0]

    # ---- Determine local view window (±8 cells around wall) ---- #
    n_cells_view = 8
    i_lo = max(0, i_cell - n_cells_view)
    i_hi = min(len(x_cc), i_cell + n_cells_view + 1)

    j_wall = int(np.argmin(np.abs(y_cc - y_wall)))
    j_lo = max(0, j_wall - n_cells_view)
    j_hi = min(len(y_cc), j_wall + n_cells_view + 1)

    x_local = x_cc[i_lo:i_hi]
    y_local = y_cc[j_lo:j_hi]
    ib_local = fields_data['ib_markers'][i_lo:i_hi, j_lo:j_hi]
    u_local = fields_data['u'][i_lo:i_hi, j_lo:j_hi]

    xx_local, yy_local = np.meshgrid(x_local, y_local, indexing='ij')

    # ================================================================ #
    fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                              gridspec_kw={'width_ratios': [1.3, 1]})

    # ---- LEFT PANEL: Physical space ---- #
    ax = axes[0]

    # Cell-center grid coloured by ib_markers
    sc = ax.scatter(xx_local.ravel(), yy_local.ravel(),
                    c=ib_local.ravel(), cmap='RdYlGn_r', vmin=0, vmax=5,
                    s=50, edgecolors='k', linewidths=0.3, zorder=3,
                    label='Cell centers (ib_markers)')
    plt.colorbar(sc, ax=ax, label='ib_markers', shrink=0.7, pad=0.02)

    # Cell boundaries
    for xv in x_cc[i_lo:i_hi+1]:
        ax.axvline(xv - dx/2, color='gray', lw=0.3, alpha=0.5)
    for yv in y_cc[j_lo:j_hi+1]:
        ax.axhline(yv - dy/2, color='gray', lw=0.3, alpha=0.5)

    # STL wall (horizontal line at y_wall near this x)
    ax.axhline(y_wall, color='lime', lw=2.5, ls='--', label=f'STL wall y={y_wall:.5f}', zorder=4)

    # Effective IBM wall (if offset detected)
    wall_offset = diag.get('wall_offset', 0.0)
    if wall_offset > 1e-6:
        y_eff_wall = y_wall + wall_offset * ny_vec
        ax.axhline(y_eff_wall, color='orange', lw=2.0, ls='-.',
                   label=f'Eff. IBM wall (+{wall_offset*1e3:.3f} mm)', zorder=4)

    # BL profile sample points
    n_show = min(n_profile_show, len(x_profile))
    ax.plot(x_profile[:n_show], y_profile_abs[:n_show],
            'b-', lw=1, alpha=0.5, zorder=5)

    # Mark fit sample points (Phase 1: separate from profile points)
    s_fit_valid = diag.get('s_fit_valid', None)
    if s_fit_valid is not None:
        x_fit_pts = x_wall + (s_fit_valid + wall_offset) * nx_vec
        y_fit_pts = y_wall + (s_fit_valid + wall_offset) * ny_vec
        ax.scatter(x_fit_pts, y_fit_pts,
                   c='red', s=80, marker='D', zorder=7,
                   label=f'Fit points (n={n_fit})')
    else:
        ax.scatter(x_profile[:n_fit], y_profile_abs[:n_fit],
                   c='red', s=80, marker='D', zorder=7, label=f'Fit points (n={n_fit})')

    # Every stencil_every-th point: show the point AND highlight stencil cells
    for k in range(0, n_show, stencil_every):
        xp, yp = x_profile[k], y_profile_abs[k]

        # Cyan circle for shown sample points
        ax.scatter(xp, yp, c='cyan', s=40, marker='o', zorder=6, edgecolors='navy', linewidths=0.5)

        # Find the 4×4 stencil for cubic interpolation
        # (neighbours: 2 cells on each side of the query point)
        ix = np.searchsorted(x_cc, xp) - 1
        iy = np.searchsorted(y_cc, yp) - 1
        ix_lo = max(0, ix - 1)
        ix_hi = min(len(x_cc) - 1, ix + 2)
        iy_lo = max(0, iy - 1)
        iy_hi = min(len(y_cc) - 1, iy + 2)

        for si in range(ix_lo, ix_hi + 1):
            for sj in range(iy_lo, iy_hi + 1):
                rect = Rectangle(
                    (x_cc[si] - dx/2, y_cc[sj] - dy/2), dx, dy,
                    linewidth=1.2, edgecolor='cyan', facecolor='cyan',
                    alpha=0.15, zorder=2
                )
                ax.add_patch(rect)

    # Normal direction arrow
    arrow_len = 3 * dy
    ax.annotate('', xy=(x_wall + arrow_len*nx_vec, y_wall + arrow_len*ny_vec),
                xytext=(x_wall, y_wall),
                arrowprops=dict(arrowstyle='->', color='magenta', lw=2),
                zorder=8)

    ax.set_xlim(x_cc[i_lo] - dx, x_cc[min(i_hi, len(x_cc)-1)] + dx)
    ax.set_ylim(y_cc[j_lo] - dy, y_cc[min(j_hi, len(y_cc)-1)] + dy)
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(f'{side} — x = {x_wall:.4f} m — Interpolation Stencil', fontsize=13)

    # ---- RIGHT PANEL: Near-wall velocity profile ---- #
    ax2 = axes[1]

    # Full profile (up to delta_99 + margin)
    i_show = min(n_profile_show, len(y_profile))
    ax2.plot(u_tang[:i_show], y_profile[:i_show] * 1e3,
             'b-o', markersize=3, lw=1.5, label='u_tangential')

    # Fit region shading (Phase 5)
    fit_lo = diag.get('fit_lo', None)
    fit_hi = diag.get('fit_hi', None)
    dy_grid = diag.get('dy_grid', dy)
    if fit_lo is not None and fit_hi is not None:
        ax2.axhspan(fit_lo * 1e3, fit_hi * 1e3, alpha=0.12, color='red',
                     label=f'Fit region ({fit_lo/dy_grid:.1f}–{fit_hi/dy_grid:.1f} dy)')

    # Fit sample points (cubic interp, from STL wall along normal)
    s_fit_valid = diag.get('s_fit_valid', None)
    u_fit_tang = diag.get('u_fit_tang', None)
    if s_fit_valid is not None and u_fit_tang is not None:
        ax2.scatter(u_fit_tang, s_fit_valid * 1e3,
                    c='red', s=80, marker='D', zorder=5,
                    label=f'Fit pts (cubic, n={n_fit})')
    else:
        ax2.scatter(u_tang[:n_fit], y_profile[:n_fit] * 1e3,
                    c='red', s=80, marker='D', zorder=5, label=f'Fit points (n={n_fit})')

    # Quadratic fit curve
    s_line_max = (fit_hi * 1.3) if fit_hi else (y_profile[n_fit-1] * 1.5)
    s_fit_line = np.linspace(0, s_line_max, 100)
    fit_coeffs = diag.get('fit_coeffs', np.array([0.0, du_dn, 0.0]))
    u_fit_line = np.polyval(fit_coeffs, s_fit_line)
    ax2.plot(u_fit_line, s_fit_line * 1e3, 'r--', lw=2,
             label=f'Quad fit (cubic interp): du/dn={du_dn:.1f} /s')

    # Effective IBM wall line
    wall_offset = diag.get('wall_offset', 0.0)
    if wall_offset > 1e-6:
        ax2.axhline(0.0, color='orange', ls='-.', lw=1.5,
                     label=f'Eff. wall (offset={wall_offset*1e3:.3f} mm)')

    # Direct-cell Cf comparison (Phase 4)
    direct_cf = diag.get('direct_cf', None)
    if direct_cf is not None:
        # Show the direct cell point on the profile
        dn1 = direct_cf['dn1']
        u_t1 = direct_cf['u_tang1']
        ax2.scatter([u_t1], [dn1 * 1e3], c='magenta', s=120, marker='*',
                    zorder=6, label=f"Direct cell (Cf={direct_cf['C_f']:.4e})")
        # Line from origin through direct cell point
        s_dc = np.linspace(0, dn1 * 1.5, 30)
        u_dc = direct_cf['du_dn_wall'] * s_dc
        ax2.plot(u_dc, s_dc * 1e3, 'm:', lw=1.5, alpha=0.7)

    # delta_99 line
    ax2.axhline(bl_results['delta_99'] * 1e3, color='green', ls=':', lw=1.5,
                label=f"δ99 = {bl_results['delta_99']*1e3:.3f} mm")

    # Annotations
    mu_w = diag['mu_wall']
    tau_w = bl_results['tau_w']
    cf = bl_results['C_f']
    textstr = (f"T_wall = {bl_results['T_wall']:.1f} K\n"
               f"μ_wall = {mu_w:.3e} Pa·s\n"
               f"du/dn = {du_dn:.1f} (m/s)/m\n"
               f"τ_w = {tau_w:.4f} Pa\n"
               f"C_f = {cf:.4e}\n"
               f"dy_grid = {dy_grid*1e3:.4f} mm")
    if wall_offset > 1e-6:
        textstr += f"\nwall_offset = {wall_offset*1e3:.3f} mm"
    if direct_cf is not None:
        textstr += f"\nCf_direct = {direct_cf['C_f']:.4e}"
    ax2.text(0.97, 0.03, textstr, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('u_tangential (m/s)', fontsize=14)
    ax2.set_ylabel('Wall-normal distance (mm)', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Near-wall velocity profile', fontsize=13)

    plt.suptitle(f'BL Diagnostic — snapshot {snapshot}, {side}, x = {x_wall:.4f} m',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    fname = f'{output_dir}/BL_diag_{side}_x{x_wall:.4f}_{snapshot}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✓ BL diagnostic plot saved: {fname}")
    return fname


def plot_surface_geometry(surfaces, fields_data, xx, yy, snapshot, 
                          output_dir='Surface_Analysis', field_key='Schlieren'):
    """
    Plot CFD field data with STL geometry outline and nearest wall cell centers overlaid.
    
    Parameters:
    -----------
    field_key : str
        Which field to plot as background (default: 'Schlieren')
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys

    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"  [DEBUG] Creating figure...")
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        # ---------------------------------------------------------- #
        #  BACKGROUND: CFD FIELD                                      #
        # ---------------------------------------------------------- #
        if field_key in fields_data:
            field = fields_data[field_key]
            vmin = np.nanpercentile(field, 1)
            vmax = np.nanpercentile(field, 99)
            cmap = 'gray' if field_key == 'Schlieren' else 'turbo'
            pcm = ax.pcolormesh(xx, yy, field,
                                cmap=cmap, shading='auto',
                                vmin=vmin, vmax=vmax,
                                rasterized=True, zorder=1)
            cbar = plt.colorbar(pcm, ax=ax, pad=0.01, fraction=0.012)
            cbar.set_label(field_key, fontsize=12)
            print(f"  [DEBUG] Plotted {field_key} field")
        else:
            print(f"  [DEBUG] Field '{field_key}' not found, plotting geometry only")
            ax.set_facecolor('black')

        # ---------------------------------------------------------- #
        #  GEOMETRY OUTLINE: exact STL wall (thin solid line)        #
        # ---------------------------------------------------------- #
        side_config = {
            'upper': {'color': 'lime',   'label': 'Upper wall (STL)'},
            'lower': {'color': 'cyan',   'label': 'Lower wall (STL)'},
        }

        for side, cfg in side_config.items():
            n_pts = len(surfaces[side]['x'])
            if n_pts == 0:
                continue

            color = cfg['color']

            # STL exact wall — thin bright line
            ax.plot(surfaces[side]['x'], surfaces[side]['y_exact'],
                    '-', color=color, linewidth=0.8, alpha=0.95,
                    label=cfg['label'], zorder=4)

            # Nearest fluid cell centers — small dots
            ax.scatter(surfaces[side]['x'], surfaces[side]['y'],
                       c=color, s=1.5, alpha=0.35, marker='o',
                       label=f'{side.capitalize()} cell centers',
                       zorder=3)

            # Connector lines (wall → cell center), every ~150th point
            n_skip = max(1, n_pts // 150)
            xw = surfaces[side]['x'][::n_skip]
            yw = surfaces[side]['y_exact'][::n_skip]
            yc = surfaces[side]['y'][::n_skip]
            for xwi, ywi, yci in zip(xw, yw, yc):
                ax.plot([xwi, xwi], [ywi, yci],
                        color=color, linewidth=0.4, alpha=0.3, zorder=2)

            print(f"  [DEBUG] {side}: {n_pts} wall pts, "
                  f"mean Δy={np.mean(surfaces[side]['dist_to_wall'])*1000:.3f} mm")

        # ---------------------------------------------------------- #
        #  FORMATTING                                                 #
        # ---------------------------------------------------------- #
        # Zoom to geometry region with small padding
        all_x = np.concatenate([surfaces['upper']['x'], surfaces['lower']['x']])
        all_y_wall = np.concatenate([surfaces['upper']['y_exact'],
                                     surfaces['lower']['y_exact']])
        all_y_cell = np.concatenate([surfaces['upper']['y'],
                                     surfaces['lower']['y']])

        if len(all_x) > 0:
            x_pad = (all_x.max() - all_x.min()) * 0.02
            y_range = max(all_y_cell.max(), all_y_wall.max()) - \
                      min(all_y_cell.min(), all_y_wall.min())
            y_pad = y_range * 0.15

            ax.set_xlim(all_x.min() - x_pad,  all_x.max() + x_pad)
            ax.set_ylim(min(all_y_cell.min(), all_y_wall.min()) - y_pad,
                        max(all_y_cell.max(), all_y_wall.max()) + y_pad)

        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        ax.set_title(f'{field_key} + STL Geometry Outline — Snapshot {snapshot}',
                     fontsize=15)
        ax.legend(fontsize=9, loc='upper left', markerscale=5,
                  framealpha=0.6)
        ax.set_aspect('equal', adjustable='box')

        print(f"  [DEBUG] Applying tight_layout...")
        plt.tight_layout()

        output_file = output_path / f'geometry_{snapshot}.png'
        sys.stdout.flush()
        plt.savefig(str(output_file), dpi=200, format='png',
                    bbox_inches='tight')
        plt.close(fig)

        if output_file.exists():
            print(f"  ✓ Saved: {output_file} ({output_file.stat().st_size} bytes)")
        else:
            print(f"  ✗ File not created!")

        return True

    except Exception as e:
        print(f"  ✗ ERROR in plot_surface_geometry:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        try:
            plt.close('all')
        except:
            pass
        return False


def build_stl_geometry_mask(xx, yy, profile):
    """
    Build a geometry mask using the STL upper/lower surface envelopes.
    
    A cell is marked as 'body' if:
        y_lower(x) <= y_cell <= y_upper(x)
    
    Parameters:
    -----------
    xx, yy : np.ndarray, shape (Nx, Ny)
        Cell-center meshgrid arrays from Silo_Read
    profile : dict
        Output from stl_to_2d_profile() — must contain 'x', 'y_upper', 'y_lower'
    
    Returns:
    --------
    mask : np.ndarray, shape (Nx, Ny)
        1.0 where cell is inside geometry, np.nan elsewhere
    """
    Nx, Ny = xx.shape
    mask = np.full((Nx, Ny), np.nan)

    x_cc = xx[:, 0]   # 1D x cell centers
    y_cc = yy[0, :]   # 1D y cell centers

    # Only process x-stations within the STL profile range
    x_in_range = (x_cc >= profile['x_min']) & (x_cc <= profile['x_max'])
    i_indices = np.where(x_in_range)[0]

    for i in i_indices:
        x_val = x_cc[i]
        y_wall_u = np.interp(x_val, profile['x'], profile['y_upper'])
        y_wall_l = np.interp(x_val, profile['x'], profile['y_lower'])

        # Mark cells between lower and upper wall as body
        body_cells = (y_cc >= y_wall_l) & (y_cc <= y_wall_u)
        mask[i, body_cells] = 1.0

    n_body = np.sum(np.isfinite(mask))
    print(f"  [STL mask] {n_body} body cells identified "
          f"({100*n_body/(Nx*Ny):.2f}% of domain)")

    return mask