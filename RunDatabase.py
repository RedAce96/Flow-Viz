# This Python file is used to setup and run the other scripts from the FuncDatabase.py
# It is not meant to be run on its own, but to be imported and used in an overall interface script.

import os
import inspect
from pathlib import Path
import matplotlib.pyplot as plt

from FuncDatabase import *

def _normalize_case_configs(case_configs, config_name):
	if case_configs is None:
		return []

	if isinstance(case_configs, (str, os.PathLike)):
		return [{'source': os.fspath(case_configs)}]

	if isinstance(case_configs, dict):
		return [dict(case_configs)]

	normalized = []
	for case_config in case_configs:
		if isinstance(case_config, (str, os.PathLike)):
			normalized.append({'source': os.fspath(case_config)})
		elif isinstance(case_config, dict):
			normalized.append(dict(case_config))
		else:
			raise TypeError(
				f"{config_name} entries must be path-like objects or dicts; got {type(case_config)!r}."
			)

	return normalized


def _get_required_source(case_config, config_name):
	source = case_config.get('source', case_config.get('path'))
	if source is None:
		raise KeyError(f"Each {config_name} config must define 'source'.")
	return source


def _infer_base_dir(base_dir=None):
	if base_dir is not None:
		base_path = Path(os.fspath(base_dir)).expanduser()
		if not base_path.is_absolute():
			base_path = Path.cwd() / base_path
		return base_path.resolve()

	this_file = Path(__file__).resolve()
	for frame_info in inspect.stack()[1:]:
		filename = frame_info.filename
		if not filename or filename.startswith('<'):
			continue

		frame_path = Path(filename).resolve()
		if frame_path != this_file:
			return frame_path.parent

	return Path.cwd().resolve()


def _resolve_path(path_value, base_dir):
	path = Path(os.fspath(path_value)).expanduser()
	if not path.is_absolute():
		path = base_dir / path
	return path.resolve()


def _load_pelec_dataset(case_config, plot_prefix, convert_to_mks, base_dir):
	source = _resolve_path(_get_required_source(case_config, 'pelec_cases'), base_dir)
	datasets = load_pelec_plotfile_series(
		os.fspath(source),
		plot_prefix=case_config.get('plot_prefix', plot_prefix),
		field_names=case_config.get('field_names'),
		convert_to_mks=case_config.get('convert_to_mks', convert_to_mks),
	)

	if len(datasets) == 0:
		raise ValueError(f"No PeleC plotfiles were found for source: {source}")

	plot_index = case_config.get('plot_index', -1)
	try:
		return datasets[plot_index]
	except IndexError as exc:
		raise IndexError(
			f"Requested plot_index={plot_index} for source {source}, but only {len(datasets)} plotfile(s) were loaded."
		) from exc


def _load_mfc_dataset(case_config, base_dir):
	source = _resolve_path(_get_required_source(case_config, 'mfc_cases'), base_dir)
	return Silo_Read(
		os.fspath(source),
		slice_plane=case_config.get('slice_plane'),
		slice_index=case_config.get('slice_index'),
		slice_coord=case_config.get('slice_coord'),
		species=case_config.get('species', True),
	)


def LineExtraction(
	pelec_cases,
	field_key,
	x_location,
	mfc_cases=None,
	surface_y=0.0,
	y_max=None,
	side='positive',
	field_label=None,
	title=None,
	output_path=None,
	x_key='distance_from_surface',
	xlabel='Distance From Surface [m]',
	plot_prefix='plt',
	convert_to_mks=True,
	base_dir=None,
	ax=None,
):
	"""
	High-level controller for wall-normal profile extraction and plotting.

	Parameters
	----------
	pelec_cases : path-like, dict, or sequence of either
		PeleC sources to load. Each dict supports:
		``source``, ``label``, ``linestyle``, ``plot_index``, ``plot_prefix``,
		``field_names``, and ``convert_to_mks``.
	field_key : str
		Field to extract, such as ``Temperature`` or ``Pressure``.
	x_location : float
		Streamwise location at which to extract the wall-normal profile.
	mfc_cases : path-like, dict, or sequence of either, optional
		Optional MFC Silo sources to compare against. Each dict supports:
		``source``, ``label``, ``linestyle``, ``slice_plane``, ``slice_index``,
		``slice_coord``, and ``species``.
	surface_y : float, default=0.0
		Physical surface location used as the wall origin.
	y_max : float, optional
		Maximum wall-normal distance to include in the extracted profile.
	side : {'positive', 'negative'}, default='positive'
		Direction of the wall-normal extraction.
	field_label : str, optional
		Y-axis label for the plot.
	title : str, optional
		Figure title.
	output_path : str or path-like, optional
		Output path for the generated figure.
	x_key : str, default='distance_from_surface'
		Profile key to use on the horizontal axis.
	xlabel : str, default='Distance From Surface [m]'
		X-axis label.
	plot_prefix : str, default='plt'
		Default PeleC plotfile prefix when a case directory is provided.
	convert_to_mks : bool, default=True
		Default PeleC unit conversion flag.
	base_dir : str or path-like, optional
		Base directory used to resolve relative source and output paths.
		If omitted, the calling script directory is used when available.
	ax : matplotlib.axes.Axes, optional
		Existing axis to draw into.

	Returns
	-------
	dict
		Contains the extracted profiles, loaded datasets, plotting axis,
		and the resolved output path.
	"""
	pelec_case_configs = _normalize_case_configs(pelec_cases, 'pelec_cases')
	mfc_case_configs = _normalize_case_configs(mfc_cases, 'mfc_cases')
	resolved_base_dir = _infer_base_dir(base_dir)

	if len(pelec_case_configs) == 0 and len(mfc_case_configs) == 0:
		raise ValueError('At least one PeleC or MFC source is required.')

	profiles = []
	pelec_results = []
	mfc_results = []

	for case_config in pelec_case_configs:
		dataset = _load_pelec_dataset(case_config, plot_prefix, convert_to_mks, resolved_base_dir)
		profile = extract_surface_normal_profile(
			dataset,
			field_key,
			x_location,
			surface_y=surface_y,
			y_max=y_max,
			side=side,
			label=case_config.get('label'),
		)
		if 'linestyle' in case_config:
			profile['linestyle'] = case_config['linestyle']
		profiles.append(profile)
		pelec_results.append({
			'config': case_config,
			'dataset': dataset,
			'profile': profile,
		})

	for case_config in mfc_case_configs:
		dataset = _load_mfc_dataset(case_config, resolved_base_dir)
		profile = extract_surface_normal_profile(
			dataset,
			field_key,
			x_location,
			surface_y=surface_y,
			y_max=y_max,
			side=side,
			label=case_config.get('label', Path(os.fspath(_get_required_source(case_config, 'mfc_cases'))).stem),
		)
		if 'linestyle' in case_config:
			profile['linestyle'] = case_config['linestyle']
		profiles.append(profile)
		mfc_results.append({
			'config': case_config,
			'dataset': dataset,
			'profile': profile,
		})

	resolved_output_path = None
	if output_path is not None:
		resolved_output_path = _resolve_path(output_path, resolved_base_dir)
		resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

	plot_axis = plot_surface_normal_profiles(
		profiles,
		field_label=xlabel,
		title=title,
		output_path=os.fspath(resolved_output_path) if resolved_output_path is not None else None,
		x_key=x_key,
		xlabel=field_label,
		ax=ax,
	)

	if resolved_output_path is not None:
		print(f"Saved line-extraction plot to: {resolved_output_path}")

	return {
		'profiles': profiles,
		'pelec_results': pelec_results,
		'mfc_results': mfc_results,
		'axis': plot_axis,
		'output_path': os.fspath(resolved_output_path) if resolved_output_path is not None else None,
	}


def StreamlinePlot(
	pelec_cases=None,
	mfc_cases=None,
	u_key='u',
	v_key='v',
	color_key=None,
	mask_key=None,
	mask_threshold=None,
	mask_mode='below',
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
	plot_prefix='plt',
	convert_to_mks=True,
	base_dir=None,
	colorbar=None,
	colorbar_label=None,
	mask_fill_color='black',
	mask_fill_alpha=1.0,
	ax=None,
):
	"""
	High-level controller for steady streamline plotting from structured 2-D data.

	Parameters
	----------
	pelec_cases : path-like, dict, or sequence of either, optional
		PeleC sources to load. Each dict supports the same keys as ``LineExtraction``.
	mfc_cases : path-like, dict, or sequence of either, optional
		Optional MFC Silo sources to compare against. Each dict supports the same
		keys as ``LineExtraction``.
	u_key, v_key : str, default=('u', 'v')
		Field names for the in-plane velocity components.
	color_key : str, optional
		Optional scalar field used to color the streamlines.
	mask_key : str, optional
		Optional scalar field used to suppress streamlines in masked cells.
	mask_threshold : float, optional
		Threshold applied to ``mask_key``.
	mask_mode : {'below', 'above'}, default='below'
		Mask cells where ``mask_key < mask_threshold`` or
		``mask_key > mask_threshold``.
	title : str, optional
		Figure title.
	output_path : str or path-like, optional
		Output path for the generated figure.
	xlim, ylim : tuple[float, float], optional
		Axis limits.
	density : float or tuple, default=1.5
		Forwarded to ``matplotlib`` streamplot.
	linewidth : float, default=1.0
		Base streamline width.
	arrowsize : float, default=1.0
		Streamline arrow scale.
	color : str, optional
		Fixed color applied when ``color_key`` is omitted.
	cmap : str, default='viridis'
		Colormap used when ``color_key`` is provided.
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
		Exponent used when ``seed_spacing='power'``.
	broken_streamlines : bool, default=True
		Forwarded to ``matplotlib`` streamplot.
	plot_prefix : str, default='plt'
		Default PeleC plotfile prefix when a case directory is provided.
	convert_to_mks : bool, default=True
		Default PeleC unit conversion flag.
	base_dir : str or path-like, optional
		Base directory used to resolve relative source and output paths.
	colorbar : bool, optional
		Whether to draw a colorbar for scalar-colored streamlines.
	colorbar_label : str, optional
		Colorbar label override.
	mask_fill_color : str, default='black'
		Fill color used to cover masked cells.
	mask_fill_alpha : float, default=1.0
		Alpha used for the masked-cell fill.
	ax : matplotlib.axes.Axes, optional
		Existing axis to draw into.

	Returns
	-------
	dict
		Contains the extracted streamline sets, loaded datasets, plotting axis,
		and the resolved output path.
	"""
	pelec_case_configs = _normalize_case_configs(pelec_cases, 'pelec_cases')
	mfc_case_configs = _normalize_case_configs(mfc_cases, 'mfc_cases')
	resolved_base_dir = _infer_base_dir(base_dir)

	if len(pelec_case_configs) == 0 and len(mfc_case_configs) == 0:
		raise ValueError('At least one PeleC or MFC source is required.')

	streamline_sets = []
	pelec_results = []
	mfc_results = []

	for case_config in pelec_case_configs:
		pelec_case_config = dict(case_config)
		requested_field_names = pelec_case_config.get('field_names')
		requested_color_key = pelec_case_config.get('color_key', color_key)
		requested_mask_key = pelec_case_config.get('mask_key', mask_key)

		if requested_field_names is None:
			requested_field_names = {}
		elif isinstance(requested_field_names, dict):
			requested_field_names = dict(requested_field_names)
		else:
			requested_field_names = {str(name): str(name) for name in requested_field_names}

		for field_name in (pelec_case_config.get('u_key', u_key), pelec_case_config.get('v_key', v_key), requested_color_key, requested_mask_key):
			if field_name is None or field_name in {'speed', 'velocity_magnitude', 'vel_mag'}:
				continue
			requested_field_names.setdefault(str(field_name), str(field_name))

		if requested_field_names:
			pelec_case_config['field_names'] = requested_field_names

		dataset = _load_pelec_dataset(pelec_case_config, plot_prefix, convert_to_mks, resolved_base_dir)
		streamline_set = extract_streamline_field(
			dataset,
			u_key=pelec_case_config.get('u_key', u_key),
			v_key=pelec_case_config.get('v_key', v_key),
			color_key=pelec_case_config.get('color_key', color_key),
			mask_key=pelec_case_config.get('mask_key', mask_key),
			mask_threshold=pelec_case_config.get('mask_threshold', mask_threshold),
			mask_mode=pelec_case_config.get('mask_mode', mask_mode),
			label=pelec_case_config.get('label'),
		)
		streamline_sets.append(streamline_set)
		pelec_results.append({
			'config': pelec_case_config,
			'dataset': dataset,
			'streamlines': streamline_set,
		})

	for case_config in mfc_case_configs:
		dataset = _load_mfc_dataset(case_config, resolved_base_dir)
		streamline_set = extract_streamline_field(
			dataset,
			u_key=case_config.get('u_key', u_key),
			v_key=case_config.get('v_key', v_key),
			color_key=case_config.get('color_key', color_key),
			mask_key=case_config.get('mask_key', mask_key),
			mask_threshold=case_config.get('mask_threshold', mask_threshold),
			mask_mode=case_config.get('mask_mode', mask_mode),
			label=case_config.get('label', Path(os.fspath(_get_required_source(case_config, 'mfc_cases'))).stem),
		)
		streamline_sets.append(streamline_set)
		mfc_results.append({
			'config': case_config,
			'dataset': dataset,
			'streamlines': streamline_set,
		})

	resolved_output_path = None
	if output_path is not None:
		resolved_output_path = _resolve_path(output_path, resolved_base_dir)
		resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

	plot_axis = plot_streamlines(
		streamline_sets,
		title=title,
		output_path=os.fspath(resolved_output_path) if resolved_output_path is not None else None,
		xlim=xlim,
		ylim=ylim,
		density=density,
		linewidth=linewidth,
		arrowsize=arrowsize,
		color=color,
		cmap=cmap,
		start_points=start_points,
		seed_lines=seed_lines,
		seed_count=seed_count,
		seed_x_position=seed_x_position,
		seed_y_min=seed_y_min,
		seed_y_max=seed_y_max,
		seed_spacing=seed_spacing,
		seed_power=seed_power,
		broken_streamlines=broken_streamlines,
		colorbar=colorbar,
		colorbar_label=colorbar_label,
		mask_fill_color=mask_fill_color,
		mask_fill_alpha=mask_fill_alpha,
		ax=ax,
	)

	if resolved_output_path is not None:
		print(f"Saved streamline plot to: {resolved_output_path}")

	return {
		'streamline_sets': streamline_sets,
		'pelec_results': pelec_results,
		'mfc_results': mfc_results,
		'axis': plot_axis,
		'output_path': os.fspath(resolved_output_path) if resolved_output_path is not None else None,
	}
