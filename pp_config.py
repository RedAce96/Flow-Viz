"""Configuration loading and validation for PeleC post-processing."""

import copy
import json
import warnings
from pathlib import Path


def _deep_merge(defaults, overlay):
    """Merge a user configuration overlay without discarding nested defaults."""
    merged = copy.deepcopy(defaults)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _unknown_configuration_paths(defaults, overlay, prefix=""):
    """Return unknown JSON keys, including misspellings in nested sections."""
    unknown = []
    for key, value in overlay.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in defaults:
            unknown.append(path)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            unknown.extend(
                _unknown_configuration_paths(defaults[key], value, path)
            )
    return unknown


def build_config(defaults, json_path=None, command_line_overrides=None):
    """Deep-copy defaults, apply a checked JSON overlay, and validate it."""
    config = copy.deepcopy(defaults)
    if json_path is not None:
        with Path(json_path).open(encoding="utf-8") as stream:
            overlay = json.load(stream)
        if not isinstance(overlay, dict):
            raise ValueError("Configuration JSON must contain one object")
        unknown = sorted(_unknown_configuration_paths(config, overlay))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        legacy_force_mapping = {
            "surface_rho_inf": ("reference", "rho_inf"),
            "surface_u_inf": ("reference", "u_inf"),
            "surface_T_inf": ("reference", "T_inf"),
            "surface_mu": ("transport", "mu_pa_s"),
            "surface_k": ("transport", "k_w_m_k"),
            "surface_wall_temperature": (
                "transport", "wall_temperature_k"
            ),
        }
        legacy_used = sorted(set(overlay) & set(legacy_force_mapping))
        if "surface_max_x" in overlay:
            legacy_used.append("surface_max_x")
        if legacy_used and overlay.get(
                "make_force_analysis",
                config.get("make_force_analysis", False)):
            force_overlay = overlay.setdefault("force", {})
            for legacy_key, (section, new_key) in legacy_force_mapping.items():
                if legacy_key not in overlay:
                    continue
                section_overlay = force_overlay.setdefault(section, {})
                section_overlay.setdefault(new_key, overlay[legacy_key])
            if "surface_max_x" in overlay:
                geometry_overlay = force_overlay.setdefault("geometry", {})
                geometry_overlay.setdefault(
                    "x_range_m", [0.0, overlay["surface_max_x"]]
                )
            warnings.warn(
                "Legacy surface_* force settings are deprecated and were "
                "translated into the nested force configuration. Set "
                "force.reference.p_inf explicitly; legacy settings never "
                "supply or infer freestream pressure.",
                DeprecationWarning,
                stacklevel=2,
            )
        config = _deep_merge(config, overlay)
    if command_line_overrides:
        config.update({
            key: value for key, value in command_line_overrides.items()
            if value is not None
        })
    validate_config(config)
    return config


def validate_config(config):
    """Reject common configuration mistakes before loading large datasets."""
    required = ("data_source", "output_dir", "plot_prefix")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    if int(config.get("snapshot_step", 1)) <= 0:
        raise ValueError("snapshot_step must be positive")
    if config.get("plot_time_mode") not in ("flow_through", "physical", "reference"):
        raise ValueError("plot_time_mode is invalid")
    if config.get("plot_time_mode") == "flow_through" and float(
            config.get("plot_flow_through_u_inf", 0.0)) <= 0:
        raise ValueError("plot_flow_through_u_inf must be positive")
    colorbar_shrink = config.get("contour_colorbar_shrink", "auto")
    if isinstance(colorbar_shrink, str):
        if colorbar_shrink.lower() != "auto":
            raise ValueError(
                "contour_colorbar_shrink must be 'auto' or a number"
            )
    elif not 0.0 < float(colorbar_shrink) <= 1.0:
        raise ValueError("contour_colorbar_shrink must lie in (0, 1]")
    if float(config.get("contour_colorbar_reference_span", 0.3)) <= 0.0:
        raise ValueError("contour_colorbar_reference_span must be positive")
    if int(config.get("fft_batch_size", 1)) < 1:
        raise ValueError("fft_batch_size must be at least 1")
    coordinate_policy = str(
        config.get("probe_coordinate_policy", "strict")
    ).lower()
    if coordinate_policy not in ("strict", "nominal", "longest_epoch"):
        raise ValueError(
            "probe_coordinate_policy must be 'strict', 'nominal', or "
            "'longest_epoch'"
        )
    if config.get("reconstruction_method") not in (
            "harmonics", "band", "top_frequencies"):
        raise ValueError("reconstruction_method is invalid")
    for key in ("reconstruction_band", "transient_band", "stability_freq_band"):
        band = config.get(key)
        if band is not None and (
                len(band) != 2 or float(band[0]) < 0
                or float(band[1]) <= float(band[0])):
            raise ValueError(f"{key} must be [f_low, f_high] with f_high > f_low")
    train_fraction = float(config.get("reconstruction_common_train_fraction", 0.6))
    if not 0.2 <= train_fraction <= 0.8:
        raise ValueError("reconstruction_common_train_fraction must lie in [0.2, 0.8]")
    overlap = float(config.get("transient_stft_noverlap", 0.5))
    if not 0 <= overlap < 1:
        raise ValueError("transient_stft_noverlap must lie in [0, 1)")
    stability_overlap = float(
        config.get("stability_wavenumber_noverlap", 0.5)
    )
    if not 0 <= stability_overlap < 1:
        raise ValueError(
            "stability_wavenumber_noverlap must lie in [0, 1)"
        )
    if int(config.get("stability_wavenumber_nperseg", 16384)) < 8:
        raise ValueError("stability_wavenumber_nperseg must be at least 8")
    if int(config.get("stability_frequency_stride", 1)) < 1:
        raise ValueError("stability_frequency_stride must be at least 1")
    if int(config.get("stability_spatial_window_size", 101)) < 5:
        raise ValueError("stability_spatial_window_size must be at least 5")
    if int(config.get("stability_spatial_step", 20)) < 1:
        raise ValueError("stability_spatial_step must be at least 1")
    for key in (
            "stability_wavenumber_min_coherence",
            "stability_wavenumber_min_coherent_fraction",
            "stability_wavenumber_min_phase_r2",
            "stability_wavenumber_min_amplitude_r2"):
        value = float(config.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1]")
    maximum_phase = float(
        config.get("stability_wavenumber_max_edge_phase_rad", 2.8)
    )
    if not 0.0 < maximum_phase <= 3.141592653589793:
        raise ValueError(
            "stability_wavenumber_max_edge_phase_rad must lie in (0, pi]"
        )
    relative_power = float(
        config.get("stability_wavenumber_min_relative_power_db", -40.0)
    )
    if relative_power != relative_power or relative_power > 0.0:
        raise ValueError(
            "stability_wavenumber_min_relative_power_db must be <= 0"
        )
    speed_bounds = config.get("stability_phase_speed_bounds")
    if speed_bounds is not None and (
            len(speed_bounds) != 2 or float(speed_bounds[0]) < 0.0
            or float(speed_bounds[1]) <= float(speed_bounds[0])):
        raise ValueError(
            "stability_phase_speed_bounds must be [min, max] with max > min"
        )
    reference_tolerance = float(
        config.get("stability_acoustic_reference_tolerance", 0.25)
    )
    if not 0.0 <= reference_tolerance <= 1.0:
        raise ValueError(
            "stability_acoustic_reference_tolerance must lie in [0, 1]"
        )
    analysis_window = config.get("stability_analysis_time_window")
    if analysis_window is not None and (
            len(analysis_window) != 2
            or float(analysis_window[1]) <= float(analysis_window[0])):
        raise ValueError(
            "stability_analysis_time_window must be [start_s, end_s]"
        )
    if int(config.get("animation_fps", 8)) < 1:
        raise ValueError("animation_fps must be at least 1")
    if (config.get("make_force_animation", False)
            and not config.get("make_surface_analysis", False)):
        raise ValueError(
            "make_force_animation requires make_surface_analysis=true"
        )
    if int(config.get("modal_probe_stride", 1)) < 1:
        raise ValueError("modal_probe_stride must be positive")
    if int(config.get("modal_n_modes", 1)) < 1:
        raise ValueError("modal_n_modes must be positive")
    modal_ranks = config.get("modal_sensitivity_dmd_ranks", [])
    if not modal_ranks:
        raise ValueError(
            "modal_sensitivity_dmd_ranks must contain at least one rank"
        )
    if any(
            int(rank) < 1
            for rank in modal_ranks):
        raise ValueError("modal_sensitivity_dmd_ranks must be positive")
    modal_windows = config.get("modal_sensitivity_windows", [])
    if len(modal_windows) < 2:
        raise ValueError(
            "modal_sensitivity_windows must contain at least two windows"
        )
    for window in modal_windows:
        if (
                not isinstance(window, (list, tuple)) or len(window) != 2
                or not 0.0 <= float(window[0]) < float(window[1]) <= 1.0):
            raise ValueError(
                "modal_sensitivity_windows entries must satisfy "
                "0 <= start < stop <= 1"
            )
    if float(config.get("modal_base_rho_kg_m3", 0.0)) <= 0.0:
        raise ValueError("modal_base_rho_kg_m3 must be positive")
    if float(config.get("modal_base_temperature_k", 0.0)) <= 0.0:
        raise ValueError("modal_base_temperature_k must be positive")
    if float(config.get("transient_arrival_noise_sigma", 0.0)) <= 0.0:
        raise ValueError("transient_arrival_noise_sigma must be positive")
    peak_fraction = float(config.get("transient_arrival_peak_fraction", 0.0))
    if not 0.0 < peak_fraction < 1.0:
        raise ValueError(
            "transient_arrival_peak_fraction must lie in (0, 1)"
        )
    if int(config.get("transient_arrival_persistent_samples", 0)) < 1:
        raise ValueError(
            "transient_arrival_persistent_samples must be positive"
        )
    edge_fraction = float(config.get("transient_filter_edge_fraction", 0.0))
    if not 0.0 <= edge_fraction < 0.25:
        raise ValueError(
            "transient_filter_edge_fraction must lie in [0, 0.25)"
        )
    spectrogram_scale = str(
        config.get("transient_spectrogram_scale", "db")
    ).lower()
    if spectrogram_scale not in ("linear", "db"):
        raise ValueError(
            "transient_spectrogram_scale must be 'linear' or 'db'"
        )
    spectrogram_minimum = config.get("transient_spectrogram_vmin")
    spectrogram_maximum = config.get("transient_spectrogram_vmax")
    if (
            spectrogram_scale == "linear"
            and spectrogram_minimum is not None
            and float(spectrogram_minimum) < 0.0):
        raise ValueError(
            "linear transient_spectrogram_vmin cannot be negative"
        )
    if (
            spectrogram_minimum is not None
            and spectrogram_maximum is not None
            and float(spectrogram_maximum) <= float(spectrogram_minimum)):
        raise ValueError(
            "transient_spectrogram_vmax must exceed vmin"
        )
    if float(config.get("transient_spectrogram_fmax_hz", 1.0)) <= 0.0:
        raise ValueError(
            "transient_spectrogram_fmax_hz must be positive"
        )
    if int(config.get("nonlinear_surrogate_count", 0)) < 19:
        raise ValueError("nonlinear_surrogate_count must be at least 19")
    if int(config.get("nonlinear_nperseg", 0)) < 8:
        raise ValueError("nonlinear_nperseg must be at least 8")
    nonlinear_overlap = float(config.get("nonlinear_noverlap", 0.5))
    if not 0.0 <= nonlinear_overlap < 1.0:
        raise ValueError("nonlinear_noverlap must lie in [0, 1)")
    surrogate_alpha = float(config.get("nonlinear_surrogate_alpha", 0.05))
    if not 0.0 < surrogate_alpha < 1.0:
        raise ValueError("nonlinear_surrogate_alpha must lie in (0, 1)")
    if int(config.get("nonlinear_minimum_independent_segments", 0)) < 2:
        raise ValueError(
            "nonlinear_minimum_independent_segments must be at least 2"
        )

    force = config.get("force")
    if config.get("make_force_analysis", False) and not isinstance(force, dict):
        raise ValueError("make_force_analysis requires a nested force configuration")
    if isinstance(force, dict):
        geometry = force.get("geometry", {})
        reference = force.get("reference", {})
        transport = force.get("transport", {})
        wall_fit = force.get("wall_fit", {})
        baseline = force.get("baseline", {})
        boundary_layer = force.get("boundary_layer", {})
        quality = force.get("quality", {})
        history = force.get("history", {})
        validation = force.get("validation", {})
        laser = force.get("laser", {})

        if geometry.get("type") != "flat_plate_one_sided":
            raise ValueError(
                "certified force.geometry.type must be 'flat_plate_one_sided'"
            )
        x_range = geometry.get("x_range_m")
        if (
            not isinstance(x_range, (list, tuple))
            or len(x_range) != 2
            or float(x_range[0]) < 0.0
            or float(x_range[1]) <= float(x_range[0])
        ):
            raise ValueError("force.geometry.x_range_m must be increasing and non-negative")
        required_reference = (
            "rho_inf", "u_inf", "p_inf", "T_inf", "R",
            "chord_m", "span_m", "moment_origin_m",
        )
        missing_force_reference = [
            key for key in required_reference if reference.get(key) is None
        ]
        if missing_force_reference:
            raise ValueError(
                "force.reference is missing: "
                + ", ".join(missing_force_reference)
            )
        for key in (
            "rho_inf", "u_inf", "p_inf", "T_inf", "R",
            "chord_m", "span_m",
        ):
            if float(reference[key]) <= 0.0:
                raise ValueError(f"force.reference.{key} must be positive")
        moment_origin = reference["moment_origin_m"]
        if not isinstance(moment_origin, (list, tuple)) or len(moment_origin) != 2:
            raise ValueError("force.reference.moment_origin_m must be [x, y]")
        if float(x_range[1]) > float(reference["chord_m"]) + 1.0e-12:
            raise ValueError(
                "force geometry interval cannot exceed the reference chord"
            )
        ideal_pressure = (
            float(reference["rho_inf"])
            * float(reference["R"])
            * float(reference["T_inf"])
        )
        pressure_error = abs(float(reference["p_inf"]) - ideal_pressure) / max(
            abs(float(reference["p_inf"])), 1.0e-300
        )
        if pressure_error > 0.01:
            warnings.warn(
                "force.reference p_inf differs from rho_inf*R*T_inf by "
                f"{100.0 * pressure_error:.2f}%",
                RuntimeWarning,
                stacklevel=2,
            )

        if transport.get("model") != "constant":
            raise ValueError(
                "the certified flat-plate force path currently requires "
                "force.transport.model='constant'"
            )
        for key in ("mu_pa_s", "k_w_m_k", "wall_temperature_k"):
            if transport.get(key) is None or float(transport[key]) <= 0.0:
                raise ValueError(f"force.transport.{key} must be positive")
        for key in ("pressure_order", "velocity_order"):
            if int(wall_fit.get(key, 0)) not in (1, 2):
                raise ValueError(f"force.wall_fit.{key} must be 1 or 2")
        if int(wall_fit.get("fluid_points", 0)) < 2:
            raise ValueError("force.wall_fit.fluid_points must be at least 2")
        if float(wall_fit.get("viscosity_relative_tolerance", 0.0)) < 0.0:
            raise ValueError(
                "force.wall_fit.viscosity_relative_tolerance must be non-negative"
            )

        baseline_mode = baseline.get("mode", "none")
        if baseline_mode not in ("none", "static", "paired"):
            raise ValueError("force.baseline.mode must be none, static, or paired")
        if baseline_mode == "static" and not baseline.get("plotfile"):
            raise ValueError("static force baseline requires force.baseline.plotfile")
        if baseline_mode == "paired" and (
            not baseline.get("paired_data_source")
            or not baseline.get("paired_plot_prefix")
        ):
            raise ValueError(
                "paired force baseline requires paired_data_source and paired_plot_prefix"
            )
        if float(baseline.get("time_tolerance_s", 0.0)) < 0.0:
            raise ValueError("force.baseline.time_tolerance_s must be non-negative")
        if int(baseline.get("drift_sample_count", 0)) < 0:
            raise ValueError(
                "force.baseline.drift_sample_count must be non-negative"
            )
        if (
            int(baseline.get("drift_sample_count", 0)) > 0
            and not baseline.get("drift_plot_prefix")
        ):
            raise ValueError(
                "force baseline drift sampling requires drift_plot_prefix"
            )

        stations = boundary_layer.get("stations_m", [])
        if boundary_layer.get("enabled", False) and not stations:
            raise ValueError(
                "enabled force boundary-layer extraction requires stations_m"
            )
        if any(float(value) < float(x_range[0])
               or float(value) > float(x_range[1]) for value in stations):
            raise ValueError(
                "force boundary-layer stations must lie in geometry.x_range_m"
            )
        if float(boundary_layer.get("maximum_height_m", 0.0)) <= 0.0:
            raise ValueError(
                "force.boundary_layer.maximum_height_m must be positive"
            )
        if not 0.0 < float(quality.get("minimum_coverage", 0.0)) <= 1.0:
            raise ValueError("force.quality.minimum_coverage must lie in (0, 1]")
        if float(quality.get("maximum_gap_widths", 0.0)) < 0.0:
            raise ValueError(
                "force.quality.maximum_gap_widths must be non-negative"
            )
        for key in ("frequency_hz", "pulse_fwhm_s", "duration_s"):
            if float(laser.get(key, 0.0)) <= 0.0:
                raise ValueError(f"force.laser.{key} must be positive")
        impulse_interval = laser.get("impulse_interval_s")
        if impulse_interval is not None and (
                not isinstance(impulse_interval, (list, tuple))
                or len(impulse_interval) != 2
                or float(impulse_interval[1]) <= float(impulse_interval[0])):
            raise ValueError(
                "force.laser.impulse_interval_s must be [start, stop] "
                "in absolute seconds or null"
            )
        history_overlap = float(history.get("spectral_noverlap", 0.5))
        if not 0.0 <= history_overlap < 1.0:
            raise ValueError(
                "force.history.spectral_noverlap must lie in [0, 1)"
            )
        if int(history.get("spectral_nperseg", 0)) < 16:
            raise ValueError(
                "force.history.spectral_nperseg must be at least 16"
            )
        if int(history.get("probe_pressure_var_col", 3)) not in (3,):
            raise ValueError(
                "force.history.probe_pressure_var_col must select pressure "
                "(legacy/chunked probe column 3)"
            )
        probe_limit = history.get("probe_max_probes")
        if probe_limit is not None and int(probe_limit) <= 0:
            raise ValueError(
                "force.history.probe_max_probes must be positive or null"
            )
        start_locations = validation.get(
            "integration_start_locations_m", []
        )
        if any(
                float(value) < float(x_range[0])
                or float(value) >= float(x_range[1])
                for value in start_locations):
            raise ValueError(
                "force.validation integration starts must lie inside "
                "geometry.x_range_m"
            )
        control_volume = validation.get("control_volume", {})
        if control_volume.get("enabled", False):
            cv_range = control_volume.get("x_range_m")
            if (
                not isinstance(cv_range, (list, tuple))
                or len(cv_range) != 2
                or float(cv_range[0]) < float(x_range[0])
                or float(cv_range[1]) > float(x_range[1])
                or float(cv_range[1]) <= float(cv_range[0])
            ):
                raise ValueError(
                    "force.validation.control_volume.x_range_m must be "
                    "inside the force interval"
                )
            if float(control_volume.get("top_y_m", 0.0)) <= float(
                    geometry.get("wall_y_m", 0.0)):
                raise ValueError(
                    "force.validation.control_volume.top_y_m must be above "
                    "the wall"
                )
            if int(control_volume.get("maximum_level", 0)) < 0:
                raise ValueError(
                    "force.validation.control_volume.maximum_level must "
                    "be non-negative"
                )

    reference_model = config.get("line_reference_model", "none")
    if reference_model not in (
            "none", "compressible_similarity", "incompressible_blasius_legacy"):
        raise ValueError("line_reference_model is invalid")
    if (config.get("make_line_profiles", False)
            and config.get("line_reference_overlay", False)
            and reference_model == "compressible_similarity"):
        reference = config.get("line_reference", {})
        transport_model = reference.get("transport_model", "constant")
        if transport_model not in ("constant", "sutherland"):
            raise ValueError("line_reference.transport_model is invalid")
        required_reference = ("u_inf", "T_inf", "rho_inf", "leading_edge_x")
        missing_reference = [
            key for key in required_reference if reference.get(key) is None
        ]
        if missing_reference:
            raise ValueError(
                "line_reference is missing: " + ", ".join(missing_reference)
            )
        if any(float(reference[key]) <= 0 for key in
               ("u_inf", "T_inf", "rho_inf", "gamma", "R", "Cp")):
            raise ValueError("line_reference freestream and gas properties must be positive")
        if transport_model == "constant":
            if float(reference.get("mu", 0.0)) <= 0 or float(reference.get("k", 0.0)) <= 0:
                raise ValueError("constant line_reference transport requires positive mu and k")
        elif float(reference.get("Pr", 0.0)) <= 0:
            raise ValueError("Sutherland line_reference transport requires positive Pr")
        if config.get("geometry_type") != "flat_plate" or not reference.get(
                "zero_pressure_gradient", False):
            raise ValueError(
                "compressible similarity reference requires geometry_type='flat_plate' "
                "and line_reference.zero_pressure_gradient=true"
            )
    return config


def write_config(config, destination):
    """Write a reusable JSON configuration containing JSON-native values."""
    destination = Path(destination)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
