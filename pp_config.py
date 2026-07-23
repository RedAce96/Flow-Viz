"""Configuration loading and validation for PeleC post-processing."""

import copy
import json
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


def build_config(defaults, json_path=None, command_line_overrides=None):
    """Deep-copy defaults, apply a checked JSON overlay, and validate it."""
    config = copy.deepcopy(defaults)
    if json_path is not None:
        with Path(json_path).open(encoding="utf-8") as stream:
            overlay = json.load(stream)
        if not isinstance(overlay, dict):
            raise ValueError("Configuration JSON must contain one object")
        unknown = sorted(set(overlay) - set(config))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
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
    if int(config.get("fft_batch_size", 1)) < 1:
        raise ValueError("fft_batch_size must be at least 1")
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
    if int(config.get("animation_fps", 8)) < 1:
        raise ValueError("animation_fps must be at least 1")
    if (config.get("make_force_animation", False)
            and not config.get("make_surface_analysis", False)):
        raise ValueError(
            "make_force_animation requires make_surface_analysis=true"
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
