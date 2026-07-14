"""Configuration loading and validation for PeleC post-processing."""

import copy
import json
from pathlib import Path


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
        config.update(overlay)
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
    return config


def write_config(config, destination):
    """Write a reusable JSON configuration containing JSON-native values."""
    destination = Path(destination)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
