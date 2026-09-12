"""Versioned contracts for numerical analysis products.

Products are still stored as ordinary NPZ/JSON files, but every numerical
artifact gets a typed description at registration time. The description is
what lets comparison and plotting distinguish measured values from axes,
labels, and validity masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


COMPARISON_POLICIES = (
    "strict", "intersection", "interpolate_to_baseline",
    "interpolate_to_comparison",
)

COORDINATE_KEYS = frozenset({
    "time_s", "raw_time_s", "relative_time_s", "frequency_hz",
    "time_center_s", "snapshot_time_s", "snapshot_requested_time_s",
    "x_m", "probe_x_m", "x_center_m", "coordinates_m", "probe_indices",
    "upstream_probe_indices", "downstream_probe_indices", "upstream_x_m",
    "downstream_x_m", "wavenumber_rad_m", "frequency_indices", "triad_labels",
})

MASK_KEYS = frozenset({
    "valid_frequency", "spectral_valid_mask", "phase_valid_mask", "growth_valid_mask",
    "active_mask", "valid_time_mask", "valid_snapshot_mask",
})

PRODUCT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "spectral.psd": {
        "axes": {"frequency": ("frequency_hz",), "space": ("x_m",)},
        "value_keys": ("psd",),
        "value_dimensions": {"psd": ("frequency", "space")},
    },
    "spectral.coherence": {
        "axes": {"frequency": ("frequency_hz",), "space": ("upstream_x_m",)},
        "value_keys": ("coherence_squared", "cross_phase_rad"),
        "value_dimensions": {
            "coherence_squared": ("frequency", "space"),
            "cross_phase_rad": ("frequency", "space"),
        },
    },
    "spectral.probe_signals": {
        "axes": {
            "time": ("raw_time_s", "time_s"),
            "frequency": ("frequency_hz",),
            "space": ("x_m",),
        },
        "value_keys": ("raw_values", "processed_values", "amplitude"),
        "value_dimensions": {
            "raw_values": ("time", "space"),
            "processed_values": ("time", "space"),
            "amplitude": ("frequency", "space"),
        },
    },
    "transient.stft": {
        "axes": {
            "frequency": ("frequency_hz",), "time": ("relative_time_s",),
            "space": ("x_m",),
        },
        "value_keys": ("complex_stft",),
        "plotting_value_keys": ("probe_complex_stft",),
        "value_dimensions": {
            "complex_stft": ("frequency", "time"),
            "probe_complex_stft": ("frequency", "time", "space"),
        },
    },
    "transient.envelope": {
        "axes": {"time": ("time_s",), "space": ("x_m",)},
        "value_keys": ("filtered_signal", "envelope", "arrival_time_s"),
        "value_dimensions": {
            "filtered_signal": ("time", "space"),
            "envelope": ("time", "space"),
            "arrival_time_s": ("space",),
        },
    },
    "wave.wavenumber": {
        "schema_version": 2,
        "axes": {"frequency": ("frequency_hz",), "space": ("x_center_m",)},
        "value_keys": (
            "alpha_real_rad_m", "alpha_imag_rad_m", "alpha_real_ci95_rad_m",
            "alpha_imag_ci95_rad_m", "amplification_rate_per_m", "phase_speed_m_s",
            "coherence_squared", "phase_fit_r_squared", "amplitude_fit_r_squared",
            "spatial_alias_margin",
        ),
        "value_dimensions": {
            key: ("frequency", "space") for key in (
                "alpha_real_rad_m", "alpha_imag_rad_m", "alpha_real_ci95_rad_m",
                "alpha_imag_ci95_rad_m", "amplification_rate_per_m", "phase_speed_m_s",
                "coherence_squared", "phase_fit_r_squared", "amplitude_fit_r_squared",
                "spatial_alias_margin",
            )
        },
        "validity_by_value": {
            "alpha_real_rad_m": ("phase_valid_mask",),
            "alpha_imag_rad_m": ("growth_valid_mask",),
            "alpha_real_ci95_rad_m": ("phase_valid_mask",),
            "alpha_imag_ci95_rad_m": ("growth_valid_mask",),
            "phase_speed_m_s": ("phase_valid_mask",),
            "phase_fit_r_squared": ("phase_valid_mask",),
            "amplification_rate_per_m": ("growth_valid_mask",),
            "amplitude_fit_r_squared": ("growth_valid_mask",),
        },
        "value_units": {
            "alpha_real_rad_m": "rad/m", "alpha_imag_rad_m": "rad/m",
            "alpha_real_ci95_rad_m": "rad/m", "alpha_imag_ci95_rad_m": "rad/m",
            "amplification_rate_per_m": "1/m", "phase_speed_m_s": "m/s",
            "coherence_squared": "1",
            "phase_fit_r_squared": "1", "amplitude_fit_r_squared": "1",
            "spatial_alias_margin": "1",
        },
    },
    "wave.spatial_spectrum": {
        "axes": {
            "frequency": ("frequency_hz",),
            "space": ("x_center_m",),
            "edge_space": ("probe_x_m",),
        },
        "value_keys": (
            "spectral_power", "relative_spectral_power_db",
            "adjacent_coherence_squared",
        ),
        "value_dimensions": {
            "spectral_power": ("frequency", "space"),
            "relative_spectral_power_db": ("frequency", "space"),
            "adjacent_coherence_squared": ("frequency", "edge_space"),
        },
        "value_units": {
            "spectral_power": "variable^2", "relative_spectral_power_db": "dB",
            "adjacent_coherence_squared": "1",
        },
    },
    "wave.komega": {
        "axes": {
            "frequency": ("frequency_hz",),
            "wavenumber": ("wavenumber_rad_m",),
        },
        "value_keys": ("power", "amplitude"),
        "value_dimensions": {
            "power": ("frequency", "wavenumber"),
            "amplitude": ("frequency", "wavenumber"),
        },
        "value_units": {"power": "variable^2", "amplitude": "variable"},
    },
    "wave.temporal_wavenumber": {
        "axes": {
            "time": ("time_center_s",),
            "wavenumber": ("wavenumber_rad_m",),
        },
        "value_keys": (
            "band_power", "band_energy", "dominant_wavenumber_rad_m",
            "dominant_frequency_hz", "phase_speed_m_s", "wavelength_m",
        ),
        "plotting_value_keys": ("relative_band_power_db",),
        "value_dimensions": {
            "band_power": ("time", "wavenumber"),
            "band_energy": ("time",),
            "dominant_wavenumber_rad_m": ("time",),
            "dominant_frequency_hz": ("time",),
            "phase_speed_m_s": ("time",),
            "wavelength_m": ("time",),
        },
        "validity_by_value": {
            "band_power": ("active_mask",),
            "band_energy": ("active_mask",),
            "dominant_wavenumber_rad_m": ("valid_time_mask",),
            "dominant_frequency_hz": ("valid_time_mask",),
            "phase_speed_m_s": ("valid_time_mask",),
            "wavelength_m": ("valid_time_mask",),
        },
        "value_units": {
            "band_power": "variable^2",
            "band_energy": "variable^2",
            "dominant_wavenumber_rad_m": "rad/m",
            "dominant_frequency_hz": "Hz",
            "phase_speed_m_s": "m/s",
            "wavelength_m": "m",
        },
    },
    "wave.komega_snapshots": {
        "axes": {
            "time": ("snapshot_time_s",),
            "frequency": ("frequency_hz",),
            "wavenumber": ("wavenumber_rad_m",),
        },
        "value_keys": ("power",),
        "plotting_value_keys": ("relative_power_db",),
        "value_dimensions": {"power": ("time", "frequency", "wavenumber")},
        "validity_by_value": {"power": ("valid_snapshot_mask",)},
        "value_units": {"power": "variable^2"},
    },
    "pulse.source_spectrum": {
        "axes": {"time": ("time_s",), "frequency": ("frequency_hz",)},
        "value_keys": (
            "source_power_w_m", "physical_complex_j_m", "physical_spectrum_j_m",
            "ideal_spectrum_j_m", "processed_complex_w_m",
        ),
        "value_dimensions": {
            "source_power_w_m": ("time",),
            "physical_complex_j_m": ("frequency",),
            "physical_spectrum_j_m": ("frequency",),
            "ideal_spectrum_j_m": ("frequency",),
            "processed_complex_w_m": ("frequency",),
        },
        "value_units": {
            "source_power_w_m": "W/m",
            "physical_complex_j_m": "J/m",
            "physical_spectrum_j_m": "J/m",
            "ideal_spectrum_j_m": "J/m",
            "processed_complex_w_m": "W/m",
        },
    },
    "pulse.transfer": {
        "schema_version": 2,
        "axes": {"frequency": ("frequency_hz",), "space": ("probe_x_m",)},
        "value_keys": (
            "source_power_spectrum_w_m", "transfer", "transfer_magnitude",
            "transfer_phase_rad", "baseline",
        ),
        "value_dimensions": {
            "source_power_spectrum_w_m": ("frequency",),
            "transfer": ("frequency", "space"),
            "transfer_magnitude": ("frequency", "space"),
            "transfer_phase_rad": ("frequency", "space"),
            "baseline": ("space",),
        },
        "value_units": {"source_power_spectrum_w_m": "W/m"},
        "validity_by_value": {
            "transfer": ("valid_frequency",),
            "transfer_magnitude": ("valid_frequency",),
            "transfer_phase_rad": ("valid_frequency",),
        },
    },
    "spectral.confidence": {
        "json_value_paths": (
            "segment_count", "approximate_degrees_of_freedom",
            "frequency_resolution_hz", "record_duration_s",
        ),
    },
    "pulse.validity": {
        "json_value_paths": (
            "baseline_sample_count", "valid_frequency_fraction",
            "minimum_relative_source_amplitude",
        ),
    },
    "transient.group_velocity": {
        "json_value_paths": (
            "group_velocity_m_s", "confidence_interval_95_m_s",
            "slope_confidence_interval_95_s_m", "arrival_time_regression_r_squared",
            "slope_s_m", "slope_standard_error_s_m", "probe_count",
            "degrees_of_freedom", "t_multiplier_95", "minimum_snr_db",
            "resolved_edge_margin_s", "baseline_sample_count",
        ),
    },
    "nonlinear.triads": {
        "json_value_paths": (
            "selection.selected_frequency_hz", "fdr_adjusted_p_value",
        ),
    },
    "wave.komega_sensitivity": {
        "json_value_paths": (
            "frequency_band_hz", "full_record_sample_count", "block_sample_count",
            "window_sensitivity.compared_frequency_count",
            "window_sensitivity.median_absolute_ridge_difference_rad_m",
            "window_sensitivity.percentile_90_absolute_ridge_difference_rad_m",
            "window_sensitivity.maximum_absolute_ridge_difference_rad_m",
            "first_vs_second_half_block_sensitivity.compared_frequency_count",
            "first_vs_second_half_block_sensitivity.median_absolute_ridge_difference_rad_m",
            "first_vs_second_half_block_sensitivity.percentile_90_absolute_ridge_difference_rad_m",
            "first_vs_second_half_block_sensitivity.maximum_absolute_ridge_difference_rad_m",
        ),
    },
    "nonlinear.bicoherence": {
        "axes": {},
        "label_keys": ("triad_labels",),
        "value_keys": (
            "observed_bicoherence_squared", "surrogate_median_bicoherence_squared",
            "surrogate_95_bicoherence_squared", "empirical_p_value",
            "fdr_adjusted_p_value", "significant_fdr",
        ),
        "value_dimensions": {},
    },
    "modal.pod": {
        "axes": {"space": ("coordinates_m",)},
        "value_keys": (
            "modes", "temporal_coefficients", "singular_values", "energy_fraction", "mean",
        ),
        "value_dimensions": {
            "modes": ("space",),
            "temporal_coefficients": (),
            "singular_values": (),
            "energy_fraction": (),
            "mean": ("space",),
        },
    },
    "modal.spod": {
        "axes": {"frequency": ("frequency_hz",), "space": ("coordinates_m",)},
        "value_keys": ("eigenvalues", "modes"),
        "value_dimensions": {
            "eigenvalues": ("frequency",),
            "modes": ("frequency", None, "space"),
        },
    },
    "modal.dmd": {
        "axes": {"frequency": ("frequency_hz",), "space": ("coordinates_m",)},
        "value_keys": (
            "modes", "eigenvalues", "growth_rate_per_s", "amplitudes",
            "retained_condition_number",
        ),
        "value_dimensions": {
            "modes": ("space", "frequency"),
            "eigenvalues": ("frequency",),
            "growth_rate_per_s": ("frequency",),
            "amplitudes": ("frequency",),
            "retained_condition_number": (),
        },
    },
    "modal.sensitivity": {
        "axes": {},
        "value_keys": (
            "pod_subspace_min_cosine_to_first_window",
            "dmd_dominant_frequency_hz", "dmd_dominant_growth_rate_per_s",
            "dmd_retained_condition_number",
        ),
        "value_dimensions": {},
    },
}


def _definition_for(artifact_id: str) -> tuple[str, dict[str, Any]]:
    for product_type, definition in PRODUCT_DEFINITIONS.items():
        if artifact_id.endswith(product_type):
            return product_type, definition
    return artifact_id.split(".", 1)[-1] if "." in artifact_id else artifact_id, {}


def product_contract(
    artifact_id: str, path: Path | None = None, *, units: str | None = None,
) -> dict[str, Any]:
    """Return the versioned numerical contract for an artifact.

    Unknown product types intentionally have no value arrays. That prevents a
    future unregistered NPZ from being compared by accidentally treating every
    numeric array as a scientific measurement.
    """
    product_type, definition = _definition_for(artifact_id)
    axes = {
        axis: tuple(keys) for axis, keys in definition.get("axes", {}).items()
    }
    value_keys = tuple(definition.get("value_keys", ()))
    if path is not None and path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                value_keys = tuple(key for key in value_keys if key in keys)
                axis_keys = {
                    axis: tuple(key for key in candidates if key in keys)
                    for axis, candidates in axes.items()
                }
                if not definition:
                    axis_keys = {}
                axes = axis_keys
        except (OSError, ValueError, KeyError):
            pass
    return {
        "schema": "pelecpost.product",
        "schema_version": int(definition.get("schema_version", 1)),
        "product_type": product_type,
        "axes": axes,
        "coordinate_arrays": axes,
        "label_keys": tuple(definition.get("label_keys", ())),
        "value_keys": value_keys,
        "required_value_keys": tuple(definition.get("value_keys", ())),
        "json_value_paths": tuple(definition.get("json_value_paths", ())),
        "required_json_value_paths": tuple(definition.get("json_value_paths", ())),
        "value_dimensions": definition.get("value_dimensions", {}),
        "value_units": definition.get("value_units", {}),
        "plotting_value_keys": tuple(definition.get("plotting_value_keys", ())),
        "validity_by_value": definition.get("validity_by_value"),
        "units": units,
        "mask_keys": tuple(sorted(MASK_KEYS)),
        "validity_fields": tuple(sorted(MASK_KEYS)),
        "coordinate_keys": tuple(sorted(COORDINATE_KEYS)),
        "complex_value_semantics": (
            "magnitude_squared" if product_type == "transient.stft" else "real_difference"
        ),
        "supported_comparison_operations": COMPARISON_POLICIES,
        "supported_plotting_operations": (
            "line", "heatmap", "complex_magnitude", "mask_overlay",
        ),
    }
