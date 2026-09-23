"""Shared physical-coordinate alignment for numerical comparisons and figures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from pelecpost.config.models import ComparisonAlignment


def axis_arrays(archive: Any, contract: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    axis_keys = contract.get("axis_keys", contract.get("axes", {}))
    for axis, candidates in axis_keys.items():
        if isinstance(candidates, str):
            candidates = (candidates,)
        for key in candidates:
            if key in archive.files:
                values = np.asarray(archive[key])
                if values.ndim == 1 and values.size:
                    result[axis] = values.astype(float, copy=False)
                    break
    return result


def _axis_for_dimension(
    array: np.ndarray,
    dimension: int,
    axes: dict[str, np.ndarray],
    product_type: str,
    value_dimensions: dict[str, tuple[str | None, ...]] | None = None,
    value_key: str | None = None,
    field_coordinates: dict[str, tuple[str, ...]] | None = None,
) -> str | None:
    if field_coordinates is not None and value_key in field_coordinates:
        coordinate_keys = field_coordinates[value_key]
        if dimension >= len(coordinate_keys):
            return None
        coordinate_key = coordinate_keys[dimension]
        aliases = {
            "raw_time_s": "time",
            "time_s": "time",
            "relative_time_s": "time",
            "frequency_hz": "frequency",
            "wavenumber_rad_m": "wavenumber",
            "x_m": "space",
            "probe_x_m": "space",
            "x_center_m": "space",
        }
        axis = aliases.get(
            coordinate_key,
            coordinate_key if coordinate_key in axes else None,
        )
        if axis is None:
            raise ValueError(
                f"product {product_type!r} is missing field coordinate {coordinate_key!r}"
            )
        if len(axes[axis]) != array.shape[dimension]:
            raise ValueError(
                f"product {product_type} value {value_key!r} dimension {dimension} "
                f"has length {array.shape[dimension]}, but {coordinate_key} has "
                f"length {len(axes[axis])}"
            )
        return axis
    if value_dimensions is not None and value_key in value_dimensions:
        declared = value_dimensions[value_key]
        if dimension >= len(declared) or declared[dimension] is None:
            return None
        axis = declared[dimension]
        if axis not in axes:
            raise ValueError(
                f"product {product_type!r} is missing declared {axis!r} axis coordinates"
            )
        if len(axes[axis]) != array.shape[dimension]:
            raise ValueError(
                f"{product_type} value {value_key!r} dimension {dimension} has length "
                f"{array.shape[dimension]}, but {axis} has length {len(axes[axis])}"
            )
        return axis
    preferred = {
        "transient.stft": ("frequency", "time"),
        "wave.komega": ("frequency", "wavenumber"),
        "wave.wavenumber": ("frequency", "space"),
        "wave.spatial_spectrum": ("frequency", "space"),
    }.get(product_type, ("time", "frequency", "space", "wavenumber"))
    candidates = [
        name
        for name in preferred
        if name in axes and len(axes[name]) == array.shape[dimension]
    ]
    return candidates[0] if candidates else None


def _axis_tolerance(alignment: ComparisonAlignment, axis: str) -> float:
    return {
        "time": alignment.time_tolerance_s,
        "frequency": alignment.frequency_tolerance_hz,
        "space": alignment.space_tolerance_m,
        "wavenumber": alignment.wavenumber_tolerance_rad_m,
    }.get(axis, 0.0)


def _reorder_exact(
    array: np.ndarray,
    old: np.ndarray,
    target: np.ndarray,
    dimension: int,
    tolerance: float,
) -> np.ndarray:
    if len(old) != len(target):
        raise ValueError("strict comparison axes have different lengths")
    if np.allclose(old, target, rtol=0.0, atol=tolerance):
        return array
    order: list[int] = []
    for value in target:
        matches = np.flatnonzero(np.isclose(old, value, rtol=0.0, atol=tolerance))
        if len(matches) != 1:
            raise ValueError("strict comparison axes differ")
        order.append(int(matches[0]))
    return np.take(array, order, axis=dimension)


def _resample_axis(
    array: np.ndarray,
    old: np.ndarray,
    target: np.ndarray,
    dimension: int,
) -> np.ndarray:
    if len(old) < 2 or len(target) == 0:
        raise ValueError("comparison interpolation requires at least two axis samples")
    if np.any(np.diff(old) <= 0.0) or np.any(np.diff(target) <= 0.0):
        raise ValueError("comparison axes must be strictly increasing")
    interpolator = interp1d(
        old,
        array,
        axis=dimension,
        kind="linear",
        bounds_error=True,
        assume_sorted=True,
    )
    return np.asarray(interpolator(target))


def align_values(
    left: np.ndarray,
    right: np.ndarray,
    left_axes: dict[str, np.ndarray],
    right_axes: dict[str, np.ndarray],
    product_type: str,
    alignment: ComparisonAlignment,
    value_dimensions: dict[str, tuple[str | None, ...]] | None = None,
    value_key: str | None = None,
    field_coordinates: dict[str, tuple[str, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align two typed arrays and return the exact target coordinates used."""
    if left.ndim != right.ndim:
        raise ValueError(
            f"comparison arrays have incompatible ranks {left.ndim} and {right.ndim}"
        )
    details: dict[str, Any] = {}
    for dimension in range(left.ndim):
        left_axis = _axis_for_dimension(
            left,
            dimension,
            left_axes,
            product_type,
            value_dimensions,
            value_key,
            field_coordinates,
        )
        right_axis = _axis_for_dimension(
            right,
            dimension,
            right_axes,
            product_type,
            value_dimensions,
            value_key,
            field_coordinates,
        )
        if left_axis is None or right_axis is None:
            if left.shape[dimension] != right.shape[dimension]:
                raise ValueError("comparison arrays have incompatible unlabelled dimensions")
            continue
        if left_axis != right_axis:
            raise ValueError("comparison arrays assign different physical axes to a dimension")
        policy = getattr(alignment, left_axis, "strict")
        x_left, x_right = left_axes[left_axis], right_axes[right_axis]
        interpolated = False
        if policy == "strict":
            right = _reorder_exact(
                right,
                x_right,
                x_left,
                dimension,
                _axis_tolerance(alignment, left_axis),
            )
            target = x_left
            target_source = "baseline"
        elif policy == "interpolate_to_baseline":
            if x_left[0] < x_right[0] or x_left[-1] > x_right[-1]:
                raise ValueError(
                    f"{left_axis} interpolation target is outside comparison coverage"
                )
            right = _resample_axis(right, x_right, x_left, dimension)
            target = x_left
            target_source = "baseline"
            interpolated = not np.array_equal(x_left, x_right)
        elif policy == "interpolate_to_comparison":
            if x_right[0] < x_left[0] or x_right[-1] > x_left[-1]:
                raise ValueError(
                    f"{left_axis} interpolation target is outside baseline coverage"
                )
            left = _resample_axis(left, x_left, x_right, dimension)
            target = x_right
            target_source = "comparison"
            interpolated = not np.array_equal(x_left, x_right)
        elif policy == "intersection":
            lower, upper = max(x_left[0], x_right[0]), min(x_left[-1], x_right[-1])
            if lower > upper:
                raise ValueError(
                    f"comparison products have no intersecting {left_axis} grid"
                )
            left_target = x_left[(x_left >= lower) & (x_left <= upper)]
            right_target = x_right[(x_right >= lower) & (x_right <= upper)]
            target = np.unique(np.concatenate((left_target, right_target)))
            if target.size == 0:
                raise ValueError(
                    f"comparison products have no intersecting {left_axis} grid"
                )
            left = _resample_axis(left, x_left, target, dimension)
            right = _resample_axis(right, x_right, target, dimension)
            target_source = "common_union"
            interpolated = not (
                np.array_equal(target, x_left) and np.array_equal(target, x_right)
            )
        else:
            raise ValueError(f"unsupported {left_axis} alignment policy {policy!r}")
        details[left_axis] = {
            "policy": policy,
            "sample_count": len(target),
            "coordinate_values": np.asarray(target, dtype=float).tolist(),
            "target_source": target_source,
            "interpolated": interpolated,
        }
    return left, right, details


def align_frequency_columns(
    baseline: np.ndarray,
    comparison: np.ndarray,
    baseline_frequency_hz: np.ndarray,
    comparison_frequency_hz: np.ndarray,
    alignment: ComparisonAlignment | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Align frequency-by-column arrays for figures and diagnostic tables."""
    alignment = alignment or ComparisonAlignment()
    left, right, details = align_values(
        np.asarray(baseline),
        np.asarray(comparison),
        {"frequency": np.asarray(baseline_frequency_hz, dtype=float)},
        {"frequency": np.asarray(comparison_frequency_hz, dtype=float)},
        "spectral.probe_signals",
        alignment,
        {"value": ("frequency", None)},
        "value",
    )
    frequency_details = details["frequency"]
    return (
        left,
        right,
        np.asarray(frequency_details["coordinate_values"], dtype=float),
        frequency_details,
    )
