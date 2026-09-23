"""Axis-aware comparison of products from current or archived analyses."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import f as f_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pelecpost.config.models import CaseComparisonAnalysis, ComparisonAlignment
from pelecpost.runtime.context import WorkflowContext

from .comparison_alignment import _axis_for_dimension as _declared_axis_for_dimension
from .comparison_alignment import align_frequency_columns
from .comparison_alignment import align_values as _shared_align_values
from .comparison_alignment import axis_arrays as _shared_axis_arrays
from .comparison_figures import (
    _amplitude_unit,
    render_komega_amplitude_frequency_slices,
    render_komega_comparison,
    render_komega_frequency_slices,
    render_probe_panels,
)
from .executors import executor
from .products import product_contract
from .spectral import probe_phase_and_symmetry_diagnostic


@dataclass(frozen=True)
class _Product:
    label: str
    metadata: dict[str, Any]
    path: Path


def _artifact_map(context: WorkflowContext) -> dict[str, Any]:
    return {artifact.id: artifact for artifact in context.artifacts.artifacts}


def _context_dpi(context: WorkflowContext) -> int:
    project = getattr(context, "project", None)
    analyses_file = getattr(project, "analyses_file", None)
    presentation = getattr(analyses_file, "presentation", None)
    figure = getattr(presentation, "figure", None)
    return int(getattr(figure, "dpi", 300))


def _fk_ratio_provenance(
    analysis: CaseComparisonAnalysis,
    *,
    normalization: str,
    scale: str,
    band: tuple[float, float],
    alignment: dict[str, Any],
    statistics_path: Path | None = None,
) -> dict[str, Any]:
    """Describe the ratio represented by a figure or its statistics."""
    options = analysis.fft_ratio_plotting
    support_basis = (
        "normalized_own_peak" if normalization == "unit_l2" else "shared_raw_peak"
    )
    result: dict[str, Any] = {
        "ratio_direction": "comparison / baseline",
        "ratio_scale": scale,
        "ratio_units": "dB" if scale == "db" else "1",
        "ratio_normalization": normalization,
        "normalization_domain": "selected_frequency_band",
        "reference_frequency_band_hz": list(band),
        "minimum_relative_amplitude": options.minimum_relative_amplitude,
        "support_basis": support_basis,
        "support_rule": (
            "both normalized powers exceed the configured fraction squared of "
            "their own normalized peaks" if normalization == "unit_l2" else
            "both raw powers exceed the configured fraction squared of the "
            "shared selected-band peak"
        ),
        "ratio_limits": list(
            options.db_ratio_limits if scale == "db" else options.linear_ratio_limits
        ),
        "ratio_color_scale": (
            "linear" if scale == "db" else options.linear_ratio_color_scale
        ),
        "coordinates": {"frequency": "Hz", "wavenumber": "rad/m"},
        "alignment": alignment,
    }
    if statistics_path is not None and statistics_path.is_file():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        result.update({
            "accepted_count": statistics["accepted_count"],
            "masked_fraction": statistics["masked_fraction"],
            "alignment": statistics.get("alignment", alignment),
        })
    return result


def _archived_metadata(context: WorkflowContext, archive_id: str) -> dict[str, dict[str, Any]]:
    try:
        return context.plan.inventory.archived_metadata[archive_id]
    except KeyError as exc:
        raise ValueError(f"archived run {archive_id!r} was not inspected") from exc


def _archived_product_path(run_dir: Path, metadata: dict[str, Any], label: str) -> Path:
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"archived product {label!r} does not declare a path")
    path = (run_dir / raw_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"archived product {label!r} points outside its run directory") from exc
    if not path.is_file():
        raise ValueError(f"archived product {label!r} is missing at {path}")
    return path


def _resolve_product(context: WorkflowContext, reference: Any, product_id: str) -> _Product:
    full_id = f"{reference.analysis_id}.{product_id}"
    if reference.archived_run_id is None:
        try:
            artifact = _artifact_map(context)[full_id]
        except KeyError as exc:
            raise ValueError(
                f"current run does not contain product {full_id!r}; "
                "place the producing analysis before case_comparison"
            ) from exc
        return _Product(full_id, artifact.__dict__, context.run_dir / artifact.path)
    archive_id = reference.archived_run_id
    try:
        run_dir = Path(context.plan.inventory.archived_runs[archive_id])
        metadata = _archived_metadata(context, archive_id)[full_id]
    except KeyError as exc:
        raise ValueError(
            f"archived run {archive_id!r} does not contain product {full_id!r}"
        ) from exc
    return _Product(
        f"{archive_id}:{full_id}", metadata,
        _archived_product_path(run_dir, metadata, f"{archive_id}:{full_id}"),
    )


def _optional_source_audit(
    context: WorkflowContext, reference: Any,
) -> dict[str, Any] | None:
    """Load the pulse source audit when the referenced analysis produced one."""
    try:
        product = _resolve_product(context, reference, "pulse.source_audit")
    except ValueError:
        return None
    try:
        payload = json.loads(product.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source audit {product.label!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"source audit {product.label!r} is not a JSON object")
    return payload


def _resolve_reference_band(
    configured: tuple[float, float] | None,
    product_pair: tuple[_Product, _Product],
    *,
    product_label: str,
) -> tuple[float, float]:
    """Resolve an omitted band from the common positive native coordinate domain."""
    if configured is not None:
        return (float(configured[0]), float(configured[1]))
    domains: list[tuple[float, float]] = []
    for product in product_pair:
        if product.path.suffix.lower() != ".npz":
            continue
        with np.load(product.path, allow_pickle=False) as data:
            if "frequency_hz" not in data.files:
                continue
            frequency = np.asarray(data["frequency_hz"], dtype=float).ravel()
        positive = frequency[np.isfinite(frequency) & (frequency > 0.0)]
        if positive.size:
            domains.append((float(np.min(positive)), float(np.max(positive))))
    if not domains:
        raise ValueError(f"{product_label} has no positive frequency coordinate for band resolution")
    lo = max(item[0] for item in domains)
    hi = min(item[1] for item in domains)
    if hi <= lo:
        raise ValueError(f"{product_label} has no common positive frequency domain")
    return (lo, hi)


def _same_forcing_value(
    left: Any, right: Any,
) -> bool | None:
    if left is None or right is None:
        return None
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)
    except (TypeError, ValueError):
        return None


def _source_forcing_comparison(
    baseline: dict[str, Any] | None, comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None or comparison is None:
        return None
    requested_left = baseline.get("configured_requested_energy_j_m")
    requested_right = comparison.get("configured_requested_energy_j_m")
    deposited_left = baseline.get("measured_deposited_energy_j_m")
    deposited_right = comparison.get("measured_deposited_energy_j_m")
    return {
        "baseline": {
            "requested_energy_j_m": requested_left,
            "deposited_energy_j_m": deposited_left,
            "source_basis": baseline.get("source_basis"),
            "status": baseline.get("status"),
        },
        "comparison": {
            "requested_energy_j_m": requested_right,
            "deposited_energy_j_m": deposited_right,
            "source_basis": comparison.get("source_basis"),
            "status": comparison.get("status"),
        },
        "equal_requested_forcing": _same_forcing_value(requested_left, requested_right),
        "equal_measured_forcing": _same_forcing_value(deposited_left, deposited_right),
    }


def _normalised_preprocessing(metadata: dict[str, Any]) -> Any:
    preprocessing = metadata.get("provenance", {}).get("preprocessing")
    if preprocessing is None:
        return None
    if not isinstance(preprocessing, dict):
        return preprocessing
    result = {
        key: value for key, value in preprocessing.items()
        if key not in {
            "probe_set_id", "probe_indices", "id", "enabled",
            "probe_plotting", "record_start_time_s", "end_time_s",
        }
    }
    temporal = result.get("temporal_wavenumber")
    if isinstance(temporal, dict):
        temporal = dict(temporal)
        temporal.pop("snapshot_times_s", None)
        temporal.pop("display_floor_db", None)
        result["temporal_wavenumber"] = temporal
    return result


def _read_json_product_schema_version(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON product {path} must contain an object")
    value = payload.get("schema_version")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_product_pair(left: _Product, right: _Product) -> dict[str, Any]:
    if left.metadata.get("kind") not in {"array", "json"}:
        raise ValueError(f"product {left.label!r} is not numerical")
    if left.metadata.get("kind") != right.metadata.get("kind"):
        raise ValueError("comparison products have different artifact kinds")
    if left.path.suffix.lower() != right.path.suffix.lower():
        raise ValueError("comparison products have different storage formats")
    if left.metadata.get("variable") != right.metadata.get("variable"):
        raise ValueError("comparison products have different variables")
    if left.metadata.get("units") != right.metadata.get("units"):
        raise ValueError("comparison products have different units")
    left_contract = left.metadata.get("provenance", {}).get("product_contract")
    right_contract = right.metadata.get("provenance", {}).get("product_contract")
    left_contract = left_contract or product_contract(
        left.label, left.path, units=left.metadata.get("units")
    )
    right_contract = right_contract or product_contract(
        right.label, right.path, units=right.metadata.get("units")
    )
    if left_contract.get("product_type") != right_contract.get("product_type"):
        raise ValueError(
            f"comparison products have different product types: "
            f"{left_contract.get('product_type')!r} != {right_contract.get('product_type')!r}"
        )
    if left_contract.get("schema_version", 1) != right_contract.get("schema_version", 1):
        raise ValueError(
            "comparison products use incompatible product-contract versions: "
            f"{left_contract.get('product_type')!r} version "
            f"{left_contract.get('schema_version', 1)} vs "
            f"{right_contract.get('schema_version', 1)}; migrate the older artifact "
            "or regenerate both products with the same contract version"
        )
    if left_contract.get("product_type") == "spectral.confidence":
        for product, contract in ((left, left_contract), (right, right_contract)):
            payload_version = _read_json_product_schema_version(product.path)
            if payload_version != contract["schema_version"]:
                raise ValueError(
                    f"{product.label!r} confidence payload schema version "
                    f"{payload_version} disagrees with product contract version "
                    f"{contract['schema_version']}"
                )
    left_basis = left.metadata.get("provenance", {}).get("source_basis")
    right_basis = right.metadata.get("provenance", {}).get("source_basis")
    if left_basis is not None and right_basis is not None and left_basis != right_basis:
        raise ValueError(
            "pulse products use incompatible source bases: "
            f"{left_basis!r} vs {right_basis!r}; migrate or regenerate both products "
            "with measured or modeled source data"
        )
    if left.path.suffix.lower() == ".npz" and not left_contract.get("value_keys"):
        raise ValueError(f"product {left.label!r} has no registered typed value arrays")
    if right.path.suffix.lower() == ".npz" and not right_contract.get("value_keys"):
        raise ValueError(f"product {right.label!r} has no registered typed value arrays")
    if left.path.suffix.lower() == ".json" and not left_contract.get("json_value_paths"):
        raise ValueError(f"product {left.label!r} has no registered typed JSON values")
    if right.path.suffix.lower() == ".json" and not right_contract.get("json_value_paths"):
        raise ValueError(f"product {right.label!r} has no registered typed JSON values")
    left_required = set(left_contract.get("required_value_keys", ()))
    right_required = set(right_contract.get("required_value_keys", ()))
    left_missing = left_required - set(left_contract.get("value_keys", ()))
    right_missing = right_required - set(right_contract.get("value_keys", ()))
    if left_missing or right_missing:
        raise ValueError(
            "product value arrays are missing: "
            f"baseline={sorted(left_missing)}, comparison={sorted(right_missing)}"
        )
    if set(left_contract.get("value_keys", ())) != set(right_contract.get("value_keys", ())):
        raise ValueError("comparison products declare different value arrays")
    if set(left_contract.get("json_value_paths", ())) != set(
        right_contract.get("json_value_paths", ())
    ):
        raise ValueError("comparison products declare different JSON value paths")
    if set(left_contract.get("label_keys", ())) != set(right_contract.get("label_keys", ())):
        raise ValueError("comparison products declare different categorical labels")
    left_units = left_contract.get("value_units", {})
    right_units = right_contract.get("value_units", {})
    if left_units and right_units and left_units != right_units:
        raise ValueError("comparison products declare different per-value units")
    label_keys = tuple(left_contract.get("label_keys", ()))
    if label_keys and left.path.suffix.lower() == ".npz":
        with np.load(left.path, allow_pickle=False) as first, np.load(
            right.path, allow_pickle=False
        ) as second:
            for key in label_keys:
                if key not in first.files or key not in second.files:
                    raise ValueError(f"comparison products are missing label array {key!r}")
                if not np.array_equal(first[key], second[key]):
                    raise ValueError(f"comparison products have different {key} labels")
    left_preprocessing = _normalised_preprocessing(left.metadata)
    right_preprocessing = _normalised_preprocessing(right.metadata)
    if left_preprocessing is None or right_preprocessing is None:
        raise ValueError("comparison products must declare preprocessing provenance")
    if left_preprocessing != right_preprocessing:
        raise ValueError("comparison products use different preprocessing settings")
    return {"left": left_contract, "right": right_contract}


def _axis_arrays(archive: Any, contract: dict[str, Any]) -> dict[str, np.ndarray]:
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
        # ``axes`` contains resolved arrays keyed by logical axis name.  The
        # contract's concrete coordinate key is therefore mapped explicitly
        # to its logical axis rather than searched as if the array were a
        # tuple of candidate field names.
        axis_aliases = {
            "raw_time_s": "time", "time_s": "time", "relative_time_s": "time",
            "frequency_hz": "frequency", "wavenumber_rad_m": "wavenumber",
            "x_m": "space", "probe_x_m": "space", "x_center_m": "space",
        }
        axis = axis_aliases.get(coordinate_key, coordinate_key if coordinate_key in axes else None)
        if axis is None:
            raise ValueError(
                f"product {product_type!r} is missing field coordinate {coordinate_key!r}"
            )
        if len(axes[axis]) != array.shape[dimension]:
            raise ValueError(
                f"product {product_type} value {value_key!r} dimension {dimension} "
                f"has length {array.shape[dimension]}, but {coordinate_key} has length {len(axes[axis])}"
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
        name for name in preferred
        if name in axes and len(axes[name]) == array.shape[dimension]
    ]
    return candidates[0] if candidates else None


def _reorder_exact(
    array: np.ndarray, old: np.ndarray, target: np.ndarray,
    dimension: int, tolerance: float,
) -> np.ndarray:
    if len(old) != len(target):
        raise ValueError("strict comparison axes have different lengths")
    if np.allclose(old, target, rtol=0.0, atol=tolerance):
        return array
    order = []
    for value in target:
        matches = np.flatnonzero(np.isclose(old, value, rtol=0.0, atol=tolerance))
        if len(matches) != 1:
            raise ValueError("strict comparison axes differ")
        order.append(int(matches[0]))
    return np.take(array, order, axis=dimension)


def _axis_tolerance(alignment: ComparisonAlignment, axis: str) -> float:
    return {
        "time": alignment.time_tolerance_s,
        "frequency": alignment.frequency_tolerance_hz,
        "space": alignment.space_tolerance_m,
        "wavenumber": alignment.wavenumber_tolerance_rad_m,
    }.get(axis, 0.0)


def _resample_axis(
    array: np.ndarray, old: np.ndarray, target: np.ndarray, dimension: int,
) -> np.ndarray:
    if len(old) < 2 or len(target) == 0:
        raise ValueError("comparison interpolation requires at least two axis samples")
    if np.any(np.diff(old) <= 0.0) or np.any(np.diff(target) <= 0.0):
        raise ValueError("comparison axes must be strictly increasing")
    interpolator = interp1d(
        old, array, axis=dimension, kind="linear", bounds_error=True,
        assume_sorted=True,
    )
    return np.asarray(interpolator(target))


def _align_values(
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
    if left.ndim != right.ndim:
        raise ValueError(f"comparison arrays have incompatible ranks {left.ndim} and {right.ndim}")
    details: dict[str, Any] = {}
    for dimension in range(left.ndim):
        left_axis = _axis_for_dimension(
            left, dimension, left_axes, product_type, value_dimensions, value_key,
            field_coordinates,
        )
        right_axis = _axis_for_dimension(
            right, dimension, right_axes, product_type, value_dimensions, value_key,
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
        if policy == "strict":
            right = _reorder_exact(
                right, x_right, x_left, dimension, _axis_tolerance(alignment, left_axis)
            )
            target = x_left
        elif policy == "interpolate_to_baseline":
            if x_left[0] < x_right[0] or x_left[-1] > x_right[-1]:
                raise ValueError(f"{left_axis} interpolation target is outside comparison coverage")
            right = _resample_axis(right, x_right, x_left, dimension)
            target = x_left
        elif policy == "interpolate_to_comparison":
            if x_right[0] < x_left[0] or x_right[-1] > x_left[-1]:
                raise ValueError(f"{left_axis} interpolation target is outside baseline coverage")
            left = _resample_axis(left, x_left, x_right, dimension)
            target = x_right
        else:
            lower, upper = max(x_left[0], x_right[0]), min(x_left[-1], x_right[-1])
            if lower > upper:
                raise ValueError(f"comparison products have no intersecting {left_axis} grid")
            left_target = x_left[(x_left >= lower) & (x_left <= upper)]
            right_target = x_right[(x_right >= lower) & (x_right <= upper)]
            target = np.unique(np.concatenate((left_target, right_target)))
            if target.size == 0:
                raise ValueError(f"comparison products have no intersecting {left_axis} grid")
            left = _resample_axis(left, x_left, target, dimension)
            right = _resample_axis(right, x_right, target, dimension)
        details[left_axis] = {
            "policy": policy,
            "sample_count": len(target),
            "coordinate_values": np.asarray(target, dtype=float).tolist(),
        }
    return left, right, details


def _validity_for_value(
    archive: Any,
    contract: dict[str, Any],
    value: np.ndarray,
    axes: dict[str, np.ndarray],
    product_type: str,
    value_key: str,
) -> np.ndarray:
    """Broadcast declared validity masks onto one typed value array."""
    valid = np.ones(value.shape, dtype=bool)
    mask_axes = {
        "valid_frequency": "frequency",
        "spectral_valid_mask": None,
        "phase_valid_mask": None,
        "growth_valid_mask": None,
    }
    validity_by_value = contract.get("validity_by_value")
    if validity_by_value is None:
        # A product-wide mask inventory is descriptive metadata, not a claim
        # that every archive must contain every possible mask.  Only an
        # explicit value-to-mask declaration makes a mask required.
        validity_fields = ()
    else:
        validity_fields = validity_by_value.get(value_key, ())
    for key in validity_fields:
        if key not in archive.files:
            raise ValueError(
                f"product {product_type!r} is missing required validity field {key!r} "
                f"for value {value_key!r}"
            )
        raw = np.asarray(archive[key])
        raw_valid = (
            np.isfinite(raw) & (raw != 0)
            if raw.dtype.kind in "fc"
            else raw.astype(bool, copy=False)
        )
        if raw.shape == value.shape:
            expanded = raw_valid
        elif raw.ndim == 1:
            declared_axis = mask_axes.get(key)
            dimensions = []
            for dimension in range(value.ndim):
                axis = _declared_axis_for_dimension(
                    value, dimension, axes, product_type,
                    contract.get("value_dimensions"), value_key,
                    contract.get("field_coordinates"),
                )
                if len(raw) == value.shape[dimension] and (
                    axis == declared_axis if declared_axis is not None else axis is not None
                ):
                    dimensions.append(dimension)
            if len(dimensions) != 1:
                raise ValueError(
                    f"validity field {key!r} cannot be unambiguously aligned to "
                    f"value {value_key!r} with shape {value.shape}"
                )
            shape = [1] * value.ndim
            shape[dimensions[0]] = len(raw)
            expanded = np.broadcast_to(raw_valid.reshape(shape), value.shape)
        else:
            try:
                expanded = np.broadcast_to(raw_valid, value.shape)
            except ValueError as exc:
                raise ValueError(
                    f"validity field {key!r} cannot be broadcast to value shape {value.shape}"
                ) from exc
        valid &= expanded
    return valid


def _numeric_metric(
    name: str, first: np.ndarray, second: np.ndarray, valid: np.ndarray | None = None,
    *, complex_semantics: str = "real_difference",
) -> dict[str, Any] | None:
    if first.shape != second.shape:
        raise ValueError(
            f"comparable quantity {name!r} has incompatible shapes {first.shape} and {second.shape}"
        )
    if first.dtype.kind not in "iufc" or second.dtype.kind not in "iufc":
        return None
    if (
        (np.iscomplexobj(first) or np.iscomplexobj(second))
        and complex_semantics == "magnitude_squared"
    ):
        first = np.abs(first) ** 2
        second = np.abs(second) ** 2
    difference = np.asarray(first - second)
    finite = np.isfinite(difference) & np.isfinite(first) & np.isfinite(second)
    if valid is not None:
        finite &= valid
    if not np.any(finite):
        return None
    reference = np.asarray(first)[finite]
    l2 = float(np.linalg.norm(difference[finite]))
    return {
        "quantity": name,
        "shape": list(first.shape),
        "l2_difference": l2,
        "linf_difference": float(np.max(np.abs(difference[finite]))),
        "relative_l2_difference": (
            float(l2 / np.linalg.norm(reference))
            if np.linalg.norm(reference) > 0.0
            else None
        ),
        "relative_l2_status": "available" if np.linalg.norm(reference) > 0.0 else "unavailable_zero_baseline",
        "finite_count": int(np.count_nonzero(finite)),
    }


def _json_path(payload: Any, path: str) -> Any:
    value = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"JSON product is missing typed value path {path!r}")
        value = value[component]
    return value


def _json_metrics(
    left: Path, right: Path, value_paths: tuple[str, ...],
) -> list[dict[str, Any]]:
    first, second = json.loads(left.read_text()), json.loads(right.read_text())
    metrics: list[dict[str, Any]] = []

    def visit(path: str, a: Any, b: Any) -> None:
        numeric_a = isinstance(a, (int, float)) and not isinstance(a, bool)
        numeric_b = isinstance(b, (int, float)) and not isinstance(b, bool)
        if numeric_a or numeric_b:
            if not numeric_a or not numeric_b:
                raise ValueError(f"JSON products differ at {path or '<root>'}")
            metric = _numeric_metric(
                path, np.asarray([a], dtype=float), np.asarray([b], dtype=float)
            )
            if metric is not None:
                metrics.append(metric)
        elif isinstance(a, list) or isinstance(b, list):
            if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
                raise ValueError(f"JSON products differ at {path or '<root>'}")
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(f"{path}[{index}]", item_a, item_b)
        elif isinstance(a, dict) or isinstance(b, dict):
            if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
                raise ValueError(f"JSON products differ at {path or '<root>'}")
            for key in sorted(a):
                visit(f"{path}.{key}" if path else str(key), a[key], b[key])

    for value_path in value_paths:
        visit(value_path, _json_path(first, value_path), _json_path(second, value_path))
    if not metrics:
        raise ValueError("JSON products share no numeric values")
    return metrics


def _array_metrics(
    left: _Product, right: _Product, alignment: ComparisonAlignment,
    frequency_band_hz: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_pair = _validate_product_pair(left, right)
    contract = contract_pair["left"]
    metrics: list[dict[str, Any]] = []
    alignment_details: dict[str, Any] = {}
    with (
        np.load(left.path, allow_pickle=False) as first,
        np.load(right.path, allow_pickle=False) as second,
    ):
        left_axes = _shared_axis_arrays(first, contract_pair["left"])
        right_axes = _shared_axis_arrays(second, contract_pair["right"])
        keys = tuple(contract.get("value_keys", ()))
        missing_left = [key for key in keys if key not in first.files]
        missing_right = [key for key in keys if key not in second.files]
        if missing_left or missing_right:
            raise ValueError(
                f"product value arrays missing: baseline={missing_left}, comparison={missing_right}"
            )
        for key in keys:
            a, b = np.asarray(first[key]), np.asarray(second[key])
            left_valid = _validity_for_value(
                first, contract_pair["left"], a, left_axes,
                contract.get("product_type", ""), key,
            )
            right_valid = _validity_for_value(
                second, contract_pair["right"], b, right_axes,
                contract.get("product_type", ""), key,
            )
            a, b, details = _shared_align_values(
                a, b, left_axes, right_axes,
                contract.get("product_type", ""), alignment,
                contract.get("value_dimensions", {}), key,
                contract.get("field_coordinates", {}),
            )
            alignment_details[key] = details
            left_valid, right_valid, _ = _shared_align_values(
                left_valid.astype(float), right_valid.astype(float),
                left_axes, right_axes, contract.get("product_type", ""), alignment,
                contract.get("value_dimensions", {}), key,
                contract.get("field_coordinates", {}),
            )
            # Interpolated masks are validity predicates, not measurements:
            # a point is valid only when interpolation stayed exactly on the
            # valid side of every contributing source sample.  A plain bool
            # cast would incorrectly turn fractional 0/1 mask values true.
            left_valid = np.isclose(left_valid, 1.0, rtol=0.0, atol=1.0e-12)
            right_valid = np.isclose(right_valid, 1.0, rtol=0.0, atol=1.0e-12)
            valid_both = left_valid & right_valid
            declared_dimensions = contract.get("value_dimensions", {}).get(key)
            has_frequency_dimension = "frequency" in (declared_dimensions or ())
            if frequency_band_hz is not None and has_frequency_dimension:
                aligned_frequency = alignment_details[key]["frequency"].get("coordinate_values")
                frequency = np.asarray(
                    aligned_frequency if aligned_frequency is not None else left_axes["frequency"],
                    dtype=float,
                )
                frequency_mask = (frequency >= frequency_band_hz[0]) & (frequency <= frequency_band_hz[1])
                dimensions = [
                    index for index, axis_name in enumerate(declared_dimensions or ())
                    if axis_name == "frequency"
                ]
                if len(dimensions) == 1 and a.shape[dimensions[0]] == len(frequency):
                    shape = [1] * a.ndim
                    shape[dimensions[0]] = len(frequency)
                    valid_both &= frequency_mask.reshape(shape)
                elif dimensions:
                    raise ValueError(
                        f"product {contract.get('product_type', '')!r} value {key!r} "
                        "frequency dimension is not aligned with its declared coordinate"
                    )
            alignment_details[key].update({
                "masked_count_baseline": int(
                    np.size(left_valid) - np.count_nonzero(left_valid)
                ),
                "masked_count_comparison": int(
                    np.size(right_valid) - np.count_nonzero(right_valid)
                ),
                "valid_count_used": int(np.count_nonzero(valid_both)),
                "validity_fields": sorted(
                    set(
                        (contract_pair["left"].get("validity_by_value") or {}).get(key, ())
                    ) & set(first.files)
                    | set(
                        (contract_pair["right"].get("validity_by_value") or {}).get(key, ())
                    ) & set(second.files)
                ),
            })
            if frequency_band_hz is not None:
                alignment_details[key]["frequency_band_hz"] = list(frequency_band_hz)
            metric = _numeric_metric(
                key, a, b, valid_both,
                complex_semantics=contract.get("complex_value_semantics", "real_difference"),
            )
            if metric is not None:
                metrics.append(metric)
    if not metrics:
        raise ValueError(f"products {left.label!r} and {right.label!r} share no finite values")
    return metrics, alignment_details


def _render_summary(context: WorkflowContext, metrics: list[dict[str, Any]]) -> Path:
    path = context.figure_dir / "comparison_relative_l2.png"
    dpi = _context_dpi(context)
    fig, axis = plt.subplots(figsize=(11, max(4, 0.4 * len(metrics))))
    usable = [item for item in metrics if item.get("relative_l2_difference") is not None]
    unavailable = [item for item in metrics if item.get("relative_l2_difference") is None]
    labels = [f"{item.get('product_id', 'product').split('.')[-1]} · {item['quantity']}"
              for item in usable]
    values = [100.0 * item["relative_l2_difference"] for item in usable]
    if unavailable:
        labels.extend([f"{item.get('product_id', 'product').split('.')[-1]} · {item['quantity']} (unavailable)"
                       for item in unavailable])
        values.extend([0.0] * len(unavailable))
    axis.barh(np.arange(len(values)), values)
    for index, value in enumerate(values):
        text = " unavailable" if index >= len(usable) else f" {value:.2g}%"
        axis.text(value, index, text, va="center", ha="left")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel(r"100 ||G − A||₂ / ||A||₂ [% of baseline]")
    axis.set_title("Baseline-to-comparison relative L2 differences")
    axis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _fft_ratio_peak_table(
    left: _Product,
    right: _Product,
    band: tuple[float, float],
    minimum_relative_amplitude: float,
    alignment: ComparisonAlignment | None = None,
) -> list[dict[str, Any]]:
    """Return accepted absolute-ratio maxima with numerator/denominator context."""
    rows: list[dict[str, Any]] = []
    with np.load(left.path, allow_pickle=False) as first, np.load(right.path, allow_pickle=False) as second:
        a, b, frequency, alignment_details = align_frequency_columns(
            np.maximum(np.asarray(first["amplitude"], dtype=float), 0.0),
            np.maximum(np.asarray(second["amplitude"], dtype=float), 0.0),
            np.asarray(first["frequency_hz"], dtype=float),
            np.asarray(second["frequency_hz"], dtype=float),
            alignment,
        )
        use = (frequency > 0.0) & (frequency >= band[0]) & (frequency <= band[1])
        for column in range(a.shape[1]):
            aa, bb = a[use, column], b[use, column]
            ff = frequency[use]
            peak_a = float(np.max(aa, initial=0.0))
            peak_b = float(np.max(bb, initial=0.0))
            valid = (aa >= peak_a * minimum_relative_amplitude) & (bb >= peak_b * minimum_relative_amplitude)
            if not np.any(valid):
                continue
            ratios = np.full(ff.shape, np.nan)
            ratios[valid] = bb[valid] / np.maximum(aa[valid], 1.0e-300)
            index = int(np.nanargmax(ratios))
            probe = int(first["probe_indices"][column]) if "probe_indices" in first else column
            rows.append({
                "probe_index": probe,
                "frequency_hz": float(ff[index]),
                "gaussian_amplitude": float(bb[index]),
                "asymmetric_amplitude": float(aa[index]),
                "absolute_amplitude_ratio_gaussian_over_asymmetric": float(ratios[index]),
                "support_rule": f"both amplitudes >= {minimum_relative_amplitude:.3g} of each own peak",
                "frequency_alignment": alignment_details,
            })
    return rows


def _welch_metadata(product: _Product) -> dict[str, Any] | None:
    value = product.metadata.get("provenance", {}).get("welch_confidence")
    return value if isinstance(value, dict) else None


def _optional_float_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _psd_ratio_confidence(
    left: _Product,
    right: _Product,
    alignment: ComparisonAlignment,
    band: tuple[float, float],
    minimum_relative_amplitude: float,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Return a pointwise Welch PSD ratio interval and plotting arrays."""
    payload: dict[str, Any] = {
        "schema": "pelecpost.psd_ratio_confidence",
        "schema_version": 1,
        "status": "unavailable",
        "ratio_direction": "comparison_over_baseline",
        "confidence_level": confidence_level,
        "frequency_band_hz": list(band),
        "assumptions": [
            "approximately stationary Gaussian process",
            "independent baseline and comparison spectral estimates",
            "pointwise intervals; not a simultaneous confidence band",
        ],
    }
    baseline_meta, comparison_meta = _welch_metadata(left), _welch_metadata(right)
    if baseline_meta is None or comparison_meta is None:
        payload["reason"] = "missing effective Welch degrees-of-freedom metadata"
        return payload, None
    baseline_segments = int(baseline_meta.get("segment_count", 0))
    comparison_segments = int(comparison_meta.get("segment_count", 0))
    if baseline_segments < 2 or comparison_segments < 2:
        payload["reason"] = "at least two Welch segments are required in both products"
        return payload, None
    baseline_dof = float(baseline_meta.get("effective_degrees_of_freedom", 0.0))
    comparison_dof = float(comparison_meta.get("effective_degrees_of_freedom", 0.0))
    if baseline_dof <= 0.0 or comparison_dof <= 0.0:
        payload["reason"] = "invalid effective Welch degrees of freedom"
        return payload, None
    with np.load(left.path, allow_pickle=False) as first, np.load(
        right.path, allow_pickle=False
    ) as second:
        if not np.array_equal(first["probe_indices"], second["probe_indices"]):
            payload["reason"] = "baseline and comparison probe identities differ"
            return payload, None
        a, b, frequency, frequency_alignment = align_frequency_columns(
            np.maximum(np.asarray(first["psd"], dtype=float), 0.0),
            np.maximum(np.asarray(second["psd"], dtype=float), 0.0),
            np.asarray(first["frequency_hz"], dtype=float),
            np.asarray(second["frequency_hz"], dtype=float),
            alignment,
        )
        probe_indices = np.asarray(first["probe_indices"], dtype=int)
        x_m = np.asarray(first["x_m"], dtype=float)
    payload["frequency_alignment"] = frequency_alignment
    if frequency_alignment.get("interpolated"):
        payload["reason"] = (
            "PSD confidence intervals are unavailable after frequency interpolation"
        )
        return payload, None
    use = (
        (frequency > 0.0)
        & (frequency >= band[0])
        & (frequency <= band[1])
    )
    if not np.any(use):
        payload["reason"] = "selected frequency band contains no positive native bins"
        return payload, None
    frequency, a, b = frequency[use], a[use], b[use]
    alpha = 1.0 - confidence_level
    lower_factor = float(
        f_distribution.ppf(alpha / 2.0, comparison_dof, baseline_dof)
    )
    upper_factor = float(
        f_distribution.ppf(1.0 - alpha / 2.0, comparison_dof, baseline_dof)
    )
    ratio = np.full_like(a, np.nan, dtype=float)
    lower = np.full_like(a, np.nan, dtype=float)
    upper = np.full_like(a, np.nan, dtype=float)
    support = np.zeros_like(a, dtype=bool)
    for column in range(a.shape[1]):
        peak_a = float(np.max(a[:, column], initial=0.0))
        peak_b = float(np.max(b[:, column], initial=0.0))
        supported = (
            (a[:, column] >= peak_a * minimum_relative_amplitude**2)
            & (b[:, column] >= peak_b * minimum_relative_amplitude**2)
            & (a[:, column] > 0.0)
            & (b[:, column] > 0.0)
        )
        support[:, column] = supported
        ratio[supported, column] = b[supported, column] / a[supported, column]
        lower[supported, column] = ratio[supported, column] / upper_factor
        upper[supported, column] = ratio[supported, column] / lower_factor
    payload.update({
        "status": "available",
        "baseline_effective_degrees_of_freedom": baseline_dof,
        "comparison_effective_degrees_of_freedom": comparison_dof,
        "minimum_relative_amplitude": minimum_relative_amplitude,
        "support_rule": "both PSDs exceed the configured fraction-squared of their own band peaks",
        "frequency_hz": frequency.tolist(),
        "probes": [
            {
                "probe_index": int(probe_indices[column]),
                "x_m": float(x_m[column]),
                "ratio": _optional_float_list(ratio[:, column]),
                "confidence_lower": _optional_float_list(lower[:, column]),
                "confidence_upper": _optional_float_list(upper[:, column]),
                "support": support[:, column].tolist(),
            }
            for column in range(a.shape[1])
        ],
        "interpretation": (
            "Pointwise Welch-model confidence intervals for comparison/baseline PSD; "
            "not a simultaneous band or evidence of physical significance."
        ),
    })
    return payload, {
        "frequency_hz": frequency,
        "ratio": ratio,
        "lower": lower,
        "upper": upper,
        "probe_indices": probe_indices,
        "x_m": x_m,
    }


def _render_psd_ratio_confidence(
    arrays: dict[str, np.ndarray],
    destination: Path,
    confidence_level: float,
    dpi: int,
) -> Path:
    frequency = arrays["frequency_hz"]
    ratio, lower, upper = arrays["ratio"], arrays["lower"], arrays["upper"]
    count = ratio.shape[1]
    rows = (count + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(12, max(3.5, 2.8 * rows)), squeeze=False)
    for column, axis in enumerate(axes.flat[:count]):
        finite = np.isfinite(ratio[:, column])
        axis.plot(frequency, ratio[:, column], color="#6a3d9a", linewidth=1.5)
        axis.fill_between(
            frequency,
            lower[:, column],
            upper[:, column],
            where=finite,
            color="#6a3d9a",
            alpha=0.2,
            linewidth=0.0,
        )
        axis.axhline(1.0, color="0.5", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(
            f"Probe {int(arrays['probe_indices'][column])} · "
            f"x={arrays['x_m'][column] * 100.0:.3f} cm"
        )
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel("PSD ratio comparison/baseline [−]")
        axis.grid(True, alpha=0.2)
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.suptitle(
        f"Welch PSD ratio with {100.0 * confidence_level:.1f}% pointwise intervals"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def _render_product_overlay(context: WorkflowContext, left: _Product, right: _Product, analysis: CaseComparisonAnalysis | None = None) -> Path:
    path = context.figure_dir / "comparison_overlay.png"
    if left.label.endswith(".spectral.probe_signals"):
        rendered = render_probe_panels(
            left.path, right.path, left.label, right.label,
            str(left.metadata.get("variable") or "signal"), path, kind="raw",
            dpi=_context_dpi(context),
            scale_policy=analysis.fft_ratio_plotting.scale_policy if analysis else "independent",
            probe_groups=tuple(tuple(group.probe_indices) for group in analysis.fft_ratio_plotting.probe_groups) if analysis else (),
            alignment=analysis.alignment if analysis else ComparisonAlignment(),
        )
        if rendered is not None:
            return rendered
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if left.path.suffix.lower() != ".npz":
        for axis in axes:
            axis.axis("off")
        axes[0].text(0.5, 0.5, "JSON product; see comparison metrics", ha="center", va="center")
    else:
        with np.load(left.path, allow_pickle=False) as first, np.load(right.path, allow_pickle=False) as second:
            contract = product_contract(left.label, left.path)
            key = next(iter(contract.get("value_keys", ())), None)
            if key is None:
                axes[0].text(0.5, 0.5, "No numeric product values", ha="center", va="center")
                axes[1].axis("off")
            else:
                for axis, values, title in (
                    (axes[0], np.asarray(first[key]), left.label),
                    (axes[1], np.asarray(second[key]), right.label),
                ):
                    if values.ndim == 1:
                        axis.plot(values)
                    else:
                        image = np.abs(values) if np.iscomplexobj(values) else values
                        axis.imshow(image, aspect="auto")
                    axis.set_title(f"{title}: {key}")
                    axis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=_context_dpi(context), bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), dpi=_context_dpi(context), bbox_inches="tight")
    plt.close(fig)
    return path


@executor("case_comparison")
def run_case_comparison(context: WorkflowContext) -> None:
    analysis = cast(CaseComparisonAnalysis, context.analysis)
    all_metrics: list[dict[str, Any]] = []
    alignment_records: dict[str, Any] = {}
    preprocessing_records: dict[str, Any] = {}
    first_pair: tuple[_Product, _Product] | None = None
    product_pairs: dict[str, tuple[_Product, _Product]] = {}
    resolved_fft_band: tuple[float, float] | None = None
    resolved_wave_band: tuple[float, float] | None = None
    for product_id in analysis.product_ids:
        left = _resolve_product(context, analysis.baseline, product_id)
        right = _resolve_product(context, analysis.comparison, product_id)
        contract_pair = _validate_product_pair(left, right)
        product_pairs[product_id] = (left, right)
        if first_pair is None:
            first_pair = (left, right)
        if left.path.suffix.lower() == ".json":
            metrics = _json_metrics(
                left.path, right.path,
                tuple(contract_pair["left"].get("json_value_paths", ())),
            )
            details = {"policy": "strict_json_structure"}
        elif left.path.suffix.lower() == ".npz":
            metric_band = None
            if product_id.startswith("wave."):
                metric_band = _resolve_reference_band(
                    analysis.fft_ratio_plotting.wave_frequency_band_hz,
                    (left, right), product_label=f"{context.analysis.id}:{product_id}"
                )
                resolved_wave_band = metric_band
            elif product_id.startswith("spectral."):
                metric_band = _resolve_reference_band(
                    analysis.fft_ratio_plotting.reference_frequency_band_hz,
                    (left, right), product_label=f"{context.analysis.id}:{product_id}"
                )
                resolved_fft_band = metric_band
            metrics, details = _array_metrics(
                left, right, analysis.alignment, frequency_band_hz=metric_band
            )
        else:
            raise ValueError(f"unsupported comparison product format: {left.path.suffix}")
        all_metrics.extend({"product_id": product_id, **metric} for metric in metrics)
        alignment_records[product_id] = details
        preprocessing_records[product_id] = {
            "baseline": left.metadata.get("provenance", {}).get("preprocessing"),
            "comparison": right.metadata.get("provenance", {}).get("preprocessing"),
        }
    source_forcing = _source_forcing_comparison(
        _optional_source_audit(context, analysis.baseline),
        _optional_source_audit(context, analysis.comparison),
    )
    payload = {
        "schema": "pelecpost.comparison",
        "schema_version": 1,
        "baseline": analysis.baseline.model_dump(mode="json"),
        "comparison": analysis.comparison.model_dump(mode="json"),
        "products": list(analysis.product_ids),
        "alignment": analysis.alignment.model_dump(mode="json"),
        "alignment_applied": alignment_records,
        "preprocessing": preprocessing_records,
        "metrics": all_metrics,
        "source_forcing_comparison": source_forcing,
        "interpretation": "Typed product differences after explicit axis and metadata validation.",
    }
    metrics_path = context.data_dir / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="comparison.metrics", path=metrics_path, kind="json", variable=None,
        units=None, coordinate_metadata={}, interpretation=payload["interpretation"],
        provenance={
            "comparison": {
                "baseline": payload["baseline"],
                "comparison": payload["comparison"],
                "alignment_requested": payload["alignment"],
                "alignment_applied": alignment_records,
                "preprocessing": preprocessing_records,
            }
        },
    )
    summary_path = _render_summary(context, all_metrics)
    context.register(
        artifact_id="comparison.figures", path=summary_path, kind="figure", variable=None,
        units="%", coordinate_metadata={},
        interpretation="Relative L2 differences for each comparable typed quantity.",
    )
    if first_pair is None:
        raise ValueError("case_comparison requires at least one product")
    overlay_path: Path | None = None
    overlay_source_artifact_id: str | None = None
    ratio_kind_names = {
        ("absolute", "db"): "fft_amplitude_ratio",
        ("absolute", "linear"): "fft_amplitude_linear_ratio",
        ("unit_l2", "db"): "fft_shape_ratio",
        ("unit_l2", "linear"): "fft_shape_linear_ratio",
    }
    wave_ratio_names = {
        ("absolute", "db"): "fk_amplitude_ratio",
        ("absolute", "linear"): "fk_amplitude_linear_ratio",
        ("unit_l2", "db"): "fk_shape_ratio",
        ("unit_l2", "linear"): "fk_shape_linear_ratio",
    }
    ratio_views = [
        (normalization, scale)
        for normalization in analysis.fft_ratio_plotting.normalizations
        for scale in analysis.fft_ratio_plotting.scales
    ]
    if analysis.product_ids[0] == "wave.komega":
        left, right = first_pair
        main_normalization, main_scale = ratio_views[0]
        main_suffix = wave_ratio_names[(main_normalization, main_scale)]
        band = resolved_wave_band or _resolve_reference_band(
            analysis.fft_ratio_plotting.wave_frequency_band_hz,
            (left, right), product_label=f"{context.analysis.id}:wave.komega"
        )
        rendered = render_komega_comparison(
            left.path, right.path, left.label, right.label,
            context.figure_dir / f"comparison_{main_suffix}.png",
            (
                context.figure_dir / "comparison_signed_k.png"
                if analysis.fft_ratio_plotting.signed_k_ratio
                else None
            ),
            band,
            str(left.metadata.get("units") or "variable²"),
            spectral_display=analysis.fft_ratio_plotting.spectral_display,
            ratio_scale=main_scale,
            ratio_normalization=main_normalization,
            minimum_relative_amplitude=(
                analysis.fft_ratio_plotting.minimum_relative_amplitude
            ),
            dpi=_context_dpi(context),
            linear_ratio_limits=analysis.fft_ratio_plotting.linear_ratio_limits,
            linear_ratio_color_scale=analysis.fft_ratio_plotting.linear_ratio_color_scale,
            db_ratio_limits=analysis.fft_ratio_plotting.db_ratio_limits,
            variable=str(left.metadata.get("variable") or "variable"),
            alignment=analysis.alignment,
        )
        if rendered is not None:
            primary_path, spectrum_path = rendered
            primary_artifact_id = f"comparison.{main_suffix}_figure"
            overlay_source_artifact_id = primary_artifact_id
            primary_provenance = _fk_ratio_provenance(
                analysis, normalization=main_normalization, scale=main_scale,
                band=band, alignment=alignment_records.get("wave.komega", {}),
                statistics_path=primary_path.with_suffix(".json"),
            )
            context.register(
                artifact_id=primary_artifact_id,
                path=primary_path,
                kind="figure",
                variable=left.metadata.get("variable"),
                units=(
                    "amplitude and ratio"
                    if analysis.fft_ratio_plotting.spectral_display == "amplitude"
                    else "relative dB and ratio"
                ),
                coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                interpretation=(
                    f"Source spectra use {analysis.fft_ratio_plotting.spectral_display}; "
                    f"the third panel is a {main_normalization} {main_scale} "
                    "comparison/baseline f-k amplitude ratio with weak bins masked."
                ),
                provenance=primary_provenance,
            )
            ratio_stats_path = primary_path.with_suffix(".json")
            if ratio_stats_path.is_file():
                semantic_stats_id = f"comparison.{main_suffix}_stats"
                context.register(
                    artifact_id=semantic_stats_id,
                    path=ratio_stats_path,
                    kind="json",
                    variable=left.metadata.get("variable"),
                    units="dB" if main_scale == "db" else "ratio",
                    coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                    interpretation=(
                        "Accepted, masked, and color-range clipping statistics for the displayed f-k ratio map."
                    ),
                    provenance=primary_provenance,
                )
                legacy_stats_path = context.data_dir / "fk_ratio_stats.json"
                shutil.copyfile(ratio_stats_path, legacy_stats_path)
                context.register(
                    artifact_id="comparison.fk_ratio_stats",
                    path=legacy_stats_path,
                    kind="json",
                    variable=left.metadata.get("variable"),
                    units="dB" if main_scale == "db" else "ratio",
                    coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                    interpretation=(
                        "Legacy reference to the primary semantic f-k ratio statistics."
                    ),
                    provenance={"derived_from": semantic_stats_id},
                )
            if spectrum_path is not None:
                context.register(
                    artifact_id="comparison.signed_k_figure", path=spectrum_path,
                    kind="figure", variable=left.metadata.get("variable"),
                    units=(
                        _amplitude_unit(str(left.metadata.get("units") or "variable²"))
                        if analysis.fft_ratio_plotting.spectral_display == "amplitude"
                        else left.metadata.get("units")
                    ),
                    coordinate_metadata={"wavenumber": "rad/m"},
                    interpretation=(
                        "Band-integrated signed-wavenumber spectra and amplitude ratio "
                        "for the two cases."
                    ),
                    provenance={
                        "ratio_direction": "comparison / baseline",
                        "ratio_scale": "linear_amplitude",
                        "ratio_units": "1",
                        "ratio_normalization": "absolute_band_sum",
                        "normalization_domain": "selected_frequency_band",
                        "reference_frequency_band_hz": list(band),
                        "minimum_relative_amplitude": (
                            analysis.fft_ratio_plotting.minimum_relative_amplitude
                        ),
                        "support_basis": "each_band_integrated_own_peak",
                        "support_rule": (
                            "each band-summed power must exceed the configured "
                            "fraction squared of its own k-bin peak"
                        ),
                        "coordinates": {"wavenumber": "rad/m"},
                        "alignment": alignment_records.get("wave.komega", {}),
                    },
                )
            overlay_path = context.figure_dir / "comparison_overlay.png"
            shutil.copyfile(primary_path, overlay_path)
            primary_pdf = primary_path.with_suffix(".pdf")
            if primary_pdf.is_file():
                shutil.copyfile(primary_pdf, overlay_path.with_suffix(".pdf"))
            detail_path = context.figure_dir / "comparison_fk_detail.png"
            detail = render_komega_comparison(
                left.path, right.path, left.label, right.label,
                detail_path, None, band,
                str(left.metadata.get("units") or "variable²"),
                spectral_display=analysis.fft_ratio_plotting.spectral_display,
                ratio_scale=main_scale,
                ratio_normalization=main_normalization,
                minimum_relative_amplitude=analysis.fft_ratio_plotting.minimum_relative_amplitude,
                dpi=_context_dpi(context),
                linear_ratio_limits=analysis.fft_ratio_plotting.linear_ratio_limits,
                linear_ratio_color_scale=analysis.fft_ratio_plotting.linear_ratio_color_scale,
                db_ratio_limits=analysis.fft_ratio_plotting.db_ratio_limits,
                variable=str(left.metadata.get("variable") or "variable"),
                wavenumber_limits_rad_m=(-2.0e4, 2.0e4),
                alignment=analysis.alignment,
            )
            if detail is not None:
                detail_provenance = {
                    **primary_provenance,
                    "derived_from": primary_artifact_id,
                    "wavenumber_display_limits_rad_m": [-2.0e4, 2.0e4],
                }
                context.register(
                    artifact_id="comparison.fk_detail_figure", path=detail[0],
                    kind="figure", variable=left.metadata.get("variable"),
                    units="dB" if main_scale == "db" else "ratio",
                    coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                    interpretation="Central signed-wavenumber detail using the same data, masks, and normalization as the overview.",
                    provenance=detail_provenance,
                )
                detail_stats_path = detail[0].with_suffix(".json")
                if detail_stats_path.is_file():
                    context.register(
                        artifact_id="comparison.fk_detail_stats",
                        path=detail_stats_path, kind="json",
                        variable=left.metadata.get("variable"),
                        units="dB" if main_scale == "db" else "ratio",
                        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                        interpretation="Support and clipping statistics for the central f-k detail.",
                        provenance=detail_provenance,
                    )
            if analysis.fft_ratio_plotting.frequency_slices:
                amplitude_slices_path = render_komega_amplitude_frequency_slices(
                    left.path, right.path, left.label, right.label,
                    context.figure_dir / "comparison_fk_frequency_slices_amplitude.png",
                    band,
                    power_unit=str(left.metadata.get("units") or "variable²"),
                    dpi=_context_dpi(context),
                    variable=str(left.metadata.get("variable") or "variable"),
                    alignment=analysis.alignment,
                )
                if amplitude_slices_path is not None:
                    context.register(
                        artifact_id="comparison.fk_frequency_slices_amplitude_figure",
                        path=amplitude_slices_path,
                        kind="figure",
                        variable=left.metadata.get("variable"),
                        units=_amplitude_unit(str(left.metadata.get("units") or "variable²")),
                        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                        interpretation=(
                            "Absolute spectral-amplitude slices at selected frequencies; "
                            "both cases use one shared linear amplitude scale."
                        ),
                        provenance={
                            "reference_frequency_band_hz": list(band),
                            "ratio_scale": "not_applicable_absolute_amplitude",
                            "ratio_normalization": "absolute",
                            "support_rule": "not_applicable_absolute_amplitude",
                            "alignment": alignment_records.get("wave.komega", {}),
                        },
                    )
                slices_path = render_komega_frequency_slices(
                    left.path, right.path, left.label, right.label,
                    context.figure_dir / "comparison_fk_frequency_slices.png", band,
                    ratio_normalization=main_normalization,
                    minimum_relative_amplitude=analysis.fft_ratio_plotting.minimum_relative_amplitude,
                    dpi=_context_dpi(context),
                    variable=str(left.metadata.get("variable") or "variable"),
                    alignment=analysis.alignment,
                )
                if slices_path is not None:
                    context.register(
                        artifact_id="comparison.fk_frequency_slices_figure", path=slices_path,
                        kind="figure", variable=left.metadata.get("variable"), units="ratio",
                        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                        interpretation="Native-bin signed-wavenumber amplitude-ratio slices near 0.5, 1, 2, and 3 MHz.",
                        provenance={
                            **_fk_ratio_provenance(
                                analysis, normalization=main_normalization,
                                scale="linear", band=band,
                                alignment=alignment_records.get("wave.komega", {}),
                                statistics_path=primary_path.with_suffix(".json"),
                            ),
                            "view_type": "frequency_slice",
                            "ratio_color_scale": "not_applicable_line",
                            "derived_from": primary_artifact_id,
                        },
                    )
            for normalization, scale in ratio_views:
                if (normalization, scale) == (main_normalization, main_scale):
                    continue
                suffix = wave_ratio_names[(normalization, scale)]
                extra = render_komega_comparison(
                    left.path, right.path, left.label, right.label,
                    context.figure_dir / f"comparison_{suffix}.png",
                    None, band,
                    str(left.metadata.get("units") or "variable²"),
                    spectral_display=analysis.fft_ratio_plotting.spectral_display,
                    ratio_scale=scale,
                    ratio_normalization=normalization,
                    minimum_relative_amplitude=(
                        analysis.fft_ratio_plotting.minimum_relative_amplitude
                    ),
                    dpi=_context_dpi(context),
                    linear_ratio_limits=analysis.fft_ratio_plotting.linear_ratio_limits,
                    linear_ratio_color_scale=analysis.fft_ratio_plotting.linear_ratio_color_scale,
                    db_ratio_limits=analysis.fft_ratio_plotting.db_ratio_limits,
                    variable=str(left.metadata.get("variable") or "variable"),
                    alignment=analysis.alignment,
                )
                if extra is not None:
                    extra_provenance = _fk_ratio_provenance(
                        analysis, normalization=normalization, scale=scale,
                        band=band, alignment=alignment_records.get("wave.komega", {}),
                        statistics_path=extra[0].with_suffix(".json"),
                    )
                    context.register(
                        artifact_id=f"comparison.{suffix}_figure", path=extra[0],
                        kind="figure", variable=left.metadata.get("variable"),
                        units="dB" if scale == "db" else "ratio",
                        coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                        interpretation=(
                            f"{normalization} {scale} comparison/baseline f-k FFT "
                            "amplitude ratio; weak bins are masked."
                        ),
                        provenance=extra_provenance,
                    )
                    extra_stats = extra[0].with_suffix(".json")
                    if extra_stats.is_file():
                        context.register(
                            artifact_id=f"comparison.{suffix}_stats",
                            path=extra_stats,
                            kind="json",
                            variable=left.metadata.get("variable"),
                            units="dB" if scale == "db" else "ratio",
                            coordinate_metadata={"frequency": "Hz", "wavenumber": "rad/m"},
                            interpretation="Accepted, masked, and color-range clipping statistics for the f-k ratio map.",
                            provenance=extra_provenance,
                        )
    if overlay_path is None:
        overlay_path = _render_product_overlay(context, *first_pair, analysis)
    context.register(
        artifact_id="comparison.overlay_figure", path=overlay_path, kind="figure", variable=None,
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Baseline and comparison visualization for the first selected product.",
        provenance=(
            {"derived_from": overlay_source_artifact_id}
            if overlay_source_artifact_id is not None else {}
        ),
    )
    probe_pair = product_pairs.get("spectral.probe_signals")
    if probe_pair is not None and (
        analysis.fft_ratio_plotting.phase_delay_diagnostics
        or analysis.fft_ratio_plotting.symmetry_diagnostics
    ):
        phase_path = context.data_dir / "phase_symmetry_diagnostic.json"
        try:
            with np.load(probe_pair[0].path, allow_pickle=False) as first, np.load(
                probe_pair[1].path, allow_pickle=False
            ) as second:
                if not np.array_equal(first["probe_indices"], second["probe_indices"]):
                    raise ValueError("baseline and comparison probe indices differ")
                if not np.allclose(first["time_s"], second["time_s"], rtol=0.0, atol=1.0e-15):
                    raise ValueError("baseline and comparison processed time grids differ")
                groups = tuple(
                    tuple(group.probe_indices)
                    for group in analysis.fft_ratio_plotting.probe_groups
                )
                if analysis.fft_ratio_plotting.symmetry_diagnostics:
                    available = set(np.asarray(first["probe_indices"], dtype=int).tolist())
                    missing = sorted(
                        {index for group in groups for index in group} - available
                    )
                    if missing:
                        raise ValueError(
                            f"symmetry diagnostics request unavailable probes {missing}"
                        )
                phase_payload = probe_phase_and_symmetry_diagnostic(
                    np.asarray(first["processed_values"], dtype=float),
                    np.asarray(second["processed_values"], dtype=float),
                    np.asarray(first["time_s"], dtype=float),
                    np.asarray(first["probe_indices"], dtype=int),
                    np.asarray(first["x_m"], dtype=float),
                    groups,
                    variable=str(probe_pair[0].metadata.get("variable") or "variable"),
                    minimum_relative_amplitude=analysis.fft_ratio_plotting.minimum_relative_amplitude,
                    reflection_center_m=analysis.fft_ratio_plotting.reflection_center_m,
                    scalar_reflection_parity=(
                        analysis.fft_ratio_plotting.scalar_reflection_parity
                    ),
                    reference_frequency_band_hz=resolved_fft_band or _resolve_reference_band(
                        analysis.fft_ratio_plotting.reference_frequency_band_hz,
                        probe_pair, product_label=f"{context.analysis.id}:spectral.probe_signals"
                    ),
                    values_are_processed=True,
                    time_origin_s=float(first["time_s"][0]),
                    phase_delay_enabled=analysis.fft_ratio_plotting.phase_delay_diagnostics,
                    symmetry_enabled=analysis.fft_ratio_plotting.symmetry_diagnostics,
                )
                phase_payload["enabled_products"] = {
                    "phase_delay": analysis.fft_ratio_plotting.phase_delay_diagnostics,
                    "symmetry": analysis.fft_ratio_plotting.symmetry_diagnostics,
                }
        except (KeyError, ValueError, OSError) as exc:
            phase_payload = {
                "schema": "pelecpost.probe_phase_symmetry",
                "schema_version": 3,
                "status": "unavailable",
                "reason": str(exc),
                "enabled_products": {
                    "phase_delay": analysis.fft_ratio_plotting.phase_delay_diagnostics,
                    "symmetry": analysis.fft_ratio_plotting.symmetry_diagnostics,
                },
            }
        phase_path.write_text(json.dumps(phase_payload, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="comparison.phase_symmetry_diagnostic",
            path=phase_path,
            kind="json",
            variable=probe_pair[0].metadata.get("variable"),
            units=None,
            coordinate_metadata={"frequency": "Hz", "time": "s"},
            interpretation="Optional complex phase/delay and signed mirrored even/odd diagnostics.",
        )
    if probe_pair is not None:
        peaks_path = context.data_dir / "fft_ratio_peak_table.json"
        probe_band = resolved_fft_band or _resolve_reference_band(
            analysis.fft_ratio_plotting.reference_frequency_band_hz,
            probe_pair, product_label=f"{context.analysis.id}:spectral.probe_signals"
        )
        peak_payload = {
            "schema_version": 1,
            "frequency_band_hz": list(probe_band),
            "interpretation": (
                "Largest accepted Gaussian/asymmetric absolute FFT amplitude ratios. "
                "A maximum can be caused by a denominator notch and is not by itself a resonance."
            ),
            "rows": _fft_ratio_peak_table(
                probe_pair[0], probe_pair[1],
                probe_band,
                analysis.fft_ratio_plotting.minimum_relative_amplitude,
                analysis.alignment,
            ),
        }
        peaks_path.write_text(json.dumps(peak_payload, indent=2) + "\n", encoding="utf-8")
        context.register(
            artifact_id="comparison.fft_ratio_peak_table",
            path=peaks_path,
            kind="json",
            variable=probe_pair[0].metadata.get("variable"),
            units="ratio",
            interpretation=peak_payload["interpretation"],
        )
        if analysis.fft_ratio_plotting.threshold_sensitivity:
            threshold_payload = {
                "schema": "pelecpost.fft_ratio_threshold_sensitivity",
                "schema_version": 1,
                "frequency_band_hz": list(probe_band),
                "normalization": "absolute",
                "thresholds": [],
                "interpretation": "Threshold sensitivity changes support only; reference band and normalization remain fixed.",
            }
            for threshold in (0.005, 0.01, 0.02):
                rows = _fft_ratio_peak_table(
                    probe_pair[0], probe_pair[1], probe_band, threshold,
                    analysis.alignment,
                )
                threshold_payload["thresholds"].append({
                    "minimum_relative_amplitude": threshold,
                    "accepted_probe_count": len(rows),
                    "rows": rows,
                })
            threshold_path = context.data_dir / "fft_ratio_threshold_sensitivity.json"
            threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8")
            context.register(
                artifact_id="comparison.fft_ratio_threshold_sensitivity",
                path=threshold_path, kind="json", variable=probe_pair[0].metadata.get("variable"),
                units="ratio", interpretation=threshold_payload["interpretation"],
            )
    psd_pair = product_pairs.get("spectral.psd")
    if psd_pair is not None:
        psd_band = resolved_fft_band or _resolve_reference_band(
            analysis.fft_ratio_plotting.reference_frequency_band_hz,
            psd_pair,
            product_label=f"{context.analysis.id}:spectral.psd",
        )
        psd_payload, psd_arrays = _psd_ratio_confidence(
            psd_pair[0],
            psd_pair[1],
            analysis.alignment,
            psd_band,
            analysis.fft_ratio_plotting.minimum_relative_amplitude,
            analysis.fft_ratio_plotting.psd_ratio_confidence_level,
        )
        psd_confidence_path = context.data_dir / "psd_ratio_confidence.json"
        psd_confidence_path.write_text(
            json.dumps(psd_payload, indent=2) + "\n", encoding="utf-8"
        )
        context.register(
            artifact_id="comparison.psd_ratio_confidence",
            path=psd_confidence_path,
            kind="json",
            variable=psd_pair[0].metadata.get("variable"),
            units="ratio",
            coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
            interpretation=psd_payload.get(
                "interpretation",
                "PSD ratio confidence is unavailable; see the structured reason.",
            ),
        )
        if psd_arrays is not None:
            psd_ratio_path = _render_psd_ratio_confidence(
                psd_arrays,
                context.figure_dir / "comparison_psd_ratio.png",
                analysis.fft_ratio_plotting.psd_ratio_confidence_level,
                _context_dpi(context),
            )
            context.register(
                artifact_id="comparison.psd_ratio_figure",
                path=psd_ratio_path,
                kind="figure",
                variable=psd_pair[0].metadata.get("variable"),
                units="ratio",
                coordinate_metadata={"frequency": "Hz", "probe_x": "m"},
                interpretation=psd_payload["interpretation"],
                provenance={
                    "confidence_artifact": "comparison.psd_ratio_confidence",
                    "confidence_level": (
                        analysis.fft_ratio_plotting.psd_ratio_confidence_level
                    ),
                },
            )
    probe_ratio_kinds = (
        tuple(
            (kind, f"comparison_{kind}.png", f"comparison.{kind}_figure")
            for normalization, scale in ratio_views
            for kind in (ratio_kind_names[(normalization, scale)],)
        )
        if analysis.fft_ratio_plotting.amplitude_ratio_panels
        else ()
    )
    for product_id, kinds in (
        ("spectral.probe_signals", (
            ("raw_zoom", "comparison_time_zoom.png", "comparison.time_zoom_figure"),
            ("fft", "comparison_fft.png", "comparison.fft_figure"),
            *probe_ratio_kinds,
            ("fft_shape", "comparison_fft_shape.png", "comparison.fft_shape_figure"),
        )),
        ("spectral.psd", (
            ("psd", "comparison_psd.png", "comparison.psd_figure"),
        )),
    ):
        if product_id not in product_pairs:
            continue
        left, right = product_pairs[product_id]
        for kind, filename, artifact_id in kinds:
            path = render_probe_panels(
                left.path, right.path, left.label, right.label,
                str(left.metadata.get("variable") or "signal"),
                context.figure_dir / filename, kind=kind,
                frequency_limit_hz=(
                    (resolved_fft_band or _resolve_reference_band(
                        analysis.fft_ratio_plotting.reference_frequency_band_hz,
                        (left, right), product_label=f"{context.analysis.id}:spectral.probe_signals"
                    ))[1]
                    if kind in {"fft", "fft_shape", *ratio_kind_names.values()}
                    else None
                ),
                minimum_relative_amplitude=(
                    analysis.fft_ratio_plotting.minimum_relative_amplitude
                ),
                reference_frequency_band_hz=(
                    resolved_fft_band or _resolve_reference_band(
                        analysis.fft_ratio_plotting.reference_frequency_band_hz,
                        (left, right), product_label=f"{context.analysis.id}:spectral.probe_signals"
                    )
                ),
                dpi=_context_dpi(context),
                scale_policy=analysis.fft_ratio_plotting.scale_policy,
                probe_groups=tuple(
                    tuple(group.probe_indices)
                    for group in analysis.fft_ratio_plotting.probe_groups
                ),
                alignment=analysis.alignment,
            )
            if path is not None:
                context.register(
                    artifact_id=artifact_id, path=path, kind="figure",
                    variable=left.metadata.get("variable"),
                    units=(
                        "dB" if kind in ("fft_amplitude_ratio", "fft_shape_ratio")
                        else "ratio" if kind in (
                            "fft_amplitude_linear_ratio", "fft_shape_linear_ratio"
                        )
                        else "artifact-dependent"
                    ),
                    coordinate_metadata={"probe_x": "m"},
                    interpretation=(
                        "Comparison/baseline single-sided FFT amplitude ratio "
                        f"({kind}); bins below the configured amplitude "
                        "threshold in either case are masked."
                        if kind in ratio_kind_names.values()
                        else f"Paired {kind} probe curves on physical axes."
                    ),
                )

    # Register all PDF companions after the complete comparison has rendered.
    # The PNG paths retain their historical artifact IDs; PDFs receive a
    # deterministic suffix so both formats are discoverable in the manifest.
    registered_paths = {Path(artifact.path).resolve() for artifact in context.artifacts.artifacts}
    png_artifacts_by_stem = {
        Path(artifact.path).stem: artifact
        for artifact in context.artifacts.artifacts
        if Path(artifact.path).suffix.lower() == ".png"
    }
    for pdf in sorted(context.figure_dir.glob("*.pdf")):
        if pdf.resolve() not in registered_paths:
            source_artifact = png_artifacts_by_stem.get(pdf.stem)
            context.register(
                artifact_id=f"comparison.pdf.{pdf.stem}",
                path=pdf,
                kind="figure",
                variable=source_artifact.variable if source_artifact else None,
                units=source_artifact.units if source_artifact else None,
                coordinate_metadata=(
                    source_artifact.coordinate_metadata if source_artifact else {}
                ),
                interpretation="PDF companion export of the corresponding comparison figure.",
                provenance={
                    **(source_artifact.provenance if source_artifact else {}),
                    "figure_format": "pdf",
                    **({"derived_from": source_artifact.id} if source_artifact else {}),
                },
            )
