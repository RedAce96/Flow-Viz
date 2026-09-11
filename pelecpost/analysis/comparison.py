"""Axis-aware comparison of products from current or archived analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pelecpost.config.models import CaseComparisonAnalysis, ComparisonAlignment
from pelecpost.runtime.context import WorkflowContext

from .executors import executor
from .products import product_contract


@dataclass(frozen=True)
class _Product:
    label: str
    metadata: dict[str, Any]
    path: Path


def _artifact_map(context: WorkflowContext) -> dict[str, Any]:
    return {artifact.id: artifact for artifact in context.artifacts.artifacts}


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
) -> str | None:
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if left.ndim != right.ndim:
        raise ValueError(f"comparison arrays have incompatible ranks {left.ndim} and {right.ndim}")
    details: dict[str, Any] = {}
    for dimension in range(left.ndim):
        left_axis = _axis_for_dimension(
            left, dimension, left_axes, product_type, value_dimensions, value_key
        )
        right_axis = _axis_for_dimension(
            right, dimension, right_axes, product_type, value_dimensions, value_key
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
        details[left_axis] = {"policy": policy, "sample_count": int(len(target))}
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
        validity_fields = contract.get("validity_fields", contract.get("mask_keys", ()))
    else:
        validity_fields = validity_by_value.get(value_key, ())
    for key in validity_fields:
        if key not in archive.files:
            continue
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
            dimensions = [
                dimension for dimension in range(value.ndim)
                if len(raw) == value.shape[dimension]
                and (
                    declared_axis is not None
                    and _axis_for_dimension(value, dimension, axes, product_type) == declared_axis
                    or declared_axis is None
                    and _axis_for_dimension(value, dimension, axes, product_type)
                )
            ]
            if len(dimensions) != 1:
                continue
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
        "relative_l2_difference": float(l2 / max(np.linalg.norm(reference), 1.0e-300)),
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_pair = _validate_product_pair(left, right)
    contract = contract_pair["left"]
    metrics: list[dict[str, Any]] = []
    alignment_details: dict[str, Any] = {}
    with (
        np.load(left.path, allow_pickle=False) as first,
        np.load(right.path, allow_pickle=False) as second,
    ):
        left_axes = _axis_arrays(first, contract_pair["left"])
        right_axes = _axis_arrays(second, contract_pair["right"])
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
            a, b, details = _align_values(
                a, b, left_axes, right_axes,
                contract.get("product_type", ""), alignment,
                contract.get("value_dimensions", {}), key,
            )
            alignment_details[key] = details
            left_valid, right_valid, _ = _align_values(
                left_valid.astype(float), right_valid.astype(float),
                left_axes, right_axes, contract.get("product_type", ""), alignment,
                contract.get("value_dimensions", {}), key,
            )
            # Interpolated masks are validity predicates, not measurements:
            # a point is valid only when interpolation stayed exactly on the
            # valid side of every contributing source sample.  A plain bool
            # cast would incorrectly turn fractional 0/1 mask values true.
            left_valid = np.isclose(left_valid, 1.0, rtol=0.0, atol=1.0e-12)
            right_valid = np.isclose(right_valid, 1.0, rtol=0.0, atol=1.0e-12)
            valid_both = left_valid & right_valid
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
    path = context.figure_dir / "comparison_linf.png"
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * len(metrics)), 5))
    labels = [item["quantity"] for item in metrics]
    values = [item["linf_difference"] for item in metrics]
    axis.bar(np.arange(len(values)), values)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylabel("Maximum absolute difference")
    axis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _render_product_overlay(context: WorkflowContext, left: _Product, right: _Product) -> Path:
    path = context.figure_dir / "comparison_overlay.png"
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
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


@executor("case_comparison")
def run_case_comparison(context: WorkflowContext) -> None:
    analysis = cast(CaseComparisonAnalysis, context.analysis)
    all_metrics: list[dict[str, Any]] = []
    alignment_records: dict[str, Any] = {}
    preprocessing_records: dict[str, Any] = {}
    first_pair: tuple[_Product, _Product] | None = None
    for product_id in analysis.product_ids:
        left = _resolve_product(context, analysis.baseline, product_id)
        right = _resolve_product(context, analysis.comparison, product_id)
        contract_pair = _validate_product_pair(left, right)
        if first_pair is None:
            first_pair = (left, right)
        if left.path.suffix.lower() == ".json":
            metrics = _json_metrics(
                left.path, right.path,
                tuple(contract_pair["left"].get("json_value_paths", ())),
            )
            details = {"policy": "strict_json_structure"}
        elif left.path.suffix.lower() == ".npz":
            metrics, details = _array_metrics(left, right, analysis.alignment)
        else:
            raise ValueError(f"unsupported comparison product format: {left.path.suffix}")
        all_metrics.extend({"product_id": product_id, **metric} for metric in metrics)
        alignment_records[product_id] = details
        preprocessing_records[product_id] = {
            "baseline": left.metadata.get("provenance", {}).get("preprocessing"),
            "comparison": right.metadata.get("provenance", {}).get("preprocessing"),
        }
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
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Maximum absolute differences for typed comparison values.",
    )
    if first_pair is None:
        raise ValueError("case_comparison requires at least one product")
    overlay_path = _render_product_overlay(context, *first_pair)
    context.register(
        artifact_id="comparison.overlay_figure", path=overlay_path, kind="figure", variable=None,
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Generic baseline/comparison visualization for the first selected product.",
    )
