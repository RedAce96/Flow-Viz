"""Strict comparisons of registered artifacts from isolated runs."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, cast

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pelecpost.config.models import CaseComparisonAnalysis
from pelecpost.runtime.context import WorkflowContext

from .executors import executor


def _run_path(context: WorkflowContext, configured: Path) -> Path:
    return configured if configured.is_absolute() else (context.project.root / configured).resolve()


def _artifacts(run: Path) -> dict[str, dict]:
    payload = json.loads((run / "artifacts.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["artifacts"]}


def _compatible(first: dict, second: dict) -> None:
    for key in ("schema_version", "variable", "units", "coordinate_metadata", "kind"):
        if first.get(key) != second.get(key):
            raise ValueError(
                f"artifact compatibility mismatch for {key}: {first.get(key)!r} != {second.get(key)!r}"
            )
    first_preprocessing = first.get("provenance", {}).get("preprocessing")
    second_preprocessing = second.get("provenance", {}).get("preprocessing")
    if first_preprocessing != second_preprocessing:
        raise ValueError("artifact preprocessing provenance differs")


def _npz_metrics(first: Path, second: Path) -> list[dict]:
    metrics = []
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        common = sorted(set(left.files) & set(right.files))
        for key in common:
            a, b = np.asarray(left[key]), np.asarray(right[key])
            if a.shape != b.shape:
                raise ValueError(f"array {key!r} has incompatible shapes {a.shape} and {b.shape}")
            if a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
                continue
            difference = np.asarray(a - b)
            finite = np.isfinite(difference)
            if not np.any(finite):
                continue
            metrics.append({
                "quantity": key,
                "shape": list(a.shape),
                "l2_difference": float(np.linalg.norm(difference[finite])),
                "linf_difference": float(np.max(np.abs(difference[finite]))),
                "reference_l2": float(np.linalg.norm(a[np.isfinite(a)])),
            })
    if not metrics:
        raise ValueError("compatible NPZ artifacts share no finite numeric arrays")
    return metrics


def _numeric_metric(name: str, first: np.ndarray, second: np.ndarray) -> dict | None:
    if first.shape != second.shape:
        raise ValueError(f"quantity {name!r} has incompatible shapes {first.shape} and {second.shape}")
    if first.dtype.kind not in "biufc" or second.dtype.kind not in "biufc":
        return None
    difference = np.asarray(first - second)
    finite = np.isfinite(difference)
    if not np.any(finite):
        return None
    return {
        "quantity": name,
        "shape": list(first.shape),
        "l2_difference": float(np.linalg.norm(difference[finite])),
        "linf_difference": float(np.max(np.abs(difference[finite]))),
        "reference_l2": float(np.linalg.norm(first[np.isfinite(first)])),
    }


def _flatten_json(value, prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_json(item, child))
    elif isinstance(value, (int, float, bool)) and not isinstance(value, str):
        result[prefix] = np.asarray(value)
    elif isinstance(value, list):
        candidate = np.asarray(value)
        if candidate.dtype.kind in "biufc":
            result[prefix] = candidate
    return result


def _json_metrics(first: Path, second: Path) -> list[dict]:
    left = _flatten_json(json.loads(first.read_text(encoding="utf-8")))
    right = _flatten_json(json.loads(second.read_text(encoding="utf-8")))
    if set(left) != set(right):
        raise ValueError("JSON numerical quantity paths differ")
    metrics = [
        metric for key in sorted(left)
        if (metric := _numeric_metric(key, left[key], right[key])) is not None
    ]
    if not metrics:
        raise ValueError("compatible JSON artifacts contain no finite numeric quantities")
    return metrics


def _csv_metrics(first: Path, second: Path) -> list[dict]:
    def read(path: Path) -> tuple[list[str], list[list[str]]]:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            rows = list(reader)
        if not rows:
            raise ValueError(f"empty CSV artifact: {path}")
        return rows[0], rows[1:]

    left_columns, left_rows = read(first)
    right_columns, right_rows = read(second)
    if left_columns != right_columns or len(left_rows) != len(right_rows):
        raise ValueError("CSV columns or row counts differ")
    metrics = []
    for index, name in enumerate(left_columns):
        try:
            left = np.asarray([float(row[index]) for row in left_rows])
            right = np.asarray([float(row[index]) for row in right_rows])
        except (ValueError, IndexError):
            continue
        metric = _numeric_metric(name, left, right)
        if metric is not None:
            metrics.append(metric)
    if not metrics:
        raise ValueError("compatible CSV artifacts contain no finite numeric columns")
    return metrics


def _hdf5_metrics(first: Path, second: Path) -> list[dict]:
    def arrays(path: Path) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as archive:
            def visit(name: str, item) -> None:
                if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc":
                    result[name] = np.asarray(item)
            archive.visititems(visit)
        return result

    left, right = arrays(first), arrays(second)
    if set(left) != set(right):
        raise ValueError("HDF5 numerical dataset paths differ")
    metrics = [
        metric for key in sorted(left)
        if (metric := _numeric_metric(key, left[key], right[key])) is not None
    ]
    if not metrics:
        raise ValueError("compatible HDF5 artifacts contain no finite numeric datasets")
    return metrics


def _artifact_metrics(first: Path, second: Path) -> list[dict]:
    if first.suffix.lower() != second.suffix.lower():
        raise ValueError("artifact file formats differ")
    suffix = first.suffix.lower()
    if suffix == ".npz":
        return _npz_metrics(first, second)
    if suffix == ".json":
        return _json_metrics(first, second)
    if suffix == ".csv":
        return _csv_metrics(first, second)
    if suffix in {".h5", ".hdf5"}:
        return _hdf5_metrics(first, second)
    raise ValueError(f"artifact format {suffix!r} is not a comparable numerical product")


@executor("case_comparison")
def run_case_comparison(context: WorkflowContext) -> None:
    analysis = cast(CaseComparisonAnalysis, context.analysis)
    baseline = _run_path(context, analysis.baseline_run)
    comparison = _run_path(context, analysis.comparison_run)
    baseline_artifacts = _artifacts(baseline)
    comparison_artifacts = _artifacts(comparison)
    all_metrics = []
    for artifact_id in analysis.artifact_ids:
        if artifact_id not in baseline_artifacts or artifact_id not in comparison_artifacts:
            raise ValueError(f"artifact {artifact_id!r} is not registered in both runs")
        first = baseline_artifacts[artifact_id]
        second = comparison_artifacts[artifact_id]
        _compatible(first, second)
        first_path = baseline / first["path"]
        second_path = comparison / second["path"]
        for metric in _artifact_metrics(first_path, second_path):
            all_metrics.append({"artifact_id": artifact_id, **metric})
    payload: dict[str, Any] = {
        "schema_version": 1, "baseline_run": str(baseline),
        "comparison_run": str(comparison), "metrics": all_metrics,
        "interpretation": "Differences are reported only after strict metadata and shape compatibility checks.",
    }
    path = context.data_dir / "comparison_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="comparison.metrics", path=path, kind="json", variable=None,
        units=None, coordinate_metadata={}, interpretation=str(payload["interpretation"]),
        provenance={"baseline_run": str(baseline), "comparison_run": str(comparison)},
    )
    figure_path = context.figure_dir / "comparison_linf.png"
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * len(all_metrics)), 5))
    labels = [f"{item['artifact_id']}:{item['quantity']}" for item in all_metrics]
    axis.bar(np.arange(len(all_metrics)), [item["linf_difference"] for item in all_metrics])
    axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
    axis.set_ylabel("Maximum absolute difference")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    context.register(
        artifact_id="comparison.figures", path=figure_path, kind="figure", variable=None,
        units="artifact-dependent", coordinate_metadata={},
        interpretation="Maximum absolute differences for compatible numerical arrays.",
    )
