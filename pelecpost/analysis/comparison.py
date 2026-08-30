"""Strict comparisons of registered artifacts from isolated runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
                "array": key,
                "shape": list(a.shape),
                "l2_difference": float(np.linalg.norm(difference[finite])),
                "linf_difference": float(np.max(np.abs(difference[finite]))),
                "reference_l2": float(np.linalg.norm(a[np.isfinite(a)])),
            })
    if not metrics:
        raise ValueError("compatible NPZ artifacts share no finite numeric arrays")
    return metrics


@executor("case_comparison")
def run_case_comparison(context: WorkflowContext) -> None:
    analysis = context.analysis
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
        if first_path.suffix.lower() != ".npz" or second_path.suffix.lower() != ".npz":
            raise ValueError(f"artifact {artifact_id!r} is not a comparable NPZ numerical product")
        for metric in _npz_metrics(first_path, second_path):
            all_metrics.append({"artifact_id": artifact_id, **metric})
    payload = {
        "schema_version": 1, "baseline_run": str(baseline),
        "comparison_run": str(comparison), "metrics": all_metrics,
        "interpretation": "Differences are reported only after strict metadata and shape compatibility checks.",
    }
    path = context.data_dir / "comparison_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    context.register(
        artifact_id="comparison.metrics", path=path, kind="json", variable=None,
        units=None, coordinate_metadata={}, interpretation=payload["interpretation"],
        provenance={"baseline_run": str(baseline), "comparison_run": str(comparison)},
    )
    figure_path = context.figure_dir / "comparison_linf.png"
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * len(all_metrics)), 5))
    labels = [f"{item['artifact_id']}:{item['array']}" for item in all_metrics]
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
