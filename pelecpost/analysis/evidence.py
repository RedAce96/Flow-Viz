"""Conservative classification of registered measurement products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pelecpost.runtime.artifacts import Artifact, ArtifactRegistry


THRESHOLDS = {
    "minimum_wave_accepted_fraction": 0.10,
    "minimum_packet_arrival_r_squared": 0.80,
    "minimum_pod_subspace_cosine": 0.90,
    "maximum_dmd_frequency_relative_range": 0.10,
    "maximum_dmd_condition_number": 1.0e8,
}


def _matching(registry: ArtifactRegistry, suffix: str) -> list[Artifact]:
    family = suffix + "."
    return [
        artifact for artifact in registry.artifacts
        if artifact.id.endswith(suffix) or family in artifact.id
    ]


def _read_json(run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    return json.loads((run_dir / artifact.path).read_text(encoding="utf-8"))


def _wave(run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    del run_dir
    phase = float(artifact.provenance.get("accepted_fraction", np.nan))
    growth = float(artifact.provenance.get("growth_accepted_fraction", np.nan))
    return {
        "analysis_id": artifact.recipe_instance,
        "status": (
            "supported_measured_propagation"
            if np.isfinite(phase) and phase >= THRESHOLDS["minimum_wave_accepted_fraction"]
            else "quality_gate_not_passed"
        ),
        "phase_fit_accepted_fraction": phase,
        "spatial_amplification_status": (
            "supported_measured_amplification"
            if np.isfinite(growth) and growth >= THRESHOLDS["minimum_wave_accepted_fraction"]
            else "quality_gate_not_passed"
        ),
        "growth_fit_accepted_fraction": growth,
        "meaning": (
            "Coherence-gated propagation measured on the selected probe line; "
            "not an instability eigenmode."
        ),
    }


def _packet(run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    value = _read_json(run_dir, artifact)
    speed = float(value.get("group_velocity_m_s", np.nan))
    r_squared = float(value.get("arrival_time_regression_r_squared", np.nan))
    supported = (
        np.isfinite(speed) and speed > 0.0 and np.isfinite(r_squared)
        and r_squared >= THRESHOLDS["minimum_packet_arrival_r_squared"]
    )
    return {
        "analysis_id": artifact.recipe_instance,
        "status": "supported_packet_kinematics" if supported else "quality_gate_not_passed",
        "group_velocity_m_s": speed,
        "arrival_time_regression_r_squared": r_squared,
        "confidence_interval_95_m_s": value.get("confidence_interval_95_m_s"),
        "meaning": "Band-limited arrival regression; not a causal attribution.",
    }


def _nonlinear(run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    value = _read_json(run_dir, artifact)
    significant = np.asarray(value.get("significant_fdr", []), dtype=bool)
    return {
        "analysis_id": artifact.recipe_instance,
        "status": (
            "fdr_significant_association_detected"
            if np.any(significant) else "no_fdr_significant_association_detected"
        ),
        "significant_triad_count": int(np.count_nonzero(significant)),
        "tested_triad_count": int(significant.size),
        "meaning": "Surrogate/FDR-controlled quadratic association; not causal transfer.",
    }


def _modal(run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    with np.load(run_dir / artifact.path, allow_pickle=False) as arrays:
        cosines = np.asarray(arrays["pod_subspace_min_cosine_to_first_window"], dtype=float)
        frequencies = np.asarray(arrays["dmd_dominant_frequency_hz"], dtype=float)
        conditions = np.asarray(arrays["dmd_retained_condition_number"], dtype=float)
    finite_cosines = cosines[np.isfinite(cosines)]
    finite_frequency = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
    finite_conditions = conditions[np.isfinite(conditions)]
    minimum_cosine = float(np.min(finite_cosines)) if finite_cosines.size else np.nan
    relative_range = (
        float(np.ptp(finite_frequency) / np.median(finite_frequency))
        if finite_frequency.size else np.nan
    )
    maximum_condition = (
        float(np.max(finite_conditions)) if finite_conditions.size else np.nan
    )
    robust = (
        np.isfinite(minimum_cosine)
        and minimum_cosine >= THRESHOLDS["minimum_pod_subspace_cosine"]
        and np.isfinite(relative_range)
        and relative_range <= THRESHOLDS["maximum_dmd_frequency_relative_range"]
        and np.isfinite(maximum_condition)
        and maximum_condition <= THRESHOLDS["maximum_dmd_condition_number"]
    )
    return {
        "analysis_id": artifact.recipe_instance,
        "status": "robust_descriptive_structure" if robust else "sensitivity_gate_not_passed",
        "minimum_pod_subspace_cosine": minimum_cosine,
        "dmd_frequency_relative_range": relative_range,
        "maximum_dmd_condition_number": maximum_condition,
        "meaning": "POD/SPOD/DMD robustness only; not an eigenmode attribution.",
    }


def _loads(artifact: Artifact) -> dict[str, Any]:
    return {
        "analysis_id": artifact.recipe_instance,
        "status": "measured_load_available",
        "designation": artifact.provenance.get("designation", "unavailable"),
        "baseline": artifact.provenance.get("baseline", "none"),
        "meaning": (
            "Integrated two-dimensional pressure/viscous load under the artifact's "
            "documented geometry, baseline, and validation designation."
        ),
    }


def build_evidence_report(run_dir: Path, registry: ArtifactRegistry) -> dict[str, Any]:
    """Classify only evidence that exists in the artifact registry."""
    domains = {
        "coherent_wave": [
            _wave(run_dir, artifact)
            for artifact in _matching(registry, ".wave.wavenumber")
        ],
        "transient_packet": [
            _packet(run_dir, artifact)
            for artifact in _matching(registry, ".transient.group_velocity")
        ],
        "quadratic_nonlinearity": [
            _nonlinear(run_dir, artifact)
            for artifact in _matching(registry, ".nonlinear.triads")
        ],
        "modal_robustness": [
            _modal(run_dir, artifact)
            for artifact in _matching(registry, ".modal.sensitivity")
        ],
        "aerodynamic_loads": [
            _loads(artifact)
            for artifact in _matching(registry, ".forces.components")
        ],
    }
    return {
        "schema": "pelecpost.measurement-evidence",
        "schema_version": 1,
        "decision_thresholds": THRESHOLDS,
        "domains": domains,
        "excluded_scope": {
            "LST_PSE": "not performed or inferred",
            "causality": "measurement association does not establish causality",
        },
        "limitations": [
            "An empty domain means that no compatible registered product was available.",
            "Statuses are non-exclusive measurement descriptions and inherit artifact quality gates.",
        ],
    }


def write_evidence_report(run_dir: Path, registry: ArtifactRegistry) -> Path:
    path = run_dir / "data" / "measurement_evidence.json"
    path.write_text(
        json.dumps(build_evidence_report(run_dir, registry), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path
