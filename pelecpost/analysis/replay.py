"""Replay supported saved products from one archived Flow Viz run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pelecpost.config.models import AnalysesFile, CaseFile
from pelecpost.runtime.artifacts import atomic_json
from pelecpost.visualization import save_figure_variants

from .comparison import (
    _array_metrics,
    _fft_ratio_peak_table,
    _fk_ratio_provenance,
    _Product,
    _psd_ratio_confidence,
    _render_psd_ratio_confidence,
    _resolve_reference_band,
)
from .comparison_figures import (
    _bin_edges,
    render_komega_amplitude_frequency_slices,
    render_komega_comparison,
    render_komega_frequency_slices,
    render_probe_panels,
)
from .probe_plotting import build_probe_overlay_figure
from .spectral import (
    _plot_probe_time_fft,
    _spectral_sampling_diagnostic,
    probe_phase_and_symmetry_diagnostic,
    spectrum_from_signal,
)

REPLAY_SCHEMA_VERSION = 4
SUPPORTED_RECIPES = frozenset({"probe_spectrum", "directional_wave", "case_comparison"})
RATIO_NAMES = {
    ("absolute", "db"): "fft_amplitude_ratio",
    ("absolute", "linear"): "fft_amplitude_linear_ratio",
    ("unit_l2", "db"): "fft_shape_ratio",
    ("unit_l2", "linear"): "fft_shape_linear_ratio",
}
WAVE_NAMES = {
    ("absolute", "db"): "fk_amplitude_ratio",
    ("absolute", "linear"): "fk_amplitude_linear_ratio",
    ("unit_l2", "db"): "fk_shape_ratio",
    ("unit_l2", "linear"): "fk_shape_linear_ratio",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_source(export: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    run_dir = export.expanduser().resolve()
    manifest = _read_json(run_dir / "manifest.json")
    registry_payload = _read_json(run_dir / "artifacts.json")
    artifacts = registry_payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"{run_dir} has no valid artifact registry")
    registry: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("id"), str):
            raise TypeError("source artifact registry contains an invalid entry")
        artifact_id = artifact["id"]
        if artifact_id in registry:
            raise ValueError(f"duplicate source artifact ID {artifact_id!r}")
        relative = artifact.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"source artifact {artifact_id!r} has no path")
        target = (run_dir / relative).resolve()
        if not target.is_relative_to(run_dir):
            raise ValueError(f"source artifact {artifact_id!r} escapes the archived run")
        registry[artifact_id] = artifact
    return manifest, registry, run_dir


def _source_artifact(
    registry: dict[str, dict[str, Any]], run_dir: Path, artifact_id: str,
) -> Path | None:
    item = registry.get(artifact_id)
    if item is None:
        return None
    path = (run_dir / item["path"]).resolve()
    return path if path.is_file() else None


def _yaml_model(path: Path, model: type) -> Any:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return model.model_validate(data)


def _configuration_differences(
    left: Any, right: Any, prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            result.extend(_configuration_differences(
                left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key)
            ))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        result = []
        for index, (old, new) in enumerate(zip(left, right)):
            result.extend(_configuration_differences(old, new, f"{prefix}[{index}]"))
        return result
    return [] if left == right else [{
        "field": prefix, "archived": left, "selected": right,
    }]


def _configuration(run_dir: Path, project_dir: Path | None) -> tuple[SimpleNamespace, dict[str, Any]]:
    archived_analyses_path = run_dir / "resolved-analyses.yaml"
    archived_case_path = run_dir / "resolved-case.yaml"
    archived_analyses = (
        _yaml_model(archived_analyses_path, AnalysesFile)
        if archived_analyses_path.is_file() else None
    )
    archived_case = _yaml_model(archived_case_path, CaseFile) if archived_case_path.is_file() else None
    if project_dir is None:
        if archived_analyses is None or archived_case is None:
            raise ValueError(
                "archived run lacks resolved analyses/case configuration; "
                "supply --project explicitly for this older archive"
            )
        analyses, case = archived_analyses, archived_case
        source = "archived_resolved_configuration"
    else:
        directory = project_dir.expanduser().resolve()
        analyses = _yaml_model(directory / "analyses.yaml", AnalysesFile)
        case = _yaml_model(directory / "case.yaml", CaseFile)
        source = "explicit_project_override"
    chosen = {
        "case": case.model_dump(mode="json"),
        "analyses": analyses.model_dump(mode="json"),
    }
    archived = (
        {
            "case": archived_case.model_dump(mode="json"),
            "analyses": archived_analyses.model_dump(mode="json"),
        }
        if archived_case is not None and archived_analyses is not None else None
    )
    details = {
        "source": source,
        "project_dir": str(project_dir.expanduser().resolve()) if project_dir else None,
        "sha256": _json_hash(chosen),
        "archived_sha256": _json_hash(archived) if archived is not None else None,
        "changed_fields_from_archive": (
            _configuration_differences(archived, chosen) if archived is not None else None
        ),
    }
    return SimpleNamespace(analyses_file=analyses, case_file=case), details


def _build_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    count = 0
    for path in sorted((*root.glob("*.py"), *root.joinpath("pelecpost").rglob("*.py"))):
        if "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(len(data).to_bytes(8, "big") + data)
        count += 1
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"commit": commit, "content_sha256": digest.hexdigest(), "file_count": count}


def _source_fingerprints(manifest: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Verify original input records only, never derived NPZ products."""
    run_dir = run_dir.resolve()
    source_provenance = manifest.get("provenance")
    records = (
        source_provenance.get("input_fingerprints", [])
        if isinstance(source_provenance, dict) else []
    )
    result = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict) or not record.get("path"):
            continue
        configured = Path(str(record["path"]))
        candidate = (configured if configured.is_absolute() else run_dir / configured).resolve()
        inside_archive = candidate.is_relative_to(run_dir)
        relative = candidate.relative_to(run_dir).as_posix() if inside_archive else None
        expected = str(record.get("checksum") or "")
        actual = _sha256(candidate) if inside_archive and candidate.is_file() else None
        result.append({
            "input_id": record.get("input_id"),
            "archived_path": str(configured),
            "run_relative_path": relative,
            "expected_sha256": expected or None,
            "actual_sha256": actual,
            "reason": (
                "original input was not archived under this run"
                if not inside_archive else
                "archived input file is absent" if actual is None else
                "archived fingerprint checksum is absent" if not expected else None
            ),
            "status": (
                "missing" if actual is None or not expected else
                "matched" if actual == expected else "mismatch"
            ),
        })
    return result


def _derived_sources(
    registry: dict[str, dict[str, Any]], run_dir: Path, selected: set[str],
) -> list[dict[str, Any]]:
    result = []
    for artifact_id, artifact in sorted(registry.items()):
        if artifact.get("recipe_instance") not in selected:
            continue
        path = _source_artifact(registry, run_dir, artifact_id)
        result.append({
            "artifact_id": artifact_id,
            "path": artifact["path"],
            "status": "available" if path else "missing",
            "sha256": _sha256(path) if path else None,
            "provenance_kind": "derived_product",
        })
    return result


def _save_overlay(
    project: SimpleNamespace, stem: Path, x: np.ndarray, values: np.ndarray,
    labels: list[str], x_label: str, y_label: str, *,
    log_x: bool = False, log_y: bool = False,
) -> None:
    figure = build_probe_overlay_figure(
        project.analyses_file.presentation, x, values, labels, x_label, y_label,
        log_x=log_x, log_y=log_y,
    )
    save_figure_variants(figure, stem, project.analyses_file.presentation.figure)


def _replay_spectrum(
    analysis: Any, source: Path, stage: Path, project: SimpleNamespace,
) -> tuple[Path, Path, dict[str, Any]]:
    with np.load(source, allow_pickle=False) as archive:
        available = {int(index): column for column, index in enumerate(archive["probe_indices"])}
        selected = np.asarray(analysis.probe_indices, dtype=int)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"saved history lacks requested probe identities {missing}")
        columns = [available[int(index)] for index in selected]
        time = np.asarray(archive["raw_time_s"], dtype=float)
        raw = np.asarray(archive["raw_values"][:, columns], dtype=float)
        x_m = np.asarray(archive["x_m"][columns], dtype=float)
        unit = str(archive["signal_unit"])
    products = spectrum_from_signal(
        time, raw,
        end_time_s=analysis.end_time_s,
        window=analysis.window,
        detrend=analysis.detrend,
        welch_segment_samples=analysis.welch_segment_samples,
        overlap_fraction=analysis.overlap_fraction,
        time_grid_policy=analysis.time_grid_policy,
        frequency_max_hz=analysis.frequency_max_hz,
        scratch_directory=None,
    )
    figure_dir = stage / "figures" / analysis.id
    data_dir = stage / "data" / analysis.id
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"Probe {index}" for index in selected]
    try:
        _save_overlay(
            project, figure_dir / "raw_history_overlay", products["raw_time"],
            products["raw_values"], labels, "Time [s]", f"{analysis.variable.value.capitalize()} [{unit}]",
        )
        _save_overlay(
            project, figure_dir / "method_ready_overlay", products["fft_time"],
            products["processed_values"], labels, "Time [s]",
            f"{analysis.variable.value.capitalize()} disturbance [{unit}]",
        )
        fft_positive = products["fft_frequency"] > 0.0
        _save_overlay(
            project, figure_dir / "fft_overlay", products["fft_frequency"][fft_positive],
            products["fft_amplitude"][fft_positive], labels, "Frequency [Hz]",
            f"Amplitude [{unit}]", log_x=True, log_y=True,
        )
        psd_positive = products["welch_frequency"] > 0.0
        _save_overlay(
            project, figure_dir / "psd_overlay", products["welch_frequency"][psd_positive],
            products["psd"][psd_positive], labels, "Frequency [Hz]",
            f"PSD [({unit})²/Hz]", log_x=True, log_y=True,
        )
        _plot_probe_time_fft(
            figure_dir / "probe_time_fft.png",
            products["raw_time"], products["raw_values"],
            products["fft_time"], products["processed_values"],
            products["fft_frequency"], products["fft_amplitude"],
            x_m, selected, unit, analysis.detrend, analysis.window,
            dpi=int(project.analyses_file.presentation.figure.dpi),
        )
        signal_path = data_dir / "probe_time_fft.npz"
        np.savez_compressed(
            signal_path,
            raw_time_s=products["raw_time"], raw_values=products["raw_values"],
            time_s=products["fft_time"], processed_values=products["processed_values"],
            frequency_hz=products["fft_frequency"], amplitude=products["fft_amplitude"],
            probe_indices=selected, x_m=x_m, signal_unit=np.array(unit),
        )
        psd_path = data_dir / "stationary_spectrum.npz"
        np.savez_compressed(
            psd_path,
            frequency_hz=products["welch_frequency"], psd=products["psd"],
            probe_indices=selected, x_m=x_m, signal_unit=np.array(unit),
            psd_unit=np.array(f"({unit})²/Hz"),
        )
        confidence = products["welch_confidence"]
        atomic_json(data_dir / "spectral_confidence.json", confidence)
        sampling = _spectral_sampling_diagnostic(
            products["raw_time"], products["raw_values"],
            products["fft_time"], products["fft_values"],
            products["processed_values"], products["weights"],
            products["dt"], analysis,
        )
        atomic_json(data_dir / "spectral_sampling_diagnostic.json", sampling)
        return signal_path, psd_path, confidence
    finally:
        if products["cleanup"] is not None:
            products["cleanup"]()


def _unavailable(
    records: list[dict[str, Any]], analysis_id: str, product_id: str, reason: str,
) -> None:
    records.append({
        "analysis_id": analysis_id,
        "product_id": product_id,
        "status": "unavailable_input",
        "reason": reason,
    })


def _comparison_pair(
    analysis: Any, product_id: str,
    registry: dict[str, dict[str, Any]], run_dir: Path,
    signals: dict[str, Path], psds: dict[str, Path],
    confidence: dict[str, dict[str, Any]],
) -> tuple[_Product, _Product] | None:
    if analysis.baseline.archived_run_id or analysis.comparison.archived_run_id:
        return None
    products: list[_Product] = []
    for reference in (analysis.baseline, analysis.comparison):
        artifact_id = f"{reference.analysis_id}.{product_id}"
        metadata = dict(registry.get(artifact_id, {}))
        if product_id == "spectral.probe_signals":
            path = signals.get(reference.analysis_id) or _source_artifact(
                registry, run_dir, artifact_id
            )
        elif product_id == "spectral.psd":
            path = psds.get(reference.analysis_id) or _source_artifact(
                registry, run_dir, artifact_id
            )
            if reference.analysis_id in confidence:
                metadata["provenance"] = {
                    **metadata.get("provenance", {}),
                    "welch_confidence": confidence[reference.analysis_id],
                }
        elif product_id == "wave.komega":
            path = _source_artifact(registry, run_dir, artifact_id)
        else:
            return None
        if path is None:
            return None
        products.append(_Product(artifact_id, metadata, path))
    return products[0], products[1]


def _replay_probe_comparison(
    analysis: Any, pair: tuple[_Product, _Product], stage: Path,
    project: SimpleNamespace, unavailable: list[dict[str, Any]],
) -> None:
    left, right = pair
    options = analysis.fft_ratio_plotting
    band = _resolve_reference_band(
        options.reference_frequency_band_hz, pair,
        product_label=f"replay:{analysis.id}:spectral.probe_signals",
    )
    figures = stage / "figures" / analysis.id
    data = stage / "data" / analysis.id
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    groups = tuple(tuple(group.probe_indices) for group in options.probe_groups)
    kinds = ["raw", "raw_zoom", "fft", "fft_shape"]
    if options.amplitude_ratio_panels:
        kinds.extend(
            RATIO_NAMES[(normalization, scale)]
            for normalization in options.normalizations for scale in options.scales
        )
    for kind in kinds:
        name = {
            "raw": "comparison_time.png",
            "raw_zoom": "comparison_time_zoom.png",
            "fft": "comparison_fft.png",
            "fft_shape": "comparison_fft_shape.png",
        }.get(kind, f"comparison_{kind}.png")
        render_probe_panels(
            left.path, right.path, left.label, right.label,
            str(left.metadata.get("variable") or "signal"), figures / name,
            kind=kind, minimum_relative_amplitude=options.minimum_relative_amplitude,
            reference_frequency_band_hz=band,
            frequency_limit_hz=band[1] if kind not in {"raw", "raw_zoom"} else None,
            dpi=int(project.analyses_file.presentation.figure.dpi),
            scale_policy=options.scale_policy, probe_groups=groups,
            alignment=analysis.alignment,
        )
    peaks = _fft_ratio_peak_table(
        left, right, band, options.minimum_relative_amplitude, analysis.alignment,
    )
    atomic_json(data / "fft_ratio_peak_table.json", {
        "schema": "pelecpost.fft_ratio_peak_table", "schema_version": 1,
        "frequency_band_hz": list(band), "rows": peaks,
    })
    if options.threshold_sensitivity:
        atomic_json(data / "fft_ratio_threshold_sensitivity.json", {
            "schema": "pelecpost.fft_ratio_threshold_sensitivity",
            "schema_version": 1,
            "frequency_band_hz": list(band),
            "thresholds": [
                {
                    "minimum_relative_amplitude": threshold,
                    "rows": _fft_ratio_peak_table(left, right, band, threshold, analysis.alignment),
                }
                for threshold in (0.005, 0.01, 0.02)
            ],
        })
    if options.phase_delay_diagnostics or options.symmetry_diagnostics:
        try:
            with np.load(left.path, allow_pickle=False) as first, np.load(
                right.path, allow_pickle=False
            ) as second:
                if not np.array_equal(first["probe_indices"], second["probe_indices"]):
                    raise ValueError("baseline and comparison probe identities differ")
                if not np.allclose(first["time_s"], second["time_s"], rtol=0.0, atol=1e-15):
                    raise ValueError("baseline and comparison processed time grids differ")
                result = probe_phase_and_symmetry_diagnostic(
                    np.asarray(first["processed_values"], dtype=float),
                    np.asarray(second["processed_values"], dtype=float),
                    np.asarray(first["time_s"], dtype=float),
                    np.asarray(first["probe_indices"], dtype=int),
                    np.asarray(first["x_m"], dtype=float),
                    groups, variable=str(left.metadata.get("variable") or "signal"),
                    minimum_relative_amplitude=options.minimum_relative_amplitude,
                    reflection_center_m=options.reflection_center_m,
                    scalar_reflection_parity=options.scalar_reflection_parity,
                    reference_frequency_band_hz=band,
                    values_are_processed=True,
                    time_origin_s=float(first["time_s"][0]),
                    phase_delay_enabled=options.phase_delay_diagnostics,
                    symmetry_enabled=options.symmetry_diagnostics,
                )
                result["enabled_products"] = {
                    "phase_delay": options.phase_delay_diagnostics,
                    "symmetry": options.symmetry_diagnostics,
                }
        except (KeyError, ValueError, OSError) as exc:
            result = {
                "schema": "pelecpost.probe_phase_symmetry", "schema_version": 3,
                "status": "unavailable", "reason": str(exc),
                "enabled_products": {
                    "phase_delay": options.phase_delay_diagnostics,
                    "symmetry": options.symmetry_diagnostics,
                },
            }
            _unavailable(unavailable, analysis.id, "comparison.phase_symmetry_diagnostic", str(exc))
        atomic_json(data / "phase_symmetry_diagnostic.json", result)


def _replay_psd_comparison(
    analysis: Any, pair: tuple[_Product, _Product], stage: Path,
    project: SimpleNamespace,
) -> None:
    left, right = pair
    options = analysis.fft_ratio_plotting
    band = _resolve_reference_band(
        options.reference_frequency_band_hz, pair,
        product_label=f"replay:{analysis.id}:spectral.psd",
    )
    figures = stage / "figures" / analysis.id
    data = stage / "data" / analysis.id
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    render_probe_panels(
        left.path, right.path, left.label, right.label,
        str(left.metadata.get("variable") or "signal"),
        figures / "comparison_psd.png", kind="psd",
        dpi=int(project.analyses_file.presentation.figure.dpi),
        scale_policy=options.scale_policy,
        probe_groups=tuple(tuple(group.probe_indices) for group in options.probe_groups),
        alignment=analysis.alignment,
    )
    payload, arrays = _psd_ratio_confidence(
        left, right, analysis.alignment, band,
        options.minimum_relative_amplitude, options.psd_ratio_confidence_level,
    )
    atomic_json(data / "psd_ratio_confidence.json", payload)
    if arrays is not None:
        _render_psd_ratio_confidence(
            arrays, figures / "comparison_psd_ratio_confidence.png",
            options.psd_ratio_confidence_level,
            int(project.analyses_file.presentation.figure.dpi),
        )


def _replay_wave_comparison(
    analysis: Any, pair: tuple[_Product, _Product], stage: Path,
    project: SimpleNamespace, provenance: dict[str, dict[str, Any]],
) -> None:
    left, right = pair
    options = analysis.fft_ratio_plotting
    band = _resolve_reference_band(
        options.wave_frequency_band_hz, pair,
        product_label=f"replay:{analysis.id}:wave.komega",
    )
    figures = stage / "figures" / analysis.id
    data = stage / "data" / analysis.id
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    wave_views = tuple(
        (normalization, scale)
        for normalization in options.normalizations for scale in options.scales
    )
    primary: Path | None = None
    primary_provenance: dict[str, Any] | None = None
    for index, (normalization, scale) in enumerate(wave_views):
        suffix = WAVE_NAMES[(normalization, scale)]
        target = figures / f"comparison_{suffix}.png"
        result = render_komega_comparison(
            left.path, right.path, left.label, right.label,
            target,
            figures / "comparison_signed_k.png"
            if index == 0 and options.signed_k_ratio else None,
            band, str(left.metadata.get("units") or "variable²"),
            spectral_display=options.spectral_display,
            ratio_scale=scale, ratio_normalization=normalization,
            minimum_relative_amplitude=options.minimum_relative_amplitude,
            dpi=int(project.analyses_file.presentation.figure.dpi),
            linear_ratio_limits=options.linear_ratio_limits,
            linear_ratio_color_scale=options.linear_ratio_color_scale,
            db_ratio_limits=options.db_ratio_limits,
            variable=str(left.metadata.get("variable") or "signal"),
            alignment=analysis.alignment,
        )
        if result is None:
            continue
        semantic_id = f"{analysis.id}.comparison.{suffix}_figure"
        view_provenance = _fk_ratio_provenance(
            analysis, normalization=normalization, scale=scale,
            band=band, alignment={}, statistics_path=target.with_suffix(".json"),
        )
        provenance[target.relative_to(stage).as_posix()] = view_provenance
        provenance[target.with_suffix(".json").relative_to(stage).as_posix()] = view_provenance
        provenance[target.with_suffix(".pdf").relative_to(stage).as_posix()] = {
            **view_provenance, "derived_from": semantic_id,
        }
        if index == 0:
            primary, primary_provenance = target, view_provenance
            if result[1] is not None:
                signed = result[1]
                provenance[signed.relative_to(stage).as_posix()] = {
                    "ratio_direction": "comparison / baseline",
                    "ratio_scale": "linear_amplitude",
                    "ratio_units": "1",
                    "ratio_normalization": "absolute_band_sum",
                    "normalization_domain": "selected_frequency_band",
                    "reference_frequency_band_hz": list(band),
                    "minimum_relative_amplitude": options.minimum_relative_amplitude,
                    "support_basis": "each_band_integrated_own_peak",
                    "support_rule": (
                        "each band-summed power must exceed the configured "
                        "fraction squared of its own k-bin peak"
                    ),
                    "coordinates": {"wavenumber": "rad/m"},
                    "alignment": view_provenance["alignment"],
                }
    if primary is None:
        return
    for legacy_name in ("comparison_fk.png", "comparison_overlay.png"):
        legacy = figures / legacy_name
        shutil.copyfile(primary, legacy)
        provenance[legacy.relative_to(stage).as_posix()] = {
            "derived_from": f"{analysis.id}.comparison.{WAVE_NAMES[wave_views[0]]}_figure"
        }
        if primary.with_suffix(".pdf").is_file():
            shutil.copyfile(primary.with_suffix(".pdf"), legacy.with_suffix(".pdf"))
            provenance[legacy.with_suffix(".pdf").relative_to(stage).as_posix()] = {
                **provenance[legacy.relative_to(stage).as_posix()], "figure_format": "pdf"
            }
    detail = render_komega_comparison(
        left.path, right.path, left.label, right.label,
        figures / "comparison_fk_detail.png", None, band,
        str(left.metadata.get("units") or "variable²"),
        spectral_display=options.spectral_display,
        ratio_scale=wave_views[0][1], ratio_normalization=wave_views[0][0],
        minimum_relative_amplitude=options.minimum_relative_amplitude,
        dpi=int(project.analyses_file.presentation.figure.dpi),
        linear_ratio_limits=options.linear_ratio_limits,
        linear_ratio_color_scale=options.linear_ratio_color_scale,
        db_ratio_limits=options.db_ratio_limits,
        variable=str(left.metadata.get("variable") or "signal"),
        wavenumber_limits_rad_m=(-2.0e4, 2.0e4),
        alignment=analysis.alignment,
    )
    if detail is not None:
        provenance[detail[0].relative_to(stage).as_posix()] = {
            **(primary_provenance or {}), "wavenumber_display_limits_rad_m": [-2.0e4, 2.0e4],
        }
    if options.frequency_slices:
        amplitude_path = render_komega_amplitude_frequency_slices(
            left.path, right.path, left.label, right.label,
            figures / "comparison_fk_frequency_slices_amplitude.png", band,
            power_unit=str(left.metadata.get("units") or "variable²"),
            dpi=int(project.analyses_file.presentation.figure.dpi),
            variable=str(left.metadata.get("variable") or "signal"),
            alignment=analysis.alignment,
        )
        if amplitude_path is not None:
            provenance[amplitude_path.relative_to(stage).as_posix()] = {
                "ratio_scale": "not_applicable_absolute_amplitude",
                "ratio_normalization": "absolute",
                "support_rule": "not_applicable_absolute_amplitude",
                "reference_frequency_band_hz": list(band),
                "display_units": str(left.metadata.get("units") or "variable²"),
                "coordinates": {"frequency": "Hz", "wavenumber": "rad/m"},
            }
        ratio_path = render_komega_frequency_slices(
            left.path, right.path, left.label, right.label,
            figures / "comparison_fk_frequency_slices.png", band,
            ratio_normalization=wave_views[0][0],
            minimum_relative_amplitude=options.minimum_relative_amplitude,
            dpi=int(project.analyses_file.presentation.figure.dpi),
            variable=str(left.metadata.get("variable") or "signal"),
            alignment=analysis.alignment,
        )
        if ratio_path is not None:
            slice_provenance = _fk_ratio_provenance(
                analysis, normalization=wave_views[0][0], scale="linear",
                band=band, alignment={},
                statistics_path=primary.with_suffix(".json"),
            )
            slice_provenance.update({
                "view_type": "frequency_slice",
                "ratio_color_scale": "not_applicable_line",
            })
            provenance[ratio_path.relative_to(stage).as_posix()] = slice_provenance
    # Save the semantic primary statistics under the familiar name as a derived copy.
    primary_stats = primary.with_suffix(".json")
    if primary_stats.is_file():
        legacy_stats = data / "fk_ratio_stats.json"
        shutil.copyfile(primary_stats, legacy_stats)
        provenance[legacy_stats.relative_to(stage).as_posix()] = {
            "derived_from": f"{analysis.id}.comparison.{WAVE_NAMES[wave_views[0]]}_stats"
        }


def _replay_wave_producer(
    analysis: Any, source: Path, stage: Path, project: SimpleNamespace,
    provenance: dict[str, dict[str, Any]], artifact_id: str,
) -> None:
    """Redraw a saved f-k map without claiming its transform was recomputed."""
    with np.load(source, allow_pickle=False) as archive:
        frequency = np.asarray(archive["frequency_hz"], dtype=float)
        wavenumber = np.asarray(archive["wavenumber_rad_m"], dtype=float)
        power = np.maximum(np.asarray(archive["power"], dtype=float), 0.0)
        amplitude = (
            np.asarray(archive["amplitude"], dtype=float)
            if "amplitude" in archive.files else np.sqrt(power)
        )
    selected = (
        (frequency > 0.0)
        & (frequency >= analysis.frequency_min_hz)
        & (frequency <= analysis.frequency_max_hz)
    )
    if np.count_nonzero(selected) < 2 or len(wavenumber) < 2:
        raise ValueError("saved f-k product has fewer than two display bins on an axis")
    figure_dir = stage / "figures" / analysis.id
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / "komega.png"
    shown_power = power[selected]
    if analysis.spectral_display == "amplitude":
        shown = amplitude[selected]
        color_label = "Saved spectral amplitude [source signal units]"
        display_units = "source signal units"
        minimum, maximum = 0.0, max(float(np.max(shown)), 1.0e-300)
    else:
        reference = max(float(np.max(shown_power)), 1.0e-300)
        shown = 10.0 * np.log10(shown_power / reference + 1.0e-300)
        color_label = "Power relative to saved map maximum [dB]"
        display_units = "relative dB"
        minimum, maximum = -60.0, 0.0
    figure, axis = plt.subplots(figsize=(10, 6.5))
    try:
        image = axis.pcolormesh(
            _bin_edges(wavenumber) / 1e3,
            _bin_edges(frequency[selected]) / 1e6,
            shown, shading="flat", vmin=minimum, vmax=maximum,
        )
        axis.set_xlabel("Signed wavenumber [10³ rad/m]")
        axis.set_ylabel("Frequency [MHz]")
        axis.set_yscale("log")
        axis.set_ylim(analysis.frequency_min_hz / 1e6 if analysis.frequency_min_hz > 0 else
                      float(frequency[selected][0]) / 1e6,
                      analysis.frequency_max_hz / 1e6)
        axis.set_title(f"{analysis.variable.value.capitalize()} saved f-k spectrum")
        figure.colorbar(image, ax=axis, label=color_label)
        figure.tight_layout()
        figure.savefig(target, dpi=int(project.analyses_file.presentation.figure.dpi), bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)
    provenance[target.relative_to(stage).as_posix()] = {
        "derived_from": artifact_id,
        "transform_recomputed": False,
        "source_array_sha256": _sha256(source),
        "display_mode": analysis.spectral_display,
        "display_units": display_units,
        "frequency_display_limits_hz": [analysis.frequency_min_hz, analysis.frequency_max_hz],
        "coordinates": {"frequency": "Hz", "wavenumber": "rad/m"},
    }


def _role_checklist(
    registry: dict[str, dict[str, Any]], stage: Path, selected: set[str],
) -> list[dict[str, Any]]:
    figures = [path for path in (stage / "figures").rglob("*") if path.is_file()]
    roles = []
    for artifact_id, artifact in sorted(registry.items()):
        if artifact.get("kind") != "figure" or artifact.get("recipe_instance") not in selected:
            continue
        source_name = Path(str(artifact["path"])).name
        same_analysis = [
            path for path in figures
            if path.parent.name == artifact.get("recipe_instance") and path.name == source_name
        ]
        roles.append({
            "role_id": artifact_id,
            "source_artifact_path": artifact["path"],
            "output_path": same_analysis[0].relative_to(stage).as_posix() if same_analysis else None,
            "status": "available" if same_analysis else "unavailable_input",
            "reason": None if same_analysis else "this source figure has no replayable saved input",
        })
    return roles


def _analysis_selection(analyses: AnalysesFile, analysis_ids: tuple[str, ...]) -> tuple[list[Any], set[str]]:
    by_id = {analysis.id: analysis for analysis in analyses.analyses}
    requested = set(analysis_ids)
    unknown = requested - by_id.keys()
    if unknown:
        raise ValueError(f"unknown analysis IDs: {sorted(unknown)}")
    disabled = sorted(analysis_id for analysis_id in requested if not by_id[analysis_id].enabled)
    if disabled:
        raise ValueError(f"selected analyses are disabled: {disabled}")
    if not requested:
        requested = {
            analysis.id for analysis in analyses.analyses
            if analysis.enabled and analysis.recipe in SUPPORTED_RECIPES
        }
    dependencies = set(requested)
    for analysis_id in requested:
        analysis = by_id[analysis_id]
        if analysis.recipe != "case_comparison":
            continue
        for reference in (analysis.baseline, analysis.comparison):
            producer = by_id.get(reference.analysis_id)
            if (
                reference.archived_run_id is None and producer is not None
                and producer.enabled and producer.recipe == "probe_spectrum"
            ):
                dependencies.add(producer.id)
    ordered = [analysis for analysis in analyses.analyses if analysis.id in dependencies]
    return ordered, requested


def _remove_incomplete_outputs(stage: Path, existing: set[Path]) -> None:
    for path in stage.rglob("*"):
        if path.is_file() and path not in existing:
            path.unlink()


def _render_selected(
    stage: Path, project: SimpleNamespace, ordered: list[Any], selected: set[str],
    registry: dict[str, dict[str, Any]], run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    unavailable: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    signals: dict[str, Path] = {}
    psds: dict[str, Path] = {}
    confidence: dict[str, dict[str, Any]] = {}
    for analysis in ordered:
        if analysis.recipe != "probe_spectrum":
            continue
        artifact_id = f"{analysis.id}.spectral.probe_signals"
        source = _source_artifact(registry, run_dir, artifact_id)
        if source is None:
            _unavailable(unavailable, analysis.id, "spectral.probe_signals", "saved raw probe history is missing")
            continue
        existing = {path for path in stage.rglob("*") if path.is_file()}
        try:
            signals[analysis.id], psds[analysis.id], confidence[analysis.id] = _replay_spectrum(
                analysis, source, stage, project,
            )
        except (KeyError, ValueError, OSError, TypeError) as exc:
            _remove_incomplete_outputs(stage, existing)
            _unavailable(unavailable, analysis.id, "spectral.probe_signals", str(exc))
    for analysis in ordered:
        if analysis.id not in selected:
            continue
        if analysis.recipe == "directional_wave":
            artifact_id = f"{analysis.id}.wave.komega"
            source = _source_artifact(registry, run_dir, artifact_id)
            if source is None:
                _unavailable(unavailable, analysis.id, "wave.komega", "saved f-k power map is missing")
            else:
                existing = {path for path in stage.rglob("*") if path.is_file()}
                try:
                    _replay_wave_producer(
                        analysis, source, stage, project, provenance, artifact_id,
                    )
                except (KeyError, ValueError, OSError, TypeError) as exc:
                    _remove_incomplete_outputs(stage, existing)
                    _unavailable(unavailable, analysis.id, "wave.komega", str(exc))
            continue
        if analysis.recipe != "case_comparison":
            if analysis.recipe != "probe_spectrum":
                _unavailable(unavailable, analysis.id, analysis.recipe, "recipe has no saved-product replay handler")
            continue
        comparison_metrics: list[dict[str, Any]] = []
        alignment_applied: dict[str, Any] = {}
        for product_id in analysis.product_ids:
            if product_id not in {"spectral.probe_signals", "spectral.psd", "wave.komega"}:
                _unavailable(
                    unavailable, analysis.id, product_id,
                    "no saved-product replay handler for this product type",
                )
                continue
            pair = _comparison_pair(
                analysis, product_id, registry, run_dir, signals, psds, confidence,
            )
            if pair is None:
                _unavailable(
                    unavailable, analysis.id, product_id,
                    "paired saved products are missing, unsupported, or refer to another archived run",
                )
                continue
            try:
                configured_band = (
                    analysis.fft_ratio_plotting.wave_frequency_band_hz
                    if product_id.startswith("wave.") else
                    analysis.fft_ratio_plotting.reference_frequency_band_hz
                )
                metric_band = _resolve_reference_band(
                    configured_band, pair,
                    product_label=f"replay:{analysis.id}:{product_id}",
                )
                rows, details = _array_metrics(
                    pair[0], pair[1], analysis.alignment,
                    frequency_band_hz=metric_band,
                )
                comparison_metrics.extend({"product_id": product_id, **row} for row in rows)
                alignment_applied[product_id] = details
            except (KeyError, ValueError, OSError, TypeError) as exc:
                _unavailable(unavailable, analysis.id, f"{product_id}.metrics", str(exc))
            existing = {path for path in stage.rglob("*") if path.is_file()}
            try:
                if product_id == "spectral.probe_signals":
                    _replay_probe_comparison(analysis, pair, stage, project, unavailable)
                elif product_id == "spectral.psd":
                    _replay_psd_comparison(analysis, pair, stage, project)
                elif product_id == "wave.komega":
                    _replay_wave_comparison(analysis, pair, stage, project, provenance)
            except (KeyError, ValueError, OSError, TypeError) as exc:
                _remove_incomplete_outputs(stage, existing)
                _unavailable(unavailable, analysis.id, product_id, str(exc))
        data_dir = stage / "data" / analysis.id
        data_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(data_dir / "comparison_metrics.json", {
            "schema": "pelecpost.comparison", "schema_version": 1,
            "status": "available" if comparison_metrics else "unavailable_no_comparable_products",
            "baseline": analysis.baseline.model_dump(mode="json"),
            "comparison": analysis.comparison.model_dump(mode="json"),
            "products": list(analysis.product_ids),
            "alignment": analysis.alignment.model_dump(mode="json"),
            "alignment_applied": alignment_applied,
            "metrics": comparison_metrics,
            "interpretation": "Typed product differences after explicit axis alignment.",
        })
    return unavailable, provenance


def _output_artifact_id(relative: Path) -> str:
    parts = relative.parts
    if len(parts) < 3 or parts[0] not in {"data", "figures"}:
        return f"replay.{relative.stem}.{relative.suffix.lstrip('.')}"
    analysis_id = parts[1]
    stem = relative.stem
    if stem == "komega" and parts[0] == "figures":
        return f"{analysis_id}.wave.komega.figure{'.pdf' if relative.suffix == '.pdf' else ''}"
    for suffix in WAVE_NAMES.values():
        if stem == f"comparison_{suffix}":
            role = "stats" if relative.suffix == ".json" else "figure"
            companion = ".pdf" if relative.suffix == ".pdf" else ""
            return f"{analysis_id}.comparison.{suffix}_{role}{companion}"
    named = {
        "comparison_fk_detail": "fk_detail",
        "comparison_fk_frequency_slices": "fk_frequency_slices",
        "comparison_fk_frequency_slices_amplitude": "fk_frequency_slices_amplitude",
        "comparison_signed_k": "signed_k",
        "comparison_psd_ratio_confidence": "psd_ratio_confidence",
    }
    if stem in named:
        role = "stats" if relative.suffix == ".json" else "figure"
        companion = ".pdf" if relative.suffix == ".pdf" else ""
        return f"{analysis_id}.comparison.{named[stem]}_{role}{companion}"
    if stem == "psd_ratio_confidence" and relative.suffix == ".json":
        return f"{analysis_id}.comparison.psd_ratio_confidence"
    if stem == "comparison_metrics" and relative.suffix == ".json":
        return f"{analysis_id}.comparison.metrics"
    if stem == "fk_ratio_stats" and relative.suffix == ".json":
        return f"{analysis_id}.comparison.fk_ratio_stats"
    if stem == "comparison_overlay":
        return f"{analysis_id}.comparison.overlay_figure{'.pdf' if relative.suffix == '.pdf' else ''}"
    return f"{analysis_id}.replay.{stem}.{relative.suffix.lstrip('.')}"


def _output_records(
    stage: Path, provenance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        relative_path = Path(relative)
        parts = relative_path.parts
        current_provenance = provenance.get(relative)
        if current_provenance is None and path.suffix in {".pdf", ".json"}:
            parent_png = relative_path.with_suffix(".png").as_posix()
            if parent_png in provenance:
                current_provenance = {
                    **provenance[parent_png],
                    "derived_from": _output_artifact_id(relative_path.with_suffix(".png")),
                }
        records.append({
            "id": _output_artifact_id(relative_path),
            "path": relative,
            "kind": "json" if path.suffix == ".json" else "figure" if parts[0] == "figures" else "array",
            "units": (current_provenance or {}).get("ratio_units") or
            (current_provenance or {}).get("display_units"),
            "coordinate_metadata": (current_provenance or {}).get("coordinates", {}),
            "sha256": _sha256(path),
            "provenance": current_provenance or {},
        })
    return records


def _write_output(
    output: Path, stage: Path, records: list[dict[str, Any]],
    previous: dict[str, Any] | None,
) -> None:
    previous_hashes = {
        item["path"]: item["sha256"]
        for item in (previous or {}).get("output_artifacts", [])
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    }
    for item in records:
        relative = Path(item["path"])
        target = output / relative
        expected = previous_hashes.get(item["path"])
        if expected and target.is_file() and _sha256(target) == expected:
            item["sha256"] = expected
            item["resume_status"] = "kept_verified"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.replay-{os.getpid()}.tmp")
        shutil.copyfile(stage / relative, temporary)
        os.replace(temporary, target)
        item["sha256"] = _sha256(target)
        item["resume_status"] = "regenerated" if previous else "created"


def replot(
    export: Path, project_dir: Path | None = None, output: Path | None = None,
    analysis_ids: tuple[str, ...] = (),
    invocation: dict[str, Any] | None = None,
) -> Path:
    """Replay available analyses, preserving an exact-identity output on resume."""
    source_manifest, registry, run_dir = _read_source(export)
    project, config_details = _configuration(run_dir, project_dir)
    ordered, selected = _analysis_selection(project.analyses_file, analysis_ids)
    build = _build_identity()
    identity = {
        "source_run_id": source_manifest.get("run_id"),
        "source_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "source_registry_sha256": _sha256(run_dir / "artifacts.json"),
        "configuration_sha256": config_details["sha256"],
        "selected_analysis_ids": sorted(selected),
        "dependency_analysis_ids": sorted({item.id for item in ordered} - selected),
        "build_sha256": build["content_sha256"],
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
    }
    identity["sha256"] = _json_hash(identity)
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = run_dir.parent / "replays" / f"{run_dir.name}_{stamp}"
    output = output.expanduser().resolve()
    if output == run_dir or output.is_relative_to(run_dir):
        raise ValueError("replay output must be outside the archived source run")
    manifest_path = output / "replot_manifest.json"
    previous: dict[str, Any] | None = None
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise ValueError("nonempty replay output has no manifest; choose a new output directory")
        previous = _read_json(manifest_path)
        if previous.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("older replay manifest lacks exact resume identity; choose a new output directory")
        if previous.get("identity") != identity:
            raise ValueError("replay identity differs; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    created = (previous or {}).get("created_utc", datetime.now(UTC).isoformat())
    atomic_json(manifest_path, {
        "schema": "pelecpost.saved_product_replay",
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "in_progress", "created_utc": created, "identity": identity,
        "output_artifacts": (previous or {}).get("output_artifacts", []),
    })
    with tempfile.TemporaryDirectory(prefix="flowviz-replay-", dir=output.parent) as temporary:
        stage = Path(temporary)
        unavailable, provenance = _render_selected(
            stage, project, ordered, selected, registry, run_dir,
        )
        roles = _role_checklist(registry, stage, selected)
        atomic_json(stage / "figure_role_checklist.json", {
            "schema": "pelecpost.figure_role_checklist", "schema_version": 2,
            "roles": roles,
        })
        records = _output_records(stage, provenance)
        _write_output(output, stage, records, previous)
    source_fingerprints = _source_fingerprints(source_manifest, run_dir)
    derived_sources = _derived_sources(registry, run_dir, {item.id for item in ordered})
    source_provenance = source_manifest.get("provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    limits = {
        item.id: {
            "variable": str(getattr(item, "variable", "")),
            "probe_indices": list(getattr(item, "probe_indices", ())),
            "frequency_max_hz": getattr(item, "frequency_max_hz", None),
            "reference_frequency_band_hz": (
                list(item.fft_ratio_plotting.reference_frequency_band_hz)
                if item.recipe == "case_comparison"
                and item.fft_ratio_plotting.reference_frequency_band_hz else None
            ),
            "wave_frequency_band_hz": (
                list(item.fft_ratio_plotting.wave_frequency_band_hz)
                if item.recipe == "case_comparison"
                and item.fft_ratio_plotting.wave_frequency_band_hz else None
            ),
        }
        for item in ordered
    }
    atomic_json(manifest_path, {
        "schema": "pelecpost.saved_product_replay",
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "partial" if unavailable else "completed",
        "created_utc": created,
        "completed_utc": datetime.now(UTC).isoformat(),
        "identity": identity,
        "source_run": str(run_dir),
        "source_run_id": source_manifest.get("run_id"),
        "source_manifest_schema": source_manifest.get("schema"),
        "source_manifest_schema_version": source_manifest.get("schema_version"),
        "source_build": {
            "analysis": source_provenance.get("analysis"),
            "software": source_provenance.get("software"),
            "solver": source_provenance.get("solver"),
        },
        "configuration": config_details,
        "invocation": invocation or {},
        "postprocessing_build": build,
        "selected_analysis_ids": sorted(selected),
        "dependency_analysis_ids": sorted({item.id for item in ordered} - selected),
        "source_input_fingerprints": source_fingerprints,
        "derived_source_artifacts": derived_sources,
        "unavailable_products": unavailable,
        "limits_by_analysis": limits,
        "figure_role_checklist": "figure_role_checklist.json",
        "figure_role_status_counts": {
            status: sum(role["status"] == status for role in roles)
            for status in sorted({role["status"] for role in roles})
        },
        "output_artifacts": records,
        "scientific_status": (
            "saved-product replay and supported processing sensitivity only; "
            "no CFD convergence or physical validation claim"
        ),
        "wave_scope": (
            "f-k figures are redrawn from saved power arrays; the original "
            "temporal/spatial transform is not recomputed"
        ),
    })
    return manifest_path
