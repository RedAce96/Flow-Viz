"""Redraw exported pressure probe and wave products without server probe files.

This reuses exact saved six-probe histories for the four selected probe plots.
The signed f-k panels reuse the exported full-line power arrays, since the
full 321-probe histories were not transferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pelecpost.analysis.comparison_figures import (
    render_komega_comparison,
    render_probe_panels,
)
from pelecpost.analysis.probe_plotting import build_probe_overlay_figure
from pelecpost.analysis.spectral import _plot_probe_time_fft, spectrum_from_signal
from pelecpost.config.loader import load_project
from pelecpost.visualization import save_figure_variants


def _analysis(project, analysis_id: str):
    return next(item for item in project.analyses_file.analyses if item.id == analysis_id)


def _save_overlay(
    project,
    stem: Path,
    x: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    x_label: str,
    y_label: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> Path:
    figure = build_probe_overlay_figure(
        project.analyses_file.presentation,
        x,
        values,
        labels,
        x_label,
        y_label,
        log_x=log_x,
        log_y=log_y,
    )
    return save_figure_variants(
        figure,
        stem,
        project.analyses_file.presentation.figure,
    )[0]


def _replot_case(export: Path, output: Path, project, case: str) -> tuple[Path, Path]:
    analysis = _analysis(project, f"{case}-pressure-spectrum")
    source = export / "data" / f"{case}-spectrum" / "probe_time_fft.npz"
    if not source.is_file():
        raise FileNotFoundError(f"exported pressure histories are missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        available = {int(index): column for column, index in enumerate(archive["probe_indices"])}
        selected = np.array(analysis.probe_indices, dtype=int)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"export does not contain requested probes {missing}")
        columns = [available[int(index)] for index in selected]
        raw_time = np.asarray(archive["raw_time_s"], dtype=float)
        raw_values = np.asarray(archive["raw_values"][:, columns], dtype=float)
        x_m = np.asarray(archive["x_m"][columns], dtype=float)
        unit = str(archive["signal_unit"])

    products = spectrum_from_signal(
        raw_time,
        raw_values,
        end_time_s=analysis.end_time_s,
        window=analysis.window,
        detrend=analysis.detrend,
        welch_segment_samples=analysis.welch_segment_samples,
        overlap_fraction=analysis.overlap_fraction,
        time_grid_policy=analysis.time_grid_policy,
        frequency_max_hz=analysis.frequency_max_hz,
        scratch_directory=None,
    )
    case_figures = output / "figures" / f"{case}-spectrum"
    case_data = output / "data" / f"{case}-spectrum"
    case_figures.mkdir(parents=True, exist_ok=True)
    case_data.mkdir(parents=True, exist_ok=True)
    labels = [f"Probe {index}" for index in selected]
    try:
        _save_overlay(
            project,
            case_figures / "raw_history_overlay",
            products["raw_time"],
            products["raw_values"],
            labels,
            "Time [s]",
            f"Pressure [{unit}]",
        )
        _save_overlay(
            project,
            case_figures / "method_ready_overlay",
            products["fft_time"],
            products["processed_values"],
            labels,
            "Time [s]",
            f"Pressure disturbance [{unit}]",
        )
        positive_fft = products["fft_frequency"] > 0.0
        _save_overlay(
            project,
            case_figures / "fft_overlay",
            products["fft_frequency"][positive_fft],
            products["fft_amplitude"][positive_fft],
            labels,
            "Frequency [Hz]",
            f"Amplitude [{unit}]",
            log_x=True,
            log_y=True,
        )
        positive_psd = products["welch_frequency"] > 0.0
        _save_overlay(
            project,
            case_figures / "psd_overlay",
            products["welch_frequency"][positive_psd],
            products["psd"][positive_psd],
            labels,
            "Frequency [Hz]",
            f"PSD [({unit})²/Hz]",
            log_x=True,
            log_y=True,
        )
        _plot_probe_time_fft(
            case_figures / "probe_time_fft.png",
            products["raw_time"],
            products["raw_values"],
            products["fft_time"],
            products["processed_values"],
            products["fft_frequency"],
            products["fft_amplitude"],
            x_m,
            selected,
            unit,
            analysis.detrend,
            analysis.window,
            dpi=int(project.analyses_file.presentation.figure.dpi),
        )
        signal_path = case_data / "probe_time_fft.npz"
        np.savez_compressed(
            signal_path,
            raw_time_s=products["raw_time"],
            raw_values=products["raw_values"],
            time_s=products["fft_time"],
            processed_values=products["processed_values"],
            frequency_hz=products["fft_frequency"],
            amplitude=products["fft_amplitude"],
            probe_indices=selected,
            x_m=x_m,
            signal_unit=np.array(unit),
        )
        psd_path = case_data / "stationary_spectrum.npz"
        np.savez_compressed(
            psd_path,
            frequency_hz=products["welch_frequency"],
            psd=products["psd"],
            probe_indices=selected,
            x_m=x_m,
            signal_unit=np.array(unit),
            psd_unit=np.array(f"({unit})²/Hz"),
        )
        return signal_path, psd_path
    finally:
        if products["cleanup"] is not None:
            products["cleanup"]()


def replot(export: Path, project_dir: Path, output: Path) -> Path:
    project = load_project(project_dir)
    signals: dict[str, Path] = {}
    psds: dict[str, Path] = {}
    for case in ("asym", "gaus"):
        signals[case], psds[case] = _replot_case(export, output, project, case)

    compare = output / "figures" / "compare-frequency"
    compare.mkdir(parents=True, exist_ok=True)
    for kind, name in (
        ("raw", "comparison_time.png"),
        ("raw_zoom", "comparison_time_zoom.png"),
        ("fft", "comparison_fft.png"),
        ("fft_shape", "comparison_fft_shape.png"),
        ("fft_shape_ratio", "comparison_fft_shape_ratio.png"),
    ):
        render_probe_panels(
            signals["asym"],
            signals["gaus"],
            "asym-pressure-spectrum.spectral.probe_signals",
            "gaus-pressure-spectrum.spectral.probe_signals",
            "pressure",
            compare / name,
            kind=kind,
        )
    render_probe_panels(
        psds["asym"],
        psds["gaus"],
        "asym-pressure-spectrum.spectral.psd",
        "gaus-pressure-spectrum.spectral.psd",
        "pressure",
        compare / "comparison_psd.png",
        kind="psd",
    )

    wave = _analysis(project, "asym-pressure-wave")
    compare_wave = output / "figures" / "compare-wave"
    compare_wave.mkdir(parents=True, exist_ok=True)
    wave_sources = [
        export / "data" / f"{case}-wavenumber" / "komega_spectrum.npz" for case in ("asym", "gaus")
    ]
    render_komega_comparison(
        wave_sources[0],
        wave_sources[1],
        "asym-pressure-wave.wave.komega",
        "gaus-pressure-wave.wave.komega",
        compare_wave / "comparison_fk.png",
        compare_wave / "comparison_signed_k.png",
        (wave.frequency_min_hz, wave.frequency_max_hz),
        "(Pa)²",
    )

    manifest = output / "replot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_export": str(export.resolve()),
                "analyses_file": str((project_dir / "analyses.yaml").resolve()),
                "variable": "pressure",
                "exact_history_scope": "first 10 µs at four selected probes per case",
                "wave_scope": "redrawn from exported 321-probe f-k power; underlying transform not recomputed",
                "selected_probes": list(_analysis(project, "asym-pressure-spectrum").probe_indices),
                "window": _analysis(project, "asym-pressure-spectrum").window,
                "wave_frequency_limit_hz": wave.frequency_max_hz,
                "probe_frequency_limit_hz": _analysis(
                    project, "asym-pressure-spectrum"
                ).frequency_max_hz,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=root / "Delta_Outputs")
    parser.add_argument("--project", type=Path, default=root / "Flow-Viz" / "Post-Processing")
    parser.add_argument(
        "--output", type=Path, default=root / "Delta_Outputs" / "replotted-frequency-wave"
    )
    args = parser.parse_args()
    print(replot(args.export, args.project, args.output))


if __name__ == "__main__":
    main()
