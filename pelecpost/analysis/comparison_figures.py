"""Coordinate-labelled figures for paired probe and directional-wave products."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pelecpost.config.models import ComparisonAlignment

from .comparison_alignment import align_frequency_columns, align_values

_FFT_RATIO_KINDS = {
    "fft_amplitude_ratio": ("absolute", "db"),
    "fft_amplitude_linear_ratio": ("absolute", "linear"),
    "fft_shape_ratio": ("unit_l2", "db"),
    "fft_shape_linear_ratio": ("unit_l2", "linear"),
}


def _bin_edges(values: np.ndarray) -> np.ndarray:
    """Return cell edges for a strictly increasing center coordinate."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("plot coordinates must contain at least two increasing values")
    interior = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - 0.5 * (values[1] - values[0])], interior,
                           [values[-1] + 0.5 * (values[-1] - values[-2])]))


def _linear_ratio_cmap(vmin: float, vmax: float):
    """Diverging map with equality (ratio 1) at its true linear position."""
    equality = (1.0 - vmin) / (vmax - vmin)
    cmap = LinearSegmentedColormap.from_list(
        "ratio_linear", [(0.0, "#2166ac"), (equality, "white"), (1.0, "#b2182b")]
    )
    return cmap.with_extremes(bad="#d0d0d0")


def _amplitude_unit(power_unit: str) -> str:
    """Convert a squared-signal unit label into its amplitude unit."""
    unit = str(power_unit).strip()
    if unit.endswith("^2"):
        unit = unit[:-2].strip()
    elif unit.endswith("²"):
        unit = unit[:-1].strip()
    if unit.startswith("(") and unit.endswith(")"):
        unit = unit[1:-1].strip()
    return unit or "signal units"


def _save_comparison_figure(figure, destination: Path, dpi: int) -> None:
    """Save the presentation PNG and a vector PDF companion."""
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if destination.suffix.lower() == ".png":
        figure.savefig(destination.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


def _fk_ratio_components(
    baseline_power: np.ndarray,
    comparison_power: np.ndarray,
    normalization: str,
    minimum_relative_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return ratio-domain powers and one scientifically consistent mask."""
    baseline = np.maximum(np.asarray(baseline_power, dtype=float), 0.0)
    comparison = np.maximum(np.asarray(comparison_power, dtype=float), 0.0)
    if normalization == "unit_l2":
        total_a, total_b = float(np.sum(baseline)), float(np.sum(comparison))
        ratio_a = baseline / total_a if total_a > 0.0 else np.zeros_like(baseline)
        ratio_b = comparison / total_b if total_b > 0.0 else np.zeros_like(comparison)
        peak_a = float(np.max(ratio_a, initial=0.0))
        peak_b = float(np.max(ratio_b, initial=0.0))
        meaningful = (
            (ratio_a >= peak_a * minimum_relative_amplitude**2)
            & (ratio_b >= peak_b * minimum_relative_amplitude**2)
            & (ratio_a > 0.0)
            & (ratio_b > 0.0)
        )
        return ratio_a, ratio_b, meaningful, "normalized_own_peak"
    if normalization != "absolute":
        raise ValueError("ratio normalization must be absolute or unit_l2")
    reference = float(max(np.max(baseline, initial=0.0), np.max(comparison, initial=0.0)))
    meaningful = (
        (baseline >= reference * minimum_relative_amplitude**2)
        & (comparison >= reference * minimum_relative_amplitude**2)
        & (baseline > 0.0)
        & (comparison > 0.0)
    )
    return baseline, comparison, meaningful, "shared_raw_peak"


def _same_axis(first: np.ndarray, second: np.ndarray, atol: float) -> bool:
    return first.shape == second.shape and np.allclose(first, second, rtol=0.0, atol=atol)


def _case_name(label: str) -> str:
    name = label.split(".spectral.")[0].split(".wave.")[0]
    if name.startswith("asym-"):
        return "Asymmetric"
    if name.startswith("gaus-"):
        return "Gaussian"
    return name


def _probe_activity(archive: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Return per-probe disturbance strength from a saved probe product."""
    for key in ("amplitude", "psd"):
        if key in archive:
            values = np.asarray(archive[key], dtype=float)
            return np.max(np.maximum(values, 0.0), axis=0)
    if "raw_values" in archive:
        values = np.asarray(archive["raw_values"], dtype=float)
        return np.ptp(values, axis=0)
    return None


def render_probe_panels(
    first_path: Path,
    second_path: Path,
    first_label: str,
    second_label: str,
    variable: str,
    destination: Path,
    *,
    kind: str,
    frequency_limit_hz: float | None = None,
    minimum_relative_amplitude: float = 0.01,
    reference_frequency_band_hz: tuple[float, float] | None = None,
    dpi: int = 300,
    scale_policy: str = "independent",
    probe_groups: tuple[tuple[int, ...], ...] = (),
    alignment: ComparisonAlignment | None = None,
) -> Path | None:
    """Draw both cases at each shared probe, using physical time or frequency."""
    if not 0.0 < minimum_relative_amplitude < 1.0:
        raise ValueError("minimum_relative_amplitude must lie in (0, 1)")
    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        x_first = np.asarray(first["x_m"], dtype=float)
        x_second = np.asarray(second["x_m"], dtype=float)
        if not _same_axis(x_first, x_second, 1e-12) or not len(x_first):
            return None
        first_activity, second_activity = _probe_activity(first), _probe_activity(second)
        if first_activity is not None and second_activity is not None:
            active = (first_activity > 0.0) | (second_activity > 0.0)
            order = (
                np.flatnonzero(active)[np.argsort(x_first[active])]
                if np.any(active)
                else np.argsort(x_first)
            )
        else:
            order = np.argsort(x_first)
        if probe_groups:
            available = {int(index): int(column) for column, index in enumerate(np.asarray(first["probe_indices"], dtype=int))}
            requested = [index for group in probe_groups for index in group]
            missing = sorted(set(requested) - set(available))
            if missing:
                raise ValueError(f"comparison probe group requests unavailable probes {missing}; available={sorted(available)}")
            chosen = np.asarray([available[index] for index in requested], dtype=int)
        else:
            chosen = order[np.unique(np.linspace(0, len(order) - 1, min(len(order), 6), dtype=int))]
        # An explicit group order is part of the presentation contract.  Do
        # not apply a second outside-in pairing pass: callers use
        # [120, 200, 150, 170] to place the mirrored pairs on rows.
        if not probe_groups:
            paired = []
            left, right = 0, len(chosen) - 1
            while left <= right:
                paired.append(int(chosen[left]))
                if right != left:
                    paired.append(int(chosen[right]))
                left += 1
                right -= 1
            chosen = np.asarray(paired, dtype=int)
        rows = (len(chosen) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, max(3.5, 2.7 * rows)), squeeze=False)
        specs = {
            "raw": ("raw_time_s", "raw_values", "Time [µs]", "Recorded history"),
            "raw_zoom": ("raw_time_s", "raw_values", "Time [µs]", "First 2 µs"),
            "fft": ("frequency_hz", "amplitude", "Frequency [Hz]", "FFT amplitude"),
            "fft_shape": (
                "frequency_hz",
                "amplitude",
                "Frequency [Hz]",
                "FFT shape (unit L2 norm)",
            ),
            "fft_shape_ratio": (
                "frequency_hz",
                "amplitude",
                "Frequency [Hz]",
                "Normalized FFT shape ratio",
            ),
            "fft_amplitude_ratio": (
                "frequency_hz",
                "amplitude",
                "Frequency [Hz]",
                "FFT amplitude ratio",
            ),
            "fft_amplitude_linear_ratio": (
                "frequency_hz",
                "amplitude",
                "Frequency [Hz]",
                "Linear FFT amplitude ratio",
            ),
            "fft_shape_linear_ratio": (
                "frequency_hz",
                "amplitude",
                "Frequency [Hz]",
                "Linear normalized FFT shape ratio",
            ),
            "psd": ("frequency_hz", "psd", "Frequency [Hz]", "Welch PSD"),
        }
        aligned_amplitude_a: np.ndarray | None = None
        aligned_amplitude_b: np.ndarray | None = None
        aligned_frequency: np.ndarray | None = None
        if kind in _FFT_RATIO_KINDS:
            aligned_amplitude_a, aligned_amplitude_b, aligned_frequency, _ = (
                align_frequency_columns(
                    np.asarray(first["amplitude"], dtype=float),
                    np.asarray(second["amplitude"], dtype=float),
                    np.asarray(first["frequency_hz"], dtype=float),
                    np.asarray(second["frequency_hz"], dtype=float),
                    alignment,
                )
            )
        axis_key, value_key, x_label, title = specs[kind]
        unit_key = "psd_unit" if kind == "psd" else "signal_unit"
        unit = str(first[unit_key]) if unit_key in first else ""
        baseline_name, comparison_name = _case_name(first_label), _case_name(second_label)
        ratio_description = (
            ("Normalized FFT amplitude" if kind.startswith("fft_shape") else "FFT amplitude")
            + f" ratio ({comparison_name} / {baseline_name})"
        )
        y_label = (
            f"{variable.capitalize()} [{unit}]"
            if kind.startswith("raw") and unit
            else f"Amplitude [{unit}]"
            if kind == "fft" and unit
            else f"PSD [{unit}]"
            if kind == "psd" and unit
            else "Normalized amplitude"
            if kind == "fft_shape"
            else f"{ratio_description} [×]"
            if kind in _FFT_RATIO_KINDS and _FFT_RATIO_KINDS[kind][1] == "linear"
            else f"{ratio_description} [dB]"
            if kind in _FFT_RATIO_KINDS
            else variable.capitalize()
        )
        if kind in _FFT_RATIO_KINDS:
            y_label = (
                "Amplitude ratio G/A [−]"
                if _FFT_RATIO_KINDS[kind][1] == "linear"
                else "Amplitude ratio G/A [dB]"
            )
        for axis, column in zip(axes.flat, chosen):
            if kind in _FFT_RATIO_KINDS:
                if (
                    aligned_frequency is None
                    or aligned_amplitude_a is None
                    or aligned_amplitude_b is None
                ):
                    raise RuntimeError("FFT ratio alignment was not prepared")
                all_x = aligned_frequency
                usable = all_x > 0.0
                reference_use = usable.copy()
                if reference_frequency_band_hz is not None:
                    band_lo, band_hi = reference_frequency_band_hz
                    reference_use &= (all_x >= band_lo) & (all_x <= band_hi)
                if frequency_limit_hz is not None:
                    usable &= all_x <= frequency_limit_hz
                x = all_x[usable]
                a_all = np.maximum(aligned_amplitude_a[:, column], 0)
                b_all = np.maximum(aligned_amplitude_b[:, column], 0)
                a = a_all[usable]
                b = b_all[usable]
                a_ref, b_ref = a_all[reference_use], b_all[reference_use]
                peak_a, peak_b = np.max(a_ref, initial=0.0), np.max(b_ref, initial=0.0)
                if peak_a > 0 and peak_b > 0:
                    normalization, scale = _FFT_RATIO_KINDS[kind]
                    if normalization == "unit_l2":
                        norm_a = np.linalg.norm(a_ref)
                        norm_b = np.linalg.norm(b_ref)
                        a = a / norm_a if norm_a > 0 else a
                        b = b / norm_b if norm_b > 0 else b
                    meaningful = (
                        (a_all[usable] >= peak_a * minimum_relative_amplitude)
                        & (b_all[usable] >= peak_b * minimum_relative_amplitude)
                    )
                    ratio = np.full(len(x), np.nan)
                    ratio[meaningful] = (
                        b[meaningful] / a[meaningful]
                        if scale == "linear"
                        else 20.0 * np.log10(b[meaningful] / a[meaningful])
                    )
                    axis.plot(x, ratio, color="#6a3d9a", linewidth=1.8)
                    missing = ~meaningful
                    changes = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
                    support_axis = axis.inset_axes([0.0, -0.13, 1.0, 0.045], transform=axis.transAxes)
                    support_axis.set_xscale("log")
                    support_axis.set_xlim(x[0], x[-1])
                    support_axis.set_ylim(0.0, 1.0)
                    support_axis.set_yticks([])
                    support_axis.set_xticks([])
                    for spine in support_axis.spines.values():
                        spine.set_visible(False)
                    for start, stop in changes.reshape(-1, 2):
                        if stop > start:
                            support_axis.axvspan(
                                x[start], x[min(stop - 1, len(x) - 1)],
                                color="#bdbdbd", alpha=0.9, linewidth=0,
                            )
                    axis.axhline(
                        1.0 if scale == "linear" else 0.0,
                        color="0.5",
                        linewidth=0.8,
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No positive spectral amplitude",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
                    axis.set_yticks([])
            else:
                for archive, label, color, linestyle in (
                    (first, first_label, "#1f77b4", "-"),
                    (second, second_label, "#ff7f0e", "--"),
                ):
                    all_x = np.asarray(archive[axis_key], dtype=float)
                    x = all_x.copy()
                    y = np.asarray(archive[value_key][:, column], dtype=float)
                    if kind.startswith("raw"):
                        x = x * 1e6
                    else:
                        usable = x > 0.0
                        if frequency_limit_hz is not None:
                            usable &= x <= frequency_limit_hz
                        # Norms are defined on the complete source reference
                        # band before applying a display cutoff.
                        reference_use = all_x > 0.0
                        if reference_frequency_band_hz is not None:
                            reference_use &= (all_x >= reference_frequency_band_hz[0]) & (all_x <= reference_frequency_band_hz[1])
                        x, y = x[usable], y[usable]
                        if kind == "fft_shape":
                            norm = float(np.linalg.norm(np.asarray(archive[value_key], dtype=float)[reference_use, column]))
                            y = y / norm if norm > 0 else y
                        y = np.where(y > 0.0, y, np.nan)
                    axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8, label=_case_name(label))
            index = int(first["probe_indices"][column]) if "probe_indices" in first else int(column)
            axis.set_title(f"Probe {index} · x={x_first[column] * 100:.3f} cm")
            axis.set_xlabel(x_label)
            if column == chosen[0]:
                axis.set_ylabel(y_label)
            if kind.startswith("raw"):
                if kind == "raw_zoom":
                    axis.set_xlim(0.0, 2.0)
            else:
                axis.set_xscale("log")
                frequency = np.asarray(first[axis_key], dtype=float)
                positive_frequency = frequency[frequency > 0.0]
                if frequency_limit_hz is not None:
                    positive_frequency = positive_frequency[
                        positive_frequency <= frequency_limit_hz
                    ]
                if len(positive_frequency):
                    axis.set_xlim(positive_frequency[0], positive_frequency[-1])
                if kind not in _FFT_RATIO_KINDS and any(
                    np.isfinite(line.get_ydata()).any() for line in axis.lines
                ):
                    axis.set_yscale("log")
                elif kind not in _FFT_RATIO_KINDS:
                    axis.text(
                        0.5,
                        0.5,
                        "No positive spectral amplitude",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
            axis.grid(True, alpha=0.2)
        if scale_policy in {"group", "all"}:
            groups = [tuple(group) for group in probe_groups] if probe_groups else [tuple(int(i) for i in np.asarray(first["probe_indices"])[chosen])]
            if scale_policy == "all":
                groups = [tuple(index for group in groups for index in group)]
            probe_to_axis = {int(np.asarray(first["probe_indices"])[column]): axis for axis, column in zip(axes.flat, chosen)}
            for group in groups:
                group_axes = [probe_to_axis[index] for index in group if index in probe_to_axis]
                values_for_limits: list[np.ndarray] = []
                for axis in group_axes:
                    for line in axis.lines:
                        values_for_limits.append(np.asarray(line.get_ydata(), dtype=float))
                finite = np.concatenate([v[np.isfinite(v)] for v in values_for_limits if np.any(np.isfinite(v))]) if values_for_limits else np.array([])
                if finite.size:
                    # Log-amplitude panels can only accept strictly positive
                    # limits.  Use multiplicative padding there; linear
                    # padding around a small positive minimum can otherwise
                    # cross zero and silently discard the requested scale.
                    log_group = any(axis.get_yscale() == "log" for axis in group_axes)
                    if log_group:
                        positive = finite[finite > 0.0]
                        if positive.size == 0:
                            continue
                        lo = float(np.min(positive)) / 1.08
                        hi = float(np.max(positive)) * 1.08
                    else:
                        lo, hi = float(np.min(finite)), float(np.max(finite))
                        pad = max((hi - lo) * 0.06, max(abs(lo), abs(hi), 1.0) * 1e-3)
                        lo, hi = lo - pad, hi + pad
                    for axis in group_axes:
                        axis.set_ylim(lo, hi)
        for axis in list(axes.flat)[len(chosen) :]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"{variable.capitalize()}: {title}", y=0.995)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
        if kind in _FFT_RATIO_KINDS:
            fig.text(
                0.5,
                0.94,
                f"{comparison_name}/{baseline_name}; ratio omitted where either amplitude "
                f"is below {minimum_relative_amplitude:.0%} of its own peak",
                ha="center",
                fontsize=9,
            )
        fig.tight_layout(
            rect=(0, 0, 1, 0.90)
        )
        _save_comparison_figure(fig, destination, dpi)
        plt.close(fig)
    return destination


def render_komega_comparison(
    first_path: Path,
    second_path: Path,
    first_label: str,
    second_label: str,
    map_path: Path,
    spectrum_path: Path | None,
    frequency_band_hz: tuple[float, float],
    power_unit: str = "variable²",
    *,
    spectral_display: str = "amplitude",
    ratio_scale: str = "db",
    ratio_normalization: str = "absolute",
    minimum_relative_amplitude: float = 0.01,
    dpi: int = 300,
    linear_ratio_limits: tuple[float, float] = (0.0, 4.0),
    linear_ratio_color_scale: str = "linear",
    db_ratio_limits: tuple[float, float] = (-12.0, 12.0),
    variable: str | None = None,
    wavenumber_limits_rad_m: tuple[float, float] | None = None,
    alignment: ComparisonAlignment | None = None,
) -> tuple[Path, Path | None] | None:
    """Show signed f-k power, a gated amplitude ratio, and optional k shapes."""
    if ratio_scale not in ("db", "linear"):
        raise ValueError("ratio_scale must be db or linear")
    if spectral_display not in ("amplitude", "relative_db"):
        raise ValueError("spectral_display must be amplitude or relative_db")
    if ratio_normalization not in ("absolute", "unit_l2"):
        raise ValueError("ratio_normalization must be absolute or unit_l2")
    if not 0.0 < minimum_relative_amplitude < 1.0:
        raise ValueError("minimum_relative_amplitude must lie in (0, 1)")
    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        raw_f_a = np.asarray(first["frequency_hz"], dtype=float)
        raw_f_b = np.asarray(second["frequency_hz"], dtype=float)
        raw_k_a = np.asarray(first["wavenumber_rad_m"], dtype=float)
        raw_k_b = np.asarray(second["wavenumber_rad_m"], dtype=float)
        power_a, power_b, alignment_details = align_values(
            np.asarray(first["power"], dtype=float),
            np.asarray(second["power"], dtype=float),
            {"frequency": raw_f_a, "wavenumber": raw_k_a},
            {"frequency": raw_f_b, "wavenumber": raw_k_b},
            "wave.komega",
            alignment or ComparisonAlignment(),
            {"power": ("frequency", "wavenumber")},
            "power",
        )
        if "amplitude" in first.files and "amplitude" in second.files:
            amplitude_a, amplitude_b, _ = align_values(
                np.asarray(first["amplitude"], dtype=float),
                np.asarray(second["amplitude"], dtype=float),
                {"frequency": raw_f_a, "wavenumber": raw_k_a},
                {"frequency": raw_f_b, "wavenumber": raw_k_b},
                "wave.komega",
                alignment or ComparisonAlignment(),
                {"amplitude": ("frequency", "wavenumber")},
                "amplitude",
            )
        else:
            amplitude_a = amplitude_b = None
        f = np.asarray(alignment_details["frequency"]["coordinate_values"], dtype=float)
        k = np.asarray(alignment_details["wavenumber"]["coordinate_values"], dtype=float)
        lo, hi = frequency_band_hz
        # The finite-record FFT axis can land a few floating-point ulps below
        # a configured bin boundary. Keep that bin, but never include DC on
        # a logarithmic frequency axis.
        use = (
            (f > 0.0)
            & (f >= lo - max(1e-6, abs(lo) * 1e-9))
            & (f <= hi + max(1e-6, abs(hi) * 1e-9))
        )
        if np.count_nonzero(use) < 2:
            return None
        use_indices = np.flatnonzero(use)
        a = np.maximum(power_a[use], 0.0)
        b = np.maximum(power_b[use], 0.0)
        reference = float(max(np.max(a), np.max(b)))
        if reference <= 0.0:
            return None
        floor = reference * 1e-6
        relative_a = np.clip(10.0 * np.log10(np.maximum(a, floor) / reference), -60, 0)
        relative_b = np.clip(10.0 * np.log10(np.maximum(b, floor) / reference), -60, 0)
        display_amplitude_a = (
            np.maximum(np.asarray(amplitude_a, dtype=float)[use], 0.0)
            if amplitude_a is not None else np.sqrt(a)
        )
        display_amplitude_b = (
            np.maximum(np.asarray(amplitude_b, dtype=float)[use], 0.0)
            if amplitude_b is not None else np.sqrt(b)
        )
        amplitude_max = float(max(np.max(display_amplitude_a), np.max(display_amplitude_b)))
        ratio = np.full(a.shape, np.nan)
        ratio_a, ratio_b, meaningful, support_basis = _fk_ratio_components(
            a, b, ratio_normalization, minimum_relative_amplitude
        )
        power_ratio = ratio_b[meaningful] / ratio_a[meaningful]
        ratio[meaningful] = (
            np.sqrt(power_ratio)
            if ratio_scale == "linear"
            else 10.0 * np.log10(power_ratio)
        )
        k_edges = _bin_edges(k) / 1e3
        # Construct frequency edges on the complete native grid, then crop
        # cells. This preserves the true boundary of the first selected bin
        # on the logarithmic axis.
        full_f_edges = _bin_edges(f) / 1e6
        f_edges = full_f_edges[use_indices[0]:use_indices[-1] + 2]
        extent = [k_edges[0], k_edges[-1], f_edges[0], f_edges[-1]]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5.8),
            sharey=True,
            constrained_layout=True,
        )
        for axis, values, title in (
            (
                axes[0], display_amplitude_a if spectral_display == "amplitude" else relative_a,
                _case_name(first_label),
            ),
            (
                axes[1], display_amplitude_b if spectral_display == "amplitude" else relative_b,
                _case_name(second_label),
            ),
        ):
            image_cmap = plt.get_cmap("viridis").with_extremes(bad="#eeeeee")
            if spectral_display == "amplitude":
                image = axis.pcolormesh(
                    k_edges, f_edges, values, shading="flat", cmap=image_cmap,
                    vmin=0.0, vmax=amplitude_max if amplitude_max > 0.0 else 1.0,
                )
            else:
                image = axis.pcolormesh(
                    k_edges, f_edges, values, shading="flat", cmap=image_cmap,
                    vmin=-60, vmax=0,
                )
            axis.set_title(title)
        if ratio_scale == "linear":
            if linear_ratio_color_scale == "log":
                ratio_norm = LogNorm(vmin=max(linear_ratio_limits[0], 1.0e-6),
                                     vmax=linear_ratio_limits[1])
                ratio_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d0d0d0")
            else:
                ratio_norm = Normalize(vmin=linear_ratio_limits[0], vmax=linear_ratio_limits[1])
                ratio_cmap = _linear_ratio_cmap(*linear_ratio_limits)
            difference = axes[2].pcolormesh(
                k_edges, f_edges, ratio, shading="flat", cmap=ratio_cmap, norm=ratio_norm,
            )
        else:
            ratio_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d0d0d0")
            difference = axes[2].pcolormesh(
                k_edges, f_edges, ratio, shading="flat", cmap=ratio_cmap,
                vmin=db_ratio_limits[0], vmax=db_ratio_limits[1],
            )
        shape = "normalized " if ratio_normalization == "unit_l2" else ""
        suffix = " [dB]" if ratio_scale == "db" else " [×]"
        axes[2].set_title(
            f"Gaussian / Asymmetric {shape}amplitude{suffix}"
        )
        fig.supxlabel("Signed wavenumber [10³ rad/m]")
        axes[0].set_ylabel("Frequency [MHz]")
        axes[0].set_yscale("log")
        # Cell edges describe geometry; requested display limits describe the
        # scientific band and must not silently grow by half a bin.
        axes[0].set_ylim(lo / 1e6, hi / 1e6)
        axes[0].set_yscale("log")
        if wavenumber_limits_rad_m is not None:
            for axis in axes:
                axis.set_xlim(wavenumber_limits_rad_m[0] / 1e3, wavenumber_limits_rad_m[1] / 1e3)
        ticks = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
        ticks = ticks[(ticks >= extent[2] - 1e-9) & (ticks <= extent[3] + 1e-9)]
        axes[0].set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        source_label = (
            f"Spectral amplitude [{_amplitude_unit(power_unit)}]"
            if spectral_display == "amplitude"
            else "Relative power [dB]"
        )
        fig.colorbar(image, ax=axes[:2], label=source_label)
        finite_ratio = ratio[np.isfinite(ratio)]
        ratio_below = np.any(finite_ratio < linear_ratio_limits[0]) if ratio_scale == "linear" else np.any(finite_ratio < db_ratio_limits[0])
        ratio_above = np.any(finite_ratio > linear_ratio_limits[1]) if ratio_scale == "linear" else np.any(finite_ratio > db_ratio_limits[1])
        fig.colorbar(
            difference, ax=axes[2],
            label=f"{shape.capitalize()}amplitude ratio G/A{suffix}",
            ticks=(list(np.arange(np.ceil(linear_ratio_limits[0]), linear_ratio_limits[1] + 0.5, 1.0))
                   if ratio_scale == "linear" else None),
            format="%.2g" if ratio_scale == "linear" else None,
            extend=("both" if ratio_below and ratio_above else "min" if ratio_below else "max" if ratio_above else "neither"),
        )
        axes[2].legend(
            handles=[Patch(facecolor="#d0d0d0", edgecolor="none", label="Unsupported ratio mask")],
            loc="upper right", fontsize=8, frameon=True,
        )
        fig.suptitle(
            f"{variable.capitalize() if variable else 'f–k'} comparison · "
            f"{_case_name(first_label)} vs {_case_name(second_label)} · "
            f"{lo / 1e6:.3g}–{hi / 1e6:.3g} MHz", fontsize=14
        )
        _save_comparison_figure(fig, map_path, dpi)
        plt.close(fig)
        displayed_limits = db_ratio_limits if ratio_scale == "db" else linear_ratio_limits
        ratio_stats = {
            "schema": "pelecpost.fk_ratio_statistics",
            "schema_version": 2,
            "status": "available" if finite_ratio.size else "unavailable_no_supported_bins",
            "frequency_band_hz": [lo, hi],
            "coordinate_units": {"frequency": "Hz", "wavenumber": "rad/m"},
            "normalization_domain": "selected_frequency_band",
            "minimum_relative_amplitude": minimum_relative_amplitude,
            "ratio_scale": ratio_scale,
            "ratio_units": "dB" if ratio_scale == "db" else "1",
            "ratio_limits": list(displayed_limits),
            "ratio_color_scale": "linear" if ratio_scale == "db" else linear_ratio_color_scale,
            "ratio_normalization": ratio_normalization,
            "alignment": alignment_details,
            "accepted_min": float(np.min(finite_ratio)) if finite_ratio.size else None,
            "accepted_max": float(np.max(finite_ratio)) if finite_ratio.size else None,
            "masked_fraction": float(1.0 - finite_ratio.size / ratio.size),
            "accepted_count": int(finite_ratio.size),
            "total_count": int(ratio.size),
            "clipped_below_fraction": (
                float(np.mean(finite_ratio < displayed_limits[0]))
                if finite_ratio.size else None
            ),
            "clipped_above_fraction": (
                float(np.mean(finite_ratio > displayed_limits[1]))
                if finite_ratio.size else None
            ),
            "support_basis": support_basis,
            "support_rule": (
                f"both normalized powers >= {minimum_relative_amplitude**2:.3g} "
                "of their own normalized maxima"
                if support_basis == "normalized_own_peak"
                else f"both powers >= {minimum_relative_amplitude**2:.3g} of shared band maximum"
            ),
        }
        map_path.with_suffix(".json").write_text(json.dumps(ratio_stats, indent=2) + "\n", encoding="utf-8")

        if spectrum_path is None:
            return map_path, None
        power_a = np.sum(a, axis=0)
        power_b = np.sum(b, axis=0)
        if spectral_display == "amplitude":
            display_a = np.sqrt(np.maximum(power_a, 0.0))
            display_b = np.sqrt(np.maximum(power_b, 0.0))
            amplitude_max = float(max(np.max(display_a), np.max(display_b)))
            amplitude_unit = _amplitude_unit(power_unit)
        else:
            display_a, display_b = power_a, power_b
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for values, label, color in (
            (display_a, _case_name(first_label), "#1f77b4"),
            (display_b, _case_name(second_label), "#ff7f0e"),
        ):
            axes[0].plot(k / 1e3, np.where(values > 0, values, np.nan), color=color, label=label)
            normalized = values / np.sum(values) if np.sum(values) > 0 else values
            axes[1].plot(
                k / 1e3, np.where(normalized > 0, normalized, np.nan), color=color, label=label
            )
        if spectral_display == "amplitude":
            axes[0].set_ylim(0.0, amplitude_max if amplitude_max > 0.0 else 1.0)
        else:
            axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        display_power_unit = power_unit.replace("(", "").replace(")", "").replace("^2", "²")
        axes[0].set_ylabel(
            f"Band-integrated amplitude [{amplitude_unit}]"
            if spectral_display == "amplitude"
            else f"Band-summed squared amplitude [{display_power_unit}]"
        )
        axes[1].set_ylabel("Fraction of band sum per k bin [−]")
        amplitude_ratio = np.full(power_a.shape, np.nan)
        support_a = power_a >= np.max(power_a, initial=0.0) * minimum_relative_amplitude**2
        support_b = power_b >= np.max(power_b, initial=0.0) * minimum_relative_amplitude**2
        support = support_a & support_b & (power_a > 0.0) & (power_b > 0.0)
        amplitude_ratio[support] = np.sqrt(power_b[support] / power_a[support])
        axes[2].plot(k / 1e3, amplitude_ratio, color="#6a3d9a", label="Gaussian / Asymmetric")
        axes[2].axhline(1.0, color="0.5", linewidth=0.8)
        axes[2].set_ylabel("Band amplitude ratio [−]")
        axes[2].set_xlabel("Signed wavenumber kₓ [10³ rad m⁻¹]")
        for axis in axes:
            axis.grid(True, alpha=0.2)
            axis.legend()
        fig.suptitle(
            f"{variable.capitalize() if variable else 'Signed'} k spectrum · "
            f"{_case_name(first_label)} vs {_case_name(second_label)} · "
            f"Signed k spectrum, {lo / 1e6:.2g}–{hi / 1e6:.2g} MHz\n"
            "S(k)=sum over selected positive-frequency bins of |C(f,k)|²"
        )
        fig.tight_layout()
        _save_comparison_figure(fig, spectrum_path, dpi)
        plt.close(fig)
    return map_path, spectrum_path


def render_komega_frequency_slices(
    first_path: Path,
    second_path: Path,
    first_label: str,
    second_label: str,
    destination: Path,
    frequency_band_hz: tuple[float, float],
    *,
    ratio_normalization: str = "absolute",
    minimum_relative_amplitude: float = 0.01,
    dpi: int = 300,
    variable: str | None = None,
    alignment: ComparisonAlignment | None = None,
) -> Path | None:
    """Plot native-bin signed-k slices near 0.5, 1, 2, and 3 MHz."""
    with np.load(first_path, allow_pickle=False) as first, np.load(second_path, allow_pickle=False) as second:
        power_a, power_b, details = align_values(
            np.asarray(first["power"], dtype=float),
            np.asarray(second["power"], dtype=float),
            {
                "frequency": np.asarray(first["frequency_hz"], dtype=float),
                "wavenumber": np.asarray(first["wavenumber_rad_m"], dtype=float),
            },
            {
                "frequency": np.asarray(second["frequency_hz"], dtype=float),
                "wavenumber": np.asarray(second["wavenumber_rad_m"], dtype=float),
            },
            "wave.komega",
            alignment or ComparisonAlignment(),
            {"power": ("frequency", "wavenumber")},
            "power",
        )
        frequency = np.asarray(details["frequency"]["coordinate_values"], dtype=float)
        k = np.asarray(details["wavenumber"]["coordinate_values"], dtype=float)
        power_a = np.maximum(power_a, 0.0)
        power_b = np.maximum(power_b, 0.0)
    use = (frequency >= frequency_band_hz[0]) & (frequency <= frequency_band_hz[1])
    indices = np.flatnonzero(use)
    if not len(indices):
        return None
    requested = np.array([0.5e6, 1.0e6, 2.0e6, 3.0e6])
    chosen: list[int] = []
    for target in requested:
        candidates = indices[np.argmin(np.abs(frequency[indices] - target))]
        if int(candidates) not in chosen:
            chosen.append(int(candidates))
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, max(6.0, 2.5 * len(chosen))), sharex=True)
    axes = np.atleast_1d(axes)
    # Norms and support are defined over the complete selected comparison
    # band, then reused by every native-bin slice.  A detail slice must not
    # acquire a new normalization or a new noise/support decision.
    band_power_a = power_a[indices]
    band_power_b = power_b[indices]
    ratio_band_a, ratio_band_b, support_band, _support_basis = _fk_ratio_components(
        band_power_a,
        band_power_b,
        ratio_normalization,
        minimum_relative_amplitude,
    )
    local_row = {int(source): row for row, source in enumerate(indices)}
    for axis, index in zip(axes, chosen):
        row = local_row[index]
        a, b = ratio_band_a[row], ratio_band_b[row]
        ratio = np.full(a.shape, np.nan)
        support = support_band[row]
        ratio[support] = np.sqrt(b[support] / a[support])
        axis.plot(k / 1e3, ratio, color="#6a3d9a", linewidth=1.6)
        axis.axhline(1.0, color="0.5", linewidth=0.8)
        axis.set_ylabel(f"{frequency[index] / 1e6:.3g} MHz\nG/A [−]")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Signed wavenumber kₓ [10³ rad m⁻¹]")
    fig.suptitle(f"{variable.capitalize() if variable else 'f–k'} amplitude-ratio slices · {_case_name(second_label)} / {_case_name(first_label)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_comparison_figure(fig, destination, dpi)
    plt.close(fig)
    return destination


def render_komega_amplitude_frequency_slices(
    first_path: Path,
    second_path: Path,
    first_label: str,
    second_label: str,
    destination: Path,
    frequency_band_hz: tuple[float, float],
    *,
    power_unit: str = "variable²",
    dpi: int = 300,
    variable: str | None = None,
    alignment: ComparisonAlignment | None = None,
) -> Path | None:
    """Overlay physical spectral amplitudes at selected frequencies."""
    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        raw_f_a = np.asarray(first["frequency_hz"], dtype=float)
        raw_f_b = np.asarray(second["frequency_hz"], dtype=float)
        raw_k_a = np.asarray(first["wavenumber_rad_m"], dtype=float)
        raw_k_b = np.asarray(second["wavenumber_rad_m"], dtype=float)
        power_a, power_b, details = align_values(
            np.asarray(first["power"], dtype=float),
            np.asarray(second["power"], dtype=float),
            {"frequency": raw_f_a, "wavenumber": raw_k_a},
            {"frequency": raw_f_b, "wavenumber": raw_k_b},
            "wave.komega",
            alignment or ComparisonAlignment(),
            {"power": ("frequency", "wavenumber")},
            "power",
        )
        if "amplitude" in first.files and "amplitude" in second.files:
            amplitude_a, amplitude_b, _ = align_values(
                np.asarray(first["amplitude"], dtype=float),
                np.asarray(second["amplitude"], dtype=float),
                {"frequency": raw_f_a, "wavenumber": raw_k_a},
                {"frequency": raw_f_b, "wavenumber": raw_k_b},
                "wave.komega",
                alignment or ComparisonAlignment(),
                {"amplitude": ("frequency", "wavenumber")},
                "amplitude",
            )
        else:
            amplitude_a = amplitude_b = None
        frequency = np.asarray(details["frequency"]["coordinate_values"], dtype=float)
        k = np.asarray(details["wavenumber"]["coordinate_values"], dtype=float)
    band_indices = np.flatnonzero(
        (frequency >= frequency_band_hz[0]) & (frequency <= frequency_band_hz[1])
    )
    if not len(band_indices):
        return None
    requested = np.array([0.5e6, 1.0e6, 2.0e6, 3.0e6])
    chosen: list[int] = []
    for target in requested:
        index = int(band_indices[np.argmin(np.abs(frequency[band_indices] - target))])
        if index not in chosen:
            chosen.append(index)
    if amplitude_a is None:
        amplitude_a = np.sqrt(np.maximum(power_a[chosen], 0.0))
        amplitude_b = np.sqrt(np.maximum(power_b[chosen], 0.0))
    else:
        amplitude_a = np.maximum(np.asarray(amplitude_a)[chosen], 0.0)
        amplitude_b = np.maximum(np.asarray(amplitude_b)[chosen], 0.0)
    amplitude_max = float(max(np.max(amplitude_a), np.max(amplitude_b)))
    fig, axes = plt.subplots(
        len(chosen), 1, figsize=(10, max(6.0, 2.5 * len(chosen))), sharex=True,
    )
    axes = np.atleast_1d(axes)
    for row, (axis, index) in enumerate(zip(axes, chosen)):
        axis.plot(k / 1e3, amplitude_a[row], label=_case_name(first_label), color="#1f77b4")
        axis.plot(k / 1e3, amplitude_b[row], label=_case_name(second_label), color="#ff7f0e")
        axis.set_ylim(0.0, amplitude_max if amplitude_max > 0.0 else 1.0)
        axis.set_ylabel(f"{frequency[index] / 1e6:.3g} MHz")
        axis.grid(True, alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel("Signed wavenumber kₓ [10³ rad m⁻¹]")
    fig.supylabel(f"Spectral amplitude [{_amplitude_unit(power_unit)}]")
    fig.suptitle(
        f"{variable.capitalize() if variable else 'f–k'} amplitude slices · "
        f"{_case_name(first_label)} vs {_case_name(second_label)}"
    )
    fig.tight_layout()
    _save_comparison_figure(fig, destination, dpi)
    plt.close(fig)
    return destination
