"""Coordinate-labelled figures for paired probe and directional-wave products."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
) -> Path | None:
    """Draw both cases at each shared probe, using physical time or frequency."""
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
        chosen = order[np.unique(np.linspace(0, len(order) - 1, min(len(order), 6), dtype=int))]
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
            "psd": ("frequency_hz", "psd", "Frequency [Hz]", "Welch PSD"),
        }
        axis_key, value_key, x_label, title = specs[kind]
        unit_key = "psd_unit" if kind == "psd" else "signal_unit"
        unit = str(first[unit_key]) if unit_key in first else ""
        y_label = (
            f"{variable.capitalize()} [{unit}]"
            if kind.startswith("raw") and unit
            else f"Amplitude [{unit}]"
            if kind == "fft" and unit
            else f"PSD [{unit}]"
            if kind == "psd" and unit
            else "Normalized amplitude"
            if kind == "fft_shape"
            else "Gaussian / Asymmetric [dB]"
            if kind == "fft_shape_ratio"
            else variable.capitalize()
        )
        for axis, column in zip(axes.flat, chosen):
            if kind == "fft_shape_ratio":
                x = np.asarray(first[axis_key], dtype=float)
                usable = x > 0.0
                if frequency_limit_hz is not None:
                    usable &= x <= frequency_limit_hz
                x = x[usable]
                a = np.maximum(np.asarray(first[value_key][:, column], dtype=float)[usable], 0)
                b = np.maximum(np.asarray(second[value_key][:, column], dtype=float)[usable], 0)
                norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
                if norm_a > 0 and norm_b > 0:
                    a, b = a / norm_a, b / norm_b
                    meaningful = (a >= np.max(a) * 0.01) & (b >= np.max(b) * 0.01)
                    ratio = np.full(len(x), np.nan)
                    ratio[meaningful] = 20.0 * np.log10(b[meaningful] / a[meaningful])
                    axis.plot(x, ratio, color="#6a3d9a", linewidth=1.8)
                    axis.axhline(0.0, color="0.5", linewidth=0.8)
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
                for archive, label, color in (
                    (first, first_label, "#1f77b4"),
                    (second, second_label, "#ff7f0e"),
                ):
                    x = np.asarray(archive[axis_key], dtype=float)
                    y = np.asarray(archive[value_key][:, column], dtype=float)
                    if kind.startswith("raw"):
                        x = x * 1e6
                    else:
                        usable = x > 0.0
                        if frequency_limit_hz is not None:
                            usable &= x <= frequency_limit_hz
                        x, y = x[usable], y[usable]
                        if kind == "fft_shape":
                            norm = float(np.linalg.norm(y))
                            y = y / norm if norm > 0 else y
                        y = np.where(y > 0.0, y, np.nan)
                    axis.plot(x, y, color=color, linewidth=1.8, label=_case_name(label))
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
                if kind != "fft_shape_ratio" and any(
                    np.isfinite(line.get_ydata()).any() for line in axis.lines
                ):
                    axis.set_yscale("log")
                elif kind != "fft_shape_ratio":
                    axis.text(
                        0.5,
                        0.5,
                        "No positive spectral amplitude",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
            axis.grid(True, alpha=0.2)
        for axis in list(axes.flat)[len(chosen) :]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"{variable.capitalize()}: {title}", y=0.995)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
        if kind == "fft_shape_ratio":
            fig.text(
                0.5,
                0.92,
                "Positive: Gaussian has more relative amplitude · both above −40 dB peak",
                ha="center",
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.85 if kind == "fft_shape_ratio" else 0.90))
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return destination


def render_komega_comparison(
    first_path: Path,
    second_path: Path,
    first_label: str,
    second_label: str,
    map_path: Path,
    spectrum_path: Path,
    frequency_band_hz: tuple[float, float],
    power_unit: str = "variable²",
) -> tuple[Path, Path] | None:
    """Show signed f-k power, a gated dB ratio, and band-integrated k shapes."""
    with (
        np.load(first_path, allow_pickle=False) as first,
        np.load(second_path, allow_pickle=False) as second,
    ):
        f = np.asarray(first["frequency_hz"], dtype=float)
        k = np.asarray(first["wavenumber_rad_m"], dtype=float)
        if not (
            _same_axis(f, np.asarray(second["frequency_hz"], dtype=float), 1e-6)
            and _same_axis(k, np.asarray(second["wavenumber_rad_m"], dtype=float), 1e-6)
        ):
            return None
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
        a = np.maximum(np.asarray(first["power"][use], dtype=float), 0.0)
        b = np.maximum(np.asarray(second["power"][use], dtype=float), 0.0)
        reference = float(max(np.max(a), np.max(b)))
        if reference <= 0.0:
            return None
        floor = reference * 1e-6
        relative_a = np.clip(10.0 * np.log10(np.maximum(a, floor) / reference), -60, 0)
        relative_b = np.clip(10.0 * np.log10(np.maximum(b, floor) / reference), -60, 0)
        ratio = np.full(a.shape, np.nan)
        meaningful = (a >= reference * 1e-4) & (b >= reference * 1e-4)
        ratio[meaningful] = 10.0 * np.log10(b[meaningful] / a[meaningful])
        extent = [k[0] / 1e3, k[-1] / 1e3, f[use][0] / 1e6, f[use][-1] / 1e6]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5.8),
            sharey=True,
            constrained_layout=True,
        )
        for axis, values, title in (
            (axes[0], relative_a, _case_name(first_label)),
            (axes[1], relative_b, _case_name(second_label)),
        ):
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=-60,
                vmax=0,
            )
            axis.set_title(title)
        difference = axes[2].imshow(
            ratio,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-12,
            vmax=12,
        )
        axes[2].set_title(f"{_case_name(second_label)} / {_case_name(first_label)} [dB]")
        fig.supxlabel("Signed wavenumber [10³ rad/m]")
        axes[0].set_ylabel("Frequency [MHz]")
        axes[0].set_yscale("log")
        axes[0].set_ylim(extent[2], extent[3])
        ticks = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])
        ticks = ticks[(ticks >= extent[2] - 1e-9) & (ticks <= extent[3] + 1e-9)]
        axes[0].set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        fig.colorbar(image, ax=axes[:2], label="Relative power [dB]")
        fig.colorbar(difference, ax=axes[2], label="Power ratio [dB]")
        fig.savefig(map_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        power_a = np.sum(a, axis=0)
        power_b = np.sum(b, axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for values, label, color in (
            (power_a, _case_name(first_label), "#1f77b4"),
            (power_b, _case_name(second_label), "#ff7f0e"),
        ):
            axes[0].plot(k / 1e3, np.where(values > 0, values, np.nan), color=color, label=label)
            normalized = values / np.sum(values) if np.sum(values) > 0 else values
            axes[1].plot(
                k / 1e3, np.where(normalized > 0, normalized, np.nan), color=color, label=label
            )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_ylabel(f"Band-integrated power [{power_unit}]")
        axes[1].set_ylabel("Fraction of band power per k bin")
        axes[1].set_xlabel("Signed wavenumber [10³ rad/m]")
        for axis in axes:
            axis.grid(True, alpha=0.2)
            axis.legend()
        fig.suptitle(f"Signed k spectrum, {lo / 1e6:.2g}–{hi / 1e6:.2g} MHz")
        fig.tight_layout()
        fig.savefig(spectrum_path, dpi=180)
        plt.close(fig)
    return map_path, spectrum_path
