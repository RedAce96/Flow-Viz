"""Fast regression tests for engineering and approved analysis corrections."""

import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_cache_root = Path(tempfile.gettempdir()) / "pelec-post-tests"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

import compressible_similarity as standalone_similarity
import pelec_post
import pp_functions_database as functions
import pp_config
import pp_modal_database as modal
import pp_plotting_database as plotting


class PlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_dataset_title_uses_flow_time(self):
        title = plotting.dataset_title({"time": 0.005}, "pressure")
        self.assertIn("Pressure", title)
        self.assertIn("5", title)
        self.assertIn("ms", title)
        self.assertNotIn("snapshot", title.lower())

    def test_flow_through_time_uses_full_domain_length(self):
        dataset = {
            "time": 0.004,
            "domain_length_x": 0.4,
            # Deliberately cropped coordinates: explicit full length must win.
            "x": np.linspace(0.0, 0.2, 21),
        }
        label = plotting.format_dataset_time(
            dataset, mode="flow_through", freestream_velocity=100.0
        )
        self.assertIn("FT", label)
        self.assertIn("1", label)

    def test_profile_coordinate_can_use_delta99(self):
        profile = {
            "y": np.array([0.0, 1.0e-3, 2.0e-3]),
            "values": np.array([0.0, 0.5, 1.0]),
            "field_key": "x_velocity",
            "label": "synthetic",
            "boundary_layer_height": 1.0e-3,
        }
        ax = plotting.plot_line_profiles(
            [profile], swap_axes=True,
            xlabel=r"$y/\delta_{99}$",
            coordinate_normalization="boundary_layer_height",
            annotate_boundary_layer=False,
            coordinate_limits=(0.0, 2.0),
        )
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 1.0, 2.0])
        self.assertIn(r"\delta", ax.get_ylabel())
        np.testing.assert_allclose(ax.get_ylim(), [0.0, 2.0])

    def test_legends_are_frameless_and_text_matches_lines(self):
        profiles = [
            {
                "y": np.array([0.0, 1.0]),
                "values": np.array([0.0, 1.0]),
                "field_key": "x_velocity",
                "label": "red profile",
                "color": "crimson",
            },
            {
                "y": np.array([0.0, 1.0]),
                "values": np.array([1.0, 0.0]),
                "field_key": "x_velocity",
                "label": "blue profile",
                "color": "navy",
            },
        ]
        ax = plotting.plot_line_profiles(profiles)
        legend = ax.get_legend()
        self.assertFalse(legend.get_frame_on())
        self.assertEqual(
            [text.get_color() for text in legend.get_texts()],
            ["crimson", "navy"],
        )

    def test_contour_accepts_canonical_xy_field_order(self):
        dataset = {
            "x": np.linspace(0.0, 1.0, 7),
            "y": np.linspace(0.0, 0.2, 3),
            "time": 0.0,
            "fields": {"pressure": np.arange(21.0).reshape(7, 3)},
        }
        ax = plotting.plot_contour(dataset, "pressure")
        ax.figure.canvas.draw()

    def test_contour_colorbar_shrink_tracks_displayed_x_span(self):
        x = np.linspace(0.0, 1.0, 5)
        self.assertAlmostEqual(
            plotting._resolve_colorbar_shrink(
                "auto", x, xlim=[0.0, 0.3], reference_span=0.3
            ),
            0.5,
        )
        self.assertAlmostEqual(
            plotting._resolve_colorbar_shrink(
                "auto", x, xlim=[0.0, 0.6], reference_span=0.3
            ),
            0.25,
        )
        self.assertAlmostEqual(
            plotting._resolve_colorbar_shrink(
                "auto", x, xlim=[0.0, 1.0], reference_span=0.3
            ),
            0.2,
        )

    def test_contour_and_surface_values_animate_in_lockstep(self):
        x = np.linspace(0.0, 1.0, 5)
        y = np.linspace(0.0, 0.2, 3)
        datasets = {}
        surfaces = {}
        surface_data = {}
        for frame, label in enumerate(("plt1", "plt2")):
            datasets[label] = {
                "x": x,
                "y": y,
                "time": frame * 1.0e-4,
                "fields": {
                    "temperature": np.add.outer(x, y) + frame,
                },
            }
            surfaces[label] = {
                "upper": {"x": x, "y_interp": np.full_like(x, 0.02)},
                "lower": {"x": np.array([]), "y_interp": np.array([])},
            }
            surface_data[label] = {
                "upper": {"x": x, "C_p": x + frame},
                "lower": {"x": np.array([]), "C_p": np.array([])},
            }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "paired.gif"
            result = plotting.animate_contour_with_surface(
                datasets,
                "temperature",
                surfaces_series=surfaces,
                surface_data_series=surface_data,
                loading_key="C_p",
                output_path=str(output),
                fps=2,
                dpi=50,
            )
            self.assertEqual(result, str(output))
            self.assertGreater(output.stat().st_size, 0)

    def test_stability_summary_mathtext_renders(self):
        x = np.array([0.05, 0.10, 0.15])
        data = {
            "freq_estimate": {
                "x": x,
                "f_omega_star": np.array([2.0e5, 1.8e5, 1.6e5]),
                "f_acoustic": np.array([2.2e5, 2.0e5, 1.7e5]),
            },
            "phase_speed": {
                "x_mid": x,
                "c_p": np.array([700.0, 710.0, 720.0]),
                "c_p_fast": np.full(3, 1100.0),
                "c_p_slow": np.full(3, 650.0),
            },
            "growth_rate": {
                "x": x,
                "alpha_i": np.array([2.0, 1.0, -1.0]),
                "alpha_i_delta": np.array([0.02, 0.01, -0.01]),
            },
            "gpi_profiles": [{
                "x": 0.10,
                "y_profile": np.linspace(0.0, 0.003, 8),
                "F": np.linspace(-1.0, 1.0, 8),
                "y_gpi": 0.001,
                "unstable": True,
            }],
            "target_freq": 1.8e5,
            "freq_band": [1.0e5, 3.0e5],
        }
        fig = plotting.plot_stability_summary(data)
        fig.canvas.draw()

    def test_wavenumber_plots_render(self):
        frequency = np.array([1.0e5, 2.0e5, 3.0e5])
        x = np.linspace(0.05, 0.25, 5)
        shape = (frequency.size, x.size)
        data = {
            "frequency_hz": frequency,
            "x_center_m": x,
            "spectral_power": np.ones(shape),
            "alpha_real_rad_per_m": np.full(shape, 500.0),
            "amplification_rate_per_m": np.linspace(
                -2.0, 3.0, np.prod(shape)
            ).reshape(shape),
            "phase_speed_m_per_s": np.full(shape, 1700.0),
            "phase_valid_mask": np.ones(shape, dtype=bool),
            "growth_valid_mask": np.ones(shape, dtype=bool),
            "mean_coherence_squared": np.full(shape, 0.95),
            "acoustic_candidate": np.tile(
                np.array([1, 1, 0, 2, 2]), (frequency.size, 1)
            ),
            "slow_acoustic_speed_m_per_s": np.full(x.size, 1500.0),
            "fast_acoustic_speed_m_per_s": np.full(x.size, 1950.0),
            "target_frequency_hz": 2.0e5,
        }
        summary = plotting.plot_wavenumber_summary(data)
        summary.canvas.draw()
        dispersion = plotting.plot_phase_speed_dispersion(data)
        dispersion.canvas.draw()


class BinaryProbeLoaderTests(unittest.TestCase):
    def test_canonical_field_selection_expands_solver_aliases(self):
        resolved = functions._resolve_field_aliases(["temperature", "density"])
        self.assertIn("Temp", resolved["temperature"])
        self.assertIn("density", resolved["density"])

    def test_vectorized_binary_loader_preserves_values(self):
        n_probes, n_fields = 3, 7
        requested_x = [1.0, 2.0, 3.0]
        requested_y = [0.1, 0.2, 0.3]
        times = [0.0, 0.25]
        records = []
        for step, time in enumerate(times):
            values = np.arange(n_probes * n_fields, dtype=float).reshape(
                n_probes, n_fields
            ) + 100.0 * step
            records.append((time, values))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.bin"
            with path.open("wb") as stream:
                stream.write(struct.pack("<q", 0x005345424F525050))
                stream.write(struct.pack("<q", n_probes))
                stream.write(struct.pack("<q", n_fields))
                stream.write(struct.pack(f"<{n_probes}d", *requested_x))
                stream.write(struct.pack(f"<{n_probes}d", *requested_y))
                for time, values in records:
                    stream.write(struct.pack("<d", time))
                    stream.write(values.astype("<f8").tobytes())

            loaded = pelec_post._load_probe_data_from_binary(
                {"probe_bin_files": [str(path)], "probe_dedup_tol": 1.0e-12},
                var_col=3,
            )

        self.assertEqual(len(loaded), n_probes)
        np.testing.assert_allclose(loaded[0]["time"], times)
        np.testing.assert_allclose(loaded[1]["signal"], [9.0, 109.0])
        self.assertEqual(loaded[2]["x"], 18.0)
        self.assertTrue(np.shares_memory(loaded[0]["time"], loaded[1]["time"]))
        time_common, matrix, dt, reused = pelec_post._probe_matrix_on_common_time(
            loaded, resample=True
        )
        self.assertTrue(reused)
        self.assertTrue(np.shares_memory(matrix, loaded[0]["signal"]))
        np.testing.assert_allclose(time_common, times)
        self.assertAlmostEqual(dt, 0.25)

    def test_chunked_binary_loader_selects_field_and_deduplicates_segments(self):
        n_probes = 3
        requested_x = np.array([1.0, 2.0, 3.0])
        requested_y = np.array([0.1, 0.2, 0.3])
        sample_x = requested_x + 0.001
        sample_y = requested_y + 0.002
        levels = np.full(n_probes, 3, dtype="<i4")
        valid = np.ones(n_probes, dtype=np.uint8)

        def write_segment(path, steps, times, pressure_offset):
            steps = np.asarray(steps, dtype="<i8")
            times = np.asarray(times, dtype="<f8")
            n_samples = len(times)
            fields = []
            for field in range(4):
                values = (
                    steps.astype(float)[:, None] * 10.0
                    + np.arange(n_probes, dtype=float)[None, :]
                    + 1000.0 * field
                )
                if field == 2:
                    values += pressure_offset
                fields.append(values.astype("<f8"))
            payload = b"".join([
                steps.tobytes(), times.tobytes(),
                sample_x.astype("<f8").tobytes(),
                sample_y.astype("<f8").tobytes(),
                levels.tobytes(), valid.tobytes(),
                *(field.tobytes() for field in fields),
            ])
            with path.open("wb") as stream:
                stream.write(b"PROBES2\0")
                stream.write(struct.pack(
                    "<8q", 2, 0x0102030405060708, n_probes, 4,
                    512, 1, 16, 24,
                ))
                for value in ("rho", "u", "p", "T"):
                    stream.write(value.encode().ljust(16, b"\0"))
                for value in ("g/cm^3", "cm/s", "dyne/cm^2", "K"):
                    stream.write(value.encode().ljust(24, b"\0"))
                stream.write(requested_x.astype("<f8").tobytes())
                stream.write(requested_y.astype("<f8").tobytes())
                stream.write(b"PRBCHNK2")
                stream.write(struct.pack(
                    "<7q2d", 0, n_samples, n_probes, 4, len(payload),
                    int(steps[0]), int(steps[-1]), times[0], times[-1],
                ))
                stream.write(payload)
                stream.write(struct.pack(
                    "<8sqQq", b"PRBEND2\0", 0, 0, len(payload)
                ))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_segment(root / "probe.segment0000.pbin", [0, 1], [0.0, 0.1], 0.0)
            # Repeat the final restart sample exactly, then add one new sample.
            write_segment(root / "probe.segment0001.pbin", [1, 2], [0.1, 0.2], 0.0)
            loaded = pelec_post._load_probe_data_from_binary(
                {
                    "probe_bin_files": [str(root / "probe.segment*.pbin")],
                    "probe_dedup_tol": 1.0e-12,
                },
                var_col=3,
            )

        self.assertEqual(len(loaded), n_probes)
        np.testing.assert_allclose(loaded[0]["time"], [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(loaded[0]["step"], [0, 1, 2])
        np.testing.assert_allclose(loaded[1]["signal"], [2001.0, 2011.0, 2021.0])
        self.assertAlmostEqual(loaded[2]["x"], sample_x[2])
        self.assertTrue(np.shares_memory(
            loaded[0]["_shared_signal_matrix"], loaded[1]["signal"]
        ))

    def test_chunked_loader_preserves_amr_mapping_epochs(self):
        n_probes = 2
        requested_x = np.array([1.0, 2.0])
        requested_y = np.array([0.1, 0.1])
        mappings = [
            (
                np.array([1.01, 2.01]), np.array([0.11, 0.11]),
                np.array([1, 1], dtype="<i4"),
            ),
            (
                np.array([1.005, 2.005]), np.array([0.105, 0.105]),
                np.array([2, 2], dtype="<i4"),
            ),
        ]

        def chunk_bytes(chunk_index, steps, times, mapping):
            steps = np.asarray(steps, dtype="<i8")
            times = np.asarray(times, dtype="<f8")
            sample_x, sample_y, levels = mapping
            valid = np.ones(n_probes, dtype=np.uint8)
            fields = []
            for field in range(4):
                values = (
                    1000.0 * field + 10.0 * steps[:, None]
                    + np.arange(n_probes)[None, :]
                ).astype("<f8")
                fields.append(values)
            payload = b"".join([
                steps.tobytes(), times.tobytes(),
                sample_x.astype("<f8").tobytes(),
                sample_y.astype("<f8").tobytes(), levels.tobytes(),
                valid.tobytes(), *(values.tobytes() for values in fields),
            ])
            header = struct.pack(
                "<7q2d", chunk_index, len(steps), n_probes, 4,
                len(payload), int(steps[0]), int(steps[-1]),
                times[0], times[-1],
            )
            footer = struct.pack(
                "<8sqQq", b"PRBEND2\0", chunk_index, 0, len(payload)
            )
            return b"PRBCHNK2" + header + payload + footer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.segment0000.pbin"
            with path.open("wb") as stream:
                stream.write(b"PROBES2\0")
                stream.write(struct.pack(
                    "<8q", 2, 0x0102030405060708, n_probes, 4,
                    512, 1, 16, 24,
                ))
                for value in ("rho", "u", "p", "T"):
                    stream.write(value.encode().ljust(16, b"\0"))
                for value in ("g/cm^3", "cm/s", "dyne/cm^2", "K"):
                    stream.write(value.encode().ljust(24, b"\0"))
                stream.write(requested_x.astype("<f8").tobytes())
                stream.write(requested_y.astype("<f8").tobytes())
                stream.write(chunk_bytes(
                    0, [0, 1], [0.0, 0.1], mappings[0]
                ))
                stream.write(chunk_bytes(
                    1, [2, 3, 4], [0.2, 0.3, 0.4], mappings[1]
                ))

            base = {"probe_bin_files": [str(path)]}
            with self.assertRaisesRegex(ValueError, "sampling map changed"):
                pelec_post._load_probe_data_from_binary(base, var_col=3)

            nominal = pelec_post._load_probe_data_from_binary(
                {**base, "probe_coordinate_policy": "nominal"}, var_col=3,
            )
            longest = pelec_post._load_probe_data_from_binary(
                {**base, "probe_coordinate_policy": "longest_epoch"},
                var_col=3,
            )
            products = pelec_post._write_probe_mapping_products(
                nominal, Path(tmp) / "output"
            )
            self.assertTrue(products[0].is_file())
            self.assertTrue(products[1].is_file())
            with np.load(products[1], allow_pickle=False) as archive:
                self.assertEqual(archive["sample_x_cm"].shape, (2, 2))
                np.testing.assert_array_equal(
                    archive["epoch_mapping_row"], [0, 1]
                )

        self.assertEqual(nominal[0]["x"], requested_x[0])
        self.assertEqual(nominal[0]["_mapping_report"]["epoch_count"], 2)
        self.assertEqual(nominal[0]["sampled_levels"], (1, 2))
        np.testing.assert_array_equal(
            nominal[0]["_mapping_epoch_id"], [0, 0, 1, 1, 1]
        )
        np.testing.assert_allclose(longest[0]["time"], [0.2, 0.3, 0.4])
        self.assertAlmostEqual(longest[0]["x"], mappings[1][0][0])
        self.assertEqual(longest[0]["_mapping_report"]["epoch_count"], 2)
        self.assertEqual(
            longest[0]["_mapping_report"]["analysis_epoch_count"], 1
        )

    def test_nonuniform_shared_record_is_reused_when_resampling_disabled(self):
        time = np.array([0.0, 1.0, 2.00001])
        matrix = np.arange(6.0).reshape(3, 2)
        probes = [
            {"time": time, "signal": matrix[:, column], "dt": 1.0,
             "x": column, "y": 0.0}
            for column in range(2)
        ]
        probes[0]["_shared_signal_matrix"] = matrix
        probes[0]["_shared_uniform_time"] = True
        _, returned, _, reused = pelec_post._probe_matrix_on_common_time(
            probes, resample=False
        )
        self.assertTrue(reused)
        self.assertTrue(np.shares_memory(returned, matrix))


class AnalysisCorrectionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_downstream_phase_speed_and_slow_acoustic_reference(self):
        frequency = 5.0e3
        expected_speed = 800.0
        probe_x = np.arange(8) * 5.0e-3
        alpha = 2.0 * np.pi * frequency / expected_speed
        spectrum = np.exp(-1j * alpha * probe_x)[None, :]
        result = functions.compute_phase_speed_from_probes(
            probe_x,
            np.array([frequency]),
            spectrum,
            frequency,
            u_edge=np.full(len(probe_x), 1000.0),
            T_edge=np.full(len(probe_x), 300.0),
        )
        np.testing.assert_allclose(result["c_p"], expected_speed, rtol=1e-12)
        expected_slow = 1000.0 - np.sqrt(1.4 * 287.05 * 300.0)
        np.testing.assert_allclose(result["c_p_slow"], expected_slow)

    def test_pprime_baseline_is_interpolated_between_grids(self):
        bx = np.linspace(0.05, 0.95, 10)
        by = np.linspace(0.05, 0.45, 5)
        tx = np.linspace(0.025, 0.975, 20)
        ty = np.linspace(0.025, 0.475, 10)
        baseline = 3.0 * bx[:, None] - 2.0 * by[None, :] + 5.0
        perturbation = np.sin(2.0 * np.pi * tx)[:, None] * np.ones((1, ty.size))
        target = 3.0 * tx[:, None] - 2.0 * ty[None, :] + 5.0 + perturbation
        result = functions.subtract_rectilinear_baseline(
            target, tx, ty, baseline, bx, by, chunk_size=3
        )
        np.testing.assert_allclose(result, perturbation, atol=1.0e-12)

    def test_harmonics_above_nyquist_are_rejected(self):
        signal = np.ones(1024)
        with self.assertRaisesRegex(ValueError, "exceeds Nyquist"):
            functions.reconstruct_from_harmonics(
                signal, dt=1.0 / 1024.0,
                harmonic_freq=300.0, num_harmonics=4,
            )
        _, _, bins = functions.reconstruct_from_harmonics(
            signal, dt=1.0 / 1024.0,
            harmonic_freq=256.0, num_harmonics=2,
        )
        self.assertEqual(bins[-1], 512)

    def test_top_frequency_reconstruction_selects_local_peaks(self):
        fs = 10_000.0
        n_samples = 1024
        time = np.arange(n_samples) / fs
        signal = np.cos(2.0 * np.pi * 503.2 * time)
        _, _, bins = functions.reconstruct_from_top_frequencies(
            signal, 1.0 / fs, n_peaks=8
        )
        self.assertEqual(len(bins), 1)
        self.assertAlmostEqual(bins[0] * fs / n_samples, 503.2, delta=fs/n_samples)

    def test_reconstruction_plot_uses_supplied_same_window_metric(self):
        time = np.linspace(0.0, 1.0, 100)
        window_time = time[20:60]
        measured = np.sin(2.0 * np.pi * time)
        reconstructed = np.zeros(len(window_time))
        residual = np.ones(len(window_time))
        fig = plotting.plot_harmonic_reconstruction(
            time, measured, reconstructed, residual, {},
            window_start=window_time[0], window_end=window_time[-1],
            windowed_signal=measured[20:60], window_time=window_time,
            relative_rms=0.25,
        )
        self.assertIn("0.250", fig.axes[1].get_title())

    def test_spectrogram_uses_psd_and_absolute_flow_time(self):
        fs = 10_000.0
        time = 0.005 + np.arange(2048) / fs
        signal = np.sin(2.0 * np.pi * 1000.0 * (time - time[0]))
        freq, _, psd_db = functions.compute_spectrogram(
            signal, fs, nperseg=256, noverlap=128
        )
        self.assertEqual(psd_db.shape[0], len(freq))
        self.assertAlmostEqual(freq[1] - freq[0], fs / 256.0)
        fig = plotting.plot_spectrogram(
            time, signal, fs, nperseg=256, noverlap=128
        )
        self.assertGreater(fig.axes[0].get_xlim()[0], 0.0)
        self.assertIn("Flow time", fig.axes[0].get_xlabel())

    def test_linear_spectrogram_has_zero_colour_minimum(self):
        fs = 4096.0
        time = 0.005 + np.arange(2048) / fs
        signal = np.sin(2.0 * np.pi * 256.0 * time)
        frequency, relative_time, psd = functions.compute_spectrogram(
            signal, fs, nperseg=256, noverlap=128,
            output_scale="linear",
        )
        self.assertTrue(np.all(psd >= 0.0))
        self.assertEqual(psd.shape, (frequency.size, relative_time.size))
        figure = plotting.plot_spectrogram(
            time, signal, fs, nperseg=256, noverlap=128,
            scale="linear", vmin=0.0, fmax=512.0,
        )
        self.assertEqual(
            figure.axes[0].collections[0].norm.vmin, 0.0
        )
        self.assertIn("PSD", figure.axes[1].get_ylabel())

    def test_fft_stft_comparison_uses_same_probe_signal(self):
        fs = 4096.0
        time = 0.005 + np.arange(4096) / fs
        signal = (
            20.0
            + np.sin(2.0 * np.pi * 256.0 * time)
            + 0.2 * np.sin(2.0 * np.pi * 512.0 * time)
        )
        figure = plotting.plot_fft_stft_comparison(
            time, signal, fs, fmax=1024.0,
            nperseg=256, noverlap=192,
            spectrogram_scale="linear",
            spectrogram_vmin=0.0,
        )
        self.assertEqual(len(figure.axes), 4)
        self.assertIn("Mean-subtracted", figure.axes[0].get_title())
        self.assertIn("Full-record FFT", figure.axes[1].get_title())
        self.assertIn("STFT spectrogram", figure.axes[2].get_title())
        self.assertAlmostEqual(
            float(np.mean(figure.axes[0].lines[0].get_ydata())),
            0.0,
            places=12,
        )
        self.assertEqual(
            figure.axes[2].collections[0].norm.vmin, 0.0
        )

    def test_short_bicoherence_record_fails_once(self):
        with self.assertRaisesRegex(ValueError, "must be at least nperseg"):
            functions.compute_bicoherence(
                np.ones(100), fs=1000.0, nperseg=256, noverlap=128
            )

    def test_bicoherence_plot_has_no_fake_significance_line(self):
        triads = [{"triad": value} for value in (0.1, 0.2, 0.3)]
        fig = plotting.plot_bicoherence_vs_x([0.0, 1.0, 2.0], triads)
        labels = [line.get_label().lower() for line in fig.axes[0].lines]
        self.assertFalse(any("sig" in label for label in labels))

    def test_direct_triad_matches_full_bicoherence_map(self):
        fs = 4096.0
        n = 4096
        time = np.arange(n) / fs
        f0 = 128.0
        signal = (
            np.cos(2.0 * np.pi * f0 * time)
            + 0.6 * np.cos(2.0 * np.pi * 2.0 * f0 * time + 0.4)
        )
        targets = [f0, 2.0 * f0]
        freq, full = functions.compute_bicoherence(
            signal, fs, nperseg=256, noverlap=128
        )
        expected = functions.extract_triad_bicoherence(freq, full, targets)
        direct = functions.compute_triad_bicoherence(
            signal, fs, targets, nperseg=256, noverlap=128
        )
        self.assertEqual(set(direct), set(expected))
        for key in direct:
            self.assertAlmostEqual(direct[key], expected[key], places=12)

    def test_main_returns_failure_when_enabled_snapshot_load_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(pelec_post.CONFIG)
            config.update({
                "data_source": tmp,
                "output_dir": str(Path(tmp) / "output"),
                "make_contour_plots": True,
                "make_line_profiles": False,
                "make_streamlines": False,
                "make_surface_analysis": False,
                "make_group_plots": False,
                "make_probe_plots": False,
                "make_fft_probes": False,
                "make_pprime_contour": False,
                "make_stability_diagnostics": False,
                "make_transient_analysis": False,
                "make_nonlinear_diagnostics": False,
                "make_force_analysis": False,
                "make_spacetime_plots": False,
                "make_force_animation": False,
            })
            with mock.patch.object(
                pelec_post.fdb, "discover_plotfile_paths",
                return_value=[str(Path(tmp) / "plt00001")],
            ), mock.patch.object(
                pelec_post.fdb, "load_pelec_plotfile",
                side_effect=OSError("synthetic load failure"),
            ):
                status = pelec_post.main(config)
            with (Path(tmp) / "output" / "run_manifest.json").open() as stream:
                manifest = __import__("json").load(stream)
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["failures"])
        self.assertEqual(status, 1)

    def test_welch_coherence_recovers_coupled_phase(self):
        fs = 8192.0
        time = np.arange(16384) / fs
        frequency = 512.0
        phase_lag = 0.4
        first = np.cos(2.0 * np.pi * frequency * time)
        second = np.cos(2.0 * np.pi * frequency * time - phase_lag)
        result = functions.compute_pair_coherence(
            first, second, fs, nperseg=2048, noverlap=1024
        )
        idx = int(np.argmin(np.abs(result["frequency_hz"] - frequency)))
        self.assertGreater(result["coherence_squared"][idx], 0.99)
        self.assertAlmostEqual(result["cross_phase_rad"][idx], -phase_lag, places=2)

    def test_target_coherence_handles_all_adjacent_probe_pairs(self):
        fs = 4096.0
        time = np.arange(8192) / fs
        signals = np.column_stack([
            np.cos(2.0 * np.pi * 256.0 * time - phase)
            for phase in (0.0, 0.2, 0.4, 0.6)
        ])
        result = functions.compute_adjacent_target_coherence(
            signals, fs, 256.0, nperseg=1024, noverlap=512,
            column_order=[3, 2, 1, 0],
        )
        np.testing.assert_allclose(result["coherence_squared"], 1.0, atol=1e-12)
        np.testing.assert_allclose(result["cross_phase_rad"], 0.2, atol=1e-12)

    def test_common_frequency_model_predicts_held_out_stationary_modes(self):
        fs = 4096.0
        length = 8192
        time = np.arange(length) / fs
        signals = np.column_stack([
            np.cos(2.0 * np.pi * 128.0 * time + phase)
            + 0.4 * np.cos(2.0 * np.pi * 320.0 * time - 0.5 * phase)
            for phase in (0.0, 0.3, 0.7)
        ])
        model = functions.fit_common_frequency_model(
            signals, 1.0 / fs, n_modes=2, train_fraction=0.5,
            freq_range=(50.0, 500.0),
        )
        np.testing.assert_allclose(
            model["selected_frequency_hz"], [128.0, 320.0], atol=1.0
        )
        self.assertLess(np.max(model["validation_relative_rms"]), 1.0e-10)

    def test_growth_fit_reports_confidence_interval(self):
        x = np.linspace(0.0, 0.1, 21)
        alpha = 12.0
        amplitude = np.exp(-alpha * x)
        result = functions.compute_growth_rate_from_probes(
            x, np.array([1000.0]), amplitude[None, :], 1000.0,
            window_size=len(x),
        )
        np.testing.assert_allclose(result["alpha_i"], alpha, atol=1.0e-10)
        np.testing.assert_allclose(
            result["amplification_rate"], -alpha, atol=1.0e-10
        )
        self.assertLess(np.nanmax(result["alpha_i_ci95"]), 1.0e-9)

    def test_frequency_resolved_complex_wavenumber(self):
        fs = 4096.0
        time = np.arange(8192) / fs
        probe_x = np.linspace(0.0, 0.3, 61)
        frequency = 256.0
        expected_speed = 8.0
        expected_amplification = 3.0
        alpha_real = 2.0 * np.pi * frequency / expected_speed
        signals = (
            np.exp(expected_amplification * probe_x)[None, :]
            * np.cos(
                2.0 * np.pi * frequency * time[:, None]
                - alpha_real * probe_x[None, :]
            )
        )
        result = functions.compute_frequency_resolved_wavenumber(
            signals,
            probe_x,
            fs,
            (255.0, 257.0),
            nperseg=1024,
            noverlap=512,
            spatial_window_size=31,
            spatial_step=10,
            min_phase_r_squared=0.99,
            min_amplitude_r_squared=0.99,
            phase_speed_bounds=(1.0, 20.0),
        )
        self.assertTrue(np.all(result["phase_valid_mask"]))
        self.assertTrue(np.all(result["growth_valid_mask"]))
        np.testing.assert_allclose(
            result["alpha_real_rad_per_m"], alpha_real, rtol=1.0e-8
        )
        np.testing.assert_allclose(
            result["phase_speed_m_per_s"], expected_speed, rtol=1.0e-8
        )
        np.testing.assert_allclose(
            result["alpha_imag_rad_per_m"],
            -expected_amplification,
            rtol=1.0e-7,
        )
        np.testing.assert_allclose(
            result["amplification_rate_per_m"],
            expected_amplification,
            rtol=1.0e-7,
        )

    def test_frequency_resolved_wavenumber_rejects_incoherent_noise(self):
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((4096, 41))
        result = functions.compute_frequency_resolved_wavenumber(
            signals,
            np.linspace(0.0, 0.2, 41),
            2048.0,
            (200.0, 220.0),
            nperseg=512,
            noverlap=256,
            spatial_window_size=21,
            spatial_step=10,
            min_coherence=0.8,
            min_coherent_fraction=0.8,
        )
        self.assertFalse(np.any(result["phase_valid_mask"]))
        self.assertFalse(np.any(result["growth_valid_mask"]))

    def test_frequency_resolved_dispersion_separates_frequencies(self):
        fs = 4096.0
        time = np.arange(8192) / fs
        probe_x = np.linspace(0.0, 0.3, 61)
        frequencies = (128.0, 320.0)
        speeds = (6.0, 9.0)
        signals = sum(
            amplitude * np.cos(
                2.0 * np.pi * frequency * time[:, None]
                - (2.0 * np.pi * frequency / speed) * probe_x[None, :]
            )
            for amplitude, frequency, speed in zip(
                (1.0, 0.6), frequencies, speeds
            )
        )
        result = functions.compute_frequency_resolved_wavenumber(
            signals,
            probe_x,
            fs,
            (120.0, 328.0),
            nperseg=1024,
            noverlap=512,
            spatial_window_size=31,
            spatial_step=10,
            min_phase_r_squared=0.99,
            min_relative_power_db=-20.0,
            phase_speed_bounds=(1.0, 20.0),
        )
        for frequency, speed in zip(frequencies, speeds):
            index = int(np.argmin(
                np.abs(result["frequency_hz"] - frequency)
            ))
            self.assertTrue(np.all(result["phase_valid_mask"][index]))
            np.testing.assert_allclose(
                result["phase_speed_m_per_s"][index], speed, rtol=1.0e-7
            )

    def test_packet_propagation_recovers_group_velocity_and_growth(self):
        time = np.linspace(0.0, 1.0, 2001)
        probe_x = np.linspace(0.0, 1.0, 11)
        expected_velocity = 2.0
        expected_log_energy_growth = 1.0
        centre = 0.25 + probe_x / expected_velocity
        envelope = np.exp(0.5 * expected_log_energy_growth * probe_x)[
            None, :
        ] * np.exp(
            -0.5 * (
                (time[:, None] - centre[None, :]) / 0.03
            ) ** 2
        )
        result = functions.estimate_packet_propagation(
            probe_x, time, envelope,
            baseline_end_time_s=0.10,
            noise_sigma=6.0,
            peak_fraction=0.05,
            persistent_samples=4,
        )
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(
            result["group_velocity_m_s"], expected_velocity, delta=0.02
        )
        self.assertGreater(result["arrival_fit_r_squared"], 0.999)
        self.assertAlmostEqual(
            result["log_energy_growth_per_m"],
            expected_log_energy_growth,
            delta=0.03,
        )
        self.assertEqual(result["baseline_source"], "configured_pre_event")

    def test_surrogate_bicoherence_detects_constructed_phase_coupling(self):
        fs = 1024.0
        nperseg = 256
        segment_count = 32
        time_segment = np.arange(nperseg) / fs
        rng = np.random.default_rng(7)
        segments = []
        for _ in range(segment_count):
            phase_one, phase_two = rng.uniform(0.0, 2.0 * np.pi, 2)
            segments.append(
                np.cos(2.0 * np.pi * 64.0 * time_segment + phase_one)
                + np.cos(
                    2.0 * np.pi * 96.0 * time_segment + phase_two
                )
                + 0.8 * np.cos(
                    2.0 * np.pi * 160.0 * time_segment
                    + phase_one + phase_two + 0.2
                )
            )
        result = functions.compute_surrogate_triad_significance(
            np.concatenate(segments),
            fs,
            [64.0, 96.0],
            nperseg=nperseg,
            noverlap=0,
            n_surrogates=99,
            fdr_alpha=0.05,
            minimum_independent_segments=8,
            random_seed=11,
            laser_frequency_hz=1000.0,
        )
        coupled = np.array([
            ("6.400e+01" in label and "9.600e+01" in label)
            for label in result["triad_labels"]
        ])
        self.assertTrue(np.any(result["significant_fdr"][coupled]))
        self.assertFalse(np.any(
            result["laser_harmonic_related"][coupled]
        ))
        self.assertIn(
            "segment-frequency", result["null_model"]
        )

    def test_surrogate_bicoherence_enforces_independent_segments(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping segments"):
            functions.compute_surrogate_triad_significance(
                np.ones(7 * 256),
                1024.0,
                [64.0, 96.0],
                nperseg=256,
                n_surrogates=19,
                minimum_independent_segments=8,
            )

    def test_evidence_matrix_keeps_lst_out_of_scope(self):
        report = functions.classify_measured_dynamics(
            wave_report={
                "phase_fit_accepted_fraction": 0.8,
                "complex_wavenumber_accepted_fraction": 0.6,
            },
            packet_report={
                "status": "complete",
                "group_velocity_m_s": 900.0,
                "arrival_fit_r_squared": 0.98,
                "baseline_source": "configured_pre_event",
            },
            nonlinear_report={
                "probe_results": [{
                    "status": "complete",
                    "significant_nonlaser_triad_count": 2,
                    "significant_laser_related_triad_count": 1,
                }]
            },
            modal_summary={
                "pod_subspace_cosines": [0.97],
                "dmd_dominant_frequency_hz": [
                    [1.0e6, 1.01e6], [0.99e6, 1.0e6]
                ],
                "dmd_condition_number": [[10.0, 20.0], [12.0, 22.0]],
            },
            force_report={
                "scientific_adequacy": {
                    "classification": "short_transient_validation",
                    "linkage": {"status": "insufficient_data"},
                    "probe_force_linkage": {"status": "complete"},
                }
            },
        )
        self.assertIn(
            "statistically_supported_quadratic_flow_coupling",
            report["supported_classifications"],
        )
        self.assertIn("LST_PSE", report["excluded_scope"])
        self.assertIn(
            "outside", report["excluded_scope"]["LST_PSE"]
        )


class ModalAnalysisTests(unittest.TestCase):
    def _traveling_wave_dataset(self):
        fs = 1024.0
        time = np.arange(2048) / fs
        x = np.linspace(0.0, 1.0, 12, endpoint=False)
        frequency = 64.0
        values = np.cos(
            2.0 * np.pi * frequency * time[:, None]
            - 2.0 * np.pi * x[None, :]
        )
        return modal.SnapshotMatrix(time, values, x, variable="pressure"), frequency

    def test_pod_recovers_rank_two_traveling_wave(self):
        dataset, _ = self._traveling_wave_dataset()
        result = modal.compute_pod(dataset, n_modes=3)
        self.assertGreater(np.sum(result["energy_fraction"][:2]), 0.999999)
        self.assertLess(result["energy_fraction"][2], 1.0e-20)

    def test_spod_identifies_known_frequency(self):
        dataset, frequency = self._traveling_wave_dataset()
        nperseg = 256
        freq = np.fft.rfftfreq(nperseg, dataset.dt)
        known = int(np.argmin(np.abs(freq - frequency)))
        comparison = int(np.argmin(np.abs(freq - 128.0)))
        result = modal.compute_spod(
            dataset, nperseg=nperseg, noverlap=128, n_modes=2,
            frequency_indices=[known, comparison],
        )
        self.assertGreater(result["eigenvalues"][0, 0],
                           1.0e6 * result["eigenvalues"][1, 0])

    def test_dmd_recovers_known_oscillation_frequency(self):
        dataset, frequency = self._traveling_wave_dataset()
        result = modal.compute_dmd(dataset, n_modes=2)
        recovered = np.sort(np.abs(result["frequency_hz"]))
        np.testing.assert_allclose(recovered, [frequency, frequency], atol=0.1)

    def test_probe_line_quadrature_integrates_constant_exactly(self):
        x = np.array([0.0, 0.1, 0.4, 1.0])
        weights = modal.trapezoidal_spatial_weights(x)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights > 0.0))

    def test_scalar_energy_weights_are_explicitly_partial(self):
        x = np.linspace(0.0, 1.0, 9)
        result = modal.scalar_compressible_energy_weights(
            x, "Pressure [Pa]",
            rho_base=0.02, temperature_base=125.0,
        )
        self.assertTrue(np.all(result["weights"] > 0.0))
        self.assertIn("not the complete Chu norm", result["scope"])
        np.testing.assert_allclose(
            result["weights"] / np.mean(result["weights"]),
            result["quadrature_weights"]
            / np.mean(result["quadrature_weights"]),
        )

    def test_modal_sensitivity_recovers_window_robust_traveling_wave(self):
        dataset, frequency = self._traveling_wave_dataset()
        result = modal.compute_modal_sensitivity(
            dataset, pod_modes=2, dmd_ranks=[2],
            window_fractions=[[0.0, 0.5], [0.5, 1.0]],
        )
        self.assertGreater(
            result["pod_subspace_min_cosine_to_first_window"][0],
            0.999,
        )
        np.testing.assert_allclose(
            result["dmd_dominant_frequency_hz"],
            frequency,
            atol=0.1,
        )
        self.assertIn("no LST", result["interpretation"])


class CompressibleReferenceTests(unittest.TestCase):
    """Regression checks for the independent laminar base-flow reference."""

    @staticmethod
    def _reference(**overrides):
        parameters = {
            "y": np.linspace(0.0, 0.005, 151),
            "x_loc": 0.30,
            "u_inf": 1726.0,
            "T_inf": 125.0,
            "rho_inf": 760.0 / (287.05 * 125.0),
            "T_wall": 293.0,
            "gamma": 1.4,
            "R": 287.05,
            "Cp": 1004.0,
            "transport_model": "constant",
            "mu": 8.65e-6,
            "k": 0.012415,
        }
        parameters.update(overrides)
        return functions.compute_compressible_flat_plate_reference_profile(
            **parameters
        )

    def test_incompressible_limit_recovers_blasius_and_skin_friction(self):
        y = np.linspace(0.0, 0.01, 301)
        u_inf, rho_inf, mu, T_inf, x_loc = 1.0, 1.0, 1.0e-5, 300.0, 0.1
        reference = functions.compute_compressible_flat_plate_reference_profile(
            y=y, x_loc=x_loc, u_inf=u_inf, T_inf=T_inf, rho_inf=rho_inf,
            T_wall=T_inf, gamma=1.4, R=287.05, Cp=1004.0,
            transport_model="constant", mu=mu, k=mu * 1004.0 / 0.71,
        )
        legacy = functions.compute_blasius_reference_profile(
            y, x_loc, u_inf, T_inf, rho_inf, T_wall=T_inf,
            const_transport=True, mu=mu, k=mu * 1004.0 / 0.71,
        )
        np.testing.assert_allclose(
            reference["profiles"]["x_velocity"]["values"],
            legacy["x_velocity"]["values"], atol=1.0e-3,
        )
        re_x = rho_inf * u_inf * x_loc / mu
        self.assertAlmostEqual(
            reference["scalars"]["C_f"], 0.664 / np.sqrt(re_x), places=5
        )

    def test_thermal_wall_conditions_and_profile_outputs(self):
        isothermal = self._reference()
        profiles = isothermal["profiles"]
        self.assertEqual(set(profiles), {"x_velocity", "temperature", "density"})
        self.assertAlmostEqual(profiles["x_velocity"]["values"][0], 0.0, places=10)
        self.assertAlmostEqual(profiles["temperature"]["values"][0], 293.0, places=8)
        self.assertLess(isothermal["scalars"]["q_w"], 0.0)  # heat enters the wall
        self.assertTrue(np.all(np.diff(isothermal["y_similarity"]) > 0.0))
        self.assertTrue(np.all(isothermal["density_similarity"] > 0.0))
        np.testing.assert_allclose(
            isothermal["theta"],
            isothermal["temperature_similarity"] / 125.0,
        )

        adiabatic = self._reference(T_wall=None)
        self.assertAlmostEqual(adiabatic["scalars"]["q_w"], 0.0, places=7)
        self.assertGreater(
            adiabatic["profiles"]["temperature"]["values"][0], 125.0
        )

    def test_density_weighted_eta_coordinate(self):
        y = np.array([0.0, 0.001, 0.002])
        density = np.full_like(y, 2.0)
        eta = functions.compute_compressible_similarity_coordinate(
            y, density, x_loc=0.5, u_inf=10.0,
            rho_inf=2.0, mu_inf=0.01,
        )
        expected = np.sqrt(10.0 / (2.0 * 2.0 * 0.01 * 0.5)) * 2.0 * y
        np.testing.assert_allclose(eta, expected)

    def test_sutherland_option_is_finite_and_distinct(self):
        constant = self._reference()
        sutherland = self._reference(transport_model="sutherland", Pr=0.71)
        self.assertTrue(np.isfinite(sutherland["scalars"]["C_f"]))
        self.assertNotAlmostEqual(
            constant["scalars"]["delta_99"],
            sutherland["scalars"]["delta_99"], places=7,
        )

    def test_reference_profiles_render_with_simulation_profiles(self):
        reference = self._reference()
        for field_key, reference_profile in reference["profiles"].items():
            simulated = dict(reference_profile)
            simulated["label"] = "synthetic simulation"
            simulated["linestyle"] = "-"
            figure = plotting.plot_line_profiles(
                [simulated, reference_profile], swap_axes=True,
                coordinate_normalization="boundary_layer_height",
                annotate_boundary_layer=False,
            ).figure
            figure.canvas.draw()

    def test_standalone_solver_matches_post_process_reference(self):
        current = self._reference()
        standalone = standalone_similarity.compute_compressible_flat_plate_reference_profile(
            y=np.linspace(0.0, 0.005, 151),
            x_loc=0.30,
            u_inf=1726.0,
            T_inf=125.0,
            rho_inf=760.0 / (287.05 * 125.0),
            T_wall=293.0,
            gamma=1.4,
            R=287.05,
            Cp=1004.0,
            transport_model="constant",
            mu=8.65e-6,
            k=0.012415,
        )
        for key in (
                "eta", "theta", "y_similarity", "u_ratio",
                "temperature_similarity", "density_similarity"):
            np.testing.assert_allclose(standalone[key], current[key])
        for key in (
                "delta_99", "delta_star", "theta", "H",
                "tau_w", "C_f", "q_w"):
            self.assertAlmostEqual(
                standalone["scalars"][key], current["scalars"][key], places=12
            )


class CertifiedForceAnalysisTests(unittest.TestCase):
    @staticmethod
    def _reference(p_inf=100.0):
        return {
            "rho_inf": 2.0,
            "u_inf": 10.0,
            "p_inf": p_inf,
            "chord": 1.0,
            "moment_origin": [0.25, 0.0],
        }

    @staticmethod
    def _wall(x_edges, pressure, shear):
        x_edges = np.asarray(x_edges, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        shear = np.asarray(shear, dtype=float)
        n = len(pressure)
        return {
            "x_left_m": x_edges[:-1],
            "x_right_m": x_edges[1:],
            "x_center_m": 0.5 * (x_edges[:-1] + x_edges[1:]),
            "wall_y_m": np.zeros(n),
            "p_wall_pa": pressure,
            "tau_wall_pa": shear,
            "valid": np.ones(n, dtype=bool),
            "amr_level": np.zeros(n, dtype=int),
            "requested_x_range_m": np.array([x_edges[0], x_edges[-1]]),
            "q_inf_pa": 100.0,
        }

    def test_wall_polynomial_recovers_pressure_and_shear(self):
        x_edges = np.linspace(0.0, 1.0, 5)
        distance = np.array([0.01, 0.02, 0.03, 0.04])
        pressure = 120.0 + 40.0 * distance + 3.0 * distance**2
        velocity = 50.0 * distance - 2.0 * distance**2
        wall = functions.reconstruct_flat_plate_wall_stencils(
            x_edges[:-1], x_edges[1:],
            np.tile(distance, (4, 1)),
            np.tile(pressure, (4, 1)),
            np.tile(velocity, (4, 1)),
            viscosity_pa_s=2.0,
            p_inf_pa=100.0, rho_inf_kg_m3=2.0, u_inf_m_s=10.0,
        )
        np.testing.assert_allclose(wall["p_wall_pa"], 120.0, atol=1.0e-10)
        np.testing.assert_allclose(wall["tau_wall_pa"], 100.0, atol=1.0e-10)
        self.assertTrue(np.all(wall["valid"]))

    def test_freestream_pressure_has_zero_pressure_load(self):
        wall = self._wall(np.linspace(0.0, 1.0, 5), np.full(4, 100.0),
                          np.zeros(4))
        force = functions.integrate_flat_plate_wall_forces(
            wall, self._reference()
        )
        self.assertAlmostEqual(force["N_pressure_N_m"], 0.0)
        self.assertAlmostEqual(force["D_total_N_m"], 0.0)

    def test_constant_pressure_and_shear_integrate_with_correct_sign(self):
        wall = self._wall(np.linspace(0.0, 1.0, 5), np.full(4, 120.0),
                          np.full(4, 3.0))
        force = functions.integrate_flat_plate_wall_forces(
            wall, self._reference()
        )
        self.assertAlmostEqual(force["D_viscous_N_m"], 3.0)
        self.assertAlmostEqual(force["N_pressure_N_m"], -20.0)
        self.assertAlmostEqual(force["C_D"], 0.03)
        self.assertAlmostEqual(force["C_N_one_sided"], -0.2)

    def test_linear_pressure_moment_is_exact(self):
        edges = np.linspace(0.0, 1.0, 101)
        centers = 0.5 * (edges[:-1] + edges[1:])
        wall = self._wall(edges, 100.0 + 10.0 * centers, np.zeros(100))
        force = functions.integrate_flat_plate_wall_forces(
            wall, self._reference()
        )
        # M = integral (x-0.25)*[-10*x] dx = -10*(1/3-1/8).
        self.assertAlmostEqual(
            force["M_pressure_N"], -10.0 * (1.0 / 3.0 - 1.0 / 8.0),
            places=3,
        )

    def test_conservative_baseline_subtraction_handles_different_faces(self):
        current = self._wall(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [110.0, 110.0, 110.0, 110.0],
            [3.0, 3.0, 3.0, 3.0],
        )
        baseline = self._wall(
            [0.0, 0.5, 1.0], [100.0, 100.0], [1.0, 1.0]
        )
        delta = functions.difference_flat_plate_wall_surfaces(
            current, baseline
        )
        result = functions.integrate_flat_plate_wall_forces(
            delta, self._reference(p_inf=0.0)
        )
        self.assertAlmostEqual(result["D_total_N_m"], 2.0)
        self.assertAlmostEqual(result["N_total_N_m"], -10.0)

    def test_invalid_wall_data_fail_instead_of_becoming_zero(self):
        wall = self._wall(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [100.0, np.nan, 100.0, 100.0],
            np.zeros(4),
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            functions.integrate_flat_plate_wall_forces(
                wall, self._reference()
            )

    def test_wall_overlap_and_large_gap_are_rejected(self):
        overlap = self._wall(
            [0.0, 0.6, 1.0], [100.0, 100.0], [1.0, 1.0]
        )
        overlap["x_left_m"] = np.array([0.0, 0.5])
        with self.assertRaisesRegex(ValueError, "overlap"):
            functions.integrate_flat_plate_wall_forces(
                overlap, self._reference()
            )
        gap = self._wall(
            [0.0, 0.1, 1.0], [100.0, 100.0], [1.0, 1.0]
        )
        gap["x_left_m"] = np.array([0.0, 0.5])
        with self.assertRaisesRegex(ValueError, "coverage|gap"):
            functions.integrate_flat_plate_wall_forces(
                gap, self._reference()
            )

    def test_force_and_coefficient_component_identities(self):
        wall = self._wall(
            [0.0, 0.5, 1.0], [110.0, 120.0], [2.0, 4.0]
        )
        result = functions.integrate_flat_plate_wall_forces(
            wall, self._reference()
        )
        self.assertAlmostEqual(
            result["D_total_N_m"],
            result["D_pressure_N_m"] + result["D_viscous_N_m"],
        )
        self.assertAlmostEqual(
            result["C_D"],
            result["C_D_pressure"] + result["C_D_viscous"],
        )
        self.assertAlmostEqual(
            result["C_N_one_sided"],
            result["C_N_pressure_one_sided"]
            + result["C_N_viscous_one_sided"],
        )
        self.assertAlmostEqual(
            result["C_M"],
            result["C_M_pressure"] + result["C_M_viscous"],
        )

    def test_density_weighted_thicknesses(self):
        distance = np.linspace(0.0, 1.0, 10001)
        u = distance
        rho = np.ones_like(distance)
        result = functions.calculate_compressible_BL_thicknesses(
            distance, u, rho, edge_velocity=1.0, edge_density=1.0
        )
        self.assertAlmostEqual(result["delta_star"], 0.5, places=7)
        self.assertAlmostEqual(result["theta"], 1.0 / 6.0, places=7)
        self.assertAlmostEqual(result["H"], 3.0, places=6)

    def test_density_weighted_thicknesses_with_variable_density(self):
        distance = np.linspace(0.0, 1.0, 10001)
        result = functions.calculate_compressible_BL_thicknesses(
            distance, distance, 1.0 + distance,
            edge_velocity=1.0, edge_density=2.0,
        )
        self.assertAlmostEqual(result["delta_star"], 7.0 / 12.0, places=7)
        self.assertAlmostEqual(result["theta"], 1.0 / 8.0, places=7)
        self.assertAlmostEqual(result["H"], 14.0 / 3.0, places=6)

    def test_gaussian_pulse_uses_fwhm_definition(self):
        fwhm = 10.0e-9
        time = np.array([0.0, 0.5 * fwhm])
        pulse = functions.gaussian_pulse_train(
            time, start_time_s=0.0, frequency_hz=1.0e6,
            pulse_fwhm_s=fwhm, duration_s=1.0e-6,
        )
        self.assertAlmostEqual(pulse["signal"][0], 1.0)
        self.assertAlmostEqual(pulse["signal"][1], 0.5, places=12)

    def test_force_spectra_require_requested_segment_count(self):
        time = np.arange(1024) / 1024.0
        signal = np.sin(2.0 * np.pi * 32.0 * time)
        with self.assertRaisesRegex(ValueError, "Welch segments"):
            functions.compute_input_output_spectra(
                signal, signal, 1024.0, nperseg=512,
                noverlap=0.5, minimum_segments=8,
            )

    def test_compact_force_history_round_trip_and_restart_deduplication(self):
        header = struct.pack(
            "<8sII6d", b"WFORCE1\0", 1, 9,
            0.0, 40.0, 1000.0, 8.65e-5, 25.0, 0.0,
        )
        records = [
            (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0, 2.0),
            (1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 1.0, 2.0),
            # Exact restart duplicate: one record is retained.
            (1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 1.0, 2.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wall-force.bin"
            with path.open("wb") as stream:
                stream.write(header)
                for record in records:
                    stream.write(struct.pack("<9d", *record))
                stream.write(b"partial")
            result = functions.load_compact_wall_force_history(
                path, self._reference()
            )
        np.testing.assert_allclose(result["time_s"], [0.0, 1.0])
        np.testing.assert_allclose(result["D_pressure_N_m"], [0.01, 0.011])
        np.testing.assert_allclose(result["M_viscous_N"], [6.0e-4, 6.1e-4])
        np.testing.assert_allclose(result["x_range_m"], [0.0, 0.4])
        self.assertEqual(result["truncated_tail_bytes"], len(b"partial"))
        self.assertEqual(result["amr_max_level"][-1], 2.0)

    def test_compact_force_history_rejects_conflicting_restart_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wall-force.bin"
            with path.open("wb") as stream:
                stream.write(struct.pack(
                    "<8sII6d", b"WFORCE1\0", 1, 9,
                    0.0, 40.0, 1000.0, 8.65e-5, 25.0, 0.0,
                ))
                stream.write(struct.pack("<9d", 1.0, *([0.0] * 8)))
                stream.write(struct.pack(
                    "<9d", 1.0, 1.0, *([0.0] * 7)
                ))
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                functions.load_compact_wall_force_history(
                    path, self._reference()
                )

    def test_compact_force_history_rejects_time_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wall-force.bin"
            with path.open("wb") as stream:
                stream.write(struct.pack(
                    "<8sII6d", b"WFORCE1\0", 1, 9,
                    0.0, 40.0, 1000.0, 8.65e-5, 25.0, 0.0,
                ))
                stream.write(struct.pack("<9d", 1.0, *([0.0] * 8)))
                stream.write(struct.pack("<9d", 0.5, *([0.0] * 8)))
            with self.assertRaisesRegex(ValueError, "time reversal"):
                functions.load_compact_wall_force_history(
                    path, self._reference()
                )

    def test_probe_force_linkage_uses_common_interval_without_extrapolation(self):
        force_time = np.arange(0.0, 2.0, 0.01)
        probe_time = np.arange(0.2, 1.8, 0.005)
        pressure = np.column_stack((
            np.sin(2.0 * np.pi * 5.0 * probe_time),
            np.sin(2.0 * np.pi * 5.0 * (probe_time - 0.02)),
        ))
        force = np.sin(2.0 * np.pi * 5.0 * (force_time - 0.04))
        result = functions.compute_probe_force_linkage(
            force_time, force, probe_time, pressure, [0.1, 0.2],
            forcing_frequency_hz=100.0, minimum_forcing_periods=1000.0,
        )
        self.assertGreaterEqual(result["time_s"][0], 0.2)
        self.assertLessEqual(result["time_s"][-1], probe_time[-1])
        self.assertEqual(result["probe_matrix"].shape[1], 2)
        self.assertEqual(result["spectral_status"], "insufficient_data")
        self.assertAlmostEqual(result["sample_rate_resampled_hz"], 100.0)


class ConfigurationTests(unittest.TestCase):
    def test_probe_coordinate_policy_is_validated(self):
        config = dict(pelec_post.CONFIG)
        config["probe_coordinate_policy"] = "ignore"
        with self.assertRaisesRegex(ValueError, "probe_coordinate_policy"):
            pp_config.build_config(config)

    def test_json_overlay_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"typo_workflow": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                pp_config.build_config(pelec_post.CONFIG, path)

    def test_json_overlay_rejects_unknown_nested_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"force": {"reference": {"p_inff": 760.0}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "force.reference.p_inff"):
                pp_config.build_config(pelec_post.CONFIG, path)

    def test_json_overlay_applies_valid_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"snapshot_start": 42}', encoding="utf-8")
            config = pp_config.build_config(pelec_post.CONFIG, path)
        self.assertEqual(config["snapshot_start"], 42)
        self.assertNotEqual(pelec_post.CONFIG["snapshot_start"], 42)

    def test_json_overlay_deep_merges_reference_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"line_reference": {"transport_model": "sutherland", "Pr": 0.72}}',
                encoding="utf-8",
            )
            config = pp_config.build_config(pelec_post.CONFIG, path)
        self.assertEqual(config["line_reference"]["transport_model"], "sutherland")
        self.assertEqual(config["line_reference"]["Pr"], 0.72)
        self.assertEqual(config["line_reference"]["u_inf"], 1726.0)

    def test_compressible_reference_rejects_nonflat_geometry(self):
        config = dict(pelec_post.CONFIG)
        config["make_line_profiles"] = True
        config["geometry_type"] = "wedge"
        with self.assertRaisesRegex(ValueError, "flat_plate"):
            pp_config.build_config(config)

    def test_synchronized_animation_requires_surface_analysis(self):
        config = dict(pelec_post.CONFIG)
        config["make_force_animation"] = True
        config["make_surface_analysis"] = False
        with self.assertRaisesRegex(ValueError, "make_surface_analysis"):
            pp_config.build_config(config)


if __name__ == "__main__":
    unittest.main()
