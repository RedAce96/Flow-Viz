from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pelecpost.analysis.comparison import (
    _array_metrics,
    _json_metrics,
    _Product,
    _psd_ratio_confidence,
    _render_product_overlay,
    _validate_product_pair,
    _validity_for_value,
)
from pelecpost.analysis.comparison_figures import (
    _fk_ratio_components,
    render_komega_amplitude_frequency_slices,
    render_komega_comparison,
    render_probe_panels,
)
from pelecpost.analysis.probe_plotting import build_probe_overlay_figure
from pelecpost.analysis.products import product_contract
from pelecpost.analysis.spectral import (
    _prepare_fft_grid,
    _si_conversion,
    spectrum_from_signal,
)
from pelecpost.analysis.temporal_wavenumber import (
    compute_temporal_wavenumber,
    register_temporal_wavenumber_figures,
)
from pelecpost.config.loader import write_project_schema
from pelecpost.config.models import (
    ComparisonAlignment,
    DirectionalWaveAnalysis,
    FFTRatioPlottingConfig,
    PresentationConfig,
    ProbeInput,
    TemporalWavenumberEnabled,
)
from pelecpost.io.signals import open_probe_signal_workspace
from pelecpost.preflight import create_plan
from pelecpost.workflows import build_workflow_graph
from tests import test_bounded_probe_workspace as bounded_fixtures
from tests import test_preflight as preflight_fixtures


def _product_metadata(path: Path, product_id: str, units: str = "Pa") -> dict:
    return {
        "kind": "array",
        "variable": "pressure",
        "units": units,
        "provenance": {
            "preprocessing": {"window": "hann", "time_grid_policy": "require_uniform"},
            "product_contract": product_contract(product_id, path, units=units),
        },
    }


class UnifiedProbeComparisonTests(unittest.TestCase):
    def test_validity_mask_uses_declared_dimension_when_axes_have_equal_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transfer.npz"
            np.savez(
                path, frequency_hz=np.array([1.0, 2.0]),
                probe_x_m=np.array([0.0, 1.0]),
                transfer=np.ones((2, 2)),
                valid_frequency=np.array([True, False]),
            )
            with np.load(path, allow_pickle=False) as archive:
                contract = product_contract("pulse.transfer", path)
                mask = _validity_for_value(
                    archive, contract, archive["transfer"],
                    {"frequency": archive["frequency_hz"], "space": archive["probe_x_m"]},
                    "pulse.transfer", "transfer",
                )
                np.testing.assert_array_equal(mask, [[True, True], [False, False]])
                ambiguous_contract = {
                    "validity_by_value": {"transfer": ("valid_frequency",)},
                }
                with self.assertRaisesRegex(ValueError, "unambiguously"):
                    _validity_for_value(
                        archive, ambiguous_contract, archive["transfer"],
                        {"frequency": archive["frequency_hz"], "space": archive["probe_x_m"]},
                        "pulse.transfer", "transfer",
                    )

    def test_confidence_contract_and_zero_support_fk_statistics(self):
        self.assertEqual(product_contract("spectral.confidence")["schema_version"], 3)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frequency = np.array([1.0e5, 2.0e5])
            wavenumber = np.array([-1.0, 0.0, 1.0])
            baseline = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            comparison = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
            left, right = directory / "left.npz", directory / "right.npz"
            np.savez(left, frequency_hz=frequency, wavenumber_rad_m=wavenumber, power=baseline)
            np.savez(right, frequency_hz=frequency, wavenumber_rad_m=wavenumber, power=comparison)
            target = directory / "ratio.png"
            render_komega_comparison(
                left, right, "left.wave.komega", "right.wave.komega",
                target, None, (1.0e5, 2.0e5), dpi=60,
            )
            statistics = json.loads(target.with_suffix(".json").read_text())
            self.assertEqual(statistics["status"], "unavailable_no_supported_bins")
            self.assertEqual(statistics["accepted_count"], 0)
            self.assertEqual(statistics["masked_fraction"], 1.0)

    def test_confidence_payload_version_must_match_its_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            older, current = directory / "older.json", directory / "current.json"
            older.write_text(json.dumps({"schema_version": 2}))
            current.write_text(json.dumps({"schema_version": 3}))
            metadata = {"kind": "json", "variable": "pressure", "units": None}
            with self.assertRaisesRegex(ValueError, "payload schema version"):
                _validate_product_pair(
                    _Product("a.spectral.confidence", metadata, older),
                    _Product("b.spectral.confidence", metadata, current),
                )

    def test_fft_ratio_plotting_choices_validate(self):
        options = FFTRatioPlottingConfig(
            scales=("linear",), normalizations=("absolute",),
            minimum_relative_amplitude=0.02,
        )
        self.assertEqual(options.scales, ("linear",))
        self.assertEqual(options.normalizations, ("absolute",))
        with self.assertRaises(ValueError):
            FFTRatioPlottingConfig(scales=("linear", "linear"))
        with self.assertRaises(ValueError):
            FFTRatioPlottingConfig(minimum_relative_amplitude=0.0)
        self.assertTrue(FFTRatioPlottingConfig().amplitude_ratio_panels)
        self.assertEqual(FFTRatioPlottingConfig().spectral_display, "amplitude")
        self.assertEqual(FFTRatioPlottingConfig().psd_ratio_confidence_level, 0.95)
        self.assertEqual(
            DirectionalWaveAnalysis(
                id="wave", recipe="directional_wave", probe_set_id="probe",
                variable="pressure", frequency_max_hz=1.0e8,
            ).spectral_display,
            "amplitude",
        )
        self.assertEqual(
            FFTRatioPlottingConfig(spectral_display="relative_db").spectral_display,
            "relative_db",
        )
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            FFTRatioPlottingConfig(symmetry_diagnostics=True)

    def test_probe_overlay_legend_stays_above_axes_and_uses_microseconds(self):
        figure = build_probe_overlay_figure(
            PresentationConfig(),
            np.array([0.0, 1e-6]),
            np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]),
            ["Probe 120", "Probe 150", "Probe 170", "Probe 200"],
            "Time [s]",
            "Pressure [Pa]",
        )
        try:
            self.assertIsNone(figure.axes[0].get_legend())
            self.assertEqual(len(figure.legends), 1)
            self.assertEqual(figure.axes[0].get_xlabel(), "Time [µs]")
            np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0.0, 1.0])
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            self.assertGreater(
                figure.legends[0].get_window_extent(renderer).y0,
                figure.axes[0].get_window_extent(renderer).y1,
            )
        finally:
            plt.close(figure)

    def test_vorticity_probe_unit_is_invariant_under_cgs_length_units(self):
        self.assertEqual(_si_conversion("vorticity", "1/s", "cgs"), (1.0, "1/s"))

    def test_frequency_comparison_uses_positive_physical_frequency(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            common = {
                "x_m": np.array([0.0235, 0.0265]),
                "probe_indices": np.array([120, 200]),
                "frequency_hz": np.array([0.0, 1e5, 2e5, 3e5]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(
                first,
                **common,
                amplitude=np.array(
                    [
                        [0.0, 0.0],
                        [10.0, 9.0],
                        [5.0, 4.0],
                        [2.0, 1.0],
                    ]
                ),
            )
            np.savez(
                second,
                **common,
                amplitude=np.array(
                    [
                        [0.0, 0.0],
                        [8.0, 7.0],
                        [6.0, 5.0],
                        [3.0, 2.0],
                    ]
                ),
            )
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first,
                    second,
                    "asym.spectral.probe_signals",
                    "gaus.spectral.probe_signals",
                    "pressure",
                    directory / "frequency.png",
                    kind="fft",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                for axis in figure.axes:
                    self.assertEqual(len(axis.lines), 2)
                    np.testing.assert_allclose(axis.lines[0].get_xdata(), [1e5, 2e5, 3e5])
                    self.assertEqual(axis.get_xscale(), "log")
            finally:
                plt.close(figure)

    def test_fft_ratio_plot_honors_frequency_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "x_m": np.array([0.025]),
                "probe_indices": np.array([160]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(
                first,
                **common,
                frequency_hz=np.array([0.0, 1.0, 2.0]),
                amplitude=np.array([[0.0], [1.0], [2.0]]),
            )
            np.savez(
                second,
                **common,
                frequency_hz=np.array([0.0, 1.5, 3.0]),
                amplitude=np.array([[0.0], [1.5], [3.0]]),
            )
            with self.assertRaises(ValueError):
                render_probe_panels(
                    first, second, "baseline", "comparison", "pressure",
                    directory / "strict.png", kind="fft_amplitude_linear_ratio",
                )
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_probe_panels(
                    first, second, "baseline", "comparison", "pressure",
                    directory / "aligned.png", kind="fft_amplitude_linear_ratio",
                    alignment=ComparisonAlignment(frequency="interpolate_to_baseline"),
                )
            figure = close.call_args.args[0]
            try:
                np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [1.0, 2.0])
                np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [1.0, 1.0])
            finally:
                plt.close(figure)

    def test_unit_l2_fk_support_is_scale_invariant(self):
        baseline = np.array([[1.0, 4.0], [9.0, 16.0]])
        for scale in (1.0e-12, 1.0e12):
            ratio_a, ratio_b, support, basis = _fk_ratio_components(
                baseline, baseline * scale, "unit_l2", 0.01
            )
            self.assertEqual(basis, "normalized_own_peak")
            self.assertTrue(np.all(support))
            np.testing.assert_allclose(
                np.sqrt(ratio_b[support] / ratio_a[support]), 1.0
            )

    def test_psd_ratio_confidence_uses_native_bins_and_effective_dof(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "frequency_hz": np.array([0.0, 1.0, 2.0]),
                "probe_indices": np.array([7]),
                "x_m": np.array([0.25]),
            }
            np.savez(first, **common, psd=np.array([[0.0], [2.0], [4.0]]))
            np.savez(second, **common, psd=np.array([[0.0], [4.0], [8.0]]))
            confidence = {
                "segment_count": 8,
                "effective_degrees_of_freedom": 12.0,
            }
            metadata = {
                "kind": "array",
                "variable": "pressure",
                "units": "Pa^2/Hz",
                "provenance": {"welch_confidence": confidence},
            }
            payload, arrays = _psd_ratio_confidence(
                _Product("baseline", metadata, first),
                _Product("comparison", metadata, second),
                ComparisonAlignment(),
                (1.0, 2.0),
                0.01,
                0.95,
            )
            self.assertEqual(payload["status"], "available")
            self.assertIsNotNone(arrays)
            np.testing.assert_allclose(arrays["ratio"], 2.0)
            self.assertTrue(np.all(arrays["lower"] < 2.0))
            self.assertTrue(np.all(arrays["upper"] > 2.0))

    def test_normalized_fft_ratio_identifies_enrichment_and_masks_weak_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            common = {
                "x_m": np.array([0.025]),
                "probe_indices": np.array([160]),
                "frequency_hz": np.array([0.0, 1e5, 2e5, 3e5]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(first, **common, amplitude=np.array([[0.0], [10.0], [1.0], [0.001]]))
            np.savez(second, **common, amplitude=np.array([[0.0], [1.0], [10.0], [0.001]]))
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first,
                    second,
                    "asym.spectral.probe_signals",
                    "gaus.spectral.probe_signals",
                    "pressure",
                    directory / "ratio.png",
                    kind="fft_shape_ratio",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                ratio = np.asarray(figure.axes[0].lines[0].get_ydata())
                self.assertLess(ratio[0], 0)
                self.assertGreater(ratio[1], 0)
                self.assertTrue(np.isnan(ratio[2]))
            finally:
                plt.close(figure)

    def test_absolute_fft_ratio_preserves_overall_gain_and_masks_weak_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            common = {
                "x_m": np.array([0.025]),
                "probe_indices": np.array([160]),
                "frequency_hz": np.array([0.0, 1e5, 2e5, 3e5]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(first, **common, amplitude=np.array([[0.0], [10.0], [5.0], [0.001]]))
            np.savez(second, **common, amplitude=np.array([[0.0], [20.0], [10.0], [0.001]]))
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first,
                    second,
                    "asym.spectral.probe_signals",
                    "gaus.spectral.probe_signals",
                    "pressure",
                    directory / "absolute_ratio.png",
                    kind="fft_amplitude_ratio",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                axis = figure.axes[0]
                np.testing.assert_allclose(axis.lines[0].get_ydata()[:2], 20 * np.log10(2))
                self.assertTrue(np.isnan(axis.lines[0].get_ydata()[2]))
                self.assertEqual(axis.get_xscale(), "log")
                self.assertEqual(axis.get_yscale(), "linear")
            finally:
                plt.close(figure)

    def test_linear_fft_ratio_is_centered_on_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            common = {
                "x_m": np.array([0.025]),
                "probe_indices": np.array([160]),
                "frequency_hz": np.array([0.0, 1e5, 2e5]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(first, **common, amplitude=np.array([[0.0], [10.0], [5.0]]))
            np.savez(second, **common, amplitude=np.array([[0.0], [20.0], [2.5]]))
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first,
                    second,
                    "asym.spectral.probe_signals",
                    "gaus.spectral.probe_signals",
                    "pressure",
                    directory / "linear_ratio.png",
                    kind="fft_amplitude_linear_ratio",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                ratio = figure.axes[0].lines[0].get_ydata()
                np.testing.assert_allclose(ratio, [2.0, 0.5])
                self.assertEqual(figure.axes[0].lines[1].get_ydata()[0], 1.0)
                self.assertEqual(figure.axes[0].get_yscale(), "linear")
            finally:
                plt.close(figure)

    def test_unit_normalized_linear_fft_ratio_removes_overall_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "x_m": np.array([0.025]),
                "probe_indices": np.array([160]),
                "frequency_hz": np.array([0.0, 1e5, 2e5]),
                "signal_unit": np.array("Pa"),
            }
            np.savez(first, **common, amplitude=np.array([[0.0], [10.0], [5.0]]))
            np.savez(second, **common, amplitude=np.array([[0.0], [20.0], [10.0]]))
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first, second,
                    "baseline.spectral.probe_signals",
                    "comparison.spectral.probe_signals",
                    "pressure", directory / "shape_linear.png",
                    kind="fft_shape_linear_ratio",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [1.0, 1.0])
                self.assertEqual(figure.axes[0].lines[1].get_ydata()[0], 1.0)
            finally:
                plt.close(figure)

    def test_paired_probe_panels_omit_two_flat_endpoint_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            common = {
                "x_m": np.linspace(0.019, 0.031, 6),
                "probe_indices": np.array([0, 120, 150, 170, 200, 320]),
                "frequency_hz": np.array([0.0, 1e5, 2e5]),
                "signal_unit": np.array("Pa"),
            }
            amplitudes = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 3.0, 4.0, 5.0, 0.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
                ]
            )
            np.savez(first, **common, amplitude=amplitudes)
            np.savez(second, **common, amplitude=0.8 * amplitudes)
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_probe_panels(
                    first,
                    second,
                    "asym-pressure-spectrum.spectral.probe_signals",
                    "gaus-pressure-spectrum.spectral.probe_signals",
                    "pressure",
                    directory / "four_probes.png",
                    kind="fft",
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                self.assertEqual(len(figure.axes), 4)
                titles = [axis.get_title() for axis in figure.axes]
                self.assertTrue(all("Probe 0 ·" not in title for title in titles))
                self.assertTrue(all("Probe 320 ·" not in title for title in titles))
                self.assertEqual(
                    [title.split(" ·")[0] for title in titles],
                    ["Probe 120", "Probe 200", "Probe 150", "Probe 170"],
                )
            finally:
                plt.close(figure)

    def test_signed_k_comparison_has_shared_axes_and_gated_power_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "asym.npz", directory / "gaus.npz"
            frequency = np.array([2e5, 3e5, 4e5])
            wavenumber = np.array([-1000.0, 0.0, 1000.0])
            a = np.array([[1.0, 4.0, 1.0], [0.1, 2.0, 0.1], [0.001, 1.0, 0.001]])
            b = 2 * a
            b[-1, -1] = 1e-6
            np.savez(first, frequency_hz=frequency, wavenumber_rad_m=wavenumber, power=a)
            np.savez(second, frequency_hz=frequency, wavenumber_rad_m=wavenumber, power=b)
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_komega_comparison(
                    first,
                    second,
                    "asym.wave.komega",
                    "gaus.wave.komega",
                    directory / "map.png",
                    directory / "signed_k.png",
                    (2e5, 4e5),
                    "(Pa)^2",
                )
            figures = [call.args[0] for call in close.call_args_list]
            try:
                self.assertIsNotNone(result)
                self.assertTrue(all(path.is_file() for path in result))
                ratio = np.asarray(figures[0].axes[2].collections[0].get_array()).reshape(3, 3)
                self.assertAlmostEqual(ratio[0, 1], 10 * np.log10(2))
                self.assertTrue(np.isnan(ratio[-1, -1]))
                self.assertEqual(figures[0].axes[0].get_ylabel(), "Frequency [MHz]")
                self.assertEqual(figures[0].axes[0].get_yscale(), "log")
                self.assertGreater(figures[0].axes[0].get_ylim()[0], 0.0)
            finally:
                for figure in figures:
                    plt.close(figure)

    def test_signed_fk_linear_ratio_uses_amplitude_not_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "frequency_hz": np.array([2e5, 3e5]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0, 1000.0]),
            }
            a = np.array([[1.0, 4.0, 1.0], [1.0, 2.0, 1.0]])
            np.savez(first, **common, power=a)
            np.savez(second, **common, power=4.0 * a)
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                result = render_komega_comparison(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "linear_map.png", None, (2e5, 3e5),
                    ratio_scale="linear",
                )
            figure = close.call_args.args[0]
            try:
                self.assertEqual(result, (directory / "linear_map.png", None))
                ratio = np.asarray(figure.axes[2].collections[0].get_array()).reshape(2, 3)
                np.testing.assert_allclose(ratio, 2.0)
                self.assertAlmostEqual(figure.axes[2].collections[0].norm(1.0), 0.25)
            finally:
                plt.close(figure)

    def test_fk_source_panels_use_shared_physical_amplitude_or_relative_db(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "frequency_hz": np.array([2e5, 3e5]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0]),
            }
            power_a = np.array([[1.0, 4.0], [9.0, 16.0]])
            power_b = 4.0 * power_a
            np.savez(first, **common, power=power_a, amplitude=np.sqrt(power_a))
            np.savez(second, **common, power=power_b, amplitude=np.sqrt(power_b))

            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_komega_comparison(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "amplitude.png", None, (2e5, 3e5), "(Pa)^2",
                )
            amplitude_figure = close.call_args.args[0]
            try:
                plotted = np.asarray(
                    amplitude_figure.axes[0].collections[0].get_array()
                ).reshape(power_a.shape)
                np.testing.assert_allclose(plotted, np.sqrt(power_a))
                self.assertEqual(amplitude_figure.axes[0].collections[0].norm.vmin, 0.0)
                self.assertEqual(amplitude_figure.axes[0].collections[0].norm.vmax, 8.0)
                self.assertEqual(amplitude_figure.axes[3].get_ylabel(), "Spectral amplitude [Pa]")
            finally:
                plt.close(amplitude_figure)

            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_komega_comparison(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "relative_db.png", None, (2e5, 3e5), "(Pa)^2",
                    spectral_display="relative_db",
                )
            db_figure = close.call_args.args[0]
            try:
                plotted = np.asarray(db_figure.axes[0].collections[0].get_array()).reshape(power_a.shape)
                np.testing.assert_allclose(plotted, 10.0 * np.log10(power_a / power_b.max()))
                self.assertEqual(db_figure.axes[0].collections[0].norm.vmin, -60.0)
                self.assertEqual(db_figure.axes[3].get_ylabel(), "Relative power [dB]")
            finally:
                plt.close(db_figure)

    def test_signed_k_amplitude_curve_and_frequency_slice_use_shared_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "frequency_hz": np.array([0.5e6, 1.0e6]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0, 1000.0]),
            }
            power_a = np.array([[1.0, 4.0, 1.0], [4.0, 9.0, 4.0]])
            power_b = 4.0 * power_a
            np.savez(first, **common, power=power_a)
            np.savez(second, **common, power=power_b)

            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_komega_comparison(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "map.png", directory / "signed_k.png",
                    (0.5e6, 1.0e6), "(Pa)^2",
                )
            figures = [call.args[0] for call in close.call_args_list]
            try:
                spectrum = figures[1]
                np.testing.assert_allclose(
                    spectrum.axes[0].lines[0].get_ydata(), np.sqrt(np.sum(power_a, axis=0))
                )
                self.assertEqual(spectrum.axes[0].get_yscale(), "linear")
                self.assertIn("[Pa]", spectrum.axes[0].get_ylabel())
            finally:
                for figure in figures:
                    plt.close(figure)

            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_komega_amplitude_frequency_slices(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "slices.png", (0.5e6, 1.0e6), power_unit="(Pa)^2",
                )
            slices = close.call_args.args[0]
            try:
                self.assertEqual(len(slices.axes[0].lines), 2)
                self.assertEqual(slices.axes[0].lines[0].get_ydata()[1], 2.0)
                self.assertEqual(slices.axes[0].lines[1].get_ydata()[1], 4.0)
                self.assertEqual(slices.axes[0].get_ylim(), slices.axes[1].get_ylim())
                self.assertTrue(
                    any(text.get_text() == "Spectral amplitude [Pa]" for text in slices.texts)
                )
            finally:
                plt.close(slices)

    def test_time_localized_amplitude_maps_share_a_linear_record_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis = DirectionalWaveAnalysis(
                id="wave", recipe="directional_wave", probe_set_id="probe",
                variable="pressure", frequency_max_hz=2.0e6,
                temporal_wavenumber=TemporalWavenumberEnabled(
                    enabled=True, window_duration_s=1.0e-6,
                ),
            )
            registered = []
            context = SimpleNamespace(
                analysis=analysis,
                project=SimpleNamespace(
                    analyses_file=SimpleNamespace(presentation=PresentationConfig())
                ),
                figure_dir=directory,
                register=lambda **record: registered.append(record),
            )
            band_power = np.array([[1.0, 4.0], [9.0, 16.0]])
            snapshot_power = np.array([
                [[1.0, 4.0], [4.0, 1.0]],
                [[9.0, 16.0], [16.0, 9.0]],
            ])
            summary = {
                "preprocessing": {"window_duration_s": 1.0e-6},
                "snapshot_roles": np.array(["early", "late"]),
                "valid_time_mask": np.array([True, True]),
                "time_center_s": np.array([1.0e-6, 2.0e-6]),
                "snapshot_time_s": np.array([1.0e-6, 2.0e-6]),
                "snapshot_valid_mask": np.array([True, True]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0]),
                "band_power": band_power,
                "relative_band_power_db": 10.0 * np.log10(band_power / 16.0),
                "dominant_wavenumber_rad_m": np.array([0.0, 0.0]),
                "dominant_frequency_hz": np.array([1.0e6, 1.0e6]),
                "wavelength_m": np.array([np.inf, np.inf]),
                "phase_speed_m_s": np.array([1.0, 1.0]),
            }
            snapshots = {
                "snapshot_time_s": np.array([1.0e-6, 2.0e-6]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0]),
                "frequency_hz": np.array([1.0e6, 2.0e6]),
                "power": snapshot_power,
                "relative_power_db": 10.0 * np.log10(snapshot_power / 16.0),
                "valid_snapshot_mask": np.array([True, True]),
            }
            with patch("pelecpost.analysis.temporal_wavenumber.plt.close") as close:
                register_temporal_wavenumber_figures(
                    context, values=np.ones((2, 2)), time_s=np.array([1.0e-6, 2.0e-6]),
                    x_m=np.array([0.0, 1.0]), variable="pressure", units="Pa",
                    summary=summary, snapshots=snapshots,
                )
            figures = list({
                id(call.args[0]): call.args[0] for call in close.call_args_list
            }.values())
            try:
                self.assertEqual(len(figures), 4)
                self.assertEqual(figures[0].axes[0].collections[0].norm.vmax, 4.0)
                snapshot_axes = figures[3].axes[:2]
                self.assertTrue(all(axis.collections[0].norm.vmin == 0.0 for axis in snapshot_axes))
                self.assertTrue(all(axis.collections[0].norm.vmax == 4.0 for axis in snapshot_axes))
                self.assertEqual(registered[0]["units"], "Pa")
                self.assertEqual(registered[-1]["units"], "Pa")
            finally:
                for figure in figures:
                    plt.close(figure)

    def test_signed_fk_unit_normalized_linear_ratio_removes_overall_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "baseline.npz", directory / "comparison.npz"
            common = {
                "frequency_hz": np.array([2e5, 3e5]),
                "wavenumber_rad_m": np.array([-1000.0, 0.0, 1000.0]),
            }
            a = np.array([[1.0, 4.0, 1.0], [1.0, 2.0, 1.0]])
            np.savez(first, **common, power=a)
            np.savez(second, **common, power=4.0 * a)
            with patch("pelecpost.analysis.comparison_figures.plt.close") as close:
                render_komega_comparison(
                    first, second, "baseline.wave.komega", "comparison.wave.komega",
                    directory / "shape_map.png", None, (2e5, 3e5),
                    ratio_scale="linear", ratio_normalization="unit_l2",
                )
            figure = close.call_args.args[0]
            try:
                ratio = np.asarray(figure.axes[2].collections[0].get_array()).reshape(2, 3)
                np.testing.assert_allclose(ratio, 1.0)
            finally:
                plt.close(figure)

    def test_probe_comparison_figure_uses_shared_time_axes_and_paired_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            figure_dir = Path(temporary)
            paths = [figure_dir / "asym.npz", figure_dir / "gaus.npz"]
            for path, peak in zip(paths, (900.0, 800.0)):
                np.savez(
                    path,
                    x_m=np.array([0.0235, 0.0265]),
                    probe_indices=np.array([120, 200]),
                    raw_time_s=np.array([0.0, 1e-6, 2e-6]),
                    raw_values=np.array([[760.0, 760.0], [peak, peak], [760.0, 760.0]]),
                    signal_unit=np.array("Pa"),
                )
            left = _Product(
                "asym-spectrum.spectral.probe_signals",
                {"variable": "pressure"},
                paths[0],
            )
            right = _Product(
                "gaus-spectrum.spectral.probe_signals",
                {"variable": "pressure"},
                paths[1],
            )
            with patch("pelecpost.analysis.comparison.plt.close") as close:
                result = _render_product_overlay(
                    SimpleNamespace(figure_dir=figure_dir),
                    left,
                    right,
                )
            figure = close.call_args.args[0]
            try:
                self.assertTrue(result.is_file())
                self.assertEqual(len(figure.axes), 2)
                for axis in figure.axes:
                    self.assertEqual(len(axis.lines), 2)
                    np.testing.assert_allclose(axis.lines[0].get_xdata(), [0, 1, 2])
                    self.assertEqual(axis.get_xlabel(), "Time [µs]")
            finally:
                plt.close(figure)

    def test_probe_input_requires_exactly_one_nonempty_source(self):
        with self.assertRaisesRegex(ValueError, "requires compact_file or binary_files"):
            ProbeInput()
        with self.assertRaisesRegex(ValueError, "not both"):
            ProbeInput(compact_file="probes.h5", binary_files=("probes.pbin",))
        with self.assertRaisesRegex(ValueError, "compact_file cannot be empty"):
            ProbeInput(compact_file="")
        with self.assertRaisesRegex(ValueError, "empty paths"):
            ProbeInput(binary_files=("",))

    def test_generated_schema_contains_source_one_of_and_new_comparison_contract(self):
        schema = __import__("pelecpost.config.models", fromlist=["MachineFile", "AnalysesFile"])
        machine_schema = schema.MachineFile.model_json_schema()
        probe_schema = machine_schema["$defs"]["ProbeInput"]
        self.assertEqual(len(probe_schema["oneOf"]), 2)
        self.assertIn("probe_sets", machine_schema["$defs"]["InputConfig"]["properties"])
        self.assertNotIn("probes", machine_schema["$defs"]["InputConfig"]["properties"])
        comparison_schema = schema.CaseComparisonAnalysis.model_json_schema()
        self.assertIn("product_ids", comparison_schema["properties"])
        self.assertIn("alignment", comparison_schema["properties"])

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "project.schema.json"
            write_project_schema(destination)
            text = destination.read_text(encoding="utf-8")
            self.assertIn('"probe_sets"', text)
            self.assertIn('"archived_runs"', text)
            self.assertIn('"interpolate_to_baseline"', text)

    def test_local_comparison_adds_analysis_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = preflight_fixtures.PreflightTests().project(
                Path(temporary),
                [
                    {"id": "asym", "recipe": "probe_spectrum", "variable": "pressure"},
                    {"id": "gaus", "recipe": "probe_spectrum", "variable": "pressure"},
                    {
                        "id": "compare",
                        "recipe": "case_comparison",
                        "baseline": {"analysis_id": "asym"},
                        "comparison": {"analysis_id": "gaus"},
                        "product_ids": ["spectral.psd"],
                    },
                ],
            )
            graph = build_workflow_graph(project)
            self.assertEqual(
                set(graph.node("analysis.compare").dependencies),
                {"analysis.asym", "analysis.gaus"},
            )

    def test_hdf5_and_probe_v2_share_the_same_si_loader(self):
        helper = bounded_fixtures.BoundedProbeWorkspaceTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact = root / "probes.h5"
            binary = root / "probes.segment0000.pbin"
            compact_values = helper.write_archive(compact, samples=17, probes=5)
            binary_values = helper.write_probe_v2(binary, samples=17, probes=5)
            selected = np.array([4, 1, 3])
            hdf5_workspace = open_probe_signal_workspace(
                compact,
                "p",
                selected,
                si_factor=0.1,
                memory_limit_gb=1.0,
            )
            pbin_workspace = open_probe_signal_workspace(
                (str(binary),),
                "p",
                selected,
                si_factor=0.1,
                memory_limit_gb=1.0,
            )
            try:
                np.testing.assert_array_equal(hdf5_workspace.time_s, pbin_workspace.time_s)
                np.testing.assert_array_equal(hdf5_workspace.x_m, pbin_workspace.x_m)
                np.testing.assert_allclose(
                    hdf5_workspace.values,
                    compact_values[:, selected] * 0.1,
                )
                np.testing.assert_allclose(
                    pbin_workspace.values,
                    binary_values[:, selected] * 0.1,
                )
                np.testing.assert_allclose(
                    hdf5_workspace.values,
                    pbin_workspace.values,
                )
            finally:
                hdf5_workspace.close()
                pbin_workspace.close()

    def test_resampling_scratch_is_removed_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            time = np.array([0.0, 1.0, 2.2, 3.2]) * 1.0e-6
            values = np.column_stack((time, time**2))
            _, _, _, resampled, cleanup = _prepare_fft_grid(
                time,
                values,
                "resample_uniform",
                scratch,
            )
            self.assertTrue(resampled)
            self.assertIsNotNone(cleanup)
            self.assertTrue(list(scratch.glob("*.mmap")))
            assert cleanup is not None
            cleanup()
            self.assertFalse(list(scratch.glob("*.mmap")))

    def test_spectrum_failure_cleans_unregistered_resampling_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            time = np.array([0.0, 1.0, 2.2, 3.2]) * 1.0e-6
            values = np.column_stack((time, time**2))
            with self.assertRaises(ValueError):
                spectrum_from_signal(
                    time,
                    values,
                    end_time_s=None,
                    window="invalid",
                    detrend="mean",
                    welch_segment_samples=None,
                    overlap_fraction=0.5,
                    time_grid_policy="resample_uniform",
                    frequency_max_hz=None,
                    scratch_directory=scratch,
                )
            self.assertFalse(list(scratch.glob("*.mmap")))

    def test_nonuniform_source_is_a_preflight_blocker_for_require_uniform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = preflight_fixtures.PreflightTests().project(
                root,
                [
                    {
                        "id": "spectrum",
                        "recipe": "probe_spectrum",
                        "variable": "pressure",
                        "time_grid_policy": "require_uniform",
                    }
                ],
            )
            with h5py.File(root / "probes.h5", "r+") as archive:
                time = np.arange(1024, dtype=float) * 1.0e-6
                time[2] = 2.1e-6
                archive["time"][...] = time
            plan = create_plan(project)
            self.assertIn("NONUNIFORM_TIME", {item.code for item in plan.blockers})

    def test_strict_alignment_matches_by_physical_coordinates_not_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            frequency = np.array([10.0, 20.0])
            x_left = np.array([0.0, 1.0, 2.0])
            left_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            # The right file reverses both physical axes and uses unrelated labels.
            np.savez(left_path, frequency_hz=frequency, x_m=x_left, psd=left_values)
            np.savez(
                right_path,
                frequency_hz=frequency[::-1],
                x_m=x_left[::-1],
                probe_indices=np.array([902, 901, 900]),
                psd=left_values[::-1, ::-1],
            )
            left = _Product("left", _product_metadata(left_path, "spectral.psd"), left_path)
            right = _Product("right", _product_metadata(right_path, "spectral.psd"), right_path)
            metrics, details = _array_metrics(left, right, ComparisonAlignment())
            self.assertEqual(metrics[0]["linf_difference"], 0.0)
            self.assertEqual(details["psd"]["frequency"]["policy"], "strict")
            self.assertEqual(details["psd"]["space"]["policy"], "strict")

    def test_intersection_and_interpolation_policies_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            np.savez(
                left_path,
                frequency_hz=np.array([1.0, 2.0, 3.0]),
                x_m=np.array([0.0, 1.0]),
                psd=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
            )
            np.savez(
                right_path,
                frequency_hz=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                x_m=np.array([0.0, 1.0]),
                psd=np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]]),
            )
            left = _Product("left", _product_metadata(left_path, "spectral.psd"), left_path)
            right = _Product("right", _product_metadata(right_path, "spectral.psd"), right_path)
            alignment = ComparisonAlignment(frequency="interpolate_to_baseline")
            metrics, details = _array_metrics(left, right, alignment)
            self.assertEqual(metrics[0]["linf_difference"], 0.0)
            self.assertEqual(details["psd"]["frequency"]["policy"], "interpolate_to_baseline")

            right_intersection = root / "right-intersection.npz"
            np.savez(
                right_intersection,
                frequency_hz=np.array([2.0, 3.0, 4.0]),
                x_m=np.array([0.0, 1.0]),
                psd=np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            )
            right = _Product(
                "right-intersection",
                _product_metadata(right_intersection, "spectral.psd"),
                right_intersection,
            )
            metrics, details = _array_metrics(
                left,
                right,
                ComparisonAlignment(frequency="intersection"),
            )
            self.assertEqual(metrics[0]["linf_difference"], 0.0)
            self.assertEqual(details["psd"]["frequency"]["sample_count"], 2)

    def test_coordinate_mismatch_and_complex_semantics_fail_or_compare_actionably(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            np.savez(
                left_path,
                frequency_hz=np.array([1.0, 2.0]),
                x_m=np.array([0.0, 1.0]),
                psd=np.ones((2, 2)),
            )
            np.savez(
                right_path,
                frequency_hz=np.array([1.0, 2.0]),
                x_m=np.array([0.0, 2.0]),
                psd=np.ones((2, 2)),
            )
            left = _Product("left", _product_metadata(left_path, "spectral.psd"), left_path)
            right = _Product("right", _product_metadata(right_path, "spectral.psd"), right_path)
            with self.assertRaisesRegex(ValueError, "strict comparison axes differ"):
                _array_metrics(left, right, ComparisonAlignment())

            complex_left = root / "complex-left.npz"
            complex_right = root / "complex-right.npz"
            np.savez(
                complex_left,
                frequency_hz=np.array([1.0, 2.0]),
                relative_time_s=np.array([0.0, 1.0]),
                complex_stft=np.array([[1 + 1j, 2 + 0j], [3 + 4j, 5 + 0j]]),
            )
            np.savez(
                complex_right,
                frequency_hz=np.array([1.0, 2.0]),
                relative_time_s=np.array([0.0, 1.0]),
                complex_stft=np.array([[1 - 1j, 2 + 0j], [3 - 4j, 5 + 0j]]),
            )
            left = _Product("left", _product_metadata(complex_left, "transient.stft"), complex_left)
            right = _Product(
                "right", _product_metadata(complex_right, "transient.stft"), complex_right
            )
            metrics, _ = _array_metrics(left, right, ComparisonAlignment())
            self.assertEqual(metrics[0]["linf_difference"], 0.0)

    def test_declared_frequency_masks_exclude_invalid_values_from_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            frequency = np.array([1.0, 2.0, 3.0])
            x = np.array([0.0, 1.0])
            common = {
                "frequency_hz": frequency,
                "probe_x_m": x,
                "source_power_spectrum_w_m": np.ones(3),
                "transfer": np.zeros((3, 2)),
                "transfer_magnitude": np.zeros((3, 2)),
                "transfer_phase_rad": np.zeros((3, 2)),
                "baseline": np.zeros(2),
                "valid_frequency": np.array([True, False, True]),
            }
            left_arrays = {key: value.copy() for key, value in common.items()}
            right_arrays = {key: value.copy() for key, value in common.items()}
            right_arrays["transfer"][1] = 1.0e9
            np.savez(left_path, **left_arrays)
            np.savez(right_path, **right_arrays)
            left = _Product("left", _product_metadata(left_path, "pulse.transfer"), left_path)
            right = _Product("right", _product_metadata(right_path, "pulse.transfer"), right_path)
            metrics, details = _array_metrics(left, right, ComparisonAlignment())
            transfer = next(item for item in metrics if item["quantity"] == "transfer")
            self.assertEqual(transfer["linf_difference"], 0.0)
            self.assertEqual(details["transfer"]["masked_count_baseline"], 2)
            self.assertEqual(details["transfer"]["valid_count_used"], 4)

    def test_growth_derived_wavenumber_values_require_growth_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "wave.npz"
            arrays = {
                "frequency_hz": np.array([1.0, 2.0]),
                "x_center_m": np.array([0.0, 1.0]),
                "alpha_real_rad_m": np.ones((2, 2)),
                "alpha_imag_rad_m": np.ones((2, 2)),
                "alpha_real_ci95_rad_m": np.ones((2, 2)),
                "alpha_imag_ci95_rad_m": np.ones((2, 2)),
                "amplification_rate_per_m": np.ones((2, 2)),
                "phase_speed_m_s": np.ones((2, 2)),
                "coherence_squared": np.ones((2, 2)),
                "phase_fit_r_squared": np.ones((2, 2)),
                "amplitude_fit_r_squared": np.ones((2, 2)),
                "spatial_alias_margin": np.ones((2, 2)),
                "phase_valid_mask": np.ones((2, 2), dtype=bool),
            }
            np.savez(path, **arrays)
            product = _Product("wave", _product_metadata(path, "wave.wavenumber"), path)
            with self.assertRaisesRegex(ValueError, "growth_valid_mask"):
                _array_metrics(product, product, ComparisonAlignment())

    def test_declared_mask_with_ambiguous_shape_fails_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "wave.npz"
            arrays = {
                "frequency_hz": np.array([1.0, 2.0]),
                "x_center_m": np.array([0.0, 1.0]),
                "alpha_real_rad_m": np.ones((2, 2)),
                "alpha_imag_rad_m": np.ones((2, 2)),
                "alpha_real_ci95_rad_m": np.ones((2, 2)),
                "alpha_imag_ci95_rad_m": np.ones((2, 2)),
                "amplification_rate_per_m": np.ones((2, 2)),
                "phase_speed_m_s": np.ones((2, 2)),
                "coherence_squared": np.ones((2, 2)),
                "phase_fit_r_squared": np.ones((2, 2)),
                "amplitude_fit_r_squared": np.ones((2, 2)),
                "spatial_alias_margin": np.ones((2, 2)),
                "phase_valid_mask": np.ones(2, dtype=bool),
                "growth_valid_mask": np.ones((2, 2), dtype=bool),
            }
            np.savez(path, **arrays)
            product = _Product("wave", _product_metadata(path, "wave.wavenumber"), path)
            with self.assertRaisesRegex(ValueError, "phase_valid_mask"):
                _array_metrics(product, product, ComparisonAlignment())

    def test_typed_product_adapters_cover_wave_nonlinear_and_modal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frequency = np.array([1.0, 2.0])
            x = np.array([0.0, 1.0, 2.0])

            products = {
                "transient.stft": {
                    "frequency_hz": frequency,
                    "relative_time_s": np.array([0.0, 1.0, 2.0]),
                    "complex_stft": np.ones((2, 3), dtype=complex),
                },
                "wave.wavenumber": {
                    "frequency_hz": frequency,
                    "x_center_m": x,
                    **{
                        key: np.ones((2, 3))
                        for key in (
                            "alpha_real_rad_m",
                            "alpha_imag_rad_m",
                            "alpha_real_ci95_rad_m",
                            "alpha_imag_ci95_rad_m",
                            "amplification_rate_per_m",
                            "phase_speed_m_s",
                            "coherence_squared",
                            "phase_fit_r_squared",
                            "amplitude_fit_r_squared",
                            "spatial_alias_margin",
                            "phase_valid_mask",
                            "growth_valid_mask",
                        )
                    },
                },
                "wave.komega": {
                    "frequency_hz": frequency,
                    "wavenumber_rad_m": np.array([-1.0, 0.0, 1.0]),
                    "power": np.ones((2, 3)),
                    "amplitude": np.ones((2, 3)),
                },
                "nonlinear.bicoherence": {
                    "triad_labels": np.array(["f1+f2", "f1+f3", "f2+f3"]),
                    **{
                        key: np.ones(3)
                        for key in (
                            "observed_bicoherence_squared",
                            "surrogate_median_bicoherence_squared",
                            "surrogate_95_bicoherence_squared",
                            "empirical_p_value",
                            "fdr_adjusted_p_value",
                            "significant_fdr",
                        )
                    },
                },
                "modal.pod": {
                    "coordinates_m": x,
                    "modes": np.ones((3, 2)),
                    "temporal_coefficients": np.ones((4, 2)),
                    "singular_values": np.ones(2),
                    "energy_fraction": np.ones(2),
                    "mean": np.ones(3),
                },
                "modal.spod": {
                    "frequency_hz": frequency,
                    "coordinates_m": x,
                    "eigenvalues": np.ones((2, 2)),
                    "modes": np.ones((2, 2, 3), dtype=complex),
                },
                "modal.dmd": {
                    "frequency_hz": frequency,
                    "coordinates_m": x,
                    "modes": np.ones((3, 2), dtype=complex),
                    "eigenvalues": np.ones(2, dtype=complex),
                    "growth_rate_per_s": np.ones(2),
                    "amplitudes": np.ones(2, dtype=complex),
                    "retained_condition_number": np.array(1.0),
                },
            }
            for product_id, arrays in products.items():
                left_path = root / f"{product_id.replace('.', '-')}-left.npz"
                right_path = root / f"{product_id.replace('.', '-')}-right.npz"
                np.savez(left_path, **arrays)
                np.savez(right_path, **arrays)
                left = _Product(
                    "left",
                    _product_metadata(left_path, product_id),
                    left_path,
                )
                right = _Product(
                    "right",
                    _product_metadata(right_path, product_id),
                    right_path,
                )
                metrics, _ = _array_metrics(left, right, ComparisonAlignment())
                self.assertTrue(metrics, product_id)

    def test_time_localized_wavenumber_recovers_signed_ridge_and_masks_tail(self):
        samples = 2048
        dt = 1.0e-6
        time = np.arange(samples) * dt
        x = np.arange(32, dtype=float) * 2.0e-3
        frequency = 25_000.0
        wavenumber = 2.0 * np.pi / (len(x) * (x[1] - x[0])) * 2.0
        envelope = np.exp(-(((time - 0.0010) / 0.00012) ** 2))
        values = envelope[:, None] * np.cos(
            2.0 * np.pi * frequency * time[:, None] - wavenumber * x[None, :]
        )
        analysis = __import__(
            "pelecpost.config.models", fromlist=["DirectionalWaveAnalysis"]
        ).DirectionalWaveAnalysis(
            id="wave",
            recipe="directional_wave",
            probe_set_id="default",
            variable="pressure",
            frequency_min_hz=15_000.0,
            frequency_max_hz=35_000.0,
            expected_speed_min_m_s=100.0,
            expected_speed_max_m_s=2_000.0,
            temporal_wavenumber={
                "enabled": True,
                "window_duration_s": 256.0e-6,
                "minimum_relative_energy_db": -20.0,
            },
        )
        result = compute_temporal_wavenumber(values, time, x, analysis)
        valid = result["valid_time_mask"]
        self.assertGreater(np.count_nonzero(valid), 0)
        recovered = result["dominant_wavenumber_rad_m"][valid]
        self.assertTrue(np.all(recovered > 0.0))
        self.assertLess(float(np.median(abs(recovered - wavenumber))), 1.0e-10)
        self.assertTrue(np.any(~result["active_mask"]))

    def test_json_products_compare_only_declared_typed_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text(
                '{"segment_count": 2, "frequency_resolution_hz": 4.0, '
                '"interpretation": "baseline"}\n',
                encoding="utf-8",
            )
            right.write_text(
                '{"segment_count": 2, "frequency_resolution_hz": 4.0, '
                '"interpretation": "different label"}\n',
                encoding="utf-8",
            )
            metrics = _json_metrics(
                left,
                right,
                ("segment_count", "frequency_resolution_hz"),
            )
            self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
