from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pelecpost.analysis.comparison import _Product, _array_metrics, _json_metrics
from pelecpost.analysis.products import product_contract
from pelecpost.analysis.spectral import _prepare_fft_grid, spectrum_from_signal
from pelecpost.analysis.temporal_wavenumber import compute_temporal_wavenumber
from pelecpost.config.models import (
    CaseComparisonAnalysis,
    ComparisonAlignment,
    ComparisonReference,
    ProbeInput,
)
from pelecpost.config.loader import write_project_schema
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
            project = preflight_fixtures.PreflightTests().project(Path(temporary), [
                {"id": "asym", "recipe": "probe_spectrum", "variable": "pressure"},
                {"id": "gaus", "recipe": "probe_spectrum", "variable": "pressure"},
                {
                    "id": "compare", "recipe": "case_comparison",
                    "baseline": {"analysis_id": "asym"},
                    "comparison": {"analysis_id": "gaus"},
                    "product_ids": ["spectral.psd"],
                },
            ])
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
                compact, "p", selected, si_factor=0.1, memory_limit_gb=1.0,
            )
            pbin_workspace = open_probe_signal_workspace(
                (str(binary),), "p", selected, si_factor=0.1, memory_limit_gb=1.0,
            )
            try:
                np.testing.assert_array_equal(hdf5_workspace.time_s, pbin_workspace.time_s)
                np.testing.assert_array_equal(hdf5_workspace.x_m, pbin_workspace.x_m)
                np.testing.assert_allclose(
                    hdf5_workspace.values, compact_values[:, selected] * 0.1,
                )
                np.testing.assert_allclose(
                    pbin_workspace.values, binary_values[:, selected] * 0.1,
                )
                np.testing.assert_allclose(
                    hdf5_workspace.values, pbin_workspace.values,
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
                time, values, "resample_uniform", scratch,
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
                    time, values, end_time_s=None, window="invalid",
                    detrend="mean", welch_segment_samples=None,
                    overlap_fraction=0.5, time_grid_policy="resample_uniform",
                    frequency_max_hz=None, scratch_directory=scratch,
                )
            self.assertFalse(list(scratch.glob("*.mmap")))

    def test_nonuniform_source_is_a_preflight_blocker_for_require_uniform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = preflight_fixtures.PreflightTests().project(root, [{
                "id": "spectrum", "recipe": "probe_spectrum", "variable": "pressure",
                "time_grid_policy": "require_uniform",
            }])
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
                right_path, frequency_hz=frequency[::-1], x_m=x_left[::-1],
                probe_indices=np.array([902, 901, 900]), psd=left_values[::-1, ::-1],
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
                left_path, frequency_hz=np.array([1.0, 2.0, 3.0]),
                x_m=np.array([0.0, 1.0]), psd=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
            )
            np.savez(
                right_path, frequency_hz=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
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
                right_intersection, frequency_hz=np.array([2.0, 3.0, 4.0]),
                x_m=np.array([0.0, 1.0]), psd=np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            )
            right = _Product(
                "right-intersection",
                _product_metadata(right_intersection, "spectral.psd"),
                right_intersection,
            )
            metrics, details = _array_metrics(
                left, right, ComparisonAlignment(frequency="intersection"),
            )
            self.assertEqual(metrics[0]["linf_difference"], 0.0)
            self.assertEqual(details["psd"]["frequency"]["sample_count"], 2)

    def test_coordinate_mismatch_and_complex_semantics_fail_or_compare_actionably(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            np.savez(left_path, frequency_hz=np.array([1.0, 2.0]), x_m=np.array([0.0, 1.0]), psd=np.ones((2, 2)))
            np.savez(right_path, frequency_hz=np.array([1.0, 2.0]), x_m=np.array([0.0, 2.0]), psd=np.ones((2, 2)))
            left = _Product("left", _product_metadata(left_path, "spectral.psd"), left_path)
            right = _Product("right", _product_metadata(right_path, "spectral.psd"), right_path)
            with self.assertRaisesRegex(ValueError, "strict comparison axes differ"):
                _array_metrics(left, right, ComparisonAlignment())

            complex_left = root / "complex-left.npz"
            complex_right = root / "complex-right.npz"
            np.savez(
                complex_left, frequency_hz=np.array([1.0, 2.0]),
                relative_time_s=np.array([0.0, 1.0]),
                complex_stft=np.array([[1 + 1j, 2 + 0j], [3 + 4j, 5 + 0j]]),
            )
            np.savez(
                complex_right, frequency_hz=np.array([1.0, 2.0]),
                relative_time_s=np.array([0.0, 1.0]),
                complex_stft=np.array([[1 - 1j, 2 + 0j], [3 - 4j, 5 + 0j]]),
            )
            left = _Product("left", _product_metadata(complex_left, "transient.stft"), complex_left)
            right = _Product("right", _product_metadata(complex_right, "transient.stft"), complex_right)
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
                "source_spectrum_j_m": np.ones(3),
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
                    "frequency_hz": frequency, "x_center_m": x,
                    **{
                        key: np.ones((2, 3)) for key in (
                            "alpha_real_rad_m", "alpha_imag_rad_m",
                            "alpha_real_ci95_rad_m", "alpha_imag_ci95_rad_m",
                            "amplification_rate_per_m", "phase_speed_m_s",
                            "coherence_squared", "phase_fit_r_squared",
                            "amplitude_fit_r_squared", "spatial_alias_margin",
                            "phase_valid_mask", "growth_valid_mask",
                        )
                    },
                },
                "wave.komega": {
                    "frequency_hz": frequency,
                    "wavenumber_rad_m": np.array([-1.0, 0.0, 1.0]),
                    "power": np.ones((2, 3)), "amplitude": np.ones((2, 3)),
                },
                "nonlinear.bicoherence": {
                    "triad_labels": np.array(["f1+f2", "f1+f3", "f2+f3"]),
                    **{
                        key: np.ones(3) for key in (
                        "observed_bicoherence_squared",
                        "surrogate_median_bicoherence_squared",
                        "surrogate_95_bicoherence_squared", "empirical_p_value",
                        "fdr_adjusted_p_value", "significant_fdr",
                        )
                    },
                },
                "modal.pod": {
                    "coordinates_m": x, "modes": np.ones((3, 2)),
                    "temporal_coefficients": np.ones((4, 2)),
                    "singular_values": np.ones(2), "energy_fraction": np.ones(2),
                    "mean": np.ones(3),
                },
                "modal.spod": {
                    "frequency_hz": frequency, "coordinates_m": x,
                    "eigenvalues": np.ones((2, 2)),
                    "modes": np.ones((2, 2, 3), dtype=complex),
                },
                "modal.dmd": {
                    "frequency_hz": frequency, "coordinates_m": x,
                    "modes": np.ones((3, 2), dtype=complex),
                    "eigenvalues": np.ones(2, dtype=complex),
                    "growth_rate_per_s": np.ones(2), "amplitudes": np.ones(2, dtype=complex),
                    "retained_condition_number": np.array(1.0),
                },
            }
            for product_id, arrays in products.items():
                left_path = root / f"{product_id.replace('.', '-')}-left.npz"
                right_path = root / f"{product_id.replace('.', '-')}-right.npz"
                np.savez(left_path, **arrays)
                np.savez(right_path, **arrays)
                left = _Product(
                    "left", _product_metadata(left_path, product_id), left_path,
                )
                right = _Product(
                    "right", _product_metadata(right_path, product_id), right_path,
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
        envelope = np.exp(-((time - 0.0010) / 0.00012) ** 2)
        values = envelope[:, None] * np.cos(
            2.0 * np.pi * frequency * time[:, None] - wavenumber * x[None, :]
        )
        analysis = __import__(
            "pelecpost.config.models", fromlist=["DirectionalWaveAnalysis"]
        ).DirectionalWaveAnalysis(
            id="wave", recipe="directional_wave", probe_set_id="default",
            variable="pressure", frequency_min_hz=15_000.0,
            frequency_max_hz=35_000.0,
            expected_speed_min_m_s=100.0, expected_speed_max_m_s=2_000.0,
            temporal_wavenumber={
                "enabled": True, "window_duration_s": 256.0e-6,
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
                '"interpretation": "baseline"}\n', encoding="utf-8",
            )
            right.write_text(
                '{"segment_count": 2, "frequency_resolution_hz": 4.0, '
                '"interpretation": "different label"}\n', encoding="utf-8",
            )
            metrics = _json_metrics(
                left, right,
                ("segment_count", "frequency_resolution_hz"),
            )
            self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
