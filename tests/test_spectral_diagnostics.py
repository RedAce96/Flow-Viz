"""Regression tests for measured probe diagnostics."""

import unittest

import numpy as np

from pelecpost.analysis.spectral import (
    _finite_record_transform,
    _komega_ridge,
    _pressure_rise_diagnostic,
    _ridge_difference,
    _welch_confidence_metadata,
    probe_phase_and_symmetry_diagnostic,
)
from pp_functions_database import compute_frequency_resolved_wavenumber


class SpectralDiagnosticTests(unittest.TestCase):
    def test_signed_direction_retains_both_speeds_inside_magnitude_bounds(self):
        sample_rate = 1.0e6
        frequency = 125_000.0
        speed = 1_000.0
        time = np.arange(512, dtype=float) / sample_rate
        x_m = np.arange(9, dtype=float) * 0.001
        wavenumber = 2.0 * np.pi * frequency / speed
        for sign in (1, -1):
            signals = np.cos(
                2.0 * np.pi * frequency * time[:, None]
                - sign * wavenumber * x_m[None, :]
            )
            common = {
                "nperseg": 128, "noverlap": 64, "spatial_window_size": 5,
                "phase_speed_bounds": (800.0, 1_200.0),
            }
            for direction in ("positive", "negative", "both"):
                result = compute_frequency_resolved_wavenumber(
                    signals, x_m, sample_rate,
                    (frequency, frequency + 1.0), direction=direction, **common,
                )
                accepted = np.asarray(result["phase_valid_mask"], dtype=bool)
                self.assertEqual(bool(np.any(accepted)), direction == "both" or
                                 direction == ("positive" if sign > 0 else "negative"))
                if np.any(accepted):
                    self.assertEqual(
                        np.sign(np.median(result["phase_speed_m_per_s"][accepted])),
                        sign,
                    )

    def test_phase_and_symmetry_switches_control_output_rows(self):
        time = np.arange(128, dtype=float) / 128.0
        values = np.column_stack((np.sin(2 * np.pi * 8 * time),
                                  np.sin(2 * np.pi * 8 * time)))
        common = {
            "variable": "pressure", "values_are_processed": True,
            "reference_frequency_band_hz": (1.0, 20.0),
        }
        phase_only = probe_phase_and_symmetry_diagnostic(
            values, values, time, np.array([1, 2]), np.array([-1.0, 1.0]),
            ((1, 2),), phase_delay_enabled=True, symmetry_enabled=False, **common,
        )
        symmetry_only = probe_phase_and_symmetry_diagnostic(
            values, values, time, np.array([1, 2]), np.array([-1.0, 1.0]),
            ((1, 2),), phase_delay_enabled=False, symmetry_enabled=True, **common,
        )
        self.assertEqual(phase_only["schema_version"], 3)
        self.assertEqual(len(phase_only["phase_rows"]), 2)
        self.assertEqual(phase_only["symmetry"], [])
        self.assertEqual(symmetry_only["phase_rows"], [])
        self.assertEqual(len(symmetry_only["symmetry"]), 1)
        nonuniform_time = time.copy()
        nonuniform_time[1:] += np.linspace(0.0, 0.001, len(time) - 1)
        symmetry_nonuniform = probe_phase_and_symmetry_diagnostic(
            values, values, nonuniform_time, np.array([1, 2]),
            np.array([-1.0, 1.0]), ((1, 2),),
            phase_delay_enabled=False, symmetry_enabled=True, **common,
        )
        self.assertEqual(len(symmetry_nonuniform["symmetry"]), 1)
        self.assertEqual(symmetry_nonuniform["frequency_hz"], [])

    def test_pressure_rise_uses_original_index_when_nonfinite_samples_are_skipped(self):
        time = np.arange(6, dtype=float) * 1.0e-9
        values = np.array(
            [[1.0], [1.1], [np.nan], [2.0], [3.0], [2.5]],
            dtype=float,
        )
        result = _pressure_rise_diagnostic(time, values)[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["peak_time_s"], time[4])
        self.assertEqual(result["crossing_10_index"], 3)
        self.assertEqual(result["crossing_90_index"], 4)

    def test_pressure_rise_rejects_missing_sample_inside_crossing_interval(self):
        time = np.arange(6, dtype=float) * 1.0e-9
        values = np.array([[0.0], [0.2], [np.nan], [0.7], [0.9], [1.0]])
        result = _pressure_rise_diagnostic(time, values)[0]
        self.assertEqual(result["status"], "crossing_interval_missing_samples")

    def test_finite_record_transform_preserves_integral_convention(self):
        values = np.zeros((8, 1)); values[1, 0] = 2.0
        frequency, transform = _finite_record_transform(values, 0.5, np.ones(8))
        self.assertEqual(len(frequency), 5)
        self.assertTrue(np.allclose(np.abs(transform[:, 0]), 1.0))

    def test_branch_ridge_difference_does_not_join_opposing_branches(self):
        frequency = np.array([1.0, 2.0])
        k = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        power = np.array([[1.0, 0.9, 0.0, 0.9, 0.99], [1.0, 0.9, 0.0, 0.9, 0.99]])
        reference = _komega_ridge({"frequency_hz": frequency, "wavenumber_rad_per_m": k, "power": power}, 1.0, 2.0)
        candidate = _komega_ridge({"frequency_hz": frequency, "wavenumber_rad_per_m": k, "power": power[:, ::-1]}, 1.0, 2.0)
        result = _ridge_difference(reference, candidate)
        self.assertIn("negative", result["branches"])
        self.assertIn("positive", result["branches"])

    def test_komega_ridge_names_zero_wavenumber_power(self):
        result = _komega_ridge(
            {
                "frequency_hz": np.array([1.0]),
                "wavenumber_rad_per_m": np.array([-1.0, 0.0, 1.0]),
                "power": np.array([[1.0, 3.0, 2.0]]),
            },
            1.0,
            1.0,
        )
        self.assertNotIn("zero_k", result)
        np.testing.assert_allclose(result["zero_k_power"], [3.0])

    def test_welch_effective_dof_accounts_for_overlap(self):
        independent = _welch_confidence_metadata(
            sample_count=1024,
            segment_samples=256,
            overlap_samples=0,
            window="hann",
            dt=1.0,
        )
        overlapped = _welch_confidence_metadata(
            sample_count=1024,
            segment_samples=256,
            overlap_samples=128,
            window="hann",
            dt=1.0,
        )
        self.assertEqual(
            independent["effective_degrees_of_freedom"],
            2 * independent["segment_count"],
        )
        self.assertLess(
            overlapped["effective_degrees_of_freedom"],
            2 * overlapped["segment_count"],
        )

    def test_phase_and_symmetry_reports_supported_signed_residuals(self):
        time = np.arange(32, dtype=float) * 1.0e-3
        x = np.array([0.024, 0.026])
        values = np.sin(2.0 * np.pi * 10.0 * time)[:, None] * np.ones((1, 2))
        result = probe_phase_and_symmetry_diagnostic(
            values, values, time, np.array([120, 200]), x, ((120, 200),), variable="pressure"
        )
        self.assertEqual(result["symmetry"][0]["baseline_odd_fraction"], 0.0)

    def test_odd_parity_reports_even_component_as_residual(self):
        time = np.arange(32, dtype=float) * 1.0e-3
        x = np.array([0.024, 0.026])
        left = np.sin(2.0 * np.pi * 10.0 * time)
        values = np.column_stack((left, -left))
        result = probe_phase_and_symmetry_diagnostic(
            values,
            values,
            time,
            np.array([120, 200]),
            x,
            ((120, 200),),
            variable="pressure",
            scalar_reflection_parity="odd",
        )
        row = result["symmetry"][0]
        self.assertEqual(row["expected_parity"], "odd")
        self.assertEqual(row["baseline_parity_residual_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
