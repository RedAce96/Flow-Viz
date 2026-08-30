from __future__ import annotations

import unittest

import numpy as np
import pp_functions_database as legacy_forces

from pelecpost.analysis.fields import add_normalized_fields, resolve_freestream_reference
from pelecpost.analysis.surface_forces import fit_wall_quantities, integrate_surface_loads
from pelecpost.config.models import RegionFreestream
from pelecpost.geometry import (
    SurfaceCurve2D,
    flat_plate_surface,
    volume_fraction_surfaces,
    wedge_surfaces,
)


class GeometryForceTests(unittest.TestCase):
    def circle(self, count: int = 2048) -> SurfaceCurve2D:
        angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        points = np.column_stack((np.cos(angle), np.sin(angle)))
        return SurfaceCurve2D.from_points(points, closed=True, fluid_side="outside")

    def loads(self, surface: SurfaceCurve2D, pressure: np.ndarray, shear=None, thermal=None):
        count = len(surface.coordinates_m)
        return integrate_surface_loads(
            surface, pressure,
            np.zeros(count) if shear is None else shear,
            np.zeros(count) if thermal is None else thermal,
            dynamic_viscosity_pa_s=2.0,
            conductivity_w_m_k=4.0,
        )

    def test_uniform_pressure_on_closed_body_has_zero_net_force(self):
        surface = self.circle()
        result = self.loads(surface, np.full(len(surface.coordinates_m), 1250.0))
        np.testing.assert_allclose(result.force_pressure_n_m, 0.0, atol=1.0e-10)
        self.assertAlmostEqual(result.moment_pressure_n, 0.0, places=10)

    def test_linear_pressure_field_recovers_analytical_resultant(self):
        surface = self.circle()
        x, y = surface.coordinates_m.T
        gradient = np.array([3.0, -5.0])
        pressure = 100.0 + gradient[0] * x + gradient[1] * y
        result = self.loads(surface, pressure)
        np.testing.assert_allclose(result.force_pressure_n_m, -np.pi * gradient, rtol=2.0e-6)
        self.assertAlmostEqual(result.moment_pressure_n, 0.0, places=9)

    def test_reversed_closed_contour_has_identical_load(self):
        surface = self.circle(512)
        pressure = 50.0 + 2.0 * surface.coordinates_m[:, 0]
        forward = self.loads(surface, pressure)
        reverse_surface = surface.reversed()
        reverse_pressure = 50.0 + 2.0 * reverse_surface.coordinates_m[:, 0]
        reverse = self.loads(reverse_surface, reverse_pressure)
        np.testing.assert_allclose(forward.force_total_n_m, reverse.force_total_n_m, atol=1.0e-12)
        self.assertAlmostEqual(forward.moment_total_n, reverse.moment_total_n, places=12)

    def test_manufactured_wall_profiles_recover_shear_and_heat_flux(self):
        surface = flat_plate_surface(0.0, 2.0, points=33)
        distance = np.linspace(0.0, 0.01, 8)
        count = len(surface.coordinates_m)
        pressure = 200.0 + 10.0 * distance[None, :] + np.zeros((count, 1))
        velocity = np.zeros((count, len(distance), 2))
        velocity[:, :, 0] = 7.0 * distance
        temperature = 300.0 + 11.0 * distance[None, :] + np.zeros((count, 1))
        fit = fit_wall_quantities(surface, distance, pressure, velocity, temperature)
        np.testing.assert_allclose(fit.pressure_pa, 200.0, atol=1.0e-12)
        np.testing.assert_allclose(fit.tangential_velocity_gradient_s, 7.0, atol=1.0e-12)
        np.testing.assert_allclose(fit.temperature_gradient_k_m, 11.0, atol=1.0e-10)
        result = self.loads(surface, fit.pressure_pa, fit.tangential_velocity_gradient_s, fit.temperature_gradient_k_m)
        np.testing.assert_allclose(result.force_viscous_n_m, [28.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(result.heat_flux_w_m2, -44.0, atol=1.0e-9)

    def test_wedge_pressure_matches_panel_result(self):
        upper, lower = wedge_surfaces((0.0, 0.0), length_m=2.0, half_angle_deg=15.0)
        pressure = 100.0
        upper_load = self.loads(upper, np.full(2, pressure))
        lower_load = self.loads(lower, np.full(2, pressure))
        total = upper_load.force_pressure_n_m + lower_load.force_pressure_n_m
        expected = np.array([2.0 * pressure * 2.0 * np.sin(np.deg2rad(15.0)), 0.0])
        np.testing.assert_allclose(total, expected, atol=1.0e-12)

    def test_wedge_inside_fluid_reverses_both_panel_normals(self):
        outside = wedge_surfaces((0.0, 0.0), 2.0, 15.0, "outside")
        inside = wedge_surfaces((0.0, 0.0), 2.0, 15.0, "inside")
        for outside_panel, inside_panel in zip(outside, inside):
            np.testing.assert_allclose(
                inside_panel.fluid_normal, -outside_panel.fluid_normal, atol=1.0e-15
            )

    def test_flat_plate_is_equivalent_to_reviewed_certified_integrator(self):
        surface = flat_plate_surface(0.0, 1.0, points=101)
        x = surface.coordinates_m[:, 0]
        pressure = 100.0 + 3.0 * x
        shear_pa = 2.0 + 0.5 * x
        origin = (0.25, 0.1)
        current = integrate_surface_loads(
            surface, pressure, shear_pa / 2.0, np.zeros_like(x),
            dynamic_viscosity_pa_s=2.0, conductivity_w_m_k=1.0,
            moment_origin_m=origin, pressure_reference_pa=90.0,
        )
        legacy = legacy_forces.integrate_flat_plate_wall_forces(
            {
                "x_left_m": x[:-1], "x_right_m": x[1:],
                "p_wall_pa": 0.5 * (pressure[:-1] + pressure[1:]),
                "tau_wall_pa": 0.5 * (shear_pa[:-1] + shear_pa[1:]),
                "valid": np.ones(len(x) - 1, dtype=bool),
                "wall_y_m": np.zeros(len(x) - 1),
            },
            {
                "rho_inf": 1.0, "u_inf": 10.0, "p_inf": 90.0,
                "chord": 1.0, "moment_origin": origin,
            },
        )
        np.testing.assert_allclose(
            current.force_total_n_m,
            [legacy["D_total_N_m"], legacy["N_total_N_m"]],
            atol=1.0e-12,
        )
        self.assertAlmostEqual(current.moment_total_n, legacy["M_total_N"], places=12)
        self.assertEqual(current.designation, "stationary_one_sided_flat_plate")

    def test_duplicate_points_and_self_intersections_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SurfaceCurve2D.from_points(
                np.array([[0, 0], [1, 0], [1, 0]]), closed=False, fluid_side="above"
            )
        with self.assertRaisesRegex(ValueError, "self-intersection"):
            SurfaceCurve2D.from_points(
                np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
                closed=True, fluid_side="outside",
            )

    def test_open_curve_requires_resolved_fluid_side(self):
        with self.assertRaisesRegex(ValueError, "open curves"):
            SurfaceCurve2D.from_points(
                np.array([[0, 0], [1, 0]]), closed=False, fluid_side="outside"
            )

    def test_volume_fraction_normals_point_into_fluid(self):
        x = np.linspace(-2.0, 2.0, 201)
        y = np.linspace(-2.0, 2.0, 201)
        xx, yy = np.meshgrid(x, y)
        fluid = (xx**2 + yy**2 >= 1.0).astype(float)
        curves = volume_fraction_surfaces(x, y, fluid, minimum_component_points=20)
        self.assertEqual(len(curves), 1)
        curve = curves[0]
        radial_alignment = np.sum(curve.coordinates_m * curve.fluid_normal, axis=1)
        self.assertGreater(float(np.median(radial_alignment)), 0.9)
        self.assertIsNotNone(curve.unsmoothed_coordinates_m)

    def test_volume_fraction_accepts_standard_dataset_xy_layout_and_records_smoothing(self):
        x = np.linspace(-2.0, 2.0, 161)
        y = np.linspace(-1.5, 1.5, 121)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        fluid_xy = (xx**2 + yy**2 >= 0.8**2).astype(float)
        curve = volume_fraction_surfaces(
            x, y, fluid_xy, minimum_component_points=20, smoothing_window=3,
        )[0]
        self.assertEqual(curve.diagnostics["smoothing_window"], 3)
        self.assertGreaterEqual(curve.diagnostics["maximum_smoothing_displacement_m"], 0.0)
        self.assertEqual(curve.coordinates_m.shape, curve.unsmoothed_coordinates_m.shape)

    def test_coarse_volume_fraction_geometry_reports_resolution_warning(self):
        x = np.linspace(-2.0, 2.0, 21)
        y = np.linspace(-2.0, 2.0, 21)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        fluid = (xx**2 + yy**2 >= 0.8**2).astype(float)
        curve = volume_fraction_surfaces(
            x, y, fluid, minimum_component_points=8,
        )[0]
        self.assertTrue(curve.diagnostics["resolution_warning"])
        self.assertGreater(curve.diagnostics["maximum_turn_angle_deg"], 20.0)
        self.assertLess(curve.confidence, 0.9)

    def test_region_freestream_uses_robust_explicit_region(self):
        x = np.linspace(0.0, 1.0, 5)
        y = np.linspace(0.0, 1.0, 5)
        fields = {
            "density": np.full((5, 5), 2.0),
            "x_velocity": np.full((5, 5), 10.0),
            "pressure": np.full((5, 5), 100.0),
            "temperature": np.full((5, 5), 300.0),
        }
        fields["pressure"][0, 0] = 1.0e9
        dataset = {"x": x, "y": y, "fields": fields}
        config = RegionFreestream.model_validate({
            "source": "region",
            "bounds": {"x_m": [0.0, 0.5], "y_m": [0.0, 0.5]},
            "statistic": "median",
        })
        reference = resolve_freestream_reference(dataset, config)
        self.assertEqual(reference.pressure_pa, 100.0)
        add_normalized_fields(dataset, reference)
        self.assertEqual(dataset["fields"]["P/Pinf"][2, 2], 1.0)
        self.assertEqual(dataset["freestream_reference"]["source"], "region")


if __name__ == "__main__":
    unittest.main()
