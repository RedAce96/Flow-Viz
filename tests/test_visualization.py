from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

from pelecpost.config.models import (
    ContourRange,
    ContourStyle,
    ContourStyleOverride,
    PresentationConfig,
    TimeAnnotationPresentation,
    SurfaceNormalStation,
)
from pelecpost.analysis.plotfiles import _resolve_normal_station
from pelecpost.geometry import flat_plate_surface, wedge_surfaces
from pelecpost.visualization import (
    artists_overlap,
    contour_layout,
    render_contour,
    resolve_contour_style,
)


class VisualizationTests(unittest.TestCase):
    @staticmethod
    def dataset():
        x = np.linspace(0.0, 1.0, 24)
        y = np.linspace(0.0, 0.2, 12)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return {"x": x, "y": y, "fields": {"temperature": 300.0 + xx + yy}}

    def test_every_external_time_and_colorbar_position_is_disjoint(self):
        for time_position in (
            "top_left",
            "top_center",
            "top_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ):
            for colorbar_position in ("top", "bottom", "left", "right"):
                with self.subTest(time=time_position, colorbar=colorbar_position):
                    config = PresentationConfig(
                        time_annotation=TimeAnnotationPresentation(
                            position=time_position
                        )
                    )
                    style = ContourStyle(colorbar={"position": colorbar_position})
                    figure, main, colorbar, time_axis, time_artist, _ = render_contour(
                        self.dataset(), "temperature", config, style, (300.0, 302.0),
                        time_text=r"$t=5.325$ ms",
                    )
                    self.assertIsNotNone(time_axis)
                    self.assertIsNotNone(time_artist)
                    figure.canvas.draw()
                    self.assertFalse(artists_overlap(figure, time_artist, colorbar))
                    self.assertFalse(artists_overlap(figure, time_artist, main))
                    # Top-left/right timestamps intentionally share the
                    # header row with the top colorbar.  Their invisible
                    # layout axes may touch the decorated colorbar extent;
                    # the visible annotation remains independently checked.
                    if not (
                        colorbar_position == "top"
                        and time_position in {"top_left", "top_right"}
                    ):
                        self.assertFalse(artists_overlap(figure, time_axis, colorbar))
                    if not (
                        colorbar_position == "top"
                        and time_position in {"top_left", "top_right"}
                    ):
                        self.assertFalse(artists_overlap(figure, time_axis, main))
                    self.assertFalse(artists_overlap(figure, colorbar, main))
                    plt.close(figure)

    def test_default_top_layout_orders_time_colorbar_and_contour(self):
        config = PresentationConfig()
        figure, main, colorbar, time_axis, _ = contour_layout(
            config,
            "top",
            0.43,
            r"$t=5.325$ ms",
        )
        figure.canvas.draw()
        assert time_axis is not None
        self.assertGreaterEqual(time_axis.get_position().y0, main.get_position().y1)
        self.assertGreaterEqual(colorbar.get_position().y0, main.get_position().y1)
        self.assertLessEqual(
            abs(time_axis.get_position().y0 - colorbar.get_position().y0),
            0.04,
        )
        self.assertGreater(
            colorbar.get_position().width, colorbar.get_position().height
        )
        plt.close(figure)

    def test_colorbar_length_fraction_controls_horizontal_extent(self):
        figure, main, colorbar, _, _ = contour_layout(
            PresentationConfig(), "top", 0.33, r"$t=5.325$ ms"
        )
        figure.canvas.draw()
        self.assertAlmostEqual(
            colorbar.get_position().width / main.get_position().width,
            0.33,
            delta=0.02,
        )
        plt.close(figure)

    def test_colorbar_length_fraction_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal to 0.2"):
            ContourStyle(colorbar={"length_fraction": 0.19})

    def test_contour_override_merges_nested_colorbar_values(self):
        default = ContourStyle(
            colorbar={"position": "right", "length_fraction": 0.43, "label": "auto"}
        )
        resolved = resolve_contour_style(
            default,
            ContourStyleOverride(
                colorbar={"length_fraction": 0.33, "label": "Temperature"}
            ),
        )
        self.assertEqual(resolved.colorbar.position, "right")
        self.assertEqual(resolved.colorbar.length_fraction, 0.33)
        self.assertEqual(resolved.colorbar.label, "Temperature")

    def test_fixed_log_contour_uses_requested_range(self):
        config = PresentationConfig()
        style = ContourStyle(
            normalization="log",
            range=ContourRange(mode="fixed", minimum=100.0, maximum=1000.0),
        )
        figure, _, _, _, _, limits = render_contour(
            self.dataset(),
            "temperature",
            config,
            style,
            (100.0, 1000.0),
            time_text=r"$t=5.325$ ms",
        )
        self.assertEqual(limits, (100.0, 1000.0))
        plt.close(figure)

    def test_log_contour_rejects_nonpositive_range(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            render_contour(
                self.dataset(),
                "temperature",
                PresentationConfig(),
                ContourStyle(normalization="log"),
                (-1.0, 1.0),
            )

    def test_surface_station_resolves_x_and_arc_length(self):
        plate = flat_plate_surface(0.0, 1.0, points=11)
        station = SurfaceNormalStation(
            id="x-quarter",
            location={"type": "x", "value_m": 0.25},
        )
        _, point, _, normal, arc = _resolve_normal_station((plate,), station)
        np.testing.assert_allclose(point, [0.25, 0.0])
        np.testing.assert_allclose(normal, [0.0, 1.0])
        self.assertAlmostEqual(arc, 0.25)
        arc_station = SurfaceNormalStation(
            id="arc-quarter",
            component_id=0,
            location={"type": "arc_length", "value_m": 0.25},
        )
        _, arc_point, _, _, _ = _resolve_normal_station((plate,), arc_station)
        np.testing.assert_allclose(arc_point, [0.25, 0.0])

    def test_ambiguous_wedge_x_station_is_rejected(self):
        surfaces = wedge_surfaces((0.0, 0.0), 1.0, 10.0)
        station = SurfaceNormalStation(
            id="ambiguous",
            location={"type": "x", "value_m": 0.5},
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _resolve_normal_station(surfaces, station)


if __name__ == "__main__":
    unittest.main()
