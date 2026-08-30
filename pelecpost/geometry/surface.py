"""Validated ordered two-dimensional surface curves and providers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np


def _validate_intersections(points: np.ndarray, closed: bool) -> None:
    count = len(points) if closed else len(points) - 1
    starts = points[:count]
    ends = np.roll(points, -1, axis=0)[:count] if closed else points[1:]
    for first in range(count):
        candidates = np.arange(first + 2, count)
        if closed and first == 0:
            candidates = candidates[candidates != count - 1]
        if not candidates.size:
            continue
        a, b = starts[first], ends[first]
        c, d = starts[candidates], ends[candidates]
        bounding = (
            (np.maximum(a[0], b[0]) >= np.minimum(c[:, 0], d[:, 0]))
            & (np.minimum(a[0], b[0]) <= np.maximum(c[:, 0], d[:, 0]))
            & (np.maximum(a[1], b[1]) >= np.minimum(c[:, 1], d[:, 1]))
            & (np.minimum(a[1], b[1]) <= np.maximum(c[:, 1], d[:, 1]))
        )
        if not np.any(bounding):
            continue
        candidate_ids = candidates[bounding]
        c, d = starts[candidate_ids], ends[candidate_ids]
        ab = b - a
        cd = d - c
        first_orientation = ab[0] * (c[:, 1] - a[1]) - ab[1] * (c[:, 0] - a[0])
        second_orientation = ab[0] * (d[:, 1] - a[1]) - ab[1] * (d[:, 0] - a[0])
        third_orientation = cd[:, 0] * (a[1] - c[:, 1]) - cd[:, 1] * (a[0] - c[:, 0])
        fourth_orientation = cd[:, 0] * (b[1] - c[:, 1]) - cd[:, 1] * (b[0] - c[:, 0])
        intersects = (first_orientation * second_orientation < 0) & (third_orientation * fourth_orientation < 0)
        if np.any(intersects):
            second = int(candidate_ids[np.flatnonzero(intersects)[0]])
            raise ValueError(f"surface self-intersection between segments {first} and {second}")


@dataclass(frozen=True)
class SurfaceCurve2D:
    coordinates_m: np.ndarray
    arc_length_m: np.ndarray
    tangent: np.ndarray
    fluid_normal: np.ndarray
    segment_length_m: np.ndarray
    segment_tangent: np.ndarray
    segment_fluid_normal: np.ndarray
    closed: bool
    component_id: str
    side_id: str
    source: str
    confidence: float
    fluid_side: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    unsmoothed_coordinates_m: np.ndarray | None = None

    @classmethod
    def from_points(
        cls,
        points_m: np.ndarray,
        *,
        closed: bool,
        fluid_side: Literal["left", "right", "outside", "inside", "above", "below"],
        component_id: str = "component-0",
        side_id: str = "surface",
        source: str = "polyline",
        confidence: float = 1.0,
        minimum_segment_m: float = 1.0e-14,
    ) -> "SurfaceCurve2D":
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("surface coordinates must have shape (n, 2)")
        if closed and len(points) > 1 and np.allclose(points[0], points[-1], rtol=0, atol=minimum_segment_m):
            points = points[:-1]
        minimum = 3 if closed else 2
        if len(points) < minimum or not np.all(np.isfinite(points)):
            raise ValueError(f"surface requires at least {minimum} finite points")
        delta = np.diff(points, axis=0)
        if closed:
            delta = np.vstack((delta, points[0] - points[-1]))
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths <= minimum_segment_m):
            raise ValueError("surface contains duplicate points or zero-length segments")
        _validate_intersections(points, closed)
        segment_tangent = delta / lengths[:, None]
        left = np.column_stack((-segment_tangent[:, 1], segment_tangent[:, 0]))
        area = 0.5 * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        ) if closed else 0.0
        if closed and fluid_side in {"outside", "inside"}:
            outside_sign = -1.0 if area > 0 else 1.0
            sign = outside_sign if fluid_side == "outside" else -outside_sign
        elif fluid_side == "left":
            sign = 1.0
        elif fluid_side == "right":
            sign = -1.0
        elif fluid_side in {"above", "below"}:
            desired = 1.0 if fluid_side == "above" else -1.0
            sign = desired if float(np.mean(left[:, 1])) >= 0 else -desired
        else:
            raise ValueError("open curves require left/right/above/below fluid-side declaration")
        segment_normal = sign * left
        if closed:
            node_tangent = segment_tangent + np.roll(segment_tangent, 1, axis=0)
            node_normal = segment_normal + np.roll(segment_normal, 1, axis=0)
        else:
            node_tangent = np.vstack((segment_tangent[0], segment_tangent[:-1] + segment_tangent[1:], segment_tangent[-1]))
            node_normal = np.vstack((segment_normal[0], segment_normal[:-1] + segment_normal[1:], segment_normal[-1]))
        node_tangent /= np.linalg.norm(node_tangent, axis=1)[:, None]
        node_normal /= np.linalg.norm(node_normal, axis=1)[:, None]
        arc = np.concatenate(([0.0], np.cumsum(lengths[:-1] if closed else lengths)))
        return cls(
            coordinates_m=points, arc_length_m=arc, tangent=node_tangent,
            fluid_normal=node_normal, segment_length_m=lengths,
            segment_tangent=segment_tangent, segment_fluid_normal=segment_normal,
            closed=closed, component_id=component_id, side_id=side_id,
            source=source, confidence=float(confidence), fluid_side=fluid_side,
            diagnostics={
                "point_count": len(points), "segment_count": len(lengths),
                "length_m": float(np.sum(lengths)), "signed_area_m2": float(area),
                "minimum_segment_m": float(np.min(lengths)),
                "maximum_segment_m": float(np.max(lengths)),
            },
        )

    def reversed(self) -> "SurfaceCurve2D":
        return SurfaceCurve2D.from_points(
            self.coordinates_m[::-1], closed=self.closed, fluid_side=self.fluid_side,
            component_id=self.component_id, side_id=self.side_id, source=self.source,
            confidence=self.confidence,
        )


def flat_plate_surface(
    leading_edge_x_m: float,
    trailing_edge_x_m: float,
    wall_y_m: float = 0.0,
    fluid_side: Literal["above", "below"] = "above",
    points: int = 257,
) -> SurfaceCurve2D:
    if trailing_edge_x_m <= leading_edge_x_m or points < 2:
        raise ValueError("flat plate needs increasing endpoints and at least two points")
    x = np.linspace(leading_edge_x_m, trailing_edge_x_m, points)
    return SurfaceCurve2D.from_points(
        np.column_stack((x, np.full_like(x, wall_y_m))), closed=False,
        fluid_side=fluid_side, component_id="flat-plate", side_id=fluid_side,
        source="flat_plate", confidence=1.0,
    )


def polyline_surface(
    points_m: np.ndarray,
    *,
    closed: bool,
    fluid_side: Literal["left", "right", "outside", "inside"],
    component_id: str = "polyline-0",
) -> SurfaceCurve2D:
    return SurfaceCurve2D.from_points(
        points_m, closed=closed, fluid_side=fluid_side,
        component_id=component_id, source="polyline", confidence=1.0,
    )


def wedge_surfaces(
    leading_edge_m: tuple[float, float], length_m: float, half_angle_deg: float,
) -> tuple[SurfaceCurve2D, SurfaceCurve2D]:
    if length_m <= 0 or not 0 < half_angle_deg < 90:
        raise ValueError("wedge length and half-angle must be physical")
    leading = np.asarray(leading_edge_m, dtype=float)
    angle = np.deg2rad(half_angle_deg)
    upper_end = leading + length_m * np.array([np.cos(angle), np.sin(angle)])
    lower_end = leading + length_m * np.array([np.cos(angle), -np.sin(angle)])
    upper = SurfaceCurve2D.from_points(
        np.vstack((leading, upper_end)), closed=False, fluid_side="left",
        component_id="wedge", side_id="upper", source="wedge", confidence=1.0,
    )
    lower = SurfaceCurve2D.from_points(
        np.vstack((leading, lower_end)), closed=False, fluid_side="right",
        component_id="wedge", side_id="lower", source="wedge", confidence=1.0,
    )
    return upper, lower


def volume_fraction_surfaces(
    x_m: np.ndarray,
    y_m: np.ndarray,
    fluid_fraction: np.ndarray,
    *,
    iso_value: float = 0.5,
    fluid_value: int = 1,
    minimum_component_points: int = 8,
    smoothing_window: int = 1,
) -> tuple[SurfaceCurve2D, ...]:
    """Extract deterministic serial isocontours without creating a figure."""
    from contourpy import contour_generator

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    values = np.asarray(fluid_fraction, dtype=float)
    if values.shape == (len(x), len(y)):
        # StandardDataset fields use indexing='ij': (x, y). contourpy and the
        # normal-direction sampler below use image order: (y, x).
        values_yx = values.T
    elif values.shape == (len(y), len(x)):
        values_yx = values
    else:
        raise ValueError(
            "volume-fraction array shape must be (len(x), len(y)) or (len(y), len(x))"
        )
    generator = contour_generator(x=x, y=y, z=values_yx, name="serial", corner_mask=False)
    lines = generator.lines(float(iso_value))
    curves: list[SurfaceCurve2D] = []
    from scipy.interpolate import RegularGridInterpolator

    sample_fraction = RegularGridInterpolator(
        (y, x), values_yx, method="linear", bounds_error=False, fill_value=np.nan
    )
    grid_scale = min(float(np.min(np.diff(x))), float(np.min(np.diff(y))))
    for index, line in enumerate(lines):
        if len(line) < minimum_component_points:
            continue
        closed = bool(np.allclose(line[0], line[-1]))
        if not closed:
            raise ValueError("open volume-fraction contour has unresolved fluid normal")
        unsmoothed = np.asarray(line[:-1] if np.allclose(line[0], line[-1]) else line, dtype=float)
        smoothed = unsmoothed.copy()
        if smoothing_window > 1:
            if smoothing_window % 2 == 0 or smoothing_window >= len(unsmoothed):
                raise ValueError("smoothing_window must be odd and smaller than the component")
            radius = smoothing_window // 2
            smoothed = np.mean(
                np.stack([np.roll(unsmoothed, offset, axis=0) for offset in range(-radius, radius + 1)]),
                axis=0,
            )
        outside = SurfaceCurve2D.from_points(
            smoothed, closed=True, fluid_side="outside",
            component_id=f"eb-{index}", source="volume_fraction", confidence=0.9,
        )
        offset = 0.35 * grid_scale
        plus_xy = outside.coordinates_m + offset * outside.fluid_normal
        minus_xy = outside.coordinates_m - offset * outside.fluid_normal
        plus = sample_fraction(np.column_stack((plus_xy[:, 1], plus_xy[:, 0])))
        minus = sample_fraction(np.column_stack((minus_xy[:, 1], minus_xy[:, 0])))
        plus_error = float(np.nanmedian(np.abs(plus - fluid_value)))
        minus_error = float(np.nanmedian(np.abs(minus - fluid_value)))
        if not np.isfinite(plus_error) or not np.isfinite(minus_error) or abs(plus_error - minus_error) < 0.05:
            raise ValueError(f"unresolved fluid normal direction for component {index}")
        curve = outside if plus_error < minus_error else SurfaceCurve2D.from_points(
            smoothed, closed=True, fluid_side="inside",
            component_id=f"eb-{index}", source="volume_fraction", confidence=0.9,
        )
        ratio = float(np.max(curve.segment_length_m) / grid_scale)
        diagnostics = {
            **curve.diagnostics,
            "fluid_side_plus_error": plus_error,
            "fluid_side_minus_error": minus_error,
            "maximum_segment_to_grid_ratio": ratio,
            "resolution_warning": ratio > 2.0,
            "smoothing_window": smoothing_window,
            "maximum_smoothing_displacement_m": float(
                np.max(np.linalg.norm(smoothed - unsmoothed, axis=1))
            ),
        }
        curve = replace(
            curve,
            confidence=0.6 if ratio > 2.0 else curve.confidence,
            diagnostics=diagnostics,
            unsmoothed_coordinates_m=unsmoothed,
        )
        curves.append(curve)
    if not curves:
        raise ValueError("no sufficiently resolved volume-fraction surface component found")
    return tuple(curves)
