"""Dimension-aware geometry providers."""

from .surface import (
    SurfaceCurve2D,
    flat_plate_surface,
    polyline_surface,
    volume_fraction_surfaces,
    wedge_surfaces,
)

__all__ = [
    "SurfaceCurve2D",
    "flat_plate_surface",
    "polyline_surface",
    "volume_fraction_surfaces",
    "wedge_surfaces",
]
