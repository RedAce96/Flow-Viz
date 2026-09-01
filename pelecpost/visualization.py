"""Scoped, configuration-driven rendering for recipe figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import pp_plotting_database as plotting_api
from pelecpost.config.models import (
    ContourStyle,
    ContourStyleOverride,
    FigurePresentation,
    LineStyle,
    LineStyleOverride,
    PresentationConfig,
    PresentationOverride,
    TimeAnnotationPresentation,
)


_LINESTYLES = {
    "solid": "-",
    "dashed": "--",
    "dashdot": "-.",
    "dotted": ":",
}
_MARKERS = {
    "none": None,
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
}
_LEGENDS = {
    "best": "best",
    "upper_left": "upper left",
    "upper_right": "upper right",
    "lower_left": "lower left",
    "lower_right": "lower right",
}


def _deep_update(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_contour_style(
    default: ContourStyle,
    override: ContourStyleOverride | None,
) -> ContourStyle:
    if override is None:
        return default
    values = override.model_dump(exclude_none=True)
    return ContourStyle.model_validate(_deep_update(default.model_dump(), values))


def resolve_line_style(
    default: LineStyle, override: LineStyleOverride | None
) -> LineStyle:
    if override is None:
        return default
    return LineStyle.model_validate(
        _deep_update(default.model_dump(), override.model_dump(exclude_none=True))
    )


def resolve_presentation(
    default: PresentationConfig,
    override: PresentationOverride | None,
) -> PresentationConfig:
    if override is None:
        return default
    return PresentationConfig.model_validate(
        _deep_update(default.model_dump(), override.model_dump(exclude_none=True))
    )


@contextmanager
def presentation_context(config: PresentationConfig) -> Iterator[None]:
    typography = config.typography
    with plt.rc_context(
        {
            "font.family": typography.font_family,
            "font.size": typography.base_size,
            "axes.labelsize": typography.axes_label_size,
            "axes.titlesize": typography.axes_label_size,
            "xtick.labelsize": typography.tick_label_size,
            "ytick.labelsize": typography.tick_label_size,
            "legend.fontsize": typography.legend_size,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    ):
        yield


def _time_axis(figure, grid_cell, config: TimeAnnotationPresentation, text: str):
    axis = figure.add_subplot(grid_cell)
    axis.set_axis_off()
    horizontal = config.position.rsplit("_", 1)[1]
    x = {"left": 0.0, "center": 0.5, "right": 1.0}[horizontal]
    alignment = {"left": "left", "center": "center", "right": "right"}[horizontal]
    bbox = None
    if config.boxed:
        bbox = {
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.7,
            "alpha": 0.9,
        }
    artist = axis.text(
        x,
        0.5,
        text,
        transform=axis.transAxes,
        ha=alignment,
        va="center",
        bbox=bbox,
    )
    return axis, artist


def _horizontal_colorbar_axis(
    figure,
    grid_cell,
    *,
    length_fraction: float,
    shrink_width: bool = True,
):
    """Create a compact horizontal colorbar without changing its typography.

    ``length_fraction`` controls the long dimension of the bar without
    changing tick or label font sizes.  Its axes are also slightly shorter.
    """
    if shrink_width:
        margin = (1.0 - length_fraction) / 2.0
        subgrid = grid_cell.subgridspec(
            3,
            3,
            width_ratios=(margin, length_fraction, margin),
            height_ratios=(0.17, 0.66, 0.17),
        )
        return figure.add_subplot(subgrid[1, 1])
    subgrid = grid_cell.subgridspec(3, 1, height_ratios=(0.17, 0.66, 0.17))
    return figure.add_subplot(subgrid[1, 0])


def _vertical_colorbar_axis(figure, grid_cell, *, length_fraction: float):
    margin = (1.0 - length_fraction) / 2.0
    subgrid = grid_cell.subgridspec(
        3, 1, height_ratios=(margin, length_fraction, margin)
    )
    return figure.add_subplot(subgrid[1, 0])


def contour_layout(
    presentation: PresentationConfig,
    colorbar_position: str,
    colorbar_length_fraction: float,
    time_text: str,
):
    """Create non-overlapping axes for time metadata, colorbar, and contour."""
    figure_config = presentation.figure
    figure = plt.figure(
        figsize=(figure_config.width_in, figure_config.height_in),
    )
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.12, top=0.96, hspace=0.50, wspace=0.5
    )
    time_config = presentation.time_annotation
    show_time = bool(time_config.enabled and time_text)
    time_top = time_config.position.startswith("top")
    horizontal = colorbar_position in {"top", "bottom"}

    time_axis = time_artist = None
    # A top-left or top-right timestamp can share the colorbar header.  This
    # keeps both immediately above the data axes while reserving independent
    # cells so the annotation can never cover the colorbar or its label.
    inline_header = (
        horizontal
        and colorbar_position == "top"
        and show_time
        and time_config.position in {"top_left", "top_right"}
    )

    if inline_header:
        # 0.30 is the smallest verified separation that clears the horizontal
        # colorbar label and the top contour ticks; it keeps the header close
        # to a shallow, equal-aspect contour.
        grid = figure.add_gridspec(2, 1, height_ratios=(0.24, 1.0), hspace=0.30)
        header = grid[0, 0].subgridspec(
            1,
            3,
            width_ratios=(0.18, colorbar_length_fraction, 0.82 - colorbar_length_fraction)
            if time_config.position == "top_left"
            else (0.82 - colorbar_length_fraction, colorbar_length_fraction, 0.18),
            wspace=0.04,
        )
        if time_config.position == "top_left":
            time_cell, colorbar_cell = header[0, 0], header[0, 1]
        else:
            colorbar_cell, time_cell = header[0, 1], header[0, 2]
        time_axis, time_artist = _time_axis(
            figure, time_cell, time_config, time_text
        )
        contour_axis = figure.add_subplot(grid[1, 0])
        colorbar_axis = _horizontal_colorbar_axis(
            figure,
            colorbar_cell,
            length_fraction=colorbar_length_fraction,
            shrink_width=False,
        )
    elif horizontal:
        rows: list[str] = []
        if show_time and time_top:
            rows.append("time")
        if colorbar_position == "top":
            rows.append("colorbar")
        rows.append("main")
        if colorbar_position == "bottom":
            rows.append("colorbar")
        if show_time and not time_top:
            rows.append("time")
        ratios = [
            0.16 if item == "time" else 0.24 if item == "colorbar" else 1.0
            for item in rows
        ]
        grid = figure.add_gridspec(len(rows), 1, height_ratios=ratios, hspace=1.0)
        axes = {}
        for index, item in enumerate(rows):
            if item == "main":
                axes["main"] = figure.add_subplot(grid[index, 0])
            elif item == "colorbar":
                axes["colorbar"] = _horizontal_colorbar_axis(
                    figure,
                    grid[index, 0],
                    length_fraction=colorbar_length_fraction,
                )
            else:
                time_axis, time_artist = _time_axis(
                    figure, grid[index, 0], time_config, time_text
                )
        contour_axis, colorbar_axis = axes["main"], axes["colorbar"]
    else:
        row_names = ["main"]
        if show_time:
            row_names = ["time", "main"] if time_top else ["main", "time"]
        row_ratios = [0.13 if item == "time" else 1.0 for item in row_names]
        outer = figure.add_gridspec(
            len(row_names), 1, height_ratios=row_ratios, hspace=1.0
        )
        main_cell = None
        for index, item in enumerate(row_names):
            if item == "time":
                time_axis, time_artist = _time_axis(
                    figure, outer[index, 0], time_config, time_text
                )
            else:
                main_cell = outer[index, 0]
        assert main_cell is not None
        columns = (
            ["colorbar", "main"]
            if colorbar_position == "left"
            else ["main", "colorbar"]
        )
        inner = main_cell.subgridspec(
            1,
            2,
            width_ratios=(0.08, 1.0) if colorbar_position == "left" else (1.0, 0.08),
        )
        contour_axis = figure.add_subplot(inner[0, columns.index("main")])
        colorbar_axis = _vertical_colorbar_axis(
            figure,
            inner[0, columns.index("colorbar")],
            length_fraction=colorbar_length_fraction,
        )
    return figure, contour_axis, colorbar_axis, time_axis, time_artist


def _normalization(style: ContourStyle, minimum: float, maximum: float):
    if style.normalization == "log":
        if minimum <= 0.0:
            raise ValueError(
                "log contour normalization requires a strictly positive range"
            )
        return mcolors.LogNorm(vmin=minimum, vmax=maximum)
    if style.normalization == "symlog":
        threshold = style.symlog_linear_threshold
        if threshold is None:
            threshold = max(abs(minimum), abs(maximum)) / 1000.0
        return mcolors.SymLogNorm(linthresh=threshold, vmin=minimum, vmax=maximum)
    return mcolors.Normalize(vmin=minimum, vmax=maximum)


def render_contour(
    dataset: dict,
    field: str,
    presentation: PresentationConfig,
    style: ContourStyle,
    limits: tuple[float, float],
    *,
    x_limits_m=None,
    y_limits_m=None,
    time_text: str = "",
):
    figure, axis, colorbar_axis, time_axis, time_artist = contour_layout(
        presentation,
        style.colorbar.position,
        style.colorbar.length_fraction,
        time_text,
    )
    values = np.asarray(dataset["fields"][field], dtype=float)
    minimum, maximum = map(float, limits)
    if style.symmetric_about_zero:
        bound = max(abs(minimum), abs(maximum))
        minimum, maximum = -bound, bound
    if style.normalization == "log" and np.any(values[np.isfinite(values)] <= 0.0):
        raise ValueError(
            f"log contour normalization requires strictly positive {field} data"
        )
    normalization = _normalization(style, minimum, maximum)
    cmap = plotting_api.resolve_cmap(style.colormap)
    levels = style.rendering.levels
    if style.rendering.mode == "discrete":
        boundaries = (
            np.linspace(minimum, maximum, int(levels) + 1)
            if isinstance(levels, int)
            else np.asarray(levels, dtype=float)
        )
        normalization = mcolors.BoundaryNorm(boundaries, cmap.N)
    artist = axis.pcolormesh(
        dataset["x"],
        dataset["y"],
        values.T,
        shading="auto",
        cmap=cmap,
        norm=normalization,
        rasterized=True,
    )
    orientation = (
        "horizontal" if style.colorbar.position in {"top", "bottom"} else "vertical"
    )
    colorbar = figure.colorbar(artist, cax=colorbar_axis, orientation=orientation)
    label = (
        plotting_api.field_label(field)
        if style.colorbar.label == "auto"
        else style.colorbar.label
    )
    colorbar.set_label(label)
    if style.colorbar.tick_format != "auto":
        from matplotlib.ticker import FormatStrFormatter

        colorbar.ax.xaxis.set_major_formatter(
            FormatStrFormatter(style.colorbar.tick_format)
        )
        colorbar.ax.yaxis.set_major_formatter(
            FormatStrFormatter(style.colorbar.tick_format)
        )
    axis.set_xlabel(r"$x$ [m]")
    axis.set_ylabel(r"$y$ [m]")
    if x_limits_m is not None:
        axis.set_xlim(x_limits_m)
    if y_limits_m is not None:
        axis.set_ylim(y_limits_m)
    axis.set_aspect("equal", adjustable="box")
    axis.set_anchor("N")
    return figure, axis, colorbar_axis, time_axis, time_artist, (minimum, maximum)


def line_figure(presentation: PresentationConfig, time_text: str):
    config = presentation.figure
    figure = plt.figure(figsize=(config.width_in, max(config.height_in, 5.0)))
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.12, top=0.95, hspace=0.4)
    time = presentation.time_annotation
    show_time = bool(time.enabled and time_text)
    if not show_time:
        return figure, figure.add_subplot(111), None, None
    top = time.position.startswith("top")
    rows = ["time", "main"] if top else ["main", "time"]
    grid = figure.add_gridspec(2, 1, height_ratios=(0.13, 1.0) if top else (1.0, 0.13))
    main_axis = None
    time_axis = time_artist = None
    for index, item in enumerate(rows):
        if item == "main":
            main_axis = figure.add_subplot(grid[index, 0])
        else:
            time_axis, time_artist = _time_axis(figure, grid[index, 0], time, time_text)
    return figure, main_axis, time_axis, time_artist


def style_line_axis(axis, style: LineStyle) -> None:
    axis.set_xscale(style.coordinate_scale)
    axis.set_yscale(style.value_scale)
    axis.grid(style.grid, alpha=0.25)


def plot_profile(axis, coordinate, values, label: str, style: LineStyle) -> None:
    axis.plot(
        coordinate,
        values,
        label=label,
        color=style.color,
        linewidth=style.linewidth,
        linestyle=_LINESTYLES[style.linestyle],
        marker=_MARKERS[style.marker],
    )


def legend_location(style: LineStyle) -> str:
    return _LEGENDS[style.legend_position]


def save_figure_variants(
    figure,
    stem: Path,
    config: FigurePresentation,
) -> tuple[Path, ...]:
    paths = []
    for output_format in config.formats:
        path = stem.with_suffix(f".{output_format}")
        figure.savefig(
            path,
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches="tight",
        )
        paths.append(path)
    plt.close(figure)
    return tuple(paths)


def artists_overlap(figure, first, second) -> bool:
    """Return whether two rendered artist/axes extents overlap."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    first_box = first.get_tightbbox(renderer)
    second_box = second.get_tightbbox(renderer)
    return bool(first_box.overlaps(second_box))
