"""Strict Pydantic models for the clean-break YAML project contract."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt,
    field_validator, model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SolverUnits(StrEnum):
    CGS = "cgs"
    SI = "si"


class Variable(StrEnum):
    DENSITY = "density"
    X_VELOCITY = "x_velocity"
    Y_VELOCITY = "y_velocity"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    VORTICITY = "vorticity"
    VORTICITY_MAGNITUDE = "vorticity_magnitude"
    MACH_NUMBER = "mach_number"
    SCHLIEREN = "schlieren"


FigureFormat = Literal["png", "pdf", "svg"]
AxisScale = Literal["linear", "log", "symlog"]


class FigurePresentation(StrictModel):
    formats: tuple[FigureFormat, ...] = ("png",)
    dpi: PositiveInt = 300
    width_in: PositiveFloat = 14.0
    height_in: PositiveFloat = 4.5
    transparent: bool = False

    @model_validator(mode="after")
    def unique_formats(self) -> "FigurePresentation":
        if not self.formats:
            raise ValueError("figure formats cannot be empty")
        if len(self.formats) != len(set(self.formats)):
            raise ValueError("figure formats must be unique")
        return self


class TypographyPresentation(StrictModel):
    font_family: str = "DejaVu Sans"
    base_size: PositiveFloat = 14.0
    axes_label_size: PositiveFloat = 16.0
    tick_label_size: PositiveFloat = 14.0
    legend_size: PositiveFloat = 13.0
    colorbar_label_size: PositiveFloat = 13.0
    colorbar_tick_label_size: PositiveFloat = 11.0
    colorbar_label_pad: float = Field(default=2.0, ge=0.0, le=24.0)
    colorbar_tick_pad: float = Field(default=2.0, ge=0.0, le=24.0)


class ContourAxesPresentation(StrictModel):
    x_tick_format: str = "auto"
    y_tick_format: str = "auto"


class TimeAnnotationPresentation(StrictModel):
    enabled: bool = True
    position: Literal[
        "top_left", "top_center", "top_right",
        "bottom_left", "bottom_center", "bottom_right",
    ] = "top_left"
    boxed: bool = True
    precision: int = Field(default=4, ge=1, le=12)


class ContourRange(StrictModel):
    mode: Literal[
        "fixed", "per_snapshot_percentile", "selected_snapshots_minmax",
        "selected_snapshots_percentile",
    ] = "per_snapshot_percentile"
    minimum: float | None = None
    maximum: float | None = None
    lower_percentile: float = Field(default=1.0, ge=0.0, le=100.0)
    upper_percentile: float = Field(default=99.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def valid_range(self) -> "ContourRange":
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be below upper_percentile")
        if self.mode == "fixed":
            if self.minimum is None or self.maximum is None:
                raise ValueError("fixed contour ranges require minimum and maximum")
            if self.maximum <= self.minimum:
                raise ValueError("fixed contour maximum must exceed minimum")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("minimum and maximum are only valid for fixed contour ranges")
        return self


class ColorbarPresentation(StrictModel):
    position: Literal["top", "bottom", "left", "right"] = "top"
    length_fraction: float = Field(default=0.43, ge=0.20, le=0.75)
    thickness_fraction: float | None = Field(default=None, ge=0.20, le=0.75)
    include_endpoints: bool = False
    tick_count: int = Field(default=4, ge=2, le=12)
    tick_format: str = "auto"
    label: str = "auto"


class ColorbarOverride(StrictModel):
    position: Literal["top", "bottom", "left", "right"] | None = None
    length_fraction: float | None = Field(default=None, ge=0.20, le=0.75)
    thickness_fraction: float | None = Field(default=None, ge=0.20, le=0.75)
    include_endpoints: bool | None = None
    tick_count: int | None = Field(default=None, ge=2, le=12)
    tick_format: str | None = None
    label: str | None = None


class ContourRendering(StrictModel):
    mode: Literal["continuous", "discrete"] = "continuous"
    levels: PositiveInt | tuple[float, ...] | None = None

    @model_validator(mode="after")
    def valid_levels(self) -> "ContourRendering":
        if self.mode == "discrete" and self.levels is None:
            raise ValueError("discrete contours require levels")
        if isinstance(self.levels, tuple):
            if len(self.levels) < 2 or any(
                second <= first for first, second in zip(self.levels, self.levels[1:])
            ):
                raise ValueError("explicit contour levels must be strictly increasing")
        return self


class ContourRenderingOverride(StrictModel):
    mode: Literal["continuous", "discrete"] | None = None
    levels: PositiveInt | tuple[float, ...] | None = None


class ContourStyle(StrictModel):
    colormap: str = "viridis"
    normalization: AxisScale = "linear"
    range: ContourRange = ContourRange()
    colorbar: ColorbarPresentation = ColorbarPresentation()
    rendering: ContourRendering = ContourRendering()
    symmetric_about_zero: bool = False
    symlog_linear_threshold: PositiveFloat | None = None

    @field_validator("colormap")
    @classmethod
    def known_colormap(cls, value: str) -> str:
        from matplotlib import colormaps
        custom = {"my_reds", "my_blues", "my_reds_r", "my_blues_r", "Blue2Red"}
        if value not in custom and value not in colormaps:
            raise ValueError(f"unknown Matplotlib colormap {value!r}")
        return value

    @model_validator(mode="after")
    def compatible_normalization(self) -> "ContourStyle":
        if self.symlog_linear_threshold is not None and self.normalization != "symlog":
            raise ValueError("symlog_linear_threshold requires symlog normalization")
        if self.symmetric_about_zero and self.normalization == "log":
            raise ValueError("log contours cannot be symmetric about zero")
        return self


class ContourStyleOverride(StrictModel):
    colormap: str | None = None
    normalization: AxisScale | None = None
    range: ContourRange | None = None
    colorbar: ColorbarOverride | None = None
    rendering: ContourRenderingOverride | None = None
    symmetric_about_zero: bool | None = None
    symlog_linear_threshold: PositiveFloat | None = None

    @field_validator("colormap")
    @classmethod
    def known_colormap(cls, value: str | None) -> str | None:
        if value is not None:
            ContourStyle.known_colormap(value)
        return value


class LineStyle(StrictModel):
    linewidth: PositiveFloat = 2.0
    linestyle: Literal["solid", "dashed", "dashdot", "dotted"] = "solid"
    marker: Literal["none", "circle", "square", "triangle", "diamond"] = "none"
    color: str | None = None
    grid: bool = True
    legend_position: Literal[
        "best", "upper_left", "upper_right", "lower_left", "lower_right",
    ] = "best"
    coordinate_scale: AxisScale = "linear"
    value_scale: AxisScale = "linear"


class LineStyleOverride(StrictModel):
    linewidth: PositiveFloat | None = None
    linestyle: Literal["solid", "dashed", "dashdot", "dotted"] | None = None
    marker: Literal["none", "circle", "square", "triangle", "diamond"] | None = None
    color: str | None = None
    grid: bool | None = None
    legend_position: Literal[
        "best", "upper_left", "upper_right", "lower_left", "lower_right",
    ] | None = None
    coordinate_scale: AxisScale | None = None
    value_scale: AxisScale | None = None


class PresentationConfig(StrictModel):
    preset: Literal["publication"] = "publication"
    figure: FigurePresentation = FigurePresentation()
    typography: TypographyPresentation = TypographyPresentation()
    contour_axes: ContourAxesPresentation = ContourAxesPresentation()
    time_annotation: TimeAnnotationPresentation = TimeAnnotationPresentation()
    contour_defaults: ContourStyle = ContourStyle()
    line_defaults: LineStyle = LineStyle()


class FigurePresentationOverride(StrictModel):
    formats: tuple[FigureFormat, ...] | None = None
    dpi: PositiveInt | None = None
    width_in: PositiveFloat | None = None
    height_in: PositiveFloat | None = None
    transparent: bool | None = None


class TypographyPresentationOverride(StrictModel):
    font_family: str | None = None
    base_size: PositiveFloat | None = None
    axes_label_size: PositiveFloat | None = None
    tick_label_size: PositiveFloat | None = None
    legend_size: PositiveFloat | None = None
    colorbar_label_size: PositiveFloat | None = None
    colorbar_tick_label_size: PositiveFloat | None = None
    colorbar_label_pad: float | None = Field(default=None, ge=0.0, le=24.0)
    colorbar_tick_pad: float | None = Field(default=None, ge=0.0, le=24.0)


class ContourAxesPresentationOverride(StrictModel):
    x_tick_format: str | None = None
    y_tick_format: str | None = None


class TimeAnnotationOverride(StrictModel):
    enabled: bool | None = None
    position: Literal[
        "top_left", "top_center", "top_right",
        "bottom_left", "bottom_center", "bottom_right",
    ] | None = None
    boxed: bool | None = None
    precision: int | None = Field(default=None, ge=1, le=12)


class PresentationOverride(StrictModel):
    figure: FigurePresentationOverride | None = None
    typography: TypographyPresentationOverride | None = None
    contour_axes: ContourAxesPresentationOverride | None = None
    time_annotation: TimeAnnotationOverride | None = None
    contour_defaults: ContourStyleOverride | None = None
    line_defaults: LineStyleOverride | None = None


class CaseIdentity(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    solver: Literal["pelec"] = "pelec"
    dimensionality: Literal[2, 3] = 2
    solver_units: SolverUnits = SolverUnits.CGS
    description: str | None = None


class GasConfig(StrictModel):
    gamma: float = Field(default=1.4, gt=1.0)
    gas_constant_j_kg_k: PositiveFloat = 287.05


class ExplicitFreestream(StrictModel):
    source: Literal["explicit"] = "explicit"
    density_kg_m3: PositiveFloat
    velocity_m_s: PositiveFloat
    pressure_pa: PositiveFloat
    temperature_k: PositiveFloat


class RegionBounds(StrictModel):
    x_m: tuple[float, float]
    y_m: tuple[float, float]

    @model_validator(mode="after")
    def increasing(self) -> "RegionBounds":
        if self.x_m[1] <= self.x_m[0] or self.y_m[1] <= self.y_m[0]:
            raise ValueError("freestream region bounds must be strictly increasing")
        return self


class RegionFreestream(StrictModel):
    source: Literal["region"] = "region"
    bounds: RegionBounds
    statistic: Literal["median", "mean"] = "median"


FreestreamConfig = Annotated[
    ExplicitFreestream | RegionFreestream,
    Field(discriminator="source"),
]


class FlatPlateGeometry(StrictModel):
    type: Literal["flat_plate"] = "flat_plate"
    leading_edge_x_m: float = 0.0
    trailing_edge_x_m: PositiveFloat | None = None
    wall_y_m: float = 0.0
    fluid_side: Literal["above", "below"] = "above"

    @model_validator(mode="after")
    def increasing_chord(self) -> "FlatPlateGeometry":
        if (
            self.trailing_edge_x_m is not None
            and self.trailing_edge_x_m <= self.leading_edge_x_m
        ):
            raise ValueError("trailing_edge_x_m must exceed leading_edge_x_m")
        return self


class WedgeGeometry(StrictModel):
    type: Literal["wedge"] = "wedge"
    leading_edge_x_m: float
    leading_edge_y_m: float
    length_m: PositiveFloat
    half_angle_deg: float = Field(gt=0.0, lt=90.0)
    fluid_side: Literal["outside", "inside"] = "outside"


class PolylineGeometry(StrictModel):
    type: Literal["polyline"] = "polyline"
    points_m: tuple[tuple[float, float], ...]
    closed: bool = False
    fluid_side: Literal["left", "right", "outside", "inside"]

    @model_validator(mode="after")
    def enough_points(self) -> "PolylineGeometry":
        minimum = 3 if self.closed else 2
        if len(self.points_m) < minimum:
            raise ValueError(f"polyline geometry requires at least {minimum} points")
        if self.closed and self.fluid_side not in {"outside", "inside"}:
            raise ValueError("closed polylines require outside/inside fluid_side")
        if not self.closed and self.fluid_side not in {"left", "right"}:
            raise ValueError("open polylines require left/right fluid_side")
        return self


class VolumeFractionGeometry(StrictModel):
    type: Literal["volume_fraction"] = "volume_fraction"
    field: str = "volume_fraction"
    fluid_value: Literal[0, 1] = 1
    iso_value: float = Field(default=0.5, gt=0.0, lt=1.0)
    minimum_component_points: PositiveInt = 8
    smoothing_window: PositiveInt = 1

    @model_validator(mode="after")
    def valid_smoothing(self) -> "VolumeFractionGeometry":
        if self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd")
        return self


GeometryConfig = Annotated[
    FlatPlateGeometry | WedgeGeometry | PolylineGeometry | VolumeFractionGeometry,
    Field(discriminator="type"),
]


class CaseFile(StrictModel):
    schema_version: Literal[1] = 1
    case: CaseIdentity
    gas: GasConfig = GasConfig()
    freestream: FreestreamConfig
    geometry: GeometryConfig
    field_aliases: dict[str, str] = Field(default_factory=dict)


class BaseAnalysis(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    enabled: bool = True
    presentation: PresentationOverride | None = None


class ProbeAnalysis(BaseAnalysis):
    probe_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def valid_probe_indices(self) -> "ProbeAnalysis":
        if any(index < 0 for index in self.probe_indices):
            raise ValueError("probe_indices cannot contain negative values")
        if len(self.probe_indices) != len(set(self.probe_indices)):
            raise ValueError("probe_indices must be unique")
        return self


class ContourConfig(StrictModel):
    fields: dict[Variable, ContourStyleOverride] = Field(default_factory=dict)


class LineProfileConfig(StrictModel):
    coordinate_range_m: tuple[float, float] | None = None
    interpolation: Literal["linear", "nearest"] = "linear"
    sample_points: PositiveInt | None = None
    layout: Literal["separate_fields", "combined"] = "separate_fields"
    normalize_values: bool = False
    coordinate_limits: tuple[float, float] | None = None
    value_limits: tuple[float, float] | None = None
    coordinate_scale: AxisScale | None = None
    value_scale: AxisScale | None = None
    grid: bool | None = None
    legend_position: Literal[
        "best", "upper_left", "upper_right", "lower_left", "lower_right",
    ] | None = None
    fields: dict[Variable, LineStyleOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def increasing_limits(self) -> "LineProfileConfig":
        for name in ("coordinate_range_m", "coordinate_limits", "value_limits"):
            limits = getattr(self, name)
            if limits is not None and limits[1] <= limits[0]:
                raise ValueError(f"{name} must be strictly increasing")
        return self


class SurfaceXLocation(StrictModel):
    type: Literal["x"] = "x"
    value_m: float


class SurfaceArcLocation(StrictModel):
    type: Literal["arc_length"] = "arc_length"
    value_m: float = Field(ge=0.0)


SurfaceProfileLocation = Annotated[
    SurfaceXLocation | SurfaceArcLocation, Field(discriminator="type"),
]


class SurfaceNormalStation(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    location: SurfaceProfileLocation
    component_id: str | int | None = None
    side_id: str | None = None
    distance_m: PositiveFloat | None = None
    sample_points: PositiveInt | None = None


class SurfaceProfileFigure(StrictModel):
    layout: Literal["separate_fields", "combined"] = "separate_fields"
    normalize_values: bool = False
    coordinate_scale: AxisScale = "linear"
    value_scale: AxisScale = "linear"
    grid: bool = True


class SurfaceNormalProfiles(StrictModel):
    fields: tuple[Variable, ...] = Field(min_length=1)
    interpolation: Literal["linear", "nearest"] = "linear"
    spacing: Literal["uniform", "wall_clustered"] = "uniform"
    clustering_exponent: PositiveFloat = 2.0
    include_wall_extrapolation: bool = True
    stations: tuple[SurfaceNormalStation, ...] = Field(min_length=1)
    figure: SurfaceProfileFigure = SurfaceProfileFigure()

    @model_validator(mode="after")
    def valid_profiles(self) -> "SurfaceNormalProfiles":
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("surface-normal fields must be unique")
        ids = [station.id for station in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError("surface-normal station ids must be unique")
        if self.spacing == "wall_clustered" and self.clustering_exponent <= 1.0:
            raise ValueError("wall_clustered spacing requires clustering_exponent > 1")
        if self.figure.layout == "combined" and not self.figure.normalize_values:
            units = {
                Variable.DENSITY: "density", Variable.PRESSURE: "pressure",
                Variable.TEMPERATURE: "temperature",
                Variable.X_VELOCITY: "velocity", Variable.Y_VELOCITY: "velocity",
                Variable.MACH_NUMBER: "dimensionless",
                Variable.VORTICITY: "frequency", Variable.VORTICITY_MAGNITUDE: "frequency",
                Variable.SCHLIEREN: "schlieren",
            }
            if len({units[field] for field in self.fields}) > 1:
                raise ValueError(
                    "combined surface-normal profiles require compatible units or "
                    "normalize_values: true"
                )
        return self


class SurfaceGeometryFigure(StrictModel):
    maximum_normal_arrows: PositiveInt = 40
    normal_arrow_length: Literal["sample_distance"] | PositiveFloat = "sample_distance"
    normal_color: str = "tab:orange"
    surface_color: str = "black"


class FlowOverviewAnalysis(BaseAnalysis):
    recipe: Literal["flow_overview"]
    fields: tuple[Variable, ...] = Field(
        default=(Variable.TEMPERATURE, Variable.PRESSURE), min_length=1
    )
    snapshot_start: int | None = None
    snapshot_end: int | None = None
    snapshot_step: PositiveInt = 1
    x_limits_m: tuple[float, float] | None = None
    y_limits_m: tuple[float, float] | None = None
    line_stations_x_m: tuple[float, ...] = ()
    contours: ContourConfig = ContourConfig()
    line_profiles: LineProfileConfig = LineProfileConfig()
    streamlines: bool = False

    @model_validator(mode="after")
    def compatible_line_layout(self) -> "FlowOverviewAnalysis":
        if self.line_profiles.layout == "combined" and not self.line_profiles.normalize_values:
            units = {
                Variable.DENSITY: "density", Variable.PRESSURE: "pressure",
                Variable.TEMPERATURE: "temperature",
                Variable.X_VELOCITY: "velocity", Variable.Y_VELOCITY: "velocity",
                Variable.MACH_NUMBER: "dimensionless",
                Variable.VORTICITY: "frequency", Variable.VORTICITY_MAGNITUDE: "frequency",
                Variable.SCHLIEREN: "schlieren",
            }
            if len({units[field] for field in self.fields}) > 1:
                raise ValueError(
                    "combined line profiles require fields with compatible units or "
                    "normalize_values: true"
                )
        return self


class BoundaryLayerAnalysis(BaseAnalysis):
    recipe: Literal["boundary_layer_reference"]
    stations_x_m: tuple[float, ...]
    maximum_height_m: PositiveFloat
    wall_temperature_k: PositiveFloat
    dynamic_viscosity_pa_s: PositiveFloat
    conductivity_w_m_k: PositiveFloat
    zero_pressure_gradient: Literal[True] = True
    laminar_reference: Literal[True] = True


class SurfaceDiagnosticsAnalysis(BaseAnalysis):
    recipe: Literal["surface_diagnostics"]
    snapshot_start: int | None = None
    snapshot_end: int | None = None
    normal_sample_distance_m: PositiveFloat = 0.001
    normal_sample_points: PositiveInt = 8
    normal_profiles: SurfaceNormalProfiles | None = None
    geometry_figure: SurfaceGeometryFigure = SurfaceGeometryFigure()


class ControlVolumeConfig(StrictModel):
    x_range_m: tuple[float, float]
    y_top_m: PositiveFloat
    bulk_viscosity_pa_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def increasing_x(self) -> "ControlVolumeConfig":
        if self.x_range_m[1] <= self.x_range_m[0]:
            raise ValueError("control-volume x_range_m must be strictly increasing")
        return self


class ForceProbeLinkageConfig(StrictModel):
    variable: Variable = Variable.PRESSURE
    force_component: Literal["x", "y", "moment"] = "y"
    forcing_frequency_hz: PositiveFloat
    minimum_forcing_periods: PositiveFloat = 10.0
    welch_segment_samples: PositiveInt = 16384
    overlap_fraction: float = Field(default=0.5, ge=0.0, lt=1.0)
    minimum_segments: PositiveInt = 8
    probe_indices: tuple[int, ...] = ()

    @model_validator(mode="after")
    def valid_probe_indices(self) -> "ForceProbeLinkageConfig":
        if any(index < 0 for index in self.probe_indices):
            raise ValueError("probe_indices cannot contain negative values")
        if len(self.probe_indices) != len(set(self.probe_indices)):
            raise ValueError("probe_indices must be unique")
        return self


class AerodynamicForcesAnalysis(BaseAnalysis):
    recipe: Literal["aerodynamic_forces"]
    reference_chord_m: PositiveFloat
    reference_span_m: PositiveFloat = 1.0
    moment_origin_m: tuple[float, float] = (0.0, 0.0)
    dynamic_viscosity_pa_s: PositiveFloat
    conductivity_w_m_k: PositiveFloat
    wall_temperature_k: PositiveFloat | None = None
    normal_sample_distance_m: PositiveFloat = 0.001
    normal_sample_points: PositiveInt = 8
    baseline: Literal["none", "static", "paired"] = "none"
    baseline_id: str | None = None
    control_volume: ControlVolumeConfig | None = None
    probe_linkage: ForceProbeLinkageConfig | None = None

    @model_validator(mode="after")
    def baseline_contract(self) -> "AerodynamicForcesAnalysis":
        if self.baseline == "none" and self.baseline_id is not None:
            raise ValueError("baseline_id is only valid for static or paired baselines")
        if self.baseline != "none" and not self.baseline_id:
            raise ValueError("static and paired baselines require baseline_id")
        return self


class ProbeSpectrumAnalysis(ProbeAnalysis):
    recipe: Literal["probe_spectrum"]
    variable: Variable
    frequency_max_hz: PositiveFloat | None = None
    window: Literal["hann", "hamming", "blackman", "rectangular"] = "hann"
    detrend: Literal["mean", "linear", "none"] = "mean"
    welch_segment_samples: PositiveInt | None = None
    overlap_fraction: float = Field(default=0.5, ge=0.0, lt=1.0)


class SinglePulseAnalysis(ProbeAnalysis):
    recipe: Literal["single_pulse_response"]
    variable: Variable
    energy_per_pulse_j_m: PositiveFloat
    pulse_fwhm_s: PositiveFloat
    pulse_period_s: PositiveFloat
    start_time_s: float = 0.0
    cutoff_sigma: PositiveFloat = 4.0
    baseline_end_time_s: float | None = None
    minimum_baseline_samples: PositiveInt = 8
    minimum_relative_source_amplitude: float = Field(default=1.0e-3, gt=0.0, lt=1.0)
    frequency_max_hz: PositiveFloat | None = None


class DirectionalWaveAnalysis(ProbeAnalysis):
    recipe: Literal["directional_wave"]
    variable: Variable
    frequency_min_hz: float = Field(default=0.0, ge=0.0)
    frequency_max_hz: PositiveFloat
    expected_speed_min_m_s: PositiveFloat | None = None
    expected_speed_max_m_s: PositiveFloat | None = None
    minimum_coherence: float = Field(default=0.8, ge=0.0, le=1.0)
    spatial_window: Literal["hann", "hamming", "blackman", "rectangular"] = "hann"
    temporal_window: Literal["hann", "hamming", "blackman", "rectangular"] = "rectangular"

    @model_validator(mode="after")
    def ranges(self) -> "DirectionalWaveAnalysis":
        if self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("frequency_max_hz must exceed frequency_min_hz")
        if (
            self.expected_speed_min_m_s is not None
            and self.expected_speed_max_m_s is not None
            and self.expected_speed_max_m_s <= self.expected_speed_min_m_s
        ):
            raise ValueError("expected speed maximum must exceed minimum")
        return self


class TransientWavepacketAnalysis(ProbeAnalysis):
    recipe: Literal["transient_wavepacket"]
    variable: Variable
    band_min_hz: float = Field(ge=0.0)
    band_max_hz: PositiveFloat
    baseline_end_time_s: float | None = None
    stft_segment_samples: PositiveInt = 2048
    overlap_fraction: float = Field(default=0.75, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def band(self) -> "TransientWavepacketAnalysis":
        if self.band_max_hz <= self.band_min_hz:
            raise ValueError("band_max_hz must exceed band_min_hz")
        return self


class NonlinearCouplingAnalysis(ProbeAnalysis):
    recipe: Literal["nonlinear_coupling"]
    variable: Variable
    segment_samples: PositiveInt = 8192
    overlap_fraction: float = Field(default=0.5, ge=0.0, lt=1.0)
    surrogate_count: int = Field(default=499, ge=19)
    fdr_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    frequency_max_hz: PositiveFloat
    target_frequencies_hz: tuple[PositiveFloat, ...] = ()
    automatic_frequency_selection: bool = False

    @model_validator(mode="after")
    def frequency_selection(self) -> "NonlinearCouplingAnalysis":
        if not self.target_frequencies_hz and not self.automatic_frequency_selection:
            raise ValueError(
                "provide target_frequencies_hz or explicitly enable automatic_frequency_selection"
            )
        return self


class ModalScreeningAnalysis(ProbeAnalysis):
    recipe: Literal["modal_screening"]
    variable: Variable
    mode_count: PositiveInt = 4
    probe_stride: PositiveInt = 1
    spod_segment_samples: PositiveInt = 2048
    dmd_ranks: tuple[PositiveInt, ...] = (2, 4, 8)
    sensitivity_windows: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.0))


class CaseComparisonAnalysis(BaseAnalysis):
    recipe: Literal["case_comparison"]
    baseline_id: str
    comparison_id: str
    artifact_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_runs_and_products(self) -> "CaseComparisonAnalysis":
        if self.baseline_id == self.comparison_id:
            raise ValueError("baseline_id and comparison_id must be different")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("artifact_ids must be unique")
        return self


AnalysisConfig = Annotated[
    FlowOverviewAnalysis
    | BoundaryLayerAnalysis
    | SurfaceDiagnosticsAnalysis
    | AerodynamicForcesAnalysis
    | ProbeSpectrumAnalysis
    | SinglePulseAnalysis
    | DirectionalWaveAnalysis
    | TransientWavepacketAnalysis
    | NonlinearCouplingAnalysis
    | ModalScreeningAnalysis
    | CaseComparisonAnalysis,
    Field(discriminator="recipe"),
]


class AnalysesFile(StrictModel):
    schema_version: Literal[1] = 1
    presentation: PresentationConfig = PresentationConfig()
    analyses: tuple[AnalysisConfig, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> "AnalysesFile":
        ids = [analysis.id for analysis in self.analyses]
        if len(ids) != len(set(ids)):
            raise ValueError("analysis ids must be unique")
        return self


class PlotfileInput(StrictModel):
    source: Path
    prefix: str = "plt"


class ProbeInput(StrictModel):
    compact_file: Path | None = None
    binary_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def one_source(self) -> "ProbeInput":
        if self.compact_file is not None and self.binary_files:
            raise ValueError("choose compact_file or binary_files, not both")
        return self


class InputConfig(StrictModel):
    plotfiles: PlotfileInput | None = None
    probes: ProbeInput | None = None
    comparison_archives: dict[str, Path] = Field(default_factory=dict)
    baselines: dict[str, PlotfileInput] = Field(default_factory=dict)


class OutputConfig(StrictModel):
    root: Path


class ComputeConfig(StrictModel):
    workers: PositiveInt = 1
    memory_limit_gb: PositiveFloat = 8.0
    fft_batch_size: PositiveInt = 32
    scratch_directory: Path | None = None


class MachineFile(StrictModel):
    schema_version: Literal[1] = 1
    inputs: InputConfig = InputConfig()
    outputs: OutputConfig
    compute: ComputeConfig = ComputeConfig()


class ResolvedProject(StrictModel):
    root: Path
    case_file: CaseFile
    analyses_file: AnalysesFile
    machine_file: MachineFile

    @property
    def enabled_analyses(self) -> tuple[AnalysisConfig, ...]:
        return tuple(item for item in self.analyses_file.analyses if item.enabled)
