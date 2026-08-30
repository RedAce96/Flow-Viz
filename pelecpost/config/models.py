"""Strict Pydantic models for the clean-break YAML project contract."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


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
        return self


class VolumeFractionGeometry(StrictModel):
    type: Literal["volume_fraction"] = "volume_fraction"
    field: str = "volume_fraction"
    fluid_value: Literal[0, 1] = 1
    iso_value: float = Field(default=0.5, gt=0.0, lt=1.0)
    minimum_component_points: PositiveInt = 8


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


class FlowOverviewAnalysis(BaseAnalysis):
    recipe: Literal["flow_overview"]
    fields: tuple[Variable, ...] = (Variable.TEMPERATURE, Variable.PRESSURE)
    snapshot_start: int | None = None
    snapshot_end: int | None = None
    snapshot_step: PositiveInt = 1
    x_limits_m: tuple[float, float] | None = None
    y_limits_m: tuple[float, float] | None = None
    line_stations_x_m: tuple[float, ...] = ()
    streamlines: bool = False


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


class ProbeSpectrumAnalysis(BaseAnalysis):
    recipe: Literal["probe_spectrum"]
    variable: Variable
    frequency_max_hz: PositiveFloat | None = None
    window: Literal["hann", "hamming", "blackman", "rectangular"] = "hann"
    detrend: Literal["mean", "linear", "none"] = "mean"
    welch_segment_samples: PositiveInt | None = None
    overlap_fraction: float = Field(default=0.5, ge=0.0, lt=1.0)
    probe_indices: tuple[int, ...] = ()


class SinglePulseAnalysis(BaseAnalysis):
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


class DirectionalWaveAnalysis(BaseAnalysis):
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


class TransientWavepacketAnalysis(BaseAnalysis):
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


class NonlinearCouplingAnalysis(BaseAnalysis):
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


class ModalScreeningAnalysis(BaseAnalysis):
    recipe: Literal["modal_screening"]
    variable: Variable
    mode_count: PositiveInt = 4
    probe_stride: PositiveInt = 1
    spod_segment_samples: PositiveInt = 2048
    dmd_ranks: tuple[PositiveInt, ...] = (2, 4, 8)
    sensitivity_windows: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.0))


class CaseComparisonAnalysis(BaseAnalysis):
    recipe: Literal["case_comparison"]
    baseline_run: Path
    comparison_run: Path
    artifact_ids: tuple[str, ...]


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
    comparison_archives: tuple[Path, ...] = ()


class OutputConfig(StrictModel):
    root: Path


class ComputeConfig(StrictModel):
    workers: PositiveInt = 1
    memory_limit_gb: PositiveFloat = 8.0
    fft_batch_size: PositiveInt = 32


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
