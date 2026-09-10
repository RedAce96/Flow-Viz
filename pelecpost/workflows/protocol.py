"""Typed workflow contract joining metadata, validation, resources, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pelecpost.config.models import AnalysisConfig, ResolvedProject
    from pelecpost.io import InputInventory
    from pelecpost.runtime.context import WorkflowContext

    from .registry import RecipeDefinition


@dataclass(frozen=True)
class WorkflowValidation:
    level: str
    code: str
    message: str


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Public contract implemented by every executable recipe workflow."""

    @property
    def metadata(self) -> "RecipeDefinition":
        ...

    @property
    def artifact_declarations(self) -> tuple[str, ...]:
        ...

    def artifact_declarations_for(self, analysis: "AnalysisConfig") -> tuple[str, ...]:
        ...

    def required_inputs_for(self, analysis: "AnalysisConfig") -> tuple[str, ...]:
        ...

    def validate(
        self,
        project: "ResolvedProject",
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> tuple[WorkflowValidation, ...]:
        ...

    def estimate_resources(
        self,
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> dict[str, float]:
        ...

    def execute(self, context: "WorkflowContext") -> None:
        ...


def _input_available(name: str, inventory: "InputInventory") -> bool:
    available = {
        "plotfiles": inventory.plotfiles is not None and inventory.plotfiles.count > 0,
        "probe_sets": any(item.sample_count > 0 for item in inventory.probe_sets.values()),
        "archived_runs": bool(inventory.archived_runs),
    }
    return available[name]


def _probe_sample_count(probes: Any, analysis: "AnalysisConfig") -> int:
    """Estimate the selected/resampled record size for one probe recipe."""
    sample_count = int(probes.sample_count)
    start = probes.time_min_s
    stop = probes.time_max_s
    configured_start = getattr(analysis, "record_start_time_s", None)
    configured_stop = getattr(analysis, "end_time_s", None)
    if configured_start is not None and start is not None:
        start = max(start, float(configured_start))
    if configured_stop is not None and stop is not None:
        stop = min(stop, float(configured_stop))
    if start is not None and stop is not None and probes.median_timestep_s:
        duration = max(0.0, stop - start)
        sample_count = min(
            sample_count, max(0, floor(duration / probes.median_timestep_s) + 1)
        )
    return sample_count


@dataclass(frozen=True)
class RegisteredWorkflow:
    """Lazy executable workflow used by planning and runtime dispatch."""

    metadata: "RecipeDefinition"

    @property
    def artifact_declarations(self) -> tuple[str, ...]:
        return self.metadata.outputs

    def required_inputs_for(self, analysis: "AnalysisConfig") -> tuple[str, ...]:
        inputs = list(self.metadata.required_inputs)
        if self.metadata.name == "aerodynamic_forces" and getattr(analysis, "probe_linkage", None):
            inputs.append("probe_sets")
        if self.metadata.name == "case_comparison":
            for reference_name in ("baseline", "comparison"):
                reference = getattr(analysis, reference_name)
                if reference.archived_run_id is not None:
                    inputs.append("archived_runs")
        return tuple(dict.fromkeys(inputs))

    def artifact_declarations_for(self, analysis: "AnalysisConfig") -> tuple[str, ...]:
        outputs = list(self.artifact_declarations)
        probe_plotting = getattr(analysis, "probe_plotting", None)
        if probe_plotting is not None:
            overlay_outputs = {
                "probe.raw_history.figure", "probe.method_ready.figure",
                "spectral.fft_overlay.figure", "spectral.psd_overlay.figure",
                "pulse.transfer_magnitude.figure", "pulse.transfer_phase.figure",
                "transient.filtered_overlay.figure", "transient.envelope_overlay.figure",
                "transient.stft_figure",
            }
            if probe_plotting.mode == "panels":
                outputs = [
                    item for item in outputs
                    if item not in overlay_outputs and not item.startswith("probe.")
                ]
            elif probe_plotting.mode == "overlay":
                outputs = [item for item in outputs if item != "spectral.probe_figure"]
        if self.metadata.name == "directional_wave":
            temporal = getattr(analysis, "temporal_wavenumber", None)
            if temporal is None or not getattr(temporal, "enabled", False):
                temporal_outputs = {
                    "wave.temporal_wavenumber", "wave.komega_snapshots",
                    "wave.space_time.figure", "wave.temporal_wavenumber.figure",
                    "wave.wavenumber_history.figure", "wave.komega_snapshots.figure",
                    "wave.dispersion.figure",
                }
                outputs = [item for item in outputs if item not in temporal_outputs]
        if self.metadata.name == "flow_overview":
            if not getattr(analysis, "line_stations_x_m", ()):
                outputs.remove("field.lines")
            if not getattr(analysis, "streamlines", False):
                outputs.remove("field.streamlines")
        if self.metadata.name == "surface_diagnostics":
            if getattr(analysis, "normal_profiles", None) is None:
                outputs.remove("surface.normal_profile")
        if self.metadata.name == "aerodynamic_forces":
            if getattr(analysis, "control_volume", None) is None:
                outputs.remove("forces.control_volume")
            if getattr(analysis, "probe_linkage", None) is None:
                outputs.remove("forces.probe_linkage")
        return tuple(outputs)

    def validate(
        self,
        project: "ResolvedProject",
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> tuple[WorkflowValidation, ...]:
        findings: list[WorkflowValidation] = []
        dimensionality = project.case_file.case.dimensionality
        if dimensionality not in self.metadata.supported_dimensions:
            findings.append(WorkflowValidation(
                "BLOCKER",
                "UNSUPPORTED_DIMENSION",
                f"{analysis.recipe} supports dimensions {self.metadata.supported_dimensions}; "
                "3-D extension interfaces are documented but no 3-D algorithm is implemented.",
            ))
        geometry = project.case_file.geometry.type
        if geometry not in self.metadata.supported_geometries:
            findings.append(WorkflowValidation(
                "BLOCKER", "UNSUPPORTED_GEOMETRY",
                f"{analysis.recipe} does not support geometry {geometry!r}.",
            ))
        required_inputs = self.required_inputs_for(analysis)
        for name in required_inputs:
            if not _input_available(name, inventory):
                findings.append(WorkflowValidation(
                    "BLOCKER", "MISSING_INPUT",
                    f"Recipe {analysis.recipe} requires configured {name} input.",
                ))
        if (
            "plotfiles" in required_inputs
            and inventory.plotfiles is not None
            and inventory.plotfiles.dimensionality is not None
            and inventory.plotfiles.dimensionality != dimensionality
        ):
            findings.append(WorkflowValidation(
                "BLOCKER", "PLOTFILE_DIMENSION_MISMATCH",
                f"case.yaml declares {dimensionality}-D but inspected plotfiles are "
                f"{inventory.plotfiles.dimensionality}-D; correct the case identity or input path.",
            ))
        if self.metadata.required_fields and inventory.plotfiles is not None:
            for field in self.metadata.required_fields:
                if field not in inventory.plotfiles.canonical_fields:
                    findings.append(WorkflowValidation(
                        "BLOCKER", "MISSING_PLOTFILE_FIELD",
                        f"Required canonical field {field!r} was not mapped from the plotfile.",
                    ))
        if self.metadata.name == "flow_overview" and inventory.plotfiles is not None:
            available_fields = set(inventory.plotfiles.canonical_fields)
            derived_requirements = {
                "mach_number": {"density", "pressure", "x_velocity", "y_velocity"},
                "schlieren": {"density"},
                "vorticity": {"x_velocity", "y_velocity"},
                "vorticity_magnitude": {"x_velocity", "y_velocity"},
            }
            for requested in getattr(analysis, "fields", ()):
                field = requested.value
                required = derived_requirements.get(field, {field})
                missing = required - available_fields
                if missing:
                    findings.append(WorkflowValidation(
                        "BLOCKER", "MISSING_REQUESTED_FIELD",
                        f"Requested flow field {field!r} requires mapped canonical field(s): "
                        f"{', '.join(sorted(missing))}.",
                    ))
        if self.metadata.name == "surface_diagnostics" and inventory.plotfiles is not None:
            normal_profiles = getattr(analysis, "normal_profiles", None)
            if normal_profiles is not None:
                available_fields = set(inventory.plotfiles.canonical_fields)
                derived_requirements = {
                    "mach_number": {"density", "pressure", "x_velocity", "y_velocity"},
                    "schlieren": {"density"},
                    "vorticity": {"x_velocity", "y_velocity"},
                    "vorticity_magnitude": {"x_velocity", "y_velocity"},
                }
                for requested in normal_profiles.fields:
                    field = requested.value
                    missing = derived_requirements.get(field, {field}) - available_fields
                    if missing:
                        findings.append(WorkflowValidation(
                            "BLOCKER", "MISSING_NORMAL_PROFILE_FIELD",
                            f"Requested surface-normal field {field!r} requires mapped "
                            f"canonical field(s): {', '.join(sorted(missing))}.",
                        ))
        return tuple(findings)

    def estimate_resources(
        self,
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> dict[str, float]:
        probes = None
        if self.metadata.name in {
            "probe_spectrum", "single_pulse_response", "directional_wave",
            "transient_wavepacket", "nonlinear_coupling", "modal_screening",
        }:
            probes = inventory.probe_sets.get(getattr(analysis, "probe_set_id", ""))
        if self.metadata.name == "aerodynamic_forces":
            linkage = getattr(analysis, "probe_linkage", None)
            if linkage is not None:
                probes = inventory.probe_sets.get(linkage.probe_set_id)
        probe_recipe = self.metadata.name in {
            "probe_spectrum", "single_pulse_response", "directional_wave",
            "transient_wavepacket", "nonlinear_coupling", "modal_screening",
        }
        force_linkage = (
            self.metadata.name == "aerodynamic_forces"
            and getattr(analysis, "probe_linkage", None) is not None
        )
        if (probe_recipe or force_linkage) and probes is not None:
            selection_source = (
                getattr(analysis, "probe_linkage") if force_linkage else analysis
            )
            selected = getattr(selection_source, "probe_indices", ())
            probe_count = len(selected) if selected else probes.probe_count
            sample_count = _probe_sample_count(probes, analysis)
            matrix = sample_count * probe_count * 8
            if force_linkage:
                return {
                    "field_slab": 1.5,
                    "plotting_workspace": 0.5,
                    "probe_signal_or_disk_workspace": matrix / 1024**3,
                    "force_probe_spectral_workspace": 3.0 * matrix / 1024**3,
                }
            multiplier = {
                "probe_spectrum": 3.0,
                "single_pulse_response": 4.0,
                "directional_wave": 6.0,
                "transient_wavepacket": 5.0,
                "nonlinear_coupling": 8.0,
                "modal_screening": 7.0,
            }[self.metadata.name]
            components = {"probe_signal_or_disk_workspace": matrix / 1024**3}
            if self.metadata.name in {
                "probe_spectrum", "single_pulse_response", "nonlinear_coupling",
            }:
                components["fft_workspace"] = matrix * (multiplier - 1.0) / 1024**3
                if self.metadata.name == "probe_spectrum":
                    components["plotting_workspace"] = 0.1
            elif self.metadata.name == "directional_wave":
                components["wavenumber_and_komega"] = matrix * (multiplier - 1.0) / 1024**3
                temporal = getattr(analysis, "temporal_wavenumber", None)
                if getattr(temporal, "enabled", False) and probes.median_timestep_s:
                    dt = float(probes.median_timestep_s)
                    n_window = max(8, int(round(temporal.window_duration_s / dt)))
                    hop = max(1, int(round(n_window * (1.0 - temporal.overlap_fraction))))
                    window_count = max(1, 1 + (max(sample_count - n_window, 0) // hop))
                    frequency_count = max(
                        1, int(np.ceil((analysis.frequency_max_hz - analysis.frequency_min_hz) * n_window * dt))
                    )
                    wavenumber_count = max(1, probe_count)
                    window_bytes = n_window * wavenumber_count * 16 + frequency_count * wavenumber_count * 16
                    summary_bytes = window_count * wavenumber_count * 8
                    snapshot_count = len(temporal.snapshot_times_s) or 4
                    snapshot_bytes = snapshot_count * frequency_count * wavenumber_count * 8
                    components["temporal_wavenumber_workspace"] = (window_bytes + snapshot_bytes) / 1024**3
                    components["temporal_wavenumber_disk"] = (summary_bytes + snapshot_bytes) / 1024**3
            elif self.metadata.name == "transient_wavepacket":
                components["stft_and_envelope"] = matrix * (multiplier - 1.0) / 1024**3
            else:
                components["modal_workspace"] = matrix * (multiplier - 1.0) / 1024**3
            return components
        if self.metadata.name == "case_comparison":
            product_count = max(1, len(getattr(analysis, "product_ids", ())))
            archived_bytes = 0
            for reference_name in ("baseline", "comparison"):
                reference = getattr(analysis, reference_name)
                if reference.archived_run_id is None:
                    continue
                metadata = inventory.archived_metadata.get(reference.archived_run_id, {})
                for product_id in getattr(analysis, "product_ids", ()):
                    item = metadata.get(f"{reference.analysis_id}.{product_id}", {})
                    path = item.get("path")
                    run_dir = inventory.archived_runs.get(reference.archived_run_id)
                    if path and run_dir:
                        try:
                            archived_bytes += (Path(run_dir) / path).stat().st_size
                        except OSError:
                            pass
            # Account for both products and keep a floor for interpolation
            # metadata even when a local product has no file-size inventory yet.
            return {
                "comparison_product_workspace": max(
                    0.25 * product_count, 2.0 * archived_bytes / 1024**3
                ),
            }
        # Plotfile readers materialize one selected region/snapshot at a time.
        return {"field_slab": 1.5, "plotting_workspace": 0.5}

    def execute(self, context: "WorkflowContext") -> None:
        # Importing here keeps configuration-only commands independent of yt,
        # Matplotlib, and the numerical executor modules.
        from pelecpost.analysis.executors import execute

        execute(context)


def assert_workflow_contract(workflow: WorkflowProtocol) -> None:
    """Fail early when registry construction omits a required public contract."""
    if not workflow.artifact_declarations:
        raise ValueError(f"workflow {workflow.metadata.name!r} declares no artifacts")
    if not workflow.metadata.assumptions:
        raise ValueError(f"workflow {workflow.metadata.name!r} declares no assumptions")
    if not workflow.metadata.limitations:
        raise ValueError(f"workflow {workflow.metadata.name!r} declares no limitations")
