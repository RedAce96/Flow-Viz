"""Typed workflow contract joining metadata, validation, resources, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    # comparison_archives remains the required-input key for backward compat,
    # but it is satisfied by either run-dir archives or direct probe sets.
    has_comparison = bool(inventory.comparison_archives) or bool(
        getattr(inventory, "comparison_probe_sets", {})
    )
    available = {
        "plotfiles": inventory.plotfiles is not None and inventory.plotfiles.count > 0,
        "probes": inventory.probes is not None and inventory.probes.sample_count > 0,
        "comparison_archives": has_comparison,
    }
    return available[name]


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
            inputs.append("probes")
        return tuple(dict.fromkeys(inputs))

    def artifact_declarations_for(self, analysis: "AnalysisConfig") -> tuple[str, ...]:
        outputs = list(self.artifact_declarations)
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
        probes = inventory.probes
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
            sample_count = probes.sample_count
            if (
                self.metadata.name == "probe_spectrum"
                and getattr(analysis, "time_grid_policy", "resample_uniform") == "resample_uniform"
                and probes.median_timestep_s
                and probes.time_min_s is not None
                and probes.time_max_s is not None
            ):
                resampled_count = int(
                    (probes.time_max_s - probes.time_min_s) / probes.median_timestep_s
                ) + 1
                sample_count = max(sample_count, resampled_count)
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
            elif self.metadata.name == "transient_wavepacket":
                components["stft_and_envelope"] = matrix * (multiplier - 1.0) / 1024**3
            else:
                components["modal_workspace"] = matrix * (multiplier - 1.0) / 1024**3
            return components
        if self.metadata.name == "case_comparison":
            return {"comparison_arrays": 0.25}
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
