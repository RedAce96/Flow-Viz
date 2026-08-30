"""Typed workflow contract joining metadata, validation, resources, and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    def metadata(self) -> "RecipeDefinition": ...

    @property
    def artifact_declarations(self) -> tuple[str, ...]: ...

    def validate(
        self,
        project: "ResolvedProject",
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> tuple[WorkflowValidation, ...]: ...

    def estimate_resources(
        self,
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> dict[str, float]: ...

    def execute(self, context: "WorkflowContext") -> None: ...


def _input_available(name: str, inventory: "InputInventory") -> bool:
    available = {
        "plotfiles": inventory.plotfiles is not None and inventory.plotfiles.count > 0,
        "probes": inventory.probes is not None and inventory.probes.sample_count > 0,
        "comparison_archives": bool(inventory.comparison_archives),
    }
    return available[name]


@dataclass(frozen=True)
class RegisteredWorkflow:
    """Lazy executable workflow used by planning and runtime dispatch."""

    metadata: "RecipeDefinition"

    @property
    def artifact_declarations(self) -> tuple[str, ...]:
        return self.metadata.outputs

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
        for name in self.metadata.required_inputs:
            if not _input_available(name, inventory):
                findings.append(WorkflowValidation(
                    "BLOCKER", "MISSING_INPUT",
                    f"Recipe {analysis.recipe} requires configured {name} input.",
                ))
        if self.metadata.required_fields and inventory.plotfiles is not None:
            for field in self.metadata.required_fields:
                if field not in inventory.plotfiles.canonical_fields:
                    findings.append(WorkflowValidation(
                        "BLOCKER", "MISSING_PLOTFILE_FIELD",
                        f"Required canonical field {field!r} was not mapped from the plotfile.",
                    ))
        return tuple(findings)

    def estimate_resources(
        self,
        analysis: "AnalysisConfig",
        inventory: "InputInventory",
    ) -> dict[str, float]:
        probes = inventory.probes
        if self.metadata.name in {
            "probe_spectrum", "single_pulse_response", "directional_wave",
            "transient_wavepacket", "nonlinear_coupling", "modal_screening",
        } and probes is not None:
            selected = getattr(analysis, "probe_indices", ())
            probe_count = len(selected) if selected else probes.probe_count
            matrix = probes.sample_count * probe_count * 8
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
