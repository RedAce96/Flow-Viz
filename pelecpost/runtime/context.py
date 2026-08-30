"""Execution context passed to independent workflow executors."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable

from pelecpost.config.models import AnalysisConfig, ResolvedProject
from pelecpost.preflight import PreflightPlan

from .artifacts import Artifact, ArtifactRegistry


@dataclass
class WorkflowContext:
    project: ResolvedProject
    plan: PreflightPlan
    run_dir: Path
    analysis: AnalysisConfig
    artifacts: ArtifactRegistry
    _resources: ExitStack | None = None
    resource_metadata: dict[str, Any] | None = None

    def __enter__(self) -> "WorkflowContext":
        self._resources = ExitStack()
        self.resource_metadata = {}
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._resources is not None:
            self._resources.__exit__(exc_type, exc_value, traceback)
            self._resources = None

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        if self._resources is None:
            raise RuntimeError("WorkflowContext must be entered before acquiring resources")
        self._resources.callback(callback)

    @property
    def data_dir(self) -> Path:
        path = self.run_dir / "data" / self.analysis.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figure_dir(self) -> Path:
        path = self.run_dir / "figures" / self.analysis.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register(
        self,
        *,
        artifact_id: str,
        path: Path,
        kind: str,
        variable: str | None,
        units: str | None,
        interpretation: str,
        coordinate_metadata: dict | None = None,
        provenance: dict | None = None,
    ) -> Artifact:
        from pelecpost.workflows import workflow_for

        required_inputs = workflow_for(self.analysis.recipe).required_inputs_for(
            self.analysis
        )
        sources: list[str] = []
        if "plotfiles" in required_inputs and self.plan.inventory.plotfiles is not None:
            sources.append(self.plan.inventory.plotfiles.source)
        if "probes" in required_inputs and self.plan.inventory.probes is not None:
            sources.append(self.plan.inventory.probes.source)
        if "comparison_archives" in required_inputs:
            sources.extend(self.plan.inventory.comparison_archives.values())
        baseline_id = (
            getattr(self.analysis, "baseline_id", None)
            if self.analysis.recipe == "aerodynamic_forces" else None
        )
        if baseline_id:
            baseline = self.project.machine_file.inputs.baselines[baseline_id]
            baseline_source = (
                baseline.source if baseline.source.is_absolute()
                else (self.project.root / baseline.source).resolve()
            )
            sources.append(str(baseline_source))
        provenance_payload = dict(provenance or {})
        provenance_payload.setdefault(
            "preprocessing",
            self.analysis.model_dump(
                mode="json", exclude={"id", "enabled"}, exclude_none=True
            ),
        )
        return self.artifacts.register(Artifact(
            id=f"{self.analysis.id}.{artifact_id}",
            schema_version=1,
            recipe_instance=self.analysis.id,
            kind=kind,
            path=str(path.relative_to(self.run_dir)),
            variable=variable,
            units=units,
            coordinate_metadata=coordinate_metadata or {},
            source_inputs=tuple(dict.fromkeys(sources)),
            interpretation=interpretation,
            provenance=provenance_payload,
        ))
