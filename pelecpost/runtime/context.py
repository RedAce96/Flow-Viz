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
        return self.artifacts.register(Artifact(
            id=f"{self.analysis.id}.{artifact_id}",
            schema_version=1,
            recipe_instance=self.analysis.id,
            kind=kind,
            path=str(path.relative_to(self.run_dir)),
            variable=variable,
            units=units,
            coordinate_metadata=coordinate_metadata or {},
            source_inputs=tuple(
                value for value in (
                    self.plan.inventory.plotfiles.source if self.plan.inventory.plotfiles else None,
                    self.plan.inventory.probes.source if self.plan.inventory.probes else None,
                ) if value
            ),
            interpretation=interpretation,
            provenance=provenance or {},
        ))
