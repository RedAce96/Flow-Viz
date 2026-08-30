"""Stable artifact records written atomically at artifact creation time."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Artifact:
    id: str
    schema_version: int
    recipe_instance: str
    kind: str
    path: str
    variable: str | None
    units: str | None
    coordinate_metadata: dict[str, Any]
    source_inputs: tuple[str, ...]
    interpretation: str
    provenance: dict[str, Any] = field(default_factory=dict)


class ArtifactRegistry:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.path = self.run_dir / "artifacts.json"
        self._artifacts: list[Artifact] = []
        self._ids: set[str] = set()
        self._paths: set[str] = set()
        self.flush()

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return tuple(self._artifacts)

    def register(self, artifact: Artifact) -> Artifact:
        target = (self.run_dir / artifact.path).resolve()
        try:
            relative = target.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError(f"artifact must remain inside run directory: {target}") from exc
        if not target.is_file():
            raise FileNotFoundError(f"artifact was not created: {target}")
        normalized = str(relative)
        if artifact.id in self._ids:
            raise ValueError(f"duplicate artifact id: {artifact.id}")
        if normalized in self._paths:
            raise ValueError(f"artifact path already registered: {normalized}")
        artifact = Artifact(**{**asdict(artifact), "path": normalized})
        self._artifacts.append(artifact)
        self._ids.add(artifact.id)
        self._paths.add(normalized)
        self.flush()
        return artifact

    def flush(self) -> None:
        atomic_json(self.path, {
            "schema": "pelecpost.artifacts",
            "schema_version": 1,
            "artifacts": [asdict(item) for item in self._artifacts],
        })

    @classmethod
    def load(cls, run_dir: Path) -> "ArtifactRegistry":
        run_dir = run_dir.resolve()
        payload = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
        instance = cls.__new__(cls)
        instance.run_dir = run_dir
        instance.path = run_dir / "artifacts.json"
        instance._artifacts = [Artifact(**item) for item in payload["artifacts"]]
        instance._ids = {item.id for item in instance._artifacts}
        instance._paths = {item.path for item in instance._artifacts}
        return instance
