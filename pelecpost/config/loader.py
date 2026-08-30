"""Load, validate, and serialize project-directory YAML files."""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from pelecpost.errors import ProjectConfigurationError

from .models import AnalysesFile, CaseFile, MachineFile, ResolvedProject, StrictModel


PROJECT_FILES = ("case.yaml", "analyses.yaml", "machine.yaml")
ModelT = TypeVar("ModelT", bound=BaseModel)


def _known_configuration_keys() -> set[str]:
    """Collect model field names for concise typo suggestions."""
    pending: list[type[StrictModel]] = [StrictModel]
    models: set[type[StrictModel]] = set()
    while pending:
        parent = pending.pop()
        for child in parent.__subclasses__():
            if child not in models:
                models.add(child)
                pending.append(child)
    return {name for model in models for name in model.model_fields}


KNOWN_CONFIGURATION_KEYS = _known_configuration_keys()


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        raise ProjectConfigurationError(
            "JSON configuration belongs to the retired interface. Create a YAML "
            "project with `pelec-post init PROJECT_DIR`."
        )
    if not path.is_file():
        if path.name == "machine.yaml":
            raise ProjectConfigurationError(
                f"Missing {path}. Copy machine.example.yaml to machine.yaml and "
                "set server-local input/output paths."
            )
        raise ProjectConfigurationError(f"Missing required project file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProjectConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ProjectConfigurationError(f"{path} must contain one YAML mapping")
    return value


def _format_validation(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid project configuration in {path}:"]
    for error in exc.errors(include_url=False):
        location = ".".join(str(item) for item in error["loc"])
        message = error["msg"]
        if error["type"] == "extra_forbidden" and error["loc"]:
            unknown = str(error["loc"][-1])
            alternatives = get_close_matches(
                unknown, KNOWN_CONFIGURATION_KEYS, n=3, cutoff=0.55
            )
            if alternatives:
                message += f"; nearest valid: {', '.join(alternatives)}"
        lines.append(f"  {location}: {message}")
    return "\n".join(lines)


def _validated_file(path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(_load_yaml(path))
    except ValidationError as exc:
        raise ProjectConfigurationError(_format_validation(path, exc)) from exc


def load_project(project_dir: str | Path) -> ResolvedProject:
    root = Path(project_dir).expanduser().resolve()
    if root.suffix.lower() == ".json":
        raise ProjectConfigurationError(
            "The clean-break interface accepts a project directory, not a JSON file."
        )
    if not root.is_dir():
        raise ProjectConfigurationError(
            f"Project directory does not exist: {root}. Run `pelec-post init {root}`."
        )
    case_file = _validated_file(root / "case.yaml", CaseFile)
    analyses_file = _validated_file(root / "analyses.yaml", AnalysesFile)
    machine_file = _validated_file(root / "machine.yaml", MachineFile)
    return ResolvedProject(
        root=root,
        case_file=case_file,
        analyses_file=analyses_file,
        machine_file=machine_file,
    )


def dump_yaml(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseException):
        raise TypeError("cannot serialize an exception as project YAML")
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    destination.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def write_project_schema(destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "case.yaml": CaseFile.model_json_schema(),
        "analyses.yaml": AnalysesFile.model_json_schema(),
        "machine.yaml": MachineFile.model_json_schema(),
    }
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
