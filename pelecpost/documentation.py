"""Deterministic reference documentation generated from public contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pelecpost.config.models import AnalysesFile, CaseFile, MachineFile
from pelecpost.workflows import RECIPES


def _schema_type(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "const" in schema:
        return repr(schema["const"])
    if "enum" in schema:
        return " | ".join(repr(item) for item in schema["enum"])
    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        return " | ".join(_schema_type(item) for item in variants)
    if schema.get("type") == "array":
        items = schema.get("items", {})
        if "prefixItems" in schema:
            contents = ", ".join(_schema_type(item) for item in schema["prefixItems"])
            return f"tuple[{contents}]"
        return f"array[{_schema_type(items)}]"
    if schema.get("type") == "object":
        values = schema.get("additionalProperties")
        return f"mapping[string, {_schema_type(values)}]" if isinstance(values, dict) else "object"
    return str(schema.get("type", "value"))


def _schema_rules(schema: dict[str, Any]) -> str:
    rules: list[str] = []
    if schema.get("description"):
        rules.append(str(schema["description"]))
    labels = {
        "minimum": "minimum",
        "exclusiveMinimum": "greater than",
        "maximum": "maximum",
        "exclusiveMaximum": "less than",
        "minLength": "minimum length",
        "maxLength": "maximum length",
        "minItems": "minimum items",
        "maxItems": "maximum items",
        "pattern": "pattern",
    }
    for key, label in labels.items():
        if key in schema:
            rules.append(f"{label}: `{schema[key]}`")
    if "discriminator" in schema:
        rules.append(f"discriminator: `{schema['discriminator']['propertyName']}`")
    return "; ".join(rules) or "—"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _model_table(title: str, schema: dict[str, Any]) -> list[str]:
    required = set(schema.get("required", ()))
    lines = [
        f"### `{title}`",
        "",
        "| Field | Required | Type | Default | Rules |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, field_schema in schema.get("properties", {}).items():
        default = (
            json.dumps(field_schema["default"], sort_keys=True)
            if "default" in field_schema else "—"
        )
        lines.append(
            "| "
            + " | ".join((
                f"`{name}`",
                "yes" if name in required else "no",
                f"`{_markdown_cell(_schema_type(field_schema))}`",
                f"`{_markdown_cell(default)}`" if default != "—" else "—",
                _markdown_cell(_schema_rules(field_schema)),
            ))
            + " |"
        )
    lines.append("")
    return lines


def render_configuration_reference() -> str:
    """Render every public YAML model and field from Pydantic JSON Schema."""
    lines = [
        "# Configuration reference",
        "",
        "This file is generated from the strict Pydantic project models. Every project requires",
        "`case.yaml`, `analyses.yaml`, and `machine.yaml`, each with `schema_version: 1`.",
        "`pelec-post init` and `validate` also write `project.schema.json` for editor completion.",
        "",
        "Unknown keys and keys placed in the wrong file are rejected. Public dimensional values",
        "use SI and unit-bearing names; PeleC CGS values are converted at the I/O boundary. Relative",
        "paths resolve from the project directory. The wizard and direct YAML editing use these same",
        "models. An empty `analyses` array is valid and runs nothing.",
        "",
        "Large selected probe matrices spill to a bounded temporary memory-mapped workspace. Set",
        "`compute.scratch_directory` to node-local storage when available; spill files are removed",
        "after each workflow and are not run artifacts.",
        "",
    ]
    roots = (
        ("case.yaml", CaseFile),
        ("analyses.yaml", AnalysesFile),
        ("machine.yaml", MachineFile),
    )
    for filename, model in roots:
        schema = model.model_json_schema()
        lines.extend((f"## `{filename}`", ""))
        lines.extend(_model_table(model.__name__, schema))
        for name, definition in schema.get("$defs", {}).items():
            lines.extend(_model_table(name, definition))
    lines.extend((
        "Repository maintainers regenerate this reference, the recipe reference, and example",
        "schemas with `python -m tools.generate_reference_docs`.",
        "",
    ))
    return "\n".join(lines)


def write_configuration_reference(destination: str | Path) -> Path:
    """Write the generated Pydantic configuration reference."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_configuration_reference(), encoding="utf-8")
    return path


def render_recipe_reference() -> str:
    """Render the recipe catalog without maintaining a second metadata list."""
    lines = [
        "# Recipe reference",
        "",
        "This file is generated from the central workflow registry. "
        "`pelec-post recipes show NAME` presents the same contract.",
        "",
    ]
    for recipe in RECIPES.values():
        lines.extend((
            f"## `{recipe.name}`",
            "",
            f"Physical question: {recipe.question}",
            "",
            recipe.summary,
            "",
            f"- Dimensions: {', '.join(str(item) + '-D' for item in recipe.supported_dimensions)}",
            f"- Geometries: {', '.join(recipe.supported_geometries)}",
            f"- Required inputs: {', '.join(recipe.required_inputs)}",
            f"- Required plotfile fields: {', '.join(recipe.required_fields) or 'none'}",
            f"- Dependencies: {', '.join(recipe.dependencies) or 'none'}",
            f"- Conflicts: {', '.join(recipe.conflicts) or 'none'}",
            f"- Artifact IDs: {', '.join(recipe.outputs)}",
            "",
            "Assumptions:",
            "",
            *(f"- {item}" for item in recipe.assumptions),
            "",
            "Interpretation limits:",
            "",
            *(f"- {item}" for item in recipe.limitations),
            "",
        ))
    lines.extend((
        "All current numerical recipes support 2-D only. Inspection recognizes 3-D datasets,",
        "but planning blocks unsupported algorithms before expensive loading.",
        "",
    ))
    return "\n".join(lines)


def write_recipe_reference(destination: str | Path) -> Path:
    """Write the generated recipe reference to a documentation tree."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_recipe_reference(), encoding="utf-8")
    return path
