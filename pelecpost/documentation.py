"""Deterministic reference documentation generated from public contracts."""

from __future__ import annotations

from pathlib import Path

from pelecpost.workflows import RECIPES


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
