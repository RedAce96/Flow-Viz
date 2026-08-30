#!/usr/bin/env python3
"""Regenerate registry and configuration references."""

from pathlib import Path

from pelecpost.config.loader import write_project_schema
from pelecpost.documentation import write_configuration_reference, write_recipe_reference


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    write_recipe_reference(ROOT / "docs" / "RECIPES.md")
    write_configuration_reference(ROOT / "docs" / "CONFIGURATION.md")
    for project in sorted((ROOT / "examples").iterdir()):
        if project.is_dir() and (project / "case.yaml").is_file():
            write_project_schema(project / "project.schema.json")


if __name__ == "__main__":
    main()
