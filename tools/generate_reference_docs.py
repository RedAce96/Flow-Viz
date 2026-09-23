"""Regenerate registry and configuration references."""

from pathlib import Path

from pelecpost.config.loader import write_project_schema
from pelecpost.documentation import write_configuration_reference, write_recipe_reference

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    write_recipe_reference(ROOT / "docs" / "RECIPES.md")
    write_configuration_reference(ROOT / "docs" / "CONFIGURATION.md")
    project_roots = (ROOT / "examples", ROOT / "verification_outputs")
    for parent in project_roots:
        for case_file in sorted(parent.glob("*/case.yaml")):
            write_project_schema(case_file.parent / "project.schema.json")


if __name__ == "__main__":
    main()
