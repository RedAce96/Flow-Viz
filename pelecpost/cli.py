"""Command-line interface for recipe-driven PeleC post-processing."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pelecpost.config import load_project, write_project_schema
from pelecpost.errors import CONFIGURATION_EXIT_CODE, PelecPostError
from pelecpost.project import initialize_project


app = typer.Typer(
    name="pelec-post",
    no_args_is_help=True,
    add_completion=False,
    help="Recipe-driven PeleC CFD post-processing.",
)
recipes_app = typer.Typer(help="List and explain analysis recipes.")
probes_app = typer.Typer(help="Create, verify, and prune compact probe archives.")
app.add_typer(recipes_app, name="recipes")
app.add_typer(probes_app, name="probes")
console = Console()


@app.command("init")
def init_command(project_dir: Path) -> None:
    """Create a strict YAML project directory."""
    root = initialize_project(project_dir)
    console.print(f"[green]Initialized PeleC post-processing project:[/] {root}")
    console.print("Edit case.yaml and machine.yaml, then run `pelec-post inspect PROJECT`.")


@app.command("validate")
def validate_command(project_dir: Path) -> None:
    """Validate all project files without reading simulation data."""
    project = load_project(project_dir)
    write_project_schema(project.root / "project.schema.json")
    console.print(
        f"[green]Valid project:[/] {project.case_file.case.id}; "
        f"{len(project.enabled_analyses)} enabled analysis recipe(s)"
    )


def main() -> None:
    try:
        app()
    except PelecPostError as exc:
        console.print(f"[red]Configuration error:[/] {exc}", highlight=False)
        raise typer.Exit(CONFIGURATION_EXIT_CODE) from exc


if __name__ == "__main__":
    main()
