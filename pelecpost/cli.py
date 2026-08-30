"""Command-line interface for recipe-driven PeleC post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pelecpost.config import load_project, write_project_schema
from pelecpost.errors import CONFIGURATION_EXIT_CODE, PelecPostError, PreflightBlockedError
from pelecpost.io import inspect_project
from pelecpost.preflight import Severity, create_plan
from pelecpost.project import initialize_project
from pelecpost.runtime import generate_report, run_project
from pelecpost.wizard import configure_project
from pelecpost.workflows import RECIPES, build_workflow_graph, recipe_for


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
    build_workflow_graph(project)
    write_project_schema(project.root / "project.schema.json")
    console.print(
        f"[green]Valid project:[/] {project.case_file.case.id}; "
        f"{len(project.enabled_analyses)} enabled analysis recipe(s)"
    )


@app.command("inspect")
def inspect_command(project_dir: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    """Inventory configured inputs without running an analysis."""
    project = load_project(project_dir)
    inventory = inspect_project(project)
    if json_output:
        console.print_json(json.dumps(inventory.as_dict()))
        return
    table = Table(title=f"Input inventory: {project.case_file.case.id}")
    table.add_column("Input")
    table.add_column("Summary")
    plotfiles = inventory.plotfiles
    probes = inventory.probes
    table.add_row(
        "Plotfiles",
        "not configured" if plotfiles is None else
        f"{plotfiles.count} files; dimension={plotfiles.dimensionality}; "
        f"levels=0..{plotfiles.maximum_amr_level}; fields={', '.join(plotfiles.fields)}",
    )
    table.add_row(
        "Probes",
        "not configured" if probes is None else
        f"{probes.format}; {probes.sample_count} samples x {probes.probe_count} probes; "
        f"fields={', '.join(probes.fields)}",
    )
    table.add_row("Comparisons", f"{len(inventory.comparison_archives)} archive(s)")
    table.add_row(
        "Baselines",
        ", ".join(f"{name} ({len(files)} files)" for name, files in inventory.baselines.items())
        or "not configured",
    )
    console.print(table)


@app.command("plan")
def plan_command(project_dir: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    """Perform scientific and resource preflight without computation."""
    project = load_project(project_dir)
    plan = create_plan(project)
    if json_output:
        console.print_json(json.dumps(plan.as_dict(), default=str))
    else:
        console.print(f"Workflow order: [cyan]{' -> '.join(plan.workflow_order) or '(empty)'}[/]")
        for finding in plan.findings:
            color = {Severity.BLOCKER: "red", Severity.WARNING: "yellow", Severity.INFO: "blue"}[finding.severity]
            owner = f" [{finding.analysis_id}]" if finding.analysis_id else ""
            console.print(f"[{color}]{finding.severity}[/] {finding.code}{owner}: {finding.message}")
        resources = plan.sampling["resource_estimate"]
        console.print(
            f"Estimated concurrent peak: {resources['estimated_concurrent_peak_gb']:.2f} GB / "
            f"{resources['memory_limit_gb']:.2f} GB configured"
        )
    if plan.blockers:
        raise PreflightBlockedError(f"preflight rejected the run with {len(plan.blockers)} blocker(s)")


@app.command("configure")
def configure_command(project_dir: Path) -> None:
    """Choose a physical question and write validated recipe YAML."""
    configure_project(str(project_dir), console)


@app.command("run")
def run_command(
    project_dir: Path,
    run_name: Optional[str] = typer.Option(None, "--name", help="Optional run-name suffix."),
) -> None:
    """Preflight and execute enabled recipes in a new isolated run."""
    result = run_project(load_project(project_dir), run_name)
    console.print(f"Run {result.status}: [cyan]{result.run_dir}[/]")
    if result.failed_workflows:
        console.print(f"[red]Failed workflows:[/] {', '.join(result.failed_workflows)}")
        raise typer.Exit(1)


@app.command("report")
def report_command(project_dir: Path, run_id: str) -> None:
    """Regenerate a run report without reading source simulation data."""
    project = load_project(project_dir)
    output = project.machine_file.outputs.root
    if not output.is_absolute():
        output = (project.root / output).resolve()
    path = generate_report(output / run_id)
    console.print(f"[green]Report regenerated:[/] {path}")


@recipes_app.command("list")
def recipes_list_command() -> None:
    """List available physical-question recipes."""
    table = Table(title="PeleC post-processing recipes")
    table.add_column("Recipe")
    table.add_column("Physical question")
    table.add_column("Inputs")
    for definition in RECIPES.values():
        table.add_row(definition.name, definition.question, ", ".join(definition.required_inputs))
    console.print(table)


@recipes_app.command("show")
def recipes_show_command(recipe: str) -> None:
    """Explain one recipe's assumptions, products, and limitations."""
    try:
        definition = recipe_for(recipe)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[bold]{definition.name}[/]: {definition.question}")
    console.print(definition.summary)
    for label, values in (
        ("Required inputs", definition.required_inputs),
        ("Required fields", definition.required_fields),
        ("Assumptions", definition.assumptions),
        ("Outputs", definition.outputs),
        ("Limitations", definition.limitations),
    ):
        console.print(f"[bold]{label}:[/] {', '.join(values) if values else 'none'}")


def _run_probe_utility(arguments: list[str]) -> None:
    from pelecpost.probe_cli import main as compact_main

    status = compact_main(arguments)
    if status:
        raise typer.Exit(1)


@probes_app.command(
    "compact",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def probes_compact_command(context: typer.Context) -> None:
    """Create a compact archive; accepts the standalone compactor options."""
    _run_probe_utility(["create", *context.args])


@probes_app.command(
    "verify",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def probes_verify_command(context: typer.Context) -> None:
    """Verify an archive exactly against its acquisition files."""
    _run_probe_utility(["verify", *context.args])


@probes_app.command(
    "prune",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def probes_prune_command(context: typer.Context) -> None:
    """Preview or prune sources covered by a verification manifest."""
    _run_probe_utility(["prune", *context.args])


def main() -> None:
    try:
        app()
    except PelecPostError as exc:
        console.print(f"[red]Configuration error:[/] {exc}", highlight=False)
        raise SystemExit(CONFIGURATION_EXIT_CODE) from None


if __name__ == "__main__":
    main()
