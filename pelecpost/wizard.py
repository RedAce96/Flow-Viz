"""Guided configuration built directly on the typed recipe models."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

from pelecpost.config.loader import dump_yaml, load_project
from pelecpost.io import inspect_project
from pelecpost.workflows import RECIPES


VARIABLES = ("temperature", "pressure", "density", "x_velocity", "y_velocity")


def _choice(prompt: str, choices: tuple[str, ...], default: int = 1) -> str:
    value = typer.prompt(prompt, default=default, type=int)
    if value < 1 or value > len(choices):
        raise typer.BadParameter(f"choose a number from 1 to {len(choices)}")
    return choices[value - 1]


def _variable() -> str:
    for index, name in enumerate(VARIABLES, 1):
        typer.echo(f"  {index}. {name}")
    return _choice("Variable", VARIABLES)


def _analysis(recipe: str, inventory: Any, console: Console) -> dict[str, Any]:
    result: dict[str, Any] = {"id": recipe.replace("_", "-"), "recipe": recipe}
    probes = inventory.probes
    if probes and probes.median_timestep_s:
        nyquist = 0.5 / probes.median_timestep_s
        duration = (probes.time_max_s or 0.0) - (probes.time_min_s or 0.0)
        console.print(
            f"Derived probe limits: Nyquist [cyan]{nyquist:.6g} Hz[/], "
            f"native resolution [cyan]{1.0 / duration:.6g} Hz[/]"
            if duration > 0 else f"Derived probe Nyquist: [cyan]{nyquist:.6g} Hz[/]"
        )
    if recipe == "flow_overview":
        result["fields"] = ["temperature", "pressure"]
    elif recipe == "boundary_layer_reference":
        result["stations_x_m"] = [typer.prompt("Streamwise station [m]", type=float)]
        result["maximum_height_m"] = typer.prompt("Maximum profile height [m]", type=float)
        result["wall_temperature_k"] = typer.prompt("Wall temperature [K]", type=float)
        result["dynamic_viscosity_pa_s"] = typer.prompt("Dynamic viscosity [Pa s]", type=float)
        result["conductivity_w_m_k"] = typer.prompt("Thermal conductivity [W/(m K)]", type=float)
    elif recipe == "surface_diagnostics":
        pass
    elif recipe == "aerodynamic_forces":
        result.update({
            "reference_chord_m": typer.prompt("Reference chord [m]", type=float),
            "dynamic_viscosity_pa_s": typer.prompt("Dynamic viscosity [Pa s]", type=float),
            "conductivity_w_m_k": typer.prompt("Thermal conductivity [W/(m K)]", type=float),
        })
        baseline = typer.prompt("Baseline mode (none/static/paired)", default="none")
        result["baseline"] = baseline
        if baseline != "none":
            result["baseline_id"] = typer.prompt("Baseline ID from machine.yaml")
    elif recipe == "probe_spectrum":
        result["variable"] = _variable()
        result["frequency_max_hz"] = typer.prompt("Maximum frequency [Hz]", type=float)
    elif recipe == "single_pulse_response":
        result.update({
            "variable": _variable(),
            "energy_per_pulse_j_m": typer.prompt("Pulse energy per unit span [J/m]", type=float),
            "pulse_fwhm_s": typer.prompt("Pulse FWHM [s]", type=float),
            "pulse_period_s": typer.prompt("Pulse period [s]", type=float),
            "start_time_s": typer.prompt("Pulse start time [s]", type=float),
        })
    elif recipe == "directional_wave":
        result.update({
            "variable": _variable(),
            "frequency_min_hz": typer.prompt("Minimum frequency [Hz]", type=float),
            "frequency_max_hz": typer.prompt("Maximum frequency [Hz]", type=float),
        })
    elif recipe == "transient_wavepacket":
        result.update({
            "variable": _variable(),
            "band_min_hz": typer.prompt("Band minimum [Hz]", type=float),
            "band_max_hz": typer.prompt("Band maximum [Hz]", type=float),
        })
    elif recipe == "nonlinear_coupling":
        result.update({
            "variable": _variable(),
            "frequency_max_hz": typer.prompt("Maximum frequency [Hz]", type=float),
            "automatic_frequency_selection": typer.confirm(
                "Automatically select candidate frequencies from stationary PSD peaks?",
                default=False,
            ),
        })
        if not result["automatic_frequency_selection"]:
            targets = typer.prompt("Comma-separated target frequencies [Hz]")
            result["target_frequencies_hz"] = [float(value) for value in targets.split(",")]
    elif recipe == "modal_screening":
        result["variable"] = _variable()
    elif recipe == "case_comparison":
        result.update({
            "baseline_run": typer.prompt("Baseline run directory"),
            "comparison_run": typer.prompt("Comparison run directory"),
            "artifact_ids": [typer.prompt("Artifact ID")],
        })
    return result


def configure_project(project_dir: str, console: Console) -> None:
    project = load_project(project_dir)
    inventory = inspect_project(project)
    available_inputs = {
        "plotfiles": inventory.plotfiles is not None and inventory.plotfiles.count > 0,
        "probes": inventory.probes is not None and inventory.probes.sample_count > 0,
        "comparison_archives": bool(inventory.comparison_archives),
    }
    compatible = tuple(
        name for name, definition in RECIPES.items()
        if project.case_file.case.dimensionality in definition.supported_dimensions
        and project.case_file.geometry.type in definition.supported_geometries
        and all(available_inputs[item] for item in definition.required_inputs)
    )
    if not compatible:
        raise typer.BadParameter(
            "No recipe is compatible with the inspected inputs; configure machine.yaml first."
        )
    console.print("Select the physical question to configure:")
    for index, name in enumerate(compatible, 1):
        console.print(f"  {index}. {RECIPES[name].question} [dim]({name})[/]")
    recipe = _choice("Question", compatible)
    analysis = _analysis(recipe, inventory, console)
    analysis["id"] = typer.prompt("Analysis ID", default=analysis["id"])
    case_path = project.root / "case.yaml"
    dump_yaml(case_path, project.case_file)
    dump_yaml(
        project.root / "analyses.yaml",
        {"schema_version": 1, "analyses": [analysis]},
    )
    # The wizard and hand editing deliberately share this exact validation path.
    load_project(project.root)
    console.print(f"[green]Configured[/] {analysis['id']} using {recipe}.")
