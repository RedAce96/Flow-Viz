"""Executor registry; numerical routines remain callable without the CLI."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.runtime.context import WorkflowContext


Executor = Callable[[WorkflowContext], None]
EXECUTORS: dict[str, Executor] = {}

_EXECUTOR_MODULES = {
    "probe_spectrum": "spectral",
    "single_pulse_response": "spectral",
    "directional_wave": "spectral",
    "modal_screening": "modal",
    "transient_wavepacket": "transient",
    "nonlinear_coupling": "nonlinear",
    "flow_overview": "plotfiles",
    "boundary_layer_reference": "plotfiles",
    "surface_diagnostics": "plotfiles",
    "aerodynamic_forces": "plotfiles",
    "case_comparison": "comparison",
}


def executor(recipe: str) -> Callable[[Executor], Executor]:
    def register(function: Executor) -> Executor:
        if recipe in EXECUTORS:
            raise RuntimeError(f"duplicate executor for {recipe}")
        EXECUTORS[recipe] = function
        return function
    return register


def execute(context: WorkflowContext) -> None:
    if context.analysis.recipe not in EXECUTORS:
        # Keep heavy numerical/plotting imports out of configuration-only CLI
        # commands and import only the domain needed by the requested recipe.
        module = _EXECUTOR_MODULES.get(context.analysis.recipe)
        if module is not None:
            with context.timed_phase(
                "executor-module-import", module=module,
                recipe=context.analysis.recipe,
            ):
                import_module(f"pelecpost.analysis.{module}")
    try:
        function = EXECUTORS[context.analysis.recipe]
    except KeyError as exc:
        raise UnsupportedCapabilityError(
            f"Recipe {context.analysis.recipe!r} is registered but its executor has not been migrated"
        ) from exc
    function(context)
