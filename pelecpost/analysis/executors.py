"""Executor registry; numerical routines remain callable without the CLI."""

from __future__ import annotations

from collections.abc import Callable

from pelecpost.errors import UnsupportedCapabilityError
from pelecpost.runtime.context import WorkflowContext


Executor = Callable[[WorkflowContext], None]
EXECUTORS: dict[str, Executor] = {}


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
        # commands and register domain executors only when they are requested.
        from . import spectral as _spectral  # noqa: F401
    try:
        function = EXECUTORS[context.analysis.recipe]
    except KeyError as exc:
        raise UnsupportedCapabilityError(
            f"Recipe {context.analysis.recipe!r} is registered but its executor has not been migrated"
        ) from exc
    function(context)

