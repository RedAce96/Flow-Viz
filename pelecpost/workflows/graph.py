"""Build and validate the workflow DAG for enabled recipe instances."""

from __future__ import annotations

from dataclasses import dataclass

from pelecpost.config.models import ResolvedProject
from pelecpost.errors import ProjectConfigurationError

from .registry import INTERNAL_WORKFLOWS, recipe_for, workflow_for


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    workflow: str
    analysis_id: str | None
    internal: bool
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowGraph:
    nodes: tuple[WorkflowNode, ...]
    order: tuple[str, ...]

    def node(self, node_id: str) -> WorkflowNode:
        return next(node for node in self.nodes if node.id == node_id)


def _topological_order(nodes: dict[str, WorkflowNode]) -> tuple[str, ...]:
    remaining = {name: set(node.dependencies) for name, node in nodes.items()}
    unknown = {
        dependency
        for dependencies in remaining.values()
        for dependency in dependencies
        if dependency not in nodes
    }
    if unknown:
        raise ProjectConfigurationError(
            f"Workflow graph contains missing dependencies: {', '.join(sorted(unknown))}"
        )
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
        if not ready:
            cycle = ", ".join(sorted(remaining))
            raise ProjectConfigurationError(f"Workflow dependency cycle detected among: {cycle}")
        for name in ready:
            order.append(name)
            remaining.pop(name)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return tuple(order)


def build_workflow_graph(project: ResolvedProject) -> WorkflowGraph:
    nodes: dict[str, WorkflowNode] = {}
    geometry_requires_plotfiles = project.case_file.geometry.type == "volume_fraction"
    enabled_recipes = {analysis.recipe for analysis in project.enabled_analyses}
    enabled_analysis_ids = {analysis.id for analysis in project.enabled_analyses}
    for analysis in project.enabled_analyses:
        definition = recipe_for(analysis.recipe)
        active_conflicts = enabled_recipes.intersection(definition.conflicts)
        if active_conflicts:
            raise ProjectConfigurationError(
                f"Recipe {analysis.recipe!r} conflicts with: {', '.join(sorted(active_conflicts))}"
            )
        dependency_ids: list[str] = []
        for input_name in workflow_for(analysis.recipe).required_inputs_for(analysis):
            node_id = f"input.{input_name}"
            if node_id not in INTERNAL_WORKFLOWS:
                raise ProjectConfigurationError(
                    f"Recipe {analysis.recipe!r} references undeclared internal workflow {node_id!r}"
                )
            nodes.setdefault(
                node_id,
                WorkflowNode(node_id, node_id, None, True, ()),
            )
            dependency_ids.append(node_id)
        for dependency in definition.dependencies:
            if dependency not in INTERNAL_WORKFLOWS:
                raise ProjectConfigurationError(
                    f"Recipe {analysis.recipe!r} references undeclared internal workflow {dependency!r}"
                )
            nested = ("input.plotfiles",) if (
                dependency == "geometry.surface" and geometry_requires_plotfiles
            ) else ()
            if nested:
                nodes.setdefault(
                    "input.plotfiles",
                    WorkflowNode("input.plotfiles", "input.plotfiles", None, True, ()),
                )
            nodes.setdefault(
                dependency,
                WorkflowNode(dependency, dependency, None, True, nested),
            )
            dependency_ids.append(dependency)
        if analysis.recipe == "case_comparison":
            for reference_name in ("baseline", "comparison"):
                reference = getattr(analysis, reference_name)
                if (
                    reference.archived_run_id is None
                    and reference.analysis_id in enabled_analysis_ids
                    and reference.analysis_id != analysis.id
                ):
                    dependency_ids.append(f"analysis.{reference.analysis_id}")
        node_id = f"analysis.{analysis.id}"
        nodes[node_id] = WorkflowNode(
            node_id,
            definition.workflow,
            analysis.id,
            False,
            tuple(dict.fromkeys(dependency_ids)),
        )
    return WorkflowGraph(tuple(nodes.values()), _topological_order(nodes))
