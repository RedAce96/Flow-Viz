"""Isolated, failure-tolerant workflow execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import os
import platform
import re
import socket
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pelecpost.config.loader import dump_yaml
from pelecpost.config.models import ResolvedProject
from pelecpost.errors import PreflightBlockedError, UnsupportedCapabilityError
from pelecpost.preflight import create_plan
from pelecpost.workflows import build_workflow_graph, workflow_for

from .artifacts import Artifact, ArtifactRegistry, atomic_json
from .context import WorkflowContext
from .report import generate_report


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    status: str
    failed_workflows: tuple[str, ...]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "run"


def _git_provenance(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _packages() -> dict[str, str]:
    names = ("numpy", "scipy", "matplotlib", "yt", "h5py", "pydantic", "typer", "rich")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


CHECKSUM_LIMIT_BYTES = 64 * 1024**2


def _fingerprint(path: Path, *, checksum_limit_bytes: int = CHECKSUM_LIMIT_BYTES) -> dict[str, Any]:
    info = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
    }
    if info.st_size <= checksum_limit_bytes:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        result.update({"checksum_algorithm": "sha256", "checksum": digest.hexdigest()})
    else:
        result["checksum_policy"] = (
            f"omitted because file exceeds {checksum_limit_bytes} byte practical limit"
        )
    return result


def _input_fingerprints(project: ResolvedProject) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in ("case.yaml", "analyses.yaml", "machine.yaml"):
        path = project.root / name
        if path.is_file():
            result.append(_fingerprint(path))
    plotfiles = project.machine_file.inputs.plotfiles
    if plotfiles:
        source = plotfiles.source if plotfiles.source.is_absolute() else project.root / plotfiles.source
        for plotfile in sorted(source.glob(f"{plotfiles.prefix}*")):
            header = plotfile / "Header"
            if header.is_file():
                result.append(_fingerprint(header))
    probes = project.machine_file.inputs.probes
    if probes and probes.compact_file:
        source = probes.compact_file if probes.compact_file.is_absolute() else project.root / probes.compact_file
        if source.is_file():
            result.append(_fingerprint(source))
    elif probes and probes.binary_files:
        from pp_probe_store import expand_paths

        patterns = []
        for pattern in probes.binary_files:
            configured = Path(pattern)
            if any(character in pattern for character in "*?["):
                parent = configured.parent
                if not parent.is_absolute():
                    parent = (project.root / parent).resolve()
                patterns.append(str(parent / configured.name))
            else:
                if not configured.is_absolute():
                    configured = (project.root / configured).resolve()
                patterns.append(str(configured))
        result.extend(_fingerprint(Path(path)) for path in expand_paths(patterns))
    for baseline in project.machine_file.inputs.baselines.values():
        source = baseline.source if baseline.source.is_absolute() else project.root / baseline.source
        for plotfile in sorted(source.glob(f"{baseline.prefix}*")):
            header = plotfile / "Header"
            if header.is_file():
                result.append(_fingerprint(header))
    for configured in project.machine_file.inputs.comparison_archives.values():
        run = configured if configured.is_absolute() else project.root / configured
        for name in ("manifest.json", "artifacts.json"):
            path = run / name
            if path.is_file():
                result.append(_fingerprint(path))
    return result


def _provenance(project: ResolvedProject) -> dict[str, Any]:
    return {
        "git": _git_provenance(project.root),
        "python": platform.python_version(),
        "packages": _packages(),
        "host": socket.gethostname(),
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
        "input_fingerprints": _input_fingerprints(project),
    }


def run_project(project: ResolvedProject, run_name: str | None = None) -> RunResult:
    plan = create_plan(project)
    if plan.blockers:
        raise PreflightBlockedError(
            f"run prohibited by {len(plan.blockers)} preflight blocker(s); use `pelec-post plan`"
        )
    graph = build_workflow_graph(project)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    parts = [stamp, _slug(project.case_file.case.id)]
    if run_name:
        parts.append(_slug(run_name))
    run_id = "_".join(parts)
    output_root = Path(plan.output_root)
    run_dir = output_root / run_id
    sequence = 2
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{sequence:02d}"
        sequence += 1
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("logs", "data", "figures", "report"):
        (run_dir / relative).mkdir()
    dump_yaml(run_dir / "resolved-case.yaml", project.case_file)
    dump_yaml(run_dir / "resolved-analyses.yaml", project.analyses_file)
    dump_yaml(run_dir / "resolved-machine.yaml", project.machine_file)
    atomic_json(run_dir / "plan.json", plan.as_dict())
    registry = ArtifactRegistry(run_dir)
    manifest: dict[str, Any] = {
        "schema": "pelecpost.run-manifest", "schema_version": 1,
        "run_id": run_id, "case_id": project.case_file.case.id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running", "workflows": {}, "provenance": _provenance(project),
    }
    atomic_json(run_dir / "manifest.json", manifest)

    def register_control(artifact_id: str, relative: str, interpretation: str) -> None:
        registry.register(Artifact(
            id=artifact_id, schema_version=1, recipe_instance="__run__",
            kind="run-metadata", path=relative, variable=None, units=None,
            coordinate_metadata={}, source_inputs=(), interpretation=interpretation,
            provenance={},
        ))

    for artifact_id, relative, interpretation in (
        ("run.resolved-case", "resolved-case.yaml", "Resolved portable case configuration."),
        ("run.resolved-analyses", "resolved-analyses.yaml", "Resolved recipe configuration."),
        ("run.resolved-machine", "resolved-machine.yaml", "Resolved machine-local configuration."),
        ("run.plan", "plan.json", "Scientific and resource preflight record."),
        ("run.manifest", "manifest.json", "Atomic workflow state and provenance manifest."),
    ):
        register_control(artifact_id, relative, interpretation)
    log_path = run_dir / "logs" / "run.log"
    dependencies_ok: dict[str, bool] = {}
    analyses = {item.id: item for item in project.enabled_analyses}

    def record(node_id: str, status: str, message: str = "") -> None:
        manifest["workflows"][node_id] = {"status": status, "message": message}
        atomic_json(run_dir / "manifest.json", manifest)

    with log_path.open("a", encoding="utf-8") as log:
        for node_id in graph.order:
            node = graph.node(node_id)
            failed_dependencies = [item for item in node.dependencies if not dependencies_ok.get(item, False)]
            if failed_dependencies:
                record(node_id, "skipped-by-dependency", f"failed dependencies: {failed_dependencies}")
                dependencies_ok[node_id] = False
                continue
            if node.internal:
                record(node_id, "completed", "preflight-validated shared resource")
                dependencies_ok[node_id] = True
                continue
            if node.analysis_id is None:
                raise RuntimeError(f"public workflow node {node_id} has no analysis owner")
            analysis = analyses[node.analysis_id]
            try:
                with WorkflowContext(project, plan, run_dir, analysis, registry) as context:
                    workflow = workflow_for(analysis.recipe)
                    workflow.execute(context)
                    produced = tuple(
                        artifact.id.removeprefix(f"{analysis.id}.")
                        for artifact in registry.artifacts
                        if artifact.recipe_instance == analysis.id
                    )
                    missing_products = [
                        expected
                        for expected in workflow.artifact_declarations_for(analysis)
                        if not any(
                            item == expected or item.startswith(expected + ".")
                            for item in produced
                        )
                    ]
                    if missing_products:
                        raise RuntimeError(
                            "workflow returned without declared artifact(s): "
                            + ", ".join(missing_products)
                        )
            except KeyboardInterrupt:
                traceback.print_exc(file=log)
                record(node_id, "interrupted", "execution interrupted by user or scheduler")
                dependencies_ok[node_id] = False
                manifest["status"] = "interrupted"
                break
            except UnsupportedCapabilityError as exc:
                traceback.print_exc(file=log)
                record(node_id, "unavailable", str(exc))
                dependencies_ok[node_id] = False
            except Exception as exc:  # independent workflows must continue
                traceback.print_exc(file=log)
                record(node_id, "failed", f"{type(exc).__name__}: {exc}")
                dependencies_ok[node_id] = False
            else:
                record(node_id, "completed")
                dependencies_ok[node_id] = True
    failed = tuple(
        name for name, item in manifest["workflows"].items()
        if item["status"] in {"failed", "unavailable", "skipped-by-dependency", "interrupted"}
    )
    if manifest["status"] != "interrupted":
        manifest["status"] = "failed" if failed else "completed"
    from pelecpost.analysis.evidence import write_evidence_report

    evidence_path = write_evidence_report(run_dir, registry)
    registry.register(Artifact(
        id="run.measurement-evidence", schema_version=1, recipe_instance="__run__",
        kind="json", path=str(evidence_path.relative_to(run_dir)), variable=None,
        units=None, coordinate_metadata={},
        source_inputs=tuple(dict.fromkeys(
            source for artifact in registry.artifacts for source in artifact.source_inputs
        )),
        interpretation=(
            "Conservative measurement-based classification with explicit exclusion of "
            "LST/PSE and causal inference."
        ),
        provenance={
            "derived_from": [artifact.id for artifact in registry.artifacts],
            "preprocessing": {
                "classifier": "pelecpost.measurement-evidence",
                "schema_version": 1,
            },
        },
    ))
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(run_dir / "manifest.json", manifest)
    register_control("run.log", "logs/run.log", "Full workflow log including tracebacks.")
    generate_report(run_dir)
    register_control("run.report", "report/index.html", "Portable HTML run report.")
    # Regenerate once so the report's product index includes its own registered entry.
    generate_report(run_dir)
    return RunResult(run_id, run_dir, manifest["status"], failed)
