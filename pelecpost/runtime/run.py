"""Isolated, failure-tolerant workflow execution."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pelecpost.analysis.executors import execute
from pelecpost.config.loader import dump_yaml
from pelecpost.config.models import ResolvedProject
from pelecpost.errors import PreflightBlockedError, UnsupportedCapabilityError
from pelecpost.preflight import create_plan
from pelecpost.workflows import build_workflow_graph

from .artifacts import ArtifactRegistry, atomic_json
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


def _fingerprint(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {"path": str(path.resolve()), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _input_fingerprints(project: ResolvedProject) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
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
                execute(WorkflowContext(project, plan, run_dir, analysis, registry))
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
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(run_dir / "manifest.json", manifest)
    generate_report(run_dir)
    return RunResult(run_id, run_dir, manifest["status"], failed)
