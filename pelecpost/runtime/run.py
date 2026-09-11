"""Isolated, failure-tolerant workflow execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import time
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


def _fingerprint(
    path: Path, *, checksum_limit_bytes: int = CHECKSUM_LIMIT_BYTES,
    input_id: str | None = None,
) -> dict[str, Any]:
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
    if input_id is not None:
        result["input_id"] = input_id
    return result


def _input_fingerprints(project: ResolvedProject) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in ("case.yaml", "analyses.yaml", "machine.yaml"):
        path = project.root / name
        if path.is_file():
            result.append(_fingerprint(path, input_id=name))
    plotfiles = project.machine_file.inputs.plotfiles
    if plotfiles:
        configured_source = plotfiles.source.expanduser()
        source = configured_source if configured_source.is_absolute() else project.root / configured_source
        for plotfile in sorted(source.glob(f"{plotfiles.prefix}*")):
            header = plotfile / "Header"
            if header.is_file():
                result.append(_fingerprint(header, input_id="inputs.plotfiles"))
    for probe_set_id, probe_set in project.machine_file.inputs.probe_sets.items():
        probes = probe_set
        if probes.compact_file:
            configured_source = probes.compact_file.expanduser()
            source = configured_source if configured_source.is_absolute() else project.root / configured_source
            if source.is_file():
                result.append(_fingerprint(
                    source, input_id=f"inputs.probe_sets.{probe_set_id}"
                ))
        elif probes.binary_files:
            from pp_probe_store import expand_paths

            patterns = []
            for pattern in probes.binary_files:
                configured = Path(pattern).expanduser()
                if any(character in pattern for character in "*?["):
                    parent = configured.parent
                    if not parent.is_absolute():
                        parent = (project.root / parent).resolve()
                    patterns.append(str(parent / configured.name))
                else:
                    if not configured.is_absolute():
                        configured = (project.root / configured).resolve()
                    patterns.append(str(configured))
            result.extend(
                _fingerprint(Path(path), input_id=f"inputs.probe_sets.{probe_set_id}")
                for path in expand_paths(patterns)
            )
    for archive_id, configured in project.machine_file.inputs.archived_runs.items():
        configured = configured.expanduser()
        run = configured if configured.is_absolute() else project.root / configured
        for name in ("manifest.json", "artifacts.json"):
            path = run / name
            if path.is_file():
                result.append(_fingerprint(path, input_id=f"inputs.archived_runs.{archive_id}"))
        artifacts_path = run / "artifacts.json"
        if artifacts_path.is_file():
            try:
                records = json.loads(artifacts_path.read_text(encoding="utf-8"))["artifacts"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                records = ()
            for record in records:
                relative = record.get("path") if isinstance(record, dict) else None
                if not isinstance(relative, str):
                    continue
                path = (run / relative).resolve()
                try:
                    path.relative_to(run.resolve())
                except ValueError:
                    continue
                if path.is_file():
                    record_id = record.get("id", relative) if isinstance(record, dict) else relative
                    result.append(_fingerprint(
                        path, input_id=f"inputs.archived_runs.{archive_id}:{record_id}"
                    ))
    for baseline_id, baseline in project.machine_file.inputs.baselines.items():
        configured_source = baseline.source.expanduser()
        source = configured_source if configured_source.is_absolute() else project.root / configured_source
        for plotfile in sorted(source.glob(f"{baseline.prefix}*")):
            header = plotfile / "Header"
            if header.is_file():
                result.append(_fingerprint(
                    header, input_id=f"inputs.baselines.{baseline_id}"
                ))
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
    run_started = time.perf_counter()
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
    log_path = run_dir / "logs" / "run.log"

    def emit(level: str, event: str, message: str, **details: Any) -> None:
        payload = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "message": message,
            **details,
        }
        # Opening for each event makes every line visible immediately to `tail -f` and
        # durable even if a batch scheduler terminates the process between phases.
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            stream.flush()

    emit("INFO", "run-start", "post-processing run initialized", run_id=run_id)
    provenance_started = time.perf_counter()
    emit("INFO", "phase-start", "collecting run provenance", phase="run-provenance")
    provenance = _provenance(project)
    emit(
        "INFO", "phase-complete", "collected run provenance",
        phase="run-provenance", elapsed_s=time.perf_counter() - provenance_started,
    )
    manifest: dict[str, Any] = {
        "schema": "pelecpost.run-manifest", "schema_version": 1,
        "run_id": run_id, "case_id": project.case_file.case.id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running", "workflows": {}, "provenance": provenance,
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
    dependencies_ok: dict[str, bool] = {}
    analyses = {item.id: item for item in project.enabled_analyses}

    def record(node_id: str, status: str, message: str = "", **details: Any) -> None:
        entry = manifest["workflows"].setdefault(node_id, {})
        entry.update({"status": status, "message": message, **details})
        atomic_json(run_dir / "manifest.json", manifest)

    for node_id in graph.order:
        node = graph.node(node_id)
        failed_dependencies = [item for item in node.dependencies if not dependencies_ok.get(item, False)]
        if failed_dependencies:
            message = f"failed dependencies: {failed_dependencies}"
            record(node_id, "skipped-by-dependency", message)
            emit("WARNING", "workflow-skipped", message, workflow=node_id)
            dependencies_ok[node_id] = False
            continue
        if node.internal:
            record(node_id, "completed", "preflight-validated shared resource")
            emit(
                "INFO", "workflow-complete", "preflight-validated shared resource",
                workflow=node_id, elapsed_s=0.0,
            )
            dependencies_ok[node_id] = True
            continue
        if node.analysis_id is None:
            raise RuntimeError(f"public workflow node {node_id} has no analysis owner")
        analysis = analyses[node.analysis_id]
        workflow_started = time.perf_counter()
        started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        active_phase: str | None = "workflow-initialization"
        phase_timings: list[dict[str, Any]] = []
        artifact_count_before = len(registry.artifacts)
        record(node_id, "running", started_utc=started_utc, active_phase=active_phase)
        emit(
            "INFO", "workflow-start", f"starting {node_id}", workflow=node_id,
            analysis_id=analysis.id, recipe=analysis.recipe,
        )

        def workflow_progress(event: str, message: str, details: dict[str, Any]) -> None:
            nonlocal active_phase
            phase = details.get("phase")
            if event == "phase-start" and isinstance(phase, str):
                active_phase = phase
            elif event == "phase-complete":
                parent_phase = details.get("parent_phase")
                active_phase = parent_phase if isinstance(parent_phase, str) else None
                phase_timings.append({"event": event, **details})
            elif event == "phase-failed":
                phase_timings.append({"event": event, **details})
            entry = manifest["workflows"][node_id]
            entry["active_phase"] = active_phase
            entry["last_progress_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_json(run_dir / "manifest.json", manifest)
            level = "ERROR" if event == "phase-failed" else "INFO"
            emit(level, event, message, workflow=node_id, **details)

        try:
            with WorkflowContext(
                project, plan, run_dir, analysis, registry,
                progress_callback=workflow_progress,
            ) as context:
                with context.timed_phase("workflow-execution"):
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
        except KeyboardInterrupt as exc:
            elapsed = time.perf_counter() - workflow_started
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
                log.flush()
            message = "execution interrupted by user or scheduler"
            record(
                node_id, "interrupted", message, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration_s=elapsed, active_phase=active_phase, phase_timings=phase_timings,
            )
            emit(
                "ERROR", "workflow-interrupted", message, workflow=node_id,
                active_phase=active_phase, elapsed_s=elapsed, error=str(exc),
            )
            dependencies_ok[node_id] = False
            manifest["status"] = "interrupted"
            break
        except UnsupportedCapabilityError as exc:
            elapsed = time.perf_counter() - workflow_started
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
                log.flush()
            record(
                node_id, "unavailable", str(exc), completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration_s=elapsed, active_phase=active_phase, phase_timings=phase_timings,
            )
            emit(
                "ERROR", "workflow-unavailable", str(exc), workflow=node_id,
                active_phase=active_phase, elapsed_s=elapsed,
            )
            dependencies_ok[node_id] = False
        except Exception as exc:  # independent workflows must continue
            elapsed = time.perf_counter() - workflow_started
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
                log.flush()
            message = f"{type(exc).__name__}: {exc}"
            record(
                node_id, "failed", message, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration_s=elapsed, active_phase=active_phase, phase_timings=phase_timings,
                error_type=type(exc).__name__,
            )
            emit(
                "ERROR", "workflow-failed", message, workflow=node_id,
                active_phase=active_phase, elapsed_s=elapsed,
            )
            dependencies_ok[node_id] = False
        else:
            elapsed = time.perf_counter() - workflow_started
            produced_count = len(registry.artifacts) - artifact_count_before
            record(
                node_id, "completed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration_s=elapsed, active_phase=None, phase_timings=phase_timings,
                artifact_count=produced_count,
            )
            emit(
                "INFO", "workflow-complete", f"completed {node_id}", workflow=node_id,
                elapsed_s=elapsed, artifact_count=produced_count,
            )
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
    manifest["duration_s"] = time.perf_counter() - run_started
    atomic_json(run_dir / "manifest.json", manifest)
    emit(
        "INFO", "run-complete", f"run finished with status {manifest['status']}",
        status=manifest["status"], elapsed_s=manifest["duration_s"],
        failed_workflows=list(failed),
    )
    register_control("run.log", "logs/run.log", "Full workflow log including tracebacks.")
    generate_report(run_dir)
    register_control("run.report", "report/index.html", "Portable HTML run report.")
    # Regenerate once so the report's product index includes its own registered entry.
    generate_report(run_dir)
    return RunResult(run_id, run_dir, manifest["status"], failed)
