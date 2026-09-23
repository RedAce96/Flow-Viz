"""Summarize recorded flow-overview worker peaks from one saved run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _external_peak_bytes(path: Path | None) -> int | None:
    if path is None:
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            return int(line.rsplit(":", 1)[1].strip()) * 1024
        if "maximum resident set size" in line:
            return int(line.strip().split()[0])
    return None


def summarize_run(run_dir: Path, time_log: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    machine_path = run_dir / "resolved-machine.yaml"
    machine = yaml.safe_load(machine_path.read_text(encoding="utf-8")) if machine_path.is_file() else {}
    analyses_path = run_dir / "resolved-analyses.yaml"
    analyses = yaml.safe_load(analyses_path.read_text(encoding="utf-8")) if analyses_path.is_file() else {}
    tasks: list[dict[str, Any]] = []
    log_path = run_dir / "logs/run.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event") != "parallel-task-complete"
                or event.get("stage") != "flow-overview-contour-render"
            ):
                continue
            tasks.append({
                "task_id": event.get("task_id"),
                "plotfile": event.get("work_unit", {}).get("plotfile"),
                "worker_pid": event.get("worker_pid"),
                "peak_rss_bytes": event.get("peak_rss_bytes"),
                "elapsed_s": event.get("elapsed_s"),
            })
    peaks = [int(task["peak_rss_bytes"]) for task in tasks if task["peak_rss_bytes"] is not None]
    pids = [int(task["worker_pid"]) for task in tasks if task["worker_pid"] is not None]
    workflows = manifest.get("workflows", {})
    overview = {
        key: item for key, item in workflows.items()
        if isinstance(item, dict)
        and "flow-overview-contour-render" in item.get("parallel_stages", {})
    }
    external_peak = _external_peak_bytes(time_log)
    return {
        "schema": "pelecpost.flow_overview_memory_observation",
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "run_status": manifest.get("status"),
        "configured_plotfile_source": machine.get("inputs", {}).get("plotfiles"),
        "flow_overview_analyses": [
            item for item in analyses.get("analyses", [])
            if isinstance(item, dict) and item.get("recipe") == "flow_overview"
        ],
        "configured_memory_limit_gb": machine.get("compute", {}).get("memory_limit_gb"),
        "completed_render_tasks": len(tasks),
        "peak_worker_rss_gb": max(peaks) / 1024**3 if peaks else None,
        "external_process_peak_rss_gb": external_peak / 1024**3 if external_peak else None,
        "unique_worker_count": len(set(pids)),
        "one_fresh_worker_per_completed_task": len(pids) == len(tasks) == len(set(pids)) if tasks else None,
        "status": (
            "measured_worker_tasks" if peaks else
            "measured_process_peak" if external_peak else
            "unavailable_no_memory_measurement"
        ),
        "tasks": tasks,
        "workflow_records": overview,
        "interpretation": (
            "Worker ru_maxrss is each process lifetime high-water mark; it is not "
            "a live RSS trace. Compare one-snapshot and repeated-snapshot runs "
            "against the same dataset and memory allocation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-log", type=Path, help="Optional /usr/bin/time -v output")
    arguments = parser.parse_args()
    report = summarize_run(arguments.run_dir, arguments.time_log)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
