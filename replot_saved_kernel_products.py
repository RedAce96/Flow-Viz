"""Redraw supported saved Flow Viz products from one archived run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from pelecpost.analysis.replay import replot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Archived run directory")
    parser.add_argument(
        "--project", type=Path,
        help="Explicit project configuration override; archived resolved settings are the default",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Replay destination; an existing output resumes only with identical identity",
    )
    parser.add_argument(
        "--analysis-id", action="append", default=[],
        help="Replay this analysis; repeat to select multiple analyses (default: all replayable)",
    )
    parser.add_argument(
        "--replay-config", type=Path,
        help="Optional replay command defaults; explicit CLI options take precedence",
    )
    args = parser.parse_args()
    file_options = {}
    config_provenance = {}
    if args.replay_config is not None:
        config_path = args.replay_config.expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("replay"), dict):
            parser.error("--replay-config must contain a replay mapping")
        file_options = payload["replay"]
        allowed = {"export_root", "project_dir", "output_root", "analysis_ids"}
        unknown = set(file_options) - allowed
        if unknown:
            parser.error(f"unknown replay config keys: {sorted(unknown)}")
        config_provenance = {
            "replay_config_path": str(config_path),
            "replay_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    export = args.export or file_options.get("export_root")
    if export is None:
        parser.error("--export or replay.export_root is required")
    project = args.project or file_options.get("project_dir")
    output = args.output or file_options.get("output_root")
    analysis_ids = args.analysis_id or file_options.get("analysis_ids", [])
    if not isinstance(analysis_ids, list) or not all(isinstance(item, str) for item in analysis_ids):
        parser.error("replay.analysis_ids must be a list of analysis IDs")
    config_provenance["precedence"] = "explicit CLI options override replay config; archived resolved analyses are default"
    print(replot(
        Path(export), Path(project) if project else None,
        Path(output) if output else None, tuple(analysis_ids),
        invocation=config_provenance,
    ))


if __name__ == "__main__":
    main()
