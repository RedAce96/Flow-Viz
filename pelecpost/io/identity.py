"""Collision-resistant input identities and trusted sidecar manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "pelecpost.input-manifest"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_payloads(source: Path, prefix: str | None = None) -> tuple[Path, ...]:
    source = source.expanduser().resolve()
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"input source does not exist: {source}")
    candidates: list[Path] = []
    top_level = sorted(source.glob(f"{prefix}*")) if prefix else [source]
    for root in top_level:
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(
                path for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    if prefix is None:
        candidates = [
            path for path in source.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
    return tuple(sorted(set(candidates)))


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _aggregate(entries: Iterable[dict[str, Any]]) -> str:
    canonical = "".join(
        f"{entry['path']}\t{int(entry['size_bytes'])}\t{entry['sha256']}\n"
        for entry in sorted(entries, key=lambda item: item["path"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_input_manifest(
    source: Path,
    *,
    prefix: str | None = None,
    payloads: Iterable[Path] | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    root = source if source.is_dir() else source.parent
    files = tuple(payloads) if payloads is not None else collect_payloads(source, prefix)
    entries: list[dict[str, Any]] = []
    for path in sorted(files):
        path = path.resolve()
        info = path.stat()
        entries.append({
            "path": _relative_path(path, root),
            "size_bytes": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
            "sha256": sha256_file(path),
        })
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "prefix": prefix,
        "entries": entries,
        "aggregate_sha256": _aggregate(entries),
    }


def write_input_manifest(
    source: Path,
    output: Path,
    *,
    prefix: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = build_input_manifest(source, prefix=prefix)
    if extra:
        payload.update(extra)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def validate_input_manifest(
    manifest_path: Path,
    *,
    expected_source: Path | None = None,
    expected_prefix: str | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read input identity manifest {manifest_path}: {exc}") from exc
    if payload.get("schema") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported input identity manifest schema in {manifest_path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries or not isinstance(payload.get("root"), str):
        raise ValueError(f"input identity manifest {manifest_path} has an incomplete inventory")
    root = Path(payload["root"]).expanduser().resolve()
    if expected_source is not None:
        source = expected_source.expanduser().resolve()
        expected_files = collect_payloads(source, expected_prefix)
        actual_files = {str(path.resolve()) for path in expected_files}
        manifest_files: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"input identity manifest {manifest_path} has a malformed entry")
            candidate = (root / entry["path"]).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"input identity manifest escapes its root: {candidate}") from exc
            manifest_files.add(str(candidate))
        if manifest_files != actual_files:
            raise ValueError(
                f"input identity manifest {manifest_path} is incomplete or has stale file inventory"
            )
    for entry in entries:
        path = (root / str(entry["path"])).resolve()
        if not path.is_file():
            raise ValueError(f"input identity manifest entry is missing: {path}")
        info = path.stat()
        if int(entry.get("size_bytes", -1)) != info.st_size or int(
            entry.get("mtime_ns", -1)
        ) != info.st_mtime_ns:
            raise ValueError(f"input identity manifest is stale for {path}")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError(f"input identity manifest has no valid SHA-256 for {path}")
    if payload.get("aggregate_sha256") != _aggregate(entries):
        raise ValueError(f"input identity manifest aggregate is inconsistent: {manifest_path}")
    nested = payload.get("source_manifest")
    if nested is not None:
        if not isinstance(nested, dict) or nested.get("schema") != SCHEMA:
            raise ValueError(f"input identity manifest has an invalid source manifest: {manifest_path}")
        nested_root = Path(str(nested.get("root", ""))).expanduser().resolve()
        nested_entries = nested.get("entries")
        if not isinstance(nested_entries, list) or nested.get("aggregate_sha256") != _aggregate(
            nested_entries
        ):
            raise ValueError(f"input identity source aggregate is inconsistent: {manifest_path}")
        for entry in nested_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"input identity source manifest has a malformed entry: {manifest_path}")
            source_path = (nested_root / entry["path"]).resolve()
            if not source_path.is_file():
                raise ValueError(f"input identity source is missing: {source_path}")
            info = source_path.stat()
            if int(entry.get("size_bytes", -1)) != info.st_size or int(
                entry.get("mtime_ns", -1)
            ) != info.st_mtime_ns:
                raise ValueError(f"input identity source manifest is stale for {source_path}")
    return payload
