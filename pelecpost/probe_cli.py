#!/usr/bin/env python3
"""Implementation for compact PeleC probe archive commands."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py

_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / (
    f"compact-probes-{os.environ.get('USER', 'user')}"
)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pp_functions_database as fdb
from pelecpost.io.identity import build_input_manifest, sha256_file
from pp_probe_store import (
    ProbeV2Collection, SCHEMA_NAME, SCHEMA_VERSION,
    create_numeric_dataset, hdf5_chunks, sha256_file,
)


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _source_metadata(paths, include_hash=False):
    result = []
    for value in paths:
        path = Path(value)
        info = path.stat()
        item = {
            "path": str(path.resolve()),
            "size_bytes": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
        }
        if include_hash:
            item["sha256"] = sha256_file(path)
        result.append(item)
    return result


def _write_diagnostics(directory, time, requested_x, results, signal_reader):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "disturbance_windows.csv"
    columns = (
        "probe", "x_cm", "status", "fallback_reason", "onset_time_s",
        "offset_time_s", "padded_start_time_s", "padded_end_time_s",
        "baseline", "noise_scale", "pulse_peak_time_s",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for probe, result in enumerate(results):
            writer.writerow({
                "probe": probe,
                "x_cm": float(requested_x[probe]),
                "status": result["status"],
                "fallback_reason": result.get("fallback_reason"),
                "onset_time_s": result.get("onset_time"),
                "offset_time_s": result.get("offset_time"),
                "padded_start_time_s": result.get("padded_start_time"),
                "padded_end_time_s": result.get("padded_end_time"),
                "baseline": result.get("baseline"),
                "noise_scale": result.get("noise_scale"),
                "pulse_peak_time_s": result.get("pulse_peak_time"),
            })

    valid = np.asarray([item["status"] == "valid" for item in results])
    onset = np.asarray([
        item.get("onset_time", np.nan) if valid[index] else np.nan
        for index, item in enumerate(results)
    ], dtype=float)
    offset = np.asarray([
        item.get("offset_time", np.nan) if valid[index] else np.nan
        for index, item in enumerate(results)
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(requested_x, onset, ".", label="onset")
    ax.plot(requested_x, offset, ".", label="offset")
    ax.set_xlabel("Requested probe x [cm]")
    ax.set_ylabel("Absolute time [s]")
    ax.set_title(f"Probe disturbance windows ({np.count_nonzero(valid)}/{len(valid)} valid)")
    ax.grid(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "window_bounds_vs_x.png", dpi=160)
    plt.close(fig)

    count = len(results)
    representatives = np.unique(np.linspace(0, count - 1, min(6, count), dtype=int))
    values = signal_reader(representatives)
    fig, axes = plt.subplots(len(representatives), 1, figsize=(11, 2.2 * len(representatives)), sharex=True)
    axes = np.atleast_1d(axes)
    for row, (axis, probe) in enumerate(zip(axes, representatives)):
        axis.plot(time, values[:, row], lw=0.7)
        item = results[int(probe)]
        if item.get("padded_start_time") is not None:
            axis.axvspan(
                item["padded_start_time"], item["padded_end_time"],
                color="C1", alpha=0.18,
            )
        axis.set_ylabel(f"p{probe}")
        axis.grid(False)
    axes[-1].set_xlabel("Absolute time [s]")
    fig.suptitle("Representative detector traces and padded windows")
    fig.tight_layout()
    fig.savefig(directory / "representative_windows.png", dpi=160)
    plt.close(fig)
    return [str(csv_path), str(directory / "window_bounds_vs_x.png"),
            str(directory / "representative_windows.png")]


def _detect_windows(collection, stage, args):
    time = collection.time
    n_time, n_probes = len(time), collection.n_probes
    duration = float(time[-1] - time[0])
    dt_value = float(np.median(np.diff(time)))
    min_active = max(args.min_active_fraction * duration, 8.0 * dt_value)
    min_quiet = max(args.min_quiet_fraction * duration, 32.0 * dt_value)
    results: list[dict[str, Any]] = [{} for _ in range(n_probes)]
    dataset = stage["detection_field"]
    for first in range(0, n_probes, args.detection_batch_size):
        last = min(first + args.detection_batch_size, n_probes)
        batch = dataset[:, first:last]
        for local in range(last - first):
            probe = first + local
            signal = batch[:, local]
            if np.count_nonzero(np.isfinite(signal)) != n_time:
                results[probe] = {
                    "status": "invalid", "fallback_reason": "nonfinite_signal",
                    "onset_time": None, "offset_time": None,
                    "padded_start_time": None, "padded_end_time": None,
                    "baseline": np.nan, "noise_scale": np.nan,
                    "pulse_peak_time": None,
                }
                continue
            detected = fdb.detect_disturbance_window(
                signal, time,
                laser_start_time=args.trigger_time,
                baseline_margin=args.baseline_margin,
                pre_event_fraction=args.pre_event_fraction,
                onset_sigma=args.onset_sigma,
                onset_peak_fraction=args.onset_peak_fraction,
                offset_sigma=args.offset_sigma,
                offset_peak_fraction=args.offset_peak_fraction,
                packet_selection=args.packet_selection,
                min_active_duration=min_active,
                min_quiet_duration=min_quiet,
                pad_before=0.0, pad_after=0.0,
                min_samples=1, max_samples=n_time,
                detector_mode=args.detector_mode,
            )
            peak_metric = detected.get("pulse_peak_metric")
            if (peak_metric is None or not np.isfinite(peak_metric)
                    or peak_metric <= detected.get("onset_threshold", np.inf)):
                detected["status"] = "invalid"
                detected["fallback_reason"] = "no_signal_above_onset"
            active_duration = max(
                dt_value,
                float((detected.get("offset_time") or time[-1])
                      - (detected.get("onset_time") or time[0])),
            )
            before = max(args.pad_before_fraction * active_duration,
                         args.min_pad_samples * dt_value)
            after = max(args.pad_after_fraction * active_duration,
                        args.min_pad_samples * dt_value)
            detected["padded_start_time"] = max(
                float(time[0]), float(detected["onset_time"] or time[0]) - before
            )
            detected["padded_end_time"] = min(
                float(time[-1]), float(detected["offset_time"] or time[-1]) + after
            )
            # The large metric array is useful inside the detector but should
            # neither remain in memory nor be serialized per probe.
            detected.pop("metric", None)
            detected.pop("metric_time", None)
            results[probe] = detected
    return results


def _write_compact_file(path, collection, fields, start, stop, results,
                        diagnostics, quality_approved, args):
    n_time = stop - start
    n_probes = collection.n_probes
    chunks = hdf5_chunks(n_time, n_probes)
    with h5py.File(path, "w", libver="latest") as archive:
        archive.attrs.update({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "quality_approved": bool(quality_approved),
            "source_start_index": int(start),
            "source_stop_index": int(stop),
            "source_sample_count": int(len(collection.time)),
            "detection_field": args.detection_field,
            "detector_configuration_json": json.dumps(vars(args), default=str, sort_keys=True),
            "diagnostic_files_json": json.dumps(diagnostics),
        })
        archive.create_dataset("time", data=collection.time[start:stop],
                               fletcher32=True)
        archive.create_dataset("step", data=collection.step[start:stop],
                               fletcher32=True)
        probes = archive.create_group("probes")
        probes.create_dataset("requested_x_cm", data=collection.header["requested_x"])
        probes.create_dataset("requested_y_cm", data=collection.header["requested_y"])
        # Nominal coordinates deliberately remain the fixed requested
        # positions. Complete mapping epochs retain actual sampled cells.
        probes.create_dataset("nominal_x_cm", data=collection.header["requested_x"])
        probes.create_dataset("nominal_y_cm", data=collection.header["requested_y"])

        field_group = archive.create_group("fields")
        for field in fields:
            dataset = create_numeric_dataset(
                field_group, field, (n_time, n_probes), np.float64,
                chunks=chunks,
            )
            dataset.attrs["unit"] = collection.field_units[
                collection.field_index(field)
            ]
            collection.write_field(dataset, field, start, stop)

        detection = archive.create_group("detection")
        string_dtype = h5py.string_dtype("utf-8")
        detection.create_dataset(
            "status", data=np.asarray([item["status"] for item in results], dtype=object),
            dtype=string_dtype,
        )
        detection.create_dataset(
            "fallback_reason",
            data=np.asarray([item.get("fallback_reason") or "" for item in results], dtype=object),
            dtype=string_dtype,
        )
        for name, key in (
                ("onset_time_s", "onset_time"),
                ("offset_time_s", "offset_time"),
                ("padded_start_time_s", "padded_start_time"),
                ("padded_end_time_s", "padded_end_time"),
                ("baseline", "baseline"), ("noise_scale", "noise_scale"),
                ("pulse_peak_time_s", "pulse_peak_time")):
            detection.create_dataset(name, data=np.asarray([
                np.nan if item.get(key) is None else item.get(key)
                for item in results
            ], dtype=float))

        epochs = collection.mapping_epochs(start, stop)
        mapping_group = archive.create_group("mapping")
        mapping_group.create_dataset("epoch_start", data=np.asarray([e[0] for e in epochs], dtype=np.int64))
        mapping_group.create_dataset("epoch_stop", data=np.asarray([e[1] for e in epochs], dtype=np.int64))
        mapping_group.create_dataset("epoch_mapping_id", data=np.asarray([e[2] for e in epochs], dtype=np.int32))
        used_ids = sorted({e[2] for e in epochs})
        mapping_group.create_dataset("mapping_id", data=np.asarray(used_ids, dtype=np.int32))
        if used_ids:
            mapping_group.create_dataset("sample_x_cm", data=np.vstack([collection.mappings[i][0] for i in used_ids]))
            mapping_group.create_dataset("sample_y_cm", data=np.vstack([collection.mappings[i][1] for i in used_ids]))
            mapping_group.create_dataset("level", data=np.vstack([collection.mappings[i][2] for i in used_ids]))
            mapping_group.create_dataset("valid", data=np.vstack([collection.mappings[i][3] for i in used_ids]))

        source = archive.create_group("source")
        source.create_dataset(
            "files_json",
            data=json.dumps(_source_metadata(collection.paths), sort_keys=True),
            dtype=string_dtype,
        )
        source.attrs["overlap_report_json"] = json.dumps(collection.overlap_report, sort_keys=True)
        archive.flush()


def command_create(args):
    if args.detection_batch_size < 1:
        raise ValueError("--detection-batch-size must be positive")
    if args.min_pad_samples < 0:
        raise ValueError("--min-pad-samples cannot be negative")
    for name in (
            "pre_event_fraction", "min_active_fraction",
            "min_quiet_fraction", "pad_before_fraction",
            "pad_after_fraction"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0.0 <= args.minimum_detection_coverage <= 1.0:
        raise ValueError("--minimum-detection-coverage must lie in [0, 1]")
    if not 0.0 < args.maximum_retained_fraction <= 1.0:
        raise ValueError("--maximum-retained-fraction must lie in (0, 1]")
    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}; use --force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path(args.diagnostics or f"{output}.diagnostics").resolve()
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    fields = list(dict.fromkeys(args.fields))
    with ProbeV2Collection(
            args.input, dedup_tol=args.dedup_tol,
            overlap_policy=args.overlap_policy) as collection:
        missing = sorted(set(fields + [args.detection_field]) - set(collection.field_names))
        if missing:
            raise ValueError(f"Fields absent from source: {missing}")
        with tempfile.NamedTemporaryFile(
                prefix="probe-detection-", suffix=".h5",
                dir=output.parent, delete=False) as handle:
            stage_path = Path(handle.name)
        try:
            with h5py.File(stage_path, "w") as stage:
                stage_probe_chunk = min(
                    collection.n_probes, args.detection_batch_size
                )
                stage_time_chunk = max(
                    1, min(
                        len(collection.time),
                        (2 * 1024 * 1024) // (8 * stage_probe_chunk),
                    )
                )
                dataset = create_numeric_dataset(
                    stage, "detection_field",
                    (len(collection.time), collection.n_probes), np.float64,
                    chunks=(stage_time_chunk, stage_probe_chunk),
                )
                collection.write_field(
                    dataset, args.detection_field, 0, len(collection.time)
                )
                results = _detect_windows(collection, stage, args)
                diagnostics = _write_diagnostics(
                    diagnostics_dir, collection.time,
                    collection.header["requested_x"], results,
                    lambda ids: stage["detection_field"][:, ids],
                )
        finally:
            stage_path.unlink(missing_ok=True)

        valid = np.asarray([item["status"] == "valid" for item in results])
        coverage = float(np.mean(valid)) if valid.size else 0.0
        if args.window is not None:
            requested_start, requested_end = map(float, args.window)
            if requested_end <= requested_start:
                raise ValueError("--window requires START < END")
            start = int(np.searchsorted(collection.time, requested_start, side="left"))
            stop = int(np.searchsorted(collection.time, requested_end, side="right"))
            start = max(0, min(start, len(collection.time) - 1))
            stop = max(start + 1, min(stop, len(collection.time)))
            quality_approved = True
        else:
            if not np.any(valid):
                raise RuntimeError(
                    f"No reliable disturbance windows; inspect {diagnostics_dir}"
                )
            starts = [results[i]["padded_start_time"] for i in np.flatnonzero(valid)]
            ends = [results[i]["padded_end_time"] for i in np.flatnonzero(valid)]
            start = int(np.searchsorted(collection.time, min(starts), side="left"))
            stop = int(np.searchsorted(collection.time, max(ends), side="right"))
            retained_fraction = (stop - start) / len(collection.time)
            quality_approved = (
                coverage >= args.minimum_detection_coverage
                and retained_fraction <= args.maximum_retained_fraction
            )
            if not quality_approved:
                raise RuntimeError(
                    f"Compaction quality gate failed: detection coverage={coverage:.1%}, "
                    f"retained={retained_fraction:.1%}; inspect {diagnostics_dir} or "
                    "supply a reviewed --window START END"
                )
        try:
            _write_compact_file(
                temporary, collection, fields, start, stop, results,
                diagnostics, quality_approved, args,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

        source_paths = tuple(Path(path).resolve() for path in collection.paths)
        common_root = Path(os.path.commonpath([str(path) for path in source_paths]))
        if common_root.is_file():
            common_root = common_root.parent
        source_manifest = build_input_manifest(
            common_root, payloads=source_paths,
        )
        archive_info = output.stat()
        sidecar = build_input_manifest(output)
        sidecar.update({
            "kind": "compact_probe_archive",
            "archive": {
                "path": str(output),
                "size_bytes": int(archive_info.st_size),
                "mtime_ns": int(archive_info.st_mtime_ns),
                "sha256": sha256_file(output),
            },
            "source_manifest": source_manifest,
        })
        sidecar_path = output.with_name(output.name + ".manifest.json")
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        source_bytes = sum(Path(path).stat().st_size for path in collection.paths)
        compact_bytes = output.stat().st_size
        report = {
            "output": str(output), "source_bytes": source_bytes,
            "compact_bytes": compact_bytes,
            "storage_ratio": compact_bytes / source_bytes if source_bytes else np.nan,
            "source_samples": len(collection.time),
            "retained_samples": stop - start,
            "retained_fraction": (stop - start) / len(collection.time),
            "detection_coverage": coverage,
            "diagnostics": diagnostics,
            "manifest": str(sidecar_path),
        }
        print(json.dumps(report, indent=2))


def _load_source_list(archive):
    raw = archive["source/files_json"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def command_verify(args):
    archive_path = Path(args.archive).resolve()
    manifest_path = Path(args.manifest or f"{archive_path}.verification.json").resolve()
    with h5py.File(archive_path, "r") as archive:
        if archive.attrs.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"Not a compact probe archive: {archive_path}")
        if not bool(archive.attrs.get("quality_approved", False)):
            raise RuntimeError("Archive has not passed the compaction quality gate")
        recorded_sources = _load_source_list(archive)
        paths = [item["path"] for item in recorded_sources]
        for item in recorded_sources:
            info = Path(item["path"]).stat()
            if info.st_size != item["size_bytes"] or info.st_mtime_ns != item["mtime_ns"]:
                raise RuntimeError(f"Source changed since compaction: {item['path']}")
        start = int(archive.attrs["source_start_index"])
        stop = int(archive.attrs["source_stop_index"])
        with ProbeV2Collection(
                paths, dedup_tol=args.dedup_tol,
                overlap_policy=args.overlap_policy) as collection:
            if not np.array_equal(archive["time"][:], collection.time[start:stop]):
                raise RuntimeError("HDF5 time values do not exactly match source")
            if not np.array_equal(archive["step"][:], collection.step[start:stop]):
                raise RuntimeError("HDF5 steps do not exactly match source")
            for field in archive["fields"]:
                if not collection.compare_field(
                        archive[f"fields/{field}"], field, start, stop):
                    raise RuntimeError(f"HDF5 field {field} does not exactly match source")
        diagnostic_files = json.loads(archive.attrs["diagnostic_files_json"])
        missing_diagnostics = [path for path in diagnostic_files if not Path(path).is_file()]
        if missing_diagnostics:
            raise RuntimeError(f"Missing compaction diagnostics: {missing_diagnostics}")

    sources = _source_metadata(paths, include_hash=True)
    manifest = {
        "schema": "pelec.compact-probes.verification",
        "version": 1,
        "verified_utc": _utc_now(),
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "sources": sources,
        "diagnostic_files": diagnostic_files,
        "exact_value_comparison": True,
        "quality_approved": True,
        "pruned": False,
    }
    _atomic_json(manifest_path, manifest)
    print(f"Verified compact archive; manifest: {manifest_path}")


def command_prune(args):
    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "pelec.compact-probes.verification":
        raise ValueError("Not a compact-probe verification manifest")
    if not manifest.get("quality_approved") or not manifest.get("exact_value_comparison"):
        raise RuntimeError("Manifest does not authorize source deletion")
    if manifest.get("pruned"):
        raise RuntimeError("Manifest records that sources were already pruned")
    archive = Path(manifest["archive"])
    if not archive.is_file() or archive.stat().st_size != manifest["archive_size_bytes"]:
        raise RuntimeError("Verified archive is missing or changed")
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise RuntimeError("Verified archive checksum changed")
    total = 0
    paths = []
    for item in manifest["sources"]:
        path = Path(item["path"])
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Refusing non-regular source path: {path}")
        if path.suffix != ".pbin":
            raise RuntimeError(f"Refusing non-.pbin source path: {path}")
        if info.st_size != item["size_bytes"] or info.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError(f"Source changed since verification: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Source checksum changed since verification: {path}")
        total += info.st_size
        paths.append(path)
    print("Verified source files eligible for permanent deletion:")
    for path in paths:
        print(f"  {path} ({path.stat().st_size} bytes)")
    print(f"Total reclaimable bytes: {total}")
    if not args.confirm_delete:
        print("Preview only; rerun with --confirm-delete after reviewing diagnostics.")
        return
    for path in paths:
        path.unlink()
    manifest.update({
        "pruned": True, "pruned_utc": _utc_now(),
        "deleted_paths": [str(path) for path in paths],
        "reclaimed_bytes": total,
    })
    _atomic_json(manifest_path, manifest)
    print(f"Deleted {len(paths)} source files; reclaimed {total} bytes")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a compact HDF5 archive")
    create.add_argument("--input", nargs="+", required=True, help="probe-v2 paths or globs")
    create.add_argument("--output", required=True)
    create.add_argument("--fields", nargs="+", default=["rho", "u", "p", "T"])
    create.add_argument("--detection-field", default="p")
    create.add_argument("--diagnostics")
    create.add_argument("--window", nargs=2, type=float, metavar=("START_S", "END_S"))
    create.add_argument("--force", action="store_true")
    create.add_argument("--dedup-tol", type=float, default=1.0e-12)
    create.add_argument("--overlap-policy", choices=("latest_segment", "error"), default="latest_segment")
    create.add_argument("--detector-mode", choices=("energy", "abs"), default="energy")
    create.add_argument("--packet-selection", choices=("dominant_peak", "first_threshold"), default="dominant_peak")
    create.add_argument("--trigger-time", type=float)
    create.add_argument("--baseline-margin", type=float, default=0.0)
    create.add_argument("--pre-event-fraction", type=float, default=0.05)
    create.add_argument("--onset-sigma", type=float, default=5.0)
    create.add_argument("--offset-sigma", type=float, default=2.0)
    create.add_argument("--onset-peak-fraction", type=float, default=0.10)
    create.add_argument("--offset-peak-fraction", type=float, default=1.0e-3)
    create.add_argument("--min-active-fraction", type=float, default=1.0e-3)
    create.add_argument("--min-quiet-fraction", type=float, default=1.0e-2)
    create.add_argument("--pad-before-fraction", type=float, default=0.10)
    create.add_argument("--pad-after-fraction", type=float, default=0.25)
    create.add_argument("--min-pad-samples", type=int, default=256)
    create.add_argument("--detection-batch-size", type=int, default=32)
    create.add_argument("--minimum-detection-coverage", type=float, default=0.90)
    create.add_argument("--maximum-retained-fraction", type=float, default=0.95)
    create.set_defaults(func=command_create)

    verify = subparsers.add_parser("verify", help="compare an archive exactly to its sources")
    verify.add_argument("archive")
    verify.add_argument("--manifest")
    verify.add_argument("--dedup-tol", type=float, default=1.0e-12)
    verify.add_argument("--overlap-policy", choices=("latest_segment", "error"), default="latest_segment")
    verify.set_defaults(func=command_verify)

    prune = subparsers.add_parser("prune", help="preview or delete verified source files")
    prune.add_argument("manifest")
    prune.add_argument("--confirm-delete", action="store_true")
    prune.set_defaults(func=command_prune)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
