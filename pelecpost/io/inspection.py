"""Metadata-only inspection of PeleC plotfiles and probe stores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pelecpost.config.models import ResolvedProject


FIELD_ALIASES = {
    "density": ("density", "rho"),
    "x_velocity": ("x_velocity", "xvel", "u"),
    "y_velocity": ("y_velocity", "yvel", "v"),
    "pressure": ("pressure", "p"),
    "temperature": ("temperature", "temp", "T"),
    "volume_fraction": ("volume_fraction", "vfrac", "volfrac"),
}


@dataclass(frozen=True)
class PlotfileInventory:
    source: str
    prefix: str
    count: int
    names: tuple[str, ...]
    time_min_s: float | None
    time_max_s: float | None
    dimensionality: int | None
    maximum_amr_level: int | None
    fields: tuple[str, ...]
    canonical_fields: dict[str, str]
    solver_units: str
    domain_bounds_m: tuple[tuple[float, float], ...] | None = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeInventory:
    source: str
    format: str
    sample_count: int
    probe_count: int
    fields: tuple[str, ...]
    field_units: dict[str, str]
    x_min_m: float | None
    x_max_m: float | None
    y_min_m: float | None
    y_max_m: float | None
    time_min_s: float | None
    time_max_s: float | None
    median_timestep_s: float | None
    timestep_std_s: float | None
    nonfinite_time_count: int
    missing_value_count: dict[str, int]
    restart_overlap_count: int
    mapping_epoch_count: int
    quality_approved: bool | None
    provenance_present: bool
    requested_x_m: tuple[float, ...] = ()
    requested_y_m: tuple[float, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputInventory:
    plotfiles: PlotfileInventory | None
    probes: ProbeInventory | None
    comparison_archives: tuple[str, ...]
    baselines: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(project: ResolvedProject, path: Path) -> Path:
    return path if path.is_absolute() else (project.root / path).resolve()


def _read_plotfile_header(
    path: Path,
) -> tuple[tuple[str, ...], int, float, int, tuple[tuple[float, float], ...] | None]:
    lines = (path / "Header").read_text(encoding="utf-8", errors="replace").splitlines()
    count = int(lines[1].strip())
    fields = tuple(line.strip() for line in lines[2 : 2 + count])
    cursor = 2 + count
    dimension = int(lines[cursor].strip())
    physical_time = float(lines[cursor + 1].strip())
    maximum_level = int(lines[cursor + 2].strip())
    bounds = None
    if len(lines) > cursor + 4:
        lower = tuple(float(value) for value in lines[cursor + 3].split())
        upper = tuple(float(value) for value in lines[cursor + 4].split())
        if len(lower) >= dimension and len(upper) >= dimension:
            bounds = tuple((lower[index], upper[index]) for index in range(dimension))
    return fields, dimension, physical_time, maximum_level, bounds


def _canonical_fields(fields: tuple[str, ...], configured: dict[str, str]) -> dict[str, str]:
    lower = {field.lower(): field for field in fields}
    result: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        selected = configured.get(canonical)
        if selected in fields:
            result[canonical] = selected
            continue
        for alias in aliases:
            if alias.lower() in lower:
                result[canonical] = lower[alias.lower()]
                break
    return result


def _inspect_plotfiles(project: ResolvedProject) -> PlotfileInventory | None:
    config = project.machine_file.inputs.plotfiles
    if config is None:
        return None
    source = _resolve(project, config.source)
    candidates = sorted(
        path for path in source.glob(f"{config.prefix}*") if (path / "Header").is_file()
    ) if source.is_dir() else []
    fields: tuple[str, ...] = ()
    dimensions: list[int] = []
    times: list[float] = []
    levels: list[int] = []
    bounds_items: list[tuple[tuple[float, float], ...]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            item_fields, dimension, physical_time, level, bounds = _read_plotfile_header(path)
            if not fields:
                fields = item_fields
            elif fields != item_fields:
                errors.append(f"field list differs in {path.name}")
            dimensions.append(dimension)
            times.append(physical_time)
            levels.append(level)
            if bounds is not None:
                bounds_items.append(bounds)
        except (OSError, ValueError, IndexError) as exc:
            errors.append(f"{path.name}: {exc}")
    unique_dimensions = set(dimensions)
    if len(unique_dimensions) > 1:
        errors.append("plotfiles report inconsistent dimensionality")
    if bounds_items and any(item != bounds_items[0] for item in bounds_items[1:]):
        errors.append("plotfiles report inconsistent physical domain bounds")
    scale = 0.01 if project.case_file.case.solver_units == "cgs" else 1.0
    domain_bounds = (
        tuple((low * scale, high * scale) for low, high in bounds_items[0])
        if bounds_items else None
    )
    return PlotfileInventory(
        source=str(source),
        prefix=config.prefix,
        count=len(candidates),
        names=tuple(path.name for path in candidates),
        time_min_s=min(times) if times else None,
        time_max_s=max(times) if times else None,
        dimensionality=dimensions[0] if dimensions else None,
        maximum_amr_level=max(levels) if levels else None,
        fields=fields,
        canonical_fields=_canonical_fields(fields, project.case_file.field_aliases),
        solver_units=project.case_file.case.solver_units,
        domain_bounds_m=domain_bounds,
        errors=tuple(errors),
    )


def _inspect_hdf5(path: Path) -> ProbeInventory:
    errors: list[str] = []
    with h5py.File(path, "r") as archive:
        schema = archive.attrs.get("schema_name", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8", errors="replace")
        if schema != "pelec.compact-probes":
            errors.append(f"unexpected compact schema {schema!r}")
        time = np.asarray(archive["time"], dtype=float)
        x_cm = np.asarray(archive["probes/requested_x_cm"], dtype=float)
        y_cm = np.asarray(archive["probes/requested_y_cm"], dtype=float)
        dt = np.diff(time)
        fields = tuple(archive["fields"].keys())
        units = {
            name: str(archive[f"fields/{name}"].attrs.get("unit", "unknown"))
            for name in fields
        }
        missing = {}
        for name in fields:
            dataset = archive[f"fields/{name}"]
            count = 0
            row_block = dataset.chunks[0] if dataset.chunks else min(4096, dataset.shape[0])
            for first in range(0, dataset.shape[0], row_block):
                values = np.asarray(dataset[first:first + row_block, :])
                count += int(np.count_nonzero(~np.isfinite(values)))
            missing[name] = count
        epochs = archive.get("mapping/epoch_start")
        provenance = "source/files_json" in archive
        quality = archive.attrs.get("quality_approved")
        overlaps = int(np.count_nonzero(dt <= 0.0)) if dt.size else 0
        return ProbeInventory(
            source=str(path), format="compact_hdf5", sample_count=int(time.size),
            probe_count=int(x_cm.size), fields=fields, field_units=units,
            x_min_m=float(np.min(x_cm) * 0.01) if x_cm.size else None,
            x_max_m=float(np.max(x_cm) * 0.01) if x_cm.size else None,
            y_min_m=float(np.min(y_cm) * 0.01) if y_cm.size else None,
            y_max_m=float(np.max(y_cm) * 0.01) if y_cm.size else None,
            time_min_s=float(np.min(time)) if time.size else None,
            time_max_s=float(np.max(time)) if time.size else None,
            median_timestep_s=float(np.median(dt)) if dt.size else None,
            timestep_std_s=float(np.std(dt)) if dt.size else None,
            nonfinite_time_count=int(np.count_nonzero(~np.isfinite(time))),
            missing_value_count=missing,
            restart_overlap_count=overlaps,
            mapping_epoch_count=int(epochs.shape[0]) if epochs is not None else 0,
            quality_approved=bool(quality) if quality is not None else None,
            provenance_present=provenance,
            requested_x_m=tuple(float(value * 0.01) for value in x_cm),
            requested_y_m=tuple(float(value * 0.01) for value in y_cm),
            errors=tuple(errors),
        )


def _inspect_binary(project: ResolvedProject, patterns: tuple[str, ...]) -> ProbeInventory:
    from pp_probe_store import ProbeV2Collection

    resolved = tuple(
        str(_resolve(project, Path(pattern))) if not any(char in pattern for char in "*?[")
        else str(_resolve(project, Path(pattern).parent) / Path(pattern).name)
        for pattern in patterns
    )
    with ProbeV2Collection(resolved) as collection:
        time = np.asarray(collection.time, dtype=float)
        dt = np.diff(time)
        x_cm = np.asarray(collection.header["requested_x"], dtype=float)
        y_cm = np.asarray(collection.header["requested_y"], dtype=float)
        epochs = collection.mapping_epochs(0, len(time))
        overlap = collection.overlap_report
        missing = {}
        for field in collection.field_names:
            count = 0
            for first in range(0, collection.n_probes, 32):
                values = collection.read_field(
                    field, probes=slice(first, min(first + 32, collection.n_probes))
                )
                count += int(np.count_nonzero(~np.isfinite(values)))
            missing[field] = count
        return ProbeInventory(
            source=", ".join(collection.paths), format="probe_v2",
            sample_count=int(time.size), probe_count=collection.n_probes,
            fields=collection.field_names,
            field_units=dict(zip(collection.field_names, collection.field_units)),
            x_min_m=float(np.min(x_cm) * 0.01), x_max_m=float(np.max(x_cm) * 0.01),
            y_min_m=float(np.min(y_cm) * 0.01), y_max_m=float(np.max(y_cm) * 0.01),
            time_min_s=float(time[0]), time_max_s=float(time[-1]),
            median_timestep_s=float(np.median(dt)) if dt.size else None,
            timestep_std_s=float(np.std(dt)) if dt.size else None,
            nonfinite_time_count=int(np.count_nonzero(~np.isfinite(time))),
            missing_value_count=missing,
            restart_overlap_count=int(overlap.get("duplicate_sample_count", 0)),
            mapping_epoch_count=len(epochs), quality_approved=None,
            provenance_present=True,
            requested_x_m=tuple(float(value * 0.01) for value in x_cm),
            requested_y_m=tuple(float(value * 0.01) for value in y_cm),
        )


def _inspect_probes(project: ResolvedProject) -> ProbeInventory | None:
    config = project.machine_file.inputs.probes
    if config is None:
        return None
    try:
        if config.compact_file is not None:
            return _inspect_hdf5(_resolve(project, config.compact_file))
        if config.binary_files:
            return _inspect_binary(project, config.binary_files)
        return ProbeInventory(
            source="", format="unconfigured", sample_count=0, probe_count=0,
            fields=(), field_units={}, x_min_m=None, x_max_m=None,
            y_min_m=None, y_max_m=None, time_min_s=None, time_max_s=None,
            median_timestep_s=None, timestep_std_s=None, nonfinite_time_count=0,
            missing_value_count={},
            restart_overlap_count=0, mapping_epoch_count=0, quality_approved=None,
            provenance_present=False, errors=("probe source is empty",),
        )
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return ProbeInventory(
            source=str(config.compact_file or config.binary_files), format="unknown",
            sample_count=0, probe_count=0, fields=(), field_units={},
            x_min_m=None, x_max_m=None, y_min_m=None, y_max_m=None,
            time_min_s=None, time_max_s=None, median_timestep_s=None,
            timestep_std_s=None, nonfinite_time_count=0, missing_value_count={},
            restart_overlap_count=0,
            mapping_epoch_count=0, quality_approved=None, provenance_present=False,
            errors=(str(exc),),
        )


def inspect_project(project: ResolvedProject) -> InputInventory:
    baselines = {}
    for name, config in project.machine_file.inputs.baselines.items():
        source = _resolve(project, config.source)
        baselines[name] = tuple(
            path.name for path in sorted(source.glob(f"{config.prefix}*"))
            if (path / "Header").is_file()
        ) if source.is_dir() else ()
    return InputInventory(
        plotfiles=_inspect_plotfiles(project),
        probes=_inspect_probes(project),
        comparison_archives=tuple(
            str(_resolve(project, path))
            for path in project.machine_file.inputs.comparison_archives
        ),
        baselines=baselines,
    )
