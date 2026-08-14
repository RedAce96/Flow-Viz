"""Chunked probe-v2 and compact HDF5 storage helpers.

The solver's ``.pbin`` files are append-friendly acquisition files.  The HDF5
file produced by :mod:`compact_probes` is a self-contained, read-optimized
archive.  This module deliberately has no dependency on ``pelec_post`` so the
command-line compactor and the post-processing driver share one implementation.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


SCHEMA_NAME = "pelec.compact-probes"
SCHEMA_VERSION = 1
FILE_MAGIC = b"PROBES2\0"
CHUNK_MAGIC = b"PRBCHNK2"
FOOTER_MAGIC = b"PRBEND2\0"
ENDIAN_MARKER = 0x0102030405060708
ASCII_COLUMNS = {1: "rho", 2: "u", 3: "p", 4: "T"}


def expand_paths(paths):
    """Expand input globs, preserve order, and return canonical paths."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    expanded = []
    for value in paths:
        matches = sorted(glob.glob(os.path.expanduser(str(value))))
        if not matches:
            raise FileNotFoundError(f"No probe files match {value!r}")
        expanded.extend(str(Path(match).resolve()) for match in matches)
    return list(dict.fromkeys(expanded))


def sha256_file(path, block_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fixed_strings(stream, count, width):
    values = []
    for _ in range(count):
        raw = stream.read(width)
        if len(raw) != width:
            raise EOFError("Incomplete fixed-string probe metadata")
        values.append(raw.split(b"\0", 1)[0].decode("ascii"))
    return values


def _read_header(stream, path):
    if stream.read(8) != FILE_MAGIC:
        raise ValueError(
            f"{path} is not a probe-v2 file; compacting legacy probe-v1 "
            "files is intentionally unsupported because they do not contain "
            "restart-safe steps or chunk mapping metadata"
        )
    raw = stream.read(64)
    if len(raw) != 64:
        raise EOFError(f"Incomplete probe-v2 header in {path}")
    values = struct.unpack("<8q", raw)
    endian = "<"
    if values[1] != ENDIAN_MARKER:
        values = struct.unpack(">8q", raw)
        endian = ">"
    if values[0] != 2 or values[1] != ENDIAN_MARKER:
        raise ValueError(f"Invalid probe-v2 header in {path}")
    version, _, n_probes, n_fields, capacity, probe_int, nw, uw = values
    names = _fixed_strings(stream, n_fields, nw)
    units = _fixed_strings(stream, n_fields, uw)
    dtype = np.dtype(f"{endian}f8")
    requested_x = np.frombuffer(
        stream.read(8 * n_probes), dtype=dtype
    ).astype(float)
    requested_y = np.frombuffer(
        stream.read(8 * n_probes), dtype=dtype
    ).astype(float)
    if requested_x.size != n_probes or requested_y.size != n_probes:
        raise EOFError(f"Incomplete probe coordinates in {path}")
    return {
        "version": version,
        "endian": endian,
        "n_probes": n_probes,
        "n_fields": n_fields,
        "chunk_capacity": capacity,
        "probe_interval": probe_int,
        "field_names": names,
        "field_units": units,
        "requested_x": requested_x,
        "requested_y": requested_y,
        "data_offset": stream.tell(),
    }


@dataclass(frozen=True)
class Chunk:
    source: int
    index: int
    n_samples: int
    payload_offset: int
    payload_bytes: int
    first_step: int
    last_step: int
    first_time: float
    last_time: float
    checksum: int


def _scan_chunks(stream, header, source, path):
    chunks = []
    endian = header["endian"]
    stream.seek(header["data_offset"])
    while True:
        offset = stream.tell()
        magic = stream.read(8)
        if not magic:
            break
        if len(magic) < 8:
            break  # crash-truncated tail
        if magic != CHUNK_MAGIC:
            raise ValueError(f"Bad chunk marker in {path} at byte {offset}")
        raw = stream.read(72)
        if len(raw) != 72:
            break
        values = struct.unpack(f"{endian}7q2d", raw)
        ci, ns, npb, nf, payload_bytes, fs, ls, ft, lt = values
        if (ns < 1 or npb != header["n_probes"]
                or nf != header["n_fields"] or payload_bytes < 0):
            raise ValueError(f"Invalid chunk header in {path} at byte {offset}")
        payload_offset = stream.tell()
        stream.seek(payload_offset + payload_bytes)
        footer = stream.read(32)
        if len(footer) != 32:
            break
        fm, fi, checksum, fb = struct.unpack(f"{endian}8sqQq", footer)
        if fm != FOOTER_MAGIC or fi != ci or fb != payload_bytes:
            raise ValueError(f"Invalid chunk footer in {path} at byte {offset}")
        chunks.append(Chunk(
            source, ci, ns, payload_offset, payload_bytes,
            fs, ls, ft, lt, checksum,
        ))
    return chunks


class ProbeV2Collection:
    """Indexed collection of restart-segmented probe-v2 files.

    The collection stores only the global sample index and compact mapping
    metadata in memory.  Field arrays are read one chunk at a time.
    """

    def __init__(self, paths, dedup_tol=1.0e-12,
                 overlap_policy="latest_segment"):
        self.paths = expand_paths(paths)
        self.dedup_tol = float(dedup_tol)
        self.overlap_policy = str(overlap_policy).lower()
        if self.overlap_policy not in ("latest_segment", "error"):
            raise ValueError("overlap_policy must be latest_segment or error")
        self._streams = []
        self.headers = []
        self.chunks = []
        reference = None
        try:
            for source, path in enumerate(self.paths):
                stream = open(path, "rb")
                self._streams.append(stream)
                header = _read_header(stream, path)
                signature = (
                    header["n_probes"], tuple(header["field_names"]),
                    tuple(header["field_units"]),
                    header["requested_x"].tobytes(),
                    header["requested_y"].tobytes(),
                )
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise ValueError(f"Probe layout mismatch in {path}")
                self.headers.append(header)
                self.chunks.extend(_scan_chunks(stream, header, source, path))
            if not self.chunks:
                raise ValueError("No complete probe chunks were found")
            self._build_index()
        except Exception:
            self.close()
            raise

    @property
    def header(self):
        return self.headers[0]

    @property
    def n_probes(self):
        return int(self.header["n_probes"])

    @property
    def field_names(self):
        return tuple(self.header["field_names"])

    @property
    def field_units(self):
        return tuple(self.header["field_units"])

    def close(self):
        for stream in getattr(self, "_streams", []):
            try:
                stream.close()
            except Exception:
                pass
        self._streams = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _chunk_arrays(self, chunk, include_mapping=False):
        header = self.headers[chunk.source]
        stream = self._streams[chunk.source]
        endian = header["endian"]
        ns, npb = chunk.n_samples, self.n_probes
        stream.seek(chunk.payload_offset)
        steps = np.frombuffer(
            stream.read(8 * ns), dtype=f"{endian}i8"
        ).astype(np.int64)
        times = np.frombuffer(
            stream.read(8 * ns), dtype=f"{endian}f8"
        ).astype(float)
        result = [steps, times]
        if include_mapping:
            sample_x = np.frombuffer(
                stream.read(8 * npb), dtype=f"{endian}f8"
            ).astype(float)
            sample_y = np.frombuffer(
                stream.read(8 * npb), dtype=f"{endian}f8"
            ).astype(float)
            level = np.frombuffer(
                stream.read(4 * npb), dtype=f"{endian}i4"
            ).astype(np.int32)
            valid = np.frombuffer(
                stream.read(npb), dtype=np.uint8
            ).astype(bool)
            result.extend((sample_x, sample_y, level, valid))
        return tuple(result)

    def _build_index(self):
        rows = []
        mappings = []
        mapping_lookup = {}
        for chunk_id, chunk in enumerate(self.chunks):
            arrays = self._chunk_arrays(chunk, include_mapping=True)
            steps, times, sx, sy, level, valid = arrays
            digest = hashlib.blake2b(digest_size=16)
            for array in (sx, sy, level, valid):
                digest.update(np.ascontiguousarray(array).tobytes())
            key = digest.digest()
            mapping_id = mapping_lookup.get(key)
            if mapping_id is None:
                mapping_id = len(mappings)
                mapping_lookup[key] = mapping_id
                mappings.append((sx, sy, level, valid))
            for local, (step, time) in enumerate(zip(steps, times)):
                rows.append((float(time), int(step), chunk.source,
                             chunk_id, local, mapping_id))
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        selected = []
        duplicates = 0
        replaced = 0
        for row in rows:
            if selected and (
                    row[1] == selected[-1][1]
                    or abs(row[0] - selected[-1][0]) < self.dedup_tol):
                duplicates += 1
                if row[2] == selected[-1][2]:
                    raise ValueError(
                        "Conflicting duplicate samples within one probe segment"
                    )
                if self.overlap_policy == "error":
                    raise ValueError("Duplicate samples found across restart segments")
                selected[-1] = row
                replaced += 1
            else:
                selected.append(row)
        self.time = np.asarray([row[0] for row in selected], dtype=float)
        self.step = np.asarray([row[1] for row in selected], dtype=np.int64)
        self._chunk_id = np.asarray([row[3] for row in selected], dtype=np.int32)
        self._local_row = np.asarray([row[4] for row in selected], dtype=np.int32)
        self.mapping_id = np.asarray([row[5] for row in selected], dtype=np.int32)
        self.mappings = mappings
        self.overlap_report = {
            "duplicate_sample_count": duplicates,
            "replaced_by_later_segment_count": replaced,
            "policy": self.overlap_policy,
        }

    def field_index(self, field):
        if field not in self.field_names:
            raise KeyError(f"Unknown probe field {field!r}; have {self.field_names}")
        return self.field_names.index(field)

    def _read_chunk_field(self, chunk, field_index):
        header = self.headers[chunk.source]
        stream = self._streams[chunk.source]
        ns, npb = chunk.n_samples, self.n_probes
        metadata_bytes = 16 * ns + 21 * npb
        field_bytes = 8 * ns * npb
        offset = chunk.payload_offset + metadata_bytes + field_index * field_bytes
        stream.seek(offset)
        raw = stream.read(field_bytes)
        if len(raw) != field_bytes:
            raise EOFError(
                f"Short {self.field_names[field_index]} block in "
                f"{self.paths[chunk.source]}"
            )
        return np.frombuffer(
            raw, dtype=f"{header['endian']}f8"
        ).reshape(ns, npb).astype(float)

    def read_field(self, field, start=0, stop=None, probes=slice(None)):
        """Read a global time interval and probe selection as float64."""
        stop = len(self.time) if stop is None else int(stop)
        start = int(start)
        probe_ids = np.arange(self.n_probes)[probes]
        probe_ids = np.atleast_1d(probe_ids).astype(int)
        output = np.empty((max(0, stop - start), probe_ids.size), dtype=float)
        field_index = self.field_index(field)
        wanted_chunks = np.unique(self._chunk_id[start:stop])
        for chunk_id in wanted_chunks:
            positions = np.flatnonzero(
                self._chunk_id[start:stop] == chunk_id
            )
            source_rows = self._local_row[start:stop][positions]
            values = self._read_chunk_field(self.chunks[int(chunk_id)], field_index)
            output[positions, :] = values[np.ix_(source_rows, probe_ids)]
        return output

    def write_field(self, dataset, field, start, stop, block_rows=2048):
        """Copy an interval in bounded contiguous blocks.

        Probe mappings can change frequently, producing very short source
        chunks. Writing those rows individually repeatedly recompresses the
        same HDF5 chunks. Contiguous blocks keep memory bounded while making
        compression and parallel-filesystem I/O efficient.
        """
        for destination_start in range(0, stop - start, int(block_rows)):
            destination_stop = min(
                destination_start + int(block_rows), stop - start
            )
            values = self.read_field(
                field, start + destination_start, start + destination_stop
            )
            dataset[destination_start:destination_stop, :] = values

    def compare_field(self, dataset, field, start, stop, block_rows=2048):
        """Return whether an HDF5 field is exactly the selected source data."""
        for destination_start in range(0, stop - start, int(block_rows)):
            destination_stop = min(
                destination_start + int(block_rows), stop - start
            )
            values = self.read_field(
                field, start + destination_start, start + destination_stop
            )
            if not np.array_equal(
                    dataset[destination_start:destination_stop, :], values,
                    equal_nan=True):
                return False
        return True

    def mapping_epochs(self, start=0, stop=None):
        stop = len(self.time) if stop is None else int(stop)
        ids = self.mapping_id[start:stop]
        if ids.size == 0:
            return []
        starts = np.r_[0, np.flatnonzero(np.diff(ids) != 0) + 1]
        stops = np.r_[starts[1:], ids.size]
        return [(int(a), int(b), int(ids[a])) for a, b in zip(starts, stops)]


class HDF5ProbeStore:
    """Lazy, list-compatible view of a compact probe HDF5 file."""

    def __init__(self, path, field="p", nt_skip=0, max_probes=None,
                 probe_ids=None):
        self.path = str(Path(path).resolve())
        self._file = h5py.File(self.path, "r")
        if self._file.attrs.get("schema_name") != SCHEMA_NAME:
            self.close()
            raise ValueError(f"Not a {SCHEMA_NAME} file: {path}")
        if int(self._file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            self.close()
            raise ValueError(f"Unsupported compact probe schema in {path}")
        if field not in self._file["fields"]:
            self.close()
            raise KeyError(f"Field {field!r} is absent from {path}")
        self.field = field
        self.nt_skip = max(0, int(nt_skip))
        total_probes = self._file["probes/requested_x_cm"].shape[0]
        if probe_ids is None:
            limit = total_probes if max_probes is None else min(
                int(max_probes), total_probes
            )
            self.probe_ids = np.arange(limit, dtype=int)
        else:
            self.probe_ids = np.asarray(probe_ids, dtype=int)
        self._time = None
        self._step = None

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        return int(self.probe_ids.size)

    def __bool__(self):
        return len(self) > 0 and self.n_samples > 0

    @property
    def n_samples(self):
        return max(0, self._file["time"].shape[0] - self.nt_skip)

    @property
    def time(self):
        if self._time is None:
            self._time = self._file["time"][self.nt_skip:].astype(float)
        return self._time

    @property
    def step(self):
        if self._step is None:
            self._step = self._file["step"][self.nt_skip:].astype(np.int64)
        return self._step

    @property
    def dt(self):
        return float(np.median(np.diff(self.time))) if self.time.size > 1 else 0.0

    @property
    def field_names(self):
        return tuple(self._file["fields"].keys())

    def _probe_values(self, name):
        return self._file[f"probes/{name}"][self.probe_ids]

    @property
    def requested_x(self):
        return self._probe_values("requested_x_cm")

    @property
    def requested_y(self):
        return self._probe_values("requested_y_cm")

    @property
    def x(self):
        name = "nominal_x_cm" if "nominal_x_cm" in self._file["probes"] else "requested_x_cm"
        return self._probe_values(name)

    @property
    def y(self):
        name = "nominal_y_cm" if "nominal_y_cm" in self._file["probes"] else "requested_y_cm"
        return self._probe_values(name)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        local = int(index)
        if local < 0:
            local += len(self)
        source = int(self.probe_ids[local])
        return {
            "x": float(self.x[local]), "y": float(self.y[local]),
            "x_req": float(self.requested_x[local]),
            "y_req": float(self.requested_y[local]),
            "time": self.time,
            "signal": self.read(probes=slice(local, local + 1))[:, 0],
            "dt": self.dt,
            "filename": f"compact_probe_{source:04d}.h5",
            "source_probe_index": source,
        }

    def select_probes(self, local_ids):
        local_ids = np.asarray(local_ids, dtype=int)
        return HDF5ProbeStore(
            self.path, field=self.field, nt_skip=self.nt_skip,
            probe_ids=self.probe_ids[local_ids],
        )

    def read(self, field=None, time_slice=slice(None), probes=slice(None)):
        """Read one field slice; probe selectors use local store indices."""
        field = self.field if field is None else field
        dataset = self._file[f"fields/{field}"]
        selected = self.probe_ids[probes]
        selected = np.atleast_1d(selected).astype(int)
        start, stop, stride = time_slice.indices(self.n_samples)
        source_time = slice(self.nt_skip + start, self.nt_skip + stop, stride)
        if selected.size == 0:
            return np.empty((len(range(start, stop, stride)), 0), dtype=float)
        # h5py requires unique increasing fancy indices. Read that canonical
        # selection and restore the caller's requested order/duplicates.
        unique, inverse = np.unique(selected, return_inverse=True)
        values = dataset[source_time, unique].astype(float)
        return values[:, inverse]

    def iter_probe_batches(self, batch_size, field=None):
        for first in range(0, len(self), int(batch_size)):
            last = min(first + int(batch_size), len(self))
            yield first, last, self.read(
                field=field, probes=slice(first, last)
            )

    def complete_probe_mask(self, batch_size=64):
        complete = np.ones(len(self), dtype=bool)
        for first, last, values in self.iter_probe_batches(batch_size):
            complete[first:last] = np.all(np.isfinite(values), axis=0)
        return complete

    def field_view(self, field=None):
        """Return a lazy two-dimensional matrix view for compatibility."""
        return HDF5FieldView(self, self.field if field is None else field)

    def source_metadata(self):
        raw = self._file["source/files_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)


class HDF5FieldView:
    """Minimal NumPy-like ``(time, probe)`` view backed by HDF5 slices."""

    def __init__(self, store, field):
        self.store = store
        self.field = field
        self.shape = (store.n_samples, len(store))
        self.ndim = 2
        self.dtype = np.dtype(np.float64)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        time_selector, probe_selector = key
        scalar_probe = isinstance(probe_selector, (int, np.integer))
        if isinstance(time_selector, (int, np.integer)):
            row = self.store.read(
                self.field,
                time_slice=slice(int(time_selector), int(time_selector) + 1),
                probes=probe_selector,
            )
            value = row[0]
            return value[0] if scalar_probe else value
        if not isinstance(time_selector, slice):
            # Rare compatibility path; avoid silently returning reordered
            # data through h5py's constrained fancy indexing.
            return np.asarray(self)[key]
        values = self.store.read(
            self.field, time_slice=time_selector, probes=probe_selector
        )
        return values[:, 0] if scalar_probe else values

    def __array__(self, dtype=None, copy=None):
        values = self.store.read(self.field)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        elif copy:
            values = values.copy()
        return values


def hdf5_chunks(n_time, n_probes, target_bytes=2 * 1024 * 1024):
    probe_chunk = max(1, min(128, n_probes))
    time_chunk = max(1, min(n_time, target_bytes // (8 * probe_chunk)))
    return time_chunk, probe_chunk


def create_numeric_dataset(group, name, shape, dtype, chunks=None):
    kwargs = {}
    if all(int(value) > 0 for value in shape):
        kwargs.update({
            "chunks": chunks,
            "compression": "gzip", "compression_opts": 4,
            "shuffle": True, "fletcher32": True,
        })
    return group.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
