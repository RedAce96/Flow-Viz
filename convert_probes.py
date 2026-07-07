#!/usr/bin/env python3
"""
convert_probes.py

Convert the single binary probe file produced by probes.cpp into per-probe
ASCII files (one per probe), matching the format of the original per-probe
.dat files.

Binary file format
------------------
Header:
  int64_t   magic       = 0x005345424F525050  ("PROBES\0\0")
  int64_t   n_probes
  int64_t   n_fields    = 7
  double    probe_x[n_probes]
  double    probe_y[n_probes]

Per-timestep record (repeated for each sample):
  double    time
  double    data[n_probes][7]
      7 fields:  rho, u, p, T, x_sample, y_sample, level

Usage
-----
  python convert_probes.py [--input PROBES.bin] [--output DIR]

Defaults:
  --input   probes/flow-probe.bin
  --output  probes/flow-probe     (creates per-probe .dat files here)
"""

import argparse
import os
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------
def read_binary_header(f):
    """Read and validate the binary header; return (n_probes, probe_x, probe_y)."""
    magic = struct.unpack('<q', f.read(8))[0]
    expected = 0x005345424F525050  # "PROBES\0\0"
    if magic != expected:
        # Try big-endian as fallback
        magic = struct.unpack('>q', f.read(8))[0]
        f.seek(-8, os.SEEK_CUR)
        if magic != expected:
            raise ValueError(
                f"Bad magic number: 0x{magic:016x}. "
                "Not a valid probe binary file."
            )
        endian = '>'
    else:
        endian = '<'

    n_probes = struct.unpack(f'{endian}q', f.read(8))[0]
    n_fields = struct.unpack(f'{endian}q', f.read(8))[0]

    if n_fields != 7:
        raise ValueError(
            f"Expected 7 fields per probe, got {n_fields}. "
            "File format mismatch."
        )

    probe_x = list(struct.unpack(f'{endian}{n_probes}d', f.read(8 * n_probes)))
    probe_y = list(struct.unpack(f'{endian}{n_probes}d', f.read(8 * n_probes)))

    return endian, n_probes, n_fields, probe_x, probe_y


def read_timestep_record(f, endian, n_probes, n_fields):
    """Read one timestep record; return (time, data_array) or None at EOF."""
    raw = f.read(8)  # time
    if len(raw) < 8:
        return None
    time = struct.unpack(f'{endian}d', raw)[0]

    raw_data = f.read(8 * n_probes * n_fields)
    if len(raw_data) < 8 * n_probes * n_fields:
        raise EOFError("Unexpected end of file in timestep record")

    data = struct.unpack(
        f'{endian}{n_probes * n_fields}d', raw_data
    )
    return time, data


# ---------------------------------------------------------------------------
# ASCII writer
# ---------------------------------------------------------------------------
def write_probe_file(path, probe_id, probe_x, probe_y, samples):
    """
    Write one per-probe ASCII file.

    samples: list of (time, rho, u, p, T, x_sample, y_sample, level)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        f.write("# Flat-plate probe history (CGS)\n")
        f.write(f"# Requested location: x={probe_x} cm, y={probe_y} cm\n")
        f.write("# Sampling rule: nearest cell center on finest available AMR level\n")
        f.write(
            "# Columns: time[s] rho[g/cm^3] u[cm/s] p[dyne/cm^2] "
            "T[K] x_sample[cm] y_sample[cm] level\n"
        )

        for s in samples:
            time, rho, uvel, pres, temp, xs, ys, lev = s
            f.write(
                f"{time:.17g} {rho:.17g} {uvel:.17g} {pres:.17g} "
                f"{temp:.17g} {xs:.17g} {ys:.17g} {lev:.0f}\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert binary probe file to per-probe ASCII files."
    )
    parser.add_argument(
        '--input',
        nargs='+',
        default=['probes/flow-probe.bin'],
        help='Path(s) to binary probe file(s). When multiple files are given '
             '(e.g. from a restarted simulation), they are merged in time '
             'order (default: probes/flow-probe.bin)',
    )
    parser.add_argument(
        '--output',
        default='probes/flow-probe',
        help='Output directory for per-probe .dat files (default: probes/flow-probe)',
    )
    parser.add_argument(
        '--skip-warnings',
        action='store_true',
        help='Suppress warnings about probes with all-zero data',
    )
    parser.add_argument(
        '--dedup-tol',
        type=float,
        default=1.0e-12,
        help='Time tolerance for duplicate detection when merging multiple '
             'files (default: 1e-12). Set to a negative value to disable.',
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output)

    for p in input_paths:
        if not p.is_file():
            print(f"Error: binary probe file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # ---- Read all binary files ----
    all_records = []
    ref_header = None  # (n_probes, n_fields, probe_x, probe_y)

    for idx, input_path in enumerate(input_paths):
        print(f"Reading binary probe file: {input_path}")
        with open(input_path, 'rb') as f:
            endian, n_probes, n_fields, probe_x, probe_y = read_binary_header(f)

        if idx == 0:
            ref_header = (n_probes, n_fields, probe_x, probe_y)
            print(f"  n_probes = {n_probes}")
            print(f"  n_fields = {n_fields}")
        else:
            # Validate header compatibility
            (rn, rf, rx, ry) = ref_header
            if n_probes != rn or n_fields != rf or probe_x != rx or probe_y != ry:
                print(f"Error: header mismatch in {input_path} -- probe layout "
                      "differs from first file.", file=sys.stderr)
                sys.exit(1)

        print(f"  file size = {os.path.getsize(input_path)} bytes")

        # Read all timesteps from this file
        records = []
        with open(input_path, 'rb') as f:
            read_binary_header(f)  # skip header
            while True:
                result = read_timestep_record(f, endian, n_probes, n_fields)
                if result is None:
                    break
                records.append(result)

        print(f"  n_timesteps = {len(records)}")
        all_records.extend(records)

    n_timesteps_total = len(all_records)
    print(f"  total n_timesteps (before dedup) = {n_timesteps_total}")
    print()

    if n_timesteps_total == 0:
        print("No timestep records found. Nothing to write.")
        sys.exit(0)

    # ---- Sort and deduplicate by time ----
    all_records.sort(key=lambda r: r[0])

    if args.dedup_tol >= 0:
        deduped = []
        n_dup = 0
        for rec in all_records:
            if deduped and (rec[0] - deduped[-1][0]) < args.dedup_tol:
                n_dup += 1
                continue
            deduped.append(rec)
        all_records = deduped
        if n_dup > 0:
            print(f"  Removed {n_dup} duplicate timestep(s) (tol={args.dedup_tol})")
    else:
        print("  Deduplication disabled")

    n_timesteps = len(all_records)
    print(f"  n_timesteps (after dedup) = {n_timesteps}")
    print()

    # ---- Transpose: group by probe ----
    # For each probe, collect (time, rho, u, p, T, x, y, level)
    probe_samples = [[] for _ in range(n_probes)]
    for time, data in all_records:
        for ip in range(n_probes):
            base = ip * n_fields
            probe_samples[ip].append((
                time,
                data[base + 0],  # rho
                data[base + 1],  # u
                data[base + 2],  # p
                data[base + 3],  # T
                data[base + 4],  # x_sample
                data[base + 5],  # y_sample
                data[base + 6],  # level
            ))

    # ---- Write per-probe ASCII files ----
    print(f"Writing per-probe ASCII files to: {output_dir}/")
    n_warn = 0
    for ip in range(n_probes):
        samples = probe_samples[ip]
        # Check for all-zero data (probe never found)
        all_zero = all(abs(s[1]) < 1e-30 for s in samples)
        if all_zero and not args.skip_warnings:
            print(f"  WARNING: probe {ip} (x={probe_x[ip]}, y={probe_y[ip]}) "
                  "has all-zero data (was never found on any AMR level)")
            n_warn += 1

        filename = output_dir / f"flow_probe_{ip:03d}.dat"
        write_probe_file(
            str(filename), ip, probe_x[ip], probe_y[ip], samples
        )

    print(f"  Wrote {n_probes} files ({n_timesteps} samples each)")
    if n_warn > 0:
        print(f"  Warnings: {n_warn} probe(s) with all-zero data")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
