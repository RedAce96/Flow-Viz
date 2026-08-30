#!/usr/bin/env python3
"""Thin standalone HPC launcher for `pelec-post probes` functionality."""

from pelecpost.probe_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
