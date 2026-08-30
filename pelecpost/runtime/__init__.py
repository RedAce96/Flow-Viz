"""Isolated execution, artifact registration, and report generation."""

from .run import RunResult, run_project
from .report import generate_report

__all__ = ["RunResult", "generate_report", "run_project"]
