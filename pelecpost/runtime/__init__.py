"""Isolated execution, artifact registration, and report generation.

Imports stay lazy so numerical executor modules can depend on the lightweight
runtime context without creating a package-initialization cycle.
"""


def run_project(*args, **kwargs):
    from .run import run_project as implementation

    return implementation(*args, **kwargs)


def generate_report(*args, **kwargs):
    from .report import generate_report as implementation

    return implementation(*args, **kwargs)


__all__ = ["generate_report", "run_project"]
