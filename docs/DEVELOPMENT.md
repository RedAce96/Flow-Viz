# Development checks

Python 3.11 and 3.12 are supported. Install the matching pinned environment with:

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -c requirements/constraints-py311.txt -e '.[dev]'
```

Regenerate constraints after an intentional dependency change with:

```bash
python -m pip install pip-tools
python -m piptools compile --strip-extras --output-file requirements/constraints-py311.txt pyproject.toml
python -m piptools compile --strip-extras --output-file requirements/constraints-py312.txt pyproject.toml
```

Run the same checks used in CI:

```bash
python -m pytest -q tests test_engineering.py
ruff check --select F401 pelecpost tests
mypy pelecpost
python -m compileall -q pelecpost tests pp_functions_database.py
git diff --check
```

The clean-install CI job builds a wheel, installs it in a fresh environment
outside the checkout, and exercises CLI help, project initialization,
validation, planning, and a probe-only execution.
