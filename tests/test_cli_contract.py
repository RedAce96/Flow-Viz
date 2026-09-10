from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "pelec_post.py"


def command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *arguments], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )


class CliContractTests(unittest.TestCase):
    def test_all_public_commands_have_help(self):
        commands = (
            ("init",), ("configure",), ("inspect",), ("validate",), ("plan",),
            ("run",), ("recipes", "list"), ("recipes", "show"), ("report",),
            ("probes", "compact"), ("probes", "verify"), ("probes", "prune"),
        )
        for parts in commands:
            with self.subTest(command=parts):
                result = command(*parts, "--help")
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_configuration_and_preflight_rejections_exit_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            self.assertEqual(command("init", str(root)).returncode, 0)
            (root / "machine.yaml").unlink()
            missing = command("validate", str(root))
            self.assertEqual(missing.returncode, 2, missing.stderr + missing.stdout)
            self.assertIn("machine.example.yaml", missing.stdout)
            (root / "machine.example.yaml").replace(root / "machine.yaml")
            analyses = {"schema_version": 1, "analyses": [{
                "id": "spectrum", "recipe": "probe_spectrum", "probe_set_id": "default", "variable": "pressure",
            }]}
            (root / "analyses.yaml").write_text(yaml.safe_dump(analyses), encoding="utf-8")
            rejected = command("plan", str(root))
            self.assertEqual(rejected.returncode, 2, rejected.stderr + rejected.stdout)
            self.assertIn("MISSING_INPUT", rejected.stdout)

    def test_runtime_workflow_failure_exits_one_and_leaves_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            self.assertEqual(command("init", str(root)).returncode, 0)
            plot = root / "inputs" / "plt00010"
            plot.mkdir(parents=True)
            fields = ["temperature"]
            (plot / "Header").write_text(
                "\n".join(["HyperCLaw-V1.1", "1", *fields, "2", "1.0e-6", "0"]) + "\n",
                encoding="utf-8",
            )
            analyses = {"schema_version": 1, "analyses": [{
                "id": "overview", "recipe": "flow_overview", "fields": ["temperature"],
            }]}
            machine = yaml.safe_load((root / "machine.yaml").read_text())
            machine["inputs"] = {
                "plotfiles": {"source": str(root / "inputs"), "prefix": "plt"}
            }
            (root / "analyses.yaml").write_text(yaml.safe_dump(analyses), encoding="utf-8")
            (root / "machine.yaml").write_text(yaml.safe_dump(machine), encoding="utf-8")
            failed = command("run", str(root), "--name", "failure-contract")
            self.assertEqual(failed.returncode, 1, failed.stderr + failed.stdout)
            reports = list((root / "runs").glob("*/report/index.html"))
            self.assertEqual(len(reports), 1)

    def test_json_interface_is_rejected_with_migration_message(self):
        result = command("validate", "/tmp/retired-config.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("clean-break interface", result.stdout)

    def test_old_flags_are_rejected_with_clean_break_migration_message(self):
        for option in ("--config", "--output-dir", "--validation-case"):
            with self.subTest(option=option):
                result = command(option, "retired-value")
                self.assertEqual(result.returncode, 2)
                self.assertIn("retired JSON/flag", result.stdout)
                self.assertIn("interface", result.stdout)
                self.assertIn("pelec-post run PROJECT_DIR", result.stdout)


if __name__ == "__main__":
    unittest.main()
