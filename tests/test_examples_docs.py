from __future__ import annotations

import unittest
from pathlib import Path

from pelecpost.config.loader import load_project
from pelecpost.workflows import RECIPES


ROOT = Path(__file__).resolve().parents[1]


class ExampleDocumentationTests(unittest.TestCase):
    def test_all_complete_examples_validate(self):
        expected = {
            "flow-overview", "boundary-layer", "flat-plate-forces", "kernel-spectrum",
            "directional-wave", "transient-packet", "nonlinear-modal", "wedge-eb-forces",
        }
        found = {
            path.name for path in (ROOT / "examples").iterdir()
            if path.is_dir() and (path / "case.yaml").is_file()
        }
        self.assertEqual(found, expected)
        for name in sorted(found):
            with self.subTest(example=name):
                self.assertTrue(load_project(ROOT / "examples" / name).enabled_analyses)

    def test_recipe_reference_tracks_registry(self):
        reference = (ROOT / "docs/RECIPES.md").read_text(encoding="utf-8")
        for name in RECIPES:
            self.assertIn(f"`{name}`", reference)


if __name__ == "__main__":
    unittest.main()
