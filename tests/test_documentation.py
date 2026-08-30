from __future__ import annotations

import unittest
from pathlib import Path

from pelecpost.documentation import render_configuration_reference, render_recipe_reference


class DocumentationTests(unittest.TestCase):
    def test_committed_recipe_reference_matches_registry(self):
        repository = Path(__file__).resolve().parents[1]
        self.assertEqual(
            (repository / "docs/RECIPES.md").read_text(encoding="utf-8"),
            render_recipe_reference(),
        )

    def test_committed_configuration_reference_matches_pydantic_models(self):
        repository = Path(__file__).resolve().parents[1]
        reference = (repository / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
        self.assertEqual(reference, render_configuration_reference())
        for field in (
            "density_kg_m3", "frequency_max_hz", "normal_sample_distance_m",
            "memory_limit_gb", "scratch_directory",
        ):
            self.assertIn(f"`{field}`", reference)


if __name__ == "__main__":
    unittest.main()
