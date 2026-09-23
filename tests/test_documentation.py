from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pelecpost.config.loader import write_project_schema
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

    def test_all_committed_project_schemas_match_models(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "project.schema.json"
            write_project_schema(generated)
            expected = generated.read_text(encoding="utf-8")
        schema_paths = sorted(repository.glob("examples/*/project.schema.json"))
        schema_paths.extend(
            sorted(repository.glob("verification_outputs/*/project.schema.json"))
        )
        self.assertEqual(len(schema_paths), 10)
        for path in schema_paths:
            with self.subTest(path=path.relative_to(repository)):
                self.assertEqual(path.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
