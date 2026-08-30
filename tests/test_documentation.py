from __future__ import annotations

import unittest
from pathlib import Path

from pelecpost.documentation import render_recipe_reference


class DocumentationTests(unittest.TestCase):
    def test_committed_recipe_reference_matches_registry(self):
        repository = Path(__file__).resolve().parents[1]
        self.assertEqual(
            (repository / "docs/RECIPES.md").read_text(encoding="utf-8"),
            render_recipe_reference(),
        )


if __name__ == "__main__":
    unittest.main()
