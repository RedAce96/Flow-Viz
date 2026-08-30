from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import typer

from pelecpost.config.models import AnalysesFile
from pelecpost.wizard import _analysis
from pelecpost.workflows import RECIPES


class WizardTests(unittest.TestCase):
    @staticmethod
    def prompt(text, default=None, type=None):
        values = {
            "Streamwise station [m]": 0.1,
            "Maximum profile height [m]": 0.01,
            "Wall temperature [K]": 300.0,
            "Dynamic viscosity [Pa s]": 1.0e-5,
            "Thermal conductivity [W/(m K)]": 0.02,
            "Reference chord [m]": 0.1,
            "Pulse energy per unit span [J/m]": 1.0,
            "Pulse FWHM [s]": 1.0e-6,
            "Pulse period [s]": 1.0e-5,
            "Pulse start time [s]": 1.0e-4,
            "Minimum frequency [Hz]": 1.0e3,
            "Maximum frequency [Hz]": 1.0e5,
            "Band minimum [Hz]": 1.0e3,
            "Band maximum [Hz]": 1.0e5,
            "Baseline mode (none/static/paired)": "none",
            "Baseline run directory": "/tmp/baseline-run",
            "Comparison run directory": "/tmp/comparison-run",
            "Artifact ID": "spectrum.spectral.psd",
            "Comma-separated target frequencies [Hz]": "10000,20000",
        }
        if text not in values:
            raise AssertionError(f"unexpected wizard prompt: {text}")
        return values[text]

    def test_every_recipe_builder_produces_typed_yaml_content(self):
        inventory = SimpleNamespace(
            probes=SimpleNamespace(
                median_timestep_s=1.0e-6, time_min_s=0.0, time_max_s=1.0e-3
            )
        )
        console = SimpleNamespace(print=lambda *args, **kwargs: None)
        with patch("pelecpost.wizard.typer.prompt", side_effect=self.prompt), \
             patch("pelecpost.wizard.typer.confirm", return_value=False), \
             patch("pelecpost.wizard._variable", return_value="pressure"):
            for recipe in RECIPES:
                with self.subTest(recipe=recipe):
                    generated = _analysis(recipe, inventory, console)
                    validated = AnalysesFile.model_validate({
                        "schema_version": 1, "analyses": [generated]
                    })
                    self.assertEqual(validated.analyses[0].recipe, recipe)

    def test_prompt_cancellation_propagates_without_partial_answer(self):
        inventory = SimpleNamespace(probes=None)
        console = SimpleNamespace(print=lambda *args, **kwargs: None)
        with patch("pelecpost.wizard.typer.prompt", side_effect=typer.Abort()):
            with self.assertRaises(typer.Abort):
                _analysis("boundary_layer_reference", inventory, console)


if __name__ == "__main__":
    unittest.main()
