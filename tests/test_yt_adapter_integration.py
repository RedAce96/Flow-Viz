from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yt

import pp_functions_database as fields_api


FIXTURE = Path(__file__).parent / "fixtures" / "pelec_2d_plt00000"


class YtAdapterIntegrationTests(unittest.TestCase):
    def test_tracked_amrex_plotfile_is_read_by_yt_and_public_si_adapter(self):
        dataset = yt.load(FIXTURE)
        self.assertEqual(dataset.dimensionality, 2)
        self.assertEqual(tuple(dataset.domain_dimensions[:2]), (8, 6))
        self.assertIn(("boxlib", "pressure"), dataset.field_list)

        result = fields_api.load_pelec_plotfile(
            FIXTURE,
            field_names=(
                "density", "x_velocity", "y_velocity", "pressure",
                "temperature", "volume_fraction",
            ),
            convert_to_mks=True,
            derive_native_vorticity=True,
        )
        self.assertEqual(result["grid_shape"], (8, 6))
        np.testing.assert_allclose(result["x"], (np.arange(8) + 0.5) * 0.01)
        np.testing.assert_allclose(result["y"], (np.arange(6) + 0.5) * 0.01)
        i, j = 3, 4
        self.assertAlmostEqual(result["fields"]["density"][i, j], 1.11)
        self.assertAlmostEqual(result["fields"]["x_velocity"][i, j], 10.3)
        self.assertAlmostEqual(result["fields"]["y_velocity"][i, j], -0.3)
        self.assertAlmostEqual(result["fields"]["pressure"][i, j], 101100.0)
        self.assertAlmostEqual(result["fields"]["temperature"][i, j], 311.0)
        np.testing.assert_allclose(result["fields"]["vorticity"], 0.0, atol=1.0e-14)

    def test_adapter_crops_covering_grid_before_materializing_fields(self):
        result = fields_api.load_pelec_plotfile(
            FIXTURE, field_names=("pressure",), convert_to_mks=True,
            region_bounds_m=((0.02, 0.05), (0.01, 0.04)),
        )
        self.assertEqual(result["grid_shape"], (3, 3))
        np.testing.assert_allclose(result["x"], [0.025, 0.035, 0.045])
        np.testing.assert_allclose(result["y"], [0.015, 0.025, 0.035])
        self.assertEqual(result["fields"]["pressure"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
