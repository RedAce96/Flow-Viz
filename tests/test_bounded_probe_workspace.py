from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pelecpost.io.signals import open_compact_signal_workspace


class BoundedProbeWorkspaceTests(unittest.TestCase):
    def write_archive(self, path: Path, samples: int = 257, probes: int = 9) -> np.ndarray:
        source = np.arange(samples * probes, dtype=np.float64).reshape(samples, probes)
        with h5py.File(path, "w") as archive:
            archive.create_dataset("time", data=np.arange(samples, dtype=float) * 1.0e-6)
            probe_group = archive.create_group("probes")
            probe_group.create_dataset("requested_x_cm", data=np.arange(probes, dtype=float))
            fields = archive.create_group("fields")
            fields.create_dataset("p", data=source, chunks=(min(17, samples), probes))
        return source

    def test_large_selection_spills_in_bounded_blocks_and_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "probes.h5"
            source = self.write_archive(archive)
            selected = np.array([7, 1, 5, 0])
            workspace = open_compact_signal_workspace(
                archive, "p", selected, si_factor=0.1, memory_limit_gb=1.0,
                spill_threshold_bytes=64, read_block_bytes=96,
                scratch_directory=root,
            )
            spill = workspace.spill_path
            self.assertEqual(workspace.storage, "disk")
            self.assertIsInstance(workspace.values, np.memmap)
            self.assertLessEqual(workspace.read_block_rows * selected.size * 8, 96)
            self.assertTrue(spill and spill.is_file())
            np.testing.assert_allclose(workspace.values, source[:, selected] * 0.1)
            workspace.close()
            self.assertFalse(spill.exists())
            workspace.close()

    def test_small_selection_stays_resident_without_spill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "probes.h5"
            source = self.write_archive(archive, samples=8, probes=3)
            selected = np.array([2, 0])
            workspace = open_compact_signal_workspace(
                archive, "p", selected, si_factor=1.0, memory_limit_gb=1.0,
                spill_threshold_bytes=1024,
            )
            self.assertEqual(workspace.storage, "memory")
            self.assertNotIsInstance(workspace.values, np.memmap)
            self.assertIsNone(workspace.spill_path)
            np.testing.assert_array_equal(workspace.values, source[:, selected])
            workspace.close()

    def test_duplicate_probe_indices_are_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "probes.h5"
            self.write_archive(archive)
            with self.assertRaisesRegex(ValueError, "unique"):
                open_compact_signal_workspace(
                    archive, "p", np.array([1, 1]), si_factor=1.0,
                    memory_limit_gb=1.0,
                )


if __name__ == "__main__":
    unittest.main()
