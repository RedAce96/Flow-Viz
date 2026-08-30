from __future__ import annotations

import tempfile
import unittest
import struct
from pathlib import Path

import h5py
import numpy as np
import yaml

from pelecpost.config.loader import load_project
from pelecpost.io.signals import (
    open_compact_signal_workspace,
    open_probe_v2_signal_workspace,
)
from pelecpost.runtime import run_project
from tests.test_preflight import PreflightTests


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

    def write_probe_v2(self, path: Path, samples: int = 17, probes: int = 5) -> np.ndarray:
        steps = np.arange(samples, dtype="<i8")
        times = np.arange(samples, dtype="<f8") * 1.0e-6
        x_cm = np.arange(probes, dtype="<f8")
        y_cm = np.zeros(probes, dtype="<f8")
        levels = np.zeros(probes, dtype="<i4")
        valid = np.ones(probes, dtype=np.uint8)
        pressure = np.arange(samples * probes, dtype="<f8").reshape(samples, probes)
        payload = b"".join((
            steps.tobytes(), times.tobytes(), x_cm.tobytes(), y_cm.tobytes(),
            levels.tobytes(), valid.tobytes(), pressure.tobytes(),
        ))
        with path.open("wb") as stream:
            stream.write(b"PROBES2\0")
            stream.write(struct.pack(
                "<8q", 2, 0x0102030405060708, probes, 1, 64, 1, 16, 24,
            ))
            stream.write(b"p".ljust(16, b"\0"))
            stream.write(b"dyne/cm^2".ljust(24, b"\0"))
            stream.write(x_cm.tobytes())
            stream.write(y_cm.tobytes())
            stream.write(b"PRBCHNK2")
            stream.write(struct.pack(
                "<7q2d", 0, samples, probes, 1, len(payload),
                int(steps[0]), int(steps[-1]), float(times[0]), float(times[-1]),
            ))
            stream.write(payload)
            stream.write(struct.pack("<8sqQq", b"PRBEND2\0", 0, 0, len(payload)))
        return pressure

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

    def test_probe_v2_selection_uses_same_bounded_workspace_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.write_probe_v2(root / "probe.segment0000.pbin")
            selected = np.array([4, 1])
            workspace = open_probe_v2_signal_workspace(
                (str(root / "probe.*.pbin"),), "p", selected,
                si_factor=0.1, memory_limit_gb=1.0,
                spill_threshold_bytes=32, read_block_bytes=64,
                scratch_directory=root,
            )
            spill = workspace.spill_path
            self.assertEqual(workspace.storage, "disk")
            np.testing.assert_allclose(workspace.values, source[:, selected] * 0.1)
            np.testing.assert_allclose(workspace.x_m, selected * 0.01)
            workspace.close()
            self.assertTrue(spill is not None)
            self.assertFalse(spill.exists())

    def test_probe_spectrum_runs_directly_from_probe_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = PreflightTests().project(root, [{
                "id": "spectrum", "recipe": "probe_spectrum",
                "variable": "pressure", "welch_segment_samples": 256,
            }])
            binary = root / "probe.segment0000.pbin"
            self.write_probe_v2(binary, samples=1024, probes=5)
            machine_path = root / "machine.yaml"
            machine = yaml.safe_load(machine_path.read_text())
            machine["inputs"]["probes"] = {"binary_files": [str(binary)]}
            machine_path.write_text(yaml.safe_dump(machine), encoding="utf-8")
            result = run_project(load_project(project.root))
            self.assertEqual(result.status, "completed")


if __name__ == "__main__":
    unittest.main()
