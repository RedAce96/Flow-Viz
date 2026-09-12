from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from pelecpost.analysis.comparison import _Product, _validate_product_pair
from pelecpost.analysis.products import product_contract
from pelecpost.analysis.transient import estimate_packet_fit
from pelecpost.io.identity import build_input_manifest, validate_input_manifest
from pelecpost.runtime.parallel import (
    ParallelStagePlan, ParallelTask, ParallelTaskError, run_parallel_stage,
)


def _category1_sleep(payload):
    time.sleep(float(payload))
    return payload


class Category1Tests(unittest.TestCase):
    def test_packet_estimator_records_student_t_and_unresolved_direction(self):
        x = np.array([0.0, 1.0, 2.0])
        envelope = np.ones((32, 3)) * 100.0
        baseline = np.zeros((16, 3))
        supported = estimate_packet_fit(
            np.array([1.0, 1.5, 2.0]), x, envelope, baseline,
            dt_s=0.01, band_min_hz=10.0, band_max_hz=20.0,
            minimum_baseline_samples=8, minimum_packet_snr_db=6.0,
            minimum_arrival_r_squared=0.8, arrival_edge_margin_s=0.1,
            require_monotonic_arrivals=True, time_start_s=0.0, time_end_s=4.0,
        )
        self.assertEqual(supported.degrees_of_freedom, 1)
        self.assertGreater(supported.t_multiplier, 1.96)
        self.assertEqual(supported.fit_status, "supported")

        unresolved = estimate_packet_fit(
            np.array([1.0, 1.0, 1.0]), x, envelope, baseline,
            dt_s=0.01, band_min_hz=10.0, band_max_hz=20.0,
            minimum_baseline_samples=8, minimum_packet_snr_db=6.0,
            minimum_arrival_r_squared=0.8, arrival_edge_margin_s=0.1,
            require_monotonic_arrivals=True, time_start_s=0.0, time_end_s=4.0,
        )
        self.assertEqual(unresolved.fit_status, "unresolved_direction")
        self.assertIsNone(unresolved.velocity_interval_m_s)

    def test_identity_manifest_is_payload_sensitive_and_staleness_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "plt00001"
            first.mkdir()
            (first / "Header").write_text("same header", encoding="utf-8")
            (first / "payload").write_bytes(b"one")
            manifest = build_input_manifest(root, prefix="plt")
            self.assertEqual(len(manifest["entries"]), 2)
            (first / "payload").write_bytes(b"two")
            changed = build_input_manifest(root, prefix="plt")
            self.assertNotEqual(manifest["aggregate_sha256"], changed["aggregate_sha256"])
            path = root / "identity.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_input_manifest(path, expected_source=root, expected_prefix="plt")

    def test_product_contract_rejects_mixed_breaking_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {
                "frequency_hz": np.arange(3.0),
                "probe_x_m": np.arange(2.0),
                "source_power_spectrum_w_m": np.ones(3),
                "transfer": np.ones((3, 2), dtype=complex),
                "transfer_magnitude": np.ones((3, 2)),
                "transfer_phase_rad": np.zeros((3, 2)),
                "baseline": np.zeros(2),
                "valid_frequency": np.ones(3, dtype=bool),
            }
            left_path = root / "left.npz"
            right_path = root / "right.npz"
            np.savez(left_path, **arrays)
            np.savez(right_path, **arrays)
            left_contract = product_contract("pulse.transfer", left_path, units="K/(W/m)")
            right_contract = dict(left_contract, schema_version=1)
            metadata = {
                "kind": "array", "variable": "temperature", "units": "K/(W/m)",
                "provenance": {"product_contract": left_contract, "preprocessing": {}},
            }
            left = _Product("left", metadata, left_path)
            right = _Product(
                "right", {**metadata, "provenance": {
                    "product_contract": right_contract, "preprocessing": {},
                }}, right_path,
            )
            with self.assertRaisesRegex(ValueError, "incompatible product-contract versions"):
                _validate_product_pair(left, right)

    def test_parallel_task_timeout_is_reported_with_identity(self):
        plan = ParallelStagePlan(
            stage="timeout", requested_workers=2, effective_workers=2,
            task_count=1, available_cpus=2, parent_resident_gb=0.0,
            per_worker_peak_gb=0.1, estimated_concurrent_peak_gb=0.2,
            limiting_reasons=(),
        )
        with self.assertRaises(ParallelTaskError) as raised:
            run_parallel_stage(
                "timeout", [ParallelTask(0, "slow", 0.5)],
                _category1_sleep, plan, task_timeout_s=0.1,
                worker_start_guard_s=5.0, heartbeat_s=0.02,
            )
        self.assertEqual(raised.exception.result.task_id, "slow")
        self.assertEqual(raised.exception.result.error_type, "TaskTimeoutError")


if __name__ == "__main__":
    unittest.main()
