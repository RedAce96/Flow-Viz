from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import pp_functions_database as reviewed_nonlinear
import pp_modal_database as reviewed_modal

from pelecpost.runtime.parallel import (
    ParallelStagePlan,
    ParallelTask,
    ParallelTaskError,
    plan_parallel_stage,
    run_parallel_stage,
    stage_readonly_array,
)


def _double(payload):
    return payload * 2


def _fail_on_two(payload):
    if payload == 2:
        raise ValueError("synthetic task failure")
    return payload


def _sleep_and_return(payload):
    time.sleep(float(payload[1]))
    return payload[0]


def _kill_worker(_payload):
    os._exit(17)


class ParallelTests(unittest.TestCase):
    def test_memory_planner_uses_safe_upper_bound(self):
        plan = plan_parallel_stage(
            "contours", requested_workers=5, task_count=20,
            memory_limit_gb=24.0, parent_resident_gb=0.5,
            per_worker_peak_gb=2.0,
        )
        self.assertLessEqual(plan.effective_workers, 5)
        self.assertAlmostEqual(plan.estimated_concurrent_peak_gb, 0.5 + 2.0 * plan.effective_workers)
        self.assertNotIn("memory_limit", plan.limiting_reasons)

    def test_memory_planner_rejects_one_worker_that_cannot_fit(self):
        with self.assertRaisesRegex(RuntimeError, "cannot fit one worker"):
            plan_parallel_stage(
                "contours", requested_workers=5, task_count=2,
                memory_limit_gb=2.0, parent_resident_gb=1.0,
                per_worker_peak_gb=1.0,
            )

    def test_cpu_allocation_limits_effective_workers(self):
        with patch("pelecpost.runtime.parallel._cpu_count", return_value=2):
            plan = plan_parallel_stage(
                "contours", requested_workers=5, task_count=20,
                memory_limit_gb=24.0, parent_resident_gb=0.5,
                per_worker_peak_gb=0.5,
            )
        self.assertEqual(plan.effective_workers, 2)
        self.assertIn("available_cpus", plan.limiting_reasons)

    def test_serial_execution_returns_ordered_results_and_events(self):
        plan = ParallelStagePlan(
            stage="serial", requested_workers=1, effective_workers=1,
            task_count=3, available_cpus=1, parent_resident_gb=0.0,
            per_worker_peak_gb=0.0, estimated_concurrent_peak_gb=0.0,
            limiting_reasons=(),
        )
        events = []
        results = run_parallel_stage(
            "serial", [ParallelTask(i, f"task-{i}", i) for i in range(3)],
            _double, plan, event_callback=lambda event, details: events.append((event, details)),
        )
        self.assertEqual([item.value for item in results], [0, 2, 4])
        self.assertEqual(
            [item[0] for item in events if item[0] == "parallel-task-start"],
            ["parallel-task-start"] * 3,
        )

    def test_spawn_execution_returns_input_order(self):
        plan = ParallelStagePlan(
            stage="spawn", requested_workers=2, effective_workers=2,
            task_count=3, available_cpus=2, parent_resident_gb=0.0,
            per_worker_peak_gb=0.1, estimated_concurrent_peak_gb=0.2,
            limiting_reasons=(),
        )
        results = run_parallel_stage(
            "spawn",
            [
                ParallelTask(0, "slow", (0, 0.08)),
                ParallelTask(1, "fast", (1, 0.01)),
                ParallelTask(2, "medium", (2, 0.03)),
            ],
            _sleep_and_return, plan, heartbeat_s=1.0,
        )
        self.assertEqual([item.value for item in results], [0, 1, 2])
        self.assertTrue(all(item.succeeded for item in results))

    def test_worker_failure_contains_task_identity_and_remote_traceback(self):
        plan = ParallelStagePlan(
            stage="failure", requested_workers=1, effective_workers=1,
            task_count=3, available_cpus=1, parent_resident_gb=0.0,
            per_worker_peak_gb=0.0, estimated_concurrent_peak_gb=0.0,
            limiting_reasons=(),
        )
        with self.assertRaises(ParallelTaskError) as raised:
            run_parallel_stage(
                "failure", [ParallelTask(i, f"task-{i}", i) for i in range(3)],
                _fail_on_two, plan,
            )
        self.assertEqual(raised.exception.result.task_id, "task-2")
        self.assertIn("synthetic task failure", str(raised.exception))
        self.assertIn("ValueError", raised.exception.result.remote_traceback)

    def test_abrupt_worker_exit_terminates_the_stage(self):
        plan = ParallelStagePlan(
            stage="worker-death", requested_workers=2, effective_workers=2,
            task_count=1, available_cpus=2, parent_resident_gb=0.0,
            per_worker_peak_gb=0.1, estimated_concurrent_peak_gb=0.2,
            limiting_reasons=(),
        )
        with self.assertRaises(ParallelTaskError) as raised:
            run_parallel_stage(
                "worker-death", [ParallelTask(0, "killed-task", None)],
                _kill_worker, plan, heartbeat_s=0.05,
            )
        self.assertEqual(raised.exception.result.task_id, "killed-task")
        self.assertEqual(raised.exception.result.error_type, "WorkerDiedError")

    def test_readonly_array_staging_is_reopenable_and_cleanable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.arange(12, dtype=float).reshape(3, 4)
            spec, cleanup = stage_readonly_array(source, directory, prefix="test-")
            self.assertEqual(spec.shape, (3, 4))
            self.assertEqual(spec.dtype, "float64")
            reopened = np.load(spec.path, mmap_mode="r")
            np.testing.assert_array_equal(reopened, source)
            self.assertFalse(reopened.flags.writeable)
            cleanup()
            self.assertFalse(Path(spec.path).exists())
            cleanup()

    def test_indexed_surrogate_batches_match_one_serial_stream(self):
        signal = np.sin(np.linspace(0.0, 80.0 * np.pi, 512))
        kwargs = {
            "fs": 1000.0, "target_freqs": np.array([50.0, 100.0]),
            "nperseg": 32, "noverlap": 16, "n_surrogates": 59,
            "fdr_alpha": 0.05, "minimum_independent_segments": 2,
        }
        prepared = reviewed_nonlinear.prepare_surrogate_triad_problem(signal, **kwargs)
        serial = reviewed_nonlinear.compute_surrogate_triad_batch(
            prepared["magnitudes"], prepared["triad_bins"], prepared["labels"],
            0, 0, kwargs["n_surrogates"],
        )
        split = np.vstack((
            reviewed_nonlinear.compute_surrogate_triad_batch(
                prepared["magnitudes"], prepared["triad_bins"], prepared["labels"],
                0, 0, 30,
            ),
            reviewed_nonlinear.compute_surrogate_triad_batch(
                prepared["magnitudes"], prepared["triad_bins"], prepared["labels"],
                0, 30, kwargs["n_surrogates"],
            ),
        ))
        np.testing.assert_allclose(serial, split)

    def test_spod_frequency_batches_match_full_reviewed_result(self):
        rng = np.random.default_rng(4)
        dataset = reviewed_modal.SnapshotMatrix(
            np.arange(256, dtype=float) / 1000.0,
            rng.normal(size=(256, 12)), np.linspace(0.0, 1.0, 12),
        )
        full = reviewed_modal.compute_spod(
            dataset, nperseg=32, noverlap=16, n_modes=3, frequency_stride=2,
        )
        prepared = reviewed_modal.prepare_spod_blocks(
            dataset, nperseg=32, noverlap=16, n_modes=3, frequency_stride=2,
        )
        batch = reviewed_modal.compute_spod_frequency_batch(
            prepared["blocks"], prepared["weights"], prepared["mode_count"],
            0, len(prepared["frequency_hz"]),
        )
        np.testing.assert_allclose(full["eigenvalues"], batch["eigenvalues"])
        np.testing.assert_allclose(full["modes"], batch["modes"])
