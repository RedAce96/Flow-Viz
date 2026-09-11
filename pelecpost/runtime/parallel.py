"""Bounded process-based execution for independent post-processing tasks.

The scheduler deliberately owns no workflow state.  Workers receive small,
picklable task payloads, write only their private outputs, and return a result
to the parent.  The parent remains responsible for manifests, artifacts, and
all user-visible logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import multiprocessing as mp
from multiprocessing.pool import ApplyResult
import os
from pathlib import Path
import queue
import resource
import sys
import time
import traceback
from typing import Any, Callable, Iterable


TaskFunction = Callable[[Any], Any]
EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ParallelTask:
    """One independent task submitted to a parallel stage."""

    index: int
    task_id: str
    payload: Any
    identity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelStagePlan:
    """Resource-limited concurrency chosen for one stage."""

    stage: str
    requested_workers: int
    effective_workers: int
    task_count: int
    available_cpus: int
    parent_resident_gb: float
    per_worker_peak_gb: float
    estimated_concurrent_peak_gb: float
    limiting_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ParallelTaskResult:
    """Serializable result and timing information from one worker."""

    index: int
    task_id: str
    value: Any = None
    elapsed_s: float = 0.0
    worker_pid: int = 0
    peak_rss_bytes: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    remote_traceback: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


@dataclass(frozen=True)
class PendingArtifact:
    """Artifact metadata returned by a worker and registered by the parent."""

    artifact_id: str
    path: str
    kind: str
    variable: str | None
    units: str | None
    coordinate_metadata: dict[str, Any]
    interpretation: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ReadonlyArraySpec:
    """Description of a read-only NumPy array staged for workers."""

    path: str
    shape: tuple[int, ...]
    dtype: str


def stage_readonly_array(
    array: Any,
    directory: str | Path,
    *,
    prefix: str,
) -> tuple[ReadonlyArraySpec, Callable[[], None]]:
    """Write one temporary NumPy array that spawned workers can reopen read-only."""

    import tempfile
    import numpy as np

    data = np.asarray(array)
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    scratch = Path(directory)
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".npy", dir=str(scratch))
    os.close(descriptor)
    path = Path(name)
    mapped = None
    closed = False
    try:
        mapped = np.lib.format.open_memmap(
            path, mode="w+", dtype=data.dtype, shape=data.shape,
        )
        if data.ndim == 0:
            mapped[...] = data
        else:
            row_bytes = max(
                1,
                int(data.dtype.itemsize)
                * max(1, int(np.prod(data.shape[1:], dtype=np.int64))),
            )
            block_rows = max(1, min(data.shape[0], (8 * 1024 * 1024) // row_bytes))
            for start in range(0, data.shape[0], block_rows):
                mapped[start:start + block_rows] = data[start:start + block_rows]
        mapped.flush()
        mmap = getattr(mapped, "_mmap", None)
        if mmap is not None:
            mmap.close()
        mapped = None
        os.chmod(path, 0o444)
    except BaseException:
        if mapped is not None:
            mmap = getattr(mapped, "_mmap", None)
            if mmap is not None:
                mmap.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise

    def cleanup() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return ReadonlyArraySpec(str(path), tuple(data.shape), str(data.dtype)), cleanup


class ParallelPlanningError(RuntimeError):
    """Raised when the configured memory budget cannot run one task."""


class ParallelTaskError(RuntimeError):
    """Raised with the identity and remote traceback of a failed task."""

    def __init__(self, stage: str, result: ParallelTaskResult):
        self.stage = stage
        self.result = result
        message = (
            f"parallel stage {stage!r} task {result.task_id!r} failed: "
            f"{result.error_type}: {result.error_message}"
        )
        if result.remote_traceback:
            message += f"\nRemote traceback:\n{result.remote_traceback}"
        super().__init__(message)


def _cpu_count() -> int:
    """Return the CPU allocation visible to this process, not host capacity."""

    candidates: list[int] = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP"):
        raw = os.environ.get(name)
        if raw:
            try:
                candidates.append(int(raw))
            except ValueError:
                pass
    candidates.append(os.cpu_count() or 1)
    return max(1, min(value for value in candidates if value > 0))


def available_cpu_count() -> int:
    """Expose the scheduler's effective CPU allocation for preflight reporting."""

    return _cpu_count()


def plan_parallel_stage(
    stage: str,
    *,
    requested_workers: int,
    task_count: int,
    memory_limit_gb: float,
    parent_resident_gb: float,
    per_worker_peak_gb: float,
) -> ParallelStagePlan:
    """Choose safe concurrency using the documented memory and CPU policy."""

    requested_workers = int(requested_workers)
    task_count = int(task_count)
    memory_limit_gb = float(memory_limit_gb)
    parent_resident_gb = max(0.0, float(parent_resident_gb))
    per_worker_peak_gb = max(0.0, float(per_worker_peak_gb))
    if requested_workers < 1:
        raise ValueError("requested_workers must be positive")
    if task_count < 0:
        raise ValueError("task_count cannot be negative")
    if not memory_limit_gb > 0.0:
        raise ValueError("memory_limit_gb must be positive")

    available_cpus = _cpu_count()
    usable_memory = 0.8 * memory_limit_gb
    available_worker_memory = usable_memory - parent_resident_gb
    if task_count and per_worker_peak_gb > 0.0 and available_worker_memory < per_worker_peak_gb:
        raise ParallelPlanningError(
            f"parallel stage {stage!r} cannot fit one worker: parent estimate "
            f"{parent_resident_gb:.3f} GB plus worker estimate "
            f"{per_worker_peak_gb:.3f} GB exceeds 80% of the configured "
            f"{memory_limit_gb:.3f} GB limit"
        )
    memory_workers = requested_workers if per_worker_peak_gb == 0.0 else max(
        1, int(available_worker_memory // per_worker_peak_gb)
    )
    effective = min(
        requested_workers,
        max(1, task_count) if task_count else 1,
        available_cpus,
        memory_workers,
    )
    reasons: list[str] = []
    if effective < requested_workers:
        if task_count < requested_workers:
            reasons.append("task_count")
        if available_cpus < requested_workers:
            reasons.append("available_cpus")
        if memory_workers < requested_workers:
            reasons.append("memory_limit")
    peak = parent_resident_gb + effective * per_worker_peak_gb
    return ParallelStagePlan(
        stage=stage,
        requested_workers=requested_workers,
        effective_workers=effective,
        task_count=task_count,
        available_cpus=available_cpus,
        parent_resident_gb=parent_resident_gb,
        per_worker_peak_gb=per_worker_peak_gb,
        estimated_concurrent_peak_gb=peak,
        limiting_reasons=tuple(reasons),
    )


_EVENT_QUEUE: Any = None
_TASK_FUNCTION: TaskFunction | None = None


def _emit_worker_event(event: str, **details: Any) -> None:
    if _EVENT_QUEUE is None:
        return
    payload = {
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "worker_pid": os.getpid(),
        **details,
    }
    try:
        _EVENT_QUEUE.put_nowait(payload)
    except (queue.Full, OSError):
        # Logging must never make a scientific task fail.  The parent still
        # receives the final result and records its completion.
        pass


def _worker_initializer(event_queue: Any, task_function: TaskFunction) -> None:
    global _EVENT_QUEUE, _TASK_FUNCTION
    _EVENT_QUEUE = event_queue
    _TASK_FUNCTION = task_function
    _configure_worker_environment()


def _configure_worker_environment() -> None:
    """Configure process-wide rendering and numerical thread behavior."""
    # This is called in the parent immediately before spawn as well as in the
    # initializer.  The parent-side call makes the settings visible before a
    # child imports a task module that itself imports Matplotlib/NumPy.
    os.environ["MPLBACKEND"] = "Agg"
    # Prevent N processes from each creating an unrestricted BLAS/OpenMP pool.
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def _worker_entry(task: ParallelTask) -> ParallelTaskResult:
    if _TASK_FUNCTION is None:
        raise RuntimeError("parallel worker was not initialized")
    started = time.perf_counter()
    _emit_worker_event(
        "parallel-task-start", task_id=task.task_id, task_index=task.index,
        current_phase="task", work_unit=task.identity,
    )
    _emit_worker_event(
        "parallel-task-phase-start", task_id=task.task_id,
        task_index=task.index, current_phase="task", work_unit=task.identity,
    )
    try:
        # Importing Matplotlib here ensures worker figures use a noninteractive
        # backend even when the parent was launched from an interactive shell.
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
        except ImportError:
            pass
        value = _TASK_FUNCTION(task.payload)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        result = ParallelTaskResult(
            index=task.index, task_id=task.task_id, elapsed_s=elapsed,
            worker_pid=os.getpid(), peak_rss_bytes=_peak_rss_bytes(),
            error_type=type(exc).__name__, error_message=str(exc),
            remote_traceback=traceback.format_exc(),
        )
        _emit_worker_event(
            "parallel-task-phase-failed", task_id=task.task_id,
            task_index=task.index, current_phase="task",
            elapsed_s=elapsed, error_type=result.error_type,
            error_message=result.error_message, work_unit=task.identity,
        )
        _emit_worker_event(
            "parallel-task-failed", task_id=task.task_id, task_index=task.index,
            elapsed_s=elapsed, error_type=result.error_type,
            error_message=result.error_message, current_phase="task",
            work_unit=task.identity,
        )
        return result
    elapsed = time.perf_counter() - started
    _emit_worker_event(
        "parallel-task-phase-complete", task_id=task.task_id,
        task_index=task.index, current_phase="task", elapsed_s=elapsed,
        work_unit=task.identity,
    )
    result = ParallelTaskResult(
        index=task.index, task_id=task.task_id, value=value, elapsed_s=elapsed,
        worker_pid=os.getpid(), peak_rss_bytes=_peak_rss_bytes(),
    )
    _emit_worker_event(
        "parallel-task-complete", task_id=task.task_id, task_index=task.index,
        elapsed_s=elapsed, peak_rss_bytes=result.peak_rss_bytes, current_phase="task",
        work_unit=task.identity,
    )
    return result


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value * 1024 if sys.platform.startswith("linux") else value
    except (AttributeError, OSError):
        return None


def _emit_parent_event(
    callback: EventCallback | None, event: str, **details: Any,
) -> None:
    if callback is not None:
        callback(event, {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        })


def _validate_task_function(function: TaskFunction) -> None:
    qualified_name = getattr(function, "__qualname__", "")
    if "<locals>" in qualified_name:
        raise TypeError(
            "parallel task functions must be defined at module scope so spawn can pickle them"
        )


def _inline_result(
    stage: str,
    task: ParallelTask,
    function: TaskFunction,
    event_callback: EventCallback | None,
) -> ParallelTaskResult:
    started = time.perf_counter()
    _emit_parent_event(
        event_callback, "parallel-task-start", stage=stage,
        task_id=task.task_id, task_index=task.index, worker_pid=os.getpid(),
        work_unit=task.identity,
    )
    _emit_parent_event(
        event_callback, "parallel-task-phase-start", stage=stage,
        task_id=task.task_id, task_index=task.index, worker_pid=os.getpid(),
        current_phase="task", work_unit=task.identity,
    )
    try:
        value = function(task.payload)
    except BaseException as exc:
        _emit_parent_event(
            event_callback, "parallel-task-phase-failed", stage=stage,
            task_id=task.task_id, task_index=task.index, worker_pid=os.getpid(),
            current_phase="task", error_type=type(exc).__name__, error_message=str(exc),
            work_unit=task.identity,
        )
        return ParallelTaskResult(
            index=task.index, task_id=task.task_id,
            elapsed_s=time.perf_counter() - started, worker_pid=os.getpid(),
            peak_rss_bytes=_peak_rss_bytes(), error_type=type(exc).__name__,
            error_message=str(exc), remote_traceback=traceback.format_exc(),
        )
    elapsed = time.perf_counter() - started
    _emit_parent_event(
        event_callback, "parallel-task-phase-complete", stage=stage,
        task_id=task.task_id, task_index=task.index, worker_pid=os.getpid(),
        current_phase="task", elapsed_s=elapsed, work_unit=task.identity,
    )
    return ParallelTaskResult(
        index=task.index, task_id=task.task_id, value=value,
        elapsed_s=elapsed, worker_pid=os.getpid(),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def run_parallel_stage(
    stage: str,
    tasks: Iterable[ParallelTask],
    function: TaskFunction,
    plan: ParallelStagePlan,
    *,
    event_callback: EventCallback | None = None,
    heartbeat_s: float = 30.0,
    recycle_after_tasks: int | None = None,
) -> tuple[ParallelTaskResult, ...]:
    """Run bounded tasks and return results in input order.

    The scheduler submits at most ``effective_workers`` tasks at once.  A
    failure terminates the stage pool and raises ``ParallelTaskError`` with
    the remote traceback, allowing the workflow to clean its private files.
    """

    _validate_task_function(function)
    task_list = tuple(tasks)
    if len(task_list) != plan.task_count:
        raise ValueError("parallel stage plan task_count does not match tasks")
    _emit_parent_event(
        event_callback, "parallel-stage-plan", stage=stage,
        requested_workers=plan.requested_workers,
        effective_workers=plan.effective_workers,
        task_count=plan.task_count, available_cpus=plan.available_cpus,
        parent_resident_gb=plan.parent_resident_gb,
        per_worker_peak_gb=plan.per_worker_peak_gb,
        estimated_concurrent_peak_gb=plan.estimated_concurrent_peak_gb,
        limiting_reasons=list(plan.limiting_reasons),
    )
    if not task_list:
        _emit_parent_event(event_callback, "parallel-stage-complete", stage=stage, task_count=0)
        return ()
    if plan.effective_workers <= 1:
        results: list[ParallelTaskResult] = []
        for task in task_list:
            result = _inline_result(stage, task, function, event_callback)
            if not result.succeeded:
                _emit_parent_event(
                    event_callback, "parallel-task-failed", stage=stage,
                    task_id=task.task_id, task_index=task.index,
                    elapsed_s=result.elapsed_s, error_type=result.error_type,
                    error_message=result.error_message, current_phase="task",
                    work_unit=task.identity,
                )
                _emit_parent_event(
                    event_callback, "parallel-stage-failed", stage=stage,
                    task_id=task.task_id, task_index=task.index,
                    completed_count=len(results), work_unit=task.identity,
                )
                raise ParallelTaskError(stage, result)
            _emit_parent_event(
                event_callback, "parallel-task-complete", stage=stage,
                task_id=task.task_id, task_index=task.index,
                elapsed_s=result.elapsed_s, worker_pid=result.worker_pid,
                peak_rss_bytes=result.peak_rss_bytes, current_phase="task",
                work_unit=task.identity,
            )
            results.append(result)
        _emit_parent_event(event_callback, "parallel-stage-complete", stage=stage, task_count=len(results))
        return tuple(results)

    _configure_worker_environment()
    multiprocessing_context = mp.get_context("spawn")
    event_queue = multiprocessing_context.Queue(maxsize=max(64, plan.effective_workers * 8))
    pool = multiprocessing_context.Pool(
        processes=plan.effective_workers,
        initializer=_worker_initializer,
        initargs=(event_queue, function),
        maxtasksperchild=recycle_after_tasks,
    )
    active: dict[ApplyResult, ParallelTask] = {}
    results = []
    next_task = 0
    last_heartbeat = time.monotonic()
    completed_count = 0

    def submit_available() -> None:
        nonlocal next_task
        while next_task < len(task_list) and len(active) < plan.effective_workers:
            task = task_list[next_task]
            active[pool.apply_async(_worker_entry, (task,))] = task
            next_task += 1

    try:
        submit_available()
        while active:
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                event_callback and event_callback(
                    str(event.get("event", "progress")),
                    {"stage": stage, **event},
                )
            finished = [item for item in active if item.ready()]
            if not finished:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_s:
                    _emit_parent_event(
                        event_callback, "parallel-stage-heartbeat", stage=stage,
                        completed_count=completed_count, total_count=len(task_list),
                        running_task_ids=[task.task_id for task in active.values()],
                    )
                    last_heartbeat = now
                time.sleep(0.02)
                continue
            for future in finished:
                task = active.pop(future)
                try:
                    result = future.get()
                except BaseException as exc:
                    result = ParallelTaskResult(
                        index=task.index, task_id=task.task_id,
                        worker_pid=0, error_type=type(exc).__name__,
                        error_message=str(exc), remote_traceback=traceback.format_exc(),
                    )
                    _emit_parent_event(
                        event_callback, "parallel-task-failed", stage=stage,
                        task_id=task.task_id, task_index=task.index,
                        error_type=result.error_type, error_message=result.error_message,
                        current_phase="task", work_unit=task.identity,
                    )
                    _emit_parent_event(
                        event_callback, "parallel-stage-failed", stage=stage,
                        task_id=task.task_id, task_index=task.index,
                        completed_count=completed_count, work_unit=task.identity,
                    )
                    pool.terminate()
                    pool.join()
                    raise ParallelTaskError(stage, result) from exc
                if not isinstance(result, ParallelTaskResult):
                    raise RuntimeError(f"parallel task {task.task_id!r} returned an invalid result")
                if not result.succeeded:
                    _emit_parent_event(
                        event_callback, "parallel-task-failed", stage=stage,
                        task_id=task.task_id, task_index=task.index,
                        elapsed_s=result.elapsed_s, worker_pid=result.worker_pid,
                        error_type=result.error_type, error_message=result.error_message,
                        current_phase="task", work_unit=task.identity,
                    )
                    _emit_parent_event(
                        event_callback, "parallel-stage-failed", stage=stage,
                        task_id=task.task_id, task_index=task.index,
                        completed_count=completed_count, work_unit=task.identity,
                    )
                    pool.terminate()
                    pool.join()
                    raise ParallelTaskError(stage, result)
                results.append(result)
                completed_count += 1
            submit_available()
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    finally:
        try:
            while True:
                event = event_queue.get_nowait()
                event_callback and event_callback(
                    str(event.get("event", "progress")),
                    {"stage": stage, **event},
                )
        except queue.Empty:
            pass
        event_queue.close()
        event_queue.join_thread()
    results.sort(key=lambda item: item.index)
    _emit_parent_event(
        event_callback, "parallel-stage-complete", stage=stage,
        task_count=len(results), completed_count=len(results),
    )
    return tuple(results)


def remove_paths(paths: Iterable[str | Path]) -> None:
    """Best-effort cleanup helper for unregistered worker outputs."""

    for raw_path in paths:
        path = Path(raw_path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
