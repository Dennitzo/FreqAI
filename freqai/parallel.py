"""Ordered spawn-process work and disjoint shared arrays on available CPU cores.

Small inputs and nested calls run serially. Worker failures are reported; only
an unavailable/broken process pool falls back to serial execution. Numerical
libraries use one thread per worker to prevent CPU oversubscription.
"""
from __future__ import annotations

import atexit
import multiprocessing
import operator
import os
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - optional affinity/physical-core details
    _psutil = None

_IN_WORKER = False
_start_lock = threading.RLock()
_thread_state = threading.local()
_executor = None
_executor_workers = 0
_executor_failed = False
_executor_error = None


def logical_cores() -> int:
    """Logical CPUs available to this process, respecting CPU affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:  # pragma: no cover
            pass
    if _psutil is not None:
        try:
            return max(1, len(_psutil.Process().cpu_affinity()))
        except (AttributeError, _psutil.Error, OSError):  # pragma: no cover
            pass
    return max(1, os.cpu_count() or 1)


def physical_cores() -> int:
    if _psutil is not None:
        count = _psutil.cpu_count(logical=False)
        if count:
            return min(logical_cores(), int(count))
    return max(1, logical_cores() // 2)


def parallel_enabled() -> bool:
    return os.environ.get("FREQAI_PARALLEL", "1").strip().lower() not in {"0", "off", "false", "no"}


def worker_count(requested=None) -> int:
    """Default to all available CPUs (Windows' executor supports at most 61)."""
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("worker count must be a positive integer or None")
    if _IN_WORKER or not parallel_enabled():
        return 1
    available = min(logical_cores(), 61) if os.name == "nt" else logical_cores()
    if requested is None:
        override = os.environ.get("FREQAI_WORKERS", "auto").strip()
        try:
            requested = max(1, int(override))
        except ValueError:
            requested = available
    return min(requested, available)


def threads_per_worker(workers=None) -> int:
    workers = worker_count() if workers is None else max(1, int(workers))
    return max(1, logical_cores() // workers)


def configure_threads(limit=None) -> int:
    """Apply real SciPy FFT and BLAS limits in the calling process/thread."""
    limit = threads_per_worker() if limit is None else max(1, int(limit))
    # These also constrain numerical libraries imported later in a worker.
    if _IN_WORKER:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(limit)
    from scipy.fft import set_workers
    previous = getattr(_thread_state, "fft_context", None)
    if previous is not None:
        previous.__exit__(None, None, None)
    context = set_workers(limit)
    context.__enter__()
    _thread_state.fft_context = context
    import threadpoolctl
    threadpoolctl.threadpool_limits(limit)
    return limit


def _worker_init() -> None:
    global _IN_WORKER
    _IN_WORKER = True
    configure_threads(1)


def _shutdown(*, wait_for_workers=False) -> None:
    global _executor, _executor_workers
    with _start_lock:
        executor, _executor = _executor, None
        _executor_workers = 0
    if executor is not None:
        executor.shutdown(wait=wait_for_workers, cancel_futures=False)


atexit.register(_shutdown)


def pool(workers: int):
    """Reuse the spawn pool; return None only when serial execution is needed."""
    global _executor, _executor_workers, _executor_failed, _executor_error
    workers = worker_count(workers)
    with _start_lock:
        if workers <= 1 or _executor_failed:
            return None
        if _executor is not None and _executor_workers == workers:
            return _executor
        if _executor is not None:
            _shutdown()
        try:
            executor = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_init)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - restricted host
            _executor_failed = True
            _executor_error = str(exc)
            warnings.warn(f"CPU process pool unavailable; running serially: {exc}", RuntimeWarning)
            return None
        _executor = executor
        _executor_workers = workers
        return executor


def fail_pool(error=None, *, executor=None) -> None:
    """Retire the failing pool without interfering with a newer concurrent pool."""
    global _executor_failed, _executor_error
    with _start_lock:
        if executor is not None and executor is not _executor:
            return
        _executor_failed = True
        _executor_error = str(error) if error is not None else "process pool unavailable"
        _shutdown(wait_for_workers=True)
    warnings.warn(f"CPU process pool failed; running serially: {_executor_error}", RuntimeWarning)


def chunk_ranges(count: int, chunks: int):
    """Non-overlapping, balanced half-open intervals."""
    count, chunks = operator.index(count), operator.index(chunks)
    if count < 0 or chunks < 1:
        raise ValueError("count must be nonnegative and chunks must be positive")
    if not count:
        return []
    chunks = min(chunks, count)
    size, rest = divmod(count, chunks)
    ranges, start = [], 0
    for index in range(chunks):
        stop = start + size + (index < rest)
        ranges.append((start, stop))
        start = stop
    return ranges


def _apply(function, chunk):
    return [function(item) for item in chunk]


def _settle(futures):
    """No submitted writer may outlive a shared buffer, including on errors."""
    for future in futures:
        future.cancel()
    if futures:
        wait(futures)


def map(function, items, *, workers=None, chunks: int | None = None,  # noqa: A001
        min_items: int = 96, chunks_per_worker: int = 4):
    """Ordered map; application and pickling errors propagate without retries."""
    items = list(items)
    if not items:
        return []
    workers = worker_count(workers)
    if workers <= 1 or len(items) < min_items:
        return [function(item) for item in items]
    packet_count = (operator.index(chunks) if chunks is not None
                    else workers * max(1, operator.index(chunks_per_worker)))
    slices = [items[start:stop] for start, stop in chunk_ranges(len(items), packet_count)]
    futures = []
    executor = None
    try:
        # A differently sized concurrent call may replace the cached pool only
        # after submission. Retired pools finish their already accepted work.
        with _start_lock:
            executor = pool(min(workers, len(slices)))
            if executor is not None:
                try:
                    for chunk in slices:
                        futures.append(executor.submit(_apply, function, chunk))
                except (BrokenProcessPool, OSError, RuntimeError) as exc:
                    fail_pool(exc, executor=executor)
                    executor = None
        if executor is not None:
            try:
                results = []
                for future in futures:
                    results.extend(future.result())
                return results
            except BrokenProcessPool as exc:
                fail_pool(exc, executor=executor)
    finally:
        _settle(futures)
    return [function(item) for item in items]


def _run_shared(name, shape, dtype, start, stop, function):
    import numpy as np
    from multiprocessing import shared_memory
    block = shared_memory.SharedMemory(name=name)
    view = None
    try:
        view = np.ndarray(shape, dtype=dtype, buffer=block.buf)
        function(start, stop, view)
    finally:
        del view
        block.close()


def shared_fill(shape, dtype, jobs, *, workers=None, min_items: int = 4):
    """Run disjoint row jobs in shared memory, returning a private array.

    Jobs must write only their own [start, stop) rows. Gaps are zero filled.
    Invalid ranges/overlaps are rejected before any work is submitted.
    """
    import math
    import numpy as np
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("shared_fill requires a dtype without Python objects")
    shape = tuple(operator.index(value) for value in
                  (shape if isinstance(shape, (tuple, list)) else (shape,)))
    if any(value < 0 for value in shape):
        raise ValueError("shape dimensions must be nonnegative")
    if not shape:
        shape = (0,)
    total = math.prod(shape)
    jobs = [(operator.index(start), operator.index(stop), function) for start, stop, function in jobs]
    previous_stop = 0
    for start, stop in sorted((start, stop) for start, stop, _ in jobs):
        if start < 0 or stop < start or stop > shape[0]:
            raise ValueError("job range is outside the target rows")
        if start < previous_stop:
            raise ValueError("shared_fill job ranges must not overlap")
        previous_stop = stop

    def serial():
        result = np.zeros(shape, dtype=dtype)
        for start, stop, function in jobs:
            function(start, stop, result)
        return result

    workers = worker_count(workers)
    if not jobs or total == 0 or workers <= 1 or len(jobs) < min_items:
        return serial()
    from multiprocessing import shared_memory
    try:
        block = shared_memory.SharedMemory(create=True, size=max(total * dtype.itemsize, 16))
    except (OSError, MemoryError) as exc:  # pragma: no cover - host shared-memory limits
        warnings.warn(f"Shared memory unavailable; running serially: {exc}", RuntimeWarning)
        return serial()
    target = None
    futures = []
    executor = None
    try:
        target = np.ndarray(shape, dtype=dtype, buffer=block.buf)
        target.fill(0)
        with _start_lock:
            executor = pool(min(workers, len(jobs)))
            if executor is not None:
                try:
                    for start, stop, function in jobs:
                        futures.append(executor.submit(_run_shared, block.name, shape, dtype,
                                                       start, stop, function))
                except (BrokenProcessPool, OSError, RuntimeError) as exc:
                    fail_pool(exc, executor=executor)
                    executor = None
        if executor is not None:
            try:
                for future in futures:
                    future.result()
                return target.copy()
            except BrokenProcessPool as exc:
                fail_pool(exc, executor=executor)
    finally:
        _settle(futures)
        del target
        block.close()
        block.unlink()
    return serial()


def describe() -> dict:
    return {"logical_cores": logical_cores(), "physical_cores": physical_cores(),
            "workers": worker_count(), "threads_per_worker": threads_per_worker(),
            "parallel_enabled": parallel_enabled(), "in_worker": bool(_IN_WORKER),
            "pool_active": _executor is not None, "pool_workers": _executor_workers,
            "pool_failed": _executor_failed, "pool_error": _executor_error}


__all__ = ["logical_cores", "physical_cores", "worker_count", "threads_per_worker",
           "configure_threads", "map", "shared_fill", "chunk_ranges", "describe",
           "parallel_enabled", "pool", "fail_pool"]
