"""Regression coverage for Windows spawn, exact parity and shared-array safety."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from freqai import parallel


def _square(value):
    return value * value


def _worker_details(value):
    from scipy.fft import get_workers
    import threadpoolctl
    return (value, os.getpid(), parallel.worker_count(), get_workers(),
            [pool["num_threads"] for pool in threadpoolctl.threadpool_info()])


def _bad_value(value):
    if not parallel._IN_WORKER:
        raise AssertionError("an application failure was retried in the parent")
    raise ValueError(f"invalid {value}")


def _die_in_worker(value):
    if parallel._IN_WORKER:
        os._exit(17)
    return value * value


class _FillRows:
    def __call__(self, start, stop, target):
        for row in range(start, stop):
            target[row] = row + np.arange(target.shape[1]) * 1j


class _FailOnce:
    def __init__(self, path):
        self.path = str(path)

    def __call__(self, start, stop, target):
        with open(self.path, "a", encoding="utf8") as stream:
            stream.write("called\n")
        raise ValueError("job failed")


class _WritePid:
    def __call__(self, start, stop, target):
        target[start:stop] = os.getpid()


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    monkeypatch.delenv("FREQAI_WORKERS", raising=False)
    monkeypatch.delenv("FREQAI_PARALLEL", raising=False)
    monkeypatch.setattr(parallel, "_executor_failed", False)
    monkeypatch.setattr(parallel, "_executor_error", None)
    yield


def test_policy_uses_available_cores_and_clamps_overrides(monkeypatch):
    monkeypatch.setattr(parallel, "logical_cores", lambda: 8)
    assert parallel.worker_count() == 8
    monkeypatch.setenv("FREQAI_WORKERS", "1000")
    assert parallel.worker_count() == 8
    monkeypatch.setenv("FREQAI_WORKERS", "4")
    assert parallel.worker_count() == 4
    monkeypatch.setenv("FREQAI_PARALLEL", "false")
    assert parallel.worker_count() == 1
    with pytest.raises(ValueError):
        parallel.worker_count(0)


def test_chunk_ranges_cover_each_item_once():
    ranges = parallel.chunk_ranges(113, 32)
    assert [item for start, stop in ranges for item in range(start, stop)] == list(range(113))
    assert max(stop - start for start, stop in ranges) - min(stop - start for start, stop in ranges) <= 1


def test_real_spawn_keeps_order_and_limits_nested_workers_and_libraries():
    result = parallel.map(_worker_details, range(24), workers=2, chunks=24, min_items=1)
    assert [item[0] for item in result] == list(range(24))
    assert all(item[1] != os.getpid() for item in result)
    assert all(item[2] == item[3] == 1 for item in result)
    assert all(item[4] and max(item[4]) == 1 for item in result)
    assert not parallel.describe()["pool_failed"]


def test_numerical_and_shared_results_equal_serial_path():
    assert parallel.map(_square, range(113), workers=2, min_items=1) == [x * x for x in range(113)]
    jobs = [(0, 3, _FillRows()), (5, 8, _FillRows())]
    serial = parallel.shared_fill((8, 12), np.complex128, jobs, workers=1)
    spawned = parallel.shared_fill((8, 12), np.complex128, jobs, workers=2, min_items=1)
    np.testing.assert_array_equal(spawned, serial)
    np.testing.assert_array_equal(spawned[3:5], 0)


def test_shared_rows_are_really_computed_in_child_processes():
    rows = parallel.shared_fill((16,), np.int64,
        [(0, 8, _WritePid()), (8, 16, _WritePid())], workers=2, min_items=1)
    assert np.all(rows > 0)
    assert np.all(rows != os.getpid())


def test_application_errors_are_not_swallowed_and_pool_stays_usable(tmp_path):
    with pytest.raises(ValueError, match="invalid 1"):
        parallel.map(_bad_value, [1, 2], workers=2, chunks=2, min_items=1)
    path = tmp_path / "calls.txt"
    with pytest.raises(ValueError, match="job failed"):
        parallel.shared_fill((2, 3), complex, [(0, 1, _FailOnce(path)), (1, 2, _FillRows())],
                             workers=2, min_items=1)
    assert path.read_text(encoding="utf8") == "called\n"
    assert not parallel.describe()["pool_failed"]
    assert parallel.map(_square, [1, 2], workers=2, min_items=1) == [1, 4]


def test_unpicklable_callable_is_reported_without_disabling_pool():
    with pytest.raises((AttributeError, TypeError)):
        parallel.map(lambda value: value, [1, 2], workers=2, min_items=1)
    assert not parallel.describe()["pool_failed"]
    assert parallel.map(_square, [1, 2], workers=2, min_items=1) == [1, 4]


@pytest.mark.parametrize("ranges", [[(-1, 1), (1, 2)], [(0, 3), (2, 4)], [(0, 5)], [(3, 2)]])
def test_invalid_shared_ranges_are_rejected(ranges):
    with pytest.raises(ValueError):
        parallel.shared_fill((4, 2), np.complex128,
                             [(start, stop, _FillRows()) for start, stop in ranges], workers=2)


def test_object_arrays_are_rejected_before_processes():
    with pytest.raises(ValueError, match="without Python objects"):
        parallel.shared_fill((3,), object, [], workers=2)


def test_concurrent_pool_size_changes_finish_every_call():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(parallel.map, _square, range(64), workers=workers, min_items=1)
                   for workers in (2, 3)]
        assert [future.result() for future in futures] == [[x * x for x in range(64)]] * 2


def test_broken_pool_reports_serial_fallback():
    with pytest.warns(RuntimeWarning, match="CPU process pool failed"):
        result = parallel.map(_die_in_worker, [2, 3], workers=2, min_items=1)
    assert result == [4, 9]
    assert parallel.describe()["pool_failed"]
    assert parallel.describe()["pool_error"]
    parallel._shutdown()
