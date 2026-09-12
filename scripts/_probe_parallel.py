"""Reproducible CPU-worker, library-thread and spectra parity evidence.

Run .venv/Scripts/python.exe scripts/_probe_parallel.py. The default corpus is
synthetic and does not change persistent memory. --memory uses the local corpus.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from freqai.memory import document_spectra
from freqai import parallel


def _worker_probe(arguments):
    from multiprocessing import shared_memory
    from scipy.fft import get_workers
    import threadpoolctl
    index, name, count = arguments
    block = shared_memory.SharedMemory(name=name)
    slots = np.ndarray((count,), dtype=np.int64, buffer=block.buf)
    try:
        slots[index] = os.getpid()
        deadline = time.monotonic() + 90
        while np.any(slots == 0):
            if time.monotonic() > deadline:
                raise RuntimeError("not every CPU worker entered the probe within 90 seconds")
            time.sleep(0.01)
        start = time.process_time()
        checksum = 0
        while time.process_time() - start < 0.1:
            for value in range(1000):
                checksum = (checksum * 31 + value) % 1000000007
        return {"pid": os.getpid(), "cpu_seconds": time.process_time() - start,
                "nested_worker_count": parallel.worker_count(), "fft_workers": get_workers(),
                "blas_threads": [item["num_threads"] for item in threadpoolctl.threadpool_info()],
                "checksum": checksum}
    finally:
        del slots
        block.close()


def main():
    from multiprocessing import shared_memory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--records", type=int, default=4096)
    parser.add_argument("--dimensions", type=int, default=512)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers:
        os.environ["FREQAI_WORKERS"] = str(args.workers)
    count = parallel.worker_count()
    evidence = {"policy": parallel.describe(), "parent_pid": os.getpid()}
    print("policy", json.dumps(evidence["policy"]), flush=True)
    block = shared_memory.SharedMemory(create=True, size=count * 8)
    slots = np.ndarray((count,), dtype=np.int64, buffer=block.buf)
    slots.fill(0)
    try:
        evidence["worker_probes"] = parallel.map(_worker_probe,
            [(index, block.name, count) for index in range(count)],
            workers=count, chunks=count, min_items=1)
    finally:
        del slots
        block.close()
        block.unlink()
    evidence["observed_worker_count"] = len({item["pid"] for item in evidence["worker_probes"]})
    assert evidence["observed_worker_count"] == count
    assert all(item["fft_workers"] == item["nested_worker_count"] == 1
               and item["blas_threads"] and max(item["blas_threads"]) == 1
               for item in evidence["worker_probes"])
    print("observed active workers", evidence["observed_worker_count"], flush=True)
    if args.memory:
        from freqai.store import DEFAULT_MEMORY_PATH, MemoryStore
        memory = MemoryStore(DEFAULT_MEMORY_PATH).load_memory()
        texts = [doc.text for doc in memory.documents][:args.records]
        dims, mode = memory.dimensions, memory.feature_mode
    else:
        texts = [f"Datensatz {index}: Stehende Wellen speichern Information durch Frequenzen. "
                 f"Parallel berechnete Spektren behalten Text {index % 97} bitgenau bei."
                 for index in range(args.records)]
        dims, mode = args.dimensions, "hybrid"
    previous = os.environ.get("FREQAI_PARALLEL")
    try:
        os.environ["FREQAI_PARALLEL"] = "0"
        start = time.perf_counter()
        serial = document_spectra(texts, dims, mode)
        serial_seconds = time.perf_counter() - start
        os.environ["FREQAI_PARALLEL"] = "1"
        start = time.perf_counter()
        result = document_spectra(texts, dims, mode)
        parallel_seconds = time.perf_counter() - start
    finally:
        if previous is None:
            os.environ.pop("FREQAI_PARALLEL", None)
        else:
            os.environ["FREQAI_PARALLEL"] = previous
    evidence.update({"records": len(texts), "dimensions": dims, "shape": list(result.shape),
                     "serial_seconds": serial_seconds, "parallel_seconds": parallel_seconds,
                     "speedup": serial_seconds / parallel_seconds,
                     "bitexact": bool(np.array_equal(serial, result)),
                     "maxdiff": float(np.max(np.abs(serial - result))) if result.size else 0.0,
                     "final_policy": parallel.describe()})
    assert evidence["bitexact"]
    assert not evidence["final_policy"]["pool_failed"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2), encoding="utf8")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
