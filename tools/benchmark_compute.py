"""Read-only numerical CPU/CUDA benchmark against the real corpus and reference.

Run from the repository root: .venv/Scripts/python tools/benchmark_compute.py
The SQLite corpus is opened with mode=ro. Only the requested JSON report is
written; no data, compiler state, or conversation is changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from freqai.compute import compute_status, cuda_for, fft_rows
from freqai.memory import Document, WaveMemory
from freqai.store import DEFAULT_MEMORY_PATH


def measure(function, repeats=5):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter()-start)
    return {"median_s": statistics.median(values), "samples_s": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/compute_backend/benchmark.json"))
    args = parser.parse_args()
    with sqlite3.connect(args.storage.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        documents = [Document(*row) for row in connection.execute("SELECT id,text,source FROM documents ORDER BY sequence")]
    # The spatial wave bank is independent of diagnostic key dimensions. Small
    # keys avoid spending this benchmark on feature hashing or a 1.3 GB matrix.
    os.environ["FREQAI_COMPUTE"] = "cpu"
    memory = WaveMemory(documents, dimensions=16)
    report = {"documents": len(documents), "modal_count": len(memory._coefficients),
              "display_points": 256, "precision": "float64/complex128"}
    reference = memory.snapshot(19.125)
    report["snapshot_cpu"] = measure(lambda: memory.snapshot(19.125))
    if cuda_for(len(memory._coefficients), backend="cuda"):
        os.environ["FREQAI_COMPUTE"] = "cuda"
        start = time.perf_counter()
        actual = memory.snapshot(19.125)
        report["snapshot_cuda_first_s"] = time.perf_counter()-start
        report["snapshot_cuda"] = measure(lambda: memory.snapshot(19.125))
        report["snapshot_speedup"] = report["snapshot_cpu"]["median_s"]/report["snapshot_cuda"]["median_s"]
        report["snapshot_max_abs_error"] = max(float(np.max(np.abs(np.asarray(reference[key])-actual[key])))
                                                 for key in ("displacement", "quadrature"))
        report["energy_relative_error"] = abs(actual["energy"]-reference["energy"])/reference["energy"]
    values = np.random.default_rng(84).normal(size=(2048, 4096))
    cpu = fft_rows(values, backend="cpu")
    report["fft_shape"] = list(values.shape)
    report["fft_cpu"] = measure(lambda: fft_rows(values, backend="cpu"))
    if cuda_for(values.size, backend="cuda"):
        gpu = fft_rows(values, backend="cuda")
        report["fft_cuda"] = measure(lambda: fft_rows(values, backend="cuda"))
        report["fft_max_abs_error"] = float(np.max(np.abs(cpu-gpu)))
    report["compute"] = compute_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
