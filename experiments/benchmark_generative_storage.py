"""Isolated synthetic storage/latency benchmark; no evaluation or live DB reads."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_working_set():
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if not get_info(get_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return int(counters.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def suffix(number):
    letters = ""
    while True:
        letters = chr(97+number % 26) + letters
        number = number//26-1
        if number < 0:
            return letters


def benchmark(count, conditioning):
    from freqai.generative import GenerativeWaveModel
    import numpy as np
    pairs = [{"prompt": f"Was macht Objekt{suffix(i)}?",
              "text": f"Objekt{suffix(i)} besucht Ort{suffix(i)}. Dort warten Freunde."}
             for i in range(count)]
    start = time.perf_counter()
    model = GenerativeWaveModel([], order=2, conditioned_pairs=pairs, conditioning=conditioning)
    compilation_seconds = time.perf_counter()-start
    gc.collect()
    stats = model.storage_stats()
    prompt = pairs[count//2]["prompt"]
    field, _ = model.prompt_field(prompt)
    # Exercise sparse, global-backoff, supported, and unsupported prefixes.
    prefixes = [[], ["dort"], ["besucht"], ["unbekannt"]]
    maximum_direct_error = 0.0
    step_times = []
    for _ in range(3):
        for prefix in prefixes:
            start = time.perf_counter()
            actual, _ = model.next_distribution(prefix, field, time_s=86400.125)
            step_times.append(time.perf_counter()-start)
            expected, _ = model.next_distribution(prefix, field, method="direct", time_s=86400.125)
            maximum_direct_error = max(maximum_direct_error, float(np.max(np.abs(actual-expected))))
    start = time.perf_counter()
    generated = model.generate(prompt, decoding="beam", max_tokens=40)
    generation_seconds = time.perf_counter()-start
    return {"record_count": count, "conditioning": conditioning, "order": 2,
            "compilation_seconds": compilation_seconds, "generation_seconds": generation_seconds,
            "single_step_median_seconds": float(np.median(step_times)), "generated_tokens": len(generated["tokens"]),
            "generated_text": generated["text"], "operator_direct_max_error": maximum_direct_error,
            "peak_process_working_set_bytes": peak_working_set(), "storage": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int)
    parser.add_argument("--conditioning", default="semantic")
    parser.add_argument("--output", type=Path, default=ROOT / "results/corpus_expansion/scaling/storage_benchmark.json")
    args = parser.parse_args()
    if args.count:
        print(json.dumps(benchmark(args.count, args.conditioning), ensure_ascii=True))
        return
    results = []
    for count in (1000, 5000, 10000):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--count", str(count),
                                  "--conditioning", args.conditioning], capture_output=True, text=True, check=True)
        result = json.loads(process.stdout)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"synthetic_only": True,
            "interpretation": "Storage and numerical scalability; not a conversational-quality evaluation.",
            "processes": "Each size runs in a separate fresh Python process.", "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
