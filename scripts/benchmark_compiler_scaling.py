"""Measure packed cold compilation and an append against an independent full rebuild."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from freqai import parallel
from freqai.unpaired import UnpairedWaveModel


def array_hash(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def measured_compile(records, previous=None):
    process = psutil.Process()
    samples, done = [], threading.Event()

    def sample():
        while not done.is_set():
            total = 0
            for child in [process] + process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            samples.append((process.memory_info().rss, total))
            done.wait(0.2)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        model = UnpairedWaveModel(records, previous_model=previous)
        elapsed = time.perf_counter() - started
    finally:
        done.set()
        monitor.join()
    run = {"records": len(records), "compile_seconds": elapsed,
           "parent_rss_bytes": process.memory_info().rss,
           "sampled_peak_parent_rss_bytes": max(row[0] for row in samples),
           "sampled_peak_process_tree_rss_bytes": max(row[1] for row in samples),
           "rss_note": "Working-set RSS sampled at 200ms; process-tree sum may count shared pages more than once.",
           "lazy_fields_after_compile": len(model.transition_spectra._cache),
           "reuse": model.compile_reuse_stats, "policy": parallel.describe()}
    print(json.dumps(run), flush=True)
    return model, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/compiler_scaling_packed.json")
    args = parser.parse_args()
    database = PROJECT_ROOT / "memory/memory.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT id,text,source FROM documents ORDER BY sequence LIMIT ?",
                                  (args.limit if args.limit else -1,)).fetchall()
    records = [dict(zip(("id", "text", "source"), row)) for row in rows]
    added = [{"id": f"compiler-scaling-{index}", "title": f"Skalonummer{index}",
              "text": f"Die Skalonummer{index} ist ein hypothetisches Testobjekt mit Kennwert {index + 21001}."}
             for index in range(args.append)]
    output = {"read_only_database": True, "synthetic_append_written_to_database": False,
              "runs": {}, "append_records": args.append}
    base, output["runs"]["cold"] = measured_compile(records)
    base_signature = base.model_signature()
    output["base_signature"] = base_signature
    baseline_path = PROJECT_ROOT / "results/compiler_benchmark_full_optimized.json"
    if args.limit == 0 and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["runs"][0]
        output["base_signature_matches_previous_architecture"] = base_signature == baseline["signature"]
        output["base_energy_matches_previous_architecture"] = array_hash(base.symbol_energy) == baseline["symbol_energy_sha256"]
        output["cold_speedup_over_previous_architecture"] = baseline["compile_seconds"] / output["runs"]["cold"]["compile_seconds"]
    old_coefficients, old_energy = array_hash(base.transition_spectra.coefficients), array_hash(base.symbol_energy)
    appended, output["runs"]["append_reuse"] = measured_compile(records + added, base)
    output["previous_arrays_unchanged"] = (old_coefficients == array_hash(base.transition_spectra.coefficients)
                                            and old_energy == array_hash(base.symbol_energy))
    del base
    gc.collect()
    fresh, output["runs"]["append_full_rebuild"] = measured_compile(records + added)
    output["append_signature_matches_full_rebuild"] = appended.model_signature() == fresh.model_signature()
    output["append_energy_matches_full_rebuild"] = array_hash(appended.symbol_energy) == array_hash(fresh.symbol_energy)
    output["append_speedup_over_full_rebuild"] = (output["runs"]["append_full_rebuild"]["compile_seconds"] /
                                                 output["runs"]["append_reuse"]["compile_seconds"])
    output["generation_matches_full_rebuild"] = True
    for prompt in ("Was ist eine Frequenz?", "Was ist Skalonummer0?"):
        actual, expected = appended.generate(prompt, max_tokens=24), fresh.generate(prompt, max_tokens=24)
        output["generation_matches_full_rebuild"] &= (actual["tokens"] == expected["tokens"] and actual["ended"] == expected["ended"]
            and [step["probability"] for step in actual["trace"]] == [step["probability"] for step in expected["trace"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in {"runs", "base_signature"}}), flush=True)
    parallel._shutdown()


if __name__ == "__main__":
    main()
