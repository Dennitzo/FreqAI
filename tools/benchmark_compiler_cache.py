"""Read-only corpus compile plus exact persistent-cache save/load verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil

from freqai.compiler_cache import CACHE_FILENAME, cache_key, load_compiled, save_compiled
from freqai.unpaired import UnpairedWaveModel


def rss():
    process = psutil.Process()
    return {"parent_bytes": process.memory_info().rss,
            "children_bytes": sum(child.memory_info().rss for child in process.children(recursive=True)
                                   if child.is_running())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path("memory/memory.sqlite3"))
    parser.add_argument("--cache-dir", type=Path, default=Path("memory/.compiler-cache/memory.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("results/compute_backend/compiler_cache_benchmark.json"))
    args = parser.parse_args()
    with sqlite3.connect(args.storage.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        records = [{"id": row[0], "text": row[1], "source": row[2]} for row in connection.execute(
            "SELECT id,text,source FROM documents ORDER BY sequence")]
    report = {"records": len(records), "storage_read_only": True, "cache_key": cache_key(records, 2),
              "memory_before": rss()}
    start = time.perf_counter()
    model = UnpairedWaveModel(records, order=2)
    report["cold_compile_s"] = time.perf_counter()-start
    report["memory_after_compile"] = rss()
    print(f"cold_compile_s={report['cold_compile_s']:.6f}", flush=True)
    start = time.perf_counter()
    if not save_compiled(model, records, 2, args.cache_dir):
        raise RuntimeError("Compiler cache save failed")
    report["cache_save_s"] = time.perf_counter()-start
    report["cache_file_bytes"] = (args.cache_dir/CACHE_FILENAME).stat().st_size
    print(f"cache_save_s={report['cache_save_s']:.6f} bytes={report['cache_file_bytes']}", flush=True)
    start = time.perf_counter()
    restored = load_compiled(records, 2, args.cache_dir)
    report["cache_load_s"] = time.perf_counter()-start
    if restored is None:
        raise RuntimeError("Compiler cache load missed identical inputs")
    report["memory_both_models"] = rss()
    print(f"cache_load_s={report['cache_load_s']:.6f}", flush=True)
    report["signature"] = model.model_signature()
    report["signature_identical"] = restored.model_signature() == report["signature"]
    report["symbol_energy_identical"] = np.array_equal(restored.symbol_energy, model.symbol_energy)
    report["compilation_energy_identical"] = np.array_equal(restored.transition_spectra.mode_energy, model.transition_spectra.mode_energy)
    report["cache_load_speedup"] = report["cold_compile_s"]/report["cache_load_s"]
    report["generated"] = []
    for prompt in ("Was ist eine Frequenz?", "Wer ist Albert Einstein gewesen?", "Hey, wie gehts dir?"):
        before = model.generate(prompt, max_tokens=40, seed=17)
        start = time.perf_counter()
        after = restored.generate(prompt, max_tokens=40, seed=17)
        report["generated"].append({"prompt": prompt, "text": after["text"],
                                    "tokens_identical": before["tokens"] == after["tokens"],
                                    "text_identical": before["text"] == after["text"],
                                    "cached_model_generate_s": time.perf_counter()-start})
    if not report["signature_identical"] or not report["symbol_energy_identical"] or not all(
            item["tokens_identical"] and item["text_identical"] for item in report["generated"]):
        raise AssertionError("Cached compiler differs from the cold compiler")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
