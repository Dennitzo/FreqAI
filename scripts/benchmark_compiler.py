"""Measure compilation against a read-only SQLite snapshot, without changing data."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from freqai import parallel
from freqai.unpaired import UnpairedWaveModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "memory/memory.sqlite3")
    parser.add_argument("--limit", type=int, default=1500, help="0 means the complete active corpus")
    parser.add_argument("--workers", type=int, default=parallel.logical_cores())
    parser.add_argument("--parallel-only", action="store_true", help="Compare a parallel run to --reference")
    parser.add_argument("--reference", type=Path, help="Previous benchmark JSON containing a serial run")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/compiler_benchmark.json")
    args = parser.parse_args()
    if args.parallel_only and args.reference is None:
        parser.error("--parallel-only requires --reference so parity is measured independently")
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        active = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        rows = connection.execute("SELECT id,text,source FROM documents ORDER BY sequence LIMIT ?",
                                  (args.limit if args.limit else -1,)).fetchall()
    records = [dict(zip(("id", "text", "source"), row)) for row in rows]
    result = {"active_records": active, "measured_records": len(records), "read_only_database": True,
              "requested_workers": args.workers, "runs": []}
    reference = None
    if args.reference is not None:
        previous = json.loads(args.reference.read_text(encoding="utf-8"))
        serial_run = next(run for run in previous["runs"] if run["mode"] == "serial")
        reference = serial_run["signature"]
        result["reference"] = str(args.reference)
        result["reference_serial_seconds"] = serial_run["compile_seconds"]
    for mode in (("parallel",) if args.parallel_only else ("serial", "parallel")):
        parallel._shutdown()
        os.environ["FREQAI_PARALLEL"] = "0" if mode == "serial" else "1"
        os.environ["FREQAI_WORKERS"] = str(args.workers)
        started = time.perf_counter()
        model = UnpairedWaveModel(records)
        elapsed = time.perf_counter() - started
        signature = model.model_signature()
        run = {"mode": mode, "compile_seconds": elapsed, "policy": parallel.describe(),
               "signature": signature, "symbol_energy_sha256": hashlib.sha256(model.symbol_energy.tobytes()).hexdigest(),
               "resonance_floor": model.resonance_floor}
        prompts = ("Was ist eine Frequenz?", "Wie heißt du?", "Hallo")
        run["answers"] = {prompt: {key: value for key, value in model.generate(prompt, max_tokens=24).items()
                                  if key in {"text", "tokens", "ended", "reason"}} for prompt in prompts}
        if reference is None:
            reference = signature
        run["signature_matches_serial"] = signature == reference
        result["runs"].append(run)
        print(json.dumps({"mode": mode, "compile_seconds": elapsed,
                          "signature_matches_serial": run["signature_matches_serial"],
                          "policy": run["policy"]}), flush=True)
        del model
        gc.collect()
    parallel._shutdown()
    baseline_path = PROJECT_ROOT / "results/parallel_baseline.json"
    if args.limit == 1500 and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        result["matches_saved_pre_refactor_signature"] = reference == ast.literal_eval(baseline["model_signature"])
    serial_seconds = (result["reference_serial_seconds"] if args.parallel_only else
                      result["runs"][0]["compile_seconds"])
    result["speedup"] = serial_seconds / result["runs"][-1]["compile_seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "speedup": result["speedup"],
                      "matches_saved_pre_refactor_signature": result.get("matches_saved_pre_refactor_signature")}), flush=True)


if __name__ == "__main__":
    main()
