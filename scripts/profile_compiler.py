"""Profile compiler phases on a read-only corpus snapshot; timings include profiler overhead."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import sqlite3
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from freqai import information, parallel, unpaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/compiler_profile.json")
    args = parser.parse_args()
    os.environ["FREQAI_PARALLEL"] = "0" if args.serial else "1"
    database = PROJECT_ROOT / "memory/memory.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT id,text,source FROM documents ORDER BY sequence LIMIT ?",
                                  (args.limit if args.limit else -1,)).fetchall()
    records = [dict(zip(("id", "text", "source"), row)) for row in rows]
    phases = {}
    original_map = information.parallel_map

    def timed_map(function, items, **kwargs):
        name = getattr(function, "__name__", type(function).__name__)
        started = time.perf_counter()
        try:
            return original_map(function, items, **kwargs)
        finally:
            phases[name] = phases.get(name, 0.0) + time.perf_counter() - started

    information.parallel_map = unpaired.parallel_map = timed_map
    profiler = cProfile.Profile()
    started = time.perf_counter()
    model = profiler.runcall(unpaired.UnpairedWaveModel, records)
    elapsed = time.perf_counter() - started
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).strip_dirs().sort_stats("cumulative").print_stats(45)
    result = {"records": len(records), "serial": args.serial, "profiled_seconds": elapsed,
              "map_phases": phases, "model_storage": model.storage_stats(), "cumulative_profile": report.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"profiled_seconds": elapsed, "map_phases": phases}), flush=True)
    print(report.getvalue(), flush=True)
    parallel._shutdown()


if __name__ == "__main__":
    main()
