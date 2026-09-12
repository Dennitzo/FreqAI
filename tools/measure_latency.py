"""Messung von Compiler-Aufbau und Antwortlaufzeit gegen den Code im Projektordner."""
from __future__ import annotations

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import freqai  # noqa: E402
from freqai.memory import WaveMemory  # noqa: E402
from freqai.store import DEFAULT_MEMORY_PATH  # noqa: E402
from freqai.unpaired_runtime import generator_for, respond_wave  # noqa: E402

PROMPTS = [
    "Hey, wie gehts dir?",
    "Wer ist Albert Einstein gewesen?",
    "Was ist eine Frequenz?",
]


def main() -> None:
    print("module", freqai.__file__, flush=True)
    started = time.perf_counter()
    memory = WaveMemory.load(DEFAULT_MEMORY_PATH)
    print(f"load={time.perf_counter()-started:.2f}s docs={len(memory.documents)}", flush=True)

    started = time.perf_counter()
    model = generator_for(memory).model
    print(f"compiler_build={time.perf_counter()-started:.2f}s vocabulary={len(model.vocabulary)} "
          f"facts={len(model.facts)}", flush=True)

    for prompt in PROMPTS:
        started = time.perf_counter()
        analysis = model.analyze_question(prompt)
        analyse_time = time.perf_counter() - started
        started = time.perf_counter()
        try:
            result, _ = respond_wave(memory, prompt)
            elapsed = time.perf_counter() - started
            print(f"analyze={analyse_time:6.2f}s generate={elapsed:8.2f}s | {prompt!r} -> "
                  f"{result['text']!r} reason={result.get('reason')!r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"analyze={analyse_time:6.2f}s | {prompt!r} -> ERROR {type(exc).__name__}: {exc}", flush=True)

    for prompt in PROMPTS[:2]:
        analysis = model.analyze_question(prompt)
        summary = {k: (v if k not in {"segments", "evidence", "temporary_roles"} else f"<{len(v)}>")
                   for k, v in analysis.items()}
        print(json.dumps(summary, ensure_ascii=False, default=str)[:600], flush=True)

    einstein = [doc.text for doc in memory.documents if "einstein" in doc.text.lower()]
    print(f"einstein_documents={len(einstein)}", flush=True)
    for text in einstein[:4]:
        print("   ", text[:260].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
