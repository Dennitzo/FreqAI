"""Prompt-Batterie gegen den aktiven Bestand laufen lassen und Ergebnisse speichern.

Aufruf:  .venv/Scripts/python.exe scripts/run_prompt_battery.py [--out results/battery/x.json]
         [--only id1,id2] [--max-tokens N]
Mehrturn-Faelle behalten ihren Gespraechskontext innerhalb des Falls.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(ROOT / "memory/evaluation/prompt_battery_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "results/battery/latest.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    from freqai.store import MemoryStore
    from freqai.unpaired_runtime import respond_wave

    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    battery = json.loads(Path(args.battery).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    store = MemoryStore(ROOT / "memory/memory.sqlite3")
    memory = store.load_memory()
    load_s = time.perf_counter() - t0

    results = []
    for case in battery["cases"]:
        if wanted and case["id"] not in wanted:
            continue
        context = None
        for turn in case["turns"]:
            started = time.perf_counter()
            record = {"case": case["id"], "kind": case["kind"], "prompt": turn["prompt"]}
            try:
                answer, state = respond_wave(memory, turn["prompt"], context=context,
                                             max_tokens=args.max_tokens)
                context = state
                record.update({
                    "answer": answer.get("answer", ""),
                    "reason": answer.get("generation_reason"),
                    "truncated": bool(answer.get("truncated")),
                    "abstained": bool(answer.get("abstained")),
                    "speech_act": (answer.get("prompt_encoding") or {}).get("speech_act", ""),
                    "supported": (answer.get("prompt_encoding") or {}).get("supported"),
                })
            except Exception as error:  # noqa: BLE001 - Diagnose soll alles erfassen
                record.update({"answer": "", "error": f"{type(error).__name__}: {error}"})
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            results.append(record)
            print(f"[{case['id']}] {turn['prompt']!r} -> {record['answer']!r} "
                  f"({record['elapsed_ms']} ms, reason={record.get('reason')})", flush=True)

    summary = {"battery": args.battery, "max_tokens": args.max_tokens,
               "load_s": round(load_s, 1), "total_s": round(time.perf_counter() - t0, 1),
               "cases": len({r['case'] for r in results}), "turns": len(results),
               "empty": sum(1 for r in results if not r["answer"].strip()),
               "truncated": sum(1 for r in results if r.get("truncated")),
               "mean_ms": round(sum(r["elapsed_ms"] for r in results) / max(1, len(results)), 1),
               "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nLEER: {summary['empty']}/{summary['turns']}  ABGESCHNITTEN: {summary['truncated']}  "
          f"MITTEL: {summary['mean_ms']} ms  GESAMT: {summary['total_s']} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
