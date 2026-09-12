"""Probe the two-wave interaction on the real corpus (never imported as data).

Usage:
    python scripts/two_wave_probe.py --part full        # 20k-document model + known probes
    python scripts/two_wave_probe.py --part randomfacts # 12 synthetic facts, 32 cases
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/two_wave_interference"


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if path.exists():
        raise SystemExit(f"Immutable result already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def full_part():
    from freqai.store import MemoryStore
    from freqai.unpaired import UnpairedWaveModel

    t0 = time.time()
    store = MemoryStore(ROOT / "memory/memory.sqlite3")
    revision, documents = store.snapshot()
    records = [{"id": d.id, "text": d.text, "source": d.source} for d in documents]
    model = UnpairedWaveModel(records, order=2)
    build_s = time.time() - t0

    data_wave = model.data_wave()
    top = sorted(zip(data_wave.indices, data_wave.amplitudes), key=lambda pair: -pair[1])[:10]
    waves_report = {
        "revision": revision,
        "documents": len(records),
        "build_seconds": round(build_s, 2),
        "vocabulary_size": model.size,
        "data_wave": {"mode_count": data_wave.mode_count, "energy": data_wave.energy,
                      "carrier_size": data_wave.carrier_size, "fingerprint": data_wave.fingerprint(),
                      "top_modes": [[model.vocabulary[int(i)], round(float(a), 6)] for i, a in top]},
    }

    def answer(prompt, context=None):
        output = model.generate(prompt, context=context)
        return {"prompt": prompt, "answer": output["text"], "ended": output["ended"],
                "reason": output.get("reason", ""), "kind": output["analysis"].get("kind"),
                "steps": len(output["trace"]),
                "operator_vs_direct_max_error": max(
                    (float(step.get("fft_roundtrip_max_error", 0.0)) for step in output["trace"]), default=0.0),
                "wave_resonance": [step.get("wave_resonance") for step in output["trace"][:1]],
                "data_wave_modes": [step.get("data_wave", {}).get("mode_count") for step in output["trace"][:1]]}

    known = []
    known.append(answer("Hallo"))
    first = model.generate("Mir geht es gut, wie geht es dir denn?")
    known.append({"prompt": "Mir geht es gut, wie geht es dir denn?", "answer": first["text"],
                  "ended": first["ended"], "reason": first.get("reason", "")})
    second = model.generate("gut und dir?", context=first["state"])
    known.append({"prompt": "gut und dir? (session)", "answer": second["text"],
                  "ended": second["ended"], "reason": second.get("reason", "")})
    known.append(answer("Wie heißt du?"))
    freq = model.generate("Was ist eine Frequenz?")
    known.append({"prompt": "Was ist eine Frequenz?", "answer": freq["text"], "ended": freq["ended"],
                  "reason": freq.get("reason", "")})
    unit = model.generate("Und welche Einheit hat sie?", context=freq["state"])
    known.append({"prompt": "Und welche Einheit hat sie? (session)", "answer": unit["text"],
                  "ended": unit["ended"], "reason": unit.get("reason", "")})

    knowledge = json.loads((ROOT / "memory/evaluation/unpaired_knowledge_regression.json").read_text(encoding="utf-8"))
    knowledge_results = []
    for case in knowledge["cases"]:
        output = model.generate(case["prompt"])
        text = output["text"]
        concepts = case.get("concepts", [])
        hit_groups = sum(1 for group in concepts if any(term.casefold() in text.casefold() for term in group))
        knowledge_results.append({"id": case["id"], "prompt": case["prompt"], "answer": text,
                                  "ended": output["ended"], "reason": output.get("reason", ""),
                                  "concept_groups_hit": hit_groups, "concept_groups_total": len(concepts)})

    result = {"waves": waves_report, "known_api_probes": known, "knowledge_regression": knowledge_results}
    write("probe_results.json", result)
    print(json.dumps({"build_seconds": round(build_s, 1), "data_wave_modes": data_wave.mode_count,
                      "known": [row["answer"] for row in known]}, ensure_ascii=False, indent=1))


def randomfacts_part():
    from freqai.unpaired import UnpairedWaveModel

    spec = json.loads((ROOT / "memory/evaluation/information_randomfacts.json").read_text(encoding="utf-8"))
    records = [{"id": r["id"], "text": r["text"], "source": r["source"]} for r in spec["records"]]
    model = UnpairedWaveModel(records)
    results = []
    for case in spec["cases"]:
        output = model.generate(case["prompt"])
        text = output["text"].casefold()
        direction = case["direction"]
        if direction in {"entity_to_value", "value_to_entity"}:
            ok = str(case["expected_value"]) in text and case["expected_entity"].casefold() in text
        elif direction == "unknown":
            ok = not output["text"].strip()
        else:  # negation: false claim; the known system behaviour is to not correct it.
            ok = None
        results.append({"id": case["id"], "direction": direction, "prompt": case["prompt"],
                        "answer": output["text"], "ended": output["ended"], "ok": bool(ok) if ok is not None else None})
    scored = [row for row in results if row["ok"] is not None]
    summary = {"model_records": len(records), "cases": len(results),
               "scored": len(scored), "passed": sum(row["ok"] for row in scored),
               "negation_cases_not_corrected": sum(
                   1 for row in results if row["direction"] == "negation" and not row["answer"].strip())}
    write("randomfacts_results.json", {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["full", "randomfacts"], required=True)
    args = parser.parse_args()
    if args.part == "full":
        full_part()
    else:
        randomfacts_part()
