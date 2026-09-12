"""Festgehaltene Referenzwerte VOR dem Parallelumbau, Vergleichsbasis der Tests."""
from __future__ import annotations
import hashlib, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from freqai.store import DEFAULT_MEMORY_PATH, MemoryStore
from freqai.unpaired import UnpairedWaveModel

PROMPTS = ["Was ist eine Frequenz?", "Wie heißt du?", "Mit welcher Einheit gibt man die Frequenz an?"]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    documents = MemoryStore(DEFAULT_MEMORY_PATH).load_memory().documents
    records = [{"id": d.id, "text": d.text, "source": d.source} for d in documents][:1500]
    out = {}
    started = time.perf_counter()
    model = UnpairedWaveModel(records, order=2)
    build = time.perf_counter() - started
    out["build_s"] = round(build, 3)
    out["model_signature"] = str(model.model_signature())
    out["storage_stats"] = json.loads(json.dumps(model.storage_stats(), default=str))
    out["vocabulary_digest"] = digest(list(model.vocabulary))
    out["facts"] = len(model.facts)
    out["roles"] = len(model.lexical_roles)
    out["groups"] = len(model.groups)
    out["spectra_digest"] = digest({str(key): [list(map(int, s.token_indices)), [[round(float(v.real),12), round(float(v.imag),12)] for v in s.local_spectrum]]
                                    for key, s in list(model.transition_spectra.items())[:40]})
    out["symbol_energy_digest"] = digest([round(float(v), 12) for v in model.symbol_energy])
    out["answers"] = {}
    for prompt in PROMPTS:
        started = time.perf_counter()
        result = model.generate(prompt, max_tokens=24, decoding="beam", beam_width=4, seed=17)
        out["answers"][prompt] = {"text": result["text"], "tokens": result["tokens"],
                                  "ended": result["ended"], "s": round(time.perf_counter()-started, 3),
                                  "trace_digest": digest([[t.get("selected_token"), round(t.get("probability", 0.0), 12)]
                                                          for t in result["trace"]])}
    with open(os.path.join(PROJECT_ROOT, "results", "parallel_baseline.json"), "w", encoding="utf-8") as stream:
        json.dump(out, stream, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:1600])


if __name__ == "__main__":
    main()
