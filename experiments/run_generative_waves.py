"""Frozen evaluation for word-by-word generation from a spectral language model.

Automatic checks are explicitly proxies, not an oracle of sentence meaning.
The separately recorded reviewer decisions determine semantic quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUITE = ROOT / "memory/evaluation/generative_waves.json"
FIXTURE = ROOT / "memory/fixtures/extension_120.jsonl"
LANGUAGE = ROOT / "memory/language/generative_corpus.jsonl"
OUTPUT = ROOT / "results/generative_waves"
IMPLEMENTATION = ["freqai/generative.py", "freqai/generation_runtime.py", "freqai/semantics.py",
                  "freqai/features.py", "freqai/memory.py", "freqai/codec.py",
                  "memory/language/generative_categories.json"]
CONFIGURATION = {"max_tokens": 40, "seed": 17}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serializable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serializable) + "\n",
                    encoding="utf-8", newline="\n")


def normalized(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def suite() -> list[dict]:
    original = read(OUTPUT / "suite_freeze.json")
    assert digest(SUITE) == original["dataset_sha256"], "Frozen evaluation suite changed"
    groups = read(SUITE)["scenarios"]
    assert Counter(s["split"] for s in groups for _ in s["turns"]) == {"development": 30, "holdout": 30}
    all_prompts = [normalized(t["prompt"]) for s in groups for t in s["turns"]]
    assert len(set(all_prompts)) == 60
    return groups


def model_manifest() -> dict:
    runtime = importlib.import_module("freqai.generation_runtime") if (ROOT / "freqai/generation_runtime.py").exists() else None
    return {"configuration": CONFIGURATION,
            "runtime_model_configuration": dict(runtime.MODEL_CONFIGURATION) if runtime else None,
            "runtime_generation_configuration": dict(runtime.GENERATION_CONFIGURATION) if runtime else None,
            "files": {path: digest(ROOT / path) for path in IMPLEMENTATION if (ROOT / path).exists()},
            "paired_corpus_sha256": digest(FIXTURE), "unpaired_corpus_sha256": digest(LANGUAGE),
            "evaluation_sha256": digest(SUITE), "runner_sha256": digest(Path(__file__))}


def freeze() -> dict:
    manifest = model_manifest()
    assert "freqai/generative.py" in manifest["files"], "No generative implementation to freeze"
    assert "freqai/generation_runtime.py" in manifest["files"], "No generation runtime to freeze"
    destination = OUTPUT / "model_freeze.json"
    if destination.exists():
        assert read(destination)["manifest"] == manifest, "The frozen model, corpus, configuration or runner changed"
        return read(destination)
    value = {"created_utc": datetime.now(timezone.utc).isoformat(), "manifest": manifest,
             "holdout_seen_by_implementer": False}
    write(destination, value)
    return value


def automatic_checks(turn: dict, answer: str) -> dict:
    """Only observable textual proxies; no semantic metadata from the model."""
    text = answer.strip()
    form_errors = []
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if not text:
        form_errors.append("empty")
    if text and not re.search(r"[.!?][\s\"'”’»]*$", text):
        form_errors.append("unfinished_punctuation")
    if text and not next((char.isupper() for char in text if char.isalpha()), False):
        form_errors.append("lowercase_sentence_start")
    if "\ufffd" in text or any(ord(char) < 32 and char not in "\n\t" for char in text):
        form_errors.append("invalid_text_character")
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", text, flags=re.I):
        form_errors.append("immediate_token_loop")
    triples = Counter(tuple(word.casefold() for word in words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    if any(n >= 3 for n in triples.values()):
        form_errors.append("repeated_trigram_loop")
    lower = text.casefold()
    topic_hint = any(word.casefold() in lower for word in turn.get("require_any", []))
    contradictions = [pattern for pattern in turn.get("forbid_patterns", []) if re.search(pattern, text, re.I)]
    return {"surface_form_ok": not form_errors, "surface_form_errors": form_errors,
            "contains_rubric_hint": topic_hint, "forbidden_text_patterns": contradictions,
            "proxy_pass": not form_errors and topic_hint and not contradictions,
            "not_a_semantic_judgment": True}


def novelty(answer: str, corpus: list[str]) -> dict:
    complete = normalized(answer)
    existing = {normalized(text) for text in corpus}
    sentences = [normalized(s) for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if normalized(s)]
    known_sentences = {normalized(s) for text in corpus for s in re.split(r"(?<=[.!?])\s+", text.strip()) if normalized(s)}
    return {"new_complete_text": bool(complete) and complete not in existing,
            "sentence_count": len(sentences), "new_sentence_count": sum(s not in known_sentences for s in sentences),
            "all_sentences_already_in_corpus": bool(sentences) and all(s in known_sentences for s in sentences),
            "is_quality_measure": False}


def engine(baseline: bool = False):
    from freqai.memory import Document, WaveMemory
    doc_records = records(FIXTURE)
    if not baseline:
        doc_records += records(LANGUAGE)
    memory = WaveMemory([Document(**record) for record in doc_records], feature_mode="morphology",
                        retrieval_policy="coverage", min_coverage=.6)
    if baseline:
        responder = importlib.import_module("freqai.dialogue").respond
    else:
        responder = importlib.import_module("freqai.generation_runtime").respond_wave
    return memory, responder


def run(split: str, *, baseline: bool = False, variant: str = "default") -> tuple[Path, dict]:
    if variant != "default":
        assert split == "development" and not baseline, "Variants are development experiments only"
        runtime = importlib.import_module("freqai.generation_runtime")
        if variant == "sample":
            runtime.MODEL_CONFIGURATION["order"] = 3
            runtime.GENERATION_CONFIGURATION.update({"decoding": "sample", "temperature": .8})
        elif variant == "order2":
            runtime.MODEL_CONFIGURATION["order"] = 2
            runtime.GENERATION_CONFIGURATION["decoding"] = "beam"
        else:
            raise ValueError("Unknown variant")
    if split == "holdout" and not baseline:
        freeze()
        assert not (OUTPUT / "holdout.json").exists(), "Holdout already evaluated; do not tune and rerun it"
    groups = [scenario for scenario in suite() if scenario["split"] == split]
    memory, responder = engine(baseline)
    corpus = [doc.text for doc in memory.documents]
    rows = []
    for scenario in groups:
        context = None
        for turn in scenario["turns"]:
            before = copy.deepcopy(context)
            started = time.perf_counter()
            extra = {} if baseline else CONFIGURATION
            result, context = responder(memory, turn["prompt"], context=context, time_s=0.0, **extra)
            answer = result["answer"]
            rows.append({"id": turn["id"], "scenario": scenario["id"], "split": split,
                         "prompt": turn["prompt"], "rubric": turn["rubric"], "topics": turn["topics"],
                         "answer": answer, "result": result, "context_before": before,
                         "context_after": copy.deepcopy(context), "elapsed_s": time.perf_counter() - started,
                         "automatic": automatic_checks(turn, answer), "novelty": novelty(answer, corpus),
                         "manual_review": None})
    summary = {"turns": len(rows), "surface_form_ok": sum(r["automatic"]["surface_form_ok"] for r in rows),
               "contains_rubric_hint": sum(r["automatic"]["contains_rubric_hint"] for r in rows),
               "proxy_pass": sum(r["automatic"]["proxy_pass"] for r in rows),
               "explicit_contradictions": sum(bool(r["automatic"]["forbidden_text_patterns"]) for r in rows),
               "new_complete_texts": sum(r["novelty"]["new_complete_text"] for r in rows),
               "texts_with_new_sentences": sum(r["novelty"]["new_sentence_count"] > 0 for r in rows),
               "semantic_pass": "pending independent review"}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "split": split,
              "baseline": baseline, "variant": variant, "manifest": model_manifest(), "summary": summary, "rows": rows}
    if baseline:
        path = OUTPUT / ("baseline_" + split + ".json")
        assert not path.exists(), "Do not overwrite the recorded baseline"
    elif split == "holdout":
        path = OUTPUT / "holdout.json"
    else:
        variant_prefix = "" if variant == "default" else variant + "_"
        path = OUTPUT / ("development_" + variant_prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write(path, result)
    return path, result


def compile_report() -> dict:
    """Combine immutable runs and independently supplied per-answer reviews."""
    reviews = []
    for name in ("baseline_review.json", "development_review.json", "holdout_review.json"):
        review_path = OUTPUT / name
        if not review_path.exists():
            continue
        review = read(review_path)
        run_path = ROOT / review["run_path"]
        assert digest(run_path) == review["run_sha256"], "Reviewed run changed"
        run_data = read(run_path)
        judgments = review["judgments"]
        assert set(judgments) == {row["id"] for row in run_data["rows"]}, "Review must cover every answer"
        for row in run_data["rows"]:
            judgment = judgments[row["id"]]
            assert all(type(judgment[key]) is bool for key in ("grammar", "relevance", "coherence", "roles_negation"))
            row["manual_review"] = judgment
            row["semantic_pass"] = all(judgment[key] for key in ("grammar", "relevance", "coherence", "roles_negation"))
        run_data["summary"]["semantic_pass"] = sum(row["semantic_pass"] for row in run_data["rows"])
        run_data["summary"]["manual_dimensions"] = {
            key: sum(row["manual_review"][key] for row in run_data["rows"])
            for key in ("grammar", "relevance", "coherence", "roles_negation")}
        reviews.append({"review": name, "reviewer": review["reviewer"], "run_path": review["run_path"], "run": run_data})
    assert reviews, "No independent reviews have been recorded"
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "reviews": reviews,
              "evaluation_policy": read(SUITE)["review_policy"]}
    write(OUTPUT / "reviewed_results.json", result)
    lines = ["# Wortweise Textgenerierung aus einem spektralen Übergangsmodell", "",
             "60 neue Gesprächseingaben wurden vor der Implementierungsbewertung festgelegt: "
             "30 Entwicklungs- und 30 zurückgehaltene Turns. Die Antworten werden einzeln auf "
             "Grammatik, Bezug, Kohärenz und korrekte Sprecherrollen/Verneinungen geprüft. "
             "Eine neue Tokenfolge ist kein Qualitätsnachweis.", "",
             "| Lauf | Sprachform (automatisch) | Text-Hinweise (automatisch) | Sinnvolle Antwort (Review) | Vollständiger Text neu |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for item in reviews:
        run_data = item["run"]
        summary = run_data["summary"]
        label = ("Bisherige Regeln, " if run_data["baseline"] else "Wellendecoder, ") + run_data["split"]
        n = summary["turns"]
        lines.append(f"| {label} | {summary['surface_form_ok']}/{n} | {summary['proxy_pass']}/{n} | "
                     f"{summary['semantic_pass']}/{n} | {summary['new_complete_texts']}/{n} |")
    lines += ["", "Die automatischen Sprachformtests prüfen nur Textoberfläche, Satzabschluss und Schleifen. "
              "Die Text-Hinweise prüfen wenige Wörter und ausdrücklich verbotene Formulierungen. "
              "Beide können unpassende oder widersprüchliche Sätze übersehen. Die Review-Urteile "
              "sind qualitative Einzelfallurteile, keine unabhängige Nutzerstudie.", "",
              "Neuheit bezieht sich auf die vollständigen Textdaten des jeweiligen Laufs. "
              "Beim bisherigen Regelmodell sind externe Grammatikfragmente nicht in diesem "
              "Vergleichskorpus enthalten; dessen Neuheitszahl ist deshalb nicht mit freier "
              "Sprachgenerierung gleichzusetzen.", "", "## Einzelbewertungen", ""]
    for item in reviews:
        run_data = item["run"]
        lines += ["### " + ("Bisherige Regeln: " if run_data["baseline"] else "Wellendecoder: ") + run_data["split"], ""]
        for row in run_data["rows"]:
            status = "bestanden" if row["semantic_pass"] else "nicht bestanden"
            lines += [f"- **{row['id']} ({status})** — {row['prompt']}",
                      "  Antwort: " + row["answer"].replace("\n", " "),
                      "  Bewertung: " + row["manual_review"]["note"]]
        lines += [""]
    lines += ["## Daten und Grenzen", "",
              "Der zusätzliche Sprachkorpus enthält 500 synthetische Kombinationen aus 200 "
              "ausdrücklich verfassten Sätzen. Das sind keine 500 unabhängig beobachteten "
              "Gespräche. Übergangszählungen sind datenabhängige Schätzung; der Verzicht "
              "auf Gradienten oder ein vortrainiertes Modell beseitigt diesen Sprachprior nicht.", "",
              "Der Algorithmus kann neue Tokenpfade bilden. Ob daraus passende Aussagen "
              "entstehen, hängt von den Daten, der Konditionierung und der Decodierung ab. "
              "Eine Fouriertransformation liefert aus sich heraus keine Semantik oder neues Wissen.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {item["review"]: item["run"]["summary"] for item in reviews}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["development", "freeze", "holdout", "baseline", "report", "sample", "order2"])
    arguments = parser.parse_args()
    if arguments.phase == "report":
        print(json.dumps(compile_report(), ensure_ascii=False, indent=2))
    elif arguments.phase == "freeze":
        print(json.dumps(freeze(), indent=2))
    else:
        split = "development" if arguments.phase in {"baseline", "sample", "order2"} else arguments.phase
        variant = arguments.phase if arguments.phase in {"sample", "order2"} else "default"
        path, result = run(split, baseline=arguments.phase == "baseline", variant=variant)
        print(json.dumps({"path": str(path), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
