"""Second dialogue iteration with a new holdout and immutable round1 baseline.

All 134 previous cases are development data now. The additional 80 cases are
split by conversation into 40 development and 40 newly withheld turns. This
runner never writes into round1 or overwrites the original result files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_dialogue as audit

ARCHIVE = ROOT / "results/dialogue/round1"
OUTPUT = ROOT / "results/dialogue/round2"
CASES = ROOT / "memory/evaluation/dialogue_generalization_v2.json"
PREVIOUS = ARCHIVE / "memory/evaluation/dialogue_generalization.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_archive() -> dict:
    manifest = read(ARCHIVE / "manifest.json")
    for name, digest in manifest["files"].items():
        assert audit.sha256(ARCHIVE / name) == digest, f"Round1 archive changed: {name}"
    return manifest


def scenarios() -> list[dict]:
    previous = read(PREVIOUS)["scenarios"]
    for scenario in previous:
        scenario["previous_split"] = scenario["split"]
        scenario["split"] = "development"
        scenario["origin"] = "round1_reused"
    new = read(CASES)["scenarios"]
    old_prompts = {t["prompt"] for s in previous for t in s["turns"]}
    new_prompts = [t["prompt"] for s in new for t in s["turns"]]
    assert len(new_prompts) == len(set(new_prompts)) == 80
    assert not (old_prompts & set(new_prompts)), "New cases overlap old prompts"
    for scenario in new:
        scenario["origin"] = "round2_new"
    all_scenarios = previous + new
    ids = [t["id"] for s in all_scenarios for t in s["turns"]]
    assert len(ids) == len(set(ids)) == 214
    return all_scenarios


def freeze_suite(groups: list[dict]) -> dict:
    verify_archive()
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "new_suite_sha256": audit.sha256(CASES), "previous_suite_sha256": audit.sha256(PREVIOUS),
                "corpus_sha256": audit.sha256(audit.CORPUS),
                "round1_manifest_sha256": audit.sha256(ARCHIVE / "manifest.json"),
                "new_turns": 80, "reused_development_turns": 134,
                "split_counts": dict(Counter(s["split"] for s in groups for _ in s["turns"]))}
    path = OUTPUT / "suite_freeze.json"
    if path.exists():
        old = read(path)
        assert {k: v for k, v in old.items() if k != "created_utc"} == {
            k: v for k, v in metadata.items() if k != "created_utc"}, "Round2 suite changed"
        return old
    audit._write(path, metadata)
    return metadata


def archived_engine():
    name = "freqai_dialogue_round1"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ARCHIVE / "freqai/__init__.py",
                                                     submodule_search_locations=[str(ARCHIVE / "freqai")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    memory_module = importlib.import_module(name + ".memory")
    dialogue_module = importlib.import_module(name + ".dialogue")
    docs = [memory_module.Document(**json.loads(line)) for line in audit.CORPUS.read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    return memory_module.WaveMemory(docs, **audit.CONFIG), dialogue_module.respond


def evaluate(expected: dict, result: dict, before: dict | None, after: dict | None) -> list[str]:
    """V1 meaning rubric with documented corrections based only on V1 review."""
    failures = audit.evaluate_answer(expected, result, before, after)
    answer = audit._norm(result["answer"])
    acts = result.get("interpretation", {}).get("acts", [])
    concepts = expected["concepts"]
    if "clarification" in concepts and re.search(r"mehr.{0,12}kontext|kontext.{0,12}(geben|fehlt)", answer):
        failures = [f for f in failures if f != "answer_missing:clarification"]
    # A negated negative adjective does not imply a positive feeling. Neutral
    # acknowledgement is valid if the correct scoped negation was interpreted.
    neutral_negation = any(a.get("kind") == "mood_statement" and a.get("target") == "user"
                          and a.get("negated") and a.get("value") == "neutral" for a in acts)
    if neutral_negation and re.search(r"einordnung|nicht|verstanden|danke|klar", answer):
        failures = [f for f in failures if f not in {"answer_missing:positive", "answer_missing:not_tired"}]
    explicit_negative = re.search(r"\bnicht\s+(?:glücklich|zufrieden|fröhlich|wohl|prima|super)\b", answer)
    if explicit_negative and any(a.get("kind") == "mood_statement" and a.get("target") == "user"
                                 and a.get("negated") and a.get("value") == "negative" for a in acts):
        failures = [f for f in failures if f != "answer_missing:negative"]
    if "third_positive" in concepts and audit._norm(expected["subject"]) in answer:
        if re.search(r"\bnicht\s+(?:traurig|schlecht|mies|unglücklich)\b", answer) and any(
                a.get("kind") == "mood_statement" and a.get("target") == "other"
                and a.get("negated") and a.get("value") == "neutral" for a in acts):
            failures = [f for f in failures if f != "answer_missing:third_positive"]
    # Pronouns can preserve third-person attribution without repeating a name.
    third_reference = re.search(r"\b(ihm|ihr|ihn|diese[rn]?|deine[mrn]?)\b", answer)
    if third_reference:
        equivalents = {"third_positive": {"positive", "neutral"}, "third_negative": {"negative", "stressed", "lonely"},
                       "third_tired": {"tired"}, "third_not_tired": {"neutral"}}
        for concept, values in equivalents.items():
            if concept in concepts and any(a.get("kind") == "mood_statement" and a.get("target") == "other"
                                            and a.get("value") in values for a in acts):
                relaxed = {**expected, "subject": next(m.group(0) for m in re.finditer(
                    r"\b(ihm|ihr|ihn|diese[rn]?|deine[mrn]?)\b", answer)), "concepts": [concept]}
                if not audit.evaluate_answer(relaxed, result, before, after):
                    failures = [f for f in failures if f != "answer_missing:" + concept]
    if expected.get("forbid_positive") and any(a.get("kind") == "mood_statement" and a.get("target") == "user"
                                               and a.get("value") == "positive" for a in acts):
        failures.append("wrong_mood:positive")
    return failures


def run_cases(memory, responder, groups: list[dict], time_s: float = 0) -> list[dict]:
    rows = []
    corpus_answers = {d.text for d in memory.documents}
    for scenario in groups:
        context = None
        for turn in scenario["turns"]:
            before = context.copy() if context else None
            started = time.perf_counter()
            result, context = responder(memory, turn["prompt"], context=context, time_s=time_s)
            elapsed = time.perf_counter() - started
            failures = evaluate(turn["expected"], result, before, context)
            decoder = result.get("decoder", {})
            checksum = hashlib.sha256(result["answer"].encode("utf-8")).hexdigest()
            if decoder.get("output_sha256") != checksum:
                failures.append("decoder_checksum_mismatch")
            rows.append({"id": turn["id"], "scenario": scenario["id"], "split": scenario["split"],
                         "origin": scenario["origin"], "previous_split": scenario.get("previous_split"),
                         "prompt": turn["prompt"], "expected": turn["expected"], "passed": not failures,
                         "failures": failures, "answer": result["answer"], "result": result,
                         "context_before": before, "context_after": context.copy() if context else None,
                         "novel_complete_answer": result["answer"] not in corpus_answers and not result.get("abstained"),
                         "elapsed_s": elapsed, "time_s": time_s})
    return rows


def source_audit(memory, rows: list[dict]) -> dict:
    language = read(ROOT / "memory/language/german.json")
    sources = {"memory:" + d.id: d.text for d in memory.documents}
    sources.update({"grammar:" + key: value for key, value in language["fragments"].items()})
    sources.update({"topic:" + key: value for key, value in language["topics"].items()})
    failures = []
    for row in rows:
        try:
            decoder = row["result"]["decoder"]
            assert decoder["source_integrity_checked"] is True
            assert decoder["method"] == "linear_spectral_composition"
            payloads = []
            for fragment in decoder["sources"]:
                source = fragment["source"]
                if source in sources:
                    candidates = [sources[source]]
                elif source == "input:prompt":
                    candidates = [row["prompt"]]
                elif source == "input:name":
                    candidates = [a["value"] for a in row["result"]["interpretation"]["acts"] if a["kind"] == "name_statement"]
                elif source == "context:name":
                    candidates = [(row["context_before"] or {}).get("user_name", ""),
                                  (row["context_after"] or {}).get("user_name", "")]
                else:
                    raise AssertionError(f"Unknown declared source: {source}")
                matching = [text.encode("utf-8") for text in candidates
                            if hashlib.sha256(text.encode("utf-8")).hexdigest() == fragment["source_sha256"]]
                assert matching, f"Unverified source: {source}"
                payloads.append(matching[0][fragment["start"]:fragment["stop"]])
            expected = b"".join(payloads)
            assert expected == row["answer"].encode("utf-8")
            assert hashlib.sha256(expected).hexdigest() == decoder["output_sha256"]
        except (AssertionError, KeyError, TypeError) as error:
            failures.append({"id": row["id"], "error": str(error)})
    return {"checked": len(rows), "passed": len(rows) - len(failures), "failures": failures}


def time_invariance(memory, responder, groups, rows) -> dict:
    checked = 0
    failures = []
    for time_s in (.013, 1.25, 86400.125, 1e12):
        for original, alternative in zip(rows, run_cases(memory, responder, groups, time_s)):
            checked += 1
            if original["answer"] != alternative["answer"] or original["context_after"] != alternative["context_after"]:
                failures.append({"id": original["id"], "time_s": time_s})
    return {"checked": checked, "passed": checked - len(failures), "failures": failures}


def implementation() -> dict:
    return {**audit._implementation_hashes(), "experiments/run_dialogue.py": audit.sha256(Path(audit.__file__)),
            "experiments/run_dialogue_v2.py": audit.sha256(Path(__file__))}


def write_report(final: dict, baseline: dict) -> None:
    lines = ["# Dialogberechnung, zweite Runde", "",
             "Die erste Runde liegt vollständig und unverändert unter `../round1/`; "
             "ihr damaliger Holdout ist jetzt ausschließlich Entwicklungsdaten. Zusätzlich "
             "wurden vor der zweiten Anpassung 80 neue Eingaben in getrennten Szenarien erstellt: "
             "40 zur Entwicklung und 40 als neuer Holdout. Insgesamt enthält Runde 2 damit 214 Turns.", "",
             "| Teil | Archiviertes V1-Modell | V2-Modell |", "|---|---:|---:|"]
    for split, item in final["summary"].items():
        old = baseline["summary"][split]
        lines.append(f"| {split} | {old['passed']}/{old['total']} | {item['passed']}/{item['total']} |")
    novel = [r["answer"] for r in final["rows"] if r["novel_complete_answer"]]
    lines += ["", f"{len(novel)} Antwortturns enthalten {len(set(novel))} verschiedene vollständige Antworten, "
              "die nicht wörtlich im unveränderten 120er-Korpus stehen. Wiederholung von Eingabeabschnitten "
              "und Kombination vorgegebener Grammatik zählen dabei ebenfalls als neue vollständige Texte; "
              "das ist kein Nachweis freier Sprachgenerierung.", "",
              f"Zeitinvarianz: {final['time_invariance']['passed']}/{final['time_invariance']['checked']}; "
              f"unabhängige Quellen-/Span-/Prüfsummenprüfung: {final['decoder_provenance']['passed']}/"
              f"{final['decoder_provenance']['checked']}. Zusätzliche UTF-8-Rückrechnung: "
              f"{final['numerical_audit']['text_roundtrip_passed']}/{final['numerical_audit']['text_roundtrip_total']}.", "",
              "Die ausführbare Rubrik prüft Bedeutungsmerkmale und Rollen, keinen exakten Antworttext. "
              "Sie wurde aufgrund der ersten Runde vor dem neuen Holdout korrigiert: eine Bitte um mehr "
              "Kontext gilt als Rückfrage, eine korrekt interpretierte doppelte/negative Verneinung darf "
              "neutral bestätigt werden, und eindeutige Pronomen dürfen Namen ersetzen. "
              "Das bleibt ein begrenztes automatisches Prüfraster und keine universelle Dialoggenauigkeit.", "",
              "## Verbleibende Fehler", ""]
    for row in final["rows"]:
        if not row["passed"]:
            lines.append(f"- `{row['id']}`: {row['prompt']} → {row['answer']} ({', '.join(row['failures'])})")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("baseline", "development", "final"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    groups = scenarios()
    suite = freeze_suite(groups)
    if args.stage == "development":
        groups = [s for s in groups if s["split"] == "development"]
    if args.stage == "baseline":
        memory, responder = archived_engine()
    else:
        _, memory = audit.load_inputs()
        from freqai.dialogue import respond as responder
    if args.stage == "final":
        if not args.freeze:
            parser.error("Final requires --freeze")
        freeze = {"created_utc": datetime.now(timezone.utc).isoformat(), "implementation": implementation(), "suite": suite}
        path = OUTPUT / "implementation_freeze.json"
        if path.exists():
            assert read(path)["implementation"] == freeze["implementation"], "Post-holdout V2 code changed"
        else:
            audit._write(path, freeze)
    rows = run_cases(memory, responder, groups)
    report = {"round": 2, "stage": args.stage, "created_utc": datetime.now(timezone.utc).isoformat(),
              "suite": suite, "implementation": implementation(), "summary": audit.summarize(rows), "rows": rows}
    if args.stage == "final":
        report["time_invariance"] = time_invariance(memory, responder, groups, rows)
        report["numerical_audit"] = audit.numerical_audit(memory)
        report["decoder_provenance"] = source_audit(memory, rows)
    audit._write(OUTPUT / f"{args.stage}.json", report)
    if args.stage == "development":
        index = len(list(OUTPUT.glob("development_round_*.json"))) + 1
        audit._write(OUTPUT / f"development_round_{index:03d}.json", report)
    if args.stage == "final":
        write_report(report, read(OUTPUT / "baseline.json"))
    print(json.dumps({"round": 2, "stage": args.stage, "suite": suite, "summary": report["summary"],
                      "time_invariance": report.get("time_invariance"),
                      "decoder_provenance": report.get("decoder_provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
