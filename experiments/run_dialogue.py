"""Independent finite dialogue audit; no corpus mutation or parameter fitting.

Run baseline, development, then final --freeze. The frozen holdout and stress
prompts must not be inspected to tune the implementation before final.
The regex oracle checks meaning-related answer properties, not exact templates;
it is deliberately a limited executable rubric, not a language-understanding
judge. Raw outputs and failed assertions are preserved for human review.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqai.memory import Document, WaveMemory

CASES = ROOT / "memory/evaluation/dialogue_generalization.json"
CORPUS = ROOT / "memory/fixtures/extension_120.jsonl"
OUTPUT = ROOT / "results/dialogue"
CONFIG = {"dimensions": 4096, "feature_mode": "morphology", "min_score": .18,
          "retrieval_policy": "coverage", "min_coverage": .6}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[dict, WaveMemory]:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    documents = [Document(**json.loads(line)) for line in CORPUS.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
    ids = [turn["id"] for scenario in data["scenarios"] for turn in scenario["turns"]]
    assert len(ids) == len(set(ids)), "Duplicate case IDs"
    assert len(documents) == 120, "Evaluation must keep the original conversation corpus"
    assert len({s["id"] for s in data["scenarios"]}) == len(data["scenarios"])
    assert {s["split"] for s in data["scenarios"]} == {"development", "holdout", "stress"}
    return data, WaveMemory(documents, **CONFIG)


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("’", "'")


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def evaluate_answer(expected: dict, result: dict, context_before: dict | None = None,
                    context_after: dict | None = None) -> list[str]:
    """Return failed observable semantic requirements, including conjunctions.

    Answer text is mandatory: the model cannot pass merely by claiming that it
    realized an act in metadata. Role and name-update checks also use structured
    state because surface words alone cannot establish correct attribution.
    """
    answer = _norm(str(result.get("answer", "")))
    acts = set(result.get("response_acts", []))
    failed = []
    positive = r"freu|schön|prima|super|positiv|zufrieden|wohl|okay|in ordnung|nicht schlecht|neutral|gut.{0,20}(geht|fühl)|geht.{0,20}gut|klingt.{0,20}gut|erleichter"
    negative = r"leid|schwer|belast|unangenehm|nicht gut|nicht leicht|schade|schlecht|schwierig|traurig|mies|anstreng|drückt|gedrückt|sorgen|stress|gereiz|bedauer"
    tired = r"müde|mued|erschöpf|erschoepf|ruhe|ruhig|pause|anstreng|ausruh|erhol|kraft"
    assistant = r"bereit|\bki\b|künstlich|kuenstlich|keine.{0,35}gefühle|keine.{0,35}gefuehle|helfen|unterstütz|funktionier|ich.{0,25}(hier|da).{0,30}(dich|gespräch)"
    clarification = r"formulier|verstand|versteh|meinst|bezieh|worum|genauer|erklär|klären|unklar|präzis|welches thema|nicht.{0,20}sicher"
    unknown = r"weiß.{0,35}nicht|weiss.{0,35}nicht|kenne.{0,35}nicht|keine.{0,40}(information|wissen|daten|antwort|text|eintrag|grundlage)|nicht.{0,30}(gespeichert|bekannt|wissen|beantwort|ableit|herleit|sicher)|kann.{0,35}nicht|kannst.{0,30}(erklär|beschreib)|dazu.{0,40}keine"
    unknown_name = r"(name|namen).{0,60}(nicht|noch keinen)|nicht.{0,60}(name|namen)|noch keinen.{0,25}namen|wie.{0,30}(heißt|heisst)|wie.{0,30}angesprochen|nenn.{0,20}namen"
    patterns = {
        "greeting": r"hallo|\bhi\b|\bhey\b|huhu|servus|moin|guten (morgen|tag|abend)|willkommen",
        "positive": positive, "negative": negative, "tired": tired,
        "not_tired": r"wach|ausgeruht|energie|fit|nicht müde|schön|freu",
        "assistant_status": assistant,
        "thanks": r"gern|danke|bitteschön|bitte schön|kein problem|nichts zu danken|freut mich",
        "farewell": r"tschüss|tschuess|bis (bald|dann|später|spaeter)|mach.s gut|auf wiedersehen|schönen.{0,15}(tag|abend)",
        "clarification": clarification, "unknown": unknown + "|" + clarification,
        "unknown_name": unknown_name,
        "unknown_reason": unknown + "|" + clarification + r"|kann.{0,35}(sagen|erklären)|was.{0,30}(passiert|beigetragen)",
        "uncertain": r"unsicher|unklar|nicht.{0,20}(sicher|genau)|benennen|bemerkst|fühlst|versteh|genauer|widersprüch|beides",
        "conditional": clarification + "|" + unknown + r"|wenn|beding|hypothet",
        "third_positive": positive, "third_negative": negative, "third_tired": tired,
        "third_not_tired": r"wach|ausgeruht|energie|fit|nicht müde",
    }
    for concept in expected["concepts"]:
        if concept in {"introduce_name", "recall_name"}:
            ok = _norm(expected.get("name", "\x00")) in answer
        elif concept == "third_person":
            ok = "user_mood" not in acts and bool(answer)
        else:
            ok = _has(answer, patterns[concept])
        if concept.startswith("third_") and concept != "third_person":
            ok = ok and _norm(expected["subject"]) in answer
        if not ok:
            failed.append(f"answer_missing:{concept}")
    interpretation = result.get("interpretation")
    if isinstance(interpretation, dict):
        semantic_acts = interpretation.get("acts", [])
        moods = {"positive": {"positive", "neutral", "excited"},
                 "negative": {"negative", "stressed", "lonely"}, "tired": {"tired"}}
        for concept in expected["concepts"]:
            if concept in moods and not any(a.get("kind") == "mood_statement"
                                            and a.get("target") == "user"
                                            and a.get("value") in moods[concept]
                                            for a in semantic_acts):
                failed.append(f"interpretation_missing:user_{concept}")
            if concept in moods and "user_mood" not in acts:
                failed.append(f"response_act_missing:user_{concept}")
    # Explicit role evidence must not turn another person's report into a claim
    # about the user. Lack of metadata in baseline is supplemented by phrasing.
    if expected.get("forbid_user_mood") and ("user_mood" in acts or _has(
            answer, r"dass es dir.{0,25}(gut|schlecht) geht|dass du.{0,20}(traurig|müde|zufrieden) bist")):
        failed.append("wrong_subject:user")
    if expected.get("forbid_negative") and _has(answer, negative):
        failed.append("wrong_mood:negative")
    for name in expected.get("forbidden_names", []):
        if _has(answer, rf"\b{re.escape(_norm(name))}\b"):
            failed.append(f"leaked_or_wrong_name:{name}")
    if expected.get("forbid_name_update") and context_after is not None:
        if (context_before or {}).get("user_name") != context_after.get("user_name"):
            failed.append("wrong_name_update")
    if "recall_name" in expected["concepts"] and context_after is not None:
        if _norm(str(context_after.get("user_name", ""))) != _norm(expected["name"]):
            failed.append("name_missing_in_session_state")
    if any(c in {"greeting", "positive", "negative", "tired", "assistant_status", "recall_name"}
           for c in expected["concepts"]) and result.get("abstained"):
        failed.append("unnecessary_abstention")
    if not answer.strip():
        failed.append("empty_answer")
    return failed


def run_cases(memory: WaveMemory, scenarios: list[dict], baseline: bool = False,
              time_s: float = 0.0) -> list[dict]:
    if not baseline:
        from freqai.dialogue import respond
    rows = []
    corpus_answers = {d.text for d in memory.documents}
    for scenario in scenarios:
        context = None  # Every scenario is an independent session, including names.
        for turn in scenario["turns"]:
            before = context.copy() if context is not None else None
            started = time.perf_counter()
            if baseline:
                result = memory.ask(turn["prompt"], time_s=time_s)
            else:
                result, context = respond(memory, turn["prompt"], context=context, time_s=time_s)
            elapsed = time.perf_counter() - started
            failures = evaluate_answer(turn["expected"], result, before, context)
            decoder = result.get("decoder", {})
            checksum = hashlib.sha256(result["answer"].encode("utf-8")).hexdigest()
            novel = result["answer"] not in corpus_answers and not result.get("abstained", False)
            if decoder.get("output_sha256") is not None and decoder["output_sha256"] != checksum:
                failures.append("decoder_checksum_mismatch")
            rows.append({"id": turn["id"], "scenario": scenario["id"], "split": scenario["split"],
                         "prompt": turn["prompt"], "expected": turn["expected"], "passed": not failures,
                         "failures": failures, "answer": result["answer"], "result": result,
                         "context_before": before, "context_after": context.copy() if context else None,
                         "novel_complete_answer": novel, "elapsed_s": elapsed, "time_s": time_s})
    return rows


def summarize(rows: list[dict]) -> dict:
    result = {}
    for split in sorted({r["split"] for r in rows}):
        group = [r for r in rows if r["split"] == split]
        result[split] = {"passed": sum(r["passed"] for r in group), "total": len(group),
                         "novel_complete_answers": sum(r["novel_complete_answer"] for r in group),
                         "abstentions": sum(r["result"].get("abstained", False) for r in group),
                         "failures": dict(Counter(f for r in group for f in r["failures"]))}
    return result


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _implementation_hashes() -> dict:
    sources = [*(ROOT / "freqai").glob("*.py"), *(ROOT / "memory/language").glob("*.json")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)}


def _freeze_suite(data: dict, memory: WaveMemory) -> dict:
    evaluation_prompts = [t["prompt"] for s in data["scenarios"] for t in s["turns"]]
    corpus_prompts = {d.prompt for d in memory.documents}
    stats = {"created_utc": datetime.now(timezone.utc).isoformat(), "suite_sha256": sha256(CASES),
             "corpus_sha256": sha256(CORPUS), "turns": len(evaluation_prompts),
             "unique_prompts": len(set(evaluation_prompts)),
             "new_unique_prompts": len(set(evaluation_prompts) - corpus_prompts),
             "scenarios": len(data["scenarios"]),
             "split_counts": dict(Counter(s["split"] for s in data["scenarios"]
                                         for _ in s["turns"]))}
    assert stats["new_unique_prompts"] >= 100
    file = OUTPUT / "suite_freeze.json"
    if file.exists():
        previous = json.loads(file.read_text(encoding="utf-8"))
        assert previous["suite_sha256"] == stats["suite_sha256"], "Frozen evaluation cases changed"
        assert previous["corpus_sha256"] == stats["corpus_sha256"], "Conversation corpus changed"
        return previous
    _write(file, stats)
    return stats


def time_invariance(memory: WaveMemory, scenarios: list[dict], reference: list[dict]) -> dict:
    times = (.013, 1.25, 86400.125, 1e12)
    failures = []
    checked = 0
    for time_s in times:
        for original, alternate in zip(reference, run_cases(memory, scenarios, time_s=time_s)):
            checked += 1
            if original["answer"] != alternate["answer"]:
                failures.append({"id": original["id"], "time_s": time_s, "kind": "answer_changed"})
            if original["context_after"] != alternate["context_after"]:
                failures.append({"id": original["id"], "time_s": time_s, "kind": "context_changed"})
    failed_comparisons = len({(item["id"], item["time_s"]) for item in failures})
    return {"checked": checked, "times_s": list(times), "passed": checked - failed_comparisons,
            "failures": failures,
            "note": "Equal symbolic answers across modal time; not an empirical physical-wave experiment."}


def numerical_audit(memory: WaveMemory) -> dict:
    """Check the arithmetic independently of language-rule success/failure."""
    import itertools
    import numpy as np
    from freqai.codec import decode_text, phase_angles
    from freqai.semantics import SemanticAct, semantic_spectrum

    concepts = [SemanticAct("mood_statement", target=target, value=value, topic="wellbeing", negated=negated)
                for target, value, negated in itertools.product(
                    ("user", "assistant", "other"),
                    ("positive", "negative", "neutral", "tired", "stressed", "lonely", "excited"),
                    (False, True))]
    spectra = np.stack([semantic_spectrum(act) for act in concepts])
    direct = np.array([[sum(getattr(a, k) == getattr(b, k)
                           for k in ("kind", "target", "value", "topic", "negated")) / 5
                        for b in concepts] for a in concepts])
    spectral = (spectra.conj() @ spectra.T).real
    waveform = np.fft.ifft(spectra, axis=1, norm="ortho")
    temporal = (waveform.conj() @ waveform.T).real
    inverse = np.fft.fft(waveform, axis=1, norm="ortho")
    phase_errors = []
    for time_s in (.013, 1.25, 86400.125, 1e12):
        phase = np.exp(1j * phase_angles(spectra.shape[1], time_s))
        rotated = spectra * phase
        phase_errors.append(float(np.max(np.abs((rotated.conj() @ rotated.T).real - spectral))))
    roundtrips = sum(decode_text(packet, time_s) == doc.text
                     for packet, doc in zip(memory.payloads, memory.documents)
                     for time_s in (0, .013, 1.25, 86400.125, 1e12))
    metrics = {"concepts": len(concepts), "pair_comparisons": len(concepts) ** 2,
               "max_symbolic_vs_spectral_error": float(np.max(np.abs(spectral - direct))),
               "max_parseval_error": float(np.max(np.abs(spectral - temporal))),
               "max_inverse_fft_error": float(np.max(np.abs(inverse - spectra))),
               "max_common_phase_error": max(phase_errors),
               "text_roundtrip_passed": roundtrips, "text_roundtrip_total": 5 * len(memory.documents),
               "note": "Different complete concepts need not be orthogonal: they share role components. "
                       "Known role/value slots are disjoint; their weighted overlaps are authored semantics. "
                       "Fourier arithmetic preserves that representation and does not discover its meanings."}
    assert all(metrics[key] < 1e-12 for key in (
        "max_symbolic_vs_spectral_error", "max_parseval_error", "max_inverse_fft_error", "max_common_phase_error"))
    assert roundtrips == metrics["text_roundtrip_total"]
    return metrics


def decoder_provenance_audit(memory: WaveMemory, rows: list[dict]) -> dict:
    """Independently rebuild checksums from declared source spans, not outputs."""
    language = json.loads((ROOT / "memory/language/german.json").read_text(encoding="utf-8"))
    sources = {"memory:" + d.id: d.text for d in memory.documents}
    sources.update({"grammar:" + key: value for key, value in language["fragments"].items()})
    sources.update({"topic:" + key: value for key, value in language["topics"].items()})
    failures = []
    for row in rows:
        result = row["result"]
        decoder = result.get("decoder", {})
        try:
            assert decoder.get("source_integrity_checked") is True
            assert decoder.get("method") == "linear_spectral_composition"
            parts = []
            for fragment in decoder.get("sources", []):
                source = fragment["source"]
                if source in sources:
                    candidates = [sources[source]]
                elif source == "input:name":
                    candidates = [act["value"] for act in result["interpretation"]["acts"]
                                  if act["kind"] == "name_statement"]
                elif source == "context:name":
                    candidates = [(row["context_before"] or {}).get("user_name", ""),
                                  (row["context_after"] or {}).get("user_name", "")]
                else:
                    raise AssertionError(f"Unrecognized source {source}")
                matching = [value.encode("utf-8") for value in candidates
                            if hashlib.sha256(value.encode("utf-8")).hexdigest() == fragment["source_sha256"]]
                assert matching, f"Source checksum does not match declared source {source}"
                parts.append(matching[0][fragment["start"]:fragment["stop"]])
            expected = b"".join(parts)
            assert expected == result["answer"].encode("utf-8"), "Source spans do not reconstruct answer"
            assert hashlib.sha256(expected).hexdigest() == decoder["output_sha256"]
        except (AssertionError, KeyError, TypeError) as error:
            failures.append({"id": row["id"], "error": str(error)})
    return {"checked": len(rows), "passed": len(rows) - len(failures), "failures": failures,
            "note": "Independent source-span, source-hash and output-hash validation. "
                    "This validates provenance; synthesis tests separately check actual inverse transforms."}


def write_report(final: dict, baseline: dict) -> None:
    lines = ["# Begrenzte Generalisierung im Dialog", "",
             "134 handgeschriebene deutsche Eingaben in 26 getrennten Gesprächsszenarien; "
             "der Speicher behält die ursprünglichen 120 Gesprächspaare. "
             "Die Entwickler sahen vor dem Freeze nur Development-Ergebnisse. "
             "Holdout und Stress wurden erst danach ausgewertet.", "",
             "Die Prüfung bewertet Antwortbedeutungsmerkmale mit regulären Ausdrücken, "
             "mehrere gleichzeitig verlangte Aussagen, Rollen und isolierte Namenszustände. "
             "Sie vergleicht keine vollständigen Antwortvorlagen. Dennoch ist das ein "
             "begrenztes automatisches Prüfraster; menschliche Bedeutungsprüfung bleibt nötig.", "",
             "| Teil | Vorher | Nachher | Neue vollständige Antworten |", "|---|---:|---:|---:|"]
    for split, item in final["summary"].items():
        old = baseline["summary"][split]
        lines.append(f"| {split} | {old['passed']}/{old['total']} | {item['passed']}/{item['total']} | {item['novel_complete_answers']} |")
    numerical = final.get("time_invariance", {})
    lines += ["", f"Zeitinvarianz: {numerical.get('passed', 0)}/{numerical.get('checked', 0)} "
              "Antwort-/Zustandsvergleiche bei vier zusätzlichen Zeitpunkten einschließlich 10^12 s.", "",
              f"Numerische Zusatzprüfung: {final['numerical_audit']['text_roundtrip_passed']}/"
              f"{final['numerical_audit']['text_roundtrip_total']} UTF-8-Rückrechnungen, "
              f"{final['numerical_audit']['pair_comparisons']} Vergleiche von expliziten Symbolüberlappungen "
              "mit Fourier-Interferenz sowie Parseval-, inverse FFT- und gemeinsame Phasenprüfung. "
              "Maximalfehler jeweils kleiner als 10^-12.", "",
              f"Quellenprüfung aller Ausgaben: {final['decoder_provenance']['passed']}/"
              f"{final['decoder_provenance']['checked']} Antworten lassen sich aus den deklarierten "
              "UTF-8-Quellabschnitten rekonstruieren; Quellen- und Ausgabeprüfsummen stimmen überein.", "",
              "Neu heißt hier: Die vollständige Antwort steht nicht wörtlich im 120er-Korpus. "
              "Sie kann trotzdem aus vorhandenen Satzteilen und expliziten Regeln bestehen. "
              "Das weist keine offene Sprachgenerierung und keine automatisch aus Physik "
              "gewonnene Textbedeutung nach.", "", "## Offen gebliebene Fälle", ""]
    failed = [r for r in final["rows"] if not r["passed"]]
    for row in failed:
        lines += [f"- `{row['id']}`: {row['prompt']} → {row['answer']} "
                  f"(Prüfung: {', '.join(row['failures'])})"]
    if not failed:
        lines.append("Keine innerhalb dieser endlichen Prüfsuite.")
    lines += ["", "Vollständige Eingaben, Ausgaben, Assertions, Laufzeiten und SHA-256-Freeze: "
              "`baseline.json`, `development.json`, `final.json`, `suite_freeze.json`, "
              "`implementation_freeze.json` im selben Verzeichnis.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("baseline", "development", "final"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    data, memory = load_inputs()
    suite = _freeze_suite(data, memory)
    scenarios = data["scenarios"]
    if args.stage == "development":
        scenarios = [s for s in scenarios if s["split"] == "development"]
    if args.stage == "final":
        if not args.freeze:
            parser.error("Final requires --freeze to record implementation before holdout is run")
        freeze_path = OUTPUT / "implementation_freeze.json"
        freeze = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "implementation": _implementation_hashes(), "scorer_sha256": sha256(Path(__file__)),
                  "suite_sha256": suite["suite_sha256"]}
        if freeze_path.exists():
            old = json.loads(freeze_path.read_text(encoding="utf-8"))
            assert old["implementation"] == freeze["implementation"], "Post-holdout implementation changed"
            assert old["scorer_sha256"] == freeze["scorer_sha256"], "Post-holdout scorer changed"
        else:
            _write(freeze_path, freeze)
    rows = run_cases(memory, scenarios, baseline=args.stage == "baseline")
    report = {"stage": args.stage, "created_utc": datetime.now(timezone.utc).isoformat(),
              "suite": suite, "configuration": CONFIG, "implementation": _implementation_hashes(),
              "scorer_sha256": sha256(Path(__file__)), "summary": summarize(rows), "rows": rows}
    if args.stage == "final":
        report["time_invariance"] = time_invariance(memory, scenarios, rows)
        report["numerical_audit"] = numerical_audit(memory)
        report["decoder_provenance"] = decoder_provenance_audit(memory, rows)
    _write(OUTPUT / f"{args.stage}.json", report)
    if args.stage == "development":
        sequence = len(list(OUTPUT.glob("development_round_*.json"))) + 1
        _write(OUTPUT / f"development_round_{sequence:03d}.json", report)
    if args.stage == "final":
        write_report(report, json.loads((OUTPUT / "baseline.json").read_text(encoding="utf-8")))
    print(json.dumps({"stage": args.stage, "summary": report["summary"],
                      "time_invariance": report.get("time_invariance"), "suite": suite},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
