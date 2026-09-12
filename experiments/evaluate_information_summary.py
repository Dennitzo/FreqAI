"""Assemble the independent information-corpus report from immutable evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import evaluate_information_corpus as evaluation
import evaluate_public_corpus as common

OUT = evaluation.OUTPUT


def reviewed(name):
    run = common.read(OUT / f"{name}.json")
    review = common.read(OUT / f"{name}_review.json")
    assert review["run_sha256"] == common.sha(OUT / f"{name}.json")
    return run, review


def metrics(run, review, split=None):
    rows = [row for row in run["rows"] if split is None or row.get("split", run["split"]) == split]
    ids = {row["id"] for row in rows}
    decisions = [row for row in review["rows"] if row["id"] in ids]
    nonempty = [row["elapsed_s"] for row in rows if row["answer"].strip()]
    return {"cases": len(rows), "correct": sum(row.get("correct", False) for row in decisions),
            "partial": sum(row.get("partial", False) for row in decisions),
            "grammatical": sum(bool(row.get("grammatical", False)) for row in decisions),
            "nonempty_answers": sum(bool(row["answer"].strip()) for row in rows),
            "empty": sum(not row["answer"].strip() for row in rows),
            "generic_nonanswers": sum(row.get("generic_nonanswer", False) for row in decisions),
            "safe_empty_unknown_abstentions": sum(row.get("safe_empty_abstention_on_unknown_value", False) for row in decisions),
            "exact_source_passages": sum(row["automatic"]["exact_normalized_source_passage"] for row in rows),
            "nonempty_median_s": statistics.median(nonempty) if nonempty else None,
            "all_median_s": statistics.median(row["elapsed_s"] for row in rows),
            "build_s": run["build_s"], "build_stages": run.get("build_stages"),
            "peak_rss_mb": run["memory_after_evaluation"].get("peak_rss_mb")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-development", default="runtime_v3_development")
    parser.add_argument("--final-holdout", default="final_holdout")
    parser.add_argument("--chat-reference", default="runtime_v2_development")
    parser.add_argument("--baseline-holdout", default="baseline_closed_holdout")
    parser.add_argument("--randomfacts", default="final_randomfacts")
    args = parser.parse_args()
    baseline, baseline_review = reviewed("baseline_development")
    closed, closed_review = reviewed(args.baseline_holdout)
    final_dev, final_dev_review = reviewed(args.final_development)
    final, final_review = reviewed(args.final_holdout)
    chat_reference, chat_review = reviewed(args.chat_reference)
    routing = common.read(OUT / "runtime_v3_routing.json")
    assert routing["known_chat_information_routes"] == 0
    assert all(routing["conversation_core_files_unchanged"].values())
    assert final_dev["manifest"]["code"] == final["manifest"]["code"]
    assert final_dev["manifest"]["additional_corpus_sha256"] == final["manifest"]["additional_corpus_sha256"]
    synthetic = common.read(OUT / f"{args.randomfacts}.json")
    synthetic_review = common.read(OUT / f"{args.randomfacts}_review.json")
    assert synthetic_review["run_sha256"] == common.sha(OUT / f"{args.randomfacts}.json")
    expected = {row["id"]: row for row in baseline["known_chat_rows"]}
    assert len(expected) == 40
    comparisons = [{"id": row["id"], "answer_equal": row["answer"] == expected[row["id"]]["answer"],
                    "tokens_equal": row["tokens"] == expected[row["id"]]["tokens"]}
                   for row in chat_reference["known_chat_rows"]]
    summary = {"created_utc": common.timestamp(), "baseline_development": metrics(baseline, baseline_review),
               "baseline_holdout": metrics(closed, closed_review),
               "final_development": metrics(final_dev, final_dev_review, "development"),
               "final_holdout": metrics(final, final_review, "holdout"),
               "known_chat_regression": {"cases": len(comparisons),
                                         "identical_answers": sum(r["answer_equal"] for r in comparisons),
                                         "identical_token_sequences": sum(r["tokens_equal"] for r in comparisons),
                                         "rows": comparisons,
                                         "reference_run_sha256": common.sha(OUT / f"{args.chat_reference}.json"),
                                         "reuse": "Authorized reuse after exact chat-route and conversation-core-hash check; no new output-quality benchmark.",
                                         "routing_check": routing},
               "default_40_token_probe": final.get("default_limit_probe"),
               "randomfacts": synthetic_review["summary"],
               "information_signature": final.get("information_signature"),
               "code": final["manifest"]["code"], "corpus_sha256": final["manifest"]["additional_corpus_sha256"],
               "development_evidence": common.read(OUT / "development_evidence_v3.json"),
               "holdout_evidence_and_independent_review": common.read(OUT / "independent_holdout_review.json"),
               "integrity": common.read(OUT / "final_integrity.json"),
               "numerical_audit": common.read(evaluation.ROOT / "results/information/numerical_audit.json"),
               "limitations": ["Independent AI-agent judgments, no human study; small manually authored suite.",
                               "User-provided frequency question is development, never unseen.",
                               "Known40 chat questions are output regression, not a new chat benchmark.",
                               "Statistical count normalization without optimizer updates is still data-derived weighting.",
                               "Existing conversation decoder retains previously chosen fixed gains; claim of no fitted gains applies only to information decoder.",
                               "Exact source passages generated token by token do not demonstrate novel factual composition.",
                               "Corpus coverage differs by question; failed answers cannot all be attributed to decoding alone.",
                               "Missing publisher formulas/symbols and limited 2023 Wikipedia coverage remain.",
                               "Compilation and latency measurements occurred alongside other local work, not a controlled speed benchmark."]}
    evaluation.immutable(OUT / "summary.json", summary)
    lines = ["# Unabhängige Evaluation des Informationskorpus", "",
             "Der neue Decoder beantwortet4 von20 Entwicklungsfragen korrekt. Im zuvor versiegelten Holdout "
             "bleibt er bei allen20 Fragen leer; die bisherige Baseline beantwortete dort einen DNA-Fall korrekt. "
             "Eine allgemeine Übertragung auf neue Informationsfragen ist damit nicht nachgewiesen.", "",
             "Die Bewertung trennt belegte Fakten, passende Antworten, sichere Enthaltung, Satzbau und Textneuheit. "
             "Es handelt sich um eine unabhängige KI-Agentenbewertung, keine menschliche Nutzerstudie.", "",
             "| Prüfung | Ausgangsmodell | Endstand |", "|---|---:|---:|"]
    for title, key in (("Informationsfragen Development", "development"), ("Versiegelte Informationsfragen", "holdout")):
        left, right = summary[f"baseline_{key}"], summary[f"final_{key}"]
        lines.append(f"| {title}: sachlich korrekt | {left['correct']}/{left['cases']} | {right['correct']}/{right['cases']} |")
        lines.append(f"| {title}: leere Antworten | {left['empty']} | {right['empty']} |")
        left_grammar = f"{left['grammatical']}/{left['nonempty_answers']}" if left['nonempty_answers'] else "nicht anwendbar"
        right_grammar = f"{right['grammatical']}/{right['nonempty_answers']}" if right['nonempty_answers'] else "nicht anwendbar"
        lines.append(f"| {title}: grammatisch unter nichtleeren Antworten | {left_grammar} | {right_grammar} |")
    chat = summary["known_chat_regression"]
    lines += ["", f"Bekannte Alltagsgespräche: **{chat['identical_answers']}/{chat['cases']} Antworten** und "
              f"**{chat['identical_token_sequences']}/{chat['cases']} Tokenfolgen** stimmen exakt mit der aktuellen Baseline überein.", "",
              "Dieser40-Fälle-Lauf wurde nach der abschließenden Routingänderung auf Basis unveränderter "
              "Gesprächscompiler-Hashes und0 von40 veränderten Gesprächsrouten wiederverwendet. Er ist ein bekannter Regressionstest.", "",
              "Das Ausgangsmodell umfasst12.000 Texte (11.500 QA-Paare plus500 synthetische Sprachprior-Texte). "
              "Der neue Import umfasst20.000 deklarative Wikipedia-Texte mit leeren Prompts. Die Fragen und Bewertungskriterien wurden nicht importiert.", "",
              "Mindestens7 der18 Wissensfragen im Development sind in finaler Prosa direkt belegt; im Holdout "
              "bestätigte ein zusätzlicher Reviewer mindestens2 vollständige Belege. Fehlende Belege und fehlende Fragebindung "
              "sind verschiedene Grenzen. Die jeweils zwei persönlichen Messfragen enthalten keine tatsächliche Messung; "
              "die leere Enthaltung verhindert dort erfundene Werte, liefert aber keinen erklärenden Antwortsatz.", "",
              "Alle vier korrekten Entwicklungsantworten sind normalisiert exakte Quellpassagen. Neue "
              "faktisch korrekte Satzkomposition wird damit nicht belegt. Die explizit vom Nutzer genannte Frequenzdefinition "
              "endet im tatsächlichen40-Token-Aufruf vollständig nach36 Tokens und entspricht exakt der64-Token-Referenz.", "",
              "## Zufallsfakten ohne QA-Paare", "",
              "Zwölf unbekannte Namen und Zahlen wurden ausschließlich als24 deklarative Sätze eingespeist. "
              "Die vorab festgelegten Fragen erreichten0/12 Vorwärts-, 0/12 Rückwärts- und0/4 Negationsantworten. "
              "Vier unbekannte Namen wurden sicher leer abgewiesen. Nach Zahlenpermutation blieben auch12 weitere "
              "Antworten leer. Alle44 Ausgaben waren leer; dies ist kein Fähigkeitsnachweis für allgemeine Faktenbindung.", "",
              "Identische Eingaben erzeugen identische Koeffizienten und Ausgaben. Durch die Zahlenpermutation "
              "ändern sich die Koeffizienten bei gleichem Vokabular. Die Datenkodierung reagiert also auf die Fakten, "
              "während die Fragegrammatik diese unabhängige Formulierung nicht erfolgreich bindet.", "",
              "## Numerik und Reproduzierbarkeit", "",
              "Im unabhängigen Toy-Audit stimmen48 endliche Operator-/Direktverteilungen nach der implementierten "
              "Rundung auf12 Nachkommastellen exakt überein; sechs destruktive Nullzustände werden korrekt abgewiesen. "
              "Zeitinvarianz, Prompt-/Datenfeldnullung und relative Phaseninterventionen sind getrennt dokumentiert. "
              "Eine neu kombinierte Toy-Folge ('Blaue Katzen schlafen.') belegt mögliche lokale Wortkombination, "
              "nicht allgemeine faktische Richtigkeit.", "",
              "Drei vollständige Aufbauten derselben20.000 Dokumente ergeben denselben Koeffizientenhash und "
              "denselben Renderhash. Es gab0 Optimierungsschritte. sqrt(count/sum(count)) ist eine deterministische "
              "statistische Datenanpassung. Die bestehende Alltagsschiene behält ihre zuvor festgelegten Verstärkungen.", "",
              f"Der abschließende Informationsaufbau benötigte{final['build_s']:.1f}s bei "
              f"{final['memory_after_evaluation']['peak_rss_mb']/1024:.2f}GiB Prozessspitze. "
              "Die rund77ms Median im Holdout messen ausschließlich Enthaltungen. Der letzte Developmentlauf "
              "benötigte etwa250s für beide Compiler; parallele lokale Arbeit beeinflusste die Zeitmessungen.", "",
              "Die vollständigen Kennzahlen, Einzelfallbegründungen, Quellbelege, Code-/Datenhashes und das "
              "Zufallsfaktenexperiment stehen in den JSON-Artefakten dieses Verzeichnisses.", "",
              "## Grenzen", "",
              "Kleine, manuell formulierte Testsuite und KI-Agentenbewertungen; keine menschliche Nutzerstudie. "
              "Der Frequenzfall ist bekanntes Development. Wikipedia-Abdeckung von2023, fehlende Formeln im "
              "Herausgeberexport und begrenzte deklarierte deutsche Grammatik bleiben Einschränkungen. "
              "Nach dem Holdout erfolgte keine weitere Qualitätsanpassung."]
    report_text = re.sub(r"(?<=[A-Za-zÄÖÜäöüß])(?=\d)|(?<=\d)(?=[A-Za-zÄÖÜäöüß])", " ", "\n".join(lines))
    (OUT / "report.md").write_text(report_text + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key in ("baseline_development", "baseline_holdout", "final_development", "final_holdout", "randomfacts")}))


if __name__ == "__main__":
    main()
