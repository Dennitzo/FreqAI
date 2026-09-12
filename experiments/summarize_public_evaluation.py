"""Combine immutable run files and separately recorded qualitative reviews."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import evaluate_public_corpus as common

OUTPUT = common.OUTPUT


def reviewed(path):
    run = common.read(path)
    review_path = path.with_name(path.stem + "_review.json")
    review = common.read(review_path) if review_path.exists() else None
    if review:
        assert review["run_sha256"] == common.sha(path), f"Review/run mismatch: {path}"
        assert [r["id"] for r in review["decisions"]] == [r["id"] for r in run["rows"]]
    return run, review


def main():
    stages = [("baseline", "621 ursprüngliche Einträge, dichter Decoder"),
              ("compact_existing", "621 Einträge, kompakter Decoder"),
              ("public500", "+500, v1-Dateianfang, überwiegend OASST, Gewicht 1"),
              ("public2000", "+2000, v1-Dateianfang, Gewicht 1"),
              ("publicall_v2", "+11.370, v2, Gewicht 1"),
              ("publicall_v3_weighted", "+11.379, v3, Quellengewichte 0,1 / 0,05"),
              ("publicall_v3_lexical", "+11.379, v3, Quellengewichte und lexikalische öffentliche Eingabemerkmale")]
    report = ["# Unabhängige Evaluation der öffentlichen Corpuserweiterung", "",
              "Die Qualitätsurteile stammen von einem separaten AI-Evaluationsagenten. Es handelt sich weder um eine Humanstudie noch um eine verblindete Bewertung. Die Stichproben sind klein; Unterschiede sind beschreibend und kein statistischer Wirksamkeitsnachweis.", "",
              "## Verfahren", "",
              "- Vor den Änderungen wurden 40 neue Alltagschat-Turns in 20 Zweiergesprächen eingefroren: 20 Development und 20 bis zum finalen Codefreeze zurückgehaltene Holdout-Turns. Keine Evaluationsfrage wurde als Importbeispiel verwendet. Der finale normalisierte Vergleich aller 40 Chatprompts mit den 11.379 öffentlichen Eingabefragen und Antworttexten ergibt jeweils 0 vollständige Übereinstimmungen; Einzelwörter und Themen dürfen sich selbstverständlich überlappen. Details: `final_chat_overlap_check.json`.",
              "- Die Ausgangsdaten enthalten 621 SQLite-Dokumente. Die Evaluierung liest SQLite nur für die eingefrorene Ausgangskopie und verändert die produktive Datenbank nicht. Der gleiche Generation-Runtimeadapter wie im Server erhält die Dokumente über einen kleinen Testhalter; der separate Legacy-Retrievalindex wird nicht in die Decodermessung einbezogen.",
              "- Jede qualitative Entscheidung ist über SHA-256 an genau einen Ergebnislauf gebunden. Semantische Passung und Satzbau werden getrennt bewertet. Eine generische Bestätigung erfüllt eine konkrete Inhaltsfrage nicht. Ein klar verständlicher kleiner Grammatikfehler kann semantisch dennoch passend sein.",
              "- Die technische Oberflächenprüfung erkennt nur leere Texte, fehlende Schlusszeichen, offensichtliche Schleifen und ungültige Zeichen. Sie ist kein Grammatik- oder Bedeutungsnachweis.",
              "- Rollen- und Negationsmarkierungen dokumentieren explizite Fehler. Ein themenfremder Satz kann ohne expliziten Rollen- oder Negationswiderspruch bleiben; diese Markierungen dürfen deshalb nicht als allgemeine Erfolgsquote gelesen werden.",
              "- Build- und Antwortzeiten sind beobachtete Prozesszeiten bei teilweise parallelen Entwicklungsarbeiten, keine kontrollierten Hardwarebenchmarks. RSS und privater Commit sind getrennte Größen. Speicherwerte schließen Python-Objekte und temporäre Compilierung ein.", "",
              "## Entwicklungsschleifen", "",
              "| Lauf | Passend | Satzbau brauchbar | Oberfläche | Build | Peak RSS | Median Antwort |", "|---|---:|---:|---:|---:|---:|---:|"]
    values = []
    for label, description in stages:
        path = OUTPUT / f"{label}_development.json"
        if not path.exists():
            continue
        run, review = reviewed(path)
        if label == "compact_existing" and review is None:
            baseline, baseline_review = reviewed(OUTPUT / "baseline_development.json")
            if [r["answer"] for r in baseline["rows"]] == [r["answer"] for r in run["rows"]]:
                review = baseline_review
        passed = str(review["summary"]["appropriate"]) if review else "offen"
        grammar = str(review["summary"].get("grammatical_sentences", "offen")) if review else "offen"
        memory = run.get("memory_after_evaluation", {})
        if label == "baseline" and (OUTPUT / "baseline_memory_development.json").exists():
            memory = common.read(OUTPUT / "baseline_memory_development.json")["memory_after_evaluation"]
        rss = f"{memory['peak_rss_mb']:.0f} MiB" if "peak_rss_mb" in memory else "nicht erfasst"
        median = statistics.median(r["elapsed_s"] for r in run["rows"])
        report.append(f"| [{description}]({path.name}) | {passed}/20 | {grammar}/20 | {run['summary']['surface_form_ok']}/20 | {run['build_s']:.2f} s | {rss} | {median:.2f} s |")
        values.append({"label": label, "run_sha256": common.sha(path), "summary": run["summary"],
                       "review": review["summary"] if review else None, "numerics": run["numerics"],
                       "build_s": run["build_s"], "median_answer_s": median, "memory": memory})
    report += ["", "Die ersten 500 und 2000 Einträge sind feste Dateipräfixe, keine zufälligen Stichproben des Gesamtbestands. v1/v2/v3 besitzen getrennte SHA-256-Hashes und unveränderte Kopien in `corpora/`. Filteränderungen wurden anhand der Quellenqualität vorgenommen, nicht durch Einfügen von Evaluationsantworten.", "",
               "Die Zwischenläufe zeigen eine deutliche Verschlechterung durch ungewichtete öffentliche Daten. Viele fachliche Eingaben enthielten Höflichkeitswörter oder andere Ausdrücke, die der bisherige Gesprächsparser fälschlich als Smalltalk-Absicht etikettierte. Die finale Compilerkorrektur beschränkt automatische öffentliche Eingabemerkmale auf lexikalische Merkmale; eigene Gesprächspaare und explizite Annotationen behalten ihre strukturierten Merkmale. Öffentliche Daten bleiben vollständig in den numerischen Feldern enthalten.", "",
               "## Numerische Kontrollen", "",
               "Pro Chatlauf wurden 60 Präfixzustände geprüft. Fourier-Interferenz und direkte punktweise Referenzrechnung werden auf derselben kompilierten Datenwelle verglichen; dies kontrolliert den Operator, nicht unabhängig die Bedeutung der importierten Texte. Die Prüfung umfasst Wahrscheinlichkeitsnormalisierung, zwei Zeitpunkte, Nullung beider Promptkanäle und sechs Zustände mit genullten komplexen Datenfeldern bei erhaltenen Referenzzählungen und Grammatikmasken.", "",
               "| Lauf | Zustände | Max. FFT/direct-Abweichung | Max. Zeit-Abweichung | Promptnull verändert Verteilung | Daten-null verweigert |", "|---|---:|---:|---:|---:|---:|"]
    for row in values:
        n = row["numerics"]
        if n.get("skipped"):
            continue
        report.append(f"| {row['label']} | {n['states']} | {n['fft_direct_max_probability_error']:.3g} | {n['time_max_probability_error']:.3g} | {n['prompt_null_distribution_changes']}/{n['states']} | {n['data_zero_rejected_states']}/{n['data_zero_tested_states']} |")
    report += ["", "Nullabweichung gilt nach der bereits dokumentierten Rundung der Decoderwahrscheinlichkeiten auf 12 Dezimalstellen. Ein unverändertes Ergebnis bei einzelnen Promptnull-Zuständen ist bei eindeutigem Grammatikübergang möglich. Die Kontrollen beweisen numerische Konsistenz und kausale Feldabhängigkeit; sie beweisen kein Sprachverständnis.", "", "## Zurückgehaltene Chatprüfung", ""]
    holdouts = []
    for path in sorted(OUTPUT.glob("*_holdout.json")):
        run, review = reviewed(path)
        passed = review["summary"]["appropriate"] if review else "Bewertung offen"
        partial = review["summary"].get("partial", 0) if review else 0
        partial_text = f" Zusätzlich {partial}/20 nur teilweise passend, konservativ nicht als volle Treffer gezählt." if partial else ""
        n = run["numerics"]
        report.append(f"- [{path.stem}]({path.name}): {passed}/20 passend; Oberfläche {run['summary']['surface_form_ok']}/20.{partial_text} FFT/direct max. {n['fft_direct_max_probability_error']:.3g}, Zeitabweichung max. {n['time_max_probability_error']:.3g}, keine Argmaxänderungen: {n['fft_direct_argmax_changes'] == n['time_argmax_changes'] == 0}.")
        holdouts.append({"run": path.name, "sha256": common.sha(path), "review": review["summary"] if review else None,
                         "build_s": run["build_s"], "median_answer_s": statistics.median(r["elapsed_s"] for r in run["rows"]),
                         "memory": run.get("memory_after_evaluation"), "numerics": n})
    if not holdouts:
        report.append("Noch nicht ausgeführt. Dieser Abschnitt wird erst nach dem finalen Code- und Corpusfreeze ergänzt.")
    elif (OUTPUT / "final_holdout_review.json").exists():
        report += ["", "Der Abschluss zeigt keine Verbesserung der allgemeinen Chatqualität: 12/20 vollständige Treffer gegenüber 13/20 in der Ausgangsversion, zusätzlich eine teilweise passende Antwort. Beide Versionen bestehen jeweils nur drei der zehn Zweiergespräche vollständig. Neue Fehler betreffen unter anderem Familientreffen und Abschied, die in fremde Sachthemen abdriften. Die Kaffee-Negation bleibt falsch. Verbessert wurden in dieser Stichprobe der Wunsch nach Ruhe und die Frage nach eigenen Gefühlen. Nach dieser Prüfung wurde kein semantisches Tuning vorgenommen."]
    report += ["", "## Separater Faktentest und bekannte Fragen", "",
               "Der offizielle GermanQuAD-Testsplit enthält 2.204 Fragen. Kein Test-Kontextabschnitt ist exakt im Trainingssplit enthalten; 2.130 Testfragen stammen aber aus bereits im Trainingssplit vertretenen Wikipedia-Artikeltiteln. Die 30 Testfragen wurden mit Seed 731 vor Kandidatausgaben fixiert. Ihre Referenzkontexte werden dem Generator nicht mitgegeben: Das ist eine geschlossene Corpusprüfung und nicht der übliche extraktive GermanQuAD-Leseverständnisbenchmark.", "",
               "Im finalen Import kommt bei 8/30 Fragen ein Referenzantwortstring vor, davon bei 4 mit gleichem Artikeltitel. Ein Stringtreffer beweist keine ausreichende Belegabdeckung. Mehrere offizielle Datensätze enthalten widersprüchliche Antwortannotationen; das qualitative Urteil berücksichtigt den Originalkontext. Historische Angaben werden gegen die Quelle bewertet und nicht als heute verifizierte Fakten ausgegeben.", "",
               "Zusätzlich wurden mit Seed 9021 genau 20 bekannte Originalfragen aus dem finalen GermanQuAD-Import fixiert. Diese Abrufdiagnose ist ausdrücklich **in-sample** und darf nicht als Generalisierungsleistung ausgegeben werden.", "",
               "| Lauf | Prüftyp | Qualitativ korrekt | Enthält Referenzstring |", "|---|---|---:|---:|"]
    facts = []
    for suffix, title in [("facts", "30 offizielle Testfragen"), ("replay", "20 bekannte Importfragen")]:
        for path in sorted(OUTPUT.glob(f"*_{suffix}.json")):
            if path.name.endswith("_review.json"):
                continue
            run, review = reviewed(path)
            count = run["summary"]["questions"]
            correct = review["summary"]["correct"] if review else "offen"
            report.append(f"| [{path.stem}]({path.name}) | {title} | {correct}/{count} | {run['summary']['contains_answer_string']}/{count} |")
            facts.append({"run": path.name, "sha256": common.sha(path), "summary": run["summary"],
                          "review": review["summary"] if review else None, "build_s": run["build_s"],
                          "median_answer_s": statistics.median(r["elapsed_s"] for r in run["rows"]), "memory": run.get("memory")})
    report += ["", "Die Reviewdateien enthalten die konkreten Fehler und Quellenbezüge. Neue vollständige Zeichenketten oder neue Sätze werden als Neuheitsdiagnose gespeichert, aber nicht mit sinnvoller oder faktisch korrekter Generierung gleichgesetzt.", ""]
    if (OUTPUT / "final_facts_review.json").exists() and (OUTPUT / "final_replay_review.json").exists():
        report += ["Der größere Bestand ermöglicht in dieser Probe acht korrekte Antworten auf bereits importierte Originalfragen. Keine der dreißig separaten offiziellen Fragen wird korrekt beantwortet; deren Belegabdeckung im Import ist allerdings größtenteils unzureichend. Ein allgemeiner Wissens- oder Verständnisgewinn ist damit nicht nachgewiesen.", ""]
    regression_path = OUTPUT / "fast_carrier_output_regression.json"
    regression = common.read(regression_path) if regression_path.exists() else None
    if regression:
        r = regression["summary"]
        report += ["## Nachträgliche numerische Beschleunigung", "",
                   f"Die Tokenmoden wurden für eine günstig faktorisierte FFT-Trägerlänge mit Nullen aufgefüllt. Die Quellenfelder, Tokenfrequenzen und semantische Konfiguration blieben erhalten. Die Ausgaberegression prüft die bereits gesehenen 40 Chat-, 30 Fakten- und 20 bekannten Importfragen: {r['exact_visible_outputs']}/90 sichtbare Antworten bytegleich, {r['token_sequence_matches']}/90 Tokenfolgen übereinstimmend. Dies ist kein neuer Holdout und kein Qualitätszuwachs.", "",
                   regression["limitation"], "",
                   f"Median über alle 90 Antworten: vorher {r['median_reference_elapsed_s']:.2f} s, nach Beschleunigung {r['median_elapsed_s']:.2f} s. Die älteren Messungen liefen teilweise parallel und sind deshalb kein kontrollierter Beschleunigungsfaktor. Der einmalige Modellaufbau der Regression dauerte {regression['build_s']:.2f} s. Details: [Ausgaberegression]({regression_path.name}).", ""]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    common.write(OUTPUT / "summary.json", {"created_utc": common.timestamp(), "development": values, "holdouts": holdouts,
                                          "facts": facts, "performance_regression": regression["summary"] if regression else None})
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
