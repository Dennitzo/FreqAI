"""Reproduce the final frozen quality-and-numerics report without model tuning."""
from __future__ import annotations

import json
from pathlib import Path

from run_generative_waves import OUTPUT, ROOT, compile_report, digest, freeze, read, write


def main() -> None:
    frozen = freeze()
    compile_report()
    reviewed = read(OUTPUT / "reviewed_results.json")
    numerics_path = OUTPUT / "numerics_20260906T090014976834Z.json"
    numerics = read(numerics_path)
    assert numerics["manifest"] == frozen["manifest"], "Final numerical audit used a different model"
    quality = {item["run"]["split"]: item["run"]["summary"] for item in reviewed["reviews"] if not item["run"]["baseline"]}
    evidence = read(OUTPUT / "new_sentence_evidence.json")
    development = next(item["run"] for item in reviewed["reviews"] if item["review"] == "development_review.json")
    assert any(row["answer"] == evidence["complete_answer"] and row["semantic_pass"] for row in development["rows"])
    summary = {"model_freeze_verified": True, "quality": quality, "numerics": numerics["summary"],
               "null_memory_explicit_refusals": numerics["null_memory_explicit_refusals"],
               "new_meaningful_sentence": evidence["new_sentence"],
               "new_sentence_is_bounded_development_example": True,
               "source_files": {name: digest(OUTPUT / name) for name in ["model_freeze.json", "holdout.json",
                                "holdout_review.json", "development_review.json", numerics_path.name,
                                "new_sentence_evidence.json"]},
               "review_identity": "Qualitative assessment by a separate Codex agent; not a human user study."}
    write(OUTPUT / "final_summary.json", summary)
    lines = ["", "## Abschluss nach dem eingefrorenen Holdout", "",
             f"Der ausgewählte Stand erreicht **{quality['development']['semantic_pass']}/30** passende Antworten in der Entwicklung "
             f"und **{quality['holdout']['semantic_pass']}/30** im einmaligen Holdout. "
             "Die Mehrheit der neuen Formulierungen erhält damit weiterhin keine passende Antwort. "
             "Die numerische Verbindung von Feldern und Tokens ist nachgewiesen; eine allgemein brauchbare Gesprächsgenerierung ist nicht erreicht.", "",
             f"Eine Ausgabe in der Entwicklung enthält einen neuen, sinnvollen Einzelsatz; im Holdout sind es **{quality['holdout']['texts_with_new_sentences']}/30**. "
             "Die vollständige Entwicklungsentscheidung und alle vorab gewählten Alternativen stehen in "
             "[development_selection.md](development_selection.md).", "",
             "> " + evidence["new_sentence"], "",
             "Dieser Satz wurde aus zwölf nacheinander berechneten Tokenmoden erzeugt und kommt unter "
             f"{evidence['corpus_sentences_checked']} unterschiedlichen Korpussätzen nicht vor. "
             "[Der Tokenbeleg](new_sentence_evidence.json) enthält Präfixe, Frequenzen und überprüfte Wahrscheinlichkeiten.", "",
             "## Kausale Kontrollen des Wortdecoders", "",
             "Die folgenden Kontrollen verwenden 90 Präfixzustände aus den 30 Entwicklungseingaben. "
             "Sie prüfen den Wortdecoder mit dem ausgewählten Sprachmodell; die Gesprächsbewertung "
             "oben prüft zusätzlich die mehrteilige Eingabeverarbeitung und den Sitzungskontext. "
             "Geänderte Verteilungen sind kein Beleg für bessere Sprache.", "",
             "| Eingriff | Verteilung geändert | Argmax geändert | Mittlere Total-Variation-Distanz |",
             "| --- | ---: | ---: | ---: |"]
    labels = {"token_channel_null": "Tokenkanal des Prompts null", "feature_channel_null": "Merkmalskanal des Prompts null",
              "both_prompt_channels_null": "Beide Promptkanäle null", "relative_phase_pi": "Relative Interferenzphase um π geändert",
              "feature_spectral_phase_scramble": "Phasen des Merkmalskanals verwürfelt",
              "data_spectral_phase_scramble": "Phasen der Datenwelle verwürfelt",
              "direct_calculation": "Direkte Produktrechnung statt spektraler Faltung",
              "time_86400_125": "Freie Zeitentwicklung bis 86.400,125 s"}
    for name, label in labels.items():
        data = numerics["summary"][name]
        lines.append(f"| {label} | {data['changed_distributions']}/{data['checked']} | "
                     f"{data['changed_argmax']}/{data['checked']} | {data['mean_total_variation']:.6f} |")
    lines += ["", f"Bei vollständig genullten unbedingten und konditionierten Datenspektren verweigert der Decoder "
              f"in **{numerics['null_memory_explicit_refusals']}/90** Fällen ausdrücklich die Ausgabe. "
              "Referenzarrays und Grammatikstützen bleiben bei diesem Test vorhanden; sie dienen nicht als verdeckter Antwortgenerator.", "",
              "Die direkte Produktrechnung und die spektrale Faltung sind nach der dokumentierten "
              "Rundung auf zwölf Nachkommastellen identisch. Beide verwenden dieselben gespeicherten "
              "Übergangsamplituden. Der größere Einfluss des Merkmalskanals zeigt, dass explizite "
              "Sprachmerkmale und annotierte Daten wesentlich zur Konditionierung beitragen. "
              "Die freie gemeinsame Phase verschwindet beim Auslesen der Intensität; bloßes längeres "
              "Schwingen erzeugt daher keine zusätzlichen Gedanken.", "",
              "Die SHA-256-Werte von Implementierung, Kategorien, Korpora, Konfiguration und Evaluierung "
              "wurden vor dem Holdout festgelegt und beim Erstellen dieses Berichts erneut überprüft. "
              "[final_summary.json](final_summary.json) bindet alle Abschlussartefakte an ihre Quelldateien.", ""]
    with (OUTPUT / "report.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"quality": quality, "model_freeze_verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
