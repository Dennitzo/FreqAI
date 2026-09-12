# Öffentliche Datenquellen und Import

Am 6. September 2026 wurden **11.379 zusätzliche deutsche Textpaare** als reproduzierbarer Import vorbereitet: 533 Gesprächsantworten aus OpenAssistant und 10.846 sachliche Antworten aus GermanQuAD. Die Daten werden gezählt und in Wellenfelder umgerechnet; es werden keine Modellgewichte durch Gradientenoptimierung trainiert. Die Quellen erweitern das beobachtete Sprachmaterial, beweisen aber keine bessere Gesprächsfähigkeit. Die tatsächliche Freigabe und Antwortprüfung steht im separaten [Ergebnisbericht](../results/public_corpus/report.md).

## Recherche und Auswahl

Alle elf Suchanfragen liefen über den lokalen SearXNG-Zugang `http://127.0.0.1:8080/v1/research/web`. Jede Antwort meldete `provider: searxng` und `isFallback: false`; einige enge Suchanfragen lieferten keine Treffer. Danach wurden Datasetkarten und Dateien direkt beim jeweiligen Herausgeber gelesen. [Recherchebelege](../results/public_corpus/research/source_verification.json) enthalten Providerprüfung und lokale Kopien der Primärquellen.

| Quelle | Eignung und Herkunft | Aktiver Ausschnitt |
|---|---|---:|
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Gesprächsbäume, Rollen und menschliche Qualitätsbewertungen; laut Datasetkarte Apache-2.0 | 533 eigenständige deutsche Root-Prompt-Antwort-Paare |
| [deepset/germanquad](https://huggingface.co/datasets/deepset/germanquad) | Menschlich annotierte deutsche Fragen mit Antwortpositionen in Wikipedia-Passagen; Datasetkarte CC-BY-4.0 | 10.846 Fragen ausschließlich aus dem offiziellen Train-Split |

OASST2 wurde statt einer Addition von OASST1 und OASST2 gewählt, weil die Veröffentlichungen große Überschneidungen haben. Der All-Export enthält 208.584 Nachrichten, darunter 8.398 deutsche. Nur positiv geprüfte, ungelöschte und ausdrücklich nicht synthetische deutsche Nachrichten mit konsistenter Rollen- und Parent-Kette kommen infrage. Negative Qualitätslabels, Code, lange Texte, Listen und fremde Assistant-Personas werden generisch gefiltert. Antwortqualität muss mindestens 0,5 betragen. Die 533 aktiven Paare können verschiedene Antworten auf denselben ursprünglichen Prompt enthalten und sind deshalb keine 533 unabhängigen Gesprächsanlässe.

Weitere 500 passende OASST-Folgeantworten bleiben in `memory/sources/openassistant-oasst2/followup_candidates_de.jsonl` **inaktiv**. Ihr vollständiger Rollenverlauf bleibt erhalten. Ein isoliertes „Warum?“ aus einem Gespräch wird damit nicht fälschlich als selbstständige Frage importiert. Der bisherige Promptencoder kann die Rollen eines fremden Gesprächsverlaufs nicht zuverlässig unterscheiden.

GermanQuAD enthält im offiziellen Train-Split 11.518 Fragen. Aus dem mitgelieferten Kontext werden vollständige Sätze extrahiert, welche die durch Zeichenposition geprüfte Originalantwort enthalten. Häufig rückverweisende Satzanfänge bekommen den vorhergehenden Satz dazu. Maximal 900 Zeichen beziehungsweise 140 Wörter werden übernommen. Wikipedia-Hervorhebungsapostrophe werden entfernt; Originalpositionen, Goldantwort, Artikelüberschrift und Kontextprüfsumme bleiben in der Provenienz. Es entstehen keine mit einem Sprachmodell umgeschriebenen Antworten. Die aktiven Fragen verteilen sich auf **2.526 Quellpassagen**; mehrere Fragen können denselben Satz verwenden. Der Test-Split wird vom Importscript weder heruntergeladen noch gelesen. Seine gesonderte Prüfung erfolgt durch den Evaluator.

## Lizenz und Quellennachweis

Die OASST2-Datasetkarte nennt Apache-2.0; Karte und Lizenztext liegen unter `memory/sources/openassistant-oasst2/`. Jeder aktive Datensatz behält Message-ID, Parent-ID, Tree-ID, Datasetrevision und Herkunft. Der kopierte Apache-Lizenztext stammt aus der ebenfalls unter Apache-2.0 veröffentlichten offiziellen OASST1-Quelle; die Lizenzzuordnung von OASST2 wird durch dessen eigene gepinnte Datasetkarte belegt.

Die GermanQuAD-Datasetkarte nennt **CC-BY-4.0**. Die extrahierten Artikeltexte stammen aus Wikipedia; deren ursprüngliche Attribution und Share-Alike-Bedingungen werden durch eine neue Datasetkarte nicht aufgehoben. Daher behalten die Einträge zusätzlich „Wikipedia contributors CC-BY-SA“, den ursprünglichen Artikeltitel und einen Artikel-Link. Die veröffentlichte Parquet-Datei enthält keine Wikipedia-Revisions-ID. Ein aktueller Artikel kann vom historischen Text abweichen. Die Rohdatei und ihr SHA-256 erhalten den tatsächlich verwendeten historischen Wortlaut. [Die wissenschaftliche Veröffentlichung](https://aclanthology.org/2021.mrqa-1.4/) beschreibt Datenerhebung und Aufteilung.

Zwei geprüfte Alternativen wurden nicht übernommen:

- `FreedomIntelligence/alpaca-gpt4-deutsch` ist laut [MultilingualSIFT](https://github.com/FreedomIntelligence/MultilingualSIFT) eine GPT-3.5-Übersetzung von GPT-4-Daten. Die Mirror-Karte nennt Apache-2.0, während die [Originalveröffentlichung der GPT-4-Daten](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) ausdrücklich CC-BY-NC-4.0 nennt. Diese widersprüchliche Herkunft wird nicht als freie Apache-Datenbasis ausgegeben.
- [mayflowergmbh/dolly-15k_de](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de) nennt einen bei der Prüfung mit HTTP 401 nicht lesbaren deutschen Vorgänger. Die Mirror-Karte dokumentiert weder Lizenz noch Übersetzungsmethode. Die klare CC-BY-SA-3.0-Lizenz der englischen [Databricks-Originaldaten](https://huggingface.co/datasets/databricks/databricks-dolly-15k) allein klärt diese konkrete deutsche Bearbeitung nicht hinreichend.

## Reproduzieren

Die benötigte Importbibliothek ist separat vom laufenden Generator:

```powershell
.venv\Scripts\python.exe -m pip install -r experiments\requirements-import.txt
.venv\Scripts\python.exe experiments\import_public_corpus.py
```

Das Script lädt ausschließlich festgeschriebene Revisionen, prüft die SHA-256-Werte und erzeugt sortierte JSONL-Dateien. Ein weiterer Lauf nutzt dieselben Rohdateien. Ohne Netzwerkzugriff:

```powershell
.venv\Scripts\python.exe experiments\import_public_corpus.py --offline
.venv\Scripts\python.exe -m pytest tests\test_public_corpus_import.py -q
```

Das Script verändert **keine SQLite-Datenbank**. Kandidat, Filterkonfiguration und Rohdaten bleiben unter dem zentralen `memory`-Ordner nachvollziehbar:

- `memory/imports/public_dialogue_oasst2_de.jsonl`
- `memory/imports/public_knowledge_germanquad_de.jsonl`
- `memory/imports/public_dialogue_expansion.jsonl`
- `memory/imports/public_dialogue_expansion.provenance.json`
- `memory/sources/` mit gepinnten Originaldateien und Datasetkarten

Der festgeschriebene Importkandidat v3 enthält 11.379 Paare und hat SHA-256 `832754653cdd1dcd0c2c0cb7043d3edae71f8e28558b5ed7a145362c0b73d9fa`. Jede Antwort trägt zusätzlich eine Herkunftszeichenkette, sodass Quelle und Original-ID auch im vereinfachten SQLite-Dokument erhalten bleiben. Die ausführlichen Metadaten einschließlich Originalspans stehen in den JSONL-Dateien daneben.

## Qualität und Grenzen

15 gezielte Importtests prüfen Rollenketten, Goldoffsets, vollständige Sätze, Abkürzungen, Listen, Pseudocode, fremde Personas, Unicode-Normalisierung und Prüfsummen. Die [vollständige Quellenprüfung](../results/public_corpus/source_numerical_audit.json) rekonstruiert alle GermanQuAD-Antworten aus ihren Originalpositionen und wiederholt die Erzeugung ohne Änderung der freigegebenen Dateien.

Eine unabhängige AI-Agent-Prüfung verwendete eine vorab festgelegte Zufallsstichprobe mit 20 OASST- und 20 GermanQuAD-Paaren, Seed 20260906. Sie ist keine menschliche Qualitätsstudie. Nach generischen Filterkorrekturen verbleiben aus genau denselben OASST-IDs 14; gefundenen flachen Listen und eine fremde Persona wurden ausgeschlossen. Unter den verbleibenden 14 Beispielen enthalten drei bereits im Original Grammatikfehler und eines eine physikalische Plausibilitätslücke. Alle 20 GermanQuAD-Beispiele bleiben erhalten; bei einem fehlt weiterhin die Auflösung eines Rückverweises auf drei Dimensionen. [Einzelurteile und Korrekturprüfung](../results/public_corpus/import_quality_review.json) legen diese Einschränkungen offen.

Die Daten sind historische Quellen von 2021 beziehungsweise bis November 2023. Ihre Fakten wurden nicht einzeln auf Aktualität geprüft. Korrekte Textpositionen und größere Datenmengen sind von richtigen, passenden generierten Antworten getrennt zu bewerten.
