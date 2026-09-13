# Deutscher Informationskorpus ohne Frage-Antwort-Paare

> Historische Dokumentation der früheren Einleitungs-Auswahl. Seit der Konsolidierung vom 13.09.2026 sind alle 569.062 Artikel der vier lokalen Shards vollständig in `memory/memory.sqlite3` gespeichert. Rohdateien und Teilimporte wurden nach vollständigem Abgleich gelöscht; Quellenunterlagen liegen jetzt in `docs/data_sources/`. Der aktuelle Stand steht im [Konsolidierungsbericht](../results/memory_consolidation.md).

Der finale Kandidat **v3** enthält **20.000 vollständige deutsche Wikipedia-Einleitungen** mit jeweils leerem `prompt`. Es werden keine Fragen, Antwortvorlagen oder ergänzten Fakten erzeugt. Die Auswahl ist unabhängig von den Evaluationsfragen. Der Importer bereitet Dateien vor und öffnet keine SQLite-Datenbank; die Integration in den aktiven Speicher erfolgt getrennt. Ergänzend kuratierte die naturwissenschaftliche Erweiterung **v1** 199 zusätzliche Artikel zu Themen und Personen; siehe unten.

## Quelle und lokal vorhandener Umfang

Quelle ist der vom Wikimedia-Konto veröffentlichte Datensatz [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), Konfiguration `20231101.de`, festgehalten auf Dataset-Revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Die [Beschreibung dieser Revision](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) erläutert die Bereinigung der Wikipedia-Artikel und nennt CC-BY-SA-3.0 und GFDL. Artikel-URLs und Titel bleiben für die Zuordnung zu den jeweiligen Wikipedia-Beiträgen erhalten.

Das Originalschema besteht aus `id`, `url`, `title`, `text` (jeweils String). Die Texte sind die vom Herausgeber exportierten vollständigen bereinigten Artikel. Wikitext-Formatierung und bestimmte Artikelabschnitte wurden bereits vom Herausgeber verarbeitet; es handelt sich nicht um einen unveränderten Wikitext-Dump. Einzelne Artikel-Revisionsnummern liefert der Datensatz nicht. Deshalb enthält `article_revision_id` ausdrücklich `null`; Dataset-Revision, Snapshot-Datum, Artikel-ID und Text-Hash identifizieren den tatsächlich verwendeten Stand.

Unter `memory/sources/information_wikipedia_20231101_de/` liegen vier vollständige, anhand der veröffentlichten Dateigröße und SHA-256 geprüfte Parquet-Shards:

| Shard | Rohartikel | Bytes |
|---|---:|---:|
| `train-00000-of-00020.parquet` | 142.266 | 781.271.996 |
| `train-00007-of-00020.parquet` | 142.266 | 258.122.020 |
| `train-00013-of-00020.parquet` | 142.265 | 228.591.696 |
| `train-00019-of-00020.parquet` | 142.265 | 228.498.582 |
| **Summe** | **569.062** | **1.496.484.294** |

Die vollständige deutsche Konfiguration hat 20 Shards mit insgesamt 5.771.317.942 Bytes. Es wurden bewusst vier Shards aus verschiedenen Bereichen heruntergeladen. Die lokale Auswahl ist dadurch weder die vollständige deutsche Wikipedia noch eine statistisch repräsentative Auswahl aus ihr. Alle später verworfenen oder nicht ausgewählten Artikel dieser vier Shards bleiben im Roharchiv verfügbar.

Die Webrecherche lief ausschließlich über `POST http://127.0.0.1:8080/v1/research/web` mit bestätigtem `provider="searxng"` und `isFallback=false`. Suchantworten, offizielle Dataset-Beschreibung und Dateimetadaten liegen unter `memory/sources/information_research/`. Das direkte Wikimedia-Dumpverzeichnis lieferte HTTP 403; der offizielle Hugging-Face-Datensatz war direkt erreichbar.

## Deterministische Auswahl und Textgrenzen

`experiments/import_information_corpus.py` liest die Parquet-Dateien in Batches von 128 Artikeln. Pro Artikel übernimmt es die vollständige erste Prosa-Einleitung. Ist ein erster vollständiger Absatz kürzer als 120 Zeichen, werden höchstens drei unmittelbar angrenzende vollständige Einleitungsabsätze zusammen übernommen. Die gesamte Prosa muss 120 bis 1.200 Zeichen lang sein. Zu lange Absätze werden verworfen und niemals passend abgeschnitten. Der ursprüngliche Titel wird mit einer Leerzeile vorangestellt und zählt nicht gegen diese Prosa-Grenze.

Die generischen Filter verwerfen unter anderem Listen- und Begriffsklärungstitel, unfertige Absätze, offensichtlichen Markup oder Links, Listenzeilen und mit einem abhängigen Pronomen beginnende Einleitungen. Abschnittsüberschriften werden nicht übersprungen, um entfernte Absätze künstlich zusammenzufügen. Einzelne Zeilenumbrüche innerhalb eines Absatzblocks und leere Exportfelder wie `(Stand: )`, `()` oder `(; …)` werden seit v2 ebenfalls verworfen. Diese Regeln entfernen auch manche ansonsten verwendbaren Artikel, vermeiden aber das Zusammenziehen einer Überschrift mit fremder Folgeprosa und offensichtlich unvollständige Exportfelder.

Eine einfache deutsche Funktionswortprüfung vermeidet offensichtlich fremdsprachige Einleitungen. Unicode wird nach NFC normalisiert, horizontale Leerzeichen werden vereinheitlicht. Seit v3 werden zusätzlich ausschließlich leere Präposition-plus-Semikolon-Anfänge einer Klammer gelöscht: `(von ; auch …)` wird zu `(auch …)`, entsprechend für `aus;` und `nach;`. Das behebt einen sichtbaren leeren Exportrest, ergänzt aber keinen ausgefallenen Begriffsursprung. Jedes entfernte Fragment ist unter `provenance.removed_empty_template_fragments` angegeben; im finalen Bestand betrifft das fünf Einträge. Andere Wörter werden nicht umgeschrieben. Diese Regeln garantieren keine sachliche Richtigkeit oder perfekte Selbstständigkeit jeder Wikipedia-Einleitung.

Artikel-IDs und identischer Titel-plus-Prosa-Inhalt werden dedupliziert. Aus den geeigneten Artikeln werden die kleinsten SHA-256-Werte über `freqai-information-v1:<article_id>` gewählt. Ein vorab festgelegter, auf höchstens 500 Artikel begrenzter Titel-Schwerpunkt umfasst Frequenz, Schwingung, Fourier, Resonanz, Akustik und grundlegende Physikbegriffe. Tatsächlich sind 57 geeignete Schwerpunkt-Titel vorhanden; die übrigen 19.943 Einträge stammen aus der breiten Hash-Auswahl. Der Schwerpunkt prüft Titelwörter, keine Wikipedia-Kategorien, und kann daher auch gleichnamige nichtphysikalische Themen enthalten. Die konkreten Regeln und ausgewählten Schwerpunkt-Titel stehen im Manifest. Keine Evaluationsfragen bestimmen diese Auswahl.

Ergebnis des finalen v3-Laufs:

| Messwert | Ergebnis |
|---|---:|
| geprüfte Rohartikel | 569.062 |
| geeignete vollständige Einleitungen | 361.698 |
| ausgewählte Einträge | 20.000 |
| Textzeichen einschließlich Titel | 6.248.987 |
| UTF-8-Textbytes einschließlich Titel | 6.352.638 |
| JSONL-Dateigröße einschließlich Provenienz | 34.889.094 Bytes |
| Importer-Laufzeit einschließlich Quellen-Hashprüfung | 60,96 Sekunden |

Die Laufzeit misst ausschließlich die Datenvorbereitung auf diesem Rechner. Sie ist keine Messung des Schwingungsaufbaus oder der Antwortgenerierung.

## Dateien und Provenienz

Der Kandidat ist `memory/imports/information_wikipedia_20000_v3.jsonl`, SHA-256:

```text
2f5b876d904346ca4b9082e05ff5e4bc5e8a729f2a479c777f47e1332a0a5674
```

Das zugehörige Manifest `memory/imports/information_wikipedia_20000_v3.provenance.json` enthält Quelldateien mit Größen und SHA-256, Filterzahlen, Auswahlsamen, Schwerpunktregeln, Script-Hash und Ergebnis-Hash.

Der erste Kandidat `information_wikipedia_20000.jsonl` bleibt unverändert mit SHA-256 `8473462b187b219f7270db292e0fdc95ee82245ad047efb21e69b4eb16c77497`. Der Zwischenstand `information_wikipedia_20000_v2.jsonl` bleibt ebenfalls unverändert mit SHA-256 `61497ef4f4ee4c09cde7b0b772e60d853e6b495d58f1e6d681b77c7469910b4c`. Die jeweilige Provenienz und die damaligen Importer-Snapshots unter `memory/sources/information_research/` halten die Iterationen nachvollziehbar auseinander. Die Änderungen entstanden aus einer festen 40er-Quellenstichprobe und dem bereits vom Nutzer vorgegebenen Frequenzbeispiel, nicht durch Auswahl nach Evaluationsfragen.

Jeder JSONL-Eintrag enthält `id`, `prompt=""`, `text`, `source`, `title` und `provenance`. Die Provenienz umfasst Artikel-ID, Titel, Original-URL, Dataset-Revision, Snapshot-Datum, SHA-256 des vollständigen Originalartikels, Rohdatei mit SHA-256, ursprüngliche Zeichenpositionen der ausgewählten Absätze und getrennte Hashes der Prosa sowie des Texts einschließlich Titel. Damit lässt sich der aktive Inhalt direkt am lokal vorhandenen Rohartikel überprüfen. Der Titel steht zusätzlich im Text und in `source`, sodass die Zuordnung auch bei Nutzung der vier bisherigen Dokumentfelder erhalten bleibt.

## Reproduzieren und prüfen

```powershell
.venv\Scripts\python.exe experiments/import_information_corpus.py --offline --count 20000
.venv\Scripts\python.exe -m pytest tests/test_information_import.py -q
.venv\Scripts\python.exe memory/sources/information_research/audit_information_candidate.py
```

Ohne `--offline` lädt das Script fehlende, fest gepinnte Quelldateien direkt herunter und prüft ihre Hashes. Vorhandene Dateien mit abweichenden Hashes führen zu einem Fehler. `--download-only` stellt ausschließlich das Roharchiv bereit. `--count 30000` erzeugt einen separaten 30k-Kandidaten aus denselben Quellen; dessen Qualität und Laufzeit wären vor der produktiven Übernahme gesondert zu prüfen.

Die 23 Importtests prüfen unter anderem leere Prompts, vollständige Absatzgrenzen, unveränderte Originalpositionen einschließlich Unicode-Normalisierung, Nichtabschneiden langer Absätze, kein Überspringen von Überschriften, leere Exportfelder und begrenzte Rest-Löschung mit Provenienz, Deduplikation, Schema- und Hashfehler sowie stabile Auswahl bei anderer Reihenfolge der Quelldatensätze. Das zusätzliche Audit überprüft alle 20.000 aktiven Einleitungen an ihren vollständigen Originalartikeln und reproduziert den Kandidaten mit umgekehrter Reihenfolge der vier Quell-Shards. Der ursprüngliche Bericht einschließlich einer festen 40er-Stichprobe bleibt unter `memory/sources/information_research/candidate_audit.json` erhalten. `candidate_audit_v3.json` prüft den finalen Bestand und dokumentiert den Verbleib genau derselben 40 Artikel-IDs. Es wurde keine bequemere neue Stichprobe nach dem Filtern gezogen.

Im finalen Audit stimmen alle 20.000 Originalstellen und Text-Hashes; der zweite Aufbau erzeugt exakt denselben Kandidaten-Hash. Aus den ursprünglichen 40 Stichproben-IDs bleiben 32 mit identischem Text erhalten, acht werden durch die generischen Quellenfilter entfernt. Vier entfernte Fälle enthielten eingebettete Abschnittsüberschriften, vier leere oder beschädigte Klammerfelder. Die verbliebenen Texte sind deshalb nicht pauschal als sprachlich fehlerfrei bewertet: Beispielsweise enthält die originale Wappenbeschreibung in `wikipedia-de-20231101-5116118-lead` elliptische Formulierungen und Grammatikfehler. Die 32 verbliebenen Fälle sind eine Teilmenge derselben Stichprobe, keine neue Zufallsstichprobe zur Schätzung einer finalen Fehlerquote.

Die Daten entsprechen dem Snapshot vom 1. November 2023. Zeitabhängige Aussagen können überholt sein. Die Bereinigung durch den Herausgeber hat teilweise Formeln, Formelzeichen oder Klammerinhalte verloren; auch ein vollständig übernommener Absatz kann deshalb bereits unvollständige Aussagen enthalten. Diese Inhalte werden nicht aus Vermutungen ergänzt. Eine gewählte Einleitung kann zudem nur eine Definition enthalten, während die Einheit oder weitere Eigenschaften ausschließlich im vollständig lokal archivierten Folgeartikel stehen. Quellen- und Hashprüfungen belegen Herkunft und nachvollziehbare Übernahme; die sprachliche und inhaltliche Qualität der aus den Schwingungen berechneten Antworten muss separat am Generator gemessen werden.

## Kuratierte naturwissenschaftliche Erweiterung v1

Ergänzend zum breiten v3-Korpus kuratiert `experiments/science_extension_titles.json` 210 deutsche Wikipedia-Titel aus Physik, Biologie, Chemie, Astronomie, Geologie und verwandten Gebieten, darunter Personen wie Albert Einstein, Marie Curie, Charles Darwin oder Emmanuelle Charpentier. `experiments/select_science_extension.py` übernimmt für jeden Titel die vollständige Einleitungsprosa nach derselben Grundrezeptur wie v3 – bevorzugt aus denselben vier geprüften Parquet-Shards, andernfalls über die MediaWiki-API von `de.wikipedia.org`. Titel, die v3 bereits führt, werden übersprungen; identischer Titel-plus-Prosa-Inhalt wird dedupliziert.

Auswahl und Abweisung:

| Messwert | Ergebnis |
|---|---:|
| kuratierte Titel | 210 |
| in den vier Shards gefunden | 168 |
| bereits produktiv in v3 | 7 |
| übernommen aus Shards | 144 |
| übernommen über MediaWiki-API | 55 |
| insgesamt übernommen | 199 |
| verworfen (Begriffsklärung oder Exportartefakt) | 4 |
| fehlende Titel | 0 |
| Textzeichen einschließlich Titel | 93.566 |

Verteilung der 199 übernommenen Einträge: Physik 71, Biologie 47, Chemie 32, Astronomie 31, Geologie 13, Naturwissenschaft 2, Physik/Chemie 2, Technik 1. Verworfen wurden `Quark`, `Virus`, `Urknall` und `Natürliche Selektion`; die ersten drei sind Begriffsklärungshauptartikel mit unvollständiger Einleitungsprosa im Export, `Urknall` zusätzlich mit leerem Exportfeld.

Der Kandidat ist `memory/imports/information_science_extension_v1.jsonl` mit 393.061 Bytes, SHA-256:

```text
ccc5ec8614f708b9eb4dbb887023518888e82b7e6c2f483de865cdd8c5c44ca6
```

Das Manifest `memory/imports/information_science_extension_v1.provenance.json` hält jeden ausgewählten Titel mit Artikel-ID, Kuratierungsdomäne, Quelle (Shard oder API), Rohdatei-, Prosa- und Text-Hashes, URL, Zeitstempel und Transformationsregel fest. Titelkette und Selektor-Skript sind dort ebenfalls mit SHA-256 verankert (`experiments/science_extension_titles.json` mit `606d89779ae4cee5d796e89b76c14e8914ec76245a6b6d62d7a8bfce882c10c7`).

Reproduzieren und Import:

```powershell
.venv\Scripts\python.exe experiments\select_science_extension.py --workers 8 --api-delay 1.5
.venv\Scripts\python.exe -m freqai import memory\imports\information_science_extension_v1.jsonl
```

Der Import in `memory/memory.sqlite3` erfolgte atomar und ohne Überschreiben: 199 Dokumente hinzugefügt, 0 vorhanden, 0 Konflikte, neue Revision 10, Bestand damit 20.292 Dokumente. Vorher wurde die Datenbank nach `memory/backups/` gesichert. Alle 199 Datensätze erfüllen den Record-Vertrag (leeres `prompt`, Titel plus vollständige Einleitungsprosa, vollständige Provenienz) und haben weder ID- noch Textüberschneidung mit den 20.000 v3-Einträgen.

Grenzen: Die 144 Shard-Artikel lassen sich wie v3 an den lokal archivierten Rohartikeln prüfen. Die 55 API-Artikel stammen aus dem Live-Stand der MediaWiki-API zum jeweiligen Abrufzeitpunkt (`retrieved_utc`) und nicht aus dem Snapshot vom 1. November 2023; für sie liegt kein lokaler Rohartikel vor. Für API-Artikel weicht die Extraktionsrezeptur dokumentiert ab: Leere Klammerartefakte werden gelöscht statt den Artikel zu verwerfen, ein Absatz über der Längenobergrenze wird nach seinem letzten vollständigen Satz gekürzt, und das Mindestmaß liegt mit 100 Zeichen leicht unter den 120 Zeichen der v3-Rezeptur. Jede Abweichung ist in der Provenienz des jeweiligen Datensatzes vermerkt. Die Auswahl ist ein kuratierter Schwerpunkt, keine vollständige oder statistisch repräsentative naturwissenschaftliche Abdeckung der deutschen Wikipedia.
