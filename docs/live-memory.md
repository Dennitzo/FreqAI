# Zentrale Memory und laufende Gesprächswellen

Der aktive Bestand befindet sich einheitlich in `memory/memory.sqlite3`. Der absolute Standardpfad wird aus dem Installationsort des Python-Pakets bestimmt, nicht aus dem aktuellen Terminalverzeichnis. Oberfläche und CLI öffnen damit dieselbe Datenbank.

## Gesprächspaare

Ein Gesprächseintrag enthält `id`, `prompt`, `text` und `source`. `prompt` ist eine beispielhafte Äußerung, `text` die dazu passende Antwort. Die festen Merkmale und Fourier-Schlüssel werden aus `prompt` gebildet. Die Ausgabe wird aus den DCT-Koeffizienten von `text` rekonstruiert. Einträge ohne `prompt` bleiben für explizite Textimporte kompatibel; dann ist der Text selbst der Suchschlüssel.

Beispiel:

```json
{"id":"beispiel-begruessung","prompt":"Guten Morgen","text":"Guten Morgen! Wie startet dein Tag?","source":"Alltag / Begrüßung"}
```

Die Gesprächssammlung enthält bewusst deklarierte Anlass-Antwort-Paare. Der zusätzliche Dialogprozessor erkennt begrenzte Gesprächsakte durch explizite Regeln und setzt Antwortteile zusammen. Sitzungskontext löst unterstützte kurze Rückfragen auf und merkt ausdrücklich genannte Namen. Das ist keine spontan aus Physik entstandene Semantik und kein allgemeines Verständnis des gesamten Verlaufs. Die [Beschreibung der Dialogberechnung](dialogue-generation.md) trennt diese Fähigkeiten von der reinen Suche.

## Ein Speicher, getrennte Aufgaben

| Inhalt | Ablage | Als Gesprächswissen abgefragt? |
| --- | --- | --- |
| Aktive Gesprächspaare | SQLite `documents` | Ja |
| Tatsächliche Eingaben und Antworten | SQLite `query_history` | Nein |
| Gesprächszustand und Revision pro Sitzung | SQLite `conversations` | Nur im jeweiligen Gespräch |
| Gesicherte frühere Sachtexte | SQLite `document_archive` | Nein |
| Explizite Typrelationen bei optionalem Import | SQLite `relations` | Nur durch den expliziten Relationsbefehl |
| Reproduzierbare Ausgangsdialoge | `memory/fixtures/extension_120.jsonl` | Erst nach Import in SQLite |
| Evaluierungsfragen und erwartete Antworten | `memory/evaluation/extension_queries.json` | Nein |
| Neue Dialogprüfung | `memory/evaluation/dialogue_generalization.json` | Nein |
| Explizite Antwortgrammatik und Themenbezeichnungen | `memory/language/german.json` | Antwortbausteine, keine importierten Fakten |

Die SQLite-Texte sind der verbindliche Bestand. Moden und Suchspektren werden daraus deterministisch abgeleitet. NPZ dient nur noch als ausdrücklich gewähltes Austauschformat. Ein Export oder eine unveränderte Fixture-Datei aktualisiert die laufende Memory nicht; `freqai import` oder die Oberfläche tun dies. Dadurch existieren keine konkurrierenden aktiven NPZ-Bestände. Historische Ergebnisse und Backups bleiben als solche gekennzeichnet.

## Ergänzen ohne Neustart

1. Die neue Antwort sowie Anlass und Quellenmetadaten werden in zusätzliche stehende Wellen übersetzt.
2. Alle vorhandenen Dokumentpakete bleiben unverändert. Es werden nur für neue Einträge FFT/DCT-Codierungen berechnet.
3. Die Einträge werden in einer SQLite-Transaktion mit einer neuen Revision gesichert. Identische Anlass/Antwort/Quelle-Kombinationen werden nicht nochmals hinzugefügt. Eine bereits vergebene ID mit anderem Inhalt erzeugt einen Fehler.
4. Der Server veröffentlicht den vorbereiteten Bestand als Ganzes. Die Modenuhr und der sichtbare Zeitverlauf bleiben erhalten; die Antwort auf den Speichervorgang enthält die aktivierte Revision.
5. Externe CLI-Importe werden anhand der Revision im laufenden Hintergrundloop erkannt und übernommen. Eine anschließende Abfrage synchronisiert zusätzlich vor der Suche.

Die Zielrate des Loops beträgt zehn Auswertungen pro Sekunde. Umfangreiche neue Importe können Rechenzeit benötigen; es gibt keine feste Echtzeitgarantie. Die teure Vorbereitung neuer Einträge läuft bei HTTP-Schreibvorgängen außerhalb der Sperre für laufende Wellenbilder. Zwei schreibende Prozesse werden durch SQLite-Transaktionen und Revisionsprüfung koordiniert. Fehlgeschlagene Datenbanktransaktionen verändern den veröffentlichten Bestand nicht.

## Kontinuität der Schwingung

Früher wurde beim Hinzufügen das gesamte JSON-Archiv neu als eine einzige DCT-Welle codiert. Ihre Länge und damit alle Frequenzen änderten sich. Nun besteht die aktive Welle aus einer **direkten Summe adressierter Dokumentwellen**. Jeder Text und seine Metadaten besitzen eigene feste Moden mit ihrer ursprünglichen Länge N und Frequenzen `omega_k = 2*pi*k*30/N`.

Bei gleicher Simulationszeit stimmen die vollständigen räumlichen Auslenkungen und Quadraturen jedes alten Pakets vor und nach einer Ergänzung exakt überein. Neue Pakete fügen neue Energie hinzu; Energieerhaltung wird bei unverändertem Datenbestand geprüft. Die gemeinsame Darstellung wird aus allen räumlichen Werten aufgebaut und anschließend nur für die Anzeige reduziert. Die Auswahl eines anderen Ausschnitts im Diagramm kann sich durch die größere Gesamtzahl von Punkten ändern; das ist kein Zurücksetzen alter Moden.

Der Suchscore bleibt ein kohärenter Fourier-Kreuzterm. Da Speicher- und Prompt-Welle dieselbe Phase besitzen, kürzt sich `U(t)* U(t) = I` analytisch heraus. Die Implementierung berechnet das verbleibende komplexe Skalarprodukt direkt und vermeidet große temporäre Zeitentwicklungsarrays pro Frage. Die Norm der Suchschlüssel wird anhand vorab summierter Spektralleistungen exakt ausgewertet. Dies spart Rechenarbeit, ohne zusätzliche semantische Fähigkeiten zu erzeugen.

## Auswahl passender Gesprächsantworten

Neben dem ursprünglichen Verfahren wird eine Variante mit festen deutschen Wortformregeln und gewichteter Abdeckung der Eingabewörter geprüft. Die Gewichtung berücksichtigt, wie häufig ein Begriff im gespeicherten Anlasskorpus vorkommt. Das ist berechnete Indexstatistik, kein Gradienten-Training. Die Annahmen und Schwellen sind im Versuchsbericht genannt; sie werden vor der Abschlussauswertung festgeschrieben.

Ein hoher Interferenzwert allein ist keine Wahrheits- oder Angemessenheitswahrscheinlichkeit. Unbekannte Begriffe, veränderte Rollen, Verneinungen und Umschreibungen können eine unpassende Antwort oder eine Nachfrage verursachen. Die Experimente berichten diese Fälle separat. Mehr gespeicherte Beispiele und mehr numerische Dimensionen garantieren kein allgemeines Gesprächsverständnis.
