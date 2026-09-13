# Großer Bestand: Start, Compiler und Laufzeittest

Stand: 13.09.2026. Zentrale Memory: 569.210 Informationstexte, Revision 11.

Der monolithische Aufbau wurde für große Bestände durch einen dateigestützten Volltext- und Schwingungscache ersetzt. Damit entfallen die vorher mindestens 146 GiB gleichzeitig gehaltenen Zahlenarrays. Start und Antwortcompiler schreiben Fortschritt sofort ins Terminal. `start.ps1` protokolliert stdout, stderr und Fehlercodes als UTF-8 unter `runtime/start-*.log`; ein nativer Testprozess unter Windows PowerShell 5 prüft ausdrücklich Fortschritt auf stderr, einen simulierten MemoryError und Exitcode 7.

## Gemessener Vollbestand

| Prüfung | Ergebnis |
|---|---|
| Erster vollständiger Aufbau einschließlich Serverstart | 575,97 Sekunden |
| Dabei beobachtete maximale Summe der Prozess-RSS | 2,36 GiB, einschließlich 32 Worker |
| Warmstart mit endgültigem GPU-Anzeigepfad | 30,31 Sekunden |
| Wiederverwendete Dokumente beim Warmstart | 569.210; neu compiliert: 0 |
| Beobachtete Warmstart-RSS-Spitze des Prozessbaums | 0,65 GiB |
| Wellenbilder im abschließenden Messfenster | 142–187 ms |
| Ausgeführte CUDA-Arbeit | GPU 0 und GPU 1, beide Quadro RTX 8000 |
| Vollständiger abgeleiteter Cache | 25.596.080.128 Bytes, rund 23,84 GiB |
| Unabhängiger Vergleich aller Indexabschnitte | 569.210 Texte, 2.466.967.143 UTF-8-Bytes, exakt gleich der zentralen Memory |
| Automatisierte Tests | 462 bestanden in 29,76 Sekunden |

Die RSS-Messung summiert den Prozessbaum; sie ist keine Messung des systemweiten Commit-Speichers oder des GPU-Speichers. Ein separater Wächter hätte den gestarteten Prozessbaum bei mehr als 12 GiB RSS beendet. Der Grenzwert wurde nicht erreicht. Die Zielrate von zehn Wellenbildern pro Sekunde wird mit den gemessenen 142–187 ms nicht durchgehend erreicht.

## Schutz gegen erneuten unbeschränkten Speicheraufbau

- Große Bestände gelangen über `MemoryStore.load_memory()` automatisch in den Festplattenmodus; direkte übergroße dichte Aufbauten werden vor der Allokation abgelehnt. Auch Wachstum während des Serverbetriebs wird geprüft.
- Der Vollbestand wird in etwa 2 MiB großen Textpaketen verarbeitet. Fertige Pakete sind transaktional bestätigt und werden nach einem Abbruch wiederverwendet. Ein Einzeltext über 16 MiB benötigt vorher eine Aufteilung.
- Vollständige Textkoeffizienten und Textabschnitte bleiben auf der Festplatte. Für die sichtbaren Ortskoordinaten werden höchstens 64 MiB gewichtete Koeffizienten vorgehalten und auf die ausgewählten GPUs verteilt. Ein CPU-Pfad bleibt verfügbar. Gesamtenergien sind zeitlich konstant und werden einmal über alle Koeffizienten berechnet.
- Die Dokument-API liefert beim großen Bestand auch ohne Seitenparameter nur 50 Dokumente. Der Server liest beim Start und beim Ergänzen keine vollständige Python-Dokumentliste mehr ein.
- Ein Anfragecompiler erhält höchstens 120.000 Zeichen aus ausgewählten Informationsabschnitten und kleinen deklarativen Basisinformationen. Zwei Arbeitsbereiche bleiben im RAM; acht persistente Cacheplätze begrenzen die Anzahl der Anfragecompiler. Die Kontextanzeige verwendet vorhandene Auswahl und Compiler wieder.

Die Tests vergleichen die CPU- und CUDA-Koordinaten mit der vollständigen inversen DCT, einschließlich großer Simulationszeiten. Weitere Regressionstests decken Abbruch/Wiederanlauf, inkrementelles Ergänzen, Cache-Wiederverwendung, Artikelenden, automatische Umschaltung, API-Seitengrenzen und ausbleibende erneute Indexabfragen für Kontextbilder ab.

## Weboberfläche und verbleibende Qualitätsschwäche

Die tatsächliche Weboberfläche wurde über den Browser bedient. Sie zeigt 569.210 Texte, 2.569.048.870 Text-/Metadatenmoden und ausgeführte Arbeit auf beiden GPUs. Prompts werden ohne Serverfehler verarbeitet; Gespräch und Kontext bleiben beim Neuladen erhalten.

- „Wie heißt du?“ → „Ich heiße FreqAI.“
- „Was ist eine Frequenz?“ → „Die Frequenz ist der Kehrwert der Periodendauer.“
- Bei „Und welche Einheit hat sie?“ wurde zunächst ein allgemeiner Einheitenabschnitt bevorzugt. Die Auswahl behält jetzt das vorherige Thema bei; ein Regressionstest prüft dies. Der echte Decoder lieferte danach dennoch die tautologische Antwort „Die Einheit der Frequenz ist die SI-Einheit der Frequenz.“ Diese fachliche Schwäche ist **nicht behoben** und wird nicht als erfolgreiche Wissensantwort gewertet.

Der neue große Modus wählt Informationsabschnitte vor der Generierung aus. Damit verwendet eine Antwort nicht sämtliche Wikipedia-Wortübergänge gleichzeitig. Vollständige Speicherung und technische Funktion belegen weiterhin keine allgemeine Sprach- oder Schlussfolgerungsfähigkeit. Es wurden keine Antwortpaare hinzugefügt und keine vollständigen Antworten hinterlegt.

Rohmessungen und die Testantworten stehen in `results/large_dataset_runtime.json`. Reproduzierbare Prüfwerkzeuge: `scripts/verify_large_start.py` und `scripts/audit_paged_cache.py`. Der erste startet tatsächlich `start.ps1` und lässt den Server nach Erfolg laufen. Der abgeleitete Cache und die Laufzeitlogs sind durch `.gitignore` ausgeschlossen.
