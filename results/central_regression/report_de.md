# FreqAI: reproduzierbare numerische Ergebnisse

Das Ergebnis ist ein trainingsfreier, exakt dekodierbarer Wellenspeicher mit lexikalischem Abruf. Die Berechnungen belegen keine natürliche physikalische Bedeutung von Text und keine allgemeine Sprachintelligenz.

## Versuchsaufbau

Zeitpunkt UTC: 2026-09-06T07:21:16.187546+00:00. Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1.

40 Speichertexte, 32 bekannte Fragen, 10 Nullfälle und 8 semantische Paraphrasen ohne gemeinsame Wörter. Fragetexte stehen ausschließlich in der Evaluierung, niemals im Speicher. Alle zwölf Kombinationen aus raw_bytes/words/hybrid und D=256/1024/4096/16384 werden vollständig ausgewertet. Schwelle 0,18 und hybride Gewichtung 75/25 sind feste Regeln; es gibt keine Optimierung von Modellparametern. Ein Leerprompt wird als abgewiesene Eingabe erfasst. Die anderen Nullfälle prüfen reguläre Nichtantworten. Der feste Wortüberlappungsfilter gehört zum System und wird in der Auswertung nicht umgangen.

Die Konfigurationen werden auf demselben kleinen, manuell erstellten Testkorpus verglichen. Eine anhand dieser Ergebnisse gewählte Konfiguration besitzt damit noch keine unabhängig bestätigte Generalisierungsleistung. Es werden weder Synonymtabellen noch vortrainierte Embeddings noch Sprachmodelle verwendet.

## Umkehrbare Textkodierung

UTF-8-Bytes werden affinlinear auf [-1,1] abgebildet und orthonormal mit DCT-II transformiert. Jeder Koeffizient steuert einen stehenden Kosinusmodus. Die normierten Quadraturen q_k(t)=a_k cos(ω_k t), p_k(t)=a_k sin(ω_k t) erlauben a_k=q_k cos(ω_k t)+p_k sin(ω_k t). Inverse DCT, Rundung und UTF-8-Dekodierung rekonstruieren den Text; SHA-256 erkennt beschädigte Nutzdaten. Die Frequenzskalierung ist frei gewählt, kein entdecktes Naturgesetz.

Exakte Textrekonstruktion: **117/117** Versuche, einschließlich Leertext, Nullbytes, Unicode, Zufallstexten und t bis 10^12 s. Rekonstruktion aus vollständigen räumlichen Quadraturgittern: **117/117**.

Maximaler Koeffizientenfehler: 8.882e-16. Maximaler relativer Fehler der normierten Quadraturenergie: 4.502e-16; der frequenzgewichteten physikalischen Modenenergie: 4.053e-16. Parseval-Fehler: 4.860e-16.

Diese Langzeittests verwenden bekannte identische Zeitstempel für Vorwärts- und Rücktransformation. Sie belegen keine Uhrengenauigkeit eines realen Oszillators bei 10^12 s. Es wird analytisch ausgewertet, nicht über 10^12 Sekunden numerisch integriert. Nur die Auslenkung reicht an Nulldurchgängen nicht aus; drei gesonderte Versuche zeigen einen unsichtbaren Modus bei vollständig erhaltener Quadratur. Die Anzeige darf heruntergesampelt werden, zur verlustfreien Dekodierung bleiben alle Modi gespeichert. Float64-Koeffizienten benötigen roh acht Bytes je UTF-8-Byte zuzüglich Metadaten; die Darstellung ist keine Kompression.

## Interferenz und Abruf

Für normierte Fourier-Schlüssel K_i und Prompt Q liefert der kohärente Kreuzterm S_i=(||K_i+Q||²-||K_i||²-||Q||²)/2=Re⟨K_i,Q⟩ den Vergleichswert. Nach Parseval entspricht er exakt dem Skalarprodukt der ursprünglichen Merkmalsvektoren. Eine gemeinsame Phasenentwicklung verändert dieses Ergebnis nicht. Die Fourierdarstellung erzeugt deshalb aus sich heraus keine neue Semantik. Der ausgewählte Antworttext wird aus seinen stehenden Modi dekodiert.

Maximaler Unterschied zu direkter Kosinusähnlichkeit: 7.994e-15. Maximaler Unterschied zur expliziten Interferenzenergieberechnung: 8.216e-15. Zeitlich unveränderte Antworten: **180/180**.

Der erste vollständige Lauf zeigte zwei zeitabhängige Antwortwechsel bei mathematisch gleichen Scores (words, D=256, Japan-Frage). Die numerische Abweichung lag bei 5,55e-17. Sortierung und Schwellwertentscheidung verwenden deshalb jetzt einheitlich auf zwölf Dezimalstellen gerundete Lesewerte (Auflösung 1e-12); die ausgegebenen Rohscores bleiben unverändert. Stabile Sortierung erhält bei Gleichstand die ursprüngliche Dokumentreihenfolge. Die negative Messung wurde unter ../iterations/01_before_tie_fix.json und .md aufbewahrt. Diese Korrektur stabilisiert die Numerik, sie verbessert nicht das Sprachverständnis.

| Merkmale | D | Bekannte Fragen | Nullfälle korrekt | Paraphrasen korrekt |
|---|---:|---:|---:|---:|
| raw_bytes | 256 | 5/32 | 1/10 | 0/8 |
| raw_bytes | 1024 | 5/32 | 1/10 | 0/8 |
| raw_bytes | 4096 | 5/32 | 1/10 | 0/8 |
| raw_bytes | 16384 | 5/32 | 1/10 | 0/8 |
| words | 256 | 31/32 | 6/10 | 0/8 |
| words | 1024 | 31/32 | 6/10 | 0/8 |
| words | 4096 | 31/32 | 6/10 | 0/8 |
| words | 16384 | 31/32 | 6/10 | 0/8 |
| hybrid | 256 | 31/32 | 6/10 | 0/8 |
| hybrid | 1024 | 31/32 | 6/10 | 0/8 |
| hybrid | 4096 | 31/32 | 6/10 | 0/8 |
| hybrid | 16384 | 31/32 | 6/10 | 0/8 |

Die Paraphrasenprüfung ist absichtlich streng: semantisch verwandte Formulierungen besitzen keine identischen Wörter. Ein fester Wortüberlappungsfilter kann hier keine richtige Antwort auswählen. Die Roh-Rangfolge ohne Antwortfilter ist zusätzlich in retrieval_summary.csv und den Einzelfällen ausgewiesen. Ein Nullfall mit bekannten Stichwörtern kann trotzdem eine falsche gespeicherte Stelle aktivieren. Die Schwelle ist kein zuverlässiger Wissens- oder Wahrheitsdetektor.

### Fehler der Referenzkonfiguration hybrid, D=4096

- q02 (known): 'Welche Hauptstadt hat Japan?'; erwartet f02, gewählt f03, höchster Score 0.4499.
- n04 (null): 'Wie heißt die Hauptstadt Brasiliens?'; erwartet None, gewählt f02, höchster Score 0.5896.
- n05 (null): 'Welche Spannung benötigt das Gerät ZXQ-987?'; erwartet None, gewählt f27, höchster Score 0.2458.
- n06 (null): 'Welcher Oktopus lebt in meinem Aquarium?'; erwartet None, gewählt f11, höchster Score 0.2617.
- n07 (null): 'Wie viel kostet morgen ein Kilogramm Gold?'; erwartet None, gewählt f30, höchster Score 0.2546.
- s01 (semantic_no_overlap): 'Welches Haustier sollte täglich spazieren gehen?'; erwartet p01, gewählt None, höchster Score 0.0599.
- s02 (semantic_no_overlap): 'Wer diagnostiziert Leiden bei Patienten?'; erwartet p02, gewählt None, höchster Score 0.0295.
- s03 (semantic_no_overlap): 'Welche Berufsgruppe steuert Wasserfahrzeuge?'; erwartet p03, gewählt None, höchster Score 0.1857.
- s04 (semantic_no_overlap): 'Womit hält man sich an frostigen Tagen warm?'; erwartet p04, gewählt None, höchster Score 0.0382.
- s05 (semantic_no_overlap): 'Was hilft dem Organismus beim nächtlichen Regenerieren?'; erwartet p05, gewählt None, höchster Score 0.0467.
- s06 (semantic_no_overlap): 'Was tankt ein benzinbetriebenes Auto?'; erwartet p06, gewählt None, höchster Score 0.0243.
- s07 (semantic_no_overlap): 'Welche Künstler erschaffen Gemälde?'; erwartet p07, gewählt None, höchster Score 0.0291.
- s08 (semantic_no_overlap): 'Wer löscht lodernde Flammen?'; erwartet p08, gewählt None, höchster Score 0.0257.

## Rauschen und endliche Speicherkapazität

352 Versuche addieren Gaußrauschen zu den Textkoeffizienten; je Rauschstärke 32 feste Seeds. Nicht erkannte falsche Texte: **0**. Eine Prüfsumme korrigiert Fehler nicht: oberhalb der Rundungsreserve wird die Dekodierung abgewiesen. noise.csv enthält jeden Versuch, die Grafik vergleicht die Messung mit der unabhängigen Gauß-Rundungswahrscheinlichkeit.

Eine zusätzliche HRR-Simulation überlagert 8, 32, 128 oder 512 zufällige Einträge in nur einem gemeinsamen Spektrum bei D=256/1024/4096 und drei Seeds. Nach Entbindung wächst die gemessene Störenergie ungefähr mit N-1. Die Auswahl benötigt weiterhin ein separat gespeichertes Wörterbuch der Kandidaten. Dies ist keine verlustfreie Alternative zum adressierten Hauptspeicher. hrr_capacity.csv zeigt auch die verlorenen Einträge; hash_collisions.csv misst Kollisionen von 2000 künstlichen Tokens.

Der Hauptspeicher vermeidet die verlustreiche Entmischung, indem er Dokumentadressen und alle Nutzdatenmodi behält. Seine Größe steigt mit der Datenmenge. Eine einzige endliche Welle ohne Adressen, Wörterbuch oder zusätzliche Struktur kann nicht beliebig viele unabhängige Texte verlustfrei speichern.

## Ergebnis und Reproduktion

Gefunden wurde eine numerisch überprüfte Lösung für Text → stehende Welle → Text und einen kohärenten, trainingsfreien Abruf aus einem endlichen Korpus. Nicht gelöst wurde die weitergehende Forderung nach allgemeinem Textverständnis oder frei erzeugten richtigen Antworten allein aus physikalischer Schwingung. Diese Einschränkung darf nicht durch längeres Wiederholen gleicher Rechnungen als gelöst dargestellt werden.

Aufruf aus dem Projektverzeichnis: `.venv\Scripts\python.exe experiments\run_experiments.py`. Die Dateien werden reproduzierbar neu erzeugt; Zeitstempel und Laufzeiten können abweichen. Der Benchmark-Hash und die Versionsangaben stehen in results.json. Numerische Integritätsverletzungen führen zu Exitcode 1; schwache Abrufleistung wird vollständig berichtet.

Dateien: codec.csv, noise.csv, position_only.csv, retrieval_cases.csv, retrieval_summary.csv, invariants.csv, hrr_capacity.csv, hash_collisions.csv, results.json sowie vier PNG-Abbildungen.
