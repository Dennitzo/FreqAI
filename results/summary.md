# Ergebnis vom 6. September 2026

**Historischer erster Versuchsstand.** Der aktuelle Bestand verwendet 120 Alltags-Gesprächspaare und zentrale Live-Persistenz. Siehe [neuer Experimentbericht](extended/report.md), [Live-Import](live_memory/active_import.json) und [aktuelle Anleitung](../README.md).

**Gefunden und implementiert:** ein verlustfrei rücklesbarer stehender Wellenspeicher, ein trainingsfreier Interferenzabruf aus Texten und ein Fourier-Algorithmus für neue Aussagen aus expliziten Typbeziehungen. **Nicht nachgewiesen:** eine natürliche Bedeutungsfrequenz von Sprache oder allgemeine richtige Antworten aus bloß fortgesetzter physikalischer Schwingung.

## Ausgeführte Berechnungen

| Untersuchung | Umfang | Beobachtung |
| --- | ---: | --- |
| UTF-8 → DCT → zeitentwickelte Quadraturen → inverse DCT → UTF-8 | 117 Versuche | 117 exakt, auch aus vollständigen räumlichen Quadraturgittern |
| Zeitinvarianz der Antwort | 180 Vergleiche im zweiten Lauf | 180 identisch; erster Lauf 178/180, Ursache und Korrektur archiviert |
| Promptabruf | 12 Konfigurationen × 50 Fragen = 600 Fälle je Lauf | Rohbytes 5/32 bekannte Antworten, Wörter und Hybrid 31/32 |
| Unbekannte Fragen beim Default | 9 Fragen + 1 Leerprompt | 5 Fragen korrekt unbeantwortet, 4 falsche Treffer; Leerprompt abgewiesen |
| Semantische Umschreibungen ohne gemeinsame Wörter | 8 Fälle je Konfiguration | Default 0/8; kein allgemeines Sprachverständnis |
| Gaußrauschen auf Wellenkoeffizienten | 352 Versuche | Beschädigungen werden abgewiesen, kein unerkannter falscher Text im Test |
| Holographische Superposition mit gemeinsamem Spektrum | 36 Konfigurationen | Endliche Kapazität und Übersprechen; Kandidatenwörterbuch weiter erforderlich |
| Typrelationen im Frequenzraum gegen klassische Graphsuche | 96 Graphen, 768 Anfragen | 768 Übereinstimmungen, darunter 311 belegte Beziehungen und 457 unbekannte |

Der maximale relative Fehler der Quadraturnorm beträgt **4,50 × 10^-16**. Die Interferenzscores unterscheiden sich höchstens **2,22 × 10^-16** von direkter Kosinusähnlichkeit. Diese Übereinstimmung ist eine Kontrolle des Basiswechsels und kein Nachweis zusätzlicher Semantik. Die getesteten großen Zeitparameter sind analytisch ausgewertete Zeitpunkte, keine tatsächlich über Jahrtausende durchlaufene Simulation.

Der erste Lauf fand ein reales Rundungsproblem: Theoretische Gleichstände kippten durch Abweichungen von wenigen 10^-17. Eine feste Ausleseauflösung von 10^-12 und stabile Korpusreihenfolge beheben dies. Merkmale, Schwelle und Standarddimension wurden dabei nicht an die Antworten angepasst. Zwei gezielte Regressionstests und der vollständig wiederholte Experimentloop bestätigen die Korrektur.

Neue Ausgabe aus expliziten Quellen: `Pudel ist ein Tier.` aus `Pudel is_a Hund` und `Hund is_a Tier`, mit zwei Frequenzraum-Schritten und geprüftem Beweispfad. Das direkte Tripel ist nicht gespeichert. Die Semantik von `is_a`, deren Transitivität und das deutsche Satzmuster sind programmiert. Zwei zusätzliche frische Prozesse bestätigten byteidentische Graphen und Falldaten des finalen Relationsexperiments.

## Laufzeitvalidierung

- **104 Tests bestanden**: Codec, Speicher/Persistenz, numerische Invarianten, Relationen sowie HTTP/Worker.
- Python-Quellen kompilierbar; `pip check` meldet keine gebrochenen Abhängigkeiten.
- Echtes Edge-Dashboard geprüft: wechselnde Welle, Promptantwort samt Quellen, sofortiges Einlesen, mobile Ansicht und keine JavaScript-Fehler.
- Demo-Endzustand läuft lokal auf `http://127.0.0.1:8765` mit acht Texten. Der Prozess wurde am Erstelltag im Hintergrund gestartet und läuft bis zu seiner Beendigung beziehungsweise dem nächsten Rechnerneustart.

## Nachweise

- [Vollständiger numerischer Bericht](numerical/report_de.md), [JSON](numerical/results.json), [Abrufgrafik](numerical/retrieval_accuracy.png)
- [Unveränderte negative erste Messung](iterations/01_before_tie_fix.json)
- [Relationale Messung](reasoning/report.md), [Reproduzierbarkeitsnachweis](reasoning/reproducibility/verification.json)
- [Browserprüfung](runtime/browser-check.json), [geprüfter laufender Endzustand](runtime/live-demo-check.json), [Live-Screenshot](runtime/live-demo.png)
- [Primärquellenrecherche](../docs/research.md), [Algorithmus und Formeln](../docs/algorithm.md)

Die Recherche umfasste vier erfolgreiche lokale SearXNG-Suchanfragen ohne Ersatzanbieter und direkte Primärquellenabrufe. Verwandte Fourier-/holographische Text- und Symbolspeicher existieren bereits. Deshalb wird weder historische Einzigartigkeit noch eine vollständige Lösung allgemeiner Sprachintelligenz behauptet.
