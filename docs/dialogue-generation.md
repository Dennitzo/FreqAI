# Dialogberechnung mit begrenzter Generalisierung

Der ursprüngliche Abruf verglich eine ganze Eingabe mit einem gespeicherten Anlass. Deshalb scheiterte eine Kombination aus eigener Stimmung und einer Rückfrage. Die neue Verarbeitung zerlegt unterstützte Äußerungen in Gesprächsakte, bindet deren Sprecherrollen an getrennte Frequenzbereiche und bildet eine Antwort aus mehreren Wellenfragmenten. Die ursprünglichen 120 Anlass-Antwort-Paare wurden dafür nicht um die gemeldeten Fehlbeispiele ergänzt.

## Beispiel

```text
Du: Hallo
FreqAI: Hallo! Schön, dass du da bist. Wie geht es dir heute?
Du: Mir geht es gut, wie geht es dir denn?
FreqAI: Das klingt schön. Ich bin bereit für unser Gespräch.
Du: gut und dir?
FreqAI: Das klingt schön. Ich bin bereit für unser Gespräch.
```

Die kombinierte Antwort steht als vollständiger Satz nicht im Ausgangskorpus. Der erste Antwortteil stammt aus der positiven Stimmungsreaktion, der zweite aus der Antwort auf eine Befindensfrage. Die Quellen und Bytebereiche sind im JSON-Feld `decoder.sources` prüfbar. Namensantworten kombinieren ebenfalls Grammatikfragmente mit einem ausdrücklich genannten Namen.

## Woher die Bedeutung stammt

`freqai/semantics.py` enthält ein sichtbares deutsches Lexikon und grammatische Regeln. Sie ordnen beispielsweise „prima“ und „gut“ derselben positiven Stimmung zu. Erkannt werden unter anderem Grüße, Stimmungen, Verneinungen, Rückfragen zum Befinden, Dank, Abschied, Namen und einige Gesprächsthemen. Andere Personen werden vom Nutzer und vom Programm unterschieden. Unbekannte, berichtete oder zitierte Klauseln können ausdrücklich unverstanden bleiben.

Jeder erkannte Akt hat fünf Rollen: Art, Zielperson, Wert, Thema und Verneinung. Diese belegen getrennte Frequenzbereiche. Bekannte Kategorien haben feste kollisionsfreie Plätze; offene Werte verwenden einen Hash und können kollidieren. Ein komplexes Skalarprodukt misst die Übereinstimmung. Harte Prüfungen von Art, Rolle und Stimmung begrenzen vorher die erlaubten Kandidaten. Das bloße Teilen von Frequenzkomponenten ist kein Beweis für eine richtige Antwort.

Generische Stimmungsantworten dürfen keine zusätzliche Situation aus einer gespeicherten Vorlage übernehmen. Deshalb werden etwa Vorlagen über eine bestimmte Präsentation nicht als allgemeine Antwort auf „Ich bin gestresst“ verwendet. Fehlt eine hinreichend allgemeine Vorlage, greift die sichtbare Antwortgrammatik in `memory/language/german.json`. Sie wird anhand von Dateistand und Größe beim nächsten Aufruf neu geladen.

Die zweite Entwicklungsrunde ergänzt grammatische Korrekturen, nachgestellte Verneinungen und weitere anaphorische Rückfragen. Ein verneintes Prädikat bleibt ausdrücklich verneint: „nicht traurig“ wird nicht zu „glücklich“. Die Zusatzfelder `predicate` und `subject` beschreiben das erkannte Prädikat und einen expliziten Bezug; sie sind Metadaten außerhalb der fünf Frequenzrollen und berechtigen nicht zu neuen Faktenbehauptungen. Aussagen über andere Menschen oder Tiere können als klar zugeordnete Aussage des Nutzers wiedergegeben werden. Eine solche Wiedergabe schafft Gesprächsbezug, ist aber noch keine eigenständige Schlussfolgerung über diese Person.

## Tatsächliche Zusammensetzung der Wellen

Sei `C_n` die orthonormale Cosinustransformation und `a_j` das Spektrum eines Quelltextes. Eine Bytebereichsprojektion `P_j` wählt einen vollständigen UTF-8-Textabschnitt aus, und `E_j` platziert ihn an seiner geordneten Position in einer neuen Antwort mit `L` Bytes:

```text
a_out = C_L · Σ_j E_j · P_j · C_njᵀ · a_j
```

`freqai/synthesis.py` berechnet dies effizient durch inverse Transformation, Bereichsauswahl, geordnete Einbettung und eine Ausgangs-DCT. Die Quellkoeffizienten werden zum Abfragezeitpunkt aus beiden Quadraturen `q` und `p` zurückgewonnen und vor der Nutzung per SHA-256 geprüft. Der fertige Antwortstring wird nicht zuerst zusammengebaut und dann als Ganzes an `encode_text` übergeben. Er entsteht durch die inverse Transformation des berechneten Ausgangsspektrums, Rundung auf Bytes und striktes UTF-8-Decodieren. Ein separat aus geprüften Quellbytes gebildeter Digest prüft das Ergebnis.

Die Reihenfolge ist Teil des Operators. Eine ungeordnete Addition beliebiger Textspektren würde keine Textverkettung liefern. Wort- und Satzgrenzen, Auswahl, Reihenfolge und dynamische Namenswerte stammen aus Regeln bzw. Gesprächsdaten. Die Quellwellen liefern den rekonstruierbaren Textinhalt. Alle 256 Bytewerte lassen sich codieren; daraus folgt nicht, dass das System die Bedeutung beliebiger Sätze kennt.

Für eine ausdrücklich verneinte Stimmung oder eine zugeordnete Wiedergabe darf auch ein unveränderter Abschnitt der aktuellen Eingabe verwendet werden. Die Quelle `input:prompt` bezeichnet dabei das ganze codierte Original mit überprüfbaren Bytegrenzen. Normalisierte oder ergänzte Inhalte werden nicht als wörtliches Zitat des Nutzers ausgegeben.

Die kontinuierliche Darstellung im Dashboard zeigt die adressierten Dokumentwellen. Die temporär zusammengesetzte Antwortwelle wird pro Antwort berechnet; der Gesprächsverlauf wird nicht automatisch als neues Antwortwissen aufgenommen.

## Gespräche und Live-Speicher

`memory/memory.sqlite3` bleibt der zentrale aktive Datenspeicher. Ein Gespräch besitzt eine Kennung, einen begrenzten strukturierten Zustand und eine Revision. Name, letztes Thema und Gesprächsrundenzähler sowie der tatsächliche Verlauf werden atomar gespeichert. Gleichzeitige Abfragen derselben Sitzung werden geordnet; Revisionsprüfungen verhindern verlorene Zustände bei mehreren Prozessen. Getrennte Sitzungen erhalten keinen fremden Kontext. Die lokale Anwendung hat keine Benutzerkonten; Gesprächskennungen sind keine Authentifizierung.

Die Oberfläche speichert ihre Kennung im Tab und stellt dessen Verlauf beim Neuladen wieder her. „Neues Gespräch“ legt eine neue Kennung an. In der CLI verbindet `--session meine-kennung` mehrere Aufrufe. Für HTTP verwendet `POST /api/ask` das optionale Feld `session_id`; ohne dieses Feld arbeitet die Abfrage ohne persistenten Gesprächskontext. `GET /api/conversation?session_id=...` liefert nur den zugehörigen Verlauf.

Neue Dokumente werden weiterhin ohne Neustart aufgenommen. Der semantische Katalog gehört zum unveränderlichen RAM-Speicherstand; nach Veröffentlichung neuer Dokumente wird er für diesen Stand neu abgeleitet. Bestehende Dokumentwellen und die laufende Modenuhr bleiben dabei erhalten.

## Was damit nicht gelöst ist

Dieses System benötigt kein Gradienten-Training. Es enthält jedoch ausdrücklich von Menschen bereitgestelltes Sprachwissen. Die Wellenrepräsentation macht Informationen numerisch verarbeitbar; sie entdeckt nicht von selbst die Bedeutung eines unbekannten Wortes, verifiziert keine Fakten und ersetzt kein allgemeines Sprachmodell. Ein frei gewähltes Thema, Ironie, komplexe Nebensätze oder ein neuer Bezug können weiter zu falscher Zuordnung oder einer Nachfrage führen. Für neue Sachinformationen braucht es gespeicherte Belege oder eine zusätzliche Wissensquelle.

Die erste unabhängige Dialogprüfung enthielt 134 Runden mit getrennten Entwicklungs-, Abschluss- und Belastungsfällen. Ihr enttäuschendes Abschlussergebnis von 24/50 führte zu einer zweiten Entwicklungsrunde. Die erste Version samt Code, Grammatik, Bewertungslogik und Ergebnissen bleibt unter `results/dialogue/round1/` nachprüfbar. Die alten Fälle gelten anschließend ausdrücklich als Entwicklungsmaterial. Zusätzlich wurden 80 neue Runden erstellt, davon 40 zur Entwicklung und 40 als frische Abschlussfälle. Der Vergleich verwendet dieselbe vorab festgelegte Bewertungslogik für den archivierten und den neuen Code.

Erfolg wird am ausgegebenen Text und an den erkennbaren Gesprächsakten gemessen; reine Decoder-Integrität ist eine getrennte numerische Prüfung. Eine neue vollständige Antwort kann auch eine einfache Kombination von Grammatik und Eingabezitat sein. Die Neuheitszahl allein beweist deshalb kein besseres Verständnis. [Aktuelle Ergebnisse und verbleibende Fehler](../results/dialogue/round2/report.md).
