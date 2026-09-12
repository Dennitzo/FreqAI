# Diagnose: beiläufige Eingabemerkmale verknüpfen Fachantworten mit Smalltalk

Die Ursache ist im Importcompiler bestätigt. `_input_features(prompt)` sammelt sämtliche erkannten Akte eines Eingabetextes. Anschließend wird **jeder Wortübergang der gesamten zugehörigen Antwort jedem dieser Merkmale zugeschlagen**. Damit kann ein abschließendes „danke“ die vollständige Fachantwort in den Smalltalk-Kanal einspeisen. Eine korrekte Fourierberechnung verhindert diese falsche sprachliche Zuordnung nicht.

Es wurden weder der eingefrorene Korpus noch produktiver Code geändert. Keine Holdout-Datei wurde gelesen. Die Diagnose verwendet den eingefrorenen v3-Korpus, SHA-256 `832754653cdd1dcd0c2c0cb7043d3edae71f8e28558b5ed7a145362c0b73d9fa`.

## Tatsächlich betroffene Quelldaten

| Kanal | Betroffene externe Paare | Grund |
|---|---:|---|
| `semantic:thanks:user::` | 2 von 533 OASST-Paaren | Dieselbe Fachfrage zum Zuschneiden und Schleifen von Cola-Flaschen endet beiläufig mit „danke“. Zwei vollständige Fachantworten werden als Dankesreaktionen eingekoppelt. |
| `semantic:mood_statement:user:positive:wellbeing` | 1 von 533 OASST-Paaren | „Erkläre das Prinzip der Markovkette für einen Schüler der 10. Klasse verständlich“ wird nach der Zahl/Punkt-Grenze als Stimmungsaussage „Klasse verständlich“ interpretiert. |
| `intent:clarification` | 1 von 10.846 GermanQuAD-Paaren | Die Frage, wo über den Ursprung serbokroatischer Sprachen gestritten wird, endet mit „oder nicht?“. Der Parser behandelt dies als unsichere Befindlichkeitsantwort, obwohl kein entsprechender Gesprächskontext besteht. |

OASST trägt keinen der untersuchten `intent:clarification`-Einträge bei; GermanQuAD trägt keine Dankes- oder positiven Stimmungseinträge bei. Alle betroffenen OASST-Einträge liegen bereits unter den ersten 500 importierten OASST-Paaren.

Betroffene IDs:

- OASST Dank: `0b264c79-186c-42d7-9656-c3efaa9a5fef`, `b7a66e4d-3d92-4b3c-b3ab-d0de4948a1c2`.
- OASST positive Stimmung: `9ce4f7b9-d92e-4641-8b62-922e871e1b95`.
- GermanQuAD Rückfrage: `germanquad-train-65873`.

Die vollständigen Texte, Parser-Akte und Kanalzählungen stehen in [OASST-Diagnose](semantic_contamination_oasst2.json) und [GermanQuAD-Diagnose](semantic_contamination_germanquad.json).

## Isolierter Vergleich im kleinen Modell

Ein separater Prozess verglich drei Modelle: 621 eigene Einträge; dieselben Einträge plus die ersten 500 OASST-Paare; dieselben 1.121 Einträge mit ausschließlich `word:`-Importmerkmalen für öffentliche Prompttexte. Dabei bleiben Texte, Wortübergänge, Wortschatz und öffentliche lexikalische Konditionierung erhalten. Inferenz-Promptmerkmale sind bei allen Varianten unverändert. Konfiguration: Ordnung 2, Beam-Decodierung, Seed 17, höchstens 40 Tokens, öffentliche Paargewichte **1,0**. Dies ist ein gezielter Kausaltest und keine Bewertung der späteren Gewichte 0,1 beziehungsweise 0,05.

| Eigener Diagnoseprompt | Basismodell | Plus 500, automatische Dialogmerkmale | Plus 500, öffentliche Importmerkmale lexikalisch |
|---|---|---|---|
| `Danke.` | „Gern. Du kannst in deinem Tempo weitererzählen.“ | Antwort über Gefahren beim Bearbeiten von Glas, Schutzbrille und Staubschutzmaske. | „Gern. Du kannst jederzeit einen neuen Gedanken einbringen.“ |
| `Mir geht es gut.` | „Es ist schön, wenn dir etwas gut gelingt. Welcher Teil daran bedeutet dir besonders viel?“ | Gleicher Text. | Gleicher Text. |
| `Hallo` | „Hallo! Schön, dass du hier bist.“ | „Hallo Heikki! Wie kann ich Ihnen heute weiterhelfen?“ | „Hallo! Schön, dass du hier bist.“ |

Zusätzlich wurde direkt aus den gespeicherten komplexen Feldern die Wortintensität nach der Demodulation gemessen:

- Im Dankeskanal erhalten `glas`, `colaflaschen` und `werkzeuge` nach dem Import jeweils ungefähr **0,0049505** Wahrscheinlichkeitsmasse. Bei lexikalischer Zuordnung öffentlicher Paare liegt ihre Masse wieder im numerischen Rauschen unter `1e-30`.
- Im positiven Stimmungskanal erhält `markov-prozess` ungefähr **0,00677966** Wahrscheinlichkeitsmasse. Die lexikalische Importvariante entfernt auch diese Kopplung bis auf numerisches Rauschen. Dass die konkrete positive Diagnoseantwort unverändert blieb, widerlegt die Speicherverunreinigung nicht; hier änderte sich lediglich noch nicht der ausgewählte Beam.

Die vollständigen Ausgaben und Werte stehen in [semantic_contamination_probe.json](semantic_contamination_probe.json). Diese Zusatzdiagnose misst keine FFT-Rundlaufabweichung und enthält keine Aussage über allgemeine Gesprächsqualität oder Holdout-Leistung.

## Grenze der temporären Unterklasse und produktive Schlussfolgerung

Die nur im Diagnoseprozess verwendete Unterklasse schaltete strukturierte Importmerkmale über eine Menge öffentlicher **Prompttexte** aus. Eine Nachkontrolle fand genau eine Überschneidung mit einem eigenen Prompt: **„Wie geht es dir?“**. Dadurch wurde in diesem Diagnosemodell auch der eigene Eintrag dieses Prompts ohne seinen strukturierten Assistant-Befindlichkeitskanal kompiliert. Dieser Fehler betrifft keinen der drei untersuchten Dankes-, User-Stimmungs- oder Begrüßungskanäle, schränkt aber eine Verallgemeinerung auf die gesamte Gesprächsfähigkeit ein. Er ist ausdrücklich in der JSON-Datei dokumentiert.

Ein produktiver Fix muss die Entscheidung deshalb **pro Paar und Quelle** treffen: eigene Gesprächspaare behalten ihre strukturierten Merkmale; öffentliche QA-Paare bekommen eine explizite Importoption für rein lexikalische Konditionierung. Der Merkmalscache muss mindestens `(prompt, include_semantics)` als Schlüssel benutzen. Gleiche Prompttexte aus verschiedenen Quellen bleiben dadurch unabhängig korrekt kodiert. Explizite Annotationen des selbst verfassten Sprachkorpus werden nicht entfernt. Ein entsprechender Root-Patch ist vorbereitet; seine Anwendung und Bewertung erfolgen außerhalb dieser Diagnose.

Globale Quellengewichte können die Folgen abschwächen, korrigieren die falsche Merkmalszuordnung aber nicht. Die finale Entscheidung über den Encoder ist anhand der separat laufenden Entwicklungsprüfung zu treffen; die hier gezeigten drei Prompts sind kein Ersatz dafür.
