# Unabhängige Evaluation der öffentlichen Corpuserweiterung

Die Qualitätsurteile stammen von einem separaten AI-Evaluationsagenten. Es handelt sich weder um eine Humanstudie noch um eine verblindete Bewertung. Die Stichproben sind klein; Unterschiede sind beschreibend und kein statistischer Wirksamkeitsnachweis.

## Verfahren

- Vor den Änderungen wurden 40 neue Alltagschat-Turns in 20 Zweiergesprächen eingefroren: 20 Development und 20 bis zum finalen Codefreeze zurückgehaltene Holdout-Turns. Keine Evaluationsfrage wurde als Importbeispiel verwendet. Der finale normalisierte Vergleich aller 40 Chatprompts mit den 11.379 öffentlichen Eingabefragen und Antworttexten ergibt jeweils 0 vollständige Übereinstimmungen; Einzelwörter und Themen dürfen sich selbstverständlich überlappen. Details: `final_chat_overlap_check.json`.
- Die Ausgangsdaten enthalten 621 SQLite-Dokumente. Die Evaluierung liest SQLite nur für die eingefrorene Ausgangskopie und verändert die produktive Datenbank nicht. Der gleiche Generation-Runtimeadapter wie im Server erhält die Dokumente über einen kleinen Testhalter; der separate Legacy-Retrievalindex wird nicht in die Decodermessung einbezogen.
- Jede qualitative Entscheidung ist über SHA-256 an genau einen Ergebnislauf gebunden. Semantische Passung und Satzbau werden getrennt bewertet. Eine generische Bestätigung erfüllt eine konkrete Inhaltsfrage nicht. Ein klar verständlicher kleiner Grammatikfehler kann semantisch dennoch passend sein.
- Die technische Oberflächenprüfung erkennt nur leere Texte, fehlende Schlusszeichen, offensichtliche Schleifen und ungültige Zeichen. Sie ist kein Grammatik- oder Bedeutungsnachweis.
- Rollen- und Negationsmarkierungen dokumentieren explizite Fehler. Ein themenfremder Satz kann ohne expliziten Rollen- oder Negationswiderspruch bleiben; diese Markierungen dürfen deshalb nicht als allgemeine Erfolgsquote gelesen werden.
- Build- und Antwortzeiten sind beobachtete Prozesszeiten bei teilweise parallelen Entwicklungsarbeiten, keine kontrollierten Hardwarebenchmarks. RSS und privater Commit sind getrennte Größen. Speicherwerte schließen Python-Objekte und temporäre Compilierung ein.

## Entwicklungsschleifen

| Lauf | Passend | Satzbau brauchbar | Oberfläche | Build | Peak RSS | Median Antwort |
|---|---:|---:|---:|---:|---:|---:|
| [621 ursprüngliche Einträge, dichter Decoder](baseline_development.json) | 11/20 | 20/20 | 20/20 | 1.83 s | 361 MiB | 0.06 s |
| [621 Einträge, kompakter Decoder](compact_existing_development.json) | 11/20 | 20/20 | 20/20 | 0.57 s | 74 MiB | 0.06 s |
| [+500, v1-Dateianfang, überwiegend OASST, Gewicht 1](public500_development.json) | 9/20 | 17/20 | 19/20 | 31.65 s | 710 MiB | 1.89 s |
| [+2000, v1-Dateianfang, Gewicht 1](public2000_development.json) | 8/20 | 15/20 | 18/20 | 60.90 s | 1302 MiB | 3.44 s |
| [+11.370, v2, Gewicht 1](publicall_v2_development.json) | 7/20 | 16/20 | 20/20 | 107.60 s | 3169 MiB | 8.64 s |
| [+11.379, v3, Quellengewichte 0,1 / 0,05](publicall_v3_weighted_development.json) | 12/20 | 17/20 | 20/20 | 88.18 s | 3159 MiB | 7.92 s |
| [+11.379, v3, Quellengewichte und lexikalische öffentliche Eingabemerkmale](publicall_v3_lexical_development.json) | 12/20 | 17/20 | 20/20 | 105.33 s | 3154 MiB | 7.11 s |

Die ersten 500 und 2000 Einträge sind feste Dateipräfixe, keine zufälligen Stichproben des Gesamtbestands. v1/v2/v3 besitzen getrennte SHA-256-Hashes und unveränderte Kopien in `corpora/`. Filteränderungen wurden anhand der Quellenqualität vorgenommen, nicht durch Einfügen von Evaluationsantworten.

Die Zwischenläufe zeigen eine deutliche Verschlechterung durch ungewichtete öffentliche Daten. Viele fachliche Eingaben enthielten Höflichkeitswörter oder andere Ausdrücke, die der bisherige Gesprächsparser fälschlich als Smalltalk-Absicht etikettierte. Die finale Compilerkorrektur beschränkt automatische öffentliche Eingabemerkmale auf lexikalische Merkmale; eigene Gesprächspaare und explizite Annotationen behalten ihre strukturierten Merkmale. Öffentliche Daten bleiben vollständig in den numerischen Feldern enthalten.

## Numerische Kontrollen

Pro Chatlauf wurden 60 Präfixzustände geprüft. Fourier-Interferenz und direkte punktweise Referenzrechnung werden auf derselben kompilierten Datenwelle verglichen; dies kontrolliert den Operator, nicht unabhängig die Bedeutung der importierten Texte. Die Prüfung umfasst Wahrscheinlichkeitsnormalisierung, zwei Zeitpunkte, Nullung beider Promptkanäle und sechs Zustände mit genullten komplexen Datenfeldern bei erhaltenen Referenzzählungen und Grammatikmasken.

| Lauf | Zustände | Max. FFT/direct-Abweichung | Max. Zeit-Abweichung | Promptnull verändert Verteilung | Daten-null verweigert |
|---|---:|---:|---:|---:|---:|
| baseline | 60 | 0 | 0 | 57/60 | 6/6 |
| compact_existing | 60 | 0 | 0 | 57/60 | 6/6 |
| public500 | 60 | 0 | 0 | 58/60 | 6/6 |
| public2000 | 60 | 0 | 0 | 57/60 | 6/6 |
| publicall_v2 | 60 | 0 | 0 | 59/60 | 6/6 |
| publicall_v3_weighted | 60 | 0 | 0 | 59/60 | 6/6 |
| publicall_v3_lexical | 60 | 0 | 0 | 59/60 | 6/6 |

Nullabweichung gilt nach der bereits dokumentierten Rundung der Decoderwahrscheinlichkeiten auf 12 Dezimalstellen. Ein unverändertes Ergebnis bei einzelnen Promptnull-Zuständen ist bei eindeutigem Grammatikübergang möglich. Die Kontrollen beweisen numerische Konsistenz und kausale Feldabhängigkeit; sie beweisen kein Sprachverständnis.

## Zurückgehaltene Chatprüfung

- [baseline_closed_holdout](baseline_closed_holdout.json): 13/20 passend; Oberfläche 19/20. FFT/direct max. 0, Zeitabweichung max. 8.17e-13, keine Argmaxänderungen: True.
- [final_holdout](final_holdout.json): 12/20 passend; Oberfläche 19/20. Zusätzlich 1/20 nur teilweise passend, konservativ nicht als volle Treffer gezählt. FFT/direct max. 0, Zeitabweichung max. 0, keine Argmaxänderungen: True.

Der Abschluss zeigt keine Verbesserung der allgemeinen Chatqualität: 12/20 vollständige Treffer gegenüber 13/20 in der Ausgangsversion, zusätzlich eine teilweise passende Antwort. Beide Versionen bestehen jeweils nur drei der zehn Zweiergespräche vollständig. Neue Fehler betreffen unter anderem Familientreffen und Abschied, die in fremde Sachthemen abdriften. Die Kaffee-Negation bleibt falsch. Verbessert wurden in dieser Stichprobe der Wunsch nach Ruhe und die Frage nach eigenen Gefühlen. Nach dieser Prüfung wurde kein semantisches Tuning vorgenommen.

## Separater Faktentest und bekannte Fragen

Der offizielle GermanQuAD-Testsplit enthält 2.204 Fragen. Kein Test-Kontextabschnitt ist exakt im Trainingssplit enthalten; 2.130 Testfragen stammen aber aus bereits im Trainingssplit vertretenen Wikipedia-Artikeltiteln. Die 30 Testfragen wurden mit Seed 731 vor Kandidatausgaben fixiert. Ihre Referenzkontexte werden dem Generator nicht mitgegeben: Das ist eine geschlossene Corpusprüfung und nicht der übliche extraktive GermanQuAD-Leseverständnisbenchmark.

Im finalen Import kommt bei 8/30 Fragen ein Referenzantwortstring vor, davon bei 4 mit gleichem Artikeltitel. Ein Stringtreffer beweist keine ausreichende Belegabdeckung. Mehrere offizielle Datensätze enthalten widersprüchliche Antwortannotationen; das qualitative Urteil berücksichtigt den Originalkontext. Historische Angaben werden gegen die Quelle bewertet und nicht als heute verifizierte Fakten ausgegeben.

Zusätzlich wurden mit Seed 9021 genau 20 bekannte Originalfragen aus dem finalen GermanQuAD-Import fixiert. Diese Abrufdiagnose ist ausdrücklich **in-sample** und darf nicht als Generalisierungsleistung ausgegeben werden.

| Lauf | Prüftyp | Qualitativ korrekt | Enthält Referenzstring |
|---|---|---:|---:|
| [baseline_facts](baseline_facts.json) | 30 offizielle Testfragen | 0/30 | 0/30 |
| [final_facts](final_facts.json) | 30 offizielle Testfragen | 0/30 | 0/30 |
| [baseline_replay](baseline_replay.json) | 20 bekannte Importfragen | 0/20 | 0/20 |
| [final_replay](final_replay.json) | 20 bekannte Importfragen | 8/20 | 9/20 |

Die Reviewdateien enthalten die konkreten Fehler und Quellenbezüge. Neue vollständige Zeichenketten oder neue Sätze werden als Neuheitsdiagnose gespeichert, aber nicht mit sinnvoller oder faktisch korrekter Generierung gleichgesetzt.

Der größere Bestand ermöglicht in dieser Probe acht korrekte Antworten auf bereits importierte Originalfragen. Keine der dreißig separaten offiziellen Fragen wird korrekt beantwortet; deren Belegabdeckung im Import ist allerdings größtenteils unzureichend. Ein allgemeiner Wissens- oder Verständnisgewinn ist damit nicht nachgewiesen.

## Nachträgliche numerische Beschleunigung

Die Tokenmoden wurden für eine günstig faktorisierte FFT-Trägerlänge mit Nullen aufgefüllt. Die Quellenfelder, Tokenfrequenzen und semantische Konfiguration blieben erhalten. Die Ausgaberegression prüft die bereits gesehenen 40 Chat-, 30 Fakten- und 20 bekannten Importfragen: 90/90 sichtbare Antworten bytegleich, 90/90 Tokenfolgen übereinstimmend. Dies ist kein neuer Holdout und kein Qualitätszuwachs.

Thirty original factual rows did not persist raw token streams. They are checked against exact frozen UTF-8 answer text and its reconstructed tokens. Sixty directly stored streams and the full model alphabet independently check renderer/tokenizer round trips. This is not a newly unseen quality evaluation.

Median über alle 90 Antworten: vorher 9.70 s, nach Beschleunigung 1.75 s. Die älteren Messungen liefen teilweise parallel und sind deshalb kein kontrollierter Beschleunigungsfaktor. Der einmalige Modellaufbau der Regression dauerte 82.29 s. Details: [Ausgaberegression](fast_carrier_output_regression.json).
