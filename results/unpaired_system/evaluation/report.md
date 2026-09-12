# Ungepaarter Informationsbestand: abschließende unabhängige Evaluation

Die vollständige Umstellung auf deklarative Informationen ist im geprüften Kandidaten technisch umgesetzt. Allgemeine Gesprächsfähigkeit ist weiterhin nicht zuverlässig erreicht: In 20 neuen, zurückgehaltenen Gesprächsrunden liefert der Kandidat **3 passende Antworten**, die historische Baseline **7**. Der Entwicklungsdatensatz verbessert sich, die unbekannten Formulierungen dagegen nicht. Nach dem endgültigen Freeze erfolgte keine weitere Qualitätsanpassung.

## Daten und Verfahren

Die vor der Migration eingefrorene historische Baseline umfasst 32.000 Dokumente: 11.500 Frage-Antwort-Paare, 500 frühere Antwortprior-Texte und 20.000 Wikipedia-Einleitungen. Der Kandidat ersetzt diesen Bestand vollständig durch **20.093 Informationsdokumente**: dieselben 20.000 Wikipedia-Einleitungen, 45 eigenständige Faktenabsätze und 48 deklarative lexikalische Klassifikationen. Er übernimmt keine belegten Promptfelder, früheren Paardokument-IDs oder Antwortpriors. Auch vollständige Textgleichheit mit früheren Paarantworten wurde nicht gefunden.

Vor Kandidatantworten wurden 20 neue Entwicklungsrunden und 20 weitere zurückgehaltene Gesprächsrunden mit Rubriken festgelegt, jeweils zehn Sitzungen mit zwei Eingaben. Die vollständigen normalisierten Prompts überschneiden sich nicht mit dem Kandidatenkorpus. Quellenautor und Decoderentwickler erhielten keine neuen Holdoutfragen. Erst nach dem endgültigen Code-/Datenfreeze wurden Kandidat-Holdouts berechnet und die zuvor ungeöffnet vorberechnete Baseline geöffnet. Fragen und Rubriken wurden nie importiert.

Die Bewertung stammt von einem unabhängigen AI-Agenten und ist keine menschliche Nutzerstudie. Sie prüft tatsächliche Gesprächseignung, Sprecherrolle, Verlauf, Negation und Satzbau. Teiltreffer zählen nicht als volle Erfolge. Eine kurze passende Zustimmung kann genügen; bloßes Wiederholen von Gefühlen oder Wünschen zählt nur teilweise. Die kleine Stichprobe erlaubt keine breite Aussage über beliebige Alltagsgespräche.

## Gesprächsergebnisse

| Prüfung | Voll passend | Zusätzlich teilweise | Leer | Ganze Zweirundensitzungen passend |
|---|---:|---:|---:|---:|
| Historische Baseline, Development | 4/20 | 3/20 | 0/20 | 0/10 |
| Trial v1, Development | 2/20 | 2/20 | 16/20 | 0/10 |
| Trial v2, Development | 10/20 | 9/20 | 1/20 | 4/10 |
| Finaler bekannter Development-Replay | 10/20 | 9/20 | 1/20 | 4/10 |
| Historische Baseline, neuer Holdout | **7/20** | 3/20 | 0/20 | 2/10 |
| Finaler Kandidat, neuer Holdout | **3/20** | 8/20 | 9/20 | 0/10 |

V1 scheiterte überwiegend an strikter Informationsbindung und pauschaler Negationsablehnung. V2 ergänzt allgemeine Rollen-, Satzstellungs-, Wunsch- und Sprachaktregeln sowie das separat verfasste Lexikon. Die Developmentverbesserung ist daher eine gemeinsame Änderung von Regeln und Daten, keine ausschließlich numerische Verbesserung. Es wurden keine fertigen Beispielantworten importiert.

Im neuen Holdout funktionieren eine Begrüßung, die Zustimmung zu einer Privatsphäregrenze und eine Dankesquittung. Viele Antworten sind nur Umformungen, etwa „Du bist noch nicht richtig wach.“. Abweichende Formulierungen zu einem freien Gespräch, Lesen, Besuch, Wiedersehen oder eigener Erfahrung bleiben teilweise leer. Inhaltliche Gesprächsfortsetzung fehlt oft. Zehn von elf nicht leeren Holdoutantworten sind grammatisch brauchbar; Satzbau allein belegt deshalb keine Gesprächsqualität.

Der abschließende Rendererfix verändert drei bekannte Developmenttexte, alle 20 Tokenfolgen bleiben gleich. Er korrigiert „platt“ und „Schreiben“, führt aber ein falsches internes „Heute“ ein. Weitere Schreibungs- und Satzstellungsprobleme bleiben dokumentiert. Es erfolgte keine Nachbesserung anhand neuer Holdouts.

## Informationen und gerichtete Faktenbindung

Alle **40 bereits bekannten Informationsfragen** behalten ihre bisherigen Antworten text- und tokengenau: **4 richtig, 36 leer**. Erhalten bleiben Frequenzdefinition, Hertz, Kehrwert der Periodendauer und Volt. Frühere Holdoutfragen sind hier ausdrücklich bereits bekannte Regression. Der bisherige schmale Funktionsumfang wird bewahrt; breites Faktenwissen wird nicht nachgewiesen.

Die bekannte Zufallsfaktendiagnostik verwendet zwölf erfundene Signalnamen und Zahlen ausschließlich in 24 deklarativen Sätzen. Fragen und Bewertungsdaten gelangen nicht in den Konstruktor. Sie verbessert sich von ausschließlich leeren Ausgaben in v1 auf:

| Bekannte Diagnose | Finales Ergebnis |
|---|---:|
| Name → deklarierter Frequenzwert | 12/12 richtig |
| Frequenzwert → deklarierter Name | 12/12 richtig |
| Zyklisch vertauschte Frequenzwerte im Datenbestand | 12/12 folgen den geänderten Daten |
| Korrektur vier falscher Wertbehauptungen | 0/4 beantwortet |
| Vier unbekannte Namen | 4/4 sicher leer |

Alle 36 richtigen Ausgaben rekonstruieren deklarierte Quellsätze. Das belegt Attribut- und Richtungsbindung ohne QA-Eingabe, keine freie Schlussfolgerung oder neue unbekannte Testleistung. Alle 32 ursprünglichen Ausgaben sind bei gleichem Neuaufbau reproduzierbar. Datenpermutation verändert Koeffizienten und Rollenhash und führt zu den neuen Werten.

## Neue Texte und Numerik

Zwei der drei voll passenden neuen Holdoutantworten sind normalisierte Quellteiltexte. Die neue Begrüßungsfolge entsteht aus expliziter Grammatik und lexikalischen Rollen. Andere neu zusammengesetzte Ausgaben sind überwiegend Nutzerparaphrasen. Neue Worttypen stammen aus dem Prompt oder expliziter Flexion, etwa „hörst“ und „möchtest“. Neue Zeichenfolgen bedeuten hier kein neues Faktenwissen.

Es erfolgen **0 Optimierungsschritte** und keine kalibrierten Mischgewinne. Koeffizienten werden unter anderem als `sqrt(count/sum(count))` berechnet. Das ist deterministische statistische Anpassung an den Bestand. Rollenparser, deutsche Grammatik und Sprachaktregeln bleiben verfasstes Sprachwissen; das System ist keine gewichtsfreie oder universelle Intelligenz.

Der numerische Audit des finalen Core prüft **255 Präfix-/Zeit-Zustände**. Fourieroperator und direkte Rechnung stimmen nach der festgelegten Rundung überein. Relative Promptphase und Datenrollenphase verändern Verteilungen jeweils mit Totalvariationsabstand 0,8 und ändern die gewählten Symbole. Nullprompt und Null-Datenrollen verhindern die entsprechende Ausgabe. Gleiche Daten erzeugen gleiche Koeffizienten; Sitzungsrollen verändern diese nicht und bleiben zwischen Sitzungen getrennt. Das sind Eigenschaften des konstruierten Algorithmus, kein Nachweis einer natürlichen physikalischen Zuordnung von Wörtern zu Schwingungen.

## Laufzeit und Grenzen

Der finale isolierte Aufbau benötigt **96,4 Sekunden**, mit **2,77 GiB Peak-RSS**. Das Modell umfasst 117.055 Vokabularsymbole, 49.143 verarbeitete Sätze und 11.219 extrahierte deklarative Fakten. Die reine numerische Nutzlast beträgt etwa 104 MB; Pythonobjekte und temporärer Speicher kommen hinzu.

Der Holdoutmedian beträgt **83 ms pro Runde**, bei neun Leerantworten und typischerweise acht ausgegebenen Symbolen unter den nicht leeren Antworten. Diese Einzelmessungen sind keine Zusage für längere brauchbare Antworten und enthalten nicht die zusätzliche Dokumentwelle, SQLite oder Browserkosten. Andere Teamprozesse liefen teilweise parallel.

Das produktionsnahe Gesprächsbudget von 40 zählt Spektralsymbole einschließlich Satzzeichen, UTF8-Bytes unbekannter Wörter und Endsymbol. Es ist kein einheitliches Wortbudget. Die 40 finalen Gesprächsausgaben enthalten keine Unicode-Ersatzzeichen; lange unbekannte Wörter können dennoch Budget verbrauchen und Antworten kürzen.

Der finale Decoderhash lautet `f99d23332e3ae7dee869b93bd4fd8aa2226f3fbb1005a2a0a9640b36b4830bba`. Nach dem Freeze wurde separat die rohe CLI-/HTTP-Importvalidierung verschärft, damit alternative Paarfelder nicht still verworfen werden. Diese Eingabevalidierung ändert weder Korpus noch Decoder; das Deployment besitzt deshalb ein eigenes Manifest. Historische Experimente wurden nicht überschrieben.

## Artefakte

- [Strukturierte Zusammenfassung](summary.json), [Versuchsprotokoll](protocol.json), [finaler Freeze](final_freeze.json)
- [Baseline-Holdout](baseline_final_holdout.json), [Baseline-Bewertung](baseline_final_holdout_review.json)
- [Finale Ausgaben: 20 Holdout, 20 bekannter Development-Replay, 40 bekannte Informationsfälle](final_holdout.json), [Holdoutbewertung](final_holdout_review.json), [Developmentbewertung](final_development_replay_review.json), [Informationsvergleich](final_knowledge_review.json)
- [Trial v1](trial_v1_development.json), [v1-Bewertung](trial_v1_development_review.json), [Trial v2](trial_v2_development.json), [v2-Bewertung](trial_v2_development_review.json)
- [Finale bekannte Zufallsfaktenregression](final_randomfacts_regression.json), [semantische Bewertung](final_randomfacts_review.json), [numerischer Audit](../numerical_audit.json)
- [Promptüberschneidungsprüfung](source_prompt_overlap_v2.json), [Evaluator](../../../experiments/evaluate_unpaired_system.py), [Faktendiagnostik](../../../experiments/evaluate_unpaired_randomfacts.py), [Zusammenfassungsprogramm](../../../experiments/evaluate_unpaired_summary.py)
