# Unabhängige Evaluation des Informationskorpus

Der neue Decoder beantwortet 4 von 20 Entwicklungsfragen korrekt. Im zuvor versiegelten Holdout bleibt er bei allen 20 Fragen leer; die bisherige Baseline beantwortete dort einen DNA-Fall korrekt. Eine allgemeine Übertragung auf neue Informationsfragen ist damit nicht nachgewiesen.

Die Bewertung trennt belegte Fakten, passende Antworten, sichere Enthaltung, Satzbau und Textneuheit. Es handelt sich um eine unabhängige KI-Agentenbewertung, keine menschliche Nutzerstudie.

| Prüfung | Ausgangsmodell | Endstand |
|---|---:|---:|
| Informationsfragen Development: sachlich korrekt | 0/20 | 4/20 |
| Informationsfragen Development: leere Antworten | 0 | 16 |
| Informationsfragen Development: grammatisch unter nichtleeren Antworten | 9/20 | 4/4 |
| Versiegelte Informationsfragen: sachlich korrekt | 1/20 | 0/20 |
| Versiegelte Informationsfragen: leere Antworten | 1 | 20 |
| Versiegelte Informationsfragen: grammatisch unter nichtleeren Antworten | 12/19 | nicht anwendbar |

Bekannte Alltagsgespräche: **40/40 Antworten** und **40/40 Tokenfolgen** stimmen exakt mit der aktuellen Baseline überein.

Dieser 40-Fälle-Lauf wurde nach der abschließenden Routingänderung auf Basis unveränderter Gesprächscompiler-Hashes und 0 von 40 veränderten Gesprächsrouten wiederverwendet. Er ist ein bekannter Regressionstest.

Das Ausgangsmodell umfasst 12.000 Texte (11.500 QA-Paare plus 500 synthetische Sprachprior-Texte). Der neue Import umfasst 20.000 deklarative Wikipedia-Texte mit leeren Prompts. Die Fragen und Bewertungskriterien wurden nicht importiert.

Mindestens 7 der 18 Wissensfragen im Development sind in finaler Prosa direkt belegt; im Holdout bestätigte ein zusätzlicher Reviewer mindestens 2 vollständige Belege. Fehlende Belege und fehlende Fragebindung sind verschiedene Grenzen. Die jeweils zwei persönlichen Messfragen enthalten keine tatsächliche Messung; die leere Enthaltung verhindert dort erfundene Werte, liefert aber keinen erklärenden Antwortsatz.

Alle vier korrekten Entwicklungsantworten sind normalisiert exakte Quellpassagen. Neue faktisch korrekte Satzkomposition wird damit nicht belegt. Die explizit vom Nutzer genannte Frequenzdefinition endet im tatsächlichen 40-Token-Aufruf vollständig nach 36 Tokens und entspricht exakt der 64-Token-Referenz.

## Zufallsfakten ohne QA-Paare

Zwölf unbekannte Namen und Zahlen wurden ausschließlich als 24 deklarative Sätze eingespeist. Die vorab festgelegten Fragen erreichten 0/12 Vorwärts-, 0/12 Rückwärts- und 0/4 Negationsantworten. Vier unbekannte Namen wurden sicher leer abgewiesen. Nach Zahlenpermutation blieben auch 12 weitere Antworten leer. Alle 44 Ausgaben waren leer; dies ist kein Fähigkeitsnachweis für allgemeine Faktenbindung.

Identische Eingaben erzeugen identische Koeffizienten und Ausgaben. Durch die Zahlenpermutation ändern sich die Koeffizienten bei gleichem Vokabular. Die Datenkodierung reagiert also auf die Fakten, während die Fragegrammatik diese unabhängige Formulierung nicht erfolgreich bindet.

## Numerik und Reproduzierbarkeit

Im unabhängigen Toy-Audit stimmen 48 endliche Operator-/Direktverteilungen nach der implementierten Rundung auf 12 Nachkommastellen exakt überein; sechs destruktive Nullzustände werden korrekt abgewiesen. Zeitinvarianz, Prompt-/Datenfeldnullung und relative Phaseninterventionen sind getrennt dokumentiert. Eine neu kombinierte Toy-Folge ('Blaue Katzen schlafen.') belegt mögliche lokale Wortkombination, nicht allgemeine faktische Richtigkeit.

Drei vollständige Aufbauten derselben 20.000 Dokumente ergeben denselben Koeffizientenhash und denselben Renderhash. Es gab 0 Optimierungsschritte. sqrt(count/sum(count)) ist eine deterministische statistische Datenanpassung. Die bestehende Alltagsschiene behält ihre zuvor festgelegten Verstärkungen.

Der abschließende Informationsaufbau benötigte 129.2 s bei 2.79 GiB Prozessspitze. Die rund 77 ms Median im Holdout messen ausschließlich Enthaltungen. Der letzte Developmentlauf benötigte etwa 250 s für beide Compiler; parallele lokale Arbeit beeinflusste die Zeitmessungen.

Die vollständigen Kennzahlen, Einzelfallbegründungen, Quellbelege, Code-/Datenhashes und das Zufallsfaktenexperiment stehen in den JSON-Artefakten dieses Verzeichnisses.

Die wiederholten Entwicklungsprüfungen nutzten dieselben 20 Fragen. Ihre Ergebnisse sind kein zusätzlicher Holdout:

| Entwicklungsstufe | Korrekte Antworten | Leere Antworten | Beobachtung |
|---|---:|---:|---|
| Ausgangsmodell | 0/20 | 0 | Unpassende alte Textgenerierungen |
| Informationsmodell v1, Prosa v1 | 3/20 | 10 | Zusätzlich zwei Teilantworten; generische Wörter als falsche Sachthemen |
| Informationsmodell v2, Prosa v3 | 2/20 | 18 | Strengere Bindung verhindert Themenfehler, blockiert auch Frageprädikate |
| Runtime v2 mit Core v3 | 4/20 | 4 | Zwölf Frageformen landen noch in der alten Gesprächsschiene |
| Abschließende Runtime v3 | 4/20 | 16 | Alle Informationsfragen korrekt geroutet, keine Themenverwechslung |

Ein erster Runtimeversuch wurde nach einem nachgewiesenen Quellenpräfixfehler abgebrochen: Prosa wäre zusätzlich in den alten Gesprächscompiler gelangt. Der korrigierte Test bestätigt anschließend exakt 12.000 unveränderte Gesprächsdokumente und 20.000 getrennte Informationstexte. Der Fehlversuch wird nicht als vollständiges Qualitätsergebnis gezählt.

Wesentliche Artefakte: [Kennzahlen](summary.json), [unabhängiges Holdout-Review](independent_holdout_review.json), [Zufallsfakten](final_randomfacts_review.json), [Integrität und vollständige Rebuild-Hashes](final_integrity.json), [numerischer Audit](../../information/numerical_audit.json).

## Grenzen

Kleine, manuell formulierte Testsuite und KI-Agentenbewertungen; keine menschliche Nutzerstudie. Der Frequenzfall ist bekanntes Development. Wikipedia-Abdeckung von 2023, fehlende Formeln im Herausgeberexport und begrenzte deklarierte deutsche Grammatik bleiben Einschränkungen. Nach dem Holdout erfolgte keine weitere Qualitätsanpassung.
