# Prüfung einer autoregressiven Wellenausgabe

Der Testdatensatz enthält 60 neue Gesprächseingaben in 20 voneinander getrennten Gesprächen. Zehn Gespräche mit 30 Turns stehen für Entwicklung zur Verfügung. Zehn weitere Gespräche mit 30 Turns bleiben bis zum Einfrieren des Algorithmus, der Parameter und des Sprachkorpus zurückgehalten. Vorhandene Tests und Speicherprompts werden auf exakte und normalisierte Überschneidungen geprüft. Die Evaluierungsdatei darf nicht in den Sprachkorpus gelangen.

## Getrennte Fragen

1. **Mechanismus:** Werden Tokens schrittweise aus der numerischen Resultatverteilung gewählt? Gibt es einen dokumentierten kausalen Effekt von Prompt, Sprachspeicher und Phase? Ein nachträgliches Kodieren eines bereits ausgewählten Satzes erfüllt dieses Kriterium nicht.
2. **Numerik:** Stimmen FFT- und direkte reelle Berechnung innerhalb einer dokumentierten Toleranz überein? Sind Gewichte endlich, normalisiert und zeitlich konsistent? Verändert der Decoder nur jene Größen, die das angegebene mathematische Modell verändern soll?
3. **Sprachform:** Ist der Text lesbar, ohne kaputte Wortgrenzen, Endlosschleifen oder unvollständige Satzabbrüche? Automatische Formtests ersetzen keine Grammatikprüfung.
4. **Bedeutung:** Passt der Satz zur Eingabe und zum bisherigen Gespräch? Bleiben Sprecherrollen und Verneinungen erhalten? Werden eigene Erlebnisse oder nicht vorhandene Gesprächsinformationen erfunden?
5. **Neuheit:** Ist der vollständige Text oder wenigstens ein Satz neu gegenüber dem Sprachkorpus? Das ist ein eigener Messwert und kein Beleg für sinnvolle Sprache.

## Ablationen

Auf den eingefrorenen Entwicklungseingaben werden dieselben Anfangszustände und deterministischen Seeds verwendet. Zusätzlich zu Textunterschieden werden die Total-Variation-Distanz der nächsten Tokenverteilungen, geänderte Argmax-Tokens und endliche/normalisierte Werte ausgewiesen.

- **Promptnull:** Die Promptwelle wird zu Null gesetzt; Sprachspeicher und bisheriger Decoderzustand bleiben identisch. Ein Textunterschied allein beweist keine sinnvolle Promptverarbeitung.
- **Memorynull:** Die Datenwelle beziehungsweise der Interaktionsoperator wird zu Null gesetzt. Falls damit kein Token definierbar ist, ist ein expliziter leerer/abgebrochener Befund korrekt; ein verdeckter Satzgenerator wäre ein Fehler.
- **Phasenverwürfelung:** Nur die Phasen der angegebenen Wellenkomponente werden mit festem Seed permutiert oder randomisiert. Amplituden bleiben unverändert. Das prüft genau diese Repräsentation; es beweist keine allgemeine physikalische Sprachtheorie.
- **FFT gegen direkte Rechnung:** Dieselbe nächste Tokenverteilung wird ohne FFT aus dem expliziten reellen Operator berechnet. Maximaler absoluter Fehler und identische Tokenwahl werden berichtet. Gleichheit zeigt eine numerisch äquivalente Darstellung, keinen zusätzlichen Wissensgewinn durch Fouriertransformation.
- **Zeit:** Falls gemeinsame freie Phasenentwicklung durch Demodulation wieder aufgehoben wird, werden identische Texte zu mehreren Zeitpunkten erwartet und die Aufhebung wird explizit dokumentiert.

## Qualitätsbewertung und Einfrieren

Die JSON-Rubriken beschreiben akzeptable Bedeutung und enthalten einfache positive und negative Textmuster. Diese liefern nur automatische Näherungswerte. Jeder erzeugte Text wird zusätzlich anhand von vier getrennten Ja/Nein-Urteilen geprüft: vollständige Grammatik, Bezug, innere Kohärenz und keine widersprechende Verneinung/Sprecherrolle. Der im Review bestätigte Gesamterfolg erfordert alle vier. Die Reviewer-Identität wird angegeben: Eine qualitative Einzelprüfung durch einen gesonderten Codex-Agenten ist keine unabhängige Studie mit menschlichen Nutzern. Eine nachvollziehbare Rückfrage ist akzeptabel, sofern die Rubrik das erlaubt; eine pauschale Rückfrage beantwortet keine klare Aufforderung oder Namensabfrage.

Vor dem ersten Holdout-Lauf werden SHA-256-Werte für Datensatz, Implementierung, Konfiguration und Sprachkorpus abgelegt. Nach Kenntnis des Holdouts wird das Modell für diesen Abschlussbefund nicht weiter abgestimmt. Fehlfälle werden vollständig berichtet. Bei weiteren Anpassungen gelten diese Fälle nur noch als Entwicklung und es wäre ein neuer Holdout nötig.

## Vorab-Befund zum vorhandenen Korpus

Die 120 Antworttexte enthalten mit einer einfachen Wort-/Interpunktionssegmentierung 1.759 Tokens und 565 verschiedene Tokens. Von 1.399 unterschiedlichen Trigrammen treten 1.324 nur einmal auf. Nur 142 von 1.124 Kontexten aus zwei Tokens erlauben mehrere Folgetokens. Ein daraus erzeugter Trigrammdecoder ist stark auf vorhandene Pfade begrenzt; Rückfall auf kürzere Kontexte erhöht die Kombinationsfreiheit, aber auch das Risiko grammatischer und semantischer Fehler.

Ein zusätzlicher ungepaarter Sprachkorpus sollte unabhängig von Testeingaben entstehen, kurze Alltagsäußerungen mit produktiven Satzmustern enthalten und Quelle sowie Erstellungsweise offenlegen. Mehr Daten und Übergangszählungen sind eine Form datenabhängiger Schätzung, auch wenn keine Gradienten und keine vortrainierten Modelle verwendet werden.
