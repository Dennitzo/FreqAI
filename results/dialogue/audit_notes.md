# Einordnung der eingefrorenen Auswertung

Die Rohwerte in `final.json` bleiben nach dem ersten Holdout-Lauf unverändert:
Development 51/51, Holdout 24/50, Stress 18/33. Implementierung,
Antwortgrammatik, Eingaben und automatische Bewertungslogik wurden vor diesem
Lauf mit SHA-256 festgehalten. Anschließend wurden ausschließlich die Resultate
gelesen und diese Einordnung geschrieben.

Die ausführbare Rubrik bewertet semantische Merkmale anhand von Wortmustern,
expliziten Interpretationsakten und Sitzungszustand. Sie ist kein unabhängiges
Sprachmodell und kein vollständiger menschlicher Bedeutungsrichter. Deshalb
dürfen die Brüche nicht als universelle Gesprächsgenauigkeit gelesen werden.

Bei fünf Stressfällen beanstandet die Rubrik eine fehlende Rückfrage, obwohl
die Ausgabe mit „Kannst du mir mehr Kontext geben?“ tatsächlich eine Rückfrage
enthält. Die generische Behauptung fehlender verlässlicher Information bleibt
bei sprachlich unklaren Fragen weniger passend als eine gezielte Rückfrage.
Die automatische Rubrik wurde nach dem Holdout trotzdem nicht erweitert.

Vier Holdoutfälle mit negiertem Befinden erhalten eine neutrale Bestätigung
(„Danke für die Einordnung“). Die interne Interpretation behandelt diese Fälle
als neutral, während die Ausgaberubrik eine explizitere Erwähnung des Befindens
verlangt. Das ist ein strenges Ausdruckskriterium und darf nicht mit einer
nachgewiesenen falschen positiven oder negativen Gefühlszuordnung verwechselt
werden. Auch diese Rohwertungen bleiben unverändert.

Deutlich reale Generalisierungslücken sind andere Formulierungen von
Gegenfragen, Dank mit zusätzlichen Satzteilen, Namenswechsel per „Nenn mich“,
nachgestellte Negation, Korrekturen einer vorigen Aussage sowie das Auflösen
elliptischer Rollenwechsel. Aussagen über andere Menschen werden häufig nur
mit einer allgemeinen Rückfrage beantwortet; ein Hund wird dabei sogar als
„Person“ bezeichnet. Diese Fälle stehen mit vollständiger Ein- und Ausgabe in
`report.md` und `final.json`.

Numerisch sind alle 536 Vergleiche der zeitlichen Invarianz und alle 134
unabhängigen Quellabschnitt-/Prüfsummenrekonstruktionen erfolgreich. Die
arithmetische Korrektheit der Fourierabbildung überträgt sich damit ausdrücklich
nicht automatisch auf die sprachliche Richtigkeit der Antworten.
