# Dialogberechnung, zweite Runde

Die erste Runde liegt vollständig und unverändert unter `../round1/`; ihr damaliger Holdout ist jetzt ausschließlich Entwicklungsdaten. Zusätzlich wurden vor der zweiten Anpassung 80 neue Eingaben in getrennten Szenarien erstellt: 40 zur Entwicklung und 40 als neuer Holdout. Insgesamt enthält Runde 2 damit 214 Turns.

| Teil | Archiviertes V1-Modell | V2-Modell |
|---|---:|---:|
| development | 119/174 | 174/174 |
| holdout | 18/40 | 32/40 |

124 Antwortturns enthalten 93 verschiedene vollständige Antworten, die nicht wörtlich im unveränderten 120er-Korpus stehen. Wiederholung von Eingabeabschnitten und Kombination vorgegebener Grammatik zählen dabei ebenfalls als neue vollständige Texte; das ist kein Nachweis freier Sprachgenerierung.

Zeitinvarianz: 856/856; unabhängige Quellen-/Span-/Prüfsummenprüfung: 214/214. Zusätzliche UTF-8-Rückrechnung: 600/600.

Die ausführbare Rubrik prüft Bedeutungsmerkmale und Rollen, keinen exakten Antworttext. Sie wurde aufgrund der ersten Runde vor dem neuen Holdout korrigiert: eine Bitte um mehr Kontext gilt als Rückfrage, eine korrekt interpretierte doppelte/negative Verneinung darf neutral bestätigt werden, und eindeutige Pronomen dürfen Namen ersetzen. Das bleibt ein begrenztes automatisches Prüfraster und keine universelle Dialoggenauigkeit.

## Verbleibende Fehler

- `v2-holdout-names-context-2`: Danke, mir geht es prima. Weißt du noch, wie ich heiße? → Gern. Das klingt schön. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:recall_name)
- `v2-holdout-second-session-1`: Habe ich dir schon gesagt, wie ich heiße? → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:unknown_name)
- `v2-holdout-second-session-4`: Wie nennst du mich? → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:recall_name, unnecessary_abstention)
- `v2-holdout-short-and-punctuation-3`: Ich fühle mich wohl – danke fürs Nachfragen. → Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? (answer_missing:thanks)
- `v2-holdout-short-and-punctuation-4`: Gut geht's mir! Und wie ist es bei dir denn? → Das klingt schön. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:assistant_status)
- `v2-holdout-conditional-uncertain-4`: Ich wünschte, mir würde es besser gehen. → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:negative, interpretation_missing:user_negative, response_act_missing:user_negative, unnecessary_abstention)
- `v2-holdout-assistant-roles-3`: Ich fühle mich schlecht. Geht's dir genauso? → Das klingt nach einem schweren Tag. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (answer_missing:assistant_status)
- `v2-holdout-assistant-roles-4`: Danke für deine Antwort! Ich bin froh und sage jetzt tschüss. → Gern. Das ist ein schöner Moment. (answer_missing:farewell)
