# Begrenzte Generalisierung im Dialog

134 handgeschriebene deutsche Eingaben in 26 getrennten Gesprächsszenarien; der Speicher behält die ursprünglichen 120 Gesprächspaare. Die Entwickler sahen vor dem Freeze nur Development-Ergebnisse. Holdout und Stress wurden erst danach ausgewertet.

Die Prüfung bewertet Antwortbedeutungsmerkmale mit regulären Ausdrücken, mehrere gleichzeitig verlangte Aussagen, Rollen und isolierte Namenszustände. Sie vergleicht keine vollständigen Antwortvorlagen. Dennoch ist das ein begrenztes automatisches Prüfraster; menschliche Bedeutungsprüfung bleibt nötig.

| Teil | Vorher | Nachher | Neue vollständige Antworten |
|---|---:|---:|---:|
| development | 16/51 | 51/51 | 15 |
| holdout | 7/50 | 24/50 | 33 |
| stress | 14/33 | 18/33 | 7 |

Zeitinvarianz: 536/536 Antwort-/Zustandsvergleiche bei vier zusätzlichen Zeitpunkten einschließlich 10^12 s.

Numerische Zusatzprüfung: 600/600 UTF-8-Rückrechnungen, 1764 Vergleiche von expliziten Symbolüberlappungen mit Fourier-Interferenz sowie Parseval-, inverse FFT- und gemeinsame Phasenprüfung. Maximalfehler jeweils kleiner als 10^-12.

Quellenprüfung aller Ausgaben: 134/134 Antworten lassen sich aus den deklarierten UTF-8-Quellabschnitten rekonstruieren; Quellen- und Ausgabeprüfsummen stimmen überein.

Neu heißt hier: Die vollständige Antwort steht nicht wörtlich im 120er-Korpus. Sie kann trotzdem aus vorhandenen Satzteilen und expliziten Regeln bestehen. Das weist keine offene Sprachgenerierung und keine automatisch aus Physik gewonnene Textbedeutung nach.

## Offen gebliebene Fälle

- `holdout-filler-composition-1`: Ach hallo, also bei mir läuft es prima, und wie geht es dir denn heute? → Ich bin bereit für unser Gespräch. Wie geht es dir heute? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:greeting, answer_missing:positive, interpretation_missing:user_positive, response_act_missing:user_positive)
- `holdout-filler-composition-2`: Ehrlich gesagt bin ich sehr zufrieden; wie sieht es bei dir aus? → Das ist ein schöner Moment. Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:assistant_status)
- `holdout-filler-composition-4`: Übrigens: Ich bin total müde, geht es dir auch so? → Dann wäre vielleicht ein ruhiger Moment angenehm. Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:assistant_status)
- `holdout-filler-composition-6`: Danke der Nachfrage, ich bin fröhlich. Wie geht's dir selbst? → Das klingt schön. Ich bin bereit für unser Gespräch. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks)
- `holdout-negation-2`: Heute fühle ich mich nicht schlecht. → Danke für die Einordnung. Magst du mehr darüber erzählen? (Prüfung: answer_missing:positive)
- `holdout-negation-4`: Ich bin nicht traurig. → Danke für die Einordnung. Magst du mehr darüber erzählen? (Prüfung: answer_missing:positive)
- `holdout-negation-6`: Nicht schlecht, und dir? → Danke für die Einordnung. Magst du mehr darüber erzählen? Ich bin bereit für unser Gespräch. (Prüfung: answer_missing:positive)
- `holdout-negation-7`: Heute bin ich nicht müde. → Danke für die Einordnung. Magst du mehr darüber erzählen? (Prüfung: answer_missing:not_tired)
- `holdout-negation-8`: Ich bin nicht unzufrieden. → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:positive, interpretation_missing:user_positive, response_act_missing:user_positive, unnecessary_abstention)
- `holdout-person-roles-1`: Anna geht es gut. → Möchtest du erzählen, was bei dieser Person los ist? (Prüfung: answer_missing:third_positive)
- `holdout-person-roles-2`: Anna geht es schlecht, mir geht es gut. → Möchtest du erzählen, was bei dieser Person los ist? Das klingt schön. (Prüfung: answer_missing:third_negative)
- `holdout-person-roles-3`: Meinem Hund geht es schlecht. → Möchtest du erzählen, was bei dieser Person los ist? (Prüfung: answer_missing:third_negative)
- `holdout-person-roles-4`: Meinem Hund geht es gut, aber ich bin traurig. → Möchtest du erzählen, was bei dieser Person los ist? Das klingt nach einem schweren Tag. (Prüfung: answer_missing:third_positive)
- `holdout-person-roles-5`: Paul ist müde und ich bin zufrieden. → Möchtest du erzählen, was bei dieser Person los ist? Das ist ein schöner Moment. (Prüfung: answer_missing:third_tired)
- `holdout-short-context-3`: Du auch? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:assistant_status, unnecessary_abstention)
- `holdout-short-context-5`: Und wie sieht es bei dir aus? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:assistant_status, unnecessary_abstention)
- `holdout-name-switch-3`: Nenn mich bitte Lukas. → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:introduce_name)
- `holdout-name-switch-4`: Kannst du dich noch an meinen Namen erinnern? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:recall_name, name_missing_in_session_state, unnecessary_abstention)
- `holdout-name-switch-6`: Wie heiße ich jetzt? → Du hast mir gesagt, dass du Meryem heißt. (Prüfung: answer_missing:recall_name, name_missing_in_session_state)
- `holdout-punctuation-case-2`: mir geht's gut… wie geht’s dir denn? → Das klingt schön. Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:assistant_status)
- `holdout-punctuation-case-4`: Gut & dir? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:positive, answer_missing:assistant_status, interpretation_missing:user_positive, response_act_missing:user_positive, unnecessary_abstention)
- `holdout-mixed-social-1`: Hi und danke, dass du da bist. Mir geht es gut! → Guten Morgen! Das klingt schön. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks)
- `holdout-mixed-social-2`: Mir geht es schlecht. Danke fürs Zuhören. → Das klingt nach einem schweren Tag. Magst du erzählen, was auf deine Stimmung drückt? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks)
- `holdout-mixed-social-3`: Ich bin zufrieden, danke und bis später! → Das ist ein schöner Moment. Was hat zu dieser Zufriedenheit beigetragen? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks, answer_missing:farewell)
- `holdout-mixed-social-4`: Hallo, wie geht es dir? Danke schon mal. → Hallo! Ich bin bereit für unser Gespräch. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks)
- `holdout-mixed-social-5`: Müde bin ich, aber danke der Nachfrage. → Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:thanks)
- `stress-context-absent-3`: Du auch? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:clarification)
- `stress-malformed-1`: dir gut es wie und geht mir? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:clarification)
- `stress-malformed-2`: wie wie wie geht es wer? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:clarification)
- `stress-malformed-5`: Wasser Uhr laut schnell weshalb? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:clarification)
- `stress-malformed-6`: Und wie geht es dem? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:clarification)
- `stress-question-scope-5`: Mir geht es gut oder vielleicht auch nicht. → Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? (Prüfung: answer_missing:uncertain)
- `stress-question-scope-7`: Ich wünschte, mir ginge es gut. → Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:negative, interpretation_missing:user_negative)
- `stress-role-and-negation-3`: Nicht Anna ist müde, sondern ich. → Möchtest du erzählen, was bei dieser Person los ist? (Prüfung: answer_missing:tired, interpretation_missing:user_tired, response_act_missing:user_tired, unnecessary_abstention)
- `stress-role-and-negation-4`: Anna ist nicht müde, ich aber schon. → Möchtest du erzählen, was bei dieser Person los ist? (Prüfung: answer_missing:tired, answer_missing:third_not_tired, interpretation_missing:user_tired, response_act_missing:user_tired, unnecessary_abstention)
- `stress-role-and-negation-5`: Mein Hund ist nicht traurig. → Möchtest du erzählen, was bei dieser Person los ist? (Prüfung: answer_missing:third_positive)
- `stress-role-and-negation-6`: Gut geht es mir nicht, dir vielleicht? → Das klingt schön. Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:negative, answer_missing:assistant_status, interpretation_missing:user_negative)
- `stress-role-and-negation-7`: Mir geht es schlecht, nein, eigentlich gut. → Das klingt nach einem schweren Tag. Alles klar. Magst du noch etwas dazu erzählen? Das klingt schön. (Prüfung: wrong_mood:negative)
- `stress-session-isolation-a-2`: Merkst du dir meinen Namen? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:recall_name, unnecessary_abstention)
- `stress-session-isolation-b-1`: Welchen Namen habe ich dir genannt? → Dafür habe ich keine verlässliche Information in meinem Speicher. Kannst du mir mehr Kontext geben? (Prüfung: answer_missing:unknown_name)
- `stress-session-isolation-b-2`: Ich bin froh, dass du da bist. → Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben? (Prüfung: answer_missing:positive, interpretation_missing:user_positive, response_act_missing:user_positive, unnecessary_abstention)

Vollständige Eingaben, Ausgaben, Assertions, Laufzeiten und SHA-256-Freeze: `baseline.json`, `development.json`, `final.json`, `suite_freeze.json`, `implementation_freeze.json` im selben Verzeichnis.
