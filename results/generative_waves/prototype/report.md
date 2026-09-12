# Numerischer Vergleich zweier Token-Wellendecoder

Das Korpus enthält ausschließlich die 120 Antworttexte. Promptschlüssel wurden nicht eingelesen.

Wortschatz: 529; Prefixfelder: 3504; HRR-Dimension: 2048.
Phaseninvarianzfehler: 1.110e-16; ungefilterte HRR-Gewinnergenauigkeit: 5.6%.
Promptablation ändert 8/25 Ausgaben.

Diese Zahlen messen Numerik und Einfluss, keine semantische Richtigkeit.

| Prompt | Exakter Operator | HRR |
| --- | --- | --- |
| Hallo | Das klingt spannend. Darf es eher kreativ, sportlich oder ganz gemütlich sein? | Das ist schön zu hören. Welcher erste schritt wäre dafür machbar? |
| Guten Morgen | Du musst das nicht sofort benennen können. Was bemerkst du gerade eher: unruhe, schwere oder vielleicht einfach leere? | Ja, ich fasse mich kürzer. Was macht dich vielleicht noch unsicher? |
| Wie geht es dir? | Das klingt nach einem schönen fund im kopf. Magst du gerade? | Eher etwas kleines zum naschen oder etwas, auf das du dich beim wiedersehen besonders freust? |
| Mir geht es gut, und dir? | Dann wäre vielleicht ein ruhiger moment angenehm. War dein tag besonders anstrengend? | Dann wäre vielleicht ein ruhiger moment angenehm. War dein tag bisher? |
| Ich bin traurig | Das klingt nach einer schönen kleinen unterbrechung. Hörst du ihnen gerade bewusst zu? | Eher etwas kleines zum naschen oder etwas, auf das du dich heute melden magst. |
| Ich bin heute müde | Und wie war es für dich? Würdest du es wieder essen? | So etwas kann lange nachklingen. Was hattest du dir anders vorgestellt? |
| Heute war die Arbeit stressig | Dann darf der tag vielleicht etwas luft schaffen? | Dann darf der tag vielleicht etwas langsamer sein. Gibt es einen konkreten auslöser oder ist einfach alles gerade zu deiner stimmung? |
| Ich freue mich auf das Wochenende | Das ist verständlich. Möchtest du erzählen, was auf deine stimmung drückt? | Das darfst du sein. Welcher schritt ist dir gelungen? |
| Was möchtest du essen? | Das kann direkt etwas an der stimmung ändern. Hast du schon zu hause, und soll es heute eher schnell oder gemütlich werden? | Gemeinsam zu essen kann schön sein. Gibt es einen ort in der nähe, an dem du dich schon ein bisschen wohlfühlst? |
| Ich koche heute Nudeln | Das kann den blick auf das eigene eng machen. Gibt es etwas kleines, das du selbst zubereiten möchtest? | Eher etwas kleines zum naschen oder etwas, worauf man sich freuen kann. Hast du lust, das du zuletzt für dich passt. Welcher erste schritt wäre dafür machbar? |
| Erzähl mir etwas über Musik | Das klingt schmerzhaft. Möchtest du etwas altes auffrischen oder neu anfangen? | Eine kurze liste machen oder weisst du schon, was du damit anfangen möchtest? |
| Hast du einen guten Film gesehen? | Das klingt unangenehm. Ist etwas bestimmtes im kopf oder ist die unruhe eher ohne klaren anlass da? | Das passiert. Lässt sich am tisch noch etwas nachwürzen? |
| Ich lese gerade ein Buch | Eine kurze ehrliche nachricht kann ein guter anfang sein. Was würdest du ihm gern sagen? | Magst du die energie nutzen? |
| Draußen regnet es | Dann hilft vielleicht ein ruhiger moment angenehm. War dein tag besonders anstrengend? | Danke für den hinweis. Welcher bereich stört dich gerade am meisten sorgen? |
| Ich möchte spazieren gehen | Das klingt nach einer schönen kleinen unterbrechung. Hörst du ihnen gerade bewusst zu? | Ein tag ohne pflichten kann sich gut anfühlen. Was beschäftigt dich daran? |
| Heute habe ich viel zu tun | Das klingt nach einer vollen zeit. Was nimmt gerade den meisten raum ein? | Hast du lust, das wachsen oder die ruhe dabei? |
| Ich brauche eine Pause | Alles gute zum geburtstag! Wie möchtest du angesprochen werden? | Bis später! Wir können uns unterhalten. Was genau beschäftigt dich daran? |
| Ich kann nicht einschlafen | Was ist dir bei der entscheidung am wichtigsten? | Willkommen zurück! Schön, dass gerade etwas ruhe da ist. Möchtest du erzählen, worum es geht? |
| Ich vermisse meine Freunde | Das passiert. Lässt sich am tisch noch etwas nachwürzen? | Eher etwas kleines zum naschen oder etwas, auf das du dich beim wiedersehen besonders freust? |
| Ich habe heute Geburtstag | Wie schön! Magst du erzählen, wie dein tag bisher? | Willkommen zurück! Schön, dass du vorbeischaust. Was liegt bei dir gerade an? |
| Danke für das Gespräch | Das klingt unangenehm. Ist etwas bestimmtes im kopf oder ist die unruhe eher ohne klaren anlass da? | Magst du regentage oder fehlt dir heute die sonne? |
| Tschüss | Das ist ein schöner blick auf alltägliche dinge. Welches motiv hat dich zuletzt überrascht? | Das ankommen kann zeit brauchen. Gibt es einen konkreten auslöser oder ist die unruhe eher ohne klaren anlass da? |
| Was machst du gern? | Das klingt nach einem entspannten plan. Möchtest du lieber auftanken oder etwas unternehmen? | Hey! Schön, dass du vorbeischaust. Was würde dir ein anruf guttun, oder denkst du gerade draussen oder schaust du aus dem fenster? |
| Ich möchte etwas Neues lernen | Das klingt spannend. Darf es eher kreativ, sportlich oder ganz gemütlich sein? | Hast du lust, das du zuletzt für dich geschafft hast? |
| Was ist ein Quantencomputer? | Gern. Schön, wenn unser gespräch dir etwas weitergeholfen hat. | Magst du eher gemeinsam rätseln, miteinander planen oder direkt gegeneinander spielen? |

## Grenzen

- No independent semantic correctness metric; outputs require human inspection.
- A Fourier change of basis adds no information or semantic knowledge.
- The response-only corpus has no conversational input/output association.
- Finite vocabulary and n-gram context cannot support unrestricted answers.
- HRR crosstalk is constrained by an explicit corpus-derived grammar mask.
- Novel whole strings and perfect wave reconstruction do not establish useful answers.
