# Bedingter spektraler Token-Generator: Entwicklungsvergleich

120 Gesprächspaare liefern gezählte Eingabemerkmal/Prefix/Folgetoken-Übergänge. 500 zusätzliche Textkombinationen aus 200 atomaren Sätzen liefern einen schwächer gewichteten Sprachprior. Das ist eine statistische Modellschätzung beim Import, ohne Gradientenoptimierung oder vortrainiertes Sprachmodell.

Runtime-Datenquelle sind beim Import berechnete komplexe Spektren F(sqrt(P)). Eine separate numerische Merkmalswelle aktiviert bedingte Übergangsfelder. Zeitentwicklung und Promptmodulation wirken als spektrale Faltungsoperatoren. Erst das resultierende Feld wird zu Tokenamplituden invers transformiert; die diskrete Auswahl geschieht pro Token mittels greedy oder Beam-Suche.

Die Wortfolge steht vor dieser Decoderrechnung nicht fest. Es gibt keine Dokument-ID, keine Auswahl einer kompletten Antwort und keine Antwortähnlichkeitswertung im Generierungsprozess. Im Vergleich werden fertige Ausgaben anschließend nur zur Neuheitsmessung mit dem Korpus verglichen.

## Numerische Kontrolle

```json
{
  "lexical": {
    "vocabulary_size": 822,
    "input_feature_modes": 283,
    "conditional_transition_spectra": 18553,
    "complete_prompt_ablation_max_error": 0.0,
    "data_operator_direct_reference_max_error": 3.3306690738754696e-16,
    "time_invariance_max_error": 4.718447854656915e-16,
    "time_invariant_beam_answers": 24,
    "prompt_ablation_changes_answers": 20,
    "feature_channel_changes_initial_distribution": 19,
    "metadata_free_field_identical_answers": 25
  },
  "semantic": {
    "vocabulary_size": 822,
    "input_feature_modes": 315,
    "conditional_transition_spectra": 20615,
    "complete_prompt_ablation_max_error": 0.0,
    "data_operator_direct_reference_max_error": 3.3306690738754696e-16,
    "time_invariance_max_error": 4.440892098500626e-16,
    "time_invariant_beam_answers": 25,
    "prompt_ablation_changes_answers": 21,
    "feature_channel_changes_initial_distribution": 21,
    "metadata_free_field_identical_answers": 25
  }
}
```

Eine vollständige Nullablation umfasst Token- und Eingabemerkmalskanal. Nur die Tokenamplituden auf null zu setzen lässt die eigene Merkmalswelle aktiv. Diagnose-Metadaten sind keine Quelle für deren Aktivierung.

## Ausgaben

| Prompt | Lexikalisch / greedy | Lexikalisch / Beam | Semantik / greedy | Semantik / Beam |
| --- | --- | --- | --- | --- |
| Hallo | Hallo! Schön, dass du da bist. | Hallo! Schön, dass du da bist. | Hallo! Schön, dass du da bist. | Hallo! Schön, dass du da bist. |
| Guten Morgen | Guten Morgen! Bist du schon richtig wach oder brauchst du noch einen ruhigen Moment? | Guten Morgen! Bist du schon richtig wach oder brauchst du noch einen ruhigen Moment? | Guten Morgen! Bist du schon richtig wach oder brauchst du noch einen ruhigen Moment? | Guten Morgen! Bist du schon richtig wach oder brauchst du noch einen ruhigen Moment? |
| Wie geht es dir? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? | Ich bin bereit für unser Gespräch. Wie geht es dir heute? | Hi! Ich bin bereit für unser Gespräch. |
| Mir geht es gut, und dir? | Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? | Wie schön! Was hat dich besonders gefreut: die Kruste, der Duft oder der erste Bissen? | Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? | Das klingt schön. Gab es heute einen besonderen Moment, der dir gutgetan hat? |
| Ich bin traurig | Das ist ein besonderer Anfang. Worauf freust du dich dabei besonders? | Nicht müde zu sein bedeutet nicht automatisch, voller Energie zu sein. Ich möchte dir keine Stimmung unterstellen. | Das ist schade, besonders wenn du dir viel davon erhofft hast. Was hattest du dir anders vorgestellt? | Das ist schade, besonders wenn du dir viel davon erhofft hast. Was hattest du dir anders vorgestellt? |
| Ich bin heute müde | Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend? | Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend? | Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend? | Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend? |
| Heute war die Arbeit stressig | Das klingt nach einer vollen Zeit. Was nimmt gerade den meisten Raum ein? | Das klingt nach einer vollen Zeit. Was nimmt gerade den meisten Raum ein? | Das klingt nach einer vollen Zeit. Was nimmt gerade den meisten Raum ein? | Das klingt nach einer vollen Zeit. Was nimmt gerade den meisten Raum ein? |
| Ich freue mich auf das Wochenende | Das klingt nach einem entspannten Plan. Möchtest du lieber auftanken oder etwas unternehmen? | Das klingt nach einem entspannten Plan. Möchtest du etwas vorbereiten oder lieber gemeinsam alles zusammenstellen? | Das klingt nach Vorfreude. Gibt es etwas, auf das du dich beim Wiedersehen besonders freust? | Das klingt nach etwas, worauf man sich freuen kann. Hast du schon eine Idee, Was du damit anfangen möchtest? |
| Was möchtest du essen? | Gemeinsam zu Essen kann schön sein. Gibt es jemanden, den du spontan fragen möchtest? | Gemeinsam zu Essen kann schön sein. Gibt es jemanden, den du spontan fragen möchtest? | Gemeinsam zu Essen kann schön sein. Gibt es jemanden, den du spontan fragen möchtest? | Gemeinsam zu Essen kann schön sein. Gibt es jemanden, den du spontan fragen möchtest? |
| Ich koche heute Nudeln | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? |
| Erzähl mir etwas über Musik | Welche Musik passt gerade zu deiner Stimmung? | Welche Musik passt gerade zu deiner Stimmung? | Welche Musik passt gerade zu deiner Stimmung? | Welche Musik passt gerade zu deiner Stimmung? |
| Hast du einen guten Film gesehen? | Was ist dir davon besonders im Kopf geblieben? | Was ist dir davon besonders im Kopf geblieben? | Was ist dir davon besonders im Kopf geblieben? | Was ist dir davon besonders im Kopf geblieben? |
| Ich lese gerade ein Buch | Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre? | Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre? | Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre? | Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre? |
| Draußen regnet es | Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne? | Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne? | Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne? | Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne? |
| Ich möchte spazieren gehen | Möchtest du erst eine kurze Liste machen oder weißt du schon, Was du brauchst? | Möchtest du erst eine kurze Liste machen oder weißt du schon, Was du brauchst? | Möchtest du erst eine kurze Liste machen oder weißt du schon, Was du brauchst? | Möchtest du erst eine kurze Liste machen oder weißt du schon, Was du brauchst? |
| Heute habe ich viel zu tun | Dann darf der Tag vielleicht etwas langsamer sein. Gibt es bei dir einen angenehm kühlen Ort? | Dann darf der Tag vielleicht etwas langsamer sein. Gibt es bei dir einen angenehm kühlen Ort? | Dann darf der Tag vielleicht etwas langsamer sein. Gibt es bei dir einen angenehm kühlen Ort? | Dann darf der Tag vielleicht etwas langsamer sein. Gibt es bei dir einen angenehm kühlen Ort? |
| Ich brauche eine Pause | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? |
| Ich kann nicht einschlafen | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? |
| Ich vermisse meine Freunde | Das kann wehtun. Würde dir ein Anruf guttun, oder möchtest du erst erzählen, wen du gerade besonders vermisst? | Das kann wehtun. Würde dir ein Anruf guttun, oder möchtest du erst erzählen, wen du gerade besonders vermisst? | Das kann wehtun. Würde dir ein Anruf guttun, oder möchtest du erst erzählen, wen du gerade besonders vermisst? | Das kann wehtun. Würde dir ein Anruf guttun, oder möchtest du erst erzählen, wen du gerade besonders vermisst? |
| Ich habe heute Geburtstag | Möchtest du ihm etwas Persönliches schreiben oder über eine kleine gemeinsame Unternehmung nachdenken? | Möchtest du ihm etwas Persönliches schreiben oder über eine kleine gemeinsame Unternehmung nachdenken? | Möchtest du ihm etwas Persönliches schreiben oder über eine kleine gemeinsame Unternehmung nachdenken? | Möchtest du ihm etwas Persönliches schreiben oder über eine kleine gemeinsame Unternehmung nachdenken? |
| Danke für das Gespräch | Gern. Du kannst in deinem Tempo weitererzählen. | Gern. Du kannst in deinem Tempo weitererzählen. | Gern. Du kannst in deinem Tempo weitererzählen. | Gern. Du kannst in deinem Tempo weitererzählen. |
| Tschüss | Tschüss! Hab einen angenehmen Tag. | Tschüss! Hab einen angenehmen Tag. | Tschüss! Hab einen angenehmen Tag. | Tschüss! Hab einen angenehmen Tag. |
| Was machst du gern? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? |
| Ich möchte etwas Neues lernen | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? | Dann ist eine Unterbrechung vielleicht genau richtig. Was würde dir gerade Abstand geben? |
| Was ist ein Quantencomputer? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? | Das klingt nach einem schweren Tag. Magst du ein wenig mehr erzählen? | Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade? |

## Messgrenzen

- Die numerische Gleichwertigkeit zum direkten Zählmodell ist erwartet: Fourierkoordinaten erzeugen kein zusätzliches Wissen.
- Semantische Merkmale sind explizite Sprachregeln am Eingang. Sie werden nicht aus der Fouriertransformation entdeckt.
- Beam-Suche bewertet nur Feldwahrscheinlichkeiten. Sie prüft weder Faktentreue noch Gesprächssinn.
- Neue Tokenfolgen können vollständige sinnvolle Sätze, unsinnige Mischungen oder unpassende Antworten sein.
- Mehrere Absichten und unbekannte Sachfragen bleiben mit diesem kleinen endlichen Modell wesentliche Grenzen.
- Diese 25 Prompts sind Entwicklungsbeispiele; sie sind kein unabhängiger Generalisierungsnachweis.
