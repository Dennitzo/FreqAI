# Auswahl des experimentellen Decoders

Vor dem Holdout wurden drei vorab festgelegte Konfigurationen auf denselben 30 Entwicklungseingaben verglichen. Der Seed bleibt überall 17, die Temperatur 0,8 und das Tokenbudget 40. Es gibt keine nachträgliche Auswahl eines günstigen Seeds oder einzelner Ausgaben. Alle drei Varianten verwenden denselben Sprachkorpus, dieselben Annotationen, dieselben Promptkanäle und dieselbe Verarbeitung mehrerer Eingabeakte.

| Konfiguration | Sinnvolle Antwort im qualitativen Review | Nichtleere grammatische Antwort | Neuer vollständiger Text | Ausgabe mit neuem Satz |
| --- | ---: | ---: | ---: | ---: |
| Ordnung 3, Beam | 12/30 | 29/30 | 3/30 | 0/30 |
| Ordnung 3, Sampling | 11/30 | 29/30 | 4/30 | 0/30 |
| Ordnung 2, Beam | 13/30 | 29/30 | 5/30 | 1/30 |

Die Review-Urteile stammen von einem gesonderten Codex-Agenten anhand der vorher festgelegten Rubriken; sie sind keine menschliche Nutzerstudie. Vollständige Einzelurteile und Hashbindungen stehen in `development_review_third.json`, `development_review_sample.json` und `development_review_order2.json`.

**Auswahl: Ordnung 2 mit Beam-Suche.** Der Vorsprung bei der Gesprächsqualität ist klein und auf Entwicklungsdaten gemessen. Ausschlaggebend für diesen Prototyp ist außerdem ein tatsächlich neuer, grammatischer und zur Eingabe passender Satz. Der Holdout bleibt von dieser Auswahl getrennt.

## Beleg für eine neue Satzbildung

Eingabe: „Nach dem langen Tag bin ich ziemlich erschöpft.“

Ausgabe: „**Wenn du erschöpft bist, kann ein ruhiger Moment angenehm sein.** Was kannst du für einen Moment liegen lassen?“

Der hervorgehobene Satz kommt in keinem der 418 unterschiedlichen Sätze aus den 120 bisherigen Antworttexten und den 500 zusätzlichen Sprachtexten vor. Er verbindet einen bekannten Anfang mit einer anderen, ebenfalls im Korpus vorhandenen Fortsetzung. Der Decoder wählt dabei zwölf Tokens einschließlich Komma und Punkt nacheinander aus den berechneten Moden; es wird kein fertiger Satz gesucht oder als Antwortfragment eingesetzt.

`new_sentence_evidence.json` enthält den reproduzierten Lauf, alle zwölf Präfixe, Frequenzen und ausgewählten Wahrscheinlichkeiten. Die Wahrscheinlichkeiten stimmen mit den erneut berechneten Ausgabefeldern überein. Die spektrale Faltung und die direkte Produktrechnung liefern nach der dokumentierten Rundung auf zwölf Nachkommastellen identische Verteilungen. Beide Rechnungen verwenden denselben kompilierten Sprachprior; der Vergleich prüft die numerische Umsetzung, nicht die Entstehung von Bedeutung.

Der Satz ist ein begrenzter positiver Befund. Die meisten Ausgaben folgen weiterhin bereits vorkommenden Sätzen, und mehr als die Hälfte der Entwicklungseingaben erhält keine passende Antwort. Neue Namen, bestimmte Verneinungen, Sprecherrollen und Themenwechsel bleiben schwierige Fälle. Aus diesem Beispiel folgt keine allgemeine Gesprächsfähigkeit.

## Frühere Iterationen

Die Entwicklungsstände wurden auf denselben 30 Eingaben geprüft und vollständig aufbewahrt. Vor der Auswahl ergaben die ersten drei qualitativen Bewertungen 13/30, 10/30 und 12/30 passende Antworten. Der Rückgang nach zusätzlichen Annotationen wurde nicht ausgeblendet: Ein Fehler in der Gewichtung semantischer Merkmale schwächte häufige Kategorien gegenüber seltenen Wörtern. Die spätere Korrektur zählt Vorkommen und normalisiert Kategorien getrennt von Lexemen. Diese Ergebnisse sind Entwicklung, kein unabhängiger Generalisierungsnachweis.
