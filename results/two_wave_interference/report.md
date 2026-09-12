# Zwei Schwingungen, eine Antwort: Ergebnis des Umbaus

Aktive Rechnung: `freqai/waves.py` (Wellenfelder, Interferenz, Messung),
`freqai/information.py` und `freqai/unpaired.py` (Ablesung pro Symbol),
`freqai/unpaired_runtime.py` (beide Wellen im Ergebnis). Beschreibung der
Mathematik: [docs/two-wave-interaction.md](../../docs/two-wave-interaction.md).

## Was jetzt rechnet

* **Datenwelle** über alle von den Daten erregten Symbolmoden. Am echten Bestand:
  116792 Moden auf einem Träger von
  117128 Stellen, Einheitsenergie
  (1.0000000000000002), Fingerabdruck `a45982a4850a4188`.
* **Promptwelle** aus Prompt (inkl. UTF-8-Bytesymbole) und gespeichertem
  Sitzungskontextfeld, ebenfalls Einheitsenergie.
* **Interferenz-Lesung**: kohärente Superposition aller resonanzgewichteten
  Datenfelder, dann Modulation durch beide Wellen, danach Intensität als
  Symbolwahrscheinlichkeit. Dieselbe Lesung für Wissensfragen und Alltagssätze.

## Numerische Prüfung am echten Bestand

| Größe | Wert |
|---|---|
| Geprüfte Zustände (alle Kontrollen, beide Ordnungsverfahren) | 1089 |
| Größte Abweichung spektraler Operator vs. direkte Rechnung | 1.112e-15 |
| Eingebaute Toleranz | 1e-11 |
| Verletzte Invarianten (Endlichkeit, Normierung auf 1) | 0 |
| Zustände mit genau einer Mode (nicht beeinflussbar) | 819 |
| Zustände mit mehreren Moden | 270 |
| Davon durch die Promptwelle messbar verändert | 172 |
| Davon durch die Datenwelle messbar verändert | 216 |
| Mittlere Resonanz beider Wellen | 0.0420 |

Das Mischungs-Theorem `spectral_convolution(F_u x, F_u y) = F_u(x·y)` wird separat
auf **1,2 × 10⁻¹⁵** geprüft; die Toleranz je Antwort ist **1 × 10⁻¹¹**. Eine Abweichung
darüber bricht die Rechnung ab, es gibt keine stille Mittelbildung.

Dass nicht jede Mehrmoden-Stufe von beiden Wellen zugleich verändert wird, ist
keine Zahlenklemme: eine Mode koppelt nur, wenn diese Welle sie überhaupt mitenergert.
Die Bedingung steht als Assertion in `tests/test_two_wave_interaction` bzw.
`tests/test_two_wave_interference.py`.

## Antworten am produktiven Bestand (20093 Informationstexte)

Compileraufbau 97.73 s bei 117055 Symbolen.
Alle bekannten API-Ausgaben werden textgenau wiederhergestellt:

| Eingabe | Berechnete Ausgabe |
|---|---|
| Hallo | Hallo! Wie geht es dir? |
| Mir geht es gut, wie geht es dir denn? | Dir geht es gut. Ich habe kein menschliches Befinden. |
| Wie heißt du? | Ich heiße FreqAI. |
| Was ist eine Frequenz? | Die Frequenz ist der Kehrwert der Periodendauer. |
| Und welche Einheit hat sie? (Sitzung) | Das Hertz (Einheitenzeichen: Hz) ist die SI-Einheit der Frequenz. |

Wissensregression (40 bekannte Fragen, nur Messung, kein Nachjustieren):
**4 beantwortet, 36 leer** – exakt dieselben vier Fälle wie vor dem Umbau
(id01, id02, id03, id09), keine neue Antwort und kein neuer Ausfall.

Zufallsfakten (12 synthetische deklarierte Texte, 32 Fälle):
**28/28** bewertete Fälle bestanden;
4 Negationsfälle bleiben wie dokumentiert
unbeantwortet, weil das System falsche Behauptungen nicht widerlegt.

## Tests

**349** technische Tests bestehen (321 bestehende unverändert + 28 neue in
`tests/test_two_wave_interference.py`). Die neuen Tests prüfen: Vollständigkeit der
Datenwelle über alle Datenfelder und Rollen, exakte Zusammensetzung der Promptwelle,
Eigenschaften der Resonanz, Operator-Identität über fünf Kopplungssätze und sechs
Feldgrößen, Normierung und Endlichkeit, Zeitinvarianz der Intensität, kausale
Sichtbarkeit beider Kopplungen in jeder Mehrmoden-Stufe, Zero-Energy-Abbruch,
Fingerprint-Stabilität, die Durchsteuerung der Regler durch `generate` sowie zwei
festgeschriebene Grenzen (Einzelmoden-Stufe, Kürzung bei Mehrordnungs-Superposition).

## Was das nicht ist

Kein vortrainiertes Modell, keine Optimierung, keine Antwortpaare. Die Interaktion
ist verifiziert; Bedeutung entsteht sie nicht. Off-diagonale Mischung über
Hashfrequenzen trägt keine Information (rund 4000 Moden pro Hertz), ein Prompt ohne
Bindung an deklarierte Daten bleibt eine leere Antwort mit Grund, und die
Mehrordnungs-Superposition `"all"` ist wegen des kumulativen Beam-Scores ohne
Längennormalisierung kürzer als die getestete Rezeptur. Gesprächsqualität wurde durch
diesen Umbau nicht neu gemessen; er ändert die Rechenbasis, nicht die Datenlage.
