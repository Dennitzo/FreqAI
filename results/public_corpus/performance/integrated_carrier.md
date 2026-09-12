# Integrierte numerische Beschleunigung

Der Tokenoperator verwendet jetzt einen FFT-Träger mit
`carrier_size = scipy.fft.next_fast_len(size)`. Die zusätzlichen Moden haben
Amplitude null. Das Padding erfolgt vor der Transformation im Tokenraum;
gespeicherte Spektren werden nicht mit zusätzlichen Frequenzbins aufgefüllt.
Wortschatz, Tokenfrequenzen, Datengewichte und semantische Regeln bleiben gleich.

Geänderte Dateien:

- `freqai/generative.py`: `carrier_size` neben der bekannten Tokenanzahl `size`;
  Basis-, Phasen- und Promptfelder des Operators verwenden dieselbe größere
  Trägerlänge. Die Wortberechnung liest nur `IFFT(result)[:size]`, während die
  numerische Roundtrip-Diagnose den vollständigen Träger prüft. `direct`, HRR
  und der Featurekanal behalten ihre bisherige Rechnung.
- `freqai/generation_runtime.py`: Die Kontextanzeige berechnet denselben
  gepaddeten Träger wie der Operator. `mode_count` bleibt die Zahl bekannter
  Tokenmoden; `carrier_size` gibt die davon getrennte Trägerlänge an. Gespeicherte
  Kontextamplituden bleiben nach Tokenname adressiert.
- `tests/test_generative_carrier.py`: 14 neue Testfälle für Compact-/Dense-
  Darstellung, ursprünglichen ungepaddeten Operator, unveränderte direkte
  Referenz, komplexe Phasenänderungen, Promptnullung, HRR, Frequenzstabilität
  und echte Kontextmoden beim Vergrößern des Wortschatzes.
- `tests/test_generation_conditioning.py`: Die Prüfung der tatsächlichen
  Kontextanzeige erzwingt jetzt einen Wortschatz mit zusätzlichem FFT-Padding
  und prüft die nullbelegten Zusatzmoden nach der inversen Transformation.

Die gezielte Generatortestsuite bestand mit **80 Testfällen**, darunter den
**14 neuen Carrier-Testfällen**. Anschließend meldete der Root-Lauf die komplette
Suite mit **475 bestandenen Tests in 26,99 Sekunden**.

## Gemessene Wirkung

Der isolierte Vergleich verwendete den finalen Korpus mit 12.000 Dokumenten,
44.614 Tokens und 10.845 Featuremoden. Der Token-Träger wächst auf 44.800 Moden.
Drei feste, bereits im Korpus vorhandene Gesprächsprompts mit einem Limit von
40 Tokens erzeugten tatsächlich 9, 16 beziehungsweise 15 Tokens:

| Ausgabe | Ursprünglicher Operator | Gepaddeter Operator | Faktor |
|---|---:|---:|---:|
| 9 Tokens | 4,186 s | 0,936 s | 4,47 |
| 16 Tokens | 6,521 s | 1,442 s | 4,52 |
| 15 Tokens | 5,807 s | 1,332 s | 4,36 |

Der Median beträgt **4,47-fache Beschleunigung**. Alle drei vollständigen
Tokenfolgen waren identisch. Zwölf reale Prefix-/Zeit-/Phasenzustände stimmten
mit dem bisherigen Operator und der direkten Referenz nach der bereits
vorhandenen Rundung exakt überein. Weitere 144 komplexe Toyzustände und zwei
Nullspeicherprüfungen bestanden im isolierten Prototyp.

Diese Zahlen betreffen die Wortberechnung. Der einmalige Modellaufbau dauerte
im gleichen Versuch 85,54 Sekunden und wird durch diesen Patch nicht beschleunigt.
Messwerte und ursprünglicher Corehash stehen in
[`operator_padding_report.json`](operator_padding_report.json).

## Vergleich mit den eingefrorenen Ausgaben

Die zusätzliche Regression der bereits vorhandenen 40 Chatfälle, 30 offiziellen
Faktenfälle und 20 Replayfälle ist abgeschlossen: **90/90 sichtbare Antworten
sind bytegleich**, alle 40 geprüften Gesprächskontexte stimmen überein. Das Ergebnis ist in
[`fast_carrier_output_regression.json`](../evaluation/fast_carrier_output_regression.json)
festgehalten. Es handelt sich um einen Vergleich gegen bereits eingefrorene
Ausgaben, nicht um eine neue unbekannte Abschlussprüfung oder Qualitätsoptimierung.

Bei 60 Chat-/Replayfällen liegen ursprüngliche rohe Tokenfolgen vor. Die alten
30 Faktenfälle speicherten nur den Antworttext; der Regressionsrunner kennzeichnet
seinen bytegenauen Textvergleich und daraus rekonstruierte Tokens gesondert.
Alle 60 gespeicherten und 30 rekonstruierten Tokenvergleiche bestehen ebenso wie
60 Rohstream- und 44.613 Alphabet-Rückrechnungen. Der Antwortmedian beträgt nun
1,75 Sekunden über alle 90 Fälle; die älteren Läufe waren teilweise parallel
belastet und liefern deshalb keinen kontrollierten Beschleunigungsfaktor.
