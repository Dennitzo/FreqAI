# Direkt berechnete Informationsfelder

Der neue Informationsdecoder verarbeitet deklarative Texte ohne vorgegebene Frage-Antwort-Paare. Seine komplexen Koeffizienten werden deterministisch aus beobachteten Wortübergängen berechnet. Es gibt keinen Optimierer, keine Gradientenupdates, keine vortrainierten Sprachmodellgewichte und keine kalibrierten Quellen- oder Mischverstärkungen. Der bestehende Alltagsdecoder bleibt ein gesonderter Pfad mit seinen bisherigen festen, ebenfalls nicht durch Gradienten trainierten Parametern.

## Was sich durch Fourierkoordinaten ersetzen lässt

Koeffizienten bestimmen die Wirkung eines Signals. In diesem mathematischen Sinn sind auch Fourierkoeffizienten Gewichte. Ein Basiswechsel allein beseitigt keine Parameterabhängigkeit. Für eine orthonormale Fouriermatrix `F` und einen linearen Operator `W` gilt:

```text
W_spektral = F W F*
x_spektral = F x
F* W_spektral x_spektral = W x
```

`F*` ist die adjungierte beziehungsweise inverse Matrix. Die Rechnung ist exakt derselbe Operator in anderen Koordinaten. Die [Kontrolle mit 64 synthetischen Matrizen](../results/information_corpus/weight_coordinate_audit.json) bestätigt dies mit maximal rund `7,33e-15` Ausgabedifferenz. Diese Kontrollmatrizen sind kein Sprachmodell und keine Behauptung über trainierte Gewichte im bisherigen Projekt.

Der umgesetzte Weg berechnet die Koeffizienten stattdessen direkt aus Daten. Auch Zählstatistik ist eine Datenanpassung und kann im weiten statistischen Sinn Lernen genannt werden. Die konkrete Eigenschaft dieses Projekts lautet: **keine iterative Parameteroptimierung**. Eine natürliche Verbindung zwischen beliebigen deutschen Begriffen und physikalischen Frequenzen folgt daraus nicht; die Symbolkodierung ist ausdrücklich definiert.

## Von Prosa zu komplexen Übergangsfeldern

Titel und vollständige Aussagen werden durch allgemeine deutsche Syntaxregeln in Begriffe und Eigenschaften gegliedert. Beispielsweise kann „Hertz ist die Einheit der Frequenz“ sowohl den beschriebenen Begriff Hertz als auch die Eigenschaft Einheit des Begriffs Frequenz adressieren. Das erzeugt keine Fragezeichenketten und keine Antwortvorlagen. Die Regeln sind vorgegebenes sprachliches Wissen; ihre Grenzen werden getrennt bewertet.

Eine Kopplungsadresse `g` enthält Begriffsrollen und eine Fragefacette, etwa Definition, Einheit oder Beziehung. Gemeinsam erwähnte Begriffe bleiben gekoppelt, damit eine Frage nach zwei Größen nicht durch beliebige getrennte Aussagen über jede Größe beantwortet wird. Für das Präfix `h` und ein mögliches Folgetoken `j` zählen wir:

```text
n[g,h,j] = beobachtete Häufigkeit des Übergangs h → j
a[g,h,j] = sqrt(n[g,h,j] / sum_k n[g,h,k])
C[g,h] = F_I(a[g,h,I])
```

`I` sind sämtliche belegten Tokenadressen des betreffenden Felds. Die orthonormale Transformation erhält die Energie: `sum |C[g,h]|² = 1`. Es bleiben komplexe Koeffizienten, Tokenadressen und Begriffsdeskriptoren gespeichert. Vollständige Aussagen, Dokument-IDs, reale Übergangszählungen und ganze Antwortkandidaten werden im kompilierten Informationsmodell nicht aufbewahrt. Die Originaltexte bleiben unverändert im zentralen Speicher und in den Quellenarchiven verfügbar.

## Frage und Interferenz

Die Frage wird mit denselben Begriffs- und Eigenschaftsregeln kodiert. Kompatible Kopplungsadressen erhalten gleiche Anfangsamplitude und gemeinsam Einheitsenergie. Zur Aktivierung werden ihre tatsächlich invers transformierten komplexen Merkmalskoeffizienten verwendet. Ungeklärte zusätzliche Begriffe und nicht unterstützte Negation führen zu einer ausgewiesenen Enthaltung.

Pro erzeugtem Präfix werden alle passenden Felder der längsten verfügbaren Präfixstufe demoduliert, symmetrisch überlagert und auf Einheitsenergie normiert. Es gibt weder eine Rangliste ähnlicher Dokumente noch eine Auswahl ganzer Antworttexte. Alle durch die so adressierte Übergangsgrammatik unterstützten Tokens bilden den lokalen Träger; dessen Größe wird nur durch Nullmoden auf eine schnelle FFT-Länge ergänzt. Bereits unterstützte Tokenmoden werden nicht per Top-k abgeschnitten.

Mit Datenamplitude `a`, Prompt-/Kontextamplitude `b` und stabiler Tokenphase `U_j(t) = exp(2π i f_j t)` berechnet der Operator:

```text
R = F(a * U) + exp(i phi) F(a * U * b)
z = inverse_F(R)
p_j = |z_j|² / sum_k |z_k|²
```

Die Multiplikationen im Inneren sind elementweise. Im Programm werden die Produkte über normalisierte spektrale Faltungen berechnet; die punktweise Formel bleibt als unabhängiger direkter Rechenpfad erhalten. Der produktive Promptfaktor beträgt eins. Ein anderer Faktor oder eine relative Phase ist ausschließlich ein expliziter numerischer Eingriff für Vergleichsversuche.

Weil `|U_j(t)|² = 1`, ändert die gemeinsame Zeitphase die idealen Wahrscheinlichkeiten nicht. Die relative Phase kann die Interferenz verändern. Die Tokenamplituden des Prompts und des vorherigen Kontexts werden jeweils auf Einheitsenergie normiert und symmetrisch verbunden.

## Wortfolge und Gesprächszustand

Beam Search mit vier Fortsetzungen akkumuliert die logarithmischen Tokenwahrscheinlichkeiten. Ein expliziter Zyklenschutz verhindert die wiederholte Traversierung desselben Trigramms. Anders als der alte Gesprächsdecoder verwendet der neue Informationspfad keine Temperatur, Wiederholungsstrafe oder kalibrierte Längengewichtung. Satzenden werden aus beobachteten Satz-Endmarkierungen gelesen, damit ein Punkt in einer Abkürzung den Text nicht vorzeitig beendet.

Nach jeder Antwort werden alter Kontext, neuer Prompt und erzeugte Wortzählung jeweils auf Einheitsenergie normiert, symmetrisch addiert und wieder normiert. Höchstens 256 stärkste Tokenamplituden bleiben unter ihren Tokenbezeichnungen gespeichert. Ein zusätzlicher expliziter Begriffsanker ermöglicht Folgefragen wie „Und welche Einheit hat sie?“. Beide Zustände sind Sitzungskontext und werden nicht als neue Wissensdaten importiert.

Die Kontextanzeige zeigt genau die nichtnull gespeicherten Kontextmoden mit ihren tatsächlichen Tokenfrequenzen. Der nächste Wortschritt projiziert diese auf seine jeweilige zulässige Übergangsmenge. Deshalb sind Kontextträger, Tokenvokabular, vollständige Dokumentwelle und früherer Abrufindex verschiedene, separat ausgewiesene Größen.

## Forschung und Nachweise

Vier Fachsuchen liefen über den lokalen SearXNG-Endpunkt; alle meldeten `provider=searxng` und `isFallback=false`. Primärseiten wurden anschließend direkt abgerufen. [Such- und Abrufprotokoll](../results/information_corpus/research/algorithm_research.json).

- [NumPy FFT-Dokumentation](https://numpy.org/doc/stable/reference/routines.fft.html) definiert die beidseitige Normierung `1/sqrt(N)` für `norm="ortho"`; diese Normierung wird tatsächlich verwendet.
- [FNet](https://aclanthology.org/2022.naacl-main.319/) ersetzt eine Tokenmischoperation durch eine unparametrisierte Fouriertransformation, trainiert das umgebende Sprachmodell aber weiterhin. Daraus folgt keine trainingsfreie Semantik.
- [Hsu, Kakade und Zhang](https://arxiv.org/abs/0811.4413) beschreiben spektrales Lernen von HMMs mit Singulärwertzerlegung und Matrixprodukten unter Modellvoraussetzungen. Auch eine geschlossene spektrale Parameterschätzung bleibt datenabhängig.
- [Associative Long Short-Term Memory](https://proceedings.mlr.press/v48/danihelka16.html) untersucht komplexe Assoziativspeicher und deren Interferenzrauschen. Die Arbeit begründet keinen Anspruch auf allgemeines Sprachverständnis allein durch Schwingungen.

Technische und sprachliche Ergebnisse, einschließlich fehlerhafter Antworten, stehen im [aktuellen Ergebnisbericht](../results/information_corpus/report.md). Aussagekräftig sind getrennte Belegabdeckung, Faktentreue, Grammatik, echte neue Sätze, Feldabhängigkeit und unveränderte bisherige Gesprächsantworten.
