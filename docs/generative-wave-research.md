# Recherche: autoregressive Textausgabe aus einem Wellenfeld

Stand: 6. September 2026. Dieses Dokument trennt die tatsächlich implementierte Variante, eine allgemeine mathematische Referenz und unabhängige Kontrollrechnungen. Es ist kein Nachweis eines allgemeinen Sprachmodells.

## Ergebnis

Eine berechenbare Verbindung zwischen Wellen und Text lässt sich durch ein **konditioniertes, autoregressives Übergangsmodell in Fourierkoordinaten** herstellen. Das Datenfeld enthält dabei Übergänge zu einzelnen nächsten Tokens. Der aktuelle Prompt und das bisher erzeugte Präfix regen dieses Feld an. Aus dem resultierenden Feld wird jeweils ein Token dekodiert und als neue Eingabe zurückgeführt. Vollständige Beispielantworten müssen dazu keine Auswahlkandidaten sein.

Die Sprachstruktur kommt aus gezählten Textübergängen, expliziten Sprachmerkmalen und dem Zustand des Gesprächs. Es gibt keine wissenschaftliche Grundlage dafür, dass eine frei gewählte Fouriertransformation diese Struktur ohne eine solche Informationsquelle erzeugt. Ein Modell ohne Gradientenoptimierung kann sinnvoll sein; Korpuszählung oder analytisches Schätzen von Übergängen ist trotzdem statistische Anpassung an Daten.

## Tatsächlich implementierte Variante

Die spätere Erweiterung um öffentliche Korpora nutzt zusätzlich eine kompakte, mathematisch äquivalente Speicherung. Für die beobachteten Tokenadressen `I` enthält ein Feld nur `C_I = F_|I|(a[I])` und die zugehörigen Adressen. Die globale Darstellung ist `S = F_V(E_I F_|I|^-1(C_I))`. Es werden weder Koeffizienten beschnitten noch ein Projektionsmodell trainiert. Der Generator demoduliert die gespeicherten komplexen Werte und bettet sie in einen gemeinsamen Träger ein. Die direkte Darstellung bleibt mit `storage="dense"` als Referenz verfügbar. Wortnachbarschaften werden dünn besetzt gespeichert. Numerische und sprachliche Auswirkungen sind im [Bericht zur öffentlichen Datenerweiterung](../results/public_corpus/report.md) getrennt dokumentiert.

[`freqai/generative.py`](../freqai/generative.py) implementiert eine Variante mit zwei Eingabekanälen und einem Intensitätsdecoder. Die Laufzeitkonfiguration verwendet `order=2`: bis zu zwei vorherige Ausgabetokens konditionieren das nächste Token. Hinzu kommen `conditioning="semantic"`, `decoding="beam"`, `prompt_gain=1.2`, `split_acts=True` und `require_conditioning=True`. Der weiter unten hergeleitete lineare Operator dient als allgemeine Referenz; seine Gleichung `z=Aq` ist nicht die vollständige Gleichung dieser konkreten Implementierung.

**Datenkanal.** Für jeden beobachteten Präfixkontext `h` werden nächste Token gezählt und zu `p_h(v)` normalisiert. Ein Datenspektrum ist `S_h = F(sqrt(p_h))`. Die zwei spezifischsten verfügbaren Präfixordnungen werden mit `0.92` und `0.08` in der Amplitude gemischt. Ein eigener Satz von Spektren `S_(f,h)` enthält dieselben lokalen Übergänge konditioniert auf Eingabemerkmale `f`. Vollständige Antworten sind bei der Ausgabe keine Kandidaten.

**Merkmalskanal.** Morphologisch vereinheitlichte Inhaltswörter, explizit erkannte Absichten, Rollen, Polaritäten und einige feste Themen aktivieren Merkmalsadressen. Die numerischen Anregungen werden per FFT kodiert, mit eigenen tokenunabhängigen Merkmalsfrequenzen fortgeschrieben und bei der Kopplung per inverser DFT ausgelesen. Wortmerkmale erhalten eine Häufigkeitskorrektur, strukturelle Merkmale ausdrücklich feste Verstärkungen. Die semantischen Anregungen kommen aus dem bestehenden regelbasierten Parser und aus den unten beschriebenen Datenannotationen; sie entstehen nicht spontan aus Textfrequenzen. Für passende Präfixe wird aus den aktivierten Merkmalskanälen ein gewichtetes konditioniertes Amplitudenfeld gebildet. Wenn sich die erlaubten Folgetokens überschneiden, mischt die Implementierung `0.04` des allgemeinen und `0.96` des konditionierten Felds. Die Schnittmenge der aus dem Korpus bekannten Folgetokens bleibt als ausdrückliche Grammatikmaske erhalten.

**Tokenkanal.** Lokale Wortnachbarschaften im Korpus und bekannte Inhaltswörter des Prompts erzeugen ein nichtnegatives Modulationsfeld `b`. Ein gespeichertes Kontextfeld `c` wird mit Gewicht `0.35` ergänzt; anschließend werden die Tokenamplituden normalisiert. Dieser Kanal ist von den Merkmalsadressen getrennt. Eine vollständige Promptablation muss beide Kanäle ausschalten. Lediglich einige Tokenamplituden auf null zu setzen prüft keine vollständige Promptunabhängigkeit.

Sei `a` die inverse DFT des gemischten Datenspektrums, `U_v(t)=exp(2πi f_v t)` die tokenabhängige Zeitphase, `γ` die Promptverstärkung und `δ` ein kontrollierter relativer Phasenfehler. Die Funktion `spectral_convolution` berechnet die zirkuläre Faltung der Frequenzindizes einschließlich des Faktors `1/sqrt(V)`. Dadurch gilt exakt:

\[
R=F(a\odot U)+\gamma e^{i\delta}F(a\odot U\odot b)
  =F\big(a\odot U\odot(1+\gamma e^{i\delta}b)\big).
\]

Der Code berechnet diese Spektraloperationen tatsächlich. Die Folgewortmasse ergibt sich aus dem resultierenden Feld:

\[
z_v=\left|F^{-1}(R)_v\right|^2
   =|a_v|^2\left(1+2\gamma\cos(\delta)b_v+\gamma^2b_v^2\right).
\]

Danach folgen die Grammatikmaske und Normalisierung. Bei synchroner Interaktion (`δ=0`) ist der Faktor `(1+γ b_v)^2`. Die gemeinsame Zeitphase fällt aus der Intensität heraus. Das System ist somit ein phaseninvarianter statistischer Decoder mit spektral berechneter Modulation. Längeres Warten erzeugt bei unverändertem Inhalt keine zusätzliche Bedeutung.

Die Ausgabe kann stichprobenweise, mit maximaler Einzelwortwahrscheinlichkeit oder per Beam Search erfolgen. Der integrierte Laufzeitmodus verwendet Beam Search. Diese sucht neu aufgebaute Präfixfolgen anhand ihrer berechneten Tokenwahrscheinlichkeiten. Temperatur, Wiederholungsdämpfung, Satzgrenze und Längennormalisierung sind weitere explizite Decoderregeln. Ein neuer Satz kann aus bekannten lokalen Übergängen entstehen; eine tatsächlich neue sinnvolle Antwort muss separat anhand des Gesprächsinhalts bewertet werden.

**Geordnete Eingabebänder.** Erkennt der Parser zwei bis vier verschiedene Gesprächsabsichten, erzeugt ein expliziter Ablaufcontroller nacheinander Merkmalsbänder in ihrer Eingabereihenfolge. Pro Band wird höchstens ein Satz tokenweise berechnet. Danach beginnt das nächste Band wieder am Satzanfang; das vorherige Ausgabepräfix wird nicht als gemeinsames Präfix weitergeführt. Die Sätze werden anschließend zusammen ausgegeben. Das sind neu berechnete Tokenfolgen, keine aus Dokumenten ausgeschnittenen Antwortbausteine. Der Ablaufcontroller und die Absichtserkennung sind dennoch zusätzliche Sprachregeln. Die Tokenmodulation berücksichtigt weiterhin den gesamten Prompt und den vorherigen Kontext; nur die Merkmalsaktivierung wird auf die aktuelle Absicht begrenzt. Mehr als vier erkannte Absichten werden nicht auf diese Weise getrennt. Das Gesamtbudget kann einzelne Bänder unvollständig oder unbeantwortet lassen.

**Numerische Eingabeschranke.** `require_conditioning=True` verhindert eine Ausgabe, wenn die Norm des numerisch kodierten Merkmalskanals unter `1e-12` liegt. Der Generator gibt dann keine Tokenfolge aus und meldet `no_supported_input_coupling`. Die Entscheidung verwendet die Feldwerte, nicht die erklärenden Merkmalsnamen im Ergebnisprotokoll. Damit führt ein rein unbekannter Input nicht automatisch zur beliebigen Fortsetzung des Sprachpriors. Die Schranke ist aber keine kalibrierte Vertrauensbewertung: Eine neue Frage mit irgendeinem bekannten Inhaltswort oder Thema kann sie passieren und trotzdem unpassend beantwortet werden. Unbeantwortete Teilbänder sind über `unanswered_bands` ausgewiesen. Ein vorhandener Text allein belegt keine vollständige Bearbeitung des Prompts.

[`freqai/generation_runtime.py`](../freqai/generation_runtime.py) speichert den Gesprächskontext als Tokenname und Amplitude, nicht als von der aktuellen Vokabularsortierung abhängigen Index. Beim Vergrößern des Vokabulars werden gespeicherte Tokens erneut den richtigen Adressen zugeordnet. Das Kontextfeld wird mit Faktor `0.6` aus dem vorherigen Zustand, der neuen Promptanregung und einem schwächeren Ausgabetokenbeitrag `0.08` aktualisiert, normalisiert und auf die stärksten 256 Adressen begrenzt. Neue Ausgabetokens werden nicht automatisch zu Korpusbelegen. Die Kontextanzeige verwendet dieselbe komplette Vokabularbasis und das Spektrum `FFT(c · U(t))`; nur die Darstellung wird auf wenige Punkte reduziert. Zusätzlich liegen einige explizite Gesprächslabels wie das zuletzt behandelte Thema außerhalb des numerischen Felds vor.

Für einen neuen veröffentlichten Speicherzustand wird der Generator aktuell neu aus den Korpusstatistiken aufgebaut und zwischengespeichert. Änderungen an der Annotationsdatei invalidieren den Cache ebenfalls anhand ihrer Änderungszeit und Größe. Die weiter unten beschriebene direkte Aktualisierung `ΔA` ist eine mögliche Optimierung; ein bereits inkrementell arbeitender Generatorkern wird damit nicht behauptet.

## Sprachprior, Annotationen und feste Grenzen

[`memory/language/generative_corpus.jsonl`](../memory/language/generative_corpus.jsonl) enthält 500 selbst verfasste synthetische Datensätze. Sie entstehen aus **200 atomaren Sätzen**: In jeder von 20 Familien werden fünf Anfänge mit fünf Fortsetzungen kombiniert. Das ergibt 25 Texte je Familie. Es sind keine 500 unabhängigen Gespräche oder Beobachtungen. Die Herkunft und Erzeugungsweise sind in [`generative_corpus.provenance.json`](../memory/language/generative_corpus.provenance.json) ausgewiesen. Zusammen mit den bisherigen 120 Gesprächspaaren ergeben sie im Versuch 620 aktive Texte. Der zusätzliche Prior besitzt keine Promptfelder; sein allgemeiner statistischer Beitrag erhält gegenüber den gepaarten Beispielen das Gewicht `0.25`.

[`memory/language/generative_categories.json`](../memory/language/generative_categories.json) ordnet diese 20 Familien ausdrücklich Eingabemerkmalen zu. Beispielsweise adressiert `positive` die positive Stimmung der sprechenden Person, `rest` Müdigkeit und Schlaf, `music` das Thema Musik und `memory` eine Namensabsicht. Die Annotationen gelten für Texte mit dem dort festgelegten Quellenpräfix. Aus ihnen werden konditionierte lokale Folgetokenfelder gezählt. Diese Beschriftung ist **vorgegebene sprachliche Supervision**, auch wenn keine Gradienten optimiert werden. Die Verbindung von Eingabeabsicht und Textart wird damit nicht aus einer neutralen Welle entdeckt.

Die Merkmalsontologie, der Parser, seine Wortlisten und die Rollenregeln sind endlich und vorgegeben. Ähnliche Formulierungen können dieselbe Merkmalsadresse erreichen; ein neuer Ausdruck außerhalb dieser Regeln muss diese Adresse nicht erreichen. Neue Weltkenntnisse oder eine allgemeine Semantik entstehen daraus nicht automatisch.

Auch das Ausgabevokabular ist endlich. Ein bislang unbekannter Name kann ohne passenden Vokabulareintrag und Übergänge nicht frei geschrieben werden. Es gibt im Wellenmodus keinen allgemeinen Zeichen- oder Kopierdecoder für neue Namen. Die Kategorie `intent:name` bezeichnet eine Gesprächsart; sie stellt keine Speicherung und korrekte Wiedergabe eines beliebigen Nutzernamens sicher. Vorhandene Korpussätze, die eine solche Fähigkeit sprachlich behaupten, sind deshalb kein Funktionsnachweis. Der bisherige Regeldialog hat andere, ausdrücklich getrennte Fähigkeiten und darf nicht als verdeckter Rückfallpfad des Generators dienen.

Ein vorab begrenzter Entwicklungsvergleich der gewählten Variante ergab laut unabhängiger Agentenprüfung **13 von 30 inhaltlich passende Antworten**, fünf neue vollständige Ausgabetexte und einen tatsächlich neuen sinnvollen Satz: „Wenn du erschöpft bist, kann ein ruhiger Moment angenehm sein.“ Dieser Satz steht so nicht im verwendeten Korpus. Das zeigt eine begrenzte lokale Kombination von Sprachmustern. Der Entwicklungsvergleich wurde für die Variantenwahl verwendet und ist kein abschließender Holdout-Nachweis. Die Belege liegen in [`development_order2_20260906T085619113266Z.json`](../results/generative_waves/development_order2_20260906T085619113266Z.json) und [`development_review_order2.json`](../results/generative_waves/development_review_order2.json). Die endgültige Bewertung wird im separaten Versuchsbericht geführt.

## Suchweg und Primärquellen

Zehn Abfragen liefen über den lokalen Endpunkt `POST http://127.0.0.1:8080/v1/research/web` mit `maximumResults=8` und `language=de-DE`. Alle Antworten meldeten HTTP 200, `provider=searxng` und `isFallback=false`. Die Fachabfragen lieferten teilweise keine beziehungsweise überwiegend irrelevante Treffer. Deshalb wurden bekannte Primärquellen zusätzlich direkt abgerufen; eine andere Suchmaschine wurde nicht direkt verwendet. Rohantworten und Abrufprotokolle einschließlich fehlgeschlagener Quellen stehen unter [`results/generative_waves/research/`](../results/generative_waves/research/).

| Primärquelle | Gesicherter Beitrag und praktische Grenze |
| --- | --- |
| [Shannon: A Mathematical Theory of Communication, 1948, Abschnitte 2–3](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) | Beschreibt Quellen, die Zeichen oder Wörter anhand vorheriger Symbole und Übergangswahrscheinlichkeiten erzeugen. Die Beispiele zeigen zunehmende lokale Sprachstruktur durch höhergradige Statistik; daraus folgt kein Verständnis beliebiger Fragen. |
| [Chen und Goodman: An Empirical Study of Smoothing Techniques for Language Modeling, 1998](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-10-98.pdf) | Untersucht Glättung von N-Gramm-Modellen. Seltene oder ungesehene lokale Kontexte verlangen Rückfall auf weniger spezifische Statistiken; zu starke Glättung kann unpassende Wörter zulassen. |
| [Hsu, Kakade und Zhang: A Spectral Algorithm for Learning Hidden Markov Models](https://arxiv.org/abs/0811.4413) | Zeigt ein spektrales Lernverfahren mit Singulärwertzerlegung und Matrixprodukten unter Voraussetzungen an das HMM. „Spektral“ bedeutet hier eine Methode der Parameterschätzung und ist kein Beweis für lernfreie Semantik. |
| [Danihelka et al.: Associative Long Short-Term Memory, ICML 2016](https://proceedings.mlr.press/v48/danihelka16.html) | Verbindet komplexe Vektoren mit holografischen Assoziativspeichern. Beim Überlagern vieler Einträge steigt Interferenzrauschen; redundante Kopien reduzieren es. Ein solches Auslesen adressierter Inhalte ist zunächst Assoziation, nicht freie Sprachgenerierung. |
| [Lee-Thorp et al.: FNet: Mixing Tokens with Fourier Transforms, NAACL 2022](https://aclanthology.org/2022.naacl-main.319/) | Ersetzt einen Teil der Tokenmischung eines Transformer-Encoders durch eine Fouriertransformation. Das gesamte Sprachmodell wird weiterhin trainiert; die unparametrisierte Fourieroperation ersetzt keine Sprachkenntnisse. |
| [Jaeger: The “echo state” approach, korrigierte Fassung 2010 des Berichts von 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf) | Ein rekurrenter Zustand kann Eingabegeschichte tragen. Der Ausgang wird durch angepasste Auslesegewichte berechnet; im beschriebenen Ansatz entspricht das einer linearen Regression. Ein zufälliges Reservoir allein liefert keine passende Textausgabe. |
| [Rusch und Mishra: Coupled Oscillatory Recurrent Neural Network, ICLR 2021](https://arxiv.org/abs/2010.00951) | Nutzt diskretisierte kontrollierte nichtlineare Oszillatoren als RNN und untersucht Gradientenstabilität. Die Architektur eignet sich für trainierte Sequenzverarbeitung; sie belegt kein universelles Textmodell aus ungelernten Schwingungen. |

Die IEEE-Seite zu Plate 1995 lieferte eine leere HTTP-202-Antwort, PubMed eine Cookie-Abfrage und die zuerst gefundene Fraunhofer-Seite eine Bot-Abfrage. Diese Antworten wurden archiviert, aber nicht als gelesener Paperinhalt ausgegeben. Für holografische Speicher wird daher die zugängliche Primärarbeit von Danihelka et al. verwendet; für Echo-State-Netze die korrigierte Autorenversion.

## Allgemeine lineare Referenz für einen Wellenalgorithmus

### 1. Diskrete Bedeutung und kompatible Wellen

Sei das Tokenvokabular `V` endlich, mit einem zusätzlichen Ende-Token. Die Zuordnung Token ↔ Index wird ausdrücklich gespeichert. Für einen Tokenindex `v` ist `e_v` der entsprechende Einheitsvektor. Mit der unitären DFT `F_V` erhält er die Wellenrepräsentation

\[
\phi_v=F_Ve_v.
\]

Die Wahl der Tokenindizes ist eine Kodierungsvereinbarung. Eine andere konsistente Permutation ändert keine Sprachbedeutung. Ein ähnlicher Frequenzwert allein bedeutet deshalb keine ähnliche Bedeutung.

Ein Eingabekanal `j` bezeichnet einen lokalen Ausgabekontext, gegebenenfalls zusammen mit Merkmalen des Prompts oder der Sprecherrolle. Die nichtnegative Anregung `q_j` entsteht aus dem aktuellen Prompt, dem Gesprächszustand und den letzten `n-1` ausgegebenen Tokens. Merkmalsregeln beziehungsweise aus Daten geschätzte Merkmalsbeziehungen sind als eigene Informationsquelle offenzulegen.

### 2. Daten als spektraler Übergangsoperator

Für jedes Folgetoken `v` und jeden Eingabekanal `j` zählt das Korpus `A[v,j]` beobachtete Übergänge. Alternativ enthält `A` bereits geglättete bedingte Massen. Rohzählungen und normalisierte Wahrscheinlichkeiten dürfen nicht unbemerkt vermischt werden. Der Datenoperator im Frequenzraum ist

\[
M=F_V A F_C^*,\qquad P=F_Cq.
\]

`C` ist die Anzahl der Eingabekanäle und `*` die konjugierte Transposition. Das Datenfeld wirkt auf die Prompt-/Kontextwelle:

\[
R=M P,\qquad z=\operatorname{Re}(F_V^*R).
\]

Die letzte Gleichung liefert eine Masse pro möglichem **nächsten Token**, keine Rangliste vollständiger Antworten. Eine große dichte Matrix dient hier zur eindeutigen mathematischen Definition. Praktisch lassen sich beobachtete Kontexte dünn besetzt speichern, irrelevante Kanäle auslassen und einzelne Fourierzeilen bei Bedarf berechnen.

### 3. Dauerhafte Schwingung und Synchronisation

Mit diagonalen Phasenoperatoren `D_C(t)` und `D_V(t)`, deren Einträge `exp(i ω_k t)` sind, kann das Feld dauerhaft schwingen:

\[
M(t)=D_V(t)M D_C(t)^*,\quad
P(t)=D_C(t)F_Cq,\quad
R(t)=M(t)P(t).
\]

Die synchrone Dekodierung ergibt

\[
F_V^*D_V(t)^*R(t)=Aq.
\]

Beide Quadraturen beziehungsweise komplexe Amplitude und Phase müssen erhalten bleiben. Die Auslese nur einer reellen Momentaufnahme kann einen Modus an seinem Nulldurchgang verlieren. Ein reiner Intensitätsdetektor ist ebenfalls nicht injektiv: `R` und `-R` haben identische Intensität.

Die Phasensynchronisation verhindert, dass identische Gesprächszustände allein aufgrund einer anderen Uhrzeit andere Bedeutungen erhalten. Der Inhalt verändert sich bei neuen Daten, neuen Eingaben und ausgegebenen Tokens. Ein Hintergrundloop oder eine animierte Kurve ist für sich genommen keine zusätzliche Schlussfolgerung.

### 4. Auslesen, Rückführen und Beenden

Aus der realen Tokenmasse entsteht nach Behandlung von numerischem Rauschen eine Verteilung. Bei einem nichtnegativen exakten Zähloperator sollten negative Werte höchstens Rundungsfehler sein; größere negative Massen sind ein Fehler und dürfen nicht stillschweigend weggeschnitten werden. Anschließend gilt zum Beispiel

\[
p_v=\frac{\max(z_v,0)}{\sum_u\max(z_u,0)}.
\]

Der Decoder wählt ein Token deterministisch oder zieht eine reproduzierbar gesetzte Zufallsstichprobe. Dieses Token wird dem Präfix hinzugefügt, `q` wird erneut berechnet und der Vorgang bis zum Ende-Token oder einer begründeten Längengrenze wiederholt. Die Auswahl eines Vokabelsymbols ist bei einem diskreten Decoder notwendig; vermieden wird die Auswahl einer vollständigen vorhandenen Antwort.

Für schwach belegte Kontexte kann man mehrere N-Gramm-Ordnungen mischen. Eine zusätzliche Wiederholungssperre, Längenbegrenzung oder grammatische Einschränkung ist ein separater Decodermechanismus und muss als solcher ausgewiesen werden. Eine Grammatik darf nicht unbemerkt eine ganze gewünschte Antwort vorschreiben und anschließend als emergente Wellenleistung erscheinen.

### 5. Ergänzungen während des Betriebs

Neue Texte ergeben additive Übergänge `ΔA` und damit

\[
M_{neu}=M+F_V\Delta A F_C^*.
\]

Ein neuer Datensatz kann so in einen neuen konsistenten Speicherzustand veröffentlicht werden, während laufende Anfragen ihren bisherigen Zustand abschließen. Bei veränderter Vokabulargröße muss die Implementierung entweder stabile reservierte Tokenplätze verwenden oder die betroffenen Fourierbasen konsistent erweitern. Ein einfaches Anhängen an Spektren mit anderer DFT-Länge ist mathematisch falsch.

## Weshalb einfache Interferenz nicht ausreicht

Für die lineare Fouriertransformation gilt `IFFT(FFT(x)+FFT(y))=x+y`. Werden Textbytes unmittelbar als Amplituden behandelt, entsteht eine Byteaddition und keine Antwortrelation. Nichtlineare Kopplungen können neue Frequenzanteile erzeugen, legen aber ohne eine sprachlich begründete Zuordnung nicht fest, welche Wörter zu einer Frage passen.

Holografisches Binden und Entbinden kann den Übergangsoperator näherungsweise verdichten: Schlüssel und Werte werden mit komplexen Phasen gebunden und überlagert. Das spart gegebenenfalls Speicher, erzeugt aber Kreuzterme. Die Qualität ist anhand von Kapazität, Fehlerraten und Wortausgaben zu messen. Das erfolgreiche Wiederfinden eines gebundenen Wortes allein ist noch kein sinnvoller Satz.

Der obige exakte Fourieroperator ist mathematisch äquivalent zu `z=Aq`. Diese Äquivalenz ist erwünscht als überprüfbare Referenz. Sie verbietet zugleich die Behauptung, die Fourierkoordinaten hätten von sich aus zusätzliche Semantik erzeugt. Ein Qualitätsvorteil durch andere Dynamik wäre erst durch einen kontrollierten Vergleich zu zeigen.

## Unabhängige Kontrollrechnungen

[`check_spectral_identities.py`](../results/generative_waves/research/check_spectral_identities.py) verwendet Seed `20260906`, 32 Eingabekanäle, 64 Ausgabetokens und 1.000 zufällige Anregungen bei Zeiten zwischen 0 und 86.400 Sekunden. Es prüft den direkten und den spektralen Operator, phasensynchrone Auslese sowie additive Erweiterungen. Das sind numerische Kontrollen; es sind ausdrücklich keine 1.000 bestandenen Gesprächsaufgaben.

| Messgröße | Ergebnis |
| --- | ---: |
| Größte Abweichung zur direkten Rechnung | `1.1102230246251565e-15` |
| Größter imaginärer Ausleserest | `1.8041124150158794e-16` |
| Größte Abweichung nach additiver Erweiterung | `1.332339907734402e-15` |
| Kleinster Ausgabeunterschied nach Permutation der Promptanregung, L2 | `0.4808012593916851` |
| Gegenphasige Felder besitzen identische Intensität | bestätigt |

Ein absichtlich ungeeigneter Gegenversuch mittelt direkt die Bytewellen von `Mir geht es gut.` und `Wie geht es dir?`. Die rekonstruierte Ausgabe lautet `Ril geht es fos6`. Das Beispiel illustriert die lineare Mischoperation, ersetzt aber keinen systematischen Sprachbenchmark. Alle Messwerte stehen in [`spectral_identities.json`](../results/generative_waves/research/spectral_identities.json).

## Kriterien für den eigentlichen Sprachversuch

Vor einer Erfolgsaussage müssen mindestens folgende Eigenschaften getrennt gemessen werden:

- **Numerische Kausalität:** Nullsetzen des Datenfelds oder der Promptkopplung verändert die Ausgabemassen wie mathematisch erwartet. Mit absichtlich falschen Phasen darf die korrekte Dekodierung nicht nur aus einem versteckten Klartextpfad stammen.
- **Autoregression:** Veränderte Präfixtokens ändern die nächste Wortverteilung. Der Decoder wird erneut pro Token aufgerufen und liefert keine fertige Antwort aus einer Dokumentliste.
- **Unabhängige Sprachevaluation:** Vollständig zurückgehaltene Gespräche prüfen passende Bedeutung, Rollen, Verneinung, Grammatik und unbekannte Fragen. Exakte Korpusantworten, neue Tokenfolgen und neu kombinierte vollständige Sätze werden getrennt gezählt.
- **Vergleich:** Ein direkter N-Gramm-Decoder mit denselben Daten und Einstellungen ist die notwendige Referenz. Eine unveränderte Trefferquote bei höherem FFT-Aufwand ist kein nachgewiesener Intelligenzgewinn.
- **Begrenzung:** Ein bisher nie gesehener vollständiger Satz kann durch bekannte Übergänge entstehen. Daraus folgt weder neues Faktenwissen noch die Fähigkeit, beliebige sinnvolle Antworten zu geben.

Die Quellen begründen eine technisch nachvollziehbare Forschungsrichtung. Sie begründen keine Neuheitsbehauptung über sämtliche existierenden Systeme und kein Versprechen, jede Frage allein durch längeres Schwingen lösen zu können.
