# Der konkrete Algorithmus

**Historische Beschreibung des ersten Versuchs.** Für den aktuellen zentralen Alltags-Gesprächsspeicher, getrennte Anlass/Antwort-Codierung und unveränderte Moden bei Ergänzungen gilt [live-memory.md](live-memory.md). Die folgenden Formeln und die Legacy-Experimente bleiben als Herleitung erhalten; die frühere aktive NPZ-/Gesamtarchiv-Architektur wurde ersetzt.

FreqAI verwendet einen verlustfreien Nutzdatenspeicher, einen separaten Fourier-Suchindex und einen expliziten Decoder. Es gibt weder Optimierer noch vortrainierte Gewichte. Die Zuordnung von Zeichen zu Moden wird konstruiert; sie ist keine gemessene Naturkonstante von Sprache.

## 1. Text in eine stehende Welle umrechnen

Für die UTF-8-Bytes \(b_n\) eines Textes setzt der Codec

\[
x_n=(b_n-127.5)/127.5,\qquad
a_k=\alpha_k\sum_{n=0}^{N-1}x_n\cos\frac{\pi k(n+1/2)}{N},
\]

mit \(\alpha_0=1/\sqrt N\), sonst \(\alpha_k=\sqrt{2/N}\). Das ist die orthonormale DCT-II. Die Implementierung verwendet [`scipy.fft.dct` und `idct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html) mit `type=2, norm="ortho"`. `idct(type=2)` ist in dieser API die zugehörige Inverse, mathematisch die passend normierte DCT-III.

Die räumliche Basis lautet \(\phi_k(x)=\alpha_k\cos(\pi kx/N)\), an den Gitterpunkten \(x=n+1/2\). Mit \(\omega_k=2\pi k\cdot30/N\) gilt

\[
u(x,t)=\sum_k a_k\phi_k(x)\cos(\omega_k t).
\]

Dies ist eine endliche Lösung der idealen Wellengleichung mit Neumann-Randbedingungen (verschwindende räumliche Ableitung) und Geschwindigkeit 60 Gittereinheiten pro Sekunde. Die Frequenzobergrenze knapp unter 30 Hz ist eine frei gewählte Simulationsskala. Sie beschreibt keine natürliche Frequenz eines Wortes. Beim Vergrößern des Speichers werden die Frequenzen neu zugeordnet; das Einlesen ist ein Neuaufbau der digitalen Darstellung, kein energieerhaltender physikalischer Schreibvorgang.

## 2. Zu jeder Zeit zurücklesen

Gespeichert und analytisch ausgewertet werden beide Modenquadraturen:

\[
q_k(t)=a_k\cos\omega_kt,\quad p_k(t)=a_k\sin\omega_kt.
\]

Für \(k>0\) ist \(p_k=-\dot q_k/\omega_k\); die DC-Mode hat \(q_0=a_0,p_0=0\). Synchrones Rücklesen ergibt

\[
a_k=q_k\cos\omega_kt+p_k\sin\omega_kt,
\qquad
b_n=\operatorname{round}(127.5\operatorname{IDCT}(a)_n+127.5).
\]

Die Phase beziehungsweise gemeinsame Uhr gehört zum Decoder. Eine einzelne Momentaufnahme der Auslenkung reicht an einem Nulldurchgang nicht. Die Bytefolge wird anschließend strikt als UTF-8 decodiert und gegen SHA-256 geprüft. Bei einer Abweichung entsteht ein Fehler, kein stillschweigend reparierter Antworttext. Leere Texte haben eine explizite Länge 0 und eine technische Puffermode.

\(\sum_k(q_k^2+p_k^2)\) bleibt konstant und wird als **Quadraturnorm** angezeigt. Die klassische normierte Wellenenergie ist \(\frac12\sum_k\omega_k^2(q_k^2+p_k^2)\); sie zählt die statische DC-Mode nicht. Die Auswertung verwendet geschlossene Sinus-/Kosinusrotationen statt eines Euler-Zeitschrittverfahrens. Es gibt deshalb keine akkumulierte Integrationsdrift, aber endliche Gleitkommagenauigkeit und begrenzte Auflösung sehr großer Zeitwerte.

## 3. Prompt-Interferenz und Antwort

Aus einem Text werden feste, vorab definierte Merkmale gebildet: normalisierte Wörter und Zeichenfolgen der Länge 3/4, mit einer sichtbaren deutschen/englischen Stopwortliste. BLAKE2b ordnet die Merkmale deterministischen vorzeichenbehafteten Kanälen zu. Es wird keine IDF aus Testfragen gelernt. Im hybriden Modus liegen Wörter und Zeichen in getrennten Teilräumen; ihre Energiegewichte betragen 0,75 und 0,25.

Der normierte Merkmalsvektor \(f_i\) wird mit einer unitären FFT in \(K_i=Ff_i\) umgerechnet. Der Prompt erhält \(Q=Ff_q\). Jede Dokumentadresse bleibt ein eigener Kanal. Mit synchroner Zeitentwicklung \(U(t)\) lautet die Interferenzmessung

\[
I_i=\|U(t)K_i+U(t)Q\|^2-\|U(t)K_i\|^2-\|U(t)Q\|^2
=2\operatorname{Re}\langle K_i,Q\rangle.
\]

Der Code berechnet den Kreuzterm direkt, um Subtraktionsverluste nahe null zu vermeiden. Ausgewählt werden die höchsten Scores \(I_i/2\), sofern sie die feste Schwelle 0,18 erreichen und wenigstens ein echtes Inhaltswort gemeinsam haben. Diese zusätzliche lexikalische Regel verhindert Antworten allein durch Hashkollisionen. Sie beweist keine inhaltliche Relevanz. Der vollständige ausgewählte Text wird aus seinen zeitentwickelten DCT-Moden zurückgerechnet. Es gibt keine vorgefertigte Frage-Antwort-Tabelle.

Der erste Rechenlauf zeigte zwei zeitabhängige Wechsel bei theoretisch gleichen Scores. Deshalb verwendet die Auswahl eine festgelegte Ausleseauflösung von 10^-12 und entscheidet Gleichstände in Korpusreihenfolge. Angezeigte Scores bleiben ungerundet. Der fehlgeschlagene erste Versuch liegt unverändert unter `results/iterations/01_before_tie_fix.json`; der folgende vollständige Lauf prüft die Korrektur.

Parseval zeigt ausdrücklich: \(\operatorname{Re}\langle Ff_i,Ff_q\rangle=f_i^Tf_q\). Die Fourier-Suche ist bei diesen Merkmalen mathematisch gleichwertig zur direkten Kosinusähnlichkeit. Ein Bedeutungsgewinn durch bloßen Basiswechsel wird weder erwartet noch behauptet. Längeres unverändertes Schwingen verbessert den Score nicht.

## 4. Warum alle Daten Adressen behalten

Für Bytefolgen gilt `AB + BA = AA + BB` als Vektorsumme. Auch ihre Fourier-Summen sind gleich. Aus einer solchen unmarkierten Summe lässt sich der Ursprung nicht eindeutig rekonstruieren. FreqAI erhält deshalb Dokumentgrenzen, IDs, Quellen, alle Nutzdatenkoeffizienten und einen separat adressierten Schlüsselindex. Die NPZ-Datei enthält den vollständigen Korpus als DCT-Koeffizienten samt Längenfeld, Prüfsumme und Decoderkonfiguration. Beim Laden werden die Texte invers rekonstruiert und der deterministische Index neu aufgebaut.

Mit insgesamt B UTF-8-Bytes und M Dokumenten hat der Laufzeitspeicher Größenordnung O(B + MD), die Suche O(MD), die Auswertung der Archivwelle O(B log B). D ist die Schlüsselgröße, standardmäßig 4096. Float64-Koeffizienten brauchen vor Dateikompression acht Bytes pro Mode; die komplexen Suchschlüssel brauchen 16MD Bytes. Dekodierte Texte und einzelne Nutzdatenspektren werden zusätzlich im RAM gehalten. Dies ist keine verlustfreie universelle Kompression und kein Kapazitätsvorteil gegenüber gewöhnlicher Textspeicherung.

Der Server berechnet standardmäßig zehnmal pro Sekunde alle Archiv- und Schlüsselmoden. Erst danach werden die räumlichen Werte für das Diagramm auf höchstens 256 Punkte reduziert. Diese Anzeige ist weder räumlich noch zeitlich ein vollständiger Messkanal: Bei bis zu 30 Hz genügt ihre Bildrate nicht für aliasfreie Rekonstruktion. Die eigentliche Rückrechnung nutzt die vollständigen Moden samt analytischer Phase. Der Server läuft bis zum Beenden des Prozesses; er richtet keinen automatischen Systemdienst ein.

## 5. Neue Aussagen durch explizite Relationen

`freqai.reasoning` untersucht zusätzlich neue, regelgebundene Aussagen. Ein explizites Tripel `Pudel is_a Hund` und ein zweites `Hund is_a Tier` bestimmen eine Adjazenzmatrix A. Mit \(H=FAF^*\) und \(v=Fe_{Pudel}\) berechnet der Operator die nächste Front im Frequenzraum. Inverse FFT, eine kontrollierte boolesche Projektion und Wiederholung ergeben die erreichbaren Begriffe. Ein gegen die Eingabetripel geprüfter Beweispfad erlaubt die Ausgabe `Pudel ist ein Tier.`.

Dieser Satz muss nicht wörtlich gespeichert sein. Die Bedeutung von `is_a`, ihre Transitivität und das Satzmuster sind jedoch explizit programmierte Regeln. Das ist ein begrenzter Schlussfolgerungsalgorithmus in einer Fourierbasis, keine aus beliebigem Text spontan entstehende Semantik. Die dichte Darstellung kostet O(V²) Speicher und Rechenarbeit pro Schritt bei V Begriffen und ist gegenüber klassischer Graphsuche kein bewiesener Geschwindigkeitsvorteil.

## 6. Was die Experimente entscheiden können

Verifiziert werden Zeichenrückrechnung, Energieerhaltung, Übereinstimmung der Interferenz mit direkter Ähnlichkeit, Treffer im festgelegten Minikorpus, Rauschgrenzen, Kollisionen und explizite relationale Ableitungen. Ein kleines synthetisches Benchmark-Ergebnis ist keine Messung allgemeiner Sprachkompetenz. Fragen nach unbekannten Fakten, Negation, Rollenwechsel und Umschreibungen ohne gemeinsame Wörter bleiben eigene Fehlerklassen. Die Messberichte enthalten diese Fehler und negative Versuche.
