# Recherche: Text, Fourierkoeffizienten und assoziativer Wellenspeicher

Recherchedatum: **6. September 2026**. Die Suchanfragen liefen ausschließlich über die lokale SearXNG-Schnittstelle `POST http://127.0.0.1:8080/v1/research/web`. Der erste Versuch schlug mit einer Verbindungsfehlermeldung fehl. Nach erneutem Prüfen war die bereits laufende Schnittstelle erreichbar; die anschließend gespeicherten Antworten bestätigen jeweils `provider: searxng` und `isFallback: false`. Direkte Abrufe bekannter Primärquellen ergänzten die Suche. Es wurde kein anderer Suchdienst als Ersatz verwendet. Rohantworten und Abrufnachweise liegen in [searxng_queries.json](../results/research/searxng_queries.json) und [sources.json](../results/research/sources.json).

## Was die Literatur bereits zeigt

| Primärquelle | Nachgewiesener Inhalt | Bedeutung für dieses Projekt |
| --- | --- | --- |
| [Plate (1995), Holographic Reduced Representations](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf), IEEE Transactions on Neural Networks 6(3), 623–641, DOI 10.1109/72.377968 | Bindet symbolische Vektoren durch zyklische Faltung, überlagert Speicherinhalte und ruft sie durch Korrelation ab. Abschnitt VIII erklärt FFT, unitäre Bindeschlüssel und die Rauschempfindlichkeit der Inversion. | Fourier-basierte symbolische Assoziation und Rückgewinnung sind etablierte Vorarbeiten. Mehrfachspeicherung verursacht Übersprechen; zuverlässige Symbolerkennung braucht einen bekannten Symbolvorrat. |
| [Jones & Mewhort (2007), Representing Word Meaning and Order Information in a Composite Holographic Lexicon](https://cseweb.ucsd.edu/~gary/PAPER-SUGGESTIONS/jones-mewhort-psych-rev-2007.pdf), Psychological Review 114(1), 1–37 | Das BEAGLE-Modell verwendet Faltung und Superposition, um Bedeutungs- und Reihenfolgeinformationen aus einem Sprachkorpus aufzubauen. | Schon vor diesem Projekt gab es holographische Sprachrepräsentationen. Das ist unüberwachtes Lernen aus Sprachdaten; die Bezeichnung „trainingsfrei“ würde diese Lernkomponente verdecken. |
| [Kanerva (2009), Hyperdimensional Computing](https://link.springer.com/article/10.1007/s12559-009-9009-8), Cognitive Computation 1, 139–159 | Beschreibt hochdimensionale Zufallsvektoren und algebraische Operationen für verteilte symbolische Repräsentationen. | Liefert den Kontext für deterministisch erzeugte Symbolschlüssel, Bindung, Superposition und Ähnlichkeitsvergleich. Gelesen wurden Abstract und Literaturverzeichnis; der Volltext hinter der Bezahlschranke wurde nicht als gelesen behandelt. |
| [Frady, Kleyko & Sommer (2018), A theory of sequence indexing and working memory in recurrent neural networks](https://arxiv.org/abs/1803.00412), [Volltext](https://arxiv.org/pdf/1803.00412) | Analysiert Sequenzspeicher, Rückrufgenauigkeit und endliche Kapazität. Der Volltext behandelt Fourier Holographic Reduced Representations mit komplexen Einheitsphasen und enthält numerische Verfahren. | Zeigt eine konkrete Vorarbeit zu Symbolfolgen in phasenbasierten Speichern. Zufällige Phasen beseitigen weder Übersprechen noch Kapazitätsgrenzen. |
| [Jiménez et al. (2023), Learning algorithms for oscillatory neural networks as associative memory for pattern recognition](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1257611/full) | Untersucht assoziative Speicher aus gekoppelten Oszillatoren und Lernregeln für ihre Kopplungen. | Physikalisch motivierte Oszillator-Speicher existieren. Diese konkrete Arbeit verwendet Lernen und demonstriert Mustererkennung, keine allgemeine trainingsfreie Textantwortmaschine. |
| [Saklakov (2025), Phase-Coded Memory and Morphological Resonance](https://arxiv.org/abs/2511.11848) | Beschreibt eine sehr ähnliche Architekturidee aus semantischen Wellenformen, holographischem Feldspeicher und interferenzgeleiteter Ausgabe. Der Eintrag bezeichnet die Arbeit ausdrücklich als konzeptionelles Whitepaper. | Besonders relevanter Vorläufer der Idee. Die weitreichenden Aussagen des Abstracts sind ohne eigene reproduzierte Messungen kein Nachweis für unbegrenzten Kontext oder allgemeine Sprachkompetenz. |
| [Shannon (1948), A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) | Trennt die technische Übertragung beziehungsweise Rekonstruktion einer Nachricht von ihrer Bedeutung; behandelt Codierung, Kanal und Kapazität. | Die Rückgewinnung korrekt codierter Textzeichen ist ein anderes Problem als das Ermitteln einer inhaltlich richtigen Antwort. |
| [NumPy, Discrete Fourier Transform](https://numpy.org/doc/stable/reference/routines.fft.html), offizielle Dokumentation | Definiert Vorwärts- und Rücktransformation, komplexe Phase, Normierung sowie die Symmetrie reeller Signale. | Referenz für die numerische Implementierung. Bei `rfft` müssen ursprüngliche Länge und die zum verwendeten Verfahren passende Normierung erhalten bleiben. |

Die Behauptung **„Es gibt bisher kein System, das Texte mit Schwingungen verbindet“ ist in dieser allgemeinen Form widerlegt**. Die Quellen belegen verwandte Symbol-, Sprach- und Oszillatorspeicher. Sie belegen weder eine naturgesetzliche, eindeutige Frequenzzuordnung für Wortbedeutungen noch eine universelle Antwortmaschine aus bloßer Fourier-Superposition. Eine vollständige Neuheits- oder Patentrecherche wurde nicht durchgeführt.

## Konkrete mathematische Lösung für die Rückwandlung

Die folgenden Herleitungen sind eigene mathematische Ableitungen für das Projekt. Sie sind keine Behauptung eines neu entdeckten Naturgesetzes.

Ein Text wird als UTF-8-Bytefolge `b[0], ..., b[N-1]` mit Werten von 0 bis 255 codiert. Nach optionaler dokumentierter Zentrierung liefert die diskrete Fouriertransformation

\[
X_k = \sum_{n=0}^{N-1} b_n\exp(-2\pi i kn/N).
\]

Die Rückwandlung lautet

\[
\hat b_n = \operatorname{round}\left(\operatorname{Re}\frac{1}{N}
\sum_{k=0}^{N-1}X_k\exp(2\pi i kn/N)\right).
\]

Liegt der numerische Fehler vor dem Runden für jedes Byte unter 0,5, erhält man die Bytefolge exakt zurück. Danach folgt strikte UTF-8-Decodierung. Ein Längenfeld bewahrt die Nachrichtengrenze; eine Prüfsumme erkennt Beschädigungen. Ein Decoder darf ungültige Bytes nicht stillschweigend ersetzen und anschließend Erfolg melden. Für leere Texte braucht es einen ausdrücklich definierten Sonderfall. Magnituden allein reichen zur Rekonstruktion im Allgemeinen nicht aus: Die Phasen gehören zu den Daten.

Für eine reale Schwingungsdarstellung kann man die Fourierkoeffizienten als Sinus- und Kosinusamplituden einer endlichen periodischen Wellenform interpretieren. Für eine stehende Welle mit festen Endpunkten eignet sich alternativ

\[
u(x,t)=\sum_{k=1}^{K}\left[a_k\cos(\omega_k t)+b_k\sin(\omega_k t)\right]
\sin(k\pi x/L),\qquad \omega_k=ck\pi/L.
\]

Diese Funktion löst die ideale eindimensionale Wellengleichung `u_tt = c² u_xx`. Ein digitaler Algorithmus kann die Koeffizienten beziehungsweise beide Zustandsgrößen eines Modus speichern und die Entwicklung mit analytischen Rotationen fortsetzen. Eine einzelne Auslenkung zu einem beliebigen Zeitpunkt genügt nicht: Bei einem zeitlichen Nulldurchgang kann sie trotz gespeicherter Information null sein. Auslenkung und Geschwindigkeit oder die äquivalente komplexe Phase erhalten den Zustand. Die physikalischen Größen `c`, `L`, Amplitudeneinheit und Frequenzmaßstab sind Modellentscheidungen.

## Wie ein Prompt auf den Speicher wirken kann

Für einen Speicherzustand `M` und eine Promptcodierung `Q` enthält die Interferenzenergie den Term

\[
|M+Q|^2-|M|^2-|Q|^2=2\operatorname{Re}(M\overline Q).
\]

Summiert über die Frequenzkanäle ergibt dies ein skalares Ähnlichkeitsmaß. Eine Antwort benötigt zusätzlich eine gespeicherte Zuordnung zwischen dem adressierbaren Inhalt und seiner Antwort oder Quellpassage. Beispielsweise kann eine frequenzbasierte Schlüssel-Wert-Speicherung die Relation `Frage/Thema → Textpassage` erhalten. Der Prompt gewichtet passende Kanäle; aus den ausgewählten Antwortkoeffizienten rekonstruiert die inverse Transformation den Text.

Eine reine Summe ohne Adressierung verliert Information. Ein einfaches Gegenbeispiel sind Bytevektoren für `AB` und `BA`: Ihre Summe ist `[131, 131]`. Genau dieselbe Summe entsteht aus `AA` und `BB`. Wegen Linearität sind auch die Summen ihrer Fouriertransformationen gleich. Kein Decoder kann aus dieser Summe entscheiden, welches Dokumentpaar ursprünglich vorlag. Getrennte Adresskanäle, gespeicherte Rollen oder zusätzliche Bindestruktur sind daher notwendig. Sie kosten Speicher.

Eine holographische Variante speichert `M = Σ K_j ⊙ V_j`, wobei `K_j` komplexe Einheitsphasen und `V_j` Antwortvektoren sind. Für eine isolierte Zuordnung ergibt `conj(K_j) ⊙ (K_j ⊙ V_j) = V_j`. Bei mehreren Zuordnungen bleiben Störterme anderer Schlüssel bestehen. Größere Dimensionen und ein bekanntes Antwortalphabet können deren Einfluss verringern; sie garantieren keine verlustfreie unbegrenzte Kompression. Eine Alternative mit orthogonalen Adresskanälen erhält die vollständigen Daten, benötigt aber mit dem Korpus wachsende Kapazität.

## Prüfkriterien und ehrliche Reichweite

Eine belastbare numerische Lösung muss mehrere getrennte Ergebnisse zeigen:

1. **Codierung:** Exakte Rückgewinnung verschiedener UTF-8-Texte, einschließlich Umlauten, Emoji, leerem Text und längeren Nachrichten.
2. **Dynamik:** Erhaltung der gespeicherten Information und Energie unter fortgesetzter Schwingung; Prüfung mehrerer Zeitpunkte einschließlich Nulldurchgängen.
3. **Assoziation:** Rückruf bekannter Textpassagen durch neue Formulierungen und transparente Fehlerraten auf vorher festgelegten, nicht zur Parameterwahl verwendeten Fragen.
4. **Grenzen:** Negativbeispiele ohne Wortüberschneidung, falsche Prämissen, vertauschte Rollen, unbekannte Fakten, steigende Speicherauslastung sowie Rauschen. Keine Umdefinition fehlgeschlagener Tests als Erfolg.
5. **Vergleich:** Vergleich gegen einen einfachen lexikalischen Suchalgorithmus. Wenn Fouriertransformation nur denselben Skalarproduktscore in anderer Basis berechnet, ist das eine Darstellungseigenschaft und kein gemessener Bedeutungsgewinn.

„Ohne Training“ lässt sich hier präzise als **keine Gradientenoptimierung, keine vortrainierten Gewichte und keine lernende Parameteranpassung** definieren. Das Einlesen, Tokenisieren, Zählen und Speichern eines Korpus bleibt notwendig. Eine Parametersuche über Testdaten wäre ebenfalls eine Form der Anpassung und muss getrennt von der abschließenden Auswertung ausgewiesen werden.

Ein solcher Algorithmus kann eine funktionierende trainingsfreie, frequenzbasierte **Speicherung und quellengebundene Antwortrekonstruktion** liefern. Ob darüber hinaus neue semantische Schlussfolgerungen möglich sind, muss an eigenen Aufgaben geprüft werden. Das bloße dauerhafte Weiterrechnen einer unveränderten linearen Welle erzeugt keine zusätzlichen Beobachtungen, Wissensrelationen oder Beweise. Allgemeine Sprachkompetenz und eine natürliche Frequenz von Wortbedeutung sind damit nicht nachgewiesen.
