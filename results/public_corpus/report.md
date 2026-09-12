# Erweiterung um öffentliche deutsche Daten

Die Erweiterung wurde gegen den tatsächlichen Ausgangsbestand von **621 aktiven Texten** geprüft. Dieser Bestand enthält auch den nach dem vorigen Versuch ergänzten Nutzereintrag. Keine alten Texte, Archive oder Gespräche werden durch den Import ersetzt.

## Daten und Herkunft

Der finale Importkandidat v3 enthält **11.379 zusätzliche Frage-Antwort-Paare**:

- 533 geprüfte deutsche OpenAssistant-Antworten auf 342 verschiedene selbstständige Gesprächsanlässe.
- 10.846 GermanQuAD-Trainfragen mit vollständigen Antwortsätzen aus 2.526 Wikipedia-Passagen; 9.292 unterschiedliche Antworttexte.

Die gemeinsame Datenmenge nach vollständiger Übernahme beträgt 12.000 aktive Einträge. Fragen, Antworttexte und Quellpassagen sind unterschiedliche Zähleinheiten. Mehrere Antworten auf denselben Anlass und mehrere Fragen zur selben Passage sind keine unabhängigen Beobachtungen.

[Quellen, Lizenzen und reproduzierbarer Import](../../docs/public-corpus-sources.md) dokumentieren die elf lokalen SearXNG-Anfragen, gepinnte Downloads, SHA-256 und Filter. Der Korpus-SHA lautet `832754653cdd1dcd0c2c0cb7043d3edae71f8e28558b5ed7a145362c0b73d9fa`. Die offiziellen Testfragen wurden nicht importiert. Weitere 500 OASST-Folgeantworten bleiben separat, weil ihr fremder Rollenverlauf bisher nicht zuverlässig codiert würde.

## Warum ein einfacher Import nicht ausreichte

Die unveränderte Konditionierung führte trotz korrekter Numerik zu schlechteren Gesprächsantworten. Beispielsweise endet eine Fachfrage über Glasbearbeitung mit „danke“. Der bisherige Parser ordnete damit die komplette Fachantwort auch dem Dankeskanal zu. Eine Frage für Schüler der „10. Klasse“ aktivierte fälschlich positive Stimmung. Das ist ein Fehler bei der Zuordnung von Daten zu Sprachmerkmalen, kein Fourierfehler.

Die [kausale Diagnose](research/semantic_contamination.md) änderte die Daten nicht: Ohne diese automatisch vergebenen Dialoglabels verschwanden die falschen Glas-/Markov-Kopplungen. Der produktive Ansatz entscheidet deshalb **pro Paar und Quelle**, nicht anhand des Prompttexts allein. Öffentliche Paare liefern lexikalische Eingabekopplungen; die eigene Dialogsammlung und ausdrücklich annotierten Sprachtexte behalten ihre semantischen Kopplungen. Der Merkmalscache enthält sowohl Prompt als auch Konditionierungsmodus, damit identische Fragen verschiedener Quellen nicht verwechselt werden.

Zusätzlich werden neue Quellen beim Zählen mit festen Gewichten berücksichtigt: OpenAssistant `0.1`, GermanQuAD `0.05`, eigene Paare weiterhin `1.0`. Alle importierten Texte und Wörter bleiben vorhanden. Die Gewichte und Quellpräfixe stehen in [generative_categories.json](../../memory/language/generative_categories.json). Sie wählen keine vollständigen Antworten aus und begründen keine Faktenkonfidenz.

## Antwortqualität und getrennte Prüfungen

Der [vollständige Evaluationsbericht](evaluation/report.md) enthält die endgültigen Entwicklungs- und Abschlusszahlen, sämtliche qualitativen Einzelfallurteile, Konfigurationen und Hashes. Entwicklung und Holdout umfassen jeweils 20 neue Gesprächsrunden. Alle 40 normalisierten Prompts haben **keine exakte Überschneidung** mit den öffentlichen Prompt- oder Antworttexten. Die qualitative Bewertung stammt von einer separaten KI-Agentenprüfung, nicht aus einer menschlichen Nutzerstudie.

| Prüfung | 621 bisherige Texte | 12.000 Texte, korrigierter Compiler |
| --- | ---: | ---: |
| Inhaltlich passende Entwicklungsantworten | 11/20 | 12/20 |
| Inhaltlich passende zurückgehaltene Chatantworten | 13/20 | 12/20 |
| Vollständig passende Zweiergespräche im Holdout | 3/10 | 3/10 |
| Korrekte Antworten auf bekannte Importfragen | 0/20 | 8/20 |
| Korrekte Antworten auf offizielle Testfragen ohne Referenzkontext | 0/30 | 0/30 |

Der Ausbau belegt **keine Verbesserung der allgemeinen Gesprächsqualität und keinen Transfer auf die geprüften unbekannten Faktenfragen**. Eine weitere Holdoutantwort ist nur teilweise passend und wird nicht als voller Treffer gezählt. Bei Familientreffen und Abschied entstehen neue Themenabweichungen; die Kaffee-Negation bleibt falsch. Der Wunsch nach Ruhe und die Frage nach eigenen Gefühlen werden in dieser Stichprobe besser beantwortet. Die Abschlusszahlen wurden vor der anschließenden, rein numerisch äquivalenten Geschwindigkeitsoptimierung ermittelt; danach findet ausschließlich eine Ausgaberegression statt, kein semantisches Tuning.

Die Entwicklung vergleicht den bestehenden Generator, dessen mathematisch äquivalente kompakte Speicherung, zunächst +500 und +2.000 öffentliche Einträge, den großen ungewichteten Import sowie die Gewichtung und den korrigierten Compiler. Die ersten Teilimporte sind deterministische Dateipräfixe mit unterschiedlicher Quellenmischung, keine repräsentativen Zufallsstichproben des Gesamtkorpus. Neue ganze Texte, neue Einzelsätze, Grammatik und passende Bedeutung werden getrennt gezählt.

Eine zusätzliche Faktenprüfung verwendet 30 festgelegte Fragen aus dem offiziellen GermanQuAD-Testsplit ohne mitgelieferten Referenzkontext. Das ist **kein regulärer GermanQuAD-Leseverständnisbenchmark**. Im importierten Korpus findet sich nur bei acht Fragen überhaupt eine Referenzantwort-Zeichenfolge, bei vier zusätzlich derselbe Artikeltitel. Selbst dieser Beleg garantiert noch keine Beantwortbarkeit. Eine weitere Prüfung auf 20 bereits importierten Originalfragen untersucht die Nutzung vorhandener Daten und ist ausdrücklich **in-sample**, kein Generalisierungsnachweis.

## Numerik und Skalierung

Die [vollständige technische Suite](deployment/technical-tests.json) besteht mit **475 Tests**, darunter 14 zusätzliche Vergleiche für den größeren FFT-Träger. Sie prüft auch Importfilter, Quellengewichte, kompakte Felder, Live-Synchronisation, Sitzungen und paginierte Datenansicht.

Die kompakten Felder speichern für beobachtete Tokenadressen `I` die komplexen Koeffizienten `C_I = F_|I|(a[I])`. Ihre globale Darstellung ist `F_V(E_I F_|I|^-1(C_I))`. Der Generator rekonstruiert die tatsächlichen komplexen Moden und mischt sie in einen gemeinsamen Träger. Kein beobachteter Koeffizient wird zur Kompression beschnitten; es gibt keine gelernte Projektion. Die dichte Darstellung bleibt als Referenz verfügbar, die Wortnachbarschaften liegen als dünn besetzte Matrix vor.

- Mit denselben bisherigen 621 Texten sind alle 20 geprüften Antworten exakt gleich. Der gemessene Modellaufbau sank von 1,83 auf 0,57 Sekunden und der Prozesspeak von 361 auf 73 MiB.
- [144 zusätzliche Kontrollen mit echtem externem Wortschatz](import_numerical_reference.json) vergleichen kompakt/dicht und Fourier/direkt bei unterschiedlichen Präfixen, Zeiten und relativen Phasen. Nach der dokumentierten Rundung gibt es keine Abweichung; zwölf vollständige Beamantworten sind identisch.
- [Alle neuen Text- und Metadatenpakete](public_codec_audit.json) wurden bei zwei Schwingungszeiten einschließlich 24 Stunden zurückgerechnet: **45.516/45.516** exakte Rückrechnungen. Das prüft die verlustfreie Speicherung sämtlicher neuer Bytes und ausdrücklich nicht die Qualität frei erzeugter Antworten.
- Der [synthetische Skalierungsversuch](../corpus_expansion/scaling/storage_benchmark.json) enthält 10.000 Paare und 20.006 Tokens. Er misst 354,6 MiB Prozesspeak und 7,19 MiB komplexe Koeffizienten gegenüber rechnerisch 87,36 GiB für eine vollständig dichte Koeffizientenbank. Diese Größen beziehen sich auf verschiedene Speicheranteile; der Versuch bewertet keine Gesprächsqualität.
- Die [Dokumentwellenmessung mit allen 12.000 Texten](deployment/final_wave_benchmark.json) berechnet sämtliche 5.964.644 Text- und Metadatenmoden. Gleiche Paketlängen werden gemeinsam transformiert; nur die für die Anzeige ausgewählten Ortswerte müssen gespeichert bleiben. Die vollständige Rechnung dauerte im gemessenen Lauf im Median rund 0,45 Sekunden pro Bild. Der 10-Hz-Zielwert des Workers wird bei dieser Datenmenge somit nicht erreicht.
- Der [isolierte Laufzeitvergleich](performance/operator_padding_report.json) füllt den gemeinsamen Token-Träger vor der FFT mit Nullamplituden von 44.614 auf 44.800 Moden auf. Das verbessert die Faktorisierung der FFT-Länge; gespeicherte komplexe Felder, bekannte Tokenfrequenzen und Sprachstatistik bleiben erhalten. Die ersten 44.614 invers transformierten Moden bestimmen weiterhin die Tokenwahrscheinlichkeiten. Drei vollständige Beamantworten waren wortgleich und im Median 4,47-mal schneller (4,19 → 0,94; 6,52 → 1,44; 5,81 → 1,33 Sekunden). 144 komplexe Kontrollzustände und zwölf Zustände des großen Korpus stimmen nach der bestehenden Rundung mit dem bisherigen Operator und der unveränderten direkten Referenz überein. Die Kontextanzeige verwendet ebenfalls diesen gemeinsamen größeren Träger und nennt bekannte Moden und Trägergröße getrennt.

Korrekte Transformationsrechnungen, mehr Wortschatz oder eine lebendige Anzeige garantieren keine sinnvolle oder wahre Antwort. Das Modell zählt Übergänge und konditioniert sie auf Eingabemerkmale. Es nutzt keine Gradientenoptimierung und kein vortrainiertes Sprachmodell, passt seine Statistik aber an die Daten an.

Die anschließende [Ausgaberegression des integrierten Operators](evaluation/fast_carrier_output_regression.json) ist abgeschlossen: **90/90 sichtbare Antworten bytegleich**, alle 40 geprüften Gesprächskontexte gleich. Bei 60 Fällen sind die ursprünglichen rohen Tokenfolgen gespeichert und identisch; bei 30 Faktenfällen erfolgt der Tokenvergleich gegen ausdrücklich rekonstruierte Referenzen aus dem gespeicherten Text. Alle 60 vorhandenen Rohstream-Rückrechnungen und 44.613 Alphabet-Rückrechnungen stimmen überein. Die gemessene mediane Antwortzeit über diese 90 Fälle sank von 9,70 auf 1,75 Sekunden. Die früheren Läufe waren teilweise parallel belastet, weshalb dieser Zeitvergleich kein kontrollierter Beschleunigungsfaktor ist. Es ist eine Regression bereits bekannter Ausgaben, kein weiterer unbekannter Holdout.

## Betrieb und Reproduktion

Die aktive Quelle bleibt `memory/memory.sqlite3`. Rohdaten, Importdateien, Evaluationsdaten und Sicherungen liegen ebenfalls unter `memory/`, werden aber nicht automatisch als aktive Antworten indexiert. Der Import ist additiv. Eine Codeaktualisierung benötigt einen Versionsneustart; spätere Datenimporte werden ohne weiteren Serverneustart übernommen. Der erste Generatoraufbau nach einer großen Ergänzung benötigt deutlich länger als eine Antwort bei bereits aufgebautem Modell.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[experiments,corpora]"
.\.venv\Scripts\python.exe experiments\import_public_corpus.py --offline
.\.venv\Scripts\python.exe -m freqai import memory\imports\public_dialogue_expansion.jsonl
.\.venv\Scripts\python.exe -m pytest -q
```

Die Speicheransicht lädt 50 Einträge pro Seite und prüft die Revision beim Weiterblättern. Bei einer Ergänzung beginnt sie mit der neuen Revision wieder auf Seite eins. Damit müssen nicht alle 12.000 Tabellenzeilen gleichzeitig im Browser dargestellt werden.

Aktivierungsbelege, Datensicherung, erhaltene Text- und Archivhashes sowie Browserprüfung werden unter `results/public_corpus/deployment/` protokolliert. Der Import ergänzte atomar 11.379 Einträge in Revision 5 bei unveränderter Serverkennung 34084. Alle bisherigen 621 Dokumente und 48 Archivtexte stimmen mit ihren gesicherten Hashes überein; der gespeicherte geprüfte Sitzungskontext bleibt erhalten. Die Antwortqualität bleibt durch den getrennten Evaluationsbericht belegt.

Der anschließende [Codeversionsneustart für den schnelleren Operator](deployment/performance-restart.json) verwendet Prozess 29008 und erhielt ebenfalls Texte, Archive und den gespeicherten Testkontext. Die [vollständige Erhaltungsprüfung](deployment/all-saved-state-preservation.json) vergleicht außerdem sämtliche 56 ursprünglichen Verlaufseinträge und zehn ursprünglichen Sitzungszustände gegen die Sicherung: alle unverändert, SQLite-Integritätsprüfung `ok`. Sieben neue Nachrichten stammen aus den ausdrücklich dafür angelegten Prüfsitzungen; keine davon wird als neues Wissen in den Korpus übernommen.

Die [abschließende Prüfung im echten Edge-Browser](deployment/browser.json) besteht auf dieser Serverversion: 12.000 Texte, maximal 50 Tabellenzeilen, Vor-/Zurückblättern bis Seite 240, sichtbare Quellen, drei gespeicherte Antworten nach Reload und keine JavaScript-Fehler. Alle geprüften Decoderschritte verwenden 44.800 Trägermoden. Bei derselben Gesprächsrevision 3 verändert sich die Kontextwelle mit der Zeit, während ihre Energie exakt gleich bleibt. Der erste Browserprüfer verglich versehentlich Poll-Antworten verschiedener Gesprächsrevisionen; die korrigierte GET-Prüfung wartet auf dieselbe Revision und benötigt keine weiteren Chatnachrichten. Das Produkt wurde dafür nicht verändert. Die Antwort auf „Mir geht es gut“ enthält weiterhin die unpräzise Formulierung „wenn dir etwas gut gelingt“ und wird nicht als Beleg besserer Gesprächsqualität ausgegeben.

Die Anzeige „Aktive Moden“ nennt 55.116.644: Sie addiert zu den 5.964.644 Text-/Metadatenmoden auch 49.152.000 Schlüsselkoordinaten des weiter verfügbaren Regeldialog-Abrufs. Diese Anzeigegröße ist weder die Wortschatzgröße noch die 44.800 Moden des generativen Token-Trägers.
