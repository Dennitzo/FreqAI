# Gemeinsame Informationsberechnung für Gespräche und Wissen

Alle Eingaben verwenden in der neuen Implementierung `UnpairedWaveModel`. Der aktive Runtimecode enthält keine Paarstatistik, Antwortbaustein-Auswahl, Quellengewichtung für QA-Bestände oder Umschaltung auf den früheren Gesprächsdecoder. Die frühere Implementierung und ihre unverändert gesicherten Tests bleiben außerhalb des aktiven Pakets als historische Artefakte erhalten.

## Daten und Migration

Die neue Rezeptur enthält 20.000 unveränderte Wikipedia-v3-Einleitungen, 45 eigenständig verfasste deklarative Gesprächsinformationen und 48 separate lexikalische Klassifikationen. Die 48 Klassifikationen wurden von einem separaten Agenten ohne Einsicht in die Entwicklungs- oder Abschlussfragen als allgemeiner Wortbestand zusammengestellt. Die ursprünglichen 45 Fakten bleiben unverändert. Neue Texte haben keine Frage-/Antwortfelder. Der getrennte Bestandsaudit identifiziert 11.500 frühere Paare und 500 alte kategorisierte Sprachprior-Texte zur Archivierung; kein alter Antworttext wurde einfach in den neuen Bestand kopiert.

`MemoryStore.migrate_information_only()` archiviert die ausgewählten Originalzeilen innerhalb derselben SQLite-Transaktion und entfernt die `prompt`-Spalte der aktiven Tabelle. Die erhaltenen Einträge behalten ihre IDs und Sequenzen. Archive, Verläufe und Sitzungskontexte bleiben getrennt erhalten. Die Migration ist idempotent. Ein Fehler beim Archivieren rollt auch den Schemawechsel zurück. Ein neuer Live-Engine-Snapshot kann die entfernten Moden ohne Rücksetzen seiner Uhr übernehmen.

Die Anwendung besitzt nur noch Texteingabe plus Quelle. `add --prompt`, die Modusauswahl und das aktive Speichern von Paaren sind entfernt; ein früheres `mode=dialogue` wird zurückgewiesen. Ein gespeicherter Verlauf darf weiterhin Benutzernachricht und berechnete Ausgabe enthalten, wird aber nicht als Wissenskorpus oder Paartraining verwendet.

## Berechnungsmodell

Der gemeinsame Compiler erzeugt Prosa-Präfixfelder und deklarative Rollenfelder. Grammatik und Sprecherperspektive sind ausdrücklich programmierte sprachliche Regeln. Symboladressen tragen die Identität und Reihenfolge der Wörter; uniforme Rollenkoeffizienten allein kodieren keine unterschiedliche Bedeutung. Pro Ausgabeschritt werden alle kompatiblen Moden überlagert, mit Prompt und Kontext interferiert und über die inverse FFT in Symbolwahrscheinlichkeiten zurückgeführt.

Der Decoder konstruiert keinen vollständigen Antwortstring vor der Fourierberechnung. Sein Grammatikablauf kann trotzdem Wortfolgen stark vorgeben, beispielsweise den Bau einer Begrüßungsfrage. Das ist explizites sprachliches Vorwissen und kein durch Fouriertransformation entstandenes Sprachverständnis. [Vollständige Herleitung](../../docs/unpaired-grammar.md).

Neue Namen und Zustandswörter werden über ein festes UTF-8-Bytealphabet als zeitweilige Rollen in der Sitzung abgebildet. Sie verändern weder das globale Vokabular noch die Korpuskoeffizienten. Das Symbolbudget einschließlich Bytes und Satzzeichen wird getrennt von sichtbaren Wörtern ausgewiesen.

## Prüfstand

Die abschließende vollständige technische Prüfung ergab **321 bestandene Tests in 14,58 Sekunden**, einschließlich der Prüfung alternativer Frage-/Antwortfelder vor ihrer Projektion auf das interne Dokumentformat. Die Anzahl ist nicht direkt mit den früheren 536 Tests vergleichbar: Veraltete Tests der entfernten Paararchitektur wurden zusammen mit ihrem Code historisch gesichert; ihre nun ungültigen Produktverträge werden nicht mehr als aktuelle Funktionen getestet. Weiterhin relevante Transaktions-, Speicher-, Numerik-, API- und Paginierungsprüfungen bleiben aktiv und wurden auf den neuen Vertrag angepasst. [Testprotokoll](technical_tests.json).

Die [numerische Prüfung](numerical_audit.json) kontrolliert 255 Präfix-/Zeit-Zustände. FFT und direkte Referenz ergeben nach der dokumentierten Rundung auf zwölf Nachkommastellen dieselben Wahrscheinlichkeiten. Eine Phasenänderung eines gespeicherten Rollenfelds verändert das nächste Inhaltswort (Totalvariation 0,8). Nullgesetzte Datenrollen verhindern die Ausgabe auch bei intakten Prosa-Übergangsfeldern. Weitere Kontrollen prüfen Promptnullung, Gegenfakten, reproduzierbaren Neuaufbau und Sitzungsisolierung. Das prüft die kausale Beteiligung der Daten- und Promptfelder; es beweist keine allgemeine Sprachfähigkeit.

Der unabhängige [vollständige Speicheraudit](storage_audit.json) prüft alle 20.093 Texte und ihre Metadaten an zwei Zeitpunkten: **80.372 bytegenaue Dekodierungen**. Alle 92.388.505 Paket- und Diagnoseschlüsselmoden wurden berücksichtigt; alle Werte waren endlich. Die relative Normabweichung betrug 1,50 × 10⁻¹⁶. Der Speicheraufbau dauerte 38,81 Sekunden, ein vollständiger instrumentierter Frame im Median 0,82 Sekunden. Diese Zeiten betreffen die Dokumentwelle, nicht den Aufbau des Generationscompilers.

## Antwortqualität nach dem Freeze

Die Numerik ist getrennt von einer neuen Gesprächsevaluation mit 20 Entwicklungs- und 20 vorab zurückgehaltenen Abschlussrunden. Je zehn Sitzungen enthalten zwei Nachrichten. Die früheren 40 Informationsfragen werden ausdrücklich als bekannte Wissensregression verwendet. Bewertet wurden die tatsächlichen Antworten durch einen unabhängigen AI-Agenten; dies ist keine menschliche Nutzerstudie.

| Gesprächsprüfung | Alte Baseline, Development | Neuer Decoder, Development | Alte Baseline, neuer Holdout | Neuer Decoder, neuer Holdout |
|---|---:|---:|---:|---:|
| Vollständig passend | 4/20 | 10/20 | 7/20 | **3/20** |
| Zusätzlich teilweise passend | 3/20 | 9/20 | 3/20 | 8/20 |
| Leer | 0/20 | 1/20 | 0/20 | **9/20** |
| Vollständig passende Zweirundensitzungen | 0/10 | 4/10 | 2/10 | 0/10 |

Der Entwicklungsgewinn überträgt sich nicht auf die unbekannten Gespräche. Viele Antworten wiederholen oder paraphrasieren nur Benutzeraussagen. Satzbau, sprachliche Abdeckung und inhaltliche Anschlussantworten bleiben begrenzt. **Die Umstellung auf Informationen ist implementiert; zuverlässige allgemeine Gesprächsfähigkeit ist nicht erreicht.** Die neue Version ist im unbekannten Gesprächstest schwächer als die frühere Paarbaseline. Nach Öffnen dieses Holdouts wurde der Decoder nicht weiter angepasst.

Alle 40 bekannten Wissensantworten bleiben text- und tokengenau erhalten: vier korrekte Antworten und 36 Leerantworten. Das erhält den bisherigen engen Informationsfunktionsumfang, erweitert ihn aber nicht allgemein. Im bekannten Zufallsfaktentest gelingt die Rekonstruktion aller zwölf Vorwärts- und zwölf Rückwärtszuordnungen sowie zwölf vertauschter Werte (36/36). Vier falsche Wertbehauptungen werden weiterhin nicht korrigiert; vier unbekannte Namen bleiben unbeantwortet. Diese bekannten Prüfungen beweisen weder neue Faktenableitung noch allgemeine Wissensgeneralisation.

[Vollständige Evaluation mit Antworten, Rubriken, Zeitmessungen und Daten-/Codehashes](evaluation/report.md). Der finale Kernhash beginnt mit `f99d23332e3a`. Nach dem Freeze wurde ausschließlich die Eingabevalidierung in CLI und HTTP ergänzt; Generationskern, Runtime und die drei Informationsdateien blieben unverändert.

## Tatsächlicher Serverbetrieb und Live-Ergänzung

Der produktive Speicher wurde vor der Umstellung mit SQLite-Backup gesichert: `memory/backups/before-final-unpaired-migration-20260906-142535.sqlite3`. Die Migration archivierte genau 12.000 aktive Alteinträge und erhielt 20.000 Wikipedia-Texte unverändert. Der Bestand enthält anschließend insgesamt 12.048 archivierte Originaltexte. Sämtliche 89 vor der Migration gespeicherten Verlaufszeilen blieben bytegenau erhalten; bei den 19 damaligen Sitzungen ging keine Revision verloren. Auch der vorher gespeicherte Informationskontext ließ sich nach dem Neustart weiterverwenden: Die Nachfrage nach der Einheit der Frequenz erzeugte korrekt Hertz.

Danach wurden im **selben Serverprozess, PID 3916**, erst 45 Gesprächsinformationen und anschließend 48 lexikalische Fakten importiert. Der aktive Bestand enthält **20.093 reine Informationstexte bei Revision 9**, ohne `prompt`-Spalte. Die archivierten Originale und alle beibehaltenen Wikipedia-Texte wurden per Hash geprüft. Ein tatsächlicher HTTP-Import mit `question`/`answer` wurde mit Status 400 abgewiesen, ohne Datenbankrevision oder Inhalt zu ändern. Uhr und Frames liefen während der Ergänzung weiter; drei vollständige Frames hatten dieselbe Energie.

Der kalte Compileraufbau aus den 20.000 Wikipedia-Texten dauerte bei der ersten Anfrage rund 81,87 Sekunden. Nach Ergänzung der 93 Texte wurde der Compiler ohne Prozessneustart in rund 79,93 Sekunden neu aufgebaut. Die danach gemessenen kurzen Antworten dauerten 0,078 bis 0,159 Sekunden. Eine Ergänzung erfordert weiterhin einen vollständigen Aufbau des abgeleiteten Generationscompilers; sie wird zur Laufzeit übernommen, ist aber keine sofortige inkrementelle Berechnung sämtlicher Compilerfelder.

Die bekannten tatsächlichen API-Ausgaben lauten unter anderem:

| Eingabe | Berechnete Ausgabe |
|---|---|
| Hallo | Hallo! Wie geht es dir? |
| Mir geht es gut, wie geht es dir denn? | Dir geht es gut. Ich habe kein menschliches Befinden. |
| Wie heißt du? | Ich heiße FreqAI. |
| Was ist eine Frequenz? | Die Frequenz ist der Kehrwert der Periodendauer. |
| Und welche Einheit hat sie? | Das Hertz (Einheitenzeichen: Hz) ist die SI-Einheit der Frequenz. |

Alle fünf Antworten verwenden denselben spektralen Informationsdecoder, ohne Paartexte, Optimierungsschritte oder Trefferlisten. Sie sind bekannte Funktionsproben. Ihre erfolgreiche Ausgabe ist nicht mit unbekannter Gesprächsqualität gleichzusetzen. [Server-/Erhaltungsprüfung, Messwerte und aktive Codehashes](deployment/active-report.json), [Migrationsprotokoll](deployment/migration.json).

Der abschließende tatsächliche Edge-Browserlauf bestand ebenfalls. Eine eigene neue Sitzung erhielt fünf berechnete Antworten und behielt sie nach dem Neuladen. Die alten Auswahlfelder für Verfahren und Gesprächsanlass sind entfernt; auch eine alte gespeicherte `dialogue`-Einstellung wird ignoriert. Alle Dokumentseiten bis Seite 402 mit den letzten 43 Einträgen wurden korrekt angezeigt. Das Kontextfeld bewegte sich bei unveränderter Gesprächsrevision weiter, bei stabiler Norm von rund 1. Es traten keine JavaScript- oder Konsolenfehler auf. Die fünf warmen Anfragen dauerten im Browser etwa 0,19 bis 0,31 Sekunden. [Browserprotokoll](deployment/browser.json), [geprüfte Bildschirmaufnahme](deployment/browser.png), [Gesprächsausschnitt](deployment/browser.conversation.png).

Der geprüfte Server läuft unter `http://127.0.0.1:8765`. Die neue Oberfläche wird nach einem Neuladen sichtbar. Für neue Informationen stehen weiterhin die Oberfläche und `freqai import`/`add` zur Verfügung; sie benötigen keinen Prozessneustart.
