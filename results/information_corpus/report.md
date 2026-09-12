# Informationskorpus und direkt berechnete Schwingungskoeffizienten

Der neue Informationspfad verarbeitet deklarative Texte ohne vorgegebene Fragen. Komplexe Übergangsfelder entstehen deterministisch aus gezählten Wortübergängen. Der Decoder berechnet Folgetokens durch tatsächliche Fourieroperationen und Interferenz; er enthält keine Rangliste vollständiger Antwortkandidaten. Die Originaltexte bleiben im zentralen SQLite-Speicher erhalten. Der bestehende Gesprächspfad bleibt separat und wird auf unveränderte Ausgaben geprüft.

Das ersetzt keine allgemeine trainierte Sprachfunktion. Fourierkoeffizienten sind mathematisch ebenfalls Gewichte. Verifiziert werden direkt berechnete Koeffizienten ohne Optimiererschritte, keine prinzipielle Gewichtslosigkeit oder allgemeines Schlussfolgern. Allgemeine deutsche Begriffs- und Eigenschaftsregeln sowie begrenzte Wortpräfixe liefern ausdrücklich vorgegebenes Sprachwissen.

## Daten und Trennung

Die vier lokal archivierten Shards des offiziellen Wikimedia-Datensatzes enthalten 569.062 vollständige deutsche Artikel und belegen 1.496.484.294 Bytes. Der eingefrorene aktive Kandidat enthält 20.000 ausgewählte Einleitungen mit Titel, sämtlich mit leerem Prompt. Das entspricht 6.352.638 UTF-8-Textbytes. Die vollständigen Rohartikel sind nicht sämtlich Teil des berechneten Informationsoperators.

Der Kandidat `memory/imports/information_wikipedia_20000_v3.jsonl` hat SHA-256 `2f5b876d904346ca4b9082e05ff5e4bc5e8a729f2a479c777f47e1332a0a5674`. Alle 20.000 übernommenen Originalstellen wurden überprüft. Ein deterministischer Neuaufbau mit umgekehrter Quelldateireihenfolge erzeugte exakt denselben Kandidaten. Herkunft, Filter, die feste Quellenstichprobe und Einschränkungen des Wikipedia-Exports stehen in der [Quellendokumentation](../../docs/information-corpus-sources.md).

Der alte aktive Bestand umfasst 12.000 Texte. Bei gemeinsamer Verwendung werden davon weiterhin exakt dieselben 12.000 Texte an den Gesprächscompiler übergeben; die 20.000 neuen Wikipedia-Texte gehen in den Informationscompiler. Damit werden die alten Gesprächsfelder durch den neuen großen Informationsbestand nicht verdünnt. Die Corpusauswahl wurde nicht nach den neuen Abschlussfragen erweitert.

## Entwicklung und Sprachprüfung

Ein separater Evaluationsagent legte vorab 20 Entwicklungsfragen und 20 zurückgehaltene Informationsfragen fest. Das vom Nutzer ausdrücklich genannte Frequenzbeispiel gehört zur Entwicklung. Bekannte 40 Gesprächsrunden sind eine Regression und kein neuer unbekannter Test. Zusätzlich wurden deklarative Zufallsfakten mit getrennten Fragen und einer späteren Datenpermutation vorbereitet.

Die inhaltlichen Bewertungen stammen von separaten KI-Agenten, nicht aus einer menschlichen oder verblindeten Studie. Die kleinen Stichproben erlauben keine allgemeine Erfolgsquote für alltägliche Fragen. Automatisch als „neu“ markierte vollständige Texte werden außerdem separat auf Quellenüberschneidung geprüft: Eine derartige Kennzahl kann trotz exakter Quellenpassage wahr sein und gilt hier ausdrücklich nicht als Nachweis neuer Formulierungen.

Die ursprüngliche Laufzeit erreichte auf den 20 neuen Entwicklungsfragen 0 korrekte Antworten. Der erste Informationsversuch erreichte 3/20 und zeigte unter anderem Abkürzungsabbrüche und falsch zugeordnete Begriffe. Strengere allgemeine Begriffsprüfung reduzierte falsche Ausgaben, enthielt sich aber im zweiten Versuch zu häufig: 2/20 korrekt, 18 Enthaltungen. Diese fehlgeschlagenen Zwischenstände bleiben in den Evaluationsartefakten erhalten.

Danach wurden allgemeine Frageprädikate, die Rollenbeziehung zwischen Einheit und Größe sowie die Darstellung mehrdeutiger Wortformen korrigiert. Die Runtimeprüfung deckte außerdem einen falschen Importquellen-Präfix und zu enge Erkennung von Wissensfragen auf. Der korrigierte Quellenfilter liefert exakt die vorgesehene 12k/20k-Aufteilung. Die allgemeine Eingaberoute erkennt jetzt 20/20 Entwicklungsfragen als Informationsfragen und belässt 40/40 bekannte Gesprächsrunden im Gesprächspfad. Erkennen einer Frage bedeutet noch nicht, ihren Inhalt beantworten zu können.

Der abschließende Runtime-v3-Entwicklungslauf erreicht **4/20 korrekte Antworten, 16 leere Enthaltungen und keine themenfremde Ausgabe**. Unterstützt sind hier die Frequenzdefinition, Hertz als Frequenzeinheit, die Kehrwertbeziehung zur Periodendauer und Volt als Spannungseinheit. Alle vier Antworten entsprechen normalisierten Quellpassagen. Ihre tokenweise Rekonstruktion belegt die neue Mechanik, aber keine neuen fachlichen Schlussfolgerungen. Im aktiven Korpus sind für mindestens sieben der 18 fachlichen Entwicklungsfragen geeignete Belege vorhanden; zwei weitere Fragen verlangen nicht bereitgestellte private Echtzeitmessungen. Datenlücken und Decoderfehler werden deshalb getrennt betrachtet.

Die vollständige Runtime-v2-Regression erzeugte **40/40 exakt gleiche Gesprächsantworten und Tokenfolgen** wie die eingefrorene Baseline. Die unveränderten Gesprächsdateien und unveränderten Routen aller 40 Runden begründen die Wiederverwendung dieses Belegs für v3. Die 40 Antworten wurden in v3 nicht noch einmal als neuer Vergleich erzeugt. Eine Beschleunigung oder Verbesserung der Gesprächssemantik wird daraus nicht abgeleitet. Nach diesem Entwicklungslauf wurden Code und Daten vor dem Öffnen der unbekannten Abschlussfragen eingefroren.

Der unabhängige Zufallsfaktentest ist fehlgeschlagen: **0/12 Vorwärtsfragen, 0/12 Rückwärtsfragen und 0/12 Fragen nach Zahlenpermutation** werden beantwortet. Sämtliche 44 Ausgaben einschließlich vier Negations- und vier unbekannter Kontrollfragen bleiben leer. Identische Eingaben reproduzieren dieselben Koeffizienten; die Zahlenpermutation verändert bei gleichbleibendem Vokabular den Koeffizientenhash. Die Daten werden also in das Feld übernommen, aber die unabhängige deklarative Form wird nicht erfolgreich an die Fragen gekoppelt. Dies ist ein klarer fehlender Fähigkeitsnachweis und wird nicht als gelöste Faktenaufgabe gewertet. Nach Bekanntwerden wurden weder Sprachregeln noch Daten dafür angepasst.

Im abschließenden, zuvor unbenutzten Informations-Holdout bleiben **alle 20 Antworten leer**. Damit werden **0/20 Fragen beantwortet**, gegenüber **1/20 korrekten Antworten der Baseline**. Der zuvor richtige Fall zur DNA-Funktion geht durch Enthaltung verloren. Ein zweiter unabhängiger KI-Reviewer bestätigt die Bewertung und findet für mindestens zwei Abschlussfragen direkte Belege im neuen Prosa-Korpus sowie für weitere Fragen Teilbelege. Datenlücken erklären somit nicht sämtliche Enthaltungen. Leere Ausgaben besitzen keine bewertbare Grammatik. Enthaltung bei nicht bereitgestellten privaten Echtzeitwerten ist angemessen, sie ist aber kein Beleg für Fachwissen. Der neue Decoder weist in diesem Versuch keine allgemeine Informationsgeneralisierung nach. Mehr Daten und numerisch korrekte Fourieroperationen allein haben die sprachliche Frage-Fakten-Bindung nicht gelöst.

## Numerische Kontrollen

| Prüfung | Ergebnis |
|---|---:|
| Gesamte technische Testsuite | 536 bestanden |
| Unabhängige synthetische Basiswechsel | 64 |
| Größte Differenz zu `W x` nach Fourier-Basiswechsel | 7,33 × 10⁻¹⁵ |
| Geprüfte Interferenzzustände des neuen Decoders | 54 |
| Zustände mit gültiger Wahrscheinlichkeitsverteilung | 48 |
| Korrekt abgewiesene destruktive Nullzustände | 6 |
| Exakte Rückrechnungen neuer Text-/Metadatenbytes | 80.000/80.000 |

Die 48 gültigen Zustände stimmen nach der offen implementierten Rundung auf zwölf Dezimalstellen exakt mit dem unabhängigen direkten Rechenpfad überein. Gemeinsame Zeitverschiebung lässt ihre Intensitätsverteilung invariant. Das Nullsetzen des tatsächlichen Datenfelds oder beider Promptkanäle verhindert eine Ausgabe. Eine relative Datenphase ändert in einem Mehrmodenversuch die Verteilung um Total Variation 0,333; eine relative Promptphase um 0,667 und damit auch das wahrscheinlichste Token. Nicht jede Phase wirkt auf jeden Präfix: entsprechende Nulländerungen bleiben ebenfalls im Bericht sichtbar.

Ein kleiner unabhängiger Prosa-Versuch erzeugt „Blaue Katzen schlafen.“ aus anders zusammengesetzten Ausgangssätzen. Das belegt begrenzte Neukombination lokaler Übergänge. Es ist kein Beweis für beliebige Antworten oder zuverlässige neue Schlussfolgerungen aus Wikipedia.

Die permanente Dokumentwelle über den kombinierten 32.000er-Bestand berechnet alle 16.267.618 Text- und Metadatenmoden. Der bisherige separate Abrufindex umfasst 131.072.000 Schlüsselmoden. Die drei gemessenen vollständigen Bilder brauchten unter paralleler Last im Median 2,82 Sekunden; der Zielwert von zehn Bildern pro Sekunde wird nicht erreicht. Ihre Energie blieb bei allen drei Zeitpunkten gleich. Diese bytegenaue Speicherprüfung ist vom semantischen Informationsdecoder getrennt.

Nachweise: [Tests](technical-tests.json), [Basiswechsel](weight_coordinate_audit.json), [Decoder-Numerik](../information/numerical_audit.json), [vollständige Speicherwelle](storage_audit.json), [mathematische Herleitung und Primärquellen](../../docs/information-wave-algorithm.md).

## Reproduzierbarkeit und Laufzeit

Der eingefrorene Informationskern `freqai/information.py` hat SHA-256 `5c4726b86f05ba794aaf8e6a4cca3a91cee82a7520893484db3dd9a44a08f2d7`. Für 20.000 Texte wurden 116.649 Vokabelsymbole und rund 2,78 Millionen Präfixfelder aufgebaut. Der isolierte Aufbau benötigte unter paralleler Last rund 141 Sekunden und bis etwa 2,71 GiB Arbeitsspeicher. Warme einzelne unterstützte Antworten entstanden im isolierten Operator in etwa 10–60 Millisekunden; das sind keine End-to-End-Browserlatenzen.

Drei vollständige Neuaufbauten aus demselben finalen 20.000er-Bestand reproduzieren exakt denselben Koeffizientenhash `475307745896c2a34a8b12b86ad25d5992205514e1fa9ddef26033e541cd98e2` und denselben Darstellungshash. Die [Integritätsprüfung](evaluation/final_integrity.json) bestätigt außerdem unveränderte Evaluationsdateien, null neue Frage-Antwort-Paare, keinen Fund vollständiger Evaluationsfragen in den importierten Prosa-Texten sowie unveränderten Livecode nach dem Freeze. Das belegt reproduzierbare direkte Berechnung, keine semantische Generalisierung.

Ein neuer Datenimport erfordert keinen Serverneustart, kann aber den betroffenen abgeleiteten Operator erneut aufbauen. Stabile Tokenfrequenzen, getrennte Sitzungskontexte und Wiederverwendung unveränderter Operatoren werden technisch geprüft. Eine Programmänderung erfordert weiterhin das Laden des neuen Codestands.

## Produktiver Import und Browserprüfung

Vor der Programmaktualisierung wurde die Datenbank mit SQLite-Backup gesichert. Nach einmaligem Laden des neuen Codestands wurde der Gesprächsoperator am bisherigen 12.000er-Bestand aufgebaut. Anschließend ergänzte der CLI-Import atomar 20.000 neue Einträge zur Revision 6. Der Server behielt während dieses Datenimports die Prozesskennung **19748** und seine laufende Zeitachse.

Die [Laufzeitprüfung](deployment/active-report.json) bestätigt alle ursprünglichen Dokumenthashes, das unveränderte Archiv mit 48 Texten, sämtliche 74 ursprünglichen Historienzeilen und alle 15 ursprünglichen Gesprächszustände. Die drei protokollierten vollständigen Wellenbilder zeigen fortschreitende Ticks und dieselbe Energie. Zusätzliche Auditgespräche sind neue, isolierte Sitzungen.

Der erste API-Aufruf des Informationsdecoders benötigte 116,59 Sekunden einschließlich Aufbau. Die Frequenzdefinition endet vollständig nach 36 Tokens und passt in das produktive Limit von 40 Tokens. Die anschließende Einheitenfrage liefert „Das Hertz (Einheitenzeichen: Hz) ist die SI-Einheit der Frequenz.“ Alle Informationsantworten tragen die neue Decodermethode, null Optimiererschritte und eine leere Liste vollständiger Antworttreffer. Eine anschließende Begrüßung verwendet wieder den bestehenden Gesprächsdecoder.

Die tatsächliche [Edge-Browserprüfung](deployment/browser.json) bestätigt in einer getrennten Sitzung Begrüßung, Definition, Einheit-Folgefrage, Erhalt aller drei Antworten nach Reload und die Speicheransicht mit jeweils 50 Zeilen auf den Seiten 1, 2 und 640. Zwei Kontextbilder bei gleicher Gesprächsrevision zeigen unterschiedliche Auslenkungen und Energie 1. Es gab keine JavaScript- oder Konsolenfehler. Die warme Frequenzantwort dauerte dort 0,48 Sekunden, die Folgefrage 5,07 Sekunden. Die erste Begrüßung während des parallelen kalten Informationsaufbaus benötigte 54,53 Sekunden; die Ursache dieser Wartezeit wurde nicht getrennt instrumentiert.

Visuell geprüfte Nachweise: [Sachantworten und Kontext](deployment/browser.context.png), [letzte Speicherseite](deployment/browser.last-page.png), [Oberfläche nach Reload](deployment/browser.png). Technische Integration ist damit bestätigt; die negative Abschlussbewertung der allgemeinen Antwortfähigkeit bleibt davon unberührt.
