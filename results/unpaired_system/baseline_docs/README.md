# FreqAI – Text aus interagierenden Fourierfeldern

FreqAI besitzt einen experimentellen **Wortgenerator**: Daten bilden komplexe Felder für mögliche Folgewörter. Prompt und Gesprächskontext modulieren diese Felder. Die inverse Fouriertransformation des Ergebnisfelds liefert Wortamplituden; deren Intensitäten bestimmen das nächste Token. Anschließend wird mit dem erzeugten Präfix weitergerechnet. Im Wellenmodus werden keine vollständigen Beispielantworten oder Antwortfragmente als Kandidaten verglichen.

Die Sprachkenntnisse stammen aus gezählten Wortübergängen, ausdrücklich vorgegebenen Eingabemerkmalen und dem Textbestand. Es gibt kein vortrainiertes Sprachmodell und keine Gradientenoptimierung. **Das Zählen und Konditionieren ist trotzdem eine statistische Anpassung an Daten.** Fourierkoordinaten allein erzeugen weder Bedeutung noch unbekanntes Wissen. Neue sinnvolle Sätze sind im begrenzten Versuch möglich; verlässliche allgemeine Gespräche sind noch nicht erreicht.

## Öffnen und ausprobieren

```powershell
.\start.ps1
```

Dashboard: **http://127.0.0.1:8765**. Nach einer Codeaktualisierung den Server neu starten und die Seite neu laden. Bei neuen Sitzungen ist der Modus **Wellen: Wort für Wort generieren** voreingestellt. Die ausdrücklich auswählbare Alternative **Regeldialog** verwendet die frühere Antwortkomposition. Die Auswahl bleibt pro Browser-Tab erhalten.

`Hallo`, `Mir geht es gut, wie geht es dir denn?` und `gut und dir?` dienen als Einstieg. Die Antwortdetails zeigen die berechneten Folgewörter und ihre Wahrscheinlichkeiten. Die Kontextanzeige zeigt beide Quadraturen desselben gespeicherten Tokenfelds, das der Generator verwendet. **Neues Gespräch** beginnt eine eigene Sitzung; Reload und Serverneustart erhalten bestehende Verläufe.

Der Generator arbeitet mit einem endlichen Wortschatz und begrenzten Sprachregeln. Unbekannte Namen können nicht beliebig kopiert werden. Einige Eingaben liefern unpassende Texte oder bei fehlender Eingabekopplung keine Ausgabe. Mehr Rechenzeit oder längeres Schwingen löst diese Wissenslücken nicht. Der Wellenmodus greift dabei nicht automatisch auf den Regeldialog zurück. Einzelne erzeugbare Selbstaussagen über Fähigkeiten sind keine Funktionsnachweise.

## Den Speicher erweitern

Alle aktiven Texte, archivierten Alttexte und Gesprächszustände liegen zentral in **`memory/memory.sqlite3`**. JSONL-Dateien sind Importvorlagen; Auswertungsfragen werden nicht als Gesprächswissen eingelesen. Der Gesprächsverlauf wird gespeichert, aber nicht automatisch als neue Evidenz in den Sprachkorpus übernommen.

In der Oberfläche:

1. Für ein Gesprächspaar einen **Gesprächsanlass** eingeben, etwa `Ich habe morgen frei`. Für reine Sachtexte dieses Feld leer lassen.
2. Unter **Antwort / Textinhalt** einen grammatischen Text ergänzen, etwa `Worauf hast du an deinem freien Tag Lust?`.
3. Eine **Quelle / Bezeichnung** vergeben und **Als Welle speichern** klicken.

Neue Einträge werden ohne Serverneustart übernommen. Im Wellenmodus ergänzen sie Wortschatz, Wortübergänge und Kopplungen zu Eingabemerkmalen. Ein Sachtext sollte mit seinem Begriff oder einem Titel beginnen und vollständige Aussagen enthalten. Beispielsweise kann die Frage „Was ist eine Frequenz?“ aus einer deklarativen Frequenzbeschreibung verarbeitet werden; eine vorgegebene Frage dazu ist nicht erforderlich. Begriffsregeln und Datenabdeckung begrenzen weiterhin die möglichen Antworten.

Die Tokenfrequenzen bleiben bei einer Ergänzung stabil. Betroffene abgeleitete Generatoren werden neu aufgebaut und zwischengespeichert. Gesprächsdaten und der neue Wikipedia-Bestand besitzen getrennte Operatoren im selben Speicher; ein reiner Wikipedia-Import erhält den bereits aufgebauten Gesprächsoperator. Der erste Aufbau dauert je nach Pfad und Last ungefähr ein bis zwei Minuten. Die bestehenden Dokumentmoden und die Schwingungszeit laufen während eines Datenimports weiter. Die Speicheransicht zeigt jeweils 50 Einträge und bietet Vor- und Zurückblättern.

```powershell
# Ein Paar live ergänzen
.\.venv\Scripts\python.exe -m freqai add --prompt "Ich habe morgen frei" --text "Worauf hast du an deinem freien Tag Lust?" --source "Alltag / Freizeit"

# Reinen Informationstext live ergänzen; kein Prompt erforderlich
.\.venv\Scripts\python.exe -m freqai add --text "Die Frequenz ist die Anzahl der Wiederholungen pro Zeitspanne." --source "Eigene Sachtexte"

# Import ergänzt atomar und überschreibt keine vorhandenen IDs
.\.venv\Scripts\python.exe -m freqai import memory\fixtures\extension_120.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\language\generative_corpus.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\imports\public_dialogue_expansion.jsonl
.\.venv\Scripts\python.exe -m freqai stats

# Wortweise Wellenberechnung mit dauerhaftem Sitzungskontext
.\.venv\Scripts\python.exe -m freqai generate "Hallo" --session mein-gespraech --json
.\.venv\Scripts\python.exe -m freqai generate "gut und dir?" --session mein-gespraech --json

# Gleichwertiger Aufruf; ask ohne --mode bleibt kompatibel zum Regeldialog
.\.venv\Scripts\python.exe -m freqai ask "Wie geht es dir?" --mode wave --json
```

JSONL enthält eine Zeile pro Eintrag; `prompt` ist bei ungepaarten Sprachtexten optional:

```json
{"id":"mein-dialog-1","prompt":"Ich habe morgen frei","text":"Worauf hast du an deinem freien Tag Lust?","source":"Alltag / Freizeit"}
```

Identische Inhalte werden beim erneuten Import nicht dupliziert. Unterschiedliche Inhalte unter derselben ID erzeugen einen Konflikt. Eine Internetadresse wird nicht automatisch geladen. NPZ-Exporte sind Austauschkopien und kein zweiter aktiver Speicher.

## Was berechnet wird

Der Laufzeitgenerator nutzt Wortpräfixe der Länge bis zwei. Für einen Präfixkontext `h` werden die beobachteten Folgewörter zu `p_h` normalisiert und als `S_h = FFT(sqrt(p_h))` gespeichert. Numerisch codierte Eingabemerkmale regen zusätzliche konditionierte Felder an. Explizite Kategorien für den ergänzten Sprachbestand stehen in [generative_categories.json](memory/language/generative_categories.json).

Mit gemischter Datenamplitude `a`, synchroner Zeitphase `U(t)`, Prompt-/Kontextmodulation `b` und Verstärkung `γ` berechnet die spektrale Faltung:

```text
R = FFT(a · U(t)) + γ · FFT(a · U(t) · b)
p(nächstes Wort) ∝ |IFFT(R)|²
```

Eine Maske für bekannte lokale Wortübergänge, Normalisierung und Beam Search bestimmen daraus eine neu aufgebaute Tokenfolge. Mehrere erkannte Eingabeabsichten werden nacheinander in getrennten Merkmalsbändern verarbeitet. Das ist eine explizite Steuerregel. Die bisher erzeugten Wörter beeinflussen sowohl weitere Folgewortfelder als auch den gespeicherten Sitzungskontext.

Die gemeinsame Zeitphase verschwindet beim Intensitätsauslesen. Daher ändert die Uhrzeit allein keine Antwort. Eine veränderte relative Interferenzphase kann die Wortwahl dagegen ändern. Der Decoder hat eine mathematisch äquivalente direkte Referenzrechnung; ein zusätzlicher Intelligenzgewinn durch die Fourierdarstellung ist nicht nachgewiesen. Herleitung, Primärquellen und Suchprotokolle: [Recherche und Algorithmus](docs/generative-wave-research.md).

## Informationsfelder ohne optimierte Gewichte

Der neue Informationsdecoder berechnet sämtliche Übergangskoeffizienten direkt aus Wortzählungen: `C = FFT(sqrt(n / sum(n)))` mit Einheitsenergie. Kompatible Begriffs- und Eigenschaftsfelder werden symmetrisch überlagert. Die inverse Transformation der Interferenz liefert pro Schritt die Tokenwahrscheinlichkeiten. Es gibt in diesem Pfad keine vortrainierten Gewichte, keine Optimiererschritte und keine angepassten Quellen- oder Mischverstärkungen. Die ältere Gesprächsberechnung behält ihre bisherigen festen Compilerparameter.

**Fourierkoeffizienten sind mathematisch ebenfalls Gewichte.** Ein Wechsel in Fourierkoordinaten ersetzt keine unbekannte Sprachfunktion: `F* (F W F*) (F x) = W x`. Der nachgewiesene Unterschied ist die direkte, reproduzierbare Berechnung aus Daten statt iterativer Parameteroptimierung. Syntaxregeln, Präfixlänge und Beam Search bleiben explizite Bestandteile des Algorithmus. [Mathematik und Primärquellen](docs/information-wave-algorithm.md) beschreiben diese Grenze und die tatsächlich implementierten Felder.

Der zentrale Speicher enthält jetzt **32.000 aktive Texte**, darunter **20.000 neue deutsche Wikipedia-Einleitungen ohne Frage-Antwort-Paare**. Die vollständigen lokalen Quelldateien enthalten **569.062 Artikel**; die ausgewählten Einleitungen werden aktiv berechnet. Roharchiv, aktive Texte und Qualitätstests sind unterschiedliche Mengen. Quelle ist der gepinnte [Wikimedia-Datensatz](https://huggingface.co/datasets/wikimedia/wikipedia), Snapshot vom November 2023. [Herkunft und reproduzierbarer Import](docs/information-corpus-sources.md).

```powershell
.\.venv\Scripts\python.exe -m freqai import memory\imports\information_wikipedia_20000_v3.jsonl
.\.venv\Scripts\python.exe -m freqai generate "Was ist eine Frequenz?" --session wissen --json
.\.venv\Scripts\python.exe -m freqai generate "Und welche Einheit hat sie?" --session wissen --json
```

Im kompilierten Informationsmodell liegen keine vollständigen Antwortkandidaten oder Dokumentranglisten. Die Ausgabe entsteht tokenweise aus adressierten Übergangsfeldern; sie kann dennoch eine vorhandene Aussage exakt rekonstruieren. Das allein ist kein Nachweis neuer Schlussfolgerungen. Unbekannte Begriffe und nicht unterstützte Beziehungen führen häufig zur Enthaltung. [Aktueller Ergebnisbericht](results/information_corpus/report.md) trennt Antwortqualität, unveränderte Gespräche, numerische Kontrollen und den laufenden Server.

**Aktueller Qualitätsbefund:** 4/20 Entwicklungsfragen werden korrekt beantwortet, 16 bleiben leer. Auf allen 20 zuvor unbenutzten Abschlussfragen enthält sich der Decoder, gegenüber einer richtigen Antwort der Baseline; eine allgemeine Wissensgeneralisierung ist nicht nachgewiesen. Auch die 24 unabhängigen Vorwärts-/Rückwärtsfragen zu deklarativen Zufallsfakten und zwölf Wiederholungen nach Zahlenpermutation bleiben unbeantwortet. Die 536 technischen Tests bestehen, und die 40 bekannten Gesprächsantworten bleiben exakt unverändert. Diese technischen Erfolge gleichen die fehlende allgemeine Antwortfähigkeit nicht aus.

Der laufende Server wurde mit 32.000 Texten im Browser geprüft: Frequenzdefinition und Einheit-Folgefrage, gespeicherter Verlauf nach Reload, bewegte Kontextwelle und Pagination bis Seite 640 funktionieren. Der erste Informationsaufbau über die API dauerte rund 117 Sekunden; beobachtete warme Browserantworten lagen bei 0,48 und 5,07 Sekunden. Die Vollberechnung der Dokumentwelle läuft parallel und erreicht weiterhin keine zehn Bilder pro Sekunde.

## Frühere Erweiterung um öffentliche Gesprächspaare

Der Ausgangsbestand von 621 Texten wurde zuvor um **11.379 öffentliche deutsche Frage-Antwort-Paare** erweitert: 533 gefilterte OpenAssistant-Antworten und 10.846 GermanQuAD-Fragen mit vollständigen Antwortsätzen aus Wikipedia. Diese damaligen 12.000 Einträge bilden weiterhin den Gesprächsbestand in SQLite. Mehrere Fragen können denselben Quellabschnitt verwenden und sind keine unabhängigen Beobachtungen. [Quellen, Lizenzen, Filter und reproduzierbarer Import](docs/public-corpus-sources.md) dokumentieren die lokalen SearXNG-Recherchen, gepinnten Downloads und Herkunft jedes Eintrags. Offizielle Testfragen werden nicht als Daten importiert.

Ein ungewichteter Import verschlechterte die Gespräche deutlich. Die korrigierte Zuordnung behandelt öffentliche Eingaben lexikalisch, damit beispielsweise ein „Danke“ am Ende einer Fachfrage nicht die gesamte Fachantwort in den Dankeskanal einträgt. Die eigenen Gesprächsannotationen bleiben wirksam. Quellengewichte von `0.1` für OpenAssistant und `0.05` für GermanQuAD begrenzen deren Einfluss gegenüber eigenen Paaren mit `1.0`. Das sind explizite Compilerregeln; sämtliche importierten Texte bleiben in den Feldern enthalten.

Die neue Evaluation hat 20 Entwicklungs- und 20 vorab zurückgehaltene Gesprächsrunden. Qualitätsurteile stammen von einer separaten KI-Agentenprüfung, nicht aus einer menschlichen oder verblindeten Studie. Die kleinen Stichproben begründen keine statistische Wirksamkeitsbehauptung. [Gesamtbericht](results/public_corpus/report.md) und [Einzelfallbewertungen](results/public_corpus/evaluation/report.md) trennen technische Korrektheit, Grammatik, Bedeutung und Faktentreue.

| Frühere Prüfung der Gesprächserweiterung | Ausgangsbestand | Erweiterter Bestand |
| --- | ---: | ---: |
| Aktive Texte | 621 | 12.000 |
| Inhaltlich passende Entwicklungsantworten | 11/20 | 12/20 |
| Inhaltlich passende zurückgehaltene Chatantworten | 13/20 | 12/20 |
| Korrekte Antworten auf 20 bekannte Importfragen | 0/20 | 8/20 |
| Korrekte Antworten auf 30 offizielle Testfragen ohne Referenzkontext | 0/30 | 0/30 |

**Eine Verbesserung der allgemeinen Gesprächsqualität ist damit nicht nachgewiesen.** Der ausgebaute Generator kann einige vorhandene Sachinformationen verwenden, verfehlt aber weiterhin zahlreiche Fragen. Die 20 bekannten Fragen sind ausdrücklich eine In-sample-Kapazitätsprüfung, keine Generalisierung. Weitere 30 offizielle GermanQuAD-Testfragen werden ohne mitgelieferte Referenzpassage geprüft; das ist kein regulärer GermanQuAD-Leseverständnisbenchmark. Fehler bei Abschied, Familientreffen und Negation sind im Bericht sichtbar. Nach Öffnung der Abschlussfälle wurden keine Sprachregeln darauf abgestimmt.

Die numerische Prüfung vergleicht tatsächliche komplexe Felder mit der direkten Referenz, kontrolliert Phasen und genullte Daten- und Promptfelder. Alle neuen Text- und Metadatenbytes wurden bei zwei Zeitpunkten zurückgerechnet: **45.516/45.516 exakte Rückrechnungen**. Die kompakte Speicherung vermeidet eine dichte Präfix-mal-Wortschatz-Matrix. Die permanente Dokumentwelle berechnet weiterhin alle rund 5,96 Millionen Text- und Metadatenmoden; gemessen wurden etwa 0,45 Sekunden pro Bild, der 10-Hz-Zielwert wird somit nicht erreicht. Solche technischen Nachweise ersetzen keine Sprachbewertung.

Die damalige technische Testsuite bestand mit **475 Tests**, einschließlich Import, numerischer Referenzen, Sitzungen, Live-Ergänzung und paginierter Oberfläche. Der aktuelle Informationsstand ergänzt diese auf 536 bestandene Tests.

Eine zusätzliche Optimierung ergänzt nur im temporären FFT-Träger Nullamplituden auf eine günstigere Transformationslänge: 44.614 bekannte Tokens verwenden 44.800 Trägermoden. Die direkte Referenz, bekannten Frequenzen und gespeicherten Daten bleiben gleich. Drei vollständig identische Vergleichsantworten benötigten dadurch etwa 0,94–1,44 statt 4,19–6,52 Sekunden. Die Kontextanzeige verwendet denselben vergrößerten Träger. Der vollständige Modellaufbau nach neuen Daten bleibt davon getrennt.

Die anschließende Regression über alle bereits ausgewerteten 40 Chat-, 30 Fakten- und 20 bekannten Importfragen bestätigt **90/90 bytegleiche Antworttexte**. Das belegt unveränderte Ausgaben durch die Beschleunigung und keine neue Sprachleistung. Im gesamten Regressionslauf dauerte eine Antwort im Median 1,75 Sekunden.

Die [Prüfung des laufenden Servers](results/public_corpus/deployment/active-report.json) protokolliert den Live-Import mit unveränderter Prozesskennung sowie den Erhalt alter Texte, Archive und Sitzungskontexte. Die Datenbank wurde vorher unter `memory/backups/` gesichert. Die ergänzenden Browsernachweise stehen ebenfalls unter `results/public_corpus/deployment/`.

Der bisherige Bestand umfasst 120 Gesprächspaare, einen Nutzereintrag und 500 synthetische Sprachtexte. Die **500 Texte bestehen aus Kombinationen von 200 handgeschriebenen Sätzen** in 20 Kategorien. Die [frühere Generierungsprüfung](results/generative_waves/report.md) mit eigenen 30/30 Fällen dokumentiert einen historischen Stand und ist kein direkter Vergleich mit den neuen 20/20 Fällen.

Die technischen Tests prüfen tatsächlichen Einfluss der komplexen Datenfelder, Promptablation beider Kanäle, Präfixreihenfolge, relative Phasen, Zeitstabilität, Sitzungsisolierung und Live-Import. Ein unabhängiger Toy-Korpus belegt die Bildung von `Blaue Katzen schlafen.` aus anderen lokalen Wortübergängen. Solche Kontrollaufgaben belegen die Mechanik und ersetzen keine Gesprächsevaluation.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe experiments\import_public_corpus.py --offline
```

Der Evaluator protokolliert Code- und Datenhashes. `freeze` und `holdout` sind Schritte eines neuen vorab getrennten Versuchs; bereits geöffnete Abschlussfälle dürfen nicht erneut als unbekannte Testdaten ausgegeben werden.

## Einrichtung und Dateien

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[experiments,corpora]"
.\.venv\Scripts\python.exe -m freqai import memory\fixtures\extension_120.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\language\generative_corpus.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\imports\public_dialogue_expansion.jsonl
.\.venv\Scripts\python.exe -m freqai serve --open
```

Ein Konsolenstart endet mit Strg+C. `stop.ps1` beendet den während der Einrichtung aufgezeichneten Hintergrundprozess nach Prüfung seiner Prozesskennung. Die SQLite-Daten bleiben erhalten; es wird kein Windows-Autostart eingerichtet.

`freqai/generative.py` enthält den Folgewortdecoder, `generation_runtime.py` koppelt ihn an zentrale Daten und Sitzungen. `store.py` verwaltet SQLite, `server.py` betreibt Schwingung, Synchronisation und HTTP. `semantics.py` liefert ausdrücklich definierte Eingabemerkmale. `codec.py` und `memory.py` enthalten die gesonderte verlustfreie Textcodierung und den früheren Abruf. `dialogue.py` und `synthesis.py` bleiben für den expliziten Regeldialog verfügbar.

Historische Nachweise: [Regeldialog V2](results/dialogue/round2/report.md), [früherer Abrufversuch](results/extended/report.md), [zentrale Ablage und Live-Import](docs/live-memory.md). Sie dokumentieren jeweils ihre eigene Version und Aufgabe.
