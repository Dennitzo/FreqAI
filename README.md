# FreqAI – Gespräche aus Informationsfeldern

FreqAI berechnet Antworten aus **deklarativen Informationstexten**, der aktuellen Nachricht und dem gespeicherten Gesprächszustand. Wissensfragen und Alltagsgespräche verwenden denselben `UnpairedWaveModel`-Compiler. Der aktive Datenbestand besitzt keine Frage-Antwort-Paare; die früheren Paar- und Antwortbausteinverfahren wurden entfernt.

Es gibt kein vortrainiertes Sprachmodell und keine iterative Parameteroptimierung. Komplexe Koeffizienten entstehen direkt aus Wortübergängen und deklarativen Rollen. **Die Rollenextraktion, Sprechaktanalyse und deutsche Grammatik sind ausdrücklich programmiert.** Fouriertransformation allein erzeugt keine Bedeutung und keine allgemeine LLM-Fähigkeit. Die Antwortqualität wird getrennt von numerischer Korrektheit geprüft.

## Starten

```powershell
.\start.ps1
```

Dashboard: **http://127.0.0.1:8765**. Nach einer Programmänderung den Server neu starten und die Browserseite neu laden. Neue Informationstexte werden dagegen bei laufendem Server übernommen. Die alten Verfahrensauswahlen und gespeicherten Browser-Moduseinstellungen werden nicht mehr verwendet.

Zum Ausprobieren eignen sich `Hallo`, `Mir geht es gut, wie geht es dir denn?`, `Wie heißt du?`, `Was ist eine Frequenz?` und anschließend `Und welche Einheit hat sie?`. **Neues Gespräch** beginnt eine getrennte Sitzung. Reload und Serverneustart erhalten den Verlauf und den gespeicherten Sitzungskontext.

Das Terminal meldet Startphasen, Compilerphasen, verarbeitete Dokumente und Prozentwerte. Während langer Phasen erscheint alle zehn Sekunden eine Laufzeitmeldung. Ausgabe und Fehler stehen zusätzlich in `runtime/start-YYYYMMDD-HHMMSS.log`; bei einem Fehler zeigt der Starter den Exitcode. `./start.ps1 -NoBrowser` startet ohne Browserfenster.

Große Bestände verwenden automatisch einen Festplattenindex. Der erste Start berechnet diesen vollständig; weitere Starts und Importe verwenden fertige Blöcke wieder. Die erste Anfrage zu einem neuen Arbeitsbereich compiliert passende Informationsabschnitte. Die Dokumentwelle läuft parallel mit einer Zielrate von zehn Bildern pro Sekunde; die tatsächlich erreichte Geschwindigkeit hängt vom Rechenmodus und der Hardware ab.

### Großer Wikipedia-Bestand

Die 569.210 Dokumente werden nicht mehr gleichzeitig als dichte Suchmatrix und mehrfach kopierte Schwingungsarrays in den RAM geladen. Ab 512 MiB geschätzten dichten Arrays oder 2 MiB Quelltext verwendet `MemoryStore.load_memory()` den dateigestützten Modus. Direkte übergroße RAM-Aufbauten werden vor der Allokation mit einer verständlichen Fehlermeldung abgelehnt.

Alle vollständigen Texte werden in höchstens 4.000 Zeichen großen Abschnitten indexiert. Die vollständigen UTF-8-Schwingungskoeffizienten liegen in einem abgeleiteten SQLite-Cache unter `memory/.compiler-cache/`. Der Aufbau verarbeitet ungefähr 2 MiB Text pro Paket, verteilt die Transformation auf CPU-Worker und bestätigt fertige Pakete transaktional. Nach einem Abbruch setzt der nächste Start bei den bestätigten Dokumenten fort. Einzeltexte über 16 MiB benötigen vor dem Aufbau eine Aufteilung; fehlender Cache-Speicherplatz führt zu einem erklärten Abbruch.

Für eine Antwort wählt der Volltextindex bis zu 24 passende Abschnitte anhand von Nachricht und vorherigen Themen. Kleine deklarative Gesprächs- und Lexikoninformationen kommen hinzu; insgesamt bleiben höchstens 120.000 Zeichen im Compiler-Arbeitsbereich. Derselbe `UnpairedWaveModel` erzeugt daraus die Antwort. **Das ist jetzt eine explizite Auswahl von Informationen vor der Generierung:** Nicht alle Wikipedia-Wortübergänge sind gleichzeitig Teil des Antwortmodells. Auswahlfehler können deshalb die Antwortqualität begrenzen. Der vollständige Artikelbestand bleibt erhalten und durchsuchbar; es entstehen keine Frage-Antwort-Paare. Höchstens zwei Arbeitsbereiche bleiben im RAM, acht Cacheplätze begrenzen die Zahl gespeicherter Anfragecompiler.

Die große Dokumentwelle verwendet einmal berechnete, zeitlich konstante Gesamtenergien und wertet nur die Pakete mit sichtbaren Koordinaten aus. Die angezeigten Werte entsprechen der vollständigen inversen DCT; es werden keine Frequenzen abgeschnitten. Die große dichte Suchspektrenmatrix entfällt. Für die Anzeige bleiben höchstens 64 MiB gewichtete Koeffizienten resident; deren Auswertung verteilt sich im Automatikmodus auf die ausgewählten CUDA-GPUs. Ohne CUDA erfolgt die Auswertung auf der CPU. Überschreitet die Anzeige dieses Budget, werden die benötigten Pakete einzeln von der Festplatte verarbeitet. Der Gesamtbestand bleibt auf der Festplatte.

## CUDA und mehrere CPU-Kerne

Im kleinen RAM-Modus verwendet FreqAI standardmäßig `FREQAI_COMPUTE=auto`: Die fortlaufende Dokumentwelle hält ihre Koeffizienten mit installiertem CuPy auf allen verfügbaren CUDA-GPUs und berechnet Auslenkung, Quadratur und Energie aller Moden. Für die Anzeige werden die benötigten Ortskoordinaten aus der vollständigen inversen DCT-Summe des jeweiligen Dokuments berechnet. Es werden keine Frequenzen abgeschnitten. FFT-Stapel, deren Ein- und Ausgabe im Hauptspeicher liegen, verwenden im Automatikmodus CPU-Threads; die GPU-Übertragung war hier teurer als diese FFT. `FREQAI_COMPUTE=cuda` aktiviert auch dafür CUDA. Der Festplattenmodus baut den vollständigen Cache in begrenzten CPU-Paketen auf und beschleunigt anschließend die sichtbaren Koordinaten mit CUDA.

Textanalyse und Compilerphasen verteilen unabhängige Arbeit über einen dauerhaften Prozesspool auf die verfügbaren logischen CPU-Kerne. Jeder Worker verwendet einen FFT-/BLAS-Thread, damit sich die Prozesse nicht gegenseitig überlasten. Kleine Aufgaben und voneinander abhängige Tokenschritte bleiben seriell. Ein defekter CUDA-Treiber oder fehlendes CuPy führt zur CPU-Berechnung; der Grund steht in der Oberfläche und der API.

```powershell
# Optionale CUDA-Unterstützung; benötigt eine passende lokale CUDA-Installation.
.\.venv\Scripts\python.exe -m pip install -e '.[cuda]'

# Standard: CUDA für große Rechenblöcke, alle verfügbaren CPU-Kerne für Textanalyse.
$env:FREQAI_COMPUTE = 'auto'
$env:FREQAI_CUDA_DEVICES = 'all'
$env:FREQAI_WORKERS = 'all'
.\start.ps1

# Reine CPU-Ausführung für Vergleich oder Rechner ohne CUDA.
$env:FREQAI_COMPUTE = 'cpu'
.\start.ps1
```

`FREQAI_WORKERS=8` begrenzt den Pool beispielsweise auf acht Prozesse; `FREQAI_PARALLEL=0` aktiviert den seriellen CPU-Vergleich. `FREQAI_CUDA_DEVICES=0,1` wählt bestimmte GPUs. Änderungen an diesen Variablen gelten nach einem Serverneustart. Unter Windows unterstützt der Prozesspool höchstens 61 Worker.

Die Oberfläche zeigt den Rechenmodus und die gemessene Dauer des letzten Wellenbilds. `/api/health` und `/api/state` liefern zusätzlich Geräte, abgeschlossene CUDA-Berechnungen pro Gerät, Poolstatus und einen eventuellen Fallback-Grund. Diese Angaben belegen ausgeführte Arbeit; sie versprechen keine dauerhaften 100 Prozent Auslastung aller Hardwarekerne. Messungen und Prüfungen dieses Umbaus stehen im [Reparaturbericht](results/repair_2026-09-13.md).

## Compiler-Cache und wachsende Bestände

Im kleinen RAM-Modus speichert der Compiler seine abgeleiteten Daten automatisch in `memory/.compiler-cache/memory.sqlite3/compiled.npz`. Nach einem Neustart lädt ein unveränderter Bestand diesen Cache. Der Cache enthält JSON-Metadaten und zusammenhängende numerische Arrays ohne Pickle. Eine atomisch ersetzte Datei pro Speicher begrenzt den Platzbedarf; die Datenbank bleibt die maßgebliche Quelle. Im Festplattenmodus gilt dieses Format für die begrenzten Anfragecompiler; der vollständige Schwingungs- und Textindex verwendet separat `paged-*/corpus.sqlite3`. Dieser Index ist auch bei `FREQAI_COMPILER_CACHE=off` notwendig. Die Variable steuert ausschließlich den Antwortcompiler-Cache.

Bei Ergänzungen werden unveränderte Übergangsgruppen wiederverwendet und ihre Wort-IDs auf das neue Vokabular abgebildet. Das funktioniert während des Betriebs sowie nach einem Import bei gestopptem Server. Ein Fingerabdruck der exakten Wortfolgen und Gruppenbeschreibungen entscheidet, welche Gruppen neu berechnet werden müssen. Neue Konzepte können die Zuordnung älterer Texte verändern, und neue Schreibweisen können alte Darstellungskontexte betreffen. Deshalb bleiben eine globale Analyse und der Abgleich der Abhängigkeiten erforderlich; die Importzeit ist nicht ausschließlich von der Anzahl neuer Texte abhängig.

Die Millionen kleiner Übergangsfelder liegen in gepackten Arrays. Einzelne Feldobjekte entstehen erst bei Bedarf zur Generierung oder Diagnose. Dadurch sinken Prozessübertragung, Speicherverwaltung und Cacheaufwand bei großen Beständen.

Die Cacheprüfung berücksichtigt die geordneten Rohdatensätze einschließlich Duplikaten, die Compilerordnung, den Quellcode sowie Python-, NumPy- und SciPy-Versionen. Bei Abweichungen oder beschädigten Dateien wird neu aufgebaut. Sitzungszustände und temporäre Grammatikrollen gehen nicht in den gespeicherten Compiler ein. Scheitert das Schreiben des Caches, bleibt die Berechnung im Arbeitsspeicher verfügbar.

```powershell
# Optional: persistenten Cache abschalten, etwa für einen Kaltstartvergleich.
$env:FREQAI_COMPILER_CACHE = 'off'

# Standard wieder aktivieren; danach Server neu starten.
$env:FREQAI_COMPILER_CACHE = 'auto'

# Alternativ ein anderes Verzeichnis für den abgeleiteten Cache verwenden.
$env:FREQAI_COMPILER_CACHE = 'D:\FreqAI-Cache'
```

In der Oberfläche steht nach einem Treffer „Compiler aus Cache geladen“. Die API zeigt unter `compiler` zusätzlich Aufbau-/Ladezeit und die Zahl wiederverwendeter beziehungsweise neu kompilierter Gruppen. Ein großer Import sollte als zusammenhängender Batch erfolgen, damit anschließend nur ein Abgleich erforderlich wird.

## Den Speicher erweitern

Die zentrale aktive Datenbank ist **`memory/memory.sqlite3`**. Die Oberfläche benötigt nur **Informationstext** und **Quelle**. Schreibe vollständige Aussagen, möglichst mit einem Titel oder klar benannten Begriff am Anfang, beispielsweise:

> Lumor
>
> Ein Lumor ist ein kupferner Behälter für blaue Steine.

Das ist ein Informationstext ohne vorgegebene Frage. Anschließend kann `Was ist ein Lumor?` die daraus berechneten Felder adressieren. Ein Eintrag garantiert keine Antwort auf jede mögliche Umschreibung.

```powershell
# Eigenen Informationstext live ergänzen
.\.venv\Scripts\python.exe -m freqai add --text "Ein Lumor ist ein kupferner Behälter für blaue Steine." --source "Eigene Informationen"

# Deklarative Gesprächsinformationen und Wikipedia-Prosa ergänzen
.\.venv\Scripts\python.exe -m freqai import memory\information\conversation_facts.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\information\conversation_lexicon_v1.jsonl
.\.venv\Scripts\python.exe -m freqai import memory\imports\information_wikipedia_20000_v3.jsonl
.\.venv\Scripts\python.exe -m freqai stats

# Ein gemeinsamer Decoder, einschließlich dauerhaftem Kontext
.\.venv\Scripts\python.exe -m freqai ask "Hallo" --session alltag --json
.\.venv\Scripts\python.exe -m freqai generate "gut und dir?" --session alltag --json
.\.venv\Scripts\python.exe -m freqai generate "Was ist eine Frequenz?" --session wissen --json
.\.venv\Scripts\python.exe -m freqai generate "Und welche Einheit hat sie?" --session wissen --json
```

Aktuelle JSONL-Einträge verwenden `id`, `text` und `source`. Zusätzliche Herkunftsangaben können in der Importdatei stehen:

```json
{"id":"lumor-info","text":"Ein Lumor ist ein kupferner Behälter für blaue Steine.","source":"Eigene Informationen"}
```

`--prompt` ist bei `add` entfernt. Ein Import mit einem nichtleeren früheren `prompt` wird zurückgewiesen. Alte Dateien dürfen ausdrücklich über `archive` aufbewahrt werden; sie gehen dann nicht in die aktiven Felder ein. Identische Informationstexte derselben Quelle werden dedupliziert. Unterschiedliche Inhalte unter derselben ID erzeugen einen Konflikt.

Neue Namen oder Zustände aus einem Gespräch bleiben **Benutzeraussagen in dieser Sitzung**. Sie werden weder zu überprüftem Weltwissen erklärt noch automatisch in den Informationskorpus importiert. Unbekannte Zeichenfolgen lassen sich über feste UTF-8-Bytesymbole abbilden. Das erzeugt keine Kenntnis ihrer Bedeutung.

## Informationen, Koeffizienten und Wortausgabe

Der Compiler erzeugt aus den Informationstexten sowohl Prosa-Präfixfelder als auch deklarative Rollenfelder. Beispiel: „Der Assistent heißt FreqAI.“ liefert Subjekt, Prädikat und Namensargument. Die explizite Grammatik kann die Sprecherperspektive ändern. Der vollständige Ausgabesatz wird dabei nicht als fertiger Antwortkandidat hinterlegt.

Für beobachtete Übergänge mit Häufigkeiten `n` gilt mit unitärer FFT:

```text
C = FFT(sqrt(n / sum(n)))
```

Eine deklarative Rolle enthält zusätzlich geordnete Symboladressen und ein normiertes komplexes Positionsfeld. Die Adressen sind Teil der Kodierung; gleiche uniforme Koeffizienten allein enthalten keine unterschiedlichen Wortbedeutungen. Ein Grammatikzustand legt fest, welche Rollenmoden für das nächste Symbol kompatibel sind.

Mit der **Datenwelle** `D` (alle von den Daten erregten Symbolmoden, Einheitsenergie), der **Promptwelle** `P` (Prompt, Byte-Symbole und gespeicherter Sitzungskontext, Einheitsenergie) und den resonanzgewichteten Kandidatenamplituden `A` wird pro Symbol gerechnet:

```text
R_F = |⟨a_F , P⟩|                                       Resonanz eines Datenfeldes mit der Promptwelle
A_k = Σ_F w_F (0.25 + R_F) a_{F,k}                       kohärente Superposition aller gebundenen Felder
psi_k = A_k (1 + g_p exp(i phi) Phat_k) (1 + g_d Dhat_k)  Interferenz mit beiden Wellen
p(Symbol) = |psi|² / sum(|psi|²)                          gemessene Intensität
```

Der produktive Operator berechnet die Produkte über spektrale Faltung. Eine unabhängige direkte Rechnung prüft das Ergebnis. Anschließend wird das gewählte Symbol in Präfix und Grammatikzustand übernommen und der nächste Schritt berechnet. Beam Search und ein begrenztes Symbolbudget steuern die endliche Ausgabe.

Auch diese Koeffizienten wirken mathematisch als Gewichte. Ein Fourier-Basiswechsel `F* (F W F*) (F x) = W x` beseitigt die Abhängigkeit von einem Operator nicht. Der nachgewiesene Unterschied lautet **direkte Berechnung ohne Parameteroptimierung**. [Ausführliche Grammatik und Mathematik](docs/unpaired-grammar.md), [Basiswechsel, Prosa-Felder und Primärquellen](docs/information-wave-algorithm.md).

## Bestand und nachvollziehbare Migration

Seit der Konsolidierung vom 13.09.2026 enthält **`memory/memory.sqlite3` als einziger Datensatzspeicher 569.210 Informationseinträge**: 569.062 vollständige Wikipedia-Artikel aus allen vier lokal vorhandenen Parquet-Dateien, 55 zusätzliche bereits lokal gespeicherte Wikipedia-API-Einträge sowie 45 deklarative Gesprächsinformationen und 48 lexikalische Einordnungen. Die 20.144 bisherigen Einleitungen aus diesen vier Dateien wurden durch vollständige Artikel ersetzt. Die 55 zusätzlichen API-Einträge behalten ihren lokal vorhandenen Textumfang; es wurden keine weiteren Artikel aus dem Internet geladen.

Jeder vollständige Artikel enthält den Originaltitel und den unveränderten gesamten Quelltext einschließlich weiterer Absätze, Überschriften und Listen. Artikel-URLs, Quellversion, Lizenzen und Textprüfsummen stehen in `information_provenance`; Datei-Prüfsummen und Importnachweise stehen in `information_imports`. Vor dem Löschen der Quelldateien wurden alle 569.062 Volltexte einzeln mit den Originalen verglichen. [Import- und Aufräumbericht](results/memory_consolidation.md).

```powershell
# Gespeicherte Artikelprüfsummen und SQLite-Integrität ohne Rohdownloads prüfen
.\.venv\Scripts\python.exe scripts\consolidate_memory.py --verify-only
```

Die bereits übernommenen JSONL-/Gzip-Datensätze, Rohdownloads, Exportstufen unter `dataset/`, Datensatz-Backups und alten Paar-/Sprachprior-Daten wurden auf Benutzerwunsch entfernt. Auch die 12.048 archivierten Trainingsdatensätze wurden aus der Datenbank gelöscht. Die 136 vorhandenen Gesprächsabfragen bleiben Gesprächsverlauf. Quellenunterlagen und Lizenzen liegen unter `docs/data_sources/`; `memory/evaluation/` enthält ausschließlich getrennte Evaluationsdefinitionen. Technische Tests erzeugen ihre kleinen Testbestände im Arbeitsspeicher; lokale Korpus-Integrationstests lesen direkt aus SQLite und werden ohne installierten zentralen Bestand übersprungen.

Die Konsolidierung prüft Speicherung und Vollständigkeit. Der vollständige Wellen-/Compileraufbau des wesentlich größeren Bestands wurde dabei nicht gestartet; die bisherigen Compiler- und CUDA-Zeitmessungen gelten für den früheren Bestand mit rund 20.000 Einträgen. Der Server blieb während der Konsolidierung beendet.

## Tests und Grenzen

Der [aktuelle Ergebnisbericht](results/unpaired_system/report.md) trennt Implementierung, Datenfreiheit von Paaren, numerische Kontrollen, Gesprächsqualität und den tatsächlichen Browserlauf. Frühere Ergebnisse, darunter der negative Wissens-Holdout des getrennten Informationsversuchs, bleiben im [historischen Bericht](results/information_corpus/report.md) erhalten.

Aktuell bestehen **462 technische Tests**. Der vollständige Festplattenaufbau und der Warmstart wurden mit allen 569.210 Dokumenten durchgeführt; Messungen und Grenzen stehen im [Laufzeitbericht zum großen Bestand](results/large_dataset_runtime.md). Im historischen, vorab zurückgehaltenen Gesprächstest waren jedoch nur **3 von 20 Antworten vollständig passend**, acht teilweise passend und neun leer. Die frühere Paarbaseline erreichte dort sieben vollständig passende Antworten. Diese Reparatur prüft die technische Funktion und numerische Parität; zuverlässige allgemeine Gesprächsfähigkeit ist damit noch nicht erreicht.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\consolidate_memory.py --verify-only
```

Die technische Suite prüft unter anderem atomare Migration, abgewiesene Paarimporte, Sitzungsisolierung, unveränderte Symbolfrequenzen bei Ergänzungen, Daten-/Promptablation und Übereinstimmung von FFT und direkter Rechnung. Gesprächs- und Wissensfragen besitzen getrennte Entwicklungs- und Abschlussmengen. Geöffnete Abschlussfragen werden bei späteren Versuchen als bekannte Regression geführt.

Der Generator verwendet einen begrenzten Satz expliziter sprachlicher Regeln. Er kann eine Information korrekt rekonstruieren, lediglich eine Benutzeraussage umformulieren, unpassend antworten oder sich enthalten. Rekonstruktion und grammatische Ausgabe beweisen kein neues Wissen. `max_tokens` zählt auch Satzzeichen, UTF-8-Bytesymbole und EOS; eine durch das Budget beendete Ausgabe ist unvollständig.

## Einrichtung

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[experiments,corpora]"
.\.venv\Scripts\python.exe -m freqai stats
```

Die vorhandene lokale SQLite-Memory wird direkt weiterverwendet. Sie ist absichtlich nicht in Git enthalten; ein neuer Checkout enthält daher keinen Wissensbestand. Neue Informationstexte können mit `freqai import DATEI.jsonl` aus einer externen Datei ergänzt werden. Historische Experimente, die entfernte Paar- oder Exportdateien voraussetzen, benötigen diese Eingaben ausdrücklich extern. Für einen manuellen Serverstart steht `start.ps1` bereit; die Größenbeschränkung der bisherigen Laufzeitmessungen ist oben dokumentiert. Ein Konsolenstart endet mit Strg+C. `stop.ps1` beendet den aufgezeichneten Hintergrundprozess nach Prüfung seiner Zugehörigkeit. Die Datenbank bleibt erhalten.
