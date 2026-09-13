# Zentrale Memory und Datensatzbereinigung – 13.09.2026

Die vier vollständig lokal vorhandenen Wikipedia-Dateien wurden ohne Sampling und ohne Kürzung der Artikeltexte in `memory/memory.sqlite3` übernommen. Erst nach dem vollständigen Vergleich mit den Quelldateien wurden die verstreuten Datensätze und Backups entfernt. Es wurde keine neue Sicherung angelegt und kein zusätzlicher Wikipedia-Download gestartet.

| Zentraler Inhalt | Anzahl |
|---|---:|
| Vollständige Artikel aus den vier Wikipedia-Shards | 569.062 |
| Zusätzliche vorher bereits gespeicherte Wikipedia-API-Einträge | 55 |
| Eigene deklarative Gesprächsinformationen | 45 |
| Lexikalische Informationstexte | 48 |
| **Aktive Informationseinträge insgesamt** | **569.210** |
| Alte archivierte Trainingsdatensätze nach Bereinigung | 0 |
| Erhaltener Gesprächsverlauf | 136 Abfragen |

Die 55 API-Einträge behalten den zuvor lokal vorhandenen Textumfang. Vollständige Originalartikel standen für diese zusätzlichen Einträge nicht in den vier Dateien zur Verfügung. Die Übernahme aller dort vorhandenen Volltexte betrifft exakt 569.062 Artikel.

Die 20.144 vorherigen Einleitungen aus diesen Shards wurden innerhalb derselben SQLite-Transaktion durch vollständige Artikel ersetzt. Dokumentadressen, Titel, Artikel-URLs, Dataset-Version, Originaltext- und Speichertextprüfsummen stehen in der zentralen Memory. Die einzige Texttransformation ist: Originaltitel, zwei Zeilenumbrüche und vollständiger unveränderter Quelltext. Weitere Absätze, Überschriften und Listen sind enthalten. Frage-/Antwortfelder wurden nicht als Informationstexte umgedeutet.

`information_provenance` enthält die Herkunft je Dokument; `information_imports` enthält Datei-Prüfsummen und Importnachweise. Der frühere `document_archive` wurde auf ausdrücklichen Benutzerwunsch geleert. Der Gesprächsverlauf ist davon getrennt. Die aktive Revision stieg von 10 auf 11.

**Nachweise vor der Löschung:** Alle 569.062 Artikel wurden nochmals aus den Parquet-Dateien gelesen und einzeln mit dem zentral gespeicherten Volltext, der Quelle und den SHA-256-Werten verglichen. SQLite meldete `integrity_check=ok` und keine Fremdschlüsselfehler. Zusätzlich wurden alle drei exportierten Korpusstufen, die früheren Wikipedia-Importversionen und die eigenen Informationsdateien gegen ihren erhaltenen zentralen Inhalt beziehungsweise vollständigen Quellartikel geprüft. [Importnachweis](/C:/Users/AMD/Documents/GitHub/FreqAI/results/memory_consolidation.json), [Abdeckungs- und Löschmanifest](/C:/Users/AMD/Documents/GitHub/FreqAI/results/memory_cleanup_plan.json).

**Aufräumen:** 90 Dateien mit 2.528.747.674 Bytes wurden entfernt. Darunter sind alle vier Wikipedia-Rohdateien, die 20k-/60k-/100k-Exportstufen, alte JSONL-Importdateien, historische Datensatzkopien in Ergebnisordnern, Paar-/Sprachprior-Datensätze, die Datensatz-Backups und der ungültig gewordene Compiler-Cache. Jede Löschdatei wurde unmittelbar vorher anhand ihres Projektpfads, ihrer Größe und ihres Hashes geprüft. Die zentrale Datenbank war vom Löschbereich ausgeschlossen. 16 Quellenunterlagen, Lizenzen und historische Importrezepte wurden nach `docs/data_sources/` verschoben. [Löschergebnis](/C:/Users/AMD/Documents/GitHub/FreqAI/results/memory_cleanup_result.json).

**Aktuelle Ablage:** Unter `memory/` liegen die zentrale SQLite-Datei und getrennte Evaluationsdefinitionen. Der Ordner `dataset/` und die alten Import-, Quell- und Backup-Verzeichnisse sind entfernt. Numerische Tests erzeugen kleine synthetische Datensätze im Arbeitsspeicher; Korpus-Integrationstests lesen gezielt aus der zentralen SQLite-Memory. Es werden keine importierten Korpuskopien als Testdateien vorgehalten. Die SQLite-Datei bleibt durch `.gitignore` vom normalen Git-Tracking ausgeschlossen. Die Bereinigung betrifft das Arbeitsverzeichnis; die Git-Versionshistorie wurde nicht umgeschrieben.

**Prüfungen:** Nach Entfernung der Datensatzdateien bestehen **449 Tests in 24,95 Sekunden**. Geprüft werden unter anderem vollständige Artikelübernahme, Wiederholungsimporte, Konflikterkennung, Rollback bei unzulässigen QA-Feldern, erhaltene Informationsmetadaten, numerische Invarianten und die Web-API mit kleinen Testbeständen. [Testprotokoll](/C:/Users/AMD/Documents/GitHub/FreqAI/runtime/consolidation_final_tests.log).

Die Datenbank wurde anschließend mit SQLite `VACUUM` auf **6.879.019.008 Bytes** komprimiert. Die enthaltenen Texte und Quellenprüfsummen können ohne die entfernten Rohdateien erneut kontrolliert werden:

```powershell
.\.venv\Scripts\python.exe scripts\consolidate_memory.py --verify-only
```

Diese spätere Kontrolle wurde nach Komprimierung und Löschung erfolgreich ausgeführt: **569.062 Artikelprüfsummen bestätigt, SQLite-Integrität und Fremdschlüsselprüfung fehlerfrei**. Sie prüft die Datenbank gegen die aufbewahrten Artikelprüfsummen und Importnachweise. Der unabhängige Vergleich mit den Originaldateien wurde ausdrücklich vor deren Löschung durchgeführt. [Kontrolle nach dem Aufräumen](/C:/Users/AMD/Documents/GitHub/FreqAI/runtime/consolidation_catalog_verification.json).

**Laufzeitgrenze:** Der Server war bereits beendet und wurde nicht gestartet. Ein vollständiger Wellen-/Compileraufbau für die 569.210 Einträge wurde nicht durchgeführt. Die früheren Compiler- und GPU-Benchmarks mit rund 20.000 Einträgen belegen keine Laufzeit oder Speichergrenze für diesen erheblich größeren Volltextbestand. Die bisherigen Gesprächsqualitätswerte sind ebenfalls keine Bewertung des neuen Bestands.
