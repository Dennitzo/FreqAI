# Fourier-Inferenz: numerischer Bericht

Zeitpunkt: 2026-09-06T06:50:32.010167+00:00; Seed: 20260906.

**768/768 Fälle bestanden**, 0 Fehler.
Verglichen wurden 96 zufällige Graphen mit 2 bis 64 Knoten gegen eine
unabhängige klassische Breitensuche. Enthalten sind Identitäten, gerichtete
Zyklen, unbekannte Knoten, nicht ableitbare Beziehungen und beschränkte Pfadlängen.
311 Fälle waren ableitbar und
457 waren unbelegt beziehungsweise unbekannt.

Der Operator ist H = F A F*, mit A[object, subject] = 1. Jeder Schritt berechnet
H @ v im Frequenzraum. Die inverse FFT liefert ganzzahlige Kantenzählungen;
eine kontrollierte boolesche Projektion bildet die nächste Front. Ein Beweispfad
aus den tatsächlichen Eingangstripeln begründet jede ausgegebene Behauptung.

Größter Ganzzahlfehler: 4.441e-15; größter imaginärer
Rest: 1.682e-15; Toleranz: 1.0e-08.
Numerische Fehlerfälle: 0.
Gesamtrechenzeit: 0.271 Sekunden.

Neue berechnete Antwort: **Pudel ist ein Tier.**
Beweis: Pudel → Hund → Tier.
Das direkte Tripel Pudel → Tier ist in den Eingangsdaten nicht enthalten.
Quellen: demo:taxonomy:1, demo:taxonomy:2. Die Quellen sind explizite lokale
Demofakten; ihre Kennungen sind keine externe wissenschaftliche Validierung.

Die Methode hat dieselbe Folgerungsfähigkeit wie Graph-Reichbarkeit. Die
Fouriertransformation allein erzeugt keine neue Semantik. Für V Knoten benötigt
der dichte Operator O(V²) Speicher und Rechenarbeit je Schritt. Semantik und
Antwortschablone sind vorgegeben. Die Lösung demonstriert eine neue, begründete
Antwort aus gespeicherten Beziehungen, keine allgemeine KI.
