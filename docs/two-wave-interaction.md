# Zwei Schwingungen und ihre Interferenz

Dieses Dokument beschreibt die aktive Berechnung nach dem Umbau: eine Schwingung
aus **allen Daten**, eine Schwingung aus **Prompt und Sitzungskontext**, und die
gemeinsame Ablesung, aus der die Antwort eines Prompts wird. Code:
`freqai/waves.py`, `freqai/information.py`, `freqai/unpaired.py`,
`freqai/unpaired_runtime.py`. Tests: `tests/test_two_wave_interference.py`.

## Das Medium

Ein Symbol \(k\) ist eine eigene Mode des Mediums. Jede Mode trägt die feste
 Frequenz \(f_k=0{,}25+29{,}75\,\mathrm{hash}(k)\) Hz aus
`spectral_ops.stable_frequency`. Diese Frequenz bleibt bei jeder Korpus-Erweiterung
gleich; sie ist eine Adressierung, keine Bedeutung.

Deshalb gilt: zwei Wellen in diesem Medium koppeln nur über **gemeinsame Moden**.
Eine Verschiebung der Trägerphase pro Mode ist in der gemessenen Intensität
unbeobachtbar – die Antwort hängt nicht von der Wanduhr ab. Das ist ein Satz, kein
Trick; `test_measured_intensity_is_invariant_to_the_carrier_clock` prüft ihn.

Ein Umweg über Summen- und Differenzfrequenzen (`f_i + f_j ≈ f_k`) trägt dagegen
**keine Information**: Bei 117.055 Moden im Bereich von rund 30 Hz liegen etwa
4000 Moden pro Hertz, also trifft jede Mischbedingung auf tausende fremde Symbole.
Solche Mischterme werden nicht in die Wahrscheinlichkeiten aufgenommen.

## Die zwei Wellen

**Datenwelle** \(D\): Amplitude pro Mode aus dem *gesamten* kompilierten Bestand.

```
E_k = Σ über alle Prosa-Übergangsfelder |a_k(Feld)|²  +  Σ über alle deklarativen Korpus-Rollen (Vorkommen/Länge)
d_k = sqrt(E_k / Σ_j E_j),   D_k(t) = d_k · exp(2πi f_k t),   Σ_k |D_k|² = 1
```

Rollen der Funktionswort-Grammatik und kurzlebige Sitzungsrollen sind **keine**
Daten: Sie beschreiben Regeln über einen Satz, nicht den Bestand. Ein Prompt aus
einer laufenden Unterhaltung verändert die Datenwelle daher nicht.

**Promptwelle** \(P\): Amplituden aus dem Prompt (unbekannte Zeichenfolgen über das
feste UTF-8-Bytealphabet) plus dem gespeicherten Sitzungskontextfeld, auf Einheitsenergie:

```
p_k aus prompt_field,   P_k(t) = p_k · exp(2πi f_k t),   Σ_k |P_k|² = 1
```

## Die Ablesung pro Symbol

Ein Feld \(a_F\) ist die kohärente Amplitude eines kompilierten Übergangsfeldes
(`sqrt(zähler/summe)`, Einheitsnorm). Mit \(\hat p_k\) als normierten Promptamplituden
und \(\hat d_k\) als auf der Kandidatenmenge normierter Datenwellen-Amplitude:

```
R_F   = |⟨a_F , P⟩|                                        Resonanz des Feldes mit der Promptwelle
A_k   = Σ_F  w_F · (ρ + R_F) · a_{F,k}                     kohärente Superposition, ρ = resonance_floor = 0.25
ψ_k   = A_k · (1 + γ_p e^{iφ} \hat p_k) · (1 + γ_d \hat d_k)    Interferenz mit beiden Wellen
p_k   = |ψ_k|² / Σ_j |ψ_j|²                                gemessene Intensität
```

\(γ_p=1.0\) (`prompt_gain`), \(γ_d=0.15\) (`data_gain`), \(φ\) (`phase_error`). Zwei
Terme wirken getrennt: Die Promptwelle verstärkt jede Kandidatenmode, die sie
miterregt; die Datenwelle wirkt als Korpusfrequenz-Erwartung über den Kandidaten.
Beide sind in `tests/test_two_wave_interference.py` einzeln ablesbar (Ablation
`prompt_gain=0` bzw. `data_gain=0`).

Im Rollenpfad (`UnpairedWaveModel`) liefert der Grammatik-Automat die Kandidaten und
das Deklarations-Gate \(\sqrt{gate/E}\) die Amplituden; danach folgt **dieselbe**
Interferenz-Ablesung. Wissensfragen und Alltagssätze sind damit zwei Antworten auf ein
und dieselbe Wechselwirkung zweier Felder.

## Rechnerische Prüfung

Der spektrale Operator rechnet nie im Modenraum, sondern nur mit Spektren. Verwendet
wird das Mischungs-Theorem für die orthonormale Transformation \(F_u\):

```
spectral_convolution(F_u x , F_u y) = F_u(x · y)
```

Damit ist der Operatorpfad eine Summe von vier Spektraltermen, einer pro Produkt der
beiden Kopplungen; der direkte Pfad multipliziert im Modenraum. Gemessene
Abweichung beider Wege: **1,2 × 10⁻¹⁵** (Theorem-Test), und in jeder einzelnen
Antwort alle Schritte ≤ **1 × 10⁻¹¹** (`VERIFY_TOLERANCE`). Weichen sie ab, bricht
die Rechnung mit `Spectral operator and direct readout disagree`; eine stille
Mittelbildung gibt es nicht.

Zusätzlich geprüft: Einheitsenergie beider Wellen (Parseval), exakte Normierung der
Intensität auf 1, Endlichkeit aller Werte, Fingerprint-Stabilität beim Neuaufbau und
Trennbarkeit verschiedener Korpora und Prompts.

## Gefundene und behobene Defekte des früheren Leseverfahrens

1. **Eins-Moden-Degeneriertheit.** Das alte Verfahren benutzte nur die längste
   verfügbare Präfixtiefe und projizierte das Promptfeld auf diese eine Mode. Nach der
   Normierung fiel jeder Interferenzterm weg: `TV(prompt an / aus) = 0.0` in fast
   jedem Schritt (gemessen). Heute sind es im Prosa-Fixkörper z. B. 11 aktive Moden pro
   Schritt, und jede Mehrmoden-Stufe reagiert messbar auf mindestens eine der beiden
   Wellen.
2. **Verlust der Datenherkunft.** Ohne Akkumulation über alle Felder gab es kein Feld
   „alle Daten“. `symbol_energy` entsteht jetzt beim Kompilieren aus allen Feldern,
   ergänzt um die Rollenenergie; die Datenwelle ist ein Objekt mit Fingerabdruck.
3. **Frühe Satzenden bei Mehrordnungs-Superposition.** Alle Ordnungen gleichzeitig zu
   superponieren bringt EOS-Kandidaten aus flachen Feldern ein. Der Beam-Score ist eine
   kumulierte Log-Wahrscheinlichkeit, und ein Pfad, der früher endet, hat weniger
   Multiplikationsterme – er gewinnt ohne Gegenmittel. Beobachtung am Fixkörper:
   `Echos folgen aufeinander, z.` statt des vollständigen Satzes. Deshalb ist
   `order_policy="deepest"` der Default (nur das spezifischste Feld darf beenden),
   `"all"` bleibt als geprüfte Experimentieroption und braucht längennormalisierte
   Suche, wenn man sie nutzen will.

## Messwerte und Grenzen

| Prüfung | Ergebnis |
|---|---|
| Technische Tests | 345 bestanden (321 bestehende + 24 neue) |
| Mischungs-Theorem | 1,2 × 10⁻¹⁵ |
| Operator vs. direct je Antwort | ≤ 1 × 10⁻¹¹ |
| Zufallsfaktten (12 Texte, 32 Fälle) | 28/28 bewertet bestanden, 4 Negationsfälle wie dokumentiert unbeantwortet |

Off-diagonale Mischung über Hashfrequenzen liefert keine Bedeutung; ein Prompt ohne
Bindung an deklarierte Daten bleibt eine leere Antwort mit Grund. Die Interaktion ist
verifiziert – Sprachverständnis ist das nicht.
