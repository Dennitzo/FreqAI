# Ungepaarte Informationen und explizite Gesprächsgrammatik

`UnpairedWaveModel` verarbeitet deklarative Texte. Ein Datensatz darf keine Frage-/Antwort- oder Dialogfelder enthalten. Es gibt keinen vortrainierten Decoder und keine Optimierung von Parametern. Die Modellkoeffizienten werden aus Textsymbolen, beobachteten Übergängen und normalisierten Rollenfeldern direkt berechnet.

Das allein ergibt keine allgemeine Sprachintelligenz: Die deutsche Rollenextraktion, Sprechaktanalyse und Grammatik sind ausdrücklich programmierte Regeln. Fouriertransformationen ersetzen diese sprachlichen Annahmen nicht. Die numerischen Prüfungen unter `results/unpaired_system/numerical_audit.json` belegen die verwendete Rechnung und Datenabhängigkeit, keine allgemeine Gesprächsqualität.

## Was aus den Daten stammt

Aus einem Satz wie „Der Assistent heißt FreqAI.“ werden Subjekt, Prädikat und Argument extrahiert. „Das Wort Hallo ist ein Begrüßungswort.“ liefert die lexikalische Mitgliedschaft eines Wortes im Sprechakt Begrüßung. Der Datensatz enthält keine dazugehörige Benutzerfrage und keinen auszugebenden Gesprächssatz.

Die Rollen speichern die geordnete Symbolfolge ihres Subjekts beziehungsweise Prädikatarguments. Das ist eine verlustfreie Repräsentation eines Teils der Information und darf nicht mit neu erschlossenem Wissen verwechselt werden. Der Compiler speichert keine vollständigen Gesprächsantworten. Die Aussagen werden nicht aufgrund ihrer Ähnlichkeit zu einer hinterlegten Frage ausgewählt.

Für längere Informationstexte besteht weiterhin ein adressierbares Feld der beobachteten Wortübergänge. Ein Konzept und die erfragte Eigenschaft adressieren dessen kompatible Präfixfelder. Ein einzelner Compiler erzeugt sowohl diese Felder als auch deklarative Rollenfelder. Es gibt keinen zweiten Decoder für frühere Dialogpaare.

## Welche Grammatik vorgegeben ist

Die Grammatik kann unter anderem einen deklarierten Assistenten aus der dritten Person in die erste Person überführen: Subjekt → „ich“, „ist“ → „bin“, „hat“ → „habe“, „heißt“ → „heiße“. Das Argument wird aus den deklarativen Rollenmoden berechnet. Die Wortfolge „Ich heiße FreqAI.“ wird deshalb nicht vorher als vollständiger Antwortstring zusammengesetzt.

Ein Grammatikzustand beschreibt lediglich die aktuelle Rolle, ihre zulässigen lexikalischen Alternativen und die Position in dieser Rolle. Für jedes nächste Symbol werden alle kompatiblen Moden berücksichtigt. Nach Auswahl eines Symbols wird der Zustand weitergeschaltet. Das kann kurze neue Äußerungen wie eine Begrüßung mit einer anschließenden Frage erzeugen, obwohl der vollständige Satz in keinem Datensatz steht.

Auch die wenigen Regeln des Gesprächsverlaufs sind explizit: Begrüßung kann Begrüßung erwidern, eine Frage nach einer deklarierten Eigenschaft kann diese Eigenschaft ausgeben, Dank kann mit einem aus dem Datensatz bekannten Höflichkeitswort bestätigt werden. Das sind allgemeine Regeln für Sprechakte, keine aus Paarbeispielen gelernten Zuordnungen. Ihr begrenzter Umfang ist eine wesentliche Einschränkung.

Die zweite Implementierungsiteration ergänzt deutsche Verbzweitstellung, geteilte Subjekte, possessive Zustände und einfache Benutzerpräferenzen. Die Grammatik unterscheidet eine negative Benutzerpräferenz von einer negierten Wissensfrage: Eine berichtete Präferenz wird im Gespräch bewahrt, während eine unbekannte Wahrheitsbedingung weiterhin zur Enthaltung führen kann. Höfliche Einleitungen fachlicher Fragen ändern deren Gegenstand nicht in eine Frage nach Fähigkeiten des Assistenten. Ein zusätzliches deklaratives Lexikon klassifiziert auch mehrteilige Wendungen; der Parser verlangt deren vollständige Wortfolge.

Bei einer bekannten Tätigkeit und offenem Gegenstand kann die Grammatik eine Rückfrage aus Fragewort, Modalverb, Anrede und der Tätigkeitsrolle des Datensatzes bilden. Sie löst damit noch keine beliebige Planungs- oder Beratungsaufgabe. Das Bestätigen einer geäußerten Einschränkung speichert deren Wortlaut als Sitzungsinformation; es ist kein allgemeiner logischer Beweis, dass alle zukünftigen Antworten jede natürlichsprachliche Einschränkung erfüllen. Bloßes Spiegeln einer Aussage zählt in der Auswertung entsprechend nur als begrenzte Gesprächsleistung.

Deklarative Attribute können sowohl direkt beim Subjekt stehen als auch über eine Besitzbeziehung angebunden sein: „Die Eigenschaft von X beträgt Y.“ Die Eigenschaft und der in einer getrennten Aussage beschriebene Typ von X werden aus diesen Aussagen übernommen. Ein eingesetzter Zahlenwert kann die Beziehung auch in Gegenrichtung adressieren. Dieser begrenzte relationale Schluss benötigt keine gespeicherte Frage, ersetzt aber keine allgemeine Schlussfolgerungslogik.

Die Ausgabeoberfläche unterscheidet tatsächliche Grammatikpositionen von Faktenrollen. Beispielsweise wird die Funktionsform „bist“ kleingeschrieben, während der gleichnamige Fluss „Bist“ seinen Eigennamen behält. Bekannte Wörter aus temporären Benutzerrollen behalten ihre im Prompt beobachtete Schreibweise. Diese Oberflächenbehandlung verändert die numerische Symbolauswahl nicht.

## Numerische Rollenfelder

Eine lexikalische Rolle besitzt $L_r$ Positionen mit Symboladressen $\pi_r(j)$. Die Einheitsamplituden werden direkt als

$$a_r(j)=1/\sqrt{L_r},\qquad C_r=\mathcal F_{L_r}(a_r)$$

gespeichert. Die Adressen verbinden die Positionen mit stabilen Symbolfrequenzen. Die Wörter werden weder durch zufällige Frequenzähnlichkeit noch durch eine unbekannte Naturkonstante zu Bedeutungen. Diese Adressierung ist eine explizite Kodierung.

Im Grammatikzustand $s_t$ ergibt die Überlagerung aller zulässigen Rollenmoden die nächsten Symbolamplituden:

$$A_t(v)=\sum_{(r,j)\in s_t:\pi_r(j)=v}\mathcal F^{-1}(C_r)_j.$$

Die tatsächliche Energie der zugehörigen deklarativen Evidenzfelder lizenziert auch die Funktionswörter. Bei vollständig gelöschten Evidenzkoeffizienten entsteht deshalb keine scheinbar datengestützte Antwort.

Für die aktiven Symbolfrequenzen gilt der Zeitträger $U_t(v)=e^{2\pi i f_v t}$. Aus dem Prompt und dem gespeicherten Gesprächsfeld entsteht eine normierte Führung $b_t(v)$. Der Operator berechnet

$$R_t=\mathcal F(A_tU_t)+e^{i\varphi}\left[\mathcal F(A_tU_t)*_F\mathcal F(b_t)\right],$$

wobei $*_F$ die zur unitären FFT passende normierte zirkuläre Faltung ist. Danach gilt

$$p_t(v)=\frac{|\mathcal F^{-1}(R_t)_v|^2}{\sum_w|\mathcal F^{-1}(R_t)_w|^2}.$$

Die direkte Kontrollrechnung verwendet $\mathcal F[A_tU_t(1+e^{i\varphi}b_t)]$. Beide Wege müssen dieselbe Verteilung ergeben. Die Zeitphase bewegt das Feld, ohne im idealen Operator die Symbolwahrscheinlichkeiten zu verändern. Eine relative Phase zwischen Prompt- und Datenfeld kann dagegen Alternativen verändern oder vollständig auslöschen. In den Kontrollen verändert eine Phase von $\pi$ das ausgewählte Begrüßungswort.

Für normale Prosa bleiben die Koeffizienten $C=\mathcal F(\sqrt{n/\sum n})$ der beobachteten Präfixübergänge erhalten. Auch hier werden keine Koeffizienten durch einen Optimierer angepasst. Normalisierung, Grammatik, Präfixordnung, Beamweite und Symbolbudget bleiben ausdrücklich vorgegebene Algorithmusparameter.

## Temporäre Benutzerinformationen und Symbolbudget

Eine Selbstaussage wie „Ich bin müde.“ oder „Ich heiße Xyrländo.“ wird als vom Benutzer geäußerte Information im jeweiligen Gespräch gespeichert. Sie wird nicht als unabhängig bestätigtes Weltwissen in den zentralen Korpus übernommen. Die daraus erzeugten Rollen befinden sich nur im Promptfeld; zwischen Sitzungen werden keine temporären Rollen im gemeinsamen Modell behalten.

Unbekannte Wörter werden über ein festes Alphabet von 256 UTF-8-Bytesymbolen kodiert. Bei vollständiger Symbolfolge ist die Wiedergabe verlustfrei. Dafür wird das globale Vokabular während eines Gesprächs nicht verändert. Diese Möglichkeit erlaubt das Wiedergeben einer expliziten neuen Zeichenfolge; sie erschließt keine unbekannte Bedeutung.

`max_tokens` begrenzt Decoderschritte einschließlich Wort-, Satzzeichen-, UTF-8-Byte- und EOS-Symbolen. Ein unbekannter Name kann daher mehr Schritte benötigen als ein bekanntes Wort. Das Ergebnis meldet `tokenization.decoder_steps`, `utf8_byte_symbols` und `rendered_lexical_tokens` getrennt. Ein Ende durch Symbolbudget ist keine abgeschlossene Antwort. Endet das Budget innerhalb eines UTF-8-Zeichens, zeigt die Oberfläche nur den gültigen Präfix und meldet die weggelassenen Bytes unter `utf8_truncated_bytes`; sie erfindet kein Ersatzzeichen. Eine ungültige Bytefolge mitten in einer vollständigen Rolle wird als Decoderfehler gemeldet.

Die Mathematik verwendet weiterhin Zahlen, die als Gewichte von Moden wirken. Der nachweisbare Unterschied zu einem trainierten Modell lautet deshalb **direkt berechnete Koeffizienten ohne Parameteroptimierung**, nicht „keine Gewichte“.
