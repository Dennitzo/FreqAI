# Wortweise Textgenerierung aus einem spektralen Übergangsmodell

60 neue Gesprächseingaben wurden vor der Implementierungsbewertung festgelegt: 30 Entwicklungs- und 30 zurückgehaltene Turns. Die Antworten werden einzeln auf Grammatik, Bezug, Kohärenz und korrekte Sprecherrollen/Verneinungen geprüft. Eine neue Tokenfolge ist kein Qualitätsnachweis.

| Lauf | Sprachform (automatisch) | Text-Hinweise (automatisch) | Sinnvolle Antwort (Review) | Vollständiger Text neu |
| --- | ---: | ---: | ---: | ---: |
| Bisherige Regeln, development | 30/30 | 16/30 | 12/30 | 27/30 |
| Wellendecoder, development | 29/30 | 13/30 | 13/30 | 5/30 |
| Wellendecoder, holdout | 25/30 | 11/30 | 6/30 | 2/30 |

Die automatischen Sprachformtests prüfen nur Textoberfläche, Satzabschluss und Schleifen. Die Text-Hinweise prüfen wenige Wörter und ausdrücklich verbotene Formulierungen. Beide können unpassende oder widersprüchliche Sätze übersehen. Die Review-Urteile sind qualitative Einzelfallurteile, keine unabhängige Nutzerstudie.

Neuheit bezieht sich auf die vollständigen Textdaten des jeweiligen Laufs. Beim bisherigen Regelmodell sind externe Grammatikfragmente nicht in diesem Vergleichskorpus enthalten; dessen Neuheitszahl ist deshalb nicht mit freier Sprachgenerierung gleichzusetzen.

## Einzelbewertungen

### Bisherige Regeln: development

- **gw-dev-01 (bestanden)** — Guten Tag! Ich hätte Lust auf ein kleines Gespräch.
  Antwort: Guten Abend! Gern. Was beschäftigt dich gerade oder worüber möchtest du sprechen?
  Bewertung: Begruessung und Gespraechseinladung passen; die Tageszeit wird ohne Bedarf konkretisiert.
- **gw-dev-02 (nicht bestanden)** — Mein Tag war angenehm, und deiner?
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die klare Aussage zum angenehmen Tag und die Rueckfrage bleiben unbeantwortet.
- **gw-dev-03 (nicht bestanden)** — Danke, das war eine nette Begrüßung.
  Antwort: Gern. Du kannst in deinem Tempo weitererzählen. Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Ein passender Anfang wird mit einer unbegruendeten Verstaendnisfrage verbunden.
- **gw-dev-04 (bestanden)** — Nach dem langen Tag bin ich ziemlich erschöpft.
  Antwort: Dann wäre vielleicht ein ruhiger Moment angenehm. War dein Tag besonders anstrengend?
  Bewertung: Muedigkeit wird passend aufgegriffen und eine ruhige Nachfrage angeboten.
- **gw-dev-05 (bestanden)** — Jetzt bin ich aber nicht mehr müde.
  Antwort: Du sagst, dass du dich nicht müde fühlst. Magst du mehr darüber erzählen?
  Bewertung: Die beendete Muedigkeit wird ohne gegenteilige Behauptung wiedergegeben.
- **gw-dev-06 (nicht bestanden)** — Ich würde heute gern noch etwas draußen machen.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die klare Absicht einer Aktivitaet draussen fuehrt nur zur pauschalen Rueckfrage.
- **gw-dev-07 (bestanden)** — Abends höre ich am liebsten ruhige Musik.
  Antwort: Was beschäftigt dich beim Thema Musik?
  Bewertung: Die Nachfrage bleibt beim genannten Musikthema.
- **gw-dev-08 (nicht bestanden)** — Gesang mag ich dabei nicht besonders.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die Abneigung gegen Gesang wird nicht aufgegriffen.
- **gw-dev-09 (nicht bestanden)** — Und was hörst du selbst gerne?
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die Frage nach eigenen Hoergewohnheiten wird nicht beantwortet.
- **gw-dev-10 (nicht bestanden)** — Ich habe Hunger und möchte etwas Einfaches kochen.
  Antwort: Was beschäftigt dich beim Thema Essen und Kochen?
  Bewertung: Die Antwort wiederholt nur die grobe Essenskategorie statt eine einfache Mahlzeit oder Zutaten zu besprechen.
- **gw-dev-11 (nicht bestanden)** — Fleisch möchte ich dafür nicht verwenden.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die fleischfreie Praeferenz wird nicht verstanden.
- **gw-dev-12 (nicht bestanden)** — Nach dem Essen würde ich gern noch plaudern.
  Antwort: Was beschäftigt dich beim Thema Essen und Kochen?
  Bewertung: Die Antwort bleibt beim Essen und geht nicht auf die gewuenschte Gespraechsfortsetzung ein.
- **gw-dev-13 (bestanden)** — Bei der Arbeit war heute ziemlich viel los.
  Antwort: Was beschäftigt dich beim Thema Arbeit?
  Bewertung: Die Nachfrage nach dem Arbeitstag ist passend.
- **gw-dev-14 (bestanden)** — Trotzdem bin ich mit dem Ergebnis zufrieden.
  Antwort: Das ist ein schöner Moment. Was hat zu dieser Zufriedenheit beigetragen?
  Bewertung: Zufriedenheit wird passend anerkannt und vertieft.
- **gw-dev-15 (nicht bestanden)** — Für morgen wünsche ich mir einen ruhigeren Start.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Der Wunsch nach einem ruhigeren Morgen wird nicht aufgegriffen.
- **gw-dev-16 (nicht bestanden)** — Hier regnet es seit Stunden, deshalb bleibe ich drinnen.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Weder Regen noch der Plan, drinnen zu bleiben, wird aufgegriffen.
- **gw-dev-17 (bestanden)** — Ein Buch und ein warmer Tee wären jetzt schön.
  Antwort: Was beschäftigt dich beim Thema Bücher?
  Bewertung: Die Nachfrage zum Lesen passt zu einem der genannten Themen.
- **gw-dev-18 (bestanden)** — Über Bücher können wir doch ein wenig reden.
  Antwort: Was beschäftigt dich beim Thema Bücher?
  Bewertung: Die Antwort oeffnet ein Gespraech ueber Buecher.
- **gw-dev-19 (nicht bestanden)** — Heute fehlt mir ein bisschen Gesellschaft.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Der Wunsch nach Gesellschaft bleibt unbeantwortet.
- **gw-dev-20 (nicht bestanden)** — Vielleicht rufe ich nachher einen Freund an.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Der geplante Kontakt zu einem Freund bleibt unbeantwortet.
- **gw-dev-21 (bestanden)** — Bis dahin können wir uns noch unterhalten.
  Antwort: Gern. Was beschäftigt dich gerade oder worüber möchtest du sprechen?
  Bewertung: Die Fortsetzung des Gespraechs wird angenommen.
- **gw-dev-22 (nicht bestanden)** — Für unser Gespräch darfst du mich Lennart nennen.
  Antwort: Gern. Was beschäftigt dich gerade oder worüber möchtest du sprechen?
  Bewertung: Die ausdrueckliche Bitte, den Namen Lennart zu verwenden, wird uebergangen.
- **gw-dev-23 (nicht bestanden)** — Mich interessiert, wie du dich selbst beschreiben würdest.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die Selbstbeschreibung des Programms fehlt.
- **gw-dev-24 (nicht bestanden)** — Welchen Namen hatte ich dir vorhin genannt?
  Antwort: Du hast mir deinen Namen in diesem Gespräch noch nicht genannt. Wie heißt du?
  Bewertung: Die Antwort behauptet faelschlich, es sei kein Name genannt worden.
- **gw-dev-25 (bestanden)** — Ich bin heute nicht traurig, nur etwas still.
  Antwort: Du sagst, dass du dich nicht traurig fühlst. Magst du mehr darüber erzählen? Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die Verneinung von Traurigkeit wird erhalten; die zusaetzliche Rueckfrage ist redundant, aber ohne Gegenbehauptung.
- **gw-dev-26 (bestanden)** — Mir tut ein ruhiges Gespräch gerade gut.
  Antwort: Gern. Was beschäftigt dich gerade oder worüber möchtest du sprechen?
  Bewertung: Eine ruhige Gespraechsfortsetzung wird angeboten.
- **gw-dev-27 (nicht bestanden)** — Dabei möchte ich gar keine Ratschläge haben.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die Bitte, keine Ratschlaege zu geben, wird nicht anerkannt.
- **gw-dev-28 (bestanden)** — Ich habe gerade eine kleine gute Nachricht bekommen.
  Antwort: Wie schön! Magst du erzählen, worum es geht?
  Bewertung: Die gute Nachricht wird passend aufgegriffen.
- **gw-dev-29 (nicht bestanden)** — Darüber freue ich mich wirklich sehr.
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die weiter geaeusserte Freude wird nicht aufgegriffen.
- **gw-dev-30 (nicht bestanden)** — Für jetzt verabschiede ich mich. Bis ein anderes Mal!
  Antwort: Diesen Teil habe ich noch nicht sicher verstanden. Magst du ihn genauer beschreiben?
  Bewertung: Die klare Verabschiedung wird nicht erkannt.

### Wellendecoder: development

- **gw-dev-01 (bestanden)** — Guten Tag! Ich hätte Lust auf ein kleines Gespräch.
  Antwort: Guten Abend, wir können gern ein wenig plaudern. Natürlich.
  Bewertung: Begruessung und Einladung passen. Das abschliessende Natuerlich ist redundant; die Tageszeit wird ohne Bedarf konkretisiert.
- **gw-dev-02 (nicht bestanden)** — Mein Tag war angenehm, und deiner?
  Antwort: Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne?
  Bewertung: Die Antwort wechselt ohne Grundlage zu Regen und Sonne und beantwortet die Rueckfrage nicht.
- **gw-dev-03 (bestanden)** — Danke, das war eine nette Begrüßung.
  Antwort: Gern. Du kannst jederzeit einen neuen Gedanken einbringen.
  Bewertung: Die Rueckmeldung wird anerkannt und ein weiterer Gedanke eingeladen.
- **gw-dev-04 (bestanden)** — Nach dem langen Tag bin ich ziemlich erschöpft.
  Antwort: Wenn du erschöpft bist, kann ein ruhiger Moment angenehm sein. Was kannst du für einen Moment liegen lassen?
  Bewertung: Der neue Satz verbindet Erschoepfung mit einem ruhigen Moment grammatisch und inhaltlich passend; er kommt so nicht im Korpus vor.
- **gw-dev-05 (nicht bestanden)** — Jetzt bin ich aber nicht mehr müde.
  Antwort: Wenn du keine Ratschläge möchtest, kann ich zunächst zuhören. Wie würdest du es mit deinen eigenen Worten beschreiben?
  Bewertung: Die Antwort wechselt zur Ablehnung von Ratschlaegen statt die beendete Muedigkeit aufzugreifen.
- **gw-dev-06 (bestanden)** — Ich würde heute gern noch etwas draußen machen.
  Antwort: Das klingt nach einer angenehmen Pause. Zieht es dich eher ins Grüne oder durch die Nachbarschaft?
  Bewertung: Die Nachfrage nach Gruenem oder Nachbarschaft passt zur gewuenschten Aktivitaet draussen.
- **gw-dev-07 (nicht bestanden)** — Abends höre ich am liebsten ruhige Musik.
  Antwort: Das klingt nach einer schönen kleinen Unterbrechung. Hörst du ihnen gerade bewusst zu?
  Bewertung: Das Pronomen ihnen hat bei der genannten Musik keinen klaren Bezug; die Antwort stammt aus einer anderen Hoersituation.
- **gw-dev-08 (nicht bestanden)** — Gesang mag ich dabei nicht besonders.
  Antwort: Das ist ein markanter Moment nach einem Schauer. Magst du dann gern noch kurz nach draußen gehen?
  Bewertung: Ein Schauer und ein Gang nach draussen haben keinen Bezug zur Gesangsabneigung.
- **gw-dev-09 (nicht bestanden)** — Und was hörst du selbst gerne?
  Antwort: Wie schön! Was hat dich besonders gefreut: die Kruste, der Duft oder der erste Bissen?
  Bewertung: Kruste und erster Bissen haben keinen Bezug zur Frage nach eigenen Hoergewohnheiten.
- **gw-dev-10 (nicht bestanden)** — Ich habe Hunger und möchte etwas Einfaches kochen.
  Antwort: Das klingt nach Freude am Experimentieren. Was hast du zuletzt ausprobiert?
  Bewertung: Die Antwort unterstellt Freude am Experimentieren statt nach der einfachen Mahlzeit oder Zutaten zu fragen.
- **gw-dev-11 (nicht bestanden)** — Fleisch möchte ich dafür nicht verwenden.
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben; das ist transparent, erfuellt aber die Gespraechsabsicht nicht.
- **gw-dev-12 (nicht bestanden)** — Nach dem Essen würde ich gern noch plaudern.
  Antwort: Das klingt nach Freude am Experimentieren. Was hast du zuletzt ausprobiert?
  Bewertung: Die Antwort behandelt Experimentieren statt die Gespraechsfortsetzung nach dem Essen.
- **gw-dev-13 (nicht bestanden)** — Bei der Arbeit war heute ziemlich viel los.
  Antwort: Das ist schön zu hören. Welcher Teil der Rückmeldung hat dich besonders gefreut?
  Bewertung: Eine erfreuliche Rueckmeldung wird unterstellt, obwohl nur ein voller Arbeitstag genannt ist.
- **gw-dev-14 (bestanden)** — Trotzdem bin ich mit dem Ergebnis zufrieden.
  Antwort: Das ist ein schöner Moment. Was hat zu dieser Zufriedenheit beigetragen?
  Bewertung: Zufriedenheit wird korrekt anerkannt und vertieft.
- **gw-dev-15 (nicht bestanden)** — Für morgen wünsche ich mir einen ruhigeren Start.
  Antwort: Du musst heute nicht alles perfekt hinbekommen. Ein kleiner Schritt und eine Pause dürfen für diesen Moment reichen.
  Bewertung: Die Antwort verlagert den Wunsch fuer morgen auf Perfektion und Pausen heute.
- **gw-dev-16 (bestanden)** — Hier regnet es seit Stunden, deshalb bleibe ich drinnen.
  Antwort: Dann wirkt der Tag vielleicht etwas langsamer. Magst du Regentage oder fehlt dir heute die Sonne?
  Bewertung: Die Rueckfrage bleibt beim Regenwetter.
- **gw-dev-17 (bestanden)** — Ein Buch und ein warmer Tee wären jetzt schön.
  Antwort: Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre?
  Bewertung: Die Frage nach Figuren, Geschichte und Atmosphaere passt zu einem Buch.
- **gw-dev-18 (bestanden)** — Über Bücher können wir doch ein wenig reden.
  Antwort: Was zieht dich daran besonders rein: die Figuren, die Geschichte oder die Atmosphäre?
  Bewertung: Die Antwort eroeffnet ein passendes Gespraech ueber Geschichten.
- **gw-dev-19 (nicht bestanden)** — Heute fehlt mir ein bisschen Gesellschaft.
  Antwort: Dann müssen wir nicht alles gleichzeitig sortieren. Was würde deinen nächsten kleinen Schritt leichter machen?
  Bewertung: Die Antwort unterstellt ein Sortierproblem statt die fehlende Gesellschaft aufzugreifen.
- **gw-dev-20 (nicht bestanden)** — Vielleicht rufe ich nachher einen Freund an.
  Antwort: Das klingt nach einer angenehmen Pause. Zieht es dich eher ins Grüne oder durch die Nachbarschaft?
  Bewertung: Die Antwort fragt nach einem Spaziergang statt nach dem geplanten Anruf.
- **gw-dev-21 (bestanden)** — Bis dahin können wir uns noch unterhalten.
  Antwort: Wir können uns gern in Ruhe unterhalten. Was ist dir an diesem Thema wichtig?
  Bewertung: Die ruhige Gespraechsfortsetzung und Rueckfrage sind passend.
- **gw-dev-22 (nicht bestanden)** — Für unser Gespräch darfst du mich Lennart nennen.
  Antwort: Natürlich. Ich wünsche dir einen angenehmen restlichen Tag.
  Bewertung: Die Bitte um den Namen Lennart wird ignoriert; die Antwort wirkt wie ein Abschied.
- **gw-dev-23 (nicht bestanden)** — Mich interessiert, wie du dich selbst beschreiben würdest.
  Antwort: Wie schön! Was hat dich besonders gefreut: die Kruste, der Duft oder der erste Bissen?
  Bewertung: Die Antwort wechselt zur Brotkruste statt das Programm zu beschreiben.
- **gw-dev-24 (nicht bestanden)** — Welchen Namen hatte ich dir vorhin genannt?
  Antwort: Wenn du mir deinen Namen nennst, kann ich ihn im Gespräch verwenden. Welche Formulierung passt am besten zu dem, was du meintest?
  Bewertung: Die Antwort erinnert den Namen Lennart nicht und fragt stattdessen allgemein nach einer passenden Formulierung.
- **gw-dev-25 (bestanden)** — Ich bin heute nicht traurig, nur etwas still.
  Antwort: Wer still ist, muss deshalb nicht traurig sein. Was ist dir an dieser Unterscheidung wichtig?
  Bewertung: Die Verneinung von Traurigkeit bei stiller Stimmung wird korrekt aufgegriffen.
- **gw-dev-26 (nicht bestanden)** — Mir tut ein ruhiges Gespräch gerade gut.
  Antwort: Natürlich. Ich wünsche dir einen angenehmen restlichen Tag.
  Bewertung: Die Antwort wirkt wie ein Abschied, obwohl gerade ein angenehmes Gespraech fortgesetzt wird.
- **gw-dev-27 (bestanden)** — Dabei möchte ich gar keine Ratschläge haben.
  Antwort: Verstanden. Ich höre zu.
  Bewertung: Die Bitte um Zuhoeren ohne Ratschlaege wird passend anerkannt.
- **gw-dev-28 (bestanden)** — Ich habe gerade eine kleine gute Nachricht bekommen.
  Antwort: Wie schön! Magst du erzählen, worum es geht?
  Bewertung: Die gute Nachricht wird passend aufgegriffen.
- **gw-dev-29 (nicht bestanden)** — Darüber freue ich mich wirklich sehr.
  Antwort: Nimm dir die Zeit. Du musst nicht sofort etwas dazu sagen.
  Bewertung: Die Antwort nimmt die Freude nicht auf und behandelt stattdessen ein Zoegern beim Sprechen.
- **gw-dev-30 (bestanden)** — Für jetzt verabschiede ich mich. Bis ein anderes Mal!
  Antwort: Bis später! Wir können beim nächsten Mal neu anknüpfen.
  Bewertung: Die Verabschiedung wird passend erwidert.

### Wellendecoder: holdout

- **gw-holdout-01 (bestanden)** — Hi zusammen, ich schaue für ein paar Minuten hier rein.
  Antwort: Hi! Ich bin bereit für unser Gespräch.
  Bewertung: Die Begruessung und Gespraechsbereitschaft sind passend.
- **gw-holdout-02 (nicht bestanden)** — Bei mir ist alles entspannt. Wie steht es um dich?
  Antwort: Wie schön, dass gerade etwas Ruhe da ist. Möchtest du den Moment einfach genießen oder darüber erzählen?
  Bewertung: Die entspannte Stimmung wird aufgegriffen, aber die ausdrueckliche Rueckfrage an das Programm bleibt unbeantwortet.
- **gw-holdout-03 (nicht bestanden)** — Nett, dass du dir Zeit zum Zuhören nimmst.
  Antwort: Was magst du daran am liebsten: das Pflegen, das Wachsen oder die Ruhe dabei?
  Bewertung: Die Antwort wechselt unbegruendet zu Pflegen und Wachsen statt die Wertschaetzung anzuerkennen.
- **gw-holdout-04 (nicht bestanden)** — Meine Energie ist nach dem Aufräumen ziemlich aufgebraucht.
  Antwort: Vielleicht hilft ein kleiner Anfang: eine Fläche oder zehn Minuten. Welcher Bereich stört dich gerade am meisten?
  Bewertung: Die Antwort empfiehlt den Anfang des Aufraeumens, obwohl es bereits stattgefunden hat und Erschoepfung beschrieben wird.
- **gw-holdout-05 (nicht bestanden)** — Nach der Pause fühle ich mich nicht mehr erschöpft.
  Antwort: Eine Vorliebe für ruhige Musik bedeutet nicht, dass alle anderen Klänge schlecht sind. Was ist dir an dieser Unterscheidung wichtig?
  Bewertung: Die Antwort wechselt von beendeter Erschoepfung zu Musikvorlieben.
- **gw-holdout-06 (bestanden)** — Ein kurzer Spaziergang würde jetzt zu meiner Stimmung passen.
  Antwort: Das klingt nach einer angenehmen Pause. Zieht es dich eher ins Grüne oder durch die Nachbarschaft?
  Bewertung: Die Rueckfrage nach Gruenem oder Nachbarschaft passt zum gewuenschten Spaziergang.
- **gw-holdout-07 (nicht bestanden)** — Zum Abschalten lege ich gern leise Klavierstücke auf.
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben.
- **gw-holdout-08 (nicht bestanden)** — Laute Lieder sind dabei überhaupt nicht mein Fall.
  Antwort: Das klingt nach einer schönen kleinen Unterbrechung. Hörst du lieber aufmerksam zu oder läuft die Musik nebenbei?
  Bewertung: Die Antwort greift die Abneigung gegen laute Lieder nicht auf und spricht stattdessen von einer schoenen Unterbrechung.
- **gw-holdout-09 (nicht bestanden)** — Hast du eigentlich selbst einen Musikgeschmack?
  Antwort: Wie schön! Was hat dich besonders gefreut: die Kruste, der Duft oder der erste Bissen?
  Bewertung: Die Antwort wechselt vom Musikgeschmack des Programms zu einer Brotkruste.
- **gw-holdout-10 (nicht bestanden)** — Für heute Abend suche ich eine unkomplizierte Essensidee.
  Antwort: Guten Abend! Möchtest du von deinem Tag erzählen oder einfach ein bisschen plaudern?
  Bewertung: Die Antwort begruesst und eroeffnet ein Gespraech statt die Essensidee aufzugreifen.
- **gw-holdout-11 (nicht bestanden)** — Etwas ohne Fleisch wäre mir dabei am liebsten.
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben.
- **gw-holdout-12 (bestanden)** — Wenn ich fertig gegessen habe, erzähle ich dir mehr von meinem Tag.
  Antwort: Das ist in Ordnung. Wir können eine Pause machen oder bei etwas ganz Alltäglichem anfangen.
  Bewertung: Die Antwort akzeptiert die geplante Pause und Fortsetzung, ohne Ereignisse aus dem Tag zu erfinden.
- **gw-holdout-13 (nicht bestanden)** — Mein Arbeitstag bestand fast nur aus kleinen Unterbrechungen.
  Antwort: Das ist ein besonderer Anfang. Worauf freust du dich, und was macht dich vielleicht noch unsicher?
  Bewertung: Die Antwort unterstellt einen besonderen Anfang und Vorfreude statt auf Unterbrechungen bei der Arbeit einzugehen.
- **gw-holdout-14 (nicht bestanden)** — Dass ich trotzdem etwas geschafft habe, macht mich froh.
  Antwort: Das klingt nach etwas, worauf man sich freuen kann. Hast du schon eine Idee, was du damit anfangen möchtest?
  Bewertung: Die Antwort behandelt Vorfreude auf etwas noch zu Verwendendes statt Freude ueber eine abgeschlossene Leistung.
- **gw-holdout-15 (nicht bestanden)** — Am nächsten Morgen will ich mir zuerst einen ruhigen Moment gönnen.
  Antwort: Ein Tag ohne Pflichten kann sich gut anfühlen. Wie sähe ein angenehmer, freier Nachmittag für dich aus?
  Bewertung: Die Antwort verlagert einen ruhigen Morgen auf einen freien Nachmittag und unterstellt fehlende Pflichten.
- **gw-holdout-16 (nicht bestanden)** — Der Wind ist mir heute zu ungemütlich, ich mache es mir zu Hause bequem.
  Antwort: Das klingt nach einer angenehmen Pause. Zieht es dich eher ins Grüne oder durch die Nachbarschaft?
  Bewertung: Die Antwort schlaegt einen Gang nach draussen vor, obwohl die Person ausdruecklich wegen des Windes drinnen bleiben will.
- **gw-holdout-17 (bestanden)** — Vielleicht lese ich dabei ein paar Seiten und trinke Kakao.
  Antwort: Das klingt nach einem gemütlichen Abendritual. Welche Sorte magst du gerade?
  Bewertung: Ein gemuetliches Ritual und die Frage nach der Sorte Kakao passen zum genannten Getraenk und Lesen.
- **gw-holdout-18 (nicht bestanden)** — Was für Geschichten bieten sich als Gesprächsthema an?
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben.
- **gw-holdout-19 (nicht bestanden)** — So ein Abend allein kann sich ganz schön leer anfühlen.
  Antwort: Das ist ein schöner Moment. Was hat zu dieser Zufriedenheit beigetragen?
  Bewertung: Die Antwort unterstellt Zufriedenheit, obwohl Leere und Alleinsein beschrieben werden.
- **gw-holdout-20 (nicht bestanden)** — Ich könnte meiner Schwester später eine Nachricht schreiben.
  Antwort: Das klingt nach einer unerwarteten Begegnung im Alltag. Wie war das Gespräch für dich?
  Bewertung: Die Antwort behandelt den geplanten schriftlichen Kontakt als bereits erlebte Begegnung mit Gespraech.
- **gw-holdout-21 (nicht bestanden)** — Vorher bleibe ich gern noch ein bisschen im Gespräch mit dir.
  Antwort: Natürlich. Ich wünsche dir einen angenehmen restlichen Tag.
  Bewertung: Die Antwort wirkt wie ein Abschied, obwohl die Gespraechsfortsetzung gewuenscht wird.
- **gw-holdout-22 (nicht bestanden)** — Du kannst mich übrigens Marit nennen.
  Antwort: Ein persönlicher Name sollte genau so wiedergegeben werden, wie du ihn angibst. Was soll ich für den weiteren Zusammenhang beachten?
  Bewertung: Die allgemeine Aussage ueber Namen bestaetigt den genannten Namen Marit nicht.
- **gw-holdout-23 (bestanden)** — Was bist du eigentlich für ein Gesprächspartner?
  Antwort: Ich habe keinen eigenen Alltag und keine persönlichen Vorlieben. Was möchtest du über dieses Gesprächssystem wissen?
  Bewertung: Die Antwort beschreibt das Programm ohne eigenen Alltag und persoenliche Vorlieben wahrheitsgemaess.
- **gw-holdout-24 (nicht bestanden)** — Erinnerst du dich noch daran, wie ich genannt werden möchte?
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben; der zuvor genannte Name Marit wird nicht erinnert.
- **gw-holdout-25 (bestanden)** — Nur weil ich wenig rede, bin ich heute nicht unglücklich.
  Antwort: Nicht unglücklich zu sein lässt noch viele andere Stimmungen offen. Wie würdest du es mit deinen eigenen Worten beschreiben?
  Bewertung: Die Verneinung von Unglueck wird korrekt als offene Stimmung eingeordnet.
- **gw-holdout-26 (nicht bestanden)** — Ein Gespräch ohne Eile passt gerade gut zu mir.
  Antwort: Natürlich. Ich wünsche dir einen angenehmen restlichen Tag.
  Bewertung: Die Antwort wirkt wie ein Abschied statt ein ruhiges Gespraech anzubieten.
- **gw-holdout-27 (nicht bestanden)** — Lösungen suche ich nicht, ich will einfach etwas erzählen.
  Antwort: Ein Tag ohne Pflichten kann sich gut anfühlen. Wie sähe ein angenehmer, freier Nachmittag für dich aus?
  Bewertung: Die Antwort greift die ausdrueckliche Bitte um blosses Erzaehlen nicht auf und wechselt zu einem freien Nachmittag.
- **gw-holdout-28 (nicht bestanden)** — Eine unerwartete kleine Überraschung hat mich heute glücklich gemacht.
  Antwort: So etwas kann lange nachklingen. Was wurde dir gesagt?
  Bewertung: Die Antwort fragt nach etwas Gesagtem und setzt damit einen nicht genannten verbalen Anlass der Ueberraschung voraus.
- **gw-holdout-29 (nicht bestanden)** — Solche Augenblicke möchte ich mir gern merken.
  Antwort: 
  Bewertung: Es wird keine Antwort ausgegeben.
- **gw-holdout-30 (nicht bestanden)** — Ich bin dann mal weg und wünsche dir noch einen schönen Abend.
  Antwort: Du musst heute nicht alles perfekt hinbekommen. Ein kleiner Schritt und eine Pause dürfen für diesen Moment reichen.
  Bewertung: Die Antwort gibt Hinweise zu Perfektion und Pausen statt sich zu verabschieden.

## Daten und Grenzen

Der zusätzliche Sprachkorpus enthält 500 synthetische Kombinationen aus 200 ausdrücklich verfassten Sätzen. Das sind keine 500 unabhängig beobachteten Gespräche. Übergangszählungen sind datenabhängige Schätzung; der Verzicht auf Gradienten oder ein vortrainiertes Modell beseitigt diesen Sprachprior nicht.

Der Algorithmus kann neue Tokenpfade bilden. Ob daraus passende Aussagen entstehen, hängt von den Daten, der Konditionierung und der Decodierung ab. Eine Fouriertransformation liefert aus sich heraus keine Semantik oder neues Wissen.

## Abschluss nach dem eingefrorenen Holdout

Der ausgewählte Stand erreicht **13/30** passende Antworten in der Entwicklung und **6/30** im einmaligen Holdout. Die Mehrheit der neuen Formulierungen erhält damit weiterhin keine passende Antwort. Die numerische Verbindung von Feldern und Tokens ist nachgewiesen; eine allgemein brauchbare Gesprächsgenerierung ist nicht erreicht.

Eine Ausgabe in der Entwicklung enthält einen neuen, sinnvollen Einzelsatz; im Holdout sind es **0/30**. Die vollständige Entwicklungsentscheidung und alle vorab gewählten Alternativen stehen in [development_selection.md](development_selection.md).

> Wenn du erschöpft bist, kann ein ruhiger Moment angenehm sein.

Dieser Satz wurde aus zwölf nacheinander berechneten Tokenmoden erzeugt und kommt unter 418 unterschiedlichen Korpussätzen nicht vor. [Der Tokenbeleg](new_sentence_evidence.json) enthält Präfixe, Frequenzen und überprüfte Wahrscheinlichkeiten.

## Kausale Kontrollen des Wortdecoders

Die folgenden Kontrollen verwenden 90 Präfixzustände aus den 30 Entwicklungseingaben. Sie prüfen den Wortdecoder mit dem ausgewählten Sprachmodell; die Gesprächsbewertung oben prüft zusätzlich die mehrteilige Eingabeverarbeitung und den Sitzungskontext. Geänderte Verteilungen sind kein Beleg für bessere Sprache.

| Eingriff | Verteilung geändert | Argmax geändert | Mittlere Total-Variation-Distanz |
| --- | ---: | ---: | ---: |
| Tokenkanal des Prompts null | 27/90 | 6/90 | 0.032779 |
| Merkmalskanal des Prompts null | 80/90 | 36/90 | 0.467127 |
| Beide Promptkanäle null | 83/90 | 37/90 | 0.478145 |
| Relative Interferenzphase um π geändert | 27/90 | 14/90 | 0.090170 |
| Phasen des Merkmalskanals verwürfelt | 59/90 | 35/90 | 0.426957 |
| Phasen der Datenwelle verwürfelt | 34/90 | 28/90 | 0.227984 |
| Direkte Produktrechnung statt spektraler Faltung | 0/90 | 0/90 | 0.000000 |
| Freie Zeitentwicklung bis 86.400,125 s | 0/90 | 0/90 | 0.000000 |

Bei vollständig genullten unbedingten und konditionierten Datenspektren verweigert der Decoder in **90/90** Fällen ausdrücklich die Ausgabe. Referenzarrays und Grammatikstützen bleiben bei diesem Test vorhanden; sie dienen nicht als verdeckter Antwortgenerator.

Die direkte Produktrechnung und die spektrale Faltung sind nach der dokumentierten Rundung auf zwölf Nachkommastellen identisch. Beide verwenden dieselben gespeicherten Übergangsamplituden. Der größere Einfluss des Merkmalskanals zeigt, dass explizite Sprachmerkmale und annotierte Daten wesentlich zur Konditionierung beitragen. Die freie gemeinsame Phase verschwindet beim Auslesen der Intensität; bloßes längeres Schwingen erzeugt daher keine zusätzlichen Gedanken.

Die SHA-256-Werte von Implementierung, Kategorien, Korpora, Konfiguration und Evaluierung wurden vor dem Holdout festgelegt und beim Erstellen dieses Berichts erneut überprüft. [final_summary.json](final_summary.json) bindet alle Abschlussartefakte an ihre Quelldateien.
