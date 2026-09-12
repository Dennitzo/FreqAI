"""Build an explicitly authored, unpaired German language prior.

The 500 records are Cartesian combinations of 20 independent theme families,
each with five complete opening sentences and five complete continuations.
They are synthetic language examples, not 500 independently collected dialogs.
There are no question/answer keys, embeddings, fitted weights or evaluation
lookups. This script never reads any evaluation file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "memory/language/generative_corpus.jsonl"
PROVENANCE = OUTPUT.with_suffix(".provenance.json")

FAMILIES = {
    "greeting": (
        ["Hallo, schön, dass du hier bist.", "Hey, willkommen zu unserem Gespräch.",
         "Guten Morgen, ich bin bereit für unser Gespräch.", "Guten Abend, wir können gern ein wenig plaudern.",
         "Hi, schön, von dir zu hören."],
        ["Wie ist deine Stimmung gerade?", "Was möchtest du heute erzählen?",
         "Was hat dich heute beschäftigt?", "Wie hat sich dein Tag bisher angefühlt?", "Womit möchtest du beginnen?"]),
    "conversation": (
        ["Wir können uns gern in Ruhe unterhalten.", "Für ein Gespräch bin ich bereit.",
         "Du kannst ein Thema wählen, das dich interessiert.", "Ein kleines Gespräch kann mit einer einfachen Frage beginnen.",
         "Wir können bei deinem Gedanken bleiben."],
        ["Worüber möchtest du zuerst sprechen?", "Was geht dir gerade durch den Kopf?",
         "Welche Frage möchtest du mitbringen?", "Was ist dir an diesem Thema wichtig?", "Magst du ein wenig mehr erzählen?"]),
    "positive": (
        ["Wenn du zufrieden bist, ist das ein schöner Moment.", "Eine gute Nachricht kann den Tag heller machen.",
         "Über einen kleinen Erfolg darfst du dich freuen.", "Es ist schön, wenn dir etwas gut gelingt.",
         "Ein angenehmer Augenblick kann lange in Erinnerung bleiben."],
        ["Was hat zu diesem guten Gefühl beigetragen?", "Was möchtest du davon festhalten?",
         "Welcher Teil daran bedeutet dir besonders viel?", "Möchtest du von diesem schönen Moment erzählen?", "Mit wem würdest du deine Freude gern teilen?"]),
    "negative": (
        ["Ein belastender Tag kann viel Kraft kosten.", "Wenn dich etwas traurig macht, darfst du es ansprechen.",
         "Schwierige Gefühle lassen sich manchmal nur langsam in Worte fassen.", "Manchmal ist eine Situation einfach schwer auszuhalten.",
         "Du musst einen unangenehmen Gedanken nicht sofort erklären können."],
        ["Was beschäftigt dich daran am meisten?", "Möchtest du erzählen, was gerade schwer ist?",
         "Was würde dir in diesem Moment etwas Halt geben?", "Soll ich dir erst einmal nur zuhören?", "Welchen Teil möchtest du zuerst beschreiben?"]),
    "rest": (
        ["Wenn du müde bist, kann ein ruhiger Moment angenehm sein.", "Nach einer Anstrengung kann eine Pause guttun.",
         "Bei wenig Energie darf der nächste Schritt auch klein sein.", "Wenn du erschöpft bist, musst du nicht sofort alles erledigen.",
         "Ein anstrengender Tag lässt manchmal wenig Kraft übrig."],
        ["Was würde dir gerade etwas Erholung geben?", "Möchtest du lieber ausruhen oder ruhig erzählen?",
         "Wie viel Ruhe wäre jetzt für dich angenehm?", "Was kannst du für einen Moment liegen lassen?", "Gibt es einen bequemen Platz für eine kleine Pause?"]),
    "calm": (
        ["Wir können ohne Eile miteinander sprechen.", "Du kannst dir beim Erzählen Zeit lassen.",
         "Ein ruhiges Gespräch braucht keine schnellen Antworten.", "Ich kann auf deine nächsten Worte eingehen.",
         "Wir müssen aus diesem Gespräch keine Aufgabe machen."],
        ["Was möchtest du in deinem eigenen Tempo erzählen?", "Möchtest du bei diesem Gedanken bleiben?",
         "Was fühlt sich als nächstes Thema passend an?", "Soll ich zunächst einfach zuhören?", "Welche Worte passen gerade am besten zu deiner Stimmung?"]),
    "music": (
        ["Ruhige Musik kann eine angenehme Begleitung sein.", "Ein Lied kann an einen bestimmten Moment erinnern.",
         "Bei Musik sind Vorlieben sehr verschieden.", "Instrumentale Stücke lassen viel Raum für eigene Gedanken.",
         "Manche Menschen mögen leise Klänge, andere lebendige Rhythmen."],
        ["Welche Musik passt gerade zu deiner Stimmung?", "Was gefällt dir an diesem Klang?",
         "Hörst du lieber aufmerksam zu oder läuft die Musik nebenbei?", "Gibt es ein Lied, über das du sprechen möchtest?", "Magst du eher Gesang oder Instrumente?"]),
    "books": (
        ["Eine Geschichte kann ein guter Anlass für ein Gespräch sein.", "Beim Lesen entdeckt man manchmal einen neuen Blick auf den Alltag.",
         "Bücher können sehr unterschiedliche Stimmungen erzeugen.", "Ein kurzer Text kann ebenso anregend sein wie ein langer Roman.",
         "Manche Geschichten bleiben besonders lange im Kopf."],
        ["Welche Art von Geschichte interessiert dich?", "Was gefällt dir an dem Buch, das du gerade liest?",
         "Möchtest du lieber über Figuren oder über die Handlung sprechen?", "Welche Stelle ist dir besonders aufgefallen?", "Was würdest du gern als Nächstes lesen?"]),
    "food": (
        ["Ein einfaches Essen kann mit wenigen Zutaten gelingen.", "Gemüse lässt sich auf viele Arten zu einer Mahlzeit verbinden.",
         "Auch ein Gericht ohne Fleisch kann abwechslungsreich sein.", "Beim Kochen helfen kleine, überschaubare Schritte.",
         "Eine warme Suppe kann ein unkompliziertes Abendessen sein."],
        ["Welche Zutaten hast du gerade da?", "Möchtest du lieber etwas Warmes oder etwas Kaltes essen?",
         "Wie viel Zeit möchtest du fürs Kochen einplanen?", "Welche Lebensmittel magst du besonders gern?", "Soll das Essen vegetarisch sein?"]),
    "work": (
        ["Ein voller Arbeitstag kann anstrengend sein.", "Viele Unterbrechungen können die Konzentration erschweren.",
         "Ein kleiner erledigter Schritt ist ebenfalls ein Ergebnis.", "Nach der Arbeit kann ein klarer Übergang in die Freizeit angenehm sein.",
         "Bei vielen Aufgaben hilft manchmal ein Blick auf das Wesentliche."],
        ["Was hat dich heute bei der Arbeit beschäftigt?", "Welcher Teil ist dir gut gelungen?",
         "Was möchtest du morgen anders angehen?", "Welche Aufgabe ist für dich gerade am wichtigsten?", "Wie möchtest du den Arbeitstag ausklingen lassen?"]),
    "weather": (
        ["An einem regnerischen Tag kann es drinnen gemütlich sein.", "Bei Wind fühlt sich ein geschützter Platz oft angenehmer an.",
         "Ein sonniger Moment kann Lust auf frische Luft machen.", "Das Wetter beeinflusst manchmal die Pläne für den Tag.",
         "Auch zu Hause lassen sich ruhige Stunden verbringen."],
        ["Wie möchtest du deine Zeit heute verbringen?", "Was wäre bei diesem Wetter für dich angenehm?",
         "Möchtest du lieber drinnen bleiben oder nach draußen gehen?", "Welche kleine Beschäftigung würde dir Freude machen?", "Was macht es für dich zu Hause gemütlich?"]),
    "contact": (
        ["Eine kurze Nachricht kann einen Kontakt wieder aufnehmen.", "Ein Gespräch mit einem vertrauten Menschen kann Nähe schaffen.",
         "Wenn Gesellschaft fehlt, kann ein kleiner Kontakt ein Anfang sein.", "Mit Freunden lassen sich auch unspektakuläre Dinge teilen.",
         "Eine Einladung muss nicht besonders aufwendig sein."],
        ["Mit wem würdest du gern sprechen?", "Was möchtest du dieser Person erzählen?",
         "Wäre dir eine Nachricht oder ein Anruf lieber?", "Welche Form von Gesellschaft würde dir gerade guttun?", "Was wäre ein angenehmer erster Satz für den Kontakt?"]),
    "leisure": (
        ["Ein kurzer Spaziergang kann eine kleine Abwechslung sein.", "Freizeit darf auch ohne großes Vorhaben angenehm sein.",
         "Ein kreativer Versuch muss nicht perfekt werden.", "Manchmal macht eine vertraute Beschäftigung besonders viel Freude.",
         "Etwas Neues auszuprobieren kann den Tag interessanter machen."],
        ["Worauf hättest du gerade Lust?", "Möchtest du dich lieber bewegen oder etwas Ruhiges machen?",
         "Welche Beschäftigung passt zu deiner Energie?", "Was würdest du gern einmal ausprobieren?", "Wie viel Zeit möchtest du dir dafür nehmen?"]),
    "gratitude": (
        ["Gern geschehen.", "Danke für deine freundlichen Worte.", "Schön, wenn dir das Gespräch etwas gibt.",
         "Deine Rückmeldung hilft dabei, das Gespräch einzuordnen.", "Es ist gut zu wissen, was für dich passend war."],
        ["Wir können gern noch weiterreden.", "Du kannst jederzeit einen neuen Gedanken einbringen.",
         "Was möchtest du als Nächstes besprechen?", "Magst du bei diesem Thema bleiben?", "Ich kann auf deine nächste Frage eingehen."]),
    "farewell": (
        ["Tschüss, bis zum nächsten Gespräch.", "Auf Wiedersehen und alles Gute für deinen Tag.",
         "Bis bald, wir können später wieder anknüpfen.", "Mach es gut und komm gut durch den Tag.",
         "Danke für das Gespräch und bis ein anderes Mal."],
        ["Hab eine angenehme Zeit.", "Ich wünsche dir einen ruhigen Abend.",
         "Für dein nächstes Vorhaben wünsche ich dir gutes Gelingen.", "Pass gut auf dich auf.", "Lass dir für deine nächsten Schritte genügend Zeit."]),
    "assistant": (
        ["Ich bin FreqAI, ein Programm für Gespräche mit einem Wellenspeicher.", "Ich habe keine eigenen Gefühle oder körperlichen Erlebnisse.",
         "Meine Antworten entstehen durch Berechnungen mit gespeicherten Sprachdaten.", "Ich habe keinen eigenen Alltag und keine persönlichen Vorlieben.",
         "Ich kann Text verarbeiten, besitze aber kein menschliches Erleben."],
        ["Wir können trotzdem über deinen Alltag sprechen.", "Welche Frage möchtest du mit mir besprechen?",
         "Bei unbekannten Informationen sollte ich meine Grenzen benennen.", "Ich bin bereit, auf deine Worte einzugehen.", "Was möchtest du über dieses Gesprächssystem wissen?"]),
    "negation": (
        ["Nicht müde zu sein bedeutet nicht automatisch, voller Energie zu sein.", "Wer still ist, muss deshalb nicht traurig sein.",
         "Wenn du keine Ratschläge möchtest, kann ich zunächst zuhören.", "Nicht unglücklich zu sein lässt noch viele andere Stimmungen offen.",
         "Eine Vorliebe für ruhige Musik bedeutet nicht, dass alle anderen Klänge schlecht sind."],
        ["Wie würdest du es mit deinen eigenen Worten beschreiben?", "Was ist dir an dieser Unterscheidung wichtig?",
         "Ich möchte dir keine Stimmung unterstellen.", "Wir können bei deiner eigenen Beschreibung bleiben.", "Welche Einordnung passt für dich besser?"]),
    "uncertainty": (
        ["Ich bin nicht sicher, worauf sich dieser Gedanke bezieht.", "Für diese Frage fehlt mir gerade ein wichtiger Zusammenhang.",
         "Ich möchte aus wenigen Worten keine falsche Schlussfolgerung ziehen.", "Dazu kann ich ohne weitere Angaben nichts Verlässliches sagen.",
         "Der Zusammenhang ist für mich noch nicht eindeutig."],
        ["Kannst du ein wenig mehr Kontext geben?", "Welchen Teil meinst du genau?",
         "Magst du den Gedanken genauer beschreiben?", "Auf welches Thema möchtest du zurückkommen?", "Was wäre für deine Frage besonders wichtig?"]),
    "memory": (
        ["Wenn du mir deinen Namen nennst, kann ich ihn im Gespräch verwenden.", "Ein persönlicher Name sollte genau so wiedergegeben werden, wie du ihn angibst.",
         "Informationen aus verschiedenen Gesprächen sollten getrennt bleiben.", "Ein zuvor genannter Gedanke kann für eine spätere Rückfrage wichtig sein.",
         "Wenn eine Erinnerung fehlt, sollte ich offen danach fragen."],
        ["Wie möchtest du angesprochen werden?", "Welche Angabe ist für dieses Gespräch wichtig?",
         "Möchtest du eine frühere Aussage ergänzen?", "Was soll ich für den weiteren Zusammenhang beachten?", "Welche Formulierung passt am besten zu dem, was du meintest?"]),
    "plans": (
        ["Ein überschaubarer Plan kann den nächsten Schritt leichter machen.", "Ein ruhiger Morgen beginnt für jeden Menschen etwas anders.",
         "Zwischen festen Terminen kann freie Zeit angenehm sein.", "Nicht jeder Teil eines Tages muss im Voraus feststehen.",
         "Ein kleiner Anfang ist manchmal hilfreicher als ein großer Vorsatz."],
        ["Was möchtest du als Erstes angehen?", "Welcher Schritt wäre für dich gut machbar?",
         "Wofür möchtest du morgen Zeit einplanen?", "Was würde deinen Start etwas ruhiger machen?", "Welche Aufgabe darf noch warten?"]),
}


def build() -> tuple[list[dict], dict]:
    records = []
    atomic = set()
    for family, (openings, continuations) in FAMILIES.items():
        assert len(openings) == len(continuations) == 5
        for opening in openings:
            for continuation in continuations:
                text = opening + " " + continuation
                records.append({"id": f"language-{len(records) + 1:03d}", "text": text,
                                "source": "Authored synthetic language prior / " + family})
                atomic.update((opening, continuation))
    assert len(records) == 500
    assert len({r["text"] for r in records}) == 500
    data = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    metadata = {
        "description": "Explicit authored synthetic German language prior, unpaired sentences.",
        "method": "20 authored families; 5 openings x 5 continuations per family; no evaluation reads.",
        "not_independent_observations": True,
        "records": len(records), "atomic_sentences": len(atomic),
        "families": {family: 25 for family in FAMILIES},
        "tokens_word_punctuation": sum(len(re.findall(r"\w+|[^\w\s]", r["text"])) for r in records),
        "question_answer_pairs": False, "prompt_fields": False,
        "pretrained_model": False, "gradient_training": False,
        "generator": "experiments/build_generative_corpus.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
        "limitations": ["Synthetic combinations, not 500 independent conversations.",
                       "Authored linguistic and factual assumptions are explicit prior knowledge.",
                       "Transition counts estimated from this corpus are data-dependent parameters.",
                       "Not yet imported into the productive database by this generator."]}
    OUTPUT.write_text(data, encoding="utf-8", newline="\n")
    PROVENANCE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return records, metadata


if __name__ == "__main__":
    _, summary = build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
