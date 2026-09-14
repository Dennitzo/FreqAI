"""Deutsche Sprachmittel fuer Eingabeverstaendnis, Satzzerlegung und Textaufbereitung.

Alle Regeln hier sind ausdruecklich geschriebene Sprachregeln: keine gelernten
Gewichte, keine Antwortbeispiele, keine Frage-Antwort-Paare. Sie ergaenzen die
deklarativen Rollen des Compilers um eine Frage- und Satzebene.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
import unicodedata

# ---------------------------------------------------------------- Normalisierung

QUOTES = {chr(0x201E): '"', chr(0x201C): '"', chr(0x201D): '"', chr(0x201F): '"',
          chr(0x201A): "'", chr(0x2018): "'", chr(0x2019): "'"}
DASHES = {chr(0x2013): "-", chr(0x2014): "-", chr(0xA0): " ", chr(0x2009): " ",
          chr(0x200A): " ", chr(0x202F): " ", chr(0xFEFF): "", chr(0xAD): ""}

#: Lautschrift- und Exportklammern aus dem Wikipedia-Export sind kein Satzinhalt.
BRACKET_ARTIFACT = re.compile(r"\[[^\[\]]*\]")
EMPTY_BRACKET = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
LOOSE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")
LOOSE_OPEN = re.compile(r"([(\[])\s+")
MULTISPACE = re.compile(r"[ \t]{2,}")


def normalize_glyphs(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    for left, right in {**QUOTES, **DASHES}.items():
        text = text.replace(left, right)
    return text


DANGLING_SOURCE = re.compile(r"\(\s*(?:von|zu|entlehnt|eigentlich)?\s*[;,]\s*")
HOLLOW_PARENTHESIS = re.compile(r"\(\s*(?:[a-zäöü]{1,4}[.;])?\s*\)")


def clean_text(text: str) -> str:
    """Lesbaren Fliesstext herstellen: Exportartefakte und Doppel leer entfernen."""
    text = normalize_glyphs(text)
    text = DANGLING_SOURCE.sub("(", text)
    text = HOLLOW_PARENTHESIS.sub("", text)
    previous = None
    while previous != text:
        previous = text
        text = BRACKET_ARTIFACT.sub("", text)
    text = EMPTY_BRACKET.sub("", text)
    text = LOOSE_PUNCT.sub(lambda match: match.group(1), text)
    text = LOOSE_OPEN.sub(lambda match: match.group(1), text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()


ABBREVIATIONS = {"z", "b", "bzw", "ca", "dr", "prof", "nr", "usw", "etc", "evtl",
                 "u", "a", "d", "h", "v", "chr", "n chr", "m", "w", "gbf", "ff",
                 "jan", "feb", "maerz", "märz", "apr", "jun", "jul", "aug", "sep",
                 "okt", "nov", "dez", "st", "bd", "tl"}


def split_sentences(text: str) -> list[str]:
    """Satzgrenzen; Abkuerzungen und Kommas werden nicht als Satzende gelesen."""
    text = normalize_glyphs(text)
    parts, current = [], ""
    tokens = re.findall(r"\S+|\s+", text)
    for position, token in enumerate(tokens):
        current += token
        stripped = current.rstrip()
        if not token.strip():
            continue
        tail = stripped[-1:]
        if tail in ".!?":
            head = re.split(r"\s+", stripped[:-1])[-1].lower().rstrip(".") if stripped[:-1] else ""
            numeric_head = bool(re.fullmatch(r"[0-9]{1,4}", head))
            # Eine Abkuerzung oder eine Zahl beendet keinen Satz.
            if head in ABBREVIATIONS or numeric_head:
                continue
            current = ""
            parts.append(stripped)
    if current.strip():
        parts.append(current.strip())
    return [clean_text(part) for part in parts if len(clean_text(part)) >= 4]


def strip_question_mark(text: str) -> str:
    return normalize_glyphs(text).strip().rstrip("?!. ").strip()


# ------------------------------------------------------------------- Fragearten

#: Rechtschreib-, Abkuerzungs- und Aliasvarianten derselben Sache.
ALIASES = {
    "sci fi": "science fiction", "scifi": "science fiction", "sci-fi": "science fiction",
    "sciencefiction": "science fiction", "sf film": "science fiction film",
    "youtube": "youtube", "yt": "youtube", "youtu be": "youtube",
    "twitch tv": "twitch", "fb": "facebook", "insta": "instagram",
    "ki": "kuenstliche intelligenz", "ai": "artificial intelligence",
    "ml": "maschinelles lernen", "machine learning": "maschinelles lernen",
    "doku": "dokumentation", "handy": "smartphone", "pc": "personalcomputer",
    "tv": "fernsehen", "email": "elektronische post", "mail": "elektronische post",
    "whatsapp": "whatsapp", "amazon": "amazon", "ebay": "ebay", "bitcoin": "bitcoin",
}

#: Einleitungs- und Hoeflichkeitsformeln, die den Zielbegriff verdecken.
LEAD_IN = [
    r"^\s*(?:okay|ok|also|na ja|nun|hmm)[,:!]?\s+",
    r"^\s*(?:bitte|mal|kurz|kurzzum|genau)[ ,]+",
    r"^\s*(?:kannst|koenntest|könntest|magst|moechtest|möchtest|willst|moechten|möchten) du "
    r"(?:mir|uns)? ?(?:mal)? ?(?:erklär\w*|erklaer\w*|beschreib\w*|sag\w*|nenn\w*|zeig\w*|"
    r"verrat\w*|erläuter\w*|erlauter\w*)[ ,]+",
    r"^\s*(?:erklär|erklaer|beschreib|definier|nenn|schilder|erläuter|erlauter)"
    r"(?:e)?(?:n)?(?:st)? (?:mir|uns|mal|bitte)? ?\w*[ ,]*",
    r"^\s*(?:kannst du mir sagen|sag mir|weisst du|weißt du),? ?(?:ob )?",
    r"^\s*(?:ich möchte|ich will|ich hätte gern|hätte gern|moechte gern|gerne) "
    r"(?:gern )?(?:wissen|erfahren|hören|hoeren|lernen)[ ,]+",
    r"^\s*(?:hast du|hast du mir)? ?(?:einen|ein)? ?"
    r"(?:tipp|rat|vorschlag|idee)(?: für mich)?\??\s*",
]

INTERROGATIVE_HEAD = [
    r"^\s*(?:was|wer|wie|welch(?:e|er|es|em|en)?|wozu|wofür|womit|wovon|wann|warum|wieso|"
    r"weshalb|wodurch|wieso)\b(?: ist| sind| war| waren| es)? ?",
]

COPULAE = ("ist", "sind", "war", "waren", "bedeutet", "heisst", "heißt", "nennt",
           "gilt als", "steht für", "umfasst", "bezeichnet", "versteht")


def _strip_repeated(text: str, patterns: list[str]) -> str:
    result = text
    changed = True
    while changed and result.strip():
        changed = False
        for pattern in patterns:
            stripped = re.sub(pattern, "", result, flags=re.I)
            if stripped != result:
                result = stripped
                changed = True
        result = re.sub(r"^\s+|[?!.,]+$", "", result)
    return result.strip()



def target_phrase(prompt: str) -> str:
    """Der Zielbegriff einer Frage, bereinigt von Fragewoertern und Hoeflichkeit."""
    text = strip_question_mark(prompt)
    candidate = _strip_repeated(text, LEAD_IN + INTERROGATIVE_HEAD)
    if len(candidate) < 2:
        candidate = text
    lowered = candidate.lower()
    for prefix in ("die ", "der ", "das ", "eine ", "ein "):
        if lowered.startswith(prefix) and len(candidate) > len(prefix) + 2:
            candidate = candidate[len(prefix):]
            break
    return candidate.strip(" ,;:")


def alias_expansions(prompt: str) -> list[str]:
    lowered = normalize_glyphs(prompt).casefold().replace("ss", "ss")
    result = []
    for key, value in ALIASES.items():
        if re.search(r"(?<![\\w-])" + re.escape(key) + r"(?![\\w-])", lowered):
            result.append(value)
    return list(dict.fromkeys(result))


def content_words(text: str, *, drop_extra=()) -> list[str]:
    """Inhaltswoerter ohne Fuell- und Fragewoerter; Reihenfolge bleibt erhalten."""
    from .features import STOPWORDS, canonical_term
    words = [canonical_term(word) for word in
             re.findall(r"[^\W_]+", normalize_glyphs(text), flags=re.UNICODE)]
    skip = {canonical_term(item) for item in drop_extra}
    return [word for word in words if word not in STOPWORDS and word not in skip]


def is_pronominal(prompt: str) -> bool:
    """Ein Fuerwort ohne eigenes Substantiv verweist auf das vorherige Thema."""
    words = re.findall(r"[^\W_]+", normalize_glyphs(prompt).casefold(), flags=re.UNICODE)
    pronouns = {"sie", "er", "es", "ihm", "ihn", "ihren", "ihre", "ihrer", "dessen",
                "deren", "diese", "dieser", "dieses", "denen", "solche", "sein", "seine"}
    return bool(set(words) & pronouns) and len(words) <= 9


def expects_yes_no(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.match(r"^\s*(?:ob|stimmt es|ist es (?:so|wahr)|kann|koennen|können|hat|"
                         r"haben|sind|war|waren|gibt es|darf|muss|müssen)\b", lowered))


def list_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:nenn|liste|aufzähl|aufzaehl|beispiele|bsp)\w*", lowered)
                or re.search(r"\bwie viele\b", lowered))


def comparison_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:unterschied|unterscheid|vergleich|verschieden)\b", lowered)) \
        or " zwischen " in f" {lowered} "


def summarize_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:fasst?|zusammenfass\\w*|kurzfassung|ueberblick|überblick|"
                          r"erzähl|erzahl|berichte|schildere)\w*", lowered))


def explain_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:erklär|erklaer|beschreib|definier|erläuter|erlauter|"
                          r"funktioniert|funktionsweise)\w*", lowered))


def how_many_request(prompt: str) -> bool:
    return bool(re.search(r"\bwie viele\b|\banzahl\b", strip_question_mark(prompt).lower()))


def temporal_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:wann|seit wann|wie lange|wie alt|in welchem jahr)\b", lowered))


def causal_request(prompt: str) -> bool:
    lowered = strip_question_mark(prompt).lower()
    return bool(re.search(r"\b(?:warum|wieso|weshalb|wodurch|wofür|wozu)\b", lowered))



# --------------------------------------------------------------- Frageklassifikation

#: Faehigkeits- und Selbstauskunftssignale fuer Fragen an den Assistenten selbst.
ASSISTANT_PRONOUNS = {"du", "dir", "dich", "dein", "deine", "deinen", "deiner", "euch", "Ihnen"}
ABILITY_WORDS = {"sehen", "siehst", "hoeren", "hören", "denken", "wissen", "weisst",
                 "koennen", "sprechen", "sprichst", "reden", "fuehlen", "gefuehl", "gefuehle",
                 "traeumen", "lieben", "wollen", "erinnern", "erinnerst", "lesen",
                 "schreiben", "gewissen", "bewusstsein", "daten", "informationen",
                 "herkunft", "aufgewachsen", "koerper", "sprache", "meinst", "haeltst",
                 "sichtest", "wahrnimmst", "erfahrungen", "meinung"}
#: Woerter, die in einem Zielbegriff nichts suchen (Auftrags- und Fuellwoerter).
TARGET_NOISE = {"artikel", "text", "absatz", "kapitel", "mir", "uns", "mal", "bitte",
                "fasse", "zusammen", "nenne", "nenn", "aufzähl", "aufzaehl", "drei",
                "zwei", "vier", "fuenf", "sechs", "mehrere", "paar", "einige", "beispiel",
                "beispiele", "list", "liste", "hat", "haben", "hatte", "gibt", "es",
                "stimmt", "dass", "ob", "wie", "viele", "viel", "lang", "lange",
                "bedeutet", "heisst", "funktioniert", "gemacht", "eigentlich", "den"}


#: Redewendungen und ihre Themen; ausdrueckliche Sprachkenntnis, keine Paare.
IDIOM_TOPICS = (
    (r"\b(?:wo|woher)\s+kommst\b|\bherkunft\b", "Herkunft"),
    (r"\bwie alt bist du\b|\bdein alter\b|\bgeburtstag\b", "Alter"),
    (r"\bwas siehst du\b|\bwelche daten\b|\bwas nimmst du wahr\b|\bwer hat dich geschrieben\b",
     "Daten"),
    (r"\bwelche sprache\b|\bsprichst du\b|\bauf welcher sprache\b", "Sprache"),
    (r"\bgewissen\b", "Gewissen"),
    (r"\bgef(?:ü|ue)hl|\bf(?:ü|ue)hlst du|\bempfinden\w*\b|\blemotionen\b", "Gefühle"),
    (r"\btr(?:ä|ae)umst du|\btr(?:ä|ae)ume\b", "Träume"),
    (r"\bmeinung\b|\bwas meinst du\b|\bdenkst du darüber\b", "Meinung"),
    (r"\bbeschreib\w* (?:dich|selbst)|\bstell dich vor\b|\bwer bist du\b|\bwas bist du\b",
     "Beschreibung"),
    (r"\bwie (?:funktionierst|berechnest|arbeitest) du\b|\bfunktionsweise\b|\bwie genau\w* du"
     r" das\b", "Funktionsweise"),
    (r"\btipp\b|\bein rat\b|\bworauf achtest\b", "Tipp"),
    (r"\bf(?:ü|ue)hre ein gespräch|\blass uns (?:reden|plaudern)\b|\bquatsch\w* mit mir\b",
     "Gespräch"),
    (r"\bdein körper\b|\bdein koerper\b|\bwie siehst du aus\b", "Körper"),
    (r"\bkannst du (?:übersetzen|rechnen)|\bübersetz\w*\b|\bformuliere um\b", "Grenzen"),
)

#: Zweitpersonformen, die eine Frage an den Assistenten richten ohne 'du'.
SECOND_PERSON_VERBS = {"bist", "hast", "kannst", "willst", "sollst", "magst", "moechtest",
                       "möchtest", "kommst", "denkst", "weisst", "weißt", "lebst", "fuehlst",
                       "fühlst", "traeumst", "träumst", "meinst", "erinnerst", "siehst",
                       "hoerst", "hörst", "liest", "schreibst", "sprichst", "urteilst",
                       "wahrnimmst", "arbeitest", "funktionierst", "berechnest"}


def idiom_topics(text: str) -> tuple:
    found = []
    for pattern, topic in IDIOM_TOPICS:
        if re.search(pattern, text, flags=re.I):
            found.append(topic)
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class QueryIntent:
    """Was eine Eingabe sprachlich will. Beschrieben, nicht gelernt."""
    kind: str
    target: str = ""
    topics: tuple = ()
    modifiers: tuple = ()
    expects_number: bool = False
    expects_person: bool = False
    attention: str = "thema"          # thema | assistent | benutzer
    parts: tuple = ()                 # Teilfragen derselben Eingabe
    raw: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "topics": list(self.topics),
                "modifiers": list(self.modifiers),
                "expects_number": self.expects_number, "expects_person": self.expects_person,
                "attention": self.attention, "parts": [item.as_dict() for item in self.parts]}


#: Woerter, die in einem Auftragsziel ganz entfallen duerfen.
INSTRUCTION_NOISE = TARGET_NOISE | {"ueber", "über", "den", "der", "die", "das", "für",
                                     "von", "mit", "zu", "ins", "zum", "zur", "im", "am",
                                     "auf", "eine", "ein", "einen", "dem", "den", "englische",
                                     "englisch", "deutsche", "französische", "übersetze",
                                     "uebersetze", "fische"}


def clean_target(text: str, *, aggressive: bool = False) -> str:
    """Zielbegriff von Auftrags- und Fuellwoertern befreien, Reihenfolge erhalten."""
    candidate = target_phrase(text)
    words = [word for word in candidate.split() if word not in {",", ";"}]
    while words and canonical_keep(words[0]) in TARGET_NOISE:
        words = words[1:]
    result = " ".join(words)
    for noise in (" gibt es", " hat ein", " hat eine", " hat ", " haben "):
        index = result.lower().find(noise)
        if index > 2:
            result = result[:index] + " " + result[index + len(noise):]
    result = re.sub(r"\s+", " ", result).strip(" ,;:.")
    if aggressive:
        kept = [word for word in result.split()
                if canonical_keep(word.strip(".,;:")) not in INSTRUCTION_NOISE]
        if kept:
            result = " ".join(kept)
    return result or candidate.strip()


def canonical_keep(word: str) -> str:
    from .features import canonical_term
    return canonical_term(word).replace("ss", "ss")


def _modifier_phrases(prompt: str) -> tuple:
    """Einschraenkungen wie 'in der Physik' oder 'aus dem Jahr 1921' getrennt halten."""
    text = normalize_glyphs(prompt)
    found = []
    for pattern in (r"\bin (?:der|dem|den|die) ([^\n?!.]{2,30}?)(?=[?.!,]|$)",
                    r"\baus (?:dem|der|den) ([^\n?!.]{2,30}?)(?=[?.!,]|$)"):
        match = re.search(pattern, text, flags=re.I)
        if match:
            phrase = match.group(0).strip().lower()
            if 3 < len(phrase) <= 40 and len(phrase.split()) <= 6:
                found.append(phrase)
    return tuple(dict.fromkeys(found))


def sub_questions(text: str) -> tuple:
    """Mehrere Teilfragen einer Eingabe, etwa Definition plus Einheit."""
    pieces = [piece.strip(" ,;.") for piece in re.split(r"[;,]|\s+und\s+", normalize_glyphs(text))
              if len(piece.strip()) >= 3]
    if len(pieces) < 2:
        return ()
    meaningful = [piece for piece in pieces if len(content_words(piece)) >= 1]
    if len(meaningful) < 2 or sum(len(content_words(piece)) for piece in meaningful) < 4:
        return ()
    return tuple(dict.fromkeys(meaningful))[:3]


def is_assistant_directed(lowered: str, word_set: set) -> bool:
    if not (word_set & ASSISTANT_PRONOUNS):
        return False
    return True


def classify(prompt: str) -> QueryIntent:
    """Frageart, Zielbegriff und Teilfragen einer Eingabe bestimmen."""
    text = normalize_glyphs(prompt).strip()
    stripped = strip_question_mark(text)
    if not stripped:
        return QueryIntent("leer", raw=text)
    lowered = stripped.lower()
    words = re.findall(r"[^\W_]+", lowered, flags=re.UNICODE)
    word_set = set(words)
    second_person = bool(word_set & SECOND_PERSON_VERBS)
    topics = idiom_topics(lowered)
    assistant = bool(word_set & ASSISTANT_PRONOUNS) or "auf dich" in lowered or (
        second_person and len(word_set) <= 12)

    if any(lowered.startswith(item) for item in
           ("hallo", "hi ", "hey", "guten morgen", "guten tag", "servus", "moin")):
        return QueryIntent("begruessung", raw=text, attention="thema")
    if lowered.startswith(("tschues", "tschüss", "ciao", "auf wiedersehen", "bye")) or \
            lowered.startswith("bis bald"):
        return QueryIntent("abschied", raw=text)
    if word_set & {"danke", "dankeschoen", "dankeschön", "verdankesehr"}:
        return QueryIntent("dank", raw=text)

    parts = tuple(QueryIntent("teil", clean_target(piece), raw=piece)
                  for piece in sub_questions(stripped))
    expects_number = how_many_request(stripped) or bool(re.search(r"\bwie viele\b", lowered))
    expects_person = bool(re.match(r"^\s*wer\b", lowered))
    modifiers = _modifier_phrases(stripped)
    target = clean_target(stripped)

    if assistant and topics:
        return QueryIntent("selbstauskunft", topics[0], topics, raw=text, attention="assistent")
    if assistant and re.search(r"\bwie heißt du\b|\bwie heisst du\b|\bdein name\b|"
                               r"\bwas bist du für ein", lowered):
        return QueryIntent("name", "name", attention="assistent", raw=text)
    if re.search(r"\bwie geht es (?:dir|heute dir)?\b|\bwie gehts\b|\bbefinden\b", lowered) \
            and assistant or lowered in ("wie geht es dir", "wie geht's"):
        return QueryIntent("befinden", "befinden", attention="assistent", raw=text)
    if assistant and re.search(r"\b(?:beschreib|stell dich|wer bist|was bist)", lowered):
        return QueryIntent("selbstbeschreibung", "", attention="assistent", raw=text)
    if assistant and (word_set & ABILITY_WORDS or second_person or re.search(
            r"\b(?:was siehst|welche daten|wovon|woran|wonach|hast du|kennst du|"
            r"erinnerst du)", lowered)):
        return QueryIntent("fahigkeit", target, (), modifiers, expects_number=expects_number,
                           attention="assistent", raw=text)

    if re.search(r"\b(?:übersetz|uebersetz|übersetze|uebersetze|schreibe|schreib|"
                 r"korrigiere|formuliere|rechne|addiere)\w*", lowered):
        return QueryIntent("auftrag", clean_target(stripped, aggressive=True), modifiers,
                           parts=parts, raw=text)
    if comparison_request(stripped):
        return QueryIntent("vergleich", target, modifiers, expects_number=expects_number,
                           parts=parts, raw=text)
    if summarize_request(stripped):
        return QueryIntent("zusammenfassung", clean_target(stripped, aggressive=True),
                           modifiers, parts=parts, raw=text)
    if list_request(stripped):
        return QueryIntent("liste", clean_target(stripped, aggressive=True), modifiers,
                           expects_number=True, parts=parts, raw=text)
    if temporal_request(stripped):
        return QueryIntent("zeit", target, modifiers, expects_number=True, parts=parts, raw=text)
    if causal_request(stripped):
        return QueryIntent("ursache", target, modifiers, parts=parts, raw=text)
    if expects_yes_no(stripped):
        return QueryIntent("entscheidung", clean_target(stripped, aggressive=True),
                           modifiers, parts=parts, raw=text)
    if explain_request(stripped):
        return QueryIntent("erklaerung", target, modifiers, expects_number=expects_number,
                           parts=parts, raw=text)
    if expects_person:
        return QueryIntent("person", target, modifiers, expects_person=True, parts=parts, raw=text)
    if re.match(r"^\s*(?:was|welch\w*)\b", lowered) and len(words) >= 3:
        return QueryIntent("definition", clean_target(stripped, aggressive=assistant),
                           modifiers, expects_number=expects_number, parts=parts, raw=text)
    if assistant:
        return QueryIntent("selbstauskunft", target, topics, modifiers,
                           expects_number=expects_number, attention="assistent", raw=text)
    if not verb_forms(lowered) and len(content_words(stripped)) <= 4:
        return QueryIntent("stichwort", target, modifiers, expects_number=expects_number,
                           parts=parts, raw=text)
    if is_pronominal(stripped):
        return QueryIntent("folgebegriff", "", modifiers, attention="assistent" if assistant else "thema",
                           expects_number=expects_number, parts=parts, raw=text)
    return QueryIntent("aussage", target, modifiers, expects_number=expects_number,
                       attention="assistent" if assistant else "thema", parts=parts, raw=text)


VERB_HINTS = ("ist", "sind", "war", "waren", "wird", "werden", "hat", "haben", "kann",
              "koennen", "muss", "will", "geht", "kommt", "heisst", "heißt", "bedeutet",
              "funktioniert", "entstehen", "gemacht", "spielt", "lagen", "betrug")


def verb_forms(text: str) -> set:
    return {word for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE) if word in VERB_HINTS}
