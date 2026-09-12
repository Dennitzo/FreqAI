"""Explicit German dialogue grammar and role-separated Fourier concepts.

This is authored linguistic prior knowledge, not learned semantics.  The Fourier
representation preserves the concepts that the grammar identifies; an FFT cannot
infer a meaning absent from those rules.  Unknown clauses remain visible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
import unicodedata

import numpy as np


@dataclass(frozen=True)
class SemanticAct:
    kind: str
    target: str = "user"
    value: str = ""
    topic: str = ""
    negated: bool = False
    confidence: float = 1.0
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Interpretation:
    acts: list[SemanticAct] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    used_context: bool = False
    question_topic: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# The lexicon is deliberately inspectable.  New words are grammar maintenance,
# not evidence that a wave has discovered their meanings autonomously.
MOODS = {
    "positive": r"gut|prima|super|toll|wunderbar|großartig|glücklich|zufrieden|bestens|fantastisch|ausgezeichnet|klasse|fröhlich",
    "negative": r"schlecht|schlimm|traurig|unglücklich|niedergeschlagen|elend|mies|beschissen|miserabel|bedrückt|gedrückt|enttäuscht",
    "neutral": r"okay|ok|mittelmäßig|durchwachsen|normal|soweit|geht so",
    "tired": r"müde|erschöpft|kaputt|ausgelaugt|schläfrig|fix und fertig",
    "stressed": r"gestresst|überfordert|angespannt|nervös|stress",
    "lonely": r"einsam|allein|alleine",
    "excited": r"aufgeregt|begeistert|gespannt|voller vorfreude",
}
TOPICS = {
    "weather": r"wetter|regen|sonne|schnee|sonnig|regnerisch",
    "food": r"essen|kochen|rezept|hunger|hungrig|pizza|pasta|frühstück|mittagessen|abendessen",
    "music": r"musik|lied|lieder|singen|konzert|band",
    "movies": r"film|filme|kino|serie|serien",
    "books": r"buch|bücher|lesen|roman",
    "work": r"arbeit|arbeiten|beruf|büro|job|schule|studium",
    "leisure": r"freizeit|hobby|hobbys|wochenende|spaziergang|spazieren|urlaub|reisen|sport",
    "sleep": r"schlaf|schlafen|bett|einschlafen",
    "conversation": r"reden|sprechen|plaudern|unterhalten|gespräch",
}


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _word(pattern: str, text: str):
    return re.search(r"\b(?:" + pattern + r")\b", text)


def _topic(text: str) -> str:
    for topic, pattern in TOPICS.items():
        if _word(pattern, text):
            return topic
    return ""


def _target(text: str) -> str:
    if _word(r"ich|mir|mich", text):
        return "user"
    if re.search(r"\bmein(?:e|er|em|es)?\s+(?:stimmung|laune|befinden|gefühle?)\b", text):
        return "user"
    if re.search(r"\bdein(?:e|er|em|es)?\s+(?:stimmung|laune|befinden|gefühle?)\b", text):
        return "assistant"
    # Possessive people remain separate persons: 'meine Schwester' is not 'ich'.
    if re.search(r"\b(?:mein|dein)(?:e|er|em|es)?\s+\w+\s+(?:geht|ist|fühlt)\b", text):
        return "other"
    if _word(r"du|dir|dich|deine?", text):
        return "assistant"
    if (_word(r"er|sie|ihm|ihr|ihnen|wir|uns", text)
            or re.search(r"\b\w+\s+(?:ist|sind|fühlt|fühlen|scheint)\b", text)
            or re.search(r"\b\w+\s+geht\s+es\b", text)
            or re.search(r"\bes\s+geht\s+(?!es\b|mir\b|dir\b)\w+\b", text)):
        return "other"
    return "user"


def _split(text: str) -> list[str]:
    # Keep compound predicates together; split conjunctions only where a new
    # speaker, explicit question, or contrast begins.
    nominal_subject = (r"(?:(?:mein|dein|sein|ihr|ein|der|die|das)(?:e|er|em|en|es)?\s+)?"
                       r"\w+\s+(?:ist|sind|geht|fühlt|fühlen|scheint)\b")
    pattern = (r"[.!;\n]+|(?<=[?])\s*|,\s*|\s+aber\s+|"
               r"\s+und\s+(?=(?:dir|du|wie|was|wer|warum|wo|ich|mir|er|sie)\b|"
               + nominal_subject + r")")
    return [part.strip(" ,.!;") for part in re.split(pattern, text, flags=re.I)
            if part.strip(" ,.!;")]


def analyze(text: str, context: dict | None = None) -> Interpretation:
    """Recognize compositional dialogue acts without training or a model service.

    Context accepts last_question_topic, last_topic, user_name and turn_count.
    Bare reciprocal questions require a wellbeing topic or a mood in this turn.
    ``unsupported`` reports unknown clauses even when other clauses are handled.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    context = context or {}
    result = Interpretation()
    clauses = _split(unicodedata.normalize("NFKC", text))
    pending_reciprocals: list[tuple[int, str]] = []

    def add(kind: str, raw: str, **kwargs) -> None:
        result.acts.append(SemanticAct(kind=kind, text=raw, **kwargs))

    for raw in clauses:
        clause = _normal(raw).strip()
        bare = clause.strip("?! ")
        if not bare:
            continue
        before = len(result.acts)

        # Quoted, conditional, and explicitly reported states are not the user's
        # assertions. A small grammar must abstain here instead of changing roles.
        if (re.search(r'["„“»«]', raw)
                or re.match(r"(?:wenn|falls|angenommen|angeblich)\b", clause)
                or re.search(r"\b(?:sagt|sagte|behauptet|meint)\b", clause)):
            result.unsupported.append(raw)
            continue

        greeting = re.match(r"^(?:hallo|hi|hey|huhu|moin|servus|grüß dich|guten morgen|guten tag|guten abend)(?:\b|$)", bare)
        if greeting:
            add("greeting", raw)
            clause = clause[greeting.end():].strip(" !,?")
            bare = clause.strip("?! ")
            if not bare:
                continue

        if re.fullmatch(r"(?:tschüss|tschüs|ciao|auf wiedersehen|bis bald|bis später|bis dann|gute nacht|mach(?:s|'s| es) gut)(?: noch)?", bare):
            add("farewell", raw)
            continue
        if re.fullmatch(r"(?:(?:vielen|besten|herzlichen|lieben)\s+){0,2}dank(?:e)?(?: dir| schön| sehr| für (?:deine|die) hilfe)?", bare):
            add("thanks", raw)
            continue
        if re.fullmatch(r"(?:ja|jawohl|genau|stimmt|richtig|nein|nee|nö|klar|verstanden|einverstanden)", bare):
            add("acknowledgement", raw, value="no" if bare in {"nein", "nee", "nö"} else "yes")
            continue

        name = re.match(r"^ich\s+(?:heiße|heisse|nenne mich)\s+([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2})[?!]*$", raw, re.I)
        if not name:
            name = re.match(r"^mein name ist\s+([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2})[?!]*$", raw, re.I)
        if name:
            add("name_statement", raw, value=name.group(1).strip(), topic="name")
            continue
        if re.search(r"\bwie\s+hei(?:ß|ss)e\s+ich\b|\b(?:kennst|weißt|weisst)\s+du\s+(?:noch\s+)?meinen namen\b|\bwie\s+(?:ist|lautet)\s+mein\s+name\b", clause):
            add("name_question", raw, target="user", topic="name")
            result.question_topic = "name"
            continue
        if re.search(r"\bwer\s+bist\s+du\b|\bwas\s+bist\s+du\b|\bwie\s+hei(?:ß|ss)t\s+du\b|\bbist\s+du\s+(?:ein\s+)?(?:mensch|ki|bot|computer)\b", clause):
            add("identity_question", raw, target="assistant", topic="identity")
            continue
        if re.search(r"\bwas\s+kannst\s+du\b|\bwie\s+funktionierst\s+du\b|\b(?:kannst|könntest)\s+du\s+(?:mir\s+)?helfen\b", clause):
            add("capability_question", raw, target="assistant", topic="capabilities")
            continue

        if re.fullmatch(r"(?:und\s+)?(?:dir|du)(?:\s+(?:so|denn|eigentlich))?\??|(?:und\s+)?bei\s+dir\??", clause):
            pending_reciprocals.append((len(result.acts), raw))
            continue

        mood_vocabulary = "|".join(MOODS.values()) + r"|wohl|in ordnung|guter laune"
        wellbeing = re.search(
            r"\bwie\s+(?:geht|ergeht)\s*(?:'s|s|es)?\s+(?:dir|euch|ihnen|ihm|ihr|mir)\b|"
            r"\bwie\s+(?:fühlst\s+du\s+dich|fühle\s+ich\s+mich|fühlt\s+(?:er|sie)\s+sich)\b|"
            r"\bwie\s+(?:ist|war)\s+dein(?:e)?\s+(?:tag|befinden|stimmung)\b|"
            r"\bwie\s+läuft\s*(?:'s|s|es)?\s+(?:bei\s+)?dir\b|"
            r"\balles\s+(?:klar|gut|okay|ok)\s+bei\s+dir\b",
            clause,
        )
        # Inverted yes/no syntax needs an affect predicate. 'Bist du in Berlin?'
        # and 'Fühlst du dich beobachtet?' are not wellbeing questions.
        wellbeing_predicate = (
            re.match(r"^(?:geht\s+es\s+dir|fühlst\s+du\s+dich|bist\s+du)\b", clause)
            and _word(mood_vocabulary, clause)
            and "?" in clause
        )
        if wellbeing or wellbeing_predicate:
            add("wellbeing_question", raw, target=_target(clause), topic="wellbeing")
            result.question_topic = "wellbeing"
            continue

        # A question about a predicate is not an assertion of that predicate.
        is_question = "?" in clause or bool(re.match(r"(?:wie|was|warum|wieso|weshalb|wann|wo|wer|welche[rsnm]?)\b", clause))
        mood_found = False
        if not is_question:
            # Verbal/nominal affect predicates share the same compositional
            # polarity rules as adjectives; words such as 'wohl' and 'Ordnung'
            # alone are not evidence of a mood.
            affect = re.search(
                r"\b(?:fühle?\w*\s+(?:mich|dich|sich)|bin|bist|ist)\s+"
                r"(?:(?:heute|gerade|sehr|wirklich|ganz|nicht|mehr|besonders|so)\s+)*wohl\b|"
                r"\b(?:geht|ist|bin|bist)\b.*\bin\s+ordnung\b|"
                r"\bfreu(?:e|st|t|en)?\s+(?:mich|dich|sich|uns|euch)\b|"
                r"\b(?:gute[rn]?|schlechte[rn]?)\s+laune\b",
                clause,
            )
            if affect:
                affect_words = affect.group(0)
                negative_base = "schlecht" in affect_words
                negated_affect = bool(_word(r"nicht|nie|keineswegs|kein(?:en|e|er|es)?", clause[:affect.end()]))
                mood = "negative" if negative_base else "positive"
                value = ("neutral" if negative_base else "negative") if negated_affect else mood
                add("mood_statement", raw, target=_target(clause), value=value,
                    topic="wellbeing", negated=negated_affect,
                    confidence=0.85 if negated_affect else 1.0)
                continue
            for mood, words in MOODS.items():
                for match in re.finditer(r"\b(?:" + words + r")\b", clause):
                    prefix = clause[:match.start()]
                    # Restrict adjectives to state predicates or short elliptical
                    # replies. 'Ein gutes Buch' is not a feeling assertion.
                    eligible = (bool(re.search(r"\b(?:geht|fühl\w*|bin|bist|ist|sind|scheint|habe|hab)\b", prefix))
                                or len(bare.split()) <= 5)
                    if not eligible:
                        continue
                    # Adjectives before nouns usually modify those nouns.
                    if re.search(r"\b(?:frage|idee|film|buch|wetter|rezept|musik)\b", clause[match.end():]):
                        continue
                    negated = bool(re.search(r"\b(?:nicht|nie|keineswegs|kein(?:en|e|er|es)?)\s+(?:(?:so|sehr|wirklich|besonders|mehr|richtig|ganz)\s+){0,2}$", prefix))
                    value = ("negative" if mood == "positive" else "neutral") if negated else mood
                    add("mood_statement", raw, target=_target(clause), value=value,
                        topic="wellbeing", negated=negated, confidence=0.85 if negated else 1.0)
                    mood_found = True
                    break
            if mood_found:
                continue

        topic = _topic(clause)
        if (re.search(r"\b(?:lass(?:t)?\s+uns|ich\s+(?:will|möchte|würde\s+gern)|können\s+wir|wollen\s+wir)\b", clause)
                and _word(r"reden|sprechen|plaudern|unterhalten", clause)):
            add("talk_request", raw, topic=topic or "conversation")
            continue
        if re.search(r"\bworüber\s+(?:können|sollen|wollen)\s+wir\s+(?:reden|sprechen|plaudern)\b", clause):
            add("talk_request", raw, topic="conversation")
            continue
        if topic:
            add("question" if is_question else "topic_statement", raw,
                target=_target(clause), topic=topic, value=bare)
            if is_question:
                result.question_topic = topic
            continue
        if is_question:
            add("question", raw, target=_target(clause), value=bare, confidence=0.3)
            result.unsupported.append(raw)
        elif len(result.acts) == before or clause:
            result.unsupported.append(raw)

    # Resolve ellipses only after all explicit clauses have been interpreted.
    local_mood = any(act.kind == "mood_statement" and act.target == "user" for act in result.acts)
    contextual = context.get("last_question_topic") == "wellbeing"
    offset = 0
    for index, raw in pending_reciprocals:
        resolved = local_mood or contextual
        result.acts.insert(index + offset, SemanticAct(
            "wellbeing_question" if resolved else "context_clarification",
            target="assistant", topic="wellbeing" if resolved else "",
            confidence=0.9 if resolved else 0.0, text=raw))
        offset += 1
        if resolved:
            result.question_topic = "wellbeing"
            result.used_context |= contextual and not local_mood
        else:
            result.unsupported.append(raw)
    return result


# Each role owns a disjoint frequency interval. A concept's slot in one role
# cannot leak into another role. Known semantic categories have collision-free
# explicit slots; open vocabulary values use deterministic hash slots and can
# collide, so waves never authorize factual inference by themselves.
_ROLE_CONCEPTS = {
    "kind": ("", "greeting", "farewell", "thanks", "wellbeing_question", "mood_statement",
             "talk_request", "topic_statement", "question", "acknowledgement", "name_statement",
             "name_question", "capability_question", "identity_question", "context_clarification"),
    "target": ("", "user", "assistant", "other"),
    "value": ("", "positive", "negative", "neutral", "tired", "stressed", "lonely", "excited", "yes", "no"),
    "topic": ("", "wellbeing", "name", "identity", "capabilities", *TOPICS.keys()),
    "negated": ("False", "True"),
}


def semantic_spectrum(act: SemanticAct, dimensions: int = 512) -> np.ndarray:
    """Return a normalized complex Fourier spectrum with isolated semantic roles.

    ``np.fft.ifft(spectrum, norm='ortho')`` is its complex oscillation.  The role
    blocks are an explicit encoding convention, not a physical language law.
    Source wording and confidence do not change an otherwise identical concept.
    """
    if not isinstance(dimensions, int) or dimensions < 160:
        raise ValueError("dimensions must be an integer of at least 160")
    width = dimensions // len(_ROLE_CONCEPTS)
    spectrum = np.zeros(dimensions, dtype=np.complex128)
    for role_index, (role, known) in enumerate(_ROLE_CONCEPTS.items()):
        concept = str(getattr(act, role))
        if concept in known:
            slot = known.index(concept)
            phase = 1.0 + 0j
        else:
            digest = hashlib.sha256(concept.encode("utf-8")).digest()
            slot = len(known) + int.from_bytes(digest[:4], "big") % (width - len(known))
            phase = np.exp(2j * np.pi * int.from_bytes(digest[4:8], "big") / 2**32)
        spectrum[role_index * width + slot] = phase / np.sqrt(len(_ROLE_CONCEPTS))
    return spectrum


def semantic_similarity(a: SemanticAct, b: SemanticAct, dimensions: int = 512) -> float:
    """Phase-sensitive overlap of role-bound concepts, in [-1, 1]."""
    return float(np.clip(np.vdot(semantic_spectrum(a, dimensions),
                                 semantic_spectrum(b, dimensions)).real, -1.0, 1.0))
