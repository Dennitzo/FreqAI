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
    # Auditable surface slots. These do not add matching authority or spectral
    # roles: subject is a speaker-adjusted dative phrase, predicate is the
    # original affect category before negation.
    subject: str = ""
    predicate: str = ""

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
    "positive": r"gut|prima|super|toll|wunderbar|großartig|glücklich|zufrieden|bestens|fantastisch|ausgezeichnet|klasse|fröhlich|froh|gut drauf",
    "negative": r"schlecht|schlimm|traurig|unglücklich|unzufrieden|niedergeschlagen|elend|mies|beschissen|miserabel|bedrückt|gedrückt|enttäuscht",
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
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.casefold()).strip()


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


def _subject(raw: str, target: str) -> str:
    """Extract only explicit third-party subjects; never guess an antecedent."""
    if target != "other":
        return ""
    match = re.search(
        r"(?:^|\b(?:nicht|bei|geht\s+es|es\s+geht)\s+)"
        r"((?:(?:mein|dein|sein|ihr|ein)(?:e|er|em|en|es)?\s+|(?:der|die|das|dem|den)\s+)?"
        r"[\wÄÖÜäöüß-]+)(?=\s+(?:ist|sind|geht|fühlt|fühlen|scheint)|\s+(?:nicht\s+)?(?:gut|schlecht|prima|müde)\b)",
        raw, re.I,
    )
    if not match:
        return ""
    subject = match.group(1)
    if _normal(subject) in {"er", "sie", "ihm", "ihr", "ihnen", "wir", "uns", "es"}:
        return ""
    # The output slot follows 'es geht ...': preserve dative forms and map
    # nominative possessors to dative, while exchanging first/second person.
    possession = {"mein": "deinem", "meine": "deiner", "meiner": "deiner", "meinem": "deinem", "meinen": "deinen", "meines": "deinem"}
    subject = re.sub(r"^mein(?:em|er|e|en|es)?\b", lambda m: possession[m.group().casefold()], subject, flags=re.I)
    # Generic definite articles lack gender information; keep the explicit
    # noun and avoid inventing a grammatical gender.
    subject = re.sub(r"^(?:der|die|das|dem|den)\s+", "", subject, flags=re.I)
    return subject


def _negated(clause: str, start: int, end: int) -> bool:
    prefix, suffix = clause[:start], clause[end:]
    before = re.search(r"\b(?:nicht|nie|keineswegs|kein(?:en|e|er|es)?)\s+"
                       r"(?:(?:so|sehr|wirklich|besonders|mehr|richtig|ganz|gerade|heute)\s+){0,3}$", prefix)
    # Predicate fronting moves negation behind the subject/verb: 'Müde bin
    # ich nicht'. Only ordinary copular material may intervene.
    after = re.match(r"\s+(?:(?:bin|bist|ist|sind|geht|es|mir|dir|ich|du|er|sie|heute|gerade|wirklich|mehr|gar)\s+)*"
                     r"(?:nicht|nie|keineswegs)\b", suffix)
    subject_focus = re.match(r"^nicht\s+(?!nur\b)(?:ich|du|\w+(?:\s+\w+)?)\s+(?:bin|bist|ist|sind)\b", clause)
    return bool(before or after or subject_focus)


def _split(text: str) -> list[str]:
    # Keep compound predicates together; split conjunctions only where a new
    # speaker, explicit question, or contrast begins.
    nominal_subject = (r"(?:(?:mein|dein|sein|ihr|ein|der|die|das)(?:e|er|em|en|es)?\s+)?"
                       r"\w+\s+(?:ist|sind|geht|fühlt|fühlen|scheint)\b")
    text = text.replace("&", " und ").replace("’", "'").replace("‘", "'")
    text = re.sub(r"\b(fragen|wissen|sagen|erfahren)\s*,\s*(?=wie\b)", r"\1 ", text, flags=re.I)
    pattern = (r"[.!;\n]+|(?<=[?])\s*|,(?!\s*dass\b)\s*|\s+aber\s+(?!schon\b|nicht\b|auch\b)|"
               r"\s+(?=sondern\b)|"
               r"\s+und\s+(?=(?:dir|du|wie|was|wer|warum|wo|ich|mir|er|sie|danke|bis|tschüss)\b|"
               r"(?:(?:vielen|besten|herzlichen|lieben)\s+){1,2}dank\b|auf\s+wiedersehen\b|"
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
    wish_intro = False

    def add(kind: str, raw: str, **kwargs) -> None:
        result.acts.append(SemanticAct(kind=kind, text=raw, **kwargs))

    for raw in clauses:
        clause = _normal(raw).strip()
        clause = re.sub(r"^(?:(?:ach|also|übrigens|na|ehrlich gesagt|nun|und|aber)\b\s*:?\s*)+", "", clause)
        bare = clause.strip("?! ")
        if not bare:
            continue
        before = len(result.acts)
        if re.fullmatch(r"ich\s+(?:wünschte|würde\s+mir\s+wünschen)", bare):
            wish_intro = True
            continue
        counterfactual = wish_intro and bool(_word(r"ginge|wäre|fühlte", clause))
        wish_intro = False

        # Disjunction of an assertion and its denial expresses uncertainty,
        # not a positive state. Keep an explicit clarification act.
        if (re.search(r"\boder\b.*\bnicht\b", clause)
                or (_word(r"vielleicht|möglicherweise|eventuell", clause)
                    and _word("|".join(MOODS.values()), clause))):
            add("context_clarification", raw, value="uncertain", topic="wellbeing", confidence=0.0)
            continue

        previous = next((a for a in reversed(result.acts) if a.kind == "mood_statement"), None)
        contrast = re.fullmatch(r"(?:sondern\s+)?(ich|mir|du|dir|er|sie)(?:\s+aber)?(?:\s+(schon|auch|nicht))?\??", bare)
        if (contrast and previous and "?" not in clause
                and (clause.startswith("sondern ") or (contrast.group(2) and contrast.group(1) not in {"du", "dir"}))):
            target = _target(contrast.group(1))
            negated = contrast.group(2) == "nicht"
            predicate = previous.predicate or previous.value
            value = ("negative" if predicate == "positive" else "neutral") if negated else predicate
            if clause.startswith("sondern "):
                result.acts.remove(previous)
            add("mood_statement", raw, target=target, value=value, predicate=predicate,
                topic="wellbeing", negated=negated)
            continue

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
        if re.fullmatch(r"(?:ich\s+)?(?:(?:vielen|besten|herzlichen|lieben)\s+){0,2}dank(?:e)?"
                        r"(?:\s+(?:dir|schön|sehr|schon mal|der nachfrage))*"
                        r"(?:\s+für(?:s)?\s+.+|\s*,?\s*dass\s+.+)?", bare):
            add("thanks", raw)
            continue
        if re.fullmatch(r"(?:ja|jawohl|genau|stimmt|richtig|nein|nee|nö|klar|verstanden|einverstanden)", bare):
            add("acknowledgement", raw, value="no" if bare in {"nein", "nee", "nö"} else "yes")
            continue
        if re.match(r"^(?:nein|nee|nö)\s+", clause):
            add("acknowledgement", raw, value="no")
            clause = re.sub(r"^(?:nein|nee|nö)\s+", "", clause)
            bare = clause.strip("?! ")

        name = re.match(r"^ich\s+(?:heiße|heisse|nenne mich)\s+([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2})[?!]*$", raw, re.I)
        if not name:
            name = re.match(r"^mein name ist\s+([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2})[?!]*$", raw, re.I)
        if not name:
            name = re.match(r"^(?:bitte\s+)?(?:nenn(?:e)?\s+mich|du\s+kannst\s+mich)\s+"
                            r"(?:(?:bitte|einfach|gern|ab jetzt)\s+)*([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2}?)"
                            r"(?:\s+nennen)?[?!]*$", raw, re.I)
        if not name:
            name = re.match(r"^ich\s+(?:möchte|will|würde)\s+(?:(?:lieber|gern|gerne|bitte|ab jetzt)\s+)*"
                            r"([\wÄÖÜäöüß-]+(?:\s+[\wÄÖÜäöüß-]+){0,2}?)\s+"
                            r"(?:genannt|angesprochen)\s+werden[?!]*$", raw, re.I)
        if name:
            if _word(r"nicht|kein|keine", name.group(1).casefold()):
                result.unsupported.append(raw)
                continue
            add("name_statement", raw, value=name.group(1).strip(), topic="name")
            continue
        if re.search(r"\bwie\s+hei(?:ß|ss)e\s+ich\b|\b(?:kennst|weißt|weisst)\s+du\s+(?:noch\s+)?meinen namen\b|\bwie\s+(?:ist|lautet)\s+mein\s+name\b|"
                     r"\b(?:merkst|erinner\w*|genannt|nannte)\b.*\b(?:meinen?\s+namen|ich)\b|"
                     r"\b(?:meinen?\s+namen|welchen\s+namen)\b.*\b(?:erinner\w*|genannt)\b", clause):
            add("name_question", raw, target="user", topic="name")
            result.question_topic = "name"
            continue
        if re.search(r"\bwer\s+bist\s+du\b|\bwas\s+bist\s+du\b|\bwie\s+hei(?:ß|ss)t\s+du\b|\bbist\s+du\s+(?:ein\s+)?(?:mensch|ki|bot|computer)\b", clause):
            add("identity_question", raw, target="assistant", topic="identity")
            continue
        if re.search(r"\bwas\s+kannst\s+du\b|\bwie\s+funktionierst\s+du\b|\b(?:kannst|könntest)\s+du\s+(?:mir\s+)?helfen\b", clause):
            add("capability_question", raw, target="assistant", topic="capabilities")
            continue

        if re.fullmatch(r"(?:und\s+)?(?:dir|du)(?:\s+(?:so|denn|eigentlich|auch|vielleicht|selbst))*\??|"
                        r"(?:und\s+)?bei\s+dir(?:\s+(?:so|denn|auch|selbst))*\??|"
                        r"(?:und\s+)?wie\s+(?:sieht|ist)\s+es\s+bei\s+dir(?:\s+aus)?\??|"
                        r"bist\s+du\s+(?:es|das)(?:\s+auch)?\??|"
                        r"geht\s+es\s+dir\s+(?:auch\s+)?so\??", clause):
            pending_reciprocals.append((len(result.acts), raw))
            continue

        mood_vocabulary = "|".join(MOODS.values()) + r"|wohl|in ordnung|guter laune"
        wellbeing = re.search(
            r"\bwie\s+(?:geht|ergeht)\s*(?:'s|s|es)?\s+(?:dir|euch|ihnen|ihm|ihr|mir)\b|"
            r"\bwie\s+(?:fühlst\s+du\s+dich|fühle\s+ich\s+mich|fühlt\s+(?:er|sie)\s+sich)\b|"
            r"\bwie\s+(?:du\s+dich|ich\s+mich|(?:er|sie)\s+sich)\s+(?:heute\s+|gerade\s+)?fühl\w*\b|"
            r"\bwie\s+(?:ist|war)\s+dein(?:e)?\s+(?:tag|befinden|stimmung)\b|"
            r"\bwie\s+läuft\s*(?:'s|s|es)?\s+(?:(?:heute|gerade|denn|so)\s+)*(?:bei\s+)?dir\b|"
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
            add("wellbeing_question", raw, target=_target(wellbeing.group(0) if wellbeing else clause), topic="wellbeing")
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
                    predicate=mood, subject=_subject(raw, _target(clause)),
                    confidence=0.85 if negated_affect else 1.0)
                continue
            for mood, words in MOODS.items():
                for match in re.finditer(r"\b(?:" + words + r")\b", clause):
                    prefix = clause[:match.start()]
                    # Restrict adjectives to state predicates or short elliptical
                    # replies. 'Ein gutes Buch' is not a feeling assertion.
                    eligible = (bool(re.search(r"\b(?:geht|ginge|läuft|fühl\w*|bin|bist|ist|sind|wäre|scheint|habe|hab)\b", clause))
                                or len(bare.split()) <= 5)
                    if not eligible:
                        continue
                    # Adjectives before nouns usually modify those nouns.
                    if re.search(r"\b(?:frage|idee|film|buch|wetter|rezept|musik)\b", clause[match.end():]):
                        continue
                    negated = _negated(clause, match.start(), match.end())
                    if counterfactual and mood == "positive" and _target(clause) == "user":
                        negated = True
                    value = ("negative" if mood == "positive" else "neutral") if negated else mood
                    add("mood_statement", text if counterfactual else raw, target=_target(clause), value=value,
                        topic="wellbeing", negated=negated, confidence=0.85 if negated else 1.0,
                        predicate=mood, subject=_subject(raw, _target(clause)))
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
    # A local self-correction replaces the prior same-speaker assertion. It
    # does not erase statements about a different person or unrelated topic.
    corrected: list[SemanticAct] = []
    for index, act in enumerate(result.acts):
        if act.kind == "mood_statement" and corrected:
            correction = _normal(act.text).startswith(("eigentlich ", "sondern "))
            if corrected[-1].kind == "acknowledgement" and corrected[-1].value == "no":
                corrected.pop()
                correction = True
            if correction:
                for old_index in range(len(corrected) - 1, -1, -1):
                    old = corrected[old_index]
                    if old.kind == "mood_statement" and old.target == act.target:
                        corrected.pop(old_index)
                        break
        corrected.append(act)
    result.acts = corrected
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
