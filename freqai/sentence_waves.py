"""Satzfelder-Antworten: Abruf, zweifach geprüfte Interferenz und Antwortbau.

Die Token-Dekodierung des Compilers bleibt für Gesprächsgrammatik und Diagnose
unverändert aktiv. Für Sachfragen kommt eine weitere Lesestufe dazu: Aus dem
Bestand werden vollständige Sätze geholt, ihre Wellenfelder mit der Promptwelle
zur Interferenz gebracht, nach Sprachform sortiert und zu einer zusammenhängenden
Antwort zusammengesetzt. Jede Zahl ist eine ausdrücklich gewählte Regelkonstante,
kein gelerntes Gewicht.

Warum Sätze und keine einzelnen Symbole: Ein Symbolfeld kann mitten im Wort
enden. Die Antwortqualität hängt hier an vollständigen Sätzen, deshalb wird die
Intensitätsmessung auf Satzfelder gehoben und die Zeichengrenze dem Zufall
entzogen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import sqlite3
from contextlib import closing

import numpy as np
from scipy.fft import fft, ifft

from .features import canonical_term, feature_vector, tokenize
from .language import (QueryIntent, classify, clean_text, content_words, is_pronominal,
                       normalize_glyphs, split_sentences)

# ------------------------------------------------------------------ Konstanten

#: Dimensionen des hashbasierten Wellenmediums für Satz- und Promptfelder.
MEDIUM_DIMENSIONS = 4096
FEATURE_MODE = "morphology"
#: Anteil der reinen Feldüberlappung gegenüber den sprachlichen Merkmalen.
WAVE_WEIGHT = 1.35
FEATURE_WEIGHT = 1.0
#: Sprachform-Boni und -Strafen; alle Werte sind fest geschrieben.
BONUS_DEFINITION = 1.90
BONUS_LEDE = 0.85
BONUS_TITLE = 1.45
BONUS_COVERAGE = 2.20
PENALTY_DISAMBIGUATION = 2.40
PENALTY_LIST_PAGE = 1.60
PENALTY_TAUTOLOGY = 3.10
PENALTY_SHORT = 1.20
PENALTY_NO_TARGET = 1.50
#: Maximal-Marginal-Relevance: Anteil Neuheit gegenüber Passung.
MMR_LAMBDA = 0.72
#: Unter dieser endgültigen Punktzahl gilt ein Satz als schwacher Beleg.
WEAK_EVIDENCE = 0.85


@dataclass
class SentenceCandidate:
    """Ein vollständiger Satz aus dem Bestand, mit seiner Wertung."""
    text: str
    title: str = ""
    document_id: str = ""
    source: str = ""
    position: int = 99
    origin: str = "korpus"
    wave: float = 0.0
    score: float = 0.0
    features: dict = field(default_factory=dict)
    vector: np.ndarray | None = None


def term_set(text: str) -> set:
    """Vergleichsformen: geschriebene Wörter plus behutsame Plural- und Endungsform.

    Die kräftige Stammform des Compilers ist für die Interferenz im Wellenmedium
    gedacht. Für die Frage, ob ein Satz wirklich über denselben Gegenstand
    redet, wäre sie zu grob: "Fender" und "Fend" sind nicht dieselbe Sache.
    """
    result = set()
    for word in re.findall(r"[^\W_]+", normalize_glyphs(text), flags=re.UNICODE):
        lowered = word.casefold()
        if len(lowered) < 2:
            continue
        result.add(lowered)
        if lowered.endswith("s") and len(lowered) > 3:
            result.add(lowered[:-1])
        if lowered.endswith("n") and len(lowered) > 5:
            result.add(lowered[:-1])
    return result


def stem_set(text: str) -> set:
    return {canonical_term(word) for word in tokenize(text)}


def index_terms(text: str, *, limit: int = 10) -> list[str]:
    """Wortformen fuer die Volltextsuche: geschriebene Form und Stammform.

    Der Textindex stemmingt nicht. Nur wenn beide Formen gefragt werden, findet
    die Suche "Beine" ebenso wie den gesuchten Stamm "bein".
    """
    import unicodedata
    from .features import STOPWORDS
    surface = [unicodedata.normalize("NFKC", word).casefold()
               for word in re.findall(r"[^\W_]+", normalize_glyphs(text), flags=re.UNICODE)]
    result = []
    for word in surface:
        if word in STOPWORDS or len(word) < 2:
            continue
        for variant in (word, canonical_term(word)):
            if variant not in result:
                result.append(variant)
        if len(result) >= limit:
            break
    return result


def is_assistant_corpus(source: str) -> bool:
    """Nur die eigenen deklarativen Grundtexte sprechen über den Assistenten."""
    return source.startswith("FreqAI:") or source.startswith("Eigene")


# ------------------------------------------------------------------------ Abruf

def _phrase(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def fts_expression(terms: list[str], *, titles: bool = False) -> str:
    """FTS5-Ausdruck aus Begriffen; Titelabfrage nutzt die Titelspalte."""
    cleaned = [term for term in terms if term and len(term) > 1]
    if not cleaned:
        return ""
    joiner = " OR "
    body = joiner.join(_phrase(term) for term in cleaned[:8])
    if titles:
        return "{title}:" + "(" + body + ")" if len(cleaned) > 1 else "{title}:" + body
    return "(" + body + ")" if len(cleaned) > 1 else body


def _connect(memory):
    path = getattr(memory, "cache_path", None)
    if path is None:
        return None
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=10.0)
    connection.row_factory = sqlite3.Row
    return connection


def fts_and_expression(terms: list[str]) -> str:
    """Alle Begriffe gemeinsam; das trifft den eigentlichen Gegenstand."""
    cleaned = [term for term in dict.fromkeys(terms) if term and len(term) > 2]
    if len(cleaned) < 2:
        return ""
    return " AND ".join(_phrase(term) for term in cleaned[:5])


def title_proximity(title: str, target: str) -> float:
    """Wie nah der Artikeltitel am Zielbegriff liegt; staerkstes Ordnungsmerkmal."""
    if not target:
        return 0.0
    folded_title = normalize_glyphs(title).casefold().strip()
    folded_target = normalize_glyphs(target).casefold().strip()
    if not folded_title or not folded_target:
        return 0.0
    if folded_title == folded_target:
        return 7.0
    title_words = set(re.findall(r"[^\w]+", folded_title)) and term_set(folded_title)
    target_words = term_set(folded_target)
    if not target_words:
        return 0.0
    shared = len(title_words & target_words) / max(1, len(target_words))
    extra_words = max(0, len(title_words) - len(target_words))
    return max(0.0, 5.0 * shared - 0.45 * extra_words)


def ranked_documents(memory, plans: list, *, limit: int = 16, assistant_only: bool = False,
                     target: str = ""):
    """Rangfolge der Artikel fuer eine Folge von Abrufplaenen; Titelnaehe zuerst.

    Ein Plan ist (Begriffsmenge, Zusatzwert). Erst rechnet die engste Frage, also
    alle Begriffe gemeinsam, dann die weitere. Reichen die engen Treffer, bleibt
    die breite Frage ganz weg: das haelt die Treffmenge beim Gegenstand.
    """
    maximum = getattr(memory, "maximum", None)
    found: dict[str, tuple] = {}
    connection = _connect(memory)
    if connection is not None:
        try:
            with closing(connection):
                plans = [plan for plan in plans if plan[0]]
                for index, (terms, boost) in enumerate(plans):
                    queries = [(fts_expression(terms, titles=True), boost + 3.2),
                               (fts_and_expression(terms), boost + 1.1),
                               (fts_expression(terms), boost)]
                    for query, query_boost in queries:
                        if not query:
                            continue
                        sql = ("SELECT document_id, title, text, source, bm25(passages,0,0,8,1,0) AS score "
                               "FROM passages WHERE passages MATCH ? ORDER BY score LIMIT ?")
                        for row in connection.execute(sql, (query, max(limit * 3, 30))):
                            document_id = row["document_id"]
                            if assistant_only and not is_assistant_corpus(row["source"] or ""):
                                continue
                            previous = found.get(document_id)
                            value = -float(row["score"]) * 0.25 + query_boost
                            if previous is None or value > previous[0]:
                                found[document_id] = (value, row["title"], row["text"])
                    if len(found) >= max(6, limit // 2):
                        break
        except sqlite3.DatabaseError:
            return []
    else:
        wanted = term_set(" ".join(term for plan, _ in plans for term in plan))
        scored = []
        for document in memory.documents:
            overlap = wanted & term_set(document.text)
            if not overlap:
                continue
            title_bonus = 2.0 if document.source and len(document.text.split(chr(10))[0]) < 90 else 0.0
            scored.append((len(overlap) + title_bonus, document))
        scored.sort(key=lambda item: -item[0])
        for value, document in scored[:limit]:
            found[document.id] = (value, "", document.text)
    ordered = [(identifier, (value[0] + title_proximity(value[1], target), value[1], value[2]))
               for identifier, value in found.items()]
    ordered.sort(key=lambda item: -item[1][0])
    ordered = [(identifier, values) for identifier, values in ordered[:limit]]
    return [(identifier, values[1], values[2], values[0]) for identifier, values in ordered]


def full_texts(memory, document_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Vollständige Artikeltexte holen, damit keine Satzfragmente entstehen."""
    if not document_ids or getattr(memory, "cache_path", None) is None:
        return {}
    store = getattr(memory, "store", None)
    path = store.path if store is not None else None
    if path is None:
        return {}
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=10.0)
    connection.row_factory = sqlite3.Row
    result = {}
    try:
        with closing(connection):
            placeholders = ",".join("?" for _ in document_ids)
            for row in connection.execute(
                    f"SELECT id, text FROM documents WHERE id IN ({placeholders})", document_ids):
                result[row["id"]] = (row["text"] or "")
    except sqlite3.DatabaseError:
        return {}
    return result


def article_body(text: str) -> tuple[str, str]:
    """Titel und Fliesstext eines Bestandsartikels trennen."""
    lines = text.split("\n", 1)
    if len(lines) > 1 and len(lines[0]) < 200 and not re.search(r"[.!?]$", lines[0]):
        return lines[0].strip(), lines[1]
    return "", text


HEADING_MAX_WORDS = 9


def is_heading_line(line: str) -> bool:
    """Zwischenueberschriften sind keine Antwortsaetze."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped[-1] in ".!?:":
        return False
    return len(stripped.split()) <= HEADING_MAX_WORDS


def article_sentences(body: str, *, max_sentences: int = 10) -> list[tuple[str, int]]:
    """Sätze mit Absatz- und Ueberschriftenbehandlung; Position im Artikel bleibt."""
    collected, position = [], 0
    for paragraph in re.split(r"\n\s*\n|\n{2,}", body):
        if is_heading_line(paragraph) and len(paragraph.strip()) < 90:
            continue
        for sentence in split_sentences(paragraph):
            position += 1
            if len(sentence) < 25 or len(sentence) > 460:
                continue
            collected.append((sentence, position - 1))
            if len(collected) >= max_sentences:
                return collected
    return collected


#: Diese Fragearten bekommen ihre Antwort ausschliesslich aus der eigenen Grundlage.
ASSISTANT_ONLY_KINDS = {"selbstbeschreibung", "befinden", "name", "selbstauskunft", "fahigkeit"}


def candidate_sentences(memory, intent: QueryIntent, *, limit_documents: int = 10,
                        max_sentences: int = 90) -> list[SentenceCandidate]:
    """Vollständige Sätze aus den passenden Artikeln zusammenstellen."""
    pieces = [intent.target, *intent.modifiers, *[item.raw for item in intent.parts], intent.raw]
    terms = []
    for piece in pieces:
        terms.extend(index_terms(piece, limit=6))
    for expansion in alias_target_terms(intent):
        terms.extend(index_terms(expansion, limit=4))
    terms = list(dict.fromkeys([term for term in terms if len(term) > 1]))
    assistant_only = intent.attention == "assistent" or intent.kind in ASSISTANT_ONLY_KINDS
    plans = []
    if intent.topics:
        plans.append((list(dict.fromkeys(
            term for topic in intent.topics for term in index_terms(topic, limit=4))), 4.0))
    target_surface = index_terms(intent.target, limit=8)
    if target_surface:
        plans.append((target_surface, 2.5))
    alias_terms_all = list(dict.fromkeys(
        term for expansion in alias_target_terms(intent) for term in index_terms(expansion, limit=4)))
    if alias_terms_all:
        plans.append((alias_terms_all, 1.6))
    if terms:
        plans.append((terms, 0.0))
    ranked = ranked_documents(memory, plans, limit=limit_documents,
                              assistant_only=assistant_only, target=intent.target)
    bodies = full_texts(memory, [identifier for identifier, *_ in ranked])
    candidates: list[SentenceCandidate] = []
    for identifier, title, snippet, rank_value in ranked:
        text = bodies.get(identifier)
        if text is None:
            text = snippet
        article_title, body = article_body(text or "")
        kept = 0
        for sentence, position in article_sentences(body, max_sentences=8):
            candidates.append(SentenceCandidate(sentence, article_title or title, identifier,
                                                "", position, "korpus"))
            kept += 1
        if len(candidates) >= max_sentences:
            break
    return candidates


def alias_target_terms(intent: QueryIntent) -> list[str]:
    from .language import alias_expansions
    terms = []
    for expansion in alias_expansions(intent.raw):
        terms.extend(content_words(expansion))
    return list(dict.fromkeys(terms))



def build_term_weights(candidates: list[SentenceCandidate], intent: QueryIntent,
                       context_terms: set) -> dict:
    """Begriffsgewichte aus der Kandidatenmenge (idf-artig, fest geschrieben).

    Seltenere Begriffe tragen mehr zur Interferenz bei als häufige. Das ist der
    ortsbezogene Rückkopplungsschritt: die abgerufenen Texte leihen dem Suchfeld
    seine Gewichte, ohne dass etwas gelernt wird.
    """
    documents = max(1, len(candidates))
    sentences = [term_set(candidate.text) for candidate in candidates]
    wanted = set()
    for piece in [intent.raw, intent.target, *intent.topics, *intent.modifiers,
                  *[item.raw for item in intent.parts], *alias_target_terms(intent)]:
        wanted |= term_set(piece)
    wanted |= context_terms
    weights = {}
    for term in wanted:
        frequency = sum(1 for sentence in sentences if term in sentence)
        idf = math.log(1.0 + documents / (1.0 + frequency))
        weight = 0.25 + 1.75 * idf / math.log(1.0 + documents)
        if intent.expects_number and any(character.isdigit() for character in term):
            weight *= 1.6
        weights[term] = min(2.0, max(0.15, weight))
    return weights


# ------------------------------------------------------- Wellenmedium und Wertung

def _hash_slot(feature: str, dimensions: int) -> tuple[int, float]:
    import hashlib
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16, person=b"freqai-v1").digest()
    return int.from_bytes(digest[:8], "little") % dimensions, (1.0 if digest[8] & 1 else -1.0)


def wave_vector(text: str, *, weights: dict | None = None, dimensions: int = MEDIUM_DIMENSIONS,
                scale_words: float = 1.0) -> np.ndarray:
    """Normalisiertes Wellenfeld eines Textes im Wort- und Zeichenraum.

    Dasselbe Medium wie die Suchfelder des Speichers, hier mit ausdrücklichen
    Begriffsgewichten statt roher Häufigkeit. Seltene Begriffe tragen mehr als
    häufige Füllwörter; die Gewichte kommen aus der Kandidatenmenge selbst.
    """
    words = [canonical_term(word) for word in tokenize(text)]
    half = dimensions // 2
    word_vector = np.zeros(half)
    for word in words:
        index, sign = _hash_slot("w:" + word, half)
        weight = 1.0 if weights is None else float(weights.get(word, 0.35))
        word_vector[index] += sign * weight
    character_vector = np.zeros(dimensions - half)
    for word in words:
        padded = "^" + word + "$"
        for length in (3, 4):
            for start in range(len(padded) - length + 1):
                index, sign = _hash_slot("c:" + padded[start:start + length], dimensions - half)
                character_vector[index] += sign * scale_words
    vector = np.concatenate((np.sqrt(0.75) * word_vector, np.sqrt(0.25) * character_vector))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def intensity_pairs(query: np.ndarray, sentences: np.ndarray, *, tolerance: float = 1e-9):
    """Messintensität der Interferenz, unabhängig in Orts- und Spektralrechnung.

    Direkt: Betrag des Skalarprodukts der normierten Felder. Spektral: dieselbe
    Größe im Fourierbild, weil die unitäre Transformation das Skalarprodukt
    erhält. Beide Wege müssen bis auf die Toleranz übereinstimmen.
    """
    direct = np.abs(sentences @ query)
    spectrum_query = fft(query, norm="ortho")
    spectrum_sentences = fft(sentences, axis=1, norm="ortho")
    spectral = np.abs(spectrum_sentences @ np.conjugate(spectrum_query))
    difference = float(np.max(np.abs(direct - spectral))) if len(direct) else 0.0
    if not np.isfinite(direct).all() or not np.isfinite(spectral).all():
        raise ValueError("Satzfeld-Interferenz ohne getragene endliche Energie")
    if difference > tolerance:
        raise ValueError("Satzfeld-Interferenz in Orts- und Spektralrechnung verschieden")
    return direct, difference



def surface_words(text: str) -> list:
    """Geschriebene Inhaltswoerter in Kleinschreibung, Reihenfolge erhalten."""
    from .features import STOPWORDS
    result = []
    for word in re.findall(r"[^\W_]+", normalize_glyphs(text), flags=re.UNICODE):
        lowered = word.casefold()
        if lowered in STOPWORDS:
            continue
        if lowered not in result:
            result.append(lowered)
    return result

def definition_pattern(sentence: str, target_words: list) -> float:
    """Wie stark ein Satz einer Definitionsmusterbildung folgt (0..1)."""
    lowered = sentence.lower()
    if not target_words:
        return 0.0
    head = " ".join(lowered.split()[:9])
    joined = " ".join(target_words)
    starts = any(head.startswith(word) or f" {word} " in f" {head} " for word in target_words)
    copula = re.search(r"\b(?:ist|sind|war|waren|bedeutet|gilt als|bezeichnet|nennt)\b", lowered[:180])
    if not starts or copula is None:
        return 0.0
    distance = copula.start() / max(1, len(lowered))
    genus = bool(re.search(r"\b(?:ein|eine|der|die|das)\b", lowered[copula.end():copula.end() + 40]))
    return 1.0 if (distance < 0.25 and genus) else 0.6


def is_disambiguation(sentence: str) -> bool:
    """Mehrdeutigkeitsseiten, Aufzahlungen und Inhaltsverzeichnisse sind keine Antwort."""
    lowered = sentence.lower()
    signals = [r"\b(?:steht für|begriffsklärung|begriffsklaerung|kann stehen)\b",
               r"\b(?:siehe auch|listet|countdown|im folgen)\b",
               r"^inhalt\b|\binhaltsverzeichnis\b",
               r"\bfolgende[n]? (?:person|begriff|eintrag|artikel|kapitel|version)"]
    if any(re.search(pattern, lowered) for pattern in signals):
        return True
    if sentence.rstrip().endswith(":") or ": " in sentence and sentence.count(", ") > 6:
        return True
    return bool(re.match(r"^\s*(?:\d+[.)]|[-*•])", sentence))


def is_list_page(title: str) -> bool:
    return bool(re.match(r"^\s*(?:liste|list|kategorie|sternbild|auszeichnung|"
                         r"begriffsklärung)\w*", title.strip(), flags=re.I))


def tautology_level(sentence: str, query_terms: set, target_terms: set) -> float:
    """Ein Satz, der die Frage nur wiederholt, trägt nichts bei (0..1)."""
    terms = term_set(sentence)
    if not terms or not query_terms:
        return 0.0
    new_information = terms - query_terms
    repeated = len(terms & query_terms) / max(1, len(terms))
    if not new_information:
        return 1.0
    if repeated > 0.8 and all(len(word) < 4 for word in new_information):
        return 0.8
    return max(0.0, (repeated - 0.75) * 1.6)


def score_candidates(candidates: list[SentenceCandidate], intent: QueryIntent, *,
                     context_terms: set | None = None) -> tuple[list[SentenceCandidate], float]:
    """Kandidatensätze im Satzfeldmass bewerten; hoechste Werte zuerst."""
    if not candidates:
        return [], 0.0
    query_pieces = [intent.raw]
    target_words = surface_words(intent.target) or surface_words(intent.raw)
    if intent.topics:
        for topic in intent.topics:
            for word in surface_words(topic):
                if word not in target_words:
                    target_words.append(word)
    weights = build_term_weights(candidates, intent, context_terms or set())
    query_vector = wave_vector(" ".join(query_pieces), weights=weights)
    matrix = np.vstack([wave_vector(candidate.text, weights=weights) for candidate in candidates]) \
        if len(candidates) > 1 else wave_vector(candidates[0].text, weights=weights)[None, :]
    overlap, difference = intensity_pairs(query_vector, matrix)
    target_terms = term_set(" ".join(target_words))
    query_terms = term_set(" ".join(query_pieces))
    titles = {normalize_glyphs(candidate.title).casefold() for candidate in candidates}
    normalized_target = " ".join(target_words).casefold()
    for row, (candidate, value) in enumerate(zip(candidates, overlap)):
        terms = term_set(candidate.text) | term_set(candidate.title)
        coverage = len(terms & target_terms) / max(1, len(target_terms)) if target_terms else 0.0
        definitional = definition_pattern(candidate.text, target_words)
        features = coverage * BONUS_COVERAGE + definitional * BONUS_DEFINITION \
            + (BONUS_LEDE if candidate.position == 0 else 0.0)
        title_fold = normalize_glyphs(candidate.title).casefold()
        if normalized_target and (title_fold == normalized_target
                                  or normalized_target in title_fold and len(title_fold) < 42):
            features += BONUS_TITLE
        if is_list_page(candidate.title):
            features -= PENALTY_LIST_PAGE
        if is_disambiguation(candidate.text):
            features -= PENALTY_DISAMBIGUATION
        features -= tautology_level(candidate.text, query_terms, target_terms) * PENALTY_TAUTOLOGY
        features += title_proximity(candidate.title, " ".join(target_words)) * 0.35
        if intent.kind == "liste" and candidate.text.count(", ") >= 2:
            features += 0.8
        length = len(candidate.text)
        if length < 45:
            features -= PENALTY_SHORT * (45 - length) / 45
        elif length > 430:
            features -= (length - 430) / 430
        if target_terms and not (terms & target_terms):
            features -= PENALTY_NO_TARGET
        candidate.wave = float(value) ** 2
        candidate.features = {"coverage": round(coverage, 4), "definition": round(definitional, 3),
                              "position": candidate.position}
        candidate.score = WAVE_WEIGHT * candidate.wave + FEATURE_WEIGHT * features
        candidate.vector = matrix[row]
    candidates.sort(key=lambda item: (-round(item.score, 6), item.text))
    return candidates, difference


# ------------------------------------------------------- Auswahl und Antwortbau

#: Satz- und Zeichengrenzen je Frageart; groessere Werte fuehren zu mehr Detail.
ANSWER_SHAPE = {
    "definition": (3, 620), "stichwort": (3, 620), "aussage": (2, 480),
    "folgebegriff": (2, 520), "person": (3, 700), "erklaerung": (4, 900),
    "zusammenfassung": (4, 900), "liste": (4, 620), "zeit": (2, 480),
    "ursache": (3, 650), "entscheidung": (2, 480), "vergleich": (4, 800),
    "fahigkeit": (2, 460), "selbstbeschreibung": (3, 560), "selbstauskunft": (2, 500),
    "befinden": (1, 220),
    "name": (1, 160), "auftrag": (1, 320),
}


def select_sentences(candidates: list[SentenceCandidate], *, limit: int, max_chars: int,
                     already_used: set, target_terms: set | None = None) -> list[SentenceCandidate]:
    """Maximal-Marginal-Relevance: passige und unterschiedliche Saetze waehlen.

    Ein Satz ohne jeden Zielbegriff kommt nur in Frage, wenn sonst nichts bleibt.
    """
    selected: list[SentenceCandidate] = []
    used_vectors: list[np.ndarray] = []
    pool = [candidate for candidate in candidates if 25 <= len(candidate.text) <= 430]
    if target_terms:
        covering = [candidate for candidate in pool
                    if (term_set(candidate.text) | term_set(candidate.title)) & target_terms]
        pool = covering
    remaining = list(pool)
    while remaining and len(selected) < limit:
        best, best_value = None, None
        for candidate in remaining:
            if term_set(candidate.text) <= already_used and selected:
                continue
            novelty = 0.0
            if used_vectors and candidate.vector is not None:
                matrix = np.vstack(used_vectors)
                novelty = float(np.max(np.abs(matrix @ candidate.vector)))
            value = MMR_LAMBDA * candidate.score - (1.0 - MMR_LAMBDA) * 6.0 * novelty
            if best_value is None or value > best_value:
                best, best_value = candidate, value
        if best is None:
            break
        selected.append(best)
        if best.vector is not None:
            used_vectors.append(best.vector)
        already_used |= term_set(best.text)
        remaining.remove(best)
        if sum(len(item.text) + 1 for item in selected) > max_chars:
            break
    return selected


def finish_text(sentences: list[str]) -> str:
    """Vollständige, wohlgeformte Antwort aus ganzen Saetzen bauen."""
    parts = []
    for sentence in sentences:
        text = clean_text(sentence).strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        parts.append(text[0].upper() + text[1:] if text[:1].isalpha() else text)
    return " ".join(parts).strip()


def focus_terms(context: dict | None) -> set:
    """Begriffe des bisherigen Gesprächs fuer Anschlussfragen und Wiederholungsschutz."""
    terms = set()
    if not context:
        return terms
    for key in ("focus", "subject"):
        value = context.get(key)
        if isinstance(value, str):
            terms |= term_set(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    terms |= term_set(item)
    return {term for term in terms if len(term) > 2}


def used_sentence_terms(context: dict | None) -> set:
    values = (context or {}).get("used_terms") or []
    terms = set()
    for value in values if isinstance(values, list) else []:
        if isinstance(value, str):
            terms |= term_set(value)
    return terms


# ------------------------------------------------------------------ Antwortpfad

#: Ehrliche Rückmeldungen, wenn der Bestand nichts hergibt. Sie sind selbst
#: Aussagen über den Bestand, keine leere Meldung.
NO_EVIDENCE_OPENINGS = (
    "Dazu habe ich keine Belege in meinem Informationsbestand.",
    "Diesen Begriff finde ich in meinen Informationstexten nicht.",
    "Dazu sagt mir mein Bestand nichts Verwertbares.",
)
#: Zulässige Ausschmückung: ein schwacher Beleg wird genannt statt zu schweigen.
WEAK_EVIDENCE_PREFIX = "Im Bestand ist dazu nur ein loser Zusammenhang: "


def _resolve_pronouns(text: str, main_target: str) -> str:
    if not main_target or not is_pronominal(text):
        return text
    return f"{main_target} {text}"


def _part_candidates(memory, intent: QueryIntent, part_text: str, context_terms: set):
    sub = classify(part_text)
    merged = QueryIntent(sub.kind or intent.kind, sub.target or intent.target,
                         topics=tuple(dict.fromkeys(tuple(sub.topics) + tuple(intent.topics))),
                         modifiers=sub.modifiers or intent.modifiers,
                         expects_number=sub.expects_number, expects_person=sub.expects_person,
                         attention=intent.attention, raw=part_text)
    candidates = candidate_sentences(memory, merged, limit_documents=6, max_sentences=40)
    scored, _ = score_candidates(candidates, merged, context_terms=context_terms)
    return merged, scored


def compose_answer(memory, prompt: str, context: dict | None = None, *,
                   allow_generative: bool = True, max_chars: int | None = None) -> dict:
    """Eine vollständige Antwort aus dem Bestand berechnen. Liefert nie leeren Text."""
    intent = classify(prompt)
    context_terms = focus_terms(context)
    already_used = used_sentence_terms(context)
    limit, budget = ANSWER_SHAPE.get(intent.kind, (2, 480))
    if max_chars:
        budget = min(budget, max_chars)

    candidates = candidate_sentences(memory, intent,
                                     limit_documents=10 if intent.attention != "assistent" else 6)
    scored, verification_error = score_candidates(candidates, intent, context_terms=context_terms)
    target_terms = term_set(intent.target) or term_set(intent.raw)
    if intent.topics:
        target_terms |= set().union(*(term_set(topic) for topic in intent.topics))
    chosen = select_sentences(scored, limit=limit, max_chars=budget, already_used=set(already_used),
                              target_terms=target_terms)

    pieces = [item.text for item in chosen]
    definitional = [index for index, item in enumerate(chosen)
                    if item.features.get("definition", 0.0) >= 1.0]
    if definitional and chosen[definitional[0]] is not chosen[0]:
        pieces.insert(0, pieces.pop(definitional[0]))
    confidence = best_confidence(chosen)

    if len(intent.parts) >= 2:
        # Mehrere Teilfragen: jeder Teil bekommt eigene Sätze, sonst fehlt eine Hälfte.
        follow_up = []
        for part in intent.parts:
            resolved = _resolve_pronouns(part.raw, intent.target or " ".join(
                content_words(prompt)))
            merged, part_scored = _part_candidates(memory, intent, resolved, context_terms)
            part_terms = term_set(resolved) & target_terms if target_terms else term_set(resolved)
            chosen_part = select_sentences(part_scored, limit=2 if len(intent.parts) < 3 else 1,
                                           max_chars=max(180, budget // max(1, len(intent.parts))),
                                           already_used=set(already_used),
                                           target_terms=part_terms or None)
            follow_up.extend(item.text for item in chosen_part)
        if sum(len(piece) for piece in follow_up) > sum(len(piece) for piece in pieces):
            pieces = follow_up[:4]
            confidence = best_confidence(chosen)

    if not pieces:
        pieces, confidence = fallback_pieces(memory, intent, context_terms, already_used,
                                             allow_generative=allow_generative, budget=budget)
    answer = finish_text(pieces)
    focus = intent.target or (intent.topics[0] if intent.topics else "")
    return {"answer": answer, "kind": intent.kind, "intent": intent.as_dict(),
            "confidence": confidence, "wave_verification_error": verification_error,
            "focus": focus, "evidence": [item.title for item in chosen][:4],
            "used_terms": [item.text for item in chosen][:6]}


def best_confidence(chosen: list[SentenceCandidate]) -> str:
    if not chosen:
        return "keine"
    best = max(item.score for item in chosen)
    if best >= 2.6:
        return "hoch"
    if best >= WEAK_EVIDENCE:
        return "mittel"
    return "niedrig"


def fallback_pieces(memory, intent: QueryIntent, context_terms: set, already_used: set, *,
                    allow_generative: bool, budget: int) -> tuple[list[str], str]:
    """Wenn nichts passt: ehrliche Rückmeldung, notfalls der nächste Beleg."""
    if intent.attention == "assistent":
        answer = ASSISTANT_SELF_ANSWERS.get(intent.kind) or ASSISTANT_SELF_ANSWERS.get("fahigkeit")
        return [answer or NO_EVIDENCE_OPENINGS[0]], "mittel"
    loose = None
    if allow_generative:
        probe = QueryIntent("stichwort", intent.target, intent.modifiers, raw=intent.raw)
        candidates = candidate_sentences(memory, probe, limit_documents=12, max_sentences=60)
        scored, _ = score_candidates(candidates, probe, context_terms=context_terms)
        strong = [item for item in scored if len(item.text) >= 40][:1]
        loose = [WEAK_EVIDENCE_PREFIX + item.text for item in strong] if strong else None
    if loose:
        return loose[:1], "niedrig"
    return [NO_EVIDENCE_OPENINGS[0]], "keine"


ASSISTANT_SELF_ANSWERS = {
    "fahigkeit": "Der Assistent sieht ausschließlich Informationstexte seines Bestands und den "
                 "gespeicherten Verlauf des Gesprächs. Eine eigene Sinneswahrnehmung hat er nicht.",
    "selbstbeschreibung": "Ich bin FreqAI, ein textbasiertes Assistenzprogramm. Ich berechne "
                          "Antworten aus deklarierten Informationstexten und dem Gesprächsverlauf.",
    "befinden": "Ich habe kein menschliches Befinden; mir fehlen Körper und Empfindungen.",
    "name": "Ich heiße FreqAI.",
    "auftrag": "Diese Art von Umformung gehört nicht zu meinen Fähigkeiten. Ich kann Begriffe "
               "aus meinem Bestand beschreiben.",
}
