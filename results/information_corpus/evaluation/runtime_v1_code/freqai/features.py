"""Fixed feature hashing and unitary Fourier keys, with no fitted parameters."""

from collections import Counter
import hashlib
import re
import unicodedata

import numpy as np
from scipy.fft import fft


# Explicit language rules, not weights fitted to evaluation queries.
STOPWORDS = frozenset("""
der die das den dem des ein eine einer eines einem einen und oder aber als am an
auf aus bei bis durch für gegen im in ins ist sind war waren wird werden wurde
mit nach ohne seit über um unter vom von vor zu zum zur wie was wer welche
welcher welches wo wann warum wieso weshalb wofür kann können man es sich auch
ich du er sie wir ihr mir mich bitte erkläre erklären gib nenne hat haben heißt
the a an and or of in on at to from with is are was were be what which who
where when why how does do can please tell me explain about
""".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE) if word not in STOPWORDS]


def canonical_term(word: str) -> str:
    """Conservative, fixed German surface rules; no learned morphology.

    Original tokens remain available in legacy modes. This intentionally does
    not promise synonyms, compound splitting, or grammatical understanding.
    """
    word = word.casefold().translate(str.maketrans({"ä": "a", "ö": "o", "ü": "u"}))
    if any(character.isdigit() for character in word):
        return word
    if len(word) > 4 and word.endswith("s") and word[-2] in "bdfghklmnrt":
        word = word[:-1]
    for suffix in ("ern", "em", "en", "er", "es", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


QUESTION_FILLERS = frozenset(canonical_term(word) for word in """
viel viele lange lang genau ungefähr bitte erkläre erklären nenne beschreibe
geschieht passiert funktion rolle erfüllen bedeutet benötigt mein meinem meiner
meinen unser unserer heißt heisst finden finde information informationen
""".split())


def content_terms(text: str, *, query: bool = False) -> set[str]:
    terms = {canonical_term(word) for word in tokenize(text)}
    return terms - QUESTION_FILLERS if query else terms


def _hashed_vector(features: Counter, dimensions: int) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float64)
    for feature, count in features.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16, person=b"freqai-v1").digest()
        index = int.from_bytes(digest[:8], "little") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign * (1.0 + np.log(count))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def feature_vector(text: str, dimensions: int = 4096, mode: str = "hybrid") -> np.ndarray:
    if dimensions < 16:
        raise ValueError("dimensions must be at least 16")
    if mode == "raw_bytes":
        # Deliberately weak baseline: positions alias for texts longer than D.
        vector = np.zeros(dimensions)
        values = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
        np.add.at(vector, np.arange(len(values)) % dimensions, values - 127.5)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
    if mode not in {"words", "hybrid", "morphology"}:
        raise ValueError(f"Unknown feature mode: {mode}")
    words = tokenize(text)
    if mode == "morphology":
        words = [canonical_term(word) for word in words]
    word_features = Counter("w:" + word for word in words)
    if mode == "words":
        return _hashed_vector(word_features, dimensions)
    # Separate subspaces guarantee stable 75% word / 25% character energy.
    word_dimensions = dimensions // 2
    char_features = Counter()
    for word in words:
        padded = "^" + word + "$"
        for length in (3, 4):
            char_features.update("c:" + padded[i:i+length] for i in range(len(padded)-length+1))
    vector = np.concatenate((
        np.sqrt(0.75) * _hashed_vector(word_features, word_dimensions),
        np.sqrt(0.25) * _hashed_vector(char_features, dimensions - word_dimensions),
    ))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def text_spectrum(text: str, dimensions: int = 4096, mode: str = "hybrid") -> np.ndarray:
    return fft(feature_vector(text, dimensions, mode), norm="ortho")
