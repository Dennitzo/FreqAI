"""Live adapter for unpaired prose fields, with no fitted mixture coefficients.

Information and conversational compilers share one authoritative document store.
Their separate derived operators have separate persisted context fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import re
import threading
from weakref import WeakKeyDictionary

import numpy as np

INFORMATION_SOURCE = "https://huggingface.co/datasets/wikimedia/wikipedia"
INFORMATION_SOURCE_PREFIXES = (INFORMATION_SOURCE, "Wikimedia Wikipedia ")
AUTHORED_SOURCE = "Authored synthetic language prior / "
_models = WeakKeyDictionary()
_models_lock = threading.RLock()

# These are declared input grammar rules, not fitted semantic similarities.
_PERSONAL = re.compile(
    r"\b(?:wie\s+(?:geht\s+es\s+dir|gehts\s+dir|heißt\s+du|heisst\s+du|fühlst\s+du)|"
    r"wer\s+bist\s+du|was\s+(?:kannst|machst|magst|würdest)\s+du|"
    r"was\s+ist\s+dein\w*\s+(?:name|lieblings\w+))\b", re.I)
_REQUEST = re.compile(
    r"\b(?:was\s+(?:ist|sind|bedeutet|bezeichnet)|wer\s+(?:ist|war)|wodurch|woraus|wofür|wozu|"
    r"welch\w*\s+(?:einheit|bedeutung|funktion|aufgabe)|"
    r"wie\s+(?:funktioniert|entsteht|entstehen|berechnet|misst|unterscheidet|hängt|hängen)|"
    r"erklär\w*|erklaer\w*|definier\w*|beschreib\w*|zusammenhang|unterschied)\b", re.I)
_FOLLOWUP = re.compile(r"\b(?:einheit|formel|symbol|warum|wieso|weshalb|wofür|wozu|bedeutet|funktioniert)\b", re.I)


def information_documents(documents):
    return [doc for doc in documents if not doc.prompt.strip() and not doc.source.startswith(AUTHORED_SOURCE)]


def conversation_documents(documents):
    # Imported Wikipedia prose has its own sparse compiler; avoid multiplying
    # the older conversational feature bank by the full encyclopedia corpus.
    return [doc for doc in documents if doc.prompt.strip() or not doc.source.startswith(INFORMATION_SOURCE_PREFIXES)]


def information_request(prompt: str, context: dict | None = None) -> bool:
    if _PERSONAL.search(prompt):
        return False
    return bool(_REQUEST.search(prompt) or (
        (context or {}).get("information", {}).get("subjects") and _FOLLOWUP.search(prompt)))


@dataclass
class CachedInformation:
    model: object
    version: tuple[int, int]
    lock: threading.RLock
    corpus_key: tuple = ()


def information_for(memory):
    version = (id(memory.documents), len(memory.documents))
    with _models_lock:
        cached = _models.get(memory)
        if cached is None or cached.version != version:
            from .information import InformationWaveModel
            documents = information_documents(memory.documents)
            corpus_key = tuple(documents)
            inherited = getattr(memory, "_information_generator", None)
            if inherited is not None and inherited.corpus_key == corpus_key:
                cached = CachedInformation(inherited.model, version, inherited.lock, corpus_key)
            else:
                model = InformationWaveModel([asdict(doc) for doc in documents], order=2)
                cached = CachedInformation(model, version, threading.RLock(), corpus_key)
            memory._information_generator = cached
            _models[memory] = cached
        return cached


def _unit(values):
    vector = np.asarray(values, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector.copy()


def _context_vector(model, context):
    # Share the strict token-name-addressed persistence validator with chat.
    from .generation_runtime import _saved_field
    values = _saved_field({"generation": {"field": (context or {}).get("information", {}).get("field", [])}})
    return _unit(np.array([values.get(token, 0.0) for token in model.vocabulary]))


def information_context_snapshot(memory, context, time_s=0.0, points=128):
    model = information_for(memory).model
    field = _context_vector(model, context)
    return model.context_snapshot(field, time_s=time_s, points=points)


def respond_information(memory, prompt, context=None, time_s=0.0, max_tokens=40, seed=17):
    state = copy.deepcopy(context or {})
    cached = information_for(memory)
    model = cached.model
    previous = state.get("information", {})
    with cached.lock:
        prior = _context_vector(model, state)
        output = model.generate(prompt, context=previous, context_field=prior, time_s=time_s,
                                max_tokens=max_tokens, seed=seed, method="operator", decoding="beam")
        prompt_field, _ = model.prompt_field(prompt, context=previous)
        generated = np.zeros(model.size)
        for token in output["tokens"]:
            if token in model.index and token.isalpha():
                generated[model.index[token]] += 1
        # Symmetric unit-energy superposition; no calibrated context/source gain.
        next_field = _unit(_unit(prior) + _unit(prompt_field.token_amplitudes) + _unit(generated))
        strongest = sorted(np.flatnonzero(next_field > 1e-8),
                           key=lambda i: (-next_field[i], model.vocabulary[i]))[:256]
        saved = [[model.vocabulary[i], float(next_field[i])] for i in strongest]
    state["information"] = {**output.get("information_state", {}), "field": saved,
                            "turns": int(previous.get("turns", 0)) + 1,
                            "kind": "unpaired_information_token_field"}
    state["active_wave_decoder"] = "information"
    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    trace = output["trace"]
    result = {
        "answer": output["text"], "tokens": output["tokens"], "ended": output["ended"],
        "abstained": not bool(output["tokens"]), "truncated": not output["ended"],
        "matches": [], "generated": True, "wave_generation": True,
        "method": "autoregressive_information_spectral_decoder", "time_s": float(time_s),
        "generation_reason": output.get("reason"), "prompt_encoding": output.get("analysis", {}),
        "decoder": {"method": "unpaired_information_field_demodulation", "steps": trace,
                    "vocabulary_size": model.size,
                    "output_sha256": hashlib.sha256(output["text"].encode("utf-8")).hexdigest(),
                    "quality_note": "Wortfolge aus Informationsfeldern; Fakten und Sprachqualität bleiben experimentell.",
                    "coefficient_policy": "observed_counts_unit_energy_no_fitted_gains"},
        "configuration": {"order": 2, "method": "operator", "decoding": "beam",
                          "max_tokens": max_tokens, "seed": seed, "optimizer_steps": 0,
                          "requires_question_answer_pairs": False},
        "note": "Koeffizienten direkt aus Informationstexten berechnet; keine Gradientenoptimierung.",
        "context_wave": information_context_snapshot(memory, state, time_s),
    }
    return result, state
