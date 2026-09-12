"""Session and live-memory adapter for the autoregressive spectral decoder.

The adapter never calls the legacy answer retrieval or fragment composer.
Context amplitudes are stored by token name, so a vocabulary extension cannot
silently reinterpret a saved mode as a different word.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import threading
from weakref import WeakKeyDictionary

import numpy as np

from .memory import WaveMemory


MODEL_CONFIGURATION = {"order": 2, "conditioning": "semantic"}
GENERATION_CONFIGURATION = {"method": "operator", "decoding": "beam", "prompt_gain": 1.2,
                            "split_acts": True, "require_conditioning": True}
_models = WeakKeyDictionary()
_models_lock = threading.RLock()
CATEGORIES_PATH = Path(__file__).resolve().parents[1] / "memory/language/generative_categories.json"


def corpus_annotations(documents) -> list[dict]:
    specification = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    prefix = specification["source_prefix"]
    result = []
    for doc in documents:
        if doc.source.startswith(prefix):
            category = doc.source[len(prefix):]
            features = specification["categories"].get(category, [])
            if features:
                result.append({"text": doc.text, "features": {feature: 8.0 for feature in features}})
    return result


def corpus_pairs(documents) -> list[dict]:
    """Explicit source gains control pooled statistics, never reply selection."""
    specification = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    gains = specification.get("paired_source_weights", {})
    if not isinstance(gains, dict) or any(
        not isinstance(prefix, str) or not prefix or isinstance(gain, (bool, str))
        or not np.isscalar(gain) or not np.isfinite(gain) or not 0 < gain <= 1
        for prefix, gain in gains.items()
    ):
        raise ValueError("Invalid paired source weights")
    ordered = sorted(gains, key=len, reverse=True)
    lexical_sources = specification.get("lexical_only_sources", [])
    if not isinstance(lexical_sources, list) or any(not isinstance(prefix, str) or not prefix for prefix in lexical_sources):
        raise ValueError("Invalid lexical-only source prefixes")
    lexical_sources = tuple(lexical_sources)
    pairs = []
    for doc in documents:
        if doc.prompt.strip():
            gain = next((gains[prefix] for prefix in ordered if doc.source.startswith(prefix)), 1.0)
            pairs.append({"prompt": doc.prompt, "text": doc.text, "weight": float(gain),
                          "semantic_features": not doc.source.startswith(lexical_sources)})
    return pairs


@dataclass
class CachedGenerator:
    model: object
    version: tuple[int, int, int, int]
    lock: threading.RLock
    corpus_key: tuple = ()


def generator_for(memory: WaveMemory) -> CachedGenerator:
    from .generative import GenerativeWaveModel
    stat = CATEGORIES_PATH.stat()
    version = (id(memory.documents), len(memory.documents), stat.st_mtime_ns, stat.st_size)
    with _models_lock:
        cached = _models.get(memory)
        if cached is None or cached.version != version:
            from .information_runtime import conversation_documents
            documents = conversation_documents(memory.documents)
            corpus_key = (tuple(documents), str(CATEGORIES_PATH.resolve()), stat.st_mtime_ns,
                          stat.st_size, tuple(sorted(MODEL_CONFIGURATION.items())))
            # with_documents_added copies this immutable memory's attributes.
            # An information-only import need not rebuild the identical chat
            # operator. Reuse is limited to that memory's own copy lineage.
            inherited = getattr(memory, "_conversation_generator", None)
            if inherited is not None and inherited.corpus_key == corpus_key:
                cached = CachedGenerator(inherited.model, version, inherited.lock, corpus_key)
            else:
                pairs = corpus_pairs(documents)
                model = GenerativeWaveModel([doc.text for doc in documents],
                                            conditioned_pairs=pairs, annotated_texts=corpus_annotations(documents),
                                            **MODEL_CONFIGURATION)
                cached = CachedGenerator(model, version, threading.RLock(), corpus_key)
            memory._conversation_generator = cached
            _models[memory] = cached
        return cached


def _saved_field(context: dict | None) -> dict[str, float]:
    values = (context or {}).get("generation", {}).get("field", [])
    if not isinstance(values, list) or len(values) > 256:
        raise ValueError("Invalid stored context field")
    result = {}
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Invalid stored context mode")
        token, value = item
        if not isinstance(token, str) or not token or len(token) > 200:
            raise ValueError("Invalid context token")
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError("Invalid context amplitude")
        result[token] = value
    return result


def _field_vector(model, context: dict | None):
    values = _saved_field(context)
    field = np.array([values.get(token, 0.0) for token in model.vocabulary])
    if field.max(initial=0) > 0:
        field /= field.max()
    return field


def context_snapshot(memory: WaveMemory, context: dict | None,
                     time_s: float = 0.0, points: int = 128) -> dict:
    """Analytic motion of the actual saved context modes, not a decorative sine."""
    if not np.isfinite(time_s) or not isinstance(points, int) or points < 2:
        raise ValueError("Context wave needs finite time and at least two points")
    if (context or {}).get("active_wave_decoder") == "information":
        from .information_runtime import information_context_snapshot
        return information_context_snapshot(memory, context, time_s=time_s, points=points)
    field = _saved_field(context)
    if not field:
        return {"time_s": float(time_s), "displacement": [], "quadrature": [],
                "energy": 0.0, "mode_count": 0, "carrier_size": 0}
    model = generator_for(memory).model
    # This is exactly the token vector consumed by generate(context_field=...),
    # in the complete model vocabulary, before spatial display decimation.
    amplitudes = _field_vector(model, context)
    frequencies = model.frequencies
    phases = np.exp(2j * np.pi * np.remainder(frequencies * time_s, 1.0))
    signal = np.fft.fft(amplitudes * phases, n=model.carrier_size, norm="ortho")
    indices = np.linspace(0, len(signal)-1, min(points, len(signal))).astype(int)
    return {"time_s": float(time_s), "displacement": signal.real[indices].tolist(),
            "quadrature": signal.imag[indices].tolist(),
            "energy": float(np.dot(amplitudes, amplitudes)), "mode_count": model.size,
            "carrier_size": model.carrier_size,
            "nonzero_modes": int(np.count_nonzero(amplitudes)),
            "representation": "persistent_token_amplitudes", "reference_phase": 0.0}


def respond_wave(memory: WaveMemory, prompt: str, context: dict | None = None,
                 time_s: float = 0.0, top_k: int = 1, max_tokens: int = 40,
                 seed: int = 17) -> tuple[dict, dict]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain text")
    if not np.isfinite(time_s) or type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("Invalid generation time or token limit")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 2**32-1")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    from .information_runtime import information_documents, information_request, respond_information
    if information_request(prompt, context) and information_documents(memory.documents):
        return respond_information(memory, prompt, context=context, time_s=time_s,
                                   max_tokens=max_tokens, seed=seed)
    state = copy.deepcopy(context or {})
    cached = generator_for(memory)
    model = cached.model
    with cached.lock:
        prior = _field_vector(model, state)
        output = model.generate(prompt, context_field=prior, time_s=time_s,
                                max_tokens=max_tokens, seed=seed, context=state, **GENERATION_CONFIGURATION)
        input_field, _ = model.prompt_field(prompt, context=state)
        next_field = 0.6 * prior + input_field
        # Generated tokens feed back into the context field with a smaller
        # gain. They remain conversation state, never new evidence in memory.
        for token in output["tokens"]:
            if token in model.index and token.isalpha():
                next_field[model.index[token]] += 0.08
        if next_field.max(initial=0) > 0:
            next_field /= next_field.max()
        strongest = sorted(np.flatnonzero(next_field > 1e-8),
                           key=lambda i: (-next_field[i], model.vocabulary[i]))[:256]
        stored_modes = [[model.vocabulary[i], float(next_field[i])] for i in strongest]
    previous = state.get("generation", {})
    state["generation"] = {"field": stored_modes, "recent_tokens": output["tokens"][-40:],
                           "turns": int(previous.get("turns", 0)) + 1,
                           "kind": "autoregressive_token_field", "reference_phase": 0.0}
    state["active_wave_decoder"] = "conversation"
    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    # These few discourse labels are input-encoding state. They never select a
    # reply string, and are explicitly separate from the numeric token field.
    from .semantics import analyze
    interpretation = analyze(prompt, context)
    topics = [act.topic for act in interpretation.acts if act.topic]
    if topics:
        state["last_question_topic"] = topics[-1]
    if any(act.kind == "greeting" for act in interpretation.acts):
        state["last_question_topic"] = "wellbeing"
    if any(act.kind == "farewell" for act in interpretation.acts):
        state.pop("last_question_topic", None)
    trace = output["trace"]
    result = {
        "answer": output["text"], "abstained": not bool(output["tokens"]), "matches": [],
        "method": "autoregressive_spectral_decoder", "wave_generation": True, "generated": True,
        "time_s": float(time_s), "tokens": output["tokens"], "ended": output["ended"],
        "truncated": not output["ended"], "prompt_encoding": output.get("prompt_field", {}),
        "unanswered_bands": output.get("unanswered_bands", 0),
        "generation_reason": output.get("reason"),
        "decoder": {"method": "token_field_demodulation", "steps": trace,
                    "vocabulary_size": model.size,
                    "max_transform_error": max((step.get("fft_roundtrip_max_error", 0.0) for step in trace), default=0.0),
                    "output_sha256": hashlib.sha256(output["text"].encode("utf-8")).hexdigest(),
                    "quality_note": "Tokenfolge aus dem Ergebnisfeld; numerische Korrektheit garantiert keine sinnvolle Antwort."},
        "configuration": {**MODEL_CONFIGURATION, **GENERATION_CONFIGURATION, "seed": seed,
                          "max_tokens": max_tokens},
        "note": "Wortweise Generierung aus gezählten spektralen Kopplungen. Keine Auswahl einer gespeicherten Antwort; keine Gradientenoptimierung.",
        "context_wave": context_snapshot(memory, state, time_s),
    }
    return result, state
