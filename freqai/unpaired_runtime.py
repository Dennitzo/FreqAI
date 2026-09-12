"""One information-only decoder for every message and a persisted wave context."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time
from weakref import WeakKeyDictionary

import numpy as np

from .waves import WAVE_VERSION, resonance

_models = WeakKeyDictionary()
_models_lock = threading.RLock()
_build_locks = WeakKeyDictionary()


def information_documents(documents):
    records = list(documents)
    if any(doc.prompt.strip() for doc in records):
        raise ValueError("Der aktive Bestand enthält noch Paare. Zuerst migrate-information ausführen.")
    if any(doc.source.startswith("Authored synthetic language prior / ") for doc in records):
        raise ValueError("Der frühere Antwortprior muss vor der Informationsberechnung archiviert werden.")
    return records


@dataclass
class CachedGenerator:
    model: object
    version: tuple
    lock: threading.RLock
    corpus_key: tuple


def generator_for(memory):
    version = (id(memory.documents), len(memory.documents))
    with _models_lock:
        cached = _models.get(memory)
        if cached is not None and cached.version == version:
            return cached
        build_lock = _build_locks.setdefault(memory, threading.RLock())
    # Compiling one corpus must not hold the registry lock for unrelated stores.
    with build_lock:
        with _models_lock:
            cached = _models.get(memory)
        if cached is not None and cached.version == version:
            return cached
        from .unpaired import UnpairedWaveModel
        documents = information_documents(memory.documents)
        key = tuple(documents)
        inherited = getattr(memory, "_unpaired_generator", None)
        if inherited is not None and inherited.corpus_key == key:
            cached = CachedGenerator(inherited.model, version, inherited.lock, key)
        else:
            from .compiler_cache import load_compiled, load_previous_compiled, save_compiled
            records = [{"id": doc.id, "text": doc.text, "source": doc.source} for doc in documents]
            option = os.environ.get("FREQAI_COMPILER_CACHE", "auto").strip()
            cache_dir = getattr(memory, "_compiler_cache_dir", None)
            if option.lower() in {"0", "off", "false", "no"}:
                cache_dir = None
            elif option.lower() not in {"auto", "1", "on", "true"}:
                cache_dir = Path(option).expanduser().resolve()
            started = time.perf_counter()
            memory._compiler_status = {"state": "loading", "cache": "pending"}
            try:
                model = load_compiled(records, 2, cache_dir) if cache_dir is not None else None
                cache_result = "hit" if model is not None else "miss" if cache_dir is not None else "disabled"
                saved = False
                previous_loaded = False
                if model is None:
                    previous = inherited.model if inherited is not None else None
                    if previous is None and cache_dir is not None:
                        previous = load_previous_compiled(2, cache_dir)
                        previous_loaded = previous is not None
                    memory._compiler_status = {"state": "building", "cache": cache_result}
                    model = UnpairedWaveModel(records, order=2, previous_model=previous)
                    if cache_dir is not None:
                        memory._compiler_status = {"state": "saving", "cache": cache_result}
                        saved = bool(save_compiled(model, records, 2, cache_dir))
                memory._compiler_status = {
                    "state": "ready", "cache": cache_result, "cache_saved": saved,
                    "previous_cache_loaded": previous_loaded,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "reuse": getattr(model, "compile_reuse_stats", {}),
                }
                cached = CachedGenerator(model, version, threading.RLock(), key)
            except Exception as error:
                memory._compiler_status = {"state": "error", "error": type(error).__name__}
                raise
        memory._unpaired_generator = cached
        with _models_lock:
            _models[memory] = cached
        return cached


def _unit(values):
    vector = np.asarray(values, dtype=float)
    norm = np.linalg.norm(vector)
    return vector/norm if norm else vector.copy()


def _previous(context):
    # Only the prior information field can migrate by token names. Old paired
    # conversation coefficients and act annotations never enter this compiler.
    return (context or {}).get("unpaired", (context or {}).get("information", {}))


def _saved_field(context):
    values = _previous(context).get("field", [])
    if not isinstance(values, list) or len(values) > 256:
        raise ValueError("Invalid stored context field")
    result = {}
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Invalid stored context mode")
        token, amplitude = item
        if not isinstance(token, str) or not token or len(token) > 200 or isinstance(amplitude, bool):
            raise ValueError("Invalid context token or amplitude")
        value = float(amplitude)
        if not np.isfinite(value) or not 0 <= value <= 1000:
            raise ValueError("Invalid context amplitude")
        result[token] = value
    return result


def _context_vector(model, context):
    values = _saved_field(context)
    return _unit([values.get(token, 0.0) for token in model.vocabulary])


def context_snapshot(memory, context=None, time_s=0.0, points=128):
    if not _saved_field(context):
        return {"time_s": float(time_s), "displacement": [], "quadrature": [],
                "energy": 0.0, "mode_count": 0, "carrier_size": 0}
    model = generator_for(memory).model
    return model.context_snapshot(_context_vector(model, context), time_s=time_s, points=points)


def respond_wave(memory, prompt, context=None, time_s=0.0, top_k=1, max_tokens=40, seed=17):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain text")
    if not np.isfinite(time_s) or type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("Invalid generation time or token limit")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 2**32-1")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    state = copy.deepcopy(context or {})
    previous = _previous(state)
    cached = generator_for(memory)
    model = cached.model
    with cached.lock:
        prior = _context_vector(model, state)
        output = model.generate(prompt, context=previous, context_field=prior, time_s=time_s,
                                max_tokens=max_tokens, seed=seed, method="operator", decoding="beam")
        input_field, _ = model.prompt_field(prompt, context=previous)
        generated = np.zeros(model.size)
        for token in output["tokens"]:
            if token in model.index and (token.isalpha() or token[:1].isdigit() or token.startswith("<byte:")):
                generated[model.index[token]] += 1
        next_field = _unit(_unit(prior)+_unit(input_field.token_amplitudes)+_unit(generated))
        strongest = sorted(np.flatnonzero(next_field > 1e-8),
                           key=lambda i: (-next_field[i], model.vocabulary[i]))[:256]
        saved = [[model.vocabulary[i], float(next_field[i])] for i in strongest]
        data_wave = model.data_wave()
        prompt_wave = model.prompt_wave(input_field)
        waves = {
            "version": WAVE_VERSION,
            "data_wave": data_wave.snapshot(time_s=time_s),
            "prompt_wave": prompt_wave.snapshot(time_s=time_s),
            "resonance": resonance(data_wave, prompt_wave),
            "operator_vs_direct_max_error": max(
                (float(step.get("fft_roundtrip_max_error", 0.0)) for step in output["trace"]),
                default=0.0),
        }
    state["unpaired"] = {**output.get("information_state", output.get("state", {})),
                         "field": saved, "turns": int(previous.get("turns", 0))+1,
                         "kind": "unpaired_information_token_field"}
    state["active_wave_decoder"] = "unpaired"
    state["turn_count"] = int(state.get("turn_count", 0))+1
    result = {
        "answer": output["text"], "tokens": output["tokens"], "ended": output["ended"],
        "abstained": not bool(output["tokens"]), "truncated": not output["ended"],
        "matches": [], "generated": True, "wave_generation": True,
        "method": "autoregressive_unpaired_spectral_decoder", "time_s": float(time_s),
        "generation_reason": output.get("reason"), "prompt_encoding": output.get("analysis", {}),
        "tokenization": output.get("tokenization", {}),
        "decoder": {"method": "information_roles_and_complex_token_fields", "steps": output["trace"],
                    "vocabulary_size": model.size,
                    "output_sha256": hashlib.sha256(output["text"].encode("utf-8")).hexdigest(),
                    "quality_note": "Informationen und explizite Grammatik; Sprachqualität und Schlussfolgern bleiben begrenzt.",
                    "coefficient_policy": "direct_information_counts_unit_energy"},
        "configuration": {"order": 2, "method": "operator", "decoding": "beam", "max_tokens": max_tokens,
                          "seed": seed, "optimizer_steps": 0, "requires_question_answer_pairs": False,
                          "paired_documents_used": 0, "single_information_compiler": True},
        "note": "Alle Eingaben verwenden denselben Informationscompiler; keine Antwortpaare.",
        "waves": waves,
        "context_wave": context_snapshot(memory, state, time_s),
    }
    return result, state
