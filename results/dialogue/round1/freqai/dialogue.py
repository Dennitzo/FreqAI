"""Inspectable, training-free dialogue grammar with compositional wave output.

Language rules supply meaning; role-bound spectral overlap selects compatible
stored responses. This is bounded symbolic generalization, not an untrained LLM.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import threading
import unicodedata
from weakref import WeakKeyDictionary

import numpy as np

from .codec import encode_text
from .memory import WaveMemory
from .semantics import MOODS, SemanticAct, analyze, semantic_spectrum
from .synthesis import WaveFragment, compose

LANGUAGE_PATH = Path(__file__).resolve().parents[1] / "memory" / "language" / "german.json"
_catalogs = WeakKeyDictionary()
_catalog_lock = threading.Lock()


def _normal(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


@lru_cache(maxsize=4)
def _language(path: str, mtime_ns: int, size: int):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ({key: encode_text(value) for key, value in data["fragments"].items()},
            {key: encode_text(value) for key, value in data["topics"].items()})


def _catalog(memory: WaveMemory):
    with _catalog_lock:
        cached = _catalogs.get(memory)
        version = (id(memory.documents), len(memory.documents))
        if cached is None or cached[0] != version:
            entries = []
            for index, doc in enumerate(memory.documents):
                if not doc.prompt:
                    continue
                acts = analyze(doc.prompt).acts
                for act in acts:
                    entries.append((index, act, semantic_spectrum(act), len(acts)))
            cached = (version, entries)
            _catalogs[memory] = cached
        return cached[1]


def _head_span(text: str) -> int:
    match = re.search(r"[.!?](?:\s|$)", text)
    return len(text[:match.end()].rstrip().encode("utf-8")) if match else len(text.encode("utf-8"))


def _compatible(query: SemanticAct, stored: SemanticAct) -> bool:
    if (query.kind, query.target) != (stored.kind, stored.target):
        return False
    if query.kind == "mood_statement":
        return query.value == stored.value and not query.negated and not stored.negated
    return query.topic == stored.topic


def _general_mood_cue(prompt: str) -> bool:
    """A generic mood reply must not import an unmentioned source situation.

    Remove recognized affect words and ordinary predicate modifiers. Any
    remaining content (presentation, plan, cause, etc.) disqualifies this cue
    for semantic substitution. Its explicitly stored exact answer still works.
    """
    remainder = re.sub(r"\b(?:" + "|".join(MOODS.values()) + r")\b", " ", prompt.casefold())
    allowed = set("ich mir mich mein meine meiner meinen meines wir sind bin ist es geht fühle fühle fühlt "
                  "habe hab gerade heute jetzt seit dem den der die das am morgen aufstehen etwas ein "
                  "bisschen ziemlich sehr ganz so richtig wirklich besonders voller mit und doch noch "
                  "schon momentan im augenblick stimmung laune energie wohl in ordnung freue freue mich".split())
    return set(re.findall(r"\w+", remainder)) <= allowed


def respond(memory: WaveMemory, prompt: str, context: dict | None = None,
            time_s: float = 0.0, top_k: int = 1) -> tuple[dict, dict]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a nonempty string")
    if not np.isfinite(time_s):
        raise ValueError("Time must be finite")
    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    state = {key: value for key, value in (context or {}).items()
             if key in {"last_question_topic", "last_topic", "user_name", "turn_count"}}
    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    interpretation = analyze(prompt, state)
    stat = LANGUAGE_PATH.stat()
    grammar, topics = _language(str(LANGUAGE_PATH), stat.st_mtime_ns, stat.st_size)
    # Document-only archives retain their existing top-k factual retrieval
    # contract. Dialogue grammar is intended for paired conversation memories.
    if memory.documents and all(not doc.prompt for doc in memory.documents):
        result = memory.ask(prompt, time_s=time_s, top_k=top_k)
        by_id = {doc.id: index for index, doc in enumerate(memory.documents)}
        pieces = []
        for match in result["matches"]:
            if pieces:
                pieces.append(WaveFragment(grammar["paragraph"], "grammar:paragraph"))
            pieces.append(WaveFragment(memory.payloads[by_id[match["id"]]], "memory:" + match["id"]))
        if not pieces:
            pieces.append(WaveFragment(grammar["unknown"], "grammar:unknown"))
        synthesis = compose(pieces, time_s)
        result.update(answer=synthesis.text, interpretation=interpretation.to_dict(),
                      response_acts=["unknown" if result["abstained"] else "retrieved"],
                      generated=not any(synthesis.text == doc.text for doc in memory.documents),
                      decoder=synthesis.diagnostics)
        state.pop("last_topic", None)
        state.pop("last_question_topic", None)
        return result, state
    fragments, response_acts, parts, matches = [], [], [], []
    scores = memory.scores(prompt, time_s=time_s) if memory.documents else []
    catalog = _catalog(memory)
    multi = len(interpretation.acts) > 1

    def literal(key: str) -> WaveFragment:
        return WaveFragment(grammar[key], "grammar:" + key)

    def emit(act: str, pieces: list[WaveFragment], source: str = "grammar"):
        if fragments:
            fragments.append(literal("space"))
        fragments.extend(pieces)
        if act not in response_acts:
            response_acts.append(act)
        parts.append({"act": act, "source": source})

    def fixed(act: str, key: str):
        emit(act, [literal(key)])

    def stored_reply(act: SemanticAct, output_act: str, short: bool = False) -> bool:
        spectrum = semantic_spectrum(act)
        candidates = []
        for index, reference, wave, act_count in catalog:
            # Hard type/role gates precede the numerical match. Interference
            # alone is not permission to substitute someone else's assertion.
            if not _compatible(act, reference) or act_count != 1:
                continue
            if act.kind == "mood_statement" and not _general_mood_cue(memory.documents[index].prompt):
                continue
            semantic_score = float(np.vdot(spectrum, wave).real)
            candidates.append((semantic_score + 0.05 * float(scores[index]), -index,
                               index, semantic_score))
        if not candidates:
            return False
        _, _, index, semantic_score = max(candidates)
        doc = memory.documents[index]
        stop = _head_span(doc.text) if short else None
        # A question alone does not acknowledge a mood or answer a question.
        if short and doc.text.encode("utf-8")[:stop].decode("utf-8").endswith("?"):
            return False
        emit(output_act, [WaveFragment(memory.payloads[index], "memory:" + doc.id, stop=stop)], doc.id)
        matches.append({"id": doc.id, "text": doc.text, "source": doc.source,
                        "score": float(scores[index]), "semantic_score": semantic_score,
                        "interference": 2.0 * semantic_score, "matched_prompt": doc.prompt})
        return True

    # Exact stored cues remain authoritative. Arbitrary factual generalization
    # still requires evidence, whereas an explicitly stored answer is evidence.
    exact = next((i for i, doc in enumerate(memory.documents)
                  if _normal(doc.prompt or doc.text) == _normal(prompt)), None)
    if exact is None and (not interpretation.acts or
                          all(act.kind == "question" for act in interpretation.acts)):
        evidence = memory.ask(prompt, time_s=time_s, top_k=top_k)
        if not evidence["abstained"] and evidence.get("matches"):
            best_id = evidence["matches"][0]["id"]
            exact = next((i for i, doc in enumerate(memory.documents)
                          if doc.id == best_id and not doc.prompt), None)
    special = any(act.kind in {"name_statement", "name_question", "context_clarification"}
                  for act in interpretation.acts)
    if exact is not None and not multi and not special:
        doc = memory.documents[exact]
        kind = interpretation.acts[0].kind if interpretation.acts else ""
        output_act = {"greeting": "greeting", "farewell": "farewell", "thanks": "thanks",
                      "mood_statement": "user_mood", "wellbeing_question": "assistant_status",
                      "talk_request": "talk", "topic_statement": "topic"}.get(kind, "retrieved")
        emit(output_act, [WaveFragment(memory.payloads[exact], "memory:" + doc.id)], doc.id)
        matches.append({"id": doc.id, "text": doc.text, "source": doc.source,
                        "score": float(scores[exact]), "interference": 2.0 * float(scores[exact]),
                        "matched_prompt": doc.prompt})
    else:
        seen = set()
        for act in interpretation.acts:
            signature = (act.kind, act.target, act.value, act.topic, act.negated)
            if signature in seen:
                continue
            seen.add(signature)
            if act.kind == "greeting":
                if not stored_reply(act, "greeting", short=multi):
                    fixed("greeting", "greeting_short" if multi else "greeting")
            elif act.kind == "mood_statement":
                if act.target == "other":
                    fixed("clarification", "other_mood")
                elif act.target == "assistant":
                    fixed("assistant_status", "assistant_status")
                elif act.negated or not stored_reply(act, "user_mood", short=multi):
                    fixed("user_mood", act.value if act.value in grammar else "neutral")
                state["last_topic"] = "wellbeing"
            elif act.kind == "wellbeing_question":
                if act.target == "assistant":
                    if not stored_reply(act, "assistant_status", short=multi):
                        fixed("assistant_status", "assistant_status")
                else:
                    fixed("clarification", "user_wellbeing" if act.target == "user" else "other_mood")
                state["last_topic"] = "wellbeing"
            elif act.kind in {"farewell", "thanks"}:
                if not stored_reply(act, act.kind, short=multi):
                    fixed(act.kind, act.kind)
            elif act.kind == "name_statement":
                state["user_name"] = act.value
                emit("remember_name", [literal("remember_name_prefix"),
                                      WaveFragment(encode_text(act.value), "input:name"),
                                      literal("remember_name_suffix")])
                state["last_topic"] = "name"
            elif act.kind == "name_question":
                if state.get("user_name"):
                    emit("recall_name", [literal("recall_name_prefix"),
                                         WaveFragment(encode_text(state["user_name"]), "context:name"),
                                         literal("recall_name_suffix")])
                else:
                    fixed("clarification", "unknown_name")
                state["last_topic"] = "name"
            elif act.kind in {"identity_question", "capability_question"}:
                key = "identity" if act.kind == "identity_question" else "capabilities"
                fixed(key, key)
                state["last_topic"] = key
            elif act.kind in {"talk_request", "topic_statement"}:
                if act.topic and act.topic != "conversation" and act.topic in topics:
                    emit("topic", [literal("topic_prefix"), WaveFragment(topics[act.topic], "topic:" + act.topic),
                                   literal("topic_suffix")])
                else:
                    fixed("talk", "talk")
                state["last_topic"] = act.topic or "conversation"
            elif act.kind == "acknowledgement":
                fixed("acknowledgement", "acknowledgement")
            elif act.kind == "context_clarification":
                fixed("clarification", "context_clarification")
            elif act.kind == "question":
                fixed("unknown", "unknown")
                state.pop("last_topic", None)
                state.pop("last_question_topic", None)

        if not fragments:
            # Preserve evidence-based lexical retrieval for cues outside the
            # small dialogue grammar; it never updates the grammar itself.
            retrieved = memory.ask(prompt, time_s=time_s, top_k=top_k)
            if not retrieved["abstained"] and retrieved.get("matches"):
                best = retrieved["matches"][0]
                index = next(i for i, doc in enumerate(memory.documents) if doc.id == best["id"])
                protected = {"mood_statement", "wellbeing_question", "name_statement", "name_question"}
                if any(act.kind in protected for act in analyze(memory.documents[index].prompt).acts):
                    fixed("clarification", "clarification")
                else:
                    emit("retrieved", [WaveFragment(memory.payloads[index], "memory:" + best["id"])], best["id"])
                    matches.extend(retrieved["matches"])
            else:
                fixed("clarification", "clarification")
        elif interpretation.unsupported and not any(act in response_acts for act in {"unknown", "clarification"}):
            fixed("clarification", "clarification")

    synthesis = compose(fragments, time_s)
    # Context records the topic actually asked by the answer, and expires it
    # on a topic switch. A name or a film conversation cannot inherit 'und dir'.
    if re.search(r"wie (?:geht es dir|läuft dein tag)", synthesis.text, re.I):
        state["last_question_topic"] = "wellbeing"
        state["last_topic"] = "wellbeing"
    elif state.get("last_topic") == "wellbeing":
        state["last_question_topic"] = "wellbeing"
    elif state.get("last_topic"):
        state["last_question_topic"] = state["last_topic"]
    if "farewell" in response_acts:
        state.pop("last_question_topic", None)
        state.pop("last_topic", None)
    unsupported_only = all(act in {"unknown", "clarification"} for act in response_acts)
    return {
        "answer": synthesis.text, "abstained": unsupported_only,
        "matches": matches, "method": "semantic_wave_dialogue",
        "feature_mode": memory.feature_mode, "retrieval_policy": memory.retrieval_policy,
        "time_s": float(time_s), "interpretation": interpretation.to_dict(),
        "response_acts": response_acts, "response_parts": parts,
        "generated": not any(synthesis.text == doc.text for doc in memory.documents),
        "decoder": synthesis.diagnostics,
        "note": "Explizite Sprachregeln und rollengebundene Wellen; neue Kombinationen gespeicherter Fragmente, keine freie Wissensherleitung.",
    }, state
