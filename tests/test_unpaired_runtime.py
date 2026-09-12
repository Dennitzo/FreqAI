"""The real API uses unpaired information for chat, facts and persistent context."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from freqai.cli import read_documents
from freqai.memory import Document, WaveMemory
from freqai.server import WaveEngine
from freqai.store import MemoryStore
from freqai.unpaired_runtime import generator_for, context_snapshot
from test_server import request, running_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def knowledge(tmp_path):
    store = MemoryStore(tmp_path / "unpaired.sqlite3")
    records = read_documents(ROOT / "memory/information/conversation_facts.jsonl")
    records += [Document("frequency", "Frequenz\n\nDie Frequenz ist die Anzahl von Wiederholungen pro Zeitspanne.", "Physik"),
                Document("hertz", "Hertz\n\nDas Hertz ist die SI-Einheit der Frequenz.", "Physik")]
    store.append_documents(records)
    return store


def test_chat_and_information_use_same_model_without_retrieval(knowledge, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Information generation must never call document retrieval")
    monkeypatch.setattr(WaveMemory, "ask", forbidden)
    with running_server(knowledge.load_memory(), central_store=knowledge) as server:
        for prompt in ["Hallo", "Wie heißt du?", "Was ist eine Frequenz?"]:
            status, output, _ = request(server, "POST", "/api/ask", {"prompt": prompt, "session_id": "one"})
            assert status == 200 and output["tokens"]
            assert output["method"] == "autoregressive_unpaired_spectral_decoder"
            assert output["matches"] == []
            assert output["configuration"]["paired_documents_used"] == 0
            assert output["configuration"]["optimizer_steps"] == 0
            assert output["decoder"]["steps"]
            model = generator_for(server.engine.memory).model
            assert model is generator_for(server.engine.memory).model
        assert all(not doc.prompt for doc in server.engine.memory.documents)
        status, rejected, _ = request(server, "POST", "/api/ask", {"prompt": "Hallo", "mode": "dialogue"})
        assert status == 400
    assert not any(name in sys.modules for name in ("freqai.generative", "freqai.dialogue", "freqai.semantics"))


def test_information_context_survives_restart_and_is_isolated(knowledge):
    with running_server(knowledge.load_memory(), central_store=knowledge) as server:
        status, first, _ = request(server, "POST", "/api/ask", {
            "prompt": "Was ist eine Frequenz?", "session_id": "science"})
        assert status == 200 and first["tokens"]
    saved = knowledge.load_conversation("science")
    reopened = MemoryStore(knowledge.path)
    with running_server(reopened.load_memory(), central_store=reopened) as server:
        status, state, _ = request(server, "GET", "/api/state?session_id=science")
        assert status == 200 and state["context_wave"]["energy"] > 0
        assert reopened.load_conversation("science") == saved
        assert "context_wave" not in request(server, "GET", "/api/state?session_id=unrelated")[1]
        status, followup, _ = request(server, "POST", "/api/ask", {
            "prompt": "Und welche Einheit hat sie?", "session_id": "science"})
        assert status == 200 and "Hertz" in followup["answer"]
    memory = reopened.load_memory()
    state = reopened.load_conversation("science")[1]
    a = context_snapshot(memory, state, time_s=0)
    b = context_snapshot(memory, state, time_s=1.234)
    assert a["displacement"] != b["displacement"]
    assert a["energy"] == pytest.approx(b["energy"], abs=1e-12)


def test_live_prose_extends_same_compiler_and_keeps_token_addresses(knowledge):
    engine = WaveEngine(knowledge.load_memory(), central_store=knowledge)
    try:
        before = generator_for(engine.memory).model
        frequencies = dict(zip(before.vocabulary, before.frequencies))
        engine.ask("Hallo", session_id="saved")
        saved = copy.deepcopy(knowledge.load_conversation("saved"))
        origin = engine._origin
        knowledge.append_documents([Document("lumor", "Ein Lumor ist ein kupferner Behälter für blaue Steine.", "Eigene Texte")])
        engine.sync()
        assert engine._origin == origin
        assert knowledge.load_conversation("saved") == saved
        output = engine.ask("Was ist ein Lumor?", session_id="new")
        assert output["tokens"] and "Behälter" in output["answer"]
        after = generator_for(engine.memory).model
        assert after is not before
        assert all(after.frequencies[after.index[token]] == frequency for token, frequency in frequencies.items())
        assert engine.ask("Hallo")["method"] == output["method"]
        assert after is generator_for(engine.memory).model
    finally:
        engine.close()


def test_prompt_observations_stay_session_state_not_knowledge(knowledge):
    engine = WaveEngine(knowledge.load_memory(), central_store=knowledge)
    try:
        original = knowledge.snapshot()
        output = engine.ask("Ich heiße Zylvara.", session_id="local")
        assert output["configuration"]["paired_documents_used"] == 0
        assert knowledge.snapshot() == original
        assert knowledge.load_conversation("other") == (0, {})
        assert knowledge.stats()["history_count"] == 1
    finally:
        engine.close()


def test_no_data_cannot_generate_from_a_hidden_answer_table():
    memory = WaveMemory([])
    engine = WaveEngine(memory)
    try:
        for prompt in ("Hallo", "Wie heißt du?", "Was ist eine Frequenz?"):
            output = engine.ask(prompt)
            assert output["abstained"] and output["tokens"] == []
    finally:
        engine.close()


def test_api_reports_partial_utf8_budget_without_replacement_characters(knowledge):
    with running_server(knowledge.load_memory(), central_store=knowledge) as server:
        status, result, _ = request(server, "POST", "/api/ask", {
            "prompt": "Ich heiße Ö.", "session_id": "byte-limit", "max_tokens": 3})
        assert status == 200 and result["truncated"] and not result["ended"]
        assert "\ufffd" not in result["answer"]
        assert result["tokenization"]["utf8_truncated_bytes"] == 1
        assert result["tokenization"]["utf8_complete"] is False
        assert result["tokenization"]["decoder_steps"] == 3
