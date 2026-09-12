"""Information-text ingestion, routing, and session behavior through the real API."""
import copy
import json

import numpy as np
import pytest

from freqai import cli
from freqai.generation_runtime import generator_for
from freqai.information_runtime import INFORMATION_SOURCE, information_for, information_request
from freqai.memory import Document, WaveMemory
from freqai.server import WaveEngine
from freqai.store import MemoryStore
from test_server import request, running_server


PROSE = "Frequenz\n\nDie Frequenz ist die Anzahl der Wiederholungen eines periodischen Vorgangs pro Sekunde. Die Einheit der Frequenz ist Hertz."


@pytest.fixture
def information_store(tmp_path):
    store = MemoryStore(tmp_path / "information.sqlite3")
    store.append_documents([
        Document("hello", "Hallo! Schön, dass du da bist.", "Dialog", "Hallo"),
        Document("frequency-prose", PROSE, "Wikimedia Wikipedia 20231101.de | article_id=1706 | title=Frequenz"),
    ])
    return store


def test_prose_answer_uses_token_fields_without_qa_or_retrieval(information_store, monkeypatch):
    from freqai import dialogue

    def forbidden(*args, **kwargs):
        raise AssertionError("Information generation must not retrieve an answer")

    monkeypatch.setattr(WaveMemory, "ask", forbidden)
    monkeypatch.setattr(dialogue, "respond", forbidden)
    with running_server(information_store.load_memory(), central_store=information_store) as server:
        status, result, _ = request(server, "POST", "/api/ask", {
            "prompt": "Was ist eine Frequenz?", "mode": "wave", "session_id": "unpaired", "max_tokens": 40})
        assert status == 200, result
        assert result["method"] == "autoregressive_information_spectral_decoder"
        assert result["matches"] == [] and result["wave_generation"] and result["tokens"]
        assert "Wiederholungen" in result["answer"]
        assert result["configuration"]["optimizer_steps"] == 0
        assert result["configuration"]["requires_question_answer_pairs"] is False
        assert result["context_wave"]["energy"] > 0
    assert all(not doc.prompt for doc in information_store.snapshot()[1] if doc.id == "frequency-prose")


def test_information_followup_survives_restart_and_has_real_moving_context(information_store):
    with running_server(information_store.load_memory(), central_store=information_store) as server:
        status, first, _ = request(server, "POST", "/api/ask", {
            "prompt": "Was ist eine Frequenz?", "mode": "wave", "session_id": "continued"})
        assert status == 200 and first["tokens"]
    saved = information_store.load_conversation("continued")
    reopened = MemoryStore(information_store.path)
    with running_server(reopened.load_memory(), central_store=reopened) as server:
        status, state, _ = request(server, "GET", "/api/state?session_id=continued")
        assert status == 200 and state["context_wave"]["energy"] > 0
        assert reopened.load_conversation("continued") == saved
        _, fresh, _ = request(server, "GET", "/api/state?session_id=other")
        assert "context_wave" not in fresh
        status, followup, _ = request(server, "POST", "/api/ask", {
            "prompt": "Und welche Einheit hat sie?", "mode": "wave", "session_id": "continued"})
        assert status == 200 and "Hertz" in followup["answer"]
        assert followup["conversation_revision"] == 2
        assert followup["method"] == "autoregressive_information_spectral_decoder"
    memory = reopened.load_memory()
    model = information_for(memory).model
    from freqai.information_runtime import _context_vector
    field = _context_vector(model, reopened.load_conversation("continued")[1])
    first = model.context_snapshot(field, time_s=0, points=128)
    later = model.context_snapshot(field, time_s=1.234, points=128)
    assert first["displacement"] != later["displacement"]
    assert first["energy"] == pytest.approx(later["energy"], abs=1e-12)


def test_new_prose_is_live_and_keeps_chat_and_old_information_frequencies(information_store):
    engine = WaveEngine(information_store.load_memory(), central_store=information_store)
    try:
        chat = engine.ask("Hallo", mode="wave")
        old_model = information_for(engine.memory).model
        frequencies = dict(zip(old_model.vocabulary, old_model.frequencies))
        old_chat_model = generator_for(engine.memory).model
        old_chat_vocabulary = old_chat_model.vocabulary
        engine.ask("Was ist eine Frequenz?", mode="wave", session_id="retained")
        saved = copy.deepcopy(information_store.load_conversation("retained"))
        information_store.append_documents([Document("new-prose", "Lumor\n\nEin Lumor ist ein kupferner Behälter für blaue Steine.", INFORMATION_SOURCE)])
        engine.sync()
        assert information_store.load_conversation("retained") == saved
        added = engine.ask("Was ist ein Lumor?", mode="wave", session_id="new")
        assert added["tokens"] and "Behälter" in added["answer"]
        model = information_for(engine.memory).model
        assert all(model.frequencies[model.index[word]] == frequency for word, frequency in frequencies.items())
        assert generator_for(engine.memory).model.vocabulary == old_chat_vocabulary
        assert generator_for(engine.memory).model is old_chat_model
        assert engine.ask("Hallo", mode="wave")["tokens"] == chat["tokens"]
    finally:
        engine.close()


def test_paired_append_reuses_information_operator_only_in_same_memory_lineage(information_store):
    memory = information_store.load_memory()
    model = information_for(memory).model
    extended = memory.with_documents_added([Document("new-chat", "Gern geschehen.", "Dialog", "Danke")])
    assert information_for(extended).model is model
    unrelated = information_store.load_memory()
    assert information_for(unrelated).model is not model


def test_cli_add_accepts_prose_without_a_prompt_and_preserves_paired_default(tmp_path, capsys):
    path = str(tmp_path / "cli-info.sqlite3")
    options = ["add", "--memory", path, "--text", PROSE]
    cli.main(options)
    assert json.loads(capsys.readouterr().out)["added"] == 1
    cli.main(options)
    assert json.loads(capsys.readouterr().out)["existing"] == 1
    cli.main(["add", "--memory", path, "--prompt", "Hallo", "--text", "Hallo zusammen!"])
    capsys.readouterr()
    docs = MemoryStore(path).snapshot()[1]
    assert docs[0].prompt == "" and docs[0].source == "Eigene Texte"
    assert docs[1].source == "Eigene Gespräche"


@pytest.mark.parametrize("prompt", ["Wie geht es dir?", "Was ist dein Name?", "Wer bist du?", "Was kannst du?"])
def test_personal_questions_keep_conversation_route(prompt):
    assert not information_request(prompt)


@pytest.mark.parametrize("prompt", [
    "Welche physikalische Eigenschaft beschreibt die Temperatur",
    "Was versteht man unter Osmose",
    "Benötigt ein Pendel eine Aufhängung?",
    "In welcher Maßeinheit wird Druck angegeben",
    "Erkläre mir bitte die Photosynthese.",
    "Wie warm ist es jetzt in meinem Zimmer?",
])
def test_general_information_grammar_routes_unknown_facts_to_support_check(prompt):
    assert information_request(prompt)


def test_unknown_question_cannot_invent_an_information_answer(information_store):
    engine = WaveEngine(information_store.load_memory(), central_store=information_store)
    try:
        result = engine.ask("Was ist ein Qztrblnx?", mode="wave")
        assert result["method"] == "autoregressive_information_spectral_decoder"
        assert result["abstained"] and result["tokens"] == []
    finally:
        engine.close()
