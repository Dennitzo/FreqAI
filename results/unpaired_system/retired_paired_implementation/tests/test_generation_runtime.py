"""Generative dispatch, persistent fields and the public HTTP/CLI boundary."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from freqai import cli
from freqai.memory import Document, WaveMemory
from freqai.server import WaveEngine
from freqai.store import MemoryStore
from test_server import request, running_server


@pytest.fixture
def generation_store(tmp_path):
    store = MemoryStore(tmp_path / "generation.sqlite3")
    store.append_documents([
        Document("greeting", "Hallo! Wie geht es dir heute?", "Dialog", "Hallo"),
        Document("wellbeing", "Ich bin bereit für unser Gespräch.", "Dialog", "Wie geht es dir?"),
        Document("mood", "Das klingt schön. Was möchtest du heute machen?", "Dialog", "Mir geht es gut."),
        Document("music", "Ich höre gerne ruhige Musik.", "Dialog", "Ich möchte Musik hören."),
    ])
    return store


def test_wave_http_uses_token_decoder_without_any_document_retrieval(generation_store, monkeypatch):
    from freqai import dialogue

    def forbidden(*args, **kwargs):
        raise AssertionError("The generative path must never select a stored answer")

    monkeypatch.setattr(WaveMemory, "ask", forbidden)
    monkeypatch.setattr(dialogue, "respond", forbidden)
    before = generation_store.snapshot()
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        status, result, _ = request(server, "POST", "/api/ask", {
            "prompt": "Hallo, wie geht es dir?", "mode": "wave", "max_tokens": 12,
            "session_id": "wave-only", "seed": 23})
        assert status == 200, result
        assert result["mode"] == "wave" and result["wave_generation"]
        assert result["method"] == "autoregressive_spectral_decoder"
        assert result["matches"] == [] and result["generated"]
        assert result["decoder"]["method"] == "token_field_demodulation"
        assert result["decoder"]["steps"] and len(result["decoder"]["steps"]) <= 12
        assert result["conversation_revision"] == 1
    assert generation_store.snapshot() == before
    assert generation_store.stats()["history_count"] == 1
    assert generation_store.load_conversation("wave-only")[1]["generation"]["turns"] == 1


def test_fields_survive_restart_and_are_isolated_between_sessions(generation_store):
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        for session, prompt in [("alpha", "Hallo Musik"), ("beta", "Mir geht es gut")]:
            status, result, _ = request(server, "POST", "/api/ask", {
                "prompt": prompt, "mode": "wave", "session_id": session, "max_tokens": 6})
            assert status == 200, result
    alpha_before = generation_store.load_conversation("alpha")
    beta_before = generation_store.load_conversation("beta")
    assert alpha_before[1]["generation"]["field"] != beta_before[1]["generation"]["field"]
    reopened = MemoryStore(generation_store.path)
    with running_server(reopened.load_memory(), central_store=reopened) as server:
        for session in ("alpha", "beta"):
            status, state, _ = request(server, "GET", "/api/state?session_id=" + session)
            assert status == 200 and state["session_id"] == session
            assert state["conversation_revision"] == 1
            assert state["context_wave"]["mode_count"] > 0
            assert state["context_wave"]["energy"] > 0
            assert len(state["context_wave"]["displacement"]) == len(state["context_wave"]["quadrature"])
        ordinary = request(server, "GET", "/api/state")[1]
        fresh = request(server, "GET", "/api/state?session_id=fresh")[1]
        assert "context_wave" not in ordinary and "context_wave" not in fresh
        assert fresh["conversation_revision"] == 0
    assert reopened.load_conversation("alpha") == alpha_before
    assert reopened.load_conversation("beta") == beta_before
    assert reopened.load_conversation("fresh") == (0, {})


def test_switching_to_legacy_preserves_the_generative_field(generation_store):
    engine = WaveEngine(generation_store.load_memory(), central_store=generation_store)
    first = engine.ask("Hallo", mode="wave", max_tokens=5, session_id="mixed")
    field = generation_store.load_conversation("mixed")[1]["generation"]
    legacy = engine.ask("Hallo", mode="dialogue", session_id="mixed")
    assert first["wave_generation"] and legacy["mode"] == "dialogue"
    assert generation_store.load_conversation("mixed")[1]["generation"] == field
    third = engine.ask("Musik", mode="wave", max_tokens=5, session_id="mixed")
    assert third["conversation_revision"] == 3
    assert generation_store.load_conversation("mixed")[1]["generation"]["turns"] == 2
    engine.close()


def test_generate_cli_and_ask_wave_share_session_state(generation_store, capsys):
    common = ["--memory", str(generation_store.path), "--session", "cli-wave", "--json", "--max-tokens", "5"]
    cli.main(["generate", "Hallo", *common])
    first = json.loads(capsys.readouterr().out)
    cli.main(["ask", "Musik", "--mode", "wave", *common])
    second = json.loads(capsys.readouterr().out)
    assert first["wave_generation"] and second["wave_generation"]
    assert first["conversation_revision"] == 1 and second["conversation_revision"] == 2
    assert generation_store.load_conversation("cli-wave")[1]["generation"]["turns"] == 2


def test_live_import_updates_decoder_vocabulary_without_losing_the_session(generation_store):
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        options = {"prompt": "Hallo", "mode": "wave", "session_id": "live-wave", "max_tokens": 4}
        status, first, _ = request(server, "POST", "/api/ask", options)
        assert status == 200, first
        saved_context = generation_store.load_conversation("live-wave")
        status, added, _ = request(server, "POST", "/api/documents", {
            "id": "new-topic", "prompt": "Erzähl von Kastanien", "text": "Kastanien glänzen kupferbraun.",
            "source": "Live-Ergänzung"})
        assert status == 201 and added["live_revision"] > first["live_revision"]
        assert generation_store.load_conversation("live-wave") == saved_context
        state = request(server, "GET", "/api/state?session_id=live-wave")[1]
        assert state["context_wave"]["energy"] > 0
        status, second, _ = request(server, "POST", "/api/ask", {**options, "prompt": "Kastanien"})
        assert status == 200, second
        assert second["decoder"]["vocabulary_size"] > first["decoder"]["vocabulary_size"]
        assert second["live_revision"] == added["live_revision"]
        assert second["conversation_revision"] == 2


def test_wave_context_updates_are_serialized(generation_store):
    engine = WaveEngine(generation_store.load_memory(), central_store=generation_store)
    with ThreadPoolExecutor(4) as pool:
        replies = list(pool.map(lambda n: engine.ask("Hallo", mode="wave", max_tokens=3,
                                                   session_id="parallel-wave", seed=n), range(8)))
    assert sorted(reply["conversation_revision"] for reply in replies) == list(range(1, 9))
    revision, state = generation_store.load_conversation("parallel-wave")
    assert revision == state["generation"]["turns"] == 8
    assert len(generation_store.conversation_history("parallel-wave")["turns"]) == 8
    engine.close()


@pytest.mark.parametrize("options", [
    {"mode": None}, {"mode": []}, {"mode": "unknown"},
    {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 129}, {"max_tokens": 1.5},
    {"seed": True}, {"seed": -1}, {"seed": 4_294_967_296},
])
def test_invalid_generator_options_do_not_log_a_turn(generation_store, options):
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        status, result, _ = request(server, "POST", "/api/ask", {
            "prompt": "Hallo", "mode": "wave", "session_id": "invalid", **options})
        assert status == 400 and result["error"]
    assert generation_store.stats()["history_count"] == 0
    assert generation_store.load_conversation("invalid") == (0, {})


@pytest.mark.parametrize("query", ["?session_id=", "?session_id=a&session_id=b", "?session_id=a&all=true",
                                      "?session_id=../a", "?session_id=" + "x" * 1100])
def test_context_field_endpoint_requires_one_valid_session(generation_store, query):
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        status, result, _ = request(server, "GET", "/api/state" + query)
        assert status == 400 and result["error"]


def test_context_field_rejects_other_web_origin(generation_store):
    with running_server(generation_store.load_memory(), central_store=generation_store) as server:
        status, result, _ = request(server, "GET", "/api/state?session_id=private",
                                    headers={"Origin": "https://unrelated.example"})
        assert status == 403 and result["error"]
