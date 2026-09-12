"""Persistent dialogue state exercised through real HTTP, CLI and SQLite."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from freqai import cli
from freqai.memory import Document
from freqai.server import WaveEngine
from freqai.store import MemoryStore, RevisionConflict
from test_server import request, running_server


@pytest.fixture
def dialogue_store(tmp_path):
    store = MemoryStore(tmp_path / "dialogue.sqlite3")
    fixture = Path(__file__).resolve().parents[1] / "memory/fixtures/extension_120.jsonl"
    store.append_documents(cli.read_documents(fixture))
    return store


def test_user_reported_three_turn_conversation(dialogue_store):
    original = dialogue_store.snapshot()
    prompts = ["Hallo", "Mir geht es gut, wie geht es dir denn?", "gut und dir?"]
    with running_server(dialogue_store.load_memory(), central_store=dialogue_store) as server:
        replies = []
        for revision, prompt in enumerate(prompts, 1):
            status, result, _ = request(server, "POST", "/api/ask", {
                "prompt": prompt, "session_id": "reported-dialogue"})
            assert status == 200
            assert not result["abstained"], result
            assert result["answer"] and result["conversation_revision"] == revision
            replies.append(result)
        assert "bereit" in replies[1]["answer"].lower()
        assert "bereit" in replies[2]["answer"].lower()
        status, history, _ = request(server, "GET", "/api/conversation?session_id=reported-dialogue")
        assert status == 200 and history["conversation_revision"] == 3
        assert [turn["prompt"] for turn in history["turns"]] == prompts
        assert [turn["answer"] for turn in history["turns"]] == [reply["answer"] for reply in replies]
    assert dialogue_store.snapshot() == original
    assert dialogue_store.stats()["history_count"] == 3
    assert dialogue_store.load_conversation("reported-dialogue")[1]["turn_count"] == 3


def test_sessions_and_restart_keep_name_context_private(dialogue_store):
    original = dialogue_store.snapshot()
    with running_server(dialogue_store.load_memory(), central_store=dialogue_store) as server:
        for session, name in [("alpha", "Elena"), ("beta", "Sofia")]:
            status, answer, _ = request(server, "POST", "/api/ask", {
                "prompt": f"Ich heiße {name}.", "session_id": session})
            assert status == 200 and not answer["abstained"]
    reopened = MemoryStore(dialogue_store.path)
    with running_server(reopened.load_memory(), central_store=reopened) as server:
        for session, own_name, other_name in [("alpha", "Elena", "Sofia"), ("beta", "Sofia", "Elena")]:
            status, answer, _ = request(server, "POST", "/api/ask", {
                "prompt": "Wie heiße ich?", "session_id": session})
            assert status == 200 and own_name in answer["answer"] and other_name not in answer["answer"]
            assert answer["conversation_revision"] == 2
            history = request(server, "GET", "/api/conversation?session_id=" + session)[1]
            assert other_name not in json.dumps(history, ensure_ascii=False)
        status, independent, _ = request(server, "POST", "/api/ask", {"prompt": "Wie heiße ich?"})
        assert status == 200 and "Elena" not in independent["answer"] and "Sofia" not in independent["answer"]
        assert "session_id" not in independent
        unknown = request(server, "GET", "/api/conversation?session_id=unseen")[1]
        assert unknown["turns"] == [] and unknown["conversation_revision"] == 0
    assert dialogue_store.load_conversation("unseen") == (0, {})
    assert dialogue_store.snapshot() == original
    assert dialogue_store.load_memory().ask("Elena Sofia exklusiver Verlauf")["abstained"]


def test_cli_and_http_share_session_state(dialogue_store, capsys):
    cli.main(["ask", "Ich heiße Elena.", "--session", "cli-session", "--json",
              "--memory", str(dialogue_store.path)])
    first = json.loads(capsys.readouterr().out)
    assert first["conversation_revision"] == 1
    with running_server(dialogue_store.load_memory(), central_store=dialogue_store) as server:
        status, answer, _ = request(server, "POST", "/api/ask", {
            "prompt": "Wie heiße ich?", "session_id": "cli-session"})
        assert status == 200 and "Elena" in answer["answer"]
        assert answer["conversation_revision"] == 2
    assert dialogue_store.stats()["history_count"] == 2


def test_store_cas_conflict_and_failed_history_write_are_atomic(tmp_path):
    store = MemoryStore(tmp_path / "central.sqlite3")
    result = {"answer": "Hallo", "abstained": False}
    assert store.commit_conversation("atomic", 0, "Hallo", result, {"turn_count": 1}) == 1
    with pytest.raises(RevisionConflict):
        store.commit_conversation("atomic", 0, "Stale", result, {"turn_count": 999})
    assert store.load_conversation("atomic") == (1, {"turn_count": 1})
    assert store.stats()["history_count"] == 1 and store.revision() == 0
    with sqlite3.connect(store.path) as connection:
        connection.execute("""CREATE TRIGGER test_history_failure BEFORE INSERT ON query_history
            WHEN NEW.prompt = 'fail' BEGIN SELECT RAISE(ABORT, 'simulated history disk failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        store.commit_conversation("atomic", 1, "fail", result, {"turn_count": 2})
    assert store.load_conversation("atomic") == (1, {"turn_count": 1})
    assert store.stats()["history_count"] == 1
    with pytest.raises(ValueError):
        store.commit_conversation("atomic", 1, "NaN", result, {"number": float("nan")})
    assert store.load_conversation("atomic") == (1, {"turn_count": 1})


def test_concurrent_turns_on_two_servers_retry_from_committed_context(tmp_path, monkeypatch):
    from freqai import dialogue
    first = MemoryStore(tmp_path / "concurrent.sqlite3")
    second = MemoryStore(first.path)
    barrier = threading.Barrier(2)
    conflicts = []

    def respond(memory, prompt, context=None, time_s=0.0, top_k=1):
        state = dict(context or {})
        state["turn_count"] = state.get("turn_count", 0) + 1
        state["seen"] = state.get("seen", []) + [prompt]
        return {"answer": str(state["turn_count"]), "matches": [], "abstained": False}, state

    monkeypatch.setattr(dialogue, "respond", respond)
    for store in (first, second):
        original = store.commit_conversation
        calls = [0]

        def commit(*args, original=original, calls=calls, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                barrier.wait(timeout=3)
            try:
                return original(*args, **kwargs)
            except RevisionConflict as error:
                conflicts.append(str(error))
                raise

        monkeypatch.setattr(store, "commit_conversation", commit)
    with running_server(first.load_memory(), central_store=first) as server_a:
        with running_server(second.load_memory(), central_store=second) as server_b:
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(request, server, "POST", "/api/ask", {
                    "prompt": name, "session_id": "same-session"})
                    for server, name in [(server_a, "first"), (server_b, "second")]]
                results = [future.result(timeout=5) for future in futures]
            assert all(result[0] == 200 for result in results)
            assert {result[1]["conversation_revision"] for result in results} == {1, 2}
    assert conflicts
    revision, state = first.load_conversation("same-session")
    assert revision == state["turn_count"] == 2
    turns = first.conversation_history("same-session")["turns"]
    assert [turn["prompt"] for turn in turns] == state["seen"]
    assert [turn["answer"] for turn in turns] == ["1", "2"]
    assert first.stats()["history_count"] == 2 and first.snapshot() == (0, [])


def test_same_engine_serializes_turns_without_lost_context(tmp_path, monkeypatch):
    from freqai import dialogue
    store = MemoryStore(tmp_path / "threads.sqlite3")

    def respond(memory, prompt, context=None, time_s=0.0, top_k=1):
        state = dict(context or {})
        state["turn_count"] = state.get("turn_count", 0) + 1
        return {"answer": str(state["turn_count"]), "matches": [], "abstained": False}, state

    monkeypatch.setattr(dialogue, "respond", respond)
    engine = WaveEngine(store.load_memory(), central_store=store)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda index: engine.ask(str(index), session_id="parallel"), range(24)))
    assert sorted(result["conversation_revision"] for result in results) == list(range(1, 25))
    assert store.load_conversation("parallel") == (24, {"turn_count": 24})
    assert store.stats()["history_count"] == 24
    assert len(engine._conversation_locks) == 64


@pytest.mark.parametrize("session", [None, "", " ", True, 42, [], {}, "x" * 101, "a/b", "a?b", "a\nb", "ä"])
def test_ask_rejects_invalid_explicit_session_before_logging(tmp_path, session):
    store = MemoryStore(tmp_path / "invalid.sqlite3")
    with running_server(store.load_memory(), central_store=store) as server:
        status, error, _ = request(server, "POST", "/api/ask", {"prompt": "Hallo", "session_id": session})
        assert status == 400 and "session_id" in error["error"]
    assert store.stats()["history_count"] == 0


@pytest.mark.parametrize("query", ["", "?session_id=", "?session_id=a&session_id=b", "?session_id=a&all=true",
                                     "?session_id=../../other", "?session_id=" + "x" * 1100])
def test_history_requires_one_valid_explicit_session(tmp_path, query):
    store = MemoryStore(tmp_path / "history.sqlite3")
    with running_server(store.load_memory(), central_store=store) as server:
        status, result, _ = request(server, "GET", "/api/conversation" + query)
        assert status == 400 and result["error"]


def test_history_rejects_other_web_origin(tmp_path):
    store = MemoryStore(tmp_path / "origin.sqlite3")
    with running_server(store.load_memory(), central_store=store) as server:
        status, result, _ = request(server, "GET", "/api/conversation?session_id=private",
                                    headers={"Origin": "https://unrelated.example"})
        assert status == 403 and result["error"]


def test_history_is_bounded_and_other_sessions_are_not_returned(tmp_path):
    store = MemoryStore(tmp_path / "bounded.sqlite3")
    for index in range(125):
        store.commit_conversation("long", index, str(index), {"answer": str(index)}, {"turn_count": index + 1})
    store.commit_conversation("other", 0, "Unrelated private prompt", {"answer": "Private answer"}, {})
    history = store.conversation_history("long")
    assert len(history["turns"]) == 100 and history["truncated"]
    assert history["turns"][0]["revision"] == 26 and history["turns"][-1]["revision"] == 125
    assert "private" not in json.dumps(history).lower()
    for index in range(5):
        store.commit_conversation("large", index, "x" * 20_000, {"answer": "y" * 240_000}, {})
    large = store.conversation_history("large")
    assert large["truncated"] and len(json.dumps(large).encode("utf-8")) < 1_048_576
    assert store.snapshot() == (0, [])


def test_history_schema_upgrade_preserves_legacy_rows_and_knowledge(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE query_history(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT 'old', prompt TEXT NOT NULL, result_json TEXT NOT NULL);
            INSERT INTO query_history(prompt,result_json) VALUES ('Legacy prompt','{"answer":"Legacy answer"}');
        """)
    store = MemoryStore(path)
    store.append_documents([Document("knowledge", "Bleibt vorhanden.")])
    assert store.stats()["history_count"] == 1
    assert store.conversation_history("new")["turns"] == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT prompt,result_json,session_id,conversation_revision FROM query_history").fetchone() == (
            "Legacy prompt", '{"answer":"Legacy answer"}', None, None)
    assert MemoryStore(path).snapshot() == store.snapshot()
