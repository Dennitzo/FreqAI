"""Actual HTTP/SQLite integration, atomic imports and continuing wave phases."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import time

import numpy as np
import pytest

from freqai.codec import sample_wave
from freqai.memory import Document, WaveMemory
from freqai.store import MemoryStore
from test_server import request, running_server


def prepared_store(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.append_documents([Document("seed", "Die Testanlage Nova arbeitet mit vier Sensoren.", "Test")])
    return store


def live_memory(store):
    return WaveMemory(store.snapshot()[1], dimensions=256)


def eventually(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.02)
    assert predicate(), "Live synchronization did not finish within timeout"


def test_batch_120_live_immediate_query_and_central_history(tmp_path):
    store = prepared_store(tmp_path)
    with running_server(live_memory(store), central_store=store) as server:
        old_origin = server.engine._origin
        rows = [{"id": f"extra-{index}", "text": f"Die Forschungsstation {index:03d} misst täglich {index + 7} Grad.",
                 "source": "120 zusätzliche Testtexte"} for index in range(120)]
        status, result, _ = request(server, "POST", "/api/documents", {"documents": rows})
        assert status == 201 and len(result["added"]) == 120 and result["existing"] == []
        assert result["document_count"] == 121
        assert result["revision"] == result["live_revision"] == 2
        assert server.engine._origin == old_origin
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": rows[-1]["text"]})
        assert status == 200 and answer["matches"][0]["id"] == "extra-119"
        assert answer["live_revision"] == 2
        status, state, _ = request(server, "GET", "/api/state")
        assert status == 200 and state["document_count"] == 121
        status, info, _ = request(server, "GET", "/api/memory")
        assert status == 200 and info["document_count"] == 121 and info["history_count"] == 1
        assert info["path"] == str(store.path) and info["revision"] == 2
        status, listed, _ = request(server, "GET", "/api/documents")
        assert status == 200 and listed["documents"][-1] == {**rows[-1], "prompt": ""}
        assert len(listed["documents"]) == 121
    assert MemoryStore(store.path).snapshot() == store.snapshot()


def test_external_import_updates_without_http_request_or_phase_reset(tmp_path):
    store = prepared_store(tmp_path)
    external = MemoryStore(store.path)
    with running_server(live_memory(store), central_store=store) as server:
        payload = server.engine.memory.payloads[0]
        spectrum = server.engine.memory.spectra[0].copy()
        original = sample_wave(payload, time_s=42.25, points=64)
        origin = server.engine._origin
        before = server.engine.state()
        external.append_documents([Document("external", "Orion versendet jeden Montag acht Messberichte.", "Import")])
        eventually(lambda: server.engine.state()["live_revision"] == 2)
        later = server.engine.state()
        assert later["time_s"] > before["time_s"] and later["document_count"] == 2
        assert server.engine._origin == origin
        assert server.engine.memory.payloads[0] is payload
        np.testing.assert_array_equal(server.engine.memory.spectra[0], spectrum)
        assert sample_wave(server.engine.memory.payloads[0], time_s=42.25, points=64) == original
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": "Orion versendet jeden Montag acht Messberichte."})
        assert status == 200 and answer["matches"][0]["id"] == "external"


def test_ask_synchronizes_external_import_immediately(tmp_path):
    store = prepared_store(tmp_path)
    with running_server(live_memory(store), central_store=store) as server:
        # Pause only the worker to prove ask itself performs synchronization.
        server.engine.close()
        MemoryStore(store.path).append_documents([Document("new", "Der Quarzsensor Takara hat sieben Anschlüsse.", "Import")])
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": "Der Quarzsensor Takara hat sieben Anschlüsse."})
        assert status == 200 and answer["matches"][0]["id"] == "new"
        assert answer["revision"] == 2


def test_two_servers_retry_concurrent_revision_conflict(tmp_path, monkeypatch):
    first = prepared_store(tmp_path)
    second = MemoryStore(first.path)
    barrier = threading.Barrier(2)
    conflicts = []
    for store in (first, second):
        original = store.append_documents
        calls = [0]

        def synchronized(documents, expected_revision=None, original=original, calls=calls):
            calls[0] += 1
            if calls[0] == 1:
                barrier.wait(timeout=2)
            try:
                return original(documents, expected_revision=expected_revision)
            except ValueError as error:
                conflicts.append(str(error))
                raise

        monkeypatch.setattr(store, "append_documents", synchronized)
    with running_server(live_memory(first), central_store=first) as server_a:
        with running_server(live_memory(second), central_store=second) as server_b:
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(request, server, "POST", "/api/documents", {
                    "id": name, "text": f"Messstation {name} benötigt acht Batterien.", "source": name})
                    for server, name in [(server_a, "alpha"), (server_b, "beta")]]
                results = [future.result(timeout=5) for future in futures]
            assert all(result[0] == 201 for result in results)
            assert conflicts and "expected revision" in conflicts[0]
            assert first.stats()["document_count"] == 3 and first.revision() == 3
            for server in (server_a, server_b):
                status, result, _ = request(server, "GET", "/api/documents")
                assert status == 200 and result["revision"] == 3
                assert {row["id"] for row in result["documents"]} == {"seed", "alpha", "beta"}


def test_database_transaction_failure_keeps_live_memory_and_revision(tmp_path):
    store = prepared_store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("""CREATE TRIGGER test_disk_failure BEFORE INSERT ON documents
                              WHEN NEW.id = 'fail' BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END""")
    with running_server(live_memory(store), central_store=store) as server:
        memory = server.engine.memory
        status, result, _ = request(server, "POST", "/api/documents", {"documents": [
            {"id": "prepared", "text": "Dieser Text darf nicht teilweise gespeichert werden."},
            {"id": "fail", "text": "Dieser Eintrag simuliert einen Datenbankfehler."},
        ]})
        assert status == 500 and result["error"]
        assert server.engine.memory is memory
        assert server.engine.state()["live_revision"] == 1
        assert store.revision() == 1 and store.stats()["document_count"] == 1


def test_local_fourier_preparation_does_not_stop_existing_wave(tmp_path, monkeypatch):
    store = prepared_store(tmp_path)
    preparing, release = threading.Event(), threading.Event()
    original = WaveMemory.with_documents_added

    def slow_prepare(memory, documents):
        if documents:
            preparing.set()
            assert release.wait(timeout=3)
        return original(memory, documents)

    with running_server(live_memory(store), central_store=store) as server:
        monkeypatch.setattr(WaveMemory, "with_documents_added", slow_prepare)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(request, server, "POST", "/api/documents", {"text": "Die neue Welle hat zwölf Knoten."})
            try:
                assert preparing.wait(timeout=2)
                initial = server.engine.state()
                eventually(lambda: server.engine.state()["ticks"] >= initial["ticks"] + 2)
                later = server.engine.state()
                assert later["time_s"] > initial["time_s"] + .1
                assert later["document_count"] == 1
                assert later["displacement"] != initial["displacement"]
            finally:
                release.set()
            assert future.result(timeout=3)[0] == 201
        assert server.engine.state()["document_count"] == 2


def test_external_fourier_preparation_has_separate_worker(tmp_path, monkeypatch):
    store = prepared_store(tmp_path)
    preparing, release = threading.Event(), threading.Event()
    original = WaveMemory.with_documents_added

    def slow_prepare(memory, documents):
        if documents:
            preparing.set()
            assert release.wait(timeout=3)
        return original(memory, documents)

    with running_server(live_memory(store), central_store=store) as server:
        monkeypatch.setattr(WaveMemory, "with_documents_added", slow_prepare)
        MemoryStore(store.path).append_documents([Document("external", "Hallo! Wie war dein Wochenende?", "Test", "Guten Tag")])
        try:
            assert preparing.wait(timeout=2)
            initial = server.engine.state()
            eventually(lambda: server.engine.state()["ticks"] >= initial["ticks"] + 2)
            later = server.engine.state()
            assert later["time_s"] > initial["time_s"] + .1
            assert later["document_count"] == 1
            assert later["displacement"] != initial["displacement"]
        finally:
            release.set()
        eventually(lambda: server.engine.state()["live_revision"] == 2)
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": "Guten Tag"})
        assert status == 200 and answer["matches"][0]["id"] == "external"


def test_sync_error_stays_visible_while_old_wave_advances_then_recovers(tmp_path, monkeypatch):
    store = prepared_store(tmp_path)
    original = store.changes_since

    def unavailable(revision):
        raise OSError("Simulated unavailable import")

    with running_server(live_memory(store), central_store=store) as server:
        monkeypatch.setattr(store, "changes_since", unavailable)
        MemoryStore(store.path).append_documents([Document("new", "Gern, erzähl weiter.", "Test", "Darf ich dir etwas erzählen?")])
        eventually(lambda: server.engine.state()["sync_error"] is not None)
        initial = server.engine.state()
        eventually(lambda: server.engine.state()["ticks"] >= initial["ticks"] + 2)
        status, health, _ = request(server, "GET", "/api/health")
        assert status == 503 and "Simulated unavailable import" in health["error"]
        assert server.engine.state()["document_count"] == 1
        monkeypatch.setattr(store, "changes_since", original)
        eventually(lambda: server.engine.state()["live_revision"] == 2)
        status, health, _ = request(server, "GET", "/api/health")
        assert status == 200 and health["error"] is None


def test_exact_repeated_post_is_idempotent(tmp_path):
    store = prepared_store(tmp_path)
    with running_server(live_memory(store), central_store=store) as server:
        row = {"text": "Ein Laser sendet kohärentes Licht aus.", "source": "Optik"}
        first = request(server, "POST", "/api/documents", row)[1]
        second = request(server, "POST", "/api/documents", row)[1]
        assert len(first["added"]) == 1 and second["added"] == []
        assert len(second["existing"]) == 1 and second["id"] == first["id"]
        assert first["revision"] == second["revision"] == 2
        assert second["document_count"] == 2


def test_dialogue_prompt_returns_paired_answer_without_self_indexing(tmp_path):
    store = MemoryStore(tmp_path / "dialogue.sqlite3")
    with running_server(live_memory(store), central_store=store) as server:
        pair = {"id": "greeting", "prompt": "Hallo", "text": "Hallo! Wie läuft dein Tag?", "source": "Alltag"}
        status, added, _ = request(server, "POST", "/api/documents", pair)
        assert status == 201 and added["added"][0] == pair
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": "Hallo"})
        assert status == 200 and answer["answer"] == pair["text"]
        assert answer["answer"] != pair["prompt"]
        assert store.stats()["document_count"] == 1 and store.stats()["history_count"] == 1
        assert request(server, "GET", "/api/documents")[1]["documents"] == [pair]
        status, second, _ = request(server, "POST", "/api/documents", {
            "prompt": "Guten Morgen", "text": pair["text"], "source": pair["source"]})
        assert status == 201 and len(second["added"]) == 1
        assert second["document_count"] == 2


def test_slow_database_commit_does_not_publish_an_older_frame(tmp_path, monkeypatch):
    store = prepared_store(tmp_path)
    original = store.append_documents
    writing, release = threading.Event(), threading.Event()

    def slow_write(documents, expected_revision=None):
        writing.set()
        assert release.wait(timeout=3)
        return original(documents, expected_revision=expected_revision)

    with running_server(live_memory(store), central_store=store) as server:
        monkeypatch.setattr(store, "append_documents", slow_write)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(request, server, "POST", "/api/documents", {"text": "Ein später Commit ergänzt die Welle."})
            try:
                assert writing.wait(timeout=2)
                before = server.engine.state()
                eventually(lambda: server.engine.state()["ticks"] >= before["ticks"] + 2)
                during = server.engine.state()
            finally:
                release.set()
            assert future.result(timeout=3)[0] == 201
        after = server.engine.state()
        assert after["document_count"] == 2
        assert after["time_s"] >= during["time_s"] > before["time_s"]


@pytest.mark.parametrize("rows", [
    [], ["wrong"], [{"text": "valid"}, {"text": ""}],
    [{"text": "valid"}, {"text": "invalid", "source": False}],
    [{"id": "same", "text": "one"}, {"id": "same", "text": "two"}],
    [{"text": "valid"}, {"text": "invalid", "id": 2}],
    [{"text": "valid"}, {"text": "invalid", "prompt": 2}],
])
def test_invalid_batch_is_all_or_nothing(tmp_path, rows):
    store = prepared_store(tmp_path)
    with running_server(live_memory(store), central_store=store) as server:
        status, result, _ = request(server, "POST", "/api/documents", {"documents": rows})
        assert status == 400 and result["error"]
        assert store.stats()["document_count"] == server.engine.state()["document_count"] == 1
        assert store.revision() == server.engine.state()["live_revision"] == 1
