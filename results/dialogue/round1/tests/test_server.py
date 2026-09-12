"""HTTP runtime coverage, including real worker shutdown and input boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import time

import pytest

from freqai.server import MAX_BODY_BYTES, make_server


class TinyMemory:
    def __init__(self) -> None:
        self.documents = [{"id": "seed", "text": "Eine stehende Welle entsteht durch Überlagerung.", "source": "Test"}]
        self.saved = None

    def snapshot(self, time_s: float, points: int = 256) -> dict:
        return {"time_s": time_s, "displacement": [time_s] * points,
                "quadrature": [1.0] * points, "energy": 1.0,
                "modal_count": len(self.documents) * 4, "document_count": len(self.documents)}

    def ask(self, prompt: str, top_k: int = 1, time_s: float = 0.0) -> dict:
        return {"answer": self.documents[-1]["text"], "abstained": False,
                "matches": [{**self.documents[-1], "score": .9, "interference": 1.8}],
                "time_s": time_s, "prompt": prompt, "top_k": top_k}

    def add_document(self, text: str, source: str = "user", document_id: str | None = None) -> str:
        document_id = document_id or str(len(self.documents))
        self.documents.append({"id": document_id, "text": text, "source": source})
        return document_id

    def save(self, path: Path) -> None:
        self.saved = path
        path.write_text(json.dumps(self.documents), encoding="utf-8")


@contextmanager
def running_server(memory=None, store_path=None, central_store=None):
    server = make_server(memory or TinyMemory(), port=0, store_path=store_path,
                         central_store=central_store)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert not server.engine._thread.is_alive()
        if server.engine._sync_thread is not None:
            assert not server.engine._sync_thread.is_alive()


def request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    try:
        request_headers = dict(headers or {})
        if body is not None and not isinstance(body, (bytes, str)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = response.read()
        content_type = response.getheader("Content-Type", "")
        data = json.loads(payload) if "application/json" in content_type else payload.decode("utf-8")
        return response.status, data, dict(response.getheaders())
    finally:
        connection.close()


def test_http_dashboard_and_worker_advance_without_requests():
    with running_server() as server:
        status, html, headers = request(server, "GET", "/")
        assert status == 200
        assert "Trainingsfreier Wellenspeicher" in html
        assert "<canvas" in html
        assert "https://" not in html
        assert headers["X-Content-Type-Options"] == "nosniff"
        status, initial, _ = request(server, "GET", "/api/state")
        assert status == 200
        time.sleep(.27)
        status, later, _ = request(server, "GET", "/api/state")
        assert later["ticks"] >= initial["ticks"] + 2
        assert later["time_s"] > initial["time_s"] + .15
        assert later["displacement"] != initial["displacement"]
        assert len(later["displacement"]) == 256
        status, health, _ = request(server, "GET", "/api/health")
        assert status == 200 and health["ok"]


def test_document_ingest_query_and_persistence(tmp_path):
    memory = TinyMemory()
    destination = tmp_path / "memory.json"
    with running_server(memory, destination) as server:
        text = "Schwingung mit Umlauten: ÄÖÜ. <script>alert('kein HTML')</script>"
        status, added, _ = request(server, "POST", "/api/documents", {"text": text, "source": "Eigener Text"})
        assert status == 201
        assert added["document_count"] == 2
        assert memory.saved == destination
        assert json.loads(destination.read_text(encoding="utf-8"))[-1]["text"] == text
        status, result, _ = request(server, "POST", "/api/ask", {"prompt": "Schwingung?", "top_k": 3})
        assert status == 200
        assert result["answer"] == text
        assert result["top_k"] == 3
        status, state, _ = request(server, "GET", "/api/state")
        assert state["document_count"] == 2 and state["modal_count"] == 8


def test_failed_persistence_does_not_change_live_memory(tmp_path):
    class FailingMemory(TinyMemory):
        def save(self, path):
            raise OSError("Simulated disk failure")

    memory = FailingMemory()
    with running_server(memory, tmp_path / "unwritten.json") as server:
        status, result, _ = request(server, "POST", "/api/documents", {"text": "Must not enter live memory."})
        assert status == 500 and result["error"]
        assert len(memory.documents) == 1
        assert server.engine.memory is memory
        status, state, _ = request(server, "GET", "/api/state")
        assert status == 200 and state["document_count"] == 1
        assert not (tmp_path / "unwritten.json").exists()


def test_unsupported_method_returns_json():
    with running_server() as server:
        status, result, _ = request(server, "DELETE", "/api/documents")
        assert status == 501 and result["error"]


@pytest.mark.parametrize("route,body,expected", [
    ("/api/ask", {"prompt": " "}, 400),
    ("/api/ask", {"prompt": 4}, 400),
    ("/api/ask", {"prompt": "Test", "top_k": True}, 400),
    ("/api/ask", {"prompt": "Test", "top_k": 11}, 400),
    ("/api/documents", {"text": "Test", "source": []}, 400),
    ("/api/documents", {"text": "x" * 250001}, 413),
    ("/api/ask", [], 400),
    ("/missing", {}, 404),
])
def test_invalid_json_fields(route, body, expected):
    with running_server() as server:
        status, result, _ = request(server, "POST", route, body)
        assert status == expected
        assert result["error"]


@pytest.mark.parametrize("body,headers,expected", [
    ("{broken", {"Content-Type": "application/json"}, 400),
    (b"\xff", {"Content-Type": "application/json"}, 400),
    ("{}", {"Content-Type": "text/plain"}, 415),
    ("{}", {"Content-Type": "application/json", "Origin": "https://example.com"}, 403),
    ("{}", {"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)}, 413),
])
def test_request_boundaries(body, headers, expected):
    with running_server() as server:
        status, result, _ = request(server, "POST", "/api/ask", body, headers)
        assert status == expected
        assert result["error"]


def test_real_wave_memory_roundtrip(tmp_path):
    from freqai.memory import WaveMemory

    memory = WaveMemory([])
    memory.add_document("Stehende Wellen entstehen durch Interferenz gegenläufiger Wellen.", source="Physik")
    with running_server(memory, tmp_path / "real-memory.json") as server:
        status, state, _ = request(server, "GET", "/api/state")
        assert status == 200 and state["document_count"] == 1
        assert state["modal_count"] > 0
        status, result, _ = request(server, "POST", "/api/ask", {"prompt": "Stehende Wellen entstehen durch Interferenz gegenläufiger Wellen."})
        assert status == 200
        assert not result["abstained"]
        assert "Interferenz" in result["answer"]
        status, added, _ = request(server, "POST", "/api/documents", {"text": "Fourierreihen bestehen aus Sinus und Kosinus.", "source": "Mathematik"})
        assert status == 201 and added["document_count"] == 2
        assert (tmp_path / "real-memory.json").is_file()


def test_legacy_wave_export_keeps_dialogue_prompt(tmp_path):
    from freqai.memory import WaveMemory

    memory = WaveMemory([])
    destination = tmp_path / "dialogue.npz"
    with running_server(memory, destination) as server:
        pair = {"id": "greeting", "prompt": "Hallo", "text": "Guten Tag! Was beschäftigt dich?", "source": "Alltag"}
        status, result, _ = request(server, "POST", "/api/documents", pair)
        assert status == 201 and result["added"][0] == pair
        assert server.engine.memory.documents[0].prompt == "Hallo"
        status, answer, _ = request(server, "POST", "/api/ask", {"prompt": "Hallo"})
        assert status == 200 and answer["answer"] == pair["text"]
        restored = WaveMemory.load(destination)
        assert restored.documents == server.engine.memory.documents
        assert restored.ask("Hallo")["answer"] == pair["text"]
