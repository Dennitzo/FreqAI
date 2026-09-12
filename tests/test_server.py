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
        assert "Informations" in html
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
        assert health["frame_ms"] >= 0
        assert health["compiler_ready"] is False
        assert health["compute"]["precision"] == "float64/complex128"
        assert 1 <= health["parallel"]["workers"] <= health["parallel"]["logical_cores"]


def test_document_ingest_persistence_and_reject_custom_retrieval_decoder(tmp_path):
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
        assert status == 400
        assert "WaveMemory" in result["error"]
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


def test_calculation_failure_is_logged_and_next_request_can_succeed(monkeypatch, caplog):
    with running_server() as server:
        def broken(*args, **kwargs):
            raise ImportError("Missing compiler symbol")

        monkeypatch.setattr(server.engine, "ask", broken)
        status, result, _ = request(server, "POST", "/api/ask", {"prompt": "Private test prompt"})
        assert status == 500
        assert "Missing compiler symbol" not in result["error"]
        record = next(record for record in caplog.records if record.getMessage() == "POST /api/ask failed")
        assert record.exc_info[0] is ImportError
        assert "Private test prompt" not in caplog.text
        monkeypatch.setattr(server.engine, "ask", lambda *args, **kwargs: {"answer": "Hallo."})
        status, result, _ = request(server, "POST", "/api/ask", {"prompt": "Hallo"})
        assert status == 200 and result["answer"] == "Hallo."


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
    memory.add_document("Eine Interferenz ist eine Überlagerung von Wellen.", source="Physik")
    with running_server(memory, tmp_path / "real-memory.json") as server:
        status, state, _ = request(server, "GET", "/api/state")
        assert status == 200 and state["document_count"] == 1
        assert state["modal_count"] > 0
        status, result, _ = request(server, "POST", "/api/ask", {"prompt": "Was ist eine Interferenz?"})
        assert status == 200
        assert not result["abstained"]
        assert "Interferenz" in result["answer"]
        status, added, _ = request(server, "POST", "/api/documents", {"text": "Fourierreihen bestehen aus Sinus und Kosinus.", "source": "Mathematik"})
        assert status == 201 and added["document_count"] == 2
        assert (tmp_path / "real-memory.json").is_file()


def test_api_cannot_reactivate_pairs_through_legacy_wave_export(tmp_path):
    from freqai.memory import WaveMemory

    memory = WaveMemory([])
    destination = tmp_path / "dialogue.npz"
    with running_server(memory, destination) as server:
        pair = {"id": "greeting", "prompt": "Hallo", "text": "Guten Tag! Was beschäftigt dich?", "source": "Alltag"}
        status, result, _ = request(server, "POST", "/api/documents", pair)
        assert status == 400 and "Frage-Antwort-Paare" in result["error"]
        assert server.engine.memory.documents == []
        assert not destination.exists()
