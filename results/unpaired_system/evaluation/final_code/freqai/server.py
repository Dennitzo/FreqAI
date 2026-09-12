"""Local HTTP dashboard and continuously evaluated wave-memory dynamics.

The server deliberately uses only the Python standard library.  ``make_server``
returns a ``WaveHTTPServer``; close it with ``server_close`` after shutting down
``serve_forever`` to stop the simulation worker as well.
"""

from __future__ import annotations

import copy
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
import webbrowser
from uuid import uuid4

from .memory import Document, WaveMemory
from .store import RevisionConflict, SnapshotRequired, validate_session_id


MAX_BODY_BYTES = 1_048_576
MAX_TEXT_CHARS = 250_000


class WaveEngine:
    """Evaluate the complete memory on a wall-clock timeline, without integration."""

    def __init__(self, memory: Any, store_path: str | Path | None = None,
                 frequency_hz: float = 10.0, points: int = 256,
                 central_store: Any | None = None) -> None:
        if not math.isfinite(frequency_hz) or not 0.0 < frequency_hz <= 100.0:
            raise ValueError("frequency_hz must be finite and in (0, 100]")
        self.memory = memory
        self.store_path = Path(store_path) if store_path is not None else None
        self.central_store = central_store
        self.frequency_hz = float(frequency_hz)
        self.points = points
        self.lock = threading.RLock()
        # Serialize preparation/commit/swap within this engine, without holding
        # the frame lock while Fourier transforms or database writes run.
        self._mutation_lock = threading.RLock()
        # A fixed lock pool bounds memory use for arbitrarily many session IDs.
        # SQLite revision checks additionally protect separate server processes.
        self._conversation_locks = [threading.RLock() for _ in range(64)]
        self._revision = 0
        self._origin = time.perf_counter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._state: dict[str, Any] = {}
        self._ticks = 0
        self._last_error: str | None = None
        self._sync_error: str | None = None
        if central_store is not None:
            self._revision, documents = central_store.snapshot()
            if memory.documents != documents:
                self.memory = type(memory)(documents, dimensions=memory.dimensions,
                                           feature_mode=memory.feature_mode,
                                           min_score=memory.min_score,
                                           retrieval_policy=memory.retrieval_policy,
                                           min_coverage=memory.min_coverage)
        self._refresh()

    def elapsed(self) -> float:
        return time.perf_counter() - self._origin

    def _refresh(self) -> None:
        if self.central_store is None:
            # Small legacy embedders may mutate their memory object in place.
            with self.lock:
                self._state = self.memory.snapshot(time_s=self.elapsed(), points=self.points)
                self._ticks += 1
                self._last_error = None
            return
        with self.lock:
            memory = self.memory
        # Each prepared memory is immutable after publication. A long new-text
        # FFT leaves existing modes and the monotonic clock free to advance.
        state = memory.snapshot(time_s=self.elapsed(), points=self.points)
        with self.lock:
            if memory is not self.memory or state["time_s"] < self._state.get("time_s", -1):
                return
            self._state = state
            self._ticks += 1
            self._last_error = None

    def sync(self, blocking: bool = True) -> None:
        """Apply committed external imports; callers never observe half a batch."""
        if self.central_store is None:
            return
        if not self._mutation_lock.acquire(blocking=blocking):
            return
        try:
            if self.central_store.revision() == self._revision:
                with self.lock:
                    self._sync_error = None
                return
            try:
                revision, additions = self.central_store.changes_since(self._revision)
                candidate = self.memory.with_documents_added(additions)
            except SnapshotRequired:
                revision, documents = self.central_store.snapshot()
                candidate = WaveMemory(documents, **self.central_store.configuration())
            state = candidate.snapshot(time_s=self.elapsed(), points=self.points)
            self._publish(candidate, revision, state)
            with self.lock:
                self._sync_error = None
        finally:
            self._mutation_lock.release()

    def _publish(self, candidate: Any, revision: int, state: dict[str, Any]) -> None:
        with self.lock:
            # A frame of the previous memory may have completed while candidate
            # sampling ran. Never publish an older displayed simulation time.
            if state["time_s"] < self._state.get("time_s", -1):
                state = candidate.snapshot(time_s=self.elapsed(), points=self.points)
            self.memory = candidate
            self._revision = revision
            self._state = state
            self._ticks += 1
            self._last_error = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="freqai-wave-engine", daemon=True)
        self._thread.start()
        if self.central_store is not None:
            self._sync_thread = threading.Thread(target=self._run_sync, name="freqai-memory-sync", daemon=True)
            self._sync_thread.start()

    def _run(self) -> None:
        period = 1.0 / self.frequency_hz
        deadline = time.perf_counter() + period
        while not self._stop.wait(max(0.0, deadline - time.perf_counter())):
            try:
                self._refresh()
            except Exception as exc:  # Keep the API available to report worker errors.
                with self.lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            deadline = max(deadline + period, time.perf_counter())

    def _run_sync(self) -> None:
        # A CLI import can require substantial Fourier preparation. Keep it off
        # the frame worker so old modes continue throughout external imports.
        period = 1.0 / self.frequency_hz
        while not self._stop.wait(period):
            try:
                self.sync(blocking=False)
            except Exception as exc:
                with self.lock:
                    self._sync_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
        if self._sync_thread is not None and self._sync_thread is not threading.current_thread():
            self._sync_thread.join(timeout=5.0)

    def state(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None:
            validate_session_id(session_id)
            if self.central_store is None:
                raise ValueError("Gesprächskontext benötigt die zentrale SQLite-Memory.")
        with self.lock:
            memory = self.memory
            state = copy.deepcopy(self._state)
            state.update({
                "ticks": self._ticks,
                "target_hz": self.frequency_hz,
                "running": (self._thread is not None and self._thread.is_alive()
                            and (self.central_store is None or self._sync_thread is not None and self._sync_thread.is_alive())),
                "error": self._last_error or self._sync_error,
                "sync_error": self._sync_error,
                "revision": self._revision,
                "live_revision": self._revision,
                "persistent": self.central_store is not None or self.store_path is not None,
            })
        if session_id is not None:
            conversation_revision, context = self.central_store.load_conversation(session_id)
            state["session_id"] = session_id
            state["conversation_revision"] = conversation_revision
            generation = context.get("unpaired")
            information = context.get("information")
            if ((isinstance(generation, dict) and generation.get("field")) or
                    (isinstance(information, dict) and information.get("field"))):
                from .unpaired_runtime import context_snapshot
                state["context_wave"] = context_snapshot(memory, context, time_s=state["time_s"])
        return state

    def ask(self, prompt: str, top_k: int = 1, session_id: str | None = None,
            time_s: float | None = None, mode: str = "wave",
            max_tokens: int = 40, seed: int = 17) -> dict[str, Any]:
        if not isinstance(mode, str) or mode != "wave":
            raise ValueError("Nur der einheitliche Informations-Wellenmodus 'wave' ist verfügbar.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 128:
            raise ValueError("max_tokens muss eine Ganzzahl von 1 bis 128 sein.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4_294_967_295:
            raise ValueError("seed muss eine Ganzzahl von 0 bis 4294967295 sein.")
        if not isinstance(self.memory, WaveMemory):
            raise ValueError("Wortweise Wellengenerierung benötigt einen WaveMemory-Speicher.")
        if session_id is not None:
            validate_session_id(session_id)
            if self.central_store is None:
                raise ValueError("Gesprächskontext benötigt die zentrale SQLite-Memory.")
        self.sync()
        with self.lock:
            memory, revision = self.memory, self._revision

        def calculate(context: dict | None = None) -> tuple[dict, dict]:
            clock = self.elapsed() if time_s is None else time_s
            from .unpaired_runtime import respond_wave
            result, state = respond_wave(memory, prompt, context=context, time_s=clock,
                                         top_k=top_k, max_tokens=max_tokens, seed=seed)
            result.update({"revision": revision, "live_revision": revision, "mode": mode})
            return result, state

        if session_id is None:
            result, _ = calculate()
            if self.central_store is not None:
                self.central_store.record_query(prompt, result)
            return result
        with self._conversation_locks[hash(session_id) % len(self._conversation_locks)]:
            for _ in range(32):
                previous_revision, context = self.central_store.load_conversation(session_id)
                result, state = calculate(context)
                result.update({"session_id": session_id, "conversation_revision": previous_revision + 1})
                try:
                    self.central_store.commit_conversation(session_id, previous_revision, prompt, result, state)
                except RevisionConflict:
                    continue
                return result
        raise RevisionConflict("Das Gespräch wurde gleichzeitig geändert. Bitte die Nachricht erneut senden.")

    def conversation(self, session_id: str) -> dict[str, Any]:
        validate_session_id(session_id)
        if self.central_store is None:
            raise ValueError("Gesprächskontext benötigt die zentrale SQLite-Memory.")
        return self.central_store.conversation_history(session_id)

    def memory_info(self) -> dict[str, Any]:
        self.sync()
        if self.central_store is not None:
            result = dict(self.central_store.stats())
        else:
            result = {"path": str(self.store_path) if self.store_path else None,
                      "revision": self._revision}
        with self.lock:
            result.update({"document_count": len(self.memory.documents),
                           "live_revision": self._revision,
                           "persistent": self.central_store is not None or self.store_path is not None})
        return result

    def documents(self, offset: int = 0, limit: int | None = None,
                  expected_revision: int | None = None) -> dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise ValueError("offset muss eine nichtnegative ganze Zahl sein.")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 200):
            raise ValueError("limit muss zwischen 1 und 200 liegen.")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError("revision muss eine nichtnegative ganze Zahl sein.")
        self.sync()
        with self.lock:
            if expected_revision is not None and expected_revision != self._revision:
                raise RevisionConflict("Der Speicher wurde ergänzt. Bitte die erste Seite neu laden.")
            total = len(self.memory.documents)
            selected = self.memory.documents if limit is None else self.memory.documents[offset:offset + limit]
            documents = [{"id": doc.id, "text": doc.text, "source": doc.source} if isinstance(doc, Document) else dict(doc)
                         for doc in selected]
            result = {"documents": documents, "document_count": total,
                      "revision": self._revision, "live_revision": self._revision}
            if limit is not None:
                result.update({"total": total, "offset": offset, "limit": limit,
                               "next_offset": offset + limit if offset + limit < total else None,
                               "previous_offset": max(0, min(offset, total) - limit) if offset > 0 and total else None})
            return result

    def add_document(self, text: str, source: str) -> dict[str, Any]:
        return self.add_documents([Document(f"user-{uuid4().hex}", text, source)])

    def add_documents(self, documents: list[Document]) -> dict[str, Any]:
        if not documents:
            raise ValueError("At least one document is required")
        if any(doc.prompt.strip() for doc in documents):
            raise ValueError("Wissenseinträge sind reine Informationstexte; Frage-Antwort-Paare sind nicht zulässig.")
        if self.central_store is not None:
            return self._add_to_store(documents)
        # Compatibility path for explicit legacy NPZ exports and small embedders.
        # Normal CLI/dashboard startup always uses the central SQLite store.
        with self.lock:
            # Stage ingestion and persistence on an independent object. A disk
            # failure must not leave an acknowledged-as-failed text in live RAM.
            if hasattr(self.memory, "with_documents_added"):
                candidate = self.memory.with_documents_added(documents)
            else:
                candidate = copy.deepcopy(self.memory)
                for document in documents:
                    candidate.add_document(document.text, source=document.source, document_id=document.id)
            next_state = candidate.snapshot(time_s=self.elapsed(), points=self.points)
            if self.store_path is not None:
                candidate.save(self.store_path)
            if hasattr(self.memory, "with_documents_added"):
                self.memory = candidate
            else:
                self.memory.__dict__.clear()
                self.memory.__dict__.update(candidate.__dict__)
            self._state = next_state
            self._ticks += 1
            self._last_error = None
            self._revision += 1
            return {"id": documents[0].id, "document_count": len(self.memory.documents),
                    "revision": self._revision, "live_revision": self._revision,
                    "added": [{"id": doc.id, "text": doc.text, "source": doc.source} for doc in documents], "existing": []}

    def _add_to_store(self, documents: list[Document]) -> dict[str, Any]:
        with self._mutation_lock:
            for _ in range(32):
                self.sync()
                revision = self._revision
                # Prepare before opening the write transaction. Concurrent
                # processes are detected by the revision precondition below.
                old_ids = {doc.id: doc for doc in self.memory.documents}
                old_content = {(doc.text, doc.source): doc for doc in self.memory.documents}
                additions = []
                for doc in documents:
                    if doc.id in old_ids and old_ids[doc.id] != doc:
                        raise ValueError(f"Document ID already exists with different content: {doc.id}")
                    if doc.id not in old_ids and (doc.text, doc.source) not in old_content:
                        additions.append(doc)
                        old_ids[doc.id] = doc
                        old_content[(doc.text, doc.source)] = doc
                candidate = self.memory.with_documents_added(additions)
                next_state = candidate.snapshot(time_s=self.elapsed(), points=self.points)
                try:
                    result = self.central_store.append_documents(documents, expected_revision=revision)
                except ValueError:
                    if self.central_store.revision() != revision:
                        continue
                    raise
                if result["added"] != additions:
                    # The store is authoritative if its idempotence rules differ.
                    # This branch is normally unnecessary, but keeps publication
                    # faithful to the committed row set for custom stores.
                    candidate = self.memory.with_documents_added(result["added"])
                # Commit may have waited for another process. Resample on the
                # unchanged clock rather than publishing its pre-commit frame.
                next_state = candidate.snapshot(time_s=self.elapsed(), points=self.points)
                self._publish(candidate, result["revision"], next_state)
                added = [{"id": doc.id, "text": doc.text, "source": doc.source} for doc in result["added"]]
                existing = [{"id": doc.id, "text": doc.text, "source": doc.source} for doc in result["existing"]]
                return {"id": (added or existing)[0]["id"], "added": added, "existing": existing,
                        "document_count": len(candidate.documents), "revision": result["revision"],
                        "live_revision": result["revision"]}
        raise RuntimeError("Central memory remained busy during 32 attempts; retry the import")


class WaveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], memory: Any,
                 store_path: str | Path | None = None, central_store: Any | None = None) -> None:
        self.dashboard = Path(__file__).with_name("dashboard.html").read_bytes()
        self.engine = WaveEngine(memory, store_path=store_path, central_store=central_store)
        super().__init__(address, WaveRequestHandler)
        self.engine.start()

    def server_close(self) -> None:
        self.engine.close()
        super().server_close()


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        self.status = status
        self.message = message


class WaveRequestHandler(BaseHTTPRequestHandler):
    server: WaveHTTPServer
    server_version = "FreqAI/1.0"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10.0)

    def log_message(self, format: str, *args: Any) -> None:
        # Polling runs ten times per second; avoid flooding the console or logging
        # potentially private prompts. Runtime failures appear in /api/health.
        return

    def _respond(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        try:
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
        except ConnectionError:
            pass

    def _json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._respond(status, payload, "application/json; charset=utf-8")

    def send_error(self, code: int, message: str | None = None,
                   explain: str | None = None) -> None:
        status = HTTPStatus(code)
        self._json(status, {"error": message or status.phrase})

    def _check_origin(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != self.headers.get("Host"):
                raise RequestError(HTTPStatus.FORBIDDEN, "Anfragen anderer Webseiten sind nicht erlaubt.")

    def _body(self) -> dict[str, Any]:
        self._check_origin()
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(HTTPStatus.BAD_REQUEST, "Transfer-Encoding wird nicht unterstützt.")
        if self.headers.get_content_type() != "application/json":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type muss application/json sein.")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length fehlt.")
        if len(lengths) != 1:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length ist mehrdeutig.")
        try:
            size = int(lengths[0])
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length ist ungültig.") from exc
        if size < 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length ist ungültig.")
        if size > MAX_BODY_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Anfrage ist größer als 1 MiB.")
        try:
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Unvollständiger Anfrageinhalt.")
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ungültiges UTF-8-JSON.") from exc
        if not isinstance(body, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "Ein JSON-Objekt wird erwartet.")
        return body

    @staticmethod
    def _text(body: dict[str, Any], key: str, max_length: int) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{key} muss nichtleerer Text sein.")
        if len(value) > max_length:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{key} ist zu lang (maximal {max_length} Zeichen).")
        return value

    def do_GET(self) -> None:
        try:
            self._get()
        except RequestError as exc:
            self._json(exc.status, {"error": exc.message})
        except RevisionConflict as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Der zentrale Speicher konnte nicht gelesen werden."})

    def _get(self) -> None:
        route = urlsplit(self.path).path
        if route == "/":
            self._respond(HTTPStatus.OK, self.server.dashboard, "text/html; charset=utf-8")
        elif route == "/api/state":
            query = urlsplit(self.path).query
            if query:
                self._check_origin()
                if len(query) > 1024:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Ungültige Gesprächsadresse.")
                parameters = parse_qs(query, keep_blank_values=True, max_num_fields=8)
                if set(parameters) != {"session_id"} or len(parameters["session_id"]) != 1:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Genau eine session_id wird benötigt.")
                state = self.server.engine.state(session_id=parameters["session_id"][0])
            else:
                state = self.server.engine.state()
            self._json(HTTPStatus.OK, state)
        elif route == "/api/memory":
            self._json(HTTPStatus.OK, self.server.engine.memory_info())
        elif route == "/api/documents":
            query = urlsplit(self.path).query
            if not query:
                result = self.server.engine.documents()
            else:
                self._check_origin()
                if len(query) > 512:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Ungültige Seitenadresse.")
                parameters = parse_qs(query, keep_blank_values=True, max_num_fields=3)
                if not parameters or set(parameters) - {"offset", "limit", "revision"} or any(
                    len(values) != 1 or not values[0].isascii() or not values[0].isdigit()
                    for values in parameters.values()
                ):
                    raise RequestError(HTTPStatus.BAD_REQUEST, "offset, limit und revision müssen jeweils einzelne ganze Zahlen sein.")
                result = self.server.engine.documents(
                    offset=int(parameters.get("offset", ["0"])[0]),
                    limit=int(parameters.get("limit", ["50"])[0]),
                    expected_revision=int(parameters["revision"][0]) if "revision" in parameters else None)
            self._json(HTTPStatus.OK, result)
        elif route == "/api/conversation":
            self._check_origin()
            query = urlsplit(self.path).query
            if len(query) > 1024:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Ungültige Gesprächsadresse.")
            parameters = parse_qs(query, keep_blank_values=True, max_num_fields=8)
            if set(parameters) != {"session_id"} or len(parameters["session_id"]) != 1:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Genau eine session_id wird benötigt.")
            self._json(HTTPStatus.OK, self.server.engine.conversation(parameters["session_id"][0]))
        elif route == "/api/health":
            state = self.server.engine.state()
            healthy = state["running"] and state["error"] is None
            self._json(HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE, {
                "ok": healthy, "running": state["running"], "ticks": state["ticks"],
                "error": state["error"], "document_count": state.get("document_count", 0),
                "revision": state["revision"], "live_revision": state["live_revision"],
            })
        elif route == "/favicon.ico":
            self._respond(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Ressource nicht gefunden."})

    def do_POST(self) -> None:
        route = urlsplit(self.path).path
        if route not in {"/api/ask", "/api/documents"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Ressource nicht gefunden."})
            return
        try:
            body = self._body()
            if route == "/api/ask":
                prompt = self._text(body, "prompt", 20_000)
                top_k = body.get("top_k", 1)
                if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "top_k muss eine Ganzzahl von 1 bis 10 sein.")
                session_id = body.get("session_id")
                if "session_id" in body:
                    validate_session_id(session_id)
                result = self.server.engine.ask(prompt, top_k, session_id=session_id,
                                               mode=body.get("mode", "wave"),
                                               max_tokens=body.get("max_tokens", 40),
                                               seed=body.get("seed", 17))
                self._json(HTTPStatus.OK, result)
            else:
                rows = body.get("documents", [body])
                if not isinstance(rows, list) or not rows or len(rows) > 4096:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "documents muss eine Liste mit 1 bis 4096 Texten sein.")
                documents = []
                for row in rows:
                    if not isinstance(row, dict):
                        raise RequestError(HTTPStatus.BAD_REQUEST, "Jeder Eintrag muss ein JSON-Objekt sein.")
                    text = self._text(row, "text", MAX_TEXT_CHARS)
                    source = row.get("source", "user")
                    if not isinstance(source, str) or not source.strip() or len(source) > 500:
                        raise RequestError(HTTPStatus.BAD_REQUEST, "source muss Text mit 1 bis 500 Zeichen sein.")
                    document_id = row.get("id", f"user-{uuid4().hex}")
                    if not isinstance(document_id, str) or not document_id.strip() or len(document_id) > 200:
                        raise RequestError(HTTPStatus.BAD_REQUEST, "id muss Text mit 1 bis 200 Zeichen sein.")
                    cue = row.get("prompt", "")
                    if not isinstance(cue, str) or cue.strip():
                        raise RequestError(HTTPStatus.BAD_REQUEST, "Informationstexte haben keinen Gesprächsanlass; Frage-Antwort-Paare sind nicht zulässig.")
                    documents.append(Document(document_id, text, source))
                result = self.server.engine.add_documents(documents)
                self._json(HTTPStatus.CREATED, result)
        except RequestError as exc:
            self._json(exc.status, {"error": exc.message})
        except (TimeoutError, ConnectionError):
            self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "Zeitüberschreitung beim Lesen der Anfrage."})
        except RevisionConflict as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Die Berechnung oder Speicherung ist fehlgeschlagen."})


def make_server(memory: Any, host: str = "127.0.0.1", port: int = 8765,
                store_path: str | Path | None = None, central_store: Any | None = None) -> WaveHTTPServer:
    """Create and bind the server, starting its continuous calculation worker."""
    return WaveHTTPServer((host, port), memory, store_path=store_path, central_store=central_store)


def serve(memory: Any, host: str = "127.0.0.1", port: int = 8765,
          store_path: str | Path | None = None, open_browser: bool = False,
          central_store: Any | None = None) -> None:
    """Serve the local dashboard until interrupted, then stop all workers."""
    server = make_server(memory, host=host, port=port, store_path=store_path, central_store=central_store)
    visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{visible_host}:{server.server_address[1]}/"
    print(f"FreqAI: {url}  (Strg+C beendet den Server)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
