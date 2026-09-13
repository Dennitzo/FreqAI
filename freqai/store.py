"""One transactional source of text data; wave coefficients are derived in RAM.

SQLite contains addressed knowledge, explicit relations and separate query
history. Only knowledge documents become retrieval keys. Legacy NPZ wave files
remain an explicit import/export format, never the live database.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator

from .memory import Document, WaveMemory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "memory" / "memory.sqlite3"
DEFAULT_FEATURE_MODE = "morphology"
DEFAULT_RETRIEVAL_POLICY = "coverage"
DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_CONFIGURATION = {
    "dimensions": 4096, "feature_mode": DEFAULT_FEATURE_MODE, "min_score": 0.18,
    "retrieval_policy": DEFAULT_RETRIEVAL_POLICY, "min_coverage": DEFAULT_MIN_COVERAGE,
}


class RevisionConflict(ValueError):
    """A writer's snapshot is stale; no part of its batch was committed."""


class SnapshotRequired(RevisionConflict):
    """An explicit archival migration invalidated the append-only change feed."""


def validate_session_id(session_id: str) -> str:
    """Opaque local conversation address, never a filename or query fragment."""
    if not isinstance(session_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id) is None:
        raise ValueError("session_id muss 1 bis 100 Buchstaben, Ziffern, _ oder - enthalten.")
    return session_id


class MemoryStore:
    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH):
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() == ".npz":
            raise ValueError("NPZ ist ein Austauschformat. Mit 'freqai import DATEI.npz' "
                             "in die zentrale SQLite-Memory importieren.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO metadata(key, value) VALUES ('revision', 0);
                CREATE TABLE IF NOT EXISTS configuration (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    added_revision INTEGER NOT NULL,
                    UNIQUE(text, source)
                );
                CREATE INDEX IF NOT EXISTS document_revision ON documents(added_revision);
                CREATE TABLE IF NOT EXISTS document_archive (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    archived_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    UNIQUE(id, prompt, text, source)
                );
                CREATE TABLE IF NOT EXISTS query_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    prompt TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS relations (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL CHECK(predicate = 'is_a'),
                    object TEXT NOT NULL,
                    source TEXT NOT NULL,
                    UNIQUE(subject, predicate, object, source)
                );
            """)
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_conversation_schema(connection)
                connection.executemany("INSERT OR IGNORE INTO configuration(key,value_json) VALUES (?,?)",
                                       [(key, json.dumps(value)) for key, value in DEFAULT_CONFIGURATION.items()])
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _ensure_conversation_schema(connection: sqlite3.Connection) -> None:
        """Migrate history in place; existing stateless rows stay unassigned."""
        columns = {row[1] for row in connection.execute("PRAGMA table_info(query_history)")}
        if "session_id" not in columns:
            connection.execute("ALTER TABLE query_history ADD COLUMN session_id TEXT")
        if "conversation_revision" not in columns:
            connection.execute("ALTER TABLE query_history ADD COLUMN conversation_revision INTEGER")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS conversation_history_revision
                              ON query_history(session_id, conversation_revision)""")

    @staticmethod
    def _document_columns(connection: sqlite3.Connection) -> str:
        # Read old stores and archives without implicitly mutating an active
        # server's database. Explicit migration removes the historical column.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        return "id,text,source,prompt" if "prompt" in columns else "id,text,source"

    def migrate_information_only(self) -> dict:
        """Atomically archive old pairs/prior and activate an unpaired schema.

        Original rows and transcripts remain recoverable. Nothing converts old
        prompt/response pairs into prose or silently relabels them as facts.
        Run after loading the new application version, not against an old server.
        """
        with self._transaction(write=True) as connection:
            columns = self._document_columns(connection)
            paired_column = columns.endswith(",prompt")
            revision = self._revision(connection)
            if not paired_column and not connection.execute(
                    "SELECT 1 FROM documents WHERE source GLOB 'Authored synthetic language prior / *' LIMIT 1").fetchone():
                count = connection.execute('SELECT count(*) FROM documents').fetchone()[0]
                return {"revision": revision, "archived": 0, "retained": count, "changed": False}
            rows = connection.execute(f"SELECT sequence,{columns},added_revision FROM documents ORDER BY sequence").fetchall()
            selected = []
            retained = []
            for row in rows:
                sequence, identifier, text, source = row[:4]
                prompt = row[4] if paired_column else ""
                added_revision = row[-1]
                if prompt.strip() or source.startswith("Authored synthetic language prior / "):
                    selected.append((identifier, text, source, prompt))
                else:
                    retained.append((sequence, identifier, text, source, added_revision))
            if not paired_column and not selected:
                return {"revision": revision, "archived": 0, "retained": len(retained), "changed": False}
            connection.executemany(
                "INSERT OR IGNORE INTO document_archive(id,text,source,prompt) VALUES (?,?,?,?)", selected)
            connection.execute("ALTER TABLE documents RENAME TO documents_before_information")
            connection.execute("""CREATE TABLE documents (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE, text TEXT NOT NULL, source TEXT NOT NULL,
                added_revision INTEGER NOT NULL, UNIQUE(text,source))""")
            connection.executemany(
                "INSERT INTO documents(sequence,id,text,source,added_revision) VALUES (?,?,?,?,?)", retained)
            connection.execute("DROP TABLE documents_before_information")
            connection.execute("CREATE INDEX document_revision ON documents(added_revision)")
            revision += 1
            connection.execute("UPDATE metadata SET value=? WHERE key='revision'", (revision,))
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('information_reset_revision',?)", (revision,))
        return {"revision": revision, "archived": len(selected), "retained": len(retained), "changed": True}

    @staticmethod
    def _validate_document(document: Document) -> None:
        if (not isinstance(document, Document) or not isinstance(document.id, str)
                or not document.id.strip() or not isinstance(document.text, str)
                or not isinstance(document.source, str) or not isinstance(document.prompt, str)):
            raise ValueError("Each document needs a nonempty string ID and string text, source and prompt")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _revision(connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

    def revision(self) -> int:
        with self._connection() as connection:
            return self._revision(connection)

    def snapshot(self) -> tuple[int, list[Document]]:
        with self._transaction() as connection:
            revision = self._revision(connection)
            rows = connection.execute(f"SELECT {self._document_columns(connection)} FROM documents ORDER BY sequence").fetchall()
            return revision, [Document(*row) for row in rows]

    def changes_since(self, revision: int) -> tuple[int, list[Document]]:
        """Read committed additions and their revision from one SQLite snapshot."""
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        with self._transaction() as connection:
            current = self._revision(connection)
            reset = connection.execute("SELECT value FROM metadata WHERE key='information_reset_revision'").fetchone()
            if reset and revision < reset[0]:
                raise SnapshotRequired("Der aktive Informationsbestand wurde umgestellt; vollständiger Snapshot erforderlich.")
            if revision > current:
                raise RevisionConflict("Requested revision is newer than this store")
            rows = connection.execute(
                f"SELECT {self._document_columns(connection)} FROM documents WHERE added_revision > ? ORDER BY sequence",
                (revision,),
            ).fetchall()
            return current, [Document(*row) for row in rows]

    def append_documents(self, documents: list[Document],
                         expected_revision: int | None = None) -> dict:
        """Append information atomically, deduplicating exact text and source.

        An existing ID with different content is always an error. Identical
        content under another ID returns the original addressed document.
        """
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of Document records")
        if expected_revision is not None and (
                isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
                or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer")
        supplied_ids: dict[str, Document] = {}
        for document in documents:
            self._validate_document(document)
            if document.prompt.strip():
                raise ValueError("Nur Informationstexte ohne Frage-Antwort-Paare sind zulässig; Altpaare können archiviert werden.")
            if document.source.startswith("Authored synthetic language prior / "):
                raise ValueError("Der frühere Antwortprior ist archiviert und kein Informationsbestand.")
            if document.id in supplied_ids and supplied_ids[document.id] != document:
                raise ValueError(f"Document ID conflict within batch: {document.id}")
            supplied_ids[document.id] = document
        added: list[Document] = []
        existing: list[Document] = []
        with self._transaction(write=True) as connection:
            revision = self._revision(connection)
            if expected_revision is not None and expected_revision != revision:
                raise RevisionConflict(f"Memory changed: expected revision {expected_revision}, got {revision}")
            for document in documents:
                row = connection.execute(f"SELECT {self._document_columns(connection)} FROM documents WHERE id=?",
                                         (document.id,)).fetchone()
                if row is not None:
                    stored = Document(*row)
                    if stored != document:
                        raise ValueError(f"Document ID conflict: {document.id}")
                    existing.append(stored)
                    continue
                row = connection.execute(
                    f"SELECT {self._document_columns(connection)} FROM documents WHERE text=? AND source=?",
                    (document.text, document.source)).fetchone()
                if row is not None:
                    existing.append(Document(*row))
                    continue
                connection.execute(
                    "INSERT INTO documents(id,text,source,added_revision) VALUES (?,?,?,?)",
                    (document.id, document.text, document.source, revision + 1),
                )
                added.append(document)
            if added:
                revision += 1
                connection.execute("UPDATE metadata SET value=? WHERE key='revision'", (revision,))
        return {"revision": revision, "added": added, "existing": existing}

    def archive_documents(self, documents: list[Document]) -> dict:
        """Preserve legacy texts centrally without activating them as answers.

        Historical files can reuse IDs for different contents. The archive
        preserves all such variants, deduplicating only completely equal rows.
        """
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of Document records")
        for document in documents:
            self._validate_document(document)
        with self._transaction(write=True) as connection:
            added = 0
            for document in documents:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO document_archive(id,text,source,prompt) VALUES (?,?,?,?)",
                    (document.id, document.text, document.source, document.prompt))
                added += cursor.rowcount
        return {"added": added, "existing": len(documents) - added}

    def archived_documents(self) -> list[Document]:
        with self._connection() as connection:
            return [Document(*row) for row in connection.execute(
                "SELECT id,text,source,prompt FROM document_archive ORDER BY sequence")]

    def load_memory(self) -> WaveMemory:
        from .progress import phase, report
        with phase('Memory-Größe und Speicherbedarf prüfen'):
            with self._connection() as connection:
                count, text_bytes = connection.execute(
                    'SELECT count(*),coalesce(sum(length(cast(text AS BLOB))),0) FROM documents').fetchone()
            configuration = self.configuration()
            lower_bound = text_bytes*32 + count*configuration['dimensions']*32
            report(f'{count:,} Dokumente; monolithische Arrays mindestens {lower_bound/1024**3:.1f} GiB')
        # A conservative fixed ceiling also protects small machines. Corpus-size
        # growth must never silently choose the unbounded in-memory constructor.
        from .memory import MAX_DENSE_BYTES
        if lower_bound > MAX_DENSE_BYTES or text_bytes > 2*1024**2:
            from .paged_memory import PagedWaveMemory
            return PagedWaveMemory(self)
        with phase('Kleine Memory: Schwingungen aufbauen'):
            memory = WaveMemory(self.snapshot()[1], **configuration)
        memory._compiler_cache_dir = self.path.parent / '.compiler-cache' / self.path.name
        return memory

    def configuration(self) -> dict:
        with self._connection() as connection:
            return {key: json.loads(value) for key, value in connection.execute(
                "SELECT key,value_json FROM configuration ORDER BY key")}

    def record_query(self, prompt: str, result: dict) -> None:
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(result, dict):
            raise ValueError("Query history needs a nonempty prompt and result dictionary")
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._transaction(write=True) as connection:
            connection.execute("INSERT INTO query_history(prompt,result_json) VALUES (?,?)", (prompt, encoded))

    def load_conversation(self, session_id: str) -> tuple[int, dict]:
        validate_session_id(session_id)
        with self._connection() as connection:
            row = connection.execute("SELECT revision,state_json FROM conversations WHERE session_id=?",
                                     (session_id,)).fetchone()
            return (int(row[0]), json.loads(row[1])) if row else (0, {})

    def commit_conversation(self, session_id: str, expected_revision: int, prompt: str,
                            result: dict, state: dict) -> int:
        """Commit exactly one turn and its derived context or reject both.

        CAS protects independent server/CLI processes as well as HTTP threads.
        Caller retries generation from the latest state on RevisionConflict.
        Conversation text is deliberately absent from documents and wave keys.
        """
        validate_session_id(session_id)
        if (isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
                or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer")
        if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(result, dict)
                or not isinstance(state, dict)):
            raise ValueError("Conversation needs a nonempty prompt, result and state dictionaries")
        encoded_result = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        encoded_state = json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._transaction(write=True) as connection:
            row = connection.execute("SELECT revision FROM conversations WHERE session_id=?",
                                     (session_id,)).fetchone()
            revision = int(row[0]) if row else 0
            if revision != expected_revision:
                raise RevisionConflict(f"Conversation changed: expected revision {expected_revision}, got {revision}")
            revision += 1
            connection.execute("""
                INSERT INTO conversations(session_id,revision,state_json) VALUES (?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET revision=excluded.revision,
                    state_json=excluded.state_json, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """, (session_id, revision, encoded_state))
            connection.execute("""INSERT INTO query_history(prompt,result_json,session_id,conversation_revision)
                                  VALUES (?,?,?,?)""", (prompt, encoded_result, session_id, revision))
        return revision

    def conversation_history(self, session_id: str, limit: int = 100) -> dict:
        """Return a bounded view of only the requested conversation.

        Large retrieval diagnostics are retained in SQLite but excluded here.
        The response includes at most 100 turns and 1 MiB of message JSON.
        """
        validate_session_id(session_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        with self._transaction() as connection:
            row = connection.execute("SELECT revision FROM conversations WHERE session_id=?",
                                     (session_id,)).fetchone()
            revision = int(row[0]) if row else 0
            rows = connection.execute("""
                SELECT prompt,result_json,conversation_revision,created_at FROM query_history
                WHERE session_id=? ORDER BY conversation_revision DESC LIMIT ?
            """, (session_id, limit)).fetchall()
        turns, encoded_bytes = [], 0
        for prompt, encoded_result, turn_revision, created_at in rows:
            result = json.loads(encoded_result)
            turn = {"prompt": prompt, "answer": result.get("answer", ""),
                    "abstained": result.get("abstained", False), "generated": result.get("generated", False),
                    "method": result.get("method", "retrieval"), "revision": turn_revision,
                    "created_at": created_at}
            size = len(json.dumps(turn, ensure_ascii=False).encode("utf-8"))
            if encoded_bytes + size > 1_000_000:
                break
            turns.append(turn)
            encoded_bytes += size
        turns.reverse()
        return {"session_id": session_id, "conversation_revision": revision, "turns": turns,
                "truncated": len(turns) < revision}

    def add_relations(self, triples: list[dict]) -> dict:
        if not isinstance(triples, list):
            raise TypeError("triples must be a list")
        prepared = []
        for index, triple in enumerate(triples):
            if not isinstance(triple, dict) or triple.get("predicate") != "is_a":
                raise ValueError("Only explicit is_a relation dictionaries are supported")
            subject, target = triple.get("subject"), triple.get("object")
            source = triple.get("source", f"input:triple:{index}")
            if any(not isinstance(value, str) or not value.strip() for value in (subject, target, source)):
                raise ValueError("Relations need nonempty subject, object and source strings")
            prepared.append((subject, "is_a", target, source))
        with self._transaction(write=True) as connection:
            revision = self._revision(connection)
            added = 0
            for triple in prepared:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO relations(subject,predicate,object,source) VALUES (?,?,?,?)", triple)
                added += cursor.rowcount
            if added:
                from .reasoning import WaveReasoner
                count = connection.execute(
                    "SELECT COUNT(*) FROM (SELECT subject AS node FROM relations UNION SELECT object FROM relations)"
                ).fetchone()[0]
                if count > WaveReasoner.MAX_NODES:
                    raise ValueError(f"dense Fourier reasoner is limited to {WaveReasoner.MAX_NODES} nodes")
                revision += 1
                connection.execute("UPDATE metadata SET value=? WHERE key='revision'", (revision,))
        return {"revision": revision, "added": added, "existing": len(prepared) - added}

    def relations(self) -> list[dict]:
        with self._connection() as connection:
            return [dict(zip(("subject", "predicate", "object", "source"), row)) for row in connection.execute(
                "SELECT subject,predicate,object,source FROM relations ORDER BY sequence")]

    def stats(self) -> dict:
        with self._transaction() as connection:
            return {
                "path": str(self.path),
                "revision": self._revision(connection),
                "document_count": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "archive_document_count": connection.execute("SELECT COUNT(*) FROM document_archive").fetchone()[0],
                "history_count": connection.execute("SELECT COUNT(*) FROM query_history").fetchone()[0],
                "relation_count": connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
                "canonical_format": "sqlite_text",
                "wave_storage": "derived_in_memory",
                "knowledge_format": "information_texts",
                "legacy_schema": self._document_columns(connection).endswith(",prompt"),
                "configuration": {key: json.loads(value) for key, value in connection.execute(
                    "SELECT key,value_json FROM configuration ORDER BY key")},
            }
