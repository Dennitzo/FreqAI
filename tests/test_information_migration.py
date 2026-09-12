"""Migration protects originals while removing paired knowledge from activity."""
import sqlite3

import pytest

from freqai.memory import Document
from freqai.store import MemoryStore, SnapshotRequired


def legacy_store(path):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value INTEGER NOT NULL);
            INSERT INTO metadata VALUES('revision',4);
            CREATE TABLE documents(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,text TEXT NOT NULL,source TEXT NOT NULL,
                prompt TEXT NOT NULL DEFAULT '',added_revision INTEGER NOT NULL,
                UNIQUE(prompt,text,source));
            INSERT INTO documents VALUES(2,'pair','Alte Antwort.','Dialog','Alte Frage?',1);
            INSERT INTO documents VALUES(5,'prior','Alter Antwortprior.','Authored synthetic language prior / greeting','',2);
            INSERT INTO documents VALUES(8,'prose','Eine Frequenz beschreibt Wiederholungen.','Wikipedia','',4);
        """)
    store = MemoryStore(path)
    store.archive_documents([Document("old-archive", "Früherer Text.", "Archiv", "Frühere Frage")])
    store.commit_conversation("retained", 0, "Hallo", {"answer": "Alter Verlauf"}, {"turn_count": 1})
    return store


def test_migration_archives_pairs_and_prior_without_converting_answers(tmp_path):
    store = legacy_store(tmp_path / "legacy.sqlite3")
    original = store.snapshot()[1]
    context = store.load_conversation("retained")
    history = store.conversation_history("retained")
    result = store.migrate_information_only()
    assert result == {"revision": 5, "archived": 2, "retained": 1, "changed": True}
    assert store.snapshot() == (5, [original[-1]])
    assert store.archived_documents()[1:] == original[:2]
    assert store.load_conversation("retained") == context
    assert store.conversation_history("retained") == history
    with sqlite3.connect(store.path) as connection:
        assert "prompt" not in {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        assert connection.execute("SELECT sequence FROM documents").fetchone()[0] == 8
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert store.migrate_information_only() == {"revision": 5, "archived": 0, "retained": 1, "changed": False}
    with pytest.raises(SnapshotRequired):
        store.changes_since(4)
    assert store.changes_since(5) == (5, [])
    with pytest.raises(ValueError, match="Frage-Antwort-Paare"):
        store.append_documents([original[0]])
    with pytest.raises(ValueError, match="Antwortprior"):
        store.append_documents([original[1]])


def test_archive_failure_rolls_back_every_schema_and_data_change(tmp_path):
    store = legacy_store(tmp_path / "failure.sqlite3")
    original = store.snapshot()
    archive = store.archived_documents()
    with sqlite3.connect(store.path) as connection:
        connection.execute("CREATE TRIGGER reject_archive BEFORE INSERT ON document_archive BEGIN SELECT RAISE(ABORT,'audit failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="audit failure"):
        store.migrate_information_only()
    assert store.snapshot() == original
    assert store.archived_documents() == archive
    with sqlite3.connect(store.path) as connection:
        assert "prompt" in {row[1] for row in connection.execute("PRAGMA table_info(documents)")}


def test_running_engine_replaces_archived_modes_but_continues_clock(tmp_path):
    from freqai.server import WaveEngine
    store = legacy_store(tmp_path / "live.sqlite3")
    engine = WaveEngine(store.load_memory(), central_store=store)
    try:
        origin = engine._origin
        store.migrate_information_only()
        engine.sync()
        assert [doc.id for doc in engine.memory.documents] == ["prose"]
        assert engine._origin == origin
        assert engine.state()["revision"] == 5
        assert engine.state()["error"] is None
    finally:
        engine.close()
