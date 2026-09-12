"""Transactional persistence, concurrent import and canonical-source boundaries."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from freqai.memory import Document, WaveMemory
from freqai.store import DEFAULT_CONFIGURATION, DEFAULT_MEMORY_PATH, MemoryStore, RevisionConflict


def test_batch_append_deduplication_and_restart(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = MemoryStore(path)
    first = Document("alpha", "Text mit Umlauten: ÄÖÜ und 日本語.", "Handbuch")
    second = Document("beta", "Ein zweiter Text.", "Handbuch")
    assert store.snapshot() == (0, [])
    result = store.append_documents([first, second, first])
    assert result == {"revision": 1, "added": [first, second], "existing": [first]}
    duplicate = store.append_documents([Document("new-id", first.text, first.source)])
    assert duplicate == {"revision": 1, "added": [], "existing": [first]}
    assert MemoryStore(path).snapshot() == (1, [first, second])
    assert store.append_documents([])["revision"] == 1
    assert store.configuration() == DEFAULT_CONFIGURATION
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT text FROM documents WHERE id='alpha'").fetchone()[0] == first.text


def test_conflicting_batch_rolls_back_earlier_inserts_without_data_loss(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    original = Document("existing", "Bleibt unverändert.", "Original")
    store.append_documents([original])
    with pytest.raises(ValueError, match="ID conflict"):
        store.append_documents([Document("new", "Darf nicht teilweise gespeichert sein."),
                                Document("existing", "Unzulässige Ersetzung.", "Original")])
    assert store.snapshot() == (1, [original])
    with pytest.raises(ValueError, match="within batch"):
        store.append_documents([Document("double", "A"), Document("double", "B")])
    assert store.snapshot() == (1, [original])
    with pytest.raises(ValueError, match="ID conflict"):
        store.append_documents([Document("existing", original.text, "Andere Quelle")])
    assert store.snapshot() == (1, [original])


def test_expected_revision_and_change_feed_are_consistent(tmp_path):
    path = tmp_path / "memory.sqlite3"
    first, second = MemoryStore(path), MemoryStore(path)
    document = Document("a", "Astronomie beschreibt Sterne.")
    first.append_documents([document], expected_revision=0)
    with pytest.raises(RevisionConflict, match="expected revision 0, got 1"):
        second.append_documents([Document("b", "Noch nicht veröffentlicht.")], expected_revision=0)
    assert second.changes_since(0) == (1, [document])
    assert second.changes_since(1) == (1, [])
    with pytest.raises(RevisionConflict):
        second.changes_since(2)
    later = Document("b", "Nun atomar ergänzt.")
    second.append_documents([later], expected_revision=1)
    assert first.changes_since(1) == (2, [later])


def test_concurrent_writers_preserve_every_document_and_duplicate_once(tmp_path):
    path = tmp_path / "memory.sqlite3"
    MemoryStore(path)
    barrier = threading.Barrier(6)
    shared = Document("common", "Gemeinsam identischer Text.", "Test")

    def append_batch(worker):
        store = MemoryStore(path)
        batch = [Document(f"{worker}-{number}", f"Messreihe {worker}, Beobachtung {number}.", "Test")
                 for number in range(20)]
        barrier.wait(timeout=10)
        return store.append_documents(batch + [shared])

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(append_batch, range(6)))
    revision, documents = MemoryStore(path).snapshot()
    assert revision == 6 and len(documents) == 121
    assert len({document.id for document in documents}) == 121
    assert sum(len(result["added"]) for result in results) == 121
    assert sum(len(result["existing"]) for result in results) == 5


def test_snapshot_never_mixes_document_count_and_revision(tmp_path):
    path = tmp_path / "memory.sqlite3"
    reader = MemoryStore(path)
    done = threading.Event()

    def write():
        writer = MemoryStore(path)
        try:
            for number in range(30):
                writer.append_documents([Document(str(number), f"Text {number}")])
        finally:
            done.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write)
        while not done.is_set():
            revision, documents = reader.snapshot()
            assert revision == len(documents)
        future.result(timeout=10)
    assert reader.revision() == 30


def test_query_history_is_saved_but_never_indexed_as_knowledge(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = MemoryStore(path)
    document = Document("seed", "Astronomie untersucht Himmelskörper.", "Wissen")
    store.append_documents([document])
    result = {"answer": "Privater Verlaufstext", "abstained": True, "matches": []}
    store.record_query("Verlaufsexklusiver Ausdruck", result)
    assert store.snapshot() == (1, [document])
    assert store.changes_since(1) == (1, [])
    assert store.stats()["history_count"] == 1
    memory = MemoryStore(path).load_memory()
    assert memory.documents == [document]
    assert memory.ask("Verlaufsexklusiver Ausdruck")["abstained"]
    with sqlite3.connect(path) as connection:
        prompt, encoded = connection.execute("SELECT prompt,result_json FROM query_history").fetchone()
    assert prompt == "Verlaufsexklusiver Ausdruck" and json.loads(encoded) == result
    with pytest.raises(ValueError):
        store.record_query("Fehlgeschlagen", {"score": float("nan")})
    assert store.stats()["history_count"] == 1


def test_explicit_relations_share_store_and_batch_validation_is_atomic(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = {"subject": "Pudel", "predicate": "is_a", "object": "Hund", "source": "Biologie"}
    second = {"subject": "Hund", "predicate": "is_a", "object": "Tier", "source": "Biologie"}
    assert store.add_relations([first, second]) == {"revision": 1, "added": 2, "existing": 0}
    assert store.add_relations([first]) == {"revision": 1, "added": 0, "existing": 1}
    assert MemoryStore(store.path).relations() == [first, second]
    assert store.changes_since(0) == (1, [])
    assert store.stats()["relation_count"] == 2
    with pytest.raises(ValueError):
        store.add_relations([{"subject": "Tier", "predicate": "is_a", "object": "Lebewesen"},
                             {"subject": "Pudel", "predicate": "looks_like", "object": "Wolke"}])
    assert store.relations() == [first, second] and store.revision() == 1


def test_npz_cannot_accidentally_become_live_database(tmp_path):
    legacy = tmp_path / "memory.npz"
    WaveMemory([Document("legacy", "Unveränderliches Altarchiv.")], dimensions=64).save(legacy)
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="Austauschformat"):
        MemoryStore(legacy)
    assert legacy.read_bytes() == before
    assert WaveMemory.load(legacy).documents[0].id == "legacy"


def test_default_store_path_is_anchored_to_project_independent_of_cwd(tmp_path, monkeypatch):
    from freqai import store as store_module
    expected = Path(store_module.__file__).resolve().parent.parent / "memory" / "memory.sqlite3"
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_MEMORY_PATH == expected
    assert DEFAULT_MEMORY_PATH.is_absolute()
    assert inspect.signature(MemoryStore).parameters["path"].default == expected


def test_dialogue_prompts_survive_restart_and_participate_in_deduplication(tmp_path):
    path = tmp_path / "dialogues.sqlite3"
    store = MemoryStore(path)
    first = Document("morning", "Guten Morgen! Wie hast du geschlafen?", "Alltag", "Guten Morgen!")
    second = Document("wake", first.text, first.source, "Ich bin gerade aufgewacht.")
    result = store.append_documents([first, second])
    assert result["added"] == [first, second]
    assert MemoryStore(path).snapshot() == (1, [first, second])
    assert store.append_documents([Document("duplicate", first.text, first.source, first.prompt)])["existing"] == [first]
    with pytest.raises(ValueError, match="ID conflict"):
        store.append_documents([Document(first.id, first.text, first.source, second.prompt)])
    result = store.load_memory().ask(second.prompt)
    assert result["matches"][0]["id"] == second.id
    assert result["answer"] == second.text


def test_legacy_sqlite_schema_migration_preserves_texts_sequences_and_revisions(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT INTO metadata VALUES ('revision', 7);
            CREATE TABLE documents (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                added_revision INTEGER NOT NULL,
                UNIQUE(text, source)
            );
            INSERT INTO documents(sequence,id,text,source,added_revision)
            VALUES (12,'legacy','Bestehender Text','Altbestand',7);
        """)
    store = MemoryStore(path)
    legacy = Document("legacy", "Bestehender Text", "Altbestand")
    assert store.snapshot() == (7, [legacy])
    assert store.changes_since(6) == (7, [legacy])
    # The old unique(text, source) index must be removed, too.
    new = Document("dialogue", legacy.text, legacy.source, "Ein neuer Gesprächsanlass")
    store.append_documents([new])
    assert MemoryStore(path).snapshot() == (8, [legacy, new])
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT sequence FROM documents WHERE id='legacy'").fetchone()[0] == 12
        assert connection.execute("SELECT sequence FROM documents WHERE id='dialogue'").fetchone()[0] > 12


def test_archived_legacy_variants_are_preserved_without_entering_active_waves(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    active = Document("conversation", "Hallo! Wie läuft dein Tag?", "Alltag", "Hallo!")
    store.append_documents([active])
    first = Document("reused-legacy-id", "Altes Technikhandbuch.", "Archiv")
    second = Document(first.id, "Abweichende historische Fassung.", "Archiv")
    assert store.archive_documents([first, second, first]) == {"added": 2, "existing": 1}
    assert MemoryStore(store.path).archived_documents() == [first, second]
    assert store.snapshot() == (1, [active])
    assert store.changes_since(1) == (1, [])
    assert store.stats()["archive_document_count"] == 2
    assert store.load_memory().ask("Technikhandbuch")["abstained"]
    with pytest.raises(ValueError):
        store.archive_documents([Document("valid", "Darf nicht teilweise hinein."),
                                 Document("", "Ungültige ID")])
    assert store.archived_documents() == [first, second]
