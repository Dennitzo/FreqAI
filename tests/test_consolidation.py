import json
import sqlite3

import pytest

from scripts.consolidate_memory import import_articles, schema, wikipedia_record, digest, verify_catalog
from freqai.memory import Document
from freqai.store import MemoryStore


def article(identifier="123", text="Eine vollständige Einleitung.\n\nGeschichte\nNoch ein Absatz.\n* Ein Listeneintrag"):
    return {"id": identifier, "title": "Testartikel", "url": "https://de.wikipedia.org/wiki/Testartikel", "text": text}


def test_complete_article_replaces_lead_without_losing_headings_or_lists(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.append_documents([Document("wikipedia-de-20231101-123-lead", "Eine Einleitung.", "Alt")])
    original = article()
    with store._transaction(write=True) as connection:
        schema(connection)
        result = import_articles(connection, [original], 2, "test.parquet")
    assert result == {"articles": 1, "inserted": 1, "existing": 0, "replaced_leads": 1}
    with store._connection() as connection:
        assert connection.execute("SELECT id,text,source FROM documents").fetchone() == wikipedia_record(original)
        metadata = json.loads(connection.execute("SELECT metadata_json FROM information_provenance").fetchone()[0])
        assert metadata["article_sha256"] == digest(original["text"])


def test_repeat_does_not_duplicate_articles(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    with store._transaction(write=True) as connection:
        schema(connection)
        import_articles(connection, [article()], 1, "test")
        repeated = import_articles(connection, [article()], 1, "test")
        assert repeated["inserted"] == 0 and repeated["existing"] == 1
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


@pytest.mark.parametrize("invalid", [dict(article(), question="Frage?"), dict(article(), text=""),
                                     dict(article(), url="file:///local"), dict(article(), id="bad")])
def test_invalid_source_rolls_back_lead_removal_and_prior_insertions(tmp_path, invalid):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    lead = Document("wikipedia-de-20231101-123-lead", "Die frühere Einleitung.", "Alt")
    store.append_documents([lead])
    before = store.snapshot()
    with pytest.raises(ValueError):
        with store._transaction(write=True) as connection:
            schema(connection)
            import_articles(connection, [article(), invalid], 2, "test")
    assert store.snapshot() == before


def test_conflicting_article_is_not_overwritten(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    with store._transaction(write=True) as connection:
        schema(connection)
        import_articles(connection, [article()], 1, "test")
    with pytest.raises(ValueError, match="Conflicting complete article"):
        with store._transaction(write=True) as connection:
            import_articles(connection, [article(text="Geänderter Text.")], 2, "test")
    with store._connection() as connection:
        assert connection.execute("SELECT text FROM documents").fetchone()[0] == wikipedia_record(article())[1]


def test_catalog_verification_detects_corruption_without_original_download(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    with store._transaction(write=True) as connection:
        schema(connection)
        import_articles(connection, [article()], 1, "test")
        connection.execute("INSERT INTO information_imports VALUES (?,?,?,?,?)",
                           ("test", "0" * 64, 100, 1, json.dumps({"kind": "complete_wikipedia"})))
        assert verify_catalog(connection)["articles_verified"] == 1
        connection.execute("UPDATE documents SET text=text || ' Verändert.'")
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_catalog(connection)
