"""Stream complete local Wikipedia articles into the central SQLite memory.

No network, sampling, paragraph truncation, QA conversion, backup or deletion
of source files. Publication is one transaction. A separate full verification
pass must succeed before the caller removes any source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.record_contract import validate_information_record
from freqai.store import MemoryStore

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def schema(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS information_provenance (
        document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
        metadata_json TEXT NOT NULL)""")
    connection.execute("""CREATE TABLE IF NOT EXISTS information_imports (
        source_key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, byte_count INTEGER NOT NULL,
        record_count INTEGER NOT NULL, metadata_json TEXT NOT NULL)""")


def remember_source(connection, key, path, count, metadata):
    connection.execute("INSERT OR REPLACE INTO information_imports VALUES (?,?,?,?,?)",
                       (key, file_digest(path), path.stat().st_size, count,
                        json.dumps(metadata, ensure_ascii=False)))


def wikipedia_record(article):
    validate_information_record(article)
    for key in ("id", "title", "url", "text"):
        if not isinstance(article.get(key), str) or not article[key].strip():
            raise ValueError(f"Invalid Wikipedia {key}: {article.get('id')}")
    if not article["id"].isdigit() or not article["url"].startswith("https://de.wikipedia.org/wiki/"):
        raise ValueError(f"Invalid Wikipedia address: {article['id']}")
    # Keep every original character, including paragraphs, headings and lists.
    identifier = f"wikipedia-de-20231101-{article['id']}"
    text = article["title"] + "\n\n" + article["text"]
    source = (f"Wikimedia Wikipedia 20231101.de | article_id={article['id']} | "
              f"{article['title']} | {article['url']}")
    return identifier, text, source


def import_articles(connection, articles, revision, source_key):
    counts = {"articles": 0, "inserted": 0, "existing": 0, "replaced_leads": 0}
    for article in articles:
        identifier, text, source = wikipedia_record(article)
        old = connection.execute("SELECT text,source FROM documents WHERE id=?", (identifier,)).fetchone()
        if old is not None and old != (text, source):
            raise ValueError(f"Conflicting complete article: {identifier}")
        # Removal is in the same transaction as the complete article insertion.
        counts["replaced_leads"] += connection.execute(
            "DELETE FROM documents WHERE id=?", (identifier + "-lead",)).rowcount
        if old is None:
            connection.execute("INSERT INTO documents(id,text,source,added_revision) VALUES (?,?,?,?)",
                               (identifier, text, source, revision))
            counts["inserted"] += 1
        else:
            counts["existing"] += 1
        metadata = {"kind": "complete_wikipedia_article", "dataset": "wikimedia/wikipedia",
                    "configuration": "20231101.de", "dataset_revision": REVISION,
                    "article_id": article["id"], "article_title": article["title"],
                    "article_url": article["url"], "source_key": source_key,
                    "article_sha256": digest(article["text"]), "text_sha256": digest(text),
                    "transform": "Original title plus two newlines plus complete unchanged source text",
                    "licenses": ["CC-BY-SA-3.0", "GFDL"]}
        connection.execute("INSERT OR REPLACE INTO information_provenance VALUES (?,?)",
                           (identifier, json.dumps(metadata, ensure_ascii=False)))
        counts["articles"] += 1
    return counts


def import_authored(connection, path, revision):
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            validate_information_record(record)
            identifier, text, source = (record[key] for key in ("id", "text", "source"))
            old = connection.execute("SELECT text,source FROM documents WHERE id=?", (identifier,)).fetchone()
            if old is not None and old != (text, source):
                raise ValueError(f"Prepared information conflicts with central memory: {identifier}")
            if old is None:
                connection.execute("INSERT INTO documents(id,text,source,added_revision) VALUES (?,?,?,?)",
                                   (identifier, text, source, revision))
            connection.execute("INSERT OR REPLACE INTO information_provenance VALUES (?,?)",
                               (identifier, json.dumps(record.get("provenance", {}), ensure_ascii=False)))
            count += 1
    manifest_path = path.with_suffix(".provenance.json")
    metadata = {"kind": "prepared_information", "manifest":
                json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}}
    remember_source(connection, path.relative_to(ROOT).as_posix(), path, count, metadata)
    return count


def read_articles(path):
    import pyarrow.parquet as parquet
    source = parquet.ParquetFile(path)
    if set(source.schema_arrow.names) != {"id", "title", "url", "text"}:
        raise ValueError(f"Not an information-only Wikipedia source: {path}")
    for batch in source.iter_batches(batch_size=1024, use_threads=True):
        yield from batch.to_pylist()


def verify(connection, paths):
    checked = 0
    for path in paths:
        count = 0
        receipt = connection.execute("SELECT sha256,record_count FROM information_imports WHERE source_key=?",
                                     (path.relative_to(ROOT).as_posix(),)).fetchone()
        if not receipt or receipt[0] != file_digest(path):
            raise ValueError(f"Missing or changed source receipt: {path}")
        for article in read_articles(path):
            identifier, text, source = wikipedia_record(article)
            row = connection.execute("SELECT text,source FROM documents WHERE id=?", (identifier,)).fetchone()
            if row != (text, source):
                raise ValueError(f"Full text verification failed: {identifier}")
            meta = json.loads(connection.execute("SELECT metadata_json FROM information_provenance WHERE document_id=?",
                                                 (identifier,)).fetchone()[0])
            if meta["text_sha256"] != digest(row[0]) or meta["article_sha256"] != digest(article["text"]):
                raise ValueError(f"Article hash mismatch: {identifier}")
            count += 1
        if count != receipt[1]:
            raise ValueError(f"Row count mismatch: {path}")
        checked += count
        print(json.dumps({"verified_source": path.name, "articles": count}), flush=True)
    integrity = connection.execute("PRAGMA integrity_check").fetchall()
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if integrity != [("ok",)] or foreign_keys:
        raise ValueError(f"SQLite integrity failure: {integrity}, {foreign_keys}")
    return {"articles_verified": checked, "integrity_check": "ok", "foreign_key_check": "ok"}


def verify_catalog(connection):
    """Verify stored text hashes after the original downloads have been removed.

The original-file comparison is recorded by verify() before cleanup. This pass
checks current database self-consistency against that retained provenance.
"""
    expected = {key: count for key, count, metadata in connection.execute(
        "SELECT source_key,record_count,metadata_json FROM information_imports")
        if json.loads(metadata).get("kind") == "complete_wikipedia"}
    actual = {key: 0 for key in expected}
    checked = 0
    for identifier, text, encoded in connection.execute("""SELECT d.id,d.text,p.metadata_json
            FROM documents d LEFT JOIN information_provenance p ON p.document_id=d.id
            WHERE d.id>=? AND d.id<?""", ("wikipedia-de-20231101-", "wikipedia-de-20231101-\uffff")):
        if encoded is None:
            raise ValueError(f"Missing provenance: {identifier}")
        metadata = json.loads(encoded)
        title, separator, original = text.partition("\n\n")
        if (not separator or title != metadata["article_title"] or
                digest(text) != metadata["text_sha256"] or
                digest(original) != metadata["article_sha256"]):
            raise ValueError(f"Stored article hash mismatch: {identifier}")
        actual[metadata["source_key"]] += 1
        checked += 1
    if not expected or actual != expected:
        raise ValueError(f"Source counts differ: {actual} != {expected}")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("SQLite integrity failure")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("SQLite foreign key failure")
    return {"articles_verified": checked, "integrity_check": "ok", "foreign_key_check": "ok",
            "source_files_required": False, "verification_scope": "stored hashes and source receipts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    paths = sorted((ROOT / "memory/sources/information_wikipedia_20231101_de").glob("*.parquet"))
    if args.verify_only and not paths:
        database = ROOT / "memory/memory.sqlite3"
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            print(json.dumps(verify_catalog(connection)), flush=True)
        return
    if not paths:
        parser.error("No local Wikipedia shards found")
    store = MemoryStore()
    started = time.perf_counter()
    with store._connection() as connection:
        before = store.stats()
        sources = []
        if not args.verify_only:
            manifest = json.loads((paths[0].parent / "source_manifest.json").read_text(encoding="utf-8"))
            expected = {Path(item["path"]).name: item["lfs"]["oid"] for item in manifest["files"]}
            for path in paths:
                if file_digest(path) != expected.get(path.name):
                    raise ValueError(f"Pinned source hash mismatch: {path.name}")
            connection.execute("BEGIN IMMEDIATE")
            try:
                schema(connection)
                revision = store._revision(connection) + 1
                # Preserve the current authored/API information and its provenance first.
                prepared = list((ROOT / "memory/information").glob("*.jsonl"))
                prepared += list((ROOT / "memory/imports").glob("information_science_extension*.jsonl"))
                for path in sorted(prepared):
                    import_authored(connection, path, revision)
                for path in paths:
                    key = path.relative_to(ROOT).as_posix()
                    counts = import_articles(connection, read_articles(path), revision, key)
                    remember_source(connection, key, path, counts["articles"],
                                    {"dataset_revision": REVISION, "kind": "complete_wikipedia", **counts})
                    sources.append({"file": key, **counts})
                    print(json.dumps(sources[-1]), flush=True)
                # The user explicitly no longer wants historical training pairs/backups.
                removed_archive = connection.execute("DELETE FROM document_archive").rowcount
                connection.execute("UPDATE metadata SET value=? WHERE key='revision'", (revision,))
                connection.execute("INSERT OR REPLACE INTO metadata VALUES ('information_reset_revision',?)", (revision,))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        checked = verify(connection, paths)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        report = {"before": before, "after": store.stats(), "sources": sources,
                  "verification": checked, "full_texts": True, "source_files_deleted": False,
                  "seconds": time.perf_counter() - started}
        output = ROOT / "results/memory_consolidation.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
