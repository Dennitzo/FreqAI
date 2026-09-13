"""Read optional local corpus integration data without a second dataset copy."""
import json
from pathlib import Path
import sqlite3

import pytest

DATABASE = Path(__file__).resolve().parents[1] / "memory/memory.sqlite3"


def connect():
    if not DATABASE.exists():
        pytest.skip("Local central corpus is not installed", allow_module_level=True)
    connection = sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True)
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='information_imports'").fetchone():
        connection.close()
        pytest.skip("Central corpus has not been consolidated", allow_module_level=True)
    return connection


def records(prefix):
    with connect() as connection:
        rows = connection.execute("""SELECT d.id,d.text,d.source,p.metadata_json
            FROM documents d JOIN information_provenance p ON p.document_id=d.id
            WHERE d.id>=? AND d.id<? ORDER BY d.id""", (prefix, prefix + "\uffff")).fetchall()
    return [dict(id=identifier, text=text, source=source, provenance=json.loads(metadata))
            for identifier, text, source, metadata in rows]


def receipt(source_key):
    with connect() as connection:
        row = connection.execute("SELECT sha256,byte_count,record_count,metadata_json FROM information_imports WHERE source_key=?",
                                 (source_key,)).fetchone()
    if row is None:
        pytest.skip(f"Local source receipt unavailable: {source_key}", allow_module_level=True)
    return dict(sha256=row[0], bytes=row[1], records=row[2], metadata=json.loads(row[3]))
