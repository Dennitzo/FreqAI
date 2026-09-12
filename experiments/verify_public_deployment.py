"""Check an additive public-corpus import through the real running local API."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.cli import read_documents
from freqai.store import MemoryStore

OUT = ROOT/"results/public_corpus/deployment"
CORPUS = ROOT/"memory/imports/public_dialogue_expansion.jsonl"
STORE = ROOT/"memory/memory.sqlite3"
URL = "http://127.0.0.1:8765"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request(path, data=None):
    encoded = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
    req = urllib.request.Request(URL+path, data=encoded, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def metadata():
    return json.loads((ROOT/"runtime/server-process.json").read_text(encoding="utf-8-sig"))


def main(stage):
    store = MemoryStore(STORE)
    if stage == "backup":
        path = ROOT/"memory/backups"/("before-public-corpus-deployment-"+time.strftime("%Y%m%d-%H%M%S")+".sqlite3")
        with sqlite3.connect(STORE) as source, sqlite3.connect(path) as destination:
            source.backup(destination)
            assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        record = {"backup": str(path), "statistics": store.stats(), "server": metadata(),
                  "document_hashes": {doc.id: digest(asdict(doc)) for doc in store.snapshot()[1]},
                  "archive_hash": digest([asdict(doc) for doc in store.archived_documents()])}
        save("backup-final.json", record)
        print(json.dumps({key: record[key] for key in ("backup", "statistics")}), flush=True)
    elif stage == "before":
        session = "public-import-audit-"+uuid.uuid4().hex
        result = request("/api/ask", {"prompt": "Hallo", "mode": "wave", "session_id": session})
        record = {"session_id": session, "server": metadata(), "statistics": store.stats(),
                  "conversation": store.load_conversation(session), "state": request("/api/state"),
                  "result": result}
        save("before-live-import.json", record)
        print(json.dumps({"session": session, "answer": result["answer"], "stats": record["statistics"]}), flush=True)
    elif stage == "preservation-final":
        backup = json.loads((OUT/"backup-final.json").read_text(encoding="utf-8"))
        preserved = {}
        with sqlite3.connect(backup["backup"]) as previous, sqlite3.connect(STORE) as current:
            previous.row_factory = current.row_factory = sqlite3.Row
            for table, key in (("query_history", "sequence"), ("conversations", "session_id"),
                               ("configuration", "key"), ("relations", "sequence")):
                old = {row[key]: dict(row) for row in previous.execute(f"SELECT * FROM {table}")}
                new = {row[key]: dict(row) for row in current.execute(f"SELECT * FROM {table}")}
                assert all(new.get(key) == row for key, row in old.items()), table
                preserved[table] = {"original_rows": len(old), "current_rows": len(new),
                                    "all_original_rows_unchanged": True,
                                    "original_sha256": digest(old)}
            assert current.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        save("all-saved-state-preservation.json", {"backup": backup["backup"], "tables": preserved,
                                                   "integrity_check": "ok", "statistics": store.stats()})
        print(json.dumps(preserved), flush=True)
    elif stage == "restart-before":
        active = json.loads((OUT/"active-report.json").read_text(encoding="utf-8"))
        session = active["session"]
        save("before-performance-restart.json", {
            "server": metadata(), "statistics": store.stats(), "session": session,
            "conversation": store.load_conversation(session),
            "document_hash": digest([asdict(doc) for doc in store.snapshot()[1]]),
            "archive_hash": digest([asdict(doc) for doc in store.archived_documents()])})
        print("Final code restart preservation snapshot saved", flush=True)
    elif stage == "restart-after":
        before = json.loads((OUT/"before-performance-restart.json").read_text(encoding="utf-8"))
        assert metadata()["pid"] != before["server"]["pid"]
        assert digest([asdict(doc) for doc in store.snapshot()[1]]) == before["document_hash"]
        assert digest([asdict(doc) for doc in store.archived_documents()]) == before["archive_hash"]
        assert list(store.load_conversation(before["session"])) == before["conversation"]
        frame = request("/api/state")
        assert frame["document_count"] == 12000 and frame["error"] is None
        save("performance-restart.json", {
            "server": metadata(), "statistics": store.stats(),
            "document_hash": before["document_hash"], "archive_hash": before["archive_hash"],
            "session": before["session"], "saved_conversation_preserved": True,
            "scope": "Code version restart after additive live data import; no data mutation.",
            "state": frame})
        print("Final server version ready; documents, archive and saved conversation unchanged", flush=True)
    elif stage == "after":
        before = json.loads((OUT/"before-live-import.json").read_text(encoding="utf-8"))
        backup = json.loads((OUT/"backup-final.json").read_text(encoding="utf-8"))
        incoming = read_documents(CORPUS)
        current = {doc.id: doc for doc in store.snapshot()[1]}
        assert all(doc.id in current and current[doc.id] == doc for doc in incoming)
        assert all(digest(asdict(current[key])) == value for key, value in backup["document_hashes"].items())
        assert digest([asdict(doc) for doc in store.archived_documents()]) == backup["archive_hash"]
        assert metadata()["pid"] == before["server"]["pid"]
        session = before["session_id"]
        assert list(store.load_conversation(session)) == before["conversation"]
        deadline = time.monotonic()+120
        while request("/api/state")["document_count"] < len(current):
            if time.monotonic() >= deadline:
                raise TimeoutError("Committed corpus was not published by the live wave worker")
            time.sleep(.5)
        ticks = []
        for _ in range(3):
            frame = request("/api/state")
            ticks.append({key: frame[key] for key in ("time_s", "ticks", "document_count", "energy", "modal_count", "error")})
            time.sleep(.6)
        assert ticks[-1]["ticks"] > ticks[0]["ticks"]
        assert ticks[0]["time_s"] > before["state"]["time_s"]
        assert all(frame["error"] is None for frame in ticks)
        print(json.dumps({"status": "live_import_and_preservation_passed", "statistics": store.stats(), "ticks": ticks}), flush=True)
        outputs = []
        for prompt in ("Mir geht es gut, wie geht es dir denn?", "gut und dir?", "Danke."):
            start = time.perf_counter()
            result = request("/api/ask", {"prompt": prompt, "mode": "wave", "session_id": session})
            assert result["wave_generation"] and result["matches"] == [] and result["tokens"]
            assert result["decoder"]["vocabulary_size"] > before["result"]["decoder"]["vocabulary_size"]
            outputs.append({"prompt": prompt, "seconds": time.perf_counter()-start, "result": result})
            print(json.dumps({"prompt": prompt, "answer": result["answer"], "seconds": outputs[-1]["seconds"]}), flush=True)
        assert store.load_conversation(session)[0] == before["conversation"][0]+3
        report = {"statistics": store.stats(), "server": metadata(), "session": session,
                  "same_process_during_import": True, "original_documents_and_archive_preserved": True,
                  "saved_context_preserved": True, "continuous_frames": ticks, "outputs": outputs,
                  "imported_pairs": len(incoming), "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                  "test_kind": "Runtime integration; broader answer quality is evaluated separately."}
        save("active-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("backup", "before", "after", "restart-before", "restart-after", "preservation-final"))
    main(parser.parse_args().stage)
