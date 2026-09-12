"""Record the real local deployment; preserve user text and existing history.

backup before restarting code, before before adding the language prior, after
after live import. The after stage checks additive data preservation and probes
the public API in its own conversation. It never treats answer fluency as proof
of conversational correctness.
"""
from __future__ import annotations

import argparse
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
from freqai.store import MemoryStore

OUT = ROOT / "results/generative_waves/deployment"
DB = ROOT / "memory/memory.sqlite3"
URL = "http://127.0.0.1:8765"


def request(path, body=None):
    payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(URL+path, data=payload,
                                 headers={"Content-Type": "application/json"} if payload else {})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def document_hashes(store):
    return {doc.id: hashlib.sha256(json.dumps([doc.prompt, doc.text, doc.source], ensure_ascii=False).encode()).hexdigest()
            for doc in store.snapshot()[1]}


def main(stage):
    store = MemoryStore(DB)
    if stage == "backup":
        target = ROOT / "memory/backups" / ("before-generative-waves-"+time.strftime("%Y%m%d-%H%M%S")+".sqlite3")
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB) as source, sqlite3.connect(target) as backup:
            source.backup(backup)
            assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        report = {"backup": str(target), "statistics": store.stats(), "documents": document_hashes(store),
                  "server": json.loads((ROOT/"runtime/server-process.json").read_text(encoding="utf-8-sig")),
                  "integrity_check": "ok"}
        save("backup.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "documents"}, ensure_ascii=True))
    elif stage == "before":
        session = "wave-deployment-"+uuid.uuid4().hex
        result = request("/api/ask", {"prompt": "Hallo", "mode": "wave", "session_id": session})
        assert result["method"] == "autoregressive_spectral_decoder" and result["matches"] == []
        report = {"session_id": session, "statistics": store.stats(), "state": request("/api/state"),
                  "first": result,
                  "server": json.loads((ROOT/"runtime/server-process.json").read_text(encoding="utf-8-sig"))}
        save("before-live-import.json", report)
        print(json.dumps({"answer": result["answer"], "vocabulary": result["decoder"]["vocabulary_size"],
                          "session_id": session, "statistics": report["statistics"]}, ensure_ascii=True))
    elif stage == "after":
        before = json.loads((OUT/"before-live-import.json").read_text(encoding="utf-8"))
        backup = json.loads((OUT/"backup.json").read_text(encoding="utf-8"))
        server = json.loads((ROOT/"runtime/server-process.json").read_text(encoding="utf-8-sig"))
        assert server["pid"] == before["server"]["pid"], "Import restarted the process"
        stats = store.stats()
        current = document_hashes(store)
        assert all(current.get(key) == value for key, value in backup["documents"].items())
        assert stats["archive_document_count"] == backup["statistics"]["archive_document_count"]
        assert stats["history_count"] >= before["statistics"]["history_count"]
        assert stats["document_count"] == backup["statistics"]["document_count"]+500
        session = before["session_id"]
        outputs = []
        for prompt in ("Mir geht es gut, wie geht es dir denn?", "gut und dir?"):
            started = time.perf_counter()
            result = request("/api/ask", {"prompt": prompt, "mode": "wave", "session_id": session})
            assert result["wave_generation"] and result["matches"] == []
            assert result["decoder"]["steps"]
            assert result["decoder"]["vocabulary_size"] > before["first"]["decoder"]["vocabulary_size"]
            outputs.append({"prompt": prompt, "seconds": time.perf_counter()-started, "result": result})
        state = request("/api/state?session_id="+session)
        assert state["time_s"] > before["state"]["time_s"]
        assert state["context_wave"]["energy"] > 0
        assert state["conversation_revision"] == before["first"]["conversation_revision"]+2
        report = {"statistics": store.stats(), "state": state, "outputs": outputs, "server": server,
                  "existing_documents_preserved": True, "archive_count_preserved": True,
                  "same_process_during_import": True, "session_and_history_preserved": True,
                  "quality": "Runtime evidence only; semantic success must be assessed separately."}
        save("active-report.json", report)
        print(json.dumps({"statistics": report["statistics"], "pid": server["pid"],
                          "outputs": [{"prompt": item["prompt"], "answer": item["result"]["answer"],
                                       "seconds": item["seconds"]} for item in outputs]}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("backup", "before", "after"))
    main(parser.parse_args().stage)
