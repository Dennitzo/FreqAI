"""Audit additive prose ingestion and generation through the running server."""
from dataclasses import asdict
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
from freqai.cli import read_documents
from freqai.store import MemoryStore

OUT = ROOT / "results/information_corpus/deployment"
CORPUS = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
DATABASE = ROOT / "memory/memory.sqlite3"
URL = "http://127.0.0.1:8765"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def request(path, data=None):
    body = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
    req = urllib.request.Request(URL + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=650) as response:
        return json.load(response)


def server():
    return read(ROOT / "runtime/server-process.json")


def main(stage):
    store = MemoryStore(DATABASE)
    if stage == "backup":
        backup = ROOT / "memory/backups" / ("before-information-corpus-" + time.strftime("%Y%m%d-%H%M%S") + ".sqlite3")
        with sqlite3.connect(DATABASE) as source, sqlite3.connect(backup) as destination:
            source.backup(destination)
            assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        record = {"backup": str(backup), "statistics": store.stats(), "server": server(),
                  "document_hashes": {doc.id: digest(asdict(doc)) for doc in store.snapshot()[1]},
                  "archive_hash": digest([asdict(doc) for doc in store.archived_documents()])}
        save("before-code-update.json", record)
        print(json.dumps({"backup": str(backup), "statistics": record["statistics"]}), flush=True)
    elif stage == "before":
        session = "information-import-audit-" + uuid.uuid4().hex
        started = time.perf_counter()
        result = request("/api/ask", {"prompt": "Hallo", "mode": "wave", "session_id": session})
        record = {"session": session, "conversation": store.load_conversation(session),
                  "server": server(), "statistics": store.stats(), "state": request("/api/state"),
                  "greeting": result, "cold_chat_seconds": time.perf_counter()-started}
        save("before-live-import.json", record)
        print(json.dumps({"session": session, "answer": result["answer"], "seconds": record["cold_chat_seconds"]}), flush=True)
    elif stage == "after":
        before = read(OUT / "before-live-import.json")
        backup = read(OUT / "before-code-update.json")
        incoming = read_documents(CORPUS)
        assert len(incoming) == 20000 and all(not doc.prompt.strip() for doc in incoming)
        current = {doc.id: doc for doc in store.snapshot()[1]}
        assert all(current.get(doc.id) == doc for doc in incoming)
        assert all(digest(asdict(current[key])) == value for key, value in backup["document_hashes"].items())
        assert digest([asdict(doc) for doc in store.archived_documents()]) == backup["archive_hash"]
        assert server()["pid"] == before["server"]["pid"]
        assert list(store.load_conversation(before["session"])) == before["conversation"]
        # Audit every original saved row, rather than only a row count.
        preserved = {}
        with sqlite3.connect(backup["backup"]) as previous, sqlite3.connect(DATABASE) as live:
            previous.row_factory = live.row_factory = sqlite3.Row
            for table, key in (("query_history", "sequence"), ("conversations", "session_id")):
                old = {row[key]: dict(row) for row in previous.execute(f"SELECT * FROM {table}")}
                new = {row[key]: dict(row) for row in live.execute(f"SELECT * FROM {table}")}
                if table == "query_history":
                    assert all(new.get(key) == value for key, value in old.items()), table
                    preserved[table] = {"original_rows": len(old), "all_unchanged": True}
                else:
                    # The user may keep chatting during this audit. Preserve
                    # immutable history and accept legitimately newer context.
                    assert all(key in new and new[key]["revision"] >= value["revision"]
                               for key, value in old.items()), table
                    assert all(new[key] == value for key, value in old.items()
                               if new[key]["revision"] == value["revision"]), table
                    advanced = sum(new[key]["revision"] > value["revision"] for key, value in old.items())
                    preserved[table] = {"original_rows": len(old), "no_revision_lost": True,
                                        "unchanged_at_same_revision": True, "advanced_during_work": advanced}
            assert live.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        deadline = time.monotonic()+240
        while request("/api/state")["document_count"] != len(current):
            if time.monotonic() > deadline:
                raise TimeoutError("New information texts were not published by the continuous wave worker")
            time.sleep(1)
        frames = []
        for _ in range(3):
            frame = request("/api/state")
            assert frame["error"] is None
            frames.append({key: frame[key] for key in ("time_s", "ticks", "modal_count", "key_mode_count", "energy")})
            time.sleep(3)
        assert frames[-1]["ticks"] > frames[0]["ticks"]
        assert frames[0]["time_s"] > before["state"]["time_s"]
        report = {"server": server(), "statistics": store.stats(), "original_saved_state": preserved,
                  "existing_documents_and_archive_unchanged": True, "same_process_during_data_import": True,
                  "saved_audit_context_preserved": True, "continuous_frames": frames,
                  "imported_records": len(incoming), "all_imported_prompts_empty": True,
                  "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(), "session": before["session"],
                  "scope": "Runtime integration; answer quality is independently evaluated."}
        print(json.dumps({"status": "preservation_and_live_wave_passed", "statistics": store.stats()}), flush=True)
        outputs = []
        for prompt in ("Was ist eine Frequenz?", "Und welche Einheit hat sie?", "Hallo"):
            started = time.perf_counter()
            result = request("/api/ask", {"prompt": prompt, "mode": "wave", "session_id": before["session"]})
            seconds = time.perf_counter()-started
            assert result["tokens"] and result["matches"] == [] and result["wave_generation"]
            if prompt != "Hallo":
                assert result["method"] == "autoregressive_information_spectral_decoder"
                assert result["configuration"]["optimizer_steps"] == 0
            else:
                assert result["method"] == "autoregressive_spectral_decoder"
            if "Einheit" in prompt:
                assert "hertz" in result["answer"].casefold()
            outputs.append({"prompt": prompt, "seconds": seconds, "result": result})
            print(json.dumps({"prompt": prompt, "answer": result["answer"], "seconds": seconds}), flush=True)
        report["outputs"] = outputs
        report["statistics"] = store.stats()
        save("active-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("backup", "before", "after"))
    main(parser.parse_args().stage)
