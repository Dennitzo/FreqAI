"""Audit migration and a subsequent live information-only import on the real API."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.cli import read_documents
from freqai.memory import Document
from freqai.store import MemoryStore

OUT = ROOT / "results/unpaired_system/deployment"
DATABASE = ROOT / "memory/memory.sqlite3"
URL = "http://127.0.0.1:8765"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request(path, data=None):
    encoded = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
    req = urllib.request.Request(URL+path, data=encoded, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=650) as response:
        return json.load(response)


def digest(document):
    return hashlib.sha256(json.dumps(asdict(document), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(stage):
    store = MemoryStore(DATABASE)
    if stage == "before-migration":
        session = "unpaired-migration-audit-"+uuid.uuid4().hex
        output = request("/api/ask", {"prompt": "Was ist eine Frequenz?", "mode": "wave", "session_id": session})
        assert output["tokens"]
        backup = ROOT/"memory/backups"/("before-final-unpaired-migration-"+time.strftime("%Y%m%d-%H%M%S")+".sqlite3")
        with sqlite3.connect(DATABASE) as source, sqlite3.connect(backup) as destination:
            source.backup(destination)
            assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        old = store.snapshot()[1]
        selected = [doc for doc in old if doc.prompt.strip() or doc.source.startswith("Authored synthetic language prior / ")]
        selected_ids = {doc.id for doc in selected}
        retained = [doc for doc in old if doc.id not in selected_ids]
        record = {"backup": str(backup), "session": session, "conversation": store.load_conversation(session),
                  "server": read(ROOT/"runtime/server-process.json"), "statistics": store.stats(),
                  "archival_candidates": {doc.id: digest(doc) for doc in selected},
                  "retained_documents": {doc.id: digest(doc) for doc in retained},
                  "prior_archive": [asdict(doc) for doc in store.archived_documents()]}
        save("before-migration.json", record)
        print(json.dumps({"backup": str(backup), "selected": len(selected), "retained": len(retained)}), flush=True)
    elif stage == "before-live-import":
        before = read(OUT/"before-migration.json")
        current = {doc.id: digest(doc) for doc in store.snapshot()[1]}
        archive = {digest(doc) for doc in store.archived_documents()}
        assert current == before["retained_documents"]
        assert set(before["archival_candidates"].values()) <= archive
        assert list(store.load_conversation(before["session"])) == before["conversation"]
        assert store.stats()["legacy_schema"] is False
        with sqlite3.connect(DATABASE) as connection:
            assert "prompt" not in {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        started = time.perf_counter()
        answer = request("/api/ask", {"prompt": "Und welche Einheit hat sie?", "session_id": before["session"]})
        assert "Hertz" in answer["answer"] and answer["method"] == "autoregressive_unpaired_spectral_decoder"
        save("before-live-import.json", {"session": before["session"], "conversation": store.load_conversation(before["session"]),
             "server": read(ROOT/"runtime/server-process.json"), "statistics": store.stats(), "state": request("/api/state"),
             "old_information_context_migrated_by_token_names": True, "first_request_seconds": time.perf_counter()-started,
             "first_response": answer})
        print(json.dumps({"status": "migration_and_old_followup_passed", "answer": answer["answer"],
                          "seconds": time.perf_counter()-started}), flush=True)
    elif stage == "after-live-import":
        before = read(OUT/"before-live-import.json")
        original = read(OUT/"before-migration.json")
        current_server = read(ROOT/"runtime/server-process.json")
        assert current_server["pid"] == before["server"]["pid"]
        assert list(store.load_conversation(before["session"])) == before["conversation"]
        inputs = [ROOT/"memory/information/conversation_facts.jsonl"]
        lexicon = ROOT/"memory/information/conversation_lexicon_v1.jsonl"
        if lexicon.exists():
            inputs.append(lexicon)
        additions = [doc for path in inputs for doc in read_documents(path)]
        current = {doc.id: doc for doc in store.snapshot()[1]}
        assert all(not doc.prompt for doc in current.values())
        assert all(current.get(doc.id) == doc for doc in additions)
        assert all(digest(current[key]) == value for key,value in original["retained_documents"].items())
        archive = {digest(doc) for doc in store.archived_documents()}
        assert set(original["archival_candidates"].values()) <= archive
        assert all(digest(Document(**doc)) in archive
                   for doc in original["prior_archive"])
        revision_before_rejection = store.revision()
        try:
            request("/api/documents", {"text": "An obsolete answer.",
                                      "question": "An obsolete question?", "answer": "An obsolete answer."})
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("The deployed server accepted question/answer fields")
        assert store.revision() == revision_before_rejection
        preserved = {}
        with sqlite3.connect(original["backup"]) as old, sqlite3.connect(DATABASE) as live:
            old.row_factory = live.row_factory = sqlite3.Row
            for table,key in (("query_history","sequence"),("conversations","session_id")):
                previous = {row[key]: dict(row) for row in old.execute(f"SELECT * FROM {table}")}
                latest = {row[key]: dict(row) for row in live.execute(f"SELECT * FROM {table}")}
                if table == "query_history":
                    assert all(latest.get(k) == v for k,v in previous.items())
                    preserved[table] = {"original_rows": len(previous), "unchanged": True}
                else:
                    assert all(k in latest and latest[k]["revision"] >= v["revision"] for k,v in previous.items())
                    assert all(latest[k] == v for k,v in previous.items() if latest[k]["revision"] == v["revision"])
                    preserved[table] = {"original_rows": len(previous), "no_revision_lost": True}
            assert live.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        deadline = time.monotonic()+240
        while request("/api/state")["document_count"] != len(current):
            if time.monotonic() > deadline:
                raise TimeoutError("New information did not enter the live wave")
            time.sleep(1)
        frames = []
        for _ in range(3):
            frame = request("/api/state")
            assert frame["error"] is None
            frames.append({key: frame[key] for key in ("time_s","ticks","energy","modal_count","key_mode_count")})
            time.sleep(3)
        assert frames[-1]["ticks"] > frames[0]["ticks"]
        assert frames[0]["time_s"] > before["state"]["time_s"]
        report = {"server": current_server,"statistics": store.stats(),"source_inputs": [str(p.relative_to(ROOT)) for p in inputs],
                  "all_active_documents_unpaired": True,"archived_originals_preserved": True,"original_state": preserved,
                  "alternate_qa_fields_rejected_without_mutation": True,
                  "same_process_during_information_import": True,"saved_context_preserved": True,"continuous_frames": frames,
                  "session": before["session"],"outputs":[]}
        report["runtime_code_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT/"freqai").glob("*.py"))
        }
        frozen = read(ROOT/"results/unpaired_system/evaluation/final_freeze.json")["manifest"]["code"]
        assert all(report["runtime_code_sha256"][key] == frozen[key]
                   for key in ("freqai\\unpaired.py", "freqai\\unpaired_runtime.py", "freqai\\information.py", "freqai\\spectral_ops.py"))
        report["decoder_matches_quality_freeze"] = True
        report["post_freeze_change"] = "Raw ingestion rejects QA/turn fields before projecting to Document; decoder unchanged."
        print(json.dumps({"status": "preservation_and_live_wave_passed", "statistics": store.stats()}), flush=True)
        for prompt in ("Hallo", "Mir geht es gut, wie geht es dir denn?", "Wie heißt du?", "Was ist eine Frequenz?", "Und welche Einheit hat sie?"):
            started = time.perf_counter()
            response = request("/api/ask", {"prompt": prompt, "session_id": before["session"]})
            seconds = time.perf_counter()-started
            assert response["method"] == "autoregressive_unpaired_spectral_decoder"
            assert response["tokens"] and response["matches"] == []
            assert response["configuration"]["paired_documents_used"] == response["configuration"]["optimizer_steps"] == 0
            report["outputs"].append({"prompt": prompt,"seconds": seconds,"response": response})
            print(json.dumps({"prompt": prompt,"answer": response["answer"],"seconds": seconds}), flush=True)
        report["statistics"] = store.stats()
        save("active-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("before-migration","before-live-import","after-live-import"))
    main(parser.parse_args().stage)
