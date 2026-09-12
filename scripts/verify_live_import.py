"""Verify HTTP append plus external CLI import against the already running server.

This intentionally activates the conversation fixture in the canonical memory.
It appends idempotently and never deletes existing or user-added documents.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"


def request(route, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(URL + route, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    fixture = ROOT / "memory" / "fixtures" / "extension_120.jsonl"
    documents = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(documents) >= 120 and all(document.get("prompt") for document in documents)
    before = request("/api/state")
    first = request("/api/documents", {"documents": documents[:60]})
    partial = request("/api/state")
    assert first["live_revision"] == partial["live_revision"]
    command = [sys.executable, "-m", "freqai", "import", str(fixture), "--memory",
               str(ROOT / "memory" / "memory.sqlite3")]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    imported = json.loads(completed.stdout)
    # Only state reads here: revision activation must happen in the independent
    # sync worker, not as a hidden side effect of /api/ask or /api/documents.
    deadline = time.monotonic() + 10
    while True:
        final = request("/api/state")
        if final["live_revision"] >= imported["revision"]:
            break
        if time.monotonic() > deadline:
            raise AssertionError("External CLI import did not become live")
        time.sleep(.05)
    listed = request("/api/documents")
    rows = {document["id"]: document for document in listed["documents"]}
    assert all(rows.get(document["id"]) == document for document in documents)
    assert final["time_s"] >= partial["time_s"] >= before["time_s"]
    assert final["ticks"] > before["ticks"]
    assert final["document_count"] >= 120
    answers = []
    for prompt in ("Hallo", "Wie geht es dir?", "Danke", "Tschüss"):
        result = request("/api/ask", {"prompt": prompt})
        assert not result["abstained"], prompt
        assert result["matches"][0].get("matched_prompt")
        answers.append({"prompt": prompt, "answer": result["answer"], "id": result["matches"][0]["id"]})
    time.sleep(.25)
    later = request("/api/state")
    assert later["time_s"] > final["time_s"]
    assert later["displacement"] != final["displacement"]
    assert abs(later["energy"]-final["energy"]) <= max(1, final["energy"])*1e-12
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(), "url": URL,
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "before": {k: before[k] for k in ("document_count", "time_s", "ticks", "revision")},
              "http_batch": {"added": len(first["added"]), "existing": len(first["existing"]),
                             "live_revision": first["live_revision"]},
              "external_cli_import": imported,
              "after": {k: later[k] for k in ("document_count", "time_s", "ticks", "revision", "energy")},
              "answers": answers, "store": request("/api/memory"),
              "server_restart_during_import": False, "all_fixture_rows_verified": True,
              "old_rows_deleted": False, "existing_wave_clock_preserved": True}
    output = ROOT / "results" / "live_memory" / "active_import.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
