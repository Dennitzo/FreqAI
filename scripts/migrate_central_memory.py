"""Preserve old data centrally without enabling it in the conversation corpus.

Run after stopping the legacy NPZ server. This never replaces active records and
never deletes inputs. It archives old documents and makes byte-for-byte backups.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from freqai.cli import read_documents
from freqai.store import MemoryStore


def main():
    store = MemoryStore()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "memory" / "backups" / f"legacy-{timestamp}"
    backup.mkdir(parents=True, exist_ok=True)
    evidence = []
    for path in [ROOT / "runtime" / "memory.npz", ROOT / "runtime" / "demo.npz",
                 ROOT / "memory" / "fixtures" / "demo.jsonl",
                 ROOT / "memory" / "fixtures" / "benchmark.json"]:
        if not path.exists():
            continue
        documents = read_documents(path)
        destination = backup / path.name
        shutil.copy2(path, destination)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        result = store.archive_documents(documents)
        archived = {(d.id, d.prompt, d.text, d.source) for d in store.archived_documents()}
        assert all((d.id, d.prompt, d.text, d.source) in archived for d in documents)
        evidence.append({"original": str(path.relative_to(ROOT)), "backup": str(destination.relative_to(ROOT)),
                         "sha256": digest, "documents": len(documents), "archive_result": result})
    result = {"timestamp_utc": timestamp, "inputs": evidence, "store": store.stats(),
              "active_records_replaced": False, "all_legacy_texts_preserved": True}
    output = ROOT / "results" / "live_memory" / "migration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
