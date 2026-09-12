"""Read-only provenance and reproducibility audit of the frozen 20k candidate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments import import_information_corpus as importer
import pyarrow.parquet as parquet


started = time.perf_counter()
candidate = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines()]
expected = {row["provenance"]["article_id"]: row for row in rows}
assert len(expected) == len(rows) == 20000
assert all(row["prompt"] == "" for row in rows)
checked = set()
for spec in importer.source_specs():
    path = importer.SOURCE_DIR / spec["name"]
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=128):
        for article in batch.to_pylist():
            row = expected.get(article["id"])
            if row is None:
                continue
            provenance = row["provenance"]
            original = article["text"]
            assert provenance["raw_file"] == spec["name"]
            assert hashlib.sha256(original.encode("utf-8")).hexdigest() == provenance["article_sha256"]
            assert provenance["article_title"] == article["title"]
            paragraphs = [importer.normalized(original[start:end])
                          for start, end in provenance["paragraph_spans"]]
            prose = "\n\n".join(paragraphs)
            assert row["text"] == importer.normalized(article["title"]) + "\n\n" + prose
            assert hashlib.sha256(prose.encode("utf-8")).hexdigest() == provenance["paragraph_sha256"]
            assert hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == provenance["text_sha256"]
            assert importer.MIN_PARAGRAPH_CHARS <= len(prose) <= importer.MAX_PARAGRAPH_CHARS
            checked.add(article["id"])
assert checked == set(expected)
print(json.dumps({"original_articles_and_spans_verified": len(checked)}), flush=True)
reproduced, _ = importer.build([(importer.SOURCE_DIR / spec["name"], spec)
                               for spec in reversed(importer.source_specs())], count=20000)
digest = hashlib.sha256()
for row in reproduced:
    digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
assert digest.hexdigest() == importer.file_hash(candidate)
first_audit = json.loads(Path(__file__).with_name("candidate_audit.json").read_text(encoding="utf-8"))
sample = []
for original in first_audit["sample"]:
    current = next((row for row in rows if row["id"] == original["id"]), None)
    item = {"id": original["id"], "title": original["title"], "v1_text": original["text"]}
    if current is None:
        _, _, reason = importer.opening_paragraphs(original["title"], original["text"].partition("\n\n")[2])
        item.update({"status": "removed", "generic_filter_reason": reason})
    else:
        item.update({"status": "unchanged" if current["text"] == original["text"] else "changed",
                     "text": current["text"]})
    sample.append(item)
report = {
    "candidate": str(candidate.relative_to(ROOT)).replace("\\", "/"),
    "candidate_sha256": importer.file_hash(candidate),
    "verified_original_articles_and_spans": len(checked),
    "all_prompts_empty": True,
    "unique_article_ids": len(expected),
    "unique_title_and_prose": len({row["text"] for row in rows}),
    "source_shard_counts": dict(Counter(row["provenance"]["raw_file"] for row in rows)),
    "text_characters": {"min": min(len(row["text"]) for row in rows),
                        "max": max(len(row["text"]) for row in rows),
                        "mean": sum(len(row["text"]) for row in rows) / len(rows)},
    "source_field_max_characters": max(len(row["source"]) for row in rows),
    "reproduced_with_reversed_source_shard_order": True,
    "reproduced_sha256": digest.hexdigest(),
    "sample_seed": 20260906,
    "sample_drawn_from": "Frozen v1 audit; these are the same 40 article IDs, not a new sample from the cleaner corpus",
    "sample_status_counts": dict(Counter(row["status"] for row in sample)),
    "sample": sample,
    "records_with_removed_empty_template_fragments": sum(bool(row["provenance"]["removed_empty_template_fragments"]) for row in rows),
    "elapsed_seconds": time.perf_counter() - started,
    "sqlite_changed": False,
    "evaluation_files_read": [],
}
Path(__file__).with_name("candidate_audit_v3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "sample"}, ensure_ascii=False, indent=2))
