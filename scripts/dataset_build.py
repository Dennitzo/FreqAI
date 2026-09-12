"""One central combined dataset for FreqAI: declarative information texts only.

The productive corpus lives scattered over an SQLite store and several import
files. This tool combines everything into one canonical location, ``dataset/``,
adds nested Wikipedia expansion steps from the pinned local Parquet archive and
records a leakage guard so that no evaluation split can ever enter the corpus.

Every step is a strict superset of the previous one: the sampling rank of an
article is fixed, so growth never reshuffles what is already there. No record
carries a question or answer field.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import import_information_corpus as wiki  # noqa: E402  pinned extraction recipe

DATASET_DIR = ROOT / "dataset"
CORPUS_DIR = DATASET_DIR / "corpus"
EVALUATION_DIR = DATASET_DIR / "evaluation"
STEPS = (20_000, 60_000, 100_000)
AUTHORED_FILES = ("memory/information/conversation_facts.jsonl",
                  "memory/information/conversation_lexicon_v1.jsonl")
EVALUATION_FILES = sorted((ROOT / "memory/evaluation").glob("*.json"))


def file_sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def authored_records():
    """The hand-written declarative conversation information, in stable order."""
    records = []
    for relative in AUTHORED_FILES:
        path = ROOT / relative
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("prompt", "").strip():
                raise ValueError(f"Authored record carries a pair field: {relative}")
            records.append({"id": item["id"], "text": item["text"], "source": item["source"],
                            "title": item.get("title", ""), "kind": "authored_declaration",
                            "origin_file": relative})
    return records


def scan_row_group(path, spec, group):
    """Eligible opening-paragraph records of one Parquet row group."""
    import pyarrow.parquet as parquet
    table = parquet.ParquetFile(path).read_row_group(group)
    out, rejected = [], {}
    for article in table.to_pylist():
        record, reason = wiki.article_record(article, spec)
        if record is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        out.append(record)
    return len(table), out, rejected


def collect(count_limit: int, workers: int):
    """Scan every pinned shard in parallel and keep the deterministic top ranks."""
    specs = wiki.source_specs()
    tasks = []
    shard_stats = {}
    for spec in specs:
        path = ROOT / "memory/sources/information_wikipedia_20231101_de" / spec["name"]
        if not path.exists():
            raise FileNotFoundError(f"Pinned source missing: {path}")
        meta = __import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(path).metadata
        shard_stats[spec["name"]] = {"rows": meta.num_rows, "row_groups": meta.num_row_groups,
                                     "bytes": path.stat().st_size, "sha256": spec["sha256"]}
        tasks.extend((path, spec, group) for group in range(meta.num_row_groups))
    scanned = 0
    rejected_total = {}
    candidates = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows, records, rejected in pool.map(lambda task: scan_row_group(*task), tasks):
            scanned += rows
            candidates.extend(records)
            for reason, count in rejected.items():
                rejected_total[reason] = rejected_total.get(reason, 0) + count
    # Identical prose under a different article id is one declaration only.
    unique, seen = [], set()
    for record in candidates:
        digest = record["provenance"]["text_sha256"]
        if digest in seen:
            rejected_total["duplicate_title_and_prose"] = rejected_total.get("duplicate_title_and_prose", 0) + 1
            continue
        seen.add(digest)
        unique.append(record)
    priority = sorted((record for record in unique if wiki.priority_title(record["title"])),
                      key=lambda record: (wiki.sample_rank(record["provenance"]["article_id"]), record["id"]))
    general = sorted((record for record in unique if not wiki.priority_title(record["title"])),
                     key=lambda record: (wiki.sample_rank(record["provenance"]["article_id"]), record["id"]))
    ordered = priority[:wiki.MAX_PRIORITY_ARTICLES] + general
    print(f"scanned {scanned} articles, eligible {len(unique)}, "
          f"priority {len(priority)}, seconds {time.perf_counter()-started:.1f}")
    return ordered, {"source_articles_scanned": scanned, "eligible_articles": len(unique),
                     "eligible_priority_articles": len(priority), "rejections": dict(sorted(rejected_total.items()))}, shard_stats


def write_step(step, records, path):
    ordered = sorted(records, key=lambda record: record["id"])
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"records": len(ordered), "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(path), "utf8_bytes": sum(len(record["text"].encode("utf-8")) for record in ordered)}


def leakage_guard(step_files):
    """No evaluation prompt, rubric or synthetic fact may appear as corpus text."""
    forbidden = set()
    for path in EVALUATION_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        if isinstance(payload, dict):
            for key in ("records", "cases", "scenarios", "queries", "examples"):
                entries.extend(payload.get(key) or [])
        def collect_texts(item):
            if isinstance(item, str):
                if len(item.strip()) > 3:
                    forbidden.add(normalized(item))
            elif isinstance(item, dict):
                for value in item.values():
                    collect_texts(value)
            elif isinstance(item, list):
                for value in item:
                    collect_texts(value)
        collect_texts(entries)
    forbidden.discard("")
    hits = []
    for path in step_files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if normalized(record["text"]) in forbidden:
                    hits.append({"file": path.name, "id": record["id"]})
    return {"evaluation_files": [{"file": path.name, "sha256": file_sha256(path)} for path in EVALUATION_FILES],
           "forbidden_texts": len(forbidden), "corpus_collisions": hits}


def productive_reference():
    """Does step one reproduce the corpus that is actually running?"""
    path = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
    if not path.exists():
        return {"available": False}
    ids = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return {"available": True, "file": path.name, "records": len(ids), "ids": sorted(ids)}


def main(argv):
    workers = int(argv[0]) if argv else 8
    DATASET_DIR.mkdir(exist_ok=True)
    CORPUS_DIR.mkdir(exist_ok=True)
    EVALUATION_DIR.mkdir(exist_ok=True)
    started = time.perf_counter()
    ordered, stats, shard_stats = collect(max(STEPS), workers)
    authored = authored_records()
    reference = productive_reference()
    steps, step_files = [], []
    for step in STEPS:
        selected = ordered[:step] + authored
        path = CORPUS_DIR / f"combined_step{len(steps)+1}_{step}.jsonl.gz"
        info = write_step(step, selected, path)
        info["wikipedia_leads"] = step
        info["authored_records"] = len(authored)
        if reference.get("available"):
            wiki_ids = {record["id"] for record in selected if record["id"].startswith("wikipedia-")}
            info["reproduces_productive_corpus"] = wiki_ids == set(reference["ids"]) if step == STEPS[0] else None
        steps.append(info)
        step_files.append(path)
    guard = leakage_guard(step_files)
    nested = all(set(json.loads(line)["id"] for line in gzip.open(a, "rt", encoding="utf-8"))
                <= set(json.loads(line)["id"] for line in gzip.open(b, "rt", encoding="utf-8"))
                for a, b in zip(step_files[:-1], step_files[1:]))
    manifest = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "license_note": "Wikipedia leads are CC-BY-SA-3.0/GFDL; authored texts are project-owned.",
                "record_contract": "declarative information text plus source, no question or answer field",
                "sources": {"dataset": wiki.DATASET, "configuration": wiki.CONFIGURATION,
                            "revision": wiki.REVISION, "shards": shard_stats},
                "sampling_seed": wiki.SAMPLING_SEED, "stats": stats,
                "nested_superset_of_previous_step": nested,
                "productive_corpus_reference": {k: v for k, v in reference.items() if k != "ids"},
                "steps": steps, "leakage_guard": guard,
                "build_seconds": round(time.perf_counter()-started, 1)}
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("nested_superset_of_previous_step", "steps", "stats")},
                     ensure_ascii=False, indent=1)[:1400])


if __name__ == "__main__":
    main(sys.argv[1:])
