"""Prepare a substantial German information corpus from pinned Wikipedia files.

No questions are generated. Every output prompt is empty. Full source articles
stay in the local Parquet archive; active entries contain a title and complete,
contiguous opening paragraphs. This script never opens application SQLite or
any evaluation file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "memory/sources/information_wikipedia_20231101_de"
IMPORT_DIR = ROOT / "memory/imports"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DATASET = "wikimedia/wikipedia"
CONFIGURATION = "20231101.de"
SAMPLING_SEED = "freqai-information-v1"
FILES = [
    (0, 781271996, "8778fa2f0907e4eb65917b27092a911a400f9f2ee2eea4271423b706c29249a6"),
    (7, 258122020, "e35fec0fe1ca3614b2a44c53db7251e94863cb3c72a20b7cce73ede0bd4fae96"),
    (13, 228591696, "69d92a94ba4b28a9ea38311a33070b7f9cebf31f0bea51ba13e4e2ef9867409f"),
    (19, 228498582, "380fdb4baf70df935d3fec62761d36c8bff2f000a2cc7b720ebde743a8352e18"),
]
MIN_PARAGRAPH_CHARS = 120
MAX_PARAGRAPH_CHARS = 1200
MAX_PRIORITY_ARTICLES = 500
PRIORITY_TITLES = frozenset({
    "frequenz", "schwingung", "welle", "welle (physik)", "stehende welle",
    "fouriertransformation", "fourier-reihe", "fourieranalyse", "resonanz",
    "harmonische schwingung", "phasenverschiebung", "amplitude", "akustik",
    "wellenlänge", "interferenz (physik)", "beugung (physik)", "hertz (einheit)",
    "periodendauer", "elektromagnetische welle", "physik", "schall", "licht",
})
PRIORITY_PATTERN = re.compile(r"\b(?:frequenz\w*|schwingung\w*|fourier\w*|resonanz\w*|akustik\w*)\b", re.IGNORECASE)
ENDING = re.compile(r"[.!?][\"'»“”’)]*$")
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
DEPENDENT_OPENING = re.compile(r"^(?:Er|Sie|Es|Dies|Diese[rsmn]?|Dadurch|Dabei|Dazu|Dort|Hierbei|Hierzu|Sein[e]?|Ihr[e]?)\b")
NON_PROSE = re.compile(r"^\s*(?:[*#•]|\d+[.)]\s)|https?://|www\.|\{\{|\}\}|\[\[|\]\]|<\/?[A-Za-z][^>]*>|\\(?:begin|end|frac|sum)\b", re.MULTILINE)
GERMAN_MARKERS = frozenset("der die das den dem des ein eine einer eines einem einen und oder aber ist sind war waren wird werden wurde wurden mit auf für von zu zum zur im in an am aus als sich nicht auch nach bei durch wie was wer warum welche welcher welches kann können er sie es wir man nur noch wenn dann diese dieser dieses haben hat hatte sein seine ihrer ihrem über unter zwischen vor bis mehr alle dass um so".split())


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_specs() -> list[dict]:
    return [{"shard": number, "name": f"train-{number:05d}-of-00020.parquet", "bytes": size, "sha256": digest,
             "url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{CONFIGURATION}/train-{number:05d}-of-00020.parquet"}
            for number, size, digest in FILES]


def download(spec: dict, offline: bool = False) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / spec["name"]
    if path.exists():
        if path.stat().st_size != spec["bytes"] or file_hash(path) != spec["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        return path
    if offline:
        raise FileNotFoundError(path)
    partial = path.with_suffix(".parquet.partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(spec["url"], timeout=90) as response, partial.open("wb") as target:
                while block := response.read(2 * 1024 * 1024):
                    target.write(block)
            if partial.stat().st_size != spec["bytes"] or file_hash(partial) != spec["sha256"]:
                raise ValueError(f"Downloaded source checksum mismatch: {partial}")
            partial.replace(path)
            return path
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def normalized(text: str) -> str:
    return re.sub(r"[\t \r\f\v]+", " ", unicodedata.normalize("NFC", text)).strip()


def paragraph_spans(text: str):
    """Original character offsets; blank lines delimit complete paragraphs."""
    start = 0
    for separator in re.finditer(r"\n\s*\n", text):
        end = separator.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end
        start = separator.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        yield start, end


def opening_paragraphs(title: str, text: str) -> tuple[str, list[list[int]], str | None]:
    """Keep complete opening prose, never truncate a long paragraph to fit."""
    paragraphs, offsets = [], []
    for start, end in paragraph_spans(text):
        paragraph = normalized(text[start:end])
        if not paragraphs and paragraph.casefold() == title.casefold():
            continue
        if not ENDING.search(paragraph) or NON_PROSE.search(paragraph):
            return "", [], "opening_not_complete_prose"
        if not paragraphs and DEPENDENT_OPENING.match(paragraph):
            return "", [], "dependent_opening"
        words = WORDS.findall(paragraph)
        if len(words) < 8:
            return "", [], "too_few_words"
        if sum(word.casefold() in GERMAN_MARKERS for word in words) < 2:
            return "", [], "language_heuristic"
        proposed = "\n\n".join(paragraphs + [paragraph])
        if len(proposed) > MAX_PARAGRAPH_CHARS:
            return "", [], "opening_too_long"
        paragraphs.append(paragraph)
        offsets.append([start, end])
        if len(proposed) >= MIN_PARAGRAPH_CHARS:
            return proposed, offsets, None
        if len(paragraphs) >= 3:
            break
    return "", [], "opening_too_short"


def article_record(article: dict, spec: dict) -> tuple[dict | None, str | None]:
    article_id, title, url = (str(article.get(key, "")).strip() for key in ("id", "title", "url"))
    text = str(article.get("text", ""))
    if not article_id or not title or not text or not url.startswith("https://de.wikipedia.org/wiki/"):
        return None, "invalid_source_fields"
    if title.startswith(("Liste ", "Liste der ", "Liste von ")) or "Begriffsklärung" in title:
        return None, "list_or_disambiguation_title"
    paragraph, offsets, reason = opening_paragraphs(title, text)
    if reason:
        return None, reason
    title = normalized(title)
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "id": f"wikipedia-de-20231101-{article_id}-lead", "prompt": "",
        "text": title + "\n\n" + paragraph,
        "source": f"Wikimedia Wikipedia {CONFIGURATION} | article_id={article_id} | {title} | {url}",
        "title": title,
        "provenance": {
            "dataset": DATASET, "configuration": CONFIGURATION, "dataset_revision": REVISION,
            "snapshot_date": "2023-11-01", "article_id": article_id, "article_title": title,
            "article_url": url, "article_revision_id": None,
            "revision_note": "The publisher exports a dated dataset revision but no individual article revision ID",
            "article_sha256": source_hash, "raw_file": spec["name"], "raw_file_sha256": spec["sha256"],
            "paragraph_spans": offsets, "paragraph_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
            "text_sha256": hashlib.sha256((title + "\n\n" + paragraph).encode("utf-8")).hexdigest(),
            "transform": "Original complete contiguous opening paragraphs; whitespace and Unicode NFC normalized; original title prepended",
            "license_from_dataset_card": ["CC-BY-SA-3.0", "GFDL"],
            "prompt_created": False,
        },
    }
    return record, None


def priority_title(title: str) -> bool:
    return title.casefold() in PRIORITY_TITLES or PRIORITY_PATTERN.search(title) is not None


def sample_rank(article_id: str) -> int:
    return int.from_bytes(hashlib.sha256((SAMPLING_SEED + ":" + article_id).encode()).digest(), "big")


def keep_smallest(heap: list, record: dict, count: int) -> None:
    rank = sample_rank(record["provenance"]["article_id"])
    item = (-rank, record["id"], record)
    if len(heap) < count:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def build(paths: list[tuple[Path, dict]], count: int = 20000) -> tuple[list[dict], dict]:
    import pyarrow.parquet as parquet
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError("count must be an integer between 1 and 100000")
    broad, priority = [], []
    counters, rejected = Counter(), Counter()
    seen_ids, seen_text = set(), set()
    per_shard = {}
    for path, spec in paths:
        reader = parquet.ParquetFile(path)
        if set(reader.schema_arrow.names) != {"id", "url", "title", "text"}:
            raise ValueError(f"Unexpected source schema: {path}")
        per_shard[spec["name"]] = {"rows": reader.metadata.num_rows, "row_groups": reader.metadata.num_row_groups,
                                   "bytes": path.stat().st_size, "sha256": spec["sha256"]}
        for batch in reader.iter_batches(batch_size=128):
            for article in batch.to_pylist():
                counters["source_articles_scanned"] += 1
                article_id = str(article.get("id", ""))
                if article_id in seen_ids:
                    rejected["duplicate_article_id"] += 1
                    continue
                seen_ids.add(article_id)
                record, reason = article_record(article, spec)
                if reason:
                    rejected[reason] += 1
                    continue
                text_hash = record["provenance"]["text_sha256"]
                if text_hash in seen_text:
                    rejected["duplicate_title_and_prose"] += 1
                    continue
                seen_text.add(text_hash)
                counters["eligible_articles"] += 1
                keep_smallest(broad, record, count)
                if priority_title(record["title"]):
                    counters["eligible_priority_articles"] += 1
                    keep_smallest(priority, record, min(count, MAX_PRIORITY_ARTICLES))
    preferred = sorted((item[2] for item in priority), key=lambda item: (sample_rank(item["provenance"]["article_id"]), item["id"]))
    selected_ids = {record["id"] for record in preferred}
    general = sorted((item[2] for item in broad if item[2]["id"] not in selected_ids), key=lambda item: (sample_rank(item["provenance"]["article_id"]), item["id"]))
    records = preferred + general[:count-len(preferred)]
    # Stable source-ID output order is independent of download/batch order.
    records.sort(key=lambda item: int(item["provenance"]["article_id"]))
    counters["selected_articles"] = len(records)
    counters["selected_priority_articles"] = len(preferred)
    counters["selected_text_characters"] = sum(len(record["text"]) for record in records)
    counters["selected_text_utf8_bytes"] = sum(len(record["text"].encode("utf-8")) for record in records)
    return records, {"counts": dict(counters), "rejections": dict(sorted(rejected.items())), "shards": per_shard,
                     "priority_titles_selected": [record["title"] for record in preferred]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    specs = source_specs()
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(lambda spec: download(spec, args.offline), specs))
    if args.download_only:
        print(json.dumps({"files": [str(path) for path in downloaded], "downloaded_bytes": sum(path.stat().st_size for path in downloaded)}))
        return
    rows, statistics = build(list(zip(downloaded, specs)), args.count)
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = IMPORT_DIR / f"information_wikipedia_{args.count}.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "dataset": DATASET, "configuration": CONFIGURATION, "revision": REVISION,
        "source_files": specs, "requested_records": args.count, "statistics": statistics,
        "selection": {"seed": SAMPLING_SEED, "method": "Smallest SHA-256 article-ID hashes among eligible complete leads, plus a bounded predefined physics/frequency title stratum",
                      "priority_titles": sorted(PRIORITY_TITLES), "priority_pattern": PRIORITY_PATTERN.pattern,
                      "maximum_priority_articles": MAX_PRIORITY_ARTICLES,
                      "paragraph_characters": [MIN_PARAGRAPH_CHARS, MAX_PARAGRAPH_CHARS]},
        "output": {"path": str(output.relative_to(ROOT)).replace("\\", "/"), "sha256": file_hash(output), "bytes": output.stat().st_size, "records": len(rows)},
        "script_sha256": file_hash(Path(__file__)), "elapsed_seconds": time.perf_counter()-started,
        "all_prompts_empty": all(row["prompt"] == "" for row in rows), "sqlite_changed": False,
        "evaluation_files_read": [], "generated_questions": 0,
        "limitation": "Dated Wikipedia text from four source shards, not a representative sample of all German Wikipedia and not independently fact-checked as current",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": manifest["output"], "statistics": statistics, "seconds": manifest["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
