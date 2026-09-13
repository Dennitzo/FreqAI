"""Curated natural-science extension for the pinned German Wikipedia archive.

The productive information corpus v3 was drawn by hash rank from four pinned
Parquet shards. That draw contains almost no well-known science articles or
scientists: a scan of the 20.000 active leads finds no article titled
"Albert Einstein", no "Quantenmechanik" and no "Relativitätstheorie".

This tool scans the same pinned shards for a curated title list of natural
science topics and scientists, reuses the pinned extraction recipe from
``experiments/import_information_corpus.py`` and writes an extension
candidate. Every record keeps the existing contract: empty prompt, title plus
complete contiguous opening prose, full provenance. The candidate provably
shares no article id and no prose with the productive corpus.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import import_information_corpus as wiki  # noqa: E402  pinned extraction recipe

TITLE_FILE = ROOT / "experiments/science_extension_titles.json"
PRODUCTIVE_CORPUS = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
RELAXED_LEAD_PARAGRAPHS = 5
STRATUM = "curated-science-extension-v1"
API_URL = "https://de.wikipedia.org/w/api.php"
USER_AGENT = "FreqAI-science-extension/1.0 (local corpus curation script)"
API_MIN_PARAGRAPH_CHARS = 100
EMPTY_BRACKET_ARTIFACT = re.compile(r"\(\s*(?:[,;]\s*)*\)|\[\s*\]")
SENTENCE_END = re.compile(r"(?<=[^\W\d_])[.!?][\"\u00bb\u201c\u201d\u2019)\]]*(?=\s+[A-Z\u00c4\u00d6\u00dc\u1e9e]|\s*$)")


def normalize_title(title: str) -> str:
    """Exact-case title key: NFC plus whitespace collapse, no case folding.

    A case fold would let the shard article "EVolution" (a bus brand) answer
    the curated title "Evolution" (biology), so the comparison is case
    sensitive.  NFC folding is kept so composed and decomposed accents agree.
    """
    text = unicodedata.normalize("NFC", str(title)).replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip()


def relaxed_opening_paragraphs(title: str, text: str):
    """Same guards as the pinned recipe, but a long lead keeps its largest
    contiguous set of complete opening paragraphs instead of failing."""
    paragraphs, offsets = [], []
    for start, end in wiki.paragraph_spans(text):
        paragraph = wiki.normalized(text[start:end])
        if not paragraphs and paragraph.casefold() == title.casefold():
            continue
        if "\n" in paragraph:
            return "", [], "opening_embedded_linebreak"
        if wiki.EMPTY_EXPORTED_FIELD.search(paragraph):
            return "", [], "empty_exported_field"
        if not wiki.ENDING.search(paragraph) or wiki.NON_PROSE.search(paragraph):
            return "", [], "opening_not_complete_prose"
        if not paragraphs and wiki.DEPENDENT_OPENING.match(paragraph):
            return "", [], "dependent_opening"
        words = wiki.WORDS.findall(paragraph)
        if len(words) < 8:
            return "", [], "too_few_words"
        if sum(word.casefold() in wiki.GERMAN_MARKERS for word in words) < 2:
            return "", [], "language_heuristic"
        proposed = "\n\n".join(paragraphs + [paragraph])
        if len(proposed) > wiki.MAX_PARAGRAPH_CHARS:
            kept = "\n\n".join(paragraphs)
            if len(kept) >= wiki.MIN_PARAGRAPH_CHARS:
                return kept, offsets, None
            return "", [], "opening_too_long"
        paragraphs.append(paragraph)
        offsets.append([start, end])
        if len(proposed) >= wiki.MIN_PARAGRAPH_CHARS:
            return proposed, offsets, None
        if len(paragraphs) >= RELAXED_LEAD_PARAGRAPHS:
            break
    return "", [], "opening_too_short"


def scan_row_group(path: Path, spec: dict, group: int, wanted: dict):
    """Curated-title articles of one Parquet row group, with raw article dicts."""
    import pyarrow.parquet as parquet
    table = parquet.ParquetFile(path).read_row_group(group)
    hits = []
    for article in table.to_pylist():
        title = normalize_title(str(article.get("title", "")))
        if title in wanted:
            hits.append((wanted[title], article, spec))
    return hits


def productive_corpus_guard(path: Path):
    """Article ids and prose hashes already present in the productive corpus."""
    ids, hashes = set(), set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.add(record["id"])
            hashes.add(hashlib.sha256(record["text"].encode("utf-8")).hexdigest())
    return ids, hashes


def build_record(article: dict, spec: dict, domain: str, *, relaxed: bool):
    article_id, title, url = (str(article.get(key, "")).strip() for key in ("id", "title", "url"))
    text = str(article.get("text", ""))
    if not article_id or not title or not text or not url.startswith("https://de.wikipedia.org/wiki/"):
        return None, "invalid_source_fields", None
    if title.startswith(("Liste ", "Liste der ", "Liste von ")) or "Begriffsklärung" in title:
        return None, "list_or_disambiguation_title", None
    if relaxed:
        paragraph, offsets, reason = relaxed_opening_paragraphs(title, text)
    else:
        paragraph, offsets, reason = wiki.opening_paragraphs(title, text)
    if reason:
        return None, reason, None
    title = wiki.normalized(title)
    transform = ("Original complete contiguous opening paragraphs; whitespace and Unicode NFC normalized; "
                 "empty template preposition-plus-semicolon prefixes deleted; original title prepended")
    if relaxed:
        transform += "; curated extension kept the largest contiguous complete lead within the length limit"
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "id": f"wikipedia-de-20231101-{article_id}-lead", "prompt": "",
        "text": title + "\n\n" + paragraph,
        "source": f"Wikimedia Wikipedia {wiki.CONFIGURATION} | article_id={article_id} | {title} | {url}",
        "title": title,
        "provenance": {
            "dataset": wiki.DATASET, "configuration": wiki.CONFIGURATION, "dataset_revision": wiki.REVISION,
            "snapshot_date": "2023-11-01", "article_id": article_id, "article_title": title,
            "article_url": url, "article_revision_id": None,
            "revision_note": "The publisher exports a dated dataset revision but no individual article revision ID",
            "article_sha256": source_hash, "raw_file": spec["name"], "raw_file_sha256": spec["sha256"],
            "paragraph_spans": offsets,
            "paragraph_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
            "text_sha256": hashlib.sha256((title + "\n\n" + paragraph).encode("utf-8")).hexdigest(),
            "transform": transform,
            "removed_empty_template_fragments": [match.group()[1:] for start, end in offsets
                                                 for match in wiki.EMPTY_TEMPLATE_PREFIX.finditer(text[start:end])],
            "license_from_dataset_card": ["CC-BY-SA-3.0", "GFDL"],
            "prompt_created": False,
            "selection_stratum": STRATUM,
            "curated_domain": domain,
            "curated_title": wiki.normalized(title),
        },
    }
    return record, None, "relaxed" if relaxed else "strict"


def api_page(title: str) -> tuple[str | None, dict | None]:
    """Fetch one lead extract from the German Wikipedia MediaWiki API."""
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "redirects": 1,
        "prop": "extracts", "explaintext": 1, "exintro": 1, "exsectionformat": "plain",
        "titles": title,
    })
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8"))
    redirects = payload.get("query", {}).get("redirects", []) or []
    pages = payload.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if int(page_id) < 0:
            return None, {"reason": "no_such_page"}
        page["_redirects"] = redirects
        return str(page_id), page
    return None, {"reason": "empty_api_answer"}


def api_normalized(text: str) -> str:
    """Whitespace-normalize one API block and fold its internal line breaks."""
    return wiki.normalized(re.sub(r"\s*\n\s*", " ", text))


def api_cleaned(text: str) -> tuple[str, list[str]]:
    """Delete empty bracket artifacts like `()` or `[]` from one API block.

    The export leaves such artifacts behind in many German article leads.
    Removing them restores readable prose; every removed fragment is reported
    so the provenance lists it like the pinned recipe does.
    """
    fragments = EMPTY_BRACKET_ARTIFACT.findall(text)
    text = EMPTY_BRACKET_ARTIFACT.sub(" ", text)
    text = re.sub(r"\s+([,;:.])", r"\1", text)
    return wiki.normalized(text), fragments


def clip_to_sentences(text: str, limit: int) -> str:
    """Keep whole sentences only, ending at the last sentence boundary.

    Sentence boundaries require a letter before the terminator, so dates like
    `14. April` do not produce a false boundary.  An empty result means no
    complete sentence fits into the window.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    ends = [match.end() for match in SENTENCE_END.finditer(window)]
    if not ends:
        return ""
    return window[:ends[-1]].strip()


def api_paragraphs(title: str, extract: str):
    """Complete contiguous opening paragraphs of one API lead extract.

    The same prose guards as the pinned recipe apply.  Differences are
    documented in the provenance transform: empty bracket artifacts are
    deleted instead of rejecting the lead, a paragraph over the length limit
    is clipped after its last complete sentence, and the minimum is the
    slightly lower API_MIN_PARAGRAPH_CHARS because the API reports the whole
    lead as one run.
    """
    paragraphs, offsets, removed = [], [], []
    bounds, position = [], 0
    for separator in re.finditer(r"\n[^\n]*\n(?:[ \t]*\n)*", extract):
        start = separator.start()
        bounds.append((position, start))
        position = start + len(separator.group())
    bounds.append((position, len(extract)))
    for start, end in bounds:
        paragraph, fragments = api_cleaned(api_normalized(extract[start:end]))
        removed.extend(fragments)
        if not paragraph:
            continue
        if paragraph.casefold() == title.casefold():
            continue
        if not wiki.ENDING.search(paragraph) or wiki.NON_PROSE.search(paragraph):
            return "", [], "opening_not_complete_prose", removed
        if not paragraphs and wiki.DEPENDENT_OPENING.match(paragraph):
            return "", [], "dependent_opening", removed
        words = wiki.WORDS.findall(paragraph)
        if len(words) < 8:
            return "", [], "too_few_words", removed
        if sum(word.casefold() in wiki.GERMAN_MARKERS for word in words) < 2:
            return "", [], "language_heuristic", removed
        if len(paragraph) > wiki.MAX_PARAGRAPH_CHARS:
            paragraph = clip_to_sentences(paragraph, wiki.MAX_PARAGRAPH_CHARS)
            if not paragraph or len(paragraph) < API_MIN_PARAGRAPH_CHARS:
                return "", [], "opening_too_long", removed
        proposed = "\n\n".join(paragraphs + [paragraph])
        if len(proposed) > wiki.MAX_PARAGRAPH_CHARS:
            kept = "\n\n".join(paragraphs)
            if len(kept) >= API_MIN_PARAGRAPH_CHARS:
                return kept, offsets, None, removed
            return "", [], "opening_too_long", removed
        paragraphs.append(paragraph)
        offsets.append([start, end])
        if len(proposed) >= API_MIN_PARAGRAPH_CHARS:
            return proposed, offsets, None, removed
        if len(paragraphs) >= RELAXED_LEAD_PARAGRAPHS:
            break
    return "", [], "opening_too_short", removed


def build_api_record(curated_title: str, domain: str, page_id: str, page: dict):
    """Build one record from an API page, keeping the exact contract."""
    article_title = str(page.get("title", "")).strip()
    url = str(page.get("fullurl", "")).strip()
    extract = str(page.get("extract", ""))
    if not article_title or not extract:
        return None, "empty_api_page"
    if "Begriffsklärung" in article_title or article_title.startswith(("Liste ", "Liste der ", "Liste von ")):
        return None, "list_or_disambiguation_title"
    paragraph, offsets, reason, removed = api_paragraphs(article_title, extract)
    if reason:
        return None, reason
    title = wiki.normalized(article_title)
    source_hash = hashlib.sha256(extract.encode("utf-8"))
    record = {
        "id": f"wikipedia-api-{page_id}-lead", "prompt": "",
        "text": title + "\n\n" + paragraph,
        "source": f"German Wikipedia live MediaWiki API | pageid={page_id} | {title} | {url}",
        "title": title,
        "provenance": {
            "dataset": "de.wikipedia.org", "configuration": "mediawiki-api-intro",
            "dataset_revision": None, "snapshot_date": None,
            "article_id": page_id, "article_title": title, "article_url": url,
            "article_revision_id": str(page.get("touched", "")) or None,
            "revision_note": "Live MediaWiki API lead fetched after the pinned dataset was checked; "
                             "the API reports the page touch timestamp, not an editorial revision id",
            "article_sha256": hashlib.sha256(extract.encode("utf-8")).hexdigest(),
            "raw_file": "mediawiki-api", "raw_file_sha256": None,
            "paragraph_spans": offsets,
            "paragraph_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
            "text_sha256": hashlib.sha256((title + "\n\n" + paragraph).encode("utf-8")).hexdigest(),
            "transform": "Complete contiguous API lead paragraphs; internal line breaks folded to spaces; "
                         "whitespace and Unicode NFC normalized; empty bracket artifacts deleted; "
                         "paragraphs over the length limit clipped after their last complete sentence; "
                         "original title prepended",
            "removed_empty_template_fragments": removed,
            "license_from_dataset_card": ["CC-BY-SA-4.0"],
            "prompt_created": False,
            "selection_stratum": STRATUM,
            "curated_domain": domain,
            "curated_title": wiki.normalized(curated_title),
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    redirects = list(page.get("_redirects", []) or [])
    if normalize_title(article_title) != normalize_title(curated_title):
        redirect_targets = {normalize_title(str(entry.get("to", "")))
                            for entry in redirects if isinstance(entry, dict)}
        if normalize_title(article_title) not in redirect_targets:
            return None, "title_mismatch"
    if redirects:
        record["provenance"]["redirected_from"] = [str(entry.get("from", ""))
                                                   for entry in redirects if isinstance(entry, dict)]
    return record, None


def fetch_api(title: str):
    """Fetch one API page with bounded retries; one failure never kills the run.

    The German Wikipedia API answers HTTP 429 when an unauthenticated client
    requests too quickly, so every 429 gets a longer exponential wait before
    the next attempt.
    """
    last = "empty_api_answer"
    for attempt in range(5):
        try:
            page_id, page = api_page(title)
            if page_id is not None:
                return page_id, page
            last = str(page.get("reason", "no_such_page"))
            if last == "no_such_page":
                break
        except urllib.error.HTTPError as error:
            last = f"api_error:HTTP{error.code}"
            time.sleep(10.0 * (2 ** attempt) if error.code == 429 else 2.0 * (attempt + 1))
        except Exception as error:  # noqa: BLE001  network resilience for many error types
            last = f"api_error:{type(error).__name__}"
            time.sleep(2.0 * (attempt + 1))
    return None, {"reason": last}


def collect(wanted: dict, workers: int):
    specs = wiki.source_specs()
    tasks = []
    for spec in specs:
        path = wiki.SOURCE_DIR / spec["name"]
        if not path.exists():
            raise FileNotFoundError(f"Pinned source missing: {path}")
        meta = __import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(path).metadata
        tasks.extend((path, spec, group) for group in range(meta.num_row_groups))
    hits = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for found in pool.map(lambda task: scan_row_group(*task, wanted=wanted), tasks):
            hits.extend(found)
    print(f"curated hits in {len(tasks)} row groups: {len(hits)}, seconds {time.perf_counter()-started:.1f}",
          flush=True)
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titles", default=str(TITLE_FILE))
    parser.add_argument("--output", default=str(ROOT / "memory/imports/information_science_extension_v1.jsonl"))
    parser.add_argument("--provenance", default=str(ROOT / "memory/imports/information_science_extension_v1.provenance.json"))
    parser.add_argument("--productive", default=str(PRODUCTIVE_CORPUS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--offline", action="store_true",
                        help="never query the MediaWiki API for titles the shards lack")
    parser.add_argument("--api-delay", type=float, default=1.0,
                        help="seconds to wait between consecutive MediaWiki API requests")
    parser.add_argument("--report-only", action="store_true", help="scan and report, write no candidate")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    wanted_raw = json.loads(Path(args.titles).read_text(encoding="utf-8"))
    wanted = {normalize_title(title): domain for title, domain in wanted_raw.items()}
    hits = collect(wanted, args.workers)

    productive_ids, productive_hashes = productive_corpus_guard(Path(args.productive))
    seen_ids, seen_hashes, records, outcomes = set(), set(), [], {}
    rejections = {}
    for domain, article, spec in hits:
        title = str(article.get("title", "")).strip()
        key = normalize_title(title)
        record, reason, mode = build_record(article, spec, domain, relaxed=False)
        if record is None and reason == "opening_too_long":
            record, reason, mode = build_record(article, spec, domain, relaxed=True)
        if record is None:
            rejections[key] = reason or "unknown"
            outcomes[key] = {"status": "rejected", "reason": reason, "title": title}
            continue
        if record["id"] in productive_ids or record["provenance"]["text_sha256"] in productive_hashes:
            outcomes[key] = {"status": "already_productive", "title": title}
            continue
        if record["id"] in seen_ids or record["provenance"]["text_sha256"] in seen_hashes:
            rejections[key] = "duplicate_within_extension"
            outcomes[key] = {"status": "rejected", "reason": "duplicate_within_extension", "title": title}
            continue
        seen_ids.add(record["id"])
        seen_hashes.add(record["provenance"]["text_sha256"])
        rejections.pop(key, None)
        records.append(record)
        outcomes[key] = {"status": "selected", "mode": mode, "title": title,
                        "article_id": record["provenance"]["article_id"]}
        outcomes[key]["characters"] = len(record["text"])

    shard_titles_found = len(outcomes)
    pending = []
    for title, key in ((t, normalize_title(t)) for t in wanted_raw):
        outcome = outcomes.get(key)
        if outcome is None:
            pending.append((title, key, None))
        elif outcome["status"] == "rejected":
            pending.append((title, key, outcome["reason"]))
    if pending and not args.offline:
        fetched = []
        for position, item in enumerate(pending):
            if position:
                time.sleep(args.api_delay)
            fetched.append(fetch_api(item[0]))
        for (title, key, shard_reason), (page_id, page) in zip(pending, fetched):
            record, api_reason = (build_api_record(title, wanted_raw[title], page_id, page)
                                  if page_id is not None else (None, str(page.get("reason"))))
            if record is None:
                outcomes[key] = {"status": "rejected", "title": title, "shard_reason": shard_reason,
                                 "api_reason": api_reason}
                rejections[key] = f"shard:{shard_reason or 'missing'} api:{api_reason}"
                continue
            if record["id"] in productive_ids or record["provenance"]["text_sha256"] in productive_hashes:
                rejections.pop(key, None)
                outcomes[key] = {"status": "already_productive", "title": title, "source": "api"}
                continue
            if record["id"] in seen_ids or record["provenance"]["text_sha256"] in seen_hashes:
                outcomes[key] = {"status": "rejected", "title": title, "reason": "duplicate_within_extension"}
                rejections[key] = "duplicate_within_extension"
                continue
            seen_ids.add(record["id"])
            seen_hashes.add(record["provenance"]["text_sha256"])
            rejections.pop(key, None)
            records.append(record)
            outcomes[key] = {"status": "selected", "mode": "api", "title": record["title"],
                             "article_id": record["provenance"]["article_id"],
                             "characters": len(record["text"])}

    missing = sorted(title for title, key in ((t, normalize_title(t)) for t in wanted_raw)
                     if key not in outcomes)
    summary = {
        "curated_titles": len(wanted_raw),
        "titles_found_in_pinned_shards": shard_titles_found,
        "titles_selected": len(records),
        "titles_selected_via_shards": sum(1 for value in outcomes.values()
                                          if value["status"] == "selected" and value.get("mode") != "api"),
        "titles_selected_via_api": sum(1 for value in outcomes.values() if value.get("mode") == "api"),
        "titles_already_productive": sum(1 for value in outcomes.values()
                                         if value["status"] == "already_productive"),
        "titles_missing_from_shards": len(missing),
        "strict_mode_selected": sum(1 for value in outcomes.values()
                                    if value.get("mode") == "strict"),
        "relaxed_mode_selected": sum(1 for value in outcomes.values() if value.get("mode") == "relaxed"),
        "selection_by_domain": {},
        "rejections": dict(sorted(rejections.items())),
    }
    for record in records:
        domain = record["provenance"]["curated_domain"]
        summary["selection_by_domain"][domain] = summary["selection_by_domain"].get(domain, 0) + 1

    if args.report_only:
        print(json.dumps({"summary": summary, "missing_titles": missing,
                          "selected_titles": [record["title"] for record in records]},
                         ensure_ascii=False, indent=1)[:14000])
        return

    records.sort(key=lambda item: (item["provenance"]["dataset"] != wiki.DATASET,
                                   int(str(item["provenance"]["article_id"]))))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    titles_path = Path(args.titles)
    manifest = {
        "schema_version": 1, "stratum": STRATUM, "purpose":
            "Targeted natural-science and scientist extension of the productive information corpus",
        "dataset": wiki.DATASET, "configuration": wiki.CONFIGURATION, "revision": wiki.REVISION,
        "title_file": str(titles_path.relative_to(ROOT)), "title_file_sha256":
            hashlib.file_digest(titles_path.open("rb"), "sha256").hexdigest(),
        "script_sha256": hashlib.file_digest(Path(__file__).resolve().open("rb"), "sha256").hexdigest(),
        "productive_corpus": {"path": str(Path(args.productive).relative_to(ROOT)),
                              "records": len(productive_ids),
                              "sha256": hashlib.file_digest(Path(args.productive).open("rb"), "sha256").hexdigest()},
        "summary": summary, "missing_titles": missing,
        "selected_titles": [{"title": record["title"], "domain": record["provenance"]["curated_domain"],
                             "article_id": record["provenance"]["article_id"]} for record in records],
        "output": {"path": str(output.relative_to(ROOT)), "records": len(records),
                   "sha256": digest, "bytes": output.stat().st_size,
                   "characters": sum(len(record["text"]) for record in records)},
        "all_prompts_empty": all(not record["prompt"].strip() for record in records),
        "limitation": "Curated natural-science titles from the dated four-shard archive; titles the archive "
                      "lacks were fetched live from the German Wikipedia MediaWiki API. Not a complete survey "
                      "of German Wikipedia science coverage and not independently fact-checked as current",
    }
    Path(args.provenance).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps({"output": manifest["output"], "summary": summary,
                      "missing_titles": len(missing)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
