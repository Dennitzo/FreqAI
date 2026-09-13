"""Verify exported corpus coverage and produce a hash-bound cleanup manifest.

This script does not delete files. apply_consolidation_cleanup.ps1 consumes the
manifest only after validating every path, hash and central database revision.
"""
import gzip
import json
from pathlib import Path
import sqlite3

from consolidate_memory import ROOT, file_digest
from freqai.record_contract import validate_information_record


def verify_export(connection, path):
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_information_record(row)
            identifier = row["id"]
            if identifier.startswith("wikipedia-de-20231101-") and identifier.endswith("-lead"):
                identifier = identifier[:-5]
                stored = connection.execute("SELECT metadata_json FROM information_provenance WHERE document_id=?",
                                            (identifier,)).fetchone()
                if not stored:
                    raise ValueError(f"Missing complete source for {path}: {identifier}")
                metadata = json.loads(stored[0])
                if metadata.get("kind") != "complete_wikipedia_article":
                    raise ValueError(identifier)
                expected_hash = row.get("provenance", {}).get("article_sha256")
                if expected_hash and metadata["article_sha256"] != expected_hash:
                    raise ValueError(f"Source content hash mismatch: {identifier}")
            else:
                stored = connection.execute("SELECT text,source FROM documents WHERE id=?", (identifier,)).fetchone()
                if stored != (row["text"], row["source"]):
                    raise ValueError(f"Information not preserved: {path}: {identifier}")
            count += 1
    return count


def main():
    report = json.loads((ROOT / "results/memory_consolidation.json").read_text(encoding="utf-8"))
    assert report["verification"]["articles_verified"] == 569062
    assert report["verification"]["integrity_check"] == "ok"
    files, moves, directories = {}, [], set()

    def remove(path, reason):
        if not path.is_file():
            return
        relative = path.relative_to(ROOT).as_posix()
        assert relative != "memory/memory.sqlite3" and not relative.startswith((".git/", ".venv/"))
        files[relative] = {"path": relative, "bytes": path.stat().st_size,
                           "sha256": file_digest(path), "reason": reason}
        parent = path.parent
        while parent != ROOT and parent.relative_to(ROOT).as_posix() not in {"memory", "results", "runtime", "docs"}:
            directories.add(parent.relative_to(ROOT).as_posix())
            parent = parent.parent

    with sqlite3.connect(ROOT / "memory/memory.sqlite3") as connection:
        revision = connection.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0]
        assert revision == report["after"]["revision"]
        candidates = list((ROOT / "memory/imports").glob("information_*.jsonl"))
        candidates += list((ROOT / "memory/information").glob("*.jsonl"))
        candidates += list((ROOT / "dataset/corpus").glob("*.jsonl.gz"))
        candidates += [p for p in (ROOT / "results").rglob("*.jsonl")
                       if p.name.startswith(("wikipedia_", "conversation_facts", "conversation_lexicon"))]
        coverage = []
        for path in sorted(candidates):
            count = verify_export(connection, path)
            relative = path.relative_to(ROOT).as_posix()
            coverage.append({"file": relative, "records": count, "covered": True})
            remove(path, "information_verified_in_central_memory")
            connection.execute("INSERT OR IGNORE INTO information_imports VALUES (?,?,?,?,?)",
                               (relative, files[relative]["sha256"], path.stat().st_size, count,
                                json.dumps({"kind": "verified_materialization", "covered_by_full_articles": True})))
            print(json.dumps(coverage[-1]), flush=True)

        # Entirely obsolete local data/temporary stores, explicitly requested.
        for relative in ("memory/backups", "memory/.compiler-cache", "unsloth-tmp", "memory/language"):
            folder = ROOT / relative
            directories.add(relative)
            for path in folder.rglob("*"):
                remove(path, "obsolete_backup_cache_or_paired_language_data")
        for path in (ROOT / "memory/imports").glob("*"):
            remove(path, "superseded_import_or_legacy_paired_dataset")
        for path in (ROOT / "memory/information").glob("*"):
            remove(path, "information_and_provenance_preserved_centrally")
        for path in (ROOT / "memory/fixtures").glob("*"):
            remove(path, "legacy_imported_dataset")
        remove(ROOT / "dataset/manifest.json", "old_export_stage_manifest_replaced_by_central_catalog")
        remove(ROOT / "recs.pkl", "empty_temporary_dataset")
        remove(ROOT / "tests/fixtures/benchmark.json", "historically_imported_benchmark_replaced_by_generated_test_records")
        # Keep research notes, source manifests, licenses and import recipes as documentation.
        for path in (ROOT / "memory/sources").rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".parquet", ".gz", ".jsonl", ".npz", ".pyc"}:
                remove(path, "verified_wikipedia_source_or_excluded_qa_download")
            else:
                destination = ROOT / "docs/data_sources" / path.relative_to(ROOT / "memory/sources")
                if destination.exists():
                    raise ValueError(f"Documentation destination already exists: {destination}")
                moves.append({"path": path.relative_to(ROOT).as_posix(),
                              "destination": destination.relative_to(ROOT).as_posix(),
                              "sha256": file_digest(path)})
                parent = path.parent
                while parent != ROOT / "memory":
                    directories.add(parent.relative_to(ROOT).as_posix())
                    parent = parent.parent
        for path in (ROOT / "results").rglob("*.jsonl"):
            if "reasoning" not in path.relative_to(ROOT / "results").parts:
                remove(path, "historical_corpus_copy_or_paired_dataset")
        for folder in (ROOT / "results").rglob("memory"):
            if folder.is_dir():
                for path in folder.rglob("*"):
                    remove(path, "historical_dataset_copy")
        for path in (ROOT / "results").rglob("*.parquet"):
            remove(path, "legacy_qa_evaluation_download")
        for folder in (ROOT / "runtime", ROOT / "results"):
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix in {".sqlite3", ".sqlite", ".db", ".npz", ".bak", ".backup"}:
                    remove(path, "obsolete_database_or_wave_export")
    manifest = {"central_database": "memory/memory.sqlite3", "revision": revision,
                "articles_verified": 569062, "coverage": coverage,
                "files": list(files.values()), "moves": moves,
                "empty_directories": sorted(directories, key=lambda p: (-len(Path(p).parts), p)),
                "delete_bytes": sum(item["bytes"] for item in files.values()),
                "delete_count": len(files)}
    (ROOT / "results/memory_cleanup_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"delete_count": len(files), "delete_bytes": manifest["delete_bytes"], "metadata_moves": len(moves)}))


if __name__ == "__main__":
    main()
