"""Immutable, separate development/holdout comparison for imported public data.

This evaluator does not write to SQLite and never uses evaluation prompts as
corpus data. A lightweight document holder calls the same generation runtime as
the server while excluding the unrelated legacy retrieval index from timings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/public_corpus/evaluation"
SUITE = ROOT / "memory/evaluation/public_corpus_queries.json"
BASE_CODE = OUTPUT / "baseline_code"
BASE_CORPUS = OUTPUT / "baseline_documents.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def code_manifest(code_root):
    files = sorted((code_root / "freqai").glob("*.py"))
    files += [code_root / "memory/language/generative_categories.json"]
    return {str(path.relative_to(code_root)): sha(path) for path in files if path.exists()}


def initialize():
    target = OUTPUT / "baseline_freeze.json"
    if target.exists():
        return read(target)
    with sqlite3.connect(f"file:{(ROOT / 'memory/memory.sqlite3').as_posix()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        records = [dict(row) for row in connection.execute("SELECT id,text,source,prompt FROM documents ORDER BY sequence")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BASE_CORPUS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    scenarios = read(SUITE)["scenarios"]
    assert Counter(s["split"] for s in scenarios for _ in s["turns"]) == {"development": 20, "holdout": 20}
    prompts = [normalize(t["prompt"]) for s in scenarios for t in s["turns"]]
    existing = {normalize(row[key]) for row in records for key in ("text", "prompt")}
    assert len(set(prompts)) == 40 and not set(prompts) & existing
    old_prompts = set()
    for path in (ROOT / "memory/evaluation").glob("*.json"):
        if path == SUITE:
            continue
        def walk(value):
            if isinstance(value, dict):
                if isinstance(value.get("prompt"), str):
                    old_prompts.add(normalize(value["prompt"]))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(read(path))
    assert not set(prompts) & old_prompts, "Fresh prompts overlap existing evaluation"
    manifest = {"created_utc": timestamp(), "code": code_manifest(BASE_CODE), "baseline_documents": len(records),
                "baseline_corpus_sha256": sha(BASE_CORPUS), "suite_sha256": sha(SUITE),
                "holdout_unseen_by_implementer": True, "review_type": "independent AI-agent qualitative review; not a human study"}
    write(target, manifest)
    return manifest


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return {}
        return {"rss_mb": counters.WorkingSetSize / 2**20, "peak_rss_mb": counters.PeakWorkingSetSize / 2**20,
                "private_commit_mb": counters.PagefileUsage / 2**20, "peak_private_commit_mb": counters.PeakPagefileUsage / 2**20}
    except AttributeError:
        return {}


def corpus_records(args):
    records = lines(BASE_CORPUS)
    external = lines(args.corpus) if args.corpus else []
    if args.limit is not None:
        external = external[:args.limit]
    ids = {row["id"] for row in records}
    for row in external:
        assert row["id"] not in ids, "Duplicate document id"
        ids.add(row["id"])
    return records + external


def automatic(answer, corpus_set, sentence_set):
    words = re.findall(r"\w+", answer)
    triples = Counter(tuple(words[i:i+3]) for i in range(max(0, len(words)-2)))
    errors = []
    if not answer.strip():
        errors.append("empty")
    if answer and not re.search(r"[.!?][\s\"'»]*$", answer):
        errors.append("unfinished_punctuation")
    if any(count >= 3 for count in triples.values()):
        errors.append("repeated_trigram_loop")
    if "\ufffd" in answer:
        errors.append("invalid_unicode")
    sentences = [normalize(s) for s in re.split(r"(?<=[.!?])\s+", answer) if normalize(s)]
    return {"surface_form_ok": not errors, "errors": errors,
            "new_complete_text": bool(answer) and normalize(answer) not in corpus_set,
            "new_sentences": sum(s not in sentence_set for s in sentences),
            "not_a_semantic_quality_judgment": True}


class EvaluationMemory:
    def __init__(self, documents):
        self.documents = documents


def load_engine(code_root, records):
    sys.path.insert(0, str(code_root))
    runtime = importlib.import_module("freqai.generation_runtime")
    from freqai.memory import Document
    memory = EvaluationMemory([Document(**{key: row.get(key, "") for key in ("id", "text", "source", "prompt")}) for row in records])
    started = time.perf_counter()
    model = runtime.generator_for(memory).model
    build_s = time.perf_counter() - started
    return memory, runtime, model, build_s


def numerical(model, rows):
    import numpy as np
    statistics = {"states": 0, "fft_direct_max_probability_error": 0.0, "time_max_probability_error": 0.0,
                  "fft_direct_argmax_changes": 0, "time_argmax_changes": 0, "prompt_null_distribution_changes": 0,
                  "prompt_null_mean_total_variation": 0.0, "finite_normalized_states": 0,
                  "data_zero_rejected_states": 0, "states_with_numerical_error": []}
    states = []
    for row in rows:
        tokens = row["tokens"]
        for prefix in ([], tokens[:min(3, len(tokens))], tokens[:min(7, len(tokens))]):
            states.append((row, prefix))
    for row, prefix in states:
        field, _ = model.prompt_field(row["prompt"], context=row["context_before"])
        try:
            actual, _ = model.next_distribution(prefix, field, method="operator", time_s=13.125)
            direct, _ = model.next_distribution(prefix, field, method="direct", time_s=13.125)
            later, _ = model.next_distribution(prefix, field, method="operator", time_s=1000000.875)
            null, _ = model.next_distribution(prefix, field * 0, method="operator", time_s=13.125)
        except Exception as exc:
            statistics["states_with_numerical_error"].append({"id": row["id"], "prefix": prefix, "error": str(exc)})
            continue
        statistics["states"] += 1
        statistics["fft_direct_max_probability_error"] = max(statistics["fft_direct_max_probability_error"], float(np.max(np.abs(actual-direct))))
        statistics["time_max_probability_error"] = max(statistics["time_max_probability_error"], float(np.max(np.abs(actual-later))))
        statistics["fft_direct_argmax_changes"] += int(actual.argmax() != direct.argmax())
        statistics["time_argmax_changes"] += int(actual.argmax() != later.argmax())
        difference = float(np.abs(actual-null).sum() / 2)
        statistics["prompt_null_distribution_changes"] += int(difference > 1e-9)
        statistics["prompt_null_mean_total_variation"] += difference
        statistics["finite_normalized_states"] += int(np.isfinite(actual).all() and actual.min() >= 0 and abs(actual.sum()-1) < 1e-10)
    class ZeroFields(Mapping):
        def __init__(self, original):
            self.original = original
        def __len__(self):
            return len(self.original)
        def __iter__(self):
            return iter(self.original)
        def __getitem__(self, key):
            field = self.original[key]
            if hasattr(field, "local_spectrum"):
                zero = field.copy()
                zero[:] = 0
                return zero
            return np.zeros_like(field)
    originals = {name: getattr(model, name) for name in ("transition_spectra", "conditioned_spectra")}
    try:
        for name, value in originals.items():
            setattr(model, name, ZeroFields(value))
        for row, prefix in states[:6]:
            field, _ = model.prompt_field(row["prompt"], context=row["context_before"])
            try:
                model.next_distribution(prefix, field, method="operator")
            except ValueError:
                statistics["data_zero_rejected_states"] += 1
    finally:
        for name, value in originals.items():
            setattr(model, name, value)
    statistics["data_zero_tested_states"] = min(6, len(states))
    statistics["prompt_null_mean_total_variation"] /= max(1, statistics["states"])
    statistics["scope"] = "Decoder algebra and causal dependence, not a semantic correctness proof. Null removes both prompt amplitude and feature spectrum. Data null retains real reference counts and supports."
    return statistics


def configuration_manifest(args):
    records = corpus_records(args)
    return {"code_root": str(Path(args.code_root).resolve()), "code": code_manifest(Path(args.code_root)),
            "baseline_corpus_sha256": sha(BASE_CORPUS), "suite_sha256": sha(SUITE),
            "additional_corpus": str(Path(args.corpus).resolve()) if args.corpus else None,
            "additional_corpus_sha256": sha(args.corpus) if args.corpus else None,
            "additional_limit": args.limit, "document_count": len(records),
            "runner_sha256": sha(Path(__file__)), "max_tokens": 40, "seed": 17}


def run(args):
    frozen = initialize()
    assert sha(SUITE) == frozen["suite_sha256"] and sha(BASE_CORPUS) == frozen["baseline_corpus_sha256"]
    manifest = configuration_manifest(args)
    if Path(args.code_root).resolve() == BASE_CODE.resolve():
        assert manifest["code"] == frozen["code"], "Baseline code changed"
    if args.split == "holdout":
        candidate_freeze = read(OUTPUT / f"{args.label}_freeze.json")
        assert candidate_freeze["manifest"] == manifest, "Code, corpus, runner or configuration changed after holdout freeze"
    destination = OUTPUT / f"{args.label}_{args.split}.json"
    assert not destination.exists(), "Recorded evaluation is immutable; use a new development label"
    records = corpus_records(args)
    corpus_set = {normalize(row["text"]) for row in records}
    sentence_set = {normalize(s) for row in records for s in re.split(r"(?<=[.!?])\s+", row["text"]) if normalize(s)}
    prompts = {normalize(turn["prompt"]) for scenario in read(SUITE)["scenarios"] for turn in scenario["turns"]}
    assert not prompts & {normalize(row.get("prompt", "")) for row in records}, "Evaluation leakage into corpus"
    print(json.dumps({"status": "compiling", "documents": len(records), "label": args.label}), flush=True)
    memory, runtime, model, build_s = load_engine(Path(args.code_root), records)
    after_build = process_memory()
    print(json.dumps({"status": "compiled", "vocabulary": model.size, "build_s": build_s, **after_build}), flush=True)
    rows = []
    for scenario in read(SUITE)["scenarios"]:
        if scenario["split"] != args.split:
            continue
        context = None
        for turn in scenario["turns"]:
            before = copy.deepcopy(context)
            started = time.perf_counter()
            result, context = runtime.respond_wave(memory, turn["prompt"], context=context, time_s=13.125, max_tokens=40, seed=17)
            row = {**turn, "scenario": scenario["id"], "answer": result["answer"], "tokens": result["tokens"],
                   "elapsed_s": time.perf_counter()-started, "context_before": before,
                   "automatic": automatic(result["answer"], corpus_set, sentence_set),
                   "decoder": {key: result["decoder"][key] for key in ("method", "max_transform_error", "vocabulary_size")},
                   "truncated": result["truncated"], "generation_reason": result.get("generation_reason"), "qualitative_review": None}
            rows.append(row)
            print(json.dumps({"status": "turn", "id": turn["id"], "elapsed_s": row["elapsed_s"]}), flush=True)
    numerics = numerical(model, rows) if not args.skip_numerics else {"skipped": True}
    summary = {"turns": len(rows), "surface_form_ok": sum(row["automatic"]["surface_form_ok"] for row in rows),
               "new_complete_texts": sum(row["automatic"]["new_complete_text"] for row in rows),
               "texts_with_new_sentences": sum(row["automatic"]["new_sentences"] > 0 for row in rows),
               "empty_answers": sum(not row["answer"] for row in rows), "qualitative": "pending independent AI-agent review"}
    value = {"created_utc": timestamp(), "manifest": manifest, "split": args.split, "label": args.label,
             "runtime_configuration": {"model": runtime.MODEL_CONFIGURATION, "generation": runtime.GENERATION_CONFIGURATION},
             "build_s": build_s, "memory_after_build": after_build, "memory_after_evaluation": process_memory(),
             "storage_statistics": model.storage_stats() if hasattr(model, "storage_stats") else None,
             "summary": summary, "numerics": numerics, "rows": rows}
    write(destination, value)
    print(json.dumps({"path": str(destination), "summary": summary, "numerics": numerics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "run", "freeze"])
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--corpus", help="Additional JSONL documents; productive SQLite is never changed")
    parser.add_argument("--limit", type=int, help="Take deterministic first N additional records")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--skip-numerics", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("Label must contain only letters, digits, underscore or dash")
    if args.limit is not None and args.limit < 0:
        parser.error("Limit cannot be negative")
    if args.action == "initialize":
        print(json.dumps(initialize(), ensure_ascii=False))
    elif args.action == "freeze":
        initialize()
        destination = OUTPUT / f"{args.label}_freeze.json"
        manifest = configuration_manifest(args)
        if destination.exists():
            assert read(destination)["manifest"] == manifest
        else:
            write(destination, {"created_utc": timestamp(), "manifest": manifest, "holdout_seen_by_implementer": False})
        print(destination)
    else:
        run(args)


if __name__ == "__main__":
    main()
