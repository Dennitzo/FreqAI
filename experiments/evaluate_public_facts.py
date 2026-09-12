"""Separate official GermanQuAD test questions; not a chat quality measure.

The fixed sample is selected before candidate output is examined. Questions are
never given their reference context during generation; this evaluates closed-
corpus recall, a harder and different task than the dataset's reading task.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import time

import evaluate_public_corpus as common

ROOT = common.ROOT
OUTPUT = common.OUTPUT
TEST = OUTPUT / "germanquad_test-0000.parquet"
TRAIN = ROOT / "memory/sources/deepset-germanquad/train-0000.parquet"
SAMPLE = OUTPUT / "germanquad_test_sample.json"
SOURCE_URL = "https://huggingface.co/datasets/deepset/germanquad/resolve/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet"


def initialize():
    if SAMPLE.exists():
        return common.read(SAMPLE)
    import pyarrow.parquet as pq
    train = pq.read_table(TRAIN).to_pylist()
    test = pq.read_table(TEST).to_pylist()
    train_contexts = {row["context"] for row in train}
    train_titles = {row["context"].splitlines()[0] for row in train}
    sample = random.Random(731).sample(test, 30)
    value = {"created_utc": common.timestamp(), "source_url": SOURCE_URL,
             "test_file_sha256": common.sha(TEST), "train_file_sha256": common.sha(TRAIN),
             "selection": "random.Random(731).sample(official_test_rows, 30), before candidate outputs",
             "test_rows": len(test), "test_exact_context_overlap_with_train": sum(row["context"] in train_contexts for row in test),
             "test_article_title_overlap_with_train": sum(row["context"].splitlines()[0] in train_titles for row in test),
             "scope": "Closed-corpus generation without reference context, not the standard extractive reading benchmark. Article-title overlap is reported; official test contexts are not imported.",
             "rows": [{"id": row["id"], "prompt": row["question"], "answers": row["answers"]["text"],
                       "context": row["context"], "context_sha256": hashlib.sha256(row["context"].encode()).hexdigest(),
                       "title": row["context"].splitlines()[0],
                       "article_title_in_train": row["context"].splitlines()[0] in train_titles,
                       "exact_context_in_train": row["context"] in train_contexts} for row in sample]}
    common.write(SAMPLE, value)
    return value


def token_f1(prediction, reference):
    pred = Counter(common.normalize(prediction).split())
    ref = Counter(common.normalize(reference).split())
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def evidence_status(row, records):
    answers = [common.normalize(a) for a in row["answers"]]
    exact_answer_rows = [r for r in records if any(a and a in common.normalize(r["text"]) for a in answers)]
    # The source importer preserves title in provenance source text. Answer
    # occurrence alone is weak evidence, e.g. a common year or place name.
    same_title = [r for r in exact_answer_rows if common.normalize(row["title"]) in common.normalize(r.get("source", ""))]
    return {"answer_string_occurs_in_corpus": bool(exact_answer_rows),
            "answer_occurs_with_same_article_title": bool(same_title),
            "matching_document_ids": [r["id"] for r in exact_answer_rows[:10]],
            "same_article_document_ids": [r["id"] for r in same_title[:10]],
            "caution": "A matching string is not proof that the corpus contains enough information to answer this question."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "freeze", "run"])
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--corpus")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("Unsafe label")
    dataset = initialize()
    if args.action == "initialize":
        print(json.dumps({key: value for key, value in dataset.items() if key != "rows"}))
        return
    manifest = common.configuration_manifest(args)
    manifest.update({"fact_runner_sha256": common.sha(Path(__file__)), "fact_sample_sha256": common.sha(SAMPLE)})
    freeze = OUTPUT / f"{args.label}_facts_freeze.json"
    if args.action == "freeze":
        if freeze.exists():
            assert common.read(freeze)["manifest"] == manifest
        else:
            common.write(freeze, {"created_utc": common.timestamp(), "manifest": manifest})
        print(freeze)
        return
    assert common.read(freeze)["manifest"] == manifest, "Implementation or data changed after freeze"
    destination = OUTPUT / f"{args.label}_facts.json"
    assert not destination.exists(), "Official test run is immutable"
    records = common.corpus_records(args)
    assert not {common.normalize(r["prompt"]) for r in dataset["rows"]} & {common.normalize(r.get("prompt", "")) for r in records}, "Test question leakage"
    memory, runtime, model, build_s = common.load_engine(Path(args.code_root), records)
    rows = []
    for reference in dataset["rows"]:
        started = time.perf_counter()
        result, _ = runtime.respond_wave(memory, reference["prompt"], context=None, time_s=13.125, max_tokens=40, seed=17)
        answer = result["answer"]
        norm = common.normalize(answer)
        row = {**reference, "answer": answer, "elapsed_s": time.perf_counter()-started,
               "corpus_evidence": evidence_status(reference, records),
               "automatic": {"exact_answer": any(norm == common.normalize(r) for r in reference["answers"]),
                             "contains_answer_string": any(common.normalize(r) in norm for r in reference["answers"]),
                             "max_token_f1": max(token_f1(answer, r) for r in reference["answers"]),
                             "not_a_factual_correctness_judgment": True},
               "qualitative_review": None}
        rows.append(row)
        print(json.dumps({"status": "turn", "id": reference["id"], "elapsed_s": row["elapsed_s"]}), flush=True)
    summary = {"questions": len(rows), "exact_answers": sum(r["automatic"]["exact_answer"] for r in rows),
               "contains_answer_string": sum(r["automatic"]["contains_answer_string"] for r in rows),
               "mean_token_f1": sum(r["automatic"]["max_token_f1"] for r in rows)/len(rows),
               "answer_string_in_corpus": sum(r["corpus_evidence"]["answer_string_occurs_in_corpus"] for r in rows),
               "answer_string_same_article_in_corpus": sum(r["corpus_evidence"]["answer_occurs_with_same_article_title"] for r in rows),
               "factual_correctness": "pending independent AI-agent qualitative review"}
    common.write(destination, {"created_utc": common.timestamp(), "manifest": manifest, "summary": summary,
                               "scope": dataset["scope"], "build_s": build_s, "memory": common.process_memory(), "rows": rows})
    print(json.dumps({"path": str(destination), "summary": summary}))


if __name__ == "__main__":
    main()
