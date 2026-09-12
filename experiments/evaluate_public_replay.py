"""In-sample capacity diagnostic, explicitly not evidence of generalization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

import evaluate_public_corpus as common
from evaluate_public_facts import token_f1

OUTPUT = common.OUTPUT
SAMPLE = OUTPUT / "germanquad_replay_sample.json"
SOURCE = common.ROOT / "memory/imports/public_knowledge_germanquad_de.jsonl"


def initialize():
    if SAMPLE.exists():
        return common.read(SAMPLE)
    corpus = common.lines(SOURCE)
    chosen = random.Random(9021).sample(corpus, 20)
    value = {"created_utc": common.timestamp(), "source_sha256": common.sha(SOURCE),
             "selection": "random.Random(9021).sample(final_v3_GermanQuAD_import_records, 20), before candidate answers",
             "scope": "Known original import questions. This is in-sample storage/generation capacity, never held-out generalization or a standard QA benchmark.",
             "rows": [{"id": row["id"], "prompt": row["prompt"], "reference_text": row["text"],
                       "reference_answer": row["provenance"]["gold_answer_text"],
                       "source": row["source"], "provenance": row["provenance"]} for row in chosen]}
    common.write(SAMPLE, value)
    return value


def run(args, engine=None):
    sample = initialize()
    records = common.corpus_records(args)
    ids = {row["id"] for row in records}
    manifest = common.configuration_manifest(args)
    manifest.update({"replay_runner_sha256": common.sha(Path(__file__)), "replay_sample_sha256": common.sha(SAMPLE)})
    destination = OUTPUT / f"{args.label}_replay.json"
    assert not destination.exists(), "Replay record is immutable"
    memory, runtime, model, build_s = engine or common.load_engine(Path(args.code_root), records)
    rows = []
    for reference in sample["rows"]:
        started = time.perf_counter()
        result, _ = runtime.respond_wave(memory, reference["prompt"], context=None, time_s=13.125, max_tokens=40, seed=17)
        answer = result["answer"]
        row = {**reference, "answer": answer, "tokens": result["tokens"], "known_import_record": reference["id"] in ids,
               "elapsed_s": time.perf_counter()-started, "truncated": result["truncated"],
               "automatic": {"exact_reference_text": common.normalize(answer) == common.normalize(reference["reference_text"]),
                             "contains_answer_string": common.normalize(reference["reference_answer"]) in common.normalize(answer),
                             "token_f1_against_answer": token_f1(answer, reference["reference_answer"]),
                             "not_a_factual_correctness_judgment": True},
               "qualitative_review": None}
        rows.append(row)
        print(json.dumps({"status": "replay_turn", "id": reference["id"], "elapsed_s": row["elapsed_s"]}), flush=True)
    summary = {"questions": len(rows), "known_import_records": sum(row["known_import_record"] for row in rows),
               "exact_reference_texts": sum(row["automatic"]["exact_reference_text"] for row in rows),
               "contains_answer_string": sum(row["automatic"]["contains_answer_string"] for row in rows),
               "mean_token_f1": sum(row["automatic"]["token_f1_against_answer"] for row in rows)/len(rows),
               "factual_correctness": "pending independent AI-agent qualitative review"}
    common.write(destination, {"created_utc": common.timestamp(), "manifest": manifest, "scope": sample["scope"],
                               "build_s": build_s, "memory": common.process_memory(), "summary": summary, "rows": rows})
    print(json.dumps({"path": str(destination), "summary": summary}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "run"])
    parser.add_argument("--code-root", default=str(common.ROOT))
    parser.add_argument("--corpus")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("Unsafe label")
    if args.action == "initialize":
        print(json.dumps({key: value for key, value in initialize().items() if key != "rows"}))
    else:
        run(args)


if __name__ == "__main__":
    main()
