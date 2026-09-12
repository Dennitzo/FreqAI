"""Independent conversation evaluation for the fully unpaired information system.

The historical combined system is a read-only baseline. Candidate corpora replace
it completely: no QA records or authored response-prior records are carried over.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
from pathlib import Path
import re
import statistics
import sys
import time

import evaluate_public_corpus as common
from evaluate_information_corpus import TextIndex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/unpaired_system/evaluation"
BASE_CODE = OUT / "baseline_code"
BASE_CORPUS = OUT / "baseline_documents.jsonl"
DEV = ROOT / "memory/evaluation/unpaired_chat_development.json"
HOLD = ROOT / "memory/evaluation/unpaired_chat_holdout.json"
KNOWLEDGE = ROOT / "memory/evaluation/unpaired_knowledge_regression.json"


def immutable(path, value):
    assert not path.exists(), f"Immutable result already exists: {path}"
    common.write(path, value)


def records_for(args):
    baseline = common.lines(BASE_CORPUS)
    if not args.corpus:
        assert Path(args.code_root).resolve() == BASE_CODE.resolve(), "Candidate requires explicit replacement corpus"
        return baseline, {"historical_baseline": True, "documents": len(baseline)}
    records = [row for path in args.corpus for row in common.lines(path)]
    assert records and len({row["id"] for row in records}) == len(records)
    old_pair_ids = {row["id"] for row in baseline if row.get("prompt", "").strip()}
    old_pair_texts = {common.normalize(row["text"]) for row in baseline if row.get("prompt", "").strip()}
    prohibited = {"question", "answer", "response", "messages", "user", "assistant", "instruction", "output"}
    for row in records:
        assert not row.get("prompt", "").strip(), "QA prompt in candidate"
        assert not prohibited.intersection(row), "Conversation-pair/instruction record in candidate"
        assert row["id"] not in old_pair_ids, "Historical paired record carried into active candidate"
        assert not row.get("source", "").startswith("Authored synthetic language prior / "), "Old response prior carried over"
        assert row.get("text", "").strip()
    exact_overlap = [row["id"] for row in records if common.normalize(row["text"]) in old_pair_texts]
    return records, {"historical_baseline": False, "documents": len(records), "paired_records": 0,
                     "historical_paired_ids_carried_over": 0, "old_authored_response_prior_records": 0,
                     "exact_text_overlap_with_old_pair_answers": exact_overlap,
                     "overlap_note": "Independent natural prose may share facts with former QA sources; provenance, IDs and input pairing are audited separately."}


def manifest(args):
    protocol = common.read(OUT / "protocol.json")
    assert common.sha(DEV) == protocol["development_sha256"]
    assert common.sha(HOLD) == protocol["sealed_holdout_sha256"]
    assert common.sha(KNOWLEDGE) == protocol["known_knowledge_regression_sha256"]
    records, audit = records_for(args)
    return {"code": common.code_manifest(Path(args.code_root)), "code_root": str(Path(args.code_root).resolve()),
            "corpora": [{"path": str(Path(path).resolve()), "sha256": common.sha(path)} for path in args.corpus],
            "baseline_corpus_sha256": common.sha(BASE_CORPUS), "protocol_sha256": common.sha(OUT / "protocol.json"),
            "runner_sha256": common.sha(__file__), "text_index_helper_sha256": common.sha(ROOT / "experiments/evaluate_information_corpus.py"),
            "runtime_helper_sha256": common.sha(ROOT / "experiments/evaluate_public_corpus.py"),
            "document_count": len(records), "data_audit": audit, "max_tokens": 40, "knowledge_max_tokens": 64,
            "known_knowledge_regression": args.knowledge_regression,
            "known_development_replay": args.replay_development, "seed": 17, "time_s": 13.125}


def engine(code_root, records):
    sys.path.insert(0, str(code_root))
    runtime = importlib.import_module("freqai.generation_runtime")
    Document = importlib.import_module("freqai.memory").Document
    allowed = inspect.signature(Document).parameters
    memory = common.EvaluationMemory([Document(**{key: value for key, value in row.items() if key in allowed}) for row in records])
    started = time.perf_counter()
    if hasattr(runtime, "generator_for"):
        holder = runtime.generator_for(memory)
    else:
        holder = importlib.import_module("freqai.information_runtime").information_for(memory)
    model = holder.model if hasattr(holder, "model") else holder
    return memory, runtime, model, time.perf_counter()-started


def run(args):
    configuration = manifest(args)
    if args.split == "holdout":
        assert common.read(OUT / f"{args.label}_freeze.json")["manifest"] == configuration
    target = OUT / f"{args.label}_{args.split}.json"
    assert not target.exists()
    records, audit = records_for(args)
    text_index = TextIndex(records)
    scenarios = [{**scenario, "evaluation_split": args.split} for scenario in common.read(DEV if args.split == "development" else HOLD)["scenarios"]]
    if args.replay_development:
        assert args.split == "holdout", "Development replay is appended only to the final frozen holdout run"
        scenarios += [{**scenario, "evaluation_split": "known_development_replay"} for scenario in common.read(DEV)["scenarios"]]
    print(json.dumps({"status": "compiling", "label": args.label, "documents": len(records)}), flush=True)
    memory, runtime, model, build_s = engine(Path(args.code_root), records)
    signature = model.model_signature() if hasattr(model, "model_signature") else None
    after_build = common.process_memory()
    print(json.dumps({"status": "compiled", "build_s": build_s, **after_build}), flush=True)
    rows = []
    knowledge_rows = []
    def checkpoint():
        common.write(OUT / f"{args.label}_{args.split}_progress.json", {
            "status": "incomplete, not a final result", "manifest": configuration,
            "rows": rows, "knowledge_rows": knowledge_rows})
    for scenario in scenarios:
        context = None
        for turn in scenario["turns"]:
            before = copy.deepcopy(context)
            start = time.perf_counter()
            result, context = runtime.respond_wave(memory, turn["prompt"], context=context, time_s=13.125, max_tokens=40, seed=17)
            rows.append({**turn, "scenario": scenario["id"], "split": scenario["evaluation_split"], "context_before": before,
                         "answer": result["answer"], "tokens": result.get("tokens", []), "elapsed_s": time.perf_counter()-start,
                         "automatic": text_index.evaluate(result["answer"]),
                         "diagnostics": {key: result.get(key) for key in ("method", "abstained", "ended", "truncated", "generation_reason", "decoder", "tokenization", "configuration", "prompt_encoding")},
                         "qualitative_review": None})
            checkpoint()
            print(json.dumps({"status": "chat_turn", "id": turn["id"], "elapsed_s": rows[-1]["elapsed_s"]}), flush=True)
    if args.knowledge_regression:
        for case in common.read(KNOWLEDGE)["cases"]:
            start = time.perf_counter()
            result, _ = runtime.respond_wave(memory, case["prompt"], context=None, time_s=13.125, max_tokens=64, seed=17)
            knowledge_rows.append({**case, "answer": result["answer"], "tokens": result.get("tokens", []),
                                   "elapsed_s": time.perf_counter()-start, "automatic": text_index.evaluate(result["answer"]),
                                   "diagnostics": {key: result.get(key) for key in ("method", "abstained", "ended", "truncated", "generation_reason", "tokenization", "prompt_encoding")},
                                   "qualitative_review": None})
            checkpoint()
    summary = {"conversation_turns": len(rows), "sessions": len(scenarios), "empty": sum(not row["answer"] for row in rows),
               "surface_form_ok": sum(row["automatic"]["surface_form_ok"] for row in rows),
               "exact_source_passages": sum(row["automatic"]["exact_normalized_source_passage"] for row in rows),
               "median_generation_s": statistics.median(row["elapsed_s"] for row in rows),
               "by_split": {split: {"turns": sum(row["split"] == split for row in rows),
                                     "empty": sum(row["split"] == split and not row["answer"] for row in rows)} for split in sorted({row["split"] for row in rows})},
               "known_knowledge_regression_cases": len(knowledge_rows), "conversation_quality": "pending independent AI-agent review"}
    value = {"created_utc": common.timestamp(), "label": args.label, "split": args.split, "manifest": configuration,
             "build_s": build_s, "memory_after_build": after_build, "memory_after_evaluation": common.process_memory(),
             "model_signature": signature, "data_audit": audit, "rows": rows, "knowledge_rows": knowledge_rows, "summary": summary}
    immutable(target, value)
    print(json.dumps({"path": str(target), "summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "freeze", "audit"])
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--corpus", action="append", default=[], help="Replacement JSONL corpus; repeat for separate prose sources")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--knowledge-regression", action="store_true")
    parser.add_argument("--replay-development", action="store_true", help="Append already known20 development turns after the fresh holdout in the same frozen model process")
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.label)
    if args.action == "freeze":
        immutable(OUT / f"{args.label}_freeze.json", {"created_utc": common.timestamp(), "manifest": manifest(args), "holdout_outputs_seen": False})
    elif args.action == "audit":
        print(json.dumps(manifest(args), ensure_ascii=False))
    else:
        run(args)


if __name__ == "__main__":
    main()
