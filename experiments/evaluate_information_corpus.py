"""Frozen, read-only evaluation of declarative-information wave generation.

Prompts and rubric concepts are evaluator-only. They are never passed to a
constructor. Statistical count normalization is data fitting without gradient
optimization, not evidence for weight-free intelligence.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

import evaluate_public_corpus as common

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/information_corpus/evaluation"
BASE_CODE = OUTPUT / "baseline_code"
BASE_CORPUS = OUTPUT / "baseline_documents.jsonl"
DEVELOPMENT = ROOT / "memory/evaluation/information_development.json"
HOLDOUT = ROOT / "memory/evaluation/information_holdout.json"
CHAT = ROOT / "memory/evaluation/public_corpus_queries.json"


def immutable(path, value):
    assert not path.exists(), f"Recorded artifact must not be overwritten: {path}"
    common.write(path, value)


def initialize():
    path = OUTPUT / "protocol.json"
    if path.exists():
        return common.read(path)
    baseline = common.read(OUTPUT / "baseline_freeze.json")
    assert baseline["code"] == common.code_manifest(BASE_CODE)
    assert baseline["baseline_corpus_sha256"] == common.sha(BASE_CORPUS)
    assert len(common.read(DEVELOPMENT)["cases"]) == len(common.read(HOLDOUT)["cases"]) == 20
    value = {"created_utc": common.timestamp(), "development_sha256": common.sha(DEVELOPMENT),
             "sealed_holdout_sha256": common.sha(HOLDOUT), "known_chat_regression_sha256": common.sha(CHAT),
             "baseline_freeze_sha256": common.sha(OUTPUT / "baseline_freeze.json"),
             "review": "Independent AI-agent semantic review, not a human study. No rubric-based automatic pass count.",
             "gate": "No holdout generation before explicit candidate freeze. No quality tuning after holdout.",
             "explicit_user_example_is_development": True, "optimization_steps": 0,
             "prior_40_chat_cases": "Previously observed cases, regression only, no new holdout claim.",
             "coefficient_claim": "New information decoder: observed counts and normalization, no tuned source/mix weights or gradient updates. Existing chat coefficients are outside this claim.",
             "new_text_warning": "New word sequences or sentences are not evidence of correct facts or understanding."}
    immutable(path, value)
    return value


def documents(args):
    baseline = common.lines(BASE_CORPUS)
    additional = common.lines(args.corpus) if args.corpus else []
    if args.limit is not None:
        additional = additional[:args.limit]
    assert all(not row.get("prompt", "").strip() for row in additional), "Information corpus must not contain QA inputs"
    assert all(row.get("text", "").strip() for row in additional)
    return baseline + additional if not args.information_only else additional


def manifest(args):
    protocol = initialize()
    assert common.sha(DEVELOPMENT) == protocol["development_sha256"]
    assert common.sha(HOLDOUT) == protocol["sealed_holdout_sha256"]
    assert common.sha(CHAT) == protocol["known_chat_regression_sha256"]
    return {"code_root": str(Path(args.code_root).resolve()), "code": common.code_manifest(Path(args.code_root)),
            "baseline_corpus_sha256": common.sha(BASE_CORPUS), "protocol_sha256": common.sha(OUTPUT / "protocol.json"),
            "additional_corpus": str(Path(args.corpus).resolve()) if args.corpus else None,
            "additional_corpus_sha256": common.sha(args.corpus) if args.corpus else None,
            "additional_limit": args.limit, "engine": args.engine, "information_only": args.information_only,
            "chat_regression": args.chat_regression,
            "probe_default_limit": args.probe_default_limit,
            "document_count": len(documents(args)), "runner_sha256": common.sha(__file__),
            "max_tokens": 64, "seed": 17, "time_s": 13.125}


def evidence(case, records):
    """Candidate evidence only: concept co-occurrence cannot prove entailment."""
    if not case.get("concepts"):
        return {"kind": "unprovided personal/current measurement", "candidates": [], "verified_support": None}
    candidates = []
    for row in records:
        text = row["text"].casefold()
        if all(any(term.casefold() in text for term in group) for group in case["concepts"]):
            candidates.append({"id": row["id"], "source": row.get("source", ""),
                               "title": row.get("title", row.get("provenance", {}).get("article_title")),
                               "text": row["text"][:5000]})
    return {"kind": "same-document concept co-occurrence, manually verify entailment", "candidate_count": len(candidates),
            "candidates": candidates[:5], "verified_support": None}


class TextIndex:
    def __init__(self, records):
        self.texts = [common.normalize(row["text"]) for row in records]
        self.whole = set(self.texts)
        self.words = {word for text in self.texts for word in text.split()}
        self.sentences = {common.normalize(sentence) for row in records
                          for sentence in re.split(r"(?<=[.!?])\s+", row["text"]) if sentence.strip()}

    def evaluate(self, answer):
        value = common.automatic(answer, self.whole, self.sentences)
        normalized = common.normalize(answer)
        value["exact_normalized_source_passage"] = bool(normalized) and any(normalized in text for text in self.texts)
        value["new_word_types"] = sorted(set(normalized.split()) - self.words)
        value["novelty_is_not_correctness"] = True
        return value


def run(args):
    run_manifest = manifest(args)
    if args.split in ("holdout", "all"):
        frozen = common.read(OUTPUT / f"{args.label}_freeze.json")
        assert frozen["manifest"] == run_manifest, "Frozen code, corpus or runner changed"
    target = OUTPUT / f"{args.label}_{args.split}.json"
    assert not target.exists(), "Immutable run; choose a new development label"
    records = documents(args)
    selected_splits = ("development", "holdout") if args.split == "all" else (args.split,)
    cases = [{**case, "split": split} for split in selected_splits
             for case in common.read(DEVELOPMENT if split == "development" else HOLDOUT)["cases"]]
    index = TextIndex(records)
    normalized_prompts = {common.normalize(case["prompt"]) for case in cases}
    existing_inputs = {common.normalize(row.get("prompt", "")) for row in records if row.get("prompt", "")}
    overlap = sorted(normalized_prompts & existing_inputs)
    print(json.dumps({"status": "compiling", "engine": args.engine, "documents": len(records), "label": args.label}), flush=True)
    if args.engine == "runtime":
        runtime = None
        if (Path(args.code_root) / "freqai/information_runtime.py").exists():
            sys.path.insert(0, str(Path(args.code_root)))
            adapter = importlib.import_module("freqai.information_runtime")
            if not args.chat_regression and all(adapter.information_request(case["prompt"]) for case in cases):
                runtime = importlib.import_module("freqai.generation_runtime")
                Document = importlib.import_module("freqai.memory").Document
                memory = common.EvaluationMemory([Document(**{key: row.get(key, "") for key in ("id", "text", "source", "prompt")}) for row in records])
                model, build_s = None, 0.0
        if runtime is None:
            memory, runtime, model, build_s = common.load_engine(Path(args.code_root), records)
        build_stages = {"conversation_s": build_s}
        information_signature = None
        if (Path(args.code_root) / "freqai/information_runtime.py").exists():
            adapter = importlib.import_module("freqai.information_runtime")
            started = time.perf_counter()
            information_model = adapter.information_for(memory).model
            if model is None:
                model = information_model
            build_stages["information_s"] = time.perf_counter() - started
            if hasattr(information_model, "model_signature"):
                information_signature = information_model.model_signature()
            build_s += build_stages["information_s"]
        def generate(prompt, context):
            result, state = runtime.respond_wave(memory, prompt, context=context, time_s=13.125, max_tokens=64, seed=17)
            return result, state
    else:
        sys.path.insert(0, str(Path(args.code_root)))
        module = importlib.import_module("freqai.information")
        started = time.perf_counter()
        model = module.InformationWaveModel(records, order=2)
        build_s = time.perf_counter() - started
        build_stages = {"information_s": build_s}
        information_signature = model.model_signature() if hasattr(model, "model_signature") else None
        def generate(prompt, context):
            result = model.generate(prompt, max_tokens=64, decoding="beam", seed=17, context=context, method="operator")
            return {**result, "answer": result.get("text", "")}, result.get("information_state")
    print(json.dumps({"status": "compiled", "build_s": build_s, **common.process_memory()}), flush=True)
    after_build = common.process_memory()
    rows = []
    chat_rows = []
    def checkpoint():
        common.write(OUTPUT / f"{args.label}_{args.split}_progress.json", {
            "status": "incomplete; never a final evaluation result", "manifest": run_manifest,
            "rows": rows, "known_chat_rows": chat_rows})
    for case in cases:
        started = time.perf_counter()
        result, _ = generate(case["prompt"], None)
        row = {**case, "answer": result["answer"], "tokens": result.get("tokens", []),
               "elapsed_s": time.perf_counter()-started, "context_before": None,
               "automatic": index.evaluate(result["answer"]), "evidence": evidence(case, records),
               "diagnostics": {key: result.get(key) for key in ("analysis", "reason", "generation_reason", "decoder", "ended", "truncated")},
               "qualitative_review": None}
        rows.append(row)
        checkpoint()
        print(json.dumps({"status": "information_turn", "id": case["id"], "elapsed_s": row["elapsed_s"]}), flush=True)
    if args.chat_regression:
        assert args.engine == "runtime", "Chat regression must exercise actual runtime routing"
        for scenario in common.read(CHAT)["scenarios"]:
            context = None
            for turn in scenario["turns"]:
                before = copy.deepcopy(context)
                started = time.perf_counter()
                result, context = generate(turn["prompt"], context)
                chat_rows.append({**turn, "scenario": scenario["id"], "answer": result["answer"], "tokens": result.get("tokens", []),
                                  "context_before": before, "elapsed_s": time.perf_counter()-started,
                                  "automatic": index.evaluate(result["answer"]), "qualitative_review": None})
                checkpoint()
                print(json.dumps({"status": "known_chat_turn", "id": turn["id"]}), flush=True)
    default_limit_probe = None
    if args.probe_default_limit:
        assert args.engine == "runtime"
        result, _ = runtime.respond_wave(memory, "Was ist eine Frequenz?", context=None, time_s=13.125, max_tokens=40, seed=17)
        reference = common.read(OUTPUT / "runtime_v2_development.json")["rows"][0]
        default_limit_probe = {"prompt": "Was ist eine Frequenz?", "max_tokens": 40,
                               "explicit_user_development_example": True, "scored_as_holdout": False,
                               "answer": result["answer"], "tokens": result["tokens"], "ended": result["ended"],
                               "truncated": result["truncated"], "token_count": len(result["tokens"]),
                               "same_text_as_64_token_development_reference": result["answer"] == reference["answer"],
                               "same_tokens_as_64_token_development_reference": result["tokens"] == reference["tokens"],
                               "reference_sha256": common.sha(OUTPUT / "runtime_v2_development.json")}
    summary = {"information_cases": len(rows), "known_chat_regression_cases": len(chat_rows),
               "empty_answers": sum(not row["answer"].strip() for row in rows),
               "surface_form_ok": sum(row["automatic"]["surface_form_ok"] for row in rows),
               "new_complete_texts": sum(row["automatic"]["new_complete_text"] for row in rows),
               "exact_source_passages": sum(row["automatic"]["exact_normalized_source_passage"] for row in rows),
               "median_generation_s": statistics.median(row["elapsed_s"] for row in rows),
               "semantic_correctness": "pending independent AI-agent review; automatic form/overlap metrics are not quality"}
    value = {"created_utc": common.timestamp(), "manifest": run_manifest, "label": args.label, "split": args.split,
             "build_s": build_s, "build_stages": build_stages, "information_signature": information_signature,
             "memory_after_build": after_build, "memory_after_evaluation": common.process_memory(),
             "observed_prompt_overlap_with_existing_qa": overlap,
             "summary": summary, "rows": rows, "known_chat_rows": chat_rows,
             "default_limit_probe": default_limit_probe}
    immutable(target, value)
    print(json.dumps({"path": str(target), "summary": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "freeze", "run"])
    parser.add_argument("--code-root", default=str(ROOT))
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--corpus")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--engine", choices=["runtime", "information"], default="runtime")
    parser.add_argument("--information-only", action="store_true")
    parser.add_argument("--chat-regression", action="store_true")
    parser.add_argument("--probe-default-limit", action="store_true")
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.label)
    if args.action == "initialize":
        print(json.dumps(initialize(), ensure_ascii=False))
    elif args.action == "freeze":
        immutable(OUTPUT / f"{args.label}_freeze.json", {"created_utc": common.timestamp(), "manifest": manifest(args),
                                                         "holdout_outputs_seen": False})
    else:
        run(args)


if __name__ == "__main__":
    main()
