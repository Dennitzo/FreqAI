"""Replay already observed outputs after a numerical performance change.

This is output regression, never a new holdout or a quality improvement test.
The old factual artifact saved visible text only; its token reference is
reconstructed explicitly and checked alongside exact visible text equality.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import evaluate_public_corpus as common

REFERENCES = [
    ("development", "publicall_v3_lexical_development.json"),
    ("holdout", "final_holdout.json"),
    ("facts", "final_facts.json"),
    ("replay", "final_replay.json"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", default=str(common.ROOT))
    parser.add_argument("--corpus", default=str(common.ROOT / "memory/imports/public_dialogue_expansion.jsonl"))
    parser.add_argument("--label", default="fast_carrier")
    args = parser.parse_args()
    args.limit = None
    destination = common.OUTPUT / f"{args.label}_output_regression.json"
    assert not destination.exists(), "Output regression artifacts are immutable"
    freeze = common.read(common.OUTPUT / "final_freeze.json")["manifest"]
    assert common.sha(args.corpus) == freeze["additional_corpus_sha256"], "Corpus changed after quality freeze"
    assert common.sha(common.BASE_CORPUS) == freeze["baseline_corpus_sha256"]
    assert common.sha(common.SUITE) == freeze["suite_sha256"]
    reference_runs = [(kind, filename, common.read(common.OUTPUT / filename)) for kind, filename in REFERENCES]
    assert sum(len(run["rows"]) for _, _, run in reference_runs) == 90
    records = common.corpus_records(args)
    print(json.dumps({"status": "compiling_once", "documents": len(records)}), flush=True)
    memory, runtime, model, build_s = common.load_engine(Path(args.code_root), records)
    from freqai.generative import EOS, tokenize
    after_build = common.process_memory()
    print(json.dumps({"status": "compiled", "build_s": build_s, "vocabulary_size": model.size,
                      "carrier_size": getattr(model, "carrier_size", model.size), **after_build}), flush=True)
    raw_rows = [row for _, _, run in reference_runs for row in run["rows"] if "tokens" in row]
    raw_roundtrip_failures = [row["id"] for row in raw_rows if tokenize(row["answer"]) != row["tokens"]]
    alphabet_roundtrip_failures = [token for token in model.vocabulary if token != EOS and tokenize(model.render([token])) != [token]]
    rows = []
    for kind, filename, reference in reference_runs:
        contexts = {}
        for row in reference["rows"]:
            scenario = row.get("scenario")
            context = contexts.get(scenario) if scenario else None
            expected_context = row.get("context_before")
            context_matches_reference = context == expected_context if "context_before" in row else None
            expected_tokens = row["tokens"] if "tokens" in row else tokenize(row["answer"])
            started = time.perf_counter()
            result, next_context = runtime.respond_wave(memory, row["prompt"], context=context,
                                                        time_s=13.125, max_tokens=40, seed=17)
            elapsed = time.perf_counter() - started
            if scenario:
                contexts[scenario] = next_context
            measured = {"kind": kind, "id": row["id"], "reference_file": filename,
                        "prompt": row["prompt"], "reference_answer": row["answer"], "answer": result["answer"],
                        "reference_tokens": expected_tokens, "tokens": result["tokens"],
                        "token_reference_kind": "stored_raw_tokens" if "tokens" in row else "reconstructed_from_frozen_visible_text",
                        "answer_bytes_equal": result["answer"].encode("utf-8") == row["answer"].encode("utf-8"),
                        "tokens_equal": result["tokens"] == expected_tokens,
                        "context_before_equal": context_matches_reference,
                        "elapsed_s": elapsed, "reference_elapsed_s": row["elapsed_s"],
                        "max_transform_error": result["decoder"]["max_transform_error"]}
            rows.append(measured)
            print(json.dumps({"status": "regression_turn", "kind": kind, "id": row["id"],
                              "answer_equal": measured["answer_bytes_equal"], "tokens_equal": measured["tokens_equal"],
                              "elapsed_s": elapsed}), flush=True)
    summary = {"cases": len(rows), "exact_visible_outputs": sum(row["answer_bytes_equal"] for row in rows),
               "token_sequence_matches": sum(row["tokens_equal"] for row in rows),
               "stored_raw_token_references": len(raw_rows), "reconstructed_token_references": len(rows)-len(raw_rows),
               "raw_reference_roundtrip_checked": len(raw_rows), "raw_reference_roundtrip_failures": raw_roundtrip_failures,
               "alphabet_roundtrip_checked": model.size-1, "alphabet_roundtrip_failure_count": len(alphabet_roundtrip_failures),
               "alphabet_roundtrip_failure_examples": alphabet_roundtrip_failures[:20],
               "context_inputs_matching": sum(row["context_before_equal"] is True for row in rows),
               "context_inputs_checked": sum(row["context_before_equal"] is not None for row in rows),
               "median_elapsed_s": statistics.median(row["elapsed_s"] for row in rows),
               "median_reference_elapsed_s": statistics.median(row["reference_elapsed_s"] for row in rows),
               "by_kind": {kind: {"cases": sum(row["kind"] == kind for row in rows),
                                   "equal_outputs": sum(row["kind"] == kind and row["answer_bytes_equal"] for row in rows),
                                   "median_elapsed_s": statistics.median(row["elapsed_s"] for row in rows if row["kind"] == kind),
                                   "median_reference_elapsed_s": statistics.median(row["reference_elapsed_s"] for row in rows if row["kind"] == kind)}
                           for kind, _ in REFERENCES}}
    passed = summary["exact_visible_outputs"] == summary["token_sequence_matches"] == 90 and not raw_roundtrip_failures and not alphabet_roundtrip_failures
    summary["passed"] = passed
    common.write(destination, {"created_utc": common.timestamp(), "scope": __doc__, "summary": summary,
                               "limitation": "Thirty original factual rows did not persist raw token streams. They are checked against exact frozen UTF-8 answer text and its reconstructed tokens. Sixty directly stored streams and the full model alphabet independently check renderer/tokenizer round trips. This is not a newly unseen quality evaluation.",
                               "reference_hashes": {filename: common.sha(common.OUTPUT / filename) for _, filename in REFERENCES},
                               "corpus_sha256": common.sha(args.corpus), "current_code": common.code_manifest(Path(args.code_root)),
                               "runner_sha256": common.sha(Path(__file__)), "build_s": build_s,
                               "memory_after_build": after_build, "memory_after_regression": common.process_memory(), "rows": rows})
    print(json.dumps({"path": str(destination), "summary": summary}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
