"""Assemble hash-bound results; never infer semantic success from keyword proxies."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import evaluate_public_corpus as common

OUT = common.ROOT / "results/unpaired_system/evaluation"


def read(name):
    return common.read(OUT / name)


def stats(rows, review):
    seen = {}
    novel_from_prompt, novel_other = set(), set()
    for row in rows:
        known = seen.setdefault(row["scenario"], set())
        known.update(common.normalize(row["prompt"]).split())
        novel = set(row["automatic"]["new_word_types"])
        novel_from_prompt.update(novel & known)
        novel_other.update(novel - known)
    nonempty = [row for row in rows if row["answer"]]
    full = {row["id"] for row in review["rows"] if row["appropriate"]}
    return {**review["summary"], "median_generation_s": statistics.median(row["elapsed_s"] for row in rows),
            "median_nonempty_generation_s": statistics.median(row["elapsed_s"] for row in nonempty) if nonempty else None,
            "median_nonempty_output_symbols": statistics.median(len(row["tokens"]) for row in nonempty) if nonempty else None,
            "exact_normalized_source_passages": sum(row["automatic"]["exact_normalized_source_passage"] for row in rows),
            "fully_appropriate_exact_source_passages": sum(row["id"] in full and row["automatic"]["exact_normalized_source_passage"] for row in rows),
            "new_complete_texts": sum(row["automatic"]["new_complete_text"] for row in nonempty),
            "novel_word_types_from_current_or_previous_user_turn": sorted(novel_from_prompt),
            "novel_word_types_neither_source_nor_seen_user_turn": sorted(novel_other),
            "replacement_character_outputs": sum("\ufffd" in row["answer"] for row in rows),
            "novelty_note": "New strings can be rule-based paraphrases or user echoes; this is not automatically new knowledge or conversation quality."}


def main():
    destination = OUT / "summary.json"
    assert not destination.exists()
    final = read("final_holdout.json")
    freeze = read("final_freeze.json")
    assert final["manifest"] == freeze["manifest"]
    assert common.code_manifest(OUT / "final_code") == freeze["manifest"]["code"]
    reviewed = {
        "baseline_development": (read("baseline_development.json"), read("baseline_development_review.json")),
        "baseline_holdout": (read("baseline_final_holdout.json"), read("baseline_final_holdout_review.json")),
        "trial_v1_development": (read("trial_v1_development.json"), read("trial_v1_development_review.json")),
        "trial_v2_development": (read("trial_v2_development.json"), read("trial_v2_development_review.json"))}
    result = {}
    for key, (run, review) in reviewed.items():
        filename = "baseline_final_holdout.json" if key == "baseline_holdout" else key + ".json"
        assert review["run_sha256"] == common.sha(OUT / filename)
        result[key] = stats(run["rows"], review)
    for key, split, review_file in (("final_holdout", "holdout", "final_holdout_review.json"),
                                    ("final_known_development_replay", "known_development_replay", "final_development_replay_review.json")):
        review = read(review_file)
        assert review["run_sha256"] == common.sha(OUT / "final_holdout.json")
        result[key] = stats([row for row in final["rows"] if row["split"] == split], review)
    v2 = read("trial_v2_development.json")
    numerics_path = common.ROOT / "results/unpaired_system/numerical_audit.json"
    numerics = common.read(numerics_path)
    normalized_code = {key.replace("\\", "/"): value for key, value in freeze["manifest"]["code"].items()}
    assert numerics["code_sha256"] == normalized_code["freqai/unpaired.py"]
    result.update({"created_utc": common.timestamp(), "status": "final frozen independent evaluation; no quality tuning after fresh holdout",
                   "review_limit": "AI-agent semantic review, no human study;20 unseen turns in10 sessions are a small fixed sample.",
                   "candidate_data_audit": final["data_audit"],
                   "final_known_knowledge_regression": read("final_knowledge_review.json")["summary"],
                   "final_known_randomfact_regression": read("final_randomfacts_review.json")["summary"],
                   "numerical_controls": {key: numerics[key] for key in (
                       "checked_prefix_time_states", "fft_direct_max_probability_error_after_rounding",
                       "prompt_phase_total_variation", "data_role_phase_total_variation",
                       "zero_roles_refuse_with_prose_transition_fields_intact", "zero_prompt_refuses",
                       "same_inputs_rebuild_identical_coefficients", "queries_do_not_modify_corpus_coefficients",
                       "no_global_session_roles_retained", "unknown_name_isolated_between_sessions", "optimizer_steps", "fitted_mixing_gains")},
                   "build": {"final_s": final["build_s"], "baseline_holdout_s": reviewed["baseline_holdout"][0]["build_s"],
                             "memory_after_build": final["memory_after_build"], "memory_after_evaluation": final["memory_after_evaluation"],
                             "scope": "Lightweight runtime Document holder, excluding separate document-wave/UI/SQLite costs; real production token budgets. Other team processes may run concurrently."},
                   "model_signature": final["model_signature"],
                   "v2_final_rebuild_equal": {key: v2["model_signature"][key] == final["model_signature"][key]
                                             for key in ("corpus_digest", "coefficient_sha256", "rendering_sha256", "declaration_role_sha256")},
                   "artifacts": {path.name: common.sha(path) for path in sorted(OUT.glob("*.json")) if not path.name.endswith("_progress.json")},
                   "numerical_audit_sha256": common.sha(numerics_path),
                   "post_freeze_change_note": "Root authorized separate raw-import-field validation in CLI/HTTP/helper after freeze. No model/corpus/word-decoder change; deployment records its own manifest."})
    common.write(destination, result)
    print(json.dumps({"path": str(destination), "baseline_holdout": result["baseline_holdout"]["appropriate"],
                      "final_holdout": result["final_holdout"]["appropriate"], "known_development": result["final_known_development_replay"]["appropriate"]}))


if __name__ == "__main__":
    main()
