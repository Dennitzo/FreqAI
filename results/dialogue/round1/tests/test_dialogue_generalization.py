"""Development regression and independent answer/state properties.

Held-out/stress prompts are intentionally not parametrized as implementation
regressions: their first use is the frozen final evaluation, including failures.
"""

import hashlib
import json

import pytest

from experiments.run_dialogue import (CASES, CORPUS, OUTPUT, evaluate_answer,
                                     load_inputs, numerical_audit, run_cases, sha256)


@pytest.fixture(scope="module")
def dialogue_inputs():
    return load_inputs()


def test_dialogue_suite_is_frozen_and_has_over_100_unstored_unique_prompts(dialogue_inputs):
    data, memory = dialogue_inputs
    frozen = json.loads((OUTPUT / "suite_freeze.json").read_text(encoding="utf-8"))
    assert sha256(CASES) == frozen["suite_sha256"]
    assert sha256(CORPUS) == frozen["corpus_sha256"]
    assert len(memory.documents) == 120
    prompts = {t["prompt"] for s in data["scenarios"] for t in s["turns"]}
    assert len(prompts - {d.prompt for d in memory.documents}) >= 100
    assert frozen["split_counts"] == {"development": 51, "holdout": 50, "stress": 33}


def test_oracle_demands_every_intent_in_a_compound_answer():
    expected = {"concepts": ["positive", "assistant_status"]}
    assert "answer_missing:assistant_status" in evaluate_answer(
        expected, {"answer": "Schön, dass es dir gut geht.", "abstained": False})
    assert evaluate_answer(expected, {
        "answer": "Für unser Gespräch bin ich bereit. Es freut mich, dass es dir gut geht!",
        "abstained": False}) == []


def test_oracle_rejects_metadata_claim_without_observable_answer():
    result = {"answer": "Das ist ein beliebiger Satz.", "abstained": False,
              "response_acts": ["user_mood", "assistant_status"],
              "interpretation": {"acts": [{"kind": "mood_statement", "target": "user", "value": "positive"}]}}
    failures = evaluate_answer({"concepts": ["positive", "assistant_status"]}, result)
    assert "answer_missing:positive" in failures
    assert "answer_missing:assistant_status" in failures


def test_oracle_rejects_wrong_subject_and_name_leakage():
    result = {"answer": "Schön, dass es dir gut geht, Clara.", "abstained": False,
              "response_acts": ["user_mood"]}
    failures = evaluate_answer({"concepts": ["third_positive"], "subject": "Anna",
                                "forbid_user_mood": True, "forbidden_names": ["Clara"]}, result)
    assert "wrong_subject:user" in failures
    assert "leaked_or_wrong_name:Clara" in failures


def test_independent_spectral_numerics_and_existing_120_payloads(dialogue_inputs):
    _, memory = dialogue_inputs
    result = numerical_audit(memory)
    assert result["pair_comparisons"] == 1764
    assert result["text_roundtrip_passed"] == result["text_roundtrip_total"] == 600


@pytest.mark.parametrize("scenario_id", [
    "original-regression", "simple-positive", "simple-negative", "direct-assistant",
    "greeting-forms", "compound-dev", "context-dev", "name-dev", "social-dev", "unsupported-dev",
])
def test_development_scenario_meanings(dialogue_inputs, scenario_id):
    data, memory = dialogue_inputs
    scenario = next(s for s in data["scenarios"] if s["id"] == scenario_id)
    assert scenario["split"] == "development"
    rows = run_cases(memory, [scenario])
    failed = [{"prompt": r["prompt"], "answer": r["answer"], "failures": r["failures"]}
              for r in rows if not r["passed"]]
    assert not failed, json.dumps(failed, ensure_ascii=False, indent=2)


def test_original_composition_is_new_and_wave_readout_is_checked(dialogue_inputs):
    from freqai.dialogue import respond

    _, memory = dialogue_inputs
    first, context = respond(memory, "Hallo")
    original_context = context.copy()
    result, updated = respond(memory, "Mir geht es gut, wie geht es dir denn?", context)
    assert context == original_context, "Respond must not mutate its caller's context"
    assert not result["abstained"]
    assert {"user_mood", "assistant_status"}.issubset(result["response_acts"])
    assert result["answer"] not in {d.text for d in memory.documents}
    assert result["generated"]
    decoder = result["decoder"]
    assert decoder["source_integrity_checked"]
    assert decoder["output_sha256"] == hashlib.sha256(result["answer"].encode("utf-8")).hexdigest()
    assert decoder["fragments"] >= 2
    assert updated["turn_count"] > context["turn_count"]


def test_separate_contexts_do_not_share_names(dialogue_inputs):
    from freqai.dialogue import respond

    _, memory = dialogue_inputs
    _, named = respond(memory, "Ich heiße Clara.")
    result, named_after = respond(memory, "Wie heiße ich?", named)
    assert "clara" in result["answer"].casefold()
    anonymous_result, anonymous = respond(memory, "Wie heiße ich?")
    assert "clara" not in anonymous_result["answer"].casefold()
    assert not anonymous.get("user_name")
    assert named_after.get("user_name") == "Clara"


def test_composition_time_invariance_over_long_runtime(dialogue_inputs):
    from freqai.dialogue import respond

    _, memory = dialogue_inputs
    _, context = respond(memory, "Hallo")
    prompt = "Hallo, mir geht es gut. Und dir?"
    baseline, baseline_context = respond(memory, prompt, context, time_s=0)
    for time_s in (.013, 1.25, 86400.125, 1e12):
        result, updated = respond(memory, prompt, context, time_s=time_s)
        assert result["answer"] == baseline["answer"]
        assert updated == baseline_context
        assert result["decoder"]["output_sha256"] == baseline["decoder"]["output_sha256"]
