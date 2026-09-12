import json
from pathlib import Path

import pytest

from freqai.dialogue import respond
from freqai.memory import Document, WaveMemory


@pytest.fixture
def memory():
    path = Path(__file__).resolve().parents[1] / "memory/fixtures/extension_120.jsonl"
    return WaveMemory([Document(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()],
                      feature_mode="morphology", retrieval_policy="coverage")


def test_semantic_catalog_refreshes_after_live_append(memory):
    first, _ = respond(memory, "Servus")
    changed = memory.with_documents_added([Document("hello-live", "Servus! Wir können plaudern.", prompt="Servus")])
    second, _ = respond(changed, "Servus")
    assert first["answer"] != second["answer"]
    assert second["answer"] == "Servus! Wir können plaudern."
    assert second["matches"][0]["id"] == "hello-live"


@pytest.mark.parametrize("prompt", ["Ich bin nicht glücklich", "Ich bin traurig", "Mir geht es miserabel"])
def test_negative_mood_never_receives_positive_response(memory, prompt):
    result, _ = respond(memory, prompt)
    assert "user_mood" in result["response_acts"]
    assert "Das klingt schön" not in result["answer"]


def test_topic_switch_expires_elliptical_wellbeing(memory):
    _, state = respond(memory, "Hallo")
    _, state = respond(memory, "Ich möchte über Musik reden", state)
    result, _ = respond(memory, "Und dir?", state)
    assert result["abstained"]
    assert "assistant_status" not in result["response_acts"]


def test_unknown_fact_never_uses_shared_question_words_as_evidence(memory):
    result, _ = respond(memory, "Wie viele Monde hat der Planet Xylophon?")
    assert result["abstained"]
    assert result["response_acts"] == ["unknown"]


def test_named_person_is_not_the_users_feeling(memory):
    result, _ = respond(memory, "Meiner Mutter geht es schlecht.")
    assert "user_mood" not in result["response_acts"]
    assert "Meiner Mutter geht es schlecht" in result["answer"]
    assert "third_mood" in result["response_acts"]


def test_pet_statement_is_attributed_without_calling_pet_person(memory):
    prompt = "Meinem Hund geht es schlecht."
    result, _ = respond(memory, prompt)
    assert "Person" not in result["answer"]
    assert "Meinem Hund geht es schlecht" in result["answer"]
    assert "input:prompt" in {source["source"] for source in result["decoder"]["sources"]}


@pytest.mark.parametrize("prompt,adjective", [("Ich bin nicht traurig", "traurig"),
                                           ("Heute bin ich nicht müde", "müde")])
def test_negated_predicate_is_explicit_without_asserting_opposite(memory, prompt, adjective):
    result, _ = respond(memory, prompt)
    assert "nicht " + adjective in result["answer"]
    assert "glücklich" not in result["answer"]
    assert "voller Energie" not in result["answer"]


@pytest.mark.parametrize("prompt,forbidden", [("Ich bin gestresst", "Einstieg"),
                                            ("Ich bin traurig", "erhofft")])
def test_mood_response_does_not_import_source_situation(memory, prompt, forbidden):
    result, _ = respond(memory, prompt)
    assert "user_mood" in result["response_acts"]
    assert forbidden not in result["answer"]


def test_supported_clause_does_not_hide_unknown_fact(memory):
    result, _ = respond(memory, "Hallo. Wie viele Monde hat Jupiter?")
    assert "greeting" in result["response_acts"]
    assert "unknown" in result["response_acts"]
    assert "keine verlässliche Information" in result["answer"]


def test_explicit_factual_evidence_remains_available():
    memory = WaveMemory([Document("nova", "Die Wartung der Testanlage Nova erfolgt jeden Montag um 10 Uhr.")])
    result, _ = respond(memory, "Wann erfolgt die Wartung der Testanlage Nova?")
    assert result["matches"][0]["id"] == "nova"
    assert "Montag" in result["answer"]


def test_factual_top_k_contract_is_preserved():
    memory = WaveMemory([Document("a", "Kupfer ist ein Metall."), Document("b", "Kupfer ist ein guter Leiter.")])
    legacy = memory.ask("Kupfer", top_k=2)
    result, _ = respond(memory, "Kupfer", top_k=2)
    assert result["answer"] == legacy["answer"]
    assert result["matches"] == legacy["matches"]
    with pytest.raises(ValueError, match="top_k"):
        respond(memory, "Kupfer", top_k=999)


def test_catalog_refreshes_for_public_mutating_append():
    memory = WaveMemory([Document("greeting", "Hallo!", prompt="Hallo")])
    respond(memory, "Ich bin super")
    memory.add_document("Eine neue positive Antwort.", prompt="Mir geht es prima")
    result, _ = respond(memory, "Ich bin gut")
    assert result["answer"] == "Eine neue positive Antwort."


def test_unknown_question_invalidates_previous_reciprocal_topic(memory):
    _, state = respond(memory, "Hallo")
    _, state = respond(memory, "Wie viele Monde hat Jupiter?", state)
    result, _ = respond(memory, "Und dir?", state)
    assert result["abstained"]
    assert "assistant_status" not in result["response_acts"]
