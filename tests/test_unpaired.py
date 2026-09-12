"""Independent invariants for declaration-only speech and coefficient coupling."""
import copy
import json

import numpy as np
import pytest

from freqai.unpaired import UnpairedPrompt, UnpairedWaveModel


DECLARATIONS = [
    {"text":"Der Assistent heißt Oszillo. Der Assistent ist ein Rechenprogramm. Der Assistent hat keine eigenen Gefühle. Der Assistent hat kein menschliches Befinden. Der Assistent kann Texte beschreiben."},
    {"text":"Das Wort Hallo ist ein Begrüßungswort. Das Wort Hi ist ein Begrüßungswort. Das Wort Danke ist ein Dankeswort. Das Wort Bitte ist ein Höflichkeitswort. Das Wort Ciao ist ein Abschiedswort."},
    {"text":"Das Befinden ist eine persönliche Wahrnehmung."},
    {"text":"Die Frequenz ist die Anzahl der Wiederholungen pro Sekunde."},
    {"text":"Das Signal Lumor hat eine Frequenz von 19 Hertz."},
    {"text":"Das Signal Tavor hat eine Frequenz von 73 Hertz."},
]


@pytest.fixture
def model():
    return UnpairedWaveModel(DECLARATIONS)


def test_pairs_rejected_at_compiler_boundary():
    with pytest.raises(ValueError,match="pairs"):
        UnpairedWaveModel([{"prompt":"Hallo", "text":"Eine Begrüßung ist ein Sprechakt."}])


def test_empty_corpus_cannot_answer_or_echo_prompt_facts():
    model = UnpairedWaveModel([])
    for prompt in ("Hallo", "Wie heißt du?", "Ich heiße Larion.", "Was ist eine Frequenz?"):
        output = model.generate(prompt)
        assert output["tokens"]==[]
        assert output["reason"]=="empty_information_corpus"


def test_user_reciprocal_question_uses_declared_assistant_fact(model):
    greeting = model.generate("Hallo")
    answer = model.generate("Mir geht es gut, wie geht es dir denn?",context=greeting["state"])
    assert "kein menschliches Befinden" in answer["text"]
    assert "ich habe" in answer["text"].casefold()
    assert "gut" in answer["state"]["user_facts"]["state"]
    assert answer["state"]["user_facts"]["state_scope"]=="session_prompt_only"
    answer2 = model.generate("gut und dir?",context=answer["state"])
    assert "kein menschliches Befinden" in answer2["text"]
    assert all(step["runtime_source"]=="complex_declaration_roles_and_explicit_grammar" for step in answer2["trace"])


def test_greeting_question_is_grammar_composition_not_a_source_sentence(model):
    output = model.generate("Hallo")
    assert output["text"]=="Hallo! Wie geht es dir?"
    assert all(output["text"].casefold() not in row["text"].casefold() for row in DECLARATIONS)
    assert output["uses_answer_candidates"] is False


@pytest.mark.parametrize("prompt,fragment",[("Wie heißt du?","Oszillo"),("Was kannst du?","Texte beschreiben"),
    ("Hast du Gefühle?","keine eigenen Gefühle"),("Danke","Bitte"),("Ciao","Ciao")])
def test_social_utterances_come_from_declarative_roles(model,prompt,fragment):
    result = model.generate(prompt)
    assert fragment in result["text"]
    assert result["ended"]
    assert result["trace"]


def test_modified_name_declaration_changes_emitted_name():
    first = UnpairedWaveModel(DECLARATIONS)
    changed = copy.deepcopy(DECLARATIONS)
    changed[0]["text"] = changed[0]["text"].replace("Oszillo","Kymara")
    second = UnpairedWaveModel(changed)
    assert "Oszillo" in first.generate("Wie heißt du?")["text"]
    answer = second.generate("Wie heißt du?")["text"]
    assert "Kymara" in answer and "Oszillo" not in answer
    assert first.model_signature()["declaration_role_sha256"] != second.model_signature()["declaration_role_sha256"]


def test_lexical_category_is_derived_from_corpus_not_a_hallo_table():
    altered = [{"text":"Das Wort Saluton ist ein Begrüßungswort. Das Befinden ist eine Wahrnehmung."}]
    model = UnpairedWaveModel(altered)
    assert model.generate("Saluton")["text"].startswith("Saluton!")
    assert model.generate("Hallo")["text"]==""


def test_unknown_name_has_lossless_byte_modes_and_is_session_scoped(model):
    signature = model.model_signature()
    output = model.generate("Ich heiße Xyrländo.")
    assert output["text"]=="Du heißt Xyrländo."
    assert any(token.startswith("<byte:") for token in output["tokens"])
    assert model.generate("Wie heiße ich?",context=output["state"])["text"]==output["text"]
    assert model.generate("Wie heiße ich?",context={})["text"]==""
    assert model.model_signature()["coefficient_sha256"]==signature["coefficient_sha256"]
    assert model.model_signature()["declaration_role_sha256"]==signature["declaration_role_sha256"]
    assert not any(role.origin=="session_prompt_only" for role in model.lexical_roles)
    assert "Xyrländo" not in model.index
    json.dumps(output,ensure_ascii=False)


def test_partial_utf8_at_symbol_budget_keeps_only_valid_prefix(model):
    output = model.generate("Ich heiße Ö.",max_tokens=3)
    assert output["ended"] is False
    assert output["tokenization"]["utf8_truncated_bytes"]==1
    assert output["tokenization"]["utf8_complete"] is False
    assert "\ufffd" not in output["text"]
    assert output["text"]=="Du heißt"


def test_interior_invalid_utf8_is_reported_as_decoder_error(model):
    with pytest.raises(ValueError,match="Invalid UTF-8"):
        model.render(["<byte:ff>","."])


@pytest.mark.parametrize("field",["question","answer","messages","instruction","input","response"])
def test_direct_constructor_rejects_qa_and_dialogue_fields(field):
    with pytest.raises(ValueError,match="question/answer"):
        UnpairedWaveModel([{"text":"Die Frequenz ist ein Messwert.",field:[]}])


def test_new_user_state_does_not_echo_stale_state(model):
    first = model.generate("Mir geht es gut.")
    second = model.generate("Ich bin müde.",context=first["state"])
    assert second["text"]=="Du bist müde."
    assert model.generate("Wie geht es mir?",context=second["state"])["text"]=="Du bist müde."
    assert model.generate("Wie geht es mir?",context={})["text"]==""


def test_negated_user_fact_preserved_without_negating_corpus(model):
    answer = model.generate("Ich bin nicht müde.")
    assert "nicht müde" in answer["text"]
    assert model.generate("Ist Lumor nicht 19 Hertz?")["text"]==""


def test_declarative_attributes_support_forward_and_inverse_binding(model):
    forward = model.generate("Welche Frequenz hat das Signal Lumor?")
    inverse = model.generate("Welches Signal hat eine Frequenz von 73 Hertz?")
    assert "Lumor" in forward["text"] and "19 Hertz" in forward["text"]
    assert "Tavor" in inverse["text"] and "73 Hertz" in inverse["text"]
    assert "Lumor" not in inverse["text"]


def test_permuted_numeric_attributes_change_both_directions():
    changed = copy.deepcopy(DECLARATIONS)
    changed[-2]["text"] = "Das Signal Lumor hat eine Frequenz von 73 Hertz."
    changed[-1]["text"] = "Das Signal Tavor hat eine Frequenz von 19 Hertz."
    model = UnpairedWaveModel(changed)
    assert "73 Hertz" in model.generate("Welche Frequenz hat das Signal Lumor?")["text"]
    assert "Tavor" in model.generate("Welches Signal hat eine Frequenz von 19 Hertz?")["text"]


def test_role_field_direct_fft_and_time_invariance(model):
    prompts = ["Hallo", "Wie heißt du?", "Ich bin müde.", "Welche Frequenz hat das Signal Lumor?"]
    states = 0
    for prompt in prompts:
        field,_ = model.prompt_field(prompt)
        assert isinstance(field,UnpairedPrompt)
        output = model.generate(prompt)
        for prefix_length in range(len(output["tokens"])+1):
            prefix = output["tokens"][:prefix_length]
            direct,_ = model.next_distribution(prefix,field,method="direct")
            fft,trace = model.next_distribution(prefix,field,method="operator",time_s=86400.125)
            assert np.array_equal(direct,fft)
            assert trace["all_supported_modes_included"]
            assert trace["fft_roundtrip_max_error"]<1e-10
            states += 1
    assert states>=25


def test_phase_changes_real_lexical_selection(model):
    field,_ = model.prompt_field("Hallo")
    baseline,_ = model.next_distribution([],field)
    phase,_ = model.next_distribution([],field,phase_error=np.pi)
    assert np.argmax(baseline)!=np.argmax(phase)
    assert np.abs(baseline-phase).sum()/2>.5


def test_zero_prompt_field_stops_role_grammar(model):
    field,_ = model.prompt_field("Hallo")
    with pytest.raises(ValueError,match="wave"):
        model.next_distribution([],field*0)


def test_zero_data_coefficients_stop_social_and_information_answers(model):
    for role in model.lexical_roles:
        role.spectrum[:]=0
    for spectrum in model.transition_spectra.values():
        spectrum[:]=0
    for prompt in ("Hallo", "Wie heißt du?", "Was ist eine Frequenz?"):
        output = model.generate(prompt)
        assert output["text"]==""
        assert output["tokens"]==[]


def test_definition_still_uses_same_complex_information_operator(model):
    result = model.generate("Was ist eine Frequenz?")
    assert "Wiederholungen" in result["text"]
    assert result["trace"][0]["runtime_source"]=="imported_complex_information_transition_spectra"
    assert result["uses_answer_candidates"] is False
    assert model.storage_stats()["optimizer_steps"]==0


def test_possessive_state_and_directed_question_compose_separate_roles(model):
    result = model.generate("Mein Vormittag war angenehm. Wie geht es dir?")
    assert result["text"].startswith("Dein Vormittag war angenehm.")
    assert "kein menschliches Befinden" in result["text"]
    assert "owned_state" in result["state"]["user_facts"]


def test_inverted_and_shared_subject_clauses_preserve_user_state(model):
    assert "erschöpft" in model.generate("Ich komme an und bin erschöpft.")["text"]
    assert "unruhig" in model.generate("Gerade bin ich unruhig.")["text"]


def test_inverted_negative_desire_preserves_fronted_argument(model):
    result = model.generate("Lärm möchte ich nicht.")
    assert "Lärm" in result["text"] and "nicht" in result["text"]
    assert result["state"]["user_facts"]["preference_scope"]=="session_prompt_only"


def test_compound_lexical_membership_matches_whole_expression():
    model = UnpairedWaveModel([{"text":"Die Wendung Guten Abend ist eine Begrüßungsformel."}])
    assert "Guten Abend" in model.generate("Guten Abend, alle zusammen.")["text"]
    assert model.generate("Guten Käse mag ich.")["analysis"].get("speech_act")!="reciprocal_greeting"


def test_compound_expression_preserves_its_declared_noun_spelling():
    model = UnpairedWaveModel([{"text":"Die Wendung Guten Morgen ist eine Begrüßungsformel. Das Wort morgen ist ein Zeitwort. Das Wort morgen bezeichnet den folgenden Tag."}])
    assert model.generate("Guten Morgen")["text"]=="Guten Morgen!"


def test_function_grammar_and_same_named_entity_have_distinct_spelling():
    model = UnpairedWaveModel([{"text":"Die Bist ist ein Fluss. Die Bist hat eine Länge von 20 Kilometern."},
                               {"text":"Der Assistent ist ein Programm."}])
    assert "Bist" in model.generate("Was ist die Bist?")["text"]
    assert "Bist" in model.generate("Welche Länge hat die Bist?")["text"]
    assert model.generate("Ich bin müde.")["text"].startswith("Du bist ")


def test_definition_wrapper_is_not_misread_as_assistant_capability(model):
    result = model.generate("Kannst du mir erklären, was eine Frequenz ist?")
    assert "Wiederholungen" in result["text"]
    assert "ich kann" not in result["text"].casefold()


def test_property_owner_and_type_are_bound_from_separate_declarations():
    records = [{"text":"Morava ist ein periodisches Lichtsignal. Die Frequenz von Morava beträgt 37 Hertz."},
               {"text":"Nerovo ist ein periodisches Lichtsignal. Die Frequenz von Nerovo beträgt 91 Hertz."}]
    model = UnpairedWaveModel(records)
    assert "37 Hertz" in model.generate("Welche Frequenz hat das Lichtsignal Morava?")["text"]
    assert "Nerovo" in model.generate("Welches Lichtsignal hat eine Frequenz von 91 Hertz?")["text"]


def test_activity_argument_question_uses_a_declared_action():
    model = UnpairedWaveModel([{"text":"Zeichnen ist das Darstellen von Formen."}])
    answer = model.generate("Ich möchte etwas zeichnen.")
    assert answer["text"]=="Was möchtest du zeichnen?"
    assert answer["analysis"]["speech_act"]=="clarify_activity_object"


def test_polite_constraint_keeps_negation_as_session_information():
    model = UnpairedWaveModel([{"text":"Das Wort Bitte ist ein Höflichkeitswort. Das Wort Einverstanden ist ein Zustimmungswort."}])
    answer = model.generate("Bitte keine langen Aufzählungen.")
    assert answer["text"]=="Einverstanden."
    assert answer["state"]["user_facts"]["conversation_constraints"]==["Bitte keine langen Aufzählungen."]
