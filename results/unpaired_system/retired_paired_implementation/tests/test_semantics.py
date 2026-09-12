import dataclasses

import numpy as np
import pytest

from freqai.semantics import SemanticAct, analyze, semantic_similarity, semantic_spectrum


def kinds(result):
    return [act.kind for act in result.acts]


@pytest.mark.parametrize("prompt", [
    "Mir geht es gut, wie geht es dir denn?", "gut und dir?",
    "Ich fühle mich wunderbar und wie fühlst du dich?",
    "Mir geht es prima. Und bei dir?", "Ich bin zufrieden und du?",
])
def test_mood_plus_reciprocal_is_compositional(prompt):
    result = analyze(prompt)
    assert kinds(result) == ["mood_statement", "wellbeing_question"]
    assert result.acts[0].target == "user"
    assert result.acts[0].value == "positive"
    assert result.acts[1].target == "assistant"
    assert result.question_topic == "wellbeing"
    assert not result.unsupported


@pytest.mark.parametrize("prompt", ["und dir?", "Und du?", "Bei dir?"])
def test_reciprocal_requires_context(prompt):
    absent = analyze(prompt)
    assert kinds(absent) == ["context_clarification"]
    assert absent.unsupported
    unrelated = analyze(prompt, {"last_question_topic": "weather"})
    assert kinds(unrelated) == ["context_clarification"]
    present = analyze(prompt, {"last_question_topic": "wellbeing"})
    assert kinds(present) == ["wellbeing_question"]
    assert present.used_context


@pytest.mark.parametrize(("text", "mood", "negated"), [
    ("Mir geht es nicht gut.", "negative", True),
    ("Ich bin nicht glücklich.", "negative", True),
    ("Mir geht es nicht schlecht.", "neutral", True),
    ("Ich bin nicht mehr traurig.", "neutral", True),
    ("Ich bin keineswegs müde.", "neutral", True),
    ("Ich fühle mich ziemlich einsam.", "lonely", False),
    ("Ich habe Stress.", "stressed", False),
    ("Ich bin erschöpft.", "tired", False),
    ("Ich bin aufgeregt.", "excited", False),
])
def test_negation_and_moods(text, mood, negated):
    result = analyze(text)
    assert len(result.acts) == 1
    assert result.acts[0].value == mood
    assert result.acts[0].negated is negated
    assert result.acts[0].target == "user"


@pytest.mark.parametrize("text", ["Anna ist traurig.", "Er ist müde.", "Ihr geht es schlecht."])
def test_other_person_does_not_become_user(text):
    result = analyze(text)
    assert result.acts[0].target == "other"
    assert "mood_statement" in kinds(result)


@pytest.mark.parametrize("text", ["Du bist traurig.", "Dir geht es schlecht."])
def test_assistant_predicates_keep_their_target(text):
    result = analyze(text)
    assert result.acts[0].target == "assistant"


def test_negation_does_not_cross_clauses_or_people():
    result = analyze("Ich bin nicht müde, aber Anna ist traurig und wie geht es dir?")
    assert [(a.kind, a.target, a.value, a.negated) for a in result.acts] == [
        ("mood_statement", "user", "neutral", True),
        ("mood_statement", "other", "negative", False),
        ("wellbeing_question", "assistant", "", False),
    ]


@pytest.mark.parametrize("text", [
    'Anna sagt "ich bin traurig".', 'Wenn ich traurig bin.',
    'Er behauptet ich bin müde.', 'Ein gutes Buch.',
])
def test_quotes_hypotheticals_and_noun_modifiers_do_not_assert_user_mood(text):
    result = analyze(text)
    assert not any(a.kind == "mood_statement" and a.target == "user" for a in result.acts)


def test_named_memory_and_multiple_intents():
    result = analyze("Hallo, ich heiße Lea. Wie heiße ich? Danke!")
    assert kinds(result) == ["greeting", "name_statement", "name_question", "thanks"]
    assert result.acts[1].value == "Lea"
    assert result.acts[2].target == "user"
    assert result.to_dict()["acts"][1]["value"] == "Lea"


def test_unknown_clauses_stay_visible_next_to_recognized_clauses():
    result = analyze("Hallo! Wie viele Monde hat Jupiter?")
    assert kinds(result) == ["greeting", "question"]
    assert result.unsupported == ["Wie viele Monde hat Jupiter?"]


def test_known_topic_and_open_question_are_separate():
    talk = analyze("Lass uns über Musik reden.")
    assert kinds(talk) == ["talk_request"]
    assert talk.acts[0].topic == "music"
    question = analyze("Was kann ich heute kochen?")
    assert kinds(question) == ["question"]
    assert question.acts[0].topic == "food"


def test_empty_input_and_invalid_type():
    assert analyze(" \n ").acts == []
    with pytest.raises(TypeError):
        analyze(None)


def test_wording_does_not_change_semantic_wave():
    a = analyze("Ich bin glücklich.").acts[0]
    b = analyze("Mir geht es prima.").acts[0]
    np.testing.assert_array_equal(semantic_spectrum(a), semantic_spectrum(b))
    assert semantic_similarity(a, b) == pytest.approx(1.0)


def test_fourier_roundtrip_preserves_role_bound_concepts():
    act = SemanticAct("mood_statement", "user", "positive", "wellbeing")
    spectrum = semantic_spectrum(act)
    oscillation = np.fft.ifft(spectrum, norm="ortho")
    np.testing.assert_allclose(np.fft.fft(oscillation, norm="ortho"), spectrum, atol=1e-15)
    assert np.linalg.norm(spectrum) == pytest.approx(1)
    assert np.linalg.norm(oscillation) == pytest.approx(1)


def test_target_and_value_roles_cannot_cross_talk():
    a = SemanticAct("mood_statement", "user", "positive", "wellbeing")
    b = dataclasses.replace(a, target="assistant")
    c = dataclasses.replace(a, value="negative")
    av, bv, cv = (semantic_spectrum(x) for x in (a, b, c))
    width = len(av) // 5
    assert semantic_similarity(a, b) == pytest.approx(0.8)
    assert semantic_similarity(a, c) == pytest.approx(0.8)
    # A perturbation in the target role has zero support in the value role.
    delta = bv - av
    np.testing.assert_array_equal(delta[2 * width:3 * width], np.zeros(width))
    np.testing.assert_allclose(np.vdot(delta, cv - av), 0, atol=1e-15)
    assert semantic_similarity(a, dataclasses.replace(c, target="assistant")) == pytest.approx(0.6)


def test_negation_is_explicit_in_the_spectrum():
    act = SemanticAct("mood_statement", "user", "neutral", "wellbeing")
    assert semantic_similarity(act, dataclasses.replace(act, negated=True)) == pytest.approx(0.8)


def test_open_vocabulary_is_deterministic_and_bound_to_its_role():
    a = SemanticAct("name_statement", value="Zoe")
    spectrum = semantic_spectrum(a)
    np.testing.assert_array_equal(spectrum, semantic_spectrum(a))
    assert semantic_similarity(a, dataclasses.replace(a, value="Mila")) < 1
    with pytest.raises(ValueError):
        semantic_spectrum(a, 8)


@pytest.mark.parametrize("prompt", [
    "Bist du morgen in Berlin?", "Bist du ein Roboter?", "Fühlst du dich beobachtet?",
])
def test_inverted_questions_need_an_affect_predicate(prompt):
    assert "wellbeing_question" not in kinds(analyze(prompt))


@pytest.mark.parametrize("prompt", ["Bist du müde?", "Fühlst du dich heute gut?", "Geht es dir gut?"])
def test_yes_no_wellbeing_questions(prompt):
    result = analyze(prompt)
    assert kinds(result) == ["wellbeing_question"]
    assert result.acts[0].target == "assistant"


@pytest.mark.parametrize(("prompt", "value", "target"), [
    ("Meine Stimmung ist heute etwas gedrückt.", "negative", "user"),
    ("Deine Stimmung ist gedrückt.", "negative", "assistant"),
    ("Meine Schwester ist traurig.", "negative", "other"),
    ("Meiner Schwester geht es schlecht.", "negative", "other"),
    ("Deiner Mutter geht es gut.", "positive", "other"),
    ("Es geht Anna gut.", "positive", "other"),
    ("Ich fühle mich wohl.", "positive", "user"),
    ("Bei mir ist alles in Ordnung.", "positive", "user"),
    ("Ich freue mich gerade.", "positive", "user"),
    ("Ich habe gerade keine gute Laune.", "negative", "user"),
    ("Ich bin nicht guter Laune.", "negative", "user"),
    ("Ich bin nicht schlechter Laune.", "neutral", "user"),
])
def test_possessive_roles_and_compositional_affect_predicates(prompt, value, target):
    result = analyze(prompt)
    assert len(result.acts) == 1
    assert result.acts[0].kind == "mood_statement"
    assert result.acts[0].value == value
    assert result.acts[0].target == target


@pytest.mark.parametrize(("prompt", "kind"), [("Huhu!", "greeting"), ("Vielen lieben Dank!", "thanks")])
def test_colloquial_greeting_and_composed_thanks(prompt, kind):
    assert kinds(analyze(prompt)) == [kind]


@pytest.mark.parametrize("other_clause", [
    "meine Mutter ist traurig", "Anna ist traurig", "der Nachbar ist traurig",
    "meiner Schwester geht es schlecht", "dein Bruder ist traurig",
])
def test_conjunction_with_nominal_subject_keeps_distinct_people(other_clause):
    result = analyze("Ich bin gut und " + other_clause)
    assert [(a.target, a.value) for a in result.acts] == [
        ("user", "positive"), ("other", "negative")]


def test_uppercase_conjunctions_and_question_are_equivalent():
    result = analyze("ICH BIN GUT UND DIR?")
    assert kinds(result) == ["mood_statement", "wellbeing_question"]
    assert result.acts[1].target == "assistant"


def test_conjoined_predicates_do_not_get_mistaken_for_a_new_person():
    result = analyze("Ich bin traurig und müde.")
    assert [(a.target, a.value) for a in result.acts] == [
        ("user", "negative"), ("user", "tired")]


@pytest.mark.parametrize(("prompt", "value", "predicate"), [
    ("Gut geht es mir nicht.", "negative", "positive"),
    ("Müde bin ich heute nicht.", "neutral", "tired"),
    ("Zufrieden bin ich wirklich nicht.", "negative", "positive"),
])
def test_fronted_predicates_preserve_postposed_negation(prompt, value, predicate):
    result = analyze(prompt)
    assert len(result.acts) == 1
    assert result.acts[0].value == value
    assert result.acts[0].negated
    assert result.acts[0].predicate == predicate


@pytest.mark.parametrize("prompt", [
    "Mir geht es schlecht, nein, eigentlich gut.",
    "Ich bin traurig; nein eigentlich glücklich.",
    "Mir geht es nicht gut sondern prima.",
])
def test_local_self_correction_replaces_retracted_state(prompt):
    result = analyze(prompt)
    assert [(a.kind, a.target, a.value) for a in result.acts] == [
        ("mood_statement", "user", "positive")]


def test_contrastive_subject_correction_and_predicate_ellipsis():
    result = analyze("Nicht Anna ist müde, sondern ich.")
    assert [(a.target, a.value) for a in result.acts] == [("user", "tired")]
    result = analyze("Anna ist nicht müde, ich aber schon.")
    assert [(a.target, a.value, a.negated, a.predicate) for a in result.acts] == [
        ("other", "neutral", True, "tired"), ("user", "tired", False, "tired")]


def test_unknown_alternative_is_not_a_positive_assertion():
    result = analyze("Mir geht es gut oder vielleicht auch nicht.")
    assert kinds(result) == ["context_clarification"]
    assert result.acts[0].value == "uncertain"


@pytest.mark.parametrize("prompt", ["Du auch?", "Dir vielleicht?", "Wie sieht es bei dir aus?", "Geht es dir auch so?"])
def test_anaphoric_question_forms_require_an_affect_antecedent(prompt):
    assert kinds(analyze(prompt)) == ["context_clarification"]
    result = analyze(prompt, {"last_question_topic": "wellbeing"})
    assert kinds(result) == ["wellbeing_question"]
    assert result.used_context


@pytest.mark.parametrize("prompt", [
    "Ach hallo, also bei mir läuft es prima, und wie geht es dir denn heute?",
    "Hallo, mir geht’s gut… wie geht’s dir?",
    "Hallo, gut & dir?",
])
def test_particles_and_written_conjunction_variants(prompt):
    assert kinds(analyze(prompt)) == ["greeting", "mood_statement", "wellbeing_question"]


@pytest.mark.parametrize("prompt", [
    "Danke der Nachfrage!", "Danke fürs Zuhören.", "Vielen Dank für den ruhigen Austausch.",
    "Danke schon mal.", "Danke, dass du da bist.",
])
def test_thanks_accepts_grammatical_complements(prompt):
    assert kinds(analyze(prompt)) == ["thanks"]


@pytest.mark.parametrize(("prompt", "name"), [
    ("Nenn mich bitte Lukas.", "Lukas"), ("Bitte nenne mich Maria.", "Maria"),
    ("Du kannst mich Aylin nennen.", "Aylin"), ("Nenn mich ab jetzt Otto.", "Otto"),
])
def test_name_request_has_explicit_naming_verb(prompt, name):
    result = analyze(prompt)
    assert kinds(result) == ["name_statement"]
    assert result.acts[0].value == name
    assert not any(a.kind == "name_statement" for a in analyze("Ich bin froh.").acts)
    assert not any(a.kind == "name_statement" for a in analyze("Ich heiße nicht Lukas.").acts)


@pytest.mark.parametrize("prompt", [
    "Kannst du dich noch an meinen Namen erinnern?", "Merkst du dir meinen Namen?",
    "Welchen Namen habe ich dir genannt?",
])
def test_name_recall_with_memory_verbs(prompt):
    assert kinds(analyze(prompt)) == ["name_question"]


def test_indirect_question_keeps_addressee_role():
    result = analyze("Darf ich fragen, wie du dich heute fühlst?")
    assert kinds(result) == ["wellbeing_question"]
    assert result.acts[0].target == "assistant"


@pytest.mark.parametrize(("prompt", "subject"), [
    ("Anna geht es schlecht.", "Anna"), ("Meinem Hund geht es gut.", "deinem Hund"),
    ("Meine Mutter ist traurig.", "deiner Mutter"), ("Er ist müde.", ""),
])
def test_third_party_surface_slot_is_explicit_and_not_assumed_human(prompt, subject):
    result = analyze(prompt)
    assert result.acts[0].subject == subject
    assert result.acts[0].predicate


def test_grounded_surface_metadata_does_not_add_matching_authority():
    original = SemanticAct("mood_statement", "other", "negative", "wellbeing")
    supplied = dataclasses.replace(original, subject="Anna", predicate="negative")
    np.testing.assert_array_equal(semantic_spectrum(original), semantic_spectrum(supplied))


def test_subject_focus_negation_does_not_assert_the_denied_predicate():
    result = analyze("Nicht Anna ist müde.")
    assert result.acts[0].negated
    assert result.acts[0].target == "other"
    assert result.acts[0].value == "neutral"
    additive = analyze("Nicht nur Anna ist müde.")
    assert not additive.acts[0].negated


def test_explicit_counterfactual_positive_wish_is_not_positive_reality():
    prompt = "Ich wünschte, mir ginge es gut."
    result = analyze(prompt)
    assert kinds(result) == ["mood_statement"]
    assert result.acts[0].value == "negative"
    assert result.acts[0].text == prompt


@pytest.mark.parametrize("prompt", ["Und bei dir so?", "Wie ist es bei dir?", "Bist du es?"])
def test_additional_anaphoric_grammars_are_context_bound(prompt):
    assert kinds(analyze(prompt)) == ["context_clarification"]
    assert kinds(analyze(prompt, {"last_question_topic": "wellbeing"})) == ["wellbeing_question"]


@pytest.mark.parametrize(("prompt", "expected"), [
    ("Hallo und vielen Dank für deine Zeit!", ["greeting", "thanks"]),
    ("Danke für das Gespräch und auf Wiedersehen!", ["thanks", "farewell"]),
    ("Müde bin ich, aber danke der Nachfrage.", ["mood_statement", "thanks"]),
    ("Mir geht es schlecht. Ich danke dir, dass du zuhörst.", ["mood_statement", "thanks"]),
])
def test_social_act_coordination_and_subordinate_complements(prompt, expected):
    result = analyze(prompt)
    assert kinds(result) == expected
    assert not result.unsupported


@pytest.mark.parametrize(("prompt", "name"), [
    ("Ich möchte lieber Emil genannt werden.", "Emil"),
    ("Ich würde gern Amina angesprochen werden.", "Amina"),
])
def test_name_preference_uses_explicit_passive_naming_verb(prompt, name):
    result = analyze(prompt)
    assert kinds(result) == ["name_statement"]
    assert result.acts[0].value == name


def test_epistemically_qualified_states_do_not_become_confident_assertions():
    result = analyze("Vielleicht geht es mir gut, vielleicht auch schlecht.")
    assert all(a.kind == "context_clarification" and a.value == "uncertain" for a in result.acts)
    assert not any(a.kind == "mood_statement" for a in result.acts)
