"""Independent prose-only correctness checks; no chat/knowledge evaluation data."""
import copy

import numpy as np
import pytest

from freqai.information import InformationWaveModel, tokenize


PROSE = [
    {"id":"f","title":"Frequenz","text":"Die Frequenz ist die Anzahl vollständiger Schwingungen pro Sekunde. "
     "Die Frequenz wird in Hertz gemessen. Für die Frequenz gilt f = 1 / T. "
     "Die Frequenz ist der Kehrwert der Periodendauer."},
    {"id":"t","title":"Periodendauer","text":"Die Periodendauer ist die Dauer einer vollständigen Schwingung. "
     "Die Einheit der Periodendauer ist die Sekunde."},
    {"id":"u","title":"Spannung","text":"Die Spannung ist eine elektrische Potentialdifferenz. "
     "Die Spannung wird in Volt gemessen."},
]


def test_prose_without_any_question_answer_pairs_supports_facets_and_followup():
    model = InformationWaveModel(PROSE)
    definition = model.generate("Was ist eine Frequenz?")
    assert definition["ended"] and "Frequenz" in definition["text"]
    assert "Volt" not in definition["text"]
    unit = model.generate("Und welche Einheit hat sie?", context=definition["information_state"])
    assert unit["text"] == "Die Frequenz wird in Hertz gemessen."
    assert unit["analysis"]["used_context"]
    formula = model.generate("Wie berechnet man die Frequenz?")
    assert formula["ended"] and ("=" in formula["text"] or "Kehrwert" in formula["text"])
    for result in (definition, unit, formula):
        assert not result["uses_answer_candidates"]
        assert all(step["runtime_source"]=="imported_complex_information_transition_spectra" for step in result["trace"])


def test_title_header_is_metadata_and_not_a_one_word_response():
    plain = InformationWaveModel([{"title":"Frequenz","text":PROSE[0]["text"]}])
    with_header = InformationWaveModel([{"title":"Frequenz","text":"Frequenz\n\n"+PROSE[0]["text"]}])
    assert plain.model_signature()==with_header.model_signature()
    assert len(with_header.generate("Erkläre Frequenz.")["tokens"])>1


def test_prompt_fields_are_ignored_when_compiling_information_records():
    first = InformationWaveModel(PROSE)
    contaminated = [{**record,"prompt":"A fabricated example answer must never be used"} for record in PROSE]
    second = InformationWaveModel(contaminated)
    assert first.model_signature()==second.model_signature()
    assert first.generate("Was ist Frequenz?")["tokens"]==second.generate("Was ist Frequenz?")["tokens"]


def test_conjunction_requires_a_shared_information_binding():
    model = InformationWaveModel(PROSE)
    related = model.generate("Wie hängen Frequenz und Periodendauer zusammen?")
    assert related["ended"] and "Kehrwert" in related["text"] and "Periodendauer" in related["text"]
    unsupported = model.generate("Wie hängen Frequenz und Spannung zusammen?")
    assert unsupported["tokens"]==[]
    assert unsupported["reason"]=="no_supported_information_binding"
    unknown = model.generate("Wie hängen Frequenz und Qztrblnx zusammen?")
    assert unknown["tokens"]==[] and unknown["analysis"]["unbound_terms"]


def test_unsupported_negation_does_not_reuse_positive_definition_fields():
    model = InformationWaveModel(PROSE)
    result = model.generate("Was ist keine Frequenz?")
    assert result["tokens"]==[] and result["analysis"]["unsupported_negation"]


def test_inverse_property_roles_compile_from_prose_and_not_a_fabricated_question():
    model = InformationWaveModel([{"title":"Hertz","text":"Hertz ist die SI-Einheit der Frequenz."}])
    result = model.generate("In welcher Einheit wird Frequenz gemessen?")
    assert result["ended"] and result["text"]=="Hertz ist die SI-Einheit der Frequenz."
    assert model.generate("Was ist Hertz?")["text"]==result["text"]


def test_exact_duplicate_prose_does_not_gain_additional_count_weight():
    original = InformationWaveModel(PROSE)
    repeated = InformationWaveModel(PROSE+[copy.deepcopy(PROSE[0])])
    assert repeated.input_record_count==original.input_record_count+1
    assert repeated.record_count==original.record_count
    assert repeated.model_signature()["coefficient_sha256"]==original.model_signature()["coefficient_sha256"]
    assert repeated.generate("Was ist Frequenz?")["tokens"]==original.generate("Was ist Frequenz?")["tokens"]


def test_abbreviation_and_ordinal_periods_do_not_end_the_answer_early():
    model = InformationWaveModel([{"title":"Takt","text":"Der Takt beginnt z. B. am 1. Januar mit einer Pause."}])
    result = model.generate("Erkläre den Takt.")
    assert result["ended"] and result["tokens"][-2:]==["pause","."]


@pytest.mark.parametrize("prompt", ["Welche Einheit verwendet man für Frequenz?",
                                   "Welche Einheit gibt man für Frequenz an?", "Was misst die Frequenz?"])
def test_question_predicates_are_grammar_but_unknown_domain_arguments_are_not(prompt):
    model = InformationWaveModel(PROSE)
    assert model.analyze_question(prompt)["supported"]
    assert not model.analyze_question(prompt+" Qztrblnx")["supported"]


def test_local_observed_word_forms_and_abbreviation_rendering():
    model = InformationWaveModel([
        {"title":"Echos","text":"Echos folgen aufeinander, z. B. bei ruhigem Wetter."},
        {"title":"Folgen","text":"Folgen sind Ergebnisse. Folgen beschreiben Auswirkungen. Folgen können wichtig sein."},
    ])
    result = model.generate("Erkläre Echos.")
    assert result["text"]=="Echos folgen aufeinander, z. B. bei ruhigem Wetter."
    assert result["ended"]


def test_multiple_question_facets_remain_bound_to_the_same_concept():
    model = InformationWaveModel(PROSE)
    result = model.generate("Was ist eine Frequenz und in welcher Einheit wird sie gemessen?")
    assert result["ended"]
    assert set(step["facet"] for step in result["trace"])=={"definition","unit"}
    assert "Hertz" in result["text"] and "Volt" not in result["text"]


@pytest.mark.parametrize("time_s,phase", [(0,0),(73.125,0.8),(1e9,np.pi)])
def test_fourier_direct_time_and_actual_complex_data_perturbations(time_s,phase):
    model = InformationWaveModel(PROSE)
    field,_ = model.prompt_field("Was ist Frequenz?")
    rng=np.random.default_rng(7428)
    for intervention in ("original","complex_phase"):
        if intervention=="complex_phase":
            for spectrum in model.transition_spectra.values():
                spectrum.local_spectrum *= np.exp(2j*np.pi*rng.random(len(spectrum.local_spectrum)))
        for prefix in ([],["die"],["frequenz","ist"]):
            try:
                actual, trace = model.next_distribution(prefix,field,time_s=time_s,phase_error=phase)
            except ValueError as error:
                # Equal opposite fields can cancel the only supported token.
                assert "Interference" in str(error)
                with pytest.raises(ValueError,match="Interference"):
                    model.next_distribution(prefix,field,time_s=time_s,phase_error=phase,method="direct")
                with pytest.raises(ValueError,match="Interference"):
                    model.next_distribution(prefix,field,time_s=0,phase_error=phase)
                continue
            direct,_ = model.next_distribution(prefix,field,time_s=time_s,phase_error=phase,method="direct")
            timeless,_ = model.next_distribution(prefix,field,time_s=0,phase_error=phase)
            np.testing.assert_allclose(actual,direct,atol=1.01e-12,rtol=0)
            np.testing.assert_allclose(actual,timeless,atol=1.01e-12,rtol=0)
            assert trace["all_supported_modes_included"]
            assert len(trace["active_token_indices"])<=trace["carrier_size"]
            assert np.count_nonzero(actual)<=len(trace["active_token_indices"])


def test_actual_data_and_prompt_nulling_cannot_be_replaced_by_text_lookup():
    model=InformationWaveModel(PROSE)
    field,_=model.prompt_field("Was ist Frequenz?")
    with pytest.raises(ValueError,match="binding"):
        model.next_distribution([],field*0)
    for spectrum in model.transition_spectra.values():
        spectrum.local_spectrum[:]=0
    with pytest.raises(ValueError,match="energy"):
        model.next_distribution([],field)


def test_prefix_ablation_changes_probabilities_and_unknown_concepts_abstain():
    model=InformationWaveModel(PROSE)
    field,_=model.prompt_field("Was ist Frequenz?")
    baseline,_=model.next_distribution(["die","frequenz"],field)
    without_prefix,_=model.next_distribution(["die","frequenz"],field,prefix_enabled=False)
    assert not np.allclose(baseline,without_prefix)
    assert not model.can_handle("Was ist ein unbekannter Quorb?")
    assert model.generate("Was ist ein unbekannter Quorb?")["tokens"]==[]


def test_context_is_consumed_symmetrically_and_snapshot_contains_exact_modes():
    model=InformationWaveModel(PROSE)
    first=np.zeros(model.size); first[model.index["hertz"]]=7
    second=np.zeros(model.size); second[model.index["frequenz"]]=3
    left,_=model.prompt_field("Frequenz",context_field=first)
    right,_=model.prompt_field("Hertz",context_field=second)
    np.testing.assert_allclose(left.token_amplitudes,right.token_amplitudes,atol=1e-14)
    snapshot=model.context_snapshot(left.token_amplitudes,time_s=17.25,points=1024)
    signal=np.array(snapshot["displacement"])+1j*np.array(snapshot["quadrature"])
    active=np.array(snapshot["active_token_indices"])
    modes=np.fft.ifft(signal,norm="ortho")
    expected=left.token_amplitudes[active]*np.exp(2j*np.pi*np.remainder(model.frequencies[active]*17.25,1.0))
    np.testing.assert_allclose(modes[:len(active)],expected,atol=1e-12)
    np.testing.assert_allclose(modes[len(active):],0,atol=1e-12)


def test_no_complete_prose_or_real_count_table_is_required_after_compilation():
    model=InformationWaveModel(PROSE)
    assert not any(hasattr(model,name) for name in ("records","texts","sentences","answers","counts","distributions"))
    result=model.generate("In welcher Einheit wird die Frequenz gemessen?")
    assert result["text"]=="Die Frequenz wird in Hertz gemessen."


def test_dynamic_carrier_uses_all_supported_tokens_and_stable_frequencies():
    original=InformationWaveModel(PROSE)
    enlarged=InformationWaveModel(PROSE+[{"title":"Drachen","text":"Drachen sind erfundene Tiere mit Schuppen."}])
    for token,index in original.index.items():
        assert original.frequencies[index]==enlarged.frequencies[enlarged.index[token]]
    for model in (original,enlarged):
        field,_=model.prompt_field("Was ist Frequenz?")
        _,trace=model.next_distribution([],field)
        support=set()
        for address in trace["field_addresses"]:
            support.update(model.supports[(address["group"],tuple(address["prefix"]))])
        assert set(trace["active_token_indices"])==support


def test_a_new_sentence_can_be_composed_from_prose_transition_fields():
    texts=["Rote Katzen schlafen.","Blaue Hunde laufen.","Blaue Katzen laufen."]
    model=InformationWaveModel([{"title":"Tiere","text":text} for text in texts],order=1)
    original={tuple(tokenize(text)) for text in texts}
    outputs=[model.generate("Erzähle über Tiere.",decoding="sample",seed=seed) for seed in range(12)]
    novel=[output for output in outputs if output["ended"] and tuple(output["tokens"]) not in original]
    assert novel
    grammatical={f"{adjective} {noun} {verb}." for adjective in ("Rote","Blaue")
                 for noun in ("Katzen","Hunde") for verb in ("schlafen","laufen")}
    assert all(output["text"] in grammatical for output in novel)
