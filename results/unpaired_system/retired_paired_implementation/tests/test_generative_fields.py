"""Independent, small-corpus causal checks for the spectral token generator.

These demonstrate algorithmic properties, not general conversational quality.
No production database or dialogue evaluation corpus is used.
"""
from collections.abc import Mapping

import numpy as np
import pytest

from freqai.generative import BOS, EOS, GenerativeWaveModel, PromptField, tokenize


TOY_TEXTS = (
    "Rote Katzen schlafen.",
    "Blaue Hunde laufen.",
    "Blaue Katzen laufen.",
)


def toy_model(**kwargs):
    return GenerativeWaveModel(TOY_TEXTS, order=1, conditioning="lexical", **kwargs)


def test_new_grammatical_sentence_is_generated_by_actual_token_fields():
    model = toy_model()
    original_sequences = {tuple(tokenize(text)) for text in TOY_TEXTS}
    # The toy grammar deliberately makes both adjectives, plural nouns and
    # plural verbs compatible. This new sentence is absent as a complete text.
    target = tokenize("Blaue Katzen schlafen.")
    assert tuple(target) not in original_sequences

    field, _ = model.prompt_field("")
    prefix = []
    for token in target + [EOS]:
        distribution, _ = model.next_distribution(prefix, field)
        assert distribution[model.index[token]] > 0
        prefix.append(token)

    generated = model.generate("", decoding="sample", seed=7, max_tokens=12, max_sentences=1)
    assert generated["tokens"] == target
    assert tuple(generated["tokens"]) not in original_sequences
    assert generated["ended"]
    assert [step["selected_token"] for step in generated["trace"]] == target
    for step in generated["trace"]:
        assert step["probability"] > 0
        assert model.vocabulary[step["selected_bin"]] == step["selected_token"]
        assert step["runtime_source"] == "imported_complex_transition_spectra"


class ForbiddenLookup(Mapping):
    def __getitem__(self, key):
        raise AssertionError("Generation accessed a forbidden corpus/count lookup")

    def __iter__(self):
        raise AssertionError("Generation iterated a forbidden corpus/count lookup")

    def __len__(self):
        raise AssertionError("Generation measured a forbidden corpus/count lookup")


def test_generation_needs_no_complete_answer_lookup_or_real_count_distribution(monkeypatch):
    model = toy_model()
    expected = model.generate("", decoding="sample", seed=7, max_tokens=12)

    from freqai import dialogue
    from freqai.memory import WaveMemory

    def forbidden(*args, **kwargs):
        raise AssertionError("The token generator called the reply retrieval pipeline")

    for name in ("ask", "scores", "coverage"):
        monkeypatch.setattr(WaveMemory, name, forbidden)
    monkeypatch.setattr(dialogue, "respond", forbidden)
    # Compilation has finished. Its original full strings and real count tables
    # are now deliberately inaccessible; the exact operator must use spectra.
    for name in ("texts", "documents", "answers", "counts", "distributions"):
        if hasattr(model, name):
            monkeypatch.setattr(model, name, ForbiddenLookup())

    actual = model.generate("", decoding="sample", seed=7, max_tokens=12)
    assert actual["tokens"] == expected["tokens"]
    assert not actual["uses_prompt_keys_for_response_comparison"]


def test_zero_complex_data_fields_cannot_be_replaced_by_intact_real_distributions():
    pairs = [
        {"prompt": "Tiere", "text": "Rote Katzen schlafen."},
        {"prompt": "Tiere", "text": "Blaue Hunde laufen."},
    ]
    model = GenerativeWaveModel([], conditioned_pairs=pairs, order=1, conditioning="lexical")
    field, _ = model.prompt_field("Tiere")
    assert model.next_distribution([], field)[0].sum() == pytest.approx(1)
    reference_tables = {key: value.copy() for key, value in model.distributions.items()}
    for bank in (model.transition_spectra, model.conditioned_spectra):
        for spectrum in bank.values():
            spectrum[:] = 0

    for key, reference in reference_tables.items():
        np.testing.assert_array_equal(model.distributions[key], reference)
    with pytest.raises(ValueError, match="field|wave|energy"):
        model.next_distribution([], field)


def test_changing_only_stored_complex_data_changes_next_token():
    model = GenerativeWaveModel(
        ["Rote Katzen schlafen.", "Blaue Hunde laufen."], order=1, conditioning="lexical"
    )
    field, _ = model.prompt_field("")
    before, _ = model.next_distribution([], field)
    intact_real_distribution = model.distributions[(BOS,)].copy()

    token_modes = np.zeros(model.size)
    token_modes[model.index["rote"]] = 1
    model.transition_spectra[(BOS,)] = np.fft.fft(token_modes, norm="ortho")
    after, _ = model.next_distribution([], field)

    np.testing.assert_array_equal(model.distributions[(BOS,)], intact_real_distribution)
    assert before[model.index["rote"]] == pytest.approx(0.5)
    assert after[model.index["rote"]] == pytest.approx(1)
    assert not np.allclose(before, after)


def test_changing_only_conditioned_complex_fields_changes_next_token():
    model = GenerativeWaveModel([], order=1, conditioning="lexical", conditioned_pairs=[
        {"prompt": "Tiere", "text": "Rote Katzen schlafen."},
        {"prompt": "Tiere", "text": "Blaue Hunde laufen."},
    ])
    field, _ = model.prompt_field("Tiere")
    before, _ = model.next_distribution([], field)
    global_fields = {key: value.copy() for key, value in model.transition_spectra.items()}
    real_distributions = {key: value.copy() for key, value in model.distributions.items()}

    token_modes = np.zeros(model.size)
    token_modes[model.index["rote"]] = 1
    for address in list(model.conditioned_spectra):
        if address[1] == (BOS,):
            model.conditioned_spectra[address] = np.fft.fft(token_modes, norm="ortho")
    after, _ = model.next_distribution([], field)

    for key, reference in global_fields.items():
        np.testing.assert_array_equal(model.transition_spectra[key], reference)
        np.testing.assert_array_equal(model.distributions[key], real_distributions[key])
    assert after[model.index["rote"]] > 0.99
    assert not np.allclose(before, after)


def test_prefix_order_changes_the_next_token_field():
    model = GenerativeWaveModel(
        ["Anna sieht Ben.", "Ben ruft Anna."], order=2, conditioning="lexical"
    )
    field, _ = model.prompt_field("")
    forward, _ = model.next_distribution(["anna", "sieht"], field)
    reversed_order, _ = model.next_distribution(["sieht", "anna"], field)

    assert forward[model.index["ben"]] > 0.99
    assert reversed_order[model.index["ben"]] == 0
    assert not np.allclose(forward, reversed_order)


def test_relative_prompt_phase_changes_token_competition():
    model = GenerativeWaveModel(
        ["Rote Katzen schlafen.", "Blaue Hunde laufen."], order=1, conditioning="lexical"
    )
    field, _ = model.prompt_field("")
    field[model.index["rote"]] = 1
    field[model.index["blaue"]] = 0.2
    constructive, _ = model.next_distribution([], field, prompt_gain=0.8, phase_error=0)
    destructive, _ = model.next_distribution([], field, prompt_gain=0.8, phase_error=np.pi)

    assert constructive[model.index["rote"]] > 0.7
    assert destructive[model.index["rote"]] < 0.1
    assert np.argmax(constructive) != np.argmax(destructive)


@pytest.mark.parametrize("decoding", ["greedy", "beam"])
@pytest.mark.parametrize("time_s", [0.125, 123.456, 345678.1234, 1e9])
def test_phase_invariance_including_equal_probability_word_choices(decoding, time_s):
    model = GenerativeWaveModel([
        "Rote Katzen schlafen.", "Blaue Hunde laufen.",
        "Rote Hunde laufen.", "Blaue Katzen schlafen.",
    ], order=2, conditioning="lexical")
    field, _ = model.prompt_field("")
    early, _ = model.next_distribution([], field, time_s=0)
    late, _ = model.next_distribution([], field, time_s=time_s)
    np.testing.assert_allclose(early, late, atol=1e-12, rtol=0)

    expected = model.generate("", decoding=decoding, time_s=0, max_tokens=12)
    actual = model.generate("", decoding=decoding, time_s=time_s, max_tokens=12)
    assert actual["tokens"] == expected["tokens"]


def test_full_prompt_null_removes_both_wave_channels():
    model = GenerativeWaveModel([], order=1, conditioning="lexical", conditioned_pairs=[
        {"prompt": "Rot", "text": "Rote Katzen schlafen."},
        {"prompt": "Blau", "text": "Blaue Hunde laufen."},
    ])
    field, _ = model.prompt_field("Rot")
    empty_field, _ = model.prompt_field("")
    assert np.linalg.norm(field.feature_spectrum) > 0
    complete_null = field*0
    assert np.count_nonzero(complete_null) == 0
    assert np.count_nonzero(complete_null.feature_spectrum) == 0
    blank_distribution, _ = model.next_distribution([], empty_field)
    complete_distribution, _ = model.next_distribution([], complete_null)
    np.testing.assert_allclose(complete_distribution, blank_distribution, atol=1e-12, rtol=0)

    feature_only = PromptField(np.zeros(model.size), field.feature_spectrum.copy())
    feature_distribution, _ = model.next_distribution([], feature_only)
    assert not np.allclose(feature_distribution, blank_distribution)


def test_token_frequencies_remain_stable_when_new_vocabulary_is_imported():
    original = toy_model()
    extended = GenerativeWaveModel(
        TOY_TEXTS + ("Grüne Drachen fliegen.",), order=1, conditioning="lexical"
    )
    for token in original.vocabulary:
        assert original.token_frequencies[original.index[token]] == extended.token_frequencies[extended.index[token]]
