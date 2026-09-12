"""Conditioning and persistent-field regressions, using small independent data."""
import numpy as np
import pytest

from freqai.generation_runtime import context_snapshot, corpus_annotations, generator_for, _field_vector
from freqai.generative import GenerativeWaveModel, PromptField
from freqai.memory import Document, WaveMemory


def greeting_model(extra=0):
    pairs = [{"prompt": "Hallo", "text": "Hallo!"}]
    annotations = [{"text": f"Guten Tag Nummer {i}.",
                    "features": {"semantic:greeting:user::": 8.0}} for i in range(extra)]
    return GenerativeWaveModel([], conditioned_pairs=pairs, annotated_texts=annotations)


def test_structural_channel_gain_is_independent_of_category_frequency():
    for additional_examples in (0, 1, 30):
        model = greeting_model(additional_examples)
        field, _ = model.prompt_field("Hallo")
        modes = np.fft.ifft(field.feature_spectrum, norm="ortho")
        index = model.feature_index["semantic:greeting:user::"]
        assert modes[index] == pytest.approx(8.0)
        assert model.feature_frequency["semantic:greeting:user::"] == 1+additional_examples


def test_coupling_gate_uses_numeric_spectrum_instead_of_diagnostic_labels():
    model = greeting_model()
    field, info = model.prompt_field("Hallo")
    null_output = model.generate("Hallo", input_field=(field*0, info), require_conditioning=True)
    assert null_output["tokens"] == []
    assert null_output["reason"] == "no_supported_input_coupling"
    # Clearing descriptive labels cannot remove an actual numerical excitation.
    info["condition_features"] = {}
    info["matched_condition_features"] = []
    actual = model.generate("Hallo", input_field=(field, info), require_conditioning=True)
    assert actual["tokens"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_numeric_feature_modes_cannot_bypass_the_input_gate(invalid):
    model = greeting_model()
    field = PromptField(np.ones(model.size), np.full(len(model.feature_vocabulary), invalid, dtype=complex))
    with pytest.raises(ValueError, match="feature spectrum"):
        model.generate("Hallo", input_field=(field, {}), require_conditioning=True)
    with pytest.raises(ValueError, match="feature spectrum"):
        model.next_distribution([], field)


@pytest.mark.parametrize("limit", [1, 2, 3, 12])
def test_multiple_input_bands_share_one_output_token_budget(limit):
    model = GenerativeWaveModel([], conditioned_pairs=[
        {"prompt": "Hallo", "text": "Hallo!"},
        {"prompt": "Mir geht es gut", "text": "Das freut mich."},
        {"prompt": "Wie geht es dir", "text": "Ich bin bereit."},
    ])
    result = model.generate("Hallo, mir geht es gut, wie geht es dir?", split_acts=True,
                            require_conditioning=True, max_tokens=limit, decoding="beam")
    assert result["input_bands"] >= 2
    assert 0 < len(result["tokens"]) <= limit
    assert [row["step"] for row in result["trace"]] == list(range(len(result["trace"])))


def test_context_display_is_the_exact_carrier_used_by_the_decoder():
    memory = WaveMemory([Document("toy", "Rote Katzen schlafen. Blaue Hunde laufen. Grüne Drachen fliegen. "
                                "Kleine Vögel singen. Fische schwimmen leise.")])
    model = generator_for(memory).model
    assert model.carrier_size > model.size
    context = {"generation": {"field": [["katzen", 0.5], ["hunde", 0.8]]}}
    field = _field_vector(model, context)
    snapshots = []
    for time_s in (0.0, 13.25):
        snapshot = context_snapshot(memory, context, time_s, points=1024)
        actual = np.array(snapshot["displacement"])+1j*np.array(snapshot["quadrature"])
        phases = np.exp(2j*np.pi*np.remainder(model.frequencies*time_s, 1.0))
        expected = np.fft.fft(field*phases, n=model.carrier_size, norm="ortho")
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert snapshot["mode_count"] == model.size
        assert snapshot["carrier_size"] == model.carrier_size
        assert len(actual) == model.carrier_size
        recovered = np.fft.ifft(actual, norm="ortho")
        np.testing.assert_allclose(recovered[:model.size], field*phases, atol=1e-12)
        np.testing.assert_allclose(recovered[model.size:], 0, atol=1e-12)
        assert snapshot["energy"] == pytest.approx(np.vdot(actual, actual).real)
        snapshots.append(actual)
    assert not np.allclose(*snapshots)


def test_only_explicit_language_source_annotations_add_category_couplings():
    documents = [Document("one", "Eine Pause hilft.", "Authored synthetic language prior / rest"),
                 Document("two", "Noch ein Satz.", "Unbekannte Quelle / rest")]
    annotations = corpus_annotations(documents)
    assert len(annotations) == 1
    assert annotations[0]["text"] == documents[0].text
    assert "semantic:mood_statement:user:tired:wellbeing" in annotations[0]["features"]


def test_unseen_input_with_no_coupling_does_not_emit_an_arbitrary_corpus_path():
    model = greeting_model()
    result = model.generate("Zyxquorbl", require_conditioning=True)
    assert result["tokens"] == [] and result["text"] == ""
    assert result["reason"] == "no_supported_input_coupling"
