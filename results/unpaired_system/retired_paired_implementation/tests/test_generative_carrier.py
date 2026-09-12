"""The larger FFT carrier must preserve the original token-mode computation."""
import copy

import numpy as np
import pytest
from scipy.fft import next_fast_len

from freqai.generative import GenerativeWaveModel, PromptField
from freqai.generation_runtime import context_snapshot, generator_for, _field_vector
from freqai.memory import Document, WaveMemory


TEXTS = (
    "Rote Katzen schlafen. Blaue Hunde laufen.",
    "Grüne Drachen fliegen. Kleine Vögel singen.",
    "Fische schwimmen leise.",
)


def model_for(storage="compact"):
    return GenerativeWaveModel([], order=2, conditioning="lexical", storage=storage, conditioned_pairs=[
        {"prompt": "Rote Tiere", "text": TEXTS[0]},
        {"prompt": "Blaue Tiere", "text": TEXTS[1]},
        {"prompt": "Wasser Tiere", "text": TEXTS[2]},
    ])


@pytest.mark.parametrize("storage", ["compact", "dense"])
@pytest.mark.parametrize("phase_time", [(0, 0), (123.456, 0.8), (1e9, np.pi)])
@pytest.mark.parametrize("perturb_data", [False, True])
def test_padded_operator_matches_original_carrier_and_direct_reference(storage, phase_time, perturb_data):
    model = model_for(storage)
    assert model.size == 17 and model.carrier_size == next_fast_len(model.size) == 18
    rng = np.random.default_rng(82719)
    if perturb_data:
        phase = np.exp(2j*np.pi*rng.random(model.size))
        for bank in (model.transition_spectra, model.conditioned_spectra):
            for key in bank:
                bank[key] = np.asarray(bank[key])*phase
    # Restore the previous runtime carrier on an otherwise identical model.
    original = copy.copy(model)
    original.carrier_size = model.size
    field, _ = model.prompt_field("Rote Tiere")
    time_s, relative_phase = phase_time
    variants = (field, field*0, PromptField(np.zeros(model.size), field.feature_spectrum))
    for numerical_field in variants:
        for prefix in ([], ["rote"], ["katzen", "schlafen"], ["unbekannt"]):
            controls = {"time_s":time_s, "phase_error":relative_phase}
            expected, _ = original.next_distribution(prefix, numerical_field, **controls)
            actual, trace = model.next_distribution(prefix, numerical_field, **controls)
            direct, direct_trace = model.next_distribution(prefix, numerical_field, method="direct", **controls)
            assert len(actual) == model.size
            assert trace["carrier_bins"] == model.carrier_size
            assert direct_trace["carrier_bins"] == model.size
            np.testing.assert_allclose(actual, expected, atol=1.01e-12, rtol=0)
            np.testing.assert_allclose(actual, direct, atol=1.01e-12, rtol=0)
            assert trace["fft_roundtrip_max_error"] < 1e-12
    for decoding in ("sample", "greedy", "beam"):
        controls = {"decoding":decoding, "time_s":time_s, "phase_error":relative_phase, "max_tokens":16}
        assert model.generate("Rote Tiere", **controls)["tokens"] == original.generate("Rote Tiere", **controls)["tokens"]


def test_padding_does_not_change_the_hrr_reference_carrier():
    model = model_for()
    original = copy.copy(model)
    original.carrier_size = model.size
    field, _ = model.prompt_field("Rote Tiere")
    actual, trace = model.next_distribution([], field, method="hrr", time_s=1.375)
    expected, _ = original.next_distribution([], field, method="hrr", time_s=1.375)
    assert trace["carrier_bins"] == model.size < model.carrier_size
    np.testing.assert_array_equal(actual, expected)


def test_vocabulary_extension_preserves_context_token_frequencies_across_carrier_resize():
    memory = WaveMemory([Document(str(i), text) for i, text in enumerate(TEXTS)])
    initial = generator_for(memory).model
    frequencies = dict(zip(initial.vocabulary, initial.frequencies))
    context = {"generation":{"field":[["katzen",0.5],["hunde",0.8]]}}
    original_context = copy.deepcopy(context)
    original_size = initial.carrier_size
    memory.add_document("Schnelle Pferde rennen leise.", document_id="added")
    extended = generator_for(memory).model
    assert extended.carrier_size != original_size
    for token, frequency in frequencies.items():
        assert extended.frequencies[extended.index[token]] == frequency
    snapshot = context_snapshot(memory, context, time_s=73.125, points=1024)
    signal = np.array(snapshot["displacement"])+1j*np.array(snapshot["quadrature"])
    phases = np.exp(2j*np.pi*np.remainder(extended.frequencies*73.125,1.0))
    recovered = np.fft.ifft(signal,norm="ortho")
    np.testing.assert_allclose(recovered[:extended.size], _field_vector(extended,context)*phases, atol=1e-12)
    np.testing.assert_allclose(recovered[extended.size:],0,atol=1e-12)
    assert snapshot["mode_count"] == extended.size
    assert snapshot["carrier_size"] == extended.carrier_size
    assert context == original_context
