"""Numerical and causal checks for support-local storage of complex fields."""
import copy

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from freqai.generative import BOS, GenerativeWaveModel
from freqai.spectral_storage import CompactSpectrum, SparseDistributionReference


PAIRS = [
    {"prompt": "Wie geht es dir?", "text": "Ich bin bereit und höre dir zu."},
    {"prompt": "Mir geht es gut.", "text": "Das freut mich. Was möchtest du erzählen?"},
    {"prompt": "Rote Tiere", "text": "Rote Katzen schlafen im Garten."},
    {"prompt": "Blaue Tiere", "text": "Blaue Hunde laufen im Garten.", "weight": 0.3},
    {"prompt": "Rote Tiere", "text": "Rote Hunde laufen leise."},
]


def pair_model(storage):
    return GenerativeWaveModel(["Im Garten wachsen Blumen."], order=2,
                               conditioned_pairs=PAIRS, storage=storage)


@pytest.mark.parametrize("time_s,phase", [(0, 0), (123.456, 0.8), (1e9, np.pi)])
def test_compact_and_dense_fields_have_same_probabilities_and_generation(time_s, phase):
    compact, dense = pair_model("compact"), pair_model("dense")
    assert isinstance(compact.association, csr_matrix)
    assert isinstance(compact.distributions, SparseDistributionReference)
    np.testing.assert_array_equal(compact.association.toarray(), dense.association)
    for prompt in ("Rote Tiere", "Wie geht es dir?", "Blumen Garten", ""):
        field, _ = compact.prompt_field(prompt)
        dense_field, _ = dense.prompt_field(prompt)
        np.testing.assert_array_equal(field, dense_field)
        np.testing.assert_array_equal(field.feature_spectrum, dense_field.feature_spectrum)
        for prefix in ([], ["rote"], ["im", "garten"], ["unbekannt"]):
            expected, _ = dense.next_distribution(prefix, dense_field, time_s=time_s, phase_error=phase)
            actual, _ = compact.next_distribution(prefix, field, time_s=time_s, phase_error=phase)
            direct, _ = compact.next_distribution(prefix, field, method="direct", time_s=time_s, phase_error=phase)
            np.testing.assert_allclose(actual, expected, atol=1.01e-12, rtol=0)
            np.testing.assert_allclose(actual, direct, atol=1.01e-12, rtol=0)
        for decoding in ("greedy", "sample", "beam"):
            expected = dense.generate(prompt, decoding=decoding, max_tokens=16, time_s=time_s, phase_error=phase)
            actual = compact.generate(prompt, decoding=decoding, max_tokens=16, time_s=time_s, phase_error=phase)
            assert actual["tokens"] == expected["tokens"]


def test_compact_complex_coefficients_are_runtime_source_and_global_edits_are_preserved():
    compact = pair_model("compact")
    field, _ = compact.prompt_field("Rote Tiere")
    original, _ = compact.next_distribution([], field)
    rng = np.random.default_rng(89364)
    global_phase = np.exp(2j*np.pi*rng.random(compact.size))
    perturbed = copy.deepcopy(compact)
    dense_perturbed = copy.deepcopy(compact)
    for bank_name in ("transition_spectra", "conditioned_spectra"):
        compact_bank = getattr(perturbed, bank_name)
        dense_bank = getattr(dense_perturbed, bank_name)
        for key, value in compact_bank.items():
            assert isinstance(value, CompactSpectrum)
            # A global spectral edit can populate previously empty token modes.
            changed = np.asarray(value)*global_phase
            value[:] = changed
            dense_bank[key] = changed.copy()
            np.testing.assert_array_equal(np.asarray(value), changed)
    actual, _ = perturbed.next_distribution([], field)
    expected, _ = dense_perturbed.next_distribution([], field)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert not np.allclose(actual, original)
    # Remove every local coefficient, retaining counts/reference probabilities.
    for bank in (compact.transition_spectra, compact.conditioned_spectra):
        for value in bank.values():
            value.local_spectrum[:] = 0
    with pytest.raises(ValueError, match="energy"):
        compact.next_distribution([], field)


def test_global_numpy_inspection_matches_dense_reference_without_changing_storage():
    compact, dense = pair_model("compact"), pair_model("dense")
    before = compact.storage_stats()
    for bank_name in ("transition_spectra", "conditioned_spectra"):
        for key, field in getattr(compact, bank_name).items():
            reference = getattr(dense, bank_name)[key]
            np.testing.assert_allclose(field, reference, atol=5e-16, rtol=1e-14)
            np.testing.assert_allclose(np.fft.ifft(field, norm="ortho"),
                                       np.fft.ifft(reference, norm="ortho"), atol=5e-16, rtol=1e-14)
            assert field.shape == reference.shape
            assert np.zeros_like(field).shape == reference.shape
    assert compact.storage_stats() == before


def test_numerical_storage_tracks_observed_edges_instead_of_vocabulary_times_fields():
    texts = [f"Objekt{i} besucht Ort{i}. Dort warten Freunde." for i in range(200)]
    compact = GenerativeWaveModel(texts, conditioning="lexical", storage="compact")
    dense = GenerativeWaveModel(texts, conditioning="lexical", storage="dense")
    compact_stats, dense_stats = compact.storage_stats(), dense.storage_stats()
    assert compact_stats["vocabulary_size"] > 400
    assert compact_stats["complex_coefficient_bytes"] < dense_stats["complex_coefficient_bytes"]/100
    assert compact_stats["numerical_payload_bytes"] < dense_stats["numerical_payload_bytes"]/40
    assert compact_stats["equivalent_dense_coefficient_bytes"] == dense_stats["complex_coefficient_bytes"]
    assert compact_stats["equivalent_dense_association_bytes"] == dense_stats["association_bytes"]


def test_pair_weight_changes_compiled_transition_energy_in_both_storages():
    for storage in ("compact", "dense"):
        model = GenerativeWaveModel([], order=1, conditioning="lexical", storage=storage, conditioned_pairs=[
            {"prompt": "Tiere", "text": "Rote Katzen schlafen.", "weight": 0.25},
            {"prompt": "Tiere", "text": "Blaue Hunde laufen.", "weight": 1.0},
        ])
        field, _ = model.prompt_field("Tiere")
        result, _ = model.next_distribution([], field)
        assert result[model.index["rote"]] == pytest.approx(0.2)
        assert result[model.index["blaue"]] == pytest.approx(0.8)
        assert model.distributions[(BOS,)][model.index["rote"]] == pytest.approx(0.2)


@pytest.mark.parametrize("weight", [0, -0.5, 1.01, float("nan"), float("inf"), True, "0.5", None, 0.5j])
def test_invalid_pair_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        GenerativeWaveModel([], conditioned_pairs=[{"prompt": "Tiere", "text": "Katzen.", "weight": weight}])
