"""Regression coverage for the interrupted compiler refactor and real spawn workers."""
import numpy as np
import pytest

from freqai import parallel
from freqai.information import InformationWaveModel
from freqai.unpaired import UnpairedWaveModel


REFERENCE_RECORDS = [
    {"title": "Hertz", "text": "Hertz ist die SI-Einheit der Frequenz."},
    {"title": "Echos", "text": "Echos folgen aufeinander, z. B. bei ruhigem Wetter."},
    {"title": "Folgen", "text": "Folgen sind Ergebnisse. Folgen beschreiben Auswirkungen."},
    {"title": "Periodendauer", "text": "Die Periodendauer ist die Dauer einer Schwingung. "
     "Die Frequenz ist der Kehrwert der Periodendauer."},
    {"title": "Hertz", "text": "Hertz ist die SI-Einheit der Frequenz."},
    {"title": "Echos", "text": "Echos\n\nEchos folgen aufeinander, z. B. bei ruhigem Wetter."},
]


def test_compiled_fields_preserve_pre_refactor_reference():
    # Independently obtained from evaluation/final_code, before parallelization.
    # This pins deduplication, unit ownership, shared prefixes and capitalization.
    model = InformationWaveModel(REFERENCE_RECORDS)
    assert not model.transition_spectra._cache
    signature = model.model_signature()
    assert not model.transition_spectra._cache
    assert signature["corpus_digest"] == "e4e89aa5e7d7a5d57e56513a1ab507bdb56b1d089f98d6234fc523ea44bae3c9"
    assert signature["coefficient_sha256"] == "85d449e425b72fd2ad8faa0e68e50513e3bda4eca6fb4b01d407fb8c3119673f"
    assert signature["rendering_sha256"] == "07b68c79da876864329aa2befdda3aee88d0f0dd3029615b2ad752e9d1a2c85e"
    assert model.input_record_count == 6
    assert model.record_count == 4
    assert model.statement_count == 6
    assert model.resonance_floor == 0.25
    expected_energy = np.zeros(model.size)
    for key, spectrum in model.transition_spectra.items():
        expected_energy[model.supports[key]] += np.abs(np.fft.ifft(spectrum.local_spectrum, norm="ortho")) ** 2
    np.testing.assert_array_equal(model.symbol_energy, expected_energy)
    assert model.generate("In welcher Einheit wird Frequenz gemessen?")["ended"]


@pytest.mark.skipif(parallel.logical_cores() < 2, reason="real process parity needs two available cores")
def test_real_process_compilation_preserves_fields_roles_and_generation(monkeypatch):
    records = REFERENCE_RECORDS + [
        {"title": f"Signal Lumo{index}",
         "text": f"Das Signal Lumo{index} hat eine Frequenz von {index + 10} Hertz. "
                 "Die Frequenz ist eine Anzahl von Schwingungen."}
        for index in range(64)
    ] + [{"text": "Der Assistent heißt Vela. Das Wort Hallo ist ein Begrüßungswort. "
                  "Das Befinden ist eine Wahrnehmung."}]
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    serial = UnpairedWaveModel(records)
    prompts = ["Was ist eine Frequenz?", "Wie heißt du?", "Hallo"]
    expected = [serial.generate(prompt) for prompt in prompts]
    parallel._shutdown()
    monkeypatch.setenv("FREQAI_PARALLEL", "1")
    monkeypatch.setenv("FREQAI_WORKERS", "2")
    try:
        actual = UnpairedWaveModel(records)
        policy = parallel.describe()
        assert policy["pool_active"]
        assert policy["pool_workers"] == min(2, parallel.logical_cores())
        assert actual.model_signature() == serial.model_signature()
        assert actual.display_tokens == serial.display_tokens
        assert actual.facts == serial.facts
        assert actual.lexicon_spellings == serial.lexicon_spellings
        np.testing.assert_array_equal(actual.symbol_energy, serial.symbol_energy)
        for prompt, reference in zip(prompts, expected):
            output = actual.generate(prompt)
            assert output["ended"] == reference["ended"]
            assert output["tokens"] == reference["tokens"]
            assert output["text"] == reference["text"]
            assert [step["probability"] for step in output["trace"]] == [
                step["probability"] for step in reference["trace"]]
    finally:
        parallel._shutdown()
