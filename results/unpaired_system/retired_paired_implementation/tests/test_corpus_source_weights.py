import json

import pytest
import numpy as np

from freqai import generation_runtime
from freqai.memory import Document
from freqai.generative import BOS, GenerativeWaveModel


def test_source_gains_are_explicit_and_use_the_most_specific_prefix(tmp_path, monkeypatch):
    config = tmp_path/"categories.json"
    config.write_text(json.dumps({"paired_source_weights": {"Public / ": .25, "Public / Dialogue": .5}}))
    monkeypatch.setattr(generation_runtime, "CATEGORIES_PATH", config)
    documents = [Document("own", "Antwort.", "Eigene Daten", "Frage?"),
                 Document("one", "Andere Antwort.", "Public / Dialogue / id42", "Andere Frage?"),
                 Document("two", "Ein Fakt.", "Public / Facts / id17", "Sachfrage?"),
                 Document("unpaired", "Ein Text.", "Public / Facts")]
    pairs = generation_runtime.corpus_pairs(documents)
    assert [pair["weight"] for pair in pairs] == [1.0, .5, .25]
    assert [pair["text"] for pair in pairs] == [doc.text for doc in documents[:3]]


def test_same_prompt_keeps_its_own_source_conditioning(tmp_path, monkeypatch):
    config = tmp_path/"categories.json"
    config.write_text(json.dumps({"lexical_only_sources": ["Public / "]}))
    monkeypatch.setattr(generation_runtime, "CATEGORIES_PATH", config)
    documents = [Document("own", "Hallo zusammen.", "Eigene Daten", "Hallo"),
                 Document("external", "Gläser zerbrechen.", "Public / Example", "Hallo")]
    pairs = generation_runtime.corpus_pairs(documents)
    assert [pair["semantic_features"] for pair in pairs] == [True, False]
    # Test both cache insertion orders: an equal string is not equal supervision.
    for ordered in (pairs, list(reversed(pairs))):
        model = GenerativeWaveModel([], conditioned_pairs=ordered, order=1)
        semantic = np.fft.ifft(model.conditioned_spectra[("semantic:greeting:user::", (BOS,))], norm="ortho")
        lexical = np.fft.ifft(model.conditioned_spectra[("word:hallo", (BOS,))], norm="ortho")
        assert abs(semantic[model.index["gläser"]]) < 1e-12
        assert abs(semantic[model.index["hallo"]]) > .99
        assert abs(lexical[model.index["gläser"]]) > .1


@pytest.mark.parametrize("flag", [0, 1, "false", None])
def test_pair_semantic_supervision_flag_requires_a_boolean(flag):
    with pytest.raises(ValueError, match="semantic_features"):
        GenerativeWaveModel([], conditioned_pairs=[{"prompt": "Hallo", "text": "Hallo.", "semantic_features": flag}])


@pytest.mark.parametrize("gain", [True, 0, -1, 2, "0.5", float("nan"), float("inf")])
def test_invalid_source_gains_are_not_silently_accepted(tmp_path, monkeypatch, gain):
    config = tmp_path/"categories.json"
    config.write_text(json.dumps({"paired_source_weights": {"Public": gain}}))
    monkeypatch.setattr(generation_runtime, "CATEGORIES_PATH", config)
    with pytest.raises(ValueError, match="source weights"):
        generation_runtime.corpus_pairs([])
