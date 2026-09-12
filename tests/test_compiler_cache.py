"""Persistent caches preserve compiler identity and reject stale/invalid state."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json

import numpy as np
import pytest

from freqai import compiler_cache as cache
from freqai.spectral_storage import PackedTransitionSpectra
from freqai.unpaired import UnpairedWaveModel


RECORDS = [
    {"id": "self", "text": "Der Assistent heißt Oszillo. Der Assistent hat kein menschliches Befinden.", "source": "Cachetest"},
    {"id": "words", "text": "Das Wort Hallo ist ein Begrüßungswort. Das Wort Bitte ist ein Höflichkeitswort.", "source": "Cachetest"},
    {"id": "frequency", "text": "Die Frequenz ist die Anzahl der Wiederholungen pro Sekunde.", "source": "Cachetest"},
]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    return UnpairedWaveModel(RECORDS)


def test_save_load_preserves_signature_packed_data_and_answers(tmp_path, model):
    signature = model.model_signature()
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    with np.load(tmp_path/cache.CACHE_FILENAME, allow_pickle=False) as archive:
        assert len(archive.files) < 20
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    restored = cache.load_compiled(RECORDS, 2, tmp_path)
    assert restored is not None
    assert isinstance(restored.transition_spectra, PackedTransitionSpectra)
    assert len(restored.transition_spectra._cache) == 0
    assert restored.model_signature() == signature
    for prompt in ("Hallo", "Wie heißt du?", "Was ist eine Frequenz?", "Mir geht es gut, und dir?"):
        expected = model.generate(prompt, seed=17, max_tokens=32)
        actual = restored.generate(prompt, seed=17, max_tokens=32)
        assert actual["text"] == expected["text"]
        assert actual["tokens"] == expected["tokens"]


def test_runtime_grammar_and_prompt_state_are_not_persisted(tmp_path, model):
    model.generate("Hallo")
    model.generate("Ich heiße Larion.")
    model.data_wave()
    assert model._grammar_roles
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    restored = cache.load_compiled(RECORDS, 2, tmp_path)
    assert restored._grammar_roles == {}
    assert restored._data_wave_cache is None
    assert all(role.origin in cache._CORPUS_ORIGINS and role.surfaces is None for role in restored.lexical_roles)
    assert "Larion" not in json.dumps(restored.model_signature())


@pytest.mark.parametrize("change", ["text", "title", "metadata", "order", "duplicate", "record_order"])
def test_raw_input_changes_invalidate_even_when_normalized_corpus_is_same(tmp_path, model, change):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    records = copy.deepcopy(RECORDS)
    order = 2
    if change == "text":
        records[0]["text"] = records[0]["text"].replace("Oszillo", "Kymara")
    elif change == "title":
        records[0]["title"] = "Assistent"
    elif change == "metadata":
        records[0]["source"] = "Geändert"
    elif change == "order":
        order = 3
    elif change == "duplicate":
        records.append(records[0].copy())
    else:
        records.reverse()
    assert cache.load_compiled(records, order, tmp_path) is None
    if order == 2:
        previous = cache.load_previous_compiled(order, tmp_path)
        assert previous is not None
        assert previous.model_signature() == model.model_signature()


def test_implementation_change_invalidates_exact_and_previous_cache(tmp_path, model, monkeypatch):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    monkeypatch.setattr(cache, "code_fingerprint", lambda: "f"*64)
    assert cache.load_compiled(RECORDS, 2, tmp_path) is None
    assert cache.load_previous_compiled(2, tmp_path) is None


@pytest.mark.parametrize("record", [{"text": "Text", "question": "Paar"}, {"text": "Text", "Answer": "Paar"},
                                     {"text": "Text", "prompt": "Paar"}, {"text": 23}])
def test_invalid_input_is_rejected_before_cache_lookup(tmp_path, record):
    with pytest.raises(ValueError):
        cache.load_compiled([record], 2, tmp_path)


def test_byte_corruption_and_truncation_are_cold_misses(tmp_path, model):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    path = tmp_path/cache.CACHE_FILENAME
    original = path.read_bytes()
    altered = bytearray(original)
    altered[len(altered)//2] ^= 1
    path.write_bytes(altered)
    assert cache.load_compiled(RECORDS, 2, tmp_path) is None
    path.write_bytes(original[:len(original)//2])
    assert cache.load_previous_compiled(2, tmp_path) is None


@pytest.mark.parametrize("change", ["offset", "dtype", "prefix", "fact_role", "unknown_attribute", "session_role"])
def test_structural_corruption_is_rejected_even_with_new_checksum(tmp_path, model, change):
    metadata, arrays = cache._pack(model, RECORDS, 2)
    if change == "offset":
        arrays["offsets"][1] = -1
    elif change == "dtype":
        arrays["coefficients"] = arrays["coefficients"].astype(np.complex64)
    elif change == "prefix":
        arrays["prefix_ids"][0, 0] = model.size+12
    elif change == "fact_role":
        metadata["facts"][0][2] = len(model.lexical_roles)+1
    elif change == "unknown_attribute":
        metadata["attributes"]["__class__"] = "exec"
    else:
        metadata["role_origins"][0] = "session_prompt_only"
    cache._write_container(tmp_path/cache.CACHE_FILENAME, metadata, arrays)
    assert cache.load_compiled(RECORDS, 2, tmp_path) is None
    assert cache.load_previous_compiled(2, tmp_path) is None


def test_atomic_failed_write_preserves_prior_snapshot(tmp_path, model, monkeypatch):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    original = (tmp_path/cache.CACHE_FILENAME).read_bytes()
    monkeypatch.setattr(cache.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("Disk unavailable")))
    assert cache.save_compiled(model, RECORDS, 2, tmp_path) is False
    assert (tmp_path/cache.CACHE_FILENAME).read_bytes() == original
    assert list(tmp_path.glob(".compiler-*.tmp")) == []


def test_repeated_revisions_replace_one_container(tmp_path, model):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    newer_records = RECORDS+[{"text": "Die Periode ist eine Zeitspanne."}]
    newer = UnpairedWaveModel(newer_records)
    assert cache.save_compiled(newer, newer_records, 2, tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == [cache.CACHE_FILENAME]
    assert cache.load_compiled(RECORDS, 2, tmp_path) is None
    assert cache.load_compiled(newer_records, 2, tmp_path).model_signature() == newer.model_signature()


def test_concurrent_readers_get_independent_runtime_caches(tmp_path, model):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        models = list(executor.map(lambda _: cache.load_compiled(RECORDS, 2, tmp_path), range(4)))
    assert all(item is not None for item in models)
    models[0].generate("Hallo")
    assert models[0]._grammar_roles
    assert all(item._grammar_roles == {} for item in models[1:])


def test_empty_compiler_roundtrip(tmp_path):
    model = UnpairedWaveModel([])
    assert cache.save_compiled(model, [], 2, tmp_path)
    restored = cache.load_compiled([], 2, tmp_path)
    assert restored.model_signature() == model.model_signature()
    assert restored.generate("Hallo")["tokens"] == []


def test_cached_groups_and_energy_support_exact_incremental_compilation(tmp_path, model):
    assert cache.save_compiled(model, RECORDS, 2, tmp_path)
    previous = cache.load_previous_compiled(2, tmp_path)
    assert previous.group_fingerprints == model.group_fingerprints
    assert previous.group_field_ranges == model.group_field_ranges
    np.testing.assert_array_equal(previous.transition_spectra.mode_energy, model.transition_spectra.mode_energy)
    additions = RECORDS+[{"text": "Ein Oszillator erzeugt eine periodische Bewegung."}]
    assert cache.load_compiled(additions, 2, tmp_path) is None
    incremental = UnpairedWaveModel(additions, previous_model=previous)
    reference = UnpairedWaveModel(additions)
    assert incremental.compile_reuse_stats["groups_reused"] > 0
    assert incremental.model_signature() == reference.model_signature()
    np.testing.assert_array_equal(incremental.symbol_energy, reference.symbol_energy)


def test_model_cannot_be_cached_under_other_same_length_raw_inputs(tmp_path, model):
    changed = copy.deepcopy(RECORDS)
    changed[0]["text"] = changed[0]["text"].replace("Oszillo", "Kymara")
    assert not cache.save_compiled(model, changed, 2, tmp_path)
    assert cache.load_compiled(changed, 2, tmp_path) is None


@pytest.mark.parametrize("buffer", ["coefficients", "token_indices", "mode_energy"])
def test_perturbed_numeric_buffers_are_not_persisted(tmp_path, model, buffer):
    table = model.transition_spectra
    target = getattr(table, buffer)
    target[0] = target[0]+1
    assert not cache.save_compiled(model, RECORDS, 2, tmp_path)


def test_false_group_ranges_are_rejected_before_incremental_reuse(tmp_path, model):
    metadata, arrays = cache._pack(model, RECORDS, 2)
    arrays["group_ranges"][0, 1] += 1
    arrays["group_ranges"][1, 0] += 1
    cache._write_container(tmp_path/cache.CACHE_FILENAME, metadata, arrays)
    assert cache.load_previous_compiled(2, tmp_path) is None
