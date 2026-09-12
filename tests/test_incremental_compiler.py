"""Incremental compilation must equal a fresh build after corpus changes."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from freqai.unpaired import UnpairedWaveModel


BASE_RECORDS = [
    "Pendel\nDas Pendel hat eine magnetische Flussdichte.",
    "Flussdichte\nDie Flussdichte ist eine physikalische Größe.",
    "Ein Laser ist Licht.",
    "Der Assistent heißt Vela. Das Wort Hallo ist ein Begrüßungswort.",
]
PROMPTS = ("Was ist ein Pendel?", "Was ist ein Laser?", "Wie heißt du?", "Hallo")


@pytest.fixture(autouse=True)
def small_serial_fixture(monkeypatch):
    # These regressions concern invalidation, not a second full-core benchmark.
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    monkeypatch.setenv("FREQAI_COMPUTE", "cpu")


def _generation(model, prompt):
    result = model.generate(prompt, max_tokens=32, time_s=1.234, seed=17)
    return {key: result.get(key) for key in
            ("text", "tokens", "ended", "reason", "trace", "information_state", "tokenization")}


def _snapshot(model):
    """Capture mutable numerical state by value, including mappings omitted by the signature."""
    return {
        "signature": model.model_signature(),
        "vocabulary": tuple(model.vocabulary),
        "index": dict(model.index),
        "concepts": frozenset(model.concepts),
        "heads": {key: tuple(value) for key, value in model._concept_heads.items()},
        "groups": model.groups,
        "group_indices": {key: frozenset(value) for key, value in model.group_indices_by_concept.items()},
        "display_tokens": dict(model.display_tokens),
        "context_forms": dict(model.context_forms),
        "frequencies": model.frequencies.tobytes(),
        "symbol_energy": model.symbol_energy.tobytes(),
        "supports": {key: value.tobytes() for key, value in model.supports.items()},
        "spectra": {key: (value.size, value.local_spectrum.tobytes())
                    for key, value in model.transition_spectra.items()},
        "facts": tuple(model.facts),
        "facts_by_subject": {key: tuple(value) for key, value in model.facts_by_subject.items()},
        "lexicon": {key: tuple(value) for key, value in model.lexicon.items()},
        "lexicon_spellings": dict(model.lexicon_spellings),
        "lexical_roles": [(role.origin, role.indices.tobytes(), role.spectrum.tobytes(),
                           copy.deepcopy(role.surfaces)) for role in model.lexical_roles],
        "role_ids": dict(model._role_ids),
        "grammar_roles": dict(model._grammar_roles),
        "action_roles": dict(model.action_roles),
        "state_vocabulary": frozenset(model.state_vocabulary),
        "assistant_aliases": frozenset(model.assistant_aliases),
    }


def _assert_equivalent(actual, fresh, prompts=PROMPTS):
    # Coefficient hashes alone omit concept descriptors, fact metadata, surface
    # defaults, and the all-data energy accumulator. Compare those explicitly.
    assert _snapshot(actual) == _snapshot(fresh)
    np.testing.assert_array_equal(actual.symbol_energy, fresh.symbol_energy)
    assert actual.input_record_count == fresh.input_record_count
    assert actual.record_count == fresh.record_count
    for prompt in prompts:
        assert actual.analyze_question(prompt) == fresh.analyze_question(prompt)
        assert _generation(actual, prompt) == _generation(fresh, prompt)


def test_new_multiword_concept_rebinds_statements_in_older_records():
    before = UnpairedWaveModel(BASE_RECORDS)
    old_state = _snapshot(before)
    records = BASE_RECORDS + [
        "Magnetische Flussdichte\nDie magnetische Flussdichte ist eine physikalische Größe."
    ]
    incremental = UnpairedWaveModel(records, previous_model=before)
    fresh = UnpairedWaveModel(records)
    assert ("pendel", "about", ("flussdicht", "pendel")) in before.groups
    assert ("pendel", "about", ("magnetisch flussdicht", "pendel")) in incremental.groups
    assert ("pendel", "about", ("flussdicht", "pendel")) not in incremental.groups
    _assert_equivalent(incremental, fresh)
    assert _snapshot(before) == old_state


def test_new_capitalization_ambiguity_recomputes_old_context_forms():
    before = UnpairedWaveModel(["Ein Laser ist Licht."])
    records = ["Ein Laser ist Licht.", "Ein LASER ist Licht."]
    incremental = UnpairedWaveModel(records, previous_model=before)
    fresh = UnpairedWaveModel(records)
    assert before.display_tokens["laser"] == incremental.display_tokens["laser"] == "Laser"
    context = (("<BOS>", "ein"), "laser")
    assert context not in before.context_forms
    assert incremental.context_forms[context] == "LASER"
    _assert_equivalent(incremental, fresh, ("Was ist ein Laser?",))


def test_duplicate_raw_records_preserve_extra_declarations_despite_same_digest():
    record = "Ein Laser ist Licht."
    before = UnpairedWaveModel([record])
    incremental = UnpairedWaveModel([record, record], previous_model=before)
    fresh = UnpairedWaveModel([record, record])
    assert before.corpus_digest == incremental.corpus_digest
    assert before.record_count == incremental.record_count == 1
    assert len(before.facts) == 1
    assert len(incremental.facts) == 2
    _assert_equivalent(incremental, fresh, ("Was ist ein Laser?",))


def test_lexically_earlier_vocabulary_remaps_reused_support_indices():
    before = UnpairedWaveModel(BASE_RECORDS)
    records = BASE_RECORDS + ["Aaron ist ein Aardvark."]
    incremental = UnpairedWaveModel(records, previous_model=before)
    fresh = UnpairedWaveModel(records)
    assert incremental.index["laser"] != before.index["laser"]
    assert incremental.compilation_stats["reused_groups"] > 0
    assert incremental.compilation_stats["reused_fields"] > 0
    _assert_equivalent(incremental, fresh, PROMPTS + ("Was ist Aaron?",))


@pytest.mark.parametrize("change", ["replace", "remove", "reorder"])
def test_nonappend_changes_equal_a_fresh_build(change):
    before = UnpairedWaveModel(BASE_RECORDS)
    old_state = _snapshot(before)
    records = list(BASE_RECORDS)
    if change == "replace":
        records[2] = "Ein Laser ist eine Strahlungsquelle."
    elif change == "remove":
        records.pop(1)
    else:
        records.reverse()
    incremental = UnpairedWaveModel(records, previous_model=before)
    fresh = UnpairedWaveModel(records)
    _assert_equivalent(incremental, fresh)
    assert _snapshot(before) == old_state


def test_empty_append_reuses_all_groups_and_keeps_previous_model_unchanged():
    before = UnpairedWaveModel(BASE_RECORDS)
    # Real generation creates grammar roles in the old instance. They must not
    # leak into the new corpus model or be mutated by incremental compilation.
    for prompt in PROMPTS:
        _generation(before, prompt)
    old_state = _snapshot(before)
    incremental = UnpairedWaveModel(list(BASE_RECORDS), previous_model=before)
    fresh = UnpairedWaveModel(BASE_RECORDS)
    assert incremental.compilation_stats["compiled_groups"] == 0
    assert incremental.compilation_stats["reused_groups"] == len(before.groups)
    _assert_equivalent(incremental, fresh)
    assert _snapshot(before) == old_state


def test_batch_append_equals_sequential_appends_and_full_recompile():
    before = UnpairedWaveModel(BASE_RECORDS)
    additions = [
        "Aaron ist ein Aardvark.",
        "Ein LASER ist Licht.",
        "Magnetische Flussdichte\nDie magnetische Flussdichte ist eine physikalische Größe.",
        "Ein Laser ist Licht.",
    ]
    batch = UnpairedWaveModel(BASE_RECORDS + additions, previous_model=before)
    current = before
    for position in range(1, len(additions) + 1):
        current = UnpairedWaveModel(BASE_RECORDS + additions[:position], previous_model=current)
    fresh = UnpairedWaveModel(BASE_RECORDS + additions)
    _assert_equivalent(batch, fresh)
    # The helper has warmed grammar state in fresh; compare a fresh reference
    # again so runtime-only additions cannot mask a compilation difference.
    _assert_equivalent(current, UnpairedWaveModel(BASE_RECORDS + additions))


def test_changed_ngram_order_does_not_reuse_incompatible_fields():
    before = UnpairedWaveModel(BASE_RECORDS, order=2)
    incremental = UnpairedWaveModel(BASE_RECORDS, order=3, previous_model=before)
    fresh = UnpairedWaveModel(BASE_RECORDS, order=3)
    _assert_equivalent(incremental, fresh)


def test_reused_arrays_do_not_alias_the_previous_model():
    before = UnpairedWaveModel(BASE_RECORDS)
    old_state = _snapshot(before)
    incremental = UnpairedWaveModel(BASE_RECORDS, previous_model=before)
    assert incremental.compilation_stats["reused_fields"] > 0
    key = next(iter(incremental.transition_spectra))
    assert not np.shares_memory(incremental.transition_spectra[key].local_spectrum,
                                before.transition_spectra[key].local_spectrum)
    assert not np.shares_memory(incremental.supports[key], before.supports[key])
    incremental.transition_spectra[key][:] = 0
    incremental.supports[key][0] = (int(incremental.supports[key][0]) + 1) % incremental.size
    assert _snapshot(before) == old_state


@pytest.mark.parametrize("mutation", ["zero_field", "coefficient_view", "support_view", "energy_view"])
def test_diagnostic_mutations_in_previous_model_do_not_become_corpus_coefficients(mutation):
    before = UnpairedWaveModel(BASE_RECORDS)
    key = next(iter(before.transition_spectra))
    if mutation == "zero_field":
        before.transition_spectra[key][:] = 0
    elif mutation == "coefficient_view":
        before.transition_spectra[key].local_spectrum[0] += 0.25j
    elif mutation == "support_view":
        before.supports[key][0] = (int(before.supports[key][0]) + 1) % before.size
    else:
        before.transition_spectra.mode_energy[0] += 0.125
    old_state = _snapshot(before)
    records = BASE_RECORDS + ["Aaron ist ein Aardvark."]
    incremental = UnpairedWaveModel(records, previous_model=before)
    _assert_equivalent(incremental, UnpairedWaveModel(records))
    assert _snapshot(before) == old_state


def test_disk_cache_restores_full_factual_and_rendering_state_then_reuses_safely(tmp_path):
    from freqai import compiler_cache

    records = BASE_RECORDS + [
        "Das Befinden ist eine ruhige Wahrnehmung.",
        "Lesen ist das Betrachten geschriebener Wörter.",
        "Die Farbe des Pendels ist blau.",
        "Die Wendung Guten Morgen ist eine Begrüßungsformel.",
        "Ein LASER ist Licht.",
        BASE_RECORDS[2],
    ]
    original = UnpairedWaveModel(records)
    for prompt in PROMPTS:
        _generation(original, prompt)
    assert compiler_cache.save_compiled(original, records, 2, tmp_path)
    restored = compiler_cache.load_compiled(records, 2, tmp_path)
    assert restored is not None
    _assert_equivalent(restored, UnpairedWaveModel(records),
                       PROMPTS + ("Welche Farbe hat das Pendel?", "Guten Morgen"))

    additions = ["Magnetische Flussdichte\nDie magnetische Flussdichte ist eine physikalische Größe.",
                 "Aaron ist ein Aardvark."]
    previous = compiler_cache.load_previous_compiled(2, tmp_path)
    incremental = UnpairedWaveModel(records + additions, previous_model=previous)
    assert incremental.compilation_stats["reused_groups"] > 0
    _assert_equivalent(incremental, UnpairedWaveModel(records + additions))


def test_two_disk_cache_loads_have_independent_numerical_storage(tmp_path):
    from freqai import compiler_cache

    original = UnpairedWaveModel(BASE_RECORDS)
    assert compiler_cache.save_compiled(original, BASE_RECORDS, 2, tmp_path)
    first = compiler_cache.load_compiled(BASE_RECORDS, 2, tmp_path)
    second = compiler_cache.load_compiled(BASE_RECORDS, 2, tmp_path)
    expected = _snapshot(second)
    assert not np.shares_memory(first.transition_spectra.coefficients,
                                second.transition_spectra.coefficients)
    assert not np.shares_memory(first.transition_spectra.token_indices,
                                second.transition_spectra.token_indices)
    assert not np.shares_memory(first.symbol_energy, second.symbol_energy)
    assert not np.shares_memory(first.lexical_roles[0].spectrum, second.lexical_roles[0].spectrum)
    key = next(iter(first.transition_spectra))
    first.transition_spectra[key][:] = 0
    first.supports[key][0] = (int(first.supports[key][0]) + 1) % first.size
    first.lexical_roles[0].spectrum[:] = 0
    assert _snapshot(second) == expected


@pytest.mark.parametrize("mutation", ["role_spectrum", "symbol_energy"])
def test_cache_does_not_persist_auxiliary_diagnostic_mutations(tmp_path, mutation):
    from freqai import compiler_cache

    model = UnpairedWaveModel(BASE_RECORDS)
    if mutation == "role_spectrum":
        # Lexical-role spectrum removal is an existing public diagnostic.
        model.lexical_roles[0].spectrum[:] = 0
    else:
        model.symbol_energy[:] = 0
    assert compiler_cache.save_compiled(model, BASE_RECORDS, 2, tmp_path) is False
    assert compiler_cache.load_compiled(BASE_RECORDS, 2, tmp_path) is None
