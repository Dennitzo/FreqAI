import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from freqai.memory import Document, WaveMemory
from freqai.paged_memory import PagedWaveMemory, WORKING_CHARS
from freqai.store import MemoryStore
from freqai.server import WaveEngine


def store_with(tmp_path, docs):
    store = MemoryStore(tmp_path / 'memory.sqlite3')
    store.migrate_information_only()
    store.append_documents(docs)
    return store


def test_paged_matches_full_wave_and_reuses_cache(tmp_path):
    docs = [Document('a', 'Frequenz\n\nDie Frequenz ist eine Anzahl von Schwingungen.', 'Quelle'),
            Document('b', 'Einheit\n\nDie Einheit der Frequenz ist Hertz.'), Document('empty', '')]
    store = store_with(tmp_path, docs)
    expected = WaveMemory(docs, **store.configuration())
    actual = PagedWaveMemory(store)
    assert actual.compiled == 3
    assert list(actual.documents) == docs
    assert actual.documents[1:] == docs[1:]
    for t in (0, .17, 12345.6):
        before, after = expected.snapshot(t, 41), actual.snapshot(t, 41)
        for field in ('displacement', 'quadrature', 'energy', 'physical_energy'):
            np.testing.assert_allclose(after[field], before[field], rtol=1e-11, atol=1e-10)
        assert after['modal_count'] == before['modal_count']
    with patch('freqai.paged_memory._compile_record', side_effect=AssertionError('cache missed')):
        warm = PagedWaveMemory(store)
    assert warm.compiled == 0
    store.append_documents([Document('new', 'Ein Kondensator ist ein elektrischer Speicher.')])
    grown = PagedWaveMemory(store)
    assert grown.compiled == 1
    assert len(actual.documents) == 3  # Old engine snapshots stay immutable.


def test_resume_after_interruption(tmp_path):
    store = store_with(tmp_path, [Document(str(i), 'Test ' * 100+str(i)) for i in range(5)])
    from freqai.paged_memory import _compile_record
    calls = 0
    def fail(row):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('interrupted')
        return _compile_record(row)
    with patch('freqai.paged_memory.BATCH_BYTES', 1), patch('freqai.paged_memory._compile_record', side_effect=fail):
        with pytest.raises(RuntimeError, match='interrupted'):
            PagedWaveMemory(store)
    resumed = PagedWaveMemory(store)
    assert resumed.compiled == 3
    assert resumed.snapshot(0)['document_count'] == 5


def test_selection_uses_article_tail_and_is_bounded(tmp_path):
    text = 'Artikel\n\n' + 'Unabhängiger Text. ' * 4000 + '\n\nEin Zebrakondensator ist ein elektrischer Speicher.'
    store = store_with(tmp_path, [Document('long', text)])
    paged = PagedWaveMemory(store)
    working = paged.working_memory('Was ist ein Zebrakondensator?')
    assert any('Zebrakondensator' in d.text for d in working.documents)
    assert sum(len(d.text) for d in working.documents) <= WORKING_CHARS
    with sqlite3.connect(paged.cache_path) as c:
        assert ''.join(row[0] for row in c.execute('select text from passages order by rowid')) == text


def test_large_store_never_calls_dense_constructor(tmp_path):
    store = store_with(tmp_path, [Document('test', 'Information')])
    config = store.configuration()
    config['dimensions'] = 262144
    # 80 moderate records exceed the dense safety budget by spectra alone.
    store.append_documents([Document(str(i), 'Ein Speicher ist ein Bauteil. '+str(i)) for i in range(80)])
    with patch.object(store, 'configuration', return_value=config), patch.object(WaveMemory, '_rebuild', side_effect=AssertionError('dense allocation')):
        memory = store.load_memory()
        engine = WaveEngine(memory, central_store=store)
        assert engine.state()['compiler_ready']
        result = engine.ask('Was ist ein Speicher?')
        assert result['configuration']['compiler_scope'] == 'retrieved_information_passages'
        assert result['generated']
        engine.close()


def test_paged_engine_append_does_not_enumerate_corpus(tmp_path):
    store = store_with(tmp_path, [Document('a', 'Ein Speicher ist ein Bauteil.')])
    engine = WaveEngine(PagedWaveMemory(store), central_store=store)
    with patch('freqai.paged_memory.DiskDocuments.__iter__', side_effect=AssertionError('whole corpus read')):
        with patch.object(store, 'load_memory', side_effect=lambda: PagedWaveMemory(store)):
            result = engine.add_document('Eine Spule ist ein Bauteil.', 'test')
    assert result['document_count'] == 2
    engine.close()


def test_direct_dense_allocation_is_rejected_before_rebuild():
    documents = [Document(str(i), 'test') for i in range(80)]
    with patch.object(WaveMemory, '_rebuild', side_effect=AssertionError('allocated')):
        with pytest.raises(ValueError, match='RAM-Budget'):
            WaveMemory(documents, dimensions=262144)


def test_growth_switches_to_disk_before_dense_allocation(tmp_path):
    store = store_with(tmp_path, [Document('first', 'Ein Speicher ist ein Bauteil.')])
    engine = WaveEngine(store.load_memory(), central_store=store)
    with patch('freqai.memory.MAX_DENSE_BYTES', 1), patch.object(WaveMemory, '_rebuild', side_effect=AssertionError('allocated')):
        result = engine.add_document('Eine Spule ist ein Bauteil.', 'test')
        assert result['document_count'] == 2
        assert engine.memory.paged
    engine.close()


def test_followup_keeps_subject_instead_of_generic_facet(tmp_path):
    docs = [Document('frequency', 'Frequenz\n\nDie Frequenz ist eine Wiederholungszahl. Die Einheit der Frequenz ist Hertz.')]
    docs += [Document(str(i), f'Einheit\n\nEine Einheit ist ein allgemeiner Begriff Nummer {i}.') for i in range(40)]
    paged = PagedWaveMemory(store_with(tmp_path, docs))
    working = paged.working_memory('Und welche Einheit hat sie?', {'unpaired': {'subjects': ['frequenz']}})
    assert any('Hertz' in doc.text for doc in working.documents)
    assert all('allgemeiner Begriff' not in doc.text for doc in working.documents)


def test_context_frames_reuse_selection_without_sql(tmp_path):
    paged = PagedWaveMemory(store_with(tmp_path, [Document('test', 'Ein Speicher ist ein Bauteil.')]))
    working = paged.working_memory('Was ist ein Speicher?')
    with patch('freqai.paged_memory.connect', side_effect=AssertionError('repeated FTS query')):
        assert paged.working_memory('Was ist ein Speicher?') is working
        assert paged.working_memory('', {'unpaired': {'working_set_key': working._paged_key}}) is working


def test_unpaginated_api_is_bounded_for_disk_store(tmp_path):
    docs = [Document(str(i), 'Information '+str(i)) for i in range(60)]
    store = store_with(tmp_path, docs)
    engine = WaveEngine(PagedWaveMemory(store), central_store=store)
    result = engine.documents()
    assert len(result['documents']) == 50
    assert result['total'] == 60 and result['next_offset'] == 50
    engine.close()
