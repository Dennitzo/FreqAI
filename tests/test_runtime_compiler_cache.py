"""Exercise persistent compiler reuse through the real runtime entry point."""
from concurrent.futures import ThreadPoolExecutor
import threading

from freqai.memory import Document, WaveMemory
from freqai.store import MemoryStore
from freqai.unpaired_runtime import generator_for


def test_restart_loads_compiler_without_calling_constructor(tmp_path, monkeypatch):
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    monkeypatch.delenv("FREQAI_COMPILER_CACHE", raising=False)
    store = MemoryStore(tmp_path / "knowledge.sqlite3")
    store.append_documents([Document("fact", "Die Frequenz ist eine Anzahl von Schwingungen.", "Test")])
    first_memory = store.load_memory()
    first = generator_for(first_memory).model
    signature = first.model_signature()
    assert first_memory._compiler_status["cache_saved"]
    expected = first.generate("Was ist eine Frequenz?")
    from freqai.unpaired import UnpairedWaveModel

    def forbidden(*args, **kwargs):
        raise AssertionError("An unchanged corpus must load the saved compiler")

    monkeypatch.setattr(UnpairedWaveModel, "__init__", forbidden)
    second_memory = MemoryStore(store.path).load_memory()
    restored = generator_for(second_memory).model
    assert second_memory._compiler_status["cache"] == "hit"
    assert restored is not first
    assert restored.model_signature() == signature
    assert restored.generate("Was ist eine Frequenz?") == expected
    assert store.stats()["document_count"] == 1


def test_live_append_reuses_previous_groups_and_persists_new_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    monkeypatch.delenv("FREQAI_COMPILER_CACHE", raising=False)
    store = MemoryStore(tmp_path / "knowledge.sqlite3")
    records = [Document("f", "Die Frequenz ist eine Anzahl von Schwingungen.", "Test")]
    store.append_documents(records)
    before = store.load_memory()
    original = generator_for(before).model
    original_signature = original.model_signature()
    addition = Document("z", "Ein Zorblin ist ein grüner Stein.", "Test")
    candidate = before.with_documents_added([addition])
    result = generator_for(candidate).model
    assert candidate._compiler_status["cache"] == "miss"
    assert candidate._compiler_status["reuse"]["groups_reused"] > 0
    assert original.model_signature() == original_signature
    from freqai.unpaired import UnpairedWaveModel
    fresh = UnpairedWaveModel([{"id": doc.id, "text": doc.text, "source": doc.source}
                               for doc in records + [addition]])
    assert result.model_signature() == fresh.model_signature()
    assert result.generate("Was ist ein Zorblin?") == fresh.generate("Was ist ein Zorblin?")


def test_unrelated_corpora_do_not_share_a_global_build_lock(monkeypatch):
    monkeypatch.setenv("FREQAI_COMPILER_CACHE", "off")
    import freqai.unpaired
    barrier = threading.Barrier(2)

    class ConcurrentModel:
        def __init__(self, *args, **kwargs):
            barrier.wait(timeout=5)

    monkeypatch.setattr(freqai.unpaired, "UnpairedWaveModel", ConcurrentModel)
    memories = [WaveMemory([]), WaveMemory([])]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(generator_for, memories))
    assert results[0].model is not results[1].model


def test_append_while_server_is_stopped_reuses_saved_base(tmp_path, monkeypatch):
    monkeypatch.setenv("FREQAI_PARALLEL", "0")
    monkeypatch.delenv("FREQAI_COMPILER_CACHE", raising=False)
    store = MemoryStore(tmp_path / "knowledge.sqlite3")
    store.append_documents([Document("f", "Die Frequenz ist eine Anzahl von Schwingungen.", "Test")])
    generator_for(store.load_memory())
    store.append_documents([Document("z", "Ein Zorblin ist ein grüner Stein.", "Test")])
    restarted = MemoryStore(store.path).load_memory()
    model = generator_for(restarted).model
    assert restarted._compiler_status["cache"] == "miss"
    assert restarted._compiler_status["previous_cache_loaded"]
    assert restarted._compiler_status["reuse"]["groups_reused"] > 0
    from freqai.unpaired import UnpairedWaveModel
    fresh = UnpairedWaveModel([{"id": doc.id, "text": doc.text, "source": doc.source}
                               for doc in restarted.documents])
    assert model.model_signature() == fresh.model_signature()


def test_simultaneous_first_requests_build_one_model(monkeypatch):
    monkeypatch.setenv("FREQAI_COMPILER_CACHE", "off")
    import freqai.unpaired
    entered = threading.Event()
    release = threading.Event()
    started = threading.Event()
    calls = []

    class OneModel:
        def __init__(self, *args, **kwargs):
            calls.append(1)
            entered.set()
            assert release.wait(timeout=5)

    monkeypatch.setattr(freqai.unpaired, "UnpairedWaveModel", OneModel)
    memory = WaveMemory([])

    def second_request():
        started.set()
        return generator_for(memory)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(generator_for, memory)
        assert entered.wait(timeout=5)
        second = executor.submit(second_request)
        assert started.wait(timeout=5)
        release.set()
        assert first.result(timeout=5) is second.result(timeout=5)
    assert len(calls) == 1
