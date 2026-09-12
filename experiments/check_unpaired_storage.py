"""Audit every final unpaired storage packet and three complete wave frames.

Only explicit corpus files are read. This process never opens SQLite or builds
a language generator. Frame timings include finite-value and transform-coverage
instrumentation, and therefore are conservative diagnostic timings.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import weakref

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.cli import read_documents
from freqai.codec import decode_text
from freqai.features import text_spectrum
from freqai import memory as memory_module
from freqai.memory import Document, WaveMemory
from freqai.store import DEFAULT_CONFIGURATION

SOURCES = (
    ("memory/imports/information_wikipedia_20000_v3.jsonl", 20000,
     "2f5b876d904346ca4b9082e05ff5e4bc5e8a729f2a479c777f47e1332a0a5674"),
    ("memory/information/conversation_facts.jsonl", 45,
     "688706468d39cc203f83cd8c2f0e4b9b8f7b65261a4469d6bcfcff3c274cea5e"),
    ("memory/information/conversation_lexicon_v1.jsonl", 48,
     "092dbbb0bf668bc6e5c404f1bfcc2e03b3f22c75ae2d9cff7c2b421606762f2e"),
)
CODEC_TIMES = (0.0, 86400.125)
FRAME_TIMES = (0.0, 13.75, 86400.125)
OUTPUT = ROOT / "results/unpaired_system/storage_audit.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_fixture_check():
    """A separate in-memory fixture; never enters the active information set."""
    document = Document("archive-check", "Ein Prüftext mit Umlauten: ä, ö, ü und ß.",
                        "Historische Prüfquelle", "Historischer Gesprächsanlass")
    memory = WaveMemory([document], **DEFAULT_CONFIGURATION)
    expected_metadata = json.dumps({"id": document.id, "source": document.source,
                                    "prompt": document.prompt}, ensure_ascii=False, separators=(",", ":"))
    expected_archive = json.dumps([{"id": document.id, "text": document.text,
                                    "source": document.source, "prompt": document.prompt}],
                                  ensure_ascii=False, separators=(",", ":"))
    for time_s in CODEC_TIMES:
        assert decode_text(memory.metadata_payloads[0], time_s).encode("utf-8") == expected_metadata.encode("utf-8")
        assert decode_text(memory.archive, time_s).encode("utf-8") == expected_archive.encode("utf-8")
    expected_key = text_spectrum(document.text, DEFAULT_CONFIGURATION["dimensions"], DEFAULT_CONFIGURATION["feature_mode"])
    legacy_key = text_spectrum(document.prompt, DEFAULT_CONFIGURATION["dimensions"], DEFAULT_CONFIGURATION["feature_mode"])
    assert np.array_equal(memory.spectra[0], expected_key)
    assert not np.array_equal(memory.spectra[0], legacy_key)
    del memory
    gc.collect()
    return {"scope": "Separate synthetic historical fixture, not a productive archive audit",
            "historical_metadata_and_export_checks": 4, "historical_prompt_bytes_preserved": True,
            "diagnostic_index_equals_text_spectrum": True, "diagnostic_index_differs_from_prompt_spectrum": True}


def frame_with_coverage(memory, time_s):
    """Observe the real snapshot implementation without changing its formulas."""
    original_idct, original_phase = memory_module.idct, memory_module.phase_angles
    transforms, phase_sizes = [], []

    def checked_idct(values, *args, **kwargs):
        assert np.all(np.isfinite(values)), "Nonfinite modal quadrature"
        result = original_idct(values, *args, **kwargs)
        assert result.shape == values.shape
        assert np.all(np.isfinite(result)), "Nonfinite full spatial result"
        transforms.append((tuple(values.shape), int(values.size)))
        return result

    def checked_phase(size, clock):
        values = original_phase(size, clock)
        assert values.shape == (size,) and np.all(np.isfinite(values))
        phase_sizes.append(size)
        return values

    memory_module.idct, memory_module.phase_angles = checked_idct, checked_phase
    started = time.perf_counter()
    try:
        frame = memory.snapshot(time_s, points=256)
    finally:
        seconds = time.perf_counter() - started
        memory_module.idct, memory_module.phase_angles = original_idct, original_phase
    expected_shapes = [tuple(addresses.shape) for _, addresses in memory._wave_groups for _ in range(2)]
    assert [shape for shape, _ in transforms] == expected_shapes
    assert sum(size for _, size in transforms) == 2 * len(memory._coefficients)
    assert phase_sizes == [size for size, _ in memory._wave_groups] + [memory.dimensions]
    assert np.all(np.isfinite(frame["displacement"])) and np.all(np.isfinite(frame["quadrature"]))
    assert all(np.isfinite(frame[key]) for key in ("energy", "physical_energy", "key_energy"))
    assert frame["document_count"] == len(memory.documents)
    assert frame["modal_count"] == len(memory._coefficients)
    assert frame["text_mode_count"] + frame["metadata_mode_count"] == frame["modal_count"]
    assert frame["key_mode_count"] == len(memory.documents) * memory.dimensions
    frame.update({"seconds_including_audit": seconds,
                  "inverse_dct_calls": len(transforms),
                  "full_spatial_values_computed": sum(size for _, size in transforms),
                  "all_packet_modes_transformed_in_both_quadratures": True,
                  "all_modal_and_spatial_values_finite": True,
                  "phase_vectors_computed": len(phase_sizes)})
    return frame


def main():
    started = time.perf_counter()
    implementation_files = ("freqai/memory.py", "freqai/codec.py", "freqai/features.py", "freqai/store.py",
                            "experiments/check_unpaired_storage.py")
    implementation_hashes = {name: digest(ROOT / name) for name in implementation_files}
    documents, sources = [], []
    for relative_path, count, expected_sha in SOURCES:
        path = ROOT / relative_path
        assert digest(path) == expected_sha, relative_path
        batch = read_documents(path)
        assert len(batch) == count and all(not document.prompt for document in batch)
        sources.append({"path": relative_path, "sha256": expected_sha, "records": count,
                        "bytes": path.stat().st_size})
        documents.extend(batch)
    assert len(documents) == len({document.id for document in documents}) == 20093
    fixture = historical_fixture_check()
    build_started = time.perf_counter()
    memory = WaveMemory(documents, **DEFAULT_CONFIGURATION)
    build_seconds = time.perf_counter() - build_started
    print(json.dumps({"stage": "full_wave_built", "records": len(documents), "seconds": build_seconds,
                      "packet_modes": len(memory._coefficients), "key_modes": memory.spectra.size}), flush=True)

    codec_started = time.perf_counter()
    text_checks = metadata_checks = 0
    expected_stored_bytes = 0
    for index, (document, payload, metadata) in enumerate(zip(documents, memory.payloads, memory.metadata_payloads), 1):
        text_bytes = document.text.encode("utf-8")
        metadata_text = json.dumps({"id": document.id, "source": document.source},
                                   ensure_ascii=False, separators=(",", ":"))
        metadata_bytes = metadata_text.encode("utf-8")
        assert payload.byte_length == len(text_bytes) and metadata.byte_length == len(metadata_bytes)
        expected_stored_bytes += len(text_bytes) + len(metadata_bytes)
        for time_s in CODEC_TIMES:
            assert decode_text(payload, time_s).encode("utf-8") == text_bytes, document.id
            recovered_metadata = decode_text(metadata, time_s)
            assert recovered_metadata.encode("utf-8") == metadata_bytes, document.id
            assert set(json.loads(recovered_metadata)) == {"id", "source"}, document.id
            text_checks += 1
            metadata_checks += 1
        if index % 5000 == 0:
            print(json.dumps({"stage": "byte_roundtrips", "records_checked": index}), flush=True)
    assert text_checks == metadata_checks == 40186
    codec_seconds = time.perf_counter() - codec_started

    coverage_started = time.perf_counter()
    visited = np.zeros(len(memory._coefficients), dtype=bool)
    address_count = 0
    for size, addresses in memory._wave_groups:
        assert addresses.shape[1] == size
        assert np.all((addresses >= 0) & (addresses < len(visited)))
        visited[addresses] = True
        address_count += addresses.size
    assert address_count == len(visited) and np.all(visited)
    del visited
    assert np.all(np.isfinite(memory._coefficients))
    verified_key_modes = 0
    recomputed_key_power = np.zeros(memory.dimensions)
    for offset in range(0, len(documents), 128):
        block = memory.spectra[offset:offset+128]
        assert np.all(np.isfinite(block))
        recomputed_key_power += np.sum(np.abs(block)**2, axis=0)
        verified_key_modes += block.size
    assert verified_key_modes == memory.spectra.size
    assert np.allclose(recomputed_key_power, memory._key_power, rtol=1e-12, atol=1e-12)
    coverage_seconds = time.perf_counter() - coverage_started
    frames = []
    for time_s in FRAME_TIMES:
        frame = frame_with_coverage(memory, time_s)
        assert frame["stored_bytes"] == expected_stored_bytes
        frames.append(frame)
        print(json.dumps({"stage": "full_frame", "time_s": time_s,
                          "seconds": frame["seconds_including_audit"], "energy": frame["energy"],
                          "all_modes": frame["all_packet_modes_transformed_in_both_quadratures"]}), flush=True)
    drift = {}
    for field in ("energy", "physical_energy", "key_energy"):
        values = [frame[field] for frame in frames]
        drift[field] = (max(values) - min(values)) / max(1.0, abs(values[0]))
        assert drift[field] < 1e-12, (field, values)
    assert frames[0]["displacement"] != frames[1]["displacement"]
    assert frames[1]["displacement"] != frames[2]["displacement"]
    generator_modules = [name for name in ("freqai.generative", "freqai.generation_runtime", "freqai.unpaired",
                                          "freqai.unpaired_runtime", "freqai.information_runtime") if name in sys.modules]
    assert not generator_modules
    report = {
        "result": "passed", "sources": sources, "records": len(documents),
        "configuration": DEFAULT_CONFIGURATION, "wave_build_seconds": build_seconds,
        "codec_times": CODEC_TIMES, "codec_checks": text_checks + metadata_checks,
        "text_checks": text_checks, "metadata_checks": metadata_checks,
        "all_exact_utf8_bytes": True, "active_metadata_keys": ["id", "source"],
        "active_metadata_contains_prompt": False, "codec_seconds": codec_seconds,
        "stored_text_and_metadata_bytes": expected_stored_bytes,
        "text_mode_count": frames[0]["text_mode_count"], "metadata_mode_count": frames[0]["metadata_mode_count"],
        "packet_mode_count": frames[0]["modal_count"], "key_mode_count": frames[0]["key_mode_count"],
        "total_packet_and_key_modes": frames[0]["modal_count"] + frames[0]["key_mode_count"],
        "packet_mode_addresses_covered_exactly_once": True,
        "key_modes_checked_finite_and_power_recomputed": verified_key_modes,
        "coverage_audit_seconds": coverage_seconds, "frames": frames,
        "median_snapshot_seconds_including_audit": statistics.median(frame["seconds_including_audit"] for frame in frames),
        "relative_energy_drifts": drift, "historical_fixture": fixture,
        "sqlite_opened": False, "productive_data_changed": False,
        "generation_models_built": 0, "generation_modules_loaded": generator_modules,
        "evaluation_files_read": [], "implementation_sha256": implementation_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
        "scope": "Lossless storage bytes, complete wave arithmetic and instrumented throughput; not language quality or evidence of semantic generalization.",
    }
    for name, expected_sha in implementation_hashes.items():
        assert digest(ROOT / name) == expected_sha, f"Implementation changed during audit: {name}"
    memory_reference = weakref.ref(memory)
    del memory, documents, batch, block, payload, metadata, recomputed_key_power
    gc.collect()
    assert memory_reference() is None
    report["temporary_wave_memory_released"] = True
    report["total_seconds"] = time.perf_counter() - started
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"frames", "sources", "implementation_sha256"}},
                     ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
