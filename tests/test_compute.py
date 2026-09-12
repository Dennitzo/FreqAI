"""Numerical parity, failure fallback, and real CUDA execution when available."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.fft import fft

from freqai import compute
from freqai.codec import decode_text
from freqai.memory import Document, WaveMemory


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setenv("FREQAI_COMPUTE", "cpu")
    if not compute.cuda_for(1, backend="cuda"):
        pytest.skip(f"CUDA unavailable: {compute.compute_status()['fallback_reason']}")
    return compute


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_fft_cpu_reference_and_precision(dtype):
    values = np.random.default_rng(7).normal(size=(23, 97)).astype(dtype)
    if dtype == np.complex128:
        values += 1j*values[:, ::-1]
    result = compute.fft_rows(values, backend="cpu")
    expected = fft(values.astype(np.complex128 if dtype == np.complex128 else np.float64), norm="ortho")
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.complex128


def test_small_auto_batches_and_status_do_not_probe_cuda(monkeypatch):
    monkeypatch.setenv("FREQAI_COMPUTE", "auto")
    monkeypatch.setattr(compute, "_probe_cuda", lambda: pytest.fail("Unexpected CUDA probe"))
    compute.compute_status()
    compute.fft_rows(np.ones((2, 16)))


def test_auto_host_fft_avoids_transfer_overhead(monkeypatch):
    monkeypatch.setenv("FREQAI_COMPUTE", "auto")
    monkeypatch.setattr(compute, "_probe_cuda", lambda: pytest.fail("Host FFT should stay on CPU"))
    values = np.ones((128, 4096))
    np.testing.assert_array_equal(compute.fft_rows(values), fft(values, norm="ortho"))
    assert compute.compute_status()["last_backend_by_operation"]["fft"] == "cpu"


def test_unavailable_cuda_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(compute, "_probe_cuda", lambda: False)
    values = np.random.default_rng(8).normal(size=(17, 127))
    np.testing.assert_array_equal(compute.fft_rows(values, backend="cuda"), fft(values, norm="ortho"))


@pytest.mark.parametrize("complex_input", [False, True])
def test_cuda_fft_matches_scipy_and_uses_selected_devices(cuda, complex_input):
    values = np.random.default_rng(11).normal(size=(128, 4096))
    if complex_input:
        values = values+1j*values[:, ::-1]
    before = compute.compute_status()
    actual = compute.fft_rows(values, backend="cuda")
    np.testing.assert_allclose(actual, fft(values, norm="ortho"), atol=1e-13, rtol=1e-13)
    after = compute.compute_status()
    assert after["completed_operations"].get("cuda_fft", 0) == before["completed_operations"].get("cuda_fft", 0)+1
    for device in after["cuda_devices"]:
        key = str(device["id"])
        assert after["device_completed_operations"][key] == before["device_completed_operations"].get(key, 0)+1


def _memory():
    texts = ["ab", "", "Grüße 中文 🌊", "Alle adressierten Moden schwingen gemeinsam. "*7,
             "Eine etwas längere und anders adressierte Information. "*11]
    return WaveMemory([Document(str(i), text, "Prüfung") for i, text in enumerate(texts*5)], dimensions=32)


@pytest.mark.parametrize("time_s", [0.0, 1/60, 7.5, 1e9, 1e12])
def test_cuda_snapshot_matches_full_cpu_idct_at_all_clock_scales(cuda, monkeypatch, time_s):
    memory = _memory()
    before = [packet.sha256 for packet in memory._packets]
    expected = memory.snapshot(time_s, points=128)
    monkeypatch.setenv("FREQAI_COMPUTE", "cuda")
    actual = memory.snapshot(time_s, points=128)
    assert compute.compute_status()["last_backend"] == "cuda"
    for key in ("displacement", "quadrature"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=2e-12)
    for key in ("energy", "physical_energy"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-14)
    assert actual["modal_count"] == expected["modal_count"]
    assert [packet.sha256 for packet in memory._packets] == before
    assert [decode_text(packet, time_s) for packet in memory.payloads] == [doc.text for doc in memory.documents]


def test_cuda_append_and_concurrent_snapshots_keep_addresses(cuda, monkeypatch):
    memory = _memory()
    candidate = memory.with_documents_added([Document("extra", "Zusätzliche Moden. "*9)])
    np.testing.assert_array_equal(candidate._coefficients[:len(memory._coefficients)], memory._coefficients)
    times = [0.0, .731, 19.125, 1e9]
    expected = [candidate.snapshot(value, points=24) for value in times]
    monkeypatch.setenv("FREQAI_COMPUTE", "cuda")
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda value: candidate.snapshot(value, points=24), times))
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left["displacement"], right["displacement"], atol=2e-12, rtol=1e-11)
        assert left["energy"] == pytest.approx(right["energy"], rel=2e-14)


def test_cuda_runtime_failure_is_visible_and_recomputed_on_cpu(cuda, monkeypatch):
    class BrokenCuda:
        @property
        def cuda(self):
            raise RuntimeError("Simulated CUDA device failure")

    monkeypatch.setattr(compute, "_cupy", BrokenCuda())
    monkeypatch.setattr(compute, "_failure", None)
    values = np.random.default_rng(13).normal(size=(17, 127))
    actual = compute.fft_rows(values, backend="cuda")
    np.testing.assert_array_equal(actual, fft(values, norm="ortho"))
    assert "Simulated CUDA device failure" in compute.compute_status()["fallback_reason"]
    assert compute.compute_status()["last_backend"] == "cpu"
