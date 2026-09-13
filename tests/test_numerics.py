"""Numerical properties and deliberately exposed limits of the wave method."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dct, idct

from freqai.codec import (
    WavePacket, decode_coefficients, decode_text, encode_text, modal_state,
    recover_coefficients, sample_wave,
)
from freqai.features import feature_vector
from freqai.memory import Document, WaveMemory


@pytest.mark.parametrize("text", ["", "\x00", "abc\x00def", "Grüße äöüß €",
                                  "中文 日本語 한국어", "🌊👩🏽‍🔬 e\u0301 é", "مرحبا",
                                  "Reihenfolge bleibt vollständig erhalten. " * 30])
@pytest.mark.parametrize("time_s", [0., .0137, 17.123, 1e12])
def test_text_roundtrip_through_full_spatial_quadrature_grid(text, time_s):
    packet = encode_text(text)
    grid = sample_wave(packet, time_s, points=max(2, len(packet.coefficients)))
    # Independent spatial analysis of both measured quadratures, followed by
    # the public decoder; this exercises more than a direct encode/decode call.
    q = dct(grid["displacement"], norm="ortho")
    p = dct(grid["quadrature"], norm="ortho")
    actual = decode_coefficients(recover_coefficients(q, p, time_s), packet.byte_length, packet.sha256)
    assert actual == text


def test_parseval_and_both_energy_definitions():
    text = "Frequenzen: λ, φ, ω. " * 10
    payload = text.encode("utf-8")
    spatial = (np.frombuffer(payload, dtype=np.uint8).astype(float) - 127.5) / 127.5
    packet = encode_text(text)
    expected_norm = float(spatial @ spatial)
    omega = 2 * np.pi * np.arange(len(payload)) * 30 / len(payload)
    expected_physical = .5 * np.sum(omega**2 * packet.coefficients**2)
    assert packet.coefficients @ packet.coefficients == pytest.approx(expected_norm, rel=1e-13)
    for time_s in np.linspace(0., 3.41, 101):
        snapshot = sample_wave(packet, float(time_s))
        assert snapshot["energy"] == pytest.approx(expected_norm, rel=1e-13)
        assert snapshot["physical_energy"] == pytest.approx(expected_physical, rel=1e-13)


def test_small_bounded_noise_preserves_utf8_and_large_noise_is_rejected():
    text = "Keine stillschweigende Beschädigung: 中文 🌊 äöüß."
    packet = encode_text(text)
    rng = np.random.default_rng(46)
    noise = rng.normal(size=len(packet.coefficients))
    noise *= (.25 / 127.5) / np.linalg.norm(noise)
    # Orthonormal inverse transform bounds each byte error by .25 < .5.
    protected = WavePacket(packet.coefficients + noise, packet.byte_length, packet.sha256)
    assert decode_text(protected, .713) == text
    broken = WavePacket(packet.coefficients + rng.normal(0., .1, len(noise)), packet.byte_length, packet.sha256)
    with pytest.raises(ValueError, match="corruption"):
        decode_text(broken)


def test_time_stamp_mismatch_is_detected():
    packet = encode_text("Die Ausleseuhr muss dieselbe Phase wie der Speicher kennen.")
    q, p = modal_state(packet, .0137)
    wrong = recover_coefficients(q, p, .0789)
    with pytest.raises(ValueError, match="corruption"):
        decode_coefficients(wrong, packet.byte_length, packet.sha256)


def test_displacement_alone_loses_a_mode_at_quarter_period():
    coefficients = np.zeros(64)
    coefficients[1] = 2.7
    packet = WavePacket(coefficients, 64, "numerical-test")
    time_s = 64 / 120.
    q, p = modal_state(packet, time_s)
    assert np.linalg.norm(q) < 1e-14
    assert np.linalg.norm(p) == pytest.approx(2.7)
    assert np.allclose(recover_coefficients(q, p, time_s), coefficients, atol=1e-14)


def test_downsampled_display_has_insufficient_rank_for_exact_codec():
    dimensions, display_points = 128, 16
    spatial_basis = idct(np.eye(dimensions), norm="ortho", axis=0)
    indices = np.linspace(0, dimensions - 1, display_points).astype(int)
    measurement_matrix = spatial_basis[indices]
    rank = np.linalg.matrix_rank(measurement_matrix)
    assert rank == display_points
    assert dimensions - rank == 112


@pytest.mark.parametrize("mode", ["raw_bytes", "words", "hybrid"])
@pytest.mark.parametrize("dimensions", [256, 1024, 4096])
def test_interference_equals_direct_dot_product_and_is_time_invariant(mode, dimensions):
    memory = WaveMemory([Document("a", "Kupfer leitet elektrischen Strom."),
                         Document("b", "Paris ist die Hauptstadt Frankreichs."),
                         Document("c", "Eine Geige hat schwingende Saiten.")],
                        dimensions=dimensions, feature_mode=mode)
    prompt = "Leitet Kupfer Strom?"
    direct = memory.direct_cosine_scores(prompt)
    query = np.fft.fft(feature_vector(prompt, dimensions, mode), norm="ortho")
    energy_cross = .5 * (np.sum(np.abs(memory.spectra + query)**2, axis=1)
                        - np.sum(np.abs(memory.spectra)**2, axis=1) - np.sum(np.abs(query)**2))
    np.testing.assert_allclose(direct, energy_cross, atol=1e-13, rtol=0)
    initial_answer = memory.ask(prompt)["answer"]
    for time_s in (0., .271, 75.09, 1e12):
        np.testing.assert_allclose(memory.scores(prompt, time_s), direct, atol=1e-13, rtol=0)
        assert memory.ask(prompt, time_s=time_s)["answer"] == initial_answer


def test_signed_hash_collision_alone_cannot_produce_an_answer():
    # 33 tokens in 16 bins with two signs guarantee an identical signed address.
    seen = {}
    pair = None
    for index in range(33):
        token = f"zznumerictoken{index}"
        vector = feature_vector(token, 16, "words")
        position = int(np.argmax(np.abs(vector)))
        address = (position, float(vector[position]))
        if address in seen:
            pair = (seen[address], token)
            break
        seen[address] = token
    assert pair is not None
    stored, probe = pair
    memory = WaveMemory([Document("collision", stored)], dimensions=16, feature_mode="words")
    assert memory.scores(probe)[0] == pytest.approx(1.)
    assert memory.ask(probe)["abstained"] is True


def test_equal_scores_have_stable_answers_despite_floating_point_phase_noise():
    memory = WaveMemory([Document("f01", "Die Hauptstadt Frankreichs ist Paris."),
                         Document("f02", "Die Hauptstadt Japans heißt Tokio."),
                         Document("f03", "Die Hauptstadt Italiens ist Rom.")],
                        dimensions=256, feature_mode="words")
    # Japan and Japans have distinct fixed word features. All three documents
    # can therefore share the same mathematical score for this prompt.
    prompt = "Welche Hauptstadt hat Japan?"
    initial = memory.ask(prompt)["answer"]
    for time_s in (0., .0137, 1., 1234.5, 1e12):
        assert memory.ask(prompt, time_s=time_s)["answer"] == initial


def test_exact_threshold_does_not_flicker_with_roundoff():
    text = "Kupfer leitet elektrischen Strom."
    memory = WaveMemory([Document("a", text)], dimensions=256, feature_mode="hybrid", min_score=1.)
    for time_s in (0., .0137, 1., 1234.5, 1e12):
        assert memory.ask(text, time_s=time_s)["matches"][0]["id"] == "a"


def test_single_hrr_binding_is_invertible_but_unlabelled_superposition_has_noise():
    rng = np.random.default_rng(4)
    size = 256
    values = rng.normal(size=(2, size))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    keys = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(2, size // 2 + 1)))
    keys[:, (0, -1)] = 1.
    bindings = keys * np.fft.rfft(values, norm="ortho", axis=1)
    one = np.fft.irfft(np.conj(keys[0]) * bindings[0], n=size, norm="ortho")
    both = np.fft.irfft(np.conj(keys[0]) * np.sum(bindings, axis=0), n=size, norm="ortho")
    np.testing.assert_allclose(one, values[0], atol=1e-14, rtol=0)
    assert np.linalg.norm(both - values[0]) == pytest.approx(1., abs=1e-13)


def test_benchmark_has_required_groups_and_no_question_leakage():
    from experiments.run_experiments import validate_benchmark

    # Validate the format with fresh synthetic records, not an imported corpus copy.
    benchmark = {"documents": [{"id": f"test-{i}", "text": f"Speichermarker{i}."}
                               for i in range(40)], "cases": []}
    for group, count in (("known", 32), ("null", 10), ("semantic_no_overlap", 8)):
        benchmark["cases"].extend({"id": f"{group}-{i}", "group": group,
                                   "prompt": f"Prüfbegriff{i}?", "expected_id":
                                   None if group == "null" else f"test-{i}"}
                                  for i in range(count))
    validate_benchmark(benchmark)
    assert len(benchmark["documents"]) == 40
    assert len(benchmark["cases"]) == 50


def test_wave_archive_roundtrip_preserves_retrieval(tmp_path):
    memory = WaveMemory([Document("a", "Unicode: 中文 🌊 und Grüße.", "fixture"),
                         Document("b", "Kupfer leitet Strom.", "fixture")], dimensions=256)
    expected = memory.ask("Leitet Kupfer Strom?", time_s=1.23)
    destination = tmp_path / "memory.npz"
    memory.save(destination)
    restored = WaveMemory.load(destination)
    assert restored.documents == memory.documents
    assert restored.ask("Leitet Kupfer Strom?", time_s=1.23) == expected


def test_noisy_archive_is_not_silently_accepted(tmp_path):
    memory = WaveMemory([Document("a", "Gespeicherte Nutzdaten müssen überprüfbar bleiben.")], dimensions=256)
    destination = tmp_path / "memory.npz"
    memory.save(destination)
    with np.load(destination, allow_pickle=False) as data:
        coefficients = data["coefficients"].copy()
        header = data["header"].copy()
    coefficients[0] += 10.
    np.savez_compressed(destination, coefficients=coefficients, header=header)
    with pytest.raises(ValueError, match="corruption"):
        WaveMemory.load(destination)
