"""Independent algebra checks for the research proposal, not a language benchmark.

Run from the repository root with the project Python interpreter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def run() -> dict:
    rng = np.random.default_rng(20260906)
    n_inputs, n_words = 32, 64
    counts = rng.poisson(0.7, size=(n_words, n_inputs)).astype(np.float64)
    input_basis = np.fft.fft(np.eye(n_inputs), axis=0, norm="ortho")
    output_basis = np.fft.fft(np.eye(n_words), axis=0, norm="ortho")
    coupling = output_basis @ counts @ input_basis.conj().T
    next_counts = rng.poisson(0.02, size=counts.shape).astype(np.float64)
    delta_coupling = output_basis @ next_counts @ input_basis.conj().T
    max_error = max_imaginary = max_incremental_error = 0.0
    prompt_effects = []
    for _ in range(1000):
        prompt = rng.random(n_inputs)
        prompt /= prompt.sum()
        time_s = float(rng.uniform(0.0, 86400.0))
        input_phase = np.exp(2j * np.pi * np.arange(n_inputs) * 0.03125 * time_s)
        output_phase = np.exp(2j * np.pi * np.arange(n_words) * 0.0625 * time_s)
        prompt_wave = input_phase * (input_basis @ prompt)
        data_wave = output_phase[:, None] * coupling * input_phase.conj()[None, :]
        result_wave = data_wave @ prompt_wave
        output = output_basis.conj().T @ (output_phase.conj() * result_wave)
        max_error = max(max_error, float(np.max(np.abs(output.real - counts @ prompt))))
        max_imaginary = max(max_imaginary, float(np.max(np.abs(output.imag))))
        incremental = output_basis.conj().T @ ((coupling + delta_coupling) @ (input_basis @ prompt))
        max_incremental_error = max(
            max_incremental_error,
            float(np.max(np.abs(incremental - (counts + next_counts) @ prompt))),
        )
        shuffled = prompt[rng.permutation(n_inputs)]
        prompt_effects.append(float(np.linalg.norm(counts @ prompt - counts @ shuffled)))

    # A scalar interference intensity loses phase/sign. These two different
    # resulting fields are observationally identical to an intensity-only decoder.
    field = rng.normal(size=n_words) + 1j * rng.normal(size=n_words)
    intensity_collision = np.array_equal(np.abs(field) ** 2, np.abs(-field) ** 2)

    # Unstructured addition of encoded bytes is a mixture, not an answer function.
    first = np.frombuffer(b"Mir geht es gut.", dtype=np.uint8).astype(float)
    second = np.frombuffer(b"Wie geht es dir?", dtype=np.uint8).astype(float)
    size = max(len(first), len(second))
    first = np.pad(first, (0, size - len(first)))
    second = np.pad(second, (0, size - len(second)))
    raw_superposition = np.fft.ifft((np.fft.fft(first) + np.fft.fft(second)) / 2).real
    raw_bytes = np.clip(np.rint(raw_superposition), 0, 255).astype(np.uint8).tobytes()
    result = {
        "scope": "Independent numerical identities; no semantic-quality claim",
        "seed": 20260906,
        "trials": 1000,
        "input_channels": n_inputs,
        "output_tokens": n_words,
        "max_direct_vs_spectral_error": max_error,
        "max_imaginary_residue": max_imaginary,
        "max_additive_update_error": max_incremental_error,
        "minimum_changed_prompt_output_l2": min(prompt_effects),
        "maximum_changed_prompt_output_l2": max(prompt_effects),
        "opposite_fields_have_identical_intensity": intensity_collision,
        "raw_byte_superposition": raw_bytes.decode("utf-8", errors="replace"),
    }
    assert max_error < 1e-11 and max_imaginary < 1e-11
    assert max_incremental_error < 1e-11 and min(prompt_effects) > 0.0
    assert intensity_collision
    return result


if __name__ == "__main__":
    result = run()
    destination = Path(__file__).with_name("spectral_identities.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
