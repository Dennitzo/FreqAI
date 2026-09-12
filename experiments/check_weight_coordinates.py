"""Separate a Fourier coordinate change from directly computed data coefficients.

The synthetic matrices are a linear algebra control, not a language model and
not an assertion that the old conversational decoder contains trained weights.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.generative import spectral_convolution


def run():
    rng = np.random.default_rng(20260906)
    errors, singular_errors, energies, changes = [], [], [], []
    for _ in range(64):
        n = 17
        transform = np.fft.fft(np.eye(n), axis=0, norm="ortho")
        operator = rng.normal(size=(n, n))
        x = rng.normal(size=n)
        spectral_operator = transform @ operator @ transform.conj().T
        actual = transform.conj().T @ spectral_operator @ (transform @ x)
        errors.append(float(np.max(np.abs(actual - operator @ x))))
        singular_errors.append(float(np.max(np.abs(np.linalg.svd(operator, compute_uv=False) -
                                                   np.linalg.svd(spectral_operator, compute_uv=False)))))
        counts = rng.integers(1, 30, size=n)
        amplitudes = np.sqrt(counts / counts.sum())
        coefficients = np.fft.fft(amplitudes, norm="ortho")
        energies.append(float(np.sum(np.abs(coefficients)**2)))
        assert np.array_equal(coefficients, np.fft.fft(np.sqrt(counts/counts.sum()), norm="ortho"))
        prompt = rng.random(n)
        prompt /= np.linalg.norm(prompt)
        spectrum = coefficients + spectral_convolution(coefficients, np.fft.fft(prompt, norm="ortho"))
        probabilities = np.abs(np.fft.ifft(spectrum, norm="ortho"))**2
        probabilities /= probabilities.sum()
        reference = amplitudes**2 * (1 + prompt)**2
        reference /= reference.sum()
        np.testing.assert_allclose(probabilities, reference, rtol=1e-12, atol=1e-12)
        changed = np.roll(amplitudes, 1)**2 * (1 + prompt)**2
        changed /= changed.sum()
        changes.append(float(np.max(np.abs(changed-probabilities))))
    assert max(errors) < 1e-12 and max(singular_errors) < 1e-12
    assert max(abs(value-1) for value in energies) < 1e-12 and all(value > 1e-6 for value in changes)
    return {"states": 64, "basis_change_max_output_error": max(errors),
            "basis_change_max_singular_value_error": max(singular_errors),
            "direct_coefficient_max_energy_error": max(abs(value-1) for value in energies),
            "deterministic_coefficient_rebuilds": 64, "data_changes_affect_output": len(changes),
            "optimizer_steps": 0, "fitted_parameters": 0,
            "conclusion": "A Fourier basis change preserves the operator and its parameter dependence. Direct count-derived unit-energy coefficients avoid fitting, but are still data-dependent numerical weights.",
            "scope": "Synthetic linear algebra control, not a conversation or factual quality benchmark."}


if __name__ == "__main__":
    result = run()
    path = ROOT / "results/information_corpus/weight_coordinate_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
