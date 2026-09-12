"""Two explicit oscillations and the measured interference between them.

The system knows exactly two kinds of waves:

``data_wave``
    One complex field over **every** symbol mode the compiled corpus excites.
    Its energy per mode is the summed squared amplitude that mode carries in all
    compiled prose transition fields and all declarative corpus roles. It is a
    wave over all data, not over one retrieved sentence.

``prompt_wave``
    One complex field over every symbol mode the current prompt and the persisted
    session context excite, including UTF-8 byte symbols for unknown strings.

Both waves live in the same medium: the same symbol modes with the same stable
carrier frequencies ``f_k``. Their interference is read out per mode. Because a
symbol *is* one eigenmode of the medium, two waves couple only where they excite
the same mode; the measurable coupling is therefore diagonal resonance plus the
coherent superposition of all compatible data fields. Off-diagonal sum-frequency
mixing over ``stable_frequency`` values carries no information (see docs), so it
is computed only as a diagnostic and never enters the probabilities.

Every readout is computed twice: once with the spectral operator (mixing theorem
``conv(F_u a, F_u m) = F_u(a * m)``) and once pointwise in mode space. The two
independent results must agree to ``verify_tolerance``.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len

from .spectral_ops import spectral_convolution

WAVE_VERSION = "two_wave_v2"
#: Largest tolerated difference between the spectral operator and the direct readout.
VERIFY_TOLERANCE = 1e-11
#: Amplitude gain of the all-data wave prior inside each interference readout.
DEFAULT_DATA_GAIN = 0.15


@dataclass(frozen=True)
class WaveField:
    """A standing wave: symbol modes, nonnegative amplitudes, carrier frequencies."""

    kind: str
    indices: np.ndarray          # sorted global symbol indices with nonzero amplitude
    amplitudes: np.ndarray       # float64, same length as indices, all > 0
    frequencies: np.ndarray      # carrier frequency of every listed mode

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.amplitudes.shape != self.indices.shape \
                or self.frequencies.shape != self.indices.shape:
            raise ValueError("Wave field needs one amplitude and frequency per mode")
        if not np.isfinite(self.amplitudes).all() or (self.amplitudes < 0).any():
            raise ValueError("Wave amplitudes must be finite and nonnegative")
        if len(self.indices) > 1 and not np.all(np.diff(self.indices) > 0):
            raise ValueError("Wave mode indices must be sorted and unique")

    @property
    def energy(self) -> float:
        return float(self.amplitudes @ self.amplitudes)

    @property
    def mode_count(self) -> int:
        return int(len(self.indices))

    @property
    def carrier_size(self) -> int:
        return next_fast_len(max(2, len(self.indices)))

    def oscillation(self, time_s: float = 0.0) -> np.ndarray:
        """Complex mode amplitudes at ``time_s`` (unit-energy phases)."""
        if not np.isfinite(time_s):
            raise ValueError("Invalid wave time")
        return self.amplitudes * np.exp(2j * np.pi * np.remainder(self.frequencies * float(time_s), 1.0))

    def spectrum(self, time_s: float = 0.0) -> np.ndarray:
        """Orthonormal spectrum of the oscillation on its own carrier."""
        return np.fft.fft(self.oscillation(time_s), norm="ortho")

    def dense(self, size: int) -> np.ndarray:
        result = np.zeros(size)
        if len(self.indices):
            result[self.indices] = self.amplitudes
        return result

    def normalized(self) -> "WaveField":
        norm = math.sqrt(self.energy)
        if norm <= 0.0:
            return self
        return WaveField(self.kind, self.indices, self.amplitudes / norm, self.frequencies)

    def restrict(self, indices: np.ndarray) -> np.ndarray:
        """Amplitudes of this wave at the requested global modes (zero elsewhere)."""
        result = np.zeros(len(indices))
        if not len(self.indices):
            return result
        position = np.searchsorted(self.indices, indices)
        found = (position < len(self.indices))
        found[found] &= self.indices[position[found]] == indices[found]
        result[found] = self.amplitudes[position[found]]
        return result

    def snapshot(self, time_s: float = 0.0, points: int = 128) -> dict:
        if type(points) is not int or points < 2:
            raise ValueError("Invalid display resolution")
        signal = self.oscillation(time_s)
        length = next_fast_len(max(2, len(signal)))
        field = np.fft.fft(signal, n=length, norm="ortho")
        selection = np.linspace(0, length - 1, min(points, length)).astype(int)
        return {"kind": self.kind, "time_s": float(time_s), "mode_count": self.mode_count,
                "carrier_size": length, "energy": self.energy,
                "displacement": field.real[selection].tolist(), "quadrature": field.imag[selection].tolist()}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.kind.encode())
        digest.update(np.ascontiguousarray(self.indices, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(np.round(self.amplitudes, 12), dtype=np.float64).tobytes())
        return digest.hexdigest()[:16]


def build_field(kind: str, amplitudes: np.ndarray, frequencies: np.ndarray) -> WaveField:
    """Collect nonzero modes of a dense amplitude vector into one wave."""
    values = np.asarray(amplitudes, dtype=float)
    active = np.flatnonzero(values > 0.0)
    return WaveField(kind, active.astype(np.intp), values[active], frequencies[active])


def resonance(left: WaveField, right: WaveField) -> float:
    """Normalized mode overlap of two waves in [0, 1]; 1 means identical modes."""
    if not len(left.indices) or not len(right.indices):
        return 0.0
    shared = left.restrict(right.indices)
    numerator = float(shared @ right.amplitudes)
    denominator = math.sqrt(left.energy) * math.sqrt(right.energy)
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, numerator / denominator))


def superpose(fields, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Coherent sum of complex mode fields given as (indices, values) pairs."""
    total: dict[int, complex] = {}
    for position, (indices, values) in enumerate(fields):
        weight = 1.0 if weights is None else weights[position]
        if weight == 0.0:
            continue
        for index, value in zip(indices, values):
            total[int(index)] = total.get(int(index), 0j) + complex(weight) * complex(value)
    if not total:
        return np.array([], dtype=np.intp), np.array([], dtype=complex)
    indices = np.array(sorted(total), dtype=np.intp)
    return indices, np.array([total[int(i)] for i in indices], dtype=complex)


def interference_pair(indices: np.ndarray, amplitudes: np.ndarray, prompt: WaveField, *,
                      coupling: float = 1.0, phase_error: float = 0.0, mode_prior=None,
                      carrier_size: int | None = None):
    """One interference readout, computed by two independent algorithms.

    Returns ``(spectral_operator_readout, direct_readout, max_difference)``.

    The direct form multiplies in mode space::

        psi_k = a_k * (1 + coupling*e^{i phase_error} * p_k) * (1 + q_k)

    with ``p`` the prompt wave amplitude and ``q`` an optional real mode prior
    taken from the all-data wave. The spectral operator form never multiplies in
    mode space: it applies the mixing theorem
    ``spectral_convolution(F_u x, F_u y) = F_u(x*y)`` once per factor and sums the
    resulting components. Four terms appear, one per product of the two couplings.
    Both algorithms must agree to :data:`VERIFY_TOLERANCE`.
    """
    amplitude_field = np.asarray(amplitudes, dtype=complex)
    indices = np.asarray(indices)
    if amplitude_field.shape != indices.shape:
        raise ValueError("Field amplitudes do not match their modes")
    if coupling < 0 or not np.isfinite([coupling, phase_error]).all():
        raise ValueError("Invalid interference controls")
    guide = prompt.restrict(indices).astype(complex) * (coupling * np.exp(1j * phase_error))
    prior = np.zeros(len(indices), dtype=complex) if mode_prior is None \
        else np.asarray(mode_prior, dtype=complex)
    if prior.shape != indices.shape:
        raise ValueError("Mode prior does not match the interfering field")
    direct_readout = amplitude_field * (1.0 + guide) * (1.0 + prior)
    size = int(carrier_size) if carrier_size else next_fast_len(max(2, len(indices)))

    def pad(values):
        result = np.zeros(size, dtype=complex)
        result[:len(values)] = values
        return result

    data_spectrum = np.fft.fft(pad(amplitude_field), norm="ortho")
    prompt_spectrum = np.fft.fft(pad(guide), norm="ortho")
    prior_spectrum = np.fft.fft(pad(prior), norm="ortho")
    mixed = data_spectrum + spectral_convolution(data_spectrum, prompt_spectrum) \
        + spectral_convolution(data_spectrum, prior_spectrum) \
        + spectral_convolution(spectral_convolution(data_spectrum, prompt_spectrum), prior_spectrum)
    operator_readout = np.fft.ifft(mixed, norm="ortho")[:len(indices)]
    if not (np.isfinite(operator_readout).all() and np.isfinite(direct_readout).all()):
        raise ValueError("Interference has no supported finite energy")
    difference = float(np.max(np.abs(operator_readout - direct_readout))) if len(indices) else 0.0
    return operator_readout, direct_readout, difference


def verify_interference(indices, amplitudes, prompt: WaveField, *, coupling=1.0, phase_error=0.0):
    """Compute the same interference by both independent algorithms and compare."""
    operator_readout, direct_readout, error = interference_pair(
        indices, amplitudes, prompt, coupling=coupling, phase_error=phase_error)
    return {"states": int(len(direct_readout)), "max_error": error, "tolerance": VERIFY_TOLERANCE,
            "agrees": bool(error <= VERIFY_TOLERANCE),
            "operator_energy": float(np.vdot(operator_readout, operator_readout).real),
            "direct_energy": float(np.vdot(direct_readout, direct_readout).real)}


def probabilities(readout: np.ndarray) -> tuple[np.ndarray, float]:
    """Intensity measurement: |psi|^2 as exact probabilities plus total energy."""
    scores = np.abs(readout) ** 2
    total = float(scores.sum())
    if not np.isfinite(scores).all() or total <= 1e-24:
        raise ValueError("Interference has no supported finite energy")
    local = np.round(scores / total, 12)
    local = local / local.sum()
    return local, total
