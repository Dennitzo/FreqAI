"""Invertible UTF-8 <-> standing cosine modes with quadrature readout.

The mapping is an engineered code, not an intrinsic physical meaning of text.
All coefficients are retained. No compression or infinite capacity is claimed.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.fft import dct, idct


@dataclass(frozen=True)
class WavePacket:
    coefficients: np.ndarray
    byte_length: int
    sha256: str


def encode_text(text: str) -> WavePacket:
    payload = text.encode("utf-8")
    samples = np.frombuffer(payload or b"\x00", dtype=np.uint8).astype(np.float64)
    coefficients = dct((samples - 127.5) / 127.5, type=2, norm="ortho")
    return WavePacket(coefficients, len(payload), hashlib.sha256(payload).hexdigest())


def phase_angles(size: int, time_s: float) -> np.ndarray:
    """Neumann standing modes: omega_k = 2*pi*k*(30/N) rad/s, DC at k=0.

    The 30 Hz ceiling is an arbitrary simulation unit. There is no claim that a
    word naturally has one of these frequencies. Long-time argument reduction
    prevents unbounded phase growth (within floating-point time resolution).
    """
    if size < 1 or not np.isfinite(time_s):
        raise ValueError("Mode count must be positive and time must be finite")
    fundamental_hz = 30.0 / size
    cycles = np.remainder(time_s * fundamental_hz, 1.0)
    return 2.0 * np.pi * np.remainder(np.arange(size) * cycles, 1.0)


def modal_state(packet: WavePacket, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    theta = phase_angles(len(packet.coefficients), time_s)
    return packet.coefficients * np.cos(theta), packet.coefficients * np.sin(theta)


def recover_coefficients(q: np.ndarray, p: np.ndarray, time_s: float) -> np.ndarray:
    q, p = np.asarray(q, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if q.ndim != 1 or q.shape != p.shape:
        raise ValueError("Quadratures must be equally sized one-dimensional arrays")
    theta = phase_angles(len(q), time_s)
    return q * np.cos(theta) + p * np.sin(theta)


def decode_coefficients(coefficients: np.ndarray, byte_length: int, sha256: str) -> str:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or len(coefficients) != max(1, byte_length):
        raise ValueError("Invalid wave size or byte length")
    if byte_length < 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Wave contains invalid or nonfinite values")
    samples = idct(coefficients, type=2, norm="ortho") * 127.5 + 127.5
    rounded = np.rint(samples)
    if np.any(rounded < 0) or np.any(rounded > 255):
        raise ValueError("Wave corruption: decoded sample outside byte range")
    payload = rounded.astype(np.uint8).tobytes()[:byte_length]
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Wave corruption: SHA-256 integrity check failed")
    return payload.decode("utf-8", errors="strict")


def decode_text(packet: WavePacket, time_s: float = 0.0) -> str:
    q, p = modal_state(packet, time_s)
    return decode_coefficients(recover_coefficients(q, p, time_s), packet.byte_length, packet.sha256)


def sample_wave(packet: WavePacket, time_s: float, points: int = 256) -> dict:
    if points < 2:
        raise ValueError("At least two display points required")
    q, p = modal_state(packet, time_s)
    # Compute the entire spatial grid before display decimation: no stored mode
    # is omitted from the dynamics. The plotted grid itself is not a full codec.
    displacement = idct(q, type=2, norm="ortho")
    quadrature = idct(p, type=2, norm="ortho")
    indices = np.linspace(0, len(q) - 1, min(points, len(q))).astype(int)
    omega = 2 * np.pi * np.arange(len(q)) * (30.0 / len(q))
    return {
        "time_s": float(time_s),
        "displacement": displacement[indices].tolist(),
        "quadrature": quadrature[indices].tolist(),
        "energy": float(np.dot(q, q) + np.dot(p, p)),
        "physical_energy": float(0.5 * np.sum(omega**2 * (q**2 + p**2))),
        "modal_count": len(q),
        "display_points": len(indices),
    }

