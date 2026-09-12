"""Shared numerical operations without paired-corpus or conversation logic."""
import hashlib
import math

import numpy as np

BOS = "<BOS>"
EOS = "<EOS>"


def stable_frequency(symbol: str) -> float:
    """Deterministic symbol frequency, unchanged by vocabulary extension."""
    number = int.from_bytes(hashlib.blake2b(symbol.encode(), digest_size=8).digest(), "little")
    return 0.25 + 29.75 * (number / float(2**64))


def spectral_convolution(left, right):
    """Return F(F^-1(left) * F^-1(right)) for orthonormal spectra."""
    return np.fft.ifft(np.fft.fft(left)*np.fft.fft(right))/math.sqrt(len(left))
