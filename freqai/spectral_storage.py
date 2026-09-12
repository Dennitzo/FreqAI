"""Lossless, support-local storage for sparse complex token wave fields.

For token indices I, a field stores C = F_|I|(a[I]), not real transition
probabilities. Its global representation is F_V(E_I(F_|I|^-1(C))). This is a
change of coordinates, without coefficient pruning, quantization or a learned
projection. Runtime interference still consumes the stored complex values.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib

import numpy as np


class CompactSpectrum(np.lib.mixins.NDArrayOperatorsMixin):
    """A global spectrum represented by local complex coefficients and indices.

    NumPy conversion and arithmetic expose the original global Fourier array,
    preserving inspection and perturbation experiments. The generator uses
    ``add_modes_to`` instead, so it never materializes V values per stored field.
    Full-array zeroing stays compact. Arbitrary global edits are retained as a
    dense override, since they can create amplitudes outside the original
    support; those edits must not silently be projected away.
    """

    __array_priority__ = 1000

    def __init__(self, size, token_indices, local_spectrum):
        self.size = int(size)
        self.token_indices = np.asarray(token_indices, dtype=np.intp)
        self.local_spectrum = np.asarray(local_spectrum, dtype=np.complex128)
        self._dense_override = None

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return 1

    @property
    def dtype(self):
        return np.dtype(np.complex128)

    @property
    def nbytes(self):
        """Allocated coefficients; the shared support indices are counted separately."""
        return self.local_spectrum.nbytes + (0 if self._dense_override is None else self._dense_override.nbytes)

    def __len__(self):
        return self.size

    def add_modes_to(self, target, weight=1.0):
        if self._dense_override is not None:
            target += weight * np.fft.ifft(self._dense_override, norm="ortho")
        elif len(self.local_spectrum):
            target[self.token_indices] += weight * np.fft.ifft(self.local_spectrum, norm="ortho")

    def __array__(self, dtype=None, copy=None):
        if self._dense_override is None:
            modes = np.zeros(self.size, dtype=np.complex128)
            self.add_modes_to(modes)
            result = np.fft.fft(modes, norm="ortho")
        else:
            result = self._dense_override.copy() if copy else self._dense_override
        return result.astype(dtype, copy=False) if dtype is not None else result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        arrays = [np.asarray(value) if isinstance(value, CompactSpectrum) else value for value in inputs]
        outputs = kwargs.get("out")
        if outputs:
            kwargs["out"] = tuple(np.asarray(value) if isinstance(value, CompactSpectrum) else value for value in outputs)
        result = getattr(ufunc, method)(*arrays, **kwargs)
        if outputs:
            for original, array in zip(outputs, kwargs["out"]):
                if isinstance(original, CompactSpectrum):
                    original._dense_override = np.asarray(array, dtype=np.complex128).copy()
        return result

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice) and key == slice(None) and np.isscalar(value) and value == 0:
            self.local_spectrum[:] = 0
            self._dense_override = None
            return
        array = np.asarray(self).copy()
        array[key] = value
        self._dense_override = array

    def copy(self):
        result = CompactSpectrum(self.size, self.token_indices, self.local_spectrum.copy())
        if self._dense_override is not None:
            result._dense_override = self._dense_override.copy()
        return result


class PackedTransitionSpectra(Mapping):
    """One contiguous coefficient/index store with lazy, mutable field views.

    Compilation and cache loading allocate a handful of numerical buffers,
    rather than two arrays and a Python object for every observed prefix.
    Accessed fields retain their identity so perturbation experiments, including
    dense overrides and replacement spectra, behave like a dictionary of fields.
    """

    def __init__(self, size, field_keys, offsets, token_indices, coefficients, *, mode_energy=None):
        self.size = int(size)
        self.field_keys = list(field_keys)
        self.offsets = np.asarray(offsets, dtype=np.intp)
        self.token_indices = np.asarray(token_indices, dtype=np.intp)
        self.coefficients = np.asarray(coefficients, dtype=np.complex128)
        self.mode_energy = None if mode_energy is None else np.asarray(mode_energy, dtype=float)
        if self.offsets.shape != (len(self.field_keys) + 1,) or self.offsets[0] != 0 \
                or self.offsets[-1] != len(self.token_indices) \
                or self.coefficients.shape != self.token_indices.shape \
                or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Invalid packed transition field boundaries")
        if self.mode_energy is not None and self.mode_energy.shape != self.coefficients.shape:
            raise ValueError("Invalid packed compilation-energy shape")
        self._positions = {key: index for index, key in enumerate(self.field_keys)}
        if len(self._positions) != len(self.field_keys):
            raise ValueError("Duplicate packed transition field key")
        self._cache = {}
        self._original_buffer_digest = self.integrity_digest()

    def integrity_digest(self):
        digest = hashlib.sha256()
        for value in (self.offsets, self.token_indices, self.coefficients, self.mode_energy):
            if value is None:
                digest.update(b"none")
                continue
            digest.update(value.nbytes.to_bytes(8, "little"))
            digest.update(memoryview(value).cast("B") if value.flags.c_contiguous else value.tobytes())
        return digest.hexdigest()

    def is_pristine(self):
        """Only unmodified compiled values may seed another compiler or cache."""
        for field in self._cache.values():
            if field._dense_override is not None \
                    or not np.shares_memory(field.local_spectrum, self.coefficients) \
                    or not np.shares_memory(field.token_indices, self.token_indices):
                return False
        return self.integrity_digest() == self._original_buffer_digest

    def __getitem__(self, key):
        position = self._positions[key]
        field = self._cache.get(position)
        if field is None:
            start, stop = self.offsets[position:position + 2]
            field = CompactSpectrum(self.size, self.token_indices[start:stop], self.coefficients[start:stop])
            self._cache[position] = field
        return field

    def __contains__(self, key):
        return key in self._positions

    def __iter__(self):
        return iter(self.field_keys)

    def __len__(self):
        return len(self.field_keys)

    def resize(self, size):
        self.size = int(size)
        for field in self._cache.values():
            field.size = self.size

    def raw_items(self):
        """Yield key, support and complex coordinates without materializing fields."""
        for position, key in enumerate(self.field_keys):
            cached = self._cache.get(position)
            if cached is not None:
                yield key, cached.token_indices, cached.local_spectrum
            else:
                start, stop = self.offsets[position:position + 2]
                yield key, self.token_indices[start:stop], self.coefficients[start:stop]

    def raw_field(self, key):
        position = self._positions[key]
        cached = self._cache.get(position)
        if cached is not None:
            return cached.token_indices, cached.local_spectrum
        start, stop = self.offsets[position:position + 2]
        return self.token_indices[start:stop], self.coefficients[start:stop]

    @property
    def coefficient_nbytes(self):
        # Most runs never create an override; diagnostic edits still count.
        extra = sum(field.nbytes - (self.offsets[position + 1] - self.offsets[position]) * 16
                    for position, field in self._cache.items())
        return int(self.coefficients.nbytes + extra)


class PackedSupportMapping(Mapping):
    """Support views backed by the same packed table, with no second key dictionary."""

    def __init__(self, spectra):
        self.spectra = spectra

    def __getitem__(self, key):
        position = self.spectra._positions[key]
        start, stop = self.spectra.offsets[position:position + 2]
        return self.spectra.token_indices[start:stop]

    def __contains__(self, key):
        return key in self.spectra

    def __iter__(self):
        return iter(self.spectra)

    def __len__(self):
        return len(self.spectra)


class SparseDistributionReference(Mapping):
    """Read-only real reference probabilities, expanded only when requested.

    The independent values are kept for diagnostics and the alternate HRR
    benchmark. Production token generation never reads this mapping.
    """

    def __init__(self, size, rows):
        self.size = size
        self.rows = rows

    def __getitem__(self, key):
        indices, probabilities = self.rows[key]
        result = np.zeros(self.size)
        result[indices] = probabilities
        return result

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    @property
    def nbytes(self):
        return sum(probabilities.nbytes for _, probabilities in self.rows.values())


def add_spectral_modes(target, spectrum, weight=1.0):
    """Decode the actual imported complex field into a shared token carrier."""
    if isinstance(spectrum, CompactSpectrum):
        spectrum.add_modes_to(target, weight)
    else:
        target += weight * np.fft.ifft(spectrum, norm="ortho")
