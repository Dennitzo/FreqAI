"""Ordered composition of UTF-8 standing waves, without encoding a finished reply.

If C_n is the orthonormal cosine transform and E_j embeds a source span into
its output byte positions, a_out = C_L sum_j E_j C_nj.T a_j.  Position channels
are essential: adding unrelated spectra without them does not concatenate text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.fft import dct, idct

from .codec import (WavePacket, decode_coefficients, decode_text, modal_state,
                    recover_coefficients)


@dataclass(frozen=True)
class WaveFragment:
    packet: WavePacket
    source: str = ""
    start: int = 0
    stop: int | None = None


@dataclass(frozen=True)
class Synthesis:
    text: str
    packet: WavePacket
    diagnostics: dict


def compose(fragments: list[WaveFragment], time_s: float = 0.0) -> Synthesis:
    """Read verified source waves, apply ordered projections and decode output.

    Spans use UTF-8 byte offsets, which must be valid character boundaries.
    The independent checksum is formed from verified source bytes. Output
    samples come from the inverse source transforms, not from that checksum.
    """
    if not fragments:
        raise ValueError("At least one wave fragment is required")
    sample_parts, payload_parts, sources = [], [], []
    for fragment in fragments:
        packet = fragment.packet
        q, p = modal_state(packet, time_s)
        coefficients = recover_coefficients(q, p, time_s)
        verified = decode_coefficients(coefficients, packet.byte_length, packet.sha256).encode("utf-8")
        stop = packet.byte_length if fragment.stop is None else fragment.stop
        if not (isinstance(fragment.start, int) and isinstance(stop, int)
                and 0 <= fragment.start <= stop <= packet.byte_length):
            raise ValueError("Invalid source byte span")
        payload = verified[fragment.start:stop]
        payload.decode("utf-8", errors="strict")
        sample_parts.append(idct(coefficients, type=2, norm="ortho")[fragment.start:stop])
        payload_parts.append(payload)
        sources.append({"source": fragment.source, "start": fragment.start, "stop": stop,
                        "source_sha256": packet.sha256})
    # Concatenation is the efficient implementation of disjoint embeddings E_j.
    samples = np.concatenate(sample_parts)
    expected = b"".join(payload_parts)
    if not len(samples):
        samples = np.array([-1.0])  # codec's empty-string sentinel
    coefficients = dct(samples, type=2, norm="ortho")
    coefficients.flags.writeable = False
    packet = WavePacket(coefficients, len(expected), hashlib.sha256(expected).hexdigest())
    output = decode_text(packet, time_s)
    reconstructed = idct(coefficients, type=2, norm="ortho") * 127.5 + 127.5
    error = float(np.max(np.abs(reconstructed - np.rint(reconstructed))))
    return Synthesis(output, packet, {
        "method": "linear_spectral_composition", "fragments": len(fragments),
        "mode_count": len(coefficients), "source_integrity_checked": True,
        "output_sha256": packet.sha256, "max_byte_roundoff": error,
        "sources": sources, "time_s": float(time_s),
    })
