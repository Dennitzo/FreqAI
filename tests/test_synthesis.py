import hashlib

import numpy as np
import pytest
from scipy.fft import dct, idct

from freqai.codec import WavePacket, decode_text, encode_text
from freqai.synthesis import WaveFragment, compose


@pytest.mark.parametrize("time_s", [0, 0.123, 1.7, 3600, 1e8])
def test_unseen_ordered_composition_from_source_waves(time_s):
    sources = [encode_text(text) for text in ["Grüße, ", "世界", "! 🌊"]]
    result = compose([WaveFragment(packet, str(i)) for i, packet in enumerate(sources)], time_s)
    assert result.text == "Grüße, 世界! 🌊"
    assert result.packet.sha256 == hashlib.sha256(result.text.encode()).hexdigest()
    assert result.diagnostics["max_byte_roundoff"] < 1e-9
    assert decode_text(result.packet, 97.2) == result.text
    assert result.diagnostics["source_integrity_checked"]


def test_matches_explicit_linear_embedding_operator():
    packets = [encode_text("ab"), encode_text("XYZ")]
    pieces = [WaveFragment(p) for p in packets]
    result = compose(pieces)
    output_transform = dct(np.eye(5), axis=0, norm="ortho")
    expected = np.zeros(5)
    offset = 0
    for packet in packets:
        size = packet.byte_length
        embedding = np.eye(5)[:, offset:offset + size]
        inverse = idct(np.eye(size), axis=0, norm="ortho")
        expected += output_transform @ embedding @ inverse @ packet.coefficients
        offset += size
    np.testing.assert_allclose(result.packet.coefficients, expected, atol=1e-14)
    reverse = compose(pieces[::-1])
    assert reverse.text == "XYZab"
    assert reverse.packet.sha256 != result.packet.sha256


def test_extract_sentence_from_existing_source_coefficients():
    packet = encode_text("Äpfel. Zweiter Satz.")
    result = compose([WaveFragment(packet, "source", 0, len("Äpfel.".encode())),
                      WaveFragment(encode_text(" Gut!"))], 29.2)
    assert result.text == "Äpfel. Gut!"


def test_reject_corrupt_sources_before_composition():
    good = encode_text("Hallo")
    corrupted = WavePacket(good.coefficients + 0.1, good.byte_length, good.sha256)
    with pytest.raises(ValueError, match="corruption"):
        compose([WaveFragment(corrupted)])


def test_reject_partial_utf8_character():
    with pytest.raises(UnicodeDecodeError):
        compose([WaveFragment(encode_text("ü"), start=1)])


def test_empty_content_and_invalid_span():
    assert compose([WaveFragment(encode_text(""))]).text == ""
    with pytest.raises(ValueError, match="span"):
        compose([WaveFragment(encode_text("a"), stop=2)])
    with pytest.raises(ValueError):
        compose([])
