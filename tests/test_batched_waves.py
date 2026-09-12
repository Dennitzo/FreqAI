"""Batching must retain the exact addressed waves, including after append."""
import numpy as np
import pytest
from scipy.fft import idct

from freqai.codec import phase_angles
from freqai.memory import Document, WaveMemory


@pytest.mark.parametrize("time_s", [0.0, 13.75, 86400.125, 1e9])
def test_batch_snapshot_matches_individual_packet_dynamics(time_s):
    memory = WaveMemory([Document(str(i), text, "Prüfung", "Ein Anlass") for i, text in enumerate(
        ["Rote Katzen.", "Blaue Hunde.", "", "Grüße!", "Ein deutlich längerer Satz."]*5)], dimensions=32)
    memory = memory.with_documents_added([Document("addition", "Blaue Hunde.", "Live")])
    q, p, spatial_q, spatial_p = [], [], [], []
    for packet in memory._packets:
        angles = phase_angles(len(packet.coefficients), time_s)
        packet_q, packet_p = packet.coefficients*np.cos(angles), packet.coefficients*np.sin(angles)
        q.extend(packet_q); p.extend(packet_p)
        spatial_q.extend(idct(packet_q, type=2, norm="ortho"))
        spatial_p.extend(idct(packet_p, type=2, norm="ortho"))
    snapshot = memory.snapshot(time_s, points=len(q))
    np.testing.assert_array_equal(snapshot["displacement"], spatial_q)
    np.testing.assert_array_equal(snapshot["quadrature"], spatial_p)
    norm = np.asarray(q)**2+np.asarray(p)**2
    assert snapshot["energy"] == float(np.sum(norm))
    assert snapshot["physical_energy"] == float(.5*np.sum((2*np.pi*30*memory._relative_frequencies)**2*norm))


def test_empty_batch_has_no_modes_or_energy():
    result = WaveMemory([], dimensions=32).snapshot(7.5)
    assert result["modal_count"] == 0 and result["energy"] == 0
    assert result["displacement"] == result["quadrature"] == []
