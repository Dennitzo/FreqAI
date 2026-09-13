import numpy as np
import pytest
from scipy.fft import idct
from freqai.codec import phase_angles
from freqai.projected_wave import ProjectedWave
from freqai import compute


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_projected_coordinates_match_full_idct(backend, monkeypatch):
    monkeypatch.setenv('FREQAI_COMPUTE', backend)
    if backend == 'cuda' and not compute.cuda_for(1, backend='cuda'):
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(21)
    first, second = rng.normal(size=1531), rng.normal(size=2123)
    a, b = np.array([0, 999, 1530]), np.array([71, 2000])
    projected = ProjectedWave([(first, a, [0, 1, 2]), (second, b, [3, 4])], 5)
    for time_s in (0, .17, 12345.6, 1e9):
        q, p = projected.snapshot(time_s)
        expected = []
        for coefficients, positions in ((first, a), (second, b)):
            phase = phase_angles(len(coefficients), time_s)
            expected.append((idct(coefficients*np.cos(phase), norm='ortho')[positions],
                             idct(coefficients*np.sin(phase), norm='ortho')[positions]))
        np.testing.assert_allclose(q, np.concatenate([x[0] for x in expected]), atol=2e-10)
        np.testing.assert_allclose(p, np.concatenate([x[1] for x in expected]), atol=2e-10)
    if backend == 'cuda':
        assert compute.compute_status()['last_backend_by_operation']['wave_snapshot'] == 'cuda'
