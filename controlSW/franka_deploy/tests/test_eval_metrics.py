import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from franka_deploy.eval.metrics import (chunk_consistency, dtw, ood_fraction, phases, relative_positions, wilson)


def test_wilson_reference_values():
    # reference: statsmodels proportion_confint(method="wilson")
    p, lo, hi = wilson(7, 10)
    assert p == 0.7 and abs(lo - 0.39677) < 1e-4 and abs(hi - 0.89220) < 1e-4
    p, lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.27753) < 1e-4
    p, lo, hi = wilson(10, 10)
    assert abs(hi - 1.0) < 1e-12 and abs(lo - 0.72247) < 1e-4


def test_dtw_identical_zero_and_time_warp_invariant():
    t = np.linspace(0, 1, 50)
    a = np.stack([np.sin(3 * t), np.cos(2 * t), t], 1)
    assert dtw(a, a) == 0.0
    slow = np.stack([np.sin(3 * t ** 1.5), np.cos(2 * t ** 1.5), t ** 1.5], 1)   # same path, different timing
    assert dtw(a, slow) < 0.02
    line = np.stack([t, 0 * t, 0 * t], 1)
    assert dtw(line, line + [0, 0.1, 0]) == pytest.approx(0.1, abs=1e-9)   # perpendicular offset: cost = offset


def test_relative_positions_frame_invariant():
    rng = np.random.default_rng(0)
    p = np.cumsum(rng.normal(0, 0.01, (30, 3)), 0)
    r0 = R.from_euler("xyz", [0.3, -0.2, 1.0])
    rel = relative_positions(p, r0)
    G = R.from_euler("z", 2.0)                        # move the whole world: rel must not change
    rel2 = relative_positions(G.apply(p) + [1, 2, 3], G * r0)
    np.testing.assert_allclose(rel, rel2, atol=1e-12)


def test_phases_on_synthetic_grasp_lift():
    t = np.arange(0, 10, 0.05)
    w = np.where(t < 3, 0.083, np.where(t < 4, 0.083 - (t - 3) * 0.033, 0.050))
    w = np.where(t > 8, 0.08, w)
    h = np.where(t < 4.5, 0.0, np.minimum((t - 4.5) * 0.05, 0.10))
    ph = phases(w, h, t)
    assert abs(ph["t_close_start"] - 3.25) < 0.1 and abs(ph["t_grasp"] - 3.95) < 0.1
    assert ph["lifted"] and abs(ph["lift_height_m"] - 0.10) < 1e-6 and abs(ph["grasp_width_m"] - 0.050) < 1e-9
    assert phases(np.full(100, 0.08), np.zeros(100), np.arange(100) * 0.05) is None


def test_chunk_consistency_and_ood():
    ts = np.arange(16) * 0.05
    a = np.stack([ts, 0 * ts, 0 * ts], 1)
    assert chunk_consistency(a, ts, a[8:], ts[8:]) == pytest.approx(0.0)
    assert chunk_consistency(a, ts, a[8:] + [0.01, 0, 0], ts[8:]) == pytest.approx(0.01)
    frac, exc = ood_fraction(np.array([[0.5], [1.5], [-0.5]]), np.array([0.0]), np.array([1.0]))
    assert frac == pytest.approx(2 / 3) and exc == pytest.approx(0.5)
