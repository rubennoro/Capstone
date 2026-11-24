"""The console's pure helpers (control_deploy/STRUCTURE.md 6.5)."""
from __future__ import annotations

import math

import numpy as np
from helpers import pose

from control_deploy.console import waypoints as w


def test_the_time_grid_is_strictly_increasing_and_covers_the_path():
    g = w.time_grid(1000, 2.5)
    assert all(b > a for a, b in zip(g, g[1:]))
    assert g[0] == 1000 and g[-1] >= 1000 + 2.5e9 and all(b - a == w.SPACING_NS for a, b in zip(g, g[1:]))


def test_requests_hold_at_most_8_and_lead_at_most_0_6_s():
    p = w.Path(10_000_000_000 + w.LEAD_NS, 2.5, lambda t: (np.eye(4), 0.0))
    sent = set()
    now = 10_000_000_000
    while True:
        times = w.next_request(p, now)
        if not times:
            break
        assert 1 <= len(times) <= w.PER_REQUEST
        assert times[0] - now >= w.LEAD_NS
        assert times[-1] - now <= 600_000_000
        sent.update(times)
        now += w.REQUEST_PERIOD_NS
    assert sent == set(w.time_grid(p.t0_ns, 2.5))   # every grid point, the end included


def test_min_jerk_endpoints_and_peak_speed():
    assert w.min_jerk(0.0) == 0.0 and w.min_jerk(1.0) == 1.0 and w.min_jerk(0.5) == 0.5
    assert w.min_jerk(-1.0) == 0.0 and w.min_jerk(2.0) == 1.0
    h = 1e-6
    peak = max((w.min_jerk(t + h) - w.min_jerk(t - h)) / (2 * h) for t in np.linspace(0.01, 0.99, 99))
    assert abs(peak - 1.875) < 1e-3
    assert abs(w.min_jerk_peak_speed(0.015, 1.0) - 0.028125) < 1e-12   # far below the gate's 0.25 m/s
    total, s = w.out_hold_back(1.0, 0.5)
    assert total == 2.5 and s(0.0) == 0.0 and s(1.0) == 1.0 and s(1.25) == 1.0 and abs(s(2.5)) < 1e-12


def test_translation_along_base_z():
    p = pose(0.45, 0.0, 0.45)
    q = w.translate(p, np.array([0.0, 0.0, 0.015]))
    assert np.allclose(q[:3, 3], [0.45, 0.0, 0.465]) and np.array_equal(q[:3, :3], p[:3, :3])


def test_rotation_about_tcp_z_keeps_the_z_axis_and_position():
    p = pose(0.45, 0.1, 0.45, yaw=0.3)
    q = w.rotate_about_tcp_z(p, math.radians(5))
    assert np.allclose(q[:3, 2], p[:3, 2])            # the z axis (the flange's too, F_T_TCP rotates about z only)
    assert np.allclose(q[:3, 3], p[:3, 3])
    assert abs(w.rotation_angle(p, q) - math.radians(5)) < 1e-12


def test_the_plan_model_interpolates_as_the_pc():
    a = pose(0.45, 0.0, 0.45)
    m = w.PlanModel(0, a, 0.08)
    b = w.translate(w.rotate_about_tcp_z(a, 0.2), np.array([0.1, 0.0, 0.0]))
    m.apply(1000, [w.PlanPoint(2000, b, 0.06)])
    p, width = m.at(1500)
    assert np.allclose(p[:3, 3], [0.5, 0.0, 0.45]) and abs(width - 0.07) < 1e-12
    assert abs(w.rotation_angle(a, p) - 0.1) < 1e-9
    assert np.allclose(m.at(500)[0], a) and np.allclose(m.at(5000)[0], b)
    q = w.quat_from_matrix(b[:3, :3])
    assert np.allclose(w.matrix_from_quat(q), b[:3, :3])


def test_chunks():
    assert w.chunks(list(range(17))) == [list(range(8)), list(range(8, 16)), [16]]
