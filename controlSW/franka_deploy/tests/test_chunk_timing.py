import numpy as np
import pytest

from franka_deploy.tasks.chunk_timing import SettleTimeoutError, chunk_scale, scale_timestamps, settle_gate


def chunk(dx_per_step, n=8, drot=0.0):
    from scipy.spatial.transform import Rotation as R

    out = []
    for k in range(n):
        q = R.from_euler("z", drot * k).as_quat()
        out.append([0.5 + dx_per_step * k, 0, 0.3, q[3], q[0], q[1], q[2]])
    return np.array(out)


def test_scale_slow_chunk_untouched():
    s, vp, wp = chunk_scale(chunk(0.005), None, 0.05, v_max=0.25, w_max=1.0)  # 0.1 m/s
    assert s == 1.0 and abs(vp - 0.1) < 1e-9


def test_scale_fast_chunk_to_margin():
    s, vp, _ = chunk_scale(chunk(0.05), None, 0.05, v_max=0.25, w_max=1.0)   # 1 m/s
    assert abs(vp - 1.0) < 1e-9 and abs(s - 1.0 / (0.9 * 0.25)) < 1e-9
    ts = scale_timestamps(np.arange(8) * 0.05 + 10.0, s)
    peak = np.max(np.abs(np.diff(chunk(0.05)[:, 0])) / np.diff(ts))
    assert peak <= 0.9 * 0.25 + 1e-9 and ts[0] == 10.0


def test_scale_rotation_and_anchor_segment():
    s, _, wp = chunk_scale(chunk(0.0, drot=0.1), None, 0.05, v_max=0.25, w_max=1.0)  # 2 rad/s
    assert abs(wp - 2.0) < 1e-6 and abs(s - 2.0 / 0.9) < 1e-6
    anchor = np.array([0.40, 0, 0.3, 1, 0, 0, 0.0])                                    # 10 cm before action 0
    s2, vp2, _ = chunk_scale(chunk(0.0), anchor, 0.05, v_max=0.25, w_max=1.0)
    assert abs(vp2 - 2.0) < 1e-9 and s2 > 8


def test_settle_gate_waits_then_passes():
    seq = iter([(0.01, 0.02), (0.005, 0.01), (0.001, 0.002)] + [(0.001, 0.002)] * 10)
    t = [0.0]
    st = settle_gate(lambda: next(seq), sleep=lambda d: t.__setitem__(0, t[0] + d), clock=lambda: t[0])
    assert st["track_dev_m"] == 0.002 and abs(st["settle_s"] - 0.02) < 1e-9


def test_settle_gate_accepts_stationary_bias():
    t = [0.0]
    st = settle_gate(lambda: (0.0005, 0.005), sleep=lambda d: t.__setitem__(0, t[0] + d), clock=lambda: t[0])
    assert st["stationary"] and st["settle_s"] < 0.1


def test_settle_gate_timeout():
    t = [0.0]
    vals = iter(np.linspace(0.05, 0.02, 150))  # still moving (0.2 mm per 10 ms poll), never < 3 mm
    with pytest.raises(SettleTimeoutError):
        settle_gate(lambda: (None, float(next(vals))), sleep=lambda d: t.__setitem__(0, t[0] + d), clock=lambda: t[0])
    t2 = [0.0]
    with pytest.raises(SettleTimeoutError):  # backlog never drains
        settle_gate(lambda: (0.01, 0.001), sleep=lambda d: t2.__setitem__(0, t2[0] + d), clock=lambda: t2[0])
