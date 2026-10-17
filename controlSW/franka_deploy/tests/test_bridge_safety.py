"""Bridge safety rules, one test each, plus the gripper driver and the TCP composition (pure, no processes)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from franka_deploy.bridge.safety import SafetyConfig, SafetyFilter
from franka_deploy.gripper.driver import FakeGripper, WidthCalibration


def T(p=(0.5, 0.0, 0.3), rotvec=(np.pi, 0, 0)):
    M = np.eye(4)
    M[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    M[:3, 3] = p
    return M


def filt(**kw):
    return SafetyFilter(SafetyConfig(**kw))


def test_nonfinite_rejected_and_last_kept():
    f = filt()
    f.reset(T(), 0.0)
    bad = T()
    bad[0, 3] = np.nan
    out, flags = f.apply(bad, 0.01)
    assert out is None and flags == {"nan": True}
    np.testing.assert_allclose(f.last, T())
    inf = T()
    inf[1, 1] = np.inf
    assert f.apply(inf, 0.02)[0] is None


def test_workspace_box_clamp():
    f = filt(v_max=100, w_max=100)
    out, flags = f.apply(T(p=(2.0, -1.0, -0.5)), 0.0)
    np.testing.assert_allclose(out[:3, 3], [0.80, -0.40, 0.05])
    assert flags["box"]


def test_orientation_cone_clamp():
    f = filt(v_max=100, w_max=100, cone_max_deg=30)
    # EE z pointing horizontally (+x): 90 deg from the down axis -> rotated back to 30 deg
    out, flags = f.apply(T(rotvec=(0, np.pi / 2, 0)), 0.0)
    z = out[:3, 2]
    ang = np.degrees(np.arccos(np.dot(z, [0, 0, -1])))
    assert flags["cone"] and abs(ang - 30) < 1e-6
    assert np.allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-9)


def test_step_clamp_position_is_clamp_not_reject():
    f = filt(v_max=0.25, slack_s=0.01)
    f.reset(T(p=(0.5, 0, 0.3)), 0.0)
    out, flags = f.apply(T(p=(0.6, 0, 0.3)), 0.005)  # 10 cm at once: bucket = full slack 2.5 mm
    assert flags["step_pos"]
    np.testing.assert_allclose(out[:3, 3], [0.5 + 0.25 * 0.01, 0, 0.3])
    t = 0.005
    for _ in range(200):                              # repeated updates converge (executed, not frozen)
        t += 0.005
        out, _ = f.apply(T(p=(0.6, 0, 0.3)), t)
    np.testing.assert_allclose(out[:3, 3], [0.6, 0, 0.3], atol=1e-9)


def test_step_clamp_rotation():
    f = filt(w_max=1.0, slack_s=0.01)
    f.reset(T(), 0.0)
    out, flags = f.apply(T(rotvec=(np.pi, 0, 0.5)), 0.001)
    d = (R.from_matrix(T()[:3, :3]).inv() * R.from_matrix(out[:3, :3])).magnitude()
    assert flags["step_rot"] and abs(d - 0.01) < 1e-6   # slack 10 ms * 1 rad/s


def test_long_gap_gives_only_the_slack():
    f = filt(v_max=0.25, slack_s=0.01)
    f.reset(T(p=(0.5, 0, 0.3)), 0.0)
    out, _ = f.apply(T(p=(0.7, 0, 0.3)), 10.0)
    assert abs(out[0, 3] - (0.5 + 0.25 * 0.01)) < 1e-9


def test_order_box_before_step():
    f = filt(v_max=0.25)
    f.reset(T(p=(0.799, 0, 0.3)), 0.0)
    out, flags = f.apply(T(p=(5.0, 0, 0.3)), 0.005)
    assert flags["box"] and not flags.get("step_pos")  # boxed first: 0.799 -> 0.80 is within the slack
    assert abs(out[0, 3] - 0.80) < 1e-12


def _max_window_motion(times, xs, window=0.1):
    times, xs = np.asarray(times), np.asarray(xs)
    best = 0.0
    for i in range(len(times)):
        j = np.searchsorted(times, times[i] + window, side="right") - 1
        best = max(best, abs(xs[j] - xs[i]))
    return best


@pytest.mark.parametrize("pattern", ["200hz", "1khz", "bursts"])
def test_speed_bound_independent_of_request_rate(pattern):
    """Commanded 2 m/s for 1 s: whatever the request pattern, any 100 ms window moves <= v_max * 0.1 * 1.1."""
    v_max = 0.25
    f = filt(v_max=v_max, slack_s=0.01, workspace_max=(5, 5, 5))
    f.reset(T(p=(0.5, 0, 0.3)), 0.0)
    if pattern == "200hz":
        times = np.arange(1, 201) * 0.005
    elif pattern == "1khz":
        times = np.arange(1, 1001) * 0.001
    else:  # 10 requests within 0.2 ms, every 20 ms
        times = np.concatenate([k * 0.02 + np.arange(10) * 2e-5 for k in range(1, 51)])
    xs = [0.5]
    ts = [0.0]
    for t in times:
        out, _ = f.apply(T(p=(0.5 + 2.0 * t, 0, 0.3)), t)
        xs.append(out[0, 3])
        ts.append(t)
    m = _max_window_motion(ts, xs)
    assert m <= v_max * 0.1 * 1.1 + 1e-9, (pattern, m)
    if pattern != "bursts":                # (requests 20 ms apart only get the 10 ms slack each: slower, never faster)
        assert m >= v_max * 0.1 * 0.9      # and the arm does move at ~v_max


def test_reply_jitter_does_not_clamp():
    """200 Hz controller commanding 0.9 v_max; requests arrive with +-2 ms jitter -> never clamped."""
    rng = np.random.default_rng(0)
    v = 0.9 * 0.25
    f = filt(v_max=0.25, slack_s=0.01, workspace_max=(5, 5, 5))
    f.reset(T(p=(0.5, 0, 0.3)), 0.0)
    for k in range(1, 400):
        t_cmd = k * 0.005
        t_arr = t_cmd + rng.uniform(-0.002, 0.002)
        out, flags = f.apply(T(p=(0.5 + v * t_cmd, 0, 0.3)), t_arr)
        assert not flags.get("step_pos"), k


# ------------------------------------------------------------------ gripper
CAL = WidthCalibration(np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.11]),
                       np.array([3000, 2700, 2400, 2100, 1800, 1500, 1350.0]))


def test_width_map_round_trip_below_0p1mm():
    for w in np.linspace(0, 0.11, 111):
        assert abs(CAL.to_width(CAL.to_position(w)) - w) < 1e-4


def test_calibration_must_be_monotonic():
    with pytest.raises(ValueError):
        WidthCalibration(np.array([0.0, 0.05, 0.04]), np.array([1, 2, 3.0]))
    with pytest.raises(ValueError):
        WidthCalibration(np.array([0.0, 0.05, 0.1]), np.array([1, 3, 2.0]))


def test_squeeze_delta_inside_driver_only():
    g = FakeGripper(CAL, squeeze_delta_m=0.005, squeeze_below_m=0.06)
    g.set_width(0.045)
    assert g.commanded_width() == pytest.approx(0.045)                 # what the controller publishes / history uses
    assert CAL.to_width(g.target_position()) == pytest.approx(0.040)   # what the servo is told
    g.set_width(0.08)
    assert CAL.to_width(g.target_position()) == pytest.approx(0.08)    # no offset when open


def test_fake_gripper_stalls_at_object():
    import time

    g = FakeGripper(CAL, object_width_m=0.045, speed=1e6, squeeze_delta_m=0.005)
    g.set_width(0.045)
    time.sleep(0.01)
    assert g.get_width() == pytest.approx(0.045, abs=1e-6)


# ------------------------------------------------------------------ TCP composition
def test_tcp_composition_against_independent_4x4():
    from franka_deploy.controllers.franka_cartesian import FrankaCartesian, pose7_from_T, T_from_pose7

    F_T_TCP = np.eye(4)
    F_T_TCP[:3, :3] = np.array([[1, 0, 0], [0, 0.966, -0.259], [0, 0.259, 0.966]])
    u, _, vt = np.linalg.svd(F_T_TCP[:3, :3])
    F_T_TCP[:3, :3] = u @ vt
    F_T_TCP[:3, 3] = [0.0, 0.0644, 0.3114]
    O_T_F = T(p=(0.45, 0.1, 0.4), rotvec=(np.pi, 0.1, -0.2))
    c = FrankaCartesian.__new__(FrankaCartesian)
    c.EE_T_TCP, c.TCP_T_EE = F_T_TCP, np.linalg.inv(F_T_TCP)
    # independent: p_tcp = p_F + R_F @ t, R_tcp = R_F @ R_ft
    p_ind = O_T_F[:3, 3] + O_T_F[:3, :3] @ F_T_TCP[:3, 3]
    R_ind = O_T_F[:3, :3] @ F_T_TCP[:3, :3]
    got = c.tcp_from_ee(O_T_F)
    np.testing.assert_allclose(got[:3, 3], p_ind, atol=1e-12)
    np.testing.assert_allclose(got[:3, :3], R_ind, atol=1e-12)
    np.testing.assert_allclose(c.ee_from_tcp(got), O_T_F, atol=1e-12)            # applied exactly once each way
    np.testing.assert_allclose(T_from_pose7(pose7_from_T(got)), got, atol=1e-12)
