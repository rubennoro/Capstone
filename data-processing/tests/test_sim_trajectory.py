"""dataproc.sim.trajectory: analytic derivatives, motion limits, pre-roll excitation, width schedule, determinism."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc.config import load_rig
from dataproc.sim.imu_synth import G_WORLD, synth_imu
from dataproc.sim.trajectory import (CALIB_WIDTH_RANGE_M, STATIC_START_S, MinJerk, WindowedSines,
                                     make_trajectory)

RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"


@pytest.fixture(scope="module")
def rig():
    return load_rig(RIG_PATH)


@pytest.mark.parametrize("term", [
    MinJerk((0.0, 1.0, 2.5, 3.0), (0.0, 0.3, -0.2, -0.1)),
    WindowedSines((0.05, 0.02), (0.7, 1.3), (0.1, 2.0), 0.3, 0.7, 2.0, 2.4),
    WindowedSines((0.01,), (0.3,), (0.5,), 0.3, 0.9),
])
def test_term_derivatives_match_finite_differences(term):
    t = np.linspace(-0.5, 3.5, 4001)
    h = 1e-6
    f, df, ddf = term(t)
    fp, fm = term(t + h)[0], term(t - h)[0]
    assert np.abs((fp - fm) / (2 * h) - df).max() < 1e-6
    dfp, dfm = term(t + h)[1], term(t - h)[1]
    assert np.abs((dfp - dfm) / (2 * h) - ddf).max() < 1e-3


@pytest.mark.parametrize("kind, duration, preroll", [("demonstration", 15.0, 2.5), ("demonstration", 4.0, 2.0),
                                                     ("calibration", 8.0, 2.5)])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_kinematics_are_exact_and_limited(rig, kind, duration, preroll, seed):
    tr = make_trajectory(rig, kind, seed, duration, preroll)
    t = np.arange(0.0, duration, 1e-3)
    k = tr.kinematics(t)
    h = 1e-3
    # velocity / acceleration vs central differences of the analytic position
    kp, km = tr.kinematics(t + h), tr.kinematics(t - h)
    assert np.abs((kp["p"] - km["p"]) / (2 * h) - k["v"]).max() < 1e-3
    h2 = 1e-6                                       # C2 knots: keep the jerk x h truncation error small
    kp2, km2 = tr.kinematics(t + h2), tr.kinematics(t - h2)
    assert np.abs((kp2["v"] - km2["v"]) / (2 * h2) - k["a"]).max() < 1e-3
    # body rate vs the relative rotation over +-h
    dR = np.einsum("nji,njk->nik", km["R"], kp["R"])
    assert np.abs(Rotation.from_matrix(dR).as_rotvec() / (2 * h) - k["omega"]).max() < 1e-4
    assert np.allclose(np.einsum("nij,nkj->nik", k["R"], k["R"]), np.eye(3), atol=1e-12)
    speed = np.linalg.norm(k["v"], axis=1)
    assert speed.max() < 1.0
    assert np.linalg.norm(k["omega"], axis=1).max() < 2.0
    static = t < STATIC_START_S
    assert speed[static].max() < 1e-12 and np.abs(k["omega"][static]).max() < 1e-12
    # SLAM-friendly pre-roll: real translation and rotation excitation on several axes
    pre = (t > STATIC_START_S) & (t < preroll)
    span = k["p"][pre].max(0) - k["p"][pre].min(0)
    assert (span > 0.03).sum() >= 2
    rot = Rotation.from_matrix(k["R"][pre])
    rel = (rot[0].inv() * rot).as_rotvec()
    assert (rel.max(0) - rel.min(0) > np.deg2rad(5)).sum() >= 2
    # stays inside the room, above the table
    assert k["p"][:, 2].min() > 0.85 and np.abs(k["p"][:, :2]).max() < 1.2
    demo = (t > tr.button_start_s) & (t < tr.button_stop_s)
    assert speed[demo].max() < 0.5
    if kind == "demonstration" and duration >= 10:
        assert speed[demo].max() > 0.2


def test_imu_matches_vendored_synth_imu(rig):
    """The exact IMU (analytic) agrees with synth_imu on the same dense poses (independent derivation)."""
    tr = make_trajectory(rig, "demonstration", 3, 8.0, 2.5)
    t0 = 1_000_000_000_000
    t_pose = t0 + np.arange(0, 8_000_000_000, 1_000_000)
    T = tr.T_W_imu((t_pose - t0) * 1e-9)
    t_imu, gyro, acc = synth_imu(t_pose, T, 400.0, noise=None, smooth=False)
    g2, a2 = tr.imu((t_imu - t0) * 1e-9)
    inner = slice(400, -400)
    assert np.abs(gyro[inner] - g2[inner]).max() < 2e-3
    assert np.abs(acc[inner] - a2[inner]).max() < 2e-2
    # static start: the accelerometer reads gravity (+g along world up)
    g_static, a_static = tr.imu(np.array([0.1]))
    assert np.allclose(g_static, 0) and np.isclose(np.linalg.norm(a_static), -G_WORLD[2])


def test_width_schedule(rig):
    tr = make_trajectory(rig, "demonstration", 0, 15.0, 2.5)
    t = np.arange(0, 15.0, 1e-3)
    w = tr.width_m(t)
    names = [p[0] for p in tr.phases]
    assert names == ["reach", "grasp", "lift", "hold", "move", "settle"]
    reach, grasp = tr.phases[0], tr.phases[1]
    assert np.allclose(w[t < reach[1]], w[0])                        # closed until the reach
    i_open = np.argmax(w)
    assert reach[1] < t[i_open] < grasp[2] and w.max() - w[0] > 0.03  # opens during the reach
    assert abs(w[-1] - w[0]) < 1e-9                                  # closed again (holding the block)
    cal = make_trajectory(rig, "calibration", 0, 8.0, 2.5)
    wc = cal.width_m(np.arange(0, 8.0, 1e-3))
    assert np.isclose(wc.min(), CALIB_WIDTH_RANGE_M[0]) and np.isclose(wc.max(), CALIB_WIDTH_RANGE_M[1])


def test_seeded_and_deterministic(rig):
    t = np.linspace(0, 10, 500)
    a = make_trajectory(rig, "demonstration", 5, 10.0).T_W_imu(t)
    b = make_trajectory(rig, "demonstration", 5, 10.0).T_W_imu(t)
    c = make_trajectory(rig, "demonstration", 6, 10.0).T_W_imu(t)
    assert np.array_equal(a, b) and not np.allclose(a, c)


@pytest.mark.parametrize("kw", [dict(kind="x"), dict(preroll_s=0.5), dict(duration_s=3.0)])
def test_bad_arguments(rig, kw):
    args = dict(kind="demonstration", seed=0, duration_s=15.0, preroll_s=2.5) | kw
    with pytest.raises(ValueError):
        make_trajectory(rig, **args)
