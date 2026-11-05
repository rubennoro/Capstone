import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc.geometry.trajectory import (
    SCALE_MIN_EXCITATION, SCALE_TOLERANCE, Trajectory, gravity_align, gravity_check, load_orbslam3_euroc, load_tum,
    motion_consistency_check, position_jitter, rig_frame, save_tum, scale_check, time_offset_check,
)
from dataproc.geometry.transforms import rot_x, rot_y, rot_z
from dataproc.sim.imu_synth import synth_imu, synth_static_imu
from test_geometry_common import make_rig, truth_poses

T0 = 5_000_000_000_000


def _traj(n=11, dt=100_000_000):
    t = T0 + np.arange(n, dtype=np.int64) * dt
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(n) * 0.1                          # 1 m/s along x
    T[:, :3, :3] = Rotation.from_euler("z", np.arange(n) * 9.0, degrees=True).as_matrix()
    return Trajectory.from_T(t, T)


def test_sample_keys_midpoints_and_validity():
    tr = _traj()
    T, ok = tr.sample(tr.t_ns, max_gap_ns=1)                 # exact keys valid regardless of the gap limit
    assert ok.all() and np.allclose(T, tr.T(), atol=1e-12)
    mid = tr.t_ns[:-1] + 50_000_000
    T, ok = tr.sample(mid, max_gap_ns=100_000_000)
    assert ok.all()
    assert np.allclose(T[:, 0, 3], np.arange(10) * 0.1 + 0.05)
    yaw = Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz", degrees=True)[:, 2]
    assert np.allclose(yaw, np.arange(10) * 9.0 + 4.5)
    _, ok = tr.sample(mid, max_gap_ns=99_999_999)            # bracket too wide
    assert not ok.any()
    _, ok = tr.sample([T0 - 1, int(tr.t_ns[-1]) + 1], max_gap_ns=10**18)
    assert not ok.any()


def test_trajectory_validation_and_frames():
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory([2, 1], np.zeros((2, 3)), [[0, 0, 0, 1]] * 2)
    with pytest.raises(ValueError, match="unknown trajectory frame"):
        Trajectory([1], np.zeros((1, 3)), [[0, 0, 0, 1]], frame="body")
    assert rig_frame("imu0") == "imu"
    q = Trajectory([1, 2], np.zeros((2, 3)), [[0, 0, 0, 1], [0, 0, 0, -1]]).q_xyzw
    assert np.allclose(q[1], [0, 0, 0, 1])                    # sign continuity


def test_tum_roundtrip_exact_ns(tmp_path):
    tr = _traj()
    tr = Trajectory(tr.t_ns + 123, tr.p, tr.q_xyzw)          # odd ns survive the seconds text format
    save_tum(tr, tmp_path / "a.tum")
    back = load_tum(tmp_path / "a.tum")
    assert np.array_equal(back.t_ns, tr.t_ns)
    assert np.allclose(back.T(), tr.T(), atol=1e-9)


def test_load_orbslam3_euroc_fixed_point_ns(tmp_path):
    # SaveTrajectoryEuRoC: `f << fixed`, time setprecision(6) as 1e9 * t_s, then x y z qx qy qz qw
    (tmp_path / "f.txt").write_text("5000000000000.000000 0.1 0.2 0.3 0 0 0 1\n"
                                    "5000033333333.000244 0.1 0.2 0.3 0 0 0.7071067811865476 0.7071067811865476\n")
    tr = load_orbslam3_euroc(tmp_path / "f.txt")
    assert tr.t_ns.tolist() == [5_000_000_000_000, 5_000_033_333_333] and tr.frame == "imu"
    assert np.allclose(tr.T()[1, :3, :3], rot_z(90))


@pytest.mark.parametrize("frame", ["imu", "ultrawide"])
def test_gravity_align_undoes_tilt(frame):
    rig = make_rig(**{"cameras.ultrawide.T_imu_cam": {"rotation": {"euler_xyz_deg": [-90, 0, -90]},
                                                      "translation_m": [0.0, 0.0, 0.0]}})
    t = T0 + np.arange(0, 4_000_000_001, 1_000_000, dtype=np.int64)
    T_true = truth_poses(t, T0)
    ti, g, a = synth_imu(t, T_true, 200, noise=None, smooth=False)
    if frame == "ultrawide":
        T_true = T_true @ rig.cameras["ultrawide"].T_imu_cam       # the same motion, as the camera pose
    tilt = np.eye(4)
    tilt[:3, :3] = rot_z(25) @ rot_x(12) @ rot_y(-7)
    tilt[:3, 3] = [1, 2, 3]
    tr = Trajectory.from_T(t, tilt @ T_true, frame=frame)
    assert not gravity_check(tr, ti, a, rig)["ok"]
    al, info = gravity_align(tr, ti, a, rig)
    assert info["angle_deg"] == pytest.approx(np.degrees(Rotation.from_matrix(rot_x(12) @ rot_y(-7)).magnitude()),
                                              abs=0.05)
    gc = gravity_check(al, ti, a, rig)
    assert gc["ok"] and gc["angle_to_up_deg"] < 0.05
    # aligned = (yaw + translation) @ truth: z axes of R_al R_true^T stay vertical, constant over time
    dR = np.einsum("nij,nkj->nik", al.T()[:, :3, :3], T_true[:, :3, :3])
    assert np.degrees(np.arccos(np.clip(dR[:, 2, 2], -1, 1))).max() < 0.05
    assert np.allclose(dR, dR[0], atol=1e-9)


def test_gravity_align_static():
    rig = make_rig()
    R = rot_x(30)
    ti, g, a = synth_static_imu(T0, T0 + 2_000_000_000, 200, R_W_S=R)
    tr = Trajectory.from_T([T0, T0 + 2_000_000_000], np.stack([np.eye(4)] * 2), frame="imu")
    al, info = gravity_align(tr, ti, a, rig)
    assert info["angle_deg"] == pytest.approx(30, abs=1e-6)
    assert np.allclose(al.T()[0, :3, :3][:, 2] @ R[:, 2], 1.0)   # IMU z axis back where gravity says


def test_time_offset_and_motion_checks():
    rig = make_rig()
    t = T0 + np.arange(0, 4_000_000_001, 1_000_000, dtype=np.int64)
    T = truth_poses(t, T0)
    ti, g, a = synth_imu(t, T, 200, noise=None, smooth=False)
    tr = Trajectory.from_T(t[::33], T[::33])
    toff = time_offset_check(tr, ti, g)
    assert toff["ok"] and abs(toff["traj_delay_ms"]) < 1.0
    late = Trajectory.from_T(tr.t_ns + 20_000_000, tr.T())
    assert time_offset_check(late, ti, g)["traj_delay_ms"] == pytest.approx(20.0, abs=1.5)
    mc = motion_consistency_check(tr, ti, g, rig)
    assert mc["ok"] and mc["gyro_resid_rms"] < 0.02
    Tj = tr.T()
    Tj[60:, 0, 3] += 0.5                                     # a 0.5 m jump in one 33 ms step
    assert not motion_consistency_check(Trajectory.from_T(tr.t_ns, Tj), ti, g, rig)["ok"]


# ---------------------------------------------------------------- metric scale vs the accelerometer

def _moving(scale=1.0, bias=(0.0, 0.0, 0.0)):
    rig = make_rig()
    t = T0 + np.arange(0, 4_000_000_001, 1_000_000, dtype=np.int64)
    T = truth_poses(t, T0)
    ti, g, a = synth_imu(t, T, 200, noise=None, smooth=False)
    Tv = T[::33].copy()
    Tv[:, :3, 3] *= scale
    return Trajectory.from_T(t[::33], Tv), ti, a + np.asarray(bias), rig


def test_scale_check_correct_and_wrong():
    tr, ti, a, rig = _moving()
    sc = scale_check(tr, ti, a, rig)
    assert sc["ok"] and sc["status"] == "ok" and sc["scale"] == pytest.approx(1.0, abs=0.01) and sc["problems"] == []
    assert sc["r2"] > 0.99 and sc["excitation_rms"] > SCALE_MIN_EXCITATION
    for s, expect in ((0.5, 2.0), (1.25, 0.8)):
        tr, ti, a, rig = _moving(scale=s)
        sc = scale_check(tr, ti, a, rig)
        assert sc["status"] == "scale_mismatch" and not sc["ok"] and sc["scale"] == pytest.approx(expect, rel=0.02)
        assert sc["problems"][0].startswith("scale_mismatch")
    tr, ti, a, rig = _moving(scale=1.0 + SCALE_TOLERANCE / 2)
    assert scale_check(tr, ti, a, rig)["status"] == "ok"


def test_scale_check_tolerates_accel_bias():
    tr, ti, a, rig = _moving(bias=(0.2, -0.1, 0.15))                 # ~20 mg, body frame
    sc = scale_check(tr, ti, a, rig)
    assert sc["status"] == "ok" and sc["scale"] == pytest.approx(1.0, abs=0.02)
    assert np.allclose(sc["accel_bias_imu"], [0.2, -0.1, 0.15], atol=0.03)


def test_scale_check_unobservable():
    rig = make_rig()
    ti, g, a = synth_static_imu(T0, T0 + 3_000_000_000, 200, R_W_S=rot_x(20))
    T = np.stack([np.eye(4)] * 91)
    T[:, :3, :3] = rot_x(20)
    tr = Trajectory.from_T(T0 + np.arange(91, dtype=np.int64) * 33_333_333, T)
    sc = scale_check(tr, ti, a, rig)
    assert sc["status"] == "scale_unobservable" and not sc["ok"] and "excitation" in sc["problems"][0]
    # gentle motion: 1 mm amplitude
    t = T0 + np.arange(0, 4_000_000_001, 1_000_000, dtype=np.int64)
    T = truth_poses(t, T0)
    T[:, :3, 3] = (T[:, :3, 3] - T[0, :3, 3]) * 0.01
    ti, g, a = synth_imu(t, T, 200, noise=None, smooth=False)
    assert scale_check(Trajectory.from_T(t[::33], T[::33]), ti, a, rig)["status"] == "scale_unobservable"
    # too short to filter
    tr, ti, a, rig = _moving()
    short = Trajectory(tr.t_ns[:20], tr.p[:20], tr.q_xyzw[:20])
    assert scale_check(short, ti, a, rig)["status"] == "scale_unobservable"


# ---------------------------------------------------------------- scale under position jitter (errors in variables)

@pytest.fixture(scope="module")
def sim_demo():
    """The full-resolution 15 s sim demonstration (rig_v1, seed 0, as ``dataproc sim`` makes it) without rendering:
    exact poses at the 30 Hz ultrawide times plus the sim's noisy 416 Hz IMU."""
    from pathlib import Path
    from dataproc.config import load_rig
    from dataproc.sim.episode import FRAME_PHASE_NS
    from dataproc.sim.imu_synth import add_noise, imu_times
    from dataproc.sim.trajectory import make_trajectory

    rig = load_rig(Path(__file__).resolve().parents[1] / "configs/rigs/rig_v1.yaml")
    sim = make_trajectory(rig, "demonstration", 0, 15.0, 2.5)
    t_cam = T0 + FRAME_PHASE_NS["ultrawide"] + np.round(np.arange(450) * 1e9 / 30).astype(np.int64)
    T = sim.T_W_imu((t_cam - T0) * 1e-9)
    ti = imu_times(T0, T0 + 15_000_000_000, 416.0)
    g, a = sim.imu((ti - T0) * 1e-9)
    n = rig.imu.noise
    nd = {k: getattr(n, k) for k in ("gyroscope_noise_density", "gyroscope_random_walk",
                                     "accelerometer_noise_density", "accelerometer_random_walk")}
    g, a = add_noise(ti, g, a, nd, seed=0x13579)                     # as generate_episode(seed=0)
    return t_cam, T, ti, a, rig


def _jittered(t, T, sigma, seed, scale=1.0):
    T = T.copy()
    T[:, :3, 3] = scale * (T[:, :3, 3] + np.random.default_rng(seed).normal(0, sigma, (len(T), 3)))
    return Trajectory.from_T(t, T)


def test_position_jitter_estimate():
    rng = np.random.default_rng(0)
    t = T0 + np.arange(600, dtype=np.int64) * 33_333_333
    p = np.stack([0.3 * np.sin(np.arange(600) / 30.0 * k) for k in (1.0, 2.0, 3.0)], 1)    # smooth motion
    tr = Trajectory(t, p + rng.normal(0, [0.002, 0.005, 0.0], (600, 3)), np.tile([0, 0, 0, 1.0], (600, 1)))
    j = position_jitter(tr)
    assert j["sigma_m"] == pytest.approx([0.002, 0.005, 0.0], rel=0.15, abs=1e-5) and j["dt_s"] == pytest.approx(1 / 30)
    p2 = tr.p.copy()
    p2[300:] += 0.2                                                  # one VIO jump: robust (MAD)
    assert position_jitter(Trajectory(t, p2, tr.q_xyzw))["sigma_m"][1] == pytest.approx(j["sigma_m"][1], rel=0.05)
    # irregular spacing: windows spanning a gap are skipped
    t3 = np.concatenate([t[:300], t[300:] + 500_000_000])
    assert position_jitter(Trajectory(t3, tr.p, tr.q_xyzw))["n"] == 600 - 4 - 4
    assert position_jitter(Trajectory(t[:5], p[:5], tr.q_xyzw[:5]))["sigma_m"] is None


def test_scale_check_corrects_jitter_attenuation(sim_demo):
    """Reviewer repro (wave 1): 5 mm white position noise on the full-res 15 s demo gave s 0.88-0.92 (plain least
    squares, one false scale_mismatch). The corrected estimate stays within 0.05 of 1, and scale errors of
    x0.85 / x1.2 are still detected."""
    t, T, ti, a, rig = sim_demo
    s, s_ols = [], []
    for seed in range(8):
        sc = scale_check(_jittered(t, T, 0.005, seed), ti, a, rig)
        assert sc["status"] == "ok", sc["problems"]
        s.append(sc["scale"])
        s_ols.append(sc["scale_uncorrected"])
        assert sc["jitter"]["sigma_m"] == pytest.approx([0.005] * 3, rel=0.2)
        assert 0.85 < sc["attenuation"] < 0.95
    assert np.max(np.abs(np.array(s) - 1)) < 0.05
    assert np.mean(s_ols) < 0.94                     # the bias the correction removes
    assert scale_check(_jittered(t, T, 0.005, 0), ti, a, rig, correct_jitter=False)["scale"] == \
        pytest.approx(s_ols[0])
    for f in (0.85, 1.2):                            # trajectory too small / too large by f -> s ~ 1 / f
        for seed in range(3):
            sc = scale_check(_jittered(t, T, 0.005, seed, scale=f), ti, a, rig)
            assert sc["status"] == "scale_mismatch" and sc["scale"] == pytest.approx(1 / f, rel=0.05)


def test_scale_check_jitter_dominates(sim_demo):
    t, T, ti, a, rig = sim_demo
    sc = scale_check(_jittered(t, T, 0.02, 0), ti, a, rig)
    assert sc["status"] == "scale_unobservable" and sc["attenuation"] < 0.5
    assert "jitter" in sc["problems"][0]
    clean = scale_check(Trajectory.from_T(t, T), ti, a, rig)        # no jitter: no correction
    assert clean["status"] == "ok" and clean["attenuation"] > 0.999
    assert clean["scale"] == pytest.approx(clean["scale_uncorrected"], abs=1e-3)
