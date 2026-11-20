"""teleop_train.trajectory: grid, interpolation, motion detection, trimming, gaps (PLAN.md §5.2 steps 1-5, §11)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import synthetic_episode as se
from teleop_train import trajectory as tj

HZ = tj.GRID_HZ
DT_NS = 1e9 / HZ


def states(truth: se.Truth, hz: float = 100.0, lead_s: float = 0.2, offset_ns: int = 0) -> np.ndarray:
    """State times at `hz` from lead_s before the episode start to lead_s after its end (as make_episode)."""
    n = int((truth.duration_s + 2 * lead_s) * hz) + 1
    return (truth.t0_ns - round(lead_s * 1e9) + offset_ns + np.round(np.arange(n) * 1e9 / hz)).astype(np.int64)


def rot_err_deg(rotvec_a: np.ndarray, R_b: Rotation) -> np.ndarray:
    return np.degrees((Rotation.from_rotvec(rotvec_a).inv() * R_b).magnitude())


def grid_poses(n: int, *, pos=None, rot: Rotation | None = None, width=None, t0: int = se.T0_NS) -> tj.GridPoses:
    """GridPoses on a 60 Hz grid with constant defaults (home pose, open gripper)."""
    t = tj.make_grid(t0, t0 + round((n - 1) * DT_NS) + 1)[:n]
    assert len(t) == n
    pos = np.tile(se.HOME_POS, (n, 1)) if pos is None else np.asarray(pos, float)
    rot = Rotation.from_quat(np.tile(se.HOME_ROT.as_quat(), (n, 1))) if rot is None else rot
    width = np.full(n, 0.08) if width is None else np.asarray(width, float)
    return tj.GridPoses(t_grid=t, pos=pos, rotvec=rot.as_rotvec(), quat=rot.as_quat(), width=width)


def ramp(n: int, start: int, stop: int) -> np.ndarray:
    """Seconds of motion so far per step: 0 before `start`, rising at 1 s/s until `stop`, then constant."""
    k = np.clip(np.arange(n), start, stop) - start
    return k / HZ


# ---------------------------------------------------------------- window and grid

def test_common_window():
    a = np.array([10, 20, 30, 40], np.int64)
    b = np.array([15, 25, 35], np.int64)
    c = np.array([5, 50], np.int64)
    assert tj.common_window(a, b, c) == (15, 35)
    assert all(isinstance(v, int) for v in tj.common_window(a, b, c))


def test_common_window_refusals():
    with pytest.raises(tj.TrajectoryError, match="do not overlap"):
        tj.common_window(np.array([0, 10]), np.array([20, 30]), np.array([0, 40]))
    with pytest.raises(tj.TrajectoryError, match="do not overlap"):          # touching is not a window
        tj.common_window(np.array([0, 10]), np.array([10, 30]), np.array([0, 40]))
    with pytest.raises(tj.TrajectoryError, match="main frames"):
        tj.common_window(np.array([0, 10]), np.array([], np.int64), np.array([0, 40]))


def test_make_grid_exact_integers():
    lo = se.T0_NS + 7
    g = tj.make_grid(lo, lo + 10**9)
    assert g.dtype == np.int64
    assert len(g) == 61 and g[0] == lo and g[-1] == lo + 10**9
    expect = [lo + round(k * 10**9 / 60) for k in range(61)]
    assert g.tolist() == expect
    # long episode: no drift, every step 16_666_666 or 16_666_667 ns
    g = tj.make_grid(lo, lo + 600 * 10**9)
    assert len(g) == 600 * 60 + 1 and g[-1] == lo + 600 * 10**9
    assert set(np.diff(g).tolist()) == {16_666_666, 16_666_667}


def test_make_grid_end_rule():
    lo = 1000
    assert tj.make_grid(lo, lo + 16_666_666).tolist() == [lo]                 # t_1 = lo + 16_666_667 > hi
    assert tj.make_grid(lo, lo + 16_666_667).tolist() == [lo, lo + 16_666_667]
    assert tj.make_grid(lo, lo).tolist() == [lo]
    assert tj.make_grid(lo, lo - 1).size == 0
    g = tj.make_grid(0, 10**9, hz=7)
    assert g.tolist() == [round(k * 10**9 / 7) for k in range(8)]


# ---------------------------------------------------------------- interpolation

def _interp_truth(truth: se.Truth):
    t_s = states(truth)
    lo, hi = int(t_s[0]), int(t_s[-1])
    g = tj.make_grid(lo, hi)
    poses = tj.interpolate(t_s, truth.target(t_s), truth.width(t_s), g)
    return t_s, g, poses


def test_interpolate_against_truth():
    truth = se.Truth(se.T0_NS, 2.0, 2.0)
    t_s, g, p = _interp_truth(truth)
    assert p.t_grid.tolist() == g.tolist() and p.t_grid.dtype == np.int64
    assert p.pos.shape == (len(g), 3) and p.rotvec.shape == (len(g), 3)
    assert p.quat.shape == (len(g), 4) and p.width.shape == (len(g),)
    e_pos = np.linalg.norm(p.pos - truth.position(g), axis=1)
    e_rot = rot_err_deg(p.rotvec, truth.rotation(g))
    e_w = np.abs(p.width - truth.width(g))
    # rotvec and quat describe the same rotation
    assert np.all(np.degrees((Rotation.from_quat(p.quat).inv() * Rotation.from_rotvec(p.rotvec)).magnitude()) < 1e-9)

    # The truth is linear in position/width and a constant-rate geodesic in rotation inside each phase, so
    # interpolation is exact (to rounding) except within one state interval (10 ms) of a phase boundary.
    s = truth.still_start_s
    bounds = truth.t0_ns + np.round(np.array([s, s + truth.move_s, s + truth.move_s + truth.rotate_s,
                                              s + truth.move_s + truth.rotate_s + truth.grip_s]) * 1e9)
    far = np.min(np.abs(g[:, None] - bounds[None, :]), axis=1) > 10_000_000
    assert far.sum() > 0.9 * len(g)
    assert e_pos[far].max() < 1e-12
    assert e_rot[far].max() < 1e-9
    assert e_w[far].max() < 1e-12
    # Near a kink the chord error is at most rate * 10 ms / 4.
    assert e_pos.max() <= truth.v_mps * 0.01 / 4 + 1e-12
    assert e_rot.max() <= truth.w_deg_s * 0.01 / 4 + 1e-9
    assert e_w.max() <= 0.06 * 0.01 / 4 + 1e-12


def test_interpolate_state_times_exact():
    truth = se.Truth(se.T0_NS, 1.0, 1.0)
    t_s = states(truth)
    T = truth.target(t_s)
    w = truth.width(t_s)
    g = t_s[::3].copy()                                   # includes the first state
    g = np.append(g, t_s[-1])                             # and the last one
    p = tj.interpolate(t_s, T, w, g)
    idx = np.searchsorted(t_s, g)
    R = Rotation.from_matrix(T[:, :3, :3])
    assert np.array_equal(p.pos, T[idx, :3, 3])
    assert np.array_equal(p.width, w[idx])
    assert np.array_equal(p.rotvec, R.as_rotvec()[idx])
    assert np.array_equal(p.quat, R.as_quat()[idx])


def test_interpolate_constant_rate_inputs_exact():
    # irregular state times, linear position/width, constant-rate rotation: exact everywhere in between
    t_s = se.T0_NS + np.cumsum([0, 9_000_000, 11_000_000, 10_000_000, 13_500_000, 7_000_000]).astype(np.int64)
    u = (t_s - t_s[0]) / 1e9
    T = np.tile(np.eye(4), (len(t_s), 1, 1))
    T[:, :3, 3] = np.array([0.3, -0.1, 0.5]) + np.outer(u, [0.02, -0.03, 0.01])
    R = Rotation.from_rotvec(np.outer(u, [0.0, 0.0, np.radians(20.0)])) * Rotation.from_euler("y", 30, degrees=True)
    T[:, :3, :3] = R.as_matrix()
    w = 0.05 - 0.04 * u
    g = np.linspace(t_s[0], t_s[-1], 23).round().astype(np.int64)
    p = tj.interpolate(t_s, T, w, g)
    ug = (g - t_s[0]) / 1e9
    np.testing.assert_allclose(p.pos, np.array([0.3, -0.1, 0.5]) + np.outer(ug, [0.02, -0.03, 0.01]), atol=1e-14)
    np.testing.assert_allclose(p.width, 0.05 - 0.04 * ug, atol=1e-14)
    R_true = Rotation.from_rotvec(np.outer(ug, [0, 0, np.radians(20.0)])) * Rotation.from_euler("y", 30, degrees=True)
    assert rot_err_deg(p.rotvec, R_true).max() < 1e-9


@pytest.mark.parametrize("a_deg,b_deg", [(-5.0, 5.0), (-170.0, 170.0)])
def test_slerp_near_pi(a_deg, b_deg):
    """TCP pointing down: rotation vectors have norm ~pi and can flip sign between neighbours."""
    t_s = np.array([se.T0_NS, se.T0_NS + 10_000_000], np.int64)
    # Rz(a) * Rx(180) is a half turn about a horizontal axis: rotvec norm exactly pi.
    R = Rotation.from_euler("z", [a_deg, b_deg], degrees=True) * se.HOME_ROT
    assert np.allclose(np.linalg.norm(R.as_rotvec(), axis=1), np.pi)
    T = np.tile(np.eye(4), (2, 1, 1))
    T[:, :3, :3] = R.as_matrix()
    g = t_s[0] + np.array([0, 2_500_000, 5_000_000, 7_500_000, 10_000_000], np.int64)
    p = tj.interpolate(t_s, T, np.array([0.08, 0.08]), g)
    # shortest path: the relative turn about base z goes from 0 to (b - a) (the short way round)
    rel = ((b_deg - a_deg + 180.0) % 360.0) - 180.0
    expect = Rotation.from_euler("z", a_deg + rel * np.array([0, 0.25, 0.5, 0.75, 1.0]), degrees=True) * se.HOME_ROT
    assert rot_err_deg(p.rotvec, expect).max() < 1e-9


def test_slerp_across_rotvec_sign_flip():
    """Rx(179.9 deg) and Rx(-179.9 deg) are 0.2 deg apart; the midpoint is Rx(180), not the identity."""
    t_s = np.array([se.T0_NS, se.T0_NS + 10_000_000], np.int64)
    R = Rotation.from_rotvec([[np.radians(179.9), 0, 0], [-np.radians(179.9), 0, 0]])
    T = np.tile(np.eye(4), (2, 1, 1))
    T[:, :3, :3] = R.as_matrix()
    p = tj.interpolate(t_s, T, np.array([0.08, 0.08]), np.array([t_s[0] + 5_000_000], np.int64))
    assert rot_err_deg(p.rotvec, Rotation.from_euler("x", [180.0], degrees=True))[0] < 1e-9
    assert abs(np.linalg.norm(p.rotvec[0]) - np.pi) < 1e-9


def test_interpolate_refusals():
    t_s = np.array([100, 200, 300], np.int64)
    T = np.tile(np.eye(4), (3, 1, 1))
    w = np.zeros(3)
    with pytest.raises(ValueError, match="outside"):
        tj.interpolate(t_s, T, w, np.array([99, 150]))
    with pytest.raises(ValueError, match="outside"):
        tj.interpolate(t_s, T, w, np.array([150, 301]))
    with pytest.raises(ValueError, match="strictly increasing"):
        tj.interpolate(np.array([100, 100, 300]), T, w, np.array([150]))
    with pytest.raises(ValueError, match="shapes"):
        tj.interpolate(t_s, T[:2], w, np.array([150]))
    p = tj.interpolate(t_s, T, w, np.zeros(0, np.int64))
    assert p.pos.shape == (0, 3) and p.quat.shape == (0, 4)
    p = tj.interpolate(t_s[:1], T[:1], w[:1], np.array([100]))
    assert np.array_equal(p.pos, np.zeros((1, 3)))


# ---------------------------------------------------------------- motion detection

def test_still_is_not_moving():
    assert not tj.moving_mask(grid_poses(300)).any()


def test_rates_match_known_speeds():
    n = 300
    v = 0.0125                                            # slow mode, one axis
    pos = np.tile(se.HOME_POS, (n, 1))
    pos[:, 1] += v * np.arange(n) / HZ
    vt, vr, vw = tj.step_rates(grid_poses(n, pos=pos))
    # make_grid steps are 16_666_666 or 16_666_667 ns, the rates assume exactly 1/60 s
    np.testing.assert_allclose(vt, v, rtol=1e-6)
    assert np.all(vr == 0) and np.all(vw == 0)


def test_pure_rotation_counts_as_moving():
    n = 300
    deg = 2.5 * ramp(n, 100, 200)                         # 2.5 deg/s (slow mode) for steps 100..200 only
    rot = Rotation.from_euler("z", deg, degrees=True) * se.HOME_ROT
    p = grid_poses(n, rot=rot)
    m = tj.moving_mask(p)
    assert m[105:196].all()
    assert not m[:85].any() and not m[216:].any()
    _, vr, _ = tj.step_rates(p)
    np.testing.assert_allclose(vr[120:180], 2.5, rtol=1e-6)


def test_pure_gripper_motion_counts_as_moving():
    n = 300
    width = 0.08 - 0.008 * ramp(n, 100, 200)              # closing at 8 mm/s, pose fixed
    m = tj.moving_mask(grid_poses(n, width=width))
    assert m[105:196].all()
    assert not m[:85].any() and not m[216:].any()
    width = 0.02 + 0.008 * ramp(n, 100, 200)              # opening counts too (|rate|)
    assert tj.moving_mask(grid_poses(n, width=width))[105:196].all()


def test_slow_translation_counts_as_moving():
    n = 300
    pos = np.tile(se.HOME_POS, (n, 1))
    pos[:, 2] -= 0.0125 * ramp(n, 100, 200)
    assert tj.moving_mask(grid_poses(n, pos=pos))[105:196].all()


@pytest.mark.parametrize("kind", ["trans", "rot", "width"])
def test_below_thresholds_is_still(kind):
    n = 300
    s = ramp(n, 0, n)
    if kind == "trans":
        pos = np.tile(se.HOME_POS, (n, 1))
        pos[:, 0] += 0.8 * tj.TRANS_SPEED_MPS * s
        p = grid_poses(n, pos=pos)
    elif kind == "rot":
        p = grid_poses(n, rot=Rotation.from_euler("z", 0.8 * tj.ROT_SPEED_DEG_S * s, degrees=True) * se.HOME_ROT)
    else:
        p = grid_poses(n, width=0.08 - 0.8 * tj.WIDTH_RATE_MPS * s)
    assert not tj.moving_mask(p).any()


def test_smoothing_window_and_mapping():
    """One isolated jump of d metres between steps j and j+1: interval rate d*60, steps j and j+1 get half of it,
    and the centred 15-step mean spreads it over steps j-7 .. j+8."""
    n, j, d = 100, 50, 0.01
    pos = np.tile(se.HOME_POS, (n, 1))
    pos[j + 1:, 0] += d
    vt, _, _ = tj.step_rates(grid_poses(n, pos=pos))
    half = round(tj.SMOOTH_S * HZ) // 2
    assert half == 7
    nz = np.flatnonzero(vt > 0)
    assert nz.tolist() == list(range(j - half, j + 1 + half + 1))
    w = 2 * half + 1
    np.testing.assert_allclose(vt[j - half], d * HZ / 2 / w)          # window holds step j only
    np.testing.assert_allclose(vt[j], d * HZ / w)                     # steps j and j + 1
    np.testing.assert_allclose(vt[j + 1], d * HZ / w)
    np.testing.assert_allclose(vt[j + 1 + half], d * HZ / 2 / w)      # step j + 1 only
    # edges: the window shrinks and averages only the samples that exist
    pos = np.tile(se.HOME_POS, (n, 1))
    pos[1:, 0] += d                                       # jump between steps 0 and 1
    vt, _, _ = tj.step_rates(grid_poses(n, pos=pos))
    # step 0 rate = d*60 (one interval), step 1 = d*60/2; window of step 0 is steps 0..7 (8 samples)
    np.testing.assert_allclose(vt[0], (d * HZ + d * HZ / 2) / (half + 1))


def test_short_inputs():
    assert tj.moving_mask(grid_poses(1)).tolist() == [False]
    p = grid_poses(1)
    empty = tj.GridPoses(p.t_grid[:0], p.pos[:0], p.rotvec[:0], p.quat[:0], p.width[:0])
    assert tj.moving_mask(empty).shape == (0,)


# ---------------------------------------------------------------- trimming

def test_trim_against_truth():
    """Sample the truth at 100 Hz, put it on the 60 Hz grid, detect motion and trim.

    Tolerance: smoothing only spreads motion outwards, so the first moving step lies between half a window
    (7 steps = 117 ms) plus one state interval (10 ms) plus one grid step before the true start, and one grid step
    after it; the same mirrored at the end. The kept bounds sit 0.5 s outside those."""
    truth = se.Truth(se.T0_NS, 2.0, 2.0)
    t_s, g, p = _interp_truth(truth)
    m = tj.moving_mask(p)
    i0, i1 = tj.trim_span(g, m)
    half_s = (round(tj.SMOOTH_S * HZ) // 2) / HZ
    early = (half_s + 0.010 + 1 / HZ) * 1e9
    late = (1 / HZ) * 1e9
    start = truth.motion_start_ns - 0.5e9
    end = truth.motion_end_ns + 0.5e9
    assert start - early - 1 <= g[i0] <= start + late + 1
    assert end - late - 1 <= g[i1 - 1] <= end + early + 1
    # the whole true motion (translate, rotate-only, gripper-only) is moving
    inside = (g >= truth.motion_start_ns) & (g <= truth.motion_end_ns)
    assert m[inside].all()
    # the still ends are not, beyond the smoothing reach
    assert not m[g < truth.motion_start_ns - early].any()
    assert not m[g > truth.motion_end_ns + early].any()


def test_trim_clamped_to_grid():
    truth = se.Truth(se.T0_NS, 0.1, 0.1)                  # motion starts 0.1 s in: margin would leave the grid
    t_s = states(truth, lead_s=0.0)
    g = tj.make_grid(int(t_s[0]), int(t_s[-1]))
    p = tj.interpolate(t_s, truth.target(t_s), truth.width(t_s), g)
    i0, i1 = tj.trim_span(g, tj.moving_mask(p))
    assert (i0, i1) == (0, len(g))


def test_trim_keeps_mid_episode_pause():
    n = 900                                               # 15 s
    pos = np.tile(se.HOME_POS, (n, 1))
    pos[:, 0] += 0.05 * (ramp(n, 120, 240) + ramp(n, 600, 720))   # move 2 s, pause 6 s, move 2 s
    p = grid_poses(n, pos=pos)
    m = tj.moving_mask(p)
    assert not m[300:540].any()                           # the pause itself is still
    i0, i1 = tj.trim_span(p.t_grid, m)
    first, last = np.flatnonzero(m)[[0, -1]]
    assert first < 120 and last > 720
    assert i0 == first - 30 and i1 == last + 30 + 1       # 0.5 s = 30 steps either side, pause kept


def test_trim_index_rule():
    t = tj.make_grid(0, 10 * 10**9)
    m = np.zeros(len(t), bool)
    m[[200, 400]] = True
    i0, i1 = tj.trim_span(t, m)
    assert t[i0] >= t[200] - 500_000_000 and t[i0 - 1] < t[200] - 500_000_000
    assert t[i1 - 1] <= t[400] + 500_000_000 and t[i1] > t[400] + 500_000_000
    assert tj.trim_span(t, m, margin_s=0.0) == (200, 401)


def test_trim_refusals():
    t = tj.make_grid(0, 10 * 10**9)
    with pytest.raises(tj.TrajectoryError, match="no motion found"):
        tj.trim_span(t, np.zeros(len(t), bool))
    m = np.zeros(len(t), bool)
    m[300:330] = True                                     # 0.5 s of motion + 2 x 0.5 s margin < 2 s
    with pytest.raises(tj.TrajectoryError, match=r"kept span 1\.48 s is shorter than 2 s"):
        tj.trim_span(t, m)
    m[300:361] = True                                     # 1.0 s + 1.0 s of margin = exactly 2 s: kept
    i0, i1 = tj.trim_span(t, m)
    assert t[i1 - 1] - t[i0] == 2 * 10**9
    assert "\n" not in str(pytest.raises(tj.TrajectoryError, tj.trim_span, t, np.zeros(len(t), bool)).value)


# ---------------------------------------------------------------- gaps

def test_max_gap_includes_neighbours_outside_span():
    t = np.array([0, 10, 20, 50, 60, 100, 110], np.int64)
    assert tj.max_gap_ns(t, 20, 60) == 30                 # samples 20..60
    assert tj.max_gap_ns(t, 25, 55) == 30                 # 20 (at/before lo) .. 60 (at/after hi)
    assert tj.max_gap_ns(t, 55, 65) == 40                 # 50 .. 100: the gap straddling hi counts
    assert tj.max_gap_ns(t, 0, 10) == 10
    assert tj.max_gap_ns(t, 105, 106) == 10
    assert tj.max_gap_ns(t, 70, 80) == 40                 # inside one gap
    assert tj.max_gap_ns(t, -50, 200) == 40               # clamped to the samples
    assert tj.max_gap_ns(t[:1], 0, 10) == 0
    assert isinstance(tj.max_gap_ns(t, 20, 60), int)


def _stream(t0: int, rate: float, dur_s: float, gap: tuple[float, float] | None = None) -> np.ndarray:
    t = (t0 + np.round(np.arange(int(dur_s * rate) + 1) * 1e9 / rate)).astype(np.int64)
    if gap is not None:
        t = t[(t < t0 + gap[0] * 1e9) | (t >= t0 + (gap[0] + gap[1]) * 1e9)]
    return t


def test_check_gaps_pass_and_fail():
    t0 = se.T0_NS
    lo, hi = t0 + 10**9, t0 + 9 * 10**9
    st, mn, uw = _stream(t0, 100, 10), _stream(t0 + 3_000_000, 30, 10), _stream(t0 + 7_000_000, 30, 10)
    tj.check_gaps(st, mn, uw, lo, hi)
    # 2 s of missing arm states (a seq gap) inside the span
    with pytest.raises(tj.TrajectoryError, match=r"gap of 2010\.0 ms between consecutive arm states"):
        tj.check_gaps(_stream(t0, 100, 10, (4.0, 2.0)), mn, uw, lo, hi)
    # 40 ms of arm-state gap passes, 60 ms does not
    tj.check_gaps(_stream(t0, 100, 10, (4.0, 0.03)), mn, uw, lo, hi)
    with pytest.raises(tj.TrajectoryError, match="arm states"):
        tj.check_gaps(_stream(t0, 100, 10, (4.0, 0.05)), mn, uw, lo, hi)
    with pytest.raises(tj.TrajectoryError, match=r"main frames inside the kept span \(limit 100 ms\)"):
        tj.check_gaps(st, _stream(t0 + 3_000_000, 30, 10, (5.0, 0.1)), uw, lo, hi)
    with pytest.raises(tj.TrajectoryError, match=r"ultrawide frames inside the kept span \(limit 200 ms\)"):
        tj.check_gaps(st, mn, _stream(t0 + 7_000_000, 30, 10, (5.0, 0.2)), lo, hi)
    # a gap outside the span (and not next to it) does not count
    tj.check_gaps(_stream(t0, 100, 10, (0.2, 0.5)), mn, uw, lo, hi)


def test_check_gaps_message_one_line():
    t0 = se.T0_NS
    with pytest.raises(tj.TrajectoryError) as e:
        tj.check_gaps(_stream(t0, 100, 10, (4.0, 2.0)), _stream(t0, 30, 10), _stream(t0, 30, 10),
                      t0 + 10**9, t0 + 9 * 10**9)
    assert "\n" not in str(e.value)
