"""T8: export grid (native vs fill), UW decimation/slotting/PTS, the 1-s fps abort."""

import numpy as np

from sensor_rig.export.grid import build_grid, fps_abort
from sensor_rig.export.slotting import select_uw, slot_uw, uw_pts

T0 = 1_000_000_000_000


def _times(fps, n, t0=T0):
    return t0 + np.round(np.arange(n) * 1e9 / fps).astype(np.int64)


def test_native_grid():
    t = _times(60, 120)
    t = np.delete(t, [50, 51])  # dropped frames stay as gaps (VFR)
    g = build_grid(t)
    assert g.mode == "native" and np.array_equal(g.t_ns, t) and np.array_equal(g.src_idx, np.arange(len(t)))


def test_fill_grid_nearest():
    t = _times(30, 60)
    g = build_grid(t)
    assert g.mode == "fill"
    assert g.t_ns[0] == t[0] and g.t_ns[-1] <= t[-1]
    assert np.all(np.abs(np.diff(g.t_ns) - 1e9 / 60) <= 1)
    assert len(g) == 2 * len(t) - 1
    # every slot shows the nearest frame (F4); exact midway ties alternate so the mean error is ~0
    err = g.t_ns - t[g.src_idx]
    assert np.all(np.abs(err) <= 1e9 / 60 + 1)
    assert abs(err.mean()) < 1e6
    assert g.src_idx[:8].tolist() == [0, 0, 1, 2, 2, 2, 3, 4]
    # with jitter there are no ties: strictly nearest
    tj = t + np.random.default_rng(0).integers(-3_000_000, 3_000_000, len(t))
    tj[0] = t[0]
    gj = build_grid(tj)
    d = np.abs(gj.t_ns[:, None] - tj[None, :])
    assert np.array_equal(gj.src_idx, d.argmin(axis=1))


def test_threshold():
    assert build_grid(_times(50, 100)).mode == "native"
    assert build_grid(_times(49, 100)).mode == "fill"
    assert build_grid(_times(49, 100), native_min_fps=40).mode == "native"


def test_fps_abort():
    t = _times(60, 300)
    assert fps_abort(t, 31) is None
    # 2 s at 20 fps after the first second -> abort
    bad = np.concatenate([t[:60], t[60:180:3], t[180:]])
    assert fps_abort(bad, 31) is not None
    # the same slowdown inside the first second is tolerated (warm-up)
    early = np.concatenate([t[:5], t[5:50:3], t[50:]])
    assert fps_abort(early, 31) is None


def test_uw_decimation_30_to_10():
    t = _times(30, 90) + 3_000_000
    sel = select_uw(t, 10.0)
    assert np.all(np.diff(sel) == 3) and len(sel) == 30
    assert np.array_equal(select_uw(t, None), np.arange(90))
    # jittered 30 fps still decimates to ~10 Hz
    tj = t + np.random.default_rng(0).integers(-2_000_000, 2_000_000, len(t))
    d = np.diff(tj[select_uw(tj, 10.0)])
    assert np.all(d > 80e6) and np.all(d < 120e6)


def test_slotting_nearest_and_pattern():
    g = _times(60, 120)
    t_uw = _times(10, 20) + 4_000_000  # 4 ms after a slot -> that slot
    slot_of = slot_uw(g, t_uw, np.arange(20))
    occ = np.flatnonzero(slot_of >= 0)
    assert occ.tolist() == list(range(0, 120, 6))
    assert np.array_equal(slot_of[occ], np.arange(20))
    t_uw2 = _times(10, 20) + 12_000_000  # 12 ms after slot k -> nearer to slot k+1
    occ2 = np.flatnonzero(slot_uw(g, t_uw2, np.arange(20)) >= 0)
    assert occ2.tolist() == list(range(1, 120, 6))


def test_slotting_collision_and_range():
    g = _times(60, 10)
    t_uw = np.array([g[3] - 6_000_000, g[3] + 2_000_000, g[9] + 30_000_000], dtype=np.int64)
    slot_of = slot_uw(g, t_uw, np.arange(3))
    assert slot_of[3] == 1  # the closer of two candidates wins, at most one per slot
    assert (slot_of >= 0).sum() == 1  # the frame 30 ms after the last slot has no slot


def test_uw_pts():
    g = _times(60, 60)
    slot_of = np.full(60, -1)
    slot_of[[2, 8, 14, 21]] = [0, 1, 2, 3]
    pts = uw_pts(g, slot_of, 600)
    assert pts.tolist() == [0, 60, 120, 190]
