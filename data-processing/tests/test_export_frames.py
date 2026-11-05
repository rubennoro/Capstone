"""dataproc.export.frames: main nearest-frame selection and the ~10 Hz ultrawide sub-stream (iPhUMI semantics)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from dataproc.export.frames import downsample_local, latest_at_or_before, nearest_frame_index, ultrawide_substream

MS = 1_000_000
GRID_NS = 16_666_667                      # 60 Hz
IPHUMI_ALIGN = Path("/home/ruben/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/process_stages/align.py")


def grid(n, t0=0):
    return t0 + np.round(np.arange(n) * 1e9 / 60).astype(np.int64)


# --------------------------------------------------------------------------------------------- main

def test_nearest_hand_timeline():
    t_cam = np.array([0, 10, 20, 30]) * MS
    t = np.array([-3, 0, 4, 5, 6, 26, 30, 41]) * MS
    idx, valid = nearest_frame_index(t, t_cam, max_dt_ns=5 * MS)
    assert idx.tolist() == [0, 0, 0, 0, 1, 3, 3, 3]          # 5 ms is a tie -> earlier frame
    assert valid.tolist() == [True, True, True, True, True, True, True, False]
    assert idx.dtype == np.int64


def test_nearest_main_at_30fps_fills_60hz_grid():
    t_cam = np.round(np.arange(61) * 1e9 / 30).astype(np.int64) + 2 * MS
    t = grid(120)
    idx, valid = nearest_frame_index(t, t_cam, max_dt_ns=17 * MS)
    assert valid.all()
    assert np.all(np.diff(idx) >= 0)
    assert np.bincount(idx).max() <= 2 and idx[-1] == 59
    # every pick is truly nearest
    for ti, i in zip(t, idx):
        assert np.abs(t_cam[i] - ti) == np.abs(t_cam - ti).min()


def test_nearest_detects_drop_and_validates_input():
    t_cam = np.array([0, 16, 33, 83, 100]) * MS                 # 50 ms drop
    idx, valid = nearest_frame_index(np.array([0, 17, 58, 100]) * MS, t_cam, max_dt_ns=12 * MS)
    assert idx.tolist() == [0, 1, 2, 4] and valid.tolist() == [True, True, False, True]
    with pytest.raises(ValueError):
        nearest_frame_index(np.array([0]), np.array([0, 0]), 1)
    with pytest.raises(TypeError):
        nearest_frame_index(np.array([0.0]), np.array([0, 1]), 1)
    with pytest.raises(ValueError):
        nearest_frame_index(np.array([0]), np.array([], np.int64), 1)


# --------------------------------------------------------------------------------------------- ultrawide

def _check_substream(t, t_uw, sel, up):
    assert sel.dtype == np.int64 and up.dtype == np.int64
    assert len(up) == len(t)
    assert np.all(np.diff(sel) > 0), "no duplicates, in time order"
    assert up[0] == 0 and np.all(np.diff(up) >= 0)
    assert up[-1] == len(sel) - 1
    ts = t_uw[sel]
    for k, ti in enumerate(t):                                  # brute force: latest selected at/before, clamp 0
        before = np.nonzero(ts <= ti)[0]
        assert up[k] == (before[-1] if len(before) else 0)


def test_ultrawide_substream_30fps_two_seconds():
    t = grid(121)                                              # 0 .. 2.0 s
    t_uw = np.round(np.arange(-3, 64) * 1e9 / 30).astype(np.int64) - 4 * MS
    sel, up = ultrawide_substream(t, t_uw, rate_hz=10)
    _check_substream(t, t_uw, sel, up)
    assert len(sel) == 21                                      # schedule 0, 0.1, ..., 2.0 s
    sched = np.arange(21) * 100 * MS
    assert np.abs(t_uw[sel] - sched).max() <= 17 * MS          # nearest frame to each schedule point
    assert np.all(np.diff(sel) == 3)
    # each selected frame is used for ~6 grid steps (60 / 10)
    counts = np.bincount(up)
    assert counts[1:-1].min() >= 5 and counts[1:-1].max() <= 7


def test_ultrawide_hand_timeline():
    t = np.array([0, 50, 100, 150, 200, 250]) * MS
    t_uw = np.array([-10, 20, 60, 95, 140, 190, 230, 260]) * MS
    sel, up = ultrawide_substream(t, t_uw, rate_hz=10)         # schedule 0, 100, 200 ms
    assert sel.tolist() == [0, 3, 5]                           # nearest: -10, 95, 190
    assert up.tolist() == [0, 0, 1, 1, 2, 2]


def test_ultrawide_clamps_and_trims():
    # first selected frame is after the grid start -> early steps clamp to 0; a trailing selection unused by the
    # grid is dropped so that M == up[-1] + 1
    t = np.array([0, 20, 40, 60, 80, 100, 120]) * MS
    t_uw = np.array([30, 110, 125]) * MS
    sel, up = ultrawide_substream(t, t_uw, rate_hz=10)         # schedule 0, 100 -> nearest 30, 110
    assert sel.tolist() == [0, 1] and up.tolist() == [0, 0, 0, 0, 0, 0, 1]
    t_uw = np.array([0, 130]) * MS                            # schedule 0, 100 -> 0, 130; 130 > t[-1] -> trimmed
    sel, up = ultrawide_substream(t, t_uw, rate_hz=10)
    assert sel.tolist() == [0] and up.tolist() == [0] * 7


def test_ultrawide_drops_dedup():
    t = grid(61)
    t_uw = np.concatenate([np.arange(0, 10) * 33 * MS, np.arange(25, 31) * 33 * MS])   # 495 ms gap
    sel, up = ultrawide_substream(t, t_uw)
    _check_substream(t, t_uw, sel, up)
    assert len(np.unique(sel)) == len(sel)


def test_ultrawide_random_timelines_with_jitter_and_drops():
    rng = np.random.default_rng(5)
    for _ in range(30):
        n = int(rng.integers(20, 400))
        t = grid(n, t0=int(rng.integers(0, 10**9)))
        dt = rng.normal(1e9 / 30, 1.5 * MS, size=n // 2 + 20).clip(5 * MS)
        t_uw = (t[0] - int(rng.integers(0, 80 * MS)) + np.cumsum(dt)).astype(np.int64)
        keep = rng.random(len(t_uw)) > 0.1
        keep[0] = True
        t_uw = t_uw[keep]
        sel, up = ultrawide_substream(t, t_uw, rate_hz=10)
        _check_substream(t, t_uw, sel, up)


def test_downsample_local_matches_iphumi_formula():
    up = np.array([0, 0, 0, 1, 1, 2, 2, 2, 4, 4])              # frame 3 unused: maps to the first step using >= 3
    ds = downsample_local(up)
    n_uw = int(up[-1]) + 1
    assert ds.tolist() == np.searchsorted(up, np.arange(n_uw), side="left").tolist() == [0, 3, 5, 8, 8]
    assert np.all(up[ds[[0, 1, 2, 4]]] == [0, 1, 2, 4])


def _iphumi_compute_frame_indices():
    """iPhUMI align.py ``_compute_frame_indices`` compiled from its source (no iphumi import needed)."""
    if not IPHUMI_ALIGN.is_file():
        pytest.skip(f"iPhUMI checkout not found at {IPHUMI_ALIGN}")
    tree = ast.parse(IPHUMI_ALIGN.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_frame_indices")
    ns = {"np": np, "datetime_fromisoformat": lambda s: type("T", (), {"timestamp": lambda self: float(s)})()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(IPHUMI_ALIGN), "exec"), ns)
    return ns["_compute_frame_indices"]


def test_upsample_rule_equals_iphumi_compute_frame_indices():
    f = _iphumi_compute_frame_indices()
    rng = np.random.default_rng(11)
    for _ in range(20):
        frames = np.sort(rng.choice(10**6, size=int(rng.integers(2, 60)), replace=False)).astype(np.int64)
        q = np.sort(rng.integers(-1000, 10**6 + 1000, size=200)).astype(np.int64)
        ref = f([str(x / 1e9) for x in frames], q / 1e9)
        assert np.array_equal(latest_at_or_before(q, frames), ref)
