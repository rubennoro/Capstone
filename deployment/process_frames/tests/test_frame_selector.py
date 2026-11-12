"""FrameSelector: picking at 30 and 60 FPS, jitter, drops, ties, failures (STRUCTURE.md §6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from process_frames.frame_buffer import BufferedFrame
from process_frames.frame_selector import FrameSelector

MS = 1_000_000
TOL = 25 * MS


def frames(times_ns):
    rgb = np.zeros((1, 1, 3), np.uint8)
    return [BufferedFrame("main", i, int(t), int(t), int(t), rgb) for i, t in enumerate(times_ns)]


def stream(fps, n=30, jitter_ms=0.0, drop=(), seed=0, start=10_000 * MS):
    rng = np.random.default_rng(seed)
    period = 1e9 / fps
    t = [start + i * period + rng.uniform(-jitter_ms, jitter_ms) * MS for i in range(n) if i not in drop]
    return frames(sorted(int(x) for x in t))


def nearest(fr, target):
    d = [abs(f.t_pc_ns - target) for f in fr]
    return fr[int(np.argmin(d))]


def test_targets():
    t = FrameSelector.targets(1000, 50, 2)
    assert t.dtype == np.int64 and t.tolist() == [950, 1000]
    assert FrameSelector.targets(1000, 100, 3).tolist() == [800, 900, 1000]


@pytest.mark.parametrize("fps", [30, 60])
@pytest.mark.parametrize("jitter", [0.0, 3.0])
@pytest.mark.parametrize("spacing_ms", [50, 100])
def test_nearest_frames_are_picked(fps, jitter, spacing_ms):
    fr = stream(fps, jitter_ms=jitter, seed=fps)
    anchor = fr[-1].t_pc_ns
    sel = FrameSelector(TOL).pick(fr, anchor, spacing_ms * MS, 2)
    assert sel.ok and sel.reason == ""
    exp = [nearest(fr, t) for t in FrameSelector.targets(anchor, spacing_ms * MS, 2)]
    assert [f.seq for f in sel.frames] == [f.seq for f in exp]
    assert sel.frames[-1] is fr[-1]
    for f, t in zip(sel.frames, FrameSelector.targets(anchor, spacing_ms * MS, 2)):
        assert abs(f.t_pc_ns - t) <= TOL


def test_ties_go_to_the_earlier_frame():
    fr = frames([0, 40 * MS, 60 * MS, 100 * MS])         # target 50 ms: 40 and 60 are both 10 ms away
    sel = FrameSelector(TOL).pick(fr, 100 * MS, 50 * MS, 2)
    assert [f.t_pc_ns for f in sel.frames] == [40 * MS, 100 * MS]


def test_a_drop_next_to_a_target_fails_or_passes_by_tolerance():
    # 30 FPS with frames around anchor-100 ms missing: nearest is 33 ms away
    fr = stream(30, n=20, drop=(15, 16))
    anchor = fr[-1].t_pc_ns
    target = anchor - 100 * MS
    d = min(abs(f.t_pc_ns - target) for f in fr)
    assert d > TOL
    sel = FrameSelector(TOL).pick(fr, anchor, 100 * MS, 2)
    assert not sel.ok and "tolerance 25.0 ms" in sel.reason and sel.frames is None
    assert FrameSelector(int(d) + 1).pick(fr, anchor, 100 * MS, 2).ok      # a close-enough neighbour passes
    # a single dropped frame at 30 FPS next to the 50 ms target: neighbour 16.7 ms away, passes
    fr = stream(30, n=20, drop=(18,))
    assert FrameSelector(TOL).pick(fr, fr[-1].t_pc_ns, 50 * MS, 2).ok


def test_empty_single_and_same_frame_twice_fail():
    sel = FrameSelector(TOL)
    assert not sel.pick([], 0, 50 * MS, 2).ok
    one = frames([1000 * MS])
    r = sel.pick(one, 1000 * MS, 50 * MS, 2)
    assert not r.ok
    r = FrameSelector(60 * MS).pick(one, 1000 * MS, 50 * MS, 2)
    assert not r.ok and "same frame" in r.reason
