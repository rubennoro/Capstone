"""FrameBuffer: keep_s, snapshot order and independence, read-only arrays, thread safety (STRUCTURE.md §5.8)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from process_frames.frame_buffer import BufferedFrame, FrameBuffer

MS = 1_000_000


def bf(t_ms, seq=0):
    return BufferedFrame(camera="main", seq=seq, t_pc_ns=int(t_ms * MS), t_pi_mid_ns=0, arrival_ns=int(t_ms * MS),
                         rgb=np.zeros((2, 2, 3), np.uint8))


def test_frames_older_than_keep_s_leave():
    b = FrameBuffer("main", keep_s=1.0)
    for i in range(61):
        b.push(bf(i * 33.3, seq=i))
    snap = b.snapshot()
    assert snap[-1].seq == 60
    assert snap[0].t_pc_ns >= snap[-1].t_pc_ns - 1000 * MS
    assert len(b) == 31 and b.newest().seq == 60


def test_snapshot_order_and_independence():
    b = FrameBuffer("main", keep_s=1.0)
    for i in range(5):
        b.push(bf(i * 10, seq=i))
    snap = b.snapshot()
    b.push(bf(100, seq=5))
    assert [f.seq for f in snap] == [0, 1, 2, 3, 4]
    assert [f.seq for f in b.snapshot()] == [0, 1, 2, 3, 4, 5]


def test_arrays_are_read_only():
    f = bf(0)
    assert not f.rgb.flags.writeable
    with pytest.raises(ValueError):
        f.rgb[0, 0, 0] = 1


def test_non_increasing_time_is_refused():
    b = FrameBuffer("main", keep_s=1.0)
    b.push(bf(10))
    with pytest.raises(ValueError):
        b.push(bf(10))
    with pytest.raises(ValueError):
        b.push(bf(5))
    assert FrameBuffer("main", 1.0).newest() is None


def test_one_thread_pushes_another_snapshots():
    b = FrameBuffer("main", keep_s=0.05)
    errors = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            s = b.snapshot()
            if any(x.t_pc_ns >= y.t_pc_ns for x, y in zip(s, s[1:])):
                errors.append("not increasing")

    t = threading.Thread(target=reader)
    t.start()
    for i in range(20000):
        b.push(bf(i * 0.1, seq=i))
    done.set()
    t.join()
    assert not errors
