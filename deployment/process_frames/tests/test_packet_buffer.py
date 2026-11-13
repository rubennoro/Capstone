"""PacketBuffer (STRUCTURE.md §5.9): old packets leave, snapshots are independent, images read-only."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from process_frames.packet_buffer import BufferedPacket, PacketBuffer

MS = 1_000_000


def bp(seq, t_ms, uw=True):
    img = np.zeros((4, 4, 3), np.uint8)
    return BufferedPacket(packet_seq=seq, t_pc_ns=int(t_ms * MS), t_pi_mid_ns=0, arrival_ns=int(t_ms * MS),
                          main_seq=seq, main_t_pi_ns=1_000 + seq, main_rgb=img, pair_seq=seq if uw else -1,
                          pair_t_pi_ns=990 + seq if uw else None, ultrawide_rgb=img.copy() if uw else None)


def test_packets_older_than_keep_s_leave():
    b = PacketBuffer(keep_s=0.1)
    for i in range(10):
        b.push(bp(i, i * 33.3))
    assert [p.packet_seq for p in b.snapshot()] == [6, 7, 8, 9]          # >= 299.7 - 100 ms
    assert b.newest().packet_seq == 9 and len(b) == 4


def test_snapshot_is_independent_and_images_read_only():
    b = PacketBuffer(keep_s=1.0)
    b.push(bp(0, 0))
    snap = b.snapshot()
    b.push(bp(1, 33))
    assert len(snap) == 1 and len(b.snapshot()) == 2
    assert not snap[0].main_rgb.flags.writeable and not snap[0].ultrawide_rgb.flags.writeable


def test_image_and_pair_gap():
    p = bp(3, 10)
    assert p.image("main") is p.main_rgb and p.image("ultrawide") is p.ultrawide_rgb and p.pair_gap_ns == 10
    q = bp(4, 20, uw=False)
    assert q.image("ultrawide") is None and q.pair_gap_ns is None
    with pytest.raises(ValueError):
        q.image("left")


def test_non_increasing_time_refused():
    b = PacketBuffer(keep_s=1.0)
    b.push(bp(0, 10))
    with pytest.raises(ValueError):
        b.push(bp(1, 10))


def test_one_thread_pushes_while_another_snapshots():
    b = PacketBuffer(keep_s=0.2)
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            times = [p.t_pc_ns for p in b.snapshot()]
            if times != sorted(times) or len(set(times)) != len(times):
                errors.append(times)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(3000):
        b.push(bp(i, i * 1.0))
    stop.set()
    t.join()
    assert not errors
