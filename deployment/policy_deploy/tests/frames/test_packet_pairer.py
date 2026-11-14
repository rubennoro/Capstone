# replaces rnoro/deployment:deployment/process_frames/tests/test_packet_pairer.py @ 91b178b (rewritten for the
# nearest-frame pairing, STRUCTURE.md 12.2)
"""PacketPairer: nearest-frame pairing as the live Pi does it, and every ultrawide frame sent exactly once."""

from __future__ import annotations

import numpy as np
import pytest
from frames_helpers import HALF_PERIOD_NS, raw_packets

from policy_deploy.frames.packet_pairer import PacketPairer

MS = 1_000_000
P = 33_333_333


def run(main_t, uw_t, lookahead_ns=HALF_PERIOD_NS):
    """Feed the pairer as the fake Pi does (ultrawide at capture, each main packet half an ultrawide period after
    its capture): per packet, (the ultrawide frame_seqs it carries, pair_seq)."""
    events = sorted([(t, 0, j) for j, t in enumerate(uw_t)] + [(t + lookahead_ns, 1, i) for i, t in enumerate(main_t)])
    pairer = PacketPairer()
    out = []
    for _, is_main, k in events:
        if is_main:
            out.append(pairer.take(main_t[k]))
        else:
            pairer.add_ultrawide(k, uw_t[k], k)
    return out


def brute_force_pair(uw_t, added, t):
    """The rule written plainly: among the added frames, the one captured nearest t; a tie to the older; else -1."""
    if not added:
        return -1
    return min(added, key=lambda j: (abs(uw_t[j] - t), uw_t[j]))


def times(n, start, period=P, jitter_ns=0, seed=0):
    rng = np.random.default_rng(seed)
    return [int(start + i * period + (rng.integers(-jitter_ns, jitter_ns + 1) if jitter_ns else 0)) for i in range(n)]


def test_the_nearer_of_the_frames_before_and_after_wins():
    p = PacketPairer()
    p.add_ultrawide(0, 1_000 * MS, "a")              # 20 ms before the main frame
    p.add_ultrawide(1, 1_033 * MS, "b")              # 13 ms after it
    assert p.take(1_020 * MS) == (["a", "b"], 1)
    p.add_ultrawide(2, 1_066 * MS, "c")
    assert p.take(1_045 * MS) == (["c"], 1)          # 12 ms after frame 1, 21 ms before frame 2


def test_a_tie_goes_to_the_older_frame():
    p = PacketPairer()
    p.add_ultrawide(4, 100, "x")
    p.add_ultrawide(5, 300, "y")
    assert p.take(200)[1] == 4


def test_the_live_case_a_later_frame_within_half_a_period_wins():
    """PI_PAYLOAD.md 0.6: the ultrawide mid-exposure 16.3 ms after the main frame's is its pair."""
    main_t = times(30, 1_000 * MS)
    uw_t = [t + 16 * MS for t in main_t]
    out = run(main_t, uw_t)
    assert [pair for _, pair in out] == list(range(30))


def test_no_age_cut_a_stalled_stream_keeps_pairing_its_last_frame():
    p = PacketPairer()
    p.add_ultrawide(7, 1_000 * MS, "z")
    assert p.take(1_000 * MS)[1] == 7
    for k in range(1, 100):                          # 3.3 s without a new ultrawide frame
        assert p.take(1_000 * MS + k * P) == ([], 7)


def test_no_pair_only_before_the_first_frame():
    p = PacketPairer()
    assert p.take(5) == ([], -1)
    p.add_ultrawide(0, 1_000_000_000, "f")
    assert p.take(5)[1] == 0                          # even 1 s away: it is the only one


@pytest.mark.parametrize("shift_ms,jitter_ms,seed", [(-30, 0, 0), (-16, 3, 1), (-5, 3, 2), (0, 0, 3), (11, 3, 4),
                                                     (16, 2, 5), (25, 4, 6), (40, 3, 7)])
def test_equals_the_brute_force_rule_and_sends_every_frame_once(shift_ms, jitter_ms, seed):
    main_t = times(60, 2_000 * MS, jitter_ns=jitter_ms * MS, seed=seed)
    uw_t = sorted(times(60, 2_000 * MS + shift_ms * MS, jitter_ns=jitter_ms * MS, seed=seed + 100))
    out = run(main_t, uw_t)
    sent: list[int] = []
    for i, (items, pair) in enumerate(out):
        sent += items
        assert pair == brute_force_pair(uw_t, sent, main_t[i]), i
    assert sent == sorted(set(sent))                   # rule 2: each frame once, in capture order
    assert sent == list(range(len(sent)))


def test_frames_must_be_added_in_capture_order():
    p = PacketPairer()
    p.add_ultrawide(3, 1000, None)
    with pytest.raises(ValueError):
        p.add_ultrawide(3, 2000, None)
    with pytest.raises(ValueError):
        p.add_ultrawide(4, 1000, None)
    assert p.pending == 1


def test_raw_packets_helper_follows_the_same_rule():
    main_t = times(10, 5_000 * MS)
    uw_t = [t + 16 * MS for t in main_t]
    pk = raw_packets(main_t, uw_t)
    assert [p.header.pair_seq for p in pk] == list(range(10))
    assert [f.info.frame_seq for p in pk for f in p.ultrawide] == list(range(10))


def test_fake_pi_messages_have_the_nearest_pair_and_the_documented_skew(pi_episode):
    """FakePiSender pi_v1 messages: the nearest pair and skew_ns = main.t_mid - ultrawide.t_mid."""
    import time

    import zmq

    from policy_deploy.contracts import PiStreamPacket
    from policy_deploy.frames.fake_pi_sender import FakePiSender
    from policy_deploy.frames.pi_recording import PiRecording

    rec = PiRecording.load(pi_episode, max_seconds=10.0, jpeg_quality=90)
    fake = FakePiSender(rec, "tcp://127.0.0.1:*", pi_clock_offset_ns=0, drift_ppm=0.0, send_hwm=30)
    fake.start()
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(fake.frames_endpoint)
    msgs = []
    deadline = time.monotonic() + 10
    try:
        while len(msgs) < 8 and time.monotonic() < deadline:
            if sub.poll(50):
                msgs.append(PiStreamPacket.from_parts(sub.recv_multipart()))
    finally:
        sub.close()
        fake.stop()
    assert len(msgs) == 8
    uw = rec.cameras["ultrawide"].frames
    t0 = rec.start_ns
    for m in msgs:
        h = m.header
        local_main = h.main.t_sensor_ns - fake.pi_clock_ns(fake._start_ns) + t0        # back to recorded time
        cands = [f.info.t_pi_ns + (h.main.frame_no // len(uw)) * rec.duration_ns for f in uw]
        nearest = min(range(len(cands)), key=lambda j: abs(cands[j] - local_main))
        assert h.ultrawide.frame_no % len(uw) == nearest
        assert h.skew_ns == h.main.t_mid_ns - h.ultrawide.t_mid_ns
