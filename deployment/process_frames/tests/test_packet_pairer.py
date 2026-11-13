"""PacketPairer: the Pi's pairing rule (PI_PAYLOAD.md §5 rule 3) and every ultrawide frame sent exactly once."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import raw_packets
from process_frames.packet_pairer import PAIR_AHEAD_NS, PAIR_MAX_AGE_NS, PacketPairer

MS = 1_000_000
P = 33_333_333


def run(main_t, uw_t):
    """Feed the pairer as the Pi does (ultrawide at capture, each main packet 5 ms after its capture): per packet,
    (the ultrawide frame_seqs it carries, pair_seq)."""
    events = sorted([(t, 0, j) for j, t in enumerate(uw_t)] + [(t + PAIR_AHEAD_NS, 1, i) for i, t in enumerate(main_t)])
    pairer = PacketPairer()
    out = []
    for _, is_main, k in events:
        if is_main:
            items, pair = pairer.take(main_t[k])
            out.append((items, pair))
        else:
            pairer.add_ultrawide(k, uw_t[k], k)
    return out


def brute_force_pair(main_t, uw_t, t):
    """The rule written plainly: latest ultrawide with capture <= t + 5 ms, at most 50 ms old, else -1."""
    ok = [j for j, u in enumerate(uw_t) if u <= t + PAIR_AHEAD_NS and t - u <= PAIR_MAX_AGE_NS]
    return max(ok, key=lambda j: uw_t[j]) if ok else -1


def times(n, start, period=P, jitter_ns=0, seed=0):
    rng = np.random.default_rng(seed)
    return [int(start + i * period + (rng.integers(-jitter_ns, jitter_ns + 1) if jitter_ns else 0)) for i in range(n)]


def test_normal_30_30_one_ultrawide_per_packet_paired_with_it():
    main_t = times(30, 1_000 * MS)
    uw_t = times(30, 1_000 * MS - 10 * MS)          # ultrawide 10 ms before main
    out = run(main_t, uw_t)
    assert [items for items, _ in out] == [[i] for i in range(30)]
    assert [pair for _, pair in out] == list(range(30))


@pytest.mark.parametrize("offset_ms", [-33, -25, -10, -5, 0, 3, 5, 6, 15, 30])
def test_phase_offsets_follow_the_rule(offset_ms):
    main_t = times(40, 1_000 * MS, jitter_ns=2 * MS, seed=offset_ms + 50)
    uw_t = times(40, 1_000 * MS + offset_ms * MS, jitter_ns=2 * MS, seed=offset_ms + 99)
    out = run(main_t, uw_t)
    sent = [j for items, _ in out for j in items]
    assert sent == sorted(sent) and len(sent) == len(set(sent))          # each exactly once, in capture order
    for i, (_, pair) in enumerate(out):
        assert pair == brute_force_pair(main_t, uw_t, main_t[i]), i
        if pair >= 0:
            gap = main_t[i] - uw_t[pair]
            assert -PAIR_AHEAD_NS <= gap <= P + 5 * MS                    # normally 0-33 ms apart (+5 ms)


def test_zero_or_two_ultrawide_frames_in_a_packet():
    main_t = [1_000 * MS, 1_033 * MS, 1_066 * MS]
    uw_t = [990 * MS, 1_020 * MS, 1_031 * MS, 1_060 * MS]   # two between main 0 and main 1, ...
    out = run(main_t, uw_t)
    assert [items for items, _ in out] == [[0], [1, 2], [3]]
    assert [pair for _, pair in out] == [0, 2, 3]
    main_t = [1_000 * MS, 1_033 * MS, 1_066 * MS]
    uw_t = [990 * MS, 1_060 * MS]                            # nothing new for main 1
    out = run(main_t, uw_t)
    assert [items for items, _ in out] == [[0], [], [1]]


def test_a_pair_from_an_earlier_packet():
    main_t = [1_000 * MS, 1_033 * MS]
    uw_t = [1_004 * MS, 1_040 * MS]       # frame 0 (4 ms after main 0) goes with packet 0; main 1's pair is still 0
    out = run(main_t, uw_t)
    assert out[0] == ([0], 0)
    assert out[1] == ([], 0)              # frame 1 is 7 ms after main 1: too late, it waits for packet 2


def test_no_pair_is_minus_one():
    out = run([1_000 * MS], [])
    assert out == [([], -1)]
    out = run([1_000 * MS, 1_100 * MS], [990 * MS])          # 110 ms old for main 1
    assert out[1] == ([], -1)
    out = run([1_000 * MS], [1_006 * MS])                    # 6 ms after main: later than + 5 ms
    assert out[0] == ([], -1)
    out = run([1_000 * MS], [1_005 * MS])                    # exactly + 5 ms: still the pair
    assert out[0] == ([0], 0)
    out = run([1_050 * MS], [1_000 * MS])                    # exactly 50 ms old: still the pair
    assert out[0] == ([0], 0)


def test_frames_out_of_order_are_refused():
    p = PacketPairer()
    p.add_ultrawide(3, 100, "a")
    with pytest.raises(ValueError):
        p.add_ultrawide(3, 200, "b")
    with pytest.raises(ValueError):
        p.add_ultrawide(4, 100, "b")


def test_raw_packets_helper_follows_the_rule():
    pk = raw_packets(times(10, 1_000 * MS), times(10, 1_000 * MS - 20 * MS), skip_uw=(4,))
    # main 4's newest ultrawide is frame 3, 53 ms older: beyond 50 ms, so no pair
    assert [p.header.pair_seq for p in pk] == [0, 1, 2, 3, -1, 5, 6, 7, 8, 9]
    assert [f.info.frame_seq for p in pk for f in p.ultrawide] == [0, 1, 2, 3, 5, 6, 7, 8, 9]
