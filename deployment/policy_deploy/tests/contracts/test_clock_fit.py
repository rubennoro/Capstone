# adapted from rnoro/deployment:deployment/process_frames/tests/test_clock_map.py @ 91b178b
# changes: module clock_fit, names local/remote (t_remote_ns, to_local_ns, at_local_ns); added: to_remote_ns, the frozen
# snapshot, stats span/newest age, the one-way bias and refit rate, constructor arguments.
"""clock_fit (was process_frames.clock_map), local = PC/laptop, remote = Pi.

ClockMap: offset and drift with uneven Wi-Fi round trips, outlier rejection by round trip (STRUCTURE.md §7).
Pure and seeded: no sockets."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from policy_deploy.contracts.clock_fit import ClockMap, ClockSample, NotSynced, OneWayClockMap

S = 1_000_000_000
MS = 1_000_000
OFFSET = 412_337_815_000           # Pi minus PC, ns
T0 = 50 * S                        # PC time of the first exchange


def make_map(**kw):
    args = dict(window_s=60.0, best_fraction=0.3, drift_min_span_s=10.0, max_drift_ppm=500.0, max_sync_age_s=10.0)
    args.update(kw)
    return ClockMap(**args)


def pi_time(pc_ns, drift_ppm):
    return pc_ns + OFFSET + int(round(drift_ppm * 1e-6 * (pc_ns - T0)))


def exchanges(rng, n, drift_ppm=0.0, outliers=(), start=T0, interval=S):
    """Wi-Fi like: each direction 1 ms + exponential jitter, uneven between directions (0.3 ms there, 1.5 ms back).
    ``outliers``: indices whose one direction is delayed 50-300 ms."""
    out = []
    for i in range(n):
        t_send = start + i * interval
        there = 1 * MS + int(rng.exponential(0.3 * MS))
        back = 1 * MS + int(rng.exponential(1.5 * MS))
        if i in outliers:
            extra = int(rng.uniform(50, 300) * MS)
            if rng.random() < 0.5:
                there += extra
            else:
                back += extra
        t_pi = pi_time(t_send + there, drift_ppm)
        out.append(ClockSample(t_send_ns=t_send, t_remote_ns=t_pi, t_recv_ns=t_send + there + back))
    return out


def mapping_error_ns(cm, drift_ppm, at):
    """|to_pc(pi(t)) - t| for a frame captured at PC time t, mapped at its arrival t."""
    return abs(cm.to_local_ns(pi_time(at, drift_ppm), at_local_ns=at) - at)


def test_not_synced_before_the_first_sample():
    cm = make_map()
    with pytest.raises(NotSynced):
        cm.offset_ns(T0)
    assert not cm.is_fresh(T0)
    assert cm.stats(T0).synced is False and cm.stats(T0).offset_ns is None


def test_sample_properties():
    s = ClockSample(t_send_ns=100, t_remote_ns=1_000, t_recv_ns=110)
    assert s.rtt_ns == 10 and s.offset_ns == 1_000 - 105


def test_offset_with_uneven_round_trips():
    rng = np.random.default_rng(1)
    cm = make_map()
    samples = exchanges(rng, 60)
    for s in samples:
        assert cm.add_sample(s)
    min_rtt = min(s.rtt_ns for s in samples)
    end = samples[-1].t_recv_ns
    for at in (end - 30 * S, end, end + S):
        assert mapping_error_ns(cm, 0.0, at) < min_rtt / 2 + 0.5 * MS
    assert cm.stats(end).drift_ppm == pytest.approx(0.0, abs=5.0)


@pytest.mark.parametrize("drift", [50.0, -50.0])
def test_drift_recovery(drift):
    rng = np.random.default_rng(2)
    cm = make_map()
    samples = exchanges(rng, 60, drift_ppm=drift)
    for s in samples:
        cm.add_sample(s)
    end = samples[-1].t_recv_ns
    st = cm.stats(end)
    assert st.drift_ppm == pytest.approx(drift, abs=3.0)
    min_rtt = min(s.rtt_ns for s in samples)
    for at in (end - 20 * S, end, end + S):
        assert mapping_error_ns(cm, drift, at) < min_rtt / 2 + 0.5 * MS


def test_drift_is_zero_before_drift_min_span():
    rng = np.random.default_rng(3)
    cm = make_map()
    samples = exchanges(rng, 8, drift_ppm=50.0)            # 7 s of samples: less than 10 s
    for s in samples:
        cm.add_sample(s)
    st = cm.stats(samples[-1].t_recv_ns)
    assert st.drift_ppm == 0.0
    best = sorted(samples, key=lambda s: s.rtt_ns)[:max(1, int(len(samples) * 0.3))]
    mean = np.mean([s.offset_ns for s in best])
    assert cm.offset_ns(samples[-1].t_recv_ns) == pytest.approx(mean, abs=1)


def test_outliers_by_round_trip_never_move_offset_or_drift():
    rng = np.random.default_rng(4)
    n = 60
    lone = set(rng.choice(n, size=3, replace=False).tolist())          # 5 % of exchanges
    burst = {30, 31, 32, 33}                                           # and a burst in a row
    clean = exchanges(np.random.default_rng(5), n, drift_ppm=20.0)
    dirty = exchanges(np.random.default_rng(5), n, drift_ppm=20.0, outliers=lone | burst)
    a, b = make_map(), make_map()
    for s in clean:
        a.add_sample(s)
    for s in dirty:
        b.add_sample(s)
    outlier_rtt = min(dirty[i].rtt_ns for i in lone | burst)
    best_b = sorted(dirty, key=lambda s: s.rtt_ns)[:int(n * 0.3)]
    assert max(s.rtt_ns for s in best_b) < outlier_rtt                 # never among the best samples
    end = dirty[-1].t_recv_ns
    assert abs(a.offset_ns(end) - b.offset_ns(end)) < 0.3 * MS
    assert b.stats(end).drift_ppm == pytest.approx(a.stats(end).drift_ppm, abs=2.0)
    assert mapping_error_ns(b, 20.0, end) < min(s.rtt_ns for s in dirty) / 2 + 0.5 * MS


def test_drift_above_max_is_noise():
    rng = np.random.default_rng(6)
    cm = make_map(max_drift_ppm=500.0)
    samples = exchanges(rng, 30, drift_ppm=2000.0)
    for s in samples:
        cm.add_sample(s)
    st = cm.stats(samples[-1].t_recv_ns)
    assert st.drift_ppm == 0.0
    best = sorted(samples, key=lambda s: s.rtt_ns)[:int(len(samples) * 0.3)]
    assert cm.offset_ns(0) == pytest.approx(np.mean([s.offset_ns for s in best]), abs=1)


def test_old_samples_leave_the_window():
    cm = make_map(window_s=10.0)
    for s in exchanges(np.random.default_rng(7), 30):
        cm.add_sample(s)
    assert cm.stats(T0 + 30 * S).samples <= 11


def test_impossible_sample_is_ignored():
    cm = make_map()
    assert cm.add_sample(ClockSample(t_send_ns=100, t_remote_ns=5, t_recv_ns=99)) is False
    assert cm.stats(100).samples == 0


def test_is_fresh_turns_false_after_max_sync_age():
    cm = make_map(max_sync_age_s=10.0)
    cm.add_sample(ClockSample(t_send_ns=T0, t_remote_ns=T0 + OFFSET, t_recv_ns=T0 + 2 * MS))
    assert cm.is_fresh(T0 + 10 * S)
    assert not cm.is_fresh(T0 + 2 * MS + 10 * S + 1)
    assert cm.stats(T0 + 20 * S).synced is False
    assert cm.offset_ns(T0) == OFFSET - MS                    # still computable, just not fresh


def test_thread_safety_one_adds_one_maps():
    cm = make_map()
    samples = exchanges(np.random.default_rng(8), 2000, interval=S // 20)
    cm.add_sample(samples[0])
    errors = []
    done = threading.Event()

    def mapper():
        while not done.is_set():
            try:
                cm.to_local_ns(T0 + OFFSET, T0)
                cm.stats(T0)
            except Exception as e:      # pragma: no cover - reported below
                errors.append(e)
                return

    t = threading.Thread(target=mapper)
    t.start()
    for s in samples[1:]:
        cm.add_sample(s)
    done.set()
    t.join()
    assert not errors


# ---------------------------------------------------------------------------------------------------------- added


def test_to_remote_ns_is_the_inverse_of_to_local_ns():
    rng = np.random.default_rng(9)
    cm = make_map()
    for s in exchanges(rng, 60, drift_ppm=30.0):
        cm.add_sample(s)
    for t in (T0, T0 + 30 * S, T0 + 70 * S):
        remote = cm.to_remote_ns(t)
        assert remote == t + cm.offset_ns(t)
        assert abs(cm.to_local_ns(remote, at_local_ns=t) - t) <= 1
        assert abs(remote - pi_time(t, 30.0)) < 2 * MS


def test_snapshot_is_frozen_across_a_refit():
    cm = make_map()
    samples = exchanges(np.random.default_rng(10), 20)
    for s in samples[:10]:
        cm.add_sample(s)
    snap = cm.snapshot()
    before = snap.offset_ns(T0)
    cm.add_sample(ClockSample(t_send_ns=T0 + 30 * S, t_remote_ns=T0 + 30 * S + OFFSET + 5 * MS,
                              t_recv_ns=T0 + 30 * S))                 # a 0-RTT sample 5 ms off: moves the fit
    assert snap.offset_ns(T0) == before
    assert cm.snapshot().offset_ns(T0) != before


def test_stats_report_span_and_newest_age():
    cm = make_map()
    samples = exchanges(np.random.default_rng(11), 5)
    for s in samples:
        cm.add_sample(s)
    st = cm.stats(samples[-1].t_recv_ns + 2 * S)
    assert st.span_s == pytest.approx((samples[-1].t_recv_ns - samples[0].t_recv_ns) / S)
    assert st.newest_age_s == pytest.approx(2.0) and st.last_sync_age_s == st.newest_age_s
    assert cm.sample_count() == 5 and cm.sample_span_ns() == samples[-1].t_recv_ns - samples[0].t_recv_ns
    assert cm.newest_ns() == samples[-1].t_recv_ns


def test_one_way_bias_is_the_minimum_latency_and_refits_are_rate_limited():
    ow = OneWayClockMap(window_s=60.0, best_fraction=0.3, drift_min_span_s=10.0, max_drift_ppm=500.0,
                        max_sync_age_s=10.0, refit_s=0.25)
    rng = np.random.default_rng(12)
    min_latency = 3 * MS
    for i in range(30 * 30):                                         # 30 s at 30 Hz
        t_send_local = T0 + i * (S // 30)
        latency = min_latency + int(rng.exponential(2 * MS))
        ow.add_one_way(pi_time(t_send_local, 0.0), t_send_local + latency)
    end = T0 + 30 * S
    # the fit says remote - local = OFFSET - (about the minimum latency)
    assert abs(ow.offset_ns(end) - (OFFSET - min_latency)) < 0.5 * MS
    assert ow.stats(end).rtt_min_ms is None
    # rate limit: a sample 0.1 s after the last refit does not move the fit
    snap = ow.snapshot()
    ow.add_one_way(pi_time(end, 0.0) + 50 * MS, end + 10)
    assert ow.snapshot() is snap


def test_constructor_arguments_are_used():
    cm = make_map(window_s=5.0, max_sync_age_s=1.0)
    for s in exchanges(np.random.default_rng(13), 20):
        cm.add_sample(s)
    assert cm.sample_count() <= 6
    assert not cm.is_fresh(T0 + 30 * S)
