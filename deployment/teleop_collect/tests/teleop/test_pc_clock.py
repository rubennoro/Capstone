# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager/test_pc_clock.py @ 2ffe641
# changes: imports; test_get_state_replies_never_enter_the_map uses TeleopHarness
"""The laptop <-> PC clock map (STRUCTURE.md 8.3, 12.2; PLAN 5.6): both estimators against a fake PC clock offset
+1000 s and drifting +-30 ppm. Pure and seeded: no sockets, no sleeping."""
from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest
from teleop_helpers import TeleopHarness

from teleop_collect.contracts import NotSynced
from teleop_collect.teleop.pc_clock import HelloSample, OneWayPcClock, RoundTripPcClock, make_pc_clock

S = 1_000_000_000
MS = 1_000_000
OFFSET = 1000 * S
T0 = 50_000 * S


class FakeLaptop:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def pc_time(t, drift_ppm):
    return t + OFFSET + int(round(drift_ppm * 1e-6 * (t - T0)))


def hellos(rng, n, drift_ppm, period_s=0.5, start=T0, uneven=False, outliers=(), base_ms=1.0):
    """Round trips: each direction base + jitter (uneven: 0.3 ms there, 1.5 ms back); outliers delayed 50-300 ms."""
    out = []
    for i in range(n):
        t_send = start + int(i * period_s * S)
        if uneven:
            up = int(base_ms * MS) + int(rng.exponential(0.3 * MS))
            down = int(base_ms * MS) + int(rng.exponential(1.5 * MS))
        else:
            up = down = int(base_ms * MS)
        if i in outliers:
            extra = int(rng.uniform(50, 300) * MS)
            if rng.random() < 0.5:
                up += extra
            else:
                down += extra
        out.append(HelloSample(t_send, pc_time(t_send + up, drift_ppm), t_send + up + down))
    return out


def feed(clock, samples):
    for s in samples:
        clock.add_hello(s)


def error_ns(pc_clock, t, drift_ppm):
    return abs(pc_clock.snapshot().to_pc_ns(t) - pc_time(t, drift_ppm))


@pytest.fixture
def cs(settings):
    return settings.clock


# ---- round_trip ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("drift", [30.0, -30.0])
def test_round_trip_symmetric_delays(cs, drift):
    laptop = FakeLaptop()
    c = RoundTripPcClock(cs, laptop)
    samples = hellos(np.random.default_rng(1), 41, drift)
    feed(c, samples)
    end = samples[-1].t_recv_ns
    for at in (end - 3 * S, end, end + S):
        assert error_ns(c, at, drift) < 0.5 * MS


@pytest.mark.parametrize("drift", [30.0, -30.0])
def test_round_trip_uneven_delays_within_half_the_minimum_rtt(cs, drift):
    c = RoundTripPcClock(cs, FakeLaptop())
    samples = hellos(np.random.default_rng(2), 41, drift, uneven=True)
    feed(c, samples)
    min_rtt = min(s.rtt_ns for s in samples)
    end = samples[-1].t_recv_ns
    for at in (end - 3 * S, end, end + S):
        assert error_ns(c, at, drift) < min_rtt / 2 + 0.5 * MS


def test_round_trip_outliers_and_a_burst_never_move_the_fit(cs):
    n = 41
    rng = np.random.default_rng(3)
    lone = set(rng.choice(n, size=2, replace=False).tolist())          # 5 %
    burst = {30, 31, 32, 33}
    clean, dirty = RoundTripPcClock(cs, FakeLaptop()), RoundTripPcClock(cs, FakeLaptop())
    feed(clean, hellos(np.random.default_rng(4), n, 30.0, uneven=True))
    dirty_samples = hellos(np.random.default_rng(4), n, 30.0, uneven=True, outliers=lone | burst)
    feed(dirty, dirty_samples)
    end = dirty_samples[-1].t_recv_ns
    assert abs(clean.snapshot().to_pc_ns(end) - dirty.snapshot().to_pc_ns(end)) < 0.3 * MS
    assert error_ns(dirty, end, 30.0) < min(s.rtt_ns for s in dirty_samples) / 2 + 0.5 * MS


def test_round_trip_ignores_c2(cs):
    c = RoundTripPcClock(cs, FakeLaptop())
    for i in range(100):
        c.add_state(pc_time(T0 + i * 10 * MS, 0.0), T0 + i * 10 * MS + MS)
    assert not c.has_fit()
    with pytest.raises(NotSynced):
        c.snapshot()


# ---- one_way ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("drift", [30.0, -30.0])
def test_one_way_bias_is_the_minimum_delay(cs, drift):
    laptop = FakeLaptop()
    c = OneWayPcClock(cs, laptop)
    rng = np.random.default_rng(5)
    min_delay = 1.5 * MS
    for i in range(1000):                                             # 10 s of C2 at 100 Hz
        t = T0 + i * 10 * MS
        c.add_state(pc_time(t, drift), t + int(min_delay) + int(rng.exponential(1.0 * MS)))
    end = T0 + 10 * S
    early = pc_time(end, drift) - c.snapshot().to_pc_ns(end)          # PC times come out early ...
    assert abs(early - min_delay) < 0.5 * MS                          # ... by the minimum delay


@pytest.mark.parametrize("drift", [30.0, -30.0])
def test_one_way_symmetric_error_small(cs, drift):
    c = OneWayPcClock(cs, FakeLaptop())
    for i in range(1000):
        t = T0 + i * 10 * MS
        c.add_state(pc_time(t, drift), t + MS)                        # a constant 1 ms delay
    end = T0 + 10 * S
    assert abs(pc_time(end, drift) - MS - c.snapshot().to_pc_ns(end)) < 0.5 * MS


def test_one_way_outliers_never_move_the_fit(cs):
    clean, dirty = OneWayPcClock(cs, FakeLaptop()), OneWayPcClock(cs, FakeLaptop())
    rng = np.random.default_rng(6)
    for i in range(1000):
        t = T0 + i * 10 * MS
        clean.add_state(pc_time(t, 0.0), t + MS)
        late = int(rng.uniform(50, 300) * MS) if i < 900 and (rng.random() < 0.05 or 600 <= i < 620) else 0
        dirty.add_state(pc_time(t, 0.0), t + MS + late)
    end = T0 + 10 * S
    assert abs(clean.snapshot().to_pc_ns(end) - dirty.snapshot().to_pc_ns(end)) < 0.3 * MS


def test_repeated_t_ns_never_become_one_way_samples(cs):
    c = OneWayPcClock(cs, FakeLaptop())
    c.add_state(pc_time(T0, 0.0), T0 + MS)
    for i in range(1, 50):                                            # the PC repeats its t_ns (a fault, a homing)
        c.add_state(pc_time(T0, 0.0), T0 + i * 10 * MS)
    assert c.stats(T0 + S).samples == 1
    assert c.snapshot().to_pc_ns(T0) == pc_time(T0, 0.0) - MS        # the late repeats did not pull the fit
    c.add_state(pc_time(T0, 0.0) - 5, T0 + S)                         # going back: not a sample either
    assert c.stats(T0 + S).samples == 1


def test_one_way_keeps_hellos_for_the_checks_but_never_fits_them(cs):
    c = OneWayPcClock(cs, FakeLaptop())
    feed(c, hellos(np.random.default_rng(7), 10, 0.0))
    assert len(c.hellos()) == 10 and not c.has_fit()


def test_get_state_replies_never_enter_the_map(settings):
    """Start-up step 6 asks get_state; only hellos (round_trip) are samples."""
    h = TeleopHarness(settings)
    h.start()
    assert "get_state" in h.pc.calls
    assert h.pc_clock.stats(h.clock()).samples <= h.count("hello")
    assert len(h.pc_clock.hellos()) == h.count("hello")


# ---- the guards ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("estimator", ["round_trip", "one_way"])
def test_ready_needs_2_s_and_10_samples_and_a_fresh_one(cs, estimator):
    laptop = FakeLaptop()
    c = make_pc_clock(estimator, cs, laptop)

    def add(t):
        if estimator == "round_trip":
            c.add_hello(HelloSample(t - MS, pc_time(t, 0.0), t + MS))
        else:
            c.add_state(pc_time(t, 0.0), t + MS)

    for i in range(9):                                                # 9 samples over 4 s: too few
        add(T0 + i * 500 * MS)
    assert not c.ready(T0 + 4 * S + MS)
    assert "need 10" in c.not_ready_reason(T0 + 4 * S + MS)
    c2 = make_pc_clock(estimator, cs, laptop)
    c, c_full = c2, c
    for i in range(96):                                               # 96 samples over 1.9 s: too short
        add(T0 + i * 20 * MS)
    assert not c.ready(T0 + 1900 * MS)
    for i in range(96, 102):                                          # now over 2 s
        add(T0 + i * 20 * MS)
    newest = T0 + 101 * 20 * MS
    assert c.ready(newest + MS)
    assert c.ready(newest + 5 * S)                                    # max_sample_age_s 5.0
    assert not c.ready(newest + 5 * S + 2 * MS)
    assert "old" in c.not_ready_reason(newest + 6 * S)
    assert not make_pc_clock(estimator, cs, laptop).ready(T0)
    del c_full


def test_drift_above_the_cap_is_noise(cs):
    cs = dataclasses.replace(cs, drift_min_span_s=1.0)       # equal round trips: the best samples span ~3 s
    c = RoundTripPcClock(cs, FakeLaptop())
    samples = hellos(np.random.default_rng(8), 41, 2000.0)           # 2000 ppm: above max_drift_ppm 200
    feed(c, samples)
    assert c.snapshot().drift_ppm == 0.0
    ok = RoundTripPcClock(cs, FakeLaptop())
    feed(ok, hellos(np.random.default_rng(8), 41, 50.0))
    assert ok.snapshot().drift_ppm == pytest.approx(50.0, abs=5.0)


def test_one_snapshot_maps_both_rows_and_now_consistently_across_a_refit(cs):
    laptop = FakeLaptop()
    c = RoundTripPcClock(cs, laptop)
    feed(c, hellos(np.random.default_rng(9), 10, 0.0))
    snap = c.snapshot()
    rows = [T0 + 4 * S, T0 + 4 * S + 50 * MS]
    before = [snap.to_pc_ns(t) for t in rows]
    c.add_hello(HelloSample(T0 + 5 * S, pc_time(T0 + 5 * S, 0.0) + 10 * MS, T0 + 5 * S))   # a refit moves it
    assert c.snapshot().to_pc_ns(rows[0]) != before[0]
    assert [snap.to_pc_ns(t) for t in rows] == before                # the snapshot did not move
    assert before[1] - before[0] == 50 * MS


def test_a_refit_jump_is_reported(cs):
    laptop = FakeLaptop(T0 + 5 * S)
    c = RoundTripPcClock(dataclasses.replace(cs, jump_log_ms=1.0), laptop)
    feed(c, hellos(np.random.default_rng(10), 10, 0.0))
    assert c.take_jumps() == []
    c.add_hello(HelloSample(T0 + 5 * S, pc_time(T0 + 5 * S, 0.0) + 10 * MS, T0 + 5 * S))   # 0-RTT, 10 ms off
    jumps = c.take_jumps()
    assert len(jumps) == 1 and jumps[0].jump_ms > 1.0 and c.take_jumps() == []


def test_agreement_with_a_hello(cs):
    c = RoundTripPcClock(cs, FakeLaptop())
    feed(c, hellos(np.random.default_rng(11), 10, 0.0))
    s = HelloSample(T0 + 6 * S, pc_time(T0 + 6 * S + MS, 0.0) + 7 * MS, T0 + 6 * S + 2 * MS)
    assert c.agreement_ms(s) == pytest.approx(7.0, abs=0.01)


def test_stats(cs):
    laptop = FakeLaptop()
    c = RoundTripPcClock(cs, laptop)
    samples = hellos(np.random.default_rng(12), 41, 0.0)
    feed(c, samples)
    st = c.stats(samples[-1].t_recv_ns)
    assert st.estimator == "round_trip" and st.ready and st.offset_ns == pytest.approx(OFFSET, abs=MS)
    assert st.rtt_min_ms == pytest.approx(2.0) and st.samples <= 21 and st.span_s == pytest.approx(10.0, abs=0.6)


def test_make_pc_clock_refuses_unknown_estimators(cs):
    assert isinstance(make_pc_clock("round_trip", cs), RoundTripPcClock)
    assert isinstance(make_pc_clock("one_way", cs), OneWayPcClock)
    with pytest.raises(ValueError):
        make_pc_clock("chrony", cs)


def test_thread_safety_one_adds_others_map(cs):
    laptop = FakeLaptop()
    rt, ow = RoundTripPcClock(cs, laptop), OneWayPcClock(cs, laptop)
    samples = hellos(np.random.default_rng(13), 2000, 0.0, period_s=0.02)
    feed(rt, samples[:1])
    ow.add_state(pc_time(T0, 0.0), T0)
    errors = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            try:
                for c in (rt, ow):
                    c.snapshot().to_pc_ns(T0)
                    c.ready(T0)
                    c.stats(T0)
                    c.hellos()
                    c.take_jumps()
            except Exception as e:      # pragma: no cover - reported below
                errors.append(e)
                return

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for i, s in enumerate(samples[1:], start=1):
        rt.add_hello(s)
        ow.add_state(pc_time(T0 + i * 20 * MS, 0.0), T0 + i * 20 * MS + MS)
    done.set()
    for t in threads:
        t.join()
    assert not errors
