"""B4: LinkEmulator schedule semantics and live thread timing (no sockets)."""
import time

import numpy as np
import pytest

from franka_deploy.stream.link import LinkConfig, LinkEmulator, schedule


def test_from_yaml_files_and_dict():
    c = LinkConfig.from_yaml("run/link/wifi_good_skew.yaml")
    assert c.latency_s == 0.008 and c.bandwidth_mbit_s == 400.0 and c.clock_offset_s == 12.345 and c.clock_drift_ppm == 30.0
    assert c.stall_rate_hz == 0.02 and c.stall_duration_s == (0.05, 0.15) and c.seed == 1
    i = LinkConfig.from_yaml("run/link/ideal.yaml")
    assert (i.latency_s, i.jitter_s, i.stall_rate_hz, i.bandwidth_mbit_s, i.clock_offset_s) == (0, 0, 0, 0, 0)
    assert LinkConfig.from_yaml({"latency_s": 0.1, "stall": {"rate_hz": 1, "duration_s": [0.1, 0.2]}}).stall_rate_hz == 1
    with pytest.raises(ValueError):
        LinkConfig.from_yaml({"latency_s": -1})


def _t(n=300, hz=60.0):
    return 100.0 + np.arange(n) / hz


def test_ideal_deliver_equals_send():
    t = _t()
    assert np.array_equal(schedule(LinkConfig(), t, np.full(len(t), 5e4)), t)


def test_latency_constant():
    t = _t()
    assert np.allclose(schedule(LinkConfig(latency_s=0.05), t, np.full(len(t), 1e4)) - t, 0.05)


def test_deterministic_and_in_order_and_never_early():
    cfg = LinkConfig.from_yaml("run/link/wifi_bad.yaml")
    t = _t(600)
    s = np.full(len(t), 6e4)
    a, b = schedule(cfg, t, s), schedule(cfg, t, s)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) >= 0)
    assert np.all(a >= t + cfg.latency_s - 1e-12)
    other = schedule(LinkConfig.from_yaml({**{"seed": 9}, "latency_s": 0.04, "jitter_s": 0.04}), t, s)
    assert not np.array_equal(a, other)


def test_jitter_mean():
    t = _t(4000)
    d = schedule(LinkConfig(jitter_s=0.004, seed=5), t, np.zeros(len(t))) - t
    assert d.min() >= 0
    # head-of-line ordering raises the mean above the raw jitter mean; check it is the right order of magnitude
    assert 0.003 < d.mean() < 0.008


def test_bandwidth_backlog_slope():
    hz, size, bw = 60.0, 100_000, 30.0  # 100 kB * 60 = 48 Mbit/s offered vs 30 Mbit/s capacity
    t = _t(600, hz)
    lag = schedule(LinkConfig(bandwidth_mbit_s=bw), t, np.full(len(t), size)) - t
    slope = np.polyfit(np.arange(len(t)), lag, 1)[0]
    expect = size * 8 / (bw * 1e6) - 1 / hz
    assert abs(slope - expect) / expect < 0.05
    # below capacity: no backlog
    lag2 = schedule(LinkConfig(bandwidth_mbit_s=100.0), t, np.full(len(t), size)) - t
    assert lag2.max() < size * 8 / 100e6 + 1e-9


def test_stall_is_head_of_line():
    cfg = LinkConfig(stall_rate_hz=1.0, stall_duration_s=(0.3, 0.3), seed=4)
    t = _t(600)
    d = schedule(cfg, t, np.zeros(len(t)))
    lag = d - t
    assert lag.max() >= 0.29
    k = int(np.argmax(lag > 0.05))
    # items sent during the stall are delivered together at (or after) its end, in order, none dropped
    held = np.flatnonzero(np.isclose(d, d[k]))
    assert len(held) >= 10
    assert np.all(np.diff(d) >= 0) and len(d) == len(t)


def test_stall_shifts_all_later_items():
    t = _t(600)
    z = np.zeros(len(t))
    cfg = LinkConfig(stall_rate_hz=1.0, stall_duration_s=(0.3, 0.3), seed=4)
    d = schedule(cfg, t, z)
    k = int(np.argmax(d - t > 0.05))               # first item caught by a stall
    assert np.all(d[k:] >= d[k])                   # nothing overtakes it
    assert np.all(d[k + 1:] >= d[k] - 1e-12)


def test_clock_formula_within_1us():
    cfg = LinkConfig(clock_offset_s=12.345, clock_drift_ppm=30.0)
    link = LinkEmulator(cfg, lambda x: None)
    try:
        t0 = link._t0_ns
        for dt in (0, 1_000_000_000, 37_123_456_789):
            now = t0 + dt
            want = now + 12.345e9 + 30e-6 * dt
            assert abs(link.sender_clock_ns(now) - want) < 1000
        # live value tracks monotonic + offset
        a = link.sender_clock_ns()
        assert abs(a - (time.monotonic_ns() + 12.345e9)) < 5e6
    finally:
        link.close()


def test_emulator_thread_timing_p95_within_2ms():
    cfg = LinkConfig(latency_s=0.01, jitter_s=0.004, seed=2)
    got = []
    link = LinkEmulator(cfg, lambda i: got.append((i, time.monotonic())))
    t0 = time.monotonic() + 0.05
    sched_in = [t0 + i / 60.0 for i in range(200)]
    exp = schedule(cfg, np.array(sched_in), np.full(200, 5e4))
    try:
        for i, ts in enumerate(sched_in):
            time.sleep(max(0.0, ts - time.monotonic()))
            link.submit(i, 50_000, t_send=ts)
        assert link.flush(5.0)
    finally:
        link.close()
    assert [g[0] for g in got] == list(range(200))
    err = np.array([g[1] for g in got]) - exp
    assert err.min() > -1e-3
    assert np.percentile(np.abs(err), 95) < 0.002, np.percentile(np.abs(err), [50, 95, 100])
    s = link.stats()
    assert s["submitted"] == s["delivered"] == 200 and s["errors"] == 0


def test_emulator_survives_failing_deliver_and_counts():
    def boom(_):
        raise RuntimeError("x")

    link = LinkEmulator(LinkConfig(), boom)
    for i in range(3):
        link.submit(i, 10)
    assert link.flush(2.0)
    assert link.stats()["errors"] == 3
    link.close()
