"""B5: ClockMap units (pure numpy; runs in RE and BP). Synthetic frame grids and B4's LinkEmulator schedule()."""
import numpy as np
import pytest

from franka_deploy.peripherals.clock_map import ClockMap, clock_map_from_config
from franka_deploy.stream.link import LinkConfig, schedule

FPS = 30.0


def _grid(n, t0=100.0, fps=FPS):
    return t0 + np.arange(n) / fps


def test_arrival_minus_latency_single_subtraction():
    cm = ClockMap("arrival_minus_latency", {"main": 0.08, "ultrawide": 0.05})
    assert cm.map("main", 10.0) == pytest.approx(9.92)
    assert cm.map("ultrawide", 10.0) == pytest.approx(9.95)
    assert cm.map("depth", 11.0) == pytest.approx(11.0)       # unknown stream: no latency
    s = cm.stats()
    assert s["mode"] == "arrival_minus_latency" and s["monotonic_clamps"] == 0 and s["n_mapped"] == 3


def test_monotonic_clamp_per_stream():
    cm = ClockMap("arrival_minus_latency", {"main": 0.0})
    assert cm.map("main", 5.0) == 5.0
    assert cm.map("main", 4.9) == 5.0            # clamped
    assert cm.map("ultrawide", 4.0) == 4.0       # other stream is independent
    assert cm.map("main", 5.1) == 5.1
    assert cm.stats()["monotonic_clamps"] == 1


def test_dejitter_exact_on_constant_delay_and_gaps():
    """Constant latency + phone-side drops: the grid is recovered exactly, except the 2 frames before a drop is
    confirmed (stamped 1.4 periods early or less: the t_arrival - 1.4 T bound, never late)."""
    t = _grid(300)
    keep = np.ones(len(t), bool)
    keep[[50, 51, 120, 200, 201, 202]] = False   # dropped on the phone
    cm = ClockMap("arrival_dejitter", {"main": 0.03}, fps_hz=FPS, window_s=2.0)
    est = np.array([cm.map("main", ti + 0.03) for ti in t[keep]])
    err = est - t[keep]
    bad = np.flatnonzero(np.abs(err) > 1e-9)
    # a drop is confirmed on the 3rd late frame: the 2 frames before it are stamped early (by the dropped periods)
    kept_idx = np.flatnonzero(keep)
    assert set(kept_idx[bad]) == {52, 53, 121, 122, 203, 204}, kept_idx[bad]
    assert np.all(err[bad] < 0) and np.all(err[bad] >= -1.4 / FPS - 1e-9)
    assert cm.stats()["dropped_frames"] == 6


@pytest.mark.parametrize("n_drop", [2, 4, 6, 8])
def test_dejitter_sender_drops_bounded(n_drop):
    """An n-frame sender drop (with Wi-Fi-like jitter on every frame): no stamp is off by more than 1.5 periods,
    including the unconfirmed frames right after the gap (review finding: -195 ms after a 6-frame drop before)."""
    rng = np.random.default_rng(n_drop)
    t = _grid(600)
    keep = np.ones(len(t), bool)
    for start in (100, 250, 400):
        keep[start:start + n_drop] = False
    lat = 0.02
    arr = t[keep] + lat + rng.gamma(2.0, 0.002, keep.sum())      # mean 4 ms jitter, in-order
    arr = np.maximum.accumulate(arr)
    cm = ClockMap("arrival_dejitter", {"main": lat}, fps_hz=FPS, window_s=2.0)
    err = np.array([cm.map("main", a) for a in arr]) - t[keep]
    print(f"drop {n_drop}: |err| max {np.abs(err).max() * 1e3:.1f} ms, p95 {np.percentile(np.abs(err), 95) * 1e3:.1f} ms")
    assert np.abs(err).max() <= 1.5 / FPS
    assert cm.stats()["dropped_frames"] == 3 * n_drop


def test_dejitter_beats_arrival_on_wifi_busy():
    """wifi_busy link (20 ms latency, gamma jitter 15 ms, stalls 0.2 Hz x 0.05-0.25 s, 150 Mbit/s): the p95 stamp
    error of the lower-envelope fit is far below plain arrival-minus-latency."""
    n = int(120 * FPS)
    t_true = _grid(n)
    sizes = np.random.default_rng(1).integers(120_000, 200_000, n)
    for name, seed in (("main", 2), ("ultrawide", 7)):
        cfg = LinkConfig.from_yaml({**_busy_dict(), "seed": seed})
        arr = schedule(cfg, t_true, sizes)
        lat = {name: cfg.latency_s}
        plain = ClockMap("arrival_minus_latency", lat)
        dj = ClockMap("arrival_dejitter", lat, fps_hz=FPS, window_s=2.0)
        e_plain = np.array([plain.map(name, a) for a in arr]) - t_true
        e_dj = np.array([dj.map(name, a) for a in arr]) - t_true
        p95_plain, p95_dj = np.percentile(np.abs(e_plain), 95), np.percentile(np.abs(e_dj), 95)
        print(f"wifi_busy {name}: |err| p50/p95 plain {np.median(np.abs(e_plain))*1e3:.2f}/{p95_plain*1e3:.2f} ms, "
              f"dejitter {np.median(np.abs(e_dj))*1e3:.2f}/{p95_dj*1e3:.2f} ms, clamps {dj.stats()['monotonic_clamps']}")
        assert p95_dj < 0.5 * p95_plain
        assert p95_dj < 0.005


def _busy_dict():
    import yaml

    from franka_deploy.stream.link import CONFIG_ROOT
    return yaml.safe_load((CONFIG_ROOT / "run/link/wifi_busy.yaml").read_text())


def test_dejitter_restarts_after_long_gap():
    cm = ClockMap("arrival_dejitter", {"main": 0.0}, fps_hz=FPS, window_s=1.0)
    for ti in _grid(30):
        cm.map("main", ti + 0.01)
    t2 = _grid(30, t0=200.123)                      # sender restarted with a new phase
    est = [cm.map("main", ti + 0.01) for ti in t2]
    np.testing.assert_allclose(est, t2 + 0.01, atol=1e-9)   # new envelope anchored on the first frame (min delay)
    assert cm.stats()["envelope_restarts"] == 1


def _ping_world(offset_s=12.345, drift_ppm=30.0, seed=0):
    rng = np.random.default_rng(seed)
    t_start = 1000.0

    def sender_ns(t_host_s):
        return int(round((t_host_s + offset_s + drift_ppm * 1e-6 * (t_host_s - t_start)) * 1e9))

    def ping(t0):
        d1 = 0.0004 + rng.gamma(2.0, 0.0005)       # asymmetric, independent forward / return delays
        d2 = 0.0004 + rng.gamma(2.0, 0.0005) + (0.02 if rng.random() < 0.2 else 0.0)   # sometimes queued
        t1 = sender_ns(t0 + d1)
        t2 = t1 + 50_000
        t3 = t0 + d1 + 50e-6 + d2
        return int(round(t0 * 1e9)), t1, t2, int(round(t3 * 1e9))

    return t_start, sender_ns, ping


def test_ping_offset_and_drift_recovery():
    """offset 12.345 s, drift 30 ppm, 10 s of pings at 2 Hz with RTT jitter: theta(t) within 0.5 ms."""
    t_start, sender_ns, ping = _ping_world()
    cm = ClockMap("sender_stamp_ping", {}, window_s=20.0, residual_s={"main": 0.0})
    assert not cm.synced and cm.theta(t_start) is None
    for t0 in t_start + np.arange(0, 10.0, 0.5):
        cm.add_ping(*ping(t0))
    assert cm.synced
    errs = []
    for t in t_start + np.linspace(0, 12.0, 50):
        true_theta = sender_ns(t) * 1e-9 - t
        errs.append(cm.theta(t) - true_theta)
    errs = np.abs(errs)
    s = cm.stats()
    print(f"ping: theta err max {errs.max()*1e3:.3f} ms, drift {s['drift_ppm']:.1f} ppm, rtt_min {s['rtt_ms_min']:.2f} ms")
    assert errs.max() < 0.5e-3
    assert s["n_pings"] == 20 and s["rtt_ms_min"] > 0
    assert abs(s["offset_s"] - 12.345) < 0.01
    # frames stamped on the sender clock map back to host capture time
    _, _, ping = _ping_world()
    cm2 = ClockMap("sender_stamp_ping", {}, window_s=20.0, residual_s={"main": 0.002})
    for t0 in t_start + np.arange(0, 10.0, 0.5):
        cm2.add_ping(*ping(t0))
    t_cap = t_start + 10.0
    est = cm2.map("main", t_cap + 0.03, t_sender_ns=sender_ns(t_cap))
    assert abs(est - (t_cap - 0.002)) < 0.5e-3


def test_ping_wifi_like_rtt_jitter():
    """Exponential RTT jitter of 4 ms per direction (review finding 3), 2 Hz pings, default 60 s window: steady-state
    |theta error| p95 <= 1 ms and the drift is near 30 ppm, for several seeds."""
    for seed in range(4):
        rng = np.random.default_rng(seed)
        t_start = 1000.0

        def sender_ns(t):
            return int(round((t + 12.345 + 30e-6 * (t - t_start)) * 1e9))

        cm = clock_map_from_config({"mode": "sender_stamp_ping", "ping_hz": 2.0})
        assert cm.window_s == 60.0
        errs = []
        for t0 in t_start + np.arange(0, 120.0, 0.5):
            d1, d2 = 0.0005 + rng.exponential(0.004), 0.0005 + rng.exponential(0.004)
            t1 = sender_ns(t0 + d1)
            cm.add_ping(int(round(t0 * 1e9)), t1, t1 + 50_000, int(round((t0 + d1 + 50e-6 + d2) * 1e9)))
            if t0 >= t_start + 60.0:
                for tq in t0 + np.arange(0, 0.5, 0.1):
                    errs.append(cm.theta(tq) - (sender_ns(tq) * 1e-9 - tq))
        p95 = np.percentile(np.abs(errs), 95)
        print(f"ping jitter seed {seed}: |theta err| p95 {p95 * 1e3:.2f} ms, drift {cm.stats()['drift_ppm']:.1f} ppm")
        assert p95 <= 1e-3
        assert abs(cm.stats()["drift_ppm"] - 30.0) < 40.0


def test_ping_unsynced_falls_back_and_bad_sample_ignored():
    cm = ClockMap("sender_stamp_ping", {}, residual_s={"main": 0.01})
    assert cm.map("main", 5.0, t_sender_ns=123) == pytest.approx(4.99)
    assert cm.stats()["n_unsynced"] == 1
    cm.add_ping(200, 50, 60, 100)                    # t3 < t0 (negative RTT) -> ignored
    assert cm.stats()["n_pings"] == 0 and not cm.synced
    with pytest.raises(ValueError):
        cm.map("main", 5.0)


def test_sender_stamp_realtime():
    cm = ClockMap("sender_stamp_realtime", {}, residual_s={"main": 0.001}, rt_minus_mono_ns=lambda: 1_700_000_000 * 10**9)
    t = cm.map("main", 0.0, t_sender_ns=1_700_000_000 * 10**9 + 5 * 10**9)
    assert t == pytest.approx(5.0 - 0.001)
    real = ClockMap("sender_stamp_realtime", {})
    import time
    now_rt = time.time_ns()
    assert abs(real.map("main", 0.0, t_sender_ns=now_rt) - time.monotonic()) < 0.01


def test_from_config_and_validation():
    cm = clock_map_from_config({"mode": "arrival_dejitter", "latency_s": {"main": 0.08}, "dejitter_window_s": 1.5},
                               {"main": {"physical_fps_hz": 30}})
    assert cm.window_s == 1.5 and cm.fps_hz == {"main": 30.0}
    cm = clock_map_from_config({"mode": "sender_stamp_ping", "ping_hz": 2.0, "ping_window_s": 20.0,
                                "residual_latency_s": {"main": 0.0, "ultrawide": 0.0}})
    assert cm.window_s == 20.0 and cm.residual_s == {"main": 0.0, "ultrawide": 0.0}
    assert clock_map_from_config({"mode": "sender_stamp_ping"}).window_s == 60.0
    with pytest.raises(ValueError):
        ClockMap("nope", {})
    with pytest.raises(ValueError):
        ClockMap("arrival_dejitter", {})
