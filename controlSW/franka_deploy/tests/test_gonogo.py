"""gonogo: evaluator gates (pass on a FakeBackend run, fail on injected bad stats) and the log parsers."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from franka_deploy.bridge.backends import HIST_BINS, cycle_hist
from franka_deploy.tools import gonogo

ROOT = Path(__file__).resolve().parents[1]


def _stats_line(success=1.0, cycles_s=None, state="running"):
    c = np.full(1000, 60e-6) if cycles_s is None else np.asarray(cycles_s)
    return {"t": 0.0, "kind": "stats", "state": state, "req": {}, "svc_us_p50": 50.0, "svc_us_p99": 200.0,
            "safety": {}, "loop": {"n": 1000, "p999_ms": 1.1, **cycle_hist(c)}, "min_success": success}


def _write(d: Path, lines):
    d.mkdir(parents=True, exist_ok=True)
    (d / "bridge.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d


def test_good_stats_go(tmp_path):
    rep = gonogo.evaluate(_write(tmp_path / "ok", [_stats_line() for _ in range(30)]))
    assert rep["go"], rep["reasons"]
    assert rep["loop"]["p999_ms"] <= 0.1


def test_injected_reflex_fails(tmp_path):
    lines = [_stats_line() for _ in range(30)] + [{"t": 1.0, "kind": "reflex", "error": "cartesian_reflex"}]
    rep = gonogo.evaluate(_write(tmp_path / "r", lines))
    assert not rep["go"] and any("reflex" in r for r in rep["reasons"])


def test_injected_low_success_fails(tmp_path):
    lines = [_stats_line() for _ in range(30)] + [_stats_line(success=0.985)]
    rep = gonogo.evaluate(_write(tmp_path / "s", lines))
    assert not rep["go"] and any("success" in r for r in rep["reasons"])


def test_injected_slow_cycles_fail(tmp_path):
    slow = np.full(1000, 60e-6)
    slow[:5] = 0.9e-3                                     # 5 of 30 000 cycles at 0.9 ms -> p99.9 (30th worst) ok
    lines = [_stats_line() for _ in range(29)] + [_stats_line(cycles_s=slow)]
    assert gonogo.evaluate(_write(tmp_path / "c_ok", lines))["go"]
    slow[:60] = 0.9e-3                                    # 60 slow cycles of 30 000 -> p99.9 = 0.9 ms
    lines = [_stats_line() for _ in range(29)] + [_stats_line(cycles_s=slow)]
    rep = gonogo.evaluate(_write(tmp_path / "c_bad", lines))
    assert not rep["go"] and any("p99.9" in r for r in rep["reasons"])


def test_watchdog_stop_and_no_timing_fail(tmp_path):
    lines = [_stats_line() for _ in range(5)] + [{"t": 1.0, "kind": "watchdog_stop", "age_s": 2.0}]
    assert not gonogo.evaluate(_write(tmp_path / "w", lines))["go"]
    no_t = _stats_line()
    no_t["loop"] = {}
    rep = gonogo.evaluate(_write(tmp_path / "n", [no_t] * 5))
    assert not rep["go"] and any("no loop timing" in r for r in rep["reasons"])
    # the cyclictest histogram is the fallback timing source (panda-py backend reports none)
    (tmp_path / "n" / "cyclictest.txt").write_text(
        "# /dev/cpu_dma_latency set to 0us\n000010 900000\n000020 99950\n000300 50\n000700 0\n# Total: 001000000\n"
        "# Min Latencies: 00005\n# Avg Latencies: 00011\n# Max Latencies: 00300\n")
    rep = gonogo.evaluate(tmp_path / "n")
    assert rep["go"], rep["reasons"]
    assert rep["loop"]["source"].startswith("cyclictest") and rep["loop"]["max_ms"] == 0.3


def test_hist_and_parsers():
    h = np.zeros(HIST_BINS)
    h[1] = 999
    h[9] = 1
    assert gonogo.hist_percentile(h, 0.999) == 0.1 and gonogo.hist_percentile(h, 1.0) == 0.5
    t = gonogo.parse_tegrastats(
        "RAM 3120/7620MB (lfb 2x4MB) SWAP 0/3810MB CPU [12%@1510,...] cpu@48.5C soc0@47C gpu@46.2C tj@49.1C\n"
        "RAM 3300/7620MB (lfb 2x4MB) SWAP 0/3810MB CPU [40%@1510,...] cpu@51C soc0@47.9C gpu@50C tj@52.25C\n")
    assert t["ram_used_max_mb"] == 3300 and t["ram_total_mb"] == 7620 and t["temp_max_c"]["tj"] == 52.25


def test_short_fakebackend_run_is_go(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run([sys.executable, "-m", "franka_deploy", "gonogo", "run", "--out", str(tmp_path / "run"),
                        "--duration-s", "6", "--hold-s", "2"], env=env, capture_output=True, text=True, timeout=60)
    rep = json.loads((tmp_path / "run" / "gonogo.json").read_text())
    assert p.returncode == 0 and rep["go"], (rep["reasons"], p.stdout[-2000:])
    assert rep["seconds"] >= 7 and rep["client"]["misses"] == 0


def test_motion_profiles_and_tracking_stats():
    amp = 0.02
    assert np.allclose(gonogo.motion_offset("sine", 0.25, amp, 1.0, 1), [0, amp, 0])
    r = amp / gonogo.STEP_V
    assert np.allclose(gonogo.motion_offset("steps", r + 1.0, amp, 1.0, 0), [amp, 0, 0])
    seg = 2 * r + 2 * gonogo.STEP_HOLD_S
    assert np.allclose(gonogo.motion_offset("steps", seg + r + 1.0, amp, 1.0, 0), [0, amp, 0])
    assert np.allclose(gonogo.motion_offset("steps", seg - 0.5, amp, 1.0, 0), 0)
    # first-order lag tau = 30 ms + 1 mm bias: recovered from the trace
    t = np.arange(0, 3 * seg, 0.005)
    cmd = np.array([gonogo.motion_offset("steps", x, amp, 1.0, 0) for x in t])
    fk = np.zeros_like(cmd)
    for i in range(1, len(t)):
        fk[i] = fk[i - 1] + (1 - np.exp(-0.005 / 0.03)) * (cmd[i] + [0.001, 0, 0] - fk[i - 1])
    st = gonogo.tracking_stats(t, cmd, fk)
    assert abs(st["steady_err_mm_median"] - 1.0) < 0.05
    assert 20 <= st["lag_ms"] <= 40
