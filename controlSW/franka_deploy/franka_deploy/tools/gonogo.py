"""Go/no-go load test for the co-located 1 kHz loop (Jetson: bridge on isolated core 5, everything else on 0-4).

run: starts franka_bridge (or uses a running one with --endpoint), activates it, holds the current pose for
--hold-s, then drives a 1 Hz sine (--amp-m along --axis, base frame) at 200 Hz through the same REQ/REP link the
controller uses (--motion steps: 2 cm ramps per axis at 0.05 m/s with 2 s holds instead of the sine; used for the
bridge-alone lag tau / steady-state error e measurement), while optional load runs alongside:
  --stress            stress-ng --cpu 5 --taskset 0-4 (skipped with a note if stress-ng is missing)
  --tegrastats        tegrastats --interval 1000 (temperatures + RAM; skipped if missing)
  --cyclictest        cyclictest on core 5 (wake-up latency histogram; the loop-timing source when the backend
                      cannot report per-cycle compute time, i.e. panda-py)
  --load-cmd "CMD"    any extra process, repeatable: e.g. the RigCamera at 60 Hz and back-to-back inference
                      (see docs/hardware_day.md for the exact commands)
Every child is stopped at the end; the bridge writes bridge.jsonl (events + per-second stats); the client writes
client.jsonl (per-second request timing + tracking error). Then `evaluate` runs on the output directory.

evaluate: GO only if all hold
  reflexes == 0 (bridge "reflex" events) and no watchdog stop,
  min control success rate (per second, worst) >= 0.99,
  loop p99.9 < 0.5 ms: overall p99.9 of the per-cycle compute time (FakeBackend cycle histogram) or, when the
    backend reports none, of the cyclictest histogram; no loop timing at all = NO-GO.
    --no-loop-gate: the loop timing is reported but not gated (meaningless for the Python MuJoCo sim backend).
Also reported (not gated): worst-second p99.9, request service p99, client RTT p99.9 / misses, tracking (lag
tau, steady-state error e, max error; from trace.npz), tegrastats max temperatures and RAM.

    python -m franka_deploy gonogo run --out DIR [--config configs/bridge.yaml] [--duration-s 1800] [--hold-s 30]
        [--amp-m 0.02] [--stress] [--tegrastats] [--cyclictest] [--load-cmd "..."]... [--no-loop-gate]
    python -m franka_deploy gonogo evaluate DIR [--no-loop-gate]
The bridge (spawned or external) is ready once get_state answers (polled for up to 10 s).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from franka_deploy.bridge.backends import HIST_BIN_US

PKG = Path(__file__).resolve().parents[1]
CRITERIA = {"max_reflexes": 0, "min_success_rate": 0.99, "loop_p999_ms": 0.5}


# ---------------------------------------------------------------- parsing
def hist_percentile(hist, q, bin_us=HIST_BIN_US) -> float:
    """Upper edge (ms) of the bin holding quantile q (conservative)."""
    h = np.asarray(hist, float)
    if h.sum() == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(h), q * h.sum()))
    return (k + 1) * bin_us / 1000.0


def parse_cyclictest(text: str) -> dict | None:
    """cyclictest -q -h N output: rows 'latency_us count [count ...]' before the '# Total' summary."""
    rows = []
    for ln in text.splitlines():
        m = re.match(r"^(\d+)\s+(\d+)", ln)
        if m:
            rows.append((int(m.group(1)), int(m.group(2))))
    mx = re.search(r"# Max Latencies:\s+(\d+)", text)
    if not rows:
        return None
    lat = np.array([r[0] for r in rows], float)
    cnt = np.array([r[1] for r in rows], float)
    if cnt.sum() == 0:
        return None
    c = np.cumsum(cnt)
    p999 = lat[int(np.searchsorted(c, 0.999 * c[-1]))] / 1000.0
    return {"n": int(c[-1]), "p999_ms": float(p999), "max_ms": float(mx.group(1)) / 1000.0 if mx else float(lat[cnt > 0].max()) / 1000}


TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
TEGRA_TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")


def parse_tegrastats(text: str) -> dict | None:
    ram, temps = [], {}
    for ln in text.splitlines():
        m = TEGRA_RAM.search(ln)
        if m:
            ram.append((int(m.group(1)), int(m.group(2))))
        for name, v in TEGRA_TEMP.findall(ln):
            temps[name] = max(temps.get(name, -1e9), float(v))
    if not ram and not temps:
        return None
    return {"samples": len(ram), "ram_used_max_mb": max(r[0] for r in ram) if ram else None,
            "ram_total_mb": ram[0][1] if ram else None, "temp_max_c": temps}


def _jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


# ---------------------------------------------------------------- motion + tracking
STEP_V = 0.05          # m/s ramp speed of the step profile
STEP_HOLD_S = 2.0


def motion_offset(kind: str, t: float, amp: float, freq: float, axis: int) -> np.ndarray:
    """Base-frame position offset of the commanded target at time t (s) after the hold phase.
    sine: amp*sin(2 pi f t) on `axis`. steps: per axis x, y, z in turn: ramp +amp at STEP_V, hold, ramp back, hold."""
    off = np.zeros(3)
    if kind == "sine":
        off[axis] = amp * np.sin(2 * np.pi * freq * t)
        return off
    r = amp / STEP_V
    seg = 2 * r + 2 * STEP_HOLD_S
    ax = int(t // seg) % 3
    u = t % seg
    if u < r:
        off[ax] = STEP_V * u
    elif u < r + STEP_HOLD_S:
        off[ax] = amp
    elif u < 2 * r + STEP_HOLD_S:
        off[ax] = amp - STEP_V * (u - r - STEP_HOLD_S)
    return off


def tracking_stats(t, cmd, fk, max_lag_s=0.2) -> dict:
    """Lag tau = time shift minimising |fk(t) - cmd(t - tau)| (moving samples), steady-state error e = |fk - cmd|
    where the command has been constant for >= 1.5 s."""
    t, cmd, fk = np.asarray(t), np.asarray(cmd), np.asarray(fk)
    if len(t) < 50:
        return {}
    moved = np.r_[True, np.any(np.abs(np.diff(cmd, axis=0)) > 1e-9, axis=1)]
    t_last_move = np.maximum.accumulate(np.where(moved, t, -np.inf))
    steady = (t - t_last_move) >= 1.5
    mv = moved if moved.sum() >= 20 else np.ones_like(moved)         # lag from the moving samples only
    lags = np.arange(0.0, max_lag_s + 1e-9, 0.001)
    errs = []
    for L in lags:
        c = np.stack([np.interp(t[mv] - L, t, cmd[:, i]) for i in range(3)], 1)
        errs.append(float(np.median(np.linalg.norm(fk[mv] - c, axis=1))))
    tau = float(lags[int(np.argmin(errs))])
    e = np.linalg.norm(fk - cmd, axis=1)
    out = {"lag_ms": tau * 1e3, "residual_after_lag_mm": min(errs) * 1e3, "track_err_mm_max": float(e.max() * 1e3)}
    if steady.any():
        out.update({"steady_err_mm_median": float(np.median(e[steady]) * 1e3),
                    "steady_err_mm_max": float(e[steady].max() * 1e3), "steady_samples": int(steady.sum())})
    return out


# ---------------------------------------------------------------- evaluate
def evaluate(out_dir, criteria=CRITERIA, loop_gate: bool = True) -> dict:
    d = Path(out_dir)
    ev = _jsonl(d / "bridge.jsonl")
    stats = [e for e in ev if e.get("kind") == "stats" and e.get("state") in ("running", "hold")]
    reflexes = [e for e in ev if e.get("kind") == "reflex"]
    wd_stops = [e for e in ev if e.get("kind") == "watchdog_stop"]
    wd_holds = [e for e in ev if e.get("kind") == "watchdog_hold"]
    rep = {"out_dir": str(d), "criteria": criteria, "loop_gate": loop_gate, "seconds": len(stats), "reflexes": len(reflexes),
           "watchdog_stops": len(wd_stops), "watchdog_holds": len(wd_holds), "reasons": []}
    succ = [float(s.get("min_success", 1.0)) for s in stats]
    rep["min_success_rate"] = min(succ) if succ else None
    hist = None
    worst = []
    for s in stats:
        h = s.get("loop", {}).get("cycle_hist_us")
        if h:
            hist = np.asarray(h, float) if hist is None else hist + np.asarray(h, float)
            worst.append(float(s["loop"].get("cycle_p999_ms", np.nan)))
    ct = None
    if (d / "cyclictest.txt").exists():
        ct = parse_cyclictest((d / "cyclictest.txt").read_text())
    if hist is not None:
        rep["loop"] = {"source": "backend cycle compute time", "n": int(hist.sum()),
                       "p999_ms": hist_percentile(hist, 0.999), "worst_second_p999_ms": max(worst),
                       "overflow_ge_5ms": int(hist[-1])}
    elif ct is not None:
        rep["loop"] = {"source": "cyclictest wake-up latency (core 5)", **ct}
    else:
        rep["loop"] = None
    if ct is not None and hist is not None:
        rep["cyclictest"] = ct
    svc = [s.get("svc_us_p99", 0.0) for s in stats]
    rep["bridge_service_p99_us_max"] = max(svc) if svc else None
    cl = _jsonl(d / "client.jsonl")
    if cl:
        rep["client"] = {"rtt_p999_ms_max": max(c["rtt_p999_ms"] for c in cl), "misses": sum(c["misses"] for c in cl),
                         "track_err_mm_max": max(c["track_err_mm_max"] for c in cl)}
    if (d / "trace.npz").exists():
        z = np.load(d / "trace.npz")
        m = z["t"] >= float(z["t_motion"])
        rep["tracking"] = tracking_stats(z["t"][m], z["cmd"][m], z["fk"][m])
    if (d / "tegrastats.log").exists():
        rep["tegrastats"] = parse_tegrastats((d / "tegrastats.log").read_text())
    meta = d / "run_meta.json"
    if meta.exists():
        rep["run_meta"] = json.loads(meta.read_text())
    # gates
    if not stats:
        rep["reasons"].append("no running-state stats lines in bridge.jsonl")
    if len(reflexes) > criteria["max_reflexes"]:
        rep["reasons"].append(f"{len(reflexes)} reflex(es)")
    if wd_stops:
        rep["reasons"].append(f"{len(wd_stops)} watchdog stop(s)")
    if succ and min(succ) < criteria["min_success_rate"]:
        rep["reasons"].append(f"min success rate {min(succ):.4f} < {criteria['min_success_rate']}")
    if not loop_gate:
        pass                       # reported above, not gated (--no-loop-gate)
    elif rep["loop"] is None:
        rep["reasons"].append("no loop timing (backend reports no cycle times and no cyclictest.txt)")
    elif not rep["loop"]["p999_ms"] < criteria["loop_p999_ms"]:
        rep["reasons"].append(f"loop p99.9 {rep['loop']['p999_ms']:.3f} ms >= {criteria['loop_p999_ms']} ms")
    rep["go"] = not rep["reasons"]
    return rep


# ---------------------------------------------------------------- run
def _spawn(name, cmd, out: Path, procs: dict, env=None):
    log = open(out / f"{name}.log", "w")
    procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True), log)


def _stop_all(procs: dict, timeout=10):
    for name, (p, _) in procs.items():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    t = time.monotonic() + timeout
    for name, (p, log) in procs.items():
        try:
            p.wait(max(0.1, t - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(2)
        log.close()


def wait_ready(endpoint: str, timeout_s: float = 10.0, procs: dict | None = None, out: Path | None = None) -> float:
    """Poll the bridge's get_state until it answers (the MuJoCo backend needs a few seconds to build its model).
    Returns the wait (s); raises RuntimeError on timeout or when the spawned bridge exits."""
    from franka_deploy.bridge.link import BridgeClient

    t0 = time.monotonic()
    cli = BridgeClient(endpoint, timeout_s=0.5)
    try:
        while True:
            try:
                cli.request("get_state", timeout_s=0.5)
                return time.monotonic() - t0
            except Exception as e:
                err = e
            if procs and "bridge" in procs and procs["bridge"][0].poll() is not None:
                raise RuntimeError(f"bridge exited during startup (see {out}/bridge.log)")
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"bridge {endpoint} not ready after {timeout_s:.0f} s: {err}")
            time.sleep(0.1)
    finally:
        cli.close()


def run(a) -> int:
    from franka_deploy.bridge.link import BridgeClient

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    procs: dict = {}
    notes = []
    env = dict(os.environ)
    try:
        if a.endpoint is None:
            endpoint = a.spawn_endpoint
            bcmd = [sys.executable, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config", a.config,
                    "--endpoint", endpoint, "--log-jsonl", str(out / "bridge.jsonl")]
            if a.bridge_core is not None and shutil.which("taskset"):
                bcmd = ["taskset", "-c", str(a.bridge_core)] + bcmd
            if a.bridge_rtprio and shutil.which("chrt"):
                bcmd = ["chrt", "-f", str(a.bridge_rtprio)] + bcmd
            _spawn("bridge", bcmd, out, procs, env)
        else:
            endpoint = a.endpoint
            notes.append("external bridge: its --log-jsonl must be <out>/bridge.jsonl for evaluate")
        if a.stress:
            if shutil.which("stress-ng"):
                _spawn("stress", ["stress-ng", "--cpu", "5", "--taskset", "0-4", "--timeout",
                                  f"{int(a.duration_s + a.hold_s + 10)}s"], out, procs)
            else:
                notes.append("stress-ng not installed: CPU stress skipped")
        if a.tegrastats:
            if shutil.which("tegrastats"):
                _spawn("tegrastats", ["tegrastats", "--interval", "1000", "--logfile", str(out / "tegrastats.log")], out, procs)
            else:
                notes.append("tegrastats not available (not a Jetson): temperature/RAM not recorded")
        if a.cyclictest:
            if shutil.which("cyclictest"):
                dur = int(a.duration_s + a.hold_s)
                cmd = ["cyclictest", "-q", "-m", "-a", str(a.bridge_core if a.bridge_core is not None else 5), "-t1",
                       "-p", "80", "-i", "1000", "-h", "2000", "-D", f"{dur}s"]
                procs["cyclictest"] = (subprocess.Popen(cmd, stdout=open(out / "cyclictest.txt", "w"),
                                                        stderr=subprocess.STDOUT, start_new_session=True),
                                       open(os.devnull, "w"))
            else:
                notes.append("cyclictest not installed: no wake-up latency histogram")
        for i, c in enumerate(a.load_cmd or []):
            _spawn(f"load{i}", shlex.split(c), out, procs, env)
        notes.append(f"bridge ready after {wait_ready(endpoint, 10.0, procs, out):.2f} s")
        for n, (p, _) in procs.items():
            if p.poll() is not None and n not in ("cyclictest",):
                raise RuntimeError(f"{n} exited during startup (see {out}/{n}.log)")
        cli = BridgeClient(endpoint, timeout_s=0.05)
        cli.request("start", timeout_s=2.0)
        st = cli.request("get_state", timeout_s=1.0)
        T0 = np.asarray(st["O_T_EE"], float)
        axis = {"x": 0, "y": 1, "z": 2}[a.axis]
        period, t_start = 1.0 / a.rate_hz, time.monotonic()
        t_sine = t_start + a.hold_s
        t_end = t_sine + a.duration_s
        nxt, sec = t_start, int(t_start)
        rtts, misses, track = [], 0, []
        tr_t, tr_c, tr_f = [], [], []
        cj = open(out / "client.jsonl", "w", buffering=1)
        stop = {"v": False}
        signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
        signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
        while not stop["v"]:
            now = time.monotonic()
            if now >= t_end:
                break
            T = T0.copy()
            if now >= t_sine:
                T[:3, 3] += motion_offset(a.motion, now - t_sine, a.amp_m, a.freq_hz, axis)
            t0 = time.perf_counter()
            try:
                cli.request("set_target", {"T": T})
                s = cli.request("get_state")
                rtts.append(time.perf_counter() - t0)
                fk = np.asarray(s["O_T_EE"])[:3, 3]
                track.append(float(np.linalg.norm(fk - T[:3, 3])))
                tr_t.append(now)
                tr_c.append(T[:3, 3].astype(np.float32))
                tr_f.append(fk.astype(np.float32))
            except Exception:
                misses += 1
            if int(now) != sec and rtts:
                r = np.array(rtts) * 1e3
                cj.write(json.dumps({"t": now, "phase": "sine" if now >= t_sine else "hold", "n": len(r),
                                     "rtt_p50_ms": float(np.percentile(r, 50)), "rtt_p999_ms": float(np.percentile(r, 99.9)),
                                     "misses": misses, "track_err_mm_max": max(track) * 1e3 if track else 0.0}) + "\n")
                sec, rtts, misses, track = int(now), [], 0, []
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 0.1:
                nxt = time.monotonic()
        cj.close()
        if tr_t:
            np.savez_compressed(out / "trace.npz", t=np.array(tr_t), cmd=np.array(tr_c), fk=np.array(tr_f), t_motion=t_sine)
        # bring the arm back to the start pose before stopping the controller
        for _ in range(int(a.rate_hz)):
            try:
                cli.request("set_target", {"T": T0})
            except Exception:
                pass
            time.sleep(period)
        try:
            cli.request("stop", timeout_s=1.0)
        except Exception:
            pass
        cli.close()
    finally:
        _stop_all(procs)
        (out / "run_meta.json").write_text(json.dumps({"args": vars(a), "notes": notes}, indent=1, default=str))
    rep = evaluate(out, loop_gate=not a.no_loop_gate)
    (out / "gonogo.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep.get(k) for k in ("go", "reasons", "seconds", "reflexes", "min_success_rate", "loop", "tracking")},
                     indent=1))
    for n in notes:
        print("note:", n)
    return 0 if rep["go"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy gonogo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=str(PKG.parent / "configs" / "bridge.yaml"))
    r.add_argument("--endpoint", default=None, help="use a running bridge instead of starting one")
    r.add_argument("--duration-s", type=float, default=1800.0)
    r.add_argument("--hold-s", type=float, default=30.0)
    r.add_argument("--amp-m", type=float, default=0.02)
    r.add_argument("--freq-hz", type=float, default=1.0)
    r.add_argument("--motion", default="sine", choices=["sine", "steps"],
                   help="after the hold: 1 Hz sine, or 2 cm steps per axis at 0.05 m/s (bridge-alone tau / e test)")
    r.add_argument("--axis", default="y", choices=["x", "y", "z"])
    r.add_argument("--rate-hz", type=float, default=200.0)
    r.add_argument("--bridge-core", type=int, default=None, help="taskset the bridge onto this core (Jetson: 5)")
    r.add_argument("--bridge-rtprio", type=int, default=None, help="chrt -f priority for the bridge (Jetson: 80)")
    r.add_argument("--stress", action="store_true")
    r.add_argument("--tegrastats", action="store_true")
    r.add_argument("--cyclictest", action="store_true")
    r.add_argument("--load-cmd", action="append")
    r.add_argument("--spawn-endpoint", default="tcp://127.0.0.1:5599", help="endpoint of the bridge this run spawns")
    r.add_argument("--no-loop-gate", action="store_true",
                   help="report the loop timing but do not gate on it (Python MuJoCo sim backend)")
    e = sub.add_parser("evaluate")
    e.add_argument("out")
    e.add_argument("--no-loop-gate", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    rep = evaluate(a.out, loop_gate=not a.no_loop_gate)
    print(json.dumps(rep, indent=1))
    return 0 if rep["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
