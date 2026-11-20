"""python -m policy_deploy.manager.pc_probe [--station FILE] [--seconds 30] [--local] [--check] [--json] [--axes]
[--start-pose] (STRUCTURE.md 8.6). The read-only link probe.

A C3 client, so never alongside a manager (TOP 2.4). It follows rule H2 first: it reads C2 and sends nothing on C3
while C2 shows running/holding (up to 2 x watchdog_limit_s, then it refuses: another client is driving) or homing (up
to manager.timeouts.pc_homing_wait_s). Then for --seconds: the C2 rate, seq gaps, t_ns repeats and newest-state age;
a hello every cycle for 2 s then every 0.5 s, heartbeats in between, with C3 round trips p50/p95/max; BOTH PC
estimators fed side by side (round_trip from the hellos, one_way from C2) with their offsets, drifts and difference
printed every second; the hello reply and compare_hello. It never sends start, stop, home or set_waypoints.

--check runs 2 s and prints one check line per STRUCTURE.md 8.4 row and per link fact (C2 rate, RTT, clock agreement)
for the launcher; with --json each line is {"name", "status", "detail"}. Clock agreement is judged with the map of
link.yaml clock.estimator, as `run` judges it at start-up. It exits 0 unless it crashes.

--axes (new) also prints, at the end, where the measured TCP's x, y, z axes point in the robot's base frame, in words:
the read-only check of contracts/frames.yaml F_T_TCP before settled: true (policy_deploy/DIRECTIONS.md).

--start-pose (new) also prints, at the end, the arm's pose from the newest C2 state, for measuring the start pose by
hand-guiding: q_start_rad (C2 q, rad, ready to paste into the PC's control_deploy/station.yaml under
`franka_control: homing:`), the fingertip (TCP) height z in the base frame, the tilt of the fingertip direction
(TCP z axis) from straight down and which way it points horizontally in words, the gripper width, and the training
demos' start for comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TextIO

from deploy_contracts import ContractError, HelloReply, RemoteError, TransportTimeout, compare_hello

from policy_deploy.contracts import Station, StationError
from policy_deploy.manager.errors import PcRestarted
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_client import PcClient
from policy_deploy.manager.pc_clock import OneWayPcClock, RoundTripPcClock
from policy_deploy.manager.pc_listener import PcListener
from policy_deploy.manager.settings import Settings
from policy_deploy.manager.startup_checks import StartupChecks
from policy_deploy.manager.state_history import StateHistory

NS_PER_S = 1_000_000_000
BURST_S = 2.0
CHECK_SECONDS = 2.0


@dataclass(frozen=True)
class PcCheck:
    name: str
    status: str           # pass | warn | fail | info
    detail: str


class PcProbe:
    def __init__(self, settings: Settings, out: TextIO, clock: Callable[[], int] = time.monotonic_ns,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.s = settings
        self._out = out
        self._clock = clock
        self._sleep = sleep
        self.stats = LinkStats(settings.link_stats_window_s, clock)
        self.round_trip = RoundTripPcClock(settings.clock, clock)
        self.one_way = OneWayPcClock(settings.clock, clock)
        self.history = StateHistory(settings.history_s)
        self._listener = PcListener(settings.state_endpoint, self.history, self.one_way, self.stats,
                                    settings.state_recv_hwm, settings.listener_poll_s, clock)
        self._pc: PcClient | None = None
        self.sent: list[str] = []          # every C3 request sent, in order
        self.reply: HelloReply | None = None
        self._repeats = 0
        self._last_t: int | None = None

    def close(self) -> None:
        self._listener.stop()
        if self._pc is not None:
            self._pc.close()

    def _print(self, text: str) -> None:
        self._out.write(text + "\n")
        self._out.flush()

    # ---- H2

    def h2_wait(self) -> PcCheck:
        """Rule H2: no C3 before C2 shows a mode other than running/holding/homing. Returns the H2 row."""
        s = self.s
        self._listener.start()
        start = self._clock()
        while self.history.latest() is None:
            if self._listener.error() is not None:
                return PcCheck("C2", "fail", f"C2 from {s.state_endpoint}: {self._listener.error()}")
            if self._clock() - start > s.timeouts.startup_s * NS_PER_S:
                return PcCheck("C2", "fail", f"no C2 state from {s.state_endpoint} within {s.timeouts.startup_s:g} s: "
                                             "check that control_deploy run is running on the PC, ASUS_5g, and "
                                             "link.yaml pc.address")
            self._sleep(s.heartbeat_period_s)
        first = self._clock()
        homing_since = None
        while True:
            mode = self.history.latest().mode
            now = self._clock()
            if mode in ("idle", "stopped", "fault"):
                return PcCheck("H2", "pass" if mode != "fault" else "warn",
                               f"C2 mode {mode}: no other C3 client" + ("; the PC is in fault" if mode == "fault"
                                                                        else ""))
            if mode == "homing":
                homing_since = homing_since or now
                if now - homing_since > s.timeouts.pc_homing_wait_s * NS_PER_S:
                    return PcCheck("H2", "fail", f"the PC is still homing after {s.timeouts.pc_homing_wait_s:g} s: "
                                                 "check the PC terminal")
            elif now - first > 2 * s.watchdog_limit_s * NS_PER_S:
                return PcCheck("H2", "fail", f"C2 still shows {mode} after {2 * s.watchdog_limit_s:g} s: the PC is "
                                             "driving an episode for another C3 client: stop the other laptop "
                                             "session, pc_probe or console")
            self._sleep(s.heartbeat_period_s)

    # ---- the probe

    def run(self, seconds: float, report_every_s: float | None = 1.0) -> list[PcCheck]:
        """H2, then hellos and heartbeats for `seconds`. Returns the link rows (C2, H2, hello, rate, RTT)."""
        s = self.s
        rows: list[PcCheck] = []
        h2 = self.h2_wait()
        rows.append(h2)
        if h2.status == "fail":
            return rows
        self._pc = PcClient(s.commands_endpoint, s.c3_timeout_s, s.timeouts.home_s, s.heartbeat_period_s,
                            self.stats, self._clock)
        start = self._clock()
        next_report = start + int((report_every_s or 1.0) * NS_PER_S)
        last_hello = None
        try:
            while self._clock() - start < seconds * NS_PER_S:
                now = self._clock()
                burst = now - start < BURST_S * NS_PER_S
                due = last_hello is None or burst or now - last_hello >= s.clock.hello_period_s * NS_PER_S
                if due:
                    self.sent.append("c3.hello/v1")
                    reply, sample = self._pc.hello()
                    self.reply = reply
                    self.round_trip.add_hello(sample)
                    self.one_way.add_hello(sample)
                    last_hello = now
                else:
                    self.sent.append("c3.heartbeat/v1")
                    self._pc.heartbeat()
                latest = self.history.latest()
                if latest is not None:
                    if self._last_t is not None and latest.t_ns == self._last_t:
                        self._repeats += 1
                    self._last_t = int(latest.t_ns)
                if report_every_s is not None and now >= next_report:
                    next_report += int(report_every_s * NS_PER_S)
                    self._print(self.report_line(now))
                if self._listener.error() is not None:
                    raise self._listener.error()
                self._sleep(s.heartbeat_period_s)
        except (TransportTimeout, RemoteError, ContractError, PcRestarted) as e:
            rows.append(PcCheck("hello", "fail", f"PC not answering or broken reply: {e}"))
            return rows
        link = self.stats.snapshot()
        rows.append(PcCheck("hello", "pass" if self.reply is not None else "fail",
                            f"answered; RTT p50 {link.rtt_p50_ms:.1f} ms, p95 {link.rtt_p95_ms:.1f} ms, max "
                            f"{link.rtt_max_ms:.1f} ms ({link.rtt_n} requests)" if link.rtt_n else "no reply"))
        rate = link.c2_rate_hz
        rows.append(PcCheck("C2 rate", "pass" if rate is not None and rate > 50 else "warn",
                            f"{rate:.0f} Hz, {link.seq_gaps} seq gaps, {self._repeats} t_ns repeats seen"
                            if rate is not None else "no C2 rate"))
        return rows

    def report_line(self, now: int) -> str:
        link = self.stats.snapshot(now)
        parts = []
        try:
            rt = self.round_trip.snapshot()
            ow = self.one_way.snapshot()
            diff_ms = (rt.to_pc_ns(now) - ow.to_pc_ns(now)) / 1e6
            parts.append(f"round_trip {rt.offset_ns(now) / 1e9:+.6f} s {rt.drift_ppm:+.1f} ppm")
            parts.append(f"one_way {ow.offset_ns(now) / 1e9:+.6f} s {ow.drift_ppm:+.1f} ppm")
            parts.append(f"diff {diff_ms:+.2f} ms")
        except Exception:
            parts.append("clock: no fit yet")
        latest = self.history.latest()
        if latest is not None and self.round_trip.has_fit():
            parts.append(f"newest C2 {(self.round_trip.snapshot().to_pc_ns(now) - latest.t_ns) / 1e6:.1f} ms old")
        if link.c2_rate_hz is not None:
            parts.append(f"C2 {link.c2_rate_hz:.0f} Hz gaps {link.seq_gaps}")
        if link.rtt_n:
            parts.append(f"rtt p50 {link.rtt_p50_ms:.1f} p95 {link.rtt_p95_ms:.1f} max {link.rtt_max_ms:.1f} ms")
        return " | ".join(parts)

    def check_rows(self, mock: bool) -> list[PcCheck]:
        """--check: 2 s of probing, then the STRUCTURE.md 8.4 rows."""
        rows = self.run(CHECK_SECONDS, report_every_s=None)
        if self.reply is None:
            return rows
        checks = StartupChecks(self.s, mock=mock)
        pc_clock = self.one_way if self.s.clock.estimator == "one_way" else self.round_trip   # what run will use
        for r in checks.rows(self.reply, pc_clock, pc_clock.hellos()):
            rows.append(PcCheck(r.name, r.status, r.detail))
        return rows


# The Franka base frame, seen standing behind the robot and facing the way it faces.
_BASE_WORDS = {(0, 1): "forward (away from the base)", (0, -1): "back (toward the base)",
               (1, 1): "to the robot's left", (1, -1): "to the robot's right", (2, 1): "up", (2, -1): "down"}


def axes_lines(O_T_TCP) -> list[str]:
    """One line per TCP axis: its direction in the base frame, and in words when within 25 deg of a base axis."""
    out = []
    for col, name in enumerate("xyz"):
        v = [float(O_T_TCP[row][col]) for row in range(3)]
        big = max(range(3), key=lambda i: abs(v[i]))
        words = _BASE_WORDS[(big, 1 if v[big] > 0 else -1)] if abs(v[big]) > 0.906 else "between base axes"
        out.append(f"TCP {name} axis -> base ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}): {words}")
    return out


_DEMOS_START = ("demos start: fingertips tilted ~48 deg from straight down (38-64), ~16 cm above the table (6-27), "
                "hand open; object ~16 cm ahead horizontally and ~17 cm below the fingertips")


def _horizontal_words(x: float, y: float) -> str:
    """Which way a horizontal direction (base x, y) points, in words: one base axis when within 25 deg of it, else the
    two it is between."""
    norm = math.hypot(x, y)
    v = (x / norm, y / norm)
    big = 0 if abs(v[0]) >= abs(v[1]) else 1
    first = _BASE_WORDS[(big, 1 if v[big] > 0 else -1)]
    if abs(v[big]) > 0.906:
        return first
    return f"between {first} and {_BASE_WORDS[(1 - big, 1 if v[1 - big] > 0 else -1)]}"


def start_pose_lines(state) -> list[str]:
    """--start-pose, from one C2 state: q_start_rad to paste, the fingertip height, the fingertip direction (tilt from
    straight down, and which way it points horizontally in words), the gripper width, and the demos' start."""
    q = ", ".join(f"{float(v):.4f}" for v in state.q)
    T = state.O_T_TCP
    v = [float(T[row][2]) for row in range(3)]                          # the TCP z axis: where the fingertips point
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -v[2]))))
    if math.hypot(v[0], v[1]) < 0.035:                                  # within 2 deg of vertical
        words = "no horizontal direction"
    else:
        words = f"pointing {_horizontal_words(v[0], v[1])}"
    return [f"q_start_rad: [{q}]   # paste under franka_control: homing: in the PC's control_deploy/station.yaml",
            f"fingertip (TCP) height: z {float(T[2][3]):.3f} m in the robot base frame",
            f"fingertip direction (TCP z axis) -> base ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}): tilted {tilt:.1f} deg "
            f"from straight down, {words}",
            f"gripper width: {float(state.gripper_width_m) * 1000:.1f} mm",
            _DEMOS_START]


def emit(rows: list[PcCheck], as_json: bool, out: TextIO) -> None:
    for r in rows:
        if as_json:
            out.write(json.dumps({"name": r.name, "status": r.status, "detail": r.detail}) + "\n")
        else:
            out.write(f"{r.status.upper():5} {r.name}: {r.detail}\n")
    out.flush()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m policy_deploy.manager.pc_probe", description=__doc__.split("\n")[0])
    ap.add_argument("--station", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--local", action="store_true", help="the PC on 127.0.0.1 (franka_control --local, or the fake PC)")
    ap.add_argument("--mock", action="store_true", help="expect hello mock true (and connect locally)")
    ap.add_argument("--check", action="store_true", help="2 s, then check lines for policy_deploy check")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--axes", action="store_true", help="also print where the TCP axes point (frames.yaml check)")
    ap.add_argument("--start-pose", action="store_true",
                    help="also print the arm's pose as the PC's start pose (q_start_rad) and the demos' start")
    a = ap.parse_args(argv)
    local = a.local or a.mock
    try:
        station = Station.load(a.station, mock=a.mock)
        settings = Settings.from_station(station)
        if local and not a.mock:
            settings = Settings.from_station(station, state_endpoint=settings.link.state_endpoint(local=True),
                                             commands_endpoint=settings.link.commands_endpoint(local=True))
    except (StationError, ValueError) as e:
        emit([PcCheck("settings", "fail", str(e))], a.json, sys.stdout)
        return 0 if a.check else 1
    probe = PcProbe(settings, sys.stderr if a.check else sys.stdout)
    try:
        if a.check:
            emit(probe.check_rows(mock=a.mock), a.json, sys.stdout)
            return 0
        rows = probe.run(a.seconds)
        if probe.reply is not None:
            mismatches = compare_hello(probe.reply, mock=a.mock, frame=settings.frame, link=settings.link)
            rows.append(PcCheck("compare_hello", "fail" if mismatches else "pass",
                                "; ".join(str(m) for m in mismatches) if mismatches else "all equal"))
            reply = {k: v for k, v in probe.reply.to_dict().items() if k not in ("schema", "F_T_TCP", "schemas")}
            print(f"hello reply: {reply}")
        emit(rows, a.json, sys.stdout)
        if a.axes:
            latest = probe.history.latest()
            for line in (axes_lines(latest.O_T_TCP) if latest is not None else ["TCP axes: no C2 state received"]):
                print(line)
        if a.start_pose:
            latest = probe.history.latest()
            for line in (start_pose_lines(latest) if latest is not None else ["start pose: no C2 state received"]):
                print(line)
        return 1 if any(r.status == "fail" for r in rows) else 0
    finally:
        probe.close()


if __name__ == "__main__":
    sys.exit(main())
