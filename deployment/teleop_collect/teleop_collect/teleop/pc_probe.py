# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/pc_probe.py @ ea9f7fb
# changes: imports; the texts name teleop and teleop_collect check instead of the manager and policy_deploy check;
# the --axes docstring points to this package's DIRECTIONS.md.
# Behaviour, rows and the --check --json line format are unchanged.
"""python -m teleop_collect.teleop.pc_probe [--station FILE] [--seconds 30] [--local] [--check] [--json]
(STRUCTURE.md 8.6). The read-only link probe.

A C3 client, so never alongside a teleop run (TOP 2.4). It follows rule H2 first: it reads C2 and sends nothing on C3
while C2 shows running/holding (up to 2 x watchdog_limit_s, then it refuses: another client is driving) or homing (up
to teleop.timeouts.pc_homing_wait_s). Then for --seconds: the C2 rate, seq gaps, t_ns repeats and newest-state age;
a hello every cycle for 2 s then every 0.5 s, heartbeats in between, with C3 round trips p50/p95/max; BOTH PC
estimators fed side by side (round_trip from the hellos, one_way from C2) with their offsets, drifts and difference
printed every second; the hello reply and compare_hello. It never sends start, stop, home or set_waypoints.

--check runs 2 s and prints one check line per STRUCTURE.md 8.4 row and per link fact (C2 rate, RTT, clock agreement)
for the launcher; with --json each line is {"name", "status", "detail"}. Clock agreement is judged with the map of
link.yaml clock.estimator, as `run` judges it at start-up. It exits 0 unless it crashes.

--axes also prints, at the end, where the measured TCP's x, y, z axes point in the robot's base frame, in words:
the read-only check of contracts/frames.yaml F_T_TCP before settled: true (DIRECTIONS.md, pc_probe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TextIO

from deploy_contracts import ContractError, HelloReply, RemoteError, TransportTimeout, compare_hello

from teleop_collect.contracts import Station, StationError
from teleop_collect.teleop.errors import PcRestarted
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_client import PcClient
from teleop_collect.teleop.pc_clock import OneWayPcClock, RoundTripPcClock
from teleop_collect.teleop.pc_listener import PcListener
from teleop_collect.teleop.settings import Settings
from teleop_collect.teleop.startup_checks import StartupChecks
from teleop_collect.teleop.state_history import StateHistory

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


def emit(rows: list[PcCheck], as_json: bool, out: TextIO) -> None:
    for r in rows:
        if as_json:
            out.write(json.dumps({"name": r.name, "status": r.status, "detail": r.detail}) + "\n")
        else:
            out.write(f"{r.status.upper():5} {r.name}: {r.detail}\n")
    out.flush()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m teleop_collect.teleop.pc_probe", description=__doc__.split("\n")[0])
    ap.add_argument("--station", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--local", action="store_true", help="the PC on 127.0.0.1 (franka_control --local, or the fake PC)")
    ap.add_argument("--mock", action="store_true", help="expect hello mock true (and connect locally)")
    ap.add_argument("--check", action="store_true", help="2 s, then check lines for teleop_collect check")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--axes", action="store_true", help="also print where the TCP axes point (frames.yaml check)")
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
        return 1 if any(r.status == "fail" for r in rows) else 0
    finally:
        probe.close()


if __name__ == "__main__":
    sys.exit(main())
