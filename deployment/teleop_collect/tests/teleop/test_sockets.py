"""teleop on real sockets (tcp://127.0.0.1:*, the real clock): the real PcClient, PcListener, U1Server and R1
FramesClient against SocketPc and a small R1 server; rule H2; pc_probe and its --check --json lines; the process
entry's FATAL line. The real fake PC (E6) and the other processes come together in wave 3."""
from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from deploy_contracts import Ack, ErrorReply, LinkSettings, RpcClient, RpcServer
from station_helpers import write_station
from teleop_helpers import COMMIT, FAST, SocketPc, frames_health, merge, summary, wait_for

from teleop_collect.contracts import (CreateTask, GetFramesHealth, GetStatus, KeysUpdate, StartEpisode,
                                      StartRecording, Station, StopEpisode, StopRecording, TeleopStatus)
from teleop_collect.teleop.episode.store import TaskStore
from teleop_collect.teleop.failure_monitor import FailureMonitor
from teleop_collect.teleop.frames_client import FramesClient
from teleop_collect.teleop.heartbeat_sender import HeartbeatSender
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_client import PcClient
from teleop_collect.teleop.pc_clock import make_pc_clock
from teleop_collect.teleop.pc_listener import PcListener
from teleop_collect.teleop.pc_probe import PcProbe, emit
from teleop_collect.teleop.problems import ProblemCode as C
from teleop_collect.teleop.session_log import SessionLog
from teleop_collect.teleop.settings import Settings
from teleop_collect.teleop.startup_checks import StartupChecks
from teleop_collect.teleop.state_history import StateHistory
from teleop_collect.teleop.teleop_loop import Phase, TeleopLoop
from teleop_collect.teleop.u1_server import U1Server

FAST_LINK = dataclasses.replace(LinkSettings.load(), watchdog_limit_s=0.5, c3_timeout_s=0.15)


class R1Server:
    """frames' R1, answering on its own thread."""

    def __init__(self) -> None:
        self.recording = False
        self.stop_delay_s = 0.0
        self.calls: list[str] = []
        self._rpc = RpcServer("tcp://127.0.0.1:*", {StartRecording: self._start, StopRecording: self._stop,
                                                    GetFramesHealth: self._health})
        self.endpoint = self._rpc.endpoint
        self._stop_ev = threading.Event()
        self._t = threading.Thread(target=self._rpc.serve_forever, args=(self._stop_ev, 0.01), daemon=True)
        self._t.start()

    def _start(self, msg):
        self.calls.append(msg.SCHEMA)
        self.recording = True
        return Ack(ok=True, reason="")

    def _stop(self, msg):
        self.calls.append(msg.SCHEMA)
        self._stop_ev.wait(self.stop_delay_s)                         # a slow drain, cut short by close()
        if not self.recording:
            return ErrorReply(request=msg.SCHEMA, error="not recording", field="")
        self.recording = False
        return summary(12)

    def _health(self, msg):
        self.calls.append(msg.SCHEMA)
        return frames_health(recording=self.recording)

    def close(self) -> None:
        self._stop_ev.set()
        self._t.join(1.0)
        self._rpc.close()


class Rig:
    def __init__(self, tmp_path: Path, link: LinkSettings | None = None, teleop: dict | None = None,
                 **pc_kw) -> None:
        self.pc = SocketPc(link=link, **pc_kw)
        self.r1 = R1Server()
        station = Station.load(write_station(tmp_path, sections={"teleop": merge(FAST, teleop or {})}), mock=True)
        s = Settings.from_station(station, link=link, state_endpoint=self.pc.state_endpoint,
                                  commands_endpoint=self.pc.commands_endpoint)
        self.settings = s
        clock = time.monotonic_ns
        stats = LinkStats(s.link_stats_window_s, clock)
        pc_clock = make_pc_clock(s.clock.estimator, s.clock, clock)
        history = StateHistory(s.history_s)
        self.listener = PcListener(s.state_endpoint, history, pc_clock, stats, s.state_recv_hwm, s.listener_poll_s,
                                   clock)
        self.client = PcClient(s.commands_endpoint, s.c3_timeout_s, s.timeouts.home_s, s.heartbeat_period_s, stats,
                               clock)
        heartbeats = HeartbeatSender(self.client, pc_clock, stats, s.clock.hello_period_s,
                                     s.clock.startup_hello_period_s, clock)
        self.frames = FramesClient(self.r1.endpoint, s.timeouts.health_s, s.timeouts.frames_s)
        self.session = tmp_path / "session" / "teleop"
        self.log = SessionLog(self.session, datetime.now, clock)
        self.loop = TeleopLoop(s, self.client, heartbeats, pc_clock, stats, self.frames, history, self.listener,
                               FailureMonitor(s.state_lost_s), self.log, StartupChecks(s, mock=True, commit=COMMIT),
                               TaskStore(s.data_root), clock, time.sleep)
        self.u1 = U1Server("tcp://127.0.0.1:*", self.loop.dispatch)
        self.loop.attach_u1(self.u1)
        self.ui = RpcClient("teleop", self.u1.endpoint)
        self.exit_code: int | None = None
        self._thread: threading.Thread | None = None

    def run_in_thread(self, ready_file: Path | None = None) -> None:
        def body() -> None:
            self.loop.startup(ready_file)
            if self.loop.phase is not Phase.DONE:
                self.exit_code = self.loop.run()
        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()

    def request(self, msg, expect=Ack):
        return self.ui.request(msg, expect, 3.0)

    def status(self) -> TeleopStatus:
        return self.request(GetStatus(after_message=0), TeleopStatus)

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.loop.terminate()
            self._thread.join(5.0)
        for close in (self.ui.close, self.u1.close, self.listener.stop, self.client.close, self.frames.close,
                      self.log.close, self.r1.close, self.pc.close):
            try:
                close()
            except Exception:
                pass


@pytest.fixture
def rigs():
    made = []

    def make(tmp_path, **kw):
        rig = Rig(tmp_path, **kw)
        made.append(rig)
        return rig
    yield make
    for rig in made:
        rig.close()


def test_startup_and_an_episode_over_u1(tmp_path, rigs):
    rig = rigs(tmp_path)
    ready = tmp_path / "teleop.ready"
    rig.run_in_thread(ready)
    assert wait_for(lambda: ready.exists(), 10.0), rig.loop.verdict
    names = rig.pc.request_names()
    first_get_state = names.index("c3.get_state/v1")
    assert set(names[:first_get_state]) == {"c3.hello/v1"}
    assert "c3.home/v1" not in names                                    # no start-up home: nothing moves the arm
    assert rig.loop.start_pose.known and not rig.loop.start_pose.confirmed
    assert rig.request(CreateTask(name="pick_block", description="pick it")).ok
    assert wait_for(lambda: rig.status().start_allowed, 5.0), rig.status().start_refusal
    assert rig.request(StartEpisode()).ok
    assert rig.loop.start_pose.confirmed                                # the PC accepted the start there
    end = time.monotonic() + 0.6
    while time.monotonic() < end:                                       # the page's key frames
        rig.request(KeysUpdate(held=("x+",), slow=True, focus=True))
        time.sleep(0.04)
    rig.request(KeysUpdate(held=(), slow=False, focus=True))
    time.sleep(0.2)
    assert rig.request(StopEpisode(outcome="success")).ok
    d = rig.settings.data_root / "pick_block" / "episode_000"
    assert (d / "COMPLETE").exists() and (d / "meta.yaml").exists()
    assert rig.pc.request_names().count("c3.set_waypoints/v1") >= 5
    assert rig.r1.calls.count("r1.start_recording/v1") == 1 and rig.r1.calls.count("r1.stop_recording/v1") == 1
    s = rig.status()
    assert [e.name for e in s.episodes] == ["episode_000"] and s.c2_hz > 50 and s.rtt_ms is not None
    rig.loop.interrupt()
    rig._thread.join(5.0)
    assert rig.exit_code == 0 and rig.pc.request_names()[-1] == "c3.stop/v1"


def test_h2_another_client_driving_is_refused_without_c3(tmp_path, rigs):
    rig = rigs(tmp_path, link=FAST_LINK, teleop={"link_jitter_allowance_s": 0.05}, start_mode="holding")
    rig.pc.hold_forever()
    t0 = time.monotonic()
    rig.loop.startup(None)
    assert rig.loop.verdict.problem.code is C.PC_BUSY
    assert 1.0 - 0.05 <= time.monotonic() - t0 < 3.0                     # 2 x watchdog
    assert rig.pc.request_names() == []


def test_h2_an_orphaned_holding_turns_stopped_and_start_up_goes_on(tmp_path, rigs):
    rig = rigs(tmp_path, link=FAST_LINK, teleop={"link_jitter_allowance_s": 0.05}, start_mode="holding")
    rig.loop.startup(None)
    assert rig.loop.phase is Phase.IDLE, rig.loop.verdict
    states = [json.loads(x)["detail"] for x in (rig.session / "events.jsonl").read_text().splitlines()]
    assert "C2 mode holding" in states and "C2 mode stopped" in states


def test_pc_probe_check_rows_and_json_lines(tmp_path):
    pc = SocketPc()
    try:
        station = Station.load(write_station(tmp_path, sections={"teleop": FAST}), mock=True)
        settings = Settings.from_station(station, state_endpoint=pc.state_endpoint,
                                         commands_endpoint=pc.commands_endpoint)
        probe = PcProbe(settings, io.StringIO())
        try:
            rows = probe.check_rows(mock=True)
        finally:
            probe.close()
        by = {r.name: r for r in rows}
        for name in ("H2", "hello", "C2 rate", "hello equality", "worst gap", "waypoint lead",
                     "clock agreement"):
            assert by[name].status == "pass", by[name]
        assert by["PC commit"].status == "warn" and "differs" in by["PC commit"].detail
        assert "home timeout" not in by                                  # teleop sends no C3 home
        assert set(pc.request_names()) <= {"c3.hello/v1", "c3.heartbeat/v1"}
        buf = io.StringIO()
        emit(rows, True, buf)
        parsed = [json.loads(x) for x in buf.getvalue().splitlines()]
        assert len(parsed) == len(rows) and all(set(p) == {"name", "status", "detail"} for p in parsed)
    finally:
        pc.close()


def _run(args, timeout=30):
    import os
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env)


def test_the_process_fails_with_one_fatal_line_on_a_bad_station(tmp_path):
    station = write_station(tmp_path, sections={"teleop": {"motion": {"v_max_m_s": 1.0}}})
    r = _run(["-m", "teleop_collect.teleop", "--station", str(station), "--session", str(tmp_path / "s"),
              "--mock"])
    assert r.returncode == 1
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("FATAL teleop: config: ") and "v_max_m_s" in last


def test_pc_probe_check_json_on_a_bad_station_is_one_fail_row(tmp_path):
    bad = tmp_path / "station.yaml"
    bad.write_text("station_version: 7\n")
    r = _run(["-m", "teleop_collect.teleop.pc_probe", "--station", str(bad), "--check", "--json"])
    assert r.returncode == 0
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert len(rows) == 1 and rows[0]["name"] == "settings" and rows[0]["status"] == "fail"


def test_sigterm_with_an_episode_open_and_a_slow_r1_stop_exits_within_2_s(tmp_path, rigs):
    rig = rigs(tmp_path)
    ready = tmp_path / "teleop.ready"
    rig.run_in_thread(ready)
    assert wait_for(lambda: ready.exists(), 10.0), rig.loop.verdict
    assert rig.request(CreateTask(name="pick_block", description="")).ok
    assert wait_for(lambda: rig.status().start_allowed, 5.0), rig.status().start_refusal
    assert rig.request(StartEpisode()).ok
    rig.r1.stop_delay_s = 3.0                                           # frames' writer drains slowly
    t0 = time.monotonic()
    rig.loop.terminate()
    rig._thread.join(5.0)
    assert not rig._thread.is_alive() and time.monotonic() - t0 < 2.0   # abort wait 1 s, not frames_s 4 s
    assert rig.exit_code == 0 and rig.loop.verdict.problem.code is C.LAUNCHER_STOP
    assert rig.pc.request_names()[-1] == "c3.stop/v1"
    assert not (rig.settings.data_root / "pick_block" / "episode_000").exists()
