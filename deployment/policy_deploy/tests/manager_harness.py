# adapted from rnoro/deployment:deployment/message_comms/tests/harness.py @ 91b178b (renamed manager_harness)
# changes: wired to the new EpisodeLoop: FakePcClient (fake PC clock = laptop + 1000 s), a real RoundTripPcClock,
# HeartbeatSender, LinkStats and StartupChecks; states are pushed with PC-clock t_ns; the cycle is link.yaml's
# heartbeat period; the session folder is session_dir(settings); conftest-style settings helpers.
"""LoopHarness: an EpisodeLoop wired to fakes, a FakeClock and a real StateHistory / PolicyStep / FailureMonitor /
PC clock map / HeartbeatSender."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from manager_fakes import (COMMIT, FakeClock, FakeFramesClient, FakeListener, FakePcClient, FakePolicyClient,
                           ManualWorker, QueueOperator, make_state, pose)
from station_helpers import write_station

from policy_deploy.manager.episode_loop import EpisodeLoop, Phase
from policy_deploy.manager.failure_monitor import FailureMonitor
from policy_deploy.manager.heartbeat_sender import HeartbeatSender
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_clock import make_pc_clock
from policy_deploy.manager.policy_step import PolicyStep
from policy_deploy.manager.pose_converter import PoseConverter
from policy_deploy.manager.session_log import SessionLog
from policy_deploy.manager.settings import Settings
from policy_deploy.manager.startup_checks import StartupChecks
from policy_deploy.manager.state_history import StateHistory


def load_settings(tmp_path: Path, mock: bool = True, **overrides) -> Settings:
    """Settings from a temporary station whose manager: section is `overrides`."""
    return Settings.load(write_station(tmp_path, manager=overrides), mock=mock)


def session_dir(settings: Settings) -> Path:
    """Where a LoopHarness writes its session files."""
    return settings.session_root / "run" / "manager"


def episodes(settings: Settings) -> list[dict]:
    path = session_dir(settings) / "episodes.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def events(settings: Settings) -> str:
    path = session_dir(settings) / "events.jsonl"
    return path.read_text() if path.exists() else ""


class LoopHarness:
    def __init__(self, settings: Settings, operator=None, mock: bool = True) -> None:
        self.settings = settings
        self.clock = FakeClock()
        self.calls: list = []                    # PC and policy calls, in order
        self.pc = FakePcClient(self.clock, self.calls, settings.heartbeat_period_s)
        self.franka = self.pc                    # the old name, kept for the ported tests
        self.health = FakeFramesClient(self.clock)
        self.frames = FakeFramesClient(self.clock)
        self.policy = FakePolicyClient(self.calls)
        self.history = StateHistory(settings.history_s)
        self.stats = LinkStats(settings.link_stats_window_s, self.clock)
        self.pc_clock = make_pc_clock(settings.clock.estimator, settings.clock, self.clock)
        self.listener = FakeListener(self.history, self.clock, self.pc_clock)
        self.heartbeats = HeartbeatSender(self.pc, self.pc_clock, self.stats, settings.clock.hello_period_s,
                                          settings.clock.startup_hello_period_s, self.clock)
        self.step = PolicyStep(self.frames, self.policy, self.history,
                               PoseConverter(settings.exec_h, settings.lookahead_ms), settings, self.clock,
                               self.pc_clock)
        self.worker = ManualWorker(self.clock, self.policy, self.step)
        self.worker.sleep = self._sleep
        self.monitor = FailureMonitor(settings.state_lost_s)
        self.operator = operator if operator is not None else QueueOperator()
        self.log = SessionLog(session_dir(settings), datetime.now, self.clock)
        self.checks = StartupChecks(settings, mock=mock, commit=COMMIT)
        self.loop = EpisodeLoop(settings, self.pc, self.heartbeats, self.pc_clock, self.stats, self.health,
                                self.worker, self.history, self.listener, self.monitor, self.operator, self.log,
                                self.checks, self.clock, self._sleep)
        self.pc.during_home = self._home_slice
        self.seq = 0
        self.T = pose()
        self.target = None
        self.mode_override: str | None = None
        self.gate = dict(shortened=0, pulled_back=0, loop_late=0)
        self.publish = True
        self._next_state_ns = 0

    def push_state(self, **kw) -> None:
        self.seq += 1
        values = dict(T=self.T, target=self.target, mode=self.mode_override or self.pc.mode, **self.gate)
        values.update(kw)
        self.listener.push(make_state(self.seq, self.pc.pc_ns(), **values))

    def _home_slice(self) -> None:
        """The PC keeps broadcasting while a C3 home blocks."""
        if self.publish:
            self.push_state()

    def _sleep(self, seconds: float) -> None:
        """The loop's sleep: fake time passes, and the PC keeps broadcasting at 100 Hz meanwhile."""
        end = self.clock() + int(seconds * 1e9)
        while self.clock() < end:
            self.clock.advance(min(0.01, (end - self.clock()) / 1e9))
            if self.publish:
                self.push_state()

    def start(self, ready_file=None) -> None:
        """0.5 s of state history first (the PC was already broadcasting), then start-up."""
        self._sleep(0.5)
        self.loop.startup(ready_file)

    def tick(self, n: int = 1) -> None:
        """n loop cycles, one heartbeat period apart, with states in between."""
        for _ in range(n):
            if self.loop.phase is Phase.DONE:
                return
            self._sleep(self.settings.heartbeat_period_s)
            self.loop.cycle()

    def run_until(self, condition, max_cycles: int = 500) -> None:
        for _ in range(max_cycles):
            if condition():
                return
            self.tick()
        raise AssertionError("condition not reached")

    def start_episode(self) -> None:
        self.operator.press("c")
        self.run_until(lambda: self.loop.phase is Phase.RUNNING)

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c == name)

    def keepalives(self) -> int:
        return self.count("heartbeat") + self.count("hello")
