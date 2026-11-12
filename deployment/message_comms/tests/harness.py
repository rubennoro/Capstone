"""LoopHarness: an EpisodeLoop wired to fakes, a FakeClock and a real StateHistory / PolicyStep / FailureMonitor."""
from __future__ import annotations

from datetime import datetime

from fakes import (FakeClock, FakeFrankaClient, FakeFramesClient, FakeListener, FakePolicyClient, ManualWorker,
                   QueueOperator, make_state, pose)
from message_comms.episode_loop import EpisodeLoop, Phase
from message_comms.failure_monitor import FailureMonitor
from message_comms.policy_step import PolicyStep
from message_comms.pose_converter import PoseConverter
from message_comms.session_log import SessionLog
from message_comms.state_history import StateHistory


class LoopHarness:
    def __init__(self, settings, operator=None) -> None:
        self.settings = settings
        self.clock = FakeClock()
        self.calls: list = []                    # franka and policy calls, in order
        self.franka = FakeFrankaClient(self.clock, self.calls, settings.loop_ms / 1000)
        self.health = FakeFramesClient(self.clock)
        self.frames = FakeFramesClient(self.clock)
        self.policy = FakePolicyClient(self.calls)
        self.history = StateHistory(settings.history_s)
        self.listener = FakeListener(self.history)
        self.step = PolicyStep(self.frames, self.policy, self.history,
                               PoseConverter(settings.exec_h, settings.lookahead_ms), settings, self.clock)
        self.worker = ManualWorker(self.clock, self.policy, self.step)
        self.worker.sleep = self._sleep
        self.monitor = FailureMonitor(settings)
        self.operator = operator if operator is not None else QueueOperator()
        self.log = SessionLog(settings.session_dir, datetime.now, self.clock)
        self.loop = EpisodeLoop(settings, self.franka, self.health, self.worker, self.history, self.listener,
                                self.monitor, self.operator, self.log, self.clock, self._sleep)
        self.seq = 0
        self.T = pose()
        self.target = None
        self.mode_override: str | None = None
        self.gate = dict(shortened=0, pulled_back=0, loop_late=0)
        self.publish = True

    def push_state(self, **kw) -> None:
        self.seq += 1
        values = dict(T=self.T, target=self.target, mode=self.mode_override or self.franka.mode, **self.gate)
        values.update(kw)
        self.listener.push(make_state(self.seq, self.clock(), **values))

    def _sleep(self, seconds: float) -> None:
        """The loop's sleep: fake time passes, and franka_control keeps broadcasting at 100 Hz meanwhile."""
        end = self.clock() + int(seconds * 1e9)
        while self.clock() < end:
            self.clock.advance(min(0.01, (end - self.clock()) / 1e9))
            if self.publish:
                self.push_state()

    def start(self, ready_file=None) -> None:
        """0.5 s of state history first (franka_control was already broadcasting), then startup."""
        self._sleep(0.5)
        self.loop.startup(ready_file)

    def tick(self, n: int = 1) -> None:
        """n loop cycles, loop_ms apart, with a state message before each."""
        for _ in range(n):
            if self.loop.phase is Phase.DONE:
                return
            self._sleep(self.settings.loop_ms / 1000)
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
