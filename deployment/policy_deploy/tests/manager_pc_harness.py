"""PcRig: the real manager pieces against an in-process FakePc on tcp://127.0.0.1:* (STRUCTURE.md 12.2).

C2 and C3 are real sockets (FakePc), C1 and C4 are in-process fakes (no sockets); the clock is the real one. The loop
runs on a background thread; tests press keys and wait for conditions with timeouts.
"""
from __future__ import annotations

import dataclasses
import io
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from deploy_contracts import LinkSettings
from manager_pc_fakepc import make_fake_pc, wait_for
from station_helpers import write_station

from policy_deploy.contracts import (ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, OBSERVATION_INPUTS, STEP_MS, ActionChunk,
                                     CameraHealthReport, FramesReply, HealthReport, PolicyConfig, Station)
from policy_deploy.manager.episode_loop import EpisodeLoop, Phase
from policy_deploy.manager.failure_monitor import FailureMonitor
from policy_deploy.manager.heartbeat_sender import HeartbeatSender
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.operator import Key, Operator
from policy_deploy.manager.pc_client import PcClient
from policy_deploy.manager.pc_clock import make_pc_clock
from policy_deploy.manager.pc_listener import PcListener
from policy_deploy.manager.policy_step import PolicyStep
from policy_deploy.manager.pose_converter import PoseConverter
from policy_deploy.manager.session_log import SessionLog
from policy_deploy.manager.settings import Settings
from policy_deploy.manager.startup_checks import StartupChecks
from policy_deploy.manager.state_history import StateHistory
from policy_deploy.manager.step_worker import StepWorker

COMMIT = "0123456789abcdef0123456789abcdef01234567"
FAST = {"clock": {"min_samples_s": 0.2, "min_samples": 5}, "timeouts": {"startup_s": 5.0, "pc_homing_wait_s": 3.0},
        "health_poll_ms": 100}
_IMAGES = np.zeros((OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def small_link(**change) -> LinkSettings:
    """link.yaml with some timing replaced (tests only; the manager and the fake PC get the same one)."""
    return dataclasses.replace(LinkSettings.load(), **change)


class QueueOperator(Operator):
    """Keys pressed by the test (thread-safe); records status lines and messages; a refused key is dropped."""

    def __init__(self) -> None:
        super().__init__(io.StringIO())
        self._lock = threading.Lock()
        self.keys: list[Key] = []
        self.messages: list[str] = []
        self.refused: list[tuple[Key, str]] = []
        self.started = False

    def press(self, *keys: str) -> None:
        with self._lock:
            self.keys.extend(Key(k) for k in keys)

    def start(self) -> None:
        self.started = True

    def poll_key(self, progress) -> Key | None:
        with self._lock:
            return self.keys.pop(0) if self.keys else None

    def key_refused(self, key: Key, hint: str) -> None:
        with self._lock:
            self.refused.append((key, hint))

    def show_status(self, line: str) -> None:
        pass

    def show_message(self, text: str) -> None:
        with self._lock:
            self.messages.append(text)


def camera(status: str = "ok") -> CameraHealthReport:
    return CameraHealthReport(status=status, last_frame_age_ms=20.0 if status == "ok" else 500.0,
                              fps=30.0 if status == "ok" else 0.0, dropped=0)


class FakeFrames:
    """C1 in-process: the newest main image 40 ms old (laptop clock), 66.7 ms apart; `status` sets both cameras."""

    def __init__(self) -> None:
        self.main = "ok"
        self.ultrawide = "ok"
        self.calls = 0

    def _health(self) -> HealthReport:
        return HealthReport(main=camera(self.main), ultrawide=camera(self.ultrawide))

    def get_frames(self) -> FramesReply:
        self.calls += 1
        h = self._health()
        if h.main.status != "ok" or h.ultrawide.status != "ok":
            return FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None, health=h)
        t = time.monotonic_ns() - 40_000_000
        main_t = np.array([t - 66_700_000, t], dtype=np.int64)
        uw_t = np.array([t - 100_000_000, t], dtype=np.int64)
        return FramesReply(main=_IMAGES, main_t_ns=main_t, ultrawide=_IMAGES, ultrawide_t_ns=uw_t, health=h)

    def get_health(self) -> HealthReport:
        return self._health()

    def close(self) -> None:
        pass


class FakePolicy:
    """C4 in-process: hold still (step 0 at the anchor), width 0.04; `delays` makes chosen infers slow."""

    def __init__(self, kind: str = "hold_still") -> None:
        self.kind = kind
        self.delays: list[float] = []          # popped per infer
        self.infers = 0

    def config(self) -> PolicyConfig:
        return PolicyConfig(obs_steps=OBS_STEPS, action_steps=ACTION_STEPS, step_ms=STEP_MS, image_size=IMAGE_SIZE,
                            inputs=OBSERVATION_INPUTS, uses_prompt=False, kind=self.kind)

    def reset(self) -> None:
        pass

    def prompt(self, task_name: str, prompt_index: int) -> None:
        pass

    def infer(self, obs) -> ActionChunk:
        self.infers += 1
        if self.delays:
            time.sleep(self.delays.pop(0))
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), (ACTION_STEPS, 1))
        return ActionChunk(eef_pos=np.zeros((ACTION_STEPS, 3), np.float32), eef_rot6d=rot,
                           gripper_width_m=np.full((ACTION_STEPS, 1), 0.04, np.float32), width_clipped=0)

    def close(self) -> None:
        pass


class PcRig:
    def __init__(self, tmp_path: Path, manager: dict | None = None, fake: dict | None = None,
                 link: LinkSettings | None = None, fake_link: LinkSettings | None = None, kind: str = "hold_still",
                 state_endpoint: str | None = None, start_fake: bool = True) -> None:
        tmp_path = Path(tmp_path)
        self.fake = make_fake_pc(tmp_path / "fake_pc_cfg", link=fake_link or link,
                                 **merge({"home_duration_s": 0.3}, fake or {}))
        if start_fake:
            self.fake.start()
        self.t_start_ns = time.monotonic_ns()
        station = Station.load(write_station(tmp_path, manager=merge(FAST, manager or {})), mock=True)
        s = Settings.from_station(station, link=link, state_endpoint=state_endpoint or self.fake.state_endpoint,
                                  commands_endpoint=self.fake.commands_endpoint)
        self.settings = s
        clock = time.monotonic_ns
        self.stats = LinkStats(s.link_stats_window_s, clock)
        self.pc_clock = make_pc_clock(s.clock.estimator, s.clock, clock)
        self.history = StateHistory(s.history_s)
        self.listener = PcListener(s.state_endpoint, self.history, self.pc_clock, self.stats, s.state_recv_hwm,
                                   s.listener_poll_s, clock)
        self.pc = PcClient(s.commands_endpoint, s.c3_timeout_s, s.timeouts.home_s, s.heartbeat_period_s, self.stats,
                           clock)
        self.heartbeats = HeartbeatSender(self.pc, self.pc_clock, self.stats, s.clock.hello_period_s,
                                          s.clock.startup_hello_period_s, clock)
        self.frames = FakeFrames()
        self.policy = FakePolicy(kind)
        step = PolicyStep(self.frames, self.policy, self.history, PoseConverter(s.exec_h, s.lookahead_ms), s, clock,
                          self.pc_clock)
        self.worker = StepWorker(step, self.policy, clock)
        self.monitor = FailureMonitor(s.state_lost_s)
        self.operator = QueueOperator()
        self.session = tmp_path / "session" / "manager"
        self.log = SessionLog(self.session, datetime.now, clock)
        self.checks = StartupChecks(s, mock=True, commit=COMMIT)
        self.loop = EpisodeLoop(s, self.pc, self.heartbeats, self.pc_clock, self.stats, self.frames, self.worker,
                                self.history, self.listener, self.monitor, self.operator, self.log, self.checks,
                                clock, time.sleep)
        self._thread: threading.Thread | None = None
        self.exit_code: int | None = None

    # ---- running

    def startup(self, ready_file: Path | None = None) -> None:
        self.loop.startup(ready_file)

    def run_in_thread(self) -> None:
        def body() -> None:
            self.exit_code = self.loop.run()
        self._thread = threading.Thread(target=body, name="test-loop", daemon=True)
        self._thread.start()

    def start(self) -> None:
        """Start-up (asserting it reached IDLE), then the loop on its thread."""
        self.startup()
        assert self.loop.phase is Phase.IDLE, self.verdict_text()
        self.run_in_thread()

    def wait(self, condition, timeout_s: float = 5.0) -> bool:
        return wait_for(condition, timeout_s)

    def wait_done(self, timeout_s: float = 8.0) -> None:
        assert self.wait(lambda: self.loop.phase is Phase.DONE, timeout_s), f"still {self.loop.phase}"
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def start_episode(self, steps: int = 3) -> None:
        self.operator.press("c")
        assert self.wait(lambda: self.loop.phase is Phase.RUNNING, 5.0), self.verdict_text()
        assert self.wait(lambda: self.loop._steps_sent >= steps, 5.0), self.verdict_text()

    def verdict_text(self) -> str:
        v = self.loop.verdict
        return "" if v is None else f"{v.problem.code}: {v.what_failed}: {v.what_to_check}"

    @property
    def code(self):
        v = self.loop.verdict
        return None if v is None else v.problem.code

    def requests(self) -> list[str]:
        return self.fake.request_names()

    def events(self) -> list[dict]:
        import json
        path = self.session / "events.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def episodes(self) -> list[dict]:
        import json
        path = self.session / "episodes.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def close(self) -> None:
        if self.loop.phase is not Phase.DONE:
            if self._thread is None and self.loop.phase is not Phase.STARTING:
                self.run_in_thread()                     # only start-up ran: let the loop see the q
            self.operator.press("q")
            self.wait(lambda: self.loop.phase is Phase.DONE, 3.0)
            if self.loop.phase is not Phase.DONE:
                self.loop.terminate()
                self.wait(lambda: self.loop.phase is Phase.DONE, 3.0)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        for close in (self.listener.stop, self.worker.stop, self.pc.close, self.log.close, self.fake.close):
            try:
                close()
            except Exception:
                pass
