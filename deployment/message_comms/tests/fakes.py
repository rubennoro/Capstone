"""Test fakes: FakeClock, message builders, FakeFramesClient, FakePolicyClient, FakeFrankaClient, FakeListener,
ManualWorker and QueueOperator. Same methods as the real classes; no sockets."""
from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np

from contracts import (ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, Ack, ActionChunk, ArmState, CameraHealthReport,
                       FramesReply, GateCounters, HealthReport, PolicyConfig, SetWaypointsReply, WaypointIssue)
from message_comms.errors import Refused, RequestCancelled
from message_comms.operator import Key, Operator
from message_comms.step_worker import WorkerResult

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


# ---- clock ---------------------------------------------------------------------------------------------------------

class FakeClock:
    """A monotonic ns clock driven by the test. sleep() advances it."""

    def __init__(self, start_ns: int = 1_000 * NS_PER_S) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += int(round(seconds * NS_PER_S))

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


# ---- builders ------------------------------------------------------------------------------------------------------

def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def pose(p=(0.5, 0.0, 0.3), R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = p
    return T


def random_pose(rng: np.random.Generator) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-1.0, 1.0, 3)
    return T


def make_state(seq: int, t_ns: int, T=None, target=None, width: float = 0.05, mode: str = "running",
               error: str = "", shortened: int = 0, pulled_back: int = 0, loop_late: int = 0) -> ArmState:
    T = pose() if T is None else T
    target = T if target is None else target
    return ArmState(t_ns=int(t_ns), seq=int(seq), O_T_TCP=np.array(T, dtype=np.float64),
                    O_T_TCP_target=np.array(target, dtype=np.float64), q=np.zeros(7), dq=np.zeros(7),
                    gripper_width_m=float(width), mode=mode, error=error,
                    gate=GateCounters(accepted=max(shortened, pulled_back), shortened=shortened,
                                      pulled_back=pulled_back, rejected=0),
                    loop_late=int(loop_late))


def camera(status: str = "ok", age_ms: float = 20.0, fps: float = 30.0) -> CameraHealthReport:
    return CameraHealthReport(status=status, last_frame_age_ms=float(age_ms), fps=float(fps), dropped=0)


def health(main: str = "ok", ultrawide: str = "ok") -> HealthReport:
    return HealthReport(main=camera(main, 20.0 if main == "ok" else 400.0), ultrawide=camera(ultrawide))


def images(value: int = 7) -> np.ndarray:
    return np.full((OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)


def make_frames(main_t_ns=None, report: HealthReport | None = None) -> FramesReply:
    report = health() if report is None else report
    ok = report.main.status == "ok" and report.ultrawide.status == "ok"
    if not ok:
        return FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None, health=report)
    t = np.asarray(main_t_ns, dtype=np.int64)
    uw = np.array([t[1] - 100 * NS_PER_MS, t[1]], dtype=np.int64)
    return FramesReply(main=images(1), main_t_ns=t, ultrawide=images(2), ultrawide_t_ns=uw, health=report)


def make_chunk(pos=None, rot=None, width: float = 0.04, width_clipped: int = 0) -> ActionChunk:
    """A chunk; default: hold still at the anchor. pos (16, 3), rot (16, 3, 3) relative rotations."""
    eef_pos = np.zeros((ACTION_STEPS, 3)) if pos is None else np.asarray(pos, dtype=np.float64)
    R = np.tile(np.eye(3), (ACTION_STEPS, 1, 1)) if rot is None else np.asarray(rot, dtype=np.float64)
    rot6d = R[:, :2, :].reshape(ACTION_STEPS, 6)
    return ActionChunk(eef_pos=eef_pos.astype(np.float32), eef_rot6d=rot6d.astype(np.float32),
                       gripper_width_m=np.full((ACTION_STEPS, 1), width, dtype=np.float32),
                       width_clipped=int(width_clipped))


def policy_config(**overrides) -> PolicyConfig:
    values = dict(obs_steps=OBS_STEPS, action_steps=ACTION_STEPS, step_ms=STEP_MS, image_size=IMAGE_SIZE,
                  inputs=("eef_pos", "eef_rot6d", "main"), uses_prompt=False)
    values.update(overrides)
    return PolicyConfig(**values)


# ---- clients -------------------------------------------------------------------------------------------------------

class FakeFramesClient:
    """Frames newest at clock() - latency, 50 ms apart. `report` sets the cameras; `error` makes calls raise."""

    def __init__(self, clock: Callable[[], int], latency_ms: float = 40.0) -> None:
        self.clock = clock
        self.latency_ms = latency_ms
        self.report = health()
        self.health_report: HealthReport | None = None      # get_health's answer (None: same as report)
        self.fixed_t_ns: tuple[int, int] | None = None       # serve these times instead of clock-based ones
        self.error: BaseException | None = None
        self.calls: list[str] = []

    def get_frames(self) -> FramesReply:
        self.calls.append("get_frames")
        if self.error is not None:
            raise self.error
        if self.fixed_t_ns is not None:
            t = self.fixed_t_ns
        else:
            t1 = self.clock() - int(self.latency_ms * NS_PER_MS)
            t = (t1 - STEP_MS * NS_PER_MS, t1)
        return make_frames(t, self.report)

    def get_health(self) -> HealthReport:
        self.calls.append("get_health")
        if self.error is not None:
            raise self.error
        return self.health_report if self.health_report is not None else self.report

    def close(self) -> None:
        pass


class FakePolicyClient:
    def __init__(self, calls: list | None = None) -> None:
        self.calls = [] if calls is None else calls
        self.config_reply = policy_config()
        self.chunk: Callable[[object], ActionChunk] = lambda obs: make_chunk()
        self.error: BaseException | None = None
        self.reset_ack = True
        self.observations: list = []

    def config(self) -> PolicyConfig:
        self.calls.append("config")
        return self.config_reply

    def reset(self) -> None:
        self.calls.append("reset")
        if not self.reset_ack:
            raise Refused("policy", "c4.reset/v1", "no")

    def prompt(self, task_name: str, prompt_index: int) -> None:
        self.calls.append(("prompt", task_name, prompt_index))

    def infer(self, obs) -> ActionChunk:
        self.calls.append("infer")
        self.observations.append(obs)
        if self.error is not None:
            raise self.error
        return self.chunk(obs)

    def close(self) -> None:
        pass


class FakeFrankaClient:
    """Records every call with the clock time; keeps the mode franka_control would broadcast."""

    def __init__(self, clock: FakeClock, calls: list | None = None, poll_s: float = 0.02) -> None:
        self.clock = clock
        self.calls = [] if calls is None else calls
        self.timed: list[tuple[str, int]] = []
        self.mode = "idle"
        self.start_ack = Ack(ok=True, reason="")
        self.home_behaviour = "ok"          # "ok" | "refuse" | "hang"
        self.home_s = 1.0                   # how long a successful home takes (fake time)
        self.poll_s = poll_s
        self.home_closed = False
        self.reply: Callable | None = None   # set_waypoints(msg) -> SetWaypointsReply; default: all accepted
        self.sent: list = []
        self.error: dict[str, BaseException] = {}
        self.during_home: Callable[[], None] | None = None     # called every slice while homing

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.timed.append((name, self.clock()))
        if name in self.error:
            raise self.error[name]

    def heartbeat(self) -> None:
        self._record("heartbeat")

    def start(self) -> Ack:
        self._record("start")
        if self.start_ack.ok:
            self.mode = "holding"
        return self.start_ack

    def stop(self) -> None:
        self._record("stop")
        self.mode = "stopped"

    def home(self, interrupted: Callable[[], bool]) -> None:
        self._record("home")
        self.mode = "homing"
        if self.home_behaviour == "refuse":
            self.mode = "fault"
            raise Refused("franka_control", "c3.home/v1", "home move would take 34 s")
        waited = 0.0
        while self.home_behaviour == "hang" or waited < self.home_s:
            self.clock.advance(self.poll_s)
            waited += self.poll_s
            if self.during_home is not None:
                self.during_home()
            if interrupted():
                self.home_closed = True
                raise RequestCancelled("franka_control", "c3.home/v1")
        self.mode = "idle"

    def set_waypoints(self, msg) -> SetWaypointsReply:
        self._record("set_waypoints")
        self.sent.append((self.clock(), msg))
        self.mode = "running"
        if self.reply is not None:
            return self.reply(msg)
        n = len(msg.t_ns)
        past = int(np.sum(msg.t_ns < self.clock()))
        return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=n - past, shortened=0, skipped=past, rejected=())

    def get_state(self):
        self._record("get_state")
        return make_state(1, self.clock(), mode=self.mode)

    def close(self) -> None:
        pass


def rejecting_reply(msg) -> SetWaypointsReply:
    n = len(msg.t_ns)
    return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=n - 1, shortened=0, skipped=0,
                             rejected=(WaypointIssue(index=3, reason="outside the box by 4 cm"),))


class FakeListener:
    """Pushes states into a real StateHistory, as the C2 thread would."""

    def __init__(self, history) -> None:
        self.history = history
        self._error: BaseException | None = None
        self._received = 0
        self.started = False
        self.stopped = False

    def push(self, state: ArmState) -> None:
        try:
            self.history.add(state)
            self._received += 1
        except Exception as exc:
            self._error = exc

    def set_error(self, exc: BaseException) -> None:
        self._error = exc

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def error(self) -> BaseException | None:
        return self._error

    def received(self) -> int:
        return self._received


# ---- worker --------------------------------------------------------------------------------------------------------

class ManualWorker:
    """StepWorker stand-in. A job's result becomes available `delay_s` (fake time) after it is submitted; with
    `hold` true it waits for release(). Steps run through `step` (a real PolicyStep) when their result is made.
    wait(timeout) advances the fake clock by timeout when nothing is ready."""

    def __init__(self, clock: FakeClock, policy: FakePolicyClient, step=None) -> None:
        self.clock = clock
        self.policy = policy
        self.step = step
        self.delay_s = 0.0
        self.hold = False
        self.jobs: list[tuple[str, object, int]] = []
        self._pending: tuple[str, object, int] | None = None
        self._released = False
        self.started = False
        self.stopped = False
        self.sleep: Callable[[float], None] = clock.advance     # how waiting lets fake time pass

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def busy(self) -> bool:
        return self._pending is not None

    def submit_config(self) -> None:
        self._submit("config", None)

    def submit_reset(self, prompt) -> None:
        self._submit("reset", prompt)

    def submit_step(self, job) -> None:
        self._submit("step", job)

    def _submit(self, kind: str, payload) -> None:
        if self._pending is not None:
            raise RuntimeError("busy")
        self._pending = (kind, payload, self.clock())
        self._released = False
        self.jobs.append(self._pending)

    def release(self) -> None:
        self._released = True

    def pending_kind(self) -> str | None:
        return None if self._pending is None else self._pending[0]

    def _ready(self) -> bool:
        if self._pending is None:
            return False
        if self.hold and self._pending[0] == "step":
            return self._released
        return self._released or self.clock() - self._pending[2] >= int(self.delay_s * NS_PER_S)

    def wait(self, timeout_s: float):
        if not self._ready():
            if timeout_s > 0:
                self.sleep(timeout_s)
            if not self._ready():
                return None
        kind, payload, started = self._pending
        self._pending = None
        value = error = None
        where = ("policy", f"c4.{kind}/v1")
        try:
            if kind == "config":
                value = self.policy.config()
            elif kind == "reset":
                self.policy.reset()
                if payload is not None:
                    self.policy.prompt(*payload)
            else:
                value = self.step.run(payload)
        except Exception as exc:
            error = exc
            if kind == "step":
                where = self.step.last_call
        return WorkerResult(kind, value, error, where[0], where[1], started, self.clock())

    def running_for_ms(self) -> float | None:
        return None if self._pending is None else (self.clock() - self._pending[2]) / NS_PER_MS


# ---- operator ------------------------------------------------------------------------------------------------------

class QueueOperator(Operator):
    """Keys pressed by the test; records status lines and messages."""

    def __init__(self) -> None:
        super().__init__(io.StringIO())
        self.keys: list[Key] = []
        self.statuses: list[str] = []
        self.messages: list[str] = []
        self.started = False
        self.closed = False

    def press(self, *keys: str) -> None:
        self.keys.extend(Key(k) for k in keys)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def poll_key(self, progress) -> Key | None:
        return self.keys.pop(0) if self.keys else None

    def show_status(self, line: str) -> None:
        self.statuses.append(line)

    def show_message(self, text: str) -> None:
        self.messages.append(text)
