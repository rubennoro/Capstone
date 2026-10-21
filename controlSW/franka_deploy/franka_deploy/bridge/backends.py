"""Bridge backends: the thing that owns the 1 kHz Franka loop.

Interface (all poses are 4x4 O_T_EE of the frame the robot controls):
    start()                      connect / start the Cartesian impedance controller holding the current pose
    set_target(T)                latest safe target (non-blocking)
    get_state() -> dict          {"O_T_EE", "q", "dq", "O_F_ext", "mode", "success_rate", "error", "t"}
    move_joints(q, speed)        blocking joint move (homing)
    stop()                       stop the controller (arm holds by brakes / idles); requires start() or recover()
    recover()                    clear a reflex (operator command only)
    inject_reflex(msg)           FakeBackend only (tests)
    loop_stats() -> dict         1 kHz loop since the last call: period percentiles, per-cycle compute time
                                 percentiles (cycle_*) and a compute-time histogram (cycle_hist_us, 50 us bins,
                                 last bin = overflow >= 5 ms) so a run's overall p99.9 can be aggregated exactly
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


HIST_BIN_US, HIST_BINS = 50, 101   # 0-5 ms in 50 us bins + overflow


def cycle_hist(c_s: np.ndarray) -> dict:
    """Per-cycle compute times (s) -> percentiles + histogram (see loop_stats)."""
    us = np.asarray(c_s, float) * 1e6
    h = np.bincount(np.minimum((us // HIST_BIN_US).astype(int), HIST_BINS - 1), minlength=HIST_BINS)
    return {"cycle_n": int(us.size), "cycle_p50_ms": float(np.percentile(us, 50) / 1e3),
            "cycle_p999_ms": float(np.percentile(us, 99.9) / 1e3), "cycle_max_ms": float(us.max() / 1e3),
            "cycle_hist_us": h.tolist()}


def _quat(Rm) -> tuple:
    x, y, z, w = Rotation.from_matrix(Rm).as_quat()
    return (w, x, y, z)


def _slerp(q0, q1, a) -> tuple:
    d = sum(i * j for i, j in zip(q0, q1))
    if d < 0:
        q1, d = tuple(-v for v in q1), -d
    if d > 0.9995:
        q = tuple(i + a * (j - i) for i, j in zip(q0, q1))
    else:
        th = math.acos(min(1.0, d))
        s0, s1 = math.sin((1 - a) * th) / math.sin(th), math.sin(a * th) / math.sin(th)
        q = tuple(s0 * i + s1 * j for i, j in zip(q0, q1))
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def _quat2mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Backend:
    def start(self): ...
    def set_target(self, T): ...
    def get_state(self) -> dict: ...
    def move_joints(self, q, speed=0.2): ...
    def stop(self): ...
    def recover(self): ...
    def loop_stats(self) -> dict: return {}
    def close(self): self.stop()


class FakeBackend(Backend):
    """1 kHz thread: the pose follows the target with a first-order lag tau (position lerp, rotation slerp), plus a
    constant base-frame bias `bias_m` (impedance steady-state error). A reflex stops tracking and sets `error`."""

    def __init__(self, home_O_T_EE: np.ndarray, tau_s: float = 0.03, bias_m=(0.0, 0.0, 0.0), rate_hz: float = 1000.0):
        T = np.array(home_O_T_EE, float)
        self._p, self._q = T[:3, 3].copy(), _quat(T[:3, :3])        # measured pose (loop state)
        self._tp, self._tq = self._p.copy(), self._q                  # target
        self.tau, self.bias = float(tau_s), np.asarray(bias_m, float)
        self.period = 1.0 / rate_hz
        self.mode = "idle"
        self.error = None
        self._lock = threading.Lock()
        self._run = True
        self._periods: list[float] = []
        self._cycles: list[float] = []
        self.q = np.zeros(7)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        t_prev = time.monotonic()
        nxt = t_prev
        while self._run:
            nxt += self.period
            time.sleep(max(0.0, nxt - time.monotonic()))
            now = time.monotonic()
            dt, t_prev = now - t_prev, now
            c0 = time.perf_counter()
            with self._lock:
                self._periods.append(dt)
                if len(self._periods) > 20000:
                    del self._periods[:10000]
                if self.mode == "running" and not self.error:
                    # plain-float math (no scipy objects per cycle): keeps the cycle well under 0.1 ms in CPython
                    a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
                    self._p = self._p + a * (self._tp + self.bias - self._p)
                    self._q = _slerp(self._q, self._tq, a)
                self._cycles.append(time.perf_counter() - c0)   # compute time of this cycle (incl. lock wait)
                if len(self._cycles) > 20000:
                    del self._cycles[:10000]
            if now - nxt > 0.05:
                nxt = now

    def start(self):
        with self._lock:
            if self.error:
                raise RuntimeError(f"cannot start: reflex active ({self.error}); recover first")
            self._tp, self._tq = self._p.copy(), self._q
            self.mode = "running"

    def set_target(self, T):
        with self._lock:
            if self.mode == "running" and not self.error:
                T = np.asarray(T, float)
                self._tp, self._tq = T[:3, 3].copy(), _quat(T[:3, :3])

    def get_state(self):
        with self._lock:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _quat2mat(self._q), self._p
            return {"O_T_EE": T, "q": self.q.copy(), "dq": np.zeros(7), "O_F_ext": np.zeros(6),
                    "mode": self.mode, "success_rate": 1.0, "error": self.error, "t": time.monotonic()}

    def move_joints(self, q, speed=0.2):
        with self._lock:
            self.q = np.asarray(q, float)

    def stop(self):
        with self._lock:
            self.mode = "stopped"

    def recover(self):
        with self._lock:
            self.error = None
            self.mode = "idle"

    def inject_reflex(self, msg="cartesian_reflex (injected)"):
        with self._lock:
            self.error = msg
            self.mode = "reflex"

    def loop_stats(self):
        with self._lock:
            p = np.array(self._periods)
            c = np.array(self._cycles)
            self._periods, self._cycles = [], []
        if p.size == 0:
            return {}
        out = {"n": int(p.size), "p50_ms": float(np.percentile(p, 50) * 1e3), "p99_ms": float(np.percentile(p, 99) * 1e3),
               "p999_ms": float(np.percentile(p, 99.9) * 1e3), "max_ms": float(p.max() * 1e3)}
        if c.size:
            out.update(cycle_hist(c))
        return out

    def close(self):
        self._run = False
        self._th.join(1.0)


def _errors_text(errs) -> str:
    """Non-empty description of a libfranka ``franka::Errors`` value (panda-py RobotState.last_motion_errors /
    current_errors), "" when it holds no error. Accepts what the binding may expose: a falsy/None value, a string
    such as "[]" / "[joint_reflex]" (libfranka's operator std::string), or an object with bool flag attributes."""
    if errs is None or errs is False:
        return ""
    if isinstance(errs, str):
        t = errs.strip()
        return "" if t in ("", "[]", "None") else t
    flags = [k for k in dir(errs) if not k.startswith("_") and isinstance(getattr(errs, k, None), bool)]
    if flags:
        return ", ".join(k for k in flags if getattr(errs, k))
    if isinstance(errs, (list, tuple, set)):
        return ", ".join(str(e) for e in errs)
    t = str(errs).strip()
    return "" if t in ("", "[]", "None") or " object at 0x" in t else t


def derive_panda_error(state, controller_expected: bool, started_s_ago: float = 1e9, grace_s: float = 0.5,
                       last_at_start: str = ""):
    """Bridge error string from a panda-py / libfranka RobotState, or None.

    reflex mode (robot_mode Reflex) > non-empty current_errors > last_motion_errors > the controller thread having
    stopped unexpectedly (a controller was started more than ``grace_s`` ago but robot_mode is no longer Move).
    libfranka keeps last_motion_errors after automaticErrorRecovery until the next motion starts, so that field only
    counts while a controller is expected to run and when it differs from its value at start() (``last_at_start``);
    otherwise a stale value would re-latch the error after recover() and at bridge boot, and start() could never run."""
    mode = getattr(state, "robot_mode", None)
    mode_s = str(getattr(mode, "name", mode))
    cur = _errors_text(getattr(state, "current_errors", None))
    last = _errors_text(getattr(state, "last_motion_errors", None))
    if "reflex" in mode_s.lower():
        return f"reflex (robot_mode {mode_s}): {cur or last or 'no error flags'}"
    if cur:
        return f"current_errors: {cur}"
    if last and controller_expected and last != last_at_start:
        return f"last_motion_errors: {last}"
    if controller_expected and started_s_ago > grace_s and "move" not in mode_s.lower():
        return f"controller stopped unexpectedly (robot_mode {mode_s})"
    return None


class PandaPyBackend(Backend):
    """panda-py CartesianImpedance backend. Hardware only (see docs/panda_py_checklist.md); the hardware calls are
    excluded from coverage, the error derivation is tested with a fake ``panda_py`` module (tests/test_sim_backend_stack.py).

    Pins: NOT fixed yet (decision B0: Panda vs FR3 and its system version). panda-py must be built against the
    libfranka version that matches the robot's system version per Franka's compatibility table (Panda / system 4.x
    -> libfranka 0.9.x; FR3 / system 5.x -> libfranka >= 0.10). Record the exact panda-py + libfranka versions in
    configs/bridge.yaml (backend.panda_py_version / libfranka_version) once chosen. aarch64 (Jetson): source build.
    The parameter names below are from panda-py's documented API and must be re-checked against the pinned version.
    panda-py names used: panda_py.Panda(hostname), Panda.get_state() -> franka.RobotState (O_T_EE column-major 16,
    q, dq, O_F_ext_hat_K, control_command_success_rate, robot_mode, current_errors, last_motion_errors),
    Panda.start_controller(ctrl), Panda.stop_controller(), Panda.recover(), Panda.move_to_joint_position(q,
    speed_factor), Panda.get_robot().set_collision_behavior(...), Panda.get_robot().set_EE(NE_T_EE),
    controllers.CartesianImpedance(impedance=6x6, damping_ratio, nullspace_stiffness, filter_coeff),
    CartesianImpedance.set_control(position (3,), orientation xyzw (4,)).
    error: derived on every get_state (derive_panda_error); latched until recover() (automatic_error_recovery).
    """

    def __init__(self, hostname: str, impedance=None, damping_ratio: float = 1.0, nullspace_stiffness: float = 0.5,
                 filter_coeff: float = 1.0, collision=None, NE_T_EE=None):
        import panda_py  # noqa: F401
        from panda_py import controllers

        self.panda_py, self.controllers = panda_py, controllers
        self.panda = panda_py.Panda(hostname)
        robot = self.panda.get_robot()
        if collision is not None:  # pragma: no cover - hardware call
            robot.set_collision_behavior(*collision)
        if NE_T_EE is not None:  # pragma: no cover - hardware call
            robot.set_EE(np.asarray(NE_T_EE, float).T.flatten().tolist())
        self.ctrl = controllers.CartesianImpedance(impedance=np.asarray(impedance) if impedance is not None else None,
                                                   damping_ratio=damping_ratio, nullspace_stiffness=nullspace_stiffness,
                                                   filter_coeff=filter_coeff)
        self.error = None
        self._ctrl_running = False
        self._t_started = 0.0
        self._last_at_start = ""

    def start(self):
        if self.error:
            raise RuntimeError(f"cannot start: reflex active ({self.error}); recover first")
        self._last_at_start = _errors_text(getattr(self.panda.get_state(), "last_motion_errors", None))
        self.panda.start_controller(self.ctrl)
        self._ctrl_running, self._t_started = True, time.monotonic()

    def set_target(self, T):  # pragma: no cover - hardware call
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.ctrl.set_control(np.asarray(T[:3, 3]), q)

    def get_state(self):
        s = self.panda.get_state()
        if self.error is None:
            self.error = derive_panda_error(s, self._ctrl_running, time.monotonic() - self._t_started,
                                            last_at_start=self._last_at_start)
        T = np.asarray(s.O_T_EE).reshape(4, 4).T
        return {"O_T_EE": T, "q": np.asarray(s.q), "dq": np.asarray(s.dq), "O_F_ext": np.asarray(s.O_F_ext_hat_K),
                "mode": str(s.robot_mode), "success_rate": float(s.control_command_success_rate),
                "error": self.error, "t": time.monotonic()}

    def move_joints(self, q, speed=0.2):  # pragma: no cover - hardware call
        self._ctrl_running = False
        self.panda.stop_controller()
        self.panda.move_to_joint_position(list(q), speed_factor=speed)
        self.start()

    def stop(self):
        self._ctrl_running = False
        self.panda.stop_controller()

    def recover(self):
        self._ctrl_running = False
        self.panda.recover()          # libfranka automaticErrorRecovery
        self.error = None


def make_backend(cfg: dict, home_O_T_EE: np.ndarray) -> Backend:
    kind = cfg.get("type", "fake")
    if kind == "fake":
        return FakeBackend(home_O_T_EE, tau_s=cfg.get("tau_s", 0.03), bias_m=cfg.get("bias_m", (0, 0, 0)))
    if kind == "panda_py":
        return PandaPyBackend(**cfg["panda_py"])
    if kind == "mujoco":   # franka_sim (controlSW/franka_sim on PYTHONPATH) + mujoco; imported only here
        try:
            from franka_sim.backend import MuJoCoBackend
        except ImportError as e:
            raise RuntimeError(f"backend type 'mujoco' needs controlSW/franka_sim on PYTHONPATH and the mujoco package "
                               f"({type(e).__name__}: {e})") from e
        return MuJoCoBackend.from_config(cfg, home_O_T_EE)
    raise ValueError(f"unknown backend {kind!r}")
