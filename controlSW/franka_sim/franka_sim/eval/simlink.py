"""The Stage 3 driver's view of the simulator: ``SimHandle`` (protocol), ``BridgeSim`` (A6, through the bridge) and
``StubSim`` (an in-process stand-in that publishes the same I-SIMSTATE stream with a scripted eraser).

Why a stub: Stage 3 needs a simulator that moves the eraser. Until A5/A6/A7 land, ``StubSim`` implements the A6
interfaces exactly (state PUB, ``sim_reset``) so the whole driver, judge and metrics path runs on ``sim-iphone-fake``
(FakeBackend + fake gripper + video replay). Part 2 swaps in ``BridgeSim`` without touching the driver.

Interfaces (INTERFACES.md I-SIMSTATE; code against these):
  * state message: one ZMQ part, ``robotmq.serialize(dict)``; subscribe ``""``; keys ``t_mono, qpos, qvel, eraser_T_base (4x4),
    hand_gap_m, grasped, contacts: [{a, b, force_n}], reflex: str | None, tau_ext``; ``contacts`` use MuJoCo geom names.
  * ``sim_reset`` bridge topic: ``{"eraser_xy_yaw": [x, y, yaw_rad], "hand_gap_m": g}`` (``hand_gap_m`` optional); a reply
    with ``ok: false`` (e.g. while grasping) is ``SimResetRefused``.
  * ``latest()`` = the bridge's ``sim_state`` topic (BridgeSim) or the stub's newest message.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Protocol

import numpy as np

SCENARIOS = ("lift_success", "no_lift", "lift_drop", "knock_over", "collision", "reflex")
Z_REST = 0.02           # eraser centre height on the table (stub)
STATE_HZ = 100.0


class SimResetRefused(RuntimeError):
    """``sim_reset`` was refused (A6: e.g. the eraser is grasped)."""


class SimHandle(Protocol):
    state_endpoint: str

    def reset(self, eraser_xy_yaw: tuple, hand_gap_m: float | None = None) -> dict: ...
    def latest(self) -> dict | None: ...
    def close(self) -> None: ...


class BridgeSim:
    """A6: the simulator behind the bridge. ``reset`` = bridge ``sim_reset``; ``latest`` = bridge ``sim_state``."""

    sim_source = "mujoco"

    def __init__(self, bridge_endpoint: str, state_endpoint: str, reset_timeout_s: float = 3.0):
        from franka_deploy.bridge.link import BridgeClient

        self.state_endpoint = state_endpoint
        self.client = BridgeClient(bridge_endpoint.replace("0.0.0.0", "127.0.0.1"), reset_timeout_s)

    def reset(self, eraser_xy_yaw, hand_gap_m=None) -> dict:
        from franka_deploy.rmq_rpc import RpcError

        data = {"eraser_xy_yaw": [float(x) for x in eraser_xy_yaw]}
        if hand_gap_m is not None:
            data["hand_gap_m"] = float(hand_gap_m)
        try:
            return self.client.request("sim_reset", data) or {}
        except RpcError as e:
            raise SimResetRefused(str(e)) from e

    def latest(self) -> dict | None:
        try:
            return self.client.request("sim_state", None, timeout_s=0.5)
        except Exception:       # noqa: BLE001
            return None

    def on_episode_start(self, t_mono: float):
        pass

    def close(self):
        self.client.close()


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


class StubSim:
    """Publishes I-SIMSTATE at 100 Hz on ``state_endpoint`` with a scripted eraser. Times are relative to the episode
    start the driver signals with ``on_episode_start(t_mono)``. Scenarios:
      lift_success  grasped from +1 s, +6 cm between 2.5 and 3 s, held (success 2 s later)
      no_lift       nothing moves
      lift_drop     as lift_success, but drops back at +4 s (never held 2 s)
      knock_over    tilts 45 deg at +2 s
      collision     lift_success plus an arm link6<->table contact from +2 s for 0.5 s
      reflex        no lift; ``reflex`` set from +2 s (the driver also sends bridge inject_reflex at +2 s)
    The state also carries the contacts a healthy episode has (pad<->eraser while grasped, eraser<->table at rest), which the
    judge must ignore."""

    sim_source = "stub"

    def __init__(self, state_endpoint: str, scenario: str = "lift_success", hz: float = STATE_HZ):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown stub scenario {scenario!r} (known: {', '.join(SCENARIOS)})")
        self.state_endpoint, self.scenario, self.hz = state_endpoint, scenario, float(hz)
        self.driver_actions = [(2.0, "inject_reflex")] if scenario == "reflex" else []
        self._lock = threading.Lock()
        self._xy_yaw = (0.45, 0.0, 0.0)
        self._t0: float | None = None
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.n_published = 0
        self.resets: list = []
        self._th = threading.Thread(target=self._run, name="stub-sim", daemon=True)
        self._th.start()
        if not self._ready.wait(5.0):
            raise RuntimeError(f"StubSim could not bind {state_endpoint}")

    # ---- SimHandle
    def reset(self, eraser_xy_yaw, hand_gap_m=None) -> dict:
        with self._lock:
            self._xy_yaw = tuple(float(x) for x in eraser_xy_yaw)
            self._t0 = None
            self.resets.append((self._xy_yaw, hand_gap_m))
        return {"ok": True, "eraser_xy_yaw": list(self._xy_yaw)}

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def on_episode_start(self, t_mono: float):
        with self._lock:
            self._t0 = float(t_mono)

    def on_episode_end(self):
        with self._lock:
            self._t0 = None

    def close(self):
        self._stop.set()
        self._th.join(2.0)

    # ---- scripted physics
    def state_at(self, t_mono: float, rel: float | None) -> dict:
        x, y, yaw = self._xy_yaw
        z, tilt, grasped, reflex, contacts = Z_REST, 0.0, False, None, [{"a": "eraser_geom", "b": "table_top", "force_n": 1.0}]
        if rel is not None:
            sc = self.scenario
            if sc in ("lift_success", "lift_drop", "collision"):
                if rel >= 1.0:
                    grasped = True
                    contacts = [{"a": "pad_left", "b": "eraser_geom", "force_n": 12.0}, {"a": "pad_right", "b": "eraser_geom", "force_n": 12.0}]
                lift = 0.06 * min(1.0, max(0.0, (rel - 2.5) / 0.5))
                if sc == "lift_drop" and rel >= 4.0:
                    lift, grasped = 0.0, False
                z += lift
                if sc == "collision" and 2.0 <= rel < 2.5:
                    contacts = contacts + [{"a": "link6_col", "b": "table_top", "force_n": 8.0}]
            elif sc == "knock_over" and rel >= 2.0:
                tilt = 45.0
            elif sc == "reflex" and rel >= 2.0:
                reflex = "cartesian_reflex (stub)"
        T = np.eye(4)
        c, s = math.cos(yaw), math.sin(yaw)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ _rx(tilt)
        T[:3, 3] = [x, y, z]
        return {"t_mono": float(t_mono), "qpos": np.zeros(9), "qvel": np.zeros(9), "eraser_T_base": T,
                "hand_gap_m": 0.03 if grasped else 0.08, "grasped": grasped, "contacts": contacts, "reflex": reflex,
                "tau_ext": np.zeros(7)}

    def _run(self):
        import zmq
        from robotmq import serialize

        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.bind(self.state_endpoint)
        except zmq.ZMQError:
            return
        self._ready.set()
        period = 1.0 / self.hz
        nxt = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    t0 = self._t0
                s = self.state_at(now, None if t0 is None else now - t0)
                with self._lock:
                    self._latest = s
                sock.send(serialize(s))
                self.n_published += 1
                nxt += period
                time.sleep(max(0.0, nxt - time.monotonic()))
        finally:
            sock.close(linger=0)
