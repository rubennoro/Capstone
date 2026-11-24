"""The sim PC's state export, ``sim.state/v1`` (DESIGN.md section 4): SUB, newest messages kept, qpos at a time.

One single-part JSON message every 5 ms of physics time::

    {"schema": "sim.state/v1", "t_ns": <CLOCK_MONOTONIC of that physics instant>, "step": <int>,
     "model_sha256": "...", "qpos": [<nq floats>]}

``StateBuffer.at(t_ns)`` interpolates the two messages around ``t_ns`` (linear; quaternions of free and ball joints
normalised). Never extrapolates: a time newer than the newest message gets the newest qpos (a hold), and the result
says how far it was held (``held_ns``), so the caller can refuse more than 10 ms. A message whose ``model_sha256`` or
qpos length differs from the loaded model is a fatal mismatch (``StateMismatch``): the sim PC runs another model file.
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import zmq

log = logging.getLogger("sim_pi.state")

SCHEMA = "sim.state/v1"
KEYS = ("schema", "t_ns", "step", "model_sha256", "qpos")


class StateMismatch(RuntimeError):
    """The state export is for another model (sha or nq) or breaks sim.state/v1."""


@dataclass(frozen=True)
class StateSample:
    t_ns: int
    step: int
    qpos: np.ndarray


@dataclass(frozen=True)
class StateAt:
    qpos: np.ndarray
    t_ns: int            # the time asked for
    held_ns: int         # > 0: newer than the newest message by this much (its qpos is used as is)
    gap_ns: int          # spacing of the two messages interpolated (0 when held or exact)


def parse(raw: bytes, model_sha256: str | None, nq: int) -> StateSample:
    try:
        d = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise StateMismatch(f"{SCHEMA}: not JSON ({e})") from None
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise StateMismatch(f"expected schema {SCHEMA!r}, got {d.get('schema') if isinstance(d, dict) else d!r}")
    missing = [k for k in KEYS if k not in d]
    if missing:
        raise StateMismatch(f"{SCHEMA}: missing keys {missing}")
    if model_sha256 is not None and d["model_sha256"] != model_sha256:
        raise StateMismatch(f"state export model_sha256 {str(d['model_sha256'])[:16]}... != the loaded model "
                            f"{model_sha256[:16]}...: the sim PC and the sim Pi load different model files")
    q = np.asarray(d["qpos"], float)
    if q.shape != (nq,):
        raise StateMismatch(f"{SCHEMA}: qpos has {q.size} values, the model has nq = {nq}")
    return StateSample(int(d["t_ns"]), int(d["step"]), q)


class StateBuffer:
    """Thread-safe ring of the newest samples (by t_ns) and the interpolation."""

    def __init__(self, quat_slices: list[slice], keep_s: float = 2.0):
        self._quat = quat_slices
        self._keep_ns = int(keep_s * 1e9)
        self._t: deque[int] = deque()
        self._s: deque[StateSample] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.received = 0
        self.out_of_order = 0
        self.last_arrival_ns: int | None = None

    def push(self, s: StateSample, arrival_ns: int) -> None:
        with self._cond:
            self.received += 1
            self.last_arrival_ns = arrival_ns
            if self._t and s.t_ns <= self._t[-1]:
                if s.t_ns < self._t[-1] - self._keep_ns:      # time went far back: the sim PC restarted
                    self._t.clear()
                    self._s.clear()
                else:
                    self.out_of_order += 1
                    return
            self._t.append(s.t_ns)
            self._s.append(s)
            while self._t and self._t[0] < s.t_ns - self._keep_ns:
                self._t.popleft()
                self._s.popleft()
            self._cond.notify_all()

    def newest_t_ns(self) -> int | None:
        with self._lock:
            return self._t[-1] if self._t else None

    def wait_for(self, t_ns: int, deadline_mono_ns: int) -> bool:
        """Wait until a sample at or after ``t_ns`` is in, or until the deadline (CLOCK_MONOTONIC). True if one is."""
        with self._cond:
            while not (self._t and self._t[-1] >= t_ns):
                left = (deadline_mono_ns - time.monotonic_ns()) / 1e9
                if left <= 0:
                    return False
                self._cond.wait(left)
            return True

    def at(self, t_ns: int) -> StateAt | None:
        """qpos at ``t_ns``, or None if there is no sample yet. Older than the oldest kept: the oldest (held_ns < 0)."""
        with self._lock:
            if not self._t:
                return None
            ts = self._t
            if t_ns >= ts[-1]:
                return StateAt(self._s[-1].qpos.copy(), t_ns, t_ns - ts[-1], 0)
            i = bisect.bisect_right(ts, t_ns)
            if i == 0:
                return StateAt(self._s[0].qpos.copy(), t_ns, t_ns - ts[0], 0)
            a, b = self._s[i - 1], self._s[i]
        w = (t_ns - a.t_ns) / (b.t_ns - a.t_ns)
        q = (1.0 - w) * a.qpos + w * b.qpos
        for sl in self._quat:
            qa, qb = a.qpos[sl], b.qpos[sl]
            if float(qa @ qb) < 0:                       # shortest path
                qb = -qb
            v = (1.0 - w) * qa + w * qb
            q[sl] = v / max(np.linalg.norm(v), 1e-12)
        return StateAt(q, t_ns, 0, b.t_ns - a.t_ns)


def quat_slices(model) -> list[slice]:
    """qpos slices holding quaternions (free joints: 3..7 of the joint, ball joints: 0..4)."""
    import mujoco
    out = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            out.append(slice(adr + 3, adr + 7))
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            out.append(slice(adr, adr + 4))
    return out


class StateSubscriber:
    """Thread "sim-pi-state": SUB to the state export, every message into a StateBuffer. A contract or model mismatch
    is kept in ``error`` (the main loop stops on it)."""

    def __init__(self, endpoint: str, buffer: StateBuffer, model_sha256: str | None, nq: int):
        self.endpoint = endpoint
        self.buffer = buffer
        self._sha = model_sha256
        self._nq = nq
        self._stop = threading.Event()
        self.error: StateMismatch | None = None
        self._thread = threading.Thread(target=self._run, name="sim-pi-state", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 1000)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(100):
                    continue
                while True:
                    try:
                        raw = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    arrival = time.monotonic_ns()
                    try:
                        self.buffer.push(parse(raw, self._sha, self._nq), arrival)
                    except StateMismatch as e:
                        self.error = e
                        log.error("state export: %s", e)
                        return
        finally:
            sock.close(linger=0)
