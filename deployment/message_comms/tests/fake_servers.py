"""Fake process_frames, policy and franka_control on real sockets (tcp://127.0.0.1:*), for test_flow.py.

Each answers its contract's requests; `faults` make it misbehave in one way at a chosen moment, so every row of
STRUCTURE.md section 7 can be driven through the real message_comms main().
"""
from __future__ import annotations

import threading
import time

import numpy as np

from contracts import (Ack, ErrorReply, HealthReply, Publisher, SetWaypointsReply, WaypointIssue, decode,
                       pack_obj)
from fakes import health, make_chunk, make_frames, make_state, policy_config
from raw_server import RawServer

MS = 1_000_000


class FakeProcessFrames:
    """C1. modes: "ok", "stale", "down", "silent" (stops answering). set_mode() changes it at once."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.calls: list[str] = []
        self.server = RawServer(self._answer)
        self.endpoint = self.server.endpoint

    def _report(self):
        if self.mode == "stale":
            return health("stale", "ok")
        if self.mode == "down":
            return health("down", "stale")
        return health()

    def _answer(self, data: bytes):
        req = decode(data)
        self.calls.append(req.SCHEMA)
        if self.mode == "silent":
            return None
        if req.SCHEMA == "c1.get_health/v1":
            return HealthReply(health=self._report())
        t1 = time.monotonic_ns() - 40 * MS
        return make_frames((t1 - 50 * MS, t1), self._report())

    def close(self) -> None:
        self.server.close()


class FakePolicy:
    """C4, a hold-still policy. on_infer(n) may return a fault for the n-th infer (1-based):
    "hang" (no reply), "error" (ErrorReply), "broken" (non-unit rot6d), "version" (c4.chunk/v2), "odd" (5 cm)."""

    def __init__(self, uses_prompt: bool = False) -> None:
        self.uses_prompt = uses_prompt
        self.calls: list[str] = []
        self.infers = 0
        self.faults: dict[int, str] = {}
        self.server = RawServer(self._answer)
        self.endpoint = self.server.endpoint

    def _answer(self, data: bytes):
        req = decode(data)
        self.calls.append(req.SCHEMA)
        if req.SCHEMA == "c4.config/v1":
            return policy_config(uses_prompt=self.uses_prompt)
        if req.SCHEMA in ("c4.reset/v1", "c4.prompt/v1"):
            return Ack(ok=True, reason="")
        self.infers += 1
        fault = self.faults.get(self.infers)
        if fault == "hang":
            return None
        if fault == "error":
            return ErrorReply(request=req.SCHEMA, error="CUDA error: device lost", field="")
        if fault == "broken":
            d = make_chunk().to_dict()
            d["eef_rot6d"] = np.full((16, 6), 3.0, dtype=np.float32)
            return pack_obj(d)
        if fault == "version":
            d = make_chunk().to_dict()
            d["schema"] = "c4.chunk/v2"
            return pack_obj(d)
        if fault == "odd":
            pos = np.zeros((16, 3))
            pos[:, 0] = 0.05
            return make_chunk(pos=pos, width_clipped=2)
        return make_chunk()

    def close(self) -> None:
        self.server.close()


class FakeFrankaControl:
    """C3 server + C2 publisher at ~100 Hz with franka_control's modes. Every C3 request is logged with its time.
    faults: "fault_after_sets" (n), "reject_at_set" (n), "broken_c2_after_sets" (n), "gap_after_sets" (n)."""

    def __init__(self) -> None:
        self.mode = "idle"
        self.error = ""
        self.requests: list[tuple[str, int]] = []
        self.sets: list = []
        self.faults: dict[str, int] = {}
        self._seq = 0
        self._t_last = time.monotonic_ns()
        self._lock = threading.Lock()
        self._pause_until = 0
        self._broken_next = False
        self._fault_next = ""
        self.publisher = Publisher("tcp://127.0.0.1:*", 1000)
        self.state_endpoint = self.publisher.endpoint
        self.server = RawServer(self._answer)
        self.commands_endpoint = self.server.endpoint
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def names(self) -> list[str]:
        return [name for name, _ in self.requests]

    def _publish_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.01)
            with self._lock:
                now = time.monotonic_ns()
                if now < self._pause_until:
                    continue
                if self.mode not in ("homing", "fault"):        # franka_control repeats its last reading then
                    self._t_last = now
                if self._fault_next:                            # a reflex: C2 shows it before C3 refuses
                    self.mode, self.error, self._fault_next = "fault", self._fault_next, ""
                self._seq += 1
                state = make_state(self._seq, self._t_last, mode=self.mode, error=self.error)
                if self._broken_next:
                    self._broken_next = False
                    d = state.to_dict()
                    d["O_T_TCP"] = np.ascontiguousarray(state.O_T_TCP.T)    # column-major, not transposed
                    self.publisher._socket.send(pack_obj(d))
                    continue
                self.publisher.publish(state)

    def _answer(self, data: bytes):
        req = decode(data)
        name = req.SCHEMA.split(".")[1].split("/")[0]
        self.requests.append((name, time.monotonic_ns()))
        with self._lock:
            if name == "start":
                if self.mode in ("idle", "stopped"):
                    self.mode = "holding"
                    return Ack(ok=True, reason="")
                return Ack(ok=False, reason=f"refused: mode is {self.mode}")
            if name == "stop":
                if self.mode != "fault":
                    self.mode = "stopped"
                return Ack(ok=True, reason="")
            if name == "get_state":
                return make_state(self._seq, self._t_last, mode=self.mode)
            if name == "home":
                self.mode = "homing"
        if name == "home":
            time.sleep(0.3)
            with self._lock:
                self.mode = "idle"
            return Ack(ok=True, reason="")
        if name == "set_waypoints":
            return self._waypoints(req)
        return Ack(ok=True, reason="")              # heartbeat

    def _waypoints(self, req):
        with self._lock:
            if self.mode not in ("holding", "running"):
                return ErrorReply(request=req.SCHEMA, error=f"set_waypoints refused: mode is {self.mode}", field="")
            self.sets.append(req)
            n_sets = len(self.sets)
            n = len(req.t_ns)
            past = int(np.sum(req.t_ns < time.monotonic_ns()))
            if self.faults.get("reject_at_set") == n_sets:
                self.mode, self.error = "fault", "gate rejected waypoint 2"
                return SetWaypointsReply(chunk_id=req.chunk_id, accepted=n - 1 - past, shortened=0, skipped=past,
                                         rejected=(WaypointIssue(index=n - 1, reason="outside the box by 3 cm"),))
            if self.faults.get("fault_after_sets") == n_sets:
                self._fault_next = "cartesian_reflex"
            if self.faults.get("gap_after_sets") == n_sets:
                self._seq += 10                          # 10 missed messages ...
                self._pause_until = time.monotonic_ns() + 100 * MS     # ... over 100 ms
            elif self.mode == "holding":
                self.mode = "running"
            if self.faults.get("broken_c2_after_sets") == n_sets:
                self._broken_next = True
        return SetWaypointsReply(chunk_id=req.chunk_id, accepted=n - past, shortened=0, skipped=past, rejected=())

    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self.server.close()
        self.publisher.close()
