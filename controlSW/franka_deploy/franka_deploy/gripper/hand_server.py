"""hand_server: the Franka Hand (libfranka ``Gripper``) behind the I-HAND REQ/REP topics, in its own process (port 5566).

Why: libfranka ``Gripper.move`` / ``grasp`` / ``homing`` block for 0.3-1.5 s, so they can sit neither in the RT bridge
nor in the 200 Hz controller loop. Here one worker thread makes the blocking calls, a state thread polls
``read_once()``, and the REP loop only queues commands and returns the cached state: it never waits on the hand.
In sim the bridge serves the same topics (franka_sim.hand_sim); FrankaHandDriver is the client in both cases.

Topics (franka_deploy.bridge.link protocol; unknown topics -> ok: false):
  hand_cmd   {kind: move|grasp|stop|homing, width_m, speed_m_s, force_n, eps_inner_m, eps_outer_m, seq}
             -> {accepted, seq}. Latest wins: a command while busy stops the running one first (``stop()`` on a
             helper thread, so the REP loop does not block on it), then replaces any queued one.
  hand_state -> {gap_m, gap_vel_m_s, busy, mode, grasped, last_result, seq_done, t}
             mode: idle | moving | grasping | holding | homing | stopped | error
             last_result: move_ok | move_failed | grasp_ok | grasp_failed | stopped | homing_ok | error:...
             t = time.monotonic() of the last state read.

Backends:
  fake       FakeHand: kinematic gap at the commanded speed; stalls at --object-width-m (libfranka width, i.e. the
             gap) while closing (grasp -> holding, grasped by the epsilons; move -> move_failed). --cmd-latency-s
             adds libfranka-like blocking time before each command so tests can prove the client never waits.
  libfranka  LibfrankaHand on panda_py.libfranka.Gripper(hostname) (hardware only; docs/panda_py_checklist.md).

    python -m franka_deploy.gripper.hand_server --endpoint tcp://localhost:5566 --backend fake|libfranka \
        [--hostname H] [--object-width-m W] [--cmd-latency-s S] [--no-homing]
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time

KINDS = ("move", "grasp", "stop", "homing")
SPEED_MAX_M_S = 0.1
FORCE_MAX_N = 70.0
GOOD = ("move_ok", "grasp_ok", "homing_ok")


def bind_endpoint(ep: str) -> str:
    """zmq cannot bind 'localhost' / '*'-style hostnames everywhere: tcp://localhost:5566 -> tcp://127.0.0.1:5566."""
    return ep.replace("://localhost:", "://127.0.0.1:")


def validate_cmd(msg) -> dict:
    if not isinstance(msg, dict) or msg.get("kind") not in KINDS:
        raise ValueError(f"hand_cmd: kind must be one of {KINDS}, got {msg!r}")
    out = {"kind": msg["kind"], "seq": int(msg.get("seq", 0))}
    for k, d in (("width_m", 0.0), ("speed_m_s", SPEED_MAX_M_S), ("force_n", 0.0), ("eps_inner_m", 0.005),
                 ("eps_outer_m", 0.005)):
        v = float(msg.get(k, d) if msg.get(k) is not None else d)
        if not math.isfinite(v):
            raise ValueError(f"hand_cmd: {k} is not finite")
        out[k] = v
    out["speed_m_s"] = min(max(out["speed_m_s"], 1e-4), SPEED_MAX_M_S)
    out["force_n"] = min(max(out["force_n"], 0.0), FORCE_MAX_N)
    out["eps_inner_m"], out["eps_outer_m"] = max(0.0, out["eps_inner_m"]), max(0.0, out["eps_outer_m"])
    return out


class WorkerHand:
    """Shared command machinery: one worker thread runs blocking device calls (``_dev_*``), latest command wins.
    Subclasses implement ``_dev_move/_dev_grasp/_dev_homing`` (blocking, return bool), ``_dev_stop`` (interrupts
    a running call), ``_dev_read`` -> (gap_m, is_grasped) and ``max_width_m``."""

    max_width_m = 0.08

    def __init__(self, state_rate_hz: float = 100.0):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending = None
        self._stopper = None
        self._run_id, self._interrupted = 0, -1
        self._busy = False
        self._mode, self._last_result, self._seq_done = "idle", "", 0
        self._gap, self._vel, self._grasped, self._t = float("nan"), 0.0, False, time.monotonic()
        self._run = True
        self.state_period = 1.0 / float(state_rate_hz)
        self._worker = threading.Thread(target=self._work, name="hand-worker", daemon=True)
        self._reader = threading.Thread(target=self._read_loop, name="hand-state", daemon=True)

    def start(self, homing: bool = True):
        self._reader.start()
        self._worker.start()
        if homing:
            self.command({"kind": "homing", "seq": 0})

    # --- REP side (never blocks on the device)
    def command(self, msg: dict) -> dict:
        cmd = validate_cmd(msg)
        with self._lock:
            self._pending = cmd
            # interrupt the running call; the worker joins this stopper before it starts the next command, so a
            # late stop() can never hit the new one
            if self._busy:
                self._interrupted = self._run_id
            if self._busy and (self._stopper is None or not self._stopper.is_alive()):
                self._stopper = threading.Thread(target=self._safe_stop, name="hand-stop", daemon=True)
                self._stopper.start()
            self._cv.notify()
        return {"accepted": True, "seq": cmd["seq"]}

    def state(self) -> dict:
        with self._lock:
            return {"gap_m": self._gap, "gap_vel_m_s": self._vel, "busy": self._busy or self._pending is not None,
                    "mode": self._mode, "grasped": bool(self._grasped) and self._mode == "holding",
                    "last_result": self._last_result,
                    "seq_done": self._seq_done, "t": self._t}

    def close(self):
        """Shutdown (SIGINT/SIGTERM). A running move/grasp/homing is stopped; a finished grasp that is HOLDING is
        left alone (no stop()): keeping the eraser in the fingers is safer than dropping it (checklist §7)."""
        self._run = False
        with self._lock:
            self._cv.notify()
            busy = self._busy
        if busy:
            self._safe_stop()
        self._worker.join(2.0)
        self._reader.join(1.0)

    # --- threads
    def _safe_stop(self):
        try:
            self._dev_stop()
        except Exception as e:  # pragma: no cover - device dependent
            print(f"[hand] stop failed: {e}", flush=True)

    def _read_loop(self):
        prev = None
        while self._run:
            try:
                gap, grasped = self._dev_read()
                now = time.monotonic()
                with self._lock:
                    if prev is not None and now > prev[1]:
                        self._vel = (gap - prev[0]) / (now - prev[1])
                    prev = (gap, now)
                    self._gap, self._t = float(gap), now
                    self._grasped = bool(grasped)          # raw; state() ANDs it with the current mode
            except Exception as e:  # pragma: no cover - device dependent
                print(f"[hand] read failed: {e}", flush=True)
            time.sleep(self.state_period)

    def _work(self):
        while True:
            with self._lock:
                while self._run and self._pending is None:
                    self._cv.wait(0.1)
                if not self._run:
                    return
                stopper = self._stopper
            if stopper is not None:
                stopper.join()
            with self._lock:
                cmd, self._pending = self._pending, None
                if cmd is None:
                    continue
                self._busy = True
                self._run_id += 1
                self._mode = {"move": "moving", "grasp": "grasping", "homing": "homing", "stop": "stopped"}[cmd["kind"]]
            try:
                result, mode = self._execute(cmd)
            except Exception as e:  # pragma: no cover - device dependent
                result, mode = f"error:{type(e).__name__}: {e}", "error"
            try:                                         # publish the result with a fresh width, not a 10 ms old one
                fresh = self._dev_read()
            except Exception:  # pragma: no cover - device dependent
                fresh = None
            with self._lock:
                if fresh is not None:
                    self._gap, self._t = float(fresh[0]), time.monotonic()
                if self._interrupted == self._run_id and mode != "error" and result not in GOOD:
                    result, mode = "stopped", "stopped"      # interrupted by a newer command / stop
                if result == "grasp_ok":
                    self._grasped = True        # the grasp's own verdict; the reader refreshes it from then on
                self._busy = False
                self._seq_done = cmd["seq"]
                self._last_result = result
                self._mode = mode

    def _execute(self, cmd) -> tuple[str, str]:
        k = cmd["kind"]
        if k == "stop":
            self._dev_stop()
            return "stopped", "stopped"
        if k == "homing":
            return ("homing_ok", "idle") if self._dev_homing() else ("error:homing failed", "error")
        width = min(max(cmd["width_m"], 0.0), self.max_width_m)
        if k == "move":
            return ("move_ok", "idle") if self._dev_move(width, cmd["speed_m_s"]) else ("move_failed", "idle")
        ok = self._dev_grasp(width, cmd["speed_m_s"], cmd["force_n"], cmd["eps_inner_m"], cmd["eps_outer_m"])
        return ("grasp_ok", "holding") if ok else ("grasp_failed", "idle")

    # --- device hooks
    def _dev_move(self, width, speed) -> bool:
        raise NotImplementedError

    def _dev_grasp(self, width, speed, force, eps_i, eps_o) -> bool:
        raise NotImplementedError

    def _dev_homing(self) -> bool:
        raise NotImplementedError

    def _dev_stop(self) -> None:
        raise NotImplementedError

    def _dev_read(self) -> tuple[float, bool]:
        raise NotImplementedError


class FakeHand(WorkerHand):
    """Kinematic fingers: the gap moves at the commanded speed; closing stalls at ``object_width_m`` (gap units).
    Each blocking call first waits ``cmd_latency_s`` (interruptible by stop), like libfranka's command round trip."""

    def __init__(self, object_width_m: float | None = None, cmd_latency_s: float = 0.0, start_gap_m: float = 0.08,
                 max_width_m: float = 0.08, dt: float = 0.002):
        self.max_width_m = float(max_width_m)
        self.object = None if object_width_m is None else float(object_width_m)
        self.latency = float(cmd_latency_s)
        self.dt = float(dt)
        self._g = float(start_gap_m)
        self._stop_ev = threading.Event()
        self._hold_grasp = None             # (lo, hi) grasp window while holding
        super().__init__()

    def _dev_read(self):
        g = self._g
        win = self._hold_grasp
        return g, bool(win and win[0] <= g <= win[1])

    def _dev_stop(self):
        self._stop_ev.set()

    def _wait_latency(self) -> bool:
        return not self._stop_ev.wait(self.latency) if self.latency > 0 else True

    def _travel(self, target, speed) -> str:
        """Move toward target; 'done' | 'blocked' (closing onto the object) | 'stopped'."""
        obj = self.object
        while True:
            if self._stop_ev.is_set():
                return "stopped"
            d = target - self._g
            if obj is not None and d < 0 and self._g <= obj + 1e-9:
                self._g = max(self._g, min(obj, self._g))
                return "blocked"
            if abs(d) < 1e-9:
                return "done"
            nxt = self._g + math.copysign(min(abs(d), speed * self.dt), d)
            if obj is not None and d < 0 and nxt < obj < self._g:
                nxt = obj
            self._g = nxt
            time.sleep(self.dt)

    def _begin(self) -> bool:
        self._stop_ev.clear()
        self._hold_grasp = None
        return self._wait_latency()

    def _dev_move(self, width, speed):
        if not self._begin():
            return False
        return self._travel(width, speed) == "done"

    def _dev_grasp(self, width, speed, force, eps_i, eps_o):
        if not self._begin():
            return False
        r = self._travel(max(0.0, width - eps_i), speed)
        if r == "stopped":
            return False
        ok = r == "blocked" and width - eps_i <= self._g <= width + eps_o
        if r == "blocked":
            self._hold_grasp = (width - eps_i, width + eps_o)
        return ok

    def _dev_homing(self):
        if not self._begin():
            return False
        return self._travel(self.max_width_m, SPEED_MAX_M_S) == "done"


class LibfrankaHand(WorkerHand):  # pragma: no cover - hardware only
    """panda_py.libfranka.Gripper. Unverified until the hardware checklist (docs/panda_py_checklist.md §4b) passes:
    GIL release during the blocking calls, stop() from another thread mid-grasp, the read_once rate, homing."""

    def __init__(self, hostname: str):
        try:
            from panda_py import libfranka
        except ImportError as e:
            raise RuntimeError("hand_server --backend libfranka needs panda_py (built against the robot's libfranka)") from e
        self.g = libfranka.Gripper(hostname)
        st = self.g.read_once()
        self.max_width_m = float(getattr(st, "max_width", 0.08)) or 0.08
        super().__init__(state_rate_hz=50.0)

    def _dev_read(self):
        st = self.g.read_once()
        return float(st.width), bool(st.is_grasped)

    def _dev_stop(self):
        self.g.stop()

    def _dev_move(self, width, speed):
        return bool(self.g.move(width, speed))

    def _dev_grasp(self, width, speed, force, eps_i, eps_o):
        return bool(self.g.grasp(width, speed, force, eps_i, eps_o))

    def _dev_homing(self):
        return bool(self.g.homing())


def serve(endpoint: str, hand: WorkerHand, stop_flag=None) -> None:
    from franka_deploy.bridge.link import BridgeServerSocket

    sock = BridgeServerSocket(bind_endpoint(endpoint))
    print(f"[hand] listening on {endpoint} ({type(hand).__name__})", flush=True)
    stop_flag = stop_flag or {"stop": False}
    try:
        while not stop_flag["stop"]:
            topic, data = sock.poll(5)
            if topic is None:
                continue
            try:
                if topic == "hand_cmd":
                    sock.reply(True, hand.command(data))
                elif topic == "hand_state":
                    sock.reply(True, hand.state())
                else:
                    sock.reply(False, error=f"unknown request {topic!r}")
            except Exception as e:
                sock.reply(False, error=f"{type(e).__name__}: {e}")
    finally:
        print("[hand] shutdown", flush=True)
        try:
            hand.close()
        finally:
            sock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default="tcp://localhost:5566")
    ap.add_argument("--backend", choices=("fake", "libfranka"), default="fake")
    ap.add_argument("--hostname", default=None, help="FCI hostname (libfranka)")
    ap.add_argument("--object-width-m", type=float, default=None, help="fake: gap at which closing stalls")
    ap.add_argument("--cmd-latency-s", type=float, default=0.0, help="fake: blocking time before each command")
    ap.add_argument("--start-gap-m", type=float, default=0.08, help="fake: initial gap")
    ap.add_argument("--no-homing", action="store_true")
    a = ap.parse_args(argv)
    if a.backend == "libfranka":
        if not a.hostname:
            ap.error("--backend libfranka needs --hostname")
        hand = LibfrankaHand(a.hostname)
    else:
        hand = FakeHand(a.object_width_m, a.cmd_latency_s, a.start_gap_m)
    flag = {"stop": False}
    signal.signal(signal.SIGINT, lambda *_: flag.__setitem__("stop", True))
    signal.signal(signal.SIGTERM, lambda *_: flag.__setitem__("stop", True))
    hand.start(homing=not a.no_homing)
    serve(a.endpoint, hand, flag)


if __name__ == "__main__":
    main()
