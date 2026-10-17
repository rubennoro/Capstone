"""franka_bridge: the process that owns the Franka's 1 kHz loop (on the Jetson's isolated core or an RT PC; same code).

Requests (franka_deploy.bridge.link, REQ/REP, one reply each):
  start                  activate the controller holding the current pose (refused while a reflex is active)
  set_target  {T}        4x4 O_T_EE -> safety filter (safety.py) -> backend; returns {"flags"}
  get_state              backend state + bridge state ("idle"|"running"|"hold"|"stopped"|"reflex") + stats
  move_joints {q, speed} blocking joint move (homing)
  recover                clear a reflex / stopped state -> "idle" (explicit operator command only)
  stop                   stop the controller -> "stopped"
  inject_reflex          test only (enabled with --allow-inject)
Watchdog (while running): no set_target for > watchdog_hold_s (0.2) -> "hold" (keep the last safe target);
> watchdog_stop_s (2.0) -> backend.stop(), "stopped" (needs recover/start). Reflex (backend error) -> "reflex";
never recovered automatically. Per-second stats line: request counts, hop service time, safety counters,
1 kHz loop percentiles, min success rate. SIGINT/SIGTERM -> backend stop, socket closed (linger 0), exit.

    python -m franka_deploy.bridge.franka_bridge --config configs/bridge.yaml [--endpoint tcp://127.0.0.1:5555]
"""

from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np
import yaml

from franka_deploy.bridge.backends import make_backend
from franka_deploy.bridge.link import BridgeServerSocket
from franka_deploy.bridge.safety import SafetyConfig, SafetyFilter


def pose7_to_T(p):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    return T


class FrankaBridge:
    def __init__(self, cfg: dict, endpoint: str, allow_inject: bool = False, log_jsonl: str | None = None):
        self.cfg = cfg
        self.sock = BridgeServerSocket(endpoint)
        self.endpoint = endpoint
        home = pose7_to_T(np.asarray(cfg.get("home_O_T_EE_xyz_wxyz", [0.45, 0.0, 0.45, 0.0, 1.0, 0.0, 0.0]), float))
        self.backend = make_backend(cfg.get("backend", {"type": "fake"}), home)
        self.safety = SafetyFilter(SafetyConfig.from_dict(cfg.get("safety", {})))
        self.hold_s = float(cfg.get("watchdog_hold_s", 0.2))
        self.stop_s = float(cfg.get("watchdog_stop_s", 2.0))
        self.allow_inject = allow_inject
        self.state = "idle"
        self.last_target_t = None
        self.events: list[dict] = []
        self._stop = False
        self._jsonl = open(log_jsonl, "a", buffering=1) if log_jsonl else None
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {"requests": {}, "service_us": [], "holds": 0, "stops": 0}
        self.min_success = 1.0
        self.t_stats = time.monotonic()

    def log(self, kind, **kw):
        ev = {"t": time.monotonic(), "kind": kind, **kw}
        self.events.append(ev)
        print(f"[bridge] {kind} {json.dumps(kw, default=str)}", flush=True)
        if self._jsonl:
            self._jsonl.write(json.dumps(ev, default=str) + "\n")

    def _activate(self):
        st = self.backend.get_state()
        if st["error"]:
            raise RuntimeError(f"reflex active: {st['error']} (recover first)")
        self.backend.start()
        self.safety.reset(st["O_T_EE"], time.monotonic())
        self.state = "running"
        self.last_target_t = time.monotonic()
        self.log("started")

    def handle(self, topic, data):
        if topic == "get_state":
            st = self.backend.get_state()
            self.min_success = min(self.min_success, float(st.get("success_rate", 1.0)))
            st["bridge_state"] = self.state
            return st
        if topic == "set_target":
            if self.state in ("stopped", "reflex", "idle"):
                return {"accepted": False, "state": self.state}
            T, flags = self.safety.apply(np.asarray(data["T"], float), time.monotonic())
            if T is None:
                if self.state == "running":
                    self.state = "hold"
                self.log("target_rejected_nonfinite")
                return {"accepted": False, "flags": flags, "state": self.state}
            self.backend.set_target(T)
            self.last_target_t = time.monotonic()
            if self.state == "hold":
                self.state = "running"
            return {"accepted": True, "flags": flags, "state": self.state, "T_safe": T}
        if topic == "start":
            if self.state in ("idle", "stopped"):
                self._activate()
            return {"state": self.state}
        if topic == "stop":
            self.backend.stop()
            self.state = "stopped"
            self.stats["stops"] += 1
            self.log("stopped", reason="command")
            return {"state": self.state}
        if topic == "recover":
            self.backend.recover()
            self.state = "idle"
            self.log("recovered")
            return {"state": self.state}
        if topic == "move_joints":
            self.backend.move_joints(data["q"], data.get("speed", 0.2))
            return {"state": self.state}
        if topic == "inject_reflex":
            if not self.allow_inject:
                raise RuntimeError("inject_reflex disabled")
            self.backend.inject_reflex(data or "cartesian_reflex (injected)")
            return {"state": self.state}
        raise ValueError(f"unknown request {topic!r}")

    def tick(self):
        now = time.monotonic()
        st = self.backend.get_state()
        if st["error"] and self.state != "reflex":
            self.state = "reflex"
            self.log("reflex", error=st["error"])
        if self.state in ("running", "hold") and self.last_target_t is not None:
            age = now - self.last_target_t
            if age > self.stop_s:
                self.backend.stop()
                self.state = "stopped"
                self.stats["stops"] += 1
                self.log("watchdog_stop", age_s=round(age, 3))
            elif age > self.hold_s and self.state == "running":
                self.state = "hold"  # keep the last safe target (backend already holds it)
                self.stats["holds"] += 1
                self.log("watchdog_hold", age_s=round(age, 3))
        if now - self.t_stats >= 1.0:
            svc = np.array(self.stats["service_us"]) if self.stats["service_us"] else np.zeros(1)
            line = {"state": self.state, "req": self.stats["requests"], "svc_us_p50": float(np.percentile(svc, 50)),
                    "svc_us_p99": float(np.percentile(svc, 99)), "safety": dict(self.safety.counts),
                    "loop": self.backend.loop_stats(), "min_success": self.min_success}
            short = {**line, "loop": {k: v for k, v in line["loop"].items() if k != "cycle_hist_us"}}
            print(f"[bridge] stats {json.dumps(short)}", flush=True)
            if self._jsonl:
                self._jsonl.write(json.dumps({"t": now, "kind": "stats", **line}, default=str) + "\n")
            self._reset_stats()

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stop", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        self.log("listening", endpoint=self.endpoint)
        next_tick = 0.0
        try:
            while not self._stop:
                topic, data = self.sock.poll(2)
                if topic is not None:
                    t0 = time.perf_counter()
                    try:
                        self.sock.reply(True, self.handle(topic, data))
                    except Exception as e:
                        self.sock.reply(False, error=f"{type(e).__name__}: {e}")
                    self.stats["service_us"].append((time.perf_counter() - t0) * 1e6)
                    self.stats["requests"][topic] = self.stats["requests"].get(topic, 0) + 1
                if time.monotonic() >= next_tick:  # watchdog/stats at 200 Hz; requests are served first
                    next_tick = time.monotonic() + 0.005
                    self.tick()
        finally:
            self.log("shutdown")
            if self._jsonl:
                self._jsonl.close()
                self._jsonl = None
            try:
                self.backend.stop()
                self.backend.close()
            finally:
                self.sock.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--allow-inject", action="store_true")
    ap.add_argument("--log-jsonl", default=None, help="append events + per-second stats as JSON lines")
    ap.add_argument("--bias", type=float, nargs=3, default=None, help="FakeBackend bias override (m)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    if a.bias is not None:
        cfg.setdefault("backend", {})["bias_m"] = a.bias
    FrankaBridge(cfg, a.endpoint or cfg.get("endpoint", "tcp://127.0.0.1:5555"), a.allow_inject, a.log_jsonl).run()


if __name__ == "__main__":
    main()
