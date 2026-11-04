"""Stage 3 judge: decides an episode's success from the sim state and writes ``y`` / ``n`` to the task's command file.

Why: the closed-loop episode has no human to press the label key. The judge watches the simulator's state stream
(I-SIMSTATE, one ``robotmq.serialize(dict)`` ZMQ part per message: ``{t_mono, qpos, qvel, eraser_T_base (4x4), hand_gap_m,
grasped, contacts: [{a, b, force_n}], reflex, tau_ext}``) and applies the success rule from ``criteria.success``
(``lift_min_m`` held ``hold_s``, ``timeout_s``, ``slip_tol_m``): none of those numbers lives in this file.

* ``SuccessTracker`` is pure (no I/O, no zmq): ``update(state) -> "y" | "n" | None``, ``result() -> dict``.
* ``classify_contacts`` is pure: the non-finger collisions of one state (arm / hand / tooling against table / eraser).
* ``Judge`` is a thread: SUB socket -> tracker -> appends the label to ``<session>/commands.txt`` exactly once.

Rules: lift = eraser z minus its z at the episode start. Success = lift >= lift_min_m continuously for hold_s (a drop
below the threshold restarts the clock). Slip = the largest drop below the running maximum of z while ``grasped``
(reported against ``slip_tol_m``; it never stops the episode). Knock-over = the eraser tilts more than 30 degrees
(as the A6 ground truth). Reflex = ``reflex`` non-null. Knock-over, reflex and ``timeout_s`` give ``n``.

Manual use: ``python -m franka_sim.eval.judge --state-endpoint E --commands FILE --deploy-config D --profile P [--set k=v]``.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

KNOCK_OVER_DEG = 30.0


class SuccessTracker:
    """Pure success rule. ``t_start`` and the states' ``t_mono`` share one clock (host monotonic seconds)."""

    def __init__(self, success: dict, t_start: float, z0: float | None = None, knock_over_deg: float = KNOCK_OVER_DEG):
        self.lift_min = float(success["lift_min_m"])
        self.hold_s = float(success["hold_s"])
        self.timeout_s = float(success["timeout_s"])
        self.slip_tol = float(success["slip_tol_m"])
        self.knock_deg = float(knock_over_deg)
        self.t_start = float(t_start)
        self.z0 = z0
        self.decision: str | None = None
        self._returned = False
        self._lift_since: float | None = None
        self._z_max_grasped: float | None = None
        self.r = {"success": None, "t_success_s": None, "grasp_t_s": None, "max_lift_m": 0.0, "slip_m": 0.0,
                  "knocked_over": False, "reflex": None, "reason": None}

    @staticmethod
    def tilt_deg(T) -> float:
        """Angle between the eraser's z axis and the base z axis."""
        zz = float(T[2][2])
        return math.degrees(math.acos(max(-1.0, min(1.0, zz))))

    def _decide(self, label: str, reason: str, t: float) -> str | None:
        if self.decision is not None:
            return None
        self.decision = label
        self.r["success"] = label == "y"
        self.r["reason"] = reason
        if label == "y":
            self.r["t_success_s"] = round(t - self.t_start, 4)
        self._returned = True
        return label

    def check_timeout(self, t_mono: float) -> str | None:
        """Time alone can end the episode (a silent simulator must not hang the run)."""
        if self.decision is None and t_mono - self.t_start >= self.timeout_s:
            return self._decide("n", "timeout", t_mono)
        return None

    def update(self, s: dict) -> str | None:
        """Feed one state; returns the label the first time the episode is decided, else None."""
        if self.decision is not None:
            return None
        t = float(s["t_mono"])
        if t < self.t_start:
            return None
        T = s["eraser_T_base"]
        z = float(T[2][3])
        if self.z0 is None:
            self.z0 = z
        lift = z - self.z0
        self.r["max_lift_m"] = max(self.r["max_lift_m"], lift)
        if s.get("reflex"):
            self.r["reflex"] = str(s["reflex"])
            return self._decide("n", "reflex", t)
        if self.tilt_deg(T) > self.knock_deg:
            self.r["knocked_over"] = True
            return self._decide("n", "knocked_over", t)
        if s.get("grasped"):
            if self.r["grasp_t_s"] is None:
                self.r["grasp_t_s"] = round(t - self.t_start, 4)
            self._z_max_grasped = z if self._z_max_grasped is None else max(self._z_max_grasped, z)
            self.r["slip_m"] = max(self.r["slip_m"], self._z_max_grasped - z)
        if lift >= self.lift_min:
            if self._lift_since is None:
                self._lift_since = t
            if t - self._lift_since >= self.hold_s:
                return self._decide("y", "lift_held", t)
        else:
            self._lift_since = None
        return self.check_timeout(t)

    def result(self) -> dict:
        out = dict(self.r)
        out["max_lift_m"] = round(out["max_lift_m"], 5)
        out["slip_m"] = round(out["slip_m"], 5)
        out["slip_exceeds_tol"] = out["slip_m"] > self.slip_tol
        out["lift_min_m"], out["hold_s"], out["timeout_s"], out["slip_tol_m"] = self.lift_min, self.hold_s, self.timeout_s, self.slip_tol
        return out


def _has(name: str, patterns) -> bool:
    return any(p in str(name) for p in patterns)


def classify_pad_table(contacts, finger_patterns, table_patterns) -> list:
    """Finger geom (pad / tag / finger) against the table: not a collision (``classify_contacts`` drops finger contacts),
    but reported on its own (A6 review finding 5: a fast approach strikes the pads on the table and may trip a reflex)."""
    out = []
    for c in contacts or []:
        a, b = str(c["a"]), str(c["b"])
        if (_has(a, finger_patterns) and _has(b, table_patterns)) or (_has(b, finger_patterns) and _has(a, table_patterns)):
            out.append({"a": a, "b": b, "force_n": float(c.get("force_n", 0.0))})
    return out


def classify_contacts(contacts, finger_patterns, ignore_pairs, min_force_n: float = 0.0) -> list:
    """The collisions in one state that are not expected: a contact is dropped when either geom is a finger geom
    (``finger_patterns`` are substrings of the geom name: pads, tags, fingers) or the pair matches ``ignore_pairs``
    (each entry two substrings, either order). What is left is arm / hand / tooling against the table or the eraser."""
    out = []
    for c in contacts or []:
        a, b = str(c["a"]), str(c["b"])
        if float(c.get("force_n", 0.0)) < min_force_n:
            continue
        if _has(a, finger_patterns) or _has(b, finger_patterns):
            continue
        if any((p in a and q in b) or (p in b and q in a) for p, q in ignore_pairs):
            continue
        out.append({"a": a, "b": b, "force_n": float(c.get("force_n", 0.0))})
    return out


class StateSubscriber:
    """SUB socket on the sim state PUB; ``recv(timeout_s)`` returns the next state dict or None."""

    def __init__(self, endpoint: str):
        import zmq

        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 10000)
        self.sock.connect(endpoint)

    def recv(self, timeout_s: float = 0.1):
        from robotmq import deserialize

        if not self.sock.poll(int(max(1, timeout_s * 1000)), self._zmq.POLLIN):
            return None
        return deserialize(self.sock.recv())

    def close(self):
        self.sock.close(linger=0)


class Judge(threading.Thread):
    """StateSubscriber -> SuccessTracker -> appends ``y`` / ``n`` to the commands file once. Also records, for the driver,
    what it saw: the non-finger collisions, the reflex, the grasped intervals and the state rate."""

    def __init__(self, state_endpoint: str, commands_file, success: dict, finger_patterns=(), ignore_pairs=(),
                 t_start: float | None = None, z0: float | None = None, write_label: bool = True, name: str = "judge",
                 table_patterns=("table",)):
        super().__init__(name=name, daemon=True)
        self.endpoint = state_endpoint
        self.commands = Path(commands_file)
        self.finger_patterns, self.ignore_pairs = list(finger_patterns), [tuple(p) for p in ignore_pairs]
        self.tracker = SuccessTracker(success, time.monotonic() if t_start is None else t_start, z0)
        self.write_label = write_label
        self.stop_evt, self.decided = threading.Event(), threading.Event()
        self.label: str | None = None
        self.collisions: list = []       # [{t_s, a, b, force_n}], one entry per distinct pair per contact episode
        self._active: set = set()
        self.table_patterns = list(table_patterns)
        self.pad_table: list = []        # [{t_s, a, b, force_n}] finger<->table contact onsets (reported, not a collision)
        self._active_pt: set = set()
        self.pad_table_force_max_n = 0.0
        self.grasped: list = []          # [(t_mono, bool)] edges
        self.n_states = 0
        self.t_first = self.t_last = None
        self.error: str | None = None

    def _write(self, label: str):
        with open(self.commands, "a") as f:
            f.write(label + "\n")

    def _observe(self, s: dict):
        self.n_states += 1
        t = float(s["t_mono"])
        self.t_first = self.t_first if self.t_first is not None else t
        self.t_last = t
        g = bool(s.get("grasped"))
        if not self.grasped or self.grasped[-1][1] != g:
            self.grasped.append((t, g))
        now = set()
        for c in classify_contacts(s.get("contacts"), self.finger_patterns, self.ignore_pairs):
            key = tuple(sorted((c["a"], c["b"])))
            now.add(key)
            if key not in self._active:
                self.collisions.append({"t_s": round(t - self.tracker.t_start, 4), **c})
        self._active = now
        now = set()
        for c in classify_pad_table(s.get("contacts"), self.finger_patterns, self.table_patterns):
            key = tuple(sorted((c["a"], c["b"])))
            now.add(key)
            self.pad_table_force_max_n = max(self.pad_table_force_max_n, c["force_n"])
            if key not in self._active_pt:
                self.pad_table.append({"t_s": round(t - self.tracker.t_start, 4), **c})
        self._active_pt = now

    def grasped_at(self, t_mono: float) -> bool:
        v = False
        for t, g in self.grasped:
            if t > t_mono:
                break
            v = g
        return v

    def run(self):
        try:
            sub = StateSubscriber(self.endpoint)
        except Exception as e:      # noqa: BLE001
            self.error = repr(e)
            return
        try:
            while not self.stop_evt.is_set() and self.label is None:
                s = sub.recv(0.05)
                if s is None:
                    lab = self.tracker.check_timeout(time.monotonic())
                else:
                    self._observe(s)
                    lab = self.tracker.update(s)
                if lab:
                    self.label = lab
                    if self.write_label:
                        self._write(lab)
                    self.decided.set()
            # keep observing for a moment after the decision is NOT needed: the task finalizes within ~0.1 s
        except Exception as e:      # noqa: BLE001
            self.error = repr(e)
        finally:
            sub.close()

    def stop(self):
        self.stop_evt.set()
        self.join(2.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-endpoint", required=True)
    ap.add_argument("--commands", required=True, help="the session's commands.txt")
    ap.add_argument("--deploy-config", default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="group.path=value")
    a = ap.parse_args(argv)
    from franka_sim.eval import runner as rn
    from franka_sim.eval import stage3

    ns = argparse.Namespace(deploy_config=a.deploy_config, profile=a.profile, policy=None, set=a.set, out=tempfile.mkdtemp(prefix="judge_cli_"),
                            seed=0)
    R = rn.resolve(ns)
    cfg = stage3.stage3_cfg(R)
    j = Judge(a.state_endpoint, a.commands, R.station.get("criteria.success"), cfg["finger_geom_patterns"], cfg["ignore_contact_pairs"])
    j.start()
    try:
        while j.is_alive():
            j.join(0.2)
    except KeyboardInterrupt:
        j.stop()
    print(f"[judge] {j.label} {j.tracker.result()}", flush=True)
    return 0 if j.label else 1


if __name__ == "__main__":
    sys.exit(main())
