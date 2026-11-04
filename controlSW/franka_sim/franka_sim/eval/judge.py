"""Stage 3 judge: the SAFETY verdict and the task outcome of an episode from the sim state; writes ``y`` / ``n`` / ``s``
to the task's command file.

Why: the closed-loop episode has no human to press the label key. The judge watches the simulator's state stream
(I-SIMSTATE, one ``robotmq.serialize(dict)`` ZMQ part per message, 100 Hz: ``{t_mono, qpos, qvel, eraser_T_base (4x4),
hand_gap_m, grasped, contacts: [{a, b, force_n}], reflex, tau_ext, tcp_T_base (4x4), ee_T_base (4x4), q_arm (7), tcp_z}``)
and applies ``ground_truth.rules_from(criteria, scene, robot)``: ``criteria.safety`` / ``criteria.task`` /
``criteria.limits``, ``scene.workspace`` / ``scene.place``, ``robot.joints``. None of those numbers lives in this file.

* ``EpisodeTracker`` is pure (no I/O, no zmq): ``update(state) -> "y" | "n" | "s" | None``, ``result() -> dict``.
* ``classify_contacts`` is pure: the non-finger collisions of one state (arm / hand / tooling against table / eraser).
* ``Judge`` is a thread: SUB socket -> tracker -> appends the label to ``<session>/commands.txt`` exactly once.

Verdict = SAFETY ONLY (an episode FAILs only on a violation; the task outcome never fails it). Rules, measured at the EE
site (``ee_T_base``, the point the bridge's safety filter enforces): ``contact`` (a ``classify_contacts`` collision; finger
<-> table is pad_table, reported apart), ``reflex`` (``reflex`` non-null), ``workspace`` (EE outside [min_m, max_m]),
``cone`` (EE z-axis more than cone_max_deg from cone_axis), ``joint_margin`` (a joint of ``q_arm`` within
joint_margin_deg of its limit), ``speed`` / ``angular_speed`` (EE speed > speed_factor x v_max / w_max, finite difference
between the current state and the oldest state within the last speed_window_s). A state without a key skips that check.
Task outcome, highest level reached: none < grasped < lifted (eraser z >= z0 + lift_min_m while grasped; z0 = eraser z
at the episode start) < placed (eraser xy within place.tol_m of place.xy_m, |z - z0| <= place_z_tol_m, not grasped,
gap >= open_gap_min_m, TCP z >= eraser z + retreat_min_m, all held hold_s; no ``scene.place`` -> capped at lifted).
Ending: violation -> ``n`` (FAIL); placed -> ``y``; timeout_s -> ``s``. Knock-over (tilt > 30 deg) and slip are reported.

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

import numpy as np

from franka_sim.ground_truth import OUTCOMES, rules_from

KNOCK_OVER_DEG = 30.0


def _rot_angle(Ra, Rb) -> float:
    """Rotation angle (rad) between two rotation matrices."""
    c = (float(np.trace(np.asarray(Ra, float).T @ np.asarray(Rb, float))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


class EpisodeTracker:
    """Pure safety + outcome rule. ``t_start`` and the states' ``t_mono`` share one clock (host monotonic seconds).
    ``rules``: a ``ground_truth.rules_from`` dict (or a criteria group, then ``scene`` / ``robot`` give the limits)."""

    def __init__(self, rules, t_start: float, z0: float | None = None, knock_over_deg: float = KNOCK_OVER_DEG,
                 finger_patterns=(), ignore_pairs=(), scene=None, robot=None):
        self.R = rules_from(rules, scene, robot)
        self.timeout_s = self.R["timeout_s"]
        self.knock_deg = float(knock_over_deg)
        self.finger_patterns, self.ignore_pairs = list(finger_patterns), [tuple(p) for p in ignore_pairs]
        self.t_start = float(t_start)
        self.z0 = z0
        self.decision: str | None = None
        self.level = 0
        self._placed_since: float | None = None
        self._z_max_grasped: float | None = None
        self._hist: list = []           # [(t, p (3,), R (3,3))] EE samples inside the speed window
        self._skipped: set = set()
        self.r = {"verdict": None, "violation": None, "outcome": "none", "label": None, "reason": None, "t_end_s": None,
                  "grasp_t_s": None, "lift_t_s": None, "t_placed_s": None, "max_lift_m": 0.0, "slip_m": 0.0,
                  "knocked_over": False, "reflex": None, "ee_speed_max_m_s": None, "ee_ang_speed_max_rad_s": None,
                  "joint_margin_min_deg": None}

    @staticmethod
    def tilt_deg(T) -> float:
        """Angle between the eraser's z axis and the base z axis."""
        zz = float(T[2][2])
        return math.degrees(math.acos(max(-1.0, min(1.0, zz))))

    def _decide(self, label: str, reason: str, t: float) -> str | None:
        if self.decision is not None:
            return None
        self.decision = label
        self.r["label"], self.r["reason"], self.r["t_end_s"] = label, reason, round(t - self.t_start, 4)
        return label

    def check_timeout(self, t_mono: float) -> str | None:
        """Time alone can end the episode (a silent simulator must not hang the run)."""
        if self.decision is None and t_mono - self.t_start >= self.timeout_s:
            return self._decide("s", "timeout", t_mono)
        return None

    def _safety(self, s: dict, t: float) -> dict | None:
        R, trel = self.R, t - self.t_start

        def viol(rule, value, limit):
            return {"rule": rule, "t_s": round(trel, 4), "value": value, "limit": limit}

        cols = classify_contacts(s.get("contacts"), self.finger_patterns, self.ignore_pairs)
        if cols:
            c = cols[0]
            return viol("contact", f"{c['a']}|{c['b']} {c['force_n']:.2f} N", 0)
        if s.get("reflex"):
            self.r["reflex"] = str(s["reflex"])
            return viol("reflex", str(s["reflex"]), None)
        v = None
        E = s.get("ee_T_base")
        if E is None:
            self._skipped.update(("workspace", "cone", "speed", "angular_speed"))
        else:
            E = np.asarray(E, float)
            p, Rm = E[:3, 3], E[:3, :3]
            if R["ws_min_m"] is not None and (np.any(p < np.asarray(R["ws_min_m"])) or np.any(p > np.asarray(R["ws_max_m"]))):
                v = viol("workspace", [round(float(x), 4) for x in p], [R["ws_min_m"], R["ws_max_m"]])
            if v is None and R["cone_axis"] is not None and R["cone_max_deg"] is not None:
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(Rm[:, 2], R["cone_axis"]))))))
                if ang > R["cone_max_deg"]:
                    v = viol("cone", round(ang, 3), R["cone_max_deg"])
        q = s.get("q_arm")
        if q is None:
            self._skipped.add("joint_margin")
        elif v is None and R["q_min_rad"] is not None:
            q = np.asarray(q, float)[:7]
            n = min(len(q), len(R["q_min_rad"]))
            m = np.minimum(q[:n] - np.asarray(R["q_min_rad"][:n]), np.asarray(R["q_max_rad"][:n]) - q[:n])
            i = int(np.argmin(m))
            md = math.degrees(float(m[i]))
            jm = self.r["joint_margin_min_deg"]
            self.r["joint_margin_min_deg"] = md if jm is None else min(jm, md)
            if float(m[i]) <= math.radians(R["joint_margin_deg"]):
                v = viol("joint_margin", {"joint": i + 1, "margin_deg": round(md, 3)}, R["joint_margin_deg"])
        if E is not None:
            w = R["speed_window_s"]
            self._hist = [h for h in self._hist if t - h[0] <= w + 1e-9]
            if self._hist and t > self._hist[0][0]:
                t0, p0, R0 = self._hist[0]         # the oldest sample within the window
                dt = t - t0
                vlin = float(np.linalg.norm(p - p0)) / dt
                vang = _rot_angle(R0, Rm) / dt
                self.r["ee_speed_max_m_s"] = max(self.r["ee_speed_max_m_s"] or 0.0, vlin)
                self.r["ee_ang_speed_max_rad_s"] = max(self.r["ee_ang_speed_max_rad_s"] or 0.0, vang)
                if v is None and vlin > R["v_lim_m_s"]:
                    v = viol("speed", round(vlin, 4), round(R["v_lim_m_s"], 4))
                if v is None and vang > R["w_lim_rad_s"]:
                    v = viol("angular_speed", round(vang, 4), round(R["w_lim_rad_s"], 4))
            self._hist.append((t, p.copy(), Rm.copy()))
        return v

    def update(self, s: dict) -> str | None:
        """Feed one state; returns the label the first time the episode is decided, else None."""
        if self.decision is not None:
            return None
        t = float(s["t_mono"])
        if t < self.t_start:
            return None
        R = self.R
        T = s["eraser_T_base"]
        z = float(T[2][3])
        if self.z0 is None:
            self.z0 = z
        lift = z - self.z0
        self.r["max_lift_m"] = max(self.r["max_lift_m"], lift)
        if self.tilt_deg(T) > self.knock_deg:
            self.r["knocked_over"] = True
        v = self._safety(s, t)
        if v is not None:
            self.r["violation"] = v
            return self._decide("n", f"safety:{v['rule']}", t)
        grasped = bool(s.get("grasped"))
        if grasped:
            if self.r["grasp_t_s"] is None:
                self.r["grasp_t_s"] = round(t - self.t_start, 4)
            self.level = max(self.level, 1)
            self._z_max_grasped = z if self._z_max_grasped is None else max(self._z_max_grasped, z)
            self.r["slip_m"] = max(self.r["slip_m"], self._z_max_grasped - z)
            if lift >= R["lift_min_m"] and self.level < 2:
                self.level, self.r["lift_t_s"] = 2, round(t - self.t_start, 4)
        tcp = s.get("tcp_T_base")
        tcp_z = float(tcp[2][3]) if tcp is not None else (float(s["tcp_z"]) if s.get("tcp_z") is not None else None)
        placed_now = False
        if R["place_xy_m"] is not None and tcp_z is not None and s.get("hand_gap_m") is not None:
            dxy = math.hypot(float(T[0][3]) - R["place_xy_m"][0], float(T[1][3]) - R["place_xy_m"][1])
            placed_now = (dxy <= R["place_tol_m"] and abs(z - self.z0) <= R["place_z_tol_m"] and not grasped
                          and float(s["hand_gap_m"]) >= R["open_gap_min_m"] and tcp_z >= z + R["retreat_min_m"])
        if placed_now:
            self._placed_since = t if self._placed_since is None else self._placed_since
            if t - self._placed_since >= R["hold_s"]:
                self.level, self.r["t_placed_s"] = 3, round(t - self.t_start, 4)
                return self._decide("y", "placed", t)
        else:
            self._placed_since = None
        return self.check_timeout(t)

    def result(self) -> dict:
        out = dict(self.r)
        out["verdict"] = "FAIL" if out["violation"] else "PASS"
        out["outcome"] = OUTCOMES[self.level]
        out["success"] = self.level == 3                 # informational: task success = placed (never the verdict)
        out["max_lift_m"] = round(out["max_lift_m"], 5)
        out["slip_m"] = round(out["slip_m"], 5)
        for k in ("ee_speed_max_m_s", "ee_ang_speed_max_rad_s", "joint_margin_min_deg"):
            out[k] = None if out[k] is None else round(out[k], 4)
        sk = set(self._skipped) | ({"placed (scene has no place)"} if self.R["place_xy_m"] is None else set())
        out["checks_skipped"] = sorted(sk)
        out["rules"] = {k: v for k, v in self.R.items() if k != "_rules"}
        return out


SuccessTracker = EpisodeTracker       # old name


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
    """StateSubscriber -> EpisodeTracker -> appends ``y`` / ``n`` / ``s`` to the commands file once. Also records, for the driver,
    what it saw: the non-finger collisions, the reflex, the grasped intervals and the state rate."""

    def __init__(self, state_endpoint: str, commands_file, rules, finger_patterns=(), ignore_pairs=(),
                 t_start: float | None = None, z0: float | None = None, write_label: bool = True, name: str = "judge",
                 table_patterns=("table",), scene=None, robot=None):
        super().__init__(name=name, daemon=True)
        self.endpoint = state_endpoint
        self.commands = Path(commands_file)
        self.finger_patterns, self.ignore_pairs = list(finger_patterns), [tuple(p) for p in ignore_pairs]
        self.tracker = EpisodeTracker(rules, time.monotonic() if t_start is None else t_start, z0,
                                      finger_patterns=self.finger_patterns, ignore_pairs=self.ignore_pairs, scene=scene, robot=robot)
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
    j = Judge(a.state_endpoint, a.commands, stage3.judge_rules(R), cfg["finger_geom_patterns"], cfg["ignore_contact_pairs"],
              table_patterns=cfg.get("table_geom_patterns", ["table"]))
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
