"""Offline ground truth of a simulated episode from its ``bridge.jsonl`` sim lines (A6).

Why: Stage 3's judge decides live from the state PUB; this module re-derives the same verdict afterwards from what
the episode dir kept, so ``franka_deploy eval`` can report a physics truth (``sim_gt``) next to the human / judge label.
PURE: stdlib + numpy + yaml only (no mujoco, no zmq), so it imports in the inference env.

The rule is the judge's (franka_sim.eval.judge.SuccessTracker), written independently:
  z0 = the eraser z of the first sample at or after t_start; lift = z - z0;
  success = lift >= success.lift_min_m held continuously for hold_s (a drop below restarts the clock), decided at the
  first sample where that holds; a non-null ``reflex`` -> n; tilt > 30 deg -> n (knocked_over); no decision by
  timeout_s -> n. Slip = the largest drop below the running max of z while ``grasped`` (reported vs slip_tol_m).

Sim lines (MuJoCoBackend.sim_tick, 25 Hz): {"t", "kind": "sim", "obj": [x,y,z,qw,qx,qy,qz], "gap", "grasped",
"hand_mode", "contacts": [{a, b, force_n}], "F_ext_max", "reflex"}.

    from franka_sim.ground_truth import evaluate_episode
    gt = evaluate_episode(ep_dir)     # None for an episode without sim lines (real arm)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

KNOCK_OVER_DEG = 30.0
FINGER_PATTERNS = ("pad_", "tag_", "finger")
IGNORE_PAIRS = (("eraser", "table"),)


def _tilt_deg(q) -> float:
    """Angle between the body z axis of quaternion q = [w, x, y, z] and the base z axis."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    zz = 1.0 - 2.0 * (x * x + y * y) / (n if n > 0 else 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def _unexpected(contacts, finger_patterns, ignore_pairs) -> list:
    """Contacts where neither geom is a finger geom and the pair is not ignored (judge.classify_contacts semantics)."""
    out = []
    for c in contacts or []:
        a, b = str(c["a"]), str(c["b"])
        if any(p in a for p in finger_patterns) or any(p in b for p in finger_patterns):
            continue
        if any((p in a and q in b) or (p in b and q in a) for p, q in ignore_pairs):
            continue
        out.append((a, b, float(c.get("force_n", 0.0))))
    return out


def _success_cfg(criteria: dict) -> dict:
    s = criteria.get("success", criteria) if isinstance(criteria, dict) else None
    if not s or "lift_min_m" not in s:
        raise ValueError("criteria needs success.{lift_min_m, hold_s, timeout_s, slip_tol_m}")
    return s


def evaluate(events: list, criteria: dict, t_start: float, *, finger_patterns=FINGER_PATTERNS,
             ignore_pairs=IGNORE_PAIRS) -> dict:
    """The verdict of one episode from its sim lines (``kind == "sim"``; other kinds are skipped)."""
    s = _success_cfg(criteria)
    lift_min, hold_s, timeout_s, slip_tol = (float(s[k]) for k in ("lift_min_m", "hold_s", "timeout_s", "slip_tol_m"))
    finger_patterns = tuple(finger_patterns)
    ignore_pairs = tuple(tuple(p) for p in ignore_pairs)
    t_start = float(t_start)
    ev = sorted((e for e in events if e.get("kind") == "sim" and float(e["t"]) >= t_start), key=lambda e: float(e["t"]))
    r = {"success": None, "t_success_s": None, "grasp_t_s": None, "max_lift_m": 0.0, "slip_m": 0.0,
         "knocked_over": False, "reflex": None, "reason": None}
    z0 = None
    since = None
    z_max_g = None
    t_dec = None
    pairs, collisions, active = set(), [], set()
    used = []

    def decide(label: str, reason: str, t: float):
        nonlocal t_dec
        r["success"], r["reason"], t_dec = label == "y", reason, t
        if label == "y":
            r["t_success_s"] = round(t - t_start, 4)

    for e in ev:
        t = float(e["t"])
        used.append(t)
        obj = e["obj"]
        z = float(obj[2])
        z0 = z if z0 is None else z0
        lift = z - z0
        r["max_lift_m"] = max(r["max_lift_m"], lift)
        now = set()
        for a, b, f in _unexpected(e.get("contacts"), finger_patterns, ignore_pairs):
            key = tuple(sorted((a, b)))
            now.add(key)
            if key not in active:
                collisions.append({"t_s": round(t - t_start, 4), "a": a, "b": b, "force_n": f})
        active = now
        for c in e.get("contacts") or []:
            pairs.add(tuple(sorted((str(c["a"]), str(c["b"])))))
        if e.get("reflex"):
            r["reflex"] = str(e["reflex"])
            decide("n", "reflex", t)
            break
        if _tilt_deg(obj[3:7]) > KNOCK_OVER_DEG:
            r["knocked_over"] = True
            decide("n", "knocked_over", t)
            break
        if e.get("grasped"):
            if r["grasp_t_s"] is None:
                r["grasp_t_s"] = round(t - t_start, 4)
            z_max_g = z if z_max_g is None else max(z_max_g, z)
            r["slip_m"] = max(r["slip_m"], z_max_g - z)
        if lift >= lift_min:
            since = t if since is None else since
            if t - since >= hold_s:
                decide("y", "lift_held", t)
                break
        else:
            since = None
        if t - t_start >= timeout_s:
            decide("n", "timeout", t)
            break
    if r["success"] is None and used:
        decide("n", "ended_undecided", used[-1])     # the episode ended (judge / operator) before success or timeout
    out = dict(r)
    out["max_lift_m"] = round(out["max_lift_m"], 5)
    out["slip_m"] = round(out["slip_m"], 5)
    out["slip_exceeds_tol"] = out["slip_m"] > slip_tol
    out.update(lift_min_m=lift_min, hold_s=hold_s, timeout_s=timeout_s, slip_tol_m=slip_tol)
    out["t_success_mono"] = None if out["t_success_s"] is None else round(t_start + out["t_success_s"], 4)
    out["t_start_mono"] = t_start
    out["arm_table_contact"] = collisions
    out["n_arm_table_contacts"] = len(collisions)
    out["contact_pairs"] = [list(p) for p in sorted(pairs)]
    out["n_samples"] = len(used)
    out["rate_hz"] = round((len(used) - 1) / (used[-1] - used[0]), 2) if len(used) > 1 and used[-1] > used[0] else None
    out["source"] = "bridge.jsonl"
    return out


def _read_jsonl(p: Path) -> list:
    out = []
    if not p.is_file():
        return out
    with open(p, errors="replace") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def _station_groups(ep_dir: Path) -> dict:
    p = ep_dir / "config" / "station_resolved.yaml"
    if not p.is_file():
        return {}
    import yaml

    doc = yaml.safe_load(p.read_text()) or {}
    return doc.get("groups", doc) or {}


def evaluate_episode(ep_dir, criteria: dict | None = None, t_start: float | None = None, **kw) -> dict | None:
    """``evaluate`` on ``<ep_dir>/bridge.jsonl``. criteria None -> the episode's ``config/station_resolved.yaml``
    criteria; t_start None -> the ``episode_start`` t of ``events.jsonl``; finger patterns / ignore pairs from that
    station's ``sim.stage3`` unless given. None when the episode has no sim lines (a real-arm episode)."""
    ep_dir = Path(ep_dir)
    events = [e for e in _read_jsonl(ep_dir / "bridge.jsonl") if e.get("kind") == "sim"]
    if not events:
        return None
    groups = None
    if criteria is None or ("finger_patterns" not in kw or "ignore_pairs" not in kw):
        groups = _station_groups(ep_dir)
    if criteria is None:
        criteria = (groups or {}).get("criteria")
        if not criteria:
            raise ValueError(f"{ep_dir}: no criteria given and no criteria in config/station_resolved.yaml")
    st3 = ((groups or {}).get("sim") or {}).get("stage3") or {}
    kw.setdefault("finger_patterns", tuple(st3.get("finger_geom_patterns") or FINGER_PATTERNS))
    kw.setdefault("ignore_pairs", tuple(tuple(p) for p in (st3.get("ignore_contact_pairs") or IGNORE_PAIRS)))
    if t_start is None:
        st = [e for e in _read_jsonl(ep_dir / "events.jsonl") if e.get("kind") == "episode_start"]
        t_start = float(st[0]["t"]) if st else float(min(float(e["t"]) for e in events))
    return evaluate(events, criteria, t_start, **kw)
