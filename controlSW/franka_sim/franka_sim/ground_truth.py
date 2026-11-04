"""Offline ground truth of a simulated episode from its ``bridge.jsonl`` sim lines (A6; safety verdict since main-sim Step 4).

Why: Stage 3's judge decides live from the state PUB; this module re-derives the same verdict afterwards from what
the episode dir kept, so ``franka_deploy eval`` can report a physics truth (``sim_gt``) next to the human / judge label.
PURE: stdlib + numpy + yaml only (no mujoco, no zmq), so it imports in the inference env.

The rules are the judge's (franka_sim.eval.judge.EpisodeTracker), written independently. ``rules_from`` (shared by both)
only turns the config into numbers: ``criteria.safety`` / ``criteria.task`` / ``criteria.limits``, ``scene.workspace`` /
``scene.place`` and ``robot.joints`` q_min_rad / q_max_rad.

Verdict = SAFETY ONLY: FAIL at the first violation, else PASS. Rules, measured at the EE site (the bridge's point):
  contact        a non-finger contact that is not an ignored pair (``_unexpected``; finger<->table = pad_table, reported);
  reflex         ``reflex`` non-null;
  workspace      EE position outside scene.workspace [min_m, max_m];
  cone           EE z-axis more than workspace.cone_max_deg from cone_axis;
  joint_margin   an arm joint within safety.joint_margin_deg of q_min / q_max;
  speed / angular_speed   EE linear / angular speed > speed_factor x limits.v_max_m_s / w_max_rad_s, finite difference
                 between the current line and the previous line at least safety.speed_window_s back (25 Hz lines).
Checks whose inputs a line lacks (old logs: no ``ee`` / ``q``; a scene without workspace) are skipped.
Task outcome (never a FAIL), the highest level reached: none < grasped (``grasped``) < lifted (eraser z >= z0 + lift_min_m
while grasped; z0 = eraser z of the first line at or after t_start) < placed (eraser xy within place.tol_m of place.xy_m,
|z - z0| <= place_z_tol_m, not grasped, gap >= open_gap_min_m, TCP z >= eraser z + retreat_min_m, all held hold_s; a scene
without ``place`` caps the outcome at lifted). The episode ends at the first violation (``n``), at placed (``y``) or at
timeout_s (``s``); otherwise ``ended_undecided`` at the last line.
Also reported (no verdict): slip, in-hand slip (``tcp_z``), knock-over (tilt > 30 deg), contact pairs, pad_table onsets.

Sim lines (MuJoCoBackend.sim_tick, 25 Hz): {"t", "kind": "sim", "obj": [x,y,z,qw,qx,qy,qz], "gap", "grasped",
"hand_mode", "contacts": [{a, b, force_n}], "F_ext_max", "reflex", "tcp_z", "tcp", "ee" ([x,y,z,qw,qx,qy,qz]), "q" (7)}.

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
TABLE_PATTERNS = ("table",)
OUTCOMES = ("none", "grasped", "lifted", "placed")
SAFETY_RULES = ("contact", "reflex", "workspace", "cone", "joint_margin", "speed", "angular_speed")


def _tilt_deg(q) -> float:
    """Angle between the body z axis of quaternion q = [w, x, y, z] and the base z axis."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    zz = 1.0 - 2.0 * (x * x + y * y) / (n if n > 0 else 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def _zaxis(q) -> tuple:
    """Body z axis (third rotation-matrix column) of quaternion q = [w, x, y, z]."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _quat_angle(q1, q2) -> float:
    """Rotation angle (rad) between two unit quaternions [w, x, y, z]."""
    a = [float(v) for v in q1]
    b = [float(v) for v in q2]
    na = math.sqrt(sum(v * v for v in a)) or 1.0
    nb = math.sqrt(sum(v * v for v in b)) or 1.0
    d = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return 2.0 * math.acos(max(-1.0, min(1.0, d)))


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


def _pad_table(contacts, finger_patterns, table_patterns) -> list:
    out = []
    for c in contacts or []:
        a, b = str(c["a"]), str(c["b"])
        fa, fb = any(p in a for p in finger_patterns), any(p in b for p in finger_patterns)
        ta, tb = any(p in a for p in table_patterns), any(p in b for p in table_patterns)
        if (fa and tb) or (fb and ta):
            out.append((a, b, float(c.get("force_n", 0.0))))
    return out


def _g(d, key, default=None):
    try:
        v = d[key]
    except (KeyError, TypeError, IndexError):
        return default
    return default if v is None else v


def rules_from(criteria, scene=None, robot=None) -> dict:
    """The numbers both judges use, from the criteria group (``safety``, ``task``, ``limits``), the scene group
    (``workspace``, ``place``) and the robot group (``joints.q_min_rad/q_max_rad``). Already-built rules pass through.
    Missing scene / robot parts give None (that check is skipped); missing criteria blocks raise ValueError."""
    if isinstance(criteria, dict) and criteria.get("_rules"):
        return criteria
    s, t, lim = _g(criteria, "safety"), _g(criteria, "task"), _g(criteria, "limits")
    if not s or not t or not lim:
        raise ValueError("criteria needs safety, task and limits blocks (run/criteria/pick_place.yaml), "
                         f"got {sorted(criteria) if hasattr(criteria, 'keys') else criteria!r}")
    sf = _g(s, "speed_factor", _g(s, "speed_factor_ratio"))
    if sf is None:
        raise ValueError("criteria.safety needs speed_factor (or speed_factor_ratio)")
    ws, place = _g(scene, "workspace"), _g(scene, "place")
    joints = _g(robot, "joints") or {}
    qmin, qmax = _g(joints, "q_min_rad"), _g(joints, "q_max_rad")
    ax = None
    if ws and _g(ws, "cone_axis") is not None:
        ax = [float(v) for v in ws["cone_axis"]]
        n = math.sqrt(sum(v * v for v in ax)) or 1.0
        ax = [v / n for v in ax]
    return {
        "_rules": True,
        "joint_margin_deg": float(s["joint_margin_deg"]),
        "speed_factor": float(sf), "speed_window_s": float(s["speed_window_s"]),
        "v_lim_m_s": float(sf) * float(lim["v_max_m_s"]), "w_lim_rad_s": float(sf) * float(lim["w_max_rad_s"]),
        "ws_min_m": None if not ws else [float(v) for v in ws["min_m"]],
        "ws_max_m": None if not ws else [float(v) for v in ws["max_m"]],
        "cone_axis": ax, "cone_max_deg": None if not ws or _g(ws, "cone_max_deg") is None else float(ws["cone_max_deg"]),
        "q_min_rad": None if qmin is None else [float(v) for v in qmin],
        "q_max_rad": None if qmax is None else [float(v) for v in qmax],
        "lift_min_m": float(t["lift_min_m"]), "timeout_s": float(t["timeout_s"]), "hold_s": float(t["hold_s"]),
        "place_z_tol_m": float(t["place_z_tol_m"]), "open_gap_min_m": float(t["open_gap_min_m"]),
        "retreat_min_m": float(t["retreat_min_m"]),
        "place_xy_m": None if not place else [float(v) for v in place["xy_m"]],
        "place_tol_m": None if not place else float(place["tol_m"]),
    }


def _violation(rule, t_rel, value, limit) -> dict:
    return {"rule": rule, "t_s": round(float(t_rel), 4), "value": value, "limit": limit}


def evaluate(events: list, criteria, t_start: float, *, scene=None, robot=None, finger_patterns=FINGER_PATTERNS,
             ignore_pairs=IGNORE_PAIRS, table_patterns=TABLE_PATTERNS) -> dict:
    """The safety verdict and task outcome of one episode from its sim lines (``kind == "sim"``; other kinds skipped).
    ``criteria`` is the criteria group (with ``scene`` / ``robot`` groups) or a ``rules_from`` dict."""
    R = rules_from(criteria, scene, robot)
    finger_patterns = tuple(finger_patterns)
    ignore_pairs = tuple(tuple(p) for p in ignore_pairs)
    t_start = float(t_start)
    ev = sorted((e for e in events if e.get("kind") == "sim" and float(e["t"]) >= t_start), key=lambda e: float(e["t"]))
    r = {"verdict": None, "violation": None, "outcome": "none", "label": None, "reason": None, "t_end_s": None,
         "grasp_t_s": None, "lift_t_s": None, "t_placed_s": None, "max_lift_m": 0.0, "slip_m": 0.0, "knocked_over": False,
         "reflex": None, "ee_speed_max_m_s": None, "ee_ang_speed_max_rad_s": None, "joint_margin_min_deg": None,
         "checks_skipped": []}
    level = 0
    z0 = None
    since = None
    z_max_g = None
    rel_max_g, in_hand = None, None
    pairs, collisions, active = set(), [], set()
    pad_table, active_pt, pt_fmax = [], set(), 0.0
    hist = []                           # [(t, ee)] for the speed rule
    used = []
    skipped = set()
    jm = math.radians(R["joint_margin_deg"])

    def end(label, reason, t):
        r["label"], r["reason"], r["t_end_s"] = label, reason, round(t - t_start, 4)

    for e in ev:
        t = float(e["t"])
        trel = t - t_start
        used.append(t)
        obj = e["obj"]
        z = float(obj[2])
        z0 = z if z0 is None else z0
        lift = z - z0
        r["max_lift_m"] = max(r["max_lift_m"], lift)
        grasped = bool(e.get("grasped"))
        unexp = _unexpected(e.get("contacts"), finger_patterns, ignore_pairs)
        now = set()
        for a, b, f in unexp:
            key = tuple(sorted((a, b)))
            now.add(key)
            if key not in active:
                collisions.append({"t_s": round(trel, 4), "a": a, "b": b, "force_n": f})
        active = now
        now = set()
        for a, b, f in _pad_table(e.get("contacts"), finger_patterns, table_patterns):
            key = tuple(sorted((a, b)))
            now.add(key)
            pt_fmax = max(pt_fmax, f)
            if key not in active_pt:
                pad_table.append({"t_s": round(trel, 4), "a": a, "b": b, "force_n": f})
        active_pt = now
        for c in e.get("contacts") or []:
            pairs.add(tuple(sorted((str(c["a"]), str(c["b"])))))
        if _tilt_deg(obj[3:7]) > KNOCK_OVER_DEG:
            r["knocked_over"] = True
        # ---- safety
        v = None
        ee = e.get("ee")
        q = e.get("q")
        if unexp:
            a, b, f = unexp[0]
            v = _violation("contact", trel, f"{a}|{b} {f:.2f} N", 0)
        elif e.get("reflex"):
            r["reflex"] = str(e["reflex"])
            v = _violation("reflex", trel, str(e["reflex"]), None)
        if v is None and ee is None:
            skipped.update(("workspace", "cone", "speed", "angular_speed"))
        if v is None and ee is not None and R["ws_min_m"] is not None:
            p = [float(x) for x in ee[:3]]
            if any(p[i] < R["ws_min_m"][i] or p[i] > R["ws_max_m"][i] for i in range(3)):
                v = _violation("workspace", trel, [round(x, 4) for x in p], [R["ws_min_m"], R["ws_max_m"]])
        if v is None and ee is not None and R["cone_axis"] is not None and R["cone_max_deg"] is not None:
            zx = _zaxis(ee[3:7])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(zx, R["cone_axis"]))))))
            if ang > R["cone_max_deg"]:
                v = _violation("cone", trel, round(ang, 3), R["cone_max_deg"])
        if v is None and q is not None and R["q_min_rad"] is not None:
            qq = [float(x) for x in q[:7]]
            m = [min(qq[i] - R["q_min_rad"][i], R["q_max_rad"][i] - qq[i]) for i in range(min(len(qq), len(R["q_min_rad"])))]
            i = min(range(len(m)), key=lambda k: m[k])
            md = math.degrees(m[i])
            r["joint_margin_min_deg"] = md if r["joint_margin_min_deg"] is None else min(r["joint_margin_min_deg"], md)
            if m[i] <= jm:
                v = _violation("joint_margin", trel, {"joint": i + 1, "margin_deg": round(md, 3)}, R["joint_margin_deg"])
        elif q is None:
            skipped.add("joint_margin")
        if ee is not None:
            prev = None
            for tp, eep in reversed(hist):
                if t - tp >= R["speed_window_s"] - 1e-9:
                    prev = (tp, eep)
                    break
            hist.append((t, ee))
            if prev is not None and t > prev[0]:
                dt = t - prev[0]
                vlin = math.dist([float(x) for x in ee[:3]], [float(x) for x in prev[1][:3]]) / dt
                vang = _quat_angle(ee[3:7], prev[1][3:7]) / dt
                r["ee_speed_max_m_s"] = max(r["ee_speed_max_m_s"] or 0.0, vlin)
                r["ee_ang_speed_max_rad_s"] = max(r["ee_ang_speed_max_rad_s"] or 0.0, vang)
                if v is None and vlin > R["v_lim_m_s"]:
                    v = _violation("speed", trel, round(vlin, 4), round(R["v_lim_m_s"], 4))
                if v is None and vang > R["w_lim_rad_s"]:
                    v = _violation("angular_speed", trel, round(vang, 4), round(R["w_lim_rad_s"], 4))
        if v is not None:
            r["violation"] = v
            end("n", f"safety:{v['rule']}", t)
            break
        # ---- task outcome
        if grasped:
            if r["grasp_t_s"] is None:
                r["grasp_t_s"] = round(trel, 4)
            level = max(level, 1)
            z_max_g = z if z_max_g is None else max(z_max_g, z)
            r["slip_m"] = max(r["slip_m"], z_max_g - z)
            if e.get("tcp_z") is not None:
                rel = z - float(e["tcp_z"])
                rel_max_g = rel if rel_max_g is None else max(rel_max_g, rel)
                in_hand = max(in_hand or 0.0, rel_max_g - rel)
            if lift >= R["lift_min_m"] and level < 2:
                level, r["lift_t_s"] = 2, round(trel, 4)
        tcp = e.get("tcp")
        tcp_z = float(tcp[2]) if tcp is not None else (float(e["tcp_z"]) if e.get("tcp_z") is not None else None)
        placed_now = False
        if R["place_xy_m"] is not None and tcp_z is not None and e.get("gap") is not None:
            dxy = math.hypot(float(obj[0]) - R["place_xy_m"][0], float(obj[1]) - R["place_xy_m"][1])
            placed_now = (dxy <= R["place_tol_m"] and abs(z - z0) <= R["place_z_tol_m"] and not grasped
                          and float(e["gap"]) >= R["open_gap_min_m"] and tcp_z >= z + R["retreat_min_m"])
        if placed_now:
            since = t if since is None else since
            if t - since >= R["hold_s"]:
                level, r["t_placed_s"] = 3, round(trel, 4)
                end("y", "placed", t)
                break
        else:
            since = None
        if trel >= R["timeout_s"]:
            end("s", "timeout", t)
            break
    if r["label"] is None and used:
        end(None, "ended_undecided", used[-1])       # the episode ended (judge / operator) before a decision
    if R["place_xy_m"] is None:
        skipped.add("placed (scene has no place)")
    out = dict(r)
    out["verdict"] = "FAIL" if r["violation"] else "PASS"
    out["outcome"] = OUTCOMES[level]
    out["success"] = level == 3                      # informational: task success = placed (never the verdict)
    out["max_lift_m"] = round(out["max_lift_m"], 5)
    out["slip_m"] = round(out["slip_m"], 5)
    out["in_hand_slip_m"] = None if in_hand is None else round(in_hand, 5)
    for k in ("ee_speed_max_m_s", "ee_ang_speed_max_rad_s", "joint_margin_min_deg"):
        out[k] = None if out[k] is None else round(out[k], 4)
    out["checks_skipped"] = sorted(skipped)
    out["rules"] = {k: v for k, v in R.items() if k != "_rules"}
    out["t_start_mono"] = t_start
    out["t_end_mono"] = None if out["t_end_s"] is None else round(t_start + out["t_end_s"], 4)
    out["arm_table_contact"] = collisions
    out["n_arm_table_contacts"] = len(collisions)
    out["pad_table"] = pad_table
    out["pad_table_force_max_n"] = pt_fmax
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


def evaluate_episode(ep_dir, criteria=None, t_start: float | None = None, **kw) -> dict | None:
    """``evaluate`` on ``<ep_dir>/bridge.jsonl``. criteria None -> the episode's ``config/station_resolved.yaml``
    criteria (and its scene / robot groups unless ``scene=`` / ``robot=`` are given); t_start None -> the
    ``episode_start`` t of ``events.jsonl``; finger patterns / ignore pairs from that station's ``sim.stage3`` unless
    given. None when the episode has no sim lines (a real-arm episode)."""
    ep_dir = Path(ep_dir)
    events = [e for e in _read_jsonl(ep_dir / "bridge.jsonl") if e.get("kind") == "sim"]
    if not events:
        return None
    built = isinstance(criteria, dict) and criteria.get("_rules")
    groups = None
    if criteria is None or not built and ("scene" not in kw or "robot" not in kw) or (
            "finger_patterns" not in kw or "ignore_pairs" not in kw):
        groups = _station_groups(ep_dir)
    if criteria is None:
        criteria = (groups or {}).get("criteria")
        if not criteria:
            raise ValueError(f"{ep_dir}: no criteria given and no criteria in config/station_resolved.yaml")
    if not built:
        kw.setdefault("scene", (groups or {}).get("scene"))
        kw.setdefault("robot", (groups or {}).get("robot"))
    st3 = ((groups or {}).get("sim") or {}).get("stage3") or {}
    kw.setdefault("finger_patterns", tuple(st3.get("finger_geom_patterns") or FINGER_PATTERNS))
    kw.setdefault("ignore_pairs", tuple(tuple(p) for p in (st3.get("ignore_contact_pairs") or IGNORE_PAIRS)))
    kw.setdefault("table_patterns", tuple(st3.get("table_geom_patterns") or TABLE_PATTERNS))
    if t_start is None:
        st = [e for e in _read_jsonl(ep_dir / "events.jsonl") if e.get("kind") == "episode_start"]
        t_start = float(st[0]["t"]) if st else float(min(float(e["t"]) for e in events))
    return evaluate(events, criteria, t_start, **kw)
