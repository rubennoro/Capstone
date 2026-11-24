"""Score one eval folder from its recorded files: ``score_eval(eval_dir) -> dict`` (score.json's content).

Reads only. Per rollout it prefers the sim_deploy.rollout/v2 record (``rollouts/NNN/gt.jsonl`` + ``rollout.json``) and
falls back to ``trace.json`` + the driver's ``rollouts.jsonl`` line for old folders; every metric the source cannot
give is null and named in ``missing``. The rules are constants.py's (README.md "Metrics"); success = the lift with
the eraser clear of the table (score schema v4, user decisions 2026-11-22; trace.json-only folders: the lift alone,
``clear_of_table`` listed in ``missing``), grasp a reported metric, ``lift_without_grasp`` flags a success without a
grasp row in its hold window.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from sim_deploy.eval.report import wilson
from sim_deploy.evaltool import constants as C

SCORER_VERSION = "4"                 # 3: + clear of the table, gpu_fault, start split, within-time rates, timing (v4)
                                     # 4: gpu_fault = overlap with the window [xid_time, cleared)


# ----------------------------------------------------------------------------------------------------- reading

def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list:
    out = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def load_eval(eval_dir: Path) -> tuple[dict, list]:
    """(eval.yaml, rollouts.jsonl records). Raises FileNotFoundError without eval.yaml."""
    eval_dir = Path(eval_dir)
    doc = yaml.safe_load((eval_dir / "eval.yaml").read_text()) or {}
    recs = _jsonl(eval_dir / "rollouts.jsonl")
    if not recs:                                   # no rollouts.jsonl: one record per rollouts/NNN/rollout.json
        for d in sorted((eval_dir / "rollouts").glob("[0-9][0-9][0-9]")):
            ej = _json(d / "rollout.json")
            if ej is not None:
                recs.append({"n": int(d.name), **ej})
    return doc, recs


def eraser_geometry(doc: dict) -> tuple[list | None, float | None]:
    """(eraser box extents [x, y, z] m, table top z) from eval.yaml ``scene.station_scene``; (None, None) if absent."""
    sc = (doc.get("scene") or {}).get("station_scene") or {}
    try:
        from sim_deploy.model.build import eraser_box
        return [float(v) for v in eraser_box(sc)], float(sc["table"]["top_z_m"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _rows_from_gt(gt: list, z0: float, box=None, top=None) -> list:
    rows = []
    for g in gt:
        if g.get("t") is None:
            continue
        ep = g.get("eraser_pose")
        tp = g.get("tcp_pose")
        sides = {C.GRASP_GEOM.match(n).group(2) for n in (g.get("contacts_eraser") or []) if C.GRASP_GEOM.match(n)}
        rows.append({
            "t": float(g["t"]),
            "lift": (float(ep[2]) - z0) if ep is not None else None,
            "clear": (C.clear_of_table(ep, box, top) if ep is not None and len(ep) >= 7 and box is not None
                      and top is not None else None),
            "grasp": sides == {"l", "r"},
            "gap": g.get("gap_m"),
            "tcp": [float(x) for x in tp[:3]] if tp else None,
            "eraser": [float(x) for x in ep[:3]] if ep else None,
            "table": list(g.get("contacts_table") or []),
            "min_z_fr": g.get("min_z_fingers_rig_m"),
            "c2_mode": g.get("c2_mode"),
            "reflex": g.get("reflex"),
        })
    return rows


def _rows_from_trace(trace: dict, eraser_xyz) -> list:
    cols = trace.get("columns") or []
    ix = {c: i for i, c in enumerate(cols)}
    rows = []
    for r in trace.get("rows") or []:
        tcp = r[ix["tcp_O_xyz_wxyz"]] if "tcp_O_xyz_wxyz" in ix and len(r) > ix["tcp_O_xyz_wxyz"] else None
        rows.append({
            "t": float(r[ix["t_s"]]),
            "lift": float(r[ix["lift_m"]]) if "lift_m" in ix else None,
            "clear": None,                                 # no eraser orientation in trace.json
            "grasp": bool(r[ix["grasped"]]) if "grasped" in ix else None,
            "gap": r[ix["finger_gap_m"]] if "finger_gap_m" in ix else None,
            "tcp": [float(x) for x in tcp[:3]] if tcp else None,
            "eraser": list(eraser_xyz) if eraser_xyz is not None else None,
            "table": None, "min_z_fr": None, "c2_mode": None, "reflex": None,
        })
    return rows


# ----------------------------------------------------------------------------------------------------- metrics

def _up(r: dict) -> bool:
    """One row of a success hold: lift >= LIFT_M and the eraser clear of the table (clear None = not measurable,
    trace.json-only rows: the lift alone)."""
    return r["lift"] is not None and r["lift"] >= C.LIFT_M and r.get("clear") is not False


def _success(rows: list) -> tuple[bool, float | None, bool | None]:
    """lift >= LIFT_M and the eraser's lowest corner >= table top + CLEAR_M in every row, held >= HOLD_S (grasp not
    required; user decisions 2026-11-22): the first run of such rows that lasts HOLD_S; t_success = the row where the
    hold is complete (the driver's convention and its early stop). Third value = lift_without_grasp: on success, no
    row of that run up to t_success is a grasp row (None when the rows carry no grasp column or there is no success)."""
    start = None
    grasp_in_run = False
    for r in rows:
        if _up(r):
            if start is None:
                start, grasp_in_run = r["t"], False
            grasp_in_run = grasp_in_run or bool(r["grasp"])
            if r["t"] - start >= C.HOLD_S - C.HOLD_TOL_S:
                return True, r["t"], (None if r["grasp"] is None else not grasp_in_run)
        else:
            start = None
    return False, None, None


def _dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(math.dist(a, b))


def _approach(rows: list) -> dict:
    """open gap = the first row's; close = first row with gap <= open - GAP_CLOSE_DROP_M. approach_error = min
    TCP-eraser-centre distance before that row (all rows if it never closes); close_error = the distance at it."""
    out = {"gap_open_m": None, "t_close": None, "approach_error_m": None, "t_approach": None, "close_error_m": None,
           "i_close": None, "i_approach": None}
    gaps = [r["gap"] for r in rows]
    g_open = next((g for g in gaps if g is not None), None)
    if g_open is None:
        return out
    g_open = float(g_open)
    out["gap_open_m"] = g_open
    i_close = next((i for i, g in enumerate(gaps) if g is not None and g <= g_open - C.GAP_CLOSE_DROP_M), None)
    d = [_dist(r["tcp"], r["eraser"]) for r in rows]
    if i_close is not None:
        out["t_close"], out["i_close"], out["close_error_m"] = rows[i_close]["t"], i_close, d[i_close]
    window = [(x, i) for i, x in enumerate(d[: (i_close if i_close is not None else len(d))]) if x is not None]
    if window:
        x, i = min(window)
        out["approach_error_m"], out["t_approach"], out["i_approach"] = x, rows[i]["t"], i
    return out


def _first_second(rows: list) -> dict:
    """TCP displacement in the first FIRST_SECOND_S projected on the unit vector TCP(0) -> eraser centre(0), and the
    drop in that distance."""
    out = {"first_second_progress_m": None, "first_second_dist_drop_m": None, "first_second_t": None}
    if not rows or rows[0]["tcp"] is None or rows[0]["eraser"] is None:
        return out
    t0 = rows[0]["t"]
    cand = [r for r in rows if r["tcp"] is not None and r["t"] - t0 <= C.FIRST_SECOND_S + 1e-6]
    if len(cand) < 2:
        return out
    r1 = cand[-1]
    p0, p1, e0 = (np.asarray(x, float) for x in (rows[0]["tcp"], r1["tcp"], rows[0]["eraser"]))
    v = e0 - p0
    n = float(np.linalg.norm(v))
    if n == 0:
        return out
    out["first_second_progress_m"] = float((p1 - p0) @ (v / n))
    out["first_second_dist_drop_m"] = n - float(np.linalg.norm(e0 - p1))
    out["first_second_t"] = r1["t"] - t0
    return out


def _classify(text: str) -> str:
    for name, rx in C.GATE_RULES:
        if rx.search(text or ""):
            return name
    return "other"


def _q(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else None


def _timing(steps: list) -> dict:
    sent = [s for s in steps if s.get("sent")]
    pol = [s.get("policy_ms") for s in sent]
    cyc = [s.get("cycle_ms") for s in sent]
    reasons: dict = {}
    for s in steps:
        if not s.get("sent"):
            k = s.get("skip_reason") or "?"
            reasons[k] = reasons.get(k, 0) + 1
    margins = [s.get("margin_ms") for s in sent if s.get("margin_ms") is not None]
    return {"chunks": len(steps), "sent": len(sent), "skipped": len(steps) - len(sent), "skip_reasons": reasons,
            "policy_ms_p50": _q(pol, 50), "policy_ms_p90": _q(pol, 90), "policy_ms_p95": _q(pol, 95),
            "cycle_ms_p50": _q(cyc, 50), "cycle_ms_p90": _q(cyc, 90), "cycle_ms_p95": _q(cyc, 95),
            "image_age_ms_p50": _q([s.get("image_age_ms") for s in sent], 50),
            "margin_ms_min": min(margins) if margins else None, "margin_ms_p50": _q(margins, 50)}


_LOG_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) ")


def _clamps_from_log(eval_dir: Path, rec: dict, se: dict | None) -> int | None:
    """manager.log lines "clamped into the workspace box" between "episode N started" and "episode N: ..." (N = the
    manager's episode_id of the rollout); else inside session_rollout's wall_start..wall_end. None without a manager.log."""
    sd = rec.get("session_dir")
    eid = rec.get("pd_rollout_id") if rec.get("pd_rollout_id") is not None else (se or {}).get("episode_id")
    if not sd or eid is None:
        return None
    sd = Path(sd)
    if not sd.is_absolute():
        sd = eval_dir / sd
    if not sd.is_dir():                              # moved folder: find the session under this eval
        cand = sorted((eval_dir / "session").glob("*")) if (eval_dir / "session").is_dir() else []
        if len(cand) != 1:
            return None
        sd = cand[0]
    logs = list(sd.rglob("manager.log"))
    if not logs:
        return None
    started = re.compile(rf"episode {eid} started")
    ended = re.compile(rf"episode {eid}: ")
    n, inside, seen = 0, False, False
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            if started.search(line):
                inside, seen = True, True
            elif inside and ended.search(line):
                inside = False
            elif inside and C.CLAMP_LINE.search(line):
                n += 1
    if seen:
        return n
    if se and se.get("wall_start") and se.get("wall_end"):
        try:
            w0, w1 = (datetime.fromisoformat(se[k]) for k in ("wall_start", "wall_end"))
        except ValueError:
            return None
        n = 0
        for log in logs:
            for line in log.read_text(errors="replace").splitlines():
                m = _LOG_TS.match(line)
                if m and C.CLAMP_LINE.search(line):
                    t = datetime.fromisoformat(f"{m.group(1)}.{m.group(2)}")
                    n += w0 <= t <= w1
        return n
    return None


def _gpu_fault(eval_dir: Path) -> dict | None:
    """<eval>/gpu_fault.json (written by sim_deploy/eval/batch.py): {"xid_time": local ISO time, "cleared": local ISO
    time or absent, "line": ..., "earlier": [older closed windows, same keys]}. Adds "_windows": [(xid, cleared|None)]:
    the fault windows [xid_time, cleared), open-ended while cleared is not written yet."""
    d = _json(Path(eval_dir) / C.GPU_FAULT_FILE)
    if not isinstance(d, dict) or not d.get("xid_time"):
        return None
    wins = []
    for w in [d, *(d.get("earlier") or [])]:
        try:
            x = datetime.fromisoformat(str(w["xid_time"]))
            c = datetime.fromisoformat(str(w["cleared"])) if w.get("cleared") else None
        except (KeyError, TypeError, ValueError):
            continue
        wins.append((x, c))
    return {**d, "_windows": wins} if wins else None


def _gpu_fault_overlap(src: dict, gpu: dict) -> str | None:
    """The validity reason when the rollout's [wall_start, wall_end] overlaps a fault window [xid_time, cleared), else
    None. No wall_end (crashed rollout): overlaps when it started before the window's end. No wall_start: None."""
    try:
        ws = datetime.fromisoformat(str(src["wall_start"]))
        we = datetime.fromisoformat(str(src["wall_end"])) if src.get("wall_end") else None
    except (KeyError, ValueError):
        return None
    for x, c in gpu["_windows"]:
        if (c is None or ws < c) and (we is None or we >= x):
            return (f"gpu_fault (ran during the GPU fault window: kernel Xid {x.isoformat(timespec='seconds')}"
                    f" .. {'not cleared' if c is None else 'cleared ' + c.isoformat(timespec='seconds')})")
    return None


def _start_info(src: dict) -> tuple[str | None, bool | None, str | None]:
    """(placement source, held_out, start id) of a rollout: random / replay / demo / real_trial."""
    pl = src.get("placement") if isinstance(src.get("placement"), dict) else {}
    source = pl.get("source")
    held = src.get("held_out", pl.get("held_out"))
    st = pl.get("start") or pl.get("demo") or {}
    sid = st.get("id") if isinstance(st, dict) else None
    return source, held, sid


def start_group(source: str | None, held_out: bool | None) -> str | None:
    """The report's start split: demo_trained / demo_held_out / real_trial (None for random / replay)."""
    if source == "demo":
        return "demo_held_out" if held_out else "demo_trained"
    if source == "real_trial":
        return "real_trial"
    return None


def score_rollout(eval_dir: Path, rec: dict, geometry: tuple | None = None, gpu: dict | None = None) -> dict:
    eval_dir = Path(eval_dir)
    n = int(rec.get("n", 0))
    rollout_dir = eval_dir / "rollouts" / f"{n:03d}"
    ej = _json(rollout_dir / "rollout.json") or {}
    src = {**rec, **ej}                                # rollout.json wins over the driver's rollouts.jsonl line
    se = _json(rollout_dir / "session_rollout.json") or src.get("session_rollout") or {}
    steps = _jsonl(rollout_dir / "steps.jsonl")
    missing: list = []
    settled = src.get("eraser_settled_pose")
    gt = _jsonl(rollout_dir / "gt.jsonl")
    if gt:
        first = next((g["eraser_pose"] for g in gt if g.get("eraser_pose")), None)
        z0 = float(settled[2]) if settled else (float(first[2]) if first else 0.0)
        if not settled and not first:
            missing.append("eraser pose")
        box, top = geometry or (None, None)
        rows, source = _rows_from_gt(gt, z0, box, top), "gt.jsonl"
    else:
        trace = _json(rollout_dir / "trace.json") or {}
        rows = _rows_from_trace(trace, settled[:3] if settled else None)
        source = "trace.json" if rows else None
        missing += ["min_z.fingers_rig_m"]       # table contacts / reflexes: the driver's poll counts instead
        if not settled:
            missing += ["approach_error_m", "close_error_m", "first_second_progress_m"]
        if rows and rows[0]["tcp"] is None:
            missing += ["min_z.tcp_m", "approach_error_m", "close_error_m", "first_second_progress_m"]

    origin = src.get("source") or "sim"
    real = origin == "real"
    if real:                                           # imported arm trial: no ground truth, labels decide
        source = "rollout.json (real)"
        missing = ["grasp", "approach_error_m", "close_error_m", "first_second_progress_m", "min_z.fingers_rig_m",
                   "table_contacts", "max_lift_m"]
    out: dict = {"n": n, "seed_key": src.get("seed_key"), "origin": origin, "source": source,
                 "outcome_driver": rec.get("outcome"), "outcome": src.get("outcome")}
    st_source, held, sid = _start_info(src)
    out.update(start_source=st_source, held_out=held, start_id=sid, start_group=start_group(st_source, held))
    if real:
        stop = src.get("stop_reason")
        out["outcome_draft"], out["stop_reason_draft"] = src.get("outcome_draft"), src.get("stop_reason_draft")
    else:
        stop = src.get("stop_reason") or C.OUTCOME_TO_STOP_REASON.get(rec.get("outcome") or "",
                                                                    C.OTHER_OUTCOME_STOP_REASON)
    out["stop_reason"] = stop
    out["rows"] = len(rows)
    out["duration_s"] = src.get("duration_s") if src.get("duration_s") is not None else (rows[-1]["t"] if rows else None)

    # grasp / success / lift
    labelled = True
    if real:
        labelled = src.get("outcome") is not None
        out["grasp"] = src.get("grasped") if labelled else None
        out["t_grasp"] = out["t_success"] = out["max_lift_m"] = out["lift_without_grasp"] = None
        out["success"] = (src.get("outcome") == "success") if labelled else None   # label: lifted clear of the table
        out["lifted"] = src.get("lifted") if labelled else None
    elif rows and rows[0]["grasp"] is not None:
        g = next((r for r in rows if r["grasp"]), None)
        out["grasp"], out["t_grasp"] = g is not None, (g["t"] if g else None)
    else:
        out["grasp"] = out["t_grasp"] = None
        missing.append("grasp")
    if real:
        pass
    elif rows and rows[0]["lift"] is not None:
        out["success"], out["t_success"], out["lift_without_grasp"] = _success(rows)
        if any(r["clear"] is None for r in rows):
            missing.append("clear_of_table")
        lifts = [r["lift"] for r in rows if r["lift"] is not None]
        out["max_lift_m"] = max(lifts) if lifts else None
        out["lifted"] = any(r["grasp"] and r["lift"] is not None and r["lift"] >= C.LIFTED_M for r in rows)
    else:
        out["success"] = out["t_success"] = out["max_lift_m"] = out["lifted"] = out["lift_without_grasp"] = None
        missing.append("success")

    # heights
    tz = [r["tcp"][2] for r in rows if r["tcp"] is not None]
    fz = [r["min_z_fr"] for r in rows if r["min_z_fr"] is not None]
    if real and src.get("min_z_tcp_m") is not None:
        tz = [float(src["min_z_tcp_m"])]
    out["min_z"] = {"tcp_m": min(tz) if tz else None, "fingers_rig_m": min(fz) if fz else None}

    # table contacts
    if real:
        out["table_contacts"] = None
    elif gt:
        tab = [r for r in rows if r["table"]]
        out["table_contacts"] = {"rows": len(tab), "geoms": sorted({x for r in tab for x in r["table"]}),
                                 "source": "gt.jsonl"}
    elif "table_contact_polls" in rec or "table_contact_geoms" in rec or source:
        out["table_contacts"] = {"rows": int(rec.get("table_contact_polls", 0)),
                                 "geoms": list(rec.get("table_contact_geoms", [])), "source": "rollouts.jsonl (driver)"}
    else:
        out["table_contacts"] = None

    ap = _approach(rows)
    out.update({k: ap[k] for k in ("gap_open_m", "t_close", "close_error_m", "approach_error_m", "t_approach")})
    out.update(_first_second(rows))

    # faults by gate rule, gate counters, clamps
    c2 = src.get("c2") or {}
    texts = [t for t in (c2.get("errors") or []) if t]
    if src.get("fault") and not any(t in src["fault"] for t in texts):
        texts.append(src["fault"])
    faults: dict = {}
    for t in texts:
        k = _classify(t)
        faults[k] = faults.get(k, 0) + 1
    if not faults and c2.get("fault_rule"):           # real import: the rule named, no text
        faults[str(c2["fault_rule"])] = 1
    out["faults"] = faults
    out["fault_text"] = texts[0] if texts else None
    gate = c2.get("gate") or c2                        # old driver: c2.gate; sim_deploy.episode/v1 and later: flat c2

    def first(*xs):
        return next((x for x in xs if x is not None), None)
    out["gate"] = {"accepted": gate.get("accepted"), "rejected": gate.get("rejected"),
                   "shortened": first(gate.get("shortened"), se.get("shortened")),
                   "pulled_back": first(gate.get("pulled_back"), se.get("pulled_back")),
                   "width_clips": first(c2.get("width_clips"), se.get("width_clips"))}
    if src.get("clamps") is not None:
        out["clamps"] = int(src["clamps"])
    else:
        out["clamps"] = _clamps_from_log(eval_dir, rec, se)
        if out["clamps"] is None:
            missing.append("clamps")
    out["timing"] = _timing(steps) if steps else (src.get("step_summary") or None)
    sent_steps = [s for s in steps if s.get("sent")]
    out["_ms"] = {"policy": [s["policy_ms"] for s in sent_steps if s.get("policy_ms") is not None],
                  "cycle": [s["cycle_ms"] for s in sent_steps if s.get("cycle_ms") is not None]}
    if out["timing"] is None:
        missing.append("timing")
    reflex_rows = sum(1 for r in rows if r["reflex"]) if gt else int(rec.get("reflex_polls", 0))
    out["reflex_rows"] = reflex_rows

    # validity
    ll = src.get("loop_late")
    if ll is None:
        ll = c2.get("loop_late", se.get("loop_late"))
    dur = out["duration_s"]
    per_s = (ll / dur) if (ll is not None and dur) else None
    reasons = []
    if stop in (C.INVALID_STOP_REASONS_REAL if real else C.INVALID_STOP_REASONS):
        reasons.append(f"stop_reason {stop}")
    if not real and per_s is not None and per_s > C.LOOP_LATE_MAX_PER_S:
        reasons.append(f"loop_late {per_s:.0f}/s > {C.LOOP_LATE_MAX_PER_S:.0f}/s")
    if not real and any(C.SIM_LAG_FAULT.search(t or "") for t in texts):
        reasons.append("sim_lag fault")
    sent = (out["timing"] or {}).get("sent")
    acc = out["gate"].get("accepted")
    if (not real and stop in C.NO_ACCEPTED_STOP_REASONS and sent and acc == 0):
        reasons.append(f"no_accepted_waypoints ({sent} chunks sent, 0 waypoints accepted)")
    if not real and gpu is not None and src.get("wall_start"):
        why = _gpu_fault_overlap(src, gpu)
        if why:
            reasons.append(why)
    if not real and not rows and not reasons:
        reasons.append("no_rows (no ground truth)")
    if not real and out["success"] is None and not reasons:
        reasons.append("no_success_measure")
    out["labelled"] = labelled
    out["validity"] = {"valid": not reasons, "reasons": reasons, "loop_late": ll,
                       "loop_late_per_s": per_s}
    if ll is None:
        missing.append("loop_late")
    out["missing"] = sorted(set(missing))
    out["video"] = (f"rollouts/{n:03d}/external.mp4" if (rollout_dir / "external.mp4").is_file() else None)
    out["tap"] = (f"rollouts/{n:03d}/tap" if (rollout_dir / "tap").is_dir() else None)
    return out


# ----------------------------------------------------------------------------------------------------- summary

def _rate(ros, key) -> dict:
    k = sum(1 for e in ros if e.get(key))
    n = len(ros)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def _stat(vals) -> dict:
    v = [x for x in vals if x is not None]
    return {"n": len(v), "p50": _q(v, 50), "mean": float(np.mean(v)) if v else None,
            "min": min(v) if v else None, "max": max(v) if v else None}


def _within(ros: list, s: float) -> dict:
    k = sum(1 for e in ros if e.get("success") and e.get("t_success") is not None
            and e["t_success"] <= s + C.HOLD_TOL_S)
    n = len(ros)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def pooled_ms(ros: list) -> dict:
    """policy_ms / cycle_ms p50 and p90 over every sent chunk of the given rollouts (steps.jsonl), and how many."""
    out = {}
    for k in ("policy", "cycle"):
        xs = [x for e in ros for x in (e.get("_ms") or {}).get(k, [])]
        out.update({f"{k}_ms_p50": _q(xs, 50), f"{k}_ms_p90": _q(xs, 90), f"{k}_chunks": len(xs)})
    return out


def summarize(ros: list) -> dict:
    rated = [e for e in ros if e["validity"]["valid"] and e.get("labelled", True)]
    valid = [e for e in ros if e["validity"]["valid"]]        # metrics: every valid rollout, labelled or not
    unlabelled = sum(1 for e in ros if e["validity"]["valid"] and not e.get("labelled", True))
    inval: dict = {}
    for e in ros:
        for r in e["validity"]["reasons"]:
            key = r.split(" ")[0] if not r.startswith("stop_reason") else r
            inval[key] = inval.get(key, 0) + 1
    faults: dict = {}
    for e in valid:
        for k, v in e["faults"].items():
            faults[k] = faults.get(k, 0) + v
    stops: dict = {}
    for e in valid:
        stops[str(e["stop_reason"])] = stops.get(str(e["stop_reason"]), 0) + 1
    tim = [e.get("timing") or {} for e in valid]
    return {
        "rollouts_recorded": len(ros), "rollouts_rated": len(rated), "unlabelled": unlabelled,
        "invalid": sum(1 for e in ros if not e["validity"]["valid"]),
        "invalid_reasons": inval,
        "success": _rate(rated, "success"),
        "success_within": {f"{s:g}s": _within(rated, s) for s in C.SUCCESS_WITHIN_S},
        "by_start": {g: _rate([e for e in rated if e.get("start_group") == g], "success")
                     for g in ("demo_trained", "demo_held_out", "real_trial")
                     if any(e.get("start_group") == g for e in ros)},
        "grasp": _rate(rated, "grasp"), "lifted": _rate(rated, "lifted"),
        "lift_without_grasp": {"k": sum(1 for e in rated if e.get("lift_without_grasp")),
                               "rollouts": [e["n"] for e in rated if e.get("lift_without_grasp")]},
        "stop_reasons": stops, "faults_by_rule": faults,
        "t_grasp_s": _stat([e["t_grasp"] for e in valid]), "t_success_s": _stat([e["t_success"] for e in valid]),
        "max_lift_m": _stat([e["max_lift_m"] for e in valid]),
        "min_z_tcp_m": _stat([e["min_z"]["tcp_m"] for e in valid]),
        "min_z_fingers_rig_m": _stat([e["min_z"]["fingers_rig_m"] for e in valid]),
        "approach_error_m": _stat([e["approach_error_m"] for e in valid]),
        "close_error_m": _stat([e["close_error_m"] for e in valid]),
        "first_second_progress_m": _stat([e["first_second_progress_m"] for e in valid]),
        "table_contact_rollouts": sum(1 for e in valid if (e.get("table_contacts") or {}).get("rows")),
        "shortened": sum((e["gate"].get("shortened") or 0) for e in valid),
        "pulled_back": sum((e["gate"].get("pulled_back") or 0) for e in valid),
        "width_clips": sum((e["gate"].get("width_clips") or 0) for e in valid),
        "clamps": sum((e.get("clamps") or 0) for e in valid),
        "policy_ms_p50": _q([t.get("policy_ms_p50") for t in tim], 50),
        "policy_ms_p95_max": max([t["policy_ms_p95"] for t in tim if t.get("policy_ms_p95") is not None],
                                 default=None),
        "margin_ms_min": min([t["margin_ms_min"] for t in tim if t.get("margin_ms_min") is not None], default=None),
        "timing_pooled": pooled_ms(valid),
        "missing": sorted({m for e in ros for m in e["missing"]}),
    }


def eval_meta(eval_dir: Path, doc: dict) -> dict:
    pol = doc.get("policy") or {}
    return {"eval_dir": str(Path(eval_dir).resolve()), "tag": doc.get("tag"), "created": doc.get("created"),
            "task": pol.get("task"), "run_folder": pol.get("run_folder"), "policy": pol.get("raw"),
            "kind_eval_yaml": pol.get("kind"), "look": (doc.get("look") or {}).get("name"),
            "record_schema": doc.get("record_schema"), "launcher": doc.get("launcher"),
            "batch_id": doc.get("batch_id"), "protocol_sha": doc.get("protocol_sha"),
            "seed": doc.get("seed"), "dry_run": doc.get("dry_run")}


def score_eval(eval_dir: Path, keep_ms: bool = False) -> dict:
    eval_dir = Path(eval_dir)
    doc, recs = load_eval(eval_dir)
    geo = eraser_geometry(doc)
    gpu = _gpu_fault(eval_dir)
    ros = [score_rollout(eval_dir, r, geo, gpu) for r in recs]
    summary = summarize(ros)
    if not keep_ms:                                    # the raw chunk times only feed the pooled timing
        for e in ros:
            e.pop("_ms", None)
    return {"schema": C.SCORE_SCHEMA, "scorer_version": SCORER_VERSION,
            "scored": datetime.now().isoformat(timespec="seconds"), "constants": C.as_dict(),
            "eval": eval_meta(eval_dir, doc), "summary": summary,
            "gpu_fault": None if gpu is None else {k: v for k, v in gpu.items() if k != "_windows"},
            "invalid_rollouts": [e["n"] for e in ros if not e["validity"]["valid"]], "rollouts": ros}


def write_score(eval_dir: Path) -> Path:
    s = score_eval(eval_dir)
    p = Path(eval_dir) / "score.json"
    tmp = p.with_name("score.json.tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(p)
    return p
