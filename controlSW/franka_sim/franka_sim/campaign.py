"""Stage 1b campaign (A8): replay every recorded demo through the whole sim stack and measure whether the simulated arm
tracks it.

Why: Stage 1 (stage1_replay.py) answers "is the demo kinematically feasible with this tooling". Stage 1b answers "does
the controller + hand + scene reproduce it without faults": launcher -> task -> bridge -> MuJoCoBackend + hand_sim +
scene, the render sender on the image side, ``policy: demo_replay`` (tools/replay_server.py serves the demo's own
actions, relative to the current pose).

Per demo d x speed s (one launcher run each):
  1. Overlay (a derived deploy.yaml next to the run + ``--set``; never an edit of a repo YAML): run.policy demo_replay
     (episode d, speed s); the bridge starts at ``q_init`` = the Stage 1 start q of d; ``robot.home_tcp`` = that start
     TCP; the eraser at d's grasp location (sim_reset ``eraser_xy_yaw``, closing axis across the 20 mm side) and the
     hand gap = gap(w0) (``tooling.width_map``); the Stage 1 tooling sets (``station_sets``: finger extension, camera
     side) and the controller TCP = the model TCP (``tooling.F_T_TCP``, see DEVIATIONS) as run overrides.
  2. ``python -m franka_deploy run --no-tty --script "1:sim_reset:{..},3:c,<T>:y,<T+3>:q"`` (T from the episode length
     / s); the campaign also sends ``y`` / ``q`` early (session commands.txt) once the replay pointer reached the last
     demo frame + ``tail_s``. Ports = FD_TEST_PORT_OFFSET (Track A 1000) + 100 x job slot.
  3. Collect: sim_gt (ground_truth.evaluate_episode), TCP path vs the demo after time alignment (``track_metric``),
     clamps, settle timeouts, reflexes, hand statistics, peak F_ext, pad-table contacts (separately from reflexes),
     system faults (launcher / sender / bridge died, timeouts, setup refused) as their own class.
  4. Demos infeasible in stage1.json are run but labelled ``stage1_infeasible`` and excluded from the tracking
     denominator.

Tracking metric (criteria.stage1b): the reference at time t is the demo pose at the frame the task scheduled for t
(tap ``reply_replay_frame`` + k x 3 x speed at ``abs_timestamps[k]`` + lookahead), expressed relative to the measured
start pose (the replay is relative), shifted by the best global lag in [0, LAG_MAX_S] (time alignment). A sample is ok
when |dp| <= track_pos_m and the rotation error <= track_rot_deg. ``track_fraction`` = the minimum over sliding
``WINDOW_S`` windows of the fraction of ok samples (a spike shorter than (1 - track_ok_fraction_min) x window passes);
``track_ok`` = track_fraction >= track_ok_fraction_min.

DEVIATIONS (recorded in campaign.json ``overrides``):
  * The controller's ``tooling.F_T_TCP`` is a placeholder 131 mm / 135 deg away from the model TCP that
    Stage 1 used for the -y design at extension 0.10 (224 mm at extension 0; 47 deg for +y); the replay is relative to
    the controller TCP, so the run sets ``tooling.F_T_TCP`` = ``model.derived_tcp(station)`` (I-TCP).
  * ``grasp_height: scene`` (default) moves the Stage 1 anchor up so the grasp TCP is at
    ``scene.grasp_tcp_height_m(sm, station)`` (the A6 pad-clearance rule, 0.02 m) instead of the station's 0.01 m
    placeholder (pads 5 mm into the table); the start q is re-solved by IK from the Stage 1 start q.
  * ``--jobs N > 1`` starts the launcher through ``python -m franka_sim.campaign --launch-child``: identical to
    ``franka_deploy run`` except that preflight ``shm_clean`` tolerates live robotmq segments of the other parallel
    launchers (it still reports them).

Usage (RE env, MUJOCO_GL=egl):
    python -m franka_sim campaign --stage1 stage1.json[.gz] --config deploy.yaml --profile sim-iphone \\
        --demos all|0,3,7 --speed 0.5[,1.0] --out DIR [--jobs N] [--set k=v ...]
    run_campaign(config, profile, stage1_json, demos, speeds, out_dir, sets=(), jobs=1) -> dict   (I-CAMPAIGN)
Outputs: DIR/campaign.json (schema sim_campaign/v1), DIR/campaign.txt, DIR/rows/<run>.json (resume), DIR/runs/<run>/.
"""

from __future__ import annotations

import argparse
import copy
import glob
import gzip
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCHEMA = "sim_campaign/v1"
SIM_PKG = Path(__file__).resolve().parents[1]                    # controlSW/franka_sim
DEPLOY_PKG = SIM_PKG.parent / "franka_deploy"                    # controlSW/franka_deploy
for _p in (SIM_PKG, DEPLOY_PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WINDOW_S = 5.0                 # sliding tracking window (design_track_a.md A8)
RESAMPLE_HZ = 50.0             # the window fraction is time based: samples on a uniform grid
LAG_MAX_S = 0.30               # time alignment: best global lag in [0, LAG_MAX_S]
LAG_STEP_S = 0.01
STRIDE = 3                     # replay_server: 3 demo frames per action step at 1x
DEMO_HZ = 60.0
TAIL_S = 4.0                   # after the replay pointer reaches the last frame: hold this long, then y, q
START_S = 3.0                  # script: c at 3 s after READY (sim_reset at 1 s)
CALL_WALL_S = 0.65             # wall time per pause-mode policy call (exec_h 8 at 20 Hz + latency + settle), for T
LAUNCH_BUDGET_S = 150.0        # launcher startup + teardown allowance on top of T
PREFLIGHT_ATTEMPTS = 2         # a launcher that fails a startup preflight (e.g. receiver fps on a loaded machine) is retried once
HOME_MAX_DIST_M = 0.10         # robot.home_tcp.max_dist_m: a start outside the 60 deg cone is pulled ~5 cm by the bridge clamp
PORT_SLOT = 100                # port offset step between parallel jobs
VALIDATE_SIM_KNOWN = {"checksums.json incomplete", "no camera video (*.zarr/*.mp4)"}
CAUSES = ("system_fault", "reflex", "pad_table", "safety_clamp", "hand", "overshoot")


# =============================================================================================== pure helpers

def load_stage1(path) -> dict:
    p = str(path)
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        d = json.load(f)
    if d.get("schema") != "stage1/v1":
        raise ValueError(f"{p}: schema {d.get('schema')!r}, expected stage1/v1")
    return d


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def parse_demos(spec, avail) -> list[int]:
    """``avail`` = the zarr demo indices stage1.json covers (or a count n = 0..n-1)."""
    avail = list(range(avail)) if isinstance(avail, int) else sorted(avail)
    if spec in (None, "", "all"):
        return avail
    out = sorted({int(x) for x in (spec if isinstance(spec, (list, tuple)) else str(spec).split(",")) if str(x).strip() != ""})
    bad = [i for i in out if i not in set(avail)]
    if bad:
        raise ValueError(f"--demos {bad}: stage1.json covers demos {avail}")
    return out


def parse_speeds(spec) -> list[float]:
    return [float(x) for x in (spec.split(",") if isinstance(spec, str) else spec)]


def run_id(demo: int, speed: float) -> str:
    return f"d{demo:02d}_s{speed:g}"


def eraser_yaw(T_tcp) -> float:
    """Eraser yaw for a grasp at TCP pose T: the closing axis (TCP x) projected on the table is the eraser y axis
    (the fingers close across the 20 mm side), i.e. eraser y = (-sin yaw, cos yaw)."""
    x = np.asarray(T_tcp, float)[:3, 0]
    n = math.hypot(x[0], x[1])
    if n < 1e-6:
        return 0.0
    return math.atan2(-x[0] / n, x[1] / n)


def gap_from_width(w: float, wm: dict) -> float:
    """Hand gap of a training width (train = a * gap + b), clipped to [gap_min_m, gap_max_m]."""
    g = (float(w) - float(wm["b"])) / float(wm["a"])
    return float(np.clip(g, float(wm.get("gap_min_m", 0.0)), float(wm.get("gap_max_m", 0.08))))


def stage1_demo_rows(s1: dict) -> list[dict]:
    """Per Stage 1 demo: name, feasible, reason, start_q, start_tcp_pose7, base_T_world, grasp_index."""
    rec = s1["recommendation"]
    out = []
    for i, r in enumerate(s1["demos"]):
        n = r["name"]
        out.append({"idx": i, "name": n, "feasible": bool(r["feasible"]), "reason": r.get("reason"),
                    "start_q": None if rec["start_q"].get(n) is None else list(rec["start_q"][n]),
                    "start_tcp_pose7": None if rec["start_tcp_pose7"].get(n) is None else list(rec["start_tcp_pose7"][n]),
                    "base_T_world": np.asarray(rec["base_T_world"][n], float), "grasp_index": int(r["grasp_index"]),
                    "n_frames": int(r.get("n_frames", 0))})
    return out


def overlay(row: dict, T_zup: np.ndarray, w: np.ndarray, wm: dict, dz: float = 0.0, ik=None, q_seed=None,
            fk=None) -> dict:
    """The run overlay of one demo: q_init, start TCP (home), eraser [x, y, yaw], hand gap, the base_T_world used.

    ``T_zup`` = the demo's z-up TCP poses, ``w`` its training widths. ``dz`` lifts the Stage 1 anchor; then
    ``ik(T_tcp, q_seed) -> (q, ok)`` re-solves the start q (seeded at the Stage 1 one). dz = 0 -> the Stage 1 q.
    A demo whose start Stage 1 could not reach (start_q None) gets the best-effort IK from ``q_seed`` and its home is
    ``fk(q)`` (the TCP where the arm really starts), so the launcher's home check passes; ik_ok is then False."""
    from franka_sim import frames as fr

    bTw = np.asarray(row["base_T_world"], float).copy()
    bTw[2, 3] += float(dz)
    T0 = bTw @ T_zup[0]
    Tg = bTw @ T_zup[int(row["grasp_index"])]
    ik_ok = True
    if row.get("start_q") is None:
        if ik is None or fk is None or q_seed is None:
            raise ValueError(f"{row['name']}: no stage1 start_q; needs ik, fk and q_seed")
        q, ik_ok = ik(T0, np.asarray(q_seed, float))
        if not ik_ok:
            T0 = np.asarray(fk(q), float)
    elif abs(dz) > 0.0:
        if ik is None:
            raise ValueError("overlay: dz != 0 needs an ik function")
        q, ik_ok = ik(T0, np.asarray(row["start_q"], float))
    else:
        q = np.asarray(row["start_q"], float)
        p0 = np.asarray(row["start_tcp_pose7"], float)
        if np.linalg.norm(p0[:3] - T0[:3, 3]) > 2e-3:
            raise ValueError(f"{row['name']}: stage1 start_tcp_pose7 {p0[:3]} != base_T_world @ demo[0] {T0[:3, 3]}")
    return {"q_init": [round(float(x), 6) for x in q], "ik_ok": bool(ik_ok),
            "start_tcp_pose7": [round(float(x), 6) for x in fr.T_to_pose7(T0)],
            "home_xyz": [round(float(x), 6) for x in T0[:3, 3]],
            "grasp_tcp_xyz": [round(float(x), 6) for x in Tg[:3, 3]],
            "eraser_xy_yaw": [round(float(Tg[0, 3]), 6), round(float(Tg[1, 3]), 6), round(eraser_yaw(Tg), 6)],
            "hand_gap_m": round(gap_from_width(float(w[0]), wm), 6), "w0": float(w[0]), "anchor_dz_m": float(dz),
            "base_T_world": bTw.tolist()}


def frame_schedule(calls: list[dict], speed: float, lookahead_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(t knots, demo frame knots) of the executed actions: call c starts at its replay_frame, action k is frame
    replay_frame + k * 3 * speed, due at abs_timestamps[k] + lookahead (the controller's schedule)."""
    ts, fs = [], []
    step = STRIDE * float(speed)
    for c in sorted(calls, key=lambda c: float(np.asarray(c["timestamps"]).reshape(-1)[0])):
        t = np.asarray(c["timestamps"], float).reshape(-1) + float(lookahead_s)
        f = float(c["replay_frame"]) + step * np.arange(len(t))
        ts.append(t)
        fs.append(f)
    if not ts:
        return np.zeros(0), np.zeros(0)
    t, f = np.concatenate(ts), np.concatenate(fs)
    o = np.argsort(t, kind="stable")
    t, f = t[o], np.maximum.accumulate(f[o])       # the pointer never moves back
    keep = np.r_[True, np.diff(t) > 1e-9]
    return t[keep], f[keep]


def _interp_T(T: np.ndarray, f: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    f = np.clip(np.asarray(f, float), 0, len(T) - 1)
    i = np.floor(f).astype(int)
    j = np.minimum(i + 1, len(T) - 1)
    a = f - i
    out = np.tile(np.eye(4), (len(f), 1, 1))
    out[:, :3, 3] = (1 - a)[:, None] * T[i, :3, 3] + a[:, None] * T[j, :3, 3]
    R = Rotation.from_matrix(T[:, :3, :3])
    sl = Slerp(np.arange(len(T)), R) if len(T) > 1 else None
    out[:, :3, :3] = sl(f).as_matrix() if sl is not None else T[0, :3, :3]
    return out


def pose_errors(T_meas: np.ndarray, T_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position error m, rotation error deg) per sample."""
    dp = np.linalg.norm(T_meas[:, :3, 3] - T_ref[:, :3, 3], axis=1)
    Rr = np.einsum("nji,njk->nik", T_ref[:, :3, :3], T_meas[:, :3, :3])
    c = np.clip((np.trace(Rr, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return dp, np.degrees(np.arccos(c))


def window_fraction(t: np.ndarray, ok: np.ndarray, window_s: float = WINDOW_S) -> tuple[float, float | None]:
    """(minimum over sliding windows of the fraction of ok samples, start time of the worst window). ``t`` uniform.
    A series shorter than the window is one window."""
    ok = np.asarray(ok, float)
    if ok.size == 0:
        return float("nan"), None
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    n = max(1, int(round(window_s / dt)))
    if len(ok) <= n:
        return float(ok.mean()), float(t[0])
    cs = np.r_[0.0, np.cumsum(ok)]
    frac = (cs[n:] - cs[:-n]) / n
    i = int(np.argmin(frac))
    return float(frac[i]), float(t[i])


def track_metric(t: np.ndarray, T_meas: np.ndarray, t_knots: np.ndarray, f_knots: np.ndarray, T_demo: np.ndarray,
                 tol_p: float, tol_r_deg: float, ok_min: float, window_s: float = WINDOW_S,
                 lag_max_s: float = LAG_MAX_S, t_end: float | None = None, T_anchor: np.ndarray | None = None) -> dict:
    """The Stage 1b tracking metric of one run (see the module docstring). ``t`` / ``T_meas`` = the measured TCP
    (controller trace), ``t_knots`` / ``f_knots`` = frame_schedule, ``T_demo`` = demo TCP poses (any world frame).
    ``T_anchor`` = the pose the replay is relative to (the commanded TCP at the first scheduled action: the task's
    ``anchor: target``); None -> the first measured sample on the grid."""
    t = np.asarray(t, float)
    if len(t) < 2 or len(t_knots) == 0:
        return {"track_fraction": None, "track_ok": None, "reason": "no trace or no policy calls"}
    t0 = float(t_knots[0]) - 0.5                     # from just before the first scheduled action
    t1 = float(t_end) if t_end is not None else float(t[-1])
    t1 = min(t1, float(t[-1]))
    if t1 - t0 < 1.0:
        return {"track_fraction": None, "track_ok": None, "reason": "episode shorter than 1 s"}
    tg = np.arange(t0, t1, 1.0 / RESAMPLE_HZ)
    tg = tg[(tg >= t[0])]
    # measured pose on the grid (nearest earlier sample is enough at the trace's ~100-200 Hz)
    idx = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)
    Tm = T_meas[idx]
    D0 = _interp_T(T_demo, np.array([float(f_knots[0])]))[0]
    A = (Tm[0] if T_anchor is None else np.asarray(T_anchor, float)) @ np.linalg.inv(D0)   # the replay is relative
    best = None
    for lag in np.arange(0.0, lag_max_s + 1e-9, LAG_STEP_S):
        f = np.interp(tg - lag, t_knots, f_knots, left=float(f_knots[0]), right=float(f_knots[-1]))
        Tr = A @ _interp_T(T_demo, f)
        dp, dr = pose_errors(Tm, Tr)
        score = float(np.median(dp))
        if best is None or score < best[0] - 1e-12:
            best = (score, float(lag), dp, dr)
    _, lag, dp, dr = best
    okp, okr = dp <= tol_p, dr <= tol_r_deg
    ok = okp & okr
    frac, t_worst = window_fraction(tg, ok, window_s)
    fp, _ = window_fraction(tg, okp, window_s)
    fr_, _ = window_fraction(tg, okr, window_s)
    return {"track_fraction": round(frac, 4), "track_ok": bool(frac >= ok_min), "lag_s": round(lag, 3),
            "pos_fraction": round(fp, 4), "rot_fraction": round(fr_, 4),
            "pos_err_mm": {"p50": round(float(np.median(dp)) * 1e3, 2), "p95": round(float(np.percentile(dp, 95)) * 1e3, 2),
                           "max": round(float(dp.max()) * 1e3, 2)},
            "rot_err_deg": {"p50": round(float(np.median(dr)), 3), "p95": round(float(np.percentile(dr, 95)), 3),
                            "max": round(float(dr.max()), 3)},
            "worst_window_t_s": None if t_worst is None else round(t_worst - t0, 2), "n_samples": int(len(tg)),
            "window_s": window_s, "_t_worst_abs": t_worst}


def classify(row: dict) -> str | None:
    """Cause class of a run that is not clean (system fault, reflex) or misses the tracking metric; None = clean.
    Order: system_fault > reflex > pad_table (pad-table contact in the worst window) > safety_clamp (bridge clamp / box
    / cone in the worst window) > hand (finger contact other than the eraser / no grasp in the worst window) > overshoot
    (controller dynamics)."""
    if row.get("system_fault"):
        return "system_fault"
    if row.get("reflexes"):
        return "reflex"
    tr = row.get("track") or {}
    if tr.get("track_ok") is not False:
        return None
    w = row.get("worst_window") or {}
    if w.get("pad_table"):
        return "pad_table"
    if w.get("safety_clamp"):
        return "safety_clamp"
    if w.get("hand"):
        return "hand"
    return "overshoot"


def fault_reasons(row: dict) -> list[str]:
    """System-fault reasons of a run from its stored fields (plumbing, not tracking). A policy exception caused by a
    reflex or a settle timeout is that outcome, not a fault."""
    out = []
    if row.get("timeout_fault"):
        out.append(row["timeout_fault"])
    lr = row.get("launcher") or {}
    rc = row.get("launcher_rc")
    if rc != 0 or (lr.get("exit_reason") not in (None, "quit")):
        out.append(f"launcher rc {rc}: {lr.get('exit_reason')}")
    if lr.get("sender_died"):
        out.append(f"sender_died {lr['sender_died']}")
    if row.get("refused_start"):
        out.append(f"start refused: {row['refused_start']}")
    sr = row.get("sim_reset") or {}
    if not sr or sr.get("kind") != "sim_reset" or not (sr.get("reply") or {}).get("ok", False):
        out.append(f"sim_reset failed: {json.dumps(sr)[:160]}")
    if not row.get("episode") and not row.get("collect_fault"):
        out.append("no episode dir")
    if row.get("collect_fault"):
        out.append(row["collect_fault"])
    if row.get("collect_error"):
        out.append(row["collect_error"])
    outcome = bool(row.get("reflexes")) or bool(row.get("settle_timeouts"))
    if row.get("policy_exceptions") and not outcome:
        out.append(f"policy_exception {row['policy_exceptions'][0][:120]}")
    if row.get("episode") and not (row.get("replay_frames") or {}).get("completed") and not outcome:
        out.append("replay did not reach the last demo frame")
    return out


def finalize(row: dict) -> dict:
    """system_fault / fault_reason / cause from the stored fields (also re-applied to cached rows)."""
    reasons = fault_reasons(row)
    row["system_fault"] = bool(reasons)
    row["fault_reason"] = "; ".join(reasons) or None
    row["cause"] = classify(row)
    row["final"] = True
    return row


def wilson(k: int, n: int) -> tuple:
    from franka_deploy.eval.metrics import wilson as _w

    p, lo, hi = _w(int(k), int(n))
    return (None if n == 0 else round(p, 4)), round(lo, 4), round(hi, 4)


def aggregate(rows: list[dict], crit: dict) -> dict:
    """Per speed: n, system faults, track_ok over the feasible (non-faulted) runs with Wilson 95 % CI, reflexes,
    pad_table, sim_gt success, cause counts, failing demos."""
    out = {}
    for s in sorted({r["speed"] for r in rows}):
        rs = [r for r in rows if r["speed"] == s]
        feas = [r for r in rs if not r["stage1_infeasible"]]
        scored = [r for r in feas if not r.get("system_fault") and (r.get("track") or {}).get("track_ok") is not None]
        k = sum(bool(r["track"]["track_ok"]) for r in scored)
        p, lo, hi = wilson(k, len(scored))
        gt = [r for r in rs if (r.get("sim_gt") or {}).get("success") is not None]
        kg = sum(bool(r["sim_gt"]["success"]) for r in gt)
        pg, log_, hig = wilson(kg, len(gt))
        causes = {c: sum(r.get("cause") == c for r in rs) for c in CAUSES}
        out[f"{s:g}"] = {
            "speed": s, "n": len(rs), "n_stage1_infeasible": len(rs) - len(feas),
            "stage1_infeasible": [r["demo"] for r in rs if r["stage1_infeasible"]],
            "system_faults": sum(bool(r.get("system_fault")) for r in rs),
            "track_ok": {"k": k, "n": len(scored), "p": p, "ci95": [lo, hi],
                         "min": crit["track_ok_fraction_min"], "holds": (p is not None and p >= crit["track_ok_fraction_min"])},
            "track_fraction_median": (round(float(np.median([r["track"]["track_fraction"] for r in scored])), 4)
                                      if scored else None),
            "runs_with_reflex": sum(bool(r.get("reflexes")) for r in rs),
            "reflexes_total": sum(len(r.get("reflexes") or []) for r in rs),
            "reflex_max": crit.get("reflex_max"),
            "runs_with_pad_table": sum(int(r.get("pad_table") or 0) > 0 for r in rs),
            "pad_table_contacts_total": sum(int(r.get("pad_table") or 0) for r in rs),
            "sim_gt_success": {"k": kg, "n": len(gt), "p": pg, "ci95": [log_, hig]},
            "clamps_total": sum(int(r.get("clamp_count") or 0) for r in rs),
            "settle_timeouts_total": sum(int(r.get("settle_timeouts") or 0) for r in rs),
            "causes": causes,
            "failing": [{"demo": r["demo"], "cause": r["cause"], "track_fraction": (r.get("track") or {}).get("track_fraction"),
                         "detail": r.get("fault_reason") or ", ".join(r.get("reflexes") or [])[:160] or None,
                         "stage1_infeasible": r["stage1_infeasible"]}
                        for r in rs if r.get("cause")],
            "run_wall_s": {"median": round(float(np.median([r["wall_s"] for r in rs])), 1),
                           "max": round(float(max(r["wall_s"] for r in rs)), 1)} if rs else None,
        }
    return out


def campaign_text(doc: dict) -> str:
    L = [f"Stage 1b campaign ({doc['schema']}), profile {doc['profile']}, robot {doc['robot']}, fingerprint {doc['fingerprint']}",
         f"overrides: {' '.join(doc['overrides'])}",
         f"criteria.stage1b: {json.dumps(doc['criteria'])}; v_max {doc['v_max_m_s']} m/s; grasp height {doc['grasp_height']}"
         f" (anchor dz {doc['anchor_dz_m']:+.3f} m); jobs {doc['jobs']}",
         "",
         "speed    n  inf  faults  track_ok        p   ci95            reflex(runs/total)  pad_table(runs/total)  gt_success  clamps  run_s(med)"]
    for key, a in doc["aggregates"].items():
        t, g = a["track_ok"], a["sim_gt_success"]
        ci = f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]"
        L.append(f"{key:>5s} {a['n']:4d} {a['n_stage1_infeasible']:4d} {a['system_faults']:7d}  {t['k']:3d}/{t['n']:<3d}"
                 f"   {('%.3f' % t['p']) if t['p'] is not None else '  -  '} {ci:16s} {a['runs_with_reflex']:5d}/{a['reflexes_total']:<5d}"
                 f"         {a['runs_with_pad_table']:5d}/{a['pad_table_contacts_total']:<5d}          {g['k']:3d}/{g['n']:<3d}"
                 f"  {a['clamps_total']:6d}  {(a['run_wall_s'] or {}).get('median', 0):6.1f}")
    L.append("")
    L.append("run       demo  s1_feas  track   pos_p95mm  rot_p95deg  lag_s  reflex  pad_tbl  gt   grasp_s  F_ext_max  cause")
    for r in doc["rows"]:
        tr = r.get("track") or {}
        gt = r.get("sim_gt") or {}
        L.append(f"{r['run']:9s} {r['demo']:4d}  {'y' if not r['stage1_infeasible'] else 'n':7s}  "
                 f"{('%.3f' % tr['track_fraction']) if tr.get('track_fraction') is not None else '  -  '}  "
                 f"{(tr.get('pos_err_mm') or {}).get('p95', float('nan')):9.1f}  {(tr.get('rot_err_deg') or {}).get('p95', float('nan')):10.2f}  "
                 f"{tr.get('lag_s', float('nan')):5.2f}  {len(r.get('reflexes') or []):6d}  {int(r.get('pad_table') or 0):7d}  "
                 f"{'y' if gt.get('success') else ('n' if gt.get('success') is False else '-'):3s}  "
                 f"{gt.get('grasp_t_s') if gt.get('grasp_t_s') is not None else '-':>7}  {r.get('F_ext_max') or 0:9.1f}  {r.get('cause') or ''}"
                 + (f" ({r['fault_reason'][:60]})" if r.get("fault_reason") else ""))
    return "\n".join(L) + "\n"


# =============================================================================================== episode reading

def _jsonl(p: Path) -> list:
    out = []
    if not p.is_file():
        return out
    with open(p, errors="replace") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def read_calls(ep: Path) -> list[dict]:
    calls = []
    for f in sorted((ep / "tap").glob("call_*.npz")):
        try:
            z = np.load(f, allow_pickle=False)
            if "reply_replay_frame" not in z.files or "abs_timestamps" not in z.files:
                continue
            calls.append({"replay_frame": float(np.asarray(z["reply_replay_frame"]).reshape(-1)[0]),
                          "timestamps": np.asarray(z["abs_timestamps"], float).reshape(-1)})
        except Exception:
            continue
    return calls


def _pose7_to_T_batch(P: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    P = np.asarray(P, float)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, 3] = P[:, :3]
    T[:, :3, :3] = Rotation.from_quat(P[:, [4, 5, 6, 3]]).as_matrix()
    return T


def window_flags(t0: float | None, window_s: float, trace: dict, sim: list, stats: list = (),
                 finger=("pad_", "finger")) -> dict:
    """What happened inside the worst tracking window [t0, t0 + window_s]: pad-table contact, safety clamp (the bridge
    clamped the target by > 5 mm, or its cone / box counters grew), hand (a finger touched something but the eraser)."""
    if t0 is None:
        return {}
    t1 = t0 + window_s
    out = {"pad_table": False, "safety_clamp": False, "hand": False}
    inside = [e.get("safety") or {} for e in stats if t0 - 1.0 <= float(e["t"]) <= t1 + 1.0 and e.get("safety")]
    if len(inside) >= 2 and any(int(inside[-1].get(k, 0)) > int(inside[0].get(k, 0)) for k in ("cone", "box")):
        out["safety_clamp"] = True
    for e in sim:
        if t0 <= float(e["t"]) <= t1:
            for c in e.get("contacts") or []:
                a, b = str(c["a"]), str(c["b"])
                if ("pad_" in a and "table" in b) or ("pad_" in b and "table" in a):
                    out["pad_table"] = True
                elif (any(p in a for p in finger) or any(p in b for p in finger)) and "eraser" not in a + b:
                    out["hand"] = True
    if trace:
        m = (trace["t"] >= t0) & (trace["t"] <= t1)
        for k in ("clamp_pos_flag", "clamp_rot_flag"):
            if k in trace and np.nansum(trace[k][m]) > 0:
                out["safety_clamp"] = True
        if "safe_target" in trace and "commanded" in trace and m.any():
            dev = np.linalg.norm(trace["safe_target"][m, :3] - trace["commanded"][m, :3], axis=1)
            if np.nanmax(dev) > 0.005:
                out["safety_clamp"] = True
    return out


def safety_counts(stats: list) -> dict:
    """Bridge SafetyFilter counters over the episode (stats lines are cumulative): updates clamped by the cone / box /
    step limits, and the cone fraction of all target updates."""
    sf = [e.get("safety") or {} for e in stats if e.get("safety")]
    if len(sf) < 2:
        return {}
    d = {k: int(sf[-1].get(k, 0)) - int(sf[0].get(k, 0)) for k in ("box", "cone", "step_pos", "step_rot", "ok", "nan")}
    tot = sum(d.values())
    d["cone_fraction"] = round(d["cone"] / tot, 3) if tot else None
    return d


def collect(run_dir: Path, plan: dict, T_demo_zup: np.ndarray, crit: dict, lookahead_s: float) -> dict:
    """Everything one run produced -> one campaign row (no system-fault decision on the launcher here, see run_one)."""
    from franka_sim.ground_truth import evaluate_episode

    work = run_dir / "work"
    sess = sorted(work.glob("session_*"))
    out: dict = {"session": str(sess[-1]) if sess else None}
    if not sess:
        out["collect_fault"] = "no session dir"
        return out
    s = sess[-1]
    summ = json.loads((s / "launcher_summary.json").read_text()) if (s / "launcher_summary.json").is_file() else {}
    out["launcher"] = {"exit_code": summ.get("exit_code"), "exit_reason": summ.get("exit_reason"),
                       "ready_s": summ.get("ready_s"), "total_s": summ.get("total_s"),
                       "sender_died": summ.get("sender_died")}
    resets = [e for e in summ.get("events", []) if e.get("kind") in ("sim_reset", "sim_reset_failed")]
    out["sim_reset"] = resets[-1] if resets else None
    eps = sorted(glob.glob(str(s / "task" / "**" / "episode_*"), recursive=True))
    eps = [Path(e) for e in eps if Path(e).is_dir()]
    out["episode"] = str(eps[0]) if eps else None
    if not eps:
        out["collect_fault"] = "no episode dir"
        tl = s / "logs" / "task.log"
        if tl.is_file():
            ref = [ln for ln in tl.read_text(errors="replace").splitlines() if "refused_start" in ln]
            if ref:
                out["refused_start"] = ref[-1].split("refused_start", 1)[1].strip()[:200]
        return out
    ep = eps[0]
    lab = json.loads((ep / "label.json").read_text()) if (ep / "label.json").is_file() else {}
    es = json.loads((ep / "summary.json").read_text()) if (ep / "summary.json").is_file() else {}
    out["label"] = lab.get("label")
    out["episode_duration_s"] = es.get("duration_s")
    out["policy_calls"] = es.get("policy_calls")
    out["clamp_count"] = es.get("clamp_count")
    out["chunk_scale_max"] = (es.get("chunk_scale") or {}).get("max")
    out["workspace_contacts"] = es.get("workspace_contacts")
    tc = es.get("task_counters") or {}
    out["settle_waits"] = tc.get("settle_waits")
    ev = _jsonl(ep / "events.jsonl")
    out["settle_timeouts"] = sum(1 for e in ev if "not settled" in json.dumps(e) or "SettleTimeout" in json.dumps(e))
    out["policy_exceptions"] = [str(e.get("error"))[:200] for e in ev if e.get("kind") == "policy_exception"]
    out["holds"] = [str(e.get("reason"))[:120] for e in ev if e.get("kind") == "hold"]
    br = _jsonl(ep / "bridge.jsonl")
    sim = [e for e in br if e.get("kind") == "sim"]
    refl = [str(e.get("error")) for e in br if e.get("kind") == "reflex"]
    for e in sim:
        if e.get("reflex") and str(e["reflex"]) not in refl:
            refl.append(str(e["reflex"]))
    out["reflexes"] = refl
    stats = [e for e in br if e.get("kind") == "stats"]
    loops = [e.get("loop") or {} for e in stats]
    rs = [lp.get("rt_resyncs") for lp in loops if lp.get("rt_resyncs") is not None]
    rtf = [lp.get("realtime_factor") for lp in loops if isinstance(lp.get("realtime_factor"), (int, float))]
    out["rt"] = {"rt_resyncs": int(max(rs) - min(rs)) if rs else None,
                 "realtime_factor_min": round(float(min(rtf)), 3) if rtf else None,
                 "loop_p99_ms_max": round(float(max((lp.get("p99_ms") or 0) for lp in loops)), 2) if loops else None}
    out["safety"] = safety_counts(stats)
    out["F_ext_max"] = round(float(max((float(e.get("F_ext_max") or 0.0) for e in sim), default=0.0)), 2)
    # pad-table contacts (reported separately from reflexes; A6 finding 5): onsets + max force
    n_on, fmax, active = 0, 0.0, False
    for e in sim:
        f = sum(float(c.get("force_n", 0.0)) for c in e.get("contacts") or []
                if ("pad_" in str(c["a"]) and "table" in str(c["b"])) or ("pad_" in str(c["b"]) and "table" in str(c["a"])))
        if f > 0 and not active:
            n_on += 1
        active = f > 0
        fmax = max(fmax, f)
    out["pad_table"] = n_on                                   # flat count (eval/stage1b.py reads it as an int)
    out["pad_table_max_force_n"] = round(fmax, 2)
    gaps = [float(e["gap"]) for e in sim if e.get("gap") is not None]
    modes = sorted({str(e.get("hand_mode")) for e in sim})
    out["hand"] = {"gap_min_m": round(min(gaps), 4) if gaps else None, "gap_max_m": round(max(gaps), 4) if gaps else None,
                   "grasped_fraction": round(sum(bool(e.get("grasped")) for e in sim) / len(sim), 3) if sim else None,
                   "modes": modes}
    try:
        gt = evaluate_episode(ep)
    except Exception as e:      # noqa: BLE001
        gt = {"error": repr(e)}
    out["sim_gt"] = None if gt is None else {k: gt.get(k) for k in (
        "success", "reason", "t_success_s", "grasp_t_s", "max_lift_m", "slip_m", "in_hand_slip_m", "knocked_over",
        "reflex", "n_arm_table_contacts", "n_samples", "rate_hz", "error")}
    trp = ep / "controller_trace.npz"
    trace = {}
    if trp.is_file():
        z = np.load(trp)
        trace = {k: z[k] for k in z.files if z[k].dtype.kind == "f"}
    calls = read_calls(ep)
    tk, fk = frame_schedule(calls, plan["speed"], lookahead_s)
    t_end = None
    if len(tk) and len(fk) and fk[-1] >= len(T_demo_zup) - 1 - 1e-6:
        t_end = float(tk[np.argmax(fk >= len(T_demo_zup) - 1 - 1e-6)]) + 1.0   # the replay end + 1 s hold
    if trace and "fk" in trace:
        anchor = None
        if "commanded" in trace and len(tk):
            i0 = int(np.clip(np.searchsorted(trace["t"], tk[0]), 0, len(trace["t"]) - 1))
            anchor = _pose7_to_T_batch(trace["commanded"][i0:i0 + 1])[0]
        tr = track_metric(trace["t"], _pose7_to_T_batch(trace["fk"]), tk, fk, T_demo_zup, float(crit["track_pos_m"]),
                          float(crit["track_rot_deg"]), float(crit["track_ok_fraction_min"]), t_end=t_end, T_anchor=anchor)
        if "commanded" in trace and "safe_target" in trace:     # where the error comes from: bridge clamp vs controller
            m = trace["t"] >= (tk[0] if len(tk) else trace["t"][0])
            sc = np.linalg.norm(trace["safe_target"][m, :3] - trace["commanded"][m, :3], axis=1)
            fs = np.linalg.norm(trace["fk"][m, :3] - trace["safe_target"][m, :3], axis=1)
            out["err_split_mm"] = {"safe_minus_cmd_p95": round(float(np.nanpercentile(sc, 95)) * 1e3, 2),
                                   "fk_minus_safe_p50": round(float(np.nanpercentile(fs, 50)) * 1e3, 2),
                                   "fk_minus_safe_p95": round(float(np.nanpercentile(fs, 95)) * 1e3, 2)}
        if len(trace["t"]) and plan.get("start_tcp_pose7"):
            out["start_err_mm"] = round(float(np.linalg.norm(trace["fk"][0, :3] - np.asarray(plan["start_tcp_pose7"][:3]))) * 1e3, 2)
    else:
        tr = {"track_fraction": None, "track_ok": None, "reason": "no controller_trace.npz"}
    out["replay_frames"] = {"last": float(fk[-1]) if len(fk) else None, "n_demo": int(len(T_demo_zup)),
                            "completed": bool(t_end is not None)}
    t_worst = tr.pop("_t_worst_abs", None)
    out["track"] = tr
    out["track_ok"], out["track_fraction"] = tr.get("track_ok"), tr.get("track_fraction")   # flat (I-CAMPAIGN rows)
    out["worst_window"] = window_flags(t_worst, WINDOW_S, trace, sim, stats) if tr.get("track_ok") is False else {}
    return out


# =============================================================================================== process side

WIRING_SET_PREFIXES = ("tooling.hand.endpoint=", "sensor.transport.")


def apply_port_offset(prof: dict, o: int, protocol: str = "v1") -> None:
    """Shift every port of a sim profile by o (receiver main/uw, camera, logger, bridge, controller, inference,
    spacemouse, sim-state PUB, the bridge-served hand endpoint). Same layout as Track B's stage2b/stage3 helper."""
    prof["station_set"] = list(prof.get("station_set") or []) + [f"sensor.transport.ports.main={5555 + o}"] + (
        [f"sensor.transport.ports.ultrawide={5556 + o}"] if protocol == "v1" else [])
    prof["station_set"] = [x.replace("tcp://127.0.0.1:5565", f"tcp://127.0.0.1:{5565 + o}")
                           if x.startswith("tooling.hand.endpoint=") else x for x in prof["station_set"]]
    prof.setdefault("camera", {}).update(endpoint=f"tcp://localhost:{16702 + o}", logger_endpoint=f"tcp://localhost:{25556 + o}")
    prof.setdefault("bridge", {})["endpoint"] = f"tcp://127.0.0.1:{5565 + o}"
    prof.setdefault("controller", {})["endpoint"] = f"tcp://localhost:{26010 + o}"
    prof.setdefault("inference", {})["endpoint"] = f"tcp://0.0.0.0:{18765 + o}"
    prof.setdefault("spacemouse", {})["endpoint"] = f"tcp://localhost:{15557 + o}"
    prof["sim_state_endpoint"] = f"tcp://127.0.0.1:{5570 + o}"


def child_env() -> dict:
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e["PYTHONPATH"] = os.pathsep.join([str(DEPLOY_PKG), str(SIM_PKG)])
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.setdefault("MUJOCO_GL", "egl")
    return e


def fmt_list(v) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def process_tree(root: int) -> list[int]:
    """root pid + all its descendants (by /proc ppid), including children that left the process group."""
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo += kids.get(pid, [])
    return out


def kill_cleanup_shm(pids) -> list[str]:
    """Remove the robotmq segments (rmq_<user>_<pid>_*) of the killed tree's pids that are dead now; live pids
    (and every other process's segments) are left alone."""
    import glob
    import re

    removed: list[str] = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            continue                                  # still alive (or a recycled pid): not ours to clean
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        for f in glob.glob(f"/dev/shm/rmq_*_{pid}_*"):      # then keep only rmq_<user>_<pid>_ (prefix-anchored)
            m = re.match(r"rmq_[^_]+_(\d+)_", os.path.basename(f))
            if m and int(m.group(1)) == pid:
                try:
                    os.remove(f)
                    removed.append(os.path.basename(f))
                except OSError:
                    pass
    return removed


class Campaign:
    """Plans, runs and collects the campaign (``run_campaign`` is the functional entry point)."""

    def __init__(self, config, profile: str, stage1_json, demos, speeds, out_dir, sets=(), jobs: int = 1,
                 grasp_height: str = "scene", zarr=None, python=None, log=None, timeout_scale: float = 1.0,
                 tail_s: float = TAIL_S, validate_n: int = 3):
        from franka_deploy.launch.launcher import load_config, task_limits
        from franka_deploy.station import station_from_deploy_cfg
        from franka_sim import frames as fr
        from franka_sim import scene as S
        from franka_sim.kinematics import Kin
        from franka_sim.model import build_model, derived_tcp
        from franka_sim.stage1_replay import load_demos

        self.log = log or (lambda msg: print(msg, flush=True))
        self.config, self.profile = str(Path(config).resolve()), str(profile)
        self.stage1_path = str(Path(stage1_json).resolve())
        self.s1 = load_stage1(stage1_json)
        self.s1_rows = stage1_demo_rows(self.s1)     # keyed by zarr demo index once the zarr is loaded (below)
        self.speeds = parse_speeds(speeds)
        self.out = Path(out_dir).resolve()
        self.jobs = max(1, int(jobs))
        self.user_sets = [str(x) for x in sets]
        self.timeout_scale, self.tail_s, self.validate_n = float(timeout_scale), float(tail_s), int(validate_n)
        self.port_base = int(os.environ.get("FD_TEST_PORT_OFFSET", 1000))
        # the campaign station = the profile + the Stage 1 tooling sets + the user's sets, policy demo_replay
        # design sets only: a Stage 1 run made by `eval` records the profile's wiring sets too (hand endpoint, transport);
        # re-applied as --set they would undo the per-job port shift of derive() (every job on bridge port 5565)
        self.tool_sets = [str(x) for x in self.s1.get("station_sets", []) if not str(x).startswith(WIRING_SET_PREFIXES)]
        base_sets = self.tool_sets + self.user_sets
        cfg = load_config(self.config, self.profile, station_set=base_sets)
        cfg["run"] = {**cfg.get("run", {}), "policy": "demo_replay"}
        st = station_from_deploy_cfg(cfg)
        if st is None:
            raise ValueError(f"profile {profile!r} has no station")
        if st.ids.get("robot") != self.s1.get("robot"):
            raise ValueError(f"stage1.json robot {self.s1.get('robot')!r} != profile robot {st.ids.get('robot')!r} "
                             f"(pass --set robot=... is not supported; use a matching stage1.json)")
        self.cfg, self.station = cfg, st
        self.python = python or cfg["python"]["realenv"]
        self.python_bp = cfg["python"].get("bp", self.python)     # eval reads the demo zarr (iphumi codecs: BP env)
        self.crit = dict(st.get("criteria.stage1b"))
        self.v_max = task_limits(cfg, st)[0]
        self.lookahead_s = float(st.task_args().get("lookahead_s", 0.1))
        self.sm = build_model(st)
        self.F_T_TCP = np.asarray(self.sm.F_T_TCP, float)       # == derived_tcp(st) unless F_T_C_main is measured
        if self.sm.info.get("F_T_TCP_source") != "F_T_C_main":
            assert np.allclose(self.F_T_TCP, derived_tcp(st), atol=1e-9)
        self.kin_tcp = Kin(self.sm, "tcp")
        top = float(st.get("scene.table.top_z_base_m"))
        h_s1 = float(self.s1.get("grasp_tcp_height_m", st.get("scene.grasp_tcp_height_m")))
        self.grasp_height = grasp_height
        h = S.grasp_tcp_height_m(self.sm, st) if grasp_height == "scene" else h_s1
        self.anchor_dz = float(h - h_s1)
        self.grasp_z = top + h
        self.wm = dict(st.get("tooling.width_map"))
        self.zarr = str(zarr or cfg.get("prompt_zarr"))
        demos_raw = load_demos(self.zarr)
        # stage1.json may cover a subset (eval --episodes N / --demos): map its demos to zarr indices by name
        zidx = {d["name"]: i for i, d in enumerate(demos_raw)}
        missing = [r["name"] for r in self.s1_rows if r["name"] not in zidx]
        if missing:
            raise ValueError(f"{self.zarr}: {len(missing)} stage1.json demo(s) not in the zarr, e.g. {missing[0]}")
        self.s1_rows = {zidx[r["name"]]: {**r, "idx": zidx[r["name"]]} for r in self.s1_rows}
        self.demos = parse_demos(demos, list(self.s1_rows))
        self.T_zup, self.width = {}, {}
        for r in self.s1_rows.values():
            d = demos_raw[r["idx"]]
            self.T_zup[r["idx"]] = fr.arkit_to_zup(d["T"])
            self.width[r["idx"]] = np.asarray(d["w"], float)
        self.F_T_TCP_set = "tooling.F_T_TCP=" + json.dumps([[round(float(x), 9) for x in row] for row in self.F_T_TCP])
        self.overrides = self.tool_sets + [self.F_T_TCP_set] + self.user_sets
        fp = {"station": st.fingerprint(), "stage1_sha": file_sha(self.stage1_path), "overrides": self.overrides,
              "grasp_height": grasp_height, "anchor_dz_m": round(self.anchor_dz, 6), "criteria": self.crit,
              "profile": self.profile, "zarr": Path(self.zarr).name, "tail_s": self.tail_s}
        self.fingerprint = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]
        self.fp_detail = fp

    # ------------------------------------------------------------------------------------------- planning
    def _ik(self, T, q0):
        q, info = self.kin_tcp.ik(np.asarray(T, float), np.asarray(q0, float), q_rest=np.asarray(q0, float))
        return q, bool(info["ok"])

    def plan(self, demo: int, speed: float) -> dict:
        r = self.s1_rows[demo]
        ov = overlay(r, self.T_zup[demo], self.width[demo], self.wm, self.anchor_dz, self._ik,
                     q_seed=self.s1["recommendation"]["home"]["q"], fk=self.kin_tcp.fk)
        n = len(self.T_zup[demo])
        adv = 7 * STRIDE * float(speed)                             # pause mode: (exec_h - 1) action steps per call
        n_calls = int(math.ceil((n - 1) / adv)) + 1
        T = START_S + self.timeout_scale * (2.0 * n_calls * CALL_WALL_S + 10.0)
        return {"run": run_id(demo, speed), "demo": demo, "name": r["name"], "speed": float(speed),
                "stage1_feasible": r["feasible"], "stage1_reason": r["reason"], "n_frames": n,
                "n_calls_est": n_calls, "T_y_s": round(T, 1), **ov}

    def script(self, p: dict) -> str:
        rst = json.dumps({"eraser_xy_yaw": p["eraser_xy_yaw"], "hand_gap_m": p["hand_gap_m"]}, separators=(",", ":"))
        return f"1:sim_reset:{rst},{START_S:g}:c,{p['T_y_s']:g}:y,{p['T_y_s'] + 3:g}:q"

    def derive(self, p: dict, run_dir: Path, slot: int) -> Path:
        import yaml

        raw = yaml.safe_load(Path(self.config).read_text())
        prof = copy.deepcopy(raw["profiles"][self.profile])
        prof["station_set"] = list(prof.get("station_set") or [])
        prof["run"] = {**prof.get("run", {}), "policy": "demo_replay"}
        prof["work_root"] = str(run_dir / "work")
        b = prof.setdefault("bridge", {})
        ov = b.setdefault("override", {}).setdefault("backend", {})
        ov.setdefault("type", "mujoco")
        ov.setdefault("mujoco", {})["q_init"] = [float(x) for x in p["q_init"]]
        apply_port_offset(prof, self.port_base + PORT_SLOT * slot,
                          "v1" if str(self.station.get("sensor.transport.protocol")).endswith("v1") else "v2")
        raw["profiles"][self.profile] = prof
        d = run_dir / "deploy_campaign.yaml"
        d.write_text(yaml.safe_dump(raw, sort_keys=False))
        return d

    def run_sets(self, p: dict) -> list[str]:
        return self.overrides + [f"robot.home_tcp.xyz_m={fmt_list(p['home_xyz'])}",
                                 f"robot.home_tcp.max_dist_m={HOME_MAX_DIST_M}", f"policy.episode_index={p['demo']}",
                                 f"policy.speed_scale={p['speed']:g}", "policy.start_frame_index=0"]

    # ------------------------------------------------------------------------------------------- one run
    def _early_stop(self, run_dir: Path, p: dict, proc: subprocess.Popen, stop: threading.Event) -> None:
        """Send y then q (session commands.txt, like `franka_deploy note`) once the replay reached the last demo frame
        and held for tail_s. The script's own y/q at T stay as the upper bound."""
        n_last = p["n_frames"] - 1 - 1e-6
        t_end = None
        sent = False
        while not stop.is_set() and proc.poll() is None:
            time.sleep(0.5)
            sess = sorted((run_dir / "work").glob("session_*"))
            if not sess:
                continue
            taps = sorted(glob.glob(str(sess[-1] / "task" / "**" / "episode_*" / "tap" / "call_*.npz"), recursive=True))
            if not taps:
                continue
            if t_end is None:
                try:
                    z = np.load(taps[-1])
                    if float(np.asarray(z["reply_replay_frame"]).reshape(-1)[0]) >= n_last:
                        t_end = time.monotonic()
                except Exception:
                    continue
            elif not sent and time.monotonic() - t_end >= self.tail_s:
                with open(sess[-1] / "commands.txt", "a") as f:
                    f.write("y\n")
                time.sleep(3.0)
                with open(sess[-1] / "commands.txt", "a") as f:
                    f.write("q\n")
                sent = True

    def run_one(self, demo: int, speed: float, slot: int) -> dict:
        p = self.plan(demo, speed)
        rd = self.out / "runs" / p["run"]
        for attempt in range(1, PREFLIGHT_ATTEMPTS + 1):
            row = self._launch(p, rd, slot)
            row["attempts"] = attempt
            pre = str((row.get("launcher") or {}).get("exit_reason") or "")
            if not (pre.startswith("preflight") and (row.get("launcher") or {}).get("ready_s") is None):
                break
            self.log(f"[campaign] {p['run']} attempt {attempt}: {pre[:120]} -> retry")
            time.sleep(3.0)
        return finalize(row)

    def _launch(self, p: dict, rd: Path, slot: int) -> dict:
        demo, speed = p["demo"], p["speed"]
        if rd.exists():
            import shutil

            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        dep = self.derive(p, rd, slot)
        cmd = [self.python, "-u"] + (["-m", "franka_sim.campaign", "--launch-child"] if self.jobs > 1 else
                                     ["-m", "franka_deploy"]) + \
            ["run", "--config", str(dep), "--profile", self.profile, "--no-tty", "--script", self.script(p)]
        for s in self.run_sets(p):
            cmd += ["--set", s]
        (rd / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
        t0 = time.monotonic()
        timeout = p["T_y_s"] + 3 + LAUNCH_BUDGET_S
        fault = None
        with open(rd / "launcher.log", "w") as lg:
            proc = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=child_env(), cwd=str(rd),
                                    start_new_session=True)
            stop = threading.Event()
            th = threading.Thread(target=self._early_stop, args=(rd, p, proc, stop), daemon=True)
            th.start()
            try:
                rc = proc.wait(timeout)
            except subprocess.TimeoutExpired:
                fault = f"launcher timeout ({timeout:.0f} s)"
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    rc = proc.wait(20)
                except subprocess.TimeoutExpired:
                    tree = process_tree(proc.pid)
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait(10)
                    fault += f"; SIGKILL, removed {len(kill_cleanup_shm(tree))} shm segments"
            stop.set()
        wall = time.monotonic() - t0
        row = {"run": p["run"], "demo": demo, "name": p["name"], "speed": float(speed),
               "stage1_infeasible": not p["stage1_feasible"], "stage1_reason": p["stage1_reason"],
               "plan": {k: p[k] for k in ("q_init", "ik_ok", "start_tcp_pose7", "home_xyz", "grasp_tcp_xyz",
                                          "eraser_xy_yaw", "hand_gap_m", "w0", "anchor_dz_m", "n_frames", "T_y_s")},
               "slot": slot, "launcher_rc": rc, "wall_s": round(wall, 1), "fingerprint": self.fingerprint}
        try:
            row.update(collect(rd, p, self.T_zup[demo], self.crit, self.lookahead_s))
        except Exception as e:      # noqa: BLE001
            row["collect_error"] = repr(e)
        row["timeout_fault"] = fault
        return row

    # ------------------------------------------------------------------------------------------- campaign
    def row_path(self, rid: str) -> Path:
        return self.out / "rows" / f"{rid}.json"

    def cached(self, rid: str) -> dict | None:
        p = self.row_path(rid)
        if not p.is_file():
            return None
        try:
            r = json.loads(p.read_text())
        except ValueError:
            return None
        if not (r.get("final") and r.get("fingerprint") == self.fingerprint):
            return None
        finalize(r)                           # the fault / cause rules of this code version
        return None if r["system_fault"] else r     # system faults are plumbing: a resume retries them

    def pending(self) -> tuple[dict, list]:
        """(cached final rows by run id, [(demo, speed)] still to run). Resume: a row counts only when it is final and
        has this campaign's fingerprint."""
        todo, rows = [], {}
        for s in self.speeds:
            for d in self.demos:
                rid = run_id(d, s)
                c = self.cached(rid)
                if c is not None:
                    rows[rid] = c
                else:
                    todo.append((d, s))
        return rows, todo

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "rows").mkdir(exist_ok=True)
        rows, todo = self.pending()
        self.log(f"[campaign] {len(rows)} cached rows, {len(todo)} to run, jobs {self.jobs}, fingerprint {self.fingerprint}, "
                 f"anchor dz {self.anchor_dz:+.3f} m, out {self.out}")
        t0 = time.monotonic()
        slots = list(range(self.jobs))
        lock = threading.Lock()

        def work(item):
            with lock:
                slot = slots.pop(0)
            try:
                r = self.run_one(item[0], item[1], slot)
            finally:
                with lock:
                    slots.append(slot)
            self.row_path(r["run"]).write_text(json.dumps(r, indent=1, default=str))
            tr = r.get("track") or {}
            self.log(f"[campaign] {r['run']} wall {r['wall_s']:.0f} s track {tr.get('track_fraction')} "
                     f"reflex {len(r.get('reflexes') or [])} pad_table {r.get('pad_table')} "
                     f"gt {(r.get('sim_gt') or {}).get('success')} cause {r.get('cause')} {r.get('fault_reason') or ''}")
            return r

        if todo:
            with ThreadPoolExecutor(self.jobs) as ex:
                futs = []
                for i, it in enumerate(todo):
                    futs.append(ex.submit(work, it))
                    if i < self.jobs - 1:
                        time.sleep(5.0)          # stagger the first starts (model builds, pip freeze)
                for f in futs:
                    r = f.result()
                    rows[r["run"]] = r
        ordered = [rows[run_id(d, s)] for s in self.speeds for d in self.demos]
        doc = self.document(ordered, time.monotonic() - t0)
        if self.validate_n > 0:
            doc["checks"] = self.validate_sample(ordered)
        (self.out / "campaign.json").write_text(json.dumps(doc, indent=1, default=str))
        (self.out / "campaign.txt").write_text(campaign_text(doc))
        return doc

    def document(self, rows: list[dict], wall: float) -> dict:
        return {"schema": SCHEMA, "fingerprint": self.fingerprint, "fingerprint_inputs": self.fp_detail,
                "config": self.config, "profile": self.profile, "robot": self.s1.get("robot"),
                "station_ids": dict(self.station.ids), "station_fingerprint": self.station.fingerprint(),
                "stage1": {"path": self.stage1_path, "design": self.s1.get("design"),
                           "station_sets": self.tool_sets, "grasp_tcp_height_m": self.s1.get("grasp_tcp_height_m"),
                           "placement": {k: self.s1["recommendation"].get(k) for k in ("grid_point", "yaw_deg", "grasp_xyz_base")},
                           "n_feasible": (self.s1.get("summary") or {}).get("n_feasible")},
                "overrides": self.overrides, "grasp_height": self.grasp_height, "anchor_dz_m": self.anchor_dz,
                "grasp_tcp_z_base_m": self.grasp_z, "criteria": self.crit, "v_max_m_s": self.v_max,
                "lookahead_s": self.lookahead_s, "window_s": WINDOW_S, "lag_max_s": LAG_MAX_S, "tail_s": self.tail_s,
                "jobs": self.jobs, "port_offset_base": self.port_base, "zarr": self.zarr,
                "demos": self.demos, "speeds": self.speeds, "wall_s_this_invocation": round(wall, 1),
                "aggregates": aggregate(rows, self.crit), "sessions": [r.get("session") for r in rows], "rows": rows}

    def validate_sample(self, rows: list[dict]) -> list[dict]:
        """franka_deploy validate + eval on the first validate_n runs that produced an episode."""
        out = []
        picks = [r for r in rows if r.get("episode") and not r.get("system_fault")][: self.validate_n]
        for r in picks:
            res = {"run": r["run"]}
            v = subprocess.run([self.python, "-m", "franka_deploy", "validate", r["episode"]], capture_output=True,
                               text=True, env=child_env(), timeout=300)
            res["validate_rc"] = v.returncode
            errs = [ln.strip()[2:] for ln in v.stdout.splitlines() if ln.strip().startswith("- ")]
            res["validate_errors"] = errs
            # a demo_replay sim session has no checkpoint sha (replay) and no video (sim profiles: camera.video false,
            # no ffmpeg): validate reports exactly these two by design; anything else is a real defect
            res["validate_ok_except_sim_known"] = v.returncode == 0 or (bool(errs) and set(errs) <= VALIDATE_SIM_KNOWN)
            res["validate_tail"] = (v.stdout + v.stderr).strip().splitlines()[-3:]
            eo = self.out / "eval" / r["run"]
            e = subprocess.run([self.python_bp, "-m", "franka_deploy", "eval", r["session"], "--out", str(eo),
                                "--zarr", self.zarr, "--normalizer", str(eo / "none.pkl")],
                               capture_output=True, text=True, env=child_env(), timeout=600)
            res["eval_rc"] = e.returncode
            res["eval_tail"] = (e.stdout + e.stderr).strip().splitlines()[-3:]
            out.append(res)
        return out


def run_campaign(config, profile, stage1_json, demos, speeds, out_dir, sets=(), jobs=1) -> dict:
    """I-CAMPAIGN: run (or resume) the Stage 1b campaign and return the ``campaign.json`` content (also written to
    ``out_dir/campaign.json`` + ``campaign.txt``). demos: "all" | "0,3,7" | [ints]; speeds: "0.5,1.0" | [floats];
    sets: extra station overrides ("group.path=value") applied to every run."""
    if not isinstance(demos, str) and demos is not None:
        demos = ",".join(str(int(d)) for d in demos)
    return Campaign(config, profile, stage1_json, demos, speeds, out_dir, sets=sets, jobs=jobs).run()


# =============================================================================================== CLI

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage1", required=True, help="stage1.json (or .json.gz) of the design to replay")
    p.add_argument("--config", default=str(DEPLOY_PKG / "configs" / "deploy.yaml"))
    p.add_argument("--profile", default="sim-iphone")
    p.add_argument("--demos", default="all")
    p.add_argument("--speed", default="0.5", help="comma list, e.g. 0.5,1.0")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--set", action="append", default=[], metavar="GROUP.PATH=VALUE")
    p.add_argument("--grasp-height", choices=("scene", "stage1"), default="scene",
                   help="scene: anchor the grasp at scene.grasp_tcp_height_m(sm, station) (A6 pad clearance); "
                        "stage1: the Stage 1 anchor as is")
    p.add_argument("--zarr", default=None, help="demo zarr (default: the profile's prompt_zarr)")
    p.add_argument("--validate", type=int, default=3, help="franka_deploy validate + eval on this many sessions")
    p.add_argument("--timeout-scale", type=float, default=1.0)


def run(args: argparse.Namespace) -> int:
    from franka_deploy.paths import check_writable_dir

    check_writable_dir(args.out, "campaign output dir")
    c = Campaign(args.config, args.profile, args.stage1, args.demos, args.speed, args.out, sets=args.set,
                 jobs=args.jobs, grasp_height=args.grasp_height, zarr=args.zarr, validate_n=args.validate,
                 timeout_scale=args.timeout_scale)
    doc = c.run()
    print(campaign_text(doc))
    print(f"wrote {c.out / 'campaign.json'}")
    return 0


def _launch_child(argv: list[str]) -> int:
    """``franka_deploy run`` with preflight shm_clean tolerant of live robotmq segments of other launchers (--jobs > 1:
    the parallel campaign runs hold their own camera segments; each launcher still removes its own at teardown)."""
    from franka_deploy.launch import preflight as pf

    orig = pf.shm_clean

    def shm_clean(pattern: str = "/dev/shm/rmq_*"):
        ok, detail = orig(pattern)
        return True, detail if ok else f"tolerated (parallel campaign): {detail}"

    pf.shm_clean = shm_clean
    from franka_deploy.__main__ import main

    return int(main(argv) or 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--launch-child"]:
        sys.exit(_launch_child(sys.argv[2:]))
    ap = argparse.ArgumentParser(prog="franka_sim.campaign")
    add_arguments(ap)
    sys.exit(run(ap.parse_args()))
