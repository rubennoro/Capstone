"""P4 (2026-11-20): the demos' TCP paths carried to a start pose and run through a Python port of franka_control's
safety gate (``deployment/control_deploy/franka_control/src/safety_gate.cpp``), offline, no rendering.

    python -m sim_deploy.scene.gate_check --q "q1,...,q7" [--dataset ZIP ...] [--json OUT]

**Carry.** Each demo's recorded TCP poses P_j (60 Hz) relative to its first pose, applied to the start pose:
W(t) = O_T_start @ inv(P_0) @ P(t) (what the policy's relative actions and the replay do when tracking is perfect).

**Requests, as the manager sends them.** One C3 request every ``CYCLE_S`` (0.1 s: the policy's 10 Hz), each with
``EXEC_H`` (12) waypoints at t_c + k * 50 ms, k = 1..12 (``STEP_S``), poses W(t_c + k * 50 ms). Each request is checked
on its own from start = W(t_c) (the anchor = the previous target, assuming it was reached), now = t_c.

**The gate (per waypoint, in the C++ order):** (2) lead <= ``max_lead_s``; (4) box: outside by <= box_margin -> pulled
back onto the box, beyond -> the request is REFUSED (a fault on the real PC); (5) cone: tilt of the flange z axis
(= TCP z: F_T_TCP only turns about z) from ``cone_axis``: > cone_max -> pulled back, > cone_max + cone_margin -> REFUSED;
(6) speed on the pulled-back poses, a leaky bucket per request: allowed = min(left, v * slack) + v * dt, position and
rotation (w_max) each -> SHORTENED. Joint limits are not a gate rule; they are checked here by IK along the path
(10 Hz, warm-started from the start q): unreachable or the margin to the Panda limits.

**Phases** (by the demo's grasp row g, the width rule of ``placement.grasp_rows``): approach [0, g), grasp
[g, g + 0.5 s), lift [g + 0.5 s, g + 2 s), retreat (the rest).

Defaults: the user's real station (2026-11-20): box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70], box_margin 0.01,
cone_axis (0, 0, -1), cone_max 60, cone_margin 2, v_max 0.25, w_max 1.0, slack 0.010, max_lead 1.5 (the copied
config.yaml's values except the box).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

GATE = {"workspace_min_m": [0.20, -0.40, -0.005], "workspace_max_m": [0.80, 0.40, 0.70], "box_margin_m": 0.01,
        "cone_axis": [0.0, 0.0, -1.0], "cone_max_deg": 60.0, "cone_margin_deg": 2.0, "v_max_m_per_s": 0.25,
        "w_max_rad_per_s": 1.0, "slack_s": 0.010, "max_lead_s": 1.5}
CYCLE_S, EXEC_H, STEP_S = 0.1, 12, 0.05
PHASES = ("approach", "grasp", "lift", "retreat")
DEFAULT_DATASETS = ("~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip",
                    "~/data/umi/datasets/multi/pick1-double-v2/pick1-double-v2.zarr.zip")


def phase_of(t: float, g_s: float) -> str:
    return "approach" if t < g_s else "grasp" if t < g_s + 0.5 else "lift" if t < g_s + 2.0 else "retreat"


def carried(T_start: np.ndarray, P: np.ndarray):
    """W(t) as a function: position (linear) and rotation (slerp) interpolation of the carried 60 Hz path."""
    W = T_start @ np.linalg.inv(P[0]) @ P
    ts = np.arange(len(W)) / 60.0
    sl = Slerp(ts, Rotation.from_matrix(W[:, :3, :3]))

    def at(t):
        t = np.clip(np.atleast_1d(t), 0.0, ts[-1])
        p = np.stack([np.interp(t, ts, W[:, a, 3]) for a in range(3)], axis=1)
        return p, sl(t)
    return at, ts[-1]


def gate_request(start_p, start_R, wps_p, wps_R, g=GATE) -> dict:
    """One request through the gate port. Returns refused (rule, value), pulled (box/cone count), shortened (count),
    and the worst values."""
    lo, hi = np.asarray(g["workspace_min_m"]), np.asarray(g["workspace_max_m"])
    axis = np.asarray(g["cone_axis"], float)
    axis /= np.linalg.norm(axis)
    out = {"refused": None, "pulled_box": 0, "pulled_cone": 0, "shortened_p": 0, "shortened_r": 0}
    cand_p, cand_R = [], []
    for k in range(len(wps_p)):
        p = wps_p[k].copy()
        R = wps_R[k]
        if (k + 1) * STEP_S > g["max_lead_s"]:
            out["refused"] = ("lead", (k + 1) * STEP_S)
            return out
        below, above = lo - p, p - hi
        outv = np.maximum(np.maximum(below, above), 0.0)
        if np.any(outv > g["box_margin_m"] + 1e-12):
            a = int(np.argmax(outv))
            out["refused"] = ("box", f"{'xyz'[a]} {'<' if below[a] > 0 else '>'} by {outv[a]:.3f} m")
            return out
        if np.any(outv > 0):
            p = np.clip(p, lo, hi)
            out["pulled_box"] += 1
        z = R.as_matrix()[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(z @ axis, -1, 1))))
        if tilt > g["cone_max_deg"]:
            if tilt > g["cone_max_deg"] + g["cone_margin_deg"] + 1e-12:
                out["refused"] = ("cone", f"tilt {tilt:.1f} deg")
                return out
            k_ax = np.cross(z, axis)
            k_ax = k_ax / np.linalg.norm(k_ax)
            R = Rotation.from_rotvec(k_ax * math.radians(tilt - g["cone_max_deg"])) * R
            out["pulled_cone"] += 1
        cand_p.append(p)
        cand_R.append(R)
    v, w, sl = g["v_max_m_per_s"], g["w_max_rad_per_s"], g["slack_s"]
    left_p, left_r = v * sl, w * sl
    pp, pR, pt = start_p, start_R, 0.0
    for k, (p, R) in enumerate(zip(cand_p, cand_R)):
        t = (k + 1) * STEP_S
        dt = max(0.0, t - pt)
        allowed_p = min(left_p, v * sl) + v * dt
        d = float(np.linalg.norm(p - pp))
        if d > allowed_p + 1e-9:
            p = pp + (p - pp) * (allowed_p / d)
            out["shortened_p"] += 1
        left_p = allowed_p - min(d, allowed_p)
        allowed_r = min(left_r, w * sl) + w * dt
        ang = float((R * pR.inv()).magnitude())
        if ang > allowed_r + 1e-9:
            R = Slerp([0, 1], Rotation.concatenate([pR, R]))([allowed_r / ang])[0]
            out["shortened_r"] += 1
        left_r = allowed_r - min(ang, allowed_r)
        pp, pR, pt = p, R, t
    return out


def check_demo(T_start, P, widths, ik=None, g=GATE) -> dict:
    from sim_deploy.scene.placement import grasp_rows

    at, t_end = carried(T_start, P)
    g_s = grasp_rows(widths)["start"] / 60.0
    res = {"refused": [], "pulled_box": {}, "pulled_cone": {}, "shortened": {}, "first_refusal": None}
    tc = 0.0
    while tc <= t_end:
        sp, sR = at(tc)
        wp, wR = at(tc + STEP_S * np.arange(1, EXEC_H + 1))
        r = gate_request(sp[0], sR[0], wp, wR, g)
        ph = phase_of(tc, g_s)
        if r["refused"]:
            res["refused"].append((round(tc, 2), ph, r["refused"][0], r["refused"][1]))
            if res["first_refusal"] is None:
                res["first_refusal"] = {"t_s": round(tc, 2), "phase": ph, "rule": r["refused"][0], "detail": r["refused"][1]}
        for key, src in (("pulled_box", r["pulled_box"]), ("pulled_cone", r["pulled_cone"]),
                         ("shortened", r["shortened_p"] + r["shortened_r"])):
            if src:
                res[key][ph] = res[key].get(ph, 0) + 1
        tc += CYCLE_S
    # worst values along the whole carried path (50 ms steps), and what a limit would need
    ts = np.arange(0.0, t_end, STEP_S)
    p, R = at(ts)
    z = R.as_matrix()[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(z @ np.asarray(g["cone_axis"], float), -1, 1)))
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rot = (R[1:] * R[:-1].inv()).magnitude()
    v_need = step / (STEP_S + g["slack_s"])          # the speed limit this step needs (with the bucket's slack)
    w_need = rot / (STEP_S + g["slack_s"])
    k_v = int(np.argmax(v_need))
    res["worst"] = {"max_tilt_deg": round(float(tilt.max()), 1), "max_tilt_phase": phase_of(float(ts[int(np.argmax(tilt))]), g_s),
                    "v_needed_m_per_s": round(float(v_need.max()), 3), "v_peak_phase": phase_of(float(ts[k_v]), g_s),
                    "w_needed_rad_per_s": round(float(w_need.max()), 3),
                    "min_x_m": round(float(p[:, 0].min()), 3), "min_z_m": round(float(p[:, 2].min()), 3),
                    "max_x_m": round(float(p[:, 0].max()), 3), "min_y_m": round(float(p[:, 1].min()), 3),
                    "max_y_m": round(float(p[:, 1].max()), 3), "max_z_m": round(float(p[:, 2].max()), 3)}
    # the manager never makes a shortened distance up (each chunk is relative to its anchor = the previous target):
    # follow W with the speed limit only, the lost distance at the grasp; and the v that keeps it <= 1 cm
    def lost(v):
        tgt = p[0].copy()
        kg = min(len(p) - 1, int(round(g_s / STEP_S)))
        for k in range(1, kg + 1):
            dv = p[k] - p[k - 1]
            n_ = float(np.linalg.norm(dv))
            cap = v * (STEP_S + g["slack_s"])
            tgt = tgt + (dv if n_ <= cap else dv * cap / n_)
        return float(np.linalg.norm(tgt - p[kg]))
    res["worst"]["lost_at_grasp_m"] = round(lost(g["v_max_m_per_s"]), 3)
    res["worst"]["v_for_lost_1cm"] = next((v for v in np.arange(0.25, 2.01, 0.05) if lost(v) <= 0.01), 2.0)
    res["grasp_t_s"] = round(g_s, 2)
    if ik is not None:
        res["ik"] = ik(at, t_end)
    return res


def make_ik(q_start):
    """IK along the carried path at 10 Hz, warm-started: unreachable samples, min margin to the Panda limits."""
    from sim_deploy.scene import start_pose as sp
    from sim_deploy.scene import start_sweep as ss
    from sim_deploy.scene.trainstart import solve_ik

    mj, m, d, _ = ss._mj()
    lo, hi = np.asarray(sp.PANDA_Q_MIN), np.asarray(sp.PANDA_Q_MAX)

    def run(at, t_end):
        q = np.asarray(q_start, float)
        bad, marg, worst_j = 0, 9.0, None
        for t in np.arange(0.0, t_end, 0.1):
            p, R = at(t)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R.as_matrix()[0], p[0]
            qn, pe, ae = solve_ik(m, d, "tcp", T, q, iters=150)
            if pe > 1e-3 or ae > 0.5:
                bad += 1
                continue
            q = qn
            mm = np.minimum(q - lo, hi - q)
            if mm.min() < marg:
                marg, worst_j = float(mm.min()), int(np.argmin(mm)) + 1
        return {"unreachable_samples": bad, "joint_margin_min_rad": round(marg, 3), "joint": worst_j}
    return run


def evaluate(q, datasets=DEFAULT_DATASETS, g=GATE, with_ik=True) -> dict:
    from sim_deploy.scene import start_pose as sp
    from sim_deploy.scene.zarr_lowdim import load_episode, n_episodes

    T0 = np.asarray(sp.fk(q)["O_T_tcp"])
    ik = make_ik(q) if with_ik else None
    out = {"q": list(q), "gate": g, "datasets": {}}
    for ds in datasets:
        path = Path(ds).expanduser()
        if not path.exists():
            continue
        demos = []
        for i in range(n_episodes(path)):
            e = load_episode(path, i)
            r = check_demo(T0, e["poses"], e["widths"], ik, g)
            r["episode"] = i
            demos.append(r)
        out["datasets"][path.name] = {"demos": demos, "summary": summarize(demos, g)}
    return out


def summarize(demos: list[dict], g=GATE) -> dict:
    n = len(demos)
    k28 = max(1, math.ceil(n * 28 / 30))                  # "28 of 30" scaled to the dataset size
    s = {"n": n, "refused_demos": sum(bool(d["refused"]) for d in demos)}
    by_rule: dict = {}
    for d in demos:
        if d["first_refusal"]:
            fr = d["first_refusal"]
            by_rule.setdefault(fr["rule"], {}).setdefault(fr["phase"], 0)
            by_rule[fr["rule"]][fr["phase"]] += 1
    s["first_refusal_by_rule_phase"] = by_rule
    for key in ("pulled_box", "pulled_cone", "shortened"):
        s[key + "_demos"] = sum(bool(d[key]) for d in demos)
        ph: dict = {}
        for d in demos:
            for p_, c in d[key].items():
                ph[p_] = ph.get(p_, 0) + 1
        s[key + "_demos_by_phase"] = ph
    W = [d["worst"] for d in demos]
    tilt = sorted(w["max_tilt_deg"] for w in W)
    vn = sorted(w["v_needed_m_per_s"] for w in W)
    wn = sorted(w["w_needed_rad_per_s"] for w in W)
    s["worst"] = {"max_tilt_deg": tilt[-1], "v_needed_max": vn[-1], "w_needed_max": wn[-1],
                  "min_x_m": min(w["min_x_m"] for w in W), "min_z_m": min(w["min_z_m"] for w in W),
                  "max_x_m": max(w["max_x_m"] for w in W), "min_y_m": min(w["min_y_m"] for w in W),
                  "max_y_m": max(w["max_y_m"] for w in W), "max_z_m": max(w["max_z_m"] for w in W)}
    s[f"for_{k28}_of_{n}"] = {
        "cone_max_deg_no_pullback": tilt[k28 - 1], "cone_max_deg_no_refusal": round(tilt[k28 - 1] - g["cone_margin_deg"], 1),
        "v_max_no_shortening": vn[k28 - 1], "w_max_no_shortening": wn[k28 - 1]}
    lost = sorted(w["lost_at_grasp_m"] for w in W)
    vl = sorted(float(w["v_for_lost_1cm"]) for w in W)
    s["lost_at_grasp_m"] = {"median": lost[n // 2], "max": lost[-1], "demos_over_2cm": sum(x > 0.02 for x in lost)}
    s[f"for_{k28}_of_{n}"]["v_max_lost_at_grasp_le_1cm"] = round(vl[k28 - 1], 2)
    s["v_peak_phase"] = {p_: sum(w["v_peak_phase"] == p_ for w in W) for p_ in PHASES}
    s["tilt_peak_phase"] = {p_: sum(w["max_tilt_phase"] == p_ for w in W) for p_ in PHASES}
    if demos and "ik" in demos[0]:
        s["ik_unreachable_demos"] = sum(d["ik"]["unreachable_samples"] > 0 for d in demos)
        s["ik_joint_margin_min_rad"] = min(d["ik"]["joint_margin_min_rad"] for d in demos)
        jm = sorted(d["ik"]["joint_margin_min_rad"] for d in demos)
        s["ik_joint_margin_p10_rad"] = jm[max(0, n // 10 - 1)]
    return s


def main(argv=None) -> int:
    from sim_deploy.scene import start_pose as sp

    ap = argparse.ArgumentParser(prog="python -m sim_deploy.scene.gate_check", description=__doc__.split("\n")[0])
    ap.add_argument("--q", required=True)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--no-ik", action="store_true")
    ap.add_argument("--no-floor", action="store_true", help="box floor at -1 m: separates carried-into-the-table from the rest")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    g = dict(GATE, workspace_min_m=[GATE["workspace_min_m"][0], GATE["workspace_min_m"][1], -1.0]) if a.no_floor else GATE
    r = evaluate(sp.parse_q_text(a.q), a.dataset or DEFAULT_DATASETS, g=g, with_ik=not a.no_ik)
    for name, d in r["datasets"].items():
        print(name, json.dumps(d["summary"], indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
