"""P4 (2026-11-20): the start-pose sweep. Candidate start poses over the demos' range, every start-pose check in the
sim and on the REAL PC, the joint path from home, and the sweep's summary table.

    python -m sim_deploy.scene.start_sweep candidates [--out DIR]     the grid, IK, checks -> DIR/candidates.{json,md},
                                                                      DIR/poses/<id>.yaml (start_pose.yaml format)
    python -m sim_deploy.scene.start_sweep summary [--out DIR]        sim_evals tagged sweep_* -> DIR/summary.{csv,md}
    python -m sim_deploy.scene.start_sweep handoff --q "q1,...,q7"     the real-PC handoff numbers for one pose (JSON)

DIR defaults to sim_evals/_start_pose_sweep/. Runs in ~/envs/sim_deploy.

**Candidates.** tilt (fingertip direction from straight down) x fingertip height above the table x heading (the
fingertip direction on the table: the robot's left = +y, away from the base = +x, the robot's right = -y, and the
two diagonals). The orientation is the stand-in's (E7: the demos' mean start orientation, closing axis on the table),
turned about the vertical to the heading and about its own left-right axis to the tilt. The position puts the demos'
median eraser centre (``randomization_start.yaml`` median: 15.9 cm ahead, 1.9 cm right of the fingertips) at a fixed
point G on the table (``GRASP_CENTRE``: (0.45, 0) for left / right = the stand-in's layout, (0.50, 0) for the
diagonals; (0.60, 0) for away: with G at 0.50 the flange ends up over the base column and no IK solution exists
(joint 4 at its limit, 39 mm residual), at 0.40 not at all (joints 1, 2, 4)). IK = ``trainstart.solve_ik``
(damped least squares), 24 seeds, the solution closest to the home in joints 1-6 with a joint-margin bonus.

**Checks** (a pose is feasible when all pass): the sim's ``start_pose.describe`` (Panda and model joint limits, TCP in
the sim box, cone, no contact, heading, the p move within homing.timeout_s), a joint margin >= ``JOINT_MARGIN_MIN``
from the Panda limits, the eraser region (every corner of ``randomization_start.yaml``'s region and its median: on
the table, in the sim box, inside BOTH cameras' views at the pose), and the joint path from home (linear joint
interpolation, as franka_control's HomingMotion: same geometric path at any speed) free of contacts with
``PATH_CLEARANCE_MIN`` to the table and between the hand / rig / fingers / links 6-7 and links 0-3.

**Real PC** (`real_pc`): the same q against the user's real station (2026-11-20 working tree: box
[0.20, -0.40, -0.005]..[0.80, 0.40, 0.70], F_T_TCP 0.23565 m = the fingertip point, cone 60 deg, v_max 0.25), as `p`
checks at its end (box and cone without margin), the joint limits, the move time at max_joint_speed 0.3 rad/s
against the 30 s homing timeout, and what `pc_probe --start-pose` would print (TCP z in the base frame, tilt,
heading). For information also with the upstream config.yaml @ ea9f7fb box [0.25, -0.40, 0.05] and F_T_TCP 0.150 m.

    python -m sim_deploy.scene.start_sweep run --stage screen|full --poses ID,...|feasible --policies RUN,...
           [--rollouts 1] [--timeout 25] [--seed 0] [--fast-homing] [--video-fps F]
       one sim per pose (merge_station with the pose file, + stations/homing_fast_sweep.yaml with --fast-homing),
       the eval driver per policy with --start-pose and tag sweep_<stage>_<pose id> (bf16 / 8 steps / exec_h 12 for
       every checkpoint), a latched fault -> restart the sim and --resume; then `summary`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO, ROOT
from sim_deploy.scene import start_pose as sp

OUT = REPO / "sim_evals" / "_start_pose_sweep"
TILTS_DEG = (38.0, 48.0, 58.0)
HEIGHTS_M = (0.10, 0.16, 0.22)
HEADINGS = {"left": 90.0, "away": 0.0, "right": -90.0, "away_left": 45.0, "away_right": -45.0}
GRASP_CENTRE = {"left": (0.45, 0.0), "right": (0.45, 0.0), "away": (0.60, 0.0), "away_left": (0.50, 0.0),
                "away_right": (0.50, 0.0)}
JOINT_MARGIN_MIN = 0.10          # rad from the Panda limits
PATH_CLEARANCE_MIN = 0.02        # m: hand / rig / fingers / links to the table and to links 0-3 along the p move
PATH_SAMPLES = 121
# The stand-in (P3, table at O z 0): its TCP orientation is the demos' mean start orientation (E7 trainstart).
STAND_IN_Z0 = [-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]
REAL_F_T_TCP_Z = 0.150           # final-deployment @ ea9f7fb contracts/frames.yaml (settled: false)
REAL_BOX = ([0.20, -0.40, -0.005], [0.80, 0.40, 0.70])  # the user's real station (2026-11-20 working tree: x 0.20,
REAL_BOX_UPSTREAM = ([0.25, -0.40, 0.05], [0.80, 0.40, 0.70])   # z -0.005); config.yaml @ ea9f7fb had these (TODO)
REAL_CONE_DEG = 60.0
REAL_MAX_JOINT_SPEED = 0.3
REAL_HOMING_TIMEOUT_S = 30.0
REAL_MIN_DURATION_S = 2.0
NEAR_BODIES = ("link0", "link1", "link2", "link3")
FAR_BODIES = ("link6", "link7", "hand", "left_finger", "right_finger", "rig")
TABLE_BODIES = ("link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger", "rig")


def _mj():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = json.loads(sp.DEFAULT_MODEL.read_text())
    m = mujoco.MjModel.from_xml_path(str(sp.DEFAULT_MODEL.with_name(model["scene_xml"])))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    return mujoco, m, d, model


def _Rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def target_pose(tilt_deg: float, height_m: float, heading: str, table_z: float, med: dict, R0: np.ndarray,
                tilt0: float, head0: float) -> np.ndarray:
    """O_T_tcp of a candidate: the stand-in's orientation turned to heading / tilt; the demos' median eraser centre
    at GRASP_CENTRE[heading]."""
    h = HEADINGS[heading]
    R = _Rz(h) @ _Ry(tilt0 - tilt_deg) @ _Rz(-head0) @ R0
    f = np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])
    lft = np.array([-f[1], f[0]])
    g = np.asarray(GRASP_CENTRE[heading], float)
    xy = g - float(med["ahead_m"]) * f - float(med["left_m"]) * lft
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [xy[0], xy[1], table_z + height_m]
    return T


def solve(m, d, T: np.ndarray, q_home: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    from sim_deploy.scene.trainstart import solve_ik

    jr = m.jnt_range[:7]
    rng = np.random.default_rng(0)
    best = None
    seeds = [q_home, np.asarray(STAND_IN_Z0)] + [np.clip(q_home + rng.normal(0, 0.6, 7), jr[:, 0], jr[:, 1])
                                                 for _ in range(22)]
    for seed in seeds:
        qk, pe, ae = solve_ik(m, d, "tcp", T, np.asarray(seed, float))
        if pe > 1e-4 or ae > 0.05:
            continue
        margin = float(np.min(np.minimum(qk - jr[:, 0], jr[:, 1] - qk)))
        cost = float(np.linalg.norm((qk - q_home)[:6])) - 0.5 * min(margin, 0.3)
        if best is None or cost < best[0]:
            best = (cost, qk, pe, ae)
    if best is None:
        return None, float("nan"), float("nan")
    return best[1], best[2], best[3]


def _geoms(m, bodies) -> list[int]:
    ids = {m.body(b).id for b in bodies}
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]


def path_check(mujoco, m, d, q_home, q, table_z: float) -> dict:
    """The p move (linear joint interpolation home -> q, HomingMotion's path): contacts, min clearance of the hand /
    rig / fingers / links 2-7 to the table, and of links 6-7 / hand / rig / fingers to links 0-3."""
    table = m.geom("table").id
    far, near, low = _geoms(m, FAR_BODIES), _geoms(m, NEAR_BODIES), _geoms(m, TABLE_BODIES)
    eraser_adr = m.jnt_qposadr[m.joint("eraser").id]
    c_tab, c_self, contacts, worst_tab, worst_self = 1e9, 1e9, [], None, None
    lowest = 1e9
    for s in np.linspace(0.0, 1.0, PATH_SAMPLES):
        d.qpos[:7] = (1 - s) * np.asarray(q_home) + s * np.asarray(q)
        d.qpos[eraser_adr:eraser_adr + 3] = [5.0, 5.0, 1.0]           # out of the way (it is placed after p)
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            names = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
            if "eraser" not in names:
                contacts.append([round(float(s), 3), sorted(names)])
        for g in low:
            dist = mujoco.mj_geomDistance(m, d, g, table, 0.5, None)
            if dist < c_tab:
                c_tab, worst_tab = dist, [round(float(s), 3), m.body(m.geom_bodyid[g]).name]
        for g in far:
            for h in near:
                dist = mujoco.mj_geomDistance(m, d, g, h, 0.5, None)
                if dist < c_self:
                    c_self, worst_self = dist, [round(float(s), 3), m.body(m.geom_bodyid[g]).name,
                                                m.body(m.geom_bodyid[h]).name]
        lowest = min(lowest, float(d.site_xpos[m.site("tcp").id][2]) - table_z)
    return {"contacts": contacts[:5], "n_contact_samples": len({c[0] for c in contacts}),
            "table_clearance_m": round(float(c_tab), 4), "table_clearance_at": worst_tab,
            "self_clearance_m": round(float(c_self), 4), "self_clearance_at": worst_self,
            "lowest_fingertip_above_table_m": round(lowest, 4),
            "ok": not contacts and c_tab >= PATH_CLEARANCE_MIN and c_self >= PATH_CLEARANCE_MIN}


def real_pc(mujoco, m, d, q, q_home, sim_tcp_z: float) -> dict:
    """The same q on the real PC: F_T_TCP 0.150 (real) and 0.23565 (sim/D7), real box and cone, move time at 0.3."""
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    sid = m.site("tcp").id
    T_sim = np.eye(4)
    T_sim[:3, :3], T_sim[:3, 3] = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    lo, hi = np.asarray(REAL_BOX[0]), np.asarray(REAL_BOX[1])
    ulo, uhi = np.asarray(REAL_BOX_UPSTREAM[0]), np.asarray(REAL_BOX_UPSTREAM[1])
    out = {}
    for name, zoff in (("tcp_150mm", REAL_F_T_TCP_Z), ("tcp_235mm", sim_tcp_z)):
        T = T_sim.copy()
        T[:3, 3] = T_sim[:3, 3] - (sim_tcp_z - zoff) * T_sim[:3, 2]   # same rotation, shorter along TCP z
        p = T[:3, 3]
        inside = np.minimum(p - lo, hi - p)                           # per axis, > 0 inside
        tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
        z = T[:3, 2]
        out[name] = {"tcp_O_m": [round(float(v), 4) for v in p], "box_ok": bool(np.all(inside >= 0)),
                     "box_margin_m": round(float(inside.min()), 4),
                     "box_limiting_axis": "xyz"[int(np.argmin(inside))],
                     "upstream_box_ok": bool(np.all(p >= ulo) and np.all(p <= uhi)),
                     "cone_ok": tilt <= REAL_CONE_DEG, "cone_margin_deg": round(REAL_CONE_DEG - tilt, 2),
                     "pc_probe": {"z_m": round(float(p[2]), 3), "dir": [round(float(v), 2) for v in z],
                                  "tilt_deg": round(tilt, 1),
                                  "words": f"pointing {sp.heading_words(z[0], z[1])}"
                                  if math.hypot(z[0], z[1]) >= 0.035 else "no horizontal direction"}}
    qa = np.asarray(q, float)
    jm = float(np.min(np.minimum(qa - sp.PANDA_Q_MIN, np.asarray(sp.PANDA_Q_MAX) - qa)))
    dur = max(REAL_MIN_DURATION_S, sp.PEAK_SPEED_FACTOR * float(np.max(np.abs(qa - np.asarray(q_home))))
              / REAL_MAX_JOINT_SPEED)
    out.update(joint_margin_rad=round(jm, 3), joints_ok=jm > 0, move_time_s=round(dur, 1),
               move_time_ok=dur < REAL_HOMING_TIMEOUT_S,
               p_accepted_real_tcp=bool(jm > 0 and dur < REAL_HOMING_TIMEOUT_S and out["tcp_150mm"]["box_ok"]
                                        and out["tcp_150mm"]["cone_ok"]),
               p_accepted_fingertip_tcp=bool(jm > 0 and dur < REAL_HOMING_TIMEOUT_S and out["tcp_235mm"]["box_ok"]
                                             and out["tcp_235mm"]["cone_ok"]))
    return out


def evaluate(q, *, pid: str, out_dir: Path, mujoco=None, m=None, d=None, model=None, extra: dict | None = None) -> dict:
    """Every check of one pose (also used by `handoff`). Writes out_dir/poses/<pid>.yaml for the region check."""
    from sim_deploy.scene.placement import Scene

    if m is None:
        mujoco, m, d, model = _mj()
    table_z = float(model["table"]["top_z_m"])
    cfg = sp.effective_config(sp.CONTROL_SIM)
    q_home = np.asarray(cfg["homing"]["q_home_rad"], float)
    info = sp.describe(q)
    qa = np.asarray(q, float)
    jm = float(np.min(np.minimum(qa - sp.PANDA_Q_MIN, np.asarray(sp.PANDA_Q_MAX) - qa)))
    jm_i = int(np.argmin(np.minimum(qa - sp.PANDA_Q_MIN, np.asarray(sp.PANDA_Q_MAX) - qa))) + 1
    path = out_dir / "poses" / f"{pid}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.write_start_pose(list(map(float, q)), f"P4 sweep candidate {pid}", info, path)
    try:
        rep = Scene.load(sp.DEFAULT_MODEL, home="start", start_pose=path).check_region()
        bad = {k: sum(not c[k] for c in rep["corners"]) for k in ("on_table", "in_workspace", "in_view_imx219",
                                                                  "in_view_imx519")}
        region = {"ok": not any(bad.values()), "failing_corners": {k: v for k, v in bad.items() if v},
                  "n_corners": len(rep["corners"])}
    except Exception as e:                                # noqa: BLE001
        region = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    pth = path_check(mujoco, m, d, q_home, q, table_z)
    real = real_pc(mujoco, m, d, q, q_home, float(np.asarray(model["F_T_TCP"])[2, 3]))
    s = cfg["safety"]
    p = np.asarray(info["tcp_O_m"])
    box_m = float(np.min(np.minimum(p - np.asarray(s["workspace_min_m"]), np.asarray(s["workspace_max_m"]) - p)))
    fails = [k for k, c in info["checks"].items() if not c["ok"]]
    if jm < JOINT_MARGIN_MIN:
        fails.append(f"joint_margin<{JOINT_MARGIN_MIN} (joint {jm_i})")
    if not region["ok"]:
        fails.append("region:" + ",".join(region.get("failing_corners", {}) or ["error"]))
    if not pth["ok"]:
        fails.append("path:" + ("contact" if pth["contacts"] else
                                "table" if pth["table_clearance_m"] < PATH_CLEARANCE_MIN else "self"))
    return {"id": pid, **(extra or {}), "q": [round(float(v), 7) for v in q], "ok": not fails, "fails": fails,
            "tcp_O_m": info["tcp_O_m"], "tilt_deg": info["tilt_deg"], "heading_deg": info["heading_deg"],
            "heading_words": info["heading_words"], "height_above_table_m": info["height_above_table_m"],
            "joint_margin_rad": round(jm, 3), "joint_margin_joint": jm_i, "sim_box_margin_m": round(box_m, 4),
            "cone_margin_deg": round(float(s["cone_max_deg"]) - info["cone_tilt_deg"], 2),
            "move_time_s_sim": info["move_time_s"], "checks": {k: c["ok"] for k, c in info["checks"].items()},
            "region": region, "path": pth, "real_pc": real, "pose_file": str(path)}


def candidates(out_dir: Path = OUT) -> list[dict]:
    mujoco, m, d, model = _mj()
    table_z = float(model["table"]["top_z_m"])
    rnd = yaml.safe_load(sp.RANDOM_START.read_text())
    med = rnd["median"]
    f0 = sp.fk(STAND_IN_Z0)
    R0 = f0["O_T_tcp"][:3, :3]
    num0 = sp.tcp_numbers(f0["O_T_tcp"], table_z)
    q_home = np.asarray(sp.effective_config(sp.CONTROL_SIM)["homing"]["q_home_rad"], float)
    rows = []
    for heading in HEADINGS:
        for tilt in TILTS_DEG:
            for h in HEIGHTS_M:
                pid = f"{heading}_t{tilt:.0f}_h{h * 100:.0f}"
                T = target_pose(tilt, h, heading, table_z, med, R0, num0["tilt_deg"], num0["heading_deg"])
                q, pe, ae = solve(m, d, T, q_home)
                extra = {"heading": heading, "tilt_target_deg": tilt, "height_target_m": h,
                         "grasp_centre_O_m": list(GRASP_CENTRE[heading]),
                         "target_tcp_O_m": [round(float(v), 4) for v in T[:3, 3]]}
                if q is None:
                    rows.append({"id": pid, **extra, "ok": False, "fails": ["ik: no solution (reach / joint limits)"]})
                else:
                    r = evaluate(q, pid=pid, out_dir=out_dir, mujoco=mujoco, m=m, d=d, model=model, extra=extra)
                    r["ik"] = {"pos_err_m": pe, "ang_err_deg": ae}
                    rows.append(r)
                print(f"{pid:<22} {'OK  ' if rows[-1]['ok'] else 'FAIL'} {', '.join(rows[-1]['fails'])}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "candidates.md").write_text(candidates_md(rows))
    return rows


def candidates_md(rows: list[dict]) -> str:
    out = ["# P4 start-pose candidates (offline checks)", "",
           "| id | ok | tilt | tip above table (cm) | TCP O (m) | joint margin (rad, joint) | sim box margin (m) | cone "
           "margin (deg) | path: table / self clearance (cm) | real PC p (150 mm TCP / fingertip TCP) | real box "
           "margin 150 mm (m, axis) | fails |", "|" + "---|" * 12]
    for r in rows:
        if "q" not in r:
            out.append(f"| {r['id']} | no | | | | | | | | | | {'; '.join(r['fails'])} |")
            continue
        rp = r["real_pc"]
        out.append(
            f"| {r['id']} | {'yes' if r['ok'] else 'no'} | {r['tilt_deg']:.1f} | {r['height_above_table_m'] * 100:.1f} "
            f"| {r['tcp_O_m']} | {r['joint_margin_rad']:.2f} (j{r['joint_margin_joint']}) | {r['sim_box_margin_m']:.3f} "
            f"| {r['cone_margin_deg']:.1f} | {r['path']['table_clearance_m'] * 100:.1f} / "
            f"{r['path']['self_clearance_m'] * 100:.1f} | {'ok' if rp['p_accepted_real_tcp'] else 'REFUSED'} / "
            f"{'ok' if rp['p_accepted_fingertip_tcp'] else 'REFUSED'} | {rp['tcp_150mm']['box_margin_m']:.3f} "
            f"({rp['tcp_150mm']['box_limiting_axis']}) | {'; '.join(r['fails'])} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ summary

FAULT_RULES = (("box", re.compile(r"outside workspace box")), ("cone", re.compile(r"from approach axis|cone")),
               ("reflex", re.compile(r"reflex", re.I)), ("v_max", re.compile(r"v_max|speed", re.I)))


def fault_rule(text: str | None) -> str:
    if not text:
        return ""
    for name, rx in FAULT_RULES:
        if rx.search(text):
            return name
    return "other"


def classify(rec: dict, rollout_dir: Path) -> dict:
    """Screen class of one rollout: fault_early (< 10 s, with the rule), no_progress, approach, grasp, lift."""
    dur = rec.get("duration_s")
    rule = fault_rule(rec.get("fault"))
    tr = rollout_dir / "trace.json"
    min_d = d0 = None
    if tr.exists() and rec.get("eraser_settled_pose"):
        e = np.asarray(rec["eraser_settled_pose"][:3], float)
        rows = [r for r in json.loads(tr.read_text())["rows"] if len(r) > 4 and r[4]]
        if rows:
            dist = [float(np.linalg.norm(np.asarray(r[4][:3]) - e)) for r in rows]
            min_d, d0 = min(dist), dist[0]
    if rec.get("outcome") == "fault" and dur is not None and dur < 10.0:
        cls = f"fault_early_{rule}"
    elif rec.get("lifted"):
        cls = "lift"
    elif rec.get("grasped"):
        cls = "grasp"
    elif min_d is not None and min_d < 0.05:
        cls = "approach"
    else:
        cls = "no_progress"
    return {"class": cls, "fault_rule": rule, "t_first_fault_s": round(dur, 1) if rec.get("fault") and dur else None,
            "min_tcp_to_eraser_m": round(min_d, 3) if min_d is not None else None,
            "start_tcp_to_eraser_m": round(d0, 3) if d0 is not None else None}


def summary(out_dir: Path = OUT, sim_evals: Path = REPO / "sim_evals") -> list[dict]:
    cand = {r["id"]: r for r in json.loads((out_dir / "candidates.json").read_text())} \
        if (out_dir / "candidates.json").exists() else {}
    rows = []
    for ev in sorted(sim_evals.glob("*/*/*_sweep*")):
        if not (ev / "metrics.json").exists() or (ev / "SUPERSEDED.txt").exists() or (ev / "INTERRUPTED.txt").exists():
            continue
        doc = yaml.safe_load((ev / "eval.yaml").read_text())
        tag = doc["tag"]
        mt = re.match(r"sweep_(screen|full)_(.+)$", tag)
        if not mt:
            continue
        stage, pid = mt.groups()
        ros = [json.loads(x) for x in (ev / "rollouts.jsonl").read_text().splitlines() if x.strip()]
        ros = [e for e in ros if e.get("outcome") not in ("not_run", "interrupted")]
        cls = [classify(e, ev / "rollouts" / f"{e['n']:03d}") for e in ros]
        c = cand.get(pid, {})
        tff = [x["t_first_fault_s"] for x in cls if x["t_first_fault_s"] is not None]
        rows.append({
            "stage": stage, "pose_id": pid, "checkpoint": f"{doc['policy']['task']}/{doc['policy']['run_folder']}",
            "kind": doc["policy"]["kind"], "rollouts": len(ros), "success": sum(bool(e["success"]) for e in ros),
            "grasp": sum(bool(e["grasped"]) for e in ros), "lift": sum(bool(e["lifted"]) for e in ros),
            "faults": sum(bool(e.get("fault")) for e in ros),
            "fault_rules": "/".join(sorted({x["fault_rule"] for x in cls if x["fault_rule"]})),
            "t_first_fault_s": min(tff) if tff else "", "classes": "/".join(x["class"] for x in cls),
            "min_tcp_to_eraser_m": min((x["min_tcp_to_eraser_m"] for x in cls if x["min_tcp_to_eraser_m"] is not None),
                                       default=""),
            "tilt_deg": c.get("tilt_deg", ""), "height_cm": round(c["height_above_table_m"] * 100, 1)
            if c.get("height_above_table_m") is not None else "", "heading": c.get("heading", ""),
            "q": json.dumps(c.get("q", [])), "eval_dir": str(ev.relative_to(sim_evals))})
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = ["# P4 start-pose sweep: per pose x checkpoint", "",
          "| stage | pose | tilt | tip (cm) | heading | checkpoint | kind | rollouts | success | grasp | lift | faults (rule) | "
          "first fault (s) | classes | closest TCP-eraser (m) |", "|" + "---|" * 15]
    for r in rows:
        md.append(f"| {r['stage']} | {r['pose_id']} | {r['tilt_deg']} | {r['height_cm']} | {r['heading']} | "
                  f"{r['checkpoint']} | {r['kind']} | {r['rollouts']} | {r['success']} | {r['grasp']} | {r['lift']} | "
                  f"{r['faults']} ({r['fault_rules']}) | {r['t_first_fault_s']} | {r['classes']} | "
                  f"{r['min_tcp_to_eraser_m']} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    return rows


# ------------------------------------------------------------------------------------------------ run


class Sim:
    """The sim PC + sim Pi for one start pose, started as run.sh does (merge_station, scene hello with the model sha,
    `ready: C2`, a pi_stream packet), kept up across the evals of that pose; restarted after a latched fault."""

    def __init__(self, pose_file: Path, logs: Path, overlay: Path | None):
        self.pose_file, self.logs, self.overlay = Path(pose_file), Path(logs), overlay
        self.station = ROOT / "build" / f"control_sim_sweep_{self.pose_file.stem}.merged.yaml"
        self.procs: list = []

    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.setdefault("MUJOCO_GL", "egl")
        return env

    def start(self, tries: int = 12, wait_s: float = 10.0) -> None:
        import subprocess
        import sys
        import time

        for i in range(tries):
            if self._start_once():
                return
            self.stop()
            print(f"[sweep] sim start failed ({i + 1}/{tries}): retrying in {wait_s:.0f} s", flush=True)
            time.sleep(wait_s)
        raise SystemExit(f"[sweep] the sim did not start (see {self.logs})")

    def _start_once(self) -> bool:
        import socket
        import subprocess
        import sys
        import time

        import zmq

        from sim_deploy.eval.sim_link import SceneClient, SceneError

        for port in (5555, 5620, 5630, 5650, 5651):
            with socket.socket() as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    print(f"[sweep] port {port} in use", flush=True)
                    return False
        self.logs.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-I", str(ROOT / "sim_pc" / "merge_station.py"), str(ROOT / "stations" / "control_sim.yaml"),
               str(self.pose_file), str(self.station)] + ([str(self.overlay)] if self.overlay else [])
        subprocess.run(cmd, check=True, env=self._env())
        fc_dir = REPO / "deployment" / "control_deploy" / "franka_control"
        pc_log = (self.logs / "sim_pc.log").open("w")
        self.procs.append(subprocess.Popen([str(fc_dir / "build-sim" / "franka_control"), "--sim", "--local",
                                            "--station", str(self.station)], cwd=fc_dir, stdout=pc_log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self._env()))
        want = json.loads(sp.DEFAULT_MODEL.read_text())["model_sha256"]
        end = time.monotonic() + 60
        ok = False
        while time.monotonic() < end and self.procs[0].poll() is None:
            if "ready: C2 on" in (self.logs / "sim_pc.log").read_text():
                try:
                    h = SceneClient(timeout_s=1.0).hello()
                    if h.get("model_sha256") != want:
                        raise SystemExit(f"[sweep] the sim PC runs model {h.get('model_sha256')}, model.json {want}")
                    ok = True
                    break
                except SceneError:
                    pass
            time.sleep(0.2)
        if not ok:
            txt = (self.logs / "sim_pc.log").read_text()
            print("[sweep] sim PC: " + next((ln for ln in txt.splitlines() if "FATAL" in ln), txt[-300:]), flush=True)
            return False
        pi_log = (self.logs / "sim_pi.log").open("w")
        self.procs.append(subprocess.Popen([sys.executable, "-m", "sim_deploy.sim_pi", "--station",
                                            str(ROOT / "stations" / "sim_rig.yaml")], cwd=REPO, stdout=pi_log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self._env()))
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.setsockopt(zmq.LINGER, 0)
        s.connect("tcp://127.0.0.1:5555")
        got = bool(s.poll(40000))
        s.close()
        if not got or any(p.poll() is not None for p in self.procs):
            print("[sweep] no pi_stream packet within 40 s", flush=True)
            return False
        return True

    def stop(self) -> None:
        import signal
        import time

        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        end = time.monotonic() + 5
        for p in self.procs:
            while p.poll() is None and time.monotonic() < end:
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()
                p.wait()
        self.procs = []


def run(stage: str, pose_ids: list[str], policies: list[str], rollouts: int, timeout: float, out_dir: Path,
        fast_homing: bool, seed: int = 0, extra: list[str] | None = None) -> list[str]:
    """For each pose: one sim, then the eval driver for every policy (tag sweep_<stage>_<pose id>, --start-pose the
    pose's file); exit 3 (latched fault) -> restart the sim and --resume the same folder (<= 5 times)."""
    import subprocess
    import sys
    import time

    overlay = ROOT / "stations" / "homing_fast_sweep.yaml" if fast_homing else None
    flags = ["--precision", "bf16", "--diffusion-steps", "8", "--exec-h", "12"] + (extra or [])
    done = []
    for pid in pose_ids:
        pose = out_dir / "poses" / f"{pid}.yaml"
        if not pose.exists():
            raise SystemExit(f"[sweep] no {pose}: run `candidates` first")
        logs = REPO / "sim_evals" / "_logs" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_sweep_{stage}_{pid}"
        sim = Sim(pose, logs, overlay)
        sim.start()
        try:
            for pol in policies:
                tag = f"sweep_{stage}_{pid}"
                cmd = [sys.executable, "-m", "sim_deploy.eval", "run", "--policy", pol, "--rollouts", str(rollouts),
                       "--seed", str(seed), "--tag", tag, "--timeout", str(timeout), "--start-pose", str(pose),
                       "--look", "current"] + flags                 # the sweep's sim runs sim_rig.yaml
                env = {**sim._env(), "SIM_DEPLOY_SIM_PC_STATION": str(sim.station)}
                folder, code, restarts = None, None, 0
                with (logs / "eval.log").open("a") as lg:
                    while True:
                        c = cmd + (["--resume", folder] if folder and code == 3 else [])
                        t0 = time.monotonic()
                        r = subprocess.run(c, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)
                        lg.write(r.stdout)
                        code = r.returncode
                        m = re.search(r"^\[eval\] eval folder: (.+)$", r.stdout, re.M)
                        folder = folder or (m.group(1).strip() if m else None)
                        last = [ln for ln in r.stdout.splitlines() if ln.startswith("[eval] rollout")]
                        print(f"[sweep] {pid} {Path(pol).name}: exit {code} in {time.monotonic() - t0:.0f} s; "
                              + " | ".join(ln[7:] for ln in last), flush=True)
                        if code == 3 and folder and restarts < 5:
                            restarts += 1
                            sim.stop()
                            time.sleep(1)
                            sim.start()
                            continue
                        break
                done.append(folder or "")
        finally:
            sim.stop()
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.scene.start_sweep", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("candidates", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(OUT))
    s = sub.add_parser("handoff")
    s.add_argument("--q", required=True)
    s.add_argument("--id", default="handoff")
    s.add_argument("--out", default=str(OUT))
    s = sub.add_parser("run", help="evals of poses x policies (one sim per pose); tag sweep_<stage>_<pose id>")
    s.add_argument("--stage", choices=("screen", "full"), required=True)
    s.add_argument("--poses", required=True, help="comma-separated pose ids (candidates.json), or 'feasible'")
    s.add_argument("--policies", required=True, help="comma-separated run folders")
    s.add_argument("--rollouts", type=int, default=1)
    s.add_argument("--timeout", type=float, default=25.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--video-fps", type=float, default=None, help="passed to the driver (0 = no external video)")
    s.add_argument("--fast-homing", action="store_true",
                   help="sim-only stations/homing_fast_sweep.yaml (h and p at 0.6 rad/s; the policy never sees them)")
    s.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "run":
        ids = ([r["id"] for r in json.loads((out / "candidates.json").read_text()) if r["ok"]]
               if a.poses == "feasible" else a.poses.split(","))
        run(a.stage, ids, [str(Path(p).expanduser()) for p in a.policies.split(",")], a.rollouts, a.timeout, out,
            a.fast_homing, a.seed, extra=["--video-fps", str(a.video_fps)] if a.video_fps is not None else None)
        summary(out)
        return 0
    if a.cmd == "candidates":
        rows = candidates(out)
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} feasible; {out / 'candidates.md'}")
    elif a.cmd == "summary":
        rows = summary(out)
        print(f"{len(rows)} evals; {out / 'summary.md'}")
    else:
        print(json.dumps(evaluate(sp.parse_q_text(a.q), pid=a.id, out_dir=out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
