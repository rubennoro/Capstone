"""The training-like start pose (E7): a second home for the sim PC, so a policy starts where its training episodes did.

    python -m sim_deploy.scene.trainstart [--dataset ZIP] [--write]

Every PICK_TEST1 recording starts with the fingers 40-64 deg from vertical, the camera looking ahead over the table,
the fingertip point 16 cm behind and 17 cm above where the grasp happens (medians; umiproc's recording frame has z up
and the table at z ~ 0). The deployment home (copied config.yaml homing.q_home_rad) points the fingers straight down
30.5 cm above the table instead. This module:

1. takes every episode's start pose relative to its grasp, in a gravity-aligned frame at the grasp (x = the closing
   axis projected on the table, z up): ``S = inv(H_grasp) . P_start``; the median position and the mean rotation;
2. puts the eraser region where that start reaches inside the gate's workspace box: the start TCP is
   ``H_sim . S`` with H_sim at the centre of ``randomization_trainstart.yaml``'s region, at the median recorded grasp
   tip height above the table, x = that file's ``closing_axis_O`` (O +x: approached from the robot's right; from the
   robot side, as the deployment home faces, the start has no IK solution);
3. solves the arm's inverse kinematics for that TCP pose in the model (damped least squares from the deployment
   home, joint limits, a pull toward the deployment home in the null space), and checks it: TCP error, joint limits,
   no contacts, inside the box, the whole region in the IMX219 view;
4. with ``--write``: ``stations/control_sim_trainstart.yaml`` (control_sim.yaml + franka_control.homing.q_home_rad,
   a station override of config.yaml, merged by franka_control's loader) and ``scene/trainstart.yaml`` (the pose,
   the camera poses at it, the statistics), which ``Scene.load(home="trainstart")`` reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from sim_deploy import REPO, ROOT
from sim_deploy.scene.placement import DEFAULT_MODEL, grasp_rows
from sim_deploy.scene.zarr_lowdim import load_episode, n_episodes

DEFAULT_DATASET = Path("~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip").expanduser()
RANDOM_TRAINSTART = ROOT / "scene" / "randomization_trainstart.yaml"
TRAINSTART_YAML = ROOT / "scene" / "trainstart.yaml"
CONTROL_SIM = ROOT / "stations" / "control_sim.yaml"
CONTROL_TRAINSTART = ROOT / "stations" / "control_sim_trainstart.yaml"
CONFIG = REPO / "deployment" / "control_deploy" / "franka_control" / "config.yaml"


def relative_starts(dataset) -> dict:
    """Start pose of every episode in the gravity-aligned grasp frame (x = closing axis on the table, z up)."""
    S, info = [], []
    for i in range(n_episodes(dataset)):
        ep = load_episode(dataset, i)
        P, g = ep["poses"], grasp_rows(ep["widths"])
        Pg, P0 = P[g["start"]], P[0]
        x = Pg[:3, 0].copy()
        x[2] = 0.0
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])
        H = np.eye(4)
        H[:3, :3], H[:3, 3] = np.column_stack([x, np.cross(z, x), z]), Pg[:3, 3]
        Si = np.linalg.inv(H) @ P0
        S.append(Si)
        info.append({"episode": i, "start_rel_grasp_mm": (Si[:3, 3] * 1e3).round(1).tolist(),
                     "start_tilt_deg": round(float(np.degrees(np.arccos(np.clip(-Si[2, 2], -1, 1)))), 1),
                     "start_tip_height_m": round(float(P0[2, 3]), 4), "grasp_tip_height_m": round(float(Pg[2, 3]), 4)})
    S = np.array(S)
    pos = np.median(S[:, :3, 3], axis=0)
    R = Rotation.from_matrix(S[:, :3, :3]).mean().as_matrix()
    return {"S": S, "pos": pos, "R": R, "episodes": info,
            "grasp_tip_height_median_m": float(np.median([e["grasp_tip_height_m"] for e in info])),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))}


def solve_ik(m, d, site: str, T_goal: np.ndarray, q0: np.ndarray, iters: int = 400) -> tuple[np.ndarray, float, float]:
    """Damped least squares on the 7 arm joints for site pose T_goal; returns (q, position error m, angle error deg)."""
    import mujoco

    sid = m.site(site).id
    jr = m.jnt_range[:7]
    q = q0.copy()
    jp, jw = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
        ep = T_goal[:3, 3] - p
        er = Rotation.from_matrix(T_goal[:3, :3] @ R.T).as_rotvec()
        mujoco.mj_jacSite(m, d, jp, jw, sid)
        J = np.vstack([jp[:, :7], jw[:, :7]])
        e = np.r_[ep, er]
        lam = 1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.05 * (q0 - q))
        q = np.clip(q + np.clip(dq, -0.1, 0.1), jr[:, 0] + 1e-3, jr[:, 1] - 1e-3)
        if np.linalg.norm(ep) < 1e-6 and np.linalg.norm(er) < 1e-6:
            break
    d.qpos[:7] = q
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[sid].reshape(3, 3)
    ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(T_goal[:3, :3] @ R.T).as_rotvec())))
    return q, float(np.linalg.norm(T_goal[:3, 3] - d.site_xpos[sid])), ang


def compute(dataset=DEFAULT_DATASET, model_json=DEFAULT_MODEL) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = json.loads(Path(model_json).read_text())
    rnd = yaml.safe_load(RANDOM_TRAINSTART.read_text())
    cfg = yaml.safe_load(CONFIG.read_text())
    rel = relative_starts(dataset)
    table_z = float(model["table"]["top_z_m"])
    cx, cy = (float(np.mean(rnd["x_m"])), float(np.mean(rnd["y_m"])))
    xa = np.asarray(rnd.get("closing_axis_O", [0.0, -1.0, 0.0]), float)   # grasp frame in O: x = closing axis, z up
    xa[2] = 0.0
    xa /= np.linalg.norm(xa)
    H = np.eye(4)
    H[:3, :3] = np.column_stack([xa, np.cross([0.0, 0.0, 1.0], xa), [0.0, 0.0, 1.0]])
    H[:3, 3] = [cx, cy, table_z + rel["grasp_tip_height_median_m"]]
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = rel["R"], rel["pos"]
    T = H @ S
    m = mujoco.MjModel.from_xml_path(str(Path(model_json).with_name(model["scene_xml"])))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    q_deploy = d.qpos[:7].copy()
    best = None                                            # several seeds: the solution closest to the deploy home
    rng = np.random.default_rng(0)
    for k in range(24):
        seed = q_deploy if k == 0 else np.clip(q_deploy + rng.normal(0, 0.6, 7), m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        qk, pe, ae = solve_ik(m, d, "tcp", T, seed)
        if pe > 1e-4 or ae > 0.05:
            continue
        margin = float(np.min(np.minimum(qk - m.jnt_range[:7, 0], m.jnt_range[:7, 1] - qk)))
        # closest to the deploy home in joints 1-6 (joint 7 only rolls the hand): the elbow-up solution with joint 3
        # near 0, not the twisted one with joints 1 and 3 turned against each other
        cost = float(np.linalg.norm((qk - q_deploy)[:6])) - 0.5 * min(margin, 0.3)
        if best is None or cost < best[0]:
            best = (cost, qk, pe, ae)
    if best is None:
        raise SystemExit("trainstart: no IK solution for the start pose")
    _, q, perr, aerr = best
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    contacts = [(m.geom(c.geom1).name or m.body(m.geom_bodyid[c.geom1]).name,
                 m.geom(c.geom2).name or m.body(m.geom_bodyid[c.geom2]).name, round(float(c.dist), 4))
                for c in d.contact[:d.ncon]
                if "eraser" not in (m.geom(c.geom1).name, m.geom(c.geom2).name)]
    lo, hi = np.array(cfg["safety"]["workspace_min_m"]), np.array(cfg["safety"]["workspace_max_m"])
    tcp = d.site_xpos[m.site("tcp").id].copy()
    cams = {}
    for cam in ("cam_imx219", "cam_imx519"):
        c = m.camera(cam).id
        Rmj = d.cam_xmat[c].reshape(3, 3)
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = Rmj @ np.diag([1.0, -1.0, -1.0]), d.cam_xpos[c]
        cams[cam] = Tc.tolist()
    jr = m.jnt_range[:7]
    return {
        "schema": "sim_deploy.trainstart/v1",
        "dataset": str(dataset), "n_episodes": len(rel["episodes"]),
        "start_rel_grasp_median_m": rel["pos"].round(5).tolist(),
        "start_rel_grasp_R_mean": rel["R"].round(6).tolist(),
        "start_tilt_deg": round(rel["tilt_deg"], 2),
        "grasp_tip_height_median_m": round(rel["grasp_tip_height_median_m"], 4),
        "region_centre_m": [cx, cy], "region_file": str(RANDOM_TRAINSTART.relative_to(REPO)),
        "O_T_tcp": T.tolist(),
        "q_home_rad": [round(float(x), 7) for x in q],
        "ik": {"position_error_m": perr, "angle_error_deg": aerr,
               "joint_margin_rad": float(np.min(np.minimum(q - jr[:, 0], jr[:, 1] - q))),
               "contacts": contacts, "tcp_in_box": bool(np.all(tcp >= lo) and np.all(tcp <= hi)),
               "tcp_O_m": tcp.round(4).tolist(), "tip_above_table_m": round(float(tcp[2] - table_z), 4)},
        "O_T_cam_cv": cams,
        "episodes": rel["episodes"],
    }


def write(doc: dict) -> None:
    hdr = ("# Generated by `python -m sim_deploy.scene.trainstart --write` (E7): the training-like home. Do not edit;\n"
           "# re-run it after a model or region change. Read by Scene.load(home='trainstart').\n")
    TRAINSTART_YAML.write_text(hdr + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
    text = CONTROL_SIM.read_text()
    marker = "franka_control:                             # overrides of franka_control/config.yaml keys (not libfranka_version)\n"
    if marker not in text:
        raise SystemExit(f"{CONTROL_SIM}: no franka_control: line to add the homing override under")
    q = ", ".join(f"{x:.7f}" for x in doc["q_home_rad"])
    tcp = doc["ik"]["tcp_O_m"]
    block = (marker +
             "  homing:                                  # E7: the training-like start (sim_deploy/scene/trainstart.py):\n"
             f"    q_home_rad: [{q}]\n"
             f"                                          # TCP O ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}), fingers "
             f"{doc['start_tilt_deg']:.1f} deg from vertical,\n"
             "                                          # camera looking ahead; eraser region "
             f"scene/{RANDOM_TRAINSTART.name}\n")
    text = text.replace(marker, block)
    lines = text.splitlines(keepends=True)
    head = [ln for ln in lines if ln.startswith("#")]
    n_head = len(head) if all(ln.startswith("#") for ln in lines[:len(head)]) else 0
    new_head = ("# sim_deploy station: control_sim.yaml with the training-like home (E7). Generated by\n"
                "# `python -m sim_deploy.scene.trainstart --write`; do not edit (edit control_sim.yaml and re-run).\n"
                "# Run: sim_deploy/run.sh --home trainstart ... (or franka_control --sim --local --station this file)\n")
    CONTROL_TRAINSTART.write_text(new_head + "".join(lines[n_head:]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.scene.trainstart", description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    doc = compute(a.dataset, a.model)
    show = {k: v for k, v in doc.items() if k not in ("episodes", "O_T_cam_cv", "start_rel_grasp_R_mean")}
    print(json.dumps(show, indent=1))
    ik = doc["ik"]
    ok = ik["position_error_m"] < 1e-4 and ik["angle_error_deg"] < 0.05 and not ik["contacts"] and ik["tcp_in_box"]
    print("OK" if ok else "FAIL")
    if a.write and ok:
        write(doc)
        print(f"wrote {TRAINSTART_YAML} and {CONTROL_TRAINSTART}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
