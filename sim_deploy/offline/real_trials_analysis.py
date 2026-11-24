"""W1 / S6 questions 3 (attribution) + 4 (predicted vs executed): reads the replays and the logs, writes one summary.

Inputs (all from this step): ``inventory.json``, ``images.json``, ``replay_<TASK>_<run>.json/.npz``,
``train_first/pick1-double-v2/``; the package low-dim (``sim_deploy.scene.zarr_lowdim.load_episode``); the sim model's
mounted-camera poses (``sim_deploy/build/sim_rig/model.json``, read only) and training's ``T_cam_tcp``
(``~/data/umi/rigs/rig_pi_v3.yaml``).

What it computes:

* (a) images: per trial and setting, the first chunk's k = 15 position (TCP frame, cm) and the fingertip's base z
  over k = 0..15 for real / training / swapped cameras, and the difference each camera makes;
* (b) start pose: every demo's own motion from its first row to its grasp (rows of the first closing,
  ``sim_deploy.scene.placement.grasp_rows``), as a pose relative to its first row, applied to each trial's real start
  pose -> where the fingertip would end in the base frame (z vs the table at 0); and the demos' start heights / tilts;
* (c) TCP / camera placement: (1) the lever arm: the chunk's own rotations R_k applied about a point offset by d from
  the TCP move the fingertip by (R_k - I) d, for d = the mounted camera's offset vs training's camera -> TCP
  (sim model) and d = (0, 0, 11.8 mm) (E3: fingertips further out than the rig study assumed); (2) the visual
  offset: a camera pose learned in training puts the robot's fingertip at T_cam_tcp(mount) instead of
  T_cam_tcp(training): that difference in the base frame at the start orientation;
* (d) width: real images with model width 0.090 (B, as ran) vs 0.040 (A) vs 0.087 (training);
* question 4: from manager.log clamp lines, the commanded descent within a chunk (z_k11 - z_k0 where k0 is clamped:
  the anchor already at the floor) vs the offline first-chunk descent; waypoints accepted/skipped/shortened per chunk.

Run (sim_deploy env for the package reader; CPU, seconds)::

    cd ~/Capstone-main-sim && unset PYTHONPATH && ~/envs/sim_deploy/bin/python -m sim_deploy.offline.real_trials_analysis

Output: ``sim_evals/_pipeline/W1/real_trials/analysis.json`` and ``analysis.md`` (tables, also printed).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from sim_deploy.offline.real_trials_common import OUT, REPO, clamps, copy_version, write_json

TF = OUT / "train_first" / "pick1-double-v2"
DATASET = Path("~/data/umi/datasets/multi/pick1-double-v2/pick1-double-v2.zarr.zip").expanduser()
RUN_DATA = {"PICK_TEST1": "PICK_TEST1", "PICK_DOUBLE": None}


def rot6d_to_R(r6):
    """behavior_prompting convention (rows a1, a2 -> Gram-Schmidt, b3 = b1 x b2; columns), as the copy's pose_math."""
    sys.path.insert(0, str(REPO / "deployment/policy_deploy"))
    from policy_deploy.manager import pose_math
    return pose_math.matrix_from_rot6d(np.asarray(r6, dtype=np.float64))


def tcp_offsets() -> dict:
    m = json.loads((REPO / "sim_deploy/build/sim_rig/model.json").read_text())
    O_tcp = np.array(m["home"]["O_T_tcp"])
    rig = yaml.safe_load(Path("~/data/umi/rigs/rig_pi_v3.yaml").expanduser().read_text())
    T_cam_tcp_tr = np.array(rig["tcp"]["T_cam_tcp"]["matrix"])
    O_cam = np.array(m["cameras"]["cam_imx219"]["O_T_cam_cv_home"])
    T_cam_tcp_mt = np.linalg.inv(O_cam) @ O_tcp
    # camera position in the TCP frame: training vs mount
    c_tr = np.linalg.inv(T_cam_tcp_tr)[:3, 3]
    c_mt = np.linalg.inv(T_cam_tcp_mt)[:3, 3]
    return {"T_cam_tcp_training": T_cam_tcp_tr, "T_cam_tcp_mount_sim": T_cam_tcp_mt,
            "cam_in_tcp_training_mm": (c_tr * 1000).round(1), "cam_in_tcp_mount_mm": (c_mt * 1000).round(1),
            "d_tcp_mm": ((c_mt - c_tr) * 1000).round(1),
            "fingertip_in_cam_training_mm": (T_cam_tcp_tr[:3, 3] * 1000).round(1),
            "fingertip_in_cam_mount_mm": (T_cam_tcp_mt[:3, 3] * 1000).round(1),
            "rot_diff_deg": round(math.degrees(math.acos(np.clip((np.trace(T_cam_tcp_tr[:3, :3].T @ T_cam_tcp_mt[:3, :3])
                                                                  - 1) / 2, -1, 1))), 2)}


def demo_motion_from_real_start(starts: dict) -> dict:
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode
    idx = json.loads((TF / "index.json").read_text())["episodes"]
    rows = []
    for e in idx:
        ep = load_episode(DATASET, e["index"])
        P, w = ep["poses"], ep["widths"]
        try:
            g = grasp_rows(w)
        except ValueError:
            continue
        gs = g["start"]
        rel = np.linalg.inv(P[0]) @ P[gs]
        zmin_to_grasp = float(P[: g["end"] + 1, 2, 3].min())
        r = {"index": e["index"], "folder": e["folder"], "start_z_m": float(P[0][2, 3]),
             "start_tilt_deg": round(math.degrees(math.acos(np.clip(-P[0][2, 2], -1, 1))), 1),
             "grasp_start_row": int(gs), "grasp_z_m": float(P[gs][2, 3]), "min_z_to_grasp_end_m": zmin_to_grasp,
             "descent_m": float(P[0][2, 3] - P[gs][2, 3]), "rel_grasp_tcp_cm": (rel[:3, 3] * 100).round(2).tolist()}
        for name, S in starts.items():
            r[f"grasp_z_from_{name}"] = float((S @ rel)[2, 3])
        rows.append(r)

    def stats(key, sel=None):
        v = np.array([r[key] for r in rows if sel is None or r["folder"] == sel])
        return {"median": round(float(np.median(v)), 4), "p10": round(float(np.percentile(v, 10)), 4),
                "p90": round(float(np.percentile(v, 90)), 4), "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4), "n": int(len(v)),
                "below_0": int((v < 0).sum()), "below_-0.003": int((v < -0.003).sum())}
    summ = {}
    for sel in ("PICK_TEST1", "PICK_TEST2", None):
        k = sel or "both"
        summ[k] = {key: stats(key, sel) for key in ["start_z_m", "start_tilt_deg", "grasp_z_m", "descent_m"] +
                   [f"grasp_z_from_{n}" for n in starts]}
    return {"per_demo": rows, "summary": summ}


def main() -> int:
    inv = json.loads((OUT / "inventory.json").read_text())
    starts = {}
    for t in inv["trials"]:
        S = np.asarray(t["episode"]["start_pose_4x4"])
        key = "real_start"
        if key in starts and not np.allclose(starts[key], S, atol=2e-3):
            key = f"real_start_{t['label']}"
        starts.setdefault(key, S)
    offs = tcp_offsets()
    d_mt = offs["d_tcp_mm"] / 1000.0
    d_e3 = np.array([0.0, 0.0, 0.0118])
    S = starts["real_start"]
    # (c2) visual offset at the start orientation: fingertip(mount) - fingertip(training), same camera pose
    R_tcp = S[:3, :3]
    R_cam = R_tcp @ np.linalg.inv(offs["T_cam_tcp_training"])[:3, :3]
    df_cam = (offs["T_cam_tcp_mount_sim"][:3, 3] - offs["T_cam_tcp_training"][:3, 3])
    visual_offset_base = R_cam @ df_cam
    # (b)
    demo = demo_motion_from_real_start(starts)

    out = {"copy_version": copy_version(), "tcp_offsets": offs,
           "visual_offset": {"fingertip_mount_minus_training_cam_mm": (df_cam * 1000).round(1),
                             "in_base_at_start_mm": (visual_offset_base * 1000).round(1),
                             "note": "same camera pose, two camera->fingertip transforms (sim mount model vs "
                                     "rig_pi_v3 T_cam_tcp): where the robot's fingertip is vs where training's "
                                     "would be"},
           "demo_motion_from_real_start": demo, "trials": {}}
    md = []
    reps = sorted(OUT.glob("replay_*.json"))
    for rp in reps:
        rj = json.loads(rp.read_text())
        raw = np.load(rp.with_suffix(".npz"))
        res = rj["results"]
        by = {(r["setting"], r["group"], r["case"]): r for r in res}
        settings = [l["setting"] for l in rj["settings_log"]]
        groups = sorted({r["group"] for r in res if not r["group"].startswith("train_ep")})
        for g in groups:
            tr = out["trials"].setdefault(g, {"run": rj["run"], "settings": {}})
            for s in settings:
                row = {}
                for case in ("real", "real_wA", "real_w087", "train", "realM_trainU", "trainM_realU"):
                    r = by.get((s, g, case))
                    if r is None:
                        continue
                    key = f"{s}|{g}|{case}"
                    pos, rot, Tab = raw[key + "|pos"], raw[key + "|rot6d"], raw[key + "|T_abs"]
                    lever_mt = []
                    lever_e3 = []
                    rot_deg = []
                    for si in range(pos.shape[0]):
                        Rk = np.stack([rot6d_to_R(rot[si, k]) for k in range(rot.shape[1])])
                        lever_mt.append(np.linalg.norm((Rk - np.eye(3)) @ d_mt, axis=1).max())
                        lever_e3.append(np.linalg.norm((Rk - np.eye(3)) @ d_e3, axis=1).max())
                        rot_deg.append(max(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))) for R in Rk))
                        # base-z effect of the lever with the start orientation
                    z_abs = Tab[:, :, 2, 3]
                    row[case] = {"k15_tcp_cm": r["k15_tcp_cm_mean"], "k15_tcp_cm_std": r["k15_tcp_cm_std"],
                                 "k15_z_tcp_cm_per_seed": r["k15_z_cm_per_seed"],
                                 "fingertip_z_k15_m": round(float(z_abs[:, -1].mean()), 4),
                                 "fingertip_z_k11_m": round(float(z_abs[:, 11].mean()), 4),
                                 "fingertip_z_min_m": round(float(z_abs.min()), 4),
                                 "base_dz_k15_cm": round(float((z_abs[:, -1] - r["anchor_z_m"]).mean() * 100), 2),
                                 "base_dz_k15_cm_per_seed": ((z_abs[:, -1] - r["anchor_z_m"]) * 100).round(2).tolist(),
                                 "width_robot_k15": r["width_robot_k0_k15_mean"][-1],
                                 "width_robot_k0": r["width_robot_k0_k15_mean"][0],
                                 "max_rot_deg": round(float(np.mean(rot_deg)), 2),
                                 "lever_mount_mm": round(float(np.mean(lever_mt)) * 1000, 2),
                                 "lever_e3_mm": round(float(np.mean(lever_e3)) * 1000, 2)}
                post = sorted([r for r in res if r["group"] == g and r["case"].startswith("post_") and r["setting"] == s],
                              key=lambda r: r["t_after_start_s"])
                row["post"] = [{"t_s": r["t_after_start_s"], "k15_tcp_cm": r["k15_tcp_cm_mean"],
                                "base_dz_k15_cm": round((r["abs_z_m_k0_k15_mean"][-1] - r["anchor_z_m"]) * 100, 2),
                                "width_robot_k15": r["width_robot_k0_k15_mean"][-1]} for r in post]
                if "real" in row and "train" in row:
                    b = lambda c: row[c]["base_dz_k15_cm"] if c in row else None  # noqa: E731
                    row["attribution_base_dz_k15_cm"] = {
                        "train_both": b("train"), "real_both": b("real"),
                        "IMX519_only_real (realM_trainU)": b("realM_trainU"),
                        "IMX219_only_real (trainM_realU)": b("trainM_realU"),
                        "effect_IMX519": None if b("realM_trainU") is None else round(b("realM_trainU") - b("train"), 2),
                        "effect_IMX219": None if b("trainM_realU") is None else round(b("trainM_realU") - b("train"), 2),
                        "effect_both": round(b("real") - b("train"), 2),
                        "width_A_minus_B": None if b("real_wA") is None else round(b("real_wA") - b("real"), 2),
                        "width_087_minus_090": None if b("real_w087") is None else round(b("real_w087") - b("real"), 2)}
                tr["settings"][s] = row
        # training baselines
        for s in settings:
            bl = []
            for g, gt in rj["ground_truth"].items():
                o, st_ = by.get((s, g, "own")), by.get((s, g, "stat"))
                if o is None:
                    continue
                gz = gt["gt_abs_z_m_k0_k15"]
                bl.append({"group": g, "gt_k15_tcp_cm": gt["gt_k15_tcp_cm"],
                           "gt_base_dz_k15_cm": round((gz[-1] - gz[0]) * 100, 2),
                           "own_k15_tcp_cm": o["k15_tcp_cm_mean"],
                           "own_base_dz_k15_cm": round((o["abs_z_m_k0_k15_mean"][-1] - o["anchor_z_m"]) * 100, 2),
                           "stat_k15_tcp_cm": st_["k15_tcp_cm_mean"],
                           "stat_base_dz_k15_cm": round((st_["abs_z_m_k0_k15_mean"][-1] - st_["anchor_z_m"]) * 100, 2),
                           "start_z_m": gz[0], "own_width_robot_k15": o["width_robot_k0_k15_mean"][-1]})
            out.setdefault("baselines", {}).setdefault(Path(rj["run"]).name, {})[s] = bl

    # question 4: commanded descent within clamped chunks
    q4 = {}
    for t in inv["trials"]:
        cl = clamps(t["session"])
        rows = [{k: v for k, ax, v, _ in c["waypoints"] if ax == "z"} for c in cl]
        full = [z for z in rows if 0 in z and 11 in z]
        dz = [z[11] - z[0] for z in full]
        e = t["episode"]
        q4[t["label"]] = {"clamped_chunks": len(rows), "chunks_k0_clamped": sum(0 in z for z in rows),
                          "dz_k11_minus_k0_mm": {"n": len(dz), "mean": round(float(np.mean(dz)) * 1000, 1) if dz else None,
                                                 "min": round(float(np.min(dz)) * 1000, 1) if dz else None},
                          "min_commanded_z_m": min((min(z.values()) for z in rows), default=None),
                          "per_chunk_accepted_mean": e["per_chunk"]["accepted_mean"],
                          "per_chunk_skipped_mean": e["per_chunk"]["skipped_mean"],
                          "shortened_total": e["waypoints"]["shortened"], "chunks_sent": e["chunks_sent"],
                          "outcome": e["outcome"], "reason": e["reason"]}
    out["executed"] = q4
    write_json(OUT / "analysis.json", out)
    md.append(report(out))
    (OUT / "analysis.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


def report(o) -> str:
    L = []
    L.append("## (c) camera -> TCP\n")
    t = o["tcp_offsets"]
    L.append(f"camera in TCP frame (mm): training {t['cam_in_tcp_training_mm'].tolist()}, sim mount "
             f"{t['cam_in_tcp_mount_mm'].tolist()}, d {t['d_tcp_mm'].tolist()}; rotation diff {t['rot_diff_deg']} deg")
    L.append(f"fingertip in camera (mm): training {t['fingertip_in_cam_training_mm'].tolist()}, mount "
             f"{t['fingertip_in_cam_mount_mm'].tolist()}; visual offset in base at the start "
             f"{o['visual_offset']['in_base_at_start_mm'].tolist()} mm\n")
    L.append("## (b) demos' own motion to the grasp, applied from the real start\n")
    for k, s in o["demo_motion_from_real_start"]["summary"].items():
        L.append(f"{k}: " + "; ".join(f"{kk} med {v['median']} [p10 {v['p10']}, p90 {v['p90']}] min {v['min']} "
                                       f"(<0: {v['below_0']}/{v['n']})" for kk, v in s.items()))
    L.append("\n## first chunk (base dz k15 cm; k15 TCP cm)\n")
    L.append("| trial | setting | real | real wA | real w087 | train | realM+trainU | trainM+realU | real k15 TCP cm | "
             "train k15 TCP cm | rot deg | lever mount / E3 mm |\n|" + "---|" * 12)
    for g, tr in o["trials"].items():
        for s, row in tr["settings"].items():
            f = lambda c: str(row[c]["base_dz_k15_cm"]) if c in row else "-"  # noqa: E731
            r = row.get("real", {})
            L.append(f"| {g} | {s} | {f('real')} | {f('real_wA')} | {f('real_w087')} | {f('train')} | "
                     f"{f('realM_trainU')} | {f('trainM_realU')} | {r.get('k15_tcp_cm')} | "
                     f"{row.get('train', {}).get('k15_tcp_cm')} | {r.get('max_rot_deg')} | "
                     f"{r.get('lever_mount_mm')} / {r.get('lever_e3_mm')} |")
    L.append("\n## training baselines (base dz k15 cm: GT / own low-dim / stationary)\n")
    for run, d in o.get("baselines", {}).items():
        for s, bl in d.items():
            L.append(f"{run} {s}: " + "; ".join(f"{b['group']} {b['gt_base_dz_k15_cm']}/{b['own_base_dz_k15_cm']}/"
                                                 f"{b['stat_base_dz_k15_cm']}" for b in bl))
    L.append("\n## executed (clamp lines)\n")
    for k, v in o["executed"].items():
        L.append(f"{k}: {v}")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
