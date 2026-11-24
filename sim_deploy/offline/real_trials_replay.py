"""W1 / S6 question 3: offline replay of the real robot's policy inputs through the deploy path, vs training's.

For ONE run folder (a checkpoint that ran on the arm on 2026-11-20): the policy is built the deploy way (a scratch
copy of ``sim_deploy/stations/policy_sim.yaml``, ``policy_deploy use RUN_FOLDER --station <copy>``,
``Settings.load(copy, mock=True)`` + ``build_policy``, as ``sim_evals/_start_pose_sweep/tools/offline_first_chunk.py``;
prompting: ``reset()`` + ``prompt(task_name, prompt_index)`` = the session's PICK_TEST1 / 0 before every sample), in
two settings: as ran (bf16, 8 diffusion steps) and fp32 with the checkpoint's own steps. The low-dim is what the
manager sends at a stationary start: the copy's ``PoseConverter.to_observation`` on the trial's ``start_pose``
(episodes.jsonl) at both observation times (eef_pos 0, rot6d identity, wrt_start identity), width = the robot's open
reading 0.040 m through ``policy.gripper_map`` exactly as the session's policy (startup.json: B, 0.040 -> 0.090).

Cases per trial (5 seeds each, ``torch.manual_seed(1000 + s)`` before every infer):

* ``real``            the real probe set written just before the episode start (both cameras);
* ``real_wA``         the same, convention A (gripper_map off: the model gets 0.040 as is; the policy's own map is
                      switched off on the object for this case only, ``_gripper_map = None``);
* ``real_w087``       the same, model width 0.087 (training's first-frame median) via robot width 0.03923 under B;
* ``train``           the nearest training first frame (images.json), both cameras, the same stationary low-dim;
* ``realM_trainU``    real IMX519 + training IMX219;  ``trainM_realU``  training IMX519 + real IMX219;
* ``post_<t>``        real probe sets 0.5 .. 3 s after the start (the arm moving: the stationary low-dim is an
                      approximation), as ran only; from 1.5 s also ``post_<t>_closed`` with the robot's
                      closed-on-nothing reading 0.0117 m (session 17-16-13's policy.log; B maps it to 0.051 =
                      "closed on the eraser").

Training baseline (8 demos of the checkpoint's data): the first sample with its own recorded low-dim (``own``) and
with the stationary low-dim + width B (``stat``), vs the demo's ground truth (train_first ``gt_rel`` / ``gt_abs``).

Absolute path: the copy's ``PoseConverter.to_waypoints`` (exec_h 16, no workspace box) with the anchor = the trial's
start_pose (base frame) -> fingertip (TCP) xyz at k = 0..15; for training: the anchor = the demo's row-6 pose (tag
frame, z = above the table). Raw chunks go to ``replay_<run>.npz``, summaries to ``replay_<run>.json``.

Run (torch env, GPU, hold the shared lock for the whole process)::

    cd ~/Capstone-main-sim/deployment/policy_deploy && unset PYTHONPATH && \
    GPU_LOCK_HELD=1 flock ~/Capstone-main-sim/sim_evals/.gpu.lock \
      ~/envs/sim_policy_deploy_torch/bin/python ~/Capstone-main-sim/sim_deploy/offline/real_trials_replay.py \
      ~/Capstone/data/PICK_DOUBLE/policy/run_002_2026-11-19 [--seeds 5] [--device cuda]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path.home() / "Capstone-main-sim"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deployment/policy_deploy"))

from sim_deploy.offline.real_trials_common import OUT, SCRATCH, copy_version, jsonable, set_paths, set_times  # noqa

STATION_TEMPLATE = REPO / "sim_deploy/stations/policy_sim.yaml"
USE_CLI = Path.home() / "envs/sim_policy_deploy/bin/policy_deploy"
ROBOT_OPEN_M = 0.040
W087_ROBOT = 0.030 + (0.087 - 0.051) * (0.040 - 0.030) / (0.090 - 0.051)   # -> model 0.087 under B
ROBOT_CLOSED_EMPTY_M = 0.0117      # session 17-16-13: the robot's reading with the fingers closed on nothing
POST_T = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def make_station(run: Path) -> Path:
    d = SCRATCH / "stations" / run.name
    d.mkdir(parents=True, exist_ok=True)
    st = d / "station.yaml"
    shutil.copyfile(STATION_TEMPLATE, st)
    r = subprocess.run([str(USE_CLI), "use", str(run), "--station", str(st)], capture_output=True, text=True,
                       cwd=str(REPO / "deployment/policy_deploy"), env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    (d / "use.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"policy_deploy use failed: {r.stdout}{r.stderr}")
    return st


def rgb(p):
    import cv2
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def load_set(folder, i):
    p = set_paths(folder, i)
    return {"main": np.stack([rgb(p["main_0"]), rgb(p["main_1"])]),
            "ultrawide": np.stack([rgb(p["ultrawide_0"]), rgb(p["ultrawide_1"])])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--settings", default="bf16:8,fp32:ckpt")
    ap.add_argument("--n-baseline", type=int, default=8)
    a = ap.parse_args()
    run = a.run.expanduser().resolve()

    import cv2
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    from policy_deploy.manager.pose_converter import PoseConverter
    from policy_deploy.contracts.c4_policy import Observation
    from policy_deploy.policy.policy_factory import build_policy
    from policy_deploy.policy.settings import Settings

    inv = json.loads((OUT / "inventory.json").read_text())
    imgs = json.loads((OUT / "images.json").read_text())
    trials = [t for t in inv["trials"] if t["run"] == run.name]
    img_by_label = {t["label"]: t for t in imgs["trials"]}
    tf = OUT / "train_first" / imgs["package"]
    tf_index = json.loads((tf / "index.json").read_text())["episodes"]
    folder = run.parts[-3]
    pool = [e["index"] for e in tf_index if folder == "PICK_DOUBLE" or e["folder"] == folder]
    base_idx = [pool[int(round(j))] for j in np.linspace(0, len(pool) - 1, a.n_baseline)]

    station = make_station(run)
    st0 = Settings.load(station, mock=True)
    conv = PoseConverter(exec_h=16, lookahead_ms=0, workspace_box=None)

    def stationary_obs(start_pose, robot_w, images):
        P = np.stack([start_pose, start_pose])
        obs, anchor = conv.to_observation(P, np.array([robot_w, robot_w]), start_pose,
                                          SimpleNamespace(main=images["main"], ultrawide=images["ultrawide"]),
                                          0, 0, "target")
        return obs, anchor

    def absolute(chunk, anchor):
        wp = conv.to_waypoints(chunk, anchor, 1)
        return np.asarray(wp.O_T_TCP)

    cases = []        # (group, name, obs, anchor_T, meta)
    for t in trials:
        start = np.asarray(t["episode"]["start_pose_4x4"], dtype=np.float64)
        it = img_by_label[t["label"]]
        i0 = t["probe"]["set_at_start"]
        real = load_set(t["folder"], i0)
        near = np.load(tf / f"ep{it['nearest_training']['index']:02d}.npz")
        trn = {"main": near["main"], "ultrawide": near["ultrawide"]}
        g = t["label"]
        for name, ims, w, gm_off in (("real", real, ROBOT_OPEN_M, False), ("real_wA", real, ROBOT_OPEN_M, True),
                                     ("real_w087", real, W087_ROBOT, False), ("train", trn, ROBOT_OPEN_M, False),
                                     ("realM_trainU", {"main": real["main"], "ultrawide": trn["ultrawide"]},
                                      ROBOT_OPEN_M, False),
                                     ("trainM_realU", {"main": trn["main"], "ultrawide": real["ultrawide"]},
                                      ROBOT_OPEN_M, False)):
            obs, anc = stationary_obs(start, w, ims)
            cases.append((g, name, obs, anc, {"gm_off": gm_off, "set": i0, "robot_w": w}))
        times = set_times(t["folder"])
        ws = t["episode"]["wall_start_epoch"]
        for dt in POST_T:
            i = int(np.argmin(np.abs(times - (ws + dt))))
            if abs(times[i] - ws - dt) < 0.2 and times[i] <= t["probe"]["t_last"]:
                obs, anc = stationary_obs(start, ROBOT_OPEN_M, load_set(t["folder"], i))
                cases.append((g, f"post_{dt:.1f}", obs, anc, {"gm_off": False, "set": i, "robot_w": ROBOT_OPEN_M,
                                                              "bf16_only": True,
                                                              "t_after_start_s": round(float(times[i] - ws), 2)}))
                if dt >= 1.5:      # the fingers had closed by then in most trials: the robot's closed reading
                    obs, anc = stationary_obs(start, ROBOT_CLOSED_EMPTY_M, load_set(t["folder"], i))
                    cases.append((g, f"post_{dt:.1f}_closed", obs, anc, {
                        "gm_off": False, "set": i, "robot_w": ROBOT_CLOSED_EMPTY_M, "bf16_only": True,
                        "t_after_start_s": round(float(times[i] - ws), 2)}))
    for j in base_idx:
        z = np.load(tf / f"ep{j:02d}.npz")
        T6 = np.asarray(z["gt_abs"][0], dtype=np.float64)
        own = Observation(**{k: z[k] for k in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m",
                                               "main", "ultrawide")})
        _, anc_own = stationary_obs(T6, ROBOT_OPEN_M, {"main": z["main"], "ultrawide": z["ultrawide"]})
        cases.append((f"train_ep{j:02d}", "own", own, anc_own, {"gm_off": True, "model_w": float(z["gripper_width_m"][1, 0]),
                                                               "note": "recorded low-dim and width (training units)"}))
        obs, anc = stationary_obs(T6, ROBOT_OPEN_M, {"main": z["main"], "ultrawide": z["ultrawide"]})
        cases.append((f"train_ep{j:02d}", "stat", obs, anc, {"gm_off": False, "robot_w": ROBOT_OPEN_M}))

    results, raw = [], {}
    log = []
    for spec in a.settings.split(","):
        prec, steps = spec.split(":")
        st = dataclasses.replace(st0, device=a.device, precision=prec,
                                 diffusion_steps=None if steps == "ckpt" else int(steps))
        t0 = time.monotonic()
        pol = build_policy(st)
        load_s = time.monotonic() - t0
        prompting = hasattr(pol, "prompt")
        task = "PICK_TEST1"
        pidx = 0
        gm = pol._gripper_map
        desc = pol.describe()
        log.append({"setting": spec, "load_s": round(load_s, 2), "describe": {k: desc.get(k) for k in (
            "kind", "epoch", "diffusion_steps_checkpoint", "diffusion_steps_in_use", "precision", "gripper_map",
            "model_target")}, "prompting": prompting})
        print(spec, log[-1], flush=True)
        for g, name, obs, anc, meta in cases:
            if meta.get("bf16_only") and prec != "bf16":
                continue
            outs = []
            for s in range(a.seeds):
                pol.reset()
                if prompting:
                    pol.prompt(task, pidx)
                torch.manual_seed(1000 + s)
                pol._gripper_map = None if meta["gm_off"] else gm
                try:
                    ch = pol.infer(obs)
                finally:
                    pol._gripper_map = gm
                T_abs = absolute(ch, anc)
                outs.append((np.asarray(ch.eef_pos), np.asarray(ch.eef_rot6d), np.asarray(ch.gripper_width_m)[:, 0],
                             T_abs))
            pos = np.stack([o[0] for o in outs])
            rot = np.stack([o[1] for o in outs])
            wid = np.stack([o[2] for o in outs])
            Tab = np.stack([o[3] for o in outs])
            key = f"{spec}|{g}|{name}"
            raw[key + "|pos"], raw[key + "|rot6d"], raw[key + "|width"], raw[key + "|T_abs"] = pos, rot, wid, Tab
            k15 = pos[:, -1] * 100
            results.append({"setting": spec, "group": g, "case": name, **{k: v for k, v in meta.items()},
                            "k15_tcp_cm_mean": k15.mean(0).round(2).tolist(), "k15_tcp_cm_std": k15.std(0).round(2).tolist(),
                            "k15_z_cm_per_seed": k15[:, 2].round(2).tolist(),
                            "abs_z_m_k0_k15_mean": Tab[:, :, 2, 3].mean(0).round(4).tolist(),
                            "abs_z_m_min_over_seeds": float(Tab[:, :, 2, 3].min()),
                            "abs_xyz_k15_mean": Tab[:, -1, :3, 3].mean(0).round(4).tolist(),
                            "anchor_z_m": float(anc.T[2, 3]),
                            "width_robot_k0_k15_mean": wid.mean(0).round(4).tolist()})
            r = results[-1]
            print(f"{spec:9s} {g:28s} {name:13s} k15 cm {r['k15_tcp_cm_mean']} z/seed {r['k15_z_cm_per_seed']} "
                  f"abs z k15 {r['abs_z_m_k0_k15_mean'][-1]:+.3f} (anchor {r['anchor_z_m']:+.3f}) "
                  f"w k15 {r['width_robot_k0_k15_mean'][-1]:.4f}", flush=True)
        del pol
        if a.device == "cuda":
            mem = torch.cuda.max_memory_allocated() / 2**20
            log[-1]["max_memory_allocated_mb"] = round(mem, 1)
            torch.cuda.empty_cache()
    gt = {}
    for j in base_idx:
        z = np.load(tf / f"ep{j:02d}.npz")
        gt[f"train_ep{j:02d}"] = {"gt_k15_tcp_cm": (z["gt_rel"][-1, :3, 3] * 100).round(2).tolist(),
                                  "gt_abs_z_m_k0_k15": z["gt_abs"][:, 2, 3].round(4).tolist(),
                                  "start_row0_z_m": float(z["start_abs"][2, 3])}
    name = run.parts[-3] + "_" + run.name
    np.savez_compressed(OUT / f"replay_{name}.npz", **raw)
    (OUT / f"replay_{name}.json").write_text(json.dumps(jsonable({
        "run": str(run), "station": str(station), "copy_version": copy_version(), "settings_log": log,
        "seeds": [1000 + s for s in range(a.seeds)], "robot_open_m": ROBOT_OPEN_M, "w087_robot_m": W087_ROBOT,
        "prompt": {"task_name": "PICK_TEST1", "prompt_index": 0} if any(l["prompting"] for l in log) else None,
        "baseline_indices": base_idx, "ground_truth": gt, "results": results}), indent=1))
    print("wrote", OUT / f"replay_{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
