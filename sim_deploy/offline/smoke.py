"""W1 step 1 smoke: the harness end to end on a few samples, plus its cross-checks. Torch env
(``~/envs/sim_policy_deploy_torch/bin/python``); the deploy stage and each checkpoint run in their own subprocesses
(one checkpoint per process, each under the GPU lock).

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.offline.smoke [--skip-policies]

Samples: pick1-v1 episodes 14 (= episode_000152, train) and 23 (val in both runs' split) and pick1-double-v2 episode
40 (day 2, PICK_TEST2, val for run_002), rows {6, 9, 36, 72} (6/36/72: a 10 Hz ultrawide frame is newest at T; 9:
not) plus each episode's grasp rows (grasp_rows start / end: the closing, for the width conventions). Checkpoints:
run_006 (prompting) and run_022 (plain) on pick1-v1, run_002 (plain) on pick1-double-v2; fp32, the checkpoint's own
diffusion steps, seeds 0 and 1. Cross-checks: the direct path with run_006's cfg (task sampling) = with run_022's
(episode sampling); the direct path at ep14 row 6 vs camera_study B1's replica (``obs.npz``) and the deploy images
vs B1's deploy images (no JPEG, ``deploy_obs.npz``). Output: ``sim_evals/_pipeline/W1/s1_smoke/`` (summary.json,
per-run json/npz, montage_ep14_r006.png).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import (CAPSTONE_DATA, ENV_TORCH, OUT, REPO, clean_env, deploy_version, read_json,
                                       write_json)

SMOKE = OUT / "s1_smoke"
B1 = REPO / "sim_evals/_camera_study/B/reference/ep14_first"
ROWS = [6, 9, 36, 72]
PLAN = {"pick1-v1": [14, 23], "pick1-double-v2": [40]}
RUNS = {"pick1-v1": ["PICK_TEST1/policy/run_006_2026-11-18", "PICK_TEST1/policy/run_022_2026-11-19"],
        "pick1-double-v2": ["PICK_DOUBLE/policy/run_002_2026-11-19"]}


def grasp_rows_of(dataset: str, index: int) -> list[int]:
    from sim_deploy.offline.episodes import dataset_path
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode
    g = grasp_rows(load_episode(dataset_path(dataset), index)["widths"])
    return [int(g["start"]), int(g["end"])]


def cross_checks() -> dict:
    from sim_deploy.offline.build_obs import direct_dataset, load_sample
    from sim_deploy.offline.compare import image_metrics
    out = {}
    z, meta = load_sample("pick1-v1", 14, 6)
    dd6 = direct_dataset("pick1-v1", CAPSTONE_DATA / RUNS["pick1-v1"][0])
    s6 = dd6.sample(14, 6)
    out["direct_task_sampling_vs_episode_sampling"] = {
        k: float(np.max(np.abs(s6["obs"][k].astype(np.float64) - z[f"direct_{k}"].astype(np.float64))))
        for k in s6["obs"]} | {"action": float(np.max(np.abs(s6["action"] - z["action"]))),
                               "split_task_cfg": s6["split_in_dataset"]}
    b1 = dict(np.load(B1 / "obs.npz"))
    b1d = dict(np.load(B1 / "deploy_obs.npz"))
    out["direct_vs_B1_replica"] = {k: float(np.max(np.abs(b1[k].astype(np.float64) - z[f"direct_{k}"].astype(np.float64))))
                                   for k in ("main", "ultrawide", "eef_pos", "eef_rot6d", "gripper_width_m")}
    out["direct_vs_B1_replica"]["eef_rot6d_wrt_start_vs_clean"] = float(np.max(np.abs(
        b1["eef_rot6d_wrt_start"] - z["direct_wrt_start_clean"])))
    out["deploy_vs_B1_deploy_no_jpeg"] = {f"{cam}_{k}": image_metrics(z[f"deploy_{cam}"][k], b1d[cam][k])
                                          for cam in ("main", "ultrawide") for k in range(2)}
    out["deploy_vs_B1_deploy_no_jpeg"]["note"] = (
        "B1 deploy images: the training-map frames (rows T-3, T; 10 Hz ultrawide), INTER_AREA 640, no JPEG; here: "
        "the selector's frames through JPEG q90. Same frame -> JPEG noise only; main_0 differs in frame at row 6")
    out["b1_frames"] = read_json(B1 / "meta.json")["raw"]["main_frame_idx"], read_json(B1 / "meta.json")["raw"]["uw_frame_idx"]
    return out


def main(argv=None) -> int:
    from sim_deploy.offline.build_obs import build_obs, load_sample
    from sim_deploy.offline.compare import montage
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-policies", action="store_true")
    a = ap.parse_args(argv)
    SMOKE.mkdir(parents=True, exist_ok=True)
    samples: dict[str, list[tuple[str, int, int]]] = {}
    for ds, idx in PLAN.items():
        for i in idx:
            rows = sorted(set(ROWS + grasp_rows_of(ds, i)))
            build_obs(ds, [i], rows)
            samples.setdefault(ds, []).extend((ds, i, r) for r in rows)
    z, _ = load_sample("pick1-v1", 14, 6)
    montage(z, SMOKE / "montage_ep14_r006.png", "pick1-v1 ep14 (episode_000152) row 6: main T-50ms, main T, uw T-100ms, uw T")
    summary = {"samples": {ds: [f"{i}:{r}" for _, i, r in s] for ds, s in samples.items()},
               "cross_checks": cross_checks(), "deploy_version": deploy_version(), "inputs": {}}
    for ds, s in samples.items():
        for _, i, r in s:
            summary["inputs"][f"{ds}:{i}:{r}"] = load_sample(ds, i, r)[1]["inputs"]
    write_json(SMOKE / "summary.json", summary)
    if a.skip_policies:
        return 0
    for ds, runs in RUNS.items():
        for run in runs:
            cmd = [str(ENV_TORCH), "-m", "sim_deploy.offline.policy_runner", "--run", str(CAPSTONE_DATA / run),
                   "--seeds", "0", "1", "--out", str(SMOKE), "--samples", *[f"{d}:{i}:{r}" for d, i, r in samples[ds]]]
            print("[smoke]", " ".join(cmd[:6]), "...", flush=True)
            subprocess.run(cmd, cwd=str(REPO), env=clean_env(), check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
