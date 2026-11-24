"""W1 / S6: training's first observation of every demo in a package, plus the demo's own first 16 action steps.

Per episode of the package (default ``~/data/umi/datasets/multi/pick1-double-v2`` = PICK_TEST1's 30 + PICK_TEST2's 32
demos, the superset of every checkpoint's data): the training sample at row 6 (the first row with two distinct
10 Hz IMX219 frames, B1's "first"), built with B1's own replica of training's sampler and conversion
(``sim_deploy.camera_study.reference._sample_obs`` / ``_convert_lowdim``: main rows T-3, T; IMX219 the two newest
10 Hz frames; relative low-dim; width as recorded), and the ground truth the demo did next: action steps k = 0..15
(rows T + 3k, 20 Hz, task/umi.yaml action_horizon 16, down_sample_steps 3) as the pose relative to row T (TCP frame)
and as the absolute TCP pose in the package's table (tag) frame (z = height above the tag plane).

Check: package index 14 (episode_000152) is compared with B1's ``ep14_first/obs.npz`` (same rows, same code).

Run (torch env for zarr + JPEG-XL codecs; CPU only, ~1 min)::

    cd ~/Capstone-main-sim && unset PYTHONPATH && \
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.offline.real_trials_train_ref [DATASET.zarr.zip]

Output: ``sim_evals/_pipeline/W1/real_trials/train_first/<package>/ep<ii>.npz`` + ``index.json``.

Second mode ``tagwidth`` (same env): the finger tags (ArUco 0/1, ``real_trials_common.TagDetector``) in every 10 Hz
IMX219 frame of the first 6 s of every episode, with the recorded width at that frame's row, so the tag position /
size can be read at any width (the robot opens less than the handheld)::

    ... -m sim_deploy.offline.real_trials_train_ref tagwidth   -> train_first/<package>/tags_vs_width.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sim_deploy.offline.real_trials_common import B1_REF, OUT, copy_version, pose_summary, write_json

DEFAULT = Path("~/data/umi/datasets/multi/pick1-double-v2/pick1-double-v2.zarr.zip").expanduser()
ROW = 6
K = 16
DOWN = 3


def tagwidth(ds: Path) -> int:
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    from sim_deploy.offline.real_trials_common import TagDetector, tags_of
    register_codecs(verbose=False)
    root = zarr.open(zarr.ZipStore(str(ds), mode="r"), mode="r")
    rb = ReplayBuffer(root=root)
    pkg = json.loads((ds.parent / "package.json").read_text())["episodes"]
    down = root["meta/downsample_index_camera_right_ultrawide_rgb"][:]
    uw_ends = root["meta/episode_ends_camera_right_ultrawide_rgb"][:]
    widths = root["data/gripper_right_gripper_width"][:, 0]
    uw = root["data/camera_right_ultrawide_rgb"]
    det = TagDetector()
    rows = []
    for i in range(rb.n_episodes):
        u0 = int(uw_ends[i - 1]) if i else 0
        for u in range(u0, min(u0 + 60, int(uw_ends[i]))):
            r = int(down[u])
            rows.append({"ep": i, "folder": pkg[i]["folder"], "uw_frame": u - u0, "row_global": r,
                         "width_m": float(widths[r]), "tags_uw": tags_of(det, uw[u])})
        print(i, len(rows), flush=True)
    out = OUT / "train_first" / ds.name.replace(".zarr.zip", "") / "tags_vs_width.json"
    write_json(out, {"dataset": str(ds), "frames": rows})
    return 0


def main(argv) -> int:
    if argv and argv[0] == "tagwidth":
        return tagwidth(Path(argv[1]).expanduser() if len(argv) > 1 else DEFAULT)
    import cv2
    cv2.setNumThreads(4)
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    from sim_deploy.camera_study.reference import _convert_lowdim, _sample_obs
    from sim_deploy.scene.zarr_lowdim import load_episode
    register_codecs(verbose=False)
    ds = Path(argv[0]).expanduser() if argv else DEFAULT
    name = ds.name.replace(".zarr.zip", "")
    pkg = json.loads((ds.parent / "package.json").read_text())["episodes"]
    root = zarr.open(zarr.ZipStore(str(ds), mode="r"), mode="r")
    rb = ReplayBuffer(root=root)
    out_dir = OUT / "train_first" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(rb.n_episodes):
        s, used = _sample_obs(rb, i, ROW)
        low = _convert_lowdim(rb, s, used["start"])
        ep = load_episode(ds, i)
        P = ep["poses"]                                     # (n, 4, 4) TCP in the tag frame
        T = P[ROW]
        rows = [min(ROW + DOWN * k, len(P) - 1) for k in range(K)]
        rel = np.linalg.inv(T)[None] @ P[rows]
        obs = dict(low, main=s["main"].astype(np.uint8), ultrawide=s["uw"].astype(np.uint8))
        np.savez_compressed(out_dir / f"ep{i:02d}.npz", **obs, gt_rel=rel, gt_abs=P[rows], start_abs=P[0],
                            widths=ep["widths"][rows])
        index.append({"index": i, "folder": pkg[i]["folder"], "slot": pkg[i]["slot"], "pi_name": pkg[i]["pi_name"],
                      "n_rows": int(len(P)), "start_row0": pose_summary(P[0]), "row6": pose_summary(T),
                      "width_row6": float(ep["widths"][ROW]),
                      "gt_k15_rel_cm": (rel[-1, :3, 3] * 100).round(2).tolist(),
                      "gt_abs_z_k0_k15_m": [round(float(P[r][2, 3]), 4) for r in rows],
                      "used": {k: v for k, v in used.items() if k != "stop"}})
        print(i, pkg[i]["folder"], pkg[i]["slot"], "h", round(float(T[2, 3]), 3), "k15 rel cm",
              index[-1]["gt_k15_rel_cm"], flush=True)
    # cross-check with B1 (PICK_TEST1 pick1-v1 index 14 = episode_000152)
    check = {}
    b1 = B1_REF / "ep14_first/obs.npz"
    j = next((e["index"] for e in index if e["pi_name"].endswith("episode_000152") and e["folder"] == "PICK_TEST1"),
             None)
    if b1.exists() and j is not None:
        a, b = np.load(out_dir / f"ep{j:02d}.npz"), np.load(b1)
        check = {"b1": str(b1), "index_here": j,
                 "max_abs_diff": {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))))
                                  for k in b.files}}
        print("B1 check", check)
    write_json(out_dir / "index.json", {"dataset": str(ds), "row": ROW, "copy_version": copy_version(),
                                        "b1_check": check, "episodes": index})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
