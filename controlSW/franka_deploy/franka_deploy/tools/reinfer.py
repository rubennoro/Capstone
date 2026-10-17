"""Re-inference check: re-run the model on the observations tapped during rollouts (images at every call:
task --tap-images-every 1) with the logged diffusion-noise seed, and compare with the logged action chunks.

    python -m franka_deploy reinfer <session_dir> --work-dir DIR [--device cuda] [--tol 1e-4]     (env_bp)

The prompt is re-applied per episode (same task + prompt index -> bitwise-identical prompt tensors, D3 fix 2).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)
    a.session = os.path.abspath(a.session)  # build_node chdirs into --work-dir
    import torch
    from franka_deploy.inference.server import attach_prompt_dataset, build_node

    eps = sorted(glob.glob(os.path.join(a.session, "task", "episodes", "*", "*", "*", "episode_*")))
    ck = json.load(open(os.path.join(eps[0], "config", "checksums.json")))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    node, _ = build_node(ck["checkpoint"], a.work_dir, None, f"tcp://127.0.0.1:{port}", a.device)
    attach_prompt_dataset(node, ck["prompt_zarr"])
    res = {"episodes": [], "max_abs_diff": 0.0, "n_calls": 0, "n_skipped_no_images": 0}
    for d in eps:
        summ = json.load(open(os.path.join(d, "summary.json")))
        task = summ.get("task_name", "block_raise")
        pidx = summ.get("prompt_index", ck.get("prompt_index", 0))
        node.policy.reset()
        node._handle_prompt(task, int(pidx))
        diffs = []
        for f in sorted(glob.glob(os.path.join(d, "tap", "call_*.npz"))):
            c = np.load(f)
            if "img224_robot0_main_camera" not in c or "reply_noise_seed" not in c:
                res["n_skipped_no_images"] += 1
                continue
            obs = {"robot0_main_camera": c["img224_robot0_main_camera"], "robot0_ultrawide_camera": c["img224_robot0_ultrawide_camera"]}
            for k in ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width", "robot0_eef_rot_axis_angle_wrt_start"):
                if f"obs_{k}" in c:
                    obs[k] = c[f"obs_{k}"]
            m = node._process_obs(obs)
            t = {k: torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0).to(node.device) for k, v in m.items()}
            torch.manual_seed(int(c["reply_noise_seed"]))
            with torch.inference_mode():
                pred = node.policy.predict_action(t)["action_pred"][0].float().cpu().numpy()
            logged = np.concatenate([c["reply_action0_eef_pos"], c["reply_action0_eef_rot_axis_angle"], c["reply_action0_gripper_width"]], 1)
            diffs.append(float(np.abs(pred - logged).max()))
        res["episodes"].append({"dir": d, "calls": len(diffs), "max_abs_diff": max(diffs) if diffs else None})
        res["n_calls"] += len(diffs)
        if diffs:
            res["max_abs_diff"] = max(res["max_abs_diff"], max(diffs))
    res["pass"] = res["n_calls"] > 0 and res["max_abs_diff"] < a.tol
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
