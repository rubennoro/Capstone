"""Inference latency / precision benchmark (same node and code path as the server, no RMQ in the timed loop).

    python -m franka_deploy.tools.bench_inference --ckpt X.ckpt --work-dir DIR --dataset-path P.zarr.zip \
        [--n 50] [--device cuda] [--out bench.json]

Reports: cold first call, warm p50/p95 over N calls for fp32 and bf16 autocast (torch.autocast, as training's
mixed_precision bf16), the fp32-vs-bf16 action difference with identical diffusion noise (same seed), fp32
repeatability, the server-side 960x720 -> 224 preprocessing time, and peak GPU memory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time

import numpy as np


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _obs(seed=0, h=720, w=960):
    rng = np.random.default_rng(seed)
    rot = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (2, 1))
    return {"robot0_main_camera": rng.integers(0, 255, (2, h, w, 3), dtype=np.uint8),
            "robot0_ultrawide_camera": rng.integers(0, 255, (2, h, w, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([[0.0, 0.0, -0.01], [0, 0, 0]], np.float32),
            "robot0_eef_rot_axis_angle": rot.copy(), "robot0_gripper_width": np.full((2, 1), 0.07, np.float32),
            "robot0_eef_rot_axis_angle_wrt_start": rot.copy()}


def _stats(xs):
    a = np.array(xs) * 1000
    return {"n": len(xs), "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
            "mean_ms": round(float(a.mean()), 1), "max_ms": round(float(a.max()), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import os
    a.out = os.path.abspath(a.out) if a.out else None  # build_node chdirs into --work-dir
    a.dataset_path = os.path.abspath(a.dataset_path)
    import torch
    from franka_deploy.inference.server import build_node

    node, rep = build_node(a.ckpt, a.work_dir, a.dataset_path, f"tcp://127.0.0.1:{_free_port()}", a.device)
    task = sorted(node.prompt_dataset.get_unique_task_name_to_dataset_indices())[0]
    cuda = node.device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    node._handle_prompt(task, a.prompt_index)
    sync()
    prompt_s = time.monotonic() - t
    obs = _obs()
    obs_model = node._process_obs(obs)
    tens = {k: torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0).to(node.device) for k, v in obs_model.items()}

    def call(precision, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
        sync()
        t0 = time.monotonic()
        with torch.inference_mode(), ctx:
            out = node.policy.predict_action(tens)["action_pred"][0].float().cpu().numpy()
        sync()
        return out, time.monotonic() - t0

    res = {"startup": rep, "prompt_s": round(prompt_s, 3), "device": torch.cuda.get_device_name(0) if cuda else "cpu"}
    _, cold = call("fp32")
    res["fp32_cold_first_call_ms"] = round(cold * 1000, 1)
    for prec in ("fp32", "bf16"):
        _, first = call(prec)
        ts = [call(prec)[1] for _ in range(a.n)]
        res[prec] = {"first_call_ms": round(first * 1000, 1), **_stats(ts)}
    a32, _ = call("fp32", seed=123)
    a32b, _ = call("fp32", seed=123)
    a16, _ = call("bf16", seed=123)
    d = np.abs(a32 - a16)
    res["fp32_repeat_max_abs"] = float(np.abs(a32 - a32b).max())
    res["bf16_vs_fp32"] = {"pos_max_abs_m": float(d[:, :3].max()), "rot6d_max_abs": float(d[:, 3:9].max()),
                           "width_max_abs_m": float(d[:, 9].max()), "pos_mean_abs_m": float(d[:, :3].mean())}
    pre = []
    for i in range(20):
        o = _obs(seed=i)
        t0 = time.monotonic()
        node._process_obs(o)
        pre.append(time.monotonic() - t0)
    res["server_preprocess_960x720"] = _stats(pre)
    if cuda:
        res["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
