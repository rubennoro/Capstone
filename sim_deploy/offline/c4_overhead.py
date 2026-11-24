"""CPU micro-bench of the per-step overhead around inference (W1 step 4, cross-check of the overhead term).

policy_ms (manager steps.jsonl) = look-up + PoseConverter.to_observation + C4 encode + ZMQ hop + C4 decode (policy) +
policy.infer + C4 encode (chunk) + ZMQ hop + C4 decode (manager). The bench (policy.bench) times only the
ModelRunner part of policy.infer. This times, in-process with the copy's own code and no socket, the pieces that are
pure CPU: to_observation on realistic poses, encode/decode of a full Observation (2 x 224x224x3 uint8 per camera) and
of an ActionChunk (with their contract checks). The ZMQ localhost hop itself is not timed (no ports may be bound).

Run (cwd ~/Capstone-main-sim/deployment/policy_deploy, light env):
    unset PYTHONPATH; ~/envs/sim_policy_deploy/bin/python ~/Capstone-main-sim/sim_deploy/offline/c4_overhead.py \
        [--n 2000] [--out ~/Capstone-main-sim/sim_evals/_pipeline/W1/bench/c4_overhead.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deploy_contracts.codec import decode, encode
from policy_deploy.contracts import ActionChunk, Observation
from policy_deploy.manager.pose_converter import PoseConverter


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(x: float, a: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(a) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.5 + x, 0.0, 0.3]
    return T


def stats(t: list[float]) -> dict:
    a = np.asarray(t) * 1e3
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    frames = SimpleNamespace(main=rng.integers(0, 255, (2, 224, 224, 3), dtype=np.uint8),
                             ultrawide=rng.integers(0, 255, (2, 224, 224, 3), dtype=np.uint8))
    conv = PoseConverter(exec_h=12, lookahead_ms=0)
    poses = np.stack([pose(0.0, 0.0), pose(0.004, 0.01)])
    start = pose(-0.05, -0.1)
    widths = np.array([0.08, 0.08])
    steps = 16
    chunk = ActionChunk(eef_pos=np.zeros((steps, 3), np.float32) + np.linspace(0, 0.02, steps, dtype=np.float32)[:, None],
                        eef_rot6d=np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (steps, 1)),
                        gripper_width_m=np.full((steps, 1), 0.04, np.float32), width_clipped=0)
    t = {k: [] for k in ("to_observation", "encode_obs", "decode_obs", "encode_chunk", "decode_chunk", "total")}
    nbytes = None
    for _ in range(a.n):
        t0 = time.perf_counter()
        obs, _anchor = conv.to_observation(poses, widths, start, frames, 1, 1, "target")
        t1 = time.perf_counter()
        b = encode(obs)
        t2 = time.perf_counter()
        decode(b, Observation)
        t3 = time.perf_counter()
        c = encode(chunk)
        t4 = time.perf_counter()
        decode(c, ActionChunk)
        t5 = time.perf_counter()
        nbytes = (len(b), len(c))
        for k, (x, y) in zip(t, ((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t0, t5))):
            t[k].append(y - x)
    result = {"n": a.n, "bytes_obs_chunk": nbytes, **{k: stats(v) for k, v in t.items()},
              "not_timed": "ZMQ localhost send/recv of ~0.6 MB each way, StateHistory look-up, policy make_chunk"}
    print(json.dumps(result, indent=1))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
