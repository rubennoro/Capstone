"""Shared bits of the W1 offline harness (step 1, S1): paths, the GPU lock, JSON helpers, and which version of the
copied deployment/ ran.

Nothing here touches the copy; ``deploy_version()`` only reads it. Import from either env
(``~/envs/sim_policy_deploy`` or ``~/envs/sim_policy_deploy_torch``); no torch, no PyAV, no zarr.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]                    # ~/Capstone-main-sim
DEPLOY = REPO / "deployment"
POLICY_DEPLOY = DEPLOY / "policy_deploy"                       # cwd for the copy (its configs resolve from here)
OUT = REPO / "sim_evals/_pipeline/W1"                          # results, caches, logs (gitignored)
OBS_CACHE = OUT / "obs"                                        # obs/<dataset>/ep<ii>_r<row>.npz + .json
STAGING = OUT / "staging"                                      # Pi-format folders of symlinks (deploy path input)
SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad/w1/s1")
GPU_LOCK = Path.home() / "Capstone-main-sim" / "sim_evals" / ".gpu.lock"   # shared with W0/W2 (flock(1) on the same file)
STATION_TEMPLATE = REPO / "sim_deploy/stations/policy_sim.yaml"   # copied, never edited
ENV_LIGHT = Path("~/envs/sim_policy_deploy/bin/python").expanduser()       # PyAV, no torch
ENV_TORCH = Path("~/envs/sim_policy_deploy_torch/bin/python").expanduser()  # torch, zarr + JPEG-XL, no PyAV
CAPSTONE_DATA = Path("~/Capstone/data").expanduser()               # <TASK>/policy/<run>/ (read only)
DATASETS = Path("~/data/umi/datasets").expanduser()                # <group>/<name>/<name>.zarr.zip (read only)
TASKS = Path("~/data/umi/tasks").expanduser()                      # <folder>/<slot>/{raw,processed} (read only)

#: copy files whose content decides the result (hashed into every result file)
VERSION_FILES = (
    "policy_deploy/policy_deploy/frames/pi_recording.py", "policy_deploy/policy_deploy/frames/packet_pairer.py",
    "policy_deploy/policy_deploy/frames/receiver.py", "policy_deploy/policy_deploy/frames/pair_rule.py",
    "policy_deploy/policy_deploy/frames/packet_buffer.py", "policy_deploy/policy_deploy/frames/packet_selector.py",
    "policy_deploy/policy_deploy/frames/frames_server.py", "policy_deploy/policy_deploy/frames/_dataproc/imageprep.py",
    "policy_deploy/policy_deploy/frames/jpeg_decoder.py", "policy_deploy/config/frames_process.yaml",
    "contracts/rigs/rig_pi_v3_640.yaml",
    "policy_deploy/policy_deploy/manager/state_history.py", "policy_deploy/policy_deploy/manager/policy_step.py",
    "policy_deploy/policy_deploy/manager/pose_converter.py", "policy_deploy/policy_deploy/manager/pose_math.py",
    "policy_deploy/config/manager.yaml",
    "policy_deploy/policy_deploy/policy/diffusion_policy.py",
    "policy_deploy/policy_deploy/policy/promptable_diffusion_policy.py",
    "policy_deploy/policy_deploy/policy/model_runner.py", "policy_deploy/policy_deploy/policy/base_policy.py",
    "policy_deploy/policy_deploy/policy/checkpoint_loader.py", "policy_deploy/policy_deploy/policy/prompt_source.py",
    "policy_deploy/policy_deploy/policy/gripper_map.py", "policy_deploy/config/policy.yaml",
)


def ensure_copy_importable() -> None:
    """The copy is installed editable in both envs; this only guards a run from another interpreter."""
    p = str(POLICY_DEPLOY)
    if p not in sys.path:
        sys.path.append(p)


def deploy_version() -> dict:
    """Which copy ran: the brief's ``git diff --stat -- deployment | tail -1`` (the copy is untracked in this repo, so
    it is usually empty), whether policy/gripper_map.py exists, and sha256[:12] + mtime of the files that matter."""
    try:
        stat = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", "--", "deployment"], capture_output=True,
                              text=True, timeout=30).stdout.strip().splitlines()
        stat_tail = stat[-1] if stat else ""
        untracked = subprocess.run(["git", "-C", str(REPO), "status", "--short", "--", "deployment"],
                                   capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[:3]
    except (OSError, subprocess.SubprocessError) as e:
        stat_tail, untracked = f"git failed: {e}", []
    files = {}
    for rel in VERSION_FILES:
        p = DEPLOY / rel
        if p.is_file():
            files[rel] = {"sha256_12": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
                          "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
        else:
            files[rel] = None
    return {"git_diff_stat_tail": stat_tail, "git_status_deployment": untracked,
            "gripper_map_py_exists": (DEPLOY / "policy_deploy/policy_deploy/policy/gripper_map.py").is_file(),
            "files": files}


@contextlib.contextmanager
def gpu_lock(what: str = ""):
    """Hold the shared GPU lock (fcntl.flock, exclusive, blocking) for the whole CUDA lifetime of a model.

    With GPU_LOCK_HELD=1 in the environment (the process runs under ``flock <GPU_LOCK> cmd``, which already holds it)
    this does nothing: a second flock on the same file from a new open file description would block forever."""
    if os.environ.get("GPU_LOCK_HELD") == "1":
        print(f"[gpu_lock] held by the outer flock on {GPU_LOCK} ({what})", flush=True)
        yield 0.0
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(GPU_LOCK, "a+")
    t0 = time.monotonic()
    print(f"[gpu_lock] waiting for {GPU_LOCK} ({what})", flush=True)
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    waited = time.monotonic() - t0
    print(f"[gpu_lock] acquired after {waited:.1f} s ({what})", flush=True)
    try:
        yield waited
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
        print(f"[gpu_lock] released ({what})", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1))
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def clean_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def set_cpu_threads() -> None:
    try:
        import cv2
        cv2.setNumThreads(4)
    except ImportError:
        pass
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(4)
