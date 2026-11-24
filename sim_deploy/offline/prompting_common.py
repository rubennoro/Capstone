"""W1 step 2 (S2, prompting): shared bits of prompting_choice.py and prompting_sweep.py (no torch at import).

Not run by itself. Holds the 5 prompting runs, the output/scratch folders, the station copy made with the copy's own
`policy_deploy use` (launcher/use_policy.use, in-process, on a copy of sim_deploy/stations/policy_sim.yaml in the
scratch folder; the original is never touched), the deploy-path version record every result file carries, and the
GPU lock (fcntl.flock on the shared lock file, held for a model's whole CUDA lifetime).

Run the scripts that use it from ~/Capstone-main-sim/deployment/policy_deploy with `unset PYTHONPATH` and
~/envs/sim_policy_deploy_torch/bin/python (see their docstrings).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

HOME = Path.home()
REPO = HOME / "Capstone-main-sim"
COPY = REPO / "deployment"
POLICY_DIR = COPY / "policy_deploy/policy_deploy/policy"
OUT = REPO / "sim_evals/_pipeline/W1/prompting"
SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad/w1/s2")
GPU_LOCK = Path.home() / "Capstone-main-sim" / "sim_evals" / ".gpu.lock"   # shared with W0/W2 (flock(1) on the same file)
STATION_SRC = REPO / "sim_deploy/stations/policy_sim.yaml"
REF_DIR = REPO / "sim_evals/_camera_study/B/reference"            # B1's 15 training-exact observations
REF_DATASET = HOME / "data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip"   # the references' package

# run -> (task folder, dataset it trained on)
RUNS = {
    "run_006_2026-11-18": ("PICK_TEST1", "pick1-v1"),
    "run_012_2026-11-18": ("PICK_TEST1", "pick1-ep15-v1"),
    "run_014_2026-11-18": ("PICK_TEST1", "pick1-ep20-v1"),
    "run_016_2026-11-18": ("PICK_TEST1", "pick1-ep25-v1"),
    "run_020_2026-11-19": ("PICK_TEST1", "pick1-ep10-v1"),
}


def run_folder(run: str) -> Path:
    return HOME / "Capstone/data" / RUNS[run][0] / "policy" / run


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def version_info() -> dict[str, Any]:
    """Which deploy-path version ran: the brief's diff-stat line, git status of the copy, gripper_map.py present,
    and short hashes of the files this step relies on."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"error: {e}"
    stat = git("diff", "--stat", "--", "deployment").splitlines()
    files = ["prompt_source.py", "promptable_diffusion_policy.py", "diffusion_policy.py", "model_runner.py",
             "checkpoint_loader.py", "gripper_map.py", "settings.py", "policy_server.py"]
    return {
        "git_diff_stat_deployment_tail": stat[-1] if stat else "",
        "git_status_deployment": git("status", "--short", "--", "deployment")[:500],
        "gripper_map_py_exists": (POLICY_DIR / "gripper_map.py").exists(),
        "file_sha256_16": {f: _sha(POLICY_DIR / f) for f in files},
        "use_policy_sha256_16": _sha(COPY / "policy_deploy/policy_deploy/launcher/use_policy.py"),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def make_station(run: str) -> tuple[Path, dict[str, Any]]:
    """A fresh copy of policy_sim.yaml in the scratch folder, pointed at `run` by the copy's own `use`. Returns the
    path and what `use` wrote (policy + manager sections, notes)."""
    import yaml
    from policy_deploy.launcher.use_policy import use
    dst = SCRATCH / "stations" / f"{run}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STATION_SRC, dst)
    _old, new, choice = use(dst, str(run_folder(run)))
    data = yaml.safe_load(new)
    return dst, {"policy_section": data.get("policy"), "manager_section": data.get("manager"),
                 "choice_policy": choice.policy, "choice_manager": choice.manager, "notes": choice.notes}


@contextlib.contextmanager
def gpu_lock(log=print):
    """Hold the shared GPU lock (blocking) for the block's whole duration; a no-op with GPU_LOCK_HELD=1 (the process
    runs under ``flock <GPU_LOCK> cmd``; a second flock from a new open file description would block forever)."""
    if os.environ.get("GPU_LOCK_HELD") == "1":
        log(f"GPU lock held by the outer flock on {GPU_LOCK}")
        yield
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(GPU_LOCK, "a+") as f:
        t0 = time.monotonic()
        log(f"waiting for GPU lock {GPU_LOCK} ...")
        fcntl.flock(f, fcntl.LOCK_EX)
        log(f"GPU lock held (waited {time.monotonic() - t0:.1f} s)")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            log("GPU lock released")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o: Any):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
