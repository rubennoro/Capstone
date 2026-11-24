# adapted from rnoro/deployment:deployment/policy/policy/self_check.py @ 91b178b
# changes: module path and the new station (Settings.load(station, mock)); --json prints the JSON check lines (without
# it, one readable line per check); for kind diffusion adds the rows "gpu memory" (free/used/total from nvidia-smi),
# "gpu processes" (other compute processes via nvidia-smi --query-compute-apps; warn above 300 MB) and "precision";
# the nvidia-smi readers (gpu_info, gpu_processes) are shared with app.py's startup.json; the environment hint names
# python.torch; with policy.prompt_dataset set, a "prompt dataset" row (exists, readable) and zarr, numcodecs and
# imagecodecs in the imports row.
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/policy/self_check.py @ f632576
# changes: for kind replay a "replay episode" row (the dataset exists; the index is checked when the policy loads)
# (sim_deploy)
"""`python -m policy_deploy.policy.self_check [--station FILE] [--mock] [--json]`: the policy rows of
`policy_deploy check` (STRUCTURE.md 5.2, 9.2).

With --json: one JSON object per line, {"name", "status", "detail"}, status pass | warn | fail | info. Exits 0 unless
it crashes. It never loads the checkpoint and never binds a socket. exec_h is the manager's and is not shown here.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from policy_deploy.policy.errors import SettingsError
from policy_deploy.policy.settings import Settings

MODEL_MODULES = ("torch", "dill", "hydra", "timm", "diffusers", "behavior_prompting")
PROMPT_MODULES = ("zarr", "numcodecs", "imagecodecs")      # reading the prompt dataset (policy.prompt_dataset)
ENV_HINT = ("run it with python.torch, the torch environment "
            "(uv pip install -e deployment/contracts -e \"deployment/policy_deploy[model,test]\" there)")
OTHER_GPU_WARN_MB = 300.0
NVIDIA_SMI_TIMEOUT_S = 5.0

Runner = Callable[[list[str]], str]


def _run_nvidia_smi(args: list[str]) -> str:
    """stdout of `nvidia-smi <args>`; raises OSError / subprocess errors."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise FileNotFoundError("nvidia-smi not found")
    done = subprocess.run([exe, *args], capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S,
                          stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        raise OSError(f"nvidia-smi exited {done.returncode}: {done.stderr.strip()[:200]}")
    return done.stdout


def _number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None                            # "[N/A]", "[Not Supported]"


def gpu_info(run: Runner = _run_nvidia_smi) -> dict[str, Any] | None:
    """The first GPU: {"name", "memory_used_mb", "memory_total_mb", "power_limit_w"} (None values where nvidia-smi
    says N/A), or None without nvidia-smi."""
    try:
        out = run(["--query-gpu=name,memory.used,memory.total,power.limit", "--format=csv,noheader,nounits"])
    except Exception:
        return None
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 4:
        return None
    return {"name": parts[0], "memory_used_mb": _number(parts[1]), "memory_total_mb": _number(parts[2]),
            "power_limit_w": _number(parts[3])}


def gpu_processes(run: Runner = _run_nvidia_smi) -> list[dict[str, Any]] | None:
    """Every compute process on the GPU: [{"pid", "name", "used_mb"}], or None without nvidia-smi."""
    try:
        out = run(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    except Exception:
        return None
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        procs.append({"pid": int(parts[0]), "name": parts[1], "used_mb": _number(parts[2])})
    return procs


class SelfCheck:
    def __init__(self, station: Path | None, mock: bool, out: TextIO, as_json: bool = True,
                 nvidia_smi: Runner = _run_nvidia_smi) -> None:
        self._station = station
        self._mock = mock
        self._out = out
        self._json = as_json
        self._smi = nvidia_smi

    def emit(self, name: str, status: str, detail: str) -> None:
        if self._json:
            self._out.write(json.dumps({"name": name, "status": status, "detail": detail}) + "\n")
        else:
            self._out.write(f"{status.upper():4}  {name}: {detail}\n")
        self._out.flush()

    def run(self) -> None:
        try:
            settings = Settings.load(self._station, self._mock)
        except SettingsError as e:
            self.emit("settings", "fail", "; ".join(e.problems))
            return
        self.emit("settings", "pass", f"{settings.station_file} (policy section valid)")
        self.emit("kind", "info", f"policy kind: {settings.active_kind}" + (" (--mock)" if settings.mock else ""))
        if settings.active_kind == "replay":
            path = settings.replay_dataset
            if path is None or not path.exists():
                self.emit("replay episode", "fail", f"replay dataset not found: {path}: check "
                                                    f"policy.replay_episode.dataset in the station file")
            else:
                self.emit("replay episode", "pass", f"{path} episode {settings.replay_index} (exec_h "
                                                    f"{settings.exec_h}; the index is checked at load)")
        if settings.active_kind == "diffusion":
            self._check_checkpoint(settings)
            if settings.prompt_dataset is not None:
                self._check_prompt_dataset(settings)
            torch = self._check_imports(MODEL_MODULES + (PROMPT_MODULES if settings.prompt_dataset else ()))
            if torch is not None:
                self._check_device(settings, torch)
            if settings.device == "cuda":
                self._check_gpu_memory()
                self._check_gpu_processes()
            self.emit("precision", "info", f"precision: {settings.precision}"
                      + (" (autocast bf16, PLAN 13 second backup)" if settings.precision == "bf16" else ""))
            if settings.diffusion_steps is not None:
                self.emit("diffusion_steps", "info",
                          f"diffusion steps: {settings.diffusion_steps}, PLAN 13 third backup")

    def _check_checkpoint(self, settings: Settings) -> None:
        path = settings.checkpoint
        if path is None:
            self.emit("checkpoint", "fail", "policy.checkpoint is not set: set it in station.yaml")
        elif not path.exists():
            self.emit("checkpoint", "fail", f"checkpoint not found: {path}: check policy.checkpoint in station.yaml")
        elif not path.is_file() or path.suffix != ".ckpt":
            self.emit("checkpoint", "fail", f"not a .ckpt file: {path}: check policy.checkpoint in station.yaml")
        else:
            try:
                with open(path, "rb") as f:
                    f.read(1)
            except OSError as e:
                self.emit("checkpoint", "fail", f"checkpoint not readable: {path}: {e}")
                return
            st = path.stat()
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            self.emit("checkpoint", "pass", f"{path} ({st.st_size / 1e9:.2f} GB, modified {mtime})")

    def _check_prompt_dataset(self, settings: Settings) -> None:
        path = settings.prompt_dataset
        hint = "check policy.prompt_dataset in station.yaml (the processed .zarr.zip of the demos)"
        if not path.exists():
            self.emit("prompt dataset", "fail", f"prompt dataset not found: {path}: {hint}")
            return
        if not (path.name.endswith(".zarr.zip") or (path.is_dir() and path.name.endswith(".zarr"))):
            self.emit("prompt dataset", "fail", f"not a .zarr.zip file: {path}: {hint}")
            return
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    f.read(1)
            except OSError as e:
                self.emit("prompt dataset", "fail", f"prompt dataset not readable: {path}: {e}")
                return
        size = f", {path.stat().st_size / 1e9:.2f} GB" if path.is_file() else ""
        self.emit("prompt dataset", "pass", f"{path} ({settings.prompt_split} split{size})")

    def _check_imports(self, names: tuple[str, ...] = MODEL_MODULES):
        missing: list[str] = []
        modules = {}
        with contextlib.redirect_stdout(sys.stderr):     # keep stdout for check lines only
            for name in names:
                try:
                    modules[name] = importlib.import_module(name)
                except Exception as e:                  # ImportError, or a broken install
                    missing.append(f"{name} ({type(e).__name__}: {e})")
        if missing:
            self.emit("imports", "fail", f"not importable: {', '.join(missing)}: {ENV_HINT}")
            return modules.get("torch")
        self.emit("imports", "pass", f"torch {modules['torch'].__version__}, " + ", ".join(names[1:]))
        return modules["torch"]

    def _check_device(self, settings: Settings, torch) -> None:
        if settings.device == "cpu":
            self.emit("device", "info", "device: cpu (PLAN 13 device backup)")
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.emit("device", "pass", f"cuda: {props.name}, {props.total_memory / 1e9:.1f} GB")
        else:
            self.emit("device", "fail", "CUDA not available: check the GPU driver, or set policy.device: cpu "
                                        "(PLAN 13 backup)")

    def _check_gpu_memory(self) -> None:
        info = gpu_info(self._smi)
        if info is None:
            self.emit("gpu memory", "info", "nvidia-smi not available: free GPU memory unknown")
            return
        used, total = info["memory_used_mb"], info["memory_total_mb"]
        if used is None or total is None:
            self.emit("gpu memory", "info", f"{info['name']}: memory not reported by nvidia-smi")
            return
        power = info["power_limit_w"]
        self.emit("gpu memory", "info", f"{info['name']}: {total - used:.0f} MiB free of {total:.0f} MiB "
                  f"({used:.0f} MiB used); power limit " + (f"{power:.0f} W" if power is not None else "n/a"))

    def _check_gpu_processes(self) -> None:
        procs = gpu_processes(self._smi)
        if procs is None:
            self.emit("gpu processes", "info", "nvidia-smi not available: other GPU processes unknown")
            return
        others = [p for p in procs if p["pid"] != os.getpid()]
        used = sum(p["used_mb"] or 0.0 for p in others)
        if not others:
            self.emit("gpu processes", "pass", "no other process on the GPU")
            return
        listing = ", ".join(f"{p['name']} (pid {p['pid']}, {p['used_mb'] or 0:.0f} MiB)" for p in others)
        status = "warn" if used > OTHER_GPU_WARN_MB else "pass"
        detail = f"{len(others)} other process(es) use {used:.0f} MiB: {listing}"
        if status == "warn":
            detail += f" (over {OTHER_GPU_WARN_MB:.0f} MiB: close them, the model needs the 4 GB)"
        self.emit("gpu processes", status, detail)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m policy_deploy.policy.self_check",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", type=Path, default=None,
                    help="the station file (default: deployment/policy_deploy/station.yaml)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--json", action="store_true", help="one JSON check line per row (for policy_deploy check)")
    a = ap.parse_args(argv)
    SelfCheck(a.station, a.mock, sys.stdout, as_json=a.json).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
