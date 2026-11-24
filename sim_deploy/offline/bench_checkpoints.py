"""Time every runnable checkpoint with the copy's own bench (W1 step 4, part 1).

For each runnable checkpoint of the catalog (`sim_deploy.ui.catalog --json --no-replays`, `runnable: true`) this
runs, from `deployment/policy_deploy` in the torch env (PYTHONPATH unset):

    python -m policy_deploy.policy.bench --checkpoint CKPT [--prompt-dataset ZIP --prompt-split train]
           --steps S,8 --precision fp32,bf16 --infers N --warmup W --device cuda --out OUT/<TASK>_<run>[/<tag>]/

S is the checkpoint's own `model.num_inference_steps` (run_config/config.yaml). ZIP, for a prompting run, is what
`policy_deploy use RUN_FOLDER` sets as `policy.prompt_dataset` (`launcher/use_policy.py training_dataset`:
data/<folder>/package/<dataset>/<dataset>.zarr.zip). The GPU lock (`fcntl.flock` on the shared gpu.lock) is taken
before the bench starts and released when it exits, one checkpoint at a time, so other agents can interleave.

Right after the lock is taken and after the bench, `env.json` records: `uptime` load average, nvidia-smi (memory,
power, SM clock, temperature, P-state, throttle reasons), every compute process on the GPU, AC power
(/sys/class/power_supply/*/online), the power profile, the copy version (`git diff --stat -- deployment | tail -1`,
HEAD, whether policy/gripper_map.py exists) and the bench's exit code and wall time. Flags: `load_1m_gt_6`,
`other_gpu_process`.

Run (cwd ~/Capstone-main-sim, any python 3 with PyYAML, e.g. the light env):

    unset PYTHONPATH; ~/envs/sim_deploy/bin/python sim_deploy/offline/bench_checkpoints.py \
        [--only PICK_TEST1/run_006_2026-11-18,...] [--tag rerun1] [--infers 150] [--warmup 5]

Without --tag the output goes to OUT/<TASK>_<run>/; with --tag (a re-run) to OUT/<TASK>_<run>/<tag>/.
bench.py appends to bench.jsonl, so a folder that already has one is skipped unless --force.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path.home() / "Capstone-main-sim"
POLICY_DEPLOY = REPO / "deployment" / "policy_deploy"
TORCH_PY = Path.home() / "envs" / "sim_policy_deploy_torch" / "bin" / "python"
LIGHT_PY = Path.home() / "envs" / "sim_deploy" / "bin" / "python"
GPU_LOCK = Path.home() / "Capstone-main-sim" / "sim_evals" / ".gpu.lock"   # shared with W0/W2 (flock(1) on the same file)
OUT = REPO / "sim_evals" / "_pipeline" / "W1" / "bench"


def sh(cmd: list[str], cwd: Path | None = None, timeout: float = 20.0) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:                                    # noqa: BLE001 (a record, not a failure)
        return f"ERROR {e}"


def copy_version() -> dict:
    return {"diff_stat_tail": sh(["bash", "-c", f"git -C {REPO} diff --stat -- deployment | tail -1"]),
            "head": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
            "gripper_map_py_exists": (POLICY_DEPLOY / "policy_deploy" / "policy" / "gripper_map.py").is_file()}


def machine_state() -> dict:
    load = Path("/proc/loadavg").read_text().split()[:3]
    smi_q = "memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu,pstate," \
            "clocks_throttle_reasons.active"
    smi = sh(["nvidia-smi", f"--query-gpu={smi_q}", "--format=csv,noheader"])
    apps = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    ac = {}
    for p in Path("/sys/class/power_supply").glob("*"):
        f = p / "online"
        if f.is_file():
            ac[p.name] = f.read_text().strip()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "loadavg_1_5_15": [float(x) for x in load],
            "uptime": sh(["uptime"]), "nvidia_smi": dict(zip(smi_q.split(","), [s.strip() for s in smi.split(",")])),
            "gpu_compute_apps": [a for a in apps.splitlines() if a.strip()], "power_supply_online": ac,
            "power_profile": sh(["powerprofilesctl", "get"])}


def runnable_checkpoints() -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([str(LIGHT_PY), "-m", "sim_deploy.ui.catalog", "--json", "--no-replays"], cwd=REPO,
                         capture_output=True, text=True, env=env, check=True).stdout
    return [p for p in json.loads(out)["policies"] if p.get("runnable")]


def run_info(p: dict) -> dict:
    run = Path(p["run_folder"])
    cfg = yaml.safe_load((run / "run_config" / "config.yaml").read_text())
    prompting = bool(cfg["task"]["shape_meta"].get("use_prompting", False))
    steps = int((cfg.get("model") or cfg.get("policy"))["num_inference_steps"])
    zip_name = Path(str(cfg["task"]["dataset"]["dataset_path"])).name
    name = zip_name.removesuffix(".zarr.zip")
    prompt_ds = run.parent.parent / "package" / name / zip_name      # as launcher/use_policy.py training_dataset
    return {"id": p["id"], "task": p["task"], "run": run.name, "run_folder": str(run), "checkpoint": p["checkpoint"],
            "kind": "prompting" if prompting else "plain", "train_config": p.get("train_config"),
            "dataset": name, "dataset_path": str(prompt_ds), "ckpt_steps": steps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="", help="comma-separated catalog ids")
    ap.add_argument("--tag", default="", help="re-run tag (sub-folder)")
    ap.add_argument("--infers", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    only = {s for s in a.only.split(",") if s}
    todo = [run_info(p) for p in runnable_checkpoints() if not only or p["id"] in only]
    print(f"{len(todo)} checkpoints", flush=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for info in todo:
        out = OUT / f"{info['task']}_{info['run']}" / a.tag if a.tag else OUT / f"{info['task']}_{info['run']}"
        if (out / "bench.jsonl").exists() and not a.force:
            print(f"skip {info['id']}: {out}/bench.jsonl exists", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        steps = ",".join(str(s) for s in dict.fromkeys([info["ckpt_steps"], 8]))
        cmd = [str(TORCH_PY), "-m", "policy_deploy.policy.bench", "--checkpoint", info["checkpoint"],
               "--steps", steps, "--precision", "fp32,bf16", "--infers", str(a.infers), "--warmup", str(a.warmup),
               "--device", "cuda", "--out", str(out)]
        if info["kind"] == "prompting":
            cmd[5:5] = ["--prompt-dataset", info["dataset_path"], "--prompt-split", "train"]
        record = {"run": info, "cmd": cmd, "cwd": str(POLICY_DEPLOY), "copy_version": copy_version(),
                  "before_lock": machine_state()}
        t_wait = time.monotonic()
        with open(GPU_LOCK, "a") as lock:
            if os.environ.get("GPU_LOCK_HELD") != "1":      # else: run under `flock GPU_LOCK ...`, which holds it
                fcntl.flock(lock, fcntl.LOCK_EX)
            record["lock_wait_s"] = round(time.monotonic() - t_wait, 1)
            record["at_lock"] = machine_state()
            print(f"{info['id']}: lock after {record['lock_wait_s']} s, load {record['at_lock']['loadavg_1_5_15']}",
                  flush=True)
            t0 = time.monotonic()
            with open(out / "bench.log", "w") as log:
                rc = subprocess.run(cmd, cwd=POLICY_DEPLOY, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            record["bench_wall_s"] = round(time.monotonic() - t0, 1)
            record["exit_code"] = rc
            record["after"] = machine_state()
            fcntl.flock(lock, fcntl.LOCK_UN)
        at = record["at_lock"]
        record["flags"] = {"load_1m_gt_6": at["loadavg_1_5_15"][0] > 6 or record["after"]["loadavg_1_5_15"][0] > 6,
                           "other_gpu_process": bool(at["gpu_compute_apps"])}
        (out / "env.json").write_text(json.dumps(record, indent=1))
        print(f"{info['id']}: exit {rc}, {record['bench_wall_s']} s, flags {record['flags']}", flush=True)
        print((out / "bench.log").read_text()[-1500:], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
