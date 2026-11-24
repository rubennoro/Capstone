"""What an eval runs: ``--policy RUN_FOLDER | hold_still | replay:<zarr>#<index>`` -> names, paths, commits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sim_deploy.scene.zarr_lowdim import n_episodes, parse_replay_target


@dataclass
class Target:
    raw: str                    # what the user passed (the `policy_deploy use` target)
    kind: str                   # hold_still | replay | diffusion | prompting
    task: str                   # sim_evals/<task>/...
    run_folder: str             # sim_evals/<task>/<run_folder>/...
    use_arg: str                # the argument for `policy_deploy use`
    checkpoint: str | None = None
    dataset: str | None = None
    training_commit: str | None = None
    info: dict = field(default_factory=dict)


def resolve(policy: str) -> Target:
    if policy == "hold_still":
        return Target(policy, "hold_still", "_baseline", "hold_still", "hold_still")
    if policy.startswith("replay:"):
        path, index = parse_replay_target(policy)
        if not path.exists():
            raise FileNotFoundError(f"replay dataset not found: {path}")
        n = n_episodes(path)
        if not 0 <= index < n:
            raise ValueError(f"{path.name} has {n} episodes (0-{n - 1}), not {index}")
        stem = path.name.split(".zarr")[0]
        task = path.parent.parent.name if path.parent.parent.name else "_replay"
        return Target(policy, "replay", f"_replay/{task}", f"{stem}_ep{index:03d}", f"replay:{path}#{index}",
                      dataset=str(path), info={"index": index, "n_episodes": n})
    run = Path(policy).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"not a run folder, hold_still or replay:...: {policy}")
    ckpts = sorted((run / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"{run}: no checkpoints/*.ckpt (fetch it with train/fetch.sh)")
    task = run.parent.parent.name if run.parent.name == "policy" else run.parent.name
    info: dict = {}
    commit = None
    for name in ("train_meta.json", "run_config/job_info.json", "job.json", "summary.json"):
        p = run / name
        if p.is_file():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            info[name] = {k: d[k] for k in ("dataset", "config_name", "clone_git_commit", "epochs", "note")
                          if k in d}
            commit = commit or d.get("clone_git_commit")
    dataset = next((v["dataset"] for v in info.values() if "dataset" in v), None)
    cfg = info.get("run_config/job_info.json", {}).get("config_name", "")
    kind = "diffusion" if cfg == "umi_policy_dunet" else "prompting" if cfg == "umi_policy_dunetp" else "diffusion"
    return Target(policy, kind, task, run.name, str(run), checkpoint=str(ckpts[-1]), dataset=dataset,
                  training_commit=commit, info=info)
