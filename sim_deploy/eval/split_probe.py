"""The training split of one checkpoint, run in the torch env (~/envs/sim_policy_deploy_torch, which has torch and
behavior_prompting). Called by ``start_set.py`` (sim_deploy env) as a subprocess; not imported by sim_deploy.

    ~/envs/sim_policy_deploy_torch/bin/python sim_deploy/eval/split_probe.py < request.json > reply.json

request: {"checkpoint": path, "sample_type": "episode"|"task", "val_ratio": float, "seed": int,
          "meta": {"episode_names": [...], "task_names": [...], "episode_ends": [...], "task_data_ends": [...],
                   "task_lengths": [...], "is_task_error_correction": [...] | null}}
reply:   {"checkpoint_split": {"<episode name>:<task name or ''>": bool, ...} | null,   (True = training)
          "recomputed_split": {... same keys ...}, "torch": version, "behavior_prompting": file}

checkpoint_split = the checkpoint's ``state_dicts.model._extra_training_split_info`` (base_policy.py state_dict(): what
the training run itself used). recomputed_split = behavior_prompting's own ``get_val_mask`` (train = not val) on the
dataset's metadata with the run's sample_type / val_ratio / seed, put into the same format by
``get_training_split_info_from_train_mask``. The checkpoint is opened with ``mmap=True`` on the CPU (~0.3 GB RSS for a
2.6 GB file).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np


def main() -> int:
    req = json.load(sys.stdin)
    import torch
    import behavior_prompting
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    from behavior_prompting.train_network.common.sampler import (get_training_split_info_from_train_mask,
                                                                  get_val_mask)

    m = req["meta"]
    rb = SimpleNamespace(
        episode_names=np.array(m["episode_names"], dtype=object), task_names=np.array(m["task_names"], dtype=object),
        episode_ends=np.array(m["episode_ends"], dtype=np.int64), task_data_ends=np.array(m["task_data_ends"], np.int64),
        task_lengths=np.array(m["task_lengths"], dtype=np.int64),
        is_task_error_correction=None if m.get("is_task_error_correction") is None
        else np.array(m["is_task_error_correction"], dtype=bool))
    rb.n_episodes = len(rb.episode_ends)
    rb.n_tasks = len(rb.task_names)
    rb.get_task_to_episode_idxs = lambda: ReplayBuffer.get_task_to_episode_idxs(rb)   # the library's own mapping
    val = get_val_mask(rb, req["sample_type"], float(req["val_ratio"]), seed=int(req["seed"]))
    recomputed = get_training_split_info_from_train_mask(rb, req["sample_type"], ~val)

    ck = torch.load(req["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    info = None
    for sd in ("model", "ema_model"):
        d = (ck.get("state_dicts") or {}).get(sd) or {}
        if "_extra_training_split_info" in d:
            info = d["_extra_training_split_info"]
            break
    json.dump({"checkpoint_split": None if info is None else {str(k): bool(v) for k, v in info.items()},
               "checkpoint_split_from": sd if info is not None else None,
               "recomputed_split": {str(k): bool(v) for k, v in recomputed.items()},
               "torch": torch.__version__, "behavior_prompting": behavior_prompting.__file__}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
