"""W1 step 2 (S2), Q1 + Q2: which prompt deploy uses, which prompts training used, and whether they are the same tensors.

CPU only (no GPU lock needed: the model is loaded on the CPU, only for its training split info, exactly as the copy's
CheckpointLoader does it). Run from the copy, one process:

    cd ~/Capstone-main-sim/deployment/policy_deploy && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python ../../sim_deploy/offline/prompting_choice.py [RUN ...]

Per prompting run (prompting_common.RUNS):

Q1 (deploy): a fresh copy of sim_deploy/stations/policy_sim.yaml pointed at the run by the copy's `policy_deploy use`
(launcher/use_policy.use, in-process); Settings.load of that copy; CheckpointLoader.load(..., "cpu"); PromptSource.load
for prompt_split train / val / all with the checkpoint's own dataset cfg + training split info (what
PromptableDiffusionPolicy.from_loaded does). Recorded: what `use` wrote, every demo of every split (prompt_index ->
dataset index -> replay-buffer task index -> episode name, steps, pick1-v1 package index), the default
(manager.task_name / prompt_index) and the warm-up demo (longest()).

Q2 (training): the training dataset rebuilt from the same cfg as train_policy_workspace.py does it
(hydra.utils.instantiate(cfg.task.dataset) with val_ratio / seed from the cfg, no split info; the .zarr.zip read in
place instead of the cluster's LMDB cache), its split compared with the checkpoint's; the training-batch prompt
dataset (only_prompt=True + the split info + the same replay buffer, as the workspace builds it for
BatchedByTaskDataset) compared index by index with PromptSource's train split; the val / train sampler's pair prompts
(sampler.prompt_indices, seed from the cfg) for the action-MSE subsets (RandomState(training.seed).choice(...,
sample_size)) and the val-loss subset; and the bit-identity check: PromptSource.get(task, i) vs the training prompt
dataset's item i (add_batch_dim, as BatchedByTaskDataset) vs a training-dataset sample whose pair prompt is that demo
(keys, shapes, dtypes, torch.equal), for the default demo, the warm-up demo and the first val demo.

Writes ~/Capstone-main-sim/sim_evals/_pipeline/W1/prompting/prompt_choice.json (all runs given; merges with the file
if it exists).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompting_common import OUT, REF_DATASET, RUNS, make_station, version_info, write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

# code references (current copy; line numbers as of this run)
CODE_REFS = {
    "use_policy": "launcher/use_policy.py choose(): prompt_dataset = training_dataset(run, cfg) = "
                  "<run>/../../package/<dataset>/<dataset>.zarr.zip; task = the dataset's only task (meta/task_names); "
                  "index = 0 if --prompt-index not given (l.228); prompt_split written only if --prompt-split given "
                  "(l.218), else config/policy.yaml prompt_split: train",
    "prompt_source": "policy/prompt_source.py PromptSource.load: hydra.instantiate(ckpt cfg.task.dataset, val_ratio=0, "
                     "cache_dir=None, only_prompt=True, training_split_info=ckpt split (train/val) or None (all)); "
                     "val = get_validation_dataset(); _dataset_index(task, i) = "
                     "get_unique_task_name_to_dataset_indices()[task][i]; get() = dataset[i]['obs']['prompt'] with a "
                     "batch dim (unsqueeze(0)) on the device; longest() = max replay_buffer.task_lengths over the split",
    "manager": "manager/episode_loop.py _key_start (l.291-292): prompt = (task_name, prompt_index) if the policy's "
               "config says uses_prompt else None -> StepWorker.submit_reset(prompt) -> worker thread: "
               "policy.reset() then policy.prompt(task_name, prompt_index) (manager/step_worker.py l.98-103), once per "
               "episode at the start key, before the PC's start and the first step; startup_sequence.py l.221-224: "
               "uses_prompt and task_name None -> startup Problem (refuses to run); prompt_index default 0 "
               "(config/manager.yaml l.58)",
    "policy_server": "policy/policy_server.py: reset -> state NEEDS_PROMPT for a PromptablePolicy (l.89); prompt before "
                     "reset / twice -> BadRequest (l.99, l.101); infer before prompt -> BadRequest 'infer before "
                     "prompt' (l.112); an unknown task / index -> BadRequest from PromptSource._dataset_index",
    "warm_up": "policy/app.py l.115-121: run_warm_up_prompt(policy) = prompt(*warm_up_prompt()) = the longest demo of "
               "the split, then warmup_inferences infers on the neutral observation, then reset() (clears it)",
    "training_pairing": "train_policy_workspace.py l.104-115: group_dataloader_by_task + pair -> a separate prompt "
                        "dataset (only_prompt, training_split_info, same replay buffer); BatchedByTaskDataset."
                        "__getitem__: ONE prompt per batch, prompt_idx = torch.randint(len(prompt_dataset[task])) "
                        "(uniform over the train demos of the batch's task, own episode included); the train sampler's "
                        "own pair prompts are ignored for the loss (set_ignore_prompt(True)); action MSE (train/val) "
                        "uses the samplers' pair prompts: sampler.py setup_indices_with_pair_prompting, "
                        "rng=default_rng(seed) choice over the SAME split's demos of the task, with replacement, fixed "
                        "for the whole run (requires_epoch_shuffle() is False once the train sampler ignores the prompt, "
                        "so shuffle_data_ordering never runs); collate_prompts pads + adds metadata.mask",
}


def _demo_table(source) -> list[dict[str, Any]]:
    ds = source._dataset
    task_idx = np.asarray(ds.sampler.prompt_indices)
    rb = ds.replay_buffer
    names = [str(n) for n in rb.episode_names[:]]
    t2e = np.asarray(ds.sampler.task_to_episode_idxs)
    rows = []
    for task, indices in source._tasks.items():
        for k, i in enumerate(indices):
            d = source.demo(task, k)
            ti = int(task_idx[i])
            rows.append({"task_name": task, "prompt_index": k, "dataset_index": int(i), "task_idx": ti,
                         "episode": d["episode"], "episode_idx": int(t2e[ti]), "steps": d["steps"],
                         "pick1_v1_index": REF_NAMES.index(names[int(t2e[ti])]) if names[int(t2e[ti])] in REF_NAMES
                         else None})
    return rows


def _cmp(a: Any, b: Any, path: str = "") -> list[str]:
    """Differences between two nested dicts of tensors (keys, shape, dtype, values)."""
    out: list[str] = []
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{path}: dict vs {type(b).__name__}"]
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: only in second")
            elif k not in b:
                out.append(f"{path}/{k}: only in first")
            else:
                out.extend(_cmp(a[k], b[k], f"{path}/{k}"))
        return out
    ta, tb = torch.as_tensor(a), torch.as_tensor(b)
    if ta.shape != tb.shape:
        return [f"{path}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}"]
    if ta.dtype != tb.dtype:
        out.append(f"{path}: dtype {ta.dtype} vs {tb.dtype}")
    if not torch.equal(ta.to(tb.dtype) if ta.dtype != tb.dtype else ta, tb):
        out.append(f"{path}: values differ, max |d| {float((ta.double() - tb.double()).abs().max()):.3g}")
    return out


def _layout(x: Any, path: str = "") -> dict[str, str]:
    if isinstance(x, dict):
        r: dict[str, str] = {}
        for k, v in x.items():
            r.update(_layout(v, f"{path}/{k}"))
        return r
    t = torch.as_tensor(x)
    return {path: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"}


def _batch(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _batch(v) for k, v in x.items()}
    return torch.as_tensor(np.array(x)).unsqueeze(0)


def _own_fraction(sampler, sel: np.ndarray) -> dict[str, Any]:
    own = sampler.indices[sel, 2] == sampler.prompt_indices[sel]
    tasks, counts = np.unique(sampler.prompt_indices[sel], return_counts=True)
    return {"n": int(len(sel)), "own_episode_as_prompt": int(own.sum()),
            "prompt_task_idx_counts": {int(t): int(c) for t, c in zip(tasks, counts)}}


def one_run(run: str) -> dict[str, Any]:
    import hydra
    import omegaconf
    from behavior_prompting.common.pytorch_util import add_batch_dim
    from policy_deploy.policy.checkpoint_loader import CheckpointLoader
    from policy_deploy.policy.prompt_source import PromptSource
    from policy_deploy.policy.settings import Settings

    t0 = time.monotonic()
    station, wrote = make_station(run)
    st = Settings.load(station, mock=True)
    loaded = CheckpointLoader(st.training_timing).load(st.checkpoint, "cpu", None)
    split_info = {str(k): bool(v) for k, v in loaded.model.get_training_split_info().items()}
    dataset_cfg = loaded.dataset_cfg
    ckpt_info = {k: loaded.info[k] for k in ("checkpoint", "epoch", "diffusion_steps_checkpoint")}
    del loaded
    gc.collect()

    task_name = wrote["manager_section"]["task_name"]
    prompt_index = wrote["manager_section"]["prompt_index"]
    sources = {s: PromptSource.load(st.prompt_dataset, s, dataset_cfg, split_info, "cpu") for s in ("train", "val", "all")}
    splits = {s: {"tasks": src.tasks(), "demos": _demo_table(src),
                  "longest": dict(zip(("task_name", "prompt_index"), src.longest())),
                  "longest_demo": src.demo(*src.longest())} for s, src in sources.items()}
    default = sources[st.prompt_split].demo(task_name, prompt_index)
    default_row = next(r for r in splits[st.prompt_split]["demos"] if r["prompt_index"] == prompt_index)

    # ---------------------------------------------------------------- training's datasets (as the workspace builds)
    cfg = omegaconf.OmegaConf.create(dict(dataset_cfg))
    trained_path = cfg.dataset_path
    cfg.dataset_path = str(st.prompt_dataset)
    cfg.cache_dir = None
    train_ds = hydra.utils.instantiate(cfg, allow_zip_file=True)
    ds_split = {str(k): bool(v) for k, v in train_ds.get_training_split_info().items()}
    val_ds = train_ds.get_validation_dataset()
    pcfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    pcfg.update(only_prompt=True, training_split_info=ds_split, replay_buffer=train_ds.replay_buffer, dataset_path=None)
    prompt_ds = hydra.utils.instantiate(pcfg)

    run_cfg = __import__("yaml").safe_load((Path(st.checkpoint).parents[1] / "run_config/config.yaml").read_text())
    tr = run_cfg["training"]
    sample_size, seed = int(tr["sample_size"]), int(tr["seed"])
    val_bs = int(run_cfg["val_dataloader"]["batch_size"])
    rng = np.random.RandomState(seed=seed)
    val_mse_sel = rng.choice(np.arange(len(val_ds)), size=sample_size, replace=True)
    rng = np.random.RandomState(seed=seed)
    train_mse_sel = rng.choice(np.arange(len(train_ds)), size=sample_size, replace=True)
    if tr.get("max_val_steps") is not None and len(val_ds) // val_bs > tr["max_val_steps"]:
        val_loss_sel = np.random.RandomState(seed=0).choice(len(val_ds), size=tr["max_val_steps"] * val_bs, replace=False)
    else:
        val_loss_sel = np.arange(len(val_ds))
    names = [str(n) for n in train_ds.replay_buffer.episode_names[:]]
    t2e = np.asarray(train_ds.sampler.task_to_episode_idxs)
    val_tasks = sorted(set(int(t) for t in val_ds.sampler.indices[:, 2]))
    training = {
        "trained_dataset_path": trained_path, "local_dataset_path": str(st.prompt_dataset),
        "val_ratio": float(cfg.val_ratio), "dataset_seed": int(cfg.seed), "training_seed": seed,
        "group_dataloader_by_task": tr["group_dataloader_by_task"], "sample_size": sample_size,
        "train_batch_size": run_cfg["dataloader"]["batch_size"], "val_batch_size": val_bs,
        "split_matches_checkpoint": ds_split == split_info,
        "n_train_demos": int(sum(split_info.values())), "n_val_demos": int(sum(not v for v in split_info.values())),
        "val_demos": [{"task_idx": t, "episode": names[t2e[t]],
                       "pick1_v1_index": REF_NAMES.index(names[t2e[t]]) if names[t2e[t]] in REF_NAMES else None}
                      for t in val_tasks],
        "train_batches_prompt_pool_equals_deploy_train_split":
            np.array_equal(np.asarray(prompt_ds.sampler.prompt_indices),
                           np.asarray(sources["train"]._dataset.sampler.prompt_indices)),
        "train_batches_prompt_pool_task_idx": np.asarray(prompt_ds.sampler.prompt_indices).tolist(),
        "val_action_mse_subset": _own_fraction(val_ds.sampler, val_mse_sel),
        "val_loss_subset": _own_fraction(val_ds.sampler, val_loss_sel),
        "train_action_mse_subset": _own_fraction(train_ds.sampler, train_mse_sel),
        "val_sampler_prompt_pool": sorted(set(int(t) for t in val_ds.sampler.prompt_indices)),
        "train_sampler_prompt_pool_size": len(set(int(t) for t in train_ds.sampler.prompt_indices)),
    }

    # ---------------------------------------------------------------- bit identity of the prompt tensors
    def check(split: str, k: int, ds, label: str) -> dict[str, Any]:
        src = sources[split]
        i = src._tasks[task_name][k]
        ti = int(src._dataset.sampler.prompt_indices[i])
        a = src.get(task_name, k)
        res: dict[str, Any] = {"label": label, "split": split, "prompt_index": k, "task_idx": ti,
                               "episode": names[t2e[ti]], "layout": _layout(a)}
        if split == "train":
            j = int(np.where(np.asarray(prompt_ds.sampler.prompt_indices) == ti)[0][0])
            b = add_batch_dim(prompt_ds[j]["obs"]["prompt"])
            res["vs_training_batch_prompt_dataset_item"] = {"item": j, "differences": _cmp(a, b)}
        hits = np.where(np.asarray(ds.sampler.prompt_indices) == ti)[0]
        if len(hits):
            s = ds[int(hits[0])]
            c = _batch(s["obs"]["prompt"])
            res["vs_training_sample_pair_prompt"] = {
                "dataset": "val" if ds is val_ds else "train", "sample": int(hits[0]),
                "sample_obs_task_idx": int(ds.sampler.indices[hits[0], 2]), "differences": _cmp(a, c)}
        else:
            res["vs_training_sample_pair_prompt"] = "no sample of that sampler pairs with this demo"
        return res

    longest_k = sources["train"].longest()[1]
    identity = [check("train", prompt_index, train_ds, "default (manager.prompt_index)")]
    if longest_k != prompt_index:
        identity.append(check("train", longest_k, train_ds, "warm-up (longest)"))
    identity.append(check("val", 0, val_ds, "val split index 0"))

    return {
        "run": run, "dataset": RUNS[run][1], "station_copy": str(station), "use_wrote": wrote,
        "settings": {"prompt_dataset": str(st.prompt_dataset), "prompt_split": st.prompt_split,
                     "precision": st.precision, "diffusion_steps": st.diffusion_steps,
                     "gripper_map": None if st.gripper_map is None else st.gripper_map.as_dict()},
        "checkpoint": ckpt_info,
        "deploy_default_prompt": {**default, "split": st.prompt_split, **{k: default_row[k] for k in
                                  ("dataset_index", "task_idx", "episode_idx", "pick1_v1_index")}},
        "warm_up_prompt": splits[st.prompt_split]["longest_demo"],
        "splits": splits, "training": training, "prompt_identity": identity,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main(argv: list[str]) -> None:
    global REF_NAMES
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    register_codecs(verbose=False)
    REF_NAMES = [str(n) for n in ReplayBuffer.create_from_path(str(REF_DATASET)).episode_names[:]]
    runs = argv or list(RUNS)
    path = OUT / "prompt_choice.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result["version"] = version_info()
    result["code_refs"] = CODE_REFS
    result.setdefault("runs", {})
    for run in runs:
        print(f"== {run}", flush=True)
        r = one_run(run)
        result["runs"][run] = r
        print(json.dumps({"default": r["deploy_default_prompt"], "warm": r["warm_up_prompt"],
                          "split_ok": r["training"]["split_matches_checkpoint"],
                          "pool_eq": r["training"]["train_batches_prompt_pool_equals_deploy_train_split"],
                          "identity": [(x["label"], x.get("vs_training_batch_prompt_dataset_item", {}).get("differences"),
                                        x["vs_training_sample_pair_prompt"] if isinstance(x["vs_training_sample_pair_prompt"], str)
                                        else x["vs_training_sample_pair_prompt"]["differences"]) for x in r["prompt_identity"]],
                          "s": r["seconds"]}, default=str), flush=True)
        write_json(path, result)
        gc.collect()
    print("wrote", path)


REF_NAMES: list[str] = []

if __name__ == "__main__":
    main(sys.argv[1:])
