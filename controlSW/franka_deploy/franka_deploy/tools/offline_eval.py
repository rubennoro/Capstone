"""Offline fidelity: the training action-MSE metric, reproduced, plus per-step errors in physical units.

    python -m franka_deploy.tools.offline_eval --ckpt X.ckpt --work-dir DIR --dataset-path P.zarr.zip \
        [--seeds 0 1 2 3 4] [--precision fp32|bf16] [--out eval.json]

Reproduces train_policy_workspace.py's `val/action_mse_error/all` (the checkpoint monitor metric):
- dataset = cfg.task.dataset (training_split_info=None -> the same 0.05 val split; checked against the checkpoint);
- val set = dataset.get_validation_dataset(); prompts paired by the sampler exactly as in training (pair mode);
- indices = np.random.RandomState(cfg.training.seed).choice(len(split), cfg.training.sample_size, replace=True);
- DataLoader(batch_size=cfg.val_dataloader.batch_size, shuffle=False, collate_fn=collate_prompts);
- pred = policy.predict_action_training(batch['obs'])['action_pred'] (prompt inside the batch, policy unprompted);
- MSE = torch.nn.functional.mse_loss over all (N,16,10) UNNORMALIZED values, plus per component.
Differences from training (documented): num_workers=0 (the .zarr.zip is read directly; forked workers would share
one zip handle), and the diffusion noise comes from torch.manual_seed(seed) per evaluation, not training's RNG state.
At epoch 50 training had not reshuffled the val ordering (requires_epoch_shuffle() is False for the train dataset
because group_dataloader_by_task sets ignore_prompt=True), so the initial sampler ordering is the right one.

Per-step errors at k in (0,4,8,15): position |dp| mm, rotation geodesic deg (rot6d -> matrix), width |dw| mm.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time

import numpy as np


def rot6d_to_mat_np(d6: np.ndarray) -> np.ndarray:
    from behavior_prompting.common.pose_util import rot6d_to_mat

    return rot6d_to_mat(d6)


def per_step_errors(pred: np.ndarray, gt: np.ndarray, ks=(0, 4, 8, 15)) -> dict:
    """pred, gt: (N, 16, 10) unnormalized relative actions (pos3, rot6d, width)."""
    out = {}
    for k in ks:
        dp = np.linalg.norm(pred[:, k, :3] - gt[:, k, :3], axis=-1) * 1000
        Rp, Rg = rot6d_to_mat_np(pred[:, k, 3:9]), rot6d_to_mat_np(gt[:, k, 3:9])
        cos = (np.einsum("nij,nij->n", Rp, Rg) - 1) / 2  # trace(Rp^T Rg) = sum_ij Rp_ij Rg_ij
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        dw = np.abs(pred[:, k, 9] - gt[:, k, 9]) * 1000
        gt_disp = np.linalg.norm(gt[:, k, :3], axis=-1) * 1000
        out[f"k{k}"] = {
            "pos_mm_median": float(np.median(dp)), "pos_mm_mean": float(dp.mean()), "pos_mm_p90": float(np.percentile(dp, 90)),
            "rot_deg_median": float(np.median(ang)), "rot_deg_mean": float(ang.mean()),
            "width_mm_median": float(np.median(dw)), "width_mm_mean": float(dw.mean()),
            "gt_displacement_mm_median": float(np.median(gt_disp)),
        }
    return out


def mse_components(pred, gt, components) -> dict:
    import torch

    p, g = torch.as_tensor(pred), torch.as_tensor(gt)
    out = {"all": torch.nn.functional.mse_loss(p, g).item()}
    d = 0
    for c in components:
        out[c["type"]] = torch.nn.functional.mse_loss(p[..., d:d + c["size"]], g[..., d:d + c["size"]]).item()
        d += c["size"]
    return out


def build_eval_datasets(cfg, dataset_path: str):
    import hydra
    from omegaconf import OmegaConf
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

    register_codecs(verbose=False)
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    dcfg.cache_dir = None
    ds = hydra.utils.instantiate(dcfg, dataset_path=dataset_path, allow_zip_file=True)
    val = ds.get_validation_dataset()
    ds.set_ignore_prompt(False)
    val.set_ignore_prompt(False)
    return ds, val


def _slice(tree, a, b):
    import torch

    if isinstance(tree, dict):
        return {k: _slice(v, a, b) for k, v in tree.items()}
    if torch.is_tensor(tree) or isinstance(tree, np.ndarray):
        return tree[a:b]
    if isinstance(tree, list):
        return tree[a:b]
    return tree


def run_metric(policy, split_ds, cfg, seed: int, precision: str = "fp32", collect=False, micro_batch: int = 16):
    import torch
    from torch.utils.data import DataLoader, Subset
    from behavior_prompting.train_network.utils.prompt_util import collate_prompts
    from behavior_prompting.common.pytorch_util import dict_apply

    rng = np.random.RandomState(seed=cfg.training.seed)
    idx = rng.choice(list(range(len(split_ds))), size=cfg.training.sample_size, replace=True)
    dl = DataLoader(Subset(split_ds, idx), batch_size=cfg.val_dataloader.batch_size, shuffle=False, num_workers=0,
                    drop_last=False, collate_fn=collate_prompts)
    torch.manual_seed(seed)
    preds, gts, batches = [], [], []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
    n = 0
    for batch in dl:
        B = batch["action"].shape[0]
        parts = []
        for a in range(0, B, micro_batch):  # micro-batching only to fit a 4 GB GPU; samples are independent in eval
            obs = _slice(batch["obs"], a, a + micro_batch)
            obs = dict_apply(obs, lambda x: x.to(policy.device) if torch.is_tensor(x) else x)
            with torch.inference_mode(), ctx:
                parts.append(policy.predict_action_training(obs)["action_pred"].float().cpu())
        pred = torch.cat(parts)
        gt = batch["action"].cpu()
        mask = batch.get("metadata", {}).get("action_mask") if isinstance(batch.get("metadata"), dict) else None
        if mask is not None:
            for b in range(gt.shape[0]):
                preds.append(pred[b, int(mask[b]):].unsqueeze(0))
                gts.append(gt[b, int(mask[b]):].unsqueeze(0))
        else:
            preds.append(pred)
            gts.append(gt)
        if collect:
            batches.append(batch)
        n += gt.shape[0]
        if n >= cfg.training.sample_size:
            break
    return torch.cat(preds).numpy(), torch.cat(gts).numpy(), idx, batches


def per_demo_eval(policy, ds, val, cfg, comps, n_per, seed, precision, micro_batch, split_info):
    """Per episode: n_per evenly spaced samples, training metric + per-step errors."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from behavior_prompting.common.pytorch_util import dict_apply
    from behavior_prompting.train_network.utils.prompt_util import collate_prompts

    names = [str(x) for x in ds.replay_buffer.meta["episode_names"][:]] if "episode_names" in ds.replay_buffer.meta else None
    split = {k.split(":")[0]: bool(v) for k, v in (split_info or {}).items()}
    out = []
    for name_split, d in (("train", ds), ("val", val)):
        ep_of = np.array([int(x[1]) for x in d.sampler.indices])
        for ep in sorted(set(ep_of.tolist())):
            idx = np.where(ep_of == ep)[0]
            sel = idx[np.linspace(0, len(idx) - 1, min(n_per, len(idx))).astype(int)]
            dl = DataLoader(Subset(d, sel), batch_size=micro_batch, shuffle=False, num_workers=0, collate_fn=collate_prompts)
            torch.manual_seed(seed)
            P, G = [], []
            for batch in dl:
                obs = dict_apply(batch["obs"], lambda x: x.to(policy.device) if torch.is_tensor(x) else x)
                with torch.inference_mode():
                    P.append(policy.predict_action_training(obs)["action_pred"].float().cpu().numpy())
                G.append(batch["action"].numpy())
            P, G = np.concatenate(P), np.concatenate(G)
            nm = names[ep] if names else str(ep)
            out.append({"episode": nm, "dataset_split": name_split,
                        "checkpoint_split": ("train" if split.get(nm) else "val") if nm in split else "new",
                        "n": int(len(sel)), "mse": mse_components(P, G, comps), "per_step": per_step_errors(P, G)})
            print(json.dumps({"episode": nm, "split": out[-1]["checkpoint_split"], "mse_all": out[-1]["mse"]["all"]}), flush=True)
    return out


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-demo", type=int, default=0,
                    help="also evaluate N samples per demo (every episode of the zarr, train and val; works on any "
                         "zarr incl. future rig recordings: episodes not in the checkpoint split are reported as 'new')")
    a = ap.parse_args(argv)
    import os
    a.out = os.path.abspath(a.out) if a.out else None  # build_node chdirs into --work-dir
    a.dataset_path = os.path.abspath(a.dataset_path)
    from omegaconf import OmegaConf
    from franka_deploy.inference.server import build_node

    node, rep = build_node(a.ckpt, a.work_dir, None, f"tcp://127.0.0.1:{_free_port()}", a.device)
    policy, cfg = node.policy, node.cfg
    policy.reset()
    ds, val = build_eval_datasets(cfg, a.dataset_path)
    split_ok = {k: bool(v) for k, v in ds.get_training_split_info().items()} == \
        {k: bool(v) for k, v in policy.get_training_split_info().items()}
    comps = OmegaConf.to_container(cfg.task.shape_meta.action.components, resolve=True)
    res = {"ckpt_epoch": node.epoch, "split_matches_checkpoint": split_ok, "precision": a.precision,
           "sample_size": int(cfg.training.sample_size), "index_seed": int(cfg.training.seed),
           "batch_size": int(cfg.val_dataloader.batch_size), "micro_batch": a.micro_batch, "len_train": len(ds), "len_val": len(val),
           "noise_seeds": a.seeds, "splits": {}}
    for name, split in (("val", val), ("train", ds)):
        per_seed, all_p, all_g = [], [], []
        t0 = time.time()
        for s in a.seeds:
            p, g, idx, _ = run_metric(policy, split, cfg, s, a.precision, micro_batch=a.micro_batch)
            per_seed.append(mse_components(p, g, comps))
            all_p.append(p)
            all_g.append(g)
        P, G = np.concatenate(all_p), np.concatenate(all_g)
        res["splits"][name] = {
            "mse_all_per_seed": [x["all"] for x in per_seed],
            "mse_all_mean": float(np.mean([x["all"] for x in per_seed])),
            "mse_components_mean": {k: float(np.mean([x[k] for x in per_seed])) for k in per_seed[0]},
            "episodes_sampled": sorted({int(split.sampler.indices[i][1]) for i in idx}),
            "per_step": per_step_errors(P, G), "seconds": round(time.time() - t0, 1),
        }
        print(name, json.dumps(res["splits"][name]["mse_components_mean"]), flush=True)
    if a.per_demo:
        res["per_demo"] = per_demo_eval(policy, ds, val, cfg, comps, a.per_demo, a.seeds[0], a.precision, a.micro_batch,
                                        policy.get_training_split_info())
    print(json.dumps({k: v for k, v in res.items() if k != "per_demo"}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
