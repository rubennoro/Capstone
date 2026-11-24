"""W1 step 2 (S2), Q3 + Q4: deploy prompting (model.prompt) vs training's prompt-in-batch, and the prompt-index sweep.

Run from the copy (GPU; takes the shared GPU lock only for the CUDA part of each run):

    cd ~/Capstone-main-sim/deployment/policy_deploy && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python ../../sim_deploy/offline/prompting_sweep.py gt          # CPU, once
    ~/envs/sim_policy_deploy_torch/bin/python ../../sim_deploy/offline/prompting_sweep.py run RUN ... # GPU
    ~/envs/sim_policy_deploy_torch/bin/python ../../sim_deploy/offline/prompting_sweep.py summary     # CPU

`gt` (CPU): B1's 15 references (sim_evals/_camera_study/B/reference/ep*/obs.npz) against behavior_prompting's own
UmiTaskDataset over pick1-v1 (val_ratio 0, the sample at the reference's row, prompt ignored): the obs keys must be
identical (images and low-dim, the training-exact check of B1 by a second method); the sample's `action` (16, 10:
pos, rot6d, width; relative to the pose at T, rows T+3k, padded as training) is the ground truth. -> gt_refs.npz.

`run RUN` per prompting run, model fp32 at the checkpoint's own diffusion steps, gripper_map None (the references are
in training units; the model's raw output is used, ModelRunner.run, before make_chunk's clip / to_robot):
- CPU, before the lock: the station copy (`policy_deploy use`), Settings, CheckpointLoader.load on the CPU, the
  training datasets (as prompting_choice.py) and 4 Q3 samples (2 val, 2 train, with their pair prompts).
- GPU (lock held): the model moved to cuda + reset (what CheckpointLoader.load(..., "cuda") does), then
  PromptableDiffusionPolicy.from_loaded (train split, the deploy default) and a second PromptableDiffusionPolicy on
  the same model + runner with a val-split PromptSource.
  Q3: deploy = policy.reset(); policy.prompt(task, k); manual_seed(s); runner.run(obs) vs training = model.reset();
  manual_seed(s); model.predict_action_training(collate_prompts([sample])["obs"]) (the val metric's path, with
  metadata.mask) and with the prompt as BatchedByTaskDataset gives it (add_batch_dim, no mask); deploy twice (the CUDA
  floor); reset() then a predict without a prompt; precision bf16: autocast state and dtypes inside the prompt and the
  predict (forward hooks on the prompt tokenizer and the UNet), prompt cache dtype.
  Batch check: model.predict_action on the 15 references stacked (B=15) vs one by one, same seed.
  Q4: every demo of the train split and of the val split as the prompt (together = prompt_split all), x the 15
  references x seeds 0, 1, 2 (manual_seed right before each runner.run). Raw chunks -> prompt_sweep_<run>.npz.
- CPU after: the metrics -> prompt_sweep_<run>.json.

`remetric RUN ...` (CPU): recompute the metrics of prompt_sweep_<run>.json from its .npz.
`summary`: summary.json (Q3 per run + the per-run headline: default prompt, its approach and error vs the
best / worst / median prompt).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompting_common import (OUT, REF_DATASET, REF_DIR, RUNS, gpu_lock, make_station, run_folder,  # noqa: E402
                              version_info, write_json)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
try:
    import cv2
    cv2.setNumThreads(4)
except ImportError:
    pass

KS = (0, 7, 15)
SEEDS = (0, 1, 2)
C4_TO_MODEL = {"eef_pos": "gripper_right_eef_pos", "eef_rot6d": "gripper_right_eef_rot_axis_angle",
               "eef_rot6d_wrt_start": "gripper_right_eef_rot_axis_angle_wrt_start",
               "gripper_width_m": "gripper_right_gripper_width", "main": "camera_right_main_rgb",
               "ultrawide": "camera_right_ultrawide_rgb"}
IMG = ("main", "ultrawide")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def refs() -> list[tuple[str, dict]]:
    out = []
    for d in sorted(REF_DIR.glob("ep*_*")):
        if (d / "obs.npz").exists():
            out.append((d.name, json.loads((d / "meta.json").read_text())))
    return out


def run_cfg_dataset(run: str) -> dict:
    import omegaconf
    cfg = omegaconf.OmegaConf.load(run_folder(run) / "run_config/config.yaml")
    return omegaconf.OmegaConf.to_container(cfg.task.dataset, resolve=True)


# ------------------------------------------------------------------------------------------------------------- gt
def cmd_gt() -> None:
    import hydra
    import omegaconf
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    register_codecs(verbose=False)
    cfg = omegaconf.OmegaConf.create(run_cfg_dataset("run_006_2026-11-18"))
    cfg.dataset_path, cfg.cache_dir, cfg.val_ratio = str(REF_DATASET), None, 0.0
    ds = hydra.utils.instantiate(cfg, allow_zip_file=True)
    ds.set_ignore_prompt(True)
    ind = ds.sampler.indices
    t2e = np.asarray(ds.sampler.task_to_episode_idxs)
    names, gts, checks = [], [], {}
    for name, meta in refs():
        pkg, row = int(meta["package_index"]), int(meta["row_in_episode"])
        task = int(np.where(t2e == pkg)[0][0])
        idx = int(np.where((ind[:, 2] == task) & (ind[:, 0] == row))[0][0])
        s = ds[idx]
        obs = np.load(REF_DIR / name / "obs.npz")
        diffs = {}
        for f, mk in C4_TO_MODEL.items():
            if f == "eef_rot6d_wrt_start":
                continue                       # training adds N(0, 0.05) noise to the start pose (ignore_by_policy)
            v = s["obs"][mk].numpy()
            if f in IMG:
                v = np.moveaxis(np.round(v * 255.0).astype(np.uint8), 1, -1)
                diffs[f] = int(np.abs(v.astype(int) - obs[f].astype(int)).max())
            else:
                diffs[f] = float(np.abs(v - obs[f]).max())
        checks[name] = {"dataset_index": idx, "task_idx": task, "row": row,
                        "episode": str(ds.replay_buffer.episode_names[t2e[task]]), "max_abs_diff_vs_obs_npz": diffs}
        names.append(name)
        gts.append(s["action"].numpy())
        log(name, checks[name])
    np.savez(OUT / "gt_refs.npz", names=np.array(names), action=np.stack(gts))
    write_json(OUT / "gt_refs.json", {"version": version_info(), "dataset": str(REF_DATASET), "refs": checks,
                                      "note": "action = UmiTaskDataset sample['action'] at the reference row "
                                              "(relative to the pose at T, rows T+3k, k=0..15)"})


# --------------------------------------------------------------------------------------------------------- helpers
def obs_from_sample(sample: dict):
    """C4 Observation from a training sample's receding obs (float32 images /255 -> uint8 exactly)."""
    from policy_deploy.contracts.c4_policy import Observation
    kw = {}
    for f, mk in C4_TO_MODEL.items():
        v = sample["obs"][mk].numpy()
        if f in IMG:
            kw[f] = np.ascontiguousarray(np.moveaxis(np.round(v * 255.0).astype(np.uint8), 1, -1))
        else:
            kw[f] = np.ascontiguousarray(v.astype(np.float32))
    return Observation(**kw)


def obs_from_npz(name: str):
    from policy_deploy.contracts.c4_policy import Observation
    d = np.load(REF_DIR / name / "obs.npz")
    return Observation(**{k: np.ascontiguousarray(d[k]) for k in d.files})


def chunk10(parts) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts], axis=-1)      # pos, rot6d, width


def to_dev(x, dev):
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x.to(dev) if torch.is_tensor(x) else x


def r6mat(d6: np.ndarray) -> np.ndarray:
    """behavior_prompting's rot6d_to_mat on any leading shape (its normalize() only broadcasts over 2-D input)."""
    from behavior_prompting.common.pose_util import rot6d_to_mat
    d6 = np.asarray(d6, np.float64)
    return rot6d_to_mat(d6.reshape(-1, 6)).reshape(d6.shape[:-1] + (3, 3))


def rot_deg(a6: np.ndarray, b6: np.ndarray) -> np.ndarray:
    ra, rb = r6mat(a6), r6mat(b6)
    tr = np.einsum("...ij,...ij->...", ra, rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# ------------------------------------------------------------------------------------------------------------- run
def cmd_run(run: str) -> None:
    import hydra
    import omegaconf
    from behavior_prompting.common.pytorch_util import add_batch_dim
    from behavior_prompting.train_network.utils.prompt_util import collate_prompts
    from policy_deploy.policy.checkpoint_loader import CheckpointLoader
    from policy_deploy.policy.model_runner import ModelRunner
    from policy_deploy.policy.key_map import KeyMap
    from policy_deploy.policy.prompt_source import PromptSource
    from policy_deploy.policy.promptable_diffusion_policy import PromptableDiffusionPolicy
    from policy_deploy.policy.settings import Settings
    from deploy_contracts import ContractLimits

    t_start = time.monotonic()
    station, wrote = make_station(run)
    st0 = Settings.load(station, mock=True)
    st = dataclasses.replace(st0, device="cuda", precision="fp32", diffusion_steps=None, gripper_map=None)
    task = wrote["manager_section"]["task_name"]
    default_k = wrote["manager_section"]["prompt_index"]
    log(run, "loading checkpoint on the CPU")
    loaded = CheckpointLoader(st.training_timing).load(st.checkpoint, "cpu", None)
    model = loaded.model
    split_info = {str(k): bool(v) for k, v in model.get_training_split_info().items()}

    # training datasets + Q3 samples (CPU)
    cfg = omegaconf.OmegaConf.create(dict(loaded.dataset_cfg))
    cfg.dataset_path, cfg.cache_dir = str(st.prompt_dataset), None
    train_ds = hydra.utils.instantiate(cfg, allow_zip_file=True)
    val_ds = train_ds.get_validation_dataset()
    q3_samples = []
    for name, ds in (("val", val_ds), ("train", train_ds)):
        smp = ds.sampler
        own = np.where(smp.indices[:, 2] == smp.prompt_indices)[0]
        other = np.where(smp.indices[:, 2] != smp.prompt_indices)[0]
        picks = ([int(own[len(own) // 2])] if len(own) else []) + ([int(other[len(other) // 3])] if len(other) else [])
        for i in picks[:2]:
            s = ds[i]
            q3_samples.append({"dataset": name, "index": i, "obs_task_idx": int(smp.indices[i, 2]),
                               "row": int(smp.indices[i, 0]), "prompt_task_idx": int(smp.prompt_indices[i]),
                               "sample": s})
    rb_names = [str(n) for n in train_ds.replay_buffer.episode_names[:]]
    del train_ds, val_ds
    gc.collect()

    ref_list = refs()
    gt = np.load(OUT / "gt_refs.npz")
    assert list(gt["names"]) == [n for n, _ in ref_list]
    ref_obs = [obs_from_npz(n) for n, _ in ref_list]

    result: dict[str, Any] = {"run": run, "dataset": RUNS[run][1], "version": version_info(),
                              "settings_used": {"device": "cuda", "precision": "fp32", "diffusion_steps":
                                                "checkpoint's own", "gripper_map": None,
                                                "station_precision": st0.precision,
                                                "station_diffusion_steps": st0.diffusion_steps},
                              "task_name": task, "default_prompt_index": default_k}
    with gpu_lock(log):
        t_gpu = time.monotonic()
        model.eval().to(torch.device("cuda"))
        model.reset()
        result["diffusion_steps_in_use"] = int(model.num_inference_steps)
        pol = PromptableDiffusionPolicy.from_loaded(loaded, st)                  # train split (deploy default)
        runner = pol._runner
        val_src = PromptSource.load(st.prompt_dataset, "val", loaded.dataset_cfg, split_info, "cuda")
        polv = PromptableDiffusionPolicy(runner, model, val_src, ContractLimits.load().gripper_width_max_m,
                                         info=loaded.info, gripper_map=None)
        src_of = {"train": pol._source, "val": val_src}
        task_to_prompt = {}
        for sp, src in src_of.items():
            pidx = np.asarray(src._dataset.sampler.prompt_indices)
            for k, i in enumerate(src._tasks[task]):
                task_to_prompt[int(pidx[i])] = (sp, k)

        def deploy_prompt(sp: str, k: int):
            (pol if sp == "train" else polv).reset()
            (pol if sp == "train" else polv).prompt(task, k)

        def deploy_run(obs, seed: int) -> np.ndarray:
            torch.manual_seed(seed)
            return chunk10(runner.run(obs))

        # ------------------------------------------------------------------ Q3
        q3 = []
        for q in q3_samples:
            s = q["sample"]
            obs = obs_from_sample(s)
            sp, k = task_to_prompt[q["prompt_task_idx"]]
            rec = {key: q[key] for key in ("dataset", "index", "obs_task_idx", "row", "prompt_task_idx")}
            rec.update(obs_episode=rb_names[q["obs_task_idx"]], prompt_episode=rb_names[q["prompt_task_idx"]],
                       prompt_split=sp, prompt_index=k, seeds={})
            for seed in (0, 1):
                deploy_prompt(sp, k)
                d1 = deploy_run(obs, seed)
                d2 = deploy_run(obs, seed)
                model.reset()
                batch = to_dev(collate_prompts([s]), "cuda")
                torch.manual_seed(seed)
                with torch.inference_mode():
                    t1 = model.predict_action_training(batch["obs"])["action_pred"][0, :16].float().cpu().numpy()
                model.reset()
                ob2 = {kk: v.unsqueeze(0) for kk, v in s["obs"].items() if kk != "prompt"}
                ob2["prompt"] = add_batch_dim(s["obs"]["prompt"])
                torch.manual_seed(seed)
                with torch.inference_mode():
                    t2 = model.predict_action_training(to_dev(ob2, "cuda"))["action_pred"][0, :16].float().cpu().numpy()
                gtA = s["action"].numpy()[:16]
                rec["seeds"][seed] = {
                    "deploy_repeat_max_abs": float(np.abs(d1 - d2).max()),
                    "deploy_vs_training_collate_max_abs": float(np.abs(d1 - t1).max()),
                    "deploy_vs_training_batchprompt_max_abs": float(np.abs(d1 - t2).max()),
                    "deploy_vs_training_collate_pos_mm_max": float(np.linalg.norm(d1[:, :3] - t1[:, :3], axis=-1).max() * 1e3),
                    "pos_err_vs_gt_mm_k15": float(np.linalg.norm(d1[15, :3] - gtA[15, :3]) * 1e3),
                }
            q3.append(rec)
            log("Q3", rec["dataset"], rec["obs_episode"][-6:], "prompt", rec["prompt_episode"][-6:], rec["seeds"])
        # seed sensitivity scale for reference: same obs, seed 0 vs 1 (deploy)
        # reset clears the prompt?
        pol.reset()
        enc = model.obs_encoder
        reset_info = {"is_prompted_after_reset": bool(enc.is_prompted),
                      "prompt_cache_is_none": enc.prompt_tokens_cache is None}
        try:
            runner.run(ref_obs[0])
            reset_info["predict_without_prompt"] = "no error (!)"
        except BaseException as e:  # noqa: BLE001
            reset_info["predict_without_prompt"] = f"{type(e).__name__}: {str(e)[:200]}"
        # bf16: autocast state and dtypes during prompt and predict
        seen: list[tuple[str, bool, str]] = []
        phase = {"now": ""}

        def hook(name):
            def f(_m, _i, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                try:
                    on = torch.is_autocast_enabled("cuda")
                except TypeError:
                    on = torch.is_autocast_enabled()
                seen.append((phase["now"] + ":" + name, bool(on), str(o.dtype)))
            return f
        hs = [enc.obs_encoder.register_forward_hook(hook("prompt_tokenizer")),
              model.model.register_forward_hook(hook("unet"))]
        runner_bf = ModelRunner(model, loaded.shape_meta, KeyMap.from_config(st.key_map), torch.device("cuda"),
                                precision="bf16")
        pol_bf = PromptableDiffusionPolicy(runner_bf, model, pol._source, ContractLimits.load().gripper_width_max_m,
                                           info=loaded.info, gripper_map=None)
        pol_bf.reset()
        phase["now"] = "prompt"
        pol_bf.prompt(task, default_k)
        cache_dtype = str(enc.prompt_tokens_cache.dtype)
        phase["now"] = "predict"
        torch.manual_seed(0)
        bf = chunk10(runner_bf.run(ref_obs[0]))
        for h in hs:
            h.remove()
        deploy_prompt("train", default_k)
        fp = deploy_run(ref_obs[0], 0)
        uniq = sorted(set(seen))
        bf16_info = {"hook_records (phase:module, autocast_enabled, out dtype)": [list(u) for u in uniq],
                     "prompt_tokens_cache_dtype": cache_dtype,
                     "bf16_vs_fp32_same_seed_ref0_pos_mm_max": float(np.linalg.norm(bf[:, :3] - fp[:, :3], -1).max() * 1e3)}
        log("reset", reset_info, "bf16", bf16_info)

        # ------------------------------------------------------------------ batch check
        deploy_prompt("train", default_k)
        one = np.stack([deploy_run(o, 0) for o in ref_obs])
        tens = [runner.model_inputs(o) for o in ref_obs]
        stacked = {kk: torch.from_numpy(np.stack([t[kk] for t in tens])).to("cuda") for kk in tens[0]}
        torch.manual_seed(0)
        with torch.inference_mode():
            many = model.predict_action(stacked)["action_pred"][:, :16].float().cpu().numpy()
        batch_info = {"max_abs_sample0": float(np.abs(one[0] - many[0]).max()),
                      "max_abs_all": float(np.abs(one - many).max()),
                      "decision": "batch 1 (per-sample noise differs with B>1)"}
        log("batch", batch_info)

        # ------------------------------------------------------------------ Q4 sweep
        prompts = [("train", k) for k in range(len(pol._source._tasks[task]))] + \
                  [("val", k) for k in range(len(val_src._tasks[task]))]
        chunks = np.zeros((len(prompts), len(ref_obs), len(SEEDS), 16, 10), np.float32)
        pinfo = []
        t_sw = time.monotonic()
        for pi, (sp, k) in enumerate(prompts):
            deploy_prompt(sp, k)
            demo = (pol if sp == "train" else polv).last_prompt
            for oi, o in enumerate(ref_obs):
                for si, seed in enumerate(SEEDS):
                    chunks[pi, oi, si] = deploy_run(o, seed)
            pinfo.append({"split": sp, "prompt_index": k, "episode": demo["episode"], "steps": demo["steps"],
                          "prompt_s": demo["prompt_s"]})
            if pi == 0 or pi % 5 == 4:
                el = time.monotonic() - t_sw
                log(f"sweep {pi + 1}/{len(prompts)} {el:.0f} s ({el / ((pi + 1) * len(ref_obs) * len(SEEDS)):.3f} s/inf)")
        sweep_s = time.monotonic() - t_sw
        del pol, polv, pol_bf, runner, runner_bf
        model.to("cpu")
        torch.cuda.empty_cache()
        gpu_s = time.monotonic() - t_gpu
    del model, loaded
    gc.collect()

    np.savez_compressed(OUT / f"prompt_sweep_{run}.npz", chunks=chunks, gt=gt["action"][:, :16],
                        refs=np.array([n for n, _ in ref_list]),
                        prompts=np.array([f"{p['split']}:{p['prompt_index']}:{p['episode']}" for p in pinfo]))
    result.update(q3=q3, reset=reset_info, bf16=bf16_info, batch=batch_info, gpu_s=round(gpu_s, 1),
                  sweep_s=round(sweep_s, 1), total_s=round(time.monotonic() - t_start, 1),
                  prompts=pinfo, default_prompt_index=default_k)
    write_json(OUT / f"prompt_sweep_{run}.json", result)              # Q3 first, in case the metrics fail
    result.update(metrics(chunks, gt["action"][:, :16], ref_list, pinfo, rb_names, split_info, default_k))
    write_json(OUT / f"prompt_sweep_{run}.json", result)
    log("wrote", OUT / f"prompt_sweep_{run}.json")


# --------------------------------------------------------------------------------------------------------- metrics
def _sd_pos(x: np.ndarray, axis: int) -> np.ndarray:
    """sqrt(sum over xyz of the unbiased variance along `axis`), mm."""
    return np.sqrt(np.var(x, axis=axis, ddof=1).sum(-1)) * 1e3


def _sd_rot(r6: np.ndarray, axis: int) -> np.ndarray:
    """RMS angle (deg) to the chordal mean along `axis` (unbiased n-1)."""
    R = r6mat(r6)                                                      # (..., 3, 3)
    M = R.mean(axis=axis, keepdims=True)
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    Rm = (U * D[..., None, :]) @ Vt
    tr = np.einsum("...ij,...ij->...", R, np.broadcast_to(Rm, R.shape))
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    n = R.shape[axis]
    return np.sqrt((ang ** 2).sum(axis=axis) / (n - 1))


def metrics(chunks, gt, ref_list, pinfo, rb_names, split_info, default_k) -> dict[str, Any]:
    P, O, S = chunks.shape[:3]
    names = [n for n, _ in ref_list]
    ref_eps = [m["pi_name"] for _, m in ref_list]
    train_eps = {k.split(":")[0] for k, v in split_info.items() if v}
    val_eps = {k.split(":")[0] for k, v in split_info.items() if not v}
    ref_status = {n: ("train" if e in train_eps else "val" if e in val_eps else "not in this run's dataset")
                  for n, e in zip(names, ref_eps)}
    out: dict[str, Any] = {"refs": names, "ref_episode_status": ref_status,
                           "prompts": pinfo, "ks": list(KS), "seeds": list(SEEDS)}
    # (a) spread
    spread = {}
    for k in KS:
        x = chunks[:, :, :, k]                                   # P, O, S, 10
        sp = {
            "across_prompts_fixed_seed": {   # sd over prompts at a fixed seed, mean over seeds and obs
                "pos_mm": float(_sd_pos(x[..., :3], 0).mean()), "rot_deg": float(_sd_rot(x[..., 3:9], 0).mean()),
                "width_mm": float((np.std(x[..., 9], axis=0, ddof=1) * 1e3).mean())},
            "across_seeds_fixed_prompt": {   # sd over seeds at a fixed prompt, mean over prompts and obs
                "pos_mm": float(_sd_pos(x[..., :3], 2).mean()), "rot_deg": float(_sd_rot(x[..., 3:9], 2).mean()),
                "width_mm": float((np.std(x[..., 9], axis=2, ddof=1) * 1e3).mean())},
        }
        sp["per_ref_across_prompts_pos_mm"] = dict(zip(names, np.round(_sd_pos(x[..., :3], 0).mean(-1), 2).tolist()))
        sp["per_ref_across_seeds_pos_mm"] = dict(zip(names, np.round(_sd_pos(x[..., :3], 2).mean(0), 2).tolist()))
        spread[f"k{k}"] = sp
    out["spread"] = spread
    # (b) approach at *_first: k=15 position in the anchor TCP frame (cm)
    first = [i for i, n in enumerate(names) if n.endswith("_first")]
    appr = {}
    for i in first:
        z = chunks[:, i, :, 15, 2] * 100                         # P, S
        appr[names[i]] = {"per_prompt_z_cm_mean": np.round(z.mean(1), 2).tolist(),
                          "per_prompt_z_cm_seeds": np.round(z, 2).tolist(),
                          "per_prompt_xyz_cm_mean": np.round(chunks[:, i, :, 15, :3].mean(1) * 100, 2).tolist(),
                          "gt_xyz_cm": np.round(gt[i, 15, :3] * 100, 2).tolist()}
    out["approach_first"] = appr
    # (c) error vs the recorded future
    perr = np.linalg.norm(chunks[..., :3] - gt[None, :, None, :, :3], axis=-1) * 1e3    # P, O, S, 16
    rerr = rot_deg(chunks[..., 3:9], np.broadcast_to(gt[None, :, None, :, 3:9], chunks[..., 3:9].shape))
    werr = np.abs(chunks[..., 9] - gt[None, :, None, :, 9]) * 1e3
    err = {}
    for k in KS:
        err[f"k{k}"] = {"pos_mm": perr[..., k].mean((1, 2)).round(2).tolist(),
                        "rot_deg": rerr[..., k].mean((1, 2)).round(3).tolist(),
                        "width_mm": werr[..., k].mean((1, 2)).round(3).tolist()}
    err["all_k_pos_mm"] = perr.mean((1, 2, 3)).round(2).tolist()
    # widths, two conventions (coordinator addendum): A = sim D8 pass-through (model width = command), B = upstream
    # policy.gripper_map (to_robot on the output). The references' recorded widths (0.0846-0.0882 m) are inside
    # [0.051, 0.090], so the model input is identical under A and B; only the command differs.
    from policy_deploy.policy.gripper_map import GripperMap
    gm = GripperMap(training_closed_m=0.051, training_open_m=0.090, robot_closed_m=0.030, robot_open_m=0.040)
    wB = gm.to_robot(chunks[..., 9])
    gB = gm.to_robot(gt[..., 9])
    err["width_conventions"] = {
        "note": "width_mm above = |model - recorded| in training units (= convention A command error); B: the "
                "robot command to_robot(model width) and its error vs to_robot(recorded), mm",
        "gripper_map_B": gm.as_dict(),
        **{f"k{k}": {"model_width_mm_mean": (chunks[..., k, 9].mean((1, 2)) * 1e3).round(2).tolist(),
                     "robot_cmd_B_mm_mean": (wB[..., k].mean((1, 2)) * 1e3).round(2).tolist(),
                     "robot_cmd_B_err_mm": (np.abs(wB[..., k] - gB[None, :, None, k]).mean((1, 2)) * 1e3).round(3).tolist()}
           for k in KS}}
    err["per_ref_k15_pos_mm"] = {n: perr[:, i, :, 15].mean(1).round(2).tolist() for i, n in enumerate(names)}
    # own-episode prompt
    own = [[pinfo[p]["episode"] == ref_eps[o] for o in range(O)] for p in range(P)]
    err["own_episode_pairs"] = [(pinfo[p]["split"], pinfo[p]["prompt_index"], names[o],
                                 float(perr[p, o, :, 15].mean()), float(perr[np.arange(P) != p, o][:, :, 15].mean()))
                                for p in range(P) for o in range(O) if own[p][o]]
    out["error_vs_gt"] = err
    # headline: the default prompt vs the rest (train split), and by split
    score = perr[..., [0, 7, 15]].mean((1, 2, 3))                 # mean pos error at k 0/7/15 over obs and seeds
    tr = [i for i, p in enumerate(pinfo) if p["split"] == "train"]
    d = next(i for i, p in enumerate(pinfo) if p["split"] == "train" and p["prompt_index"] == default_k)
    order = sorted(tr, key=lambda i: score[i])
    zf = {names[i]: chunks[:, i, :, 15, 2].mean(1) * 100 for i in first}
    zmean = np.mean([zf[n] for n in zf], axis=0)                 # per prompt, mean over the *_first refs
    out["headline"] = {
        "score": "mean pos error (mm) at k 0/7/15 over the 15 refs x 3 seeds",
        "default": {"prompt": pinfo[d], "score_mm": float(score[d]), "rank_in_train": order.index(d) + 1,
                    "n_train": len(tr), "approach_z_cm_mean_first_refs": float(zmean[d])},
        "best": {"prompt": pinfo[order[0]], "score_mm": float(score[order[0]]),
                 "approach_z_cm": float(zmean[order[0]])},
        "median": {"prompt": pinfo[order[len(order) // 2]], "score_mm": float(score[order[len(order) // 2]]),
                   "approach_z_cm": float(zmean[order[len(order) // 2]])},
        "worst": {"prompt": pinfo[order[-1]], "score_mm": float(score[order[-1]]),
                  "approach_z_cm": float(zmean[order[-1]])},
        "train_score_mm_range": [float(score[tr].min()), float(score[tr].max())],
        "train_score_mm_mean": float(score[tr].mean()),
        "val_prompts": [{"prompt": pinfo[i], "score_mm": float(score[i]), "approach_z_cm": float(zmean[i])}
                        for i, p in enumerate(pinfo) if p["split"] == "val"],
        "approach_z_cm_range_over_prompts": [float(zmean.min()), float(zmean.max())],
        "demo_gt_approach_z_cm_first_refs": float(np.mean([gt[i, 15, 2] * 100 for i in first])),
        "score_seed_noise_mm": float(np.std([perr[d][:, s][:, [0, 7, 15]].mean() for s in range(S)], ddof=1)),
    }
    return out


def cmd_summary() -> None:
    out: dict[str, Any] = {"version": version_info(), "runs": {}}
    choice = json.loads((OUT / "prompt_choice.json").read_text()) if (OUT / "prompt_choice.json").exists() else {}
    for run in RUNS:
        p = OUT / f"prompt_sweep_{run}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c = choice.get("runs", {}).get(run, {})
        out["runs"][run] = {
            "dataset": r["dataset"], "deploy_default_prompt": c.get("deploy_default_prompt"),
            "warm_up_prompt": c.get("warm_up_prompt"),
            "q3": {"samples": r["q3"], "reset": r["reset"], "bf16": r["bf16"], "batch": r["batch"]},
            "headline": r["headline"],
            "spread_k15": r["spread"]["k15"], "spread_k7": r["spread"]["k7"], "spread_k0": r["spread"]["k0"],
            "val_metric_prompting": {k: c.get("training", {}).get(k) for k in
                                     ("n_val_demos", "val_demos", "val_action_mse_subset")},
            "gpu_s": r["gpu_s"],
        }
    write_json(OUT / "summary.json", out)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "gt":
        cmd_gt()
    elif cmd == "run":
        for r in sys.argv[2:]:
            cmd_run(r)
            gc.collect()
    elif cmd == "remetric":            # recompute the metrics of prompt_sweep_<run>.json from its .npz (CPU)
        for r in sys.argv[2:]:
            p = OUT / f"prompt_sweep_{r}.json"
            res = json.loads(p.read_text())
            z = np.load(OUT / f"prompt_sweep_{r}.npz")
            names = [str(n) for n in z["refs"]]
            rl = [x for x in refs() if x[0] in names]
            split = {}
            for d in json.loads((OUT / "prompt_choice.json").read_text())["runs"][r]["splits"]["all"]["demos"]:
                tr = any(x["episode"] == d["episode"] for x in
                         json.loads((OUT / "prompt_choice.json").read_text())["runs"][r]["splits"]["train"]["demos"])
                split[f"{d['episode']}:{d['task_name']}"] = tr
            res.update(metrics(z["chunks"], z["gt"], rl, res["prompts"], None, split, res["default_prompt_index"]))
            write_json(p, res)
            log("re-metric", p)
    elif cmd == "summary":
        cmd_summary()
    else:
        print(__doc__)
        sys.exit(2)
