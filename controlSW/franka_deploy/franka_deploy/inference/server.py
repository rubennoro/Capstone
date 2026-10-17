"""Inference server launcher: upstream detached_inference.PolicyInferenceNodeRealEnv, unchanged, with
- the light workspace (light_workspace.py) instead of TrainPolicyWorkspace (no LIBERO/robosuite/HF download),
- the checkpoint staged as a symlink in --work-dir and cwd = --work-dir, so upstream's `<ckpt>.yaml` and `tmp_vis/`
  writes land there (paths under results/, data/, behavior_prompting/, iPhUMI/, scripts/ are refused),
- startup checks (strict load report, CLIP normalisation, no heavy sim modules imported),
- the prompt dataset read directly from the .zarr.zip (upstream would build an LMDB copy at the training cfg's
  cache_dir, /tmp/noroian.r/zarr_cache for run_001),
- a warm-up inference (prompt + one predict_action, then policy.reset()) before printing READY / writing --ready-file.

    python -m franka_deploy.inference.server --ckpt <run>/checkpoints/best.ckpt --work-dir <dir> \
        --dataset-path <prompts>.zarr.zip [--endpoint tcp://0.0.0.0:18765] [--device cuda]

The model math, topics and replies are exactly upstream's (run_node is called unchanged; the prompt dataset is the
same UmiTaskDataset with the same split info).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from franka_deploy.paths import check_writable_dir, stage_checkpoint


def dummy_deploy_obs(shape_meta, robot_prefixes) -> dict:
    """A deployment-format observation (robot{i}_* keys, uint8 THWC images, pos3+rot6d proprio) for warm-up."""
    import numpy as np

    obs = {}
    for i, prefix in enumerate(robot_prefixes):
        arm = prefix[len("gripper_"):]
        for cam in ("main", "ultrawide"):
            k = f"camera_{arm}_{cam}_rgb"
            if k in shape_meta["obs"]:
                c, h, w = shape_meta["obs"][k]["shape"]
                obs[f"robot{i}_{cam}_camera"] = np.zeros((shape_meta["obs"][k]["horizon"], h, w, c), np.uint8)
        hz = shape_meta["obs"][f"{prefix}_eef_pos"]["horizon"]
        rot6d_id = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (hz, 1))
        obs[f"robot{i}_eef_pos"] = np.zeros((hz, 3), np.float32)
        obs[f"robot{i}_eef_rot_axis_angle"] = rot6d_id.copy()
        obs[f"robot{i}_gripper_width"] = np.full((hz, 1), 0.07, np.float32)
        obs[f"robot{i}_eef_rot_axis_angle_wrt_start"] = rot6d_id.copy()
    return obs


def node_infer(node, obs_deploy: dict) -> dict:
    """Exactly the policy_inference branch of upstream run_node (minus RMQ and attention logging)."""
    import numpy as np
    import torch

    obs_model = node._process_obs(obs_deploy)
    obs_tensors = {
        k: torch.from_numpy(np.array(v, dtype=np.int64 if k == "task_language" else np.float32)).unsqueeze(0).to(node.device)
        for k, v in obs_model.items()
    }
    with torch.inference_mode():
        result = node.policy.predict_action(obs_tensors)
    return node._process_action(result["action_pred"][0].detach().cpu().numpy())


def attach_prompt_dataset(node, dataset_path: str, prompt_from_dataset: str = "train", eval_dataset: bool = False,
                          cache_dir: str | None = None):
    """Same as the prompt-dataset block of upstream PolicyInferenceNodeRealEnv.__init__, except `cache_dir`.

    Upstream keeps the training cfg's `task.dataset.cache_dir` (run_001: /tmp/noroian.r/zarr_cache), so it would copy
    the whole zarr into an LMDB cache at that path on this machine. Here cache_dir defaults to None (read the
    .zarr.zip directly, read-only), or a directory that must pass check_writable_dir.
    """
    import hydra
    import omegaconf

    if not bool(node.shape_meta.get("use_prompting", False)):
        raise ValueError("--dataset-path was provided but this policy was not trained with prompting")
    assert prompt_from_dataset in ("train", "val"), prompt_from_dataset
    if cache_dir is not None:
        cache_dir = str(check_writable_dir(cache_dir, "zarr cache dir"))
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

    register_codecs(verbose=False)
    dataset_cfg = omegaconf.OmegaConf.create(dict(node.cfg.task.dataset))
    dataset_cfg.val_ratio = 0
    dataset_cfg.cache_dir = cache_dir
    ds = hydra.utils.instantiate(
        dataset_cfg, only_prompt=True, dataset_path=dataset_path,
        training_split_info=None if eval_dataset else node.policy.get_training_split_info(), allow_zip_file=True,
    )
    if not eval_dataset and prompt_from_dataset == "val":
        ds = ds.get_validation_dataset()
    node.prompt_dataset = ds
    tasks = {k: len(v) for k, v in ds.get_unique_task_name_to_dataset_indices().items()}
    print(f"[franka_deploy] prompt dataset ({prompt_from_dataset} split): {tasks}", flush=True)
    return tasks


IMAGECODECS_PIN = "2023.9.18"  # training env pin; JPEG-XL prompt decoding differs slightly in other versions


def build_node(ckpt: str, work_dir: str, dataset_path: str | None, endpoint: str, device: str,
               prompt_from_dataset: str = "train", vis_prompt: bool = False, attention_vis: bool = False,
               eval_dataset: bool = False):
    """Stage, chdir, install the light workspace, construct the upstream node. Returns (node, startup_report)."""
    work = check_writable_dir(work_dir, "work dir")
    work.mkdir(parents=True, exist_ok=True)
    link = stage_checkpoint(ckpt, work)
    os.chdir(work)
    from franka_deploy.inference import light_workspace

    light_workspace.install()
    import torch
    from behavior_prompting.deployment.detached_inference import PolicyInferenceNodeRealEnv

    t0 = time.monotonic()
    node = PolicyInferenceNodeRealEnv(
        ckpt_path=str(link), server_endpoint=endpoint, dataset_path=None, device=device,
        prompt_from_dataset=prompt_from_dataset, vis_prompt_flag=vis_prompt, attention_vis=attention_vis,
        is_eval_dataset=eval_dataset,
    )
    prompt_tasks = None
    if dataset_path:
        prompt_tasks = attach_prompt_dataset(node, dataset_path, prompt_from_dataset, eval_dataset)
    ws = node.workspace
    if not getattr(type(ws), "__module__", "") == light_workspace.UPSTREAM_MODULE or not hasattr(ws, "load_report"):
        raise RuntimeError(f"workspace is {type(ws)}, expected the light workspace")
    heavy = light_workspace.loaded_heavy_modules()
    if heavy:
        raise RuntimeError(f"heavy simulation modules were imported: {heavy}")
    n_norm = light_workspace.check_vision_normalization(node.policy)
    import imagecodecs

    if imagecodecs.__version__ != IMAGECODECS_PIN:
        print(f"[franka_deploy] WARNING: imagecodecs {imagecodecs.__version__} != training pin {IMAGECODECS_PIN}: "
              "prompt images decode slightly differently (see requirements-inference.txt)", flush=True)
    report = {
        "imagecodecs": imagecodecs.__version__,
        "ckpt": str(Path(ckpt).resolve()), "staged_ckpt": str(link), "work_dir": str(work), "cwd": os.getcwd(),
        "load_s": round(time.monotonic() - t0, 1), "load_report": ws.load_report, "epoch": node.epoch,
        "clip_normalization_checked": n_norm, "heavy_modules_loaded": heavy, "robot_prefixes": node.robot_prefixes, "prompt_tasks": prompt_tasks,
        "torch": torch.__version__, "device": str(node.device),
    }
    return node, report


def seed_noise_per_call(node, base_seed: int) -> None:
    """Reproducible diffusion noise: before every predict_action, torch.manual_seed(base_seed + call index), and the
    seed is added to the reply as "noise_seed" (PolicyAgent ignores extra keys; the tap records it). The noise
    distribution, model and math are unchanged; only the RNG state becomes known, so a logged call can be re-run
    exactly (tools/reinfer.py)."""
    import numpy as np
    import torch

    state = {"k": 0, "seed": None}
    orig_predict, orig_process = node.policy.predict_action, node._process_action

    def predict_action(obs, *a, **kw):
        state["seed"] = int(base_seed + state["k"])
        state["k"] += 1
        torch.manual_seed(state["seed"])
        return orig_predict(obs, *a, **kw)

    def process_action(action_pred):
        out = orig_process(action_pred)
        out["noise_seed"] = np.array(state["seed"] if state["seed"] is not None else -1, dtype=np.int64)
        return out

    node.policy.predict_action = predict_action
    node._process_action = process_action


def warm_up(node, task_name: str | None, prompt_index: int, n: int = 2) -> dict:
    import torch

    t0 = time.monotonic()
    if node.prompt_dataset is not None:
        if task_name is None:
            task_name = sorted(node.prompt_dataset.get_unique_task_name_to_dataset_indices())[0]
        node._handle_prompt(task_name, prompt_index)
    obs = dummy_deploy_obs(node.shape_meta, node.robot_prefixes)
    times = []
    for _ in range(n):
        t = time.monotonic()
        node_infer(node, obs)
        if node.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(round(time.monotonic() - t, 3))
    node.policy.reset()  # clears the prompt: clients must prompt per episode, as upstream
    node.input_prompt_dict = None
    return {"warmup_s": round(time.monotonic() - t0, 2), "warmup_infer_s": times, "warmup_task": task_name}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--endpoint", default="tcp://0.0.0.0:18765")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-from-dataset", default="train", choices=["train", "val"])
    ap.add_argument("--warmup-task", default=None)
    ap.add_argument("--warmup-prompt-index", type=int, default=0)
    ap.add_argument("--vis-prompt", action="store_true")
    ap.add_argument("--attention-vis", action="store_true")
    ap.add_argument("--eval-dataset", action="store_true")
    ap.add_argument("--ready-file", default=None, help="JSON written when the server is ready (after warm-up)")
    ap.add_argument("--noise-seed-base", type=int, default=int(time.time()) % 1_000_000,
                    help="seed the diffusion noise per call (base + call index; -1 = upstream RNG state)")
    a = ap.parse_args(argv)
    ready_file = Path(os.path.abspath(a.ready_file)) if a.ready_file else None
    if ready_file:
        check_writable_dir(ready_file.parent, "ready-file dir")
    dataset_path = os.path.abspath(a.dataset_path) if a.dataset_path else None
    node, report = build_node(a.ckpt, a.work_dir, dataset_path, a.endpoint, a.device, a.prompt_from_dataset,
                              a.vis_prompt, a.attention_vis, a.eval_dataset)
    with open(os.path.join(report["work_dir"], "split_info.json"), "w") as f:  # for evaluation (prompt index -> demo)
        json.dump({k: bool(v) for k, v in (node.policy.get_training_split_info() or {}).items()}, f)
    report.update(warm_up(node, a.warmup_task, a.warmup_prompt_index))
    if a.noise_seed_base >= 0:
        seed_noise_per_call(node, a.noise_seed_base)
        report["noise_seed_base"] = a.noise_seed_base
    report.update({"endpoint": a.endpoint, "pid": os.getpid(), "ready_monotonic": time.monotonic()})
    print("[franka_deploy] READY " + json.dumps(report), flush=True)
    if ready_file:
        tmp = ready_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(ready_file)
    node.run_node()
    return 0


if __name__ == "__main__":
    sys.exit(main())
