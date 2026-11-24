"""W1 outputs: one checkpoint, built the deploy way, on cached obs pairs (build_obs.py), with fixed diffusion noise.
Torch env, GPU (``~/envs/sim_policy_deploy_torch/bin/python``); the CLI holds the shared GPU lock for the whole
process (the model's CUDA lifetime).

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.offline.policy_runner --run RUN_FOLDER \
        --samples pick1-v1:14:6 pick1-v1:14:9 [--seeds 0 1] [--precision fp32] [--diffusion-steps N] --out DIR

Build (as the eval driver / the robot do it): copy ``sim_deploy/stations/policy_sim.yaml`` to scratch, the copy's
``launcher.use_policy.use(copy, RUN_FOLDER, precision=, diffusion_steps=)`` (= ``policy_deploy use``: checkpoint,
kind diffusion, and for prompting prompt_dataset + manager task_name / prompt_index), ``policy.settings.Settings.load(
copy, mock=True)`` (device cuda), ``policy_factory.build_policy``. Per episode, as episode_loop does at each start:
``reset()`` then, for a prompting policy, ``prompt(task_name, prompt_index)`` (the station's manager section).
``CheckpointLoader.load`` is wrapped (observed, not changed) to keep its ``LoadedModel`` (the checkpoint's resolved
``cfg.task.dataset`` and ``get_training_split_info()``).

Per sample and seed (``torch.manual_seed(seed)`` right before every call):
* deploy, convention A (C2 width = training width): ``policy.infer(obs)`` with the policy's gripper map off ->
  the robot chunk (widths clipped to [0, gripper_width_max_m] by make_chunk); convention B (copy f632576 gripper
  map): ``policy.infer(obs with deploy_gripper_width_m_B)`` with the map on (to_model in, to_robot out). The model's
  own (16, 10) output for each = ``ModelRunner.run`` on the exact model-space obs infer builds (checked: the robot
  chunk is that output through the same width mapping and clip);
* direct: ``ModelRunner.run(Observation(direct obs))``: the same model, only the inputs differ;
* the noise floor: the direct call repeated with the same seed (CUDA determinism).
Then errors per horizon step (pos mm, rot deg, width mm) deploy vs direct and each vs the ground truth; robot
command widths (A, B) for the predicted and the ground-truth chunks; the manager's ``PoseConverter.to_waypoints``
of the deploy chunk on the deploy anchor (station exec_h); the alignment check (ground truth through to_waypoints with
exec_h 16, on the row-T anchor and on the deploy anchor, vs the recorded rows T+3k and vs the recorded trajectory at
the waypoint times); the first-chunk approach (approach.py).

API for step 3: ``PolicyRunner(run_folder, precision="fp32", diffusion_steps=None)`` (``.new_episode()``,
``.infer_deploy(obs, seed, convention)``, ``.infer_model(obs, seed)``, ``.close()``) and
``evaluate_samples(runner, samples, seeds) -> list[dict]``; the CLI writes ``<out>/<run>.json`` (+ ``.npz`` of every
chunk). One checkpoint per process (the CLI) keeps the GPU lock exact and frees all memory between checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import (OBS_CACHE, SCRATCH, STATION_TEMPLATE, deploy_version, gpu_lock, set_cpu_threads,
                                       write_json)

CONVENTIONS = ("A", "B")


def checkpoint_steps(run_folder: Path) -> int:
    """The checkpoint's own diffusion step count (run_config/config.yaml model.num_inference_steps)."""
    import yaml
    cfg = yaml.safe_load((Path(run_folder) / "run_config" / "config.yaml").read_text())
    return int((cfg.get("model") or cfg.get("policy"))["num_inference_steps"])


def resolve_steps(run_folder: Path, diffusion_steps: int | None) -> int:
    """Always an explicit step count. None means the checkpoint's own steps. Passing None on to `use` would keep the
    station template's value (policy_sim.yaml: 8), which is how s1_smoke's "_fp32_ckpt" file came to run at 8 steps."""
    return checkpoint_steps(run_folder) if diffusion_steps is None else int(diffusion_steps)


def make_station(run_folder: Path, precision: str = "fp32", diffusion_steps: int | None = None) -> Path:
    """A scratch copy of the sim station pointed at the run by the copy's own `policy_deploy use`; diffusion_steps
    None = the checkpoint's own (resolved here and always passed to `use` explicitly)."""
    from policy_deploy.launcher.use_policy import use
    run_folder = Path(run_folder)
    diffusion_steps = resolve_steps(run_folder, diffusion_steps)
    dst = SCRATCH / "stations" / f"{run_folder.name}_{precision}_s{diffusion_steps}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STATION_TEMPLATE, dst)
    use(dst, str(run_folder), precision=precision, diffusion_steps=diffusion_steps)
    return dst


class PolicyRunner:
    """The deploy-built policy of one run folder on CUDA. The caller must hold the GPU lock (``common.gpu_lock``) from
    before construction until after ``close()``; the CLI does."""

    def __init__(self, run_folder: Path, precision: str = "fp32", diffusion_steps: int | None = None,
                 device: str = "cuda") -> None:
        import torch
        from policy_deploy.manager.settings import Settings as ManagerSettings
        from policy_deploy.policy import checkpoint_loader
        from policy_deploy.policy.policy_factory import build_policy
        from policy_deploy.policy.settings import Settings as PolicySettings

        set_cpu_threads()
        self.run_folder = Path(run_folder)
        self.diffusion_steps = resolve_steps(self.run_folder, diffusion_steps)
        self.station = make_station(self.run_folder, precision, self.diffusion_steps)
        st = PolicySettings.load(self.station, mock=True)
        assert st.diffusion_steps == self.diffusion_steps, (st.diffusion_steps, self.diffusion_steps)
        self.settings = dataclasses.replace(st, device=device)
        self.manager_settings = ManagerSettings.load(self.station, mock=True)
        captured = []
        orig = checkpoint_loader.CheckpointLoader.load

        def observed(this, *a, **k):
            lm = orig(this, *a, **k)
            captured.append(lm)
            return lm

        checkpoint_loader.CheckpointLoader.load = observed
        t0 = time.monotonic()
        try:
            self.policy = build_policy(self.settings)
        finally:
            checkpoint_loader.CheckpointLoader.load = orig
        self.load_s = round(time.monotonic() - t0, 1)
        self.loaded = captured[0]
        self.runner = self.policy._runner                       # the policy's ModelRunner (read, not changed)
        self.gripper_map = self.policy._gripper_map            # convention B's map as configured (None: not set)
        if self.gripper_map is None:
            from sim_deploy.offline.deploy_path import gripper_map_for_b
            self.gripper_map, _ = gripper_map_for_b(None)
        self.uses_prompt = bool(getattr(self.policy, "USES_PROMPT", False))
        self.prompt_choice = ((self.manager_settings.task_name, self.manager_settings.prompt_index)
                              if self.uses_prompt else None)
        self.split_info = {str(k): bool(v) for k, v in (self.loaded.model.get_training_split_info() or {}).items()}
        self.torch = torch
        self.last_prompt = None

    def describe(self) -> dict:
        from sim_deploy.offline.direct_path import cfg_signature
        d = self.policy.describe()
        return {"run_folder": str(self.run_folder), "station_copy": str(self.station), "load_s": self.load_s,
                "kind": d.get("kind"), "uses_prompt": self.uses_prompt, "prompt_choice": self.prompt_choice,
                "prompt_dataset": str(self.settings.prompt_dataset) if self.settings.prompt_dataset else None,
                "prompt_split": self.settings.prompt_split, "precision": self.runner.precision,
                "diffusion_steps_in_use": d.get("diffusion_steps_in_use"),
                "diffusion_steps_checkpoint": d.get("diffusion_steps_checkpoint"), "epoch": d.get("epoch"),
                "checkpoint": d.get("checkpoint"), "inputs": list(self.runner.inputs),
                "gripper_map_configured": d.get("gripper_map"), "gripper_map_B": self.gripper_map.as_dict(),
                "dataset_cfg_signature": cfg_signature(self.loaded.dataset_cfg),
                "dataset_path_in_cfg": self.loaded.dataset_cfg.get("dataset_path"),
                "sample_type": self.loaded.dataset_cfg.get("sample_type"), "n_split_entries": len(self.split_info),
                "gpu": d.get("gpu"), "memory": self.policy.memory_stats()}

    def new_episode(self) -> None:
        """As episode_loop at every episode start: reset, then (prompting) prompt(task_name, prompt_index)."""
        self.policy.reset()
        if self.uses_prompt:
            self.policy.prompt(*self.prompt_choice)
            self.last_prompt = dict(self.policy.last_prompt)

    def _seed(self, seed: int) -> None:
        self.torch.manual_seed(int(seed))

    def infer_deploy(self, obs, seed: int, convention: str):
        """policy.infer(obs) with the gripper map off (A) or on (B); returns (robot ActionChunk after a C4 round
        trip, the model's own (16, 10) output for the same inputs and seed, model-space width input)."""
        from deploy_contracts import decode, encode
        from policy_deploy.contracts import ActionChunk
        if convention not in CONVENTIONS:
            raise ValueError(convention)
        saved = self.policy._gripper_map
        self.policy._gripper_map = None if convention == "A" else self.gripper_map
        try:
            self._seed(seed)
            chunk = decode(encode(self.policy.infer(obs)), ActionChunk)
        finally:
            self.policy._gripper_map = saved
        model_obs = obs if convention == "A" else dataclasses.replace(
            obs, gripper_width_m=self.gripper_map.to_model(obs.gripper_width_m))
        raw = self.infer_model(model_obs, seed)
        return chunk, raw, np.asarray(model_obs.gripper_width_m).ravel()

    def infer_model(self, obs, seed: int) -> np.ndarray:
        """ModelRunner.run(obs) (no width mapping, no clip) -> (16, 10) float32."""
        from sim_deploy.offline.compare import chunk_array
        self._seed(seed)
        pos, rot, w = self.runner.run(obs)
        return chunk_array(pos, rot, w)

    def robot_widths(self, model_width: np.ndarray, convention: str) -> np.ndarray:
        """The robot command widths make_chunk would send for model widths (A: clip; B: to_robot, clip)."""
        w = np.asarray(model_width, np.float32).reshape(-1, 1)
        if convention == "B":
            w = self.gripper_map.to_robot(w)
        return np.clip(w, np.float32(0), self.policy.width_max).ravel()

    def close(self) -> None:
        del self.policy, self.runner, self.loaded
        self.torch.cuda.empty_cache()


def _obs(z: dict, prefix: str, width_key: str = "gripper_width_m"):
    from deploy_contracts import decode, encode
    from policy_deploy.contracts import Observation
    o = Observation(eef_pos=z[f"{prefix}_eef_pos"].astype(np.float32),
                    eef_rot6d=z[f"{prefix}_eef_rot6d"].astype(np.float32),
                    eef_rot6d_wrt_start=z[f"{prefix}_eef_rot6d_wrt_start"].astype(np.float32),
                    gripper_width_m=z[f"{prefix}_{width_key}"].astype(np.float32).reshape(2, 1),
                    main=z[f"{prefix}_main"].astype(np.uint8), ultrawide=z[f"{prefix}_ultrawide"].astype(np.uint8))
    return decode(encode(o), Observation)                      # C4 round trip (validated)


def _anchor(meta: dict):
    from policy_deploy.manager.pose_converter import Anchor
    a = meta["deploy"]["anchor"]
    return Anchor(t_ns=int(a["t_ns"]), t_pc_ns=int(a["t_pc_ns"]), T=np.asarray(a["T"], np.float64), source=a["source"])


def _chunk_from(arr: np.ndarray, width: np.ndarray):
    from policy_deploy.contracts import ActionChunk
    return ActionChunk(eef_pos=arr[:, :3].astype(np.float32), eef_rot6d=arr[:, 3:9].astype(np.float32),
                       gripper_width_m=np.asarray(width, np.float32).reshape(-1, 1), width_clipped=0)


def _width_report(runner: PolicyRunner, model_w: np.ndarray) -> dict:
    out = {}
    for c in CONVENTIONS:
        r = runner.robot_widths(model_w, c)
        raw = model_w if c == "A" else runner.gripper_map.to_robot(np.asarray(model_w, np.float32))
        out[c] = {"robot_m": np.round(r, 5).tolist(), "min_m": float(r.min()), "max_m": float(r.max()),
                  "clipped_steps": int(np.count_nonzero(np.abs(np.asarray(raw).ravel() - r) > 0)),
                  "closes_below_0.035": bool(r.min() < 0.035)}
    return out


def evaluate_sample(runner: PolicyRunner, z: dict, meta: dict, seeds=(0,)) -> tuple[dict, dict]:
    """Everything for one cached sample; returns (json record, arrays)."""
    from policy_deploy.manager.pose_converter import Anchor, PoseConverter
    from sim_deploy.offline.approach import approach_metrics
    from sim_deploy.offline.compare import chunk_errors, pose_errors
    from sim_deploy.offline.episodes import split_of

    s = meta["sample"]
    rec: dict = {"sample": s, "checkpoint_split": split_of(runner.split_info, s["pi_name"]), "seeds": {}}
    arrays: dict = {}
    if meta["deploy"]["step"]["type"] != "StepDone":
        rec["skipped"] = meta["deploy"]["step"]
        return rec, arrays
    gt = z["action"].astype(np.float32)
    obs_a = _obs(z, "deploy")
    obs_b = _obs(z, "deploy", "gripper_width_m_B")
    obs_d = _obs(z, "direct")
    anchor = _anchor(meta)
    ms = runner.manager_settings
    conv = PoseConverter(ms.exec_h, ms.lookahead_ms)
    conv16 = PoseConverter(16, ms.lookahead_ms)
    for seed in seeds:
        r: dict = {}
        raw_d = runner.infer_model(obs_d, seed)
        raw_d2 = runner.infer_model(obs_d, seed)
        r["noise_floor_same_seed"] = chunk_errors(raw_d2, raw_d)["max"]
        arrays[f"s{seed}_direct"] = raw_d
        r["direct_vs_gt"] = chunk_errors(raw_d, gt)
        for c, obs in (("A", obs_a), ("B", obs_b)):
            chunk, raw, model_w = runner.infer_deploy(obs, seed, c)
            arrays[f"s{seed}_deploy_{c}"] = raw
            arrays[f"s{seed}_deploy_{c}_robot_width"] = np.asarray(chunk.gripper_width_m).ravel()
            expect = runner.robot_widths(raw[:, 9], c)
            r[f"deploy_{c}"] = {
                "model_width_input_m": model_w.tolist(),
                "direct_width_input_m": z["direct_gripper_width_m"].ravel().tolist(),
                "infer_equals_runner": {"pos_rot_max_abs": float(np.max(np.abs(
                    np.concatenate([chunk.eef_pos, chunk.eef_rot6d], -1) - raw[:, :9]))),
                    "robot_width_max_abs": float(np.max(np.abs(np.asarray(chunk.gripper_width_m).ravel() - expect))),
                    "width_clipped": int(chunk.width_clipped)},
                "vs_direct": chunk_errors(raw, raw_d), "vs_gt": chunk_errors(raw, gt),
                "robot_widths_pred": _width_report(runner, raw[:, 9])[c],
                "approach": approach_metrics(raw[:, :3], anchor.T, meta, z),
            }
            if c == "A" and seed == seeds[0]:
                wp = conv.to_waypoints(chunk, anchor, 1)
                arrays["waypoints_A_O_T_TCP"] = wp.O_T_TCP
                r["waypoints_A"] = {"exec_h": ms.exec_h, "t_minus_anchor_ms": ((wp.t_ns - anchor.t_pc_ns) / 1e6).tolist(),
                                    "pos_first_last_m": [wp.O_T_TCP[0, :3, 3].tolist(), wp.O_T_TCP[-1, :3, 3].tolist()]}
        r["direct_approach"] = approach_metrics(raw_d[:, :3], z["row_T"], meta, z)
        rec["seeds"][str(seed)] = r
    if len(seeds) > 1:
        rec["direct_seed_spread"] = chunk_errors(arrays[f"s{seeds[1]}_direct"], arrays[f"s{seeds[0]}_direct"])["max"]
    rec["gt"] = {"action0": gt[0].tolist(), "robot_widths": _width_report(runner, gt[:, 9]),
                 "approach": approach_metrics(gt[:, :3], z["row_T"], meta, z)}
    # alignment: training's action k = row T+3k; the deploy times waypoint k at anchor + k*STEP_MS + lookahead
    gt_chunk = _chunk_from(gt, runner.robot_widths(gt[:, 9], "A"))
    row_anchor = Anchor(t_ns=anchor.t_ns, t_pc_ns=int(meta["deploy"]["t_row_ns"]), T=z["row_T"], source="target")
    wp_row = conv16.to_waypoints(gt_chunk, row_anchor, 1)
    wp_dep = conv16.to_waypoints(gt_chunk, anchor, 1)
    fut_t = np.asarray(meta["deploy"]["future_row_t_ns"], np.int64)
    rec["alignment"] = {
        "gt_on_row_anchor_vs_rows_T+3k": pose_errors(wp_row.O_T_TCP, z["future_T"]),
        "gt_on_row_anchor_time_minus_row_times_ms": ((wp_row.t_ns - fut_t) / 1e6).tolist(),
        "gt_on_deploy_anchor_vs_rows_T+3k": pose_errors(wp_dep.O_T_TCP, z["future_T"]),
        "gt_on_deploy_anchor_vs_traj_at_waypoint_times": pose_errors(wp_dep.O_T_TCP, z["deploy_traj_at_waypoint_times"]),
        "deploy_waypoint_time_minus_row_times_ms": ((wp_dep.t_ns - fut_t) / 1e6).tolist(),
        "deploy_anchor_vs_row_T": pose_errors(anchor.T[None], z["row_T"][None]),
        "future_rows_padded": int(sum(1 for k, x in enumerate(meta["deploy"]["future_rows"]) if x != s["row"] + 3 * k)),
    }
    return rec, arrays


def evaluate_samples(runner: PolicyRunner, samples, seeds=(0,), root: Path = OBS_CACHE) -> tuple[list[dict], dict]:
    """samples: [(dataset, index, row)], grouped by episode; new_episode() (reset + prompt) once per episode."""
    from sim_deploy.offline.build_obs import load_sample
    recs, arrays = [], {}
    last_ep = None
    for dataset, index, row in samples:
        if (dataset, index) != last_ep:
            runner.new_episode()
            last_ep = (dataset, index)
        z, meta = load_sample(dataset, index, row, root)
        t0 = time.monotonic()
        rec, arr = evaluate_sample(runner, z, meta, seeds)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        rec["prompt"] = runner.last_prompt
        recs.append(rec)
        for k, v in arr.items():
            arrays[f"{dataset}_ep{index:02d}_r{row:03d}_{k}"] = v
    return recs, arrays


def _parse(spec: str) -> tuple[str, int, int]:
    ds, i, r = spec.rsplit(":", 2)
    return ds, int(i), int(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", required=True, help="dataset:index:row (cached by build_obs)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--diffusion-steps", type=int, default=None, help="default: the checkpoint's own (explicit)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=OBS_CACHE, help="obs cache root (variants: W1/obs_100hz, ...)")
    a = ap.parse_args(argv)
    samples = sorted({_parse(s) for s in a.samples})
    with gpu_lock(f"W1 S1 policy_runner {a.run.name}") as waited:
        runner = PolicyRunner(a.run, a.precision, a.diffusion_steps)
        try:
            t0 = time.monotonic()
            recs, arrays = evaluate_samples(runner, samples, a.seeds, a.root)
            desc = runner.describe()
        finally:
            runner.close()
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.run.name}_{a.precision}_s{desc['diffusion_steps_in_use']}"   # the step count that actually ran
    np.savez(a.out / f"{stem}.npz", **arrays)
    write_json(a.out / f"{stem}.json", {"policy": desc, "gpu_lock_wait_s": waited, "seeds": a.seeds,
                                        "eval_s": round(time.monotonic() - t0, 1), "samples": recs,
                                        "deploy_version": deploy_version()})
    print(f"[policy_runner] {a.run.name}: {len(recs)} samples -> {a.out / stem}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
