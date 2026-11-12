# from rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd
# changes: only PolicyInferenceNodeRealEnv.__init__'s loading steps, plus main-sim inference/server.py @ 4be9982
# build_node's start-up checks; no <ckpt>.yaml, robotmq, prompt dataset or tokenizer; adds the C4 checks and the
# DDIM-steps override (STRUCTURE.md 4.6, 5.9)
"""CheckpointLoader, LoadedModel: checkpoint -> model on the device, with the load checks (torch)."""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from policy.errors import PolicyLoadError

CKPT_HINT = "check policy.checkpoint in station.yaml"
ENV_HINT = "run policy with python.policy, the torch environment"


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module                 # in eval mode, on the device, reset
    shape_meta: Mapping[str, Any]          # cfg.task.shape_meta, as plain dicts
    uses_prompting: bool                   # shape_meta.use_prompting
    info: Mapping[str, Any]                # for startup.json (STRUCTURE.md 3.6)


class CheckpointLoader:
    def __init__(self, training_timing: Mapping[str, int]) -> None:
        self._timing = dict(training_timing)

    def load(self, path: Path, device: str, diffusion_steps: int | None) -> LoadedModel:
        """Load, move to the device, reset and check. Raises PolicyLoadError."""
        t0 = time.monotonic()
        path = Path(path)
        self.check_file(path)
        self.check_device(device)
        try:
            import dill
            import hydra
            import omegaconf
            from policy import light_workspace
            light_workspace.install()
        except ImportError as e:
            raise PolicyLoadError(f"{e.name or 'a model library'} is not importable: {e}", ENV_HINT) from e

        try:
            with open(path, "rb") as f:                # read-only; nothing is written beside the checkpoint
                payload = torch.load(f, map_location="cpu", pickle_module=dill, weights_only=False)
            cfg = payload["cfg"]
            workspace = hydra.utils.get_class(cfg._target_)(cfg)
        except Exception as e:
            raise PolicyLoadError(f"checkpoint failed to load: {path}: {type(e).__name__}: {e}",
                                  "is it a behavior_prompting training checkpoint? " + CKPT_HINT) from e
        if not light_workspace.is_light_workspace(workspace):
            raise PolicyLoadError(f"workspace is {type(workspace)}, expected the light workspace",
                                  "policy.light_workspace.install() must run before behavior_prompting's workspace "
                                  "is imported")
        try:
            workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
        except Exception as e:
            raise PolicyLoadError(f"checkpoint weights do not match the model (strict load): "
                                  f"{type(e).__name__}: {e}",
                                  "the checkpoint and its own cfg disagree; use the checkpoint training wrote") from e

        model = workspace.model.eval().to(torch.device(device))
        model.reset()

        heavy = light_workspace.loaded_heavy_modules()
        if heavy:
            raise PolicyLoadError(f"heavy simulation modules were imported while loading: {heavy}",
                                  "the light workspace must replace TrainPolicyWorkspace; see policy/light_workspace.py")
        try:
            norm_checked = light_workspace.check_vision_normalization(model)
        except AssertionError as e:
            raise PolicyLoadError(f"colour normalisation check failed: {e}",
                                  "see REFERENCE_NORMALIZATION in policy/light_workspace.py") from e

        cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        shape_meta = cfg_dict["task"]["shape_meta"]
        self.check_pose_repr(cfg_dict)
        self.check_timing(cfg_dict)
        steps_ckpt = int(model.num_inference_steps)
        self.apply_diffusion_steps(model, diffusion_steps)

        epoch = payload.get("pickles", {}).get("epoch", None)
        if isinstance(epoch, bytes):
            epoch = dill.loads(epoch)
        st = path.stat()
        info = {
            "checkpoint": str(path.resolve()),
            "checkpoint_bytes": st.st_size,
            "checkpoint_mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "epoch": epoch,
            "name": cfg_dict.get("name"),
            "model_target": cfg_dict.get("model", {}).get("_target_"),
            "diffusion_steps_checkpoint": steps_ckpt,
            "diffusion_steps_in_use": int(model.num_inference_steps),
            "num_train_timesteps": _num_train_timesteps(model),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__,
            "load_s": round(time.monotonic() - t0, 2),
            "load_report": dict(workspace.load_report),
            "colour_normalisation_checked": norm_checked,
        }
        return LoadedModel(model=model, shape_meta=shape_meta,
                           uses_prompting=bool(shape_meta.get("use_prompting", False)), info=info)

    # -- the individual checks (each raises PolicyLoadError) ------------------------------------------------------

    @staticmethod
    def check_file(path: Path) -> None:
        if not path.exists():
            raise PolicyLoadError(f"checkpoint not found: {path}", CKPT_HINT)
        if not path.is_file() or path.suffix != ".ckpt":
            raise PolicyLoadError(f"checkpoint is not a .ckpt file: {path}", CKPT_HINT)

    @staticmethod
    def check_device(device: str) -> None:
        if device == "cpu":
            return
        if device != "cuda":
            raise PolicyLoadError(f"unknown device {device!r}", "set policy.device to cuda or cpu")
        if not torch.cuda.is_available():
            raise PolicyLoadError("CUDA not available", "check the GPU driver, or set policy.device: cpu "
                                                        "(PLAN §9 backup)")

    @staticmethod
    def check_pose_repr(cfg: Mapping[str, Any]) -> None:
        task = cfg.get("task", {})
        pose = task.get("pose_repr", {}) or {}
        problems = []
        for key in ("obs_pose_repr", "action_pose_repr"):
            if pose.get(key) != "relative":
                problems.append(f"task.pose_repr.{key} is {pose.get(key)!r}, C4 needs 'relative'")
        rot = (task.get("shape_meta", {}).get("action", {}) or {}).get("rotation_rep")
        if rot != "rotation_6d":
            problems.append(f"task.shape_meta.action.rotation_rep is {rot!r}, C4 needs 'rotation_6d'")
        if problems:
            raise PolicyLoadError("checkpoint pose representation does not fit C4: " + "; ".join(problems),
                                  "use a checkpoint trained with relative poses and rotation_6d")

    def check_timing(self, cfg: Mapping[str, Any]) -> None:
        task = cfg.get("task", {})
        problems = [f"task.{key} is {task.get(key)!r}, expected {want} (policy.training_timing.{key})"
                    for key, want in self._timing.items() if task.get(key) != want]
        if problems:
            raise PolicyLoadError("checkpoint step timing does not match C1/C4: " + "; ".join(problems),
                                  "use a checkpoint trained with the rig's timing")

    @staticmethod
    def apply_diffusion_steps(model: Any, diffusion_steps: int | None) -> None:
        """Set model.num_inference_steps (PLAN §9 fewer-steps backup); only for a DDIM scheduler."""
        if diffusion_steps is None:
            return
        scheduler = getattr(model, "noise_scheduler", None)
        try:
            from diffusers import DDIMScheduler
        except ImportError as e:
            raise PolicyLoadError(f"diffusers is not importable: {e}", ENV_HINT) from e
        if not isinstance(scheduler, DDIMScheduler):
            raise PolicyLoadError(f"policy.diffusion_steps is set but the scheduler is {type(scheduler).__name__}, "
                                  f"not DDIMScheduler", "remove policy.diffusion_steps for this checkpoint")
        n_train = _num_train_timesteps(model)
        if not 1 <= diffusion_steps <= n_train:
            raise PolicyLoadError(f"policy.diffusion_steps {diffusion_steps} is outside 1..{n_train} "
                                  f"(num_train_timesteps)", "choose a smaller policy.diffusion_steps")
        model.num_inference_steps = int(diffusion_steps)


def _num_train_timesteps(model: Any) -> int | None:
    scheduler = getattr(model, "noise_scheduler", None)
    config = getattr(scheduler, "config", None)
    value = getattr(config, "num_train_timesteps", None)
    return int(value) if value is not None else None
