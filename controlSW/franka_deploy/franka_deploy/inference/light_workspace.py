"""Inference-only replacement for behavior_prompting's TrainPolicyWorkspace.

Why: a checkpoint's cfg._target_ is
`behavior_prompting.train_network.workspace.train_policy_workspace.TrainPolicyWorkspace`. Importing that module pulls
in `train_network/utils/load_env.py`, which imports LIBERO at import time (LIBERO's __init__ creates
~/.libero/config.yaml and calls input()), robosuite/mujoco, the draw env, wandb... none of which inference needs. Its
constructor also downloads the CLIP backbone from the HF hub (`pretrained: true`) only to overwrite it with the
checkpoint weights.

What this changes: ONLY how the workspace object is built. The model is still `hydra.utils.instantiate(cfg.model)`
with the checkpoint's own cfg (same seeds as TrainPolicyWorkspace.__init__), the weights are loaded with a strict
`load_state_dict` from the checkpoint's `state_dicts`, and `pickles` are restored exactly like
BaseWorkspace.load_payload. During construction `timm.create_model` is called with pretrained=False (same
architecture; no download); `pretrained=True` stays set on the encoder, so every pretrained-dependent branch in
TransformerObsEncoder (no GroupNorm swap, no re-init, vision-norm assert) is identical to training. The vision
normalisation constants come from timm's pretrained_cfg, as in training; `check_vision_normalization` asserts them.

`install()` registers this class under the upstream module name in sys.modules, so upstream
`detached_inference.PolicyInferenceNodeRealEnv` (unchanged) resolves cfg._target_ to it.
"""

from __future__ import annotations

import random
import sys
import types
from typing import Any

UPSTREAM_MODULE = "behavior_prompting.train_network.workspace.train_policy_workspace"
HEAVY_MODULES = ("libero", "robosuite", "mujoco", "mujoco_py", "wandb", "pymunk", "pygame", "gym")

# OpenAI CLIP image normalisation (timm vit_base_patch16_clip_224.openai pretrained_cfg).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _make_class():
    import hydra
    import numpy as np
    import torch
    import dill
    from behavior_prompting.train_network.workspace.base_workspace import BaseWorkspace

    class LightTrainPolicyWorkspace(BaseWorkspace):
        include_keys = ["global_step", "epoch"]
        exclude_keys = tuple()

        def __init__(self, cfg, output_dir=None):
            super().__init__(cfg, output_dir=output_dir)
            seed = cfg.training.seed  # same seeding as TrainPolicyWorkspace.__init__
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            import timm

            orig = timm.create_model

            def create_model_no_download(*args, **kwargs):
                kwargs["pretrained"] = False
                return orig(*args, **kwargs)

            timm.create_model = create_model_no_download
            try:
                self.model = hydra.utils.instantiate(cfg.model)
            finally:
                timm.create_model = orig
            self.load_report: dict[str, Any] = {}

        def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
            # Same semantics as BaseWorkspace.load_payload, strict, plus a report of what was loaded.
            exclude_keys = tuple() if exclude_keys is None else tuple(exclude_keys)
            if include_keys is None:
                include_keys = payload["pickles"].keys()
            loaded = {}
            for key, value in payload["state_dicts"].items():
                if key in exclude_keys:
                    continue
                n_tensors = sum(1 for v in value.values() if torch.is_tensor(v))
                res = self.__dict__[key].load_state_dict(value, strict=True, **kwargs)
                missing = list(getattr(res, "missing_keys", []) or [])
                unexpected = list(getattr(res, "unexpected_keys", []) or [])
                if missing or unexpected:
                    raise RuntimeError(f"{key}: missing {missing[:5]} unexpected {unexpected[:5]}")
                loaded[key] = {"tensors": n_tensors, "missing": len(missing), "unexpected": len(unexpected)}
            for key in include_keys:
                if key in payload["pickles"]:
                    self.__dict__[key] = dill.loads(payload["pickles"][key])
            self.load_report = loaded

    LightTrainPolicyWorkspace.__module__ = UPSTREAM_MODULE
    LightTrainPolicyWorkspace.__qualname__ = "TrainPolicyWorkspace"
    return LightTrainPolicyWorkspace


def install() -> type:
    """Register the light workspace as `<UPSTREAM_MODULE>.TrainPolicyWorkspace`. Idempotent.

    Refuses if the real upstream module was already imported (then the heavy stack is loaded anyway and
    hydra would resolve to the real class).
    """
    existing = sys.modules.get(UPSTREAM_MODULE)
    if existing is not None:
        if getattr(existing, "__franka_deploy_light__", False):
            return existing.TrainPolicyWorkspace
        raise RuntimeError(f"{UPSTREAM_MODULE} is already imported; install() must run before it")
    cls = _make_class()
    mod = types.ModuleType(UPSTREAM_MODULE, doc="franka_deploy inference-only stand-in (see light_workspace.py)")
    mod.TrainPolicyWorkspace = cls
    mod.__franka_deploy_light__ = True
    sys.modules[UPSTREAM_MODULE] = mod
    import behavior_prompting.train_network.workspace as ws_pkg

    setattr(ws_pkg, "train_policy_workspace", mod)
    return cls


def loaded_heavy_modules() -> list[str]:
    return sorted(m for m in sys.modules if m.split(".")[0] in HEAVY_MODULES)


def check_vision_normalization(policy, atol: float = 1e-6) -> list[str]:
    """F2: every TransformerObsEncoder must have use_vision_norm=True and a Normalize with the CLIP constants
    (one Normalize per encoder, applied to every camera key: transformer_obs_encoder.py:110, :339-340); any other
    torchvision Normalize in the policy must use them too. Returns the names of the checked modules."""
    import torch
    import torchvision
    from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder

    def check(name, m):
        mean = torch.as_tensor(m.mean, dtype=torch.float64).flatten()
        std = torch.as_tensor(m.std, dtype=torch.float64).flatten()
        if not (torch.allclose(mean, torch.tensor(CLIP_MEAN, dtype=torch.float64), atol=atol)
                and torch.allclose(std, torch.tensor(CLIP_STD, dtype=torch.float64), atol=atol)):
            raise AssertionError(f"{name}: Normalize mean {mean.tolist()} std {std.tolist()} != CLIP constants")

    checked = []
    encoders = [(n, m) for n, m in policy.named_modules() if isinstance(m, TransformerObsEncoder)]
    if not encoders:
        raise AssertionError("no TransformerObsEncoder in the policy")
    for name, enc in encoders:
        if not enc.use_vision_norm:
            raise AssertionError(f"{name}: use_vision_norm is False")
        check(f"{name}.model_normalization_transform", enc.model_normalization_transform)
        checked.append(f"{name}.model_normalization_transform (cameras: {list(enc.ordered_rgb_keys)}, separate backbones: {not enc.share_rgb_model})")
    for name, m in policy.named_modules(remove_duplicate=False):
        if isinstance(m, torchvision.transforms.Normalize):
            check(name, m)
    return checked
