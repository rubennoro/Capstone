# from rnoro/main-sim:controlSW/franka_deploy/franka_deploy/inference/light_workspace.py @ 4be9982
# changes: the vision-normalisation check accepts both TimmObsEncoder and TransformerObsEncoder (reference constants
# per backbone, REFERENCE_NORMALIZATION); install marker and docstrings renamed for policy/
"""Inference-only replacement for behavior_prompting's TrainPolicyWorkspace (torch).

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
architecture; no download); `pretrained=True` stays set on the encoder, so every pretrained-dependent branch of the
observation encoders (no GroupNorm swap, no re-init) is identical to training. The vision normalisation constants come
from timm's pretrained_cfg, as in training; `check_vision_normalization` checks them against the backbone's reference
constants (they are not stored in the checkpoint).

`install()` registers this class under the upstream module name in sys.modules, so `hydra.utils.get_class(cfg._target_)`
resolves to it (policy.checkpoint_loader calls it before anything imports the real module).
"""

from __future__ import annotations

import random
import sys
import types
from typing import Any

UPSTREAM_MODULE = "behavior_prompting.train_network.workspace.train_policy_workspace"
HEAVY_MODULES = ("libero", "robosuite", "mujoco", "mujoco_py", "wandb", "pymunk", "pygame", "gym")
LIGHT_MARKER = "__policy_light_workspace__"

# OpenAI CLIP image normalisation (timm vit_base_patch16_clip_224.openai pretrained_cfg).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Reference normalisation per backbone family: (substring of the timm model name, mean, std). A backbone that matches
# none is refused: add its constants here (from its timm pretrained_cfg) before using it.
REFERENCE_NORMALIZATION = (
    ("clip", CLIP_MEAN, CLIP_STD),
)


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
        if getattr(existing, LIGHT_MARKER, False):
            return existing.TrainPolicyWorkspace
        raise RuntimeError(f"{UPSTREAM_MODULE} is already imported; install() must run before it")
    cls = _make_class()
    mod = types.ModuleType(UPSTREAM_MODULE, doc="policy/ inference-only stand-in (see policy/light_workspace.py)")
    mod.TrainPolicyWorkspace = cls
    setattr(mod, LIGHT_MARKER, True)
    sys.modules[UPSTREAM_MODULE] = mod
    import behavior_prompting.train_network.workspace as ws_pkg

    setattr(ws_pkg, "train_policy_workspace", mod)
    return cls


def is_light_workspace(workspace: object) -> bool:
    return type(workspace).__module__ == UPSTREAM_MODULE and hasattr(workspace, "load_report")


def loaded_heavy_modules() -> list[str]:
    return sorted(m for m in sys.modules if m.split(".")[0] in HEAVY_MODULES)


def reference_normalization(model_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The (mean, std) a backbone must be normalised with. AssertionError for an unknown backbone."""
    for pattern, mean, std in REFERENCE_NORMALIZATION:
        if pattern in model_name.lower():
            return mean, std
    raise AssertionError(f"no reference colour normalisation for backbone {model_name!r}: add its mean/std "
                         f"(timm pretrained_cfg) to REFERENCE_NORMALIZATION in policy/light_workspace.py")


def check_vision_normalization(policy, atol: float = 1e-6) -> list[str]:
    """Every observation encoder (TimmObsEncoder: plain diffusion; TransformerObsEncoder: prompting) must have
    use_vision_norm=True and a Normalize with its backbone's reference constants (one Normalize per encoder, applied to
    every camera key); any other torchvision Normalize in the policy must use one of the encoders' constants too.
    Returns the names of the checked modules; raises AssertionError on a mismatch."""
    import torch
    import torchvision
    from behavior_prompting.train_network.model.vision.timm_obs_encoder import TimmObsEncoder
    from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder

    def matches(m, mean_ref, std_ref) -> bool:
        mean = torch.as_tensor(m.mean, dtype=torch.float64).flatten()
        std = torch.as_tensor(m.std, dtype=torch.float64).flatten()
        return (mean.shape == (len(mean_ref),) and std.shape == (len(std_ref),)
                and torch.allclose(mean, torch.tensor(mean_ref, dtype=torch.float64), atol=atol)
                and torch.allclose(std, torch.tensor(std_ref, dtype=torch.float64), atol=atol))

    def describe(m) -> str:
        return (f"mean {torch.as_tensor(m.mean).flatten().tolist()} std {torch.as_tensor(m.std).flatten().tolist()}")

    checked = []
    references = []
    encoders = [(n, m) for n, m in policy.named_modules() if isinstance(m, (TimmObsEncoder, TransformerObsEncoder))]
    if not encoders:
        raise AssertionError("no TimmObsEncoder or TransformerObsEncoder in the policy")
    encoder_norms = set()
    for name, enc in encoders:
        if not enc.use_vision_norm:
            raise AssertionError(f"{name}: use_vision_norm is False")
        mean_ref, std_ref = reference_normalization(enc.model_name)
        norm = enc.model_normalization_transform
        if not matches(norm, mean_ref, std_ref):
            raise AssertionError(f"{name}.model_normalization_transform: Normalize {describe(norm)} != the reference "
                                 f"constants of {enc.model_name!r}")
        references.append((mean_ref, std_ref))
        encoder_norms.add(id(norm))
        cameras = getattr(enc, "ordered_rgb_keys", None) or getattr(enc, "rgb_keys", [])
        checked.append(f"{name or '<root>'}.model_normalization_transform ({type(enc).__name__}, "
                       f"backbone {enc.model_name}, cameras: {list(cameras)}, separate backbones: "
                       f"{not enc.share_rgb_model})")
    for name, m in policy.named_modules(remove_duplicate=False):
        if isinstance(m, torchvision.transforms.Normalize) and id(m) not in encoder_norms:
            if not any(matches(m, mean_ref, std_ref) for mean_ref, std_ref in references):
                raise AssertionError(f"{name}: Normalize {describe(m)} != the encoders' reference constants")
    return checked
