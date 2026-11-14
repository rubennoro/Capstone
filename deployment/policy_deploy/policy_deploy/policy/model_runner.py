# adapted from rnoro/deployment:deployment/policy/policy/model_runner.py @ 91b178b
# (there: from rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd)
# changes: precision (fp32 | bf16): bf16 wraps predict_action in torch.autocast(bfloat16), weights stay fp32, output
# .float() as before; imports
# from rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd
# changes: _process_obs (image layout), the policy_inference branch of run_node and _process_action (with main-sim
# inference/server.py @ 4be9982 node_infer); no resize (images arrive 224x224); keys from KeyMap, not robot{i}_*
"""ModelRunner: Observation -> tensors -> predict_action -> chunk arrays (torch; STRUCTURE.md 5.2).

precision "bf16" wraps predict_action in torch.autocast(<device type>, dtype=torch.bfloat16) (training's
mixed_precision: bf16); the weights stay fp32 and the output is cast back to float32 (.float()) as before.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping

import numpy as np
import torch

from policy_deploy.contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, Observation
from policy_deploy.policy.errors import PolicyLoadError
from policy_deploy.policy.key_map import ACTION_FIELDS, IMAGE_FIELDS, LOW_DIM_SIZES, KeyMap


def check_shape_meta(shape_meta: Mapping[str, Any], key_map: KeyMap) -> list[str]:
    """The shape rows of the checkpoint checks, for the obs keys the model uses: problems as text."""
    problems: list[str] = []
    model_obs = shape_meta["obs"]
    for field in key_map.used_fields(shape_meta):
        key = key_map.model_key(field)
        attr = model_obs[key]
        if attr.get("horizon") != OBS_STEPS:
            problems.append(f"shape_meta.obs.{key}.horizon is {attr.get('horizon')}, C4 sends {OBS_STEPS} steps")
        shape = list(attr.get("shape", []))
        want = [3, IMAGE_SIZE, IMAGE_SIZE] if field in IMAGE_FIELDS else [LOW_DIM_SIZES[field]]
        if shape != want:
            extra = " (images are never resized here)" if field in IMAGE_FIELDS else ""
            problems.append(f"shape_meta.obs.{key}.shape is {shape}, C4 {field} needs {want}{extra}")
    action = shape_meta["action"]
    horizon = action.get("horizon")
    if not isinstance(horizon, int) or horizon < ACTION_STEPS:
        problems.append(f"model action horizon {horizon} < {ACTION_STEPS} (C4 action_steps): use a checkpoint "
                        f"trained with action_horizon >= {ACTION_STEPS}")
    total = sum(int(c["size"]) for c in action.get("components", []))
    if list(action.get("shape", [])) != [total]:
        problems.append(f"shape_meta.action.shape is {action.get('shape')}, but the components add up to {total}")
    return problems


PRECISIONS = ("fp32", "bf16")


class ModelRunner:
    def __init__(self, model: torch.nn.Module, shape_meta: Mapping, key_map: KeyMap, device: torch.device,
                 precision: str = "fp32") -> None:
        """Checks shape_meta against C4 and the key map; raises PolicyLoadError."""
        if precision not in PRECISIONS:
            raise PolicyLoadError(f"unknown precision {precision!r}",
                                  f"set policy.precision to {' or '.join(PRECISIONS)}")
        key_map.check(shape_meta)
        problems = check_shape_meta(shape_meta, key_map)
        if problems:
            raise PolicyLoadError("checkpoint does not fit C4: " + "; ".join(problems),
                                  "use a checkpoint trained for the rig's C4 layout")
        self.model = model
        self.model.eval()
        self.key_map = key_map
        self.device = torch.device(device)
        self.inputs: tuple[str, ...] = key_map.used_fields(shape_meta)
        self._slices = key_map.action_slices(shape_meta)
        self.precision = precision

    def model_inputs(self, obs: Observation) -> dict[str, np.ndarray]:
        """The numpy arrays the model gets, under its own key names (identical to detached_inference._process_obs)."""
        result: dict[str, np.ndarray] = {}
        for field in self.inputs:
            value = getattr(obs, field)
            if field in IMAGE_FIELDS:
                imgs = np.array(value)  # (T, H, W, 3) uint8, a writable copy
                result[self.key_map.model_key(field)] = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
            else:
                result[self.key_map.model_key(field)] = np.array(value, dtype=np.float32)
        return result

    def run(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """eef_pos (16, 3), eef_rot6d (16, 6), gripper_width_m (16, 1): float32, C-contiguous, unclipped."""
        tensors = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in self.model_inputs(obs).items()}
        with torch.inference_mode(), self._autocast():
            result = self.model.predict_action(tensors)
        action = result["action_pred"][0].detach().float().cpu().numpy()     # (H, D); waits for the GPU
        parts = self.key_map.split_action(action[:ACTION_STEPS], self._slices)
        return tuple(np.ascontiguousarray(parts[f], dtype=np.float32) for f in ACTION_FIELDS)   # type: ignore

    def _autocast(self):
        """bf16: autocast on the runner's device (weights stay fp32); fp32: nothing."""
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()
