# new (STRUCTURE.md 5.2, 12)
"""precision (STRUCTURE.md 5.2, 5.3; decision 5): bf16 = autocast around predict_action, weights stay fp32, output
float32. Torch env only."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch tests run in the torch environment")

from policy_fakes import plain_diffusion_shape_meta  # noqa: E402
from policy_fakes_torch import TinyDiffusionModel  # noqa: E402

from policy_deploy.contracts import load_config  # noqa: E402
from policy_deploy.policy.errors import PolicyLoadError  # noqa: E402
from policy_deploy.policy.key_map import KeyMap  # noqa: E402
from policy_deploy.policy.model_runner import ModelRunner  # noqa: E402
from policy_deploy.policy.neutral_observation import neutral_observation  # noqa: E402

EXPECTED = {"gripper_right_eef_pos": (1, 2, 3), "gripper_right_eef_rot_axis_angle": (1, 2, 6),
            "gripper_right_gripper_width": (1, 2, 1), "camera_right_main_rgb": (1, 2, 3, 224, 224),
            "camera_right_ultrawide_rgb": (1, 2, 3, 224, 224)}


class AutocastProbe(TinyDiffusionModel):
    def predict_action(self, obs):
        self.autocast_seen = torch.is_autocast_cpu_enabled() or torch.is_autocast_enabled()
        out = super().predict_action(obs)
        return {"action_pred": out["action_pred"].to(torch.bfloat16) if self.autocast_seen else out["action_pred"]}


def _runner(precision):
    model = AutocastProbe(EXPECTED)
    key_map = KeyMap.from_config(load_config("policy", None)["key_map"])
    return ModelRunner(model, plain_diffusion_shape_meta(), key_map, torch.device("cpu"), precision=precision), model


@pytest.mark.parametrize("precision,autocast", [("fp32", False), ("bf16", True)])
def test_autocast_only_for_bf16_and_output_float32(precision, autocast):
    runner, model = _runner(precision)
    pos, rot, width = runner.run(neutral_observation(0.05))
    assert model.autocast_seen is autocast
    assert pos.dtype == rot.dtype == width.dtype == np.float32
    assert model.linear.weight.dtype == torch.float32                    # weights stay fp32


def test_unknown_precision_refused():
    with pytest.raises(PolicyLoadError):
        _runner("fp16")
