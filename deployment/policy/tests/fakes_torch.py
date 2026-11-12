"""TinyDiffusionModel: a stand-in for a behavior_prompting policy (torch; imported only by torch tests)."""

from __future__ import annotations

from typing import Mapping

import torch

ACTION_DIM = 10


class TinyDiffusionModel(torch.nn.Module):
    """One small linear layer. predict_action checks its inputs, records them, and returns a valid chunk:
    positions k * 1 mm along x, identity rot6d, width `width` (set it outside the range to test clipping)."""

    def __init__(self, expected: Mapping[str, tuple[int, ...]], horizon: int = 16, width: float = 0.05) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.expected = dict(expected)          # model key -> shape incl. batch, e.g. (1, 2, 3, 224, 224)
        self.horizon = horizon
        self.width = width
        self.num_inference_steps = 16
        self.reset_calls = 0
        self.received: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert set(obs) == set(self.expected), (sorted(obs), sorted(self.expected))
        for key, value in obs.items():
            assert value.dtype == torch.float32, (key, value.dtype)
            assert tuple(value.shape) == self.expected[key], (key, tuple(value.shape))
            if len(value.shape) == 5:
                assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0, key
        self.received.append({k: v.clone() for k, v in obs.items()})
        action = torch.zeros((1, self.horizon, ACTION_DIM), dtype=torch.float32)
        action[0, :, 0] = torch.arange(self.horizon, dtype=torch.float32) * 0.001
        action[0, :, 3] = 1.0
        action[0, :, 7] = 1.0
        action[0, :, 9] = self.width
        return {"action_pred": action}
