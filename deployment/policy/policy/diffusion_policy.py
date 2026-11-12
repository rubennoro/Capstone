"""DiffusionPolicy: a trained plain diffusion checkpoint (torch; STRUCTURE.md 4.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from contracts import ActionChunk, ContractLimits, Observation

from policy.base_policy import Policy
from policy.checkpoint_loader import CheckpointLoader
from policy.errors import PolicyLoadError
from policy.key_map import KeyMap
from policy.model_runner import ModelRunner

if TYPE_CHECKING:
    from policy.settings import Settings


class DiffusionPolicy(Policy):
    KIND = "diffusion"

    def __init__(self, runner: ModelRunner, model_reset: Callable[[], None], gripper_width_max_m: float,
                 info: Mapping[str, Any] | None = None) -> None:
        super().__init__(gripper_width_max_m)
        self._runner = runner
        self._model_reset = model_reset
        self._info = dict(info or {})

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DiffusionPolicy":
        loaded = CheckpointLoader(settings.training_timing).load(settings.checkpoint, settings.device,
                                                                 settings.diffusion_steps)
        if loaded.uses_prompting:
            raise PolicyLoadError("checkpoint uses prompting (shape_meta.use_prompting: true)",
                                  "kind diffusion needs a plain diffusion checkpoint")
        runner = ModelRunner(loaded.model, loaded.shape_meta, KeyMap.from_config(settings.key_map),
                             torch.device(settings.device))
        return cls(runner, loaded.model.reset, ContractLimits.load().gripper_width_max_m, info=loaded.info)

    def inputs(self) -> tuple[str, ...]:
        return self._runner.inputs

    def reset(self) -> None:
        self._model_reset()

    def infer(self, obs: Observation) -> ActionChunk:
        return self.make_chunk(*self._runner.run(obs))

    def describe(self) -> dict[str, Any]:
        return {"kind": self.KIND, **self._info, "device": str(self._runner.device),
                "key_map": self._runner.key_map.as_dict(), "inputs": list(self.inputs())}
