# adapted from rnoro/deployment:deployment/policy/policy/diffusion_policy.py @ 91b178b
# changes: passes policy.precision to the runner (and reports it in describe()); memory_stats() gives the CUDA
# allocator's peaks for startup.json; imports
"""DiffusionPolicy: a trained plain diffusion checkpoint (torch; STRUCTURE.md 5.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from deploy_contracts import ContractLimits

from policy_deploy.contracts import ActionChunk, Observation
from policy_deploy.policy.base_policy import Policy
from policy_deploy.policy.checkpoint_loader import CheckpointLoader
from policy_deploy.policy.errors import PolicyLoadError
from policy_deploy.policy.key_map import KeyMap
from policy_deploy.policy.model_runner import ModelRunner

if TYPE_CHECKING:
    from policy_deploy.policy.settings import Settings


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
                             torch.device(settings.device), precision=settings.precision)
        return cls(runner, loaded.model.reset, ContractLimits.load().gripper_width_max_m, info=loaded.info)

    def inputs(self) -> tuple[str, ...]:
        return self._runner.inputs

    def reset(self) -> None:
        self._model_reset()

    def infer(self, obs: Observation) -> ActionChunk:
        return self.make_chunk(*self._runner.run(obs))

    def describe(self) -> dict[str, Any]:
        return {"kind": self.KIND, **self._info, "device": str(self._runner.device),
                "precision": self._runner.precision, "key_map": self._runner.key_map.as_dict(),
                "inputs": list(self.inputs())}

    def memory_stats(self) -> dict[str, Any]:
        """The CUDA allocator's peaks since start (after the warm-up: the model's real need), in MB."""
        if self._runner.device.type != "cuda" or not torch.cuda.is_available():
            return {}
        return {"max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
