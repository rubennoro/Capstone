# adapted from rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd
# changes: PolicyInferenceNodeRealEnv's policy_prompt / policy_reset topics as a PromptablePolicy over DiffusionPolicy
# (the same ModelRunner and predict_action); prompts from PromptSource; the warm-up prompt is the longest demo; no
# --vis-prompt, attention visualisation or tokenizer (STRUCTURE.md 5.2); passes policy.gripper_map on to DiffusionPolicy
"""PromptableDiffusionPolicy: a trained prompting diffusion checkpoint, kind diffusion (torch; STRUCTURE.md 5.2).

The order is reset -> prompt -> predict_action..., as in training's evaluator (umi_prompting_runner.py) and
detached_inference: model.prompt(prompt_dict) once after each reset (it normalises the prompt itself and keeps it in
the observation encoder); model.reset() clears it. The prompt runs under torch.inference_mode in fp32, as there, also
when policy.precision is bf16 (the autocast wraps predict_action only, ModelRunner).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Mapping

import torch
from deploy_contracts import ContractLimits

from policy_deploy.policy.checkpoint_loader import LoadedModel
from policy_deploy.policy.diffusion_policy import DiffusionPolicy
from policy_deploy.policy.gripper_map import GripperMap
from policy_deploy.policy.key_map import KeyMap
from policy_deploy.policy.model_runner import ModelRunner
from policy_deploy.policy.prompt_source import PromptSource
from policy_deploy.policy.promptable_policy import PromptablePolicy

if TYPE_CHECKING:
    from policy_deploy.policy.settings import Settings

log = logging.getLogger("policy.prompt")


class PromptableDiffusionPolicy(DiffusionPolicy, PromptablePolicy):
    KIND = "diffusion"
    USES_PROMPT = True

    def __init__(self, runner: ModelRunner, model: torch.nn.Module, source: PromptSource, gripper_width_max_m: float,
                 info: Mapping[str, Any] | None = None, gripper_map: GripperMap | None = None) -> None:
        super().__init__(runner, model.reset, gripper_width_max_m, info=info, gripper_map=gripper_map)
        self._model = model
        self._source = source
        self.last_prompt: dict[str, Any] | None = None       # the prompt in effect (None after a reset)

    @classmethod
    def from_loaded(cls, loaded: LoadedModel, settings: "Settings") -> "PromptableDiffusionPolicy":
        """The runner's C4 checks first, then the prompt dataset. Raises PolicyLoadError."""
        runner = ModelRunner(loaded.model, loaded.shape_meta, KeyMap.from_config(settings.key_map),
                             torch.device(settings.device), precision=settings.precision)
        t0 = time.monotonic()
        source = PromptSource.load(settings.prompt_dataset, settings.prompt_split, loaded.dataset_cfg,
                                   loaded.model.get_training_split_info(), settings.device)
        log.info("prompt dataset %s (%s split) loaded in %.2f s: %s", settings.prompt_dataset, settings.prompt_split,
                 time.monotonic() - t0, ", ".join(f"{t} ({n} demos)" for t, n in source.tasks().items()))
        return cls(runner, loaded.model, source, ContractLimits.load().gripper_width_max_m, info=loaded.info,
                   gripper_map=settings.gripper_map)

    def reset(self) -> None:
        self.last_prompt = None
        super().reset()                                      # model.reset(): clears the prompt

    def prompt(self, task_name: str, prompt_index: int) -> None:
        """Raises BadRequest for an unknown task or an index out of range (from PromptSource). Returns once the
        prompt has finished on the device (CUDA synchronised), so callers' timings are real."""
        demo = self._source.demo(task_name, prompt_index)
        t0 = time.monotonic()
        prompt = self._source.get(task_name, prompt_index)
        with torch.inference_mode():
            self._model.prompt(prompt)
        if self._runner.device.type == "cuda":
            torch.cuda.synchronize(self._runner.device)       # the prompt has finished on the GPU (prompt_s)
        demo["prompt_s"] = round(time.monotonic() - t0, 3)
        self.last_prompt = demo
        log.info("prompt applied: task %r, index %d (%s steps, episode %s) in %.3f s", task_name, prompt_index,
                 demo["steps"], demo["episode"], demo["prompt_s"])

    def warm_up_prompt(self) -> tuple[str, int]:
        return self._source.longest()

    def describe(self) -> dict[str, Any]:
        warm = self._source.demo(*self._source.longest())
        return {**super().describe(), **self._source.describe(), "prompt_warm_up_demo": warm}
