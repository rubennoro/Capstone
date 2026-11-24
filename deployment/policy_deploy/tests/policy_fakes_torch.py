# copied from rnoro/deployment:deployment/policy/tests/fakes_torch.py @ 91b178b
# changes: renamed policy_fakes_torch; + TinyPromptModel and FakePromptDataset (the prompting tests)
"""TinyDiffusionModel, TinyPromptModel: stand-ins for a behavior_prompting policy; FakePromptDataset: a stand-in for
UmiTaskDataset(only_prompt=True) (torch; imported only by torch tests)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
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


class TinyPromptModel(TinyDiffusionModel):
    """A prompting model: prompt() keeps the prompt dict (and whether inference_mode was on), reset() clears it, and
    predict_action refuses to run without a prompt (as a prompt encoder would)."""

    def __init__(self, expected: Mapping[str, tuple[int, ...]], **kwargs: Any) -> None:
        super().__init__(expected, **kwargs)
        self.prompted: dict[str, Any] | None = None
        self.prompts: list[dict[str, Any]] = []
        self.prompt_inference_mode: list[bool] = []
        self.events: list[str] = []
        self.split_info: dict[str, bool] | None = {"ep0:block_raise": True}

    def get_training_split_info(self):
        return self.split_info

    def prompt(self, prompt_dict: Mapping[str, Any]) -> None:
        self.prompted = dict(prompt_dict)
        self.prompts.append(self.prompted)
        self.prompt_inference_mode.append(torch.is_inference_mode_enabled())
        self.events.append("prompt")

    def reset(self) -> None:
        super().reset()
        self.prompted = None
        self.events.append("reset")

    def predict_action(self, obs):
        assert self.prompted is not None, "predict_action without a prompt"
        self.events.append("predict")
        return super().predict_action(obs)


class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class FakePromptDataset:
    """UmiTaskDataset(only_prompt=True) as PromptSource uses it, for pair prompting: `tasks` {name: [demo lengths]};
    replay-buffer task i is demo i in the order given. Records its constructor kwargs (hydra.utils.instantiate builds
    it from a dataset_cfg whose _target_ is policy_fakes_torch.FakePromptDataset)."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, tasks: Mapping[str, list[int]] | None = None, missing_key: str | None = None,
                 empty_val: bool = False, **kwargs: Any) -> None:
        """missing_key: raise KeyError(missing_key) (as zarr does for a missing group); empty_val:
        get_validation_dataset() asserts, as the sampler does for a model trained with val_ratio 0."""
        FakePromptDataset.last_kwargs = dict(kwargs)
        if missing_key is not None:
            raise KeyError(missing_key)
        self.empty_val = empty_val
        tasks = dict(tasks) if tasks is not None else {"block_raise": [100, 250, 180], "other": [300]}
        split = kwargs.get("training_split_info")
        self._names: list[str] = []
        lengths: list[int] = []
        for name, task_lengths in tasks.items():
            for length in task_lengths:
                episode = f"ep{len(lengths)}"
                if split is not None:
                    split[f"{episode}:{name}"]           # KeyError for a demo the split does not know
                self._names.append(name)
                lengths.append(int(length))
        self.sampler = _Obj(prompt_indices=np.arange(len(lengths)), task_to_episode_idxs=np.arange(len(lengths)))
        self.replay_buffer = _Obj(task_lengths=np.array(lengths), episode_names=np.array(
            [f"ep{i}" for i in range(len(lengths))]))
        self.got: list[int] = []

    def get_unique_task_name_to_dataset_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, name in enumerate(self._names):
            out.setdefault(name, []).append(i)
        return out

    def get_validation_dataset(self) -> "FakePromptDataset":
        assert not self.empty_val, "No non-error-correction demonstrations found in the prompt-only dataset split."
        val = FakePromptDataset.__new__(FakePromptDataset)
        val.__dict__.update(self.__dict__)
        val._names = self._names[:1]
        val.sampler = _Obj(prompt_indices=self.sampler.prompt_indices[:1],
                           task_to_episode_idxs=self.sampler.task_to_episode_idxs)
        val.is_validation = True
        return val

    def __getitem__(self, i: int) -> dict[str, Any]:
        self.got.append(i)
        steps = int(self.replay_buffer.task_lengths[i]) // 60
        prompt = {"obs": {"camera_right_main_rgb": torch.zeros((steps, 3, 4, 4)),
                          "gripper_right_gripper_width": torch.full((steps, 1), float(i))},
                  "action": torch.zeros((steps, 20, 10)),
                  "metadata": {"task_indices": torch.tensor([i]), "episode_indices": torch.tensor([i])}}
        return {"obs": {"prompt": prompt}, "metadata": {}}
