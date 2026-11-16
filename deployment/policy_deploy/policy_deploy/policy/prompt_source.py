# adapted from rnoro/deployment:behavior_prompting/behavior_prompting/deployment/detached_inference.py @ edfe8bd
# changes: only PolicyInferenceNodeRealEnv.__init__'s prompt-dataset loading and _handle_prompt's prompt dict, as a
# class of their own; prompt_split train | val | all (all = the original --eval-dataset); cache_dir None (the .zarr.zip
# is read in place, no LMDB cache is written); BadRequest instead of assert; the longest demo for the warm-up (from the
# replay buffer's task_lengths, no demo loaded); no --vis-prompt, attention logger or tokenizer (STRUCTURE.md 5.2)
"""PromptSource: the processed demonstration dataset -> one prompt dict for model.prompt() (torch).

A prompt is a whole recorded demonstration of the processed training dataset (.zarr.zip), not text: `task_name`
selects the task, `prompt_index` the demo of that task (0-based, in the dataset's own order).

prompt_split (policy.prompt_split):
- train: the demos the model trained on (the model's training split info; the original's default);
- val: the demos the model held out for validation (`get_validation_dataset()` of the train set);
- all: every demo of the dataset, without the model's split info (the original's --eval-dataset: a dataset the
  model never trained on).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from policy_deploy.policy.errors import BadRequest, PolicyLoadError
from policy_deploy.policy.settings import PROMPT_SPLITS as SPLITS

log = logging.getLogger("policy.prompt")

DATASET_HINT = "check policy.prompt_dataset in station.yaml (the processed .zarr.zip of the demos, on this laptop)"
ENV_HINT = "run the policy with python.torch, the torch environment (station.yaml)"


class PromptSource:
    def __init__(self, dataset: Any, device: torch.device | str, path: Path | str | None = None,
                 split: str = "train") -> None:
        """`dataset`: a behavior_prompting UmiTaskDataset built with only_prompt=True (or a stand-in with
        get_unique_task_name_to_dataset_indices(), __getitem__ and, optionally, sampler.prompt_indices and
        replay_buffer.task_lengths / episode_names)."""
        self._dataset = dataset
        self.device = torch.device(device)
        self.path = None if path is None else Path(path)
        self.split = split
        self._tasks: dict[str, list[int]] = {name: list(indices) for name, indices in
                                             dataset.get_unique_task_name_to_dataset_indices().items()}
        if not self._tasks:
            raise PolicyLoadError(f"prompt dataset has no demos in split {split!r}: {self.path}",
                                  "choose another policy.prompt_split (train | val | all), or " + DATASET_HINT)
        self._lengths, self._names = _demo_lengths_and_names(dataset)

    @classmethod
    def load(cls, path: Path, split: str, dataset_cfg: Mapping[str, Any] | None,
             training_split_info: Mapping[str, Any] | None, device: str) -> "PromptSource":
        """Build the dataset from the checkpoint's resolved cfg.task.dataset, as detached_inference does. Raises
        PolicyLoadError."""
        path = Path(path)
        if split not in SPLITS:
            raise PolicyLoadError(f"unknown prompt split {split!r}", f"set policy.prompt_split to {' | '.join(SPLITS)}")
        if not path.exists():
            raise PolicyLoadError(f"prompt dataset not found: {path}", DATASET_HINT)
        if not (path.name.endswith(".zarr.zip") or (path.is_dir() and path.name.endswith(".zarr"))):
            raise PolicyLoadError(f"prompt dataset is not a .zarr.zip file (or a .zarr folder): {path}", DATASET_HINT)
        if not dataset_cfg or "_target_" not in dataset_cfg:
            raise PolicyLoadError("the checkpoint has no task.dataset config to build the prompt dataset from",
                                  "use a checkpoint written by behavior_prompting's training")
        if split != "all" and training_split_info is None:
            raise PolicyLoadError(f"prompt_split {split} needs the model's training split info, and the checkpoint "
                                  f"has none", "set policy.prompt_split: all")
        try:
            import hydra
            import omegaconf
            from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
        except ImportError as e:
            raise PolicyLoadError(f"{e.name or 'a dataset library'} is not importable: {e}", ENV_HINT) from e

        register_codecs(verbose=False)
        cfg = omegaconf.OmegaConf.create(dict(dataset_cfg))
        cfg.val_ratio = 0          # as detached_inference: no demo of a dataset loaded for prompting is held out again
        cfg.cache_dir = None       # read the .zarr.zip in place; a cache_dir would write an LMDB copy (cluster path)
        try:
            dataset = hydra.utils.instantiate(
                cfg, only_prompt=True, dataset_path=str(path), allow_zip_file=True,
                training_split_info=None if split == "all" else {str(k): bool(v)            # np.bool_ -> bool
                                                                 for k, v in training_split_info.items()})
            if split == "val":
                dataset = dataset.get_validation_dataset()
        except Exception as e:
            cause = e if isinstance(e, KeyError) else e.__cause__          # hydra wraps the target's exception
            if split != "all" and isinstance(cause, KeyError):
                raise PolicyLoadError(f"prompt dataset {path} has a demo the model's training split info does not "
                                      f"know ({cause})", "is it the dataset the model trained on? For any other "
                                      "dataset set policy.prompt_split: all") from e
            raise PolicyLoadError(f"prompt dataset failed to load: {path}: {type(e).__name__}: {e}",
                                  "is it the processed .zarr.zip of this task's demos (the training dataset)? "
                                  + DATASET_HINT) from e
        return cls(dataset, device, path, split)

    def tasks(self) -> dict[str, int]:
        """{task_name: number of demos} in this split."""
        return {name: len(indices) for name, indices in self._tasks.items()}

    def _dataset_index(self, task_name: str, prompt_index: int) -> int:
        indices = self._tasks.get(task_name)
        if indices is None:
            raise BadRequest(f"prompt: unknown task {task_name!r}; the prompt dataset ({self.split} split) has "
                             f"{', '.join(repr(t) for t in self._tasks)}")
        if not 0 <= prompt_index < len(indices):
            raise BadRequest(f"prompt: prompt_index {prompt_index} out of range for task {task_name!r}: "
                             f"0..{len(indices) - 1} ({len(indices)} demos in the {self.split} split)")
        return indices[prompt_index]

    def demo(self, task_name: str, prompt_index: int) -> dict[str, Any]:
        """{"task_name", "prompt_index", "steps", "episode"} of one demo (steps and episode None if unknown)."""
        i = self._dataset_index(task_name, prompt_index)
        return {"task_name": task_name, "prompt_index": prompt_index,
                "steps": None if self._lengths is None else int(self._lengths[i]),
                "episode": None if self._names is None else self._names[i]}

    def longest(self) -> tuple[str, int]:
        """(task_name, prompt_index) of the longest demo (the warm-up's prompt: the largest prompt tensors). Read
        from the replay buffer's task_lengths, nothing is loaded. Without lengths (sequence prompting, where every
        prompt is prompt_sequence_length steps, or a stand-in dataset): the first demo of the first task."""
        best: tuple[str, int] | None = None
        best_len = -1
        for name, indices in self._tasks.items():
            for k, i in enumerate(indices):
                length = 0 if self._lengths is None else int(self._lengths[i])
                if length > best_len:
                    best, best_len = (name, k), length
        assert best is not None
        return best

    def get(self, task_name: str, prompt_index: int) -> dict[str, Any]:
        """The prompt dict for model.prompt(): every leaf a tensor with a batch dimension, on the device (as
        detached_inference._handle_prompt). Raises BadRequest for an unknown task or index."""
        sample = self._dataset[self._dataset_index(task_name, prompt_index)]
        return _apply(sample["obs"]["prompt"], lambda x: torch.from_numpy(np.array(x)).unsqueeze(0).to(self.device))

    def describe(self) -> dict[str, Any]:
        return {"prompt_dataset": None if self.path is None else str(self.path), "prompt_split": self.split,
                "prompt_tasks": self.tasks()}


def _apply(x: Mapping[str, Any], fn) -> dict[str, Any]:
    """behavior_prompting.common.pytorch_util.dict_apply."""
    return {k: _apply(v, fn) if isinstance(v, Mapping) else fn(v) for k, v in x.items()}


def _demo_lengths_and_names(dataset: Any) -> tuple[np.ndarray | None, list[str] | None]:
    """Per dataset index: the demo's recorded steps and its episode name (pair prompting: dataset index ->
    sampler.prompt_indices -> replay-buffer task). (None, None) when the dataset does not expose them."""
    sampler = getattr(dataset, "sampler", None)
    rb = getattr(dataset, "replay_buffer", None)
    task_idx = getattr(sampler, "prompt_indices", None)
    if task_idx is None or rb is None:
        return None, None
    task_idx = np.asarray(task_idx)
    try:
        lengths = np.asarray(rb.task_lengths[:])[task_idx]
    except Exception:
        return None, None
    try:
        episode_of_task = np.asarray(sampler.task_to_episode_idxs)
        names = [str(n) for n in np.asarray(rb.episode_names[:])[episode_of_task[task_idx]]]
    except Exception:
        names = None
    return lengths, names
