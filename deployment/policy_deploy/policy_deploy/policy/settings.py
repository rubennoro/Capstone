# adapted from rnoro/deployment:deployment/policy/policy/settings.py @ 91b178b
# changes: the station schema of STRUCTURE.md 2.4 (Station.load + load_config("policy", station) from
# policy_deploy.contracts replace the local merge); the endpoint is station.policy_endpoint() (ports.policy); the
# checkpoint path through contracts.resolve_path; adds precision (fp32 | bf16) and mock_kind (the kind that runs
# under --mock); the session folder is the process's (ProcessArgs), not part of the settings; session_root kept;
# adds prompt_dataset (a path, like checkpoint) and prompt_split (train | val | all) for a prompting checkpoint;
# adds gripper_map (handheld <-> robot widths, policy/gripper_map.py).
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/policy/settings.py @ f632576
# changes: replay_dataset / replay_index (policy.replay_episode.dataset / .index, required for kind replay) and
# exec_h (the station's manager.exec_h, which the replay advances by) (sim_deploy); gripper_map also for kind replay
"""Settings: config/policy.yaml + the station's `policy:` section, validated (STRUCTURE.md 5.3)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from policy_deploy.contracts import Station, StationError, load_config, resolve_path
from policy_deploy.policy.errors import SettingsError
from policy_deploy.policy.gripper_map import GripperMap
from policy_deploy.policy.policy_factory import KINDS

SECTION = "policy"
PRECISIONS = ("fp32", "bf16")
PROMPT_SPLITS = ("train", "val", "all")         # policy.prompt_split (policy/prompt_source.py)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Settings:
    kind: str | None
    mock: bool
    mock_kind: str
    active_kind: str                       # mock_kind if mock else kind: the kind that runs
    endpoint: str                          # tcp://127.0.0.1:<ports.policy>
    session_root: Path
    station_file: Path
    checkpoint: Path | None
    device: str
    diffusion_steps: int | None
    precision: str                         # fp32 | bf16 (autocast around predict_action)
    prompt_dataset: Path | None            # the demos a prompting checkpoint is prompted with (.zarr.zip)
    prompt_split: str                      # train | val | all
    gripper_map: GripperMap | None         # handheld <-> robot widths (kinds diffusion and replay); None: unchanged
    replay_dataset: Path | None            # kind replay: the packaged dataset (.zarr.zip or .zarr folder)
    replay_index: int                      # kind replay: the episode's index in it
    exec_h: int                            # the station's manager.exec_h (kind replay advances by it per infer)
    warmup_inferences: int
    warmup_gripper_width_m: float
    reset_max_s: float
    poll_s: float
    self_check_timeout_s: float
    training_timing: Mapping[str, int]
    key_map: Mapping[str, Mapping[str, str]]

    @classmethod
    def load(cls, station_file: Path | None, mock: bool) -> "Settings":
        """Raises SettingsError listing every problem (the station's and this section's)."""
        try:
            station = Station.load(station_file, mock=mock)
            cfg = load_config(SECTION, station)
            exec_h = load_config("manager", station).get("exec_h")
        except StationError as e:
            raise SettingsError(e.problems) from None
        source = station.path.name
        problems: list[str] = []
        kind_names = " or ".join(sorted(KINDS))

        kind = cfg.get("kind")
        if kind is None:
            if not mock:
                problems.append(f"{SECTION}.kind is not set: choose {kind_names} in {source} ({SECTION}.kind)")
        elif not isinstance(kind, str) or kind not in KINDS:
            problems.append(f"{source}: {SECTION}.kind: unknown kind {kind!r} (choose {kind_names})")
        mock_kind = cfg.get("mock_kind")
        if not isinstance(mock_kind, str) or mock_kind not in KINDS:
            problems.append(f"{source}: {SECTION}.mock_kind: unknown kind {mock_kind!r} (choose {kind_names})")
            mock_kind = "hold_still"
        active_kind = mock_kind if mock else kind

        checkpoint = cfg.get("checkpoint")
        checkpoint_path: Path | None = None
        if checkpoint is not None:
            if isinstance(checkpoint, str) and checkpoint:
                checkpoint_path = resolve_path(station, SECTION, cfg, "checkpoint")
                checkpoint_path = checkpoint_path.resolve() if checkpoint_path is not None else None
            else:
                problems.append(f"{source}: {SECTION}.checkpoint: expected a path or null, got {checkpoint!r}")
        elif active_kind == "diffusion":
            problems.append(f"{source}: {SECTION}.checkpoint: required for kind diffusion "
                            f"(the trained .ckpt file)")

        prompt_dataset = cfg.get("prompt_dataset")
        prompt_dataset_path: Path | None = None
        if prompt_dataset is not None:
            if isinstance(prompt_dataset, str) and prompt_dataset:
                prompt_dataset_path = resolve_path(station, SECTION, cfg, "prompt_dataset")
                prompt_dataset_path = prompt_dataset_path.resolve() if prompt_dataset_path is not None else None
            else:
                problems.append(f"{source}: {SECTION}.prompt_dataset: expected a path or null, got {prompt_dataset!r}")
        prompt_split = cfg.get("prompt_split")
        _check(problems, source, "prompt_split", prompt_split, prompt_split in PROMPT_SPLITS,
               " or ".join(PROMPT_SPLITS))

        gripper_cfg = cfg.get("gripper_map")
        gripper_map: GripperMap | None = None
        if gripper_cfg is not None:
            gripper_problems = GripperMap.problems(gripper_cfg)
            problems.extend(f"{source}: {SECTION}.gripper_map: {p}" for p in gripper_problems)
            if not gripper_problems:
                gripper_map = GripperMap.from_config(gripper_cfg)

        replay = cfg.get("replay_episode")
        replay_path: Path | None = None
        replay_index = 0
        if isinstance(replay, Mapping):
            dataset = replay.get("dataset")
            if dataset is not None:
                if isinstance(dataset, str) and dataset:
                    replay_path = resolve_path(station, SECTION, cfg, "replay_episode.dataset")
                    replay_path = replay_path.resolve() if replay_path is not None else None
                else:
                    problems.append(f"{source}: {SECTION}.replay_episode.dataset: expected a path or null, "
                                    f"got {dataset!r}")
            elif active_kind == "replay":
                problems.append(f"{source}: {SECTION}.replay_episode.dataset: required for kind replay "
                                f"(a packaged .zarr.zip or .zarr)")
            replay_index = replay.get("index")
            _check(problems, source, "replay_episode.index", replay_index, _is_int(replay_index) and replay_index >= 0,
                   "an integer >= 0")
        else:
            _check(problems, source, "replay_episode", replay, False, "a map")
        if active_kind == "replay" and not (_is_int(exec_h) and exec_h >= 1):
            problems.append(f"{source}: manager.exec_h: expected an integer >= 1 (kind replay advances by it), "
                            f"got {exec_h!r}")

        _check(problems, source, "device", cfg.get("device"), cfg.get("device") in ("cuda", "cpu"), "cuda or cpu")
        steps = cfg.get("diffusion_steps")
        _check(problems, source, "diffusion_steps", steps, steps is None or (_is_int(steps) and steps >= 1),
               "an integer >= 1 or null")
        precision = cfg.get("precision")
        _check(problems, source, "precision", precision, precision in PRECISIONS, " or ".join(PRECISIONS))
        warmups = cfg.get("warmup_inferences")
        _check(problems, source, "warmup_inferences", warmups, _is_int(warmups) and warmups >= 1, "an integer >= 1")
        width = cfg.get("warmup_gripper_width_m")
        _check(problems, source, "warmup_gripper_width_m", width, _is_number(width) and width >= 0, "a number >= 0")
        for key in ("reset_max_s", "poll_s", "self_check_timeout_s"):
            value = cfg.get(key)
            _check(problems, source, key, value, _is_number(value) and value > 0, "a number > 0")

        timing = cfg.get("training_timing")
        if isinstance(timing, Mapping):
            for key, value in timing.items():
                _check(problems, source, f"training_timing.{key}", value, _is_int(value), "an integer")
        else:
            _check(problems, source, "training_timing", timing, False, "a map")
        key_map = cfg.get("key_map")
        if isinstance(key_map, Mapping):
            for part in ("obs", "action"):
                names = key_map.get(part)
                if not isinstance(names, Mapping):
                    _check(problems, source, f"key_map.{part}", names, False, "a map")
                    continue
                for field, model_name in names.items():
                    _check(problems, source, f"key_map.{part}.{field}", model_name,
                           isinstance(model_name, str) and bool(model_name), "a non-empty string")
        else:
            _check(problems, source, "key_map", key_map, False, "a map")

        if problems:
            raise SettingsError(problems)
        return cls(
            kind=kind, mock=mock, mock_kind=mock_kind, active_kind=active_kind, endpoint=station.policy_endpoint(),
            session_root=station.session_root, station_file=station.path, checkpoint=checkpoint_path,
            device=cfg["device"], diffusion_steps=steps, precision=precision, prompt_dataset=prompt_dataset_path,
            prompt_split=prompt_split, gripper_map=gripper_map, replay_dataset=replay_path,
            replay_index=replay_index, exec_h=exec_h, warmup_inferences=warmups,
            warmup_gripper_width_m=float(width), reset_max_s=float(cfg["reset_max_s"]), poll_s=float(cfg["poll_s"]),
            self_check_timeout_s=float(cfg["self_check_timeout_s"]), training_timing=dict(timing),
            key_map={part: dict(key_map[part]) for part in ("obs", "action")},
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy (paths as strings), for startup.json."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, GripperMap):
                value = value.as_dict()
            elif isinstance(value, Mapping):
                value = copy.deepcopy(dict(value))
            out[name] = value
        return out


def _check(problems: list[str], source: str, key: str, value: Any, ok: bool, expected: str) -> None:
    if not ok:
        problems.append(f"{source}: {SECTION}.{key}: expected {expected}, got {value!r}")
