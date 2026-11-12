"""Shared fixtures: shape_meta dicts (run_001's and a plain-diffusion one), clocks, station files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from fakes import FakeClock

_LOW_DIM = {"type": "low_dim", "horizon": 2, "ignore_by_policy": False}
_RGB = {"type": "rgb", "horizon": 2, "ignore_by_policy": False, "shape": [3, 224, 224]}

# cfg.task.shape_meta of run_001 (results/block_raise/run_001_2026-10-05/run_config/config.yaml), resolved, main keys.
RUN_001_SHAPE_META: dict[str, Any] = {
    "image_resolution": 224,
    "use_prompting": True,
    "action": {
        "shape": [10], "horizon": 16, "down_sample_steps": 3, "rotation_rep": "rotation_6d",
        "components": [{"type": "right_pos", "size": 3}, {"type": "right_rot", "size": 6},
                       {"type": "right_gripper_width", "size": 1}],
    },
    "obs": {
        "camera_right_main_rgb": {**_RGB, "down_sample_steps": 3},
        "camera_right_ultrawide_rgb": {**_RGB, "down_sample_steps": 1},
        "gripper_right_eef_pos": {**_LOW_DIM, "shape": [3], "down_sample_steps": 3},
        "gripper_right_eef_rot_axis_angle": {**_LOW_DIM, "shape": [6], "raw_shape": [3], "down_sample_steps": 3,
                                             "rotation_rep": "rotation_6d"},
        "gripper_right_gripper_width": {**_LOW_DIM, "shape": [1], "down_sample_steps": 3},
        "gripper_right_eef_rot_axis_angle_wrt_start": {**_LOW_DIM, "shape": [6], "raw_shape": [3],
                                                       "down_sample_steps": 3, "ignore_by_policy": True},
    },
}


def plain_diffusion_shape_meta() -> dict[str, Any]:
    """A hand-written plain-diffusion shape_meta (the default UMI task config, use_prompting false)."""
    meta = copy.deepcopy(RUN_001_SHAPE_META)
    meta["use_prompting"] = False
    return meta


@pytest.fixture
def run_001_shape_meta() -> dict[str, Any]:
    return copy.deepcopy(RUN_001_SHAPE_META)


@pytest.fixture
def plain_shape_meta() -> dict[str, Any]:
    return plain_diffusion_shape_meta()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def write_station(folder: Path, policy_section: dict[str, Any] | None = None, endpoint: str | None =
                  "tcp://127.0.0.1:5640", session_root: str | None = "sessions", **extra: Any) -> Path:
    """A minimal station file (only the keys the policy reads, plus `extra`)."""
    data: dict[str, Any] = {"station_version": 1}
    if session_root is not None:
        data["session_root"] = session_root
    if endpoint is not None:
        data["endpoints"] = {"policy": endpoint}
    if policy_section is not None:
        data["policy"] = policy_section
    data.update(extra)
    path = folder / "station.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def station_file(tmp_path: Path) -> Callable[..., Path]:
    """station_file(policy_section, endpoint=..., session_root=...) -> path of a station file in tmp_path."""
    def make(policy_section: dict[str, Any] | None = None, **kwargs: Any) -> Path:
        return write_station(tmp_path, policy_section, **kwargs)
    return make
