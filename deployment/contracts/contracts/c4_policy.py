"""C4: message_comms <-> policy (STRUCTURE.md section 4.6).

Everything relative is relative to the anchor (the hand pose at the newest main image time), oldest step first.
Value checks on float32 arrays compare in float64 (cast first), so float32 rounding can't slip past a limit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from contracts._rot6d import rot6d_problem
from contracts.fields import array, scalar, str_list
from contracts.limits import ContractLimits
from contracts.message import Message
from contracts.registry import register
from contracts.sizes import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS

IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_IMAGES = (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3)


def _f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


@register
@dataclass(frozen=True, eq=False)
class PolicyConfigRequest(Message):
    SCHEMA = "c4.config/v1"


@register
@dataclass(frozen=True, eq=False)
class Reset(Message):
    """Clears the policy's history; answered with Ack."""

    SCHEMA = "c4.reset/v1"


@register
@dataclass(frozen=True, eq=False)
class Prompt(Message):
    """Answered with Ack. Only sent when PolicyConfig.uses_prompt, once after each reset."""

    SCHEMA = "c4.prompt/v1"
    task_name: str = scalar("str")
    prompt_index: int = scalar("int", min=0)


@register
@dataclass(frozen=True, eq=False)
class Observation(Message):
    """The infer request."""

    SCHEMA = "c4.infer/v1"
    eef_pos: np.ndarray = array("float32", (OBS_STEPS, 3))              # m, relative to the anchor; last row 0
    eef_rot6d: np.ndarray = array("float32", (OBS_STEPS, 6))            # relative to the anchor; last row identity
    eef_rot6d_wrt_start: np.ndarray = array("float32", (OBS_STEPS, 6))  # relative to the episode start pose
    gripper_width_m: np.ndarray = array("float32", (OBS_STEPS, 1))      # absolute opening, m
    main: np.ndarray = array("uint8", _IMAGES)                          # straight from C1
    ultrawide: np.ndarray = array("uint8", _IMAGES)                     # straight from C1

    def check_values(self) -> None:
        limits = ContractLimits.load()
        for name in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m"):
            if not np.all(np.isfinite(getattr(self, name))):
                self.fail(name, "not finite")
        tol = limits.anchor_row_tol
        if np.any(np.abs(_f64(self.eef_pos[-1])) > tol):
            self.fail("eef_pos", f"last row {self.eef_pos[-1].tolist()} must be 0 (the anchor) within {tol}")
        if np.any(np.abs(_f64(self.eef_rot6d[-1]) - np.array(IDENTITY_ROT6D)) > tol):
            self.fail("eef_rot6d", f"last row {self.eef_rot6d[-1].tolist()} must be [1,0,0,0,1,0] within {tol}")
        for name in ("eef_rot6d", "eef_rot6d_wrt_start"):
            problem = rot6d_problem(getattr(self, name), limits.rot6d_norm_tol, limits.rot6d_dot_max)
            if problem is not None:
                self.fail(name, problem)
        if np.any(_f64(self.gripper_width_m) < 0.0):
            self.fail("gripper_width_m", f"negative width: {float(_f64(self.gripper_width_m).min())}")


OBSERVATION_INPUTS = tuple(f.name for f in dataclasses.fields(Observation))


@register
@dataclass(frozen=True, eq=False)
class PolicyConfig(Message):
    """The config reply. message_comms compares the four sizes with contracts.sizes and concludes if they differ."""

    SCHEMA = "c4.config.reply/v1"
    obs_steps: int = scalar("int", min=1)
    action_steps: int = scalar("int", min=1)
    step_ms: int = scalar("int", min=1)
    image_size: int = scalar("int", min=1)
    inputs: tuple[str, ...] = str_list(choices=OBSERVATION_INPUTS)   # which Observation fields the model uses
    uses_prompt: bool = scalar("bool")


@register
@dataclass(frozen=True, eq=False)
class ActionChunk(Message):
    """The infer reply. A chunk that fails these checks is broken and ends the run. Step 0 near the anchor is not a
    contract rule (message_comms checks it). Widths are clipped by the policy before building; the validator refuses
    an out-of-range width instead of clipping."""

    SCHEMA = "c4.chunk/v1"
    eef_pos: np.ndarray = array("float32", (ACTION_STEPS, 3))           # m, relative to the anchor, STEP_MS apart
    eef_rot6d: np.ndarray = array("float32", (ACTION_STEPS, 6))         # relative to the anchor
    gripper_width_m: np.ndarray = array("float32", (ACTION_STEPS, 1))   # absolute, in [0, gripper_width_max_m]
    width_clipped: int = scalar("uint64", max=ACTION_STEPS)             # widths the policy clipped into range

    def check_values(self) -> None:
        limits = ContractLimits.load()
        for name in ("eef_pos", "eef_rot6d", "gripper_width_m"):
            if not np.all(np.isfinite(getattr(self, name))):
                self.fail(name, "not finite")
        problem = rot6d_problem(self.eef_rot6d, limits.rot6d_norm_tol, limits.rot6d_dot_max)
        if problem is not None:
            self.fail("eef_rot6d", problem)
        widths = _f64(self.gripper_width_m)
        top = limits.gripper_width_max_m
        bad = np.flatnonzero((widths < 0.0) | (widths > top))
        if bad.size:
            i = int(bad[0])
            self.fail("gripper_width_m", f"step {i}: width {float(widths.flat[i])!r} m outside [0, {top}]")
