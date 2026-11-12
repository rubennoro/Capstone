"""Policy: the interface every policy implements, and the shared chunk builder that clips widths (STRUCTURE.md 4.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, ActionChunk, Observation, PolicyConfig

if TYPE_CHECKING:
    from policy.settings import Settings


def float32_max_at_most(limit: float) -> np.float32:
    """The largest float32 value <= `limit`.

    The width is checked again in float64 downstream (franka_control's gate, and the C4 validator), so a clipped
    float32 width must not round above the limit: np.float32(0.07) is 0.0700000003 > 0.07.
    """
    value = np.float32(limit)
    if float(value) > limit:
        value = np.nextafter(value, np.float32(0))
    return value


class Policy(ABC):
    KIND: ClassVar[str]
    USES_PROMPT: ClassVar[bool] = False

    def __init__(self, gripper_width_max_m: float) -> None:
        self._width_max = float32_max_at_most(gripper_width_max_m)

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: "Settings") -> "Policy":
        """Build it (and load what it needs). Raises PolicyLoadError."""

    @abstractmethod
    def inputs(self) -> tuple[str, ...]:
        """The Observation field names it actually reads."""

    @abstractmethod
    def reset(self) -> None:
        """Clear episode state; must take < reset_max_s."""

    @abstractmethod
    def infer(self, obs: Observation) -> ActionChunk:
        """One chunk. Raises BadRequest for a problem caused by the request; anything else is its own failure."""

    def config(self) -> PolicyConfig:
        return PolicyConfig(obs_steps=OBS_STEPS, action_steps=ACTION_STEPS, step_ms=STEP_MS, image_size=IMAGE_SIZE,
                            inputs=tuple(self.inputs()), uses_prompt=self.USES_PROMPT)

    def describe(self) -> dict[str, Any]:
        """Details for startup.json."""
        return {"kind": self.KIND}

    @property
    def width_max(self) -> np.float32:
        """The clip value: the largest float32 <= the max opening."""
        return self._width_max

    def make_chunk(self, eef_pos: np.ndarray, eef_rot6d: np.ndarray, gripper_width_m: np.ndarray) -> ActionChunk:
        """Clip the widths into [0, max opening], count the clipped steps, build (and so validate) the chunk."""
        clipped = np.clip(gripper_width_m, np.float32(0), self._width_max)
        n = int(np.count_nonzero(clipped != gripper_width_m))
        return ActionChunk(eef_pos=eef_pos, eef_rot6d=eef_rot6d, gripper_width_m=clipped, width_clipped=n)
