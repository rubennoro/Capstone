# new; behaves like rnoro/main-sim:controlSW/franka_deploy/franka_deploy/mocks/fake_policy_server.py @ 935c0b3
# (--no-prompt-rules: identity chunk, no prompt needed)
# changes: width = the newest observed width instead of a fixed 0.07 m
"""HoldStillPolicy: "stay at the anchor" (STRUCTURE.md 4.3). No torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from contracts import ACTION_STEPS, ActionChunk, ContractLimits, Observation

from policy.base_policy import Policy

if TYPE_CHECKING:
    from policy.settings import Settings

_ROT6D_IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


class HoldStillPolicy(Policy):
    KIND = "hold_still"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "HoldStillPolicy":
        return cls(ContractLimits.load().gripper_width_max_m)

    def inputs(self) -> tuple[str, ...]:
        return ("gripper_width_m",)

    def reset(self) -> None:
        pass

    def infer(self, obs: Observation) -> ActionChunk:
        eef_pos = np.zeros((ACTION_STEPS, 3), dtype=np.float32)
        eef_rot6d = np.tile(_ROT6D_IDENTITY, (ACTION_STEPS, 1))
        width = np.full((ACTION_STEPS, 1), obs.gripper_width_m[-1, 0], dtype=np.float32)
        return self.make_chunk(eef_pos, eef_rot6d, width)
