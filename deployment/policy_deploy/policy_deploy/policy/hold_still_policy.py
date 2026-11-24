# copied from rnoro/deployment:deployment/policy/policy/hold_still_policy.py @ 91b178b
# (there: new; behaves like rnoro/main-sim:controlSW/franka_deploy/franka_deploy/mocks/fake_policy_server.py @ 935c0b3)
# changes: imports only
"""HoldStillPolicy: "stay at the anchor". No torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from deploy_contracts import ContractLimits

from policy_deploy.contracts import ACTION_STEPS, ActionChunk, Observation
from policy_deploy.policy.base_policy import Policy

if TYPE_CHECKING:
    from policy_deploy.policy.settings import Settings

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
