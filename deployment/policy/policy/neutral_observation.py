"""neutral_observation(): a valid made-up C4 Observation, for the warm-up and the probe (STRUCTURE.md 3.3)."""

from __future__ import annotations

import numpy as np
from contracts import IMAGE_SIZE, OBS_STEPS, Observation

_ROT6D_IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def neutral_observation(gripper_width_m: float) -> Observation:
    """Black images, both pose rows at the anchor, wrt_start identity, the given width."""
    image = np.zeros((OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    rot6d = np.tile(_ROT6D_IDENTITY, (OBS_STEPS, 1))
    return Observation(
        eef_pos=np.zeros((OBS_STEPS, 3), dtype=np.float32),
        eef_rot6d=rot6d,
        eef_rot6d_wrt_start=rot6d.copy(),
        gripper_width_m=np.full((OBS_STEPS, 1), gripper_width_m, dtype=np.float32),
        main=image,
        ultrawide=image.copy(),
    )
