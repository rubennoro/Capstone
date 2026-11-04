"""franka_sim: MuJoCo Franka (Panda / FR3) simulation used to evaluate policies before hardware day.

This package must stay importable without mujoco (so ``python -m franka_sim --help`` works in the inference env):
modules that need mujoco import it inside functions.
"""

from __future__ import annotations
