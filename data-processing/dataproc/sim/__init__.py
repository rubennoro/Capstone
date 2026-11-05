"""Synthetic episode generation (docs/SIM.md): ``trajectory`` (seeded hand-like IMU trajectories with exact
derivatives), ``render`` (numba ray-cast textured room through the rig's camera models, ArUco finger tags),
``episode`` (writes a ``dataproc.raw.v1`` episode + ``truth.npz``), ``imu_synth`` (vendored).

The names below are imported lazily, so ``import dataproc.sim.imu_synth`` does not pull in numba.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "generate_episode": "episode", "run_sim": "episode", "Faults": "episode", "SimEpisode": "episode",
    "fast_rig": "episode", "read_truth": "episode", "finger_tag_poses_virt": "episode", "TRUTH_FILE": "episode",
    "TRUTH_VERSION": "episode",
    "make_trajectory": "trajectory", "Trajectory": "trajectory",
    "build_scene": "render", "CameraRenderer": "render", "Quad": "render", "camera_model": "render",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"dataproc.sim.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module 'dataproc.sim' has no attribute {name!r}")
