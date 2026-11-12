# adapted from robot-utils==0.1.4:robot_utils/pose_utils.py (get_relative_pose, get_absolute_pose, interpolate_xyz_wxyz)
# and behavior_prompting:behavior_prompting/common/pose_util.py @ edfe8bd (mat_to_rot6d, rot6d_to_mat)
# changes: rewritten with 4x4 matrices instead of xyz+wxyz quaternions. Same math, checked against the copied
#          originals in tests/reference_pose_utils.py.
"""PoseConverter: absolute -> relative for the policy, relative -> absolute timed waypoints (STRUCTURE.md 6.2, 6.5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import STEP_MS, ActionChunk, FramesReply, Observation, SetWaypoints
from message_comms import pose_math

NS_PER_MS = 1_000_000
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Anchor:
    t_ns: int                 # main_t_ns[-1]: the newest main image time
    T: np.ndarray             # (4, 4) float64: the pose the observation was made relative to
    source: str               # "target" or "measured" (after a fallback, section 6.7)


class PoseConverter:
    """Keeps no state between calls. The Anchor returned by to_observation is the only one to_waypoints accepts."""

    def __init__(self, exec_h: int, lookahead_ms: int) -> None:
        self._exec_h = exec_h
        self._lookahead_ns = int(lookahead_ms) * NS_PER_MS

    def to_observation(self, poses: np.ndarray, widths: np.ndarray, start_pose: np.ndarray,
                       frames: FramesReply, anchor_t_ns: int, source: str) -> tuple[Observation, Anchor]:
        """poses (2, 4, 4) at main_t_ns[0], main_t_ns[1]; widths (2,); start_pose (4, 4)."""
        poses = np.asarray(poses, dtype=np.float64)
        T_a = poses[-1].copy()
        rel = pose_math.relative(T_a, poses)
        eef_pos = rel[:, :3, 3].copy()
        eef_rot6d = pose_math.rot6d_from_matrix(rel[:, :3, :3])
        eef_pos[-1] = 0.0                     # the anchor relative to itself is the identity, exactly
        eef_rot6d[-1] = IDENTITY_ROT6D
        R_s = np.asarray(start_pose, dtype=np.float64)[:3, :3]
        wrt_start = pose_math.rot6d_from_matrix(R_s.T @ poses[:, :3, :3])
        obs = Observation(
            eef_pos=eef_pos.astype(np.float32),
            eef_rot6d=eef_rot6d.astype(np.float32),
            eef_rot6d_wrt_start=wrt_start.astype(np.float32),
            gripper_width_m=np.asarray(widths, dtype=np.float64).reshape(-1, 1).astype(np.float32),
            main=frames.main,
            ultrawide=frames.ultrawide,
        )
        return obs, Anchor(t_ns=int(anchor_t_ns), T=T_a, source=source)

    def to_waypoints(self, chunk: ActionChunk, anchor: Anchor, chunk_id: int) -> SetWaypoints:
        """The first exec_h steps, multiplied onto the same anchor, at t_a + k * STEP_MS + lookahead."""
        if not isinstance(anchor, Anchor):
            raise TypeError("to_waypoints needs the Anchor returned by to_observation")
        h = self._exec_h
        T_rel = np.zeros((h, 4, 4))
        T_rel[:, :3, :3] = pose_math.matrix_from_rot6d(np.asarray(chunk.eef_rot6d[:h], dtype=np.float64))
        T_rel[:, :3, 3] = np.asarray(chunk.eef_pos[:h], dtype=np.float64)
        T_rel[:, 3, 3] = 1.0
        T_abs = pose_math.absolute(anchor.T, T_rel)
        k = np.arange(h, dtype=np.int64)
        t_ns = np.int64(anchor.t_ns) + k * np.int64(STEP_MS * NS_PER_MS) + np.int64(self._lookahead_ns)
        width = np.asarray(chunk.gripper_width_m[:h, 0], dtype=np.float64)
        return SetWaypoints(chunk_id=int(chunk_id), t_ns=t_ns.astype(np.int64), O_T_TCP=np.ascontiguousarray(T_abs),
                            gripper_width_m=np.ascontiguousarray(width))
