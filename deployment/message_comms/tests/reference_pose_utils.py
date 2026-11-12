# from robot-utils==0.1.4:robot_utils/pose_utils.py (installed package; no git commit) and
# from behavior_prompting:behavior_prompting/common/pose_util.py @ edfe8bd
# changes: only the functions listed kept, bodies unchanged. Both files define normalize() with the same body
#          (behavior_prompting's second definition, the one in effect there; only the type hints differ), so it
#          appears once, as robot_utils wrote it, and serves rot_6d_to_mat and rot6d_to_mat alike.
"""Test-only reference: the training-side pose functions, copied unchanged (message_comms STRUCTURE.md section 11).

Never imported by the package; the light environment does not need robot-utils.
"""
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation


# ---- robot_utils/pose_utils.py (robot-utils==0.1.4) ----

def qmult(q1: npt.NDArray[Any], q2: npt.NDArray[Any]) -> npt.NDArray[Any]:
    q = np.array(
        [
            q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        ]
    )

    return q


def qconjugate(q: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def to_xyzw(wxyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if wxyz.ndim == 1:
        return np.concatenate([wxyz[1:], wxyz[0:1]])
    elif wxyz.ndim == 2:
        return np.concatenate([wxyz[:, 1:], wxyz[:, 0:1]], axis=1)
    else:
        raise ValueError("wxyz must be a 1D or 2D array")


def to_wxyz(xyzw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if xyzw.ndim == 1:
        return np.concatenate([xyzw[3:], xyzw[:3]])
    elif xyzw.ndim == 2:
        return np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
    else:
        raise ValueError("xyzw must be a 1D or 2D array")


def interpolate_xyz_wxyz(
    pose_left: npt.NDArray[np.float64],
    pose_right: npt.NDArray[np.float64],
    timestamp_left: float,
    timestamp_right: float,
    timestamp: float,
) -> npt.NDArray[np.float64]:
    assert (
        pose_left.shape == pose_right.shape == (7,)
    ), f"pose_left.shape: {pose_left.shape}, pose_right.shape: {pose_right.shape}"
    assert timestamp_left <= timestamp < timestamp_right
    ratio = (timestamp - timestamp_left) / (timestamp_right - timestamp_left)
    pos = pose_left[:3] + ratio * (pose_right[:3] - pose_left[:3])
    rot_left = st.Rotation.from_quat(to_xyzw(pose_left[3:]))
    rot_right = st.Rotation.from_quat(to_xyzw(pose_right[3:]))
    rots = st.Rotation.concatenate([rot_left, rot_right])
    rot = st.Slerp([0, 1], rots)([ratio])
    return np.concatenate([pos, to_wxyz(rot.as_quat().squeeze())])


def get_absolute_pose(
    init_pose_xyz_wxyz: npt.NDArray[Any],
    relative_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The new pose is in the same frame of reference as the initial pose"""
    new_pose_xyz_wxyz = np.zeros(7, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz = np.zeros(4, init_pose_xyz_wxyz.dtype)
    relative_pos_in_init_frame_as_quat_wxyz[1:] = relative_pose_xyz_wxyz[:3]
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pos_in_world_frame_as_quat_wxyz = qmult(
        qmult(init_pose_xyz_wxyz[3:], relative_pos_in_init_frame_as_quat_wxyz),
        init_rot_qinv,
    )
    new_pose_xyz_wxyz[:3] = (
        init_pose_xyz_wxyz[:3] + relative_pos_in_world_frame_as_quat_wxyz[1:]
    )
    quat = qmult(init_pose_xyz_wxyz[3:], relative_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    new_pose_xyz_wxyz[3:] = quat
    return new_pose_xyz_wxyz


def get_relative_pose(
    new_pose_xyz_wxyz: npt.NDArray[Any],
    init_pose_xyz_wxyz: npt.NDArray[Any],
):
    """The two poses are in the same frame of reference"""
    relative_pose_xyz_wxyz = np.zeros(7, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz = np.zeros(4, new_pose_xyz_wxyz.dtype)
    relative_pos_in_world_frame_as_quat_wxyz[1:] = (
        new_pose_xyz_wxyz[:3] - init_pose_xyz_wxyz[:3]
    )
    init_rot_qinv = qconjugate(init_pose_xyz_wxyz[3:])
    relative_pose_xyz_wxyz[:3] = qmult(
        qmult(init_rot_qinv, relative_pos_in_world_frame_as_quat_wxyz),
        init_pose_xyz_wxyz[3:],
    )[1:]
    quat = qmult(init_rot_qinv, new_pose_xyz_wxyz[3:])
    if quat[0] < 0:
        quat = -quat
    relative_pose_xyz_wxyz[3:] = quat
    return relative_pose_xyz_wxyz


def normalize(vec: npt.NDArray[Any], eps: float = 1e-12) -> npt.NDArray[Any]:
    norm: npt.NDArray[Any] = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out: npt.NDArray[Any] = (vec.T / norm).T
    return out


def mat_to_rot_6d(mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    batch_dim = mat.shape[:-2]
    out: npt.NDArray[Any] = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def rot_6d_to_mat(d6: npt.NDArray[Any]) -> npt.NDArray[Any]:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def pose_9d_to_xyz_wxyz(pose_9d: npt.NDArray[Any]) -> npt.NDArray[Any]:
    assert pose_9d.shape[-1] == 9, f"pose_9d.shape: {pose_9d.shape}"
    pos_xyz = pose_9d[..., :3]
    rot_mqt = rot_6d_to_mat(pose_9d[..., 3:])
    rot_wxyz = to_wxyz(Rotation.from_matrix(rot_mqt).as_quat())
    return np.concatenate([pos_xyz, rot_wxyz], axis=-1)


def xyz_wxyz_to_pose_9d(pose_xyz_wxyz: npt.NDArray[Any]) -> npt.NDArray[Any]:
    assert pose_xyz_wxyz.shape[-1] == 7, f"pose_xyz_wxyz.shape: {pose_xyz_wxyz.shape}"
    pos_xyz = pose_xyz_wxyz[..., :3]
    rot_mqt = Rotation.from_quat(to_xyzw(pose_xyz_wxyz[..., 3:])).as_matrix()
    rot_9d = mat_to_rot_6d(rot_mqt)
    return np.concatenate([pos_xyz, rot_9d], axis=-1)


# ---- behavior_prompting/common/pose_util.py @ edfe8bd ----

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out
