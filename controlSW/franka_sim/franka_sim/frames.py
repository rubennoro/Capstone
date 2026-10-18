"""Pure-numpy pose helpers: ARKit (y-up) -> robot-base (z-up) placement and the repo's rot6d / pose7 conventions.

Why: Stage 1 replays recorded iPhone/rig TCP trajectories on a simulated robot. Those trajectories live in the ARKit
world (gravity aligned, +y up), the robot base frame is z-up, and the demo may be shifted/yawed anywhere on the table.
This module holds the few conversions needed, with no mujoco/scipy import so it runs in the policy (BP) environment.

Conventions (INTERFACES.md section 3): pose7 = [x, y, z, qw, qx, qy, qz]; rot6d = the first two ROWS of R
(behavior_prompting ``pose_util``; identical to ``franka_deploy.eval.report.rot6d_to_R``); ``T_a_b`` style 4x4 matrices.
``T_YUP_ZUP`` mirrors ``sensor_rig.transforms.T_YUP_ZUP`` (a test asserts equality with the literal values).

Usage: ``T_zup = arkit_to_zup(T_arkit)``; ``base_T_world = anchor_placement(T_grasp_zup, grasp_xyz_base, yaw_rad)``;
       ``base_T_tcp = base_T_world @ T_zup``.
"""

from __future__ import annotations

import numpy as np

#: Maps points from a z-up world to a y-up world (x -> x, z -> y, y -> -z), as in sensor_rig.
T_YUP_ZUP = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
T_YUP_ZUP.flags.writeable = False
#: ARKit y-up world -> z-up world: the inverse of ``T_YUP_ZUP`` (exact transpose of the rotation, no translation).
T_ZUP_YUP = np.linalg.inv(T_YUP_ZUP)
T_ZUP_YUP.flags.writeable = False


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, float)
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def quat_to_R(q) -> np.ndarray:
    """Rotation matrix from a unit-normalised quaternion [qw, qx, qy, qz]."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R) -> np.ndarray:
    """Quaternion [qw, qx, qy, qz] (qw >= 0) from a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = [(R[k, j] - R[j, k]) / s, *v]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def pose7_to_T(p) -> np.ndarray:
    """4x4 from [x, y, z, qw, qx, qy, qz]."""
    p = np.asarray(p, float).reshape(7)
    return make_T(quat_to_R(p[3:]), p[:3])


def T_to_pose7(T) -> np.ndarray:
    T = np.asarray(T, float)
    return np.r_[T[:3, 3], R_to_quat(T[:3, :3])]


def rotvec_to_R(rv) -> np.ndarray:
    """Rodrigues: rotation matrix from an axis-angle vector (angle = norm)."""
    rv = np.asarray(rv, float).reshape(3)
    th = np.linalg.norm(rv)
    K = np.array([[0, -rv[2], rv[1]], [rv[2], 0, -rv[0]], [-rv[1], rv[0], 0]])
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def rotvec_pos_to_T(pos, rotvec) -> np.ndarray:
    """4x4 from the dataset's (eef_pos, eef_rot_axis_angle)."""
    return make_T(rotvec_to_R(rotvec), np.asarray(pos, float).reshape(3))


def rot6d_to_R(r6) -> np.ndarray:
    """Gram-Schmidt on the two stored rows; same arithmetic as ``franka_deploy.eval.report.rot6d_to_R``."""
    d6 = np.asarray(r6, float)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def R_to_rot6d(R) -> np.ndarray:
    """First two ROWS of R, flattened (repo convention)."""
    R = np.asarray(R, float)
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def arkit_to_zup(T_arkit) -> np.ndarray:
    """Pose(s) given in the ARKit y-up world -> same physical pose in a z-up world (+y up becomes +z up).

    ``T_arkit`` is ``world_T_x`` with the ARKit world on the left; the result is ``zup_T_x = T_ZUP_YUP @ T_arkit``.
    Note: the spec text writes ``T_YUP_ZUP @ T``; ``T_YUP_ZUP`` maps z-up points INTO y-up, so the y-up -> z-up
    direction needs its inverse (a test checks that +y maps to +z).
    """
    return T_ZUP_YUP @ np.asarray(T_arkit, float)


def yaw_about_z(T, yaw_rad: float) -> np.ndarray:
    """Rotate the pose(s) T about the world z axis through the origin: ``Rz(yaw) @ T``."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))
    return Rz @ np.asarray(T, float)


def anchor_placement(T_grasp_zup, grasp_xyz_base, yaw_rad: float) -> np.ndarray:
    """``base_T_world``: rigid map from the demo's z-up world into the robot base frame that sends the demo grasp TCP
    position to ``grasp_xyz_base`` using a yaw rotation about z only (so gravity stays along -z)."""
    p = np.asarray(T_grasp_zup, float)[:3, 3]
    T = yaw_about_z(np.eye(4), yaw_rad)
    T[:3, 3] = np.asarray(grasp_xyz_base, float) - T[:3, :3] @ p
    return T


def relative(T_from, T_to) -> np.ndarray:
    """``inv(T_from) @ T_to``."""
    return inv_T(T_from) @ np.asarray(T_to, float)
