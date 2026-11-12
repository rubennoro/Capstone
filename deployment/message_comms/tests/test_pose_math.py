import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import reference_pose_utils as ref
from fakes import pose, random_pose, rz
from message_comms import pose_math as pm


def to_xyz_wxyz(T):
    return np.concatenate([T[:3, 3], ref.to_wxyz(Rotation.from_matrix(T[:3, :3]).as_quat())])


def from_xyz_wxyz(v):
    return pose(v[:3], Rotation.from_quat(ref.to_xyzw(v[3:])).as_matrix())


def test_invert_is_the_inverse():
    rng = np.random.default_rng(0)
    T = np.stack([random_pose(rng) for _ in range(20)])
    assert np.allclose(pm.invert(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(pm.invert(T), np.linalg.inv(T), atol=1e-12)


def test_rot6d_takes_rows():
    d6 = pm.rot6d_from_matrix(rz(10.0))
    assert np.allclose(d6, [0.98481, -0.17365, 0, 0.17365, 0.98481, 0], atol=1e-5)
    assert not np.allclose(d6, [0.98481, 0.17365, 0, -0.17365, 0.98481, 0], atol=1e-3)   # columns would give this


def test_rot6d_round_trip():
    R = Rotation.random(50, random_state=1).as_matrix()
    assert np.allclose(pm.matrix_from_rot6d(pm.rot6d_from_matrix(R)), R, atol=1e-12)


def test_gram_schmidt_makes_orthonormal():
    R = pm.matrix_from_rot6d(np.array([2.0, 0.1, 0.0, 0.3, 1.5, 0.2]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[0], np.array([2.0, 0.1, 0.0]) / np.linalg.norm([2.0, 0.1, 0.0]))


def test_interpolate_worked_example():
    T0 = pose((0.4950, 0, 0.30), rz(89.0))
    T1 = pose((0.5000, 0, 0.30), rz(90.0))
    assert np.allclose(pm.interpolate(T0, T1, 0.0), T0, atol=1e-12)
    assert np.allclose(pm.interpolate(T0, T1, 1.0), T1, atol=1e-12)
    T = pm.interpolate(T0, T1, 0.4)
    assert np.allclose(T[:3, 3], [0.4970, 0, 0.30], atol=1e-12)
    assert np.allclose(T[:3, :3], rz(89.4), atol=1e-12)


def test_slerp_halfway_of_90_is_45():
    T = pm.interpolate(pose(R=np.eye(3)), pose(R=rz(90.0)), 0.5)
    assert np.allclose(T[:3, :3], rz(45.0), atol=1e-12)


def test_position_distance_and_angle():
    assert pm.position_distance_m(pose((0, 0, 0)), pose((0.03, 0.04, 0))) == pytest.approx(0.05)
    assert pm.rotation_angle_deg(rz(6.0)) == pytest.approx(6.0)


def test_equal_to_reference_on_random_poses():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        A, B = random_pose(rng), random_pose(rng)
        a, b = to_xyz_wxyz(A), to_xyz_wxyz(B)
        # relative: get_relative_pose(new=B, init=A) == inv(A) @ B
        rel = pm.relative(A, B)
        assert np.allclose(rel, from_xyz_wxyz(ref.get_relative_pose(b, a)), atol=1e-9)
        # absolute: get_absolute_pose(init=A, rel) == A @ rel
        assert np.allclose(pm.absolute(A, rel), from_xyz_wxyz(ref.get_absolute_pose(a, to_xyz_wxyz(rel))), atol=1e-9)
        # rot6d both ways, both reference implementations
        R = B[:3, :3]
        d6 = pm.rot6d_from_matrix(R)
        assert np.allclose(d6, ref.mat_to_rot6d(R), atol=1e-12)
        assert np.allclose(d6, ref.mat_to_rot_6d(R), atol=1e-12)
        noisy = d6 + rng.normal(0, 0.05, 6)
        assert np.allclose(pm.matrix_from_rot6d(noisy), ref.rot6d_to_mat(noisy), atol=1e-9)
        assert np.allclose(pm.matrix_from_rot6d(noisy), ref.rot_6d_to_mat(noisy), atol=1e-9)
        # interpolation
        t = rng.uniform(0, 1)
        got = pm.interpolate(A, B, t)
        want = from_xyz_wxyz(ref.interpolate_xyz_wxyz(a, b, 0.0, 1.0, t))
        assert np.allclose(got, want, atol=1e-9)
    # pose 9d helpers agree with rot6d rows
    v = to_xyz_wxyz(B)
    assert np.allclose(ref.xyz_wxyz_to_pose_9d(v)[3:], pm.rot6d_from_matrix(B[:3, :3]), atol=1e-12)
    assert np.allclose(ref.pose_9d_to_xyz_wxyz(ref.xyz_wxyz_to_pose_9d(v))[3:] * np.sign(v[3]),
                       v[3:] * np.sign(v[3]), atol=1e-9)


def test_batched_stacks():
    rng = np.random.default_rng(3)
    A = random_pose(rng)
    Ts = np.stack([random_pose(rng) for _ in range(5)])
    assert np.allclose(pm.relative(A, Ts), np.stack([pm.relative(A, T) for T in Ts]))
