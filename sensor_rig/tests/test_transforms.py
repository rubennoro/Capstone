import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sensor_rig.transforms import (
    D_CV_ARKIT, R_from_euler_xyz_deg, T_YUP_ZUP, T_from_tum, arkit_to_cv, compose, cv_to_arkit, inv_T, is_rigid,
    make_T, orthonormalize, rot_x, rot_y, rot_z, rotation_angle_deg, tum_from_T,
)


def _random_T(rng):
    return make_T(Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))


def test_inv_compose_rigid():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A, B = _random_T(rng), _random_T(rng)
        assert np.allclose(A @ inv_T(A), np.eye(4), atol=1e-12)
        assert np.allclose(inv_T(compose(A, B)), inv_T(B) @ inv_T(A), atol=1e-12)
        assert is_rigid(A)
    stack = np.stack([_random_T(rng) for _ in range(5)])
    assert np.allclose(inv_T(stack) @ stack, np.eye(4), atol=1e-12)
    assert np.array_equal(compose(), np.eye(4))
    shear = np.eye(4)
    shear[0, 1] = 0.01
    assert not is_rigid(shear)
    assert not is_rigid(np.diag([1.0, 1.0, -1.0, 1.0]))          # reflection, det = -1
    bad_row = np.eye(4)
    bad_row[3, 0] = 0.1
    assert not is_rigid(bad_row)
    noisy = Rotation.random(random_state=rng).as_matrix() + 1e-3 * rng.normal(size=(3, 3))
    R = orthonormalize(noisy)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)


def test_elementary_rotations():
    assert np.allclose(rot_z(90) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    assert np.allclose(rot_x(90) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    assert np.allclose(rot_y(90) @ [0, 0, 1], [1, 0, 0], atol=1e-15)
    assert np.allclose(R_from_euler_xyz_deg(10, 20, 30), rot_z(30) @ rot_y(20) @ rot_x(10), atol=1e-15)
    assert np.array_equal(R_from_euler_xyz_deg(0, 0, 0), np.eye(3))
    assert rotation_angle_deg(rot_y(5)) == pytest.approx(5.0, abs=1e-12)
    assert rotation_angle_deg(make_T(rot_x(-170))) == pytest.approx(170.0, abs=1e-10)


def test_arkit_cv():
    rng = np.random.default_rng(1)
    T = _random_T(rng)
    assert np.array_equal(cv_to_arkit(arkit_to_cv(T)), T)
    T_ark = cv_to_arkit(T)
    # ARKit +z (backward) is minus the OpenCV optical axis; +y (up) is minus OpenCV's +y (down); x is shared.
    assert np.array_equal(T_ark[:3, 2], -T[:3, 2])
    assert np.array_equal(T_ark[:3, 1], -T[:3, 1])
    assert np.array_equal(T_ark[:3, 0], T[:3, 0])
    assert np.array_equal(T_ark[:3, 3], T[:3, 3])
    assert np.array_equal(D_CV_ARKIT @ D_CV_ARKIT, np.eye(4))


def test_yup_from_zup():
    assert np.array_equal(T_YUP_ZUP[:3, :3] @ [0, 0, -1], [0, -1, 0])     # gravity
    assert np.array_equal(T_YUP_ZUP[:3, :3] @ [1, 0, 0], [1, 0, 0])
    assert np.array_equal(T_YUP_ZUP[:3, :3] @ [0, 1, 0], [0, 0, -1])
    assert np.linalg.det(T_YUP_ZUP[:3, :3]) == 1.0
    assert is_rigid(T_YUP_ZUP)


def test_tum_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(20):
        T = _random_T(rng)
        row = tum_from_T(T)
        assert row.shape == (7,) and row[6] >= 0
        assert np.allclose(T_from_tum(0.0, *row), T, atol=1e-12)
    # scalar-last convention: 90 deg about z
    T = T_from_tum(1.0, 1, 2, 3, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(T[:3, :3], rot_z(90), atol=1e-12) and np.array_equal(T[:3, 3], [1, 2, 3])
    # vectorised
    Ts = np.stack([_random_T(rng) for _ in range(10)])
    rows = tum_from_T(Ts)
    assert rows.shape == (10, 7)
    assert np.allclose(T_from_tum(np.zeros(10), *rows.T), Ts, atol=1e-12)
    with pytest.raises(ValueError):
        T_from_tum(0, 0, 0, 0, 0, 0, 0, 0.5)
