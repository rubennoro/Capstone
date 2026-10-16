"""T11: frame conventions of the export pose chain."""

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_rig.pose.frames import (
    A_YUP_FROM_ZUP,
    CV_TO_ARKIT,
    T_YUP_ZUP,
    arkit_main_to_vio,
    hom,
    inv_T,
    orthonormalize,
    rig_vio_to_arkit_main,
    vio_to_arkit_main,
)


def _rand_T(rng, n=None):
    shape = () if n is None else (n,)
    R = Rotation.random(n, random_state=rng.integers(1 << 30)).as_matrix()
    T = np.zeros(shape + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = rng.normal(size=shape + (3,))
    T[..., 3, 3] = 1
    return T


def test_axes():
    assert np.allclose(A_YUP_FROM_ZUP @ [0, 0, 1], [0, 1, 0])
    assert np.allclose(A_YUP_FROM_ZUP @ [1, 0, 0], [1, 0, 0])
    assert np.isclose(np.linalg.det(A_YUP_FROM_ZUP), 1)
    assert np.array_equal(CV_TO_ARKIT @ CV_TO_ARKIT, np.eye(4))
    # ARKit camera looks along -z: the OpenCV optical axis (+z_cv) is -z_arkit
    T_cv = hom(np.eye(3))
    assert np.allclose((T_cv @ CV_TO_ARKIT)[:3, 2], [0, 0, -1])


def test_round_trip():
    rng = np.random.default_rng(0)
    T_arkit = _rand_T(rng, 20)
    X = _rand_T(rng)
    C = hom(Rotation.from_euler("z", 73, degrees=True).as_matrix(), [0.4, -1.2, 0.3])
    T_Wz_S = arkit_main_to_vio(T_arkit, X, C)
    offset = inv_T(T_YUP_ZUP @ C @ inv_T(T_YUP_ZUP))
    back = vio_to_arkit_main(T_Wz_S, X, offset)
    assert np.abs(back - T_arkit).max() < 1e-12


def test_world_change_is_left_multiplication():
    """A different VIO origin/heading changes every exported pose by the same rotation about +y (gravity kept)."""
    rng = np.random.default_rng(1)
    T_Wz_S = _rand_T(rng, 10)
    X = _rand_T(rng)
    C = hom(Rotation.from_euler("z", 40, degrees=True).as_matrix(), [1, 2, 3])
    A = vio_to_arkit_main(T_Wz_S, X, np.eye(4))
    B = vio_to_arkit_main(C @ T_Wz_S, X, np.eye(4))
    D = B @ inv_T(A)
    assert np.abs(D - D[0]).max() < 1e-12
    assert np.allclose(D[0][:3, 1], [0, 1, 0])  # rotation about +y only


def test_rig_chain(test_rig):
    rng = np.random.default_rng(2)
    T_Wz_S = _rand_T(rng, 5)
    got = rig_vio_to_arkit_main(test_rig, T_Wz_S, "imu0")
    want = test_rig.export.T_world_offset @ T_YUP_ZUP @ T_Wz_S @ test_rig.T("imu", "virtual_main") @ CV_TO_ARKIT
    assert np.abs(got - want).max() < 1e-12
    # same physical motion described with the ultrawide as trajectory frame gives the same export
    T_Wz_U = T_Wz_S @ test_rig.T("imu", "ultrawide")
    assert np.abs(rig_vio_to_arkit_main(test_rig, T_Wz_U, "ultrawide") - want).max() < 1e-12


def test_orthonormalize():
    rng = np.random.default_rng(3)
    R = Rotation.random(4, random_state=3).as_matrix() + rng.normal(scale=1e-3, size=(4, 3, 3))
    Q = orthonormalize(R)
    assert np.allclose(np.einsum("nji,njk->nik", Q, Q), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(Q), 1)
