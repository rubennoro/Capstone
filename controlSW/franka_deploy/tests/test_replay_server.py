"""Single-arm action replay: training convention (action[k] = demo[t + 3k] relative to demo[t]) and pointer advance."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from franka_deploy.tools.replay_server import ReplayServer


def _rot6_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def _demo(n=300):
    t = np.arange(n) / 60.0
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = np.c_[0.4 + 0.1 * np.sin(t), 0.05 * t, 0.2 + 0.02 * np.cos(2 * t)]
    T[:, :3, :3] = R.from_rotvec(np.c_[0.3 * np.sin(t), 0.2 * t, 0.1 * np.ones(n)]).as_matrix()
    return T, 0.02 + 0.06 * (t / t[-1])


def test_chunk_matches_training_convention():
    T, w = _demo()
    rs = ReplayServer(T, w, start=30, speed=1.0, exec_h=8)
    c = rs.chunk()
    assert c["action0_eef_pos"].shape == (16, 3) and c["action0_eef_rot_axis_angle"].shape == (16, 6)
    np.testing.assert_allclose(c["action0_eef_pos"][0], 0, atol=1e-7)
    np.testing.assert_allclose(_rot6_to_mat(c["action0_eef_rot_axis_angle"][0]), np.eye(3), atol=1e-6)
    for k in (1, 5, 15):
        rel = np.linalg.inv(T[30]) @ T[30 + 3 * k]
        np.testing.assert_allclose(c["action0_eef_pos"][k], rel[:3, 3], atol=1e-6)
        np.testing.assert_allclose(_rot6_to_mat(c["action0_eef_rot_axis_angle"][k]), rel[:3, :3], atol=1e-5)
        assert abs(c["action0_gripper_width"][k, 0] - w[30 + 3 * k]) < 1e-6
    assert rs.t == 30 + 7 * 3                             # pause mode: the next chunk starts at action exec_h - 1


def test_half_speed_and_continuous_advance():
    T, w = _demo()
    rs = ReplayServer(T, w, start=0, speed=0.5, exec_h=8)
    c = rs.chunk()
    rel = np.linalg.inv(T[0]) @ T[3]                      # action 2 at half speed = frame 3
    np.testing.assert_allclose(c["action0_eef_pos"][2], rel[:3, 3], atol=1e-6)
    assert rs.t == 7 * 1.5
    rc = ReplayServer(T, w, start=0, speed=1.0, exec_h=8, continuous=True)
    rc.chunk()
    assert rc.t == 24


def test_chained_chunks_integrate_to_the_demo():
    """Composing each chunk's action exec_h-1 (what pause mode reaches) reproduces the demo's relative motion."""
    T, w = _demo()
    rs = ReplayServer(T, w, start=0, speed=1.0, exec_h=8)
    P = np.eye(4)
    for _ in range(10):
        c = rs.chunk()
        rel = np.eye(4)
        rel[:3, 3], rel[:3, :3] = c["action0_eef_pos"][7], _rot6_to_mat(c["action0_eef_rot_axis_angle"][7])
        P = P @ rel
    ref = np.linalg.inv(T[0]) @ T[210]
    assert np.linalg.norm(P[:3, 3] - ref[:3, 3]) < 1e-4    # float32 replies
