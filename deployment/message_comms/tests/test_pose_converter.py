import numpy as np
import pytest

from contracts import ACTION_STEPS, Observation
from fakes import make_chunk, make_frames, pose, random_pose, rz
from message_comms import pose_math as pm
from message_comms.pose_converter import Anchor, PoseConverter
from message_comms.state_history import StateHistory
from fakes import make_state

T_NS = (9_950_000_000, 10_000_000_000)


def observe(conv, poses, start=None, widths=(0.05, 0.06), source="target"):
    start = poses[-1] if start is None else start
    return conv.to_observation(np.stack(poses), np.array(widths), start, make_frames(T_NS), T_NS[1], source)


def chunk_from_absolute(T_anchor, targets):
    rel = pm.relative(T_anchor, np.stack(targets))
    return make_chunk(pos=rel[:, :3, 3], rot=rel[:, :3, :3])


def test_round_trip_absolute_relative_absolute():
    rng = np.random.default_rng(7)
    conv = PoseConverter(exec_h=ACTION_STEPS, lookahead_ms=0)
    for _ in range(50):
        anchor_T = random_pose(rng)
        anchor_T[:3, 3] *= 0.5
        targets = [anchor_T @ pose(rng.uniform(-0.05, 0.05, 3), random_pose(rng)[:3, :3]) for _ in range(ACTION_STEPS)]
        _, anchor = observe(conv, [random_pose(rng), anchor_T])
        rel = pm.relative(anchor_T, np.stack(targets))
        # through float64 directly (the math) ...
        assert np.allclose(pm.absolute(anchor.T, rel), np.stack(targets), atol=1e-9)
        # ... and through the float32 chunk (the contract's dtype), back to the same poses within float32 precision
        wp = conv.to_waypoints(chunk_from_absolute(anchor_T, targets), anchor, chunk_id=1)
        assert np.allclose(wp.O_T_TCP, np.stack(targets), atol=1e-6)


def test_worked_example_input():
    conv = PoseConverter(8, 0)
    T_a = pose((0.50, 0.00, 0.30), rz(90.0))
    T_0 = pose((0.49, 0.00, 0.30), rz(85.0))
    T_s = pose((0.40, 0.10, 0.30), rz(80.0))
    obs, anchor = observe(conv, [T_0, T_a], start=T_s)
    assert np.allclose(obs.eef_pos, [[0.0, 0.01, 0.0], [0, 0, 0]], atol=1e-7)
    assert np.allclose(obs.eef_rot6d[0], [0.99619, 0.08716, 0, -0.08716, 0.99619, 0], atol=1e-5)
    assert np.array_equal(obs.eef_rot6d[1], np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    assert np.allclose(obs.eef_rot6d_wrt_start[0], [0.99619, -0.08716, 0, 0.08716, 0.99619, 0], atol=1e-5)
    assert np.allclose(obs.eef_rot6d_wrt_start[1], [0.98481, -0.17365, 0, 0.17365, 0.98481, 0], atol=1e-5)
    assert np.allclose(obs.gripper_width_m[:, 0], [0.05, 0.06])
    assert anchor.t_ns == T_NS[1] and np.array_equal(anchor.T, T_a)


def test_worked_example_output():
    conv = PoseConverter(8, 0)
    T_a = pose((0.50, 0.00, 0.30), rz(90.0))
    anchor = Anchor(t_ns=10_000_000_000, T=T_a, source="target")
    pos = np.zeros((ACTION_STEPS, 3))
    rot = np.tile(np.eye(3), (ACTION_STEPS, 1, 1))
    pos[4] = (0.02, 0, 0)
    rot[4] = rz(10.0)
    wp = conv.to_waypoints(make_chunk(pos=pos, rot=rot), anchor, 3)
    assert np.allclose(wp.O_T_TCP[4][:3, 3], [0.50, 0.02, 0.30], atol=1e-7)
    assert np.allclose(wp.O_T_TCP[4][:3, :3], rz(100.0), atol=1e-6)
    assert np.allclose(wp.O_T_TCP[4][0, :3], [-0.17365, -0.98481, 0], atol=1e-5)
    assert wp.t_ns[4] == 10_200_000_000
    assert list(wp.t_ns) == [10_000_000_000 + k * 50_000_000 for k in range(8)]


def test_rotation_not_just_translation():
    conv = PoseConverter(ACTION_STEPS, 0)
    T_a = pose((0.3, -0.2, 0.4), rz(30.0))
    # pure rotation about the anchor: position unchanged, orientation turned in the anchor's frame
    R_rel = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)      # 90 deg about x
    wp = conv.to_waypoints(make_chunk(rot=np.tile(R_rel, (ACTION_STEPS, 1, 1))), Anchor(0, T_a, "target"), 1)
    assert np.allclose(wp.O_T_TCP[0][:3, 3], T_a[:3, 3], atol=1e-9)
    assert np.allclose(wp.O_T_TCP[0][:3, :3], rz(30.0) @ R_rel, atol=1e-6)
    # mixed: an older pose rotated and moved: relative position is in the anchor's axes
    T_0 = pose((0.3, -0.2, 0.4) + rz(30.0) @ np.array([0.0, 0.0, 0.01]), rz(30.0) @ R_rel)
    obs, _ = observe(conv, [T_0, T_a])
    assert np.allclose(obs.eef_pos[0], [0, 0, 0.01], atol=1e-7)
    assert np.allclose(pm.matrix_from_rot6d(obs.eef_rot6d[0]), R_rel, atol=1e-6)


def test_same_anchor_both_ways_even_if_the_history_changes():
    conv = PoseConverter(8, 0)
    history = StateHistory(1.0)
    T_old = pose((0.5, 0.0, 0.3), rz(90.0))
    history.add(make_state(1, T_NS[0] - 10_000_000, T=T_old))
    history.add(make_state(2, T_NS[1] + 10_000_000, T=T_old))
    p0 = history.pose_at(T_NS[0], "target").T
    p1 = history.pose_at(T_NS[1], "target").T
    obs, anchor = observe(conv, [p0, p1])
    wp_before = conv.to_waypoints(make_chunk(), anchor, 1)
    history.add(make_state(3, T_NS[1] + 30_000_000, T=pose((0.6, 0.1, 0.3), rz(45.0))))  # the arm moved since
    wp_after = conv.to_waypoints(make_chunk(), anchor, 1)
    assert np.array_equal(wp_before.O_T_TCP, wp_after.O_T_TCP)
    assert np.allclose(wp_after.O_T_TCP[0], T_old, atol=1e-7)
    with pytest.raises(TypeError):
        conv.to_waypoints(make_chunk(), T_old, 1)


def test_wrt_start_identity_when_start_equals_pose():
    conv = PoseConverter(8, 0)
    T = pose((0.4, 0.1, 0.2), rz(33.0))
    obs, _ = observe(conv, [T, T], start=T)
    assert np.allclose(obs.eef_rot6d_wrt_start, [[1, 0, 0, 0, 1, 0]] * 2, atol=1e-7)


def test_last_rows_exact_and_valid_observation():
    rng = np.random.default_rng(5)
    conv = PoseConverter(8, 0)
    obs, _ = observe(conv, [random_pose(rng), random_pose(rng)])
    assert isinstance(obs, Observation)
    assert np.array_equal(obs.eef_pos[-1], np.zeros(3, dtype=np.float32))
    assert np.array_equal(obs.eef_rot6d[-1], np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    for name in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m"):
        assert getattr(obs, name).dtype == np.float32


def test_waypoints_dtypes_times_lookahead_exec_h_and_width():
    conv = PoseConverter(exec_h=5, lookahead_ms=30)
    widths = np.linspace(0.0, 0.075, ACTION_STEPS)
    chunk = make_chunk()
    chunk = type(chunk)(eef_pos=chunk.eef_pos, eef_rot6d=chunk.eef_rot6d,
                        gripper_width_m=widths.reshape(-1, 1).astype(np.float32), width_clipped=0)
    wp = conv.to_waypoints(chunk, Anchor(1_000_000_000, pose(), "measured"), 9)
    assert wp.t_ns.dtype == np.int64 and wp.O_T_TCP.dtype == np.float64 and wp.gripper_width_m.dtype == np.float64
    assert len(wp.t_ns) == 5 and wp.O_T_TCP.shape == (5, 4, 4)
    assert list(wp.t_ns) == [1_000_000_000 + k * 50_000_000 + 30_000_000 for k in range(5)]
    assert np.array_equal(wp.gripper_width_m, widths[:5].astype(np.float32).astype(np.float64))
    assert wp.chunk_id == 9
