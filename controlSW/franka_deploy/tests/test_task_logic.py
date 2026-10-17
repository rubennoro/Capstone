"""FrankaPromptTask policy-step logic with fakes (no processes): proprio history selection, relative->absolute,
timestamps, and prompt bookkeeping. Runs in the real-env env (needs real_env + robot_utils)."""
import time

import numpy as np
import pytest

pytest.importorskip("real_env")
pytest.importorskip("robot_utils")
from scipy.spatial.transform import Rotation as R  # noqa: E402

from real_env.tasks.base_task import TaskControlMode  # noqa: E402
from robot_utils.pose_utils import get_relative_pose  # noqa: E402

from franka_deploy.tasks.franka_task import FrankaPromptTask  # noqa: E402
from franka_deploy.tools.agent_capture import FakePolicyClient, make_agent  # noqa: E402

EXEC_H = 8


def T_of(xyz_wxyz):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(np.r_[xyz_wxyz[4:], xyz_wxyz[3]]).as_matrix()
    M[:3, 3] = xyz_wxyz[:3]
    return M


def xyz_wxyz_of(M):
    q = R.from_matrix(M[:3, :3]).as_quat()
    return np.r_[M[:3, 3], q[3], q[:3]]


class FakeArm:
    def __init__(self):
        self.eef, self.joint = [], []

    def schedule_eef_traj(self, eef_traj_xyz_wxyz, timestamps, use_relative_timestamps):
        self.eef.append((eef_traj_xyz_wxyz.copy(), timestamps.copy()))

    def schedule_joint_traj(self, joint_traj_pos, timestamps, use_relative_timestamps):
        self.joint.append((joint_traj_pos.copy(), timestamps.copy()))


def rel_chunk(k_shift=0.0):
    pos = np.array([[0.001 * k + k_shift, 0.0005 * k, 0.0] for k in range(16)])
    rot = np.array([R.from_euler("z", 0.01 * k).as_matrix()[:2].reshape(-1) for k in range(16)])  # rot6d = first 2 rows
    return {"action0_eef_pos": pos, "action0_eef_rot_axis_angle": rot, "action0_gripper_width": np.full((16, 1), 0.05)}


@pytest.fixture
def task():
    agent = make_agent(20.0)
    agent.action_execution_horizon = EXEC_H
    t = FrankaPromptTask.__new__(FrankaPromptTask)
    t.policy_agent = agent
    t.arx5_client = FakeArm()
    t.arx5_lookahead_time_s = 0.1
    t.camera_latency_s = 0.0
    t.control_mode = TaskControlMode.POLICY
    t.last_control_mode = TaskControlMode.SPACEMOUSE
    t.pause_when_policy_inference = False
    t._stop_requested = None
    t._prompted = True
    t.tap_dir = None
    t._last_request = None
    t._obs_info = {}
    t.counters = {"policy_calls": 0, "obs_retries": 0}
    t.width_proprio = "commanded"
    t.history_source = "commanded"
    t.v_max, t.w_max, t.scale_margin = 10.0, 100.0, 0.9   # no time-scaling in these tests (tested separately)
    t.settle = False
    t._ep = None
    t.episode_dir = None
    t.episode_active = False
    t._ctrl_msg = lambda msg: None
    t.home_xyz = None
    t.obs_retries = 3
    t.policy_history_actions = {}
    t.measured = np.array([0.45, 0.02, 0.30, *np.r_[R.from_euler("xyz", [3.1, 0.05, 0.2]).as_quat()[3], R.from_euler("xyz", [3.1, 0.05, 0.2]).as_quat()[:3]]])
    t.measured_width = np.array([0.062])
    t.spacemouse_history_actions = {"action0_eef_xyz_wxyz": t.measured[None].copy(),
                                    "action0_gripper_width": t.measured_width[None].copy(),
                                    "timestamps": np.array([time.monotonic()])}

    def get_obs():
        now = time.monotonic()
        return {"robot0_eef_xyz_wxyz": np.stack([t.measured, t.measured]), "robot0_gripper_width": np.stack([t.measured_width] * 2),
                "robot0_main_camera": np.zeros((2, 4, 4, 3), np.uint8), "robot0_ultrawide_camera": np.zeros((2, 4, 4, 3), np.uint8),
                "timestamps": np.array([now - 0.05, now]), "task_name": "block_raise"}

    t.get_observations = get_obs
    return t


def run_call(task, rel):
    FakePolicyClient.reply_actions = rel
    task.run_policy_control()
    return FakePolicyClient.captured


def test_first_call_after_spacemouse_identity_and_measured_width(task):
    sent = run_call(task, rel_chunk())
    np.testing.assert_allclose(sent["robot0_eef_pos"], 0, atol=1e-7)
    np.testing.assert_allclose(sent["robot0_eef_rot_axis_angle"], [[1, 0, 0, 0, 1, 0]] * 2, atol=1e-6)
    np.testing.assert_allclose(sent["robot0_gripper_width"], [[0.062], [0.062]], atol=1e-7)


def test_relative_to_absolute_hand_computed(task):
    rel = rel_chunk()
    run_call(task, rel)
    sched, ts = task.arx5_client.eef[-1]
    A = T_of(task.measured)
    for k in range(16):
        Rk = np.eye(4)
        r6 = rel["action0_eef_rot_axis_angle"][k]
        b1, b2 = r6[:3], r6[3:]
        Rk[:3, :3] = np.stack([b1, b2, np.cross(b1, b2)])
        Rk[:3, 3] = rel["action0_eef_pos"][k]
        expect = A @ Rk
        got = T_of(sched[k])
        np.testing.assert_allclose(got[:3, 3], expect[:3, 3], atol=1e-9)
        np.testing.assert_allclose(got[:3, :3], expect[:3, :3], atol=1e-9)
    # first chunk after spacemouse: re-based timestamps now + k/20, arm + lookahead, gripper without
    np.testing.assert_allclose(np.diff(ts), 0.05, atol=1e-9)
    np.testing.assert_allclose(ts - task.arx5_client.joint[-1][1], 0.1, atol=1e-9)


def _history_check(task, expect_idx):
    run_call(task, rel_chunk())
    abs1 = task.arx5_client.eef[-1][0]
    task.last_control_mode = task.control_mode
    sent = run_call(task, rel_chunk(0.002))
    h0, h1 = abs1[expect_idx[0]], abs1[expect_idx[1]]
    rel = get_relative_pose(h0, h1)
    np.testing.assert_allclose(sent["robot0_eef_pos"][0], rel[:3], atol=1e-6)
    np.testing.assert_allclose(sent["robot0_eef_pos"][1], 0, atol=1e-7)


def test_history_continuous_uses_exec_h_minus_1_and_exec_h(task):
    task.pause_when_policy_inference = False
    _history_check(task, (EXEC_H - 1, EXEC_H))


def test_history_pause_uses_exec_h_minus_2_and_exec_h_minus_1(task):
    task.pause_when_policy_inference = True
    _history_check(task, (EXEC_H - 2, EXEC_H - 1))


def test_pause_executes_exec_h_actions_only(task):
    task.pause_when_policy_inference = True
    run_call(task, rel_chunk())
    assert task.arx5_client.eef[-1][0].shape == (EXEC_H, 7)


def test_stop_during_inference_is_not_scheduled(task):
    def stopping_obs(orig=task.get_observations):
        task._stop_requested = "s"
        return orig()

    task.get_observations = stopping_obs
    run_call(task, rel_chunk())
    assert task.arx5_client.eef == []


# ------------------------------------------------------------------ anchor (decision a) + width_proprio (decision b)
class FakeArmState:
    def __init__(self, fk, target, width=0.06):
        self.fk, self.target, self.width = np.asarray(fk, float), np.asarray(target, float), width
        self.eef_timestamps = [time.monotonic()]

    def get_eef_xyz_wxyz(self, timestamp):
        self.eef_timestamps = [time.monotonic()]
        return self.fk

    def get_eef_target_xyz_wxyz(self, timestamp):
        return self.target

    def get_joint_pos(self, timestamp):
        return np.array([self.width])

    def state_age_s(self):
        return 0.001


class FakeCam:
    camera_names = ["main", "ultrawide"]
    config = {"camera_configs": {"main": {"fps": 60}, "ultrawide": {"fps": 60}}}

    def get_latest_images_dict_THWC(self, n, **kw):
        ts = time.monotonic() - np.arange(n + 4)[::-1] / 60
        img = np.zeros((len(ts), 4, 4, 3), np.uint8)
        return {"main": img, "ultrawide": img}, ts


def anchor_task(task, fk, target):
    from franka_deploy.tasks import franka_task as ft

    task.wrist_camera_client = FakeCam()
    task.arx5_client = FakeArmState(fk, target)
    task.match_camera_latency = False
    task.state_max_age_s = 0.05
    task.anchor = "target"
    task.anchor_max_dev_m = 0.02
    task.anchor_hold_m = 0.05
    task.policy_start_eef_xyz_wxyz = None
    task.task_name = "block_raise"
    task.counters.update({"stale_state": 0, "anchor_fallbacks": 0})
    reseeds = []
    task._reseed_controller = lambda: reseeds.append(1)
    task.controller_status = lambda: None
    task.get_observations = ft.FrankaPromptTask.get_observations.__get__(task)
    return reseeds


def test_target_anchor_used_when_close(task):
    fk = np.array([0.45, 0, 0.3, 0, 1, 0, 0.0])
    tgt = fk.copy(); tgt[0] += 0.01
    reseeds = anchor_task(task, fk, tgt)
    obs = task.get_observations()
    np.testing.assert_allclose(obs["robot0_eef_xyz_wxyz"][-1], tgt)
    assert task._obs_info["anchor_used"] == "target" and not reseeds
    assert abs(task._obs_info["anchor_dev_m"] - 0.01) < 1e-12


def test_target_anchor_falls_back_and_reseeds_beyond_2cm(task):
    fk = np.array([0.45, 0, 0.3, 0, 1, 0, 0.0])
    tgt = fk.copy(); tgt[0] += 0.03
    reseeds = anchor_task(task, fk, tgt)
    obs = task.get_observations()
    np.testing.assert_allclose(obs["robot0_eef_xyz_wxyz"][-1], fk)
    assert task._obs_info["anchor_used"] == "measured" and reseeds == [1] and task.counters["anchor_fallbacks"] == 1


def test_target_anchor_holds_beyond_5cm(task):
    from franka_deploy.tasks.franka_task import AnchorDeviationError

    fk = np.array([0.45, 0, 0.3, 0, 1, 0, 0.0])
    tgt = fk.copy(); tgt[0] += 0.06
    anchor_task(task, fk, tgt)
    with pytest.raises(AnchorDeviationError):
        task.get_observations()


def test_width_proprio_measured_replaces_history_width(task):
    task.width_proprio = "measured"
    task.history_source = "commanded"
    task.measured_width = np.array([0.071])
    sent = run_call(task, rel_chunk())
    np.testing.assert_allclose(sent["robot0_gripper_width"], [[0.071], [0.071]], atol=1e-7)


def test_clamp_backlog_is_not_a_fallback(task):
    """Franka: commanded target 3 cm ahead because the bridge clamps speed, but the arm tracks the SAFE target
    (track_dev 2 mm) -> keep the target anchor, no reseed."""
    fk = np.array([0.45, 0, 0.3, 0, 1, 0, 0.0])
    tgt = fk.copy(); tgt[2] -= 0.03
    reseeds = anchor_task(task, fk, tgt)
    task.controller_status = lambda: {"track_dev_m": 0.002, "clamp_backlog_m": 0.028}
    obs = task.get_observations()
    np.testing.assert_allclose(obs["robot0_eef_xyz_wxyz"][-1], tgt)
    assert task._obs_info["anchor_used"] == "target" and not reseeds
    assert abs(task._obs_info["target_fk_dev_m"] - 0.03) < 1e-12


@pytest.mark.parametrize("status,ok", [({"fault": None, "bridge_state": "running"}, True),
                                       ({"fault": "robot reflex", "bridge_state": "reflex"}, False),
                                       ({"fault": None, "bridge_state": "stopped"}, False),
                                       ({"fault": None}, True),                 # mock controller (no bridge)
                                       (None, False)])
def test_refuse_start_unless_bridge_running_without_fault(task, status, ok):
    task.controller_status = lambda: status
    task.events = []
    started = []
    task.start_episode = lambda: started.append(1)
    task.log_event = lambda kind, **kw: task.events.append({"kind": kind, **kw})
    task.handle_key("c")
    assert bool(started) == ok
    assert any(e["kind"] == "refused_start" for e in task.events) == (not ok)


def test_pause_action0_executed_as_anchor_without_delay(task):
    """Fix 1 (D6): action 0 = the anchor (training: action 0 is the identity), action k at t0 + k*step, width 0 =
    last commanded, history = the predicted chunk (unchanged)."""
    task.pause_when_policy_inference = True
    orig = task.get_observations

    def obs_with_anchor():
        o = orig()
        task._obs_info = {"anchor_pose": task.measured.copy()}
        return o

    task.get_observations = obs_with_anchor
    rel = rel_chunk(k_shift=0.002)                          # action 0 off by 2 mm from the anchor
    t_before = time.monotonic()
    run_call(task, rel)
    poses, ts = task.arx5_client.eef[-1]
    widths, wts = task.arx5_client.joint[-1]
    np.testing.assert_allclose(poses[0], task.measured, atol=1e-12)
    np.testing.assert_allclose(np.diff(ts), 0.05, atol=1e-9)
    assert t_before + 0.1 <= ts[0] <= time.monotonic() + 0.1   # arm schedule = t0 + lookahead, t0 = now
    assert abs(widths[0, 0] - 0.062) < 1e-12 and abs(widths[1, 0] - 0.05) < 1e-12
    hist = task.policy_history_actions["action0_eef_xyz_wxyz"]
    assert np.linalg.norm(hist[0, :3] - task.measured[:3]) > 0.0015  # history keeps the predicted action 0
