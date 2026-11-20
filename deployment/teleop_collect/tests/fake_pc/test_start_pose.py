"""FakePc and the start pose (2026-11-20, final-deployment's start-pose work, 927a5ba): the PC's p (keyboard_start_pose) with the
real PC's refusal texts and order, the C3 start refused away from p_pose, p_pose null as before, and the operator
stand-in (auto_h_p_after_stop_s: h, then p, after an episode's stop). On a fake laptop clock, no threads or sockets."""
from __future__ import annotations

import numpy as np
import pytest
from deploy_contracts import Start, Stop
from launcher_helpers import FakeLaptopClock, fake_pc_settings, make_fake_pc
from scipy.spatial.transform import Rotation

from teleop_collect.contracts import StationError, load_config
from teleop_collect.fake_pc.fake_pc import NOT_AT_HOME, NOT_AT_START_POSE, START_POSE_UNSET

UNSET = ("refused: homing.q_start_rad is not set: measure it and set it in control_deploy/station.yaml "
         "(franka_control: homing:)")


class Recorder:
    def __init__(self) -> None:
        self.states = []

    def publish(self, msg) -> None:
        self.states.append(msg)

    def close(self) -> None:
        pass


@pytest.fixture
def clock():
    return FakeLaptopClock()


def manual(tmp_path, clock, **overrides):
    """config/fake_pc.yaml's start pose (p_pose = start_pose), the operator stand-in off unless given."""
    fake = make_fake_pc(tmp_path, laptop_clock=clock, **{"auto_h_p_after_stop_s": None, **overrides})
    fake._pub = Recorder()
    fake.tick()
    return fake


def run(fake, clock, seconds, step=0.01):
    for _ in range(int(round(seconds / step))):
        clock.advance(step)
        fake.tick()


def home(fake, clock):
    assert fake.keyboard_home().ok
    run(fake, clock, fake.s.home_duration_s + 0.05)
    assert fake.mode == "idle" and np.allclose(fake.target, fake.s.home_pose)


def test_the_texts_are_the_real_pc_s():
    assert START_POSE_UNSET == UNSET
    assert NOT_AT_HOME == "refused: not at home: press h first"
    assert NOT_AT_START_POSE == "refused: not at the start pose: press h, then p on the PC"


# ---------------------------------------------------------------------------------------------- p refused

def test_p_refused_while_p_pose_is_unset_first(tmp_path, clock):
    fake = manual(tmp_path, clock, p_pose=None)
    assert fake.keyboard_start_pose().reason == UNSET
    home(fake, clock)
    assert not fake.keyboard_start_pose().ok and fake.keyboard_start_pose().reason == UNSET
    assert fake.handle(Start()).ok                             # unset: start never checks the pose
    assert fake.keyboard_start_pose().reason == UNSET          # unset comes before the mode check
    assert fake.mode == "holding"


def test_p_refused_away_from_home(tmp_path, clock):
    fake = manual(tmp_path, clock)
    assert np.allclose(fake.target, fake.s.p_pose)             # starts at the start pose, not at home
    ack = fake.keyboard_start_pose()
    assert not ack.ok and ack.reason == "refused: not at home: press h first"
    assert fake.mode == "idle"


@pytest.mark.parametrize("mode, reason", [
    ("holding", "refused: mode is holding: send stop first"),
    ("running", "refused: mode is running: send stop first"),
    ("homing", "refused: mode is homing: homing in progress"),
    ("fault", "refused: mode is fault: reflex"),
])
def test_p_refused_outside_idle_and_stopped_before_the_home_check(tmp_path, clock, mode, reason):
    fake = manual(tmp_path, clock)                             # away from home: the mode refusal comes first
    if mode == "homing":
        assert fake.keyboard_home().ok
    elif mode == "fault":
        fake.inject_fault("reflex")
    else:
        fake.set_mode(mode)
    ack = fake.keyboard_start_pose()
    assert not ack.ok and ack.reason == reason
    assert fake.mode == mode


def test_p_allowed_when_stopped_at_home(tmp_path, clock):
    fake = manual(tmp_path, clock)
    home(fake, clock)
    assert fake.keyboard_stop().ok and fake.mode == "stopped"
    assert fake.keyboard_start_pose().ok                       # stopped at home: allowed, as h


# ---------------------------------------------------------------------------------------------- p's move

def test_p_from_home_reaches_p_pose_homing_then_idle_with_the_hand_open(tmp_path, clock):
    fake = manual(tmp_path, clock)
    home(fake, clock)
    assert fake.width_m == pytest.approx(0.04)
    assert fake.keyboard_start_pose().ok
    assert fake.mode == "homing"
    run(fake, clock, 0.5)
    z = float(fake.target[2, 3])
    assert fake.mode == "homing" and 0.30 < z < 0.35
    assert fake._pub.states[-1].mode == "homing"
    assert fake.handle(Start()).reason == "refused: mode is homing: homing in progress"
    run(fake, clock, 0.6)
    assert fake.mode == "idle"
    assert np.allclose(fake.target, fake.s.p_pose)
    assert fake.width_m == pytest.approx(fake.s.limits.gripper_width_max_m)
    assert fake._pub.states[-1].gripper_width_m == pytest.approx(fake.s.limits.gripper_width_max_m)
    assert fake.handle(Start()).ok


@pytest.mark.parametrize("who", ["C3", "PC keyboard"])
def test_a_stop_during_p_aborts_it_as_a_home(tmp_path, clock, who):
    fake = manual(tmp_path, clock)
    home(fake, clock)
    assert fake.keyboard_start_pose().ok
    run(fake, clock, 0.4)
    assert fake.handle(Stop()).ok if who == "C3" else fake.keyboard_stop().ok
    assert fake.mode == "stopped"
    held, width = fake.target, fake.width_m
    assert 0.30 < held[2, 3] < 0.35 and width < fake.s.limits.gripper_width_max_m
    run(fake, clock, 2.0)
    assert np.allclose(fake.target, held) and fake.width_m == pytest.approx(width)
    assert fake.handle(Start()).reason == NOT_AT_START_POSE
    assert fake._stopped_detail == f"stop during homing ({who})"     # the real PC's detail (shared with h)


# ---------------------------------------------------------------------------------------------- C3 start

def test_c3_start_refused_away_from_p_pose_and_accepted_at_it(tmp_path, clock):
    fake = manual(tmp_path, clock)
    home(fake, clock)
    ack = fake.handle(Start())
    assert not ack.ok and ack.reason == "refused: not at the start pose: press h, then p on the PC"
    assert fake.mode == "idle"
    assert fake.keyboard_start_pose().ok
    run(fake, clock, 1.1)
    assert fake.handle(Start()).ok and fake.mode == "holding"


def test_c3_start_mode_refusal_comes_before_the_pose(tmp_path, clock):
    fake = manual(tmp_path, clock)
    home(fake, clock)
    fake.set_mode("holding")
    assert fake.handle(Start()).reason == "refused: mode is holding: already started"


@pytest.mark.parametrize("dx, deg, ok", [(0.001, 0.0, True), (0.003, 0.0, False), (0.0, 0.5, True),
                                         (0.0, 2.0, False)])
def test_at_the_start_pose_within_at_pose_tol_m_and_1_deg(tmp_path, clock, dx, deg, ok):
    fake = manual(tmp_path, clock)
    T = fake.s.p_pose.copy()
    T[0, 3] += dx
    T[:3, :3] = Rotation.from_euler("x", deg, degrees=True).as_matrix() @ T[:3, :3]
    fake._follower.set_target(T)
    ack = fake.handle(Start())
    assert ack.ok == ok and (ok or ack.reason == NOT_AT_START_POSE)


def test_p_pose_null_keeps_the_old_behaviour(tmp_path, clock):
    fake = manual(tmp_path, clock, p_pose=None, auto_h_p_after_stop_s=1.0)
    home(fake, clock)
    assert fake.handle(Start()).ok                             # anywhere
    assert fake.handle(Stop()).ok
    run(fake, clock, 1.2)                                       # the stand-in's h
    assert fake.mode == "homing"
    run(fake, clock, 2.0)                                       # its p is refused (unset): idle at home
    assert fake.mode == "idle" and np.allclose(fake.target, fake.s.home_pose)


# ---------------------------------------------------------------------------------------------- the operator stand-in

def episode(fake, clock, dx=0.02):
    """Start at the start pose, move the target away from it, stay holding."""
    assert fake.handle(Start()).ok
    T = fake.target
    T[0, 3] += dx
    fake._follower.set_target(T)
    run(fake, clock, 0.1)
    assert fake.mode == "holding"


def test_after_an_episode_s_c3_stop_the_stand_in_presses_h_then_p(tmp_path, clock):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=1.0)
    episode(fake, clock)
    assert fake.handle(Stop()).ok and fake.mode == "stopped"
    assert fake.handle(Start()).reason == NOT_AT_START_POSE
    run(fake, clock, 0.9)
    assert fake.mode == "stopped"
    run(fake, clock, 0.2)
    assert fake.mode == "homing"                               # h
    run(fake, clock, 0.5)
    assert fake.mode == "homing" and fake.target[2, 3] > 0.32  # h, on its way to home (z 0.35)
    run(fake, clock, 1.0)                                       # h ended, p runs
    assert fake.mode == "homing" and 0.30 < fake.target[2, 3] < 0.35
    run(fake, clock, 1.0)
    assert fake.mode == "idle" and np.allclose(fake.target, fake.s.p_pose)
    assert fake.width_m == pytest.approx(fake.s.limits.gripper_width_max_m)
    assert fake.handle(Start()).ok                             # the next episode


@pytest.mark.parametrize("how", ["PC keyboard", "watchdog"])
def test_every_end_of_an_episode_counts(tmp_path, clock, how):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=1.0)
    episode(fake, clock)
    if how == "PC keyboard":
        assert fake.keyboard_stop().ok
    else:
        run(fake, clock, fake.s.watchdog_limit_s + 0.05)
    assert fake.mode == "stopped"
    assert fake.handle(Stop()).ok                              # teleop's conclude: a stop in stopped changes nothing
    run(fake, clock, 4.0)
    assert fake.mode == "idle" and np.allclose(fake.target, fake.s.p_pose)


def test_no_stand_in_after_a_stop_outside_an_episode(tmp_path, clock):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=1.0)
    assert fake.handle(Stop()).ok and fake.mode == "stopped"   # idle -> stopped: no episode
    run(fake, clock, 4.0)
    assert fake.mode == "stopped" and np.allclose(fake.target, fake.s.p_pose)


def test_a_stop_during_the_stand_in_s_h_cancels_its_p(tmp_path, clock):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=0.5)
    episode(fake, clock)
    assert fake.handle(Stop()).ok
    run(fake, clock, 0.8)
    assert fake.mode == "homing"
    assert fake.handle(Stop()).ok and fake.mode == "stopped"
    run(fake, clock, 4.0)
    assert fake.mode == "stopped"


def test_a_start_before_the_stand_in_s_h_cancels_it(tmp_path, clock):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=1.0)
    assert fake.handle(Start()).ok                             # an episode that never moved the arm
    assert fake.handle(Stop()).ok
    run(fake, clock, 0.5)
    assert fake.handle(Start()).ok                             # still at the start pose: accepted
    run(fake, clock, 0.1)
    fake.hold_forever()
    run(fake, clock, 2.0)
    assert fake.mode == "holding"


def test_the_stand_in_s_presses_are_logged(tmp_path, clock, caplog):
    fake = manual(tmp_path, clock, auto_h_p_after_stop_s=0.2)
    episode(fake, clock)
    assert fake.handle(Stop()).ok
    with caplog.at_level("INFO", logger="fake_pc"):
        run(fake, clock, 3.0)
    assert [r.getMessage() for r in caplog.records] == ["operator stand-in: h", "operator stand-in: p"]


# ---------------------------------------------------------------------------------------------- settings

def test_the_yaml_keys_load(tmp_path):
    s = fake_pc_settings(tmp_path)
    assert np.allclose(s.p_pose, s.start_pose)
    assert s.at_pose_tol_m == 0.002 and s.auto_h_p_after_stop_s == 1.0
    off = fake_pc_settings(tmp_path, p_pose=None, auto_h_p_after_stop_s=None)
    assert off.p_pose is None and off.auto_h_p_after_stop_s is None


@pytest.mark.parametrize("changes, problem", [
    ({"p_pose": [[1, 0], [0, 1]]}, "fake_pc.p_pose: expected a rigid 4x4 pose"),
    ({"at_pose_tol_m": 0.0}, "fake_pc.at_pose_tol_m: expected a number > 0"),
    ({"auto_h_p_after_stop_s": -1.0}, "fake_pc.auto_h_p_after_stop_s: expected null or a number >= 0"),
    ({"auto_h_p_after_stop_s": "soon"}, "fake_pc.auto_h_p_after_stop_s: expected null or a number >= 0"),
])
def test_bad_values_are_named(tmp_path, changes, problem):
    with pytest.raises(StationError) as e:
        fake_pc_settings(tmp_path, **changes)
    assert any(p.startswith(problem) for p in e.value.problems), e.value.problems


@pytest.mark.parametrize("key", ["p_pose", "auto_h_p_after_stop_s"])
def test_a_missing_key_is_named(tmp_path, key):
    from teleop_collect.fake_pc.settings import FakePcSettings
    config = load_config("fake_pc", None)
    del config[key]
    with pytest.raises(StationError) as e:
        FakePcSettings.from_config(config, tmp_path, state_endpoint="tcp://127.0.0.1:*",
                                   commands_endpoint="tcp://127.0.0.1:*")
    assert f"fake_pc.{key}: missing (null turns it off)" in e.value.problems
