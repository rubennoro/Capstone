"""StartPoseSettings and StartPoseReference (PLAN 8): learned at the end of start-up, follows the PC's h/p until the
PC accepts a start, dropped when the PC refuses one away from its start pose."""
from __future__ import annotations

import numpy as np
import pytest
from core_helpers import Q_HOME, home_pose, start_pose_settings, teleop_config

from teleop_collect.teleop.motion import StartPoseReference, StartPoseSettings

Q_START = Q_HOME + np.array([0.0, 0.2, 0.0, 0.3, 0.0, -0.1, 0.0])
T_START = home_pose(x=0.55, z=0.25)


def ref() -> StartPoseReference:
    return StartPoseReference(start_pose_settings())


def h_then_p(r: StartPoseReference, q_from, T_from, q_to=Q_START, T_to=T_START) -> None:
    """The PC's h (to home), then p (home -> start pose), as C2 shows them."""
    for mode, q, T in (("idle", q_from, T_from), ("homing", q_from, T_from), ("homing", Q_HOME + 0.05, home_pose()),
                       ("idle", Q_HOME, home_pose()), ("homing", Q_HOME, home_pose()), ("homing", q_to, T_to),
                       ("idle", q_to, T_to)):
        r.observe(mode, q, T)


# ---- settings

def test_settings_from_config():
    s = start_pose_settings()
    assert s.tolerance_rad == 0.01 and s.tolerance_m == 0.005


@pytest.mark.parametrize("key, value", [("tolerance_rad", 0.0), ("tolerance_m", -1.0), ("tolerance_rad", True),
                                        ("tolerance_m", float("nan")), ("tolerance_m", "0.005")])
def test_bad_settings_refused(key, value):
    with pytest.raises(ValueError, match=f"start_pose.{key}"):
        start_pose_settings(**{key: value})


def test_missing_and_unknown_keys():
    cfg = dict(teleop_config()["start_pose"])
    del cfg["tolerance_m"]
    with pytest.raises(ValueError, match="start_pose: missing tolerance_m"):
        StartPoseSettings.from_config(cfg)
    with pytest.raises(ValueError, match="start_pose: unknown on_start"):
        StartPoseSettings.from_config({**teleop_config()["start_pose"], "on_start": True})


# ---- learn, tolerances

def test_nothing_known_at_first():
    r = ref()
    assert not r.known and not r.confirmed and r.q_ref is None and r.p_ref is None
    assert not r.at_start(Q_START, T_START) and not r.at_refused_pose(Q_START, T_START)


def test_learn_is_a_candidate_not_confirmed():
    r = ref()
    r.learn(Q_START, T_START)
    assert r.known and not r.confirmed and r.at_start(Q_START, T_START)
    assert np.allclose(r.q_ref, Q_START) and np.allclose(r.p_ref, T_START[:3, 3])


def test_tolerances():
    r = ref()
    r.learn(Q_START, T_START)
    q = Q_START.copy()
    q[3] += 0.0099
    assert r.at_start(q, T_START)
    q[3] += 0.0002
    assert not r.at_start(q, T_START)
    assert r.at_start(Q_START, home_pose(x=0.55 + 0.0049, z=0.25))
    assert not r.at_start(Q_START, home_pose(x=0.55 + 0.004, y=0.004, z=0.25))     # 5.7 mm away


def test_bad_shapes():
    r = ref()
    with pytest.raises(ValueError):
        r.learn(np.zeros(6), T_START)
    with pytest.raises(ValueError):
        r.learn(Q_START, np.eye(3))


# ---- observe: the PC's h and p

def test_h_then_p_replaces_the_candidate():
    """Start-up learned a wrong pose (the operator forgot p); h, then p on the PC: the pose after p is the candidate."""
    r = ref()
    away_q, away_T = Q_HOME + 0.3, home_pose(x=0.6, z=0.2)
    r.learn(away_q, away_T)
    h_then_p(r, away_q, away_T)
    assert r.known and not r.confirmed and r.at_start(Q_START, T_START) and not r.at_start(away_q, away_T)


def test_a_moving_homing_sets_a_candidate_when_none_is_known():
    r = ref()
    r.observe("idle", Q_HOME, home_pose())
    r.observe("homing", Q_HOME, home_pose())
    r.observe("homing", Q_START, T_START)
    r.observe("idle", Q_START, T_START)
    assert r.known and r.at_start(Q_START, T_START)


def test_g_never_counts():
    """The PC's g (the hand alone) shows homing -> idle too, moving neither the joints nor the TCP target."""
    r = ref()
    away_q, away_T = Q_HOME + 0.3, home_pose(x=0.6, z=0.2)
    for mode in ("idle", "homing", "homing", "idle"):
        r.observe(mode, away_q, away_T)
    assert not r.known                                              # none known: g does not make one
    r.learn(Q_START, T_START)
    for mode, q, T in (("holding", away_q, away_T), ("idle", away_q, away_T), ("homing", away_q, away_T),
                       ("idle", away_q, away_T)):
        r.observe(mode, q, T)
    assert r.at_start(Q_START, T_START) and not r.at_start(away_q, away_T)


def test_idle_to_idle_and_a_stopped_homing_do_not_count():
    r = ref()
    r.observe("idle", Q_HOME, home_pose())
    r.observe("idle", Q_START, T_START)
    assert not r.known
    r.observe("homing", Q_HOME, home_pose())
    r.observe("stopped", Q_START, T_START)                          # s on the PC mid-move
    r.observe("idle", Q_START, T_START)
    assert not r.known


def test_movement_within_tolerance_does_not_count():
    r = ref()
    r.observe("homing", Q_START, T_START)
    q2 = Q_START.copy()
    q2[0] += 0.009
    r.observe("idle", q2, home_pose(x=0.55 + 0.004, z=0.25))
    assert not r.known


def test_fake_pc_moves_only_the_tcp_target():
    """The fake PC reports q = 0 throughout: its p moves the TCP target, which is enough."""
    r = ref()
    zero = np.zeros(7)
    r.observe("homing", zero, home_pose())
    r.observe("idle", zero, T_START)
    assert r.known and r.at_start(zero, T_START) and not r.at_start(zero, home_pose())


# ---- confirm, refused_here

def test_confirm_fixes_the_pose():
    r = ref()
    r.learn(Q_START, T_START)
    r.confirm()
    assert r.confirmed and r.known
    h_then_p(r, Q_START, T_START, Q_HOME + 0.2, home_pose(z=0.5))   # a later h/p elsewhere: ignored
    r.learn(Q_HOME, home_pose())                                    # and so is a learn
    assert r.at_start(Q_START, T_START) and not r.at_start(Q_HOME + 0.2, home_pose(z=0.5))


def test_confirm_with_nothing_known_does_nothing():
    r = ref()
    r.confirm()
    assert not r.confirmed and not r.known


def test_refused_here_drops_the_candidate_and_keeps_the_pose_for_the_hint():
    r = ref()
    wrong_q, wrong_T = Q_HOME + 0.3, home_pose(x=0.6, z=0.2)
    r.learn(wrong_q, wrong_T)
    r.refused_here(wrong_q, wrong_T)
    assert not r.known and not r.confirmed and not r.at_start(wrong_q, wrong_T)
    assert r.at_refused_pose(wrong_q, wrong_T) and not r.at_refused_pose(Q_START, T_START)
    h_then_p(r, wrong_q, wrong_T)                                   # the next h, p: the new candidate
    assert r.known and r.at_start(Q_START, T_START) and not r.confirmed
    r.confirm()
    assert r.confirmed and r.at_start(Q_START, T_START)


def test_refused_pose_tolerances():
    r = ref()
    r.refused_here(Q_START, T_START)
    assert r.at_refused_pose(Q_START, home_pose(x=0.55 + 0.0049, z=0.25))
    assert not r.at_refused_pose(Q_START, home_pose(x=0.55 + 0.006, z=0.25))
    q = Q_START.copy()
    q[6] += 0.011
    assert not r.at_refused_pose(q, T_START)


def test_refused_here_once_confirmed_keeps_the_pose():
    r = ref()
    r.learn(Q_START, T_START)
    r.confirm()
    r.refused_here(Q_HOME, home_pose())
    assert r.confirmed and r.known and r.at_start(Q_START, T_START) and r.at_refused_pose(Q_HOME, home_pose())
