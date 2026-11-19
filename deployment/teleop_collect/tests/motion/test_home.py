"""HomeSettings and HomeReference (PLAN 8)."""
from __future__ import annotations

import numpy as np
import pytest
from core_helpers import Q_HOME, home_pose, home_settings, teleop_config

from teleop_collect.teleop.motion import HomeReference, HomeSettings


def test_settings_from_config():
    s = home_settings()
    assert s.on_start is True and s.tolerance_rad == 0.01 and s.tolerance_m == 0.005


@pytest.mark.parametrize("key, value", [("on_start", "yes"), ("tolerance_rad", 0.0), ("tolerance_m", -1.0),
                                        ("tolerance_rad", True)])
def test_bad_settings_refused(key, value):
    with pytest.raises(ValueError, match=key):
        home_settings(**{key: value})


def test_missing_and_unknown_keys():
    cfg = dict(teleop_config()["home"])
    del cfg["on_start"]
    with pytest.raises(ValueError, match="missing on_start"):
        HomeSettings.from_config(cfg)
    with pytest.raises(ValueError, match="unknown x"):
        HomeSettings.from_config({**teleop_config()["home"], "x": 1})


def test_unknown_reference_is_never_at_home():
    ref = HomeReference(home_settings())
    assert not ref.known and not ref.at_home(Q_HOME, home_pose())


def test_tolerances():
    ref = HomeReference(home_settings())
    ref.set_from(Q_HOME, home_pose())
    assert ref.known and ref.at_home(Q_HOME, home_pose())
    q = Q_HOME.copy()
    q[3] += 0.0099
    assert ref.at_home(q, home_pose())
    q[3] += 0.0002
    assert not ref.at_home(q, home_pose())
    assert ref.at_home(Q_HOME, home_pose(x=0.45 + 0.0049))
    assert not ref.at_home(Q_HOME, home_pose(x=0.45 + 0.004, y=0.004))      # 5.7 mm away


def test_observe_learns_from_the_pcs_own_home():
    ref = HomeReference(home_settings())
    q2 = Q_HOME + 0.1
    ref.observe("idle", Q_HOME, home_pose())
    ref.observe("idle", Q_HOME, home_pose())
    assert not ref.known                                           # idle -> idle is not a home
    ref.observe("homing", Q_HOME, home_pose())
    ref.observe("stopped", Q_HOME, home_pose())
    assert not ref.known                                           # a stopped home is not a home
    ref.observe("homing", q2, home_pose())
    ref.observe("idle", q2, home_pose(z=0.4))
    assert ref.known and ref.at_home(q2, home_pose(z=0.4)) and not ref.at_home(Q_HOME, home_pose())
    assert np.allclose(ref.q_ref, q2)


def test_bad_shapes():
    ref = HomeReference(home_settings())
    with pytest.raises(ValueError):
        ref.set_from(np.zeros(6), home_pose())
    with pytest.raises(ValueError):
        ref.set_from(Q_HOME, np.eye(3))


def test_hand_homing_away_from_home_does_not_count():
    """The PC's g (hand only) shows homing -> idle too; once a home is known it never moves the reference."""
    ref = HomeReference(home_settings())
    ref.set_from(Q_HOME, home_pose())                              # the start-up C3 home
    away_q, away_T = Q_HOME + 0.3, home_pose(x=0.6, z=0.2)         # after an episode, away from home
    ref.observe("holding", away_q, away_T)
    ref.observe("idle", away_q, away_T)
    ref.observe("homing", away_q, away_T)                          # g on the PC
    ref.observe("idle", away_q, away_T)
    assert not ref.at_home(away_q, away_T) and ref.at_home(Q_HOME, home_pose())


def test_observed_reference_is_kept_until_a_c3_home():
    ref = HomeReference(home_settings())
    away = Q_HOME + 0.3
    ref.observe("homing", away, home_pose(x=0.6))
    ref.observe("homing", Q_HOME + 0.1, home_pose(x=0.5))          # moving
    ref.observe("idle", Q_HOME, home_pose())                       # h on the PC moved the arm: learned
    assert ref.at_home(Q_HOME, home_pose())
    ref.observe("homing", Q_HOME + 0.2, home_pose(z=0.5))
    ref.observe("idle", Q_HOME, home_pose())                       # a later homing -> idle: ignored
    ref.observe("homing", Q_HOME, home_pose())
    ref.observe("idle", Q_HOME + 0.2, home_pose(z=0.5))
    assert ref.at_home(Q_HOME, home_pose()) and not ref.at_home(Q_HOME + 0.2, home_pose(z=0.5))
    ref.set_from(Q_HOME + 0.2, home_pose(z=0.5))                   # a C3 home teleop sent replaces it
    assert ref.at_home(Q_HOME + 0.2, home_pose(z=0.5)) and not ref.at_home(Q_HOME, home_pose())


def test_hand_homing_before_any_home_is_not_accepted():
    """g on the PC before any home: homing -> idle, but neither q nor the TCP target moves."""
    ref = HomeReference(home_settings())
    away_q, away_T = Q_HOME + 0.3, home_pose(x=0.6, z=0.2)
    for mode in ("idle", "homing", "homing", "idle"):
        ref.observe(mode, away_q, away_T)
    assert not ref.known and not ref.at_home(away_q, away_T)


def test_pc_h_that_moves_the_arm_is_accepted():
    ref = HomeReference(home_settings())
    ref.observe("idle", Q_HOME + 0.3, home_pose(x=0.6))
    ref.observe("homing", Q_HOME + 0.3, home_pose(x=0.6))
    ref.observe("homing", Q_HOME + 0.1, home_pose(x=0.5))
    ref.observe("idle", Q_HOME, home_pose())
    assert ref.known and ref.at_home(Q_HOME, home_pose())


def test_fake_pc_home_moves_only_the_tcp_target():
    """The fake PC reports q = 0 throughout; its arm home moves the TCP target, which is enough."""
    ref = HomeReference(home_settings())
    zero = np.zeros(7)
    ref.observe("homing", zero, home_pose(x=0.6))
    ref.observe("idle", zero, home_pose())
    assert ref.known and ref.at_home(zero, home_pose())


def test_movement_within_tolerance_does_not_count():
    ref = HomeReference(home_settings())
    q = Q_HOME.copy()
    ref.observe("homing", q, home_pose())
    q2 = q.copy()
    q2[0] += 0.009
    ref.observe("idle", q2, home_pose(x=0.45 + 0.004))
    assert not ref.known


def test_pc_h_already_at_home_is_not_accepted():
    """h while the arm already stands at home moves nothing, so it looks like a g: not accepted, and Start stays
    refused until a C3 home (set_from)."""
    ref = HomeReference(home_settings())
    for mode in ("idle", "homing", "homing", "idle"):
        ref.observe(mode, Q_HOME, home_pose())
    assert not ref.known and not ref.at_home(Q_HOME, home_pose())
    ref.set_from(Q_HOME, home_pose())
    assert ref.at_home(Q_HOME, home_pose())


def test_homing_ending_stopped_then_idle_is_not_a_home():
    ref = HomeReference(home_settings())
    ref.observe("homing", Q_HOME + 0.3, home_pose(x=0.6))
    ref.observe("stopped", Q_HOME + 0.1, home_pose(x=0.5))         # s on the PC mid-home
    ref.observe("idle", Q_HOME + 0.1, home_pose(x=0.5))
    assert not ref.known
