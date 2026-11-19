"""KeyState (deadman, changes) and twist_from_keys (signs, cancelling, slow, tilt normalisation)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from core_helpers import MS, home_pose, motion_settings, z_tilt_deg

from teleop_collect.contracts import MOTION_KEYS
from teleop_collect.teleop.motion import NO_KEYS, HeldKeys, KeyState, Twist, advance, twist_from_keys

S = motion_settings()
V = S.v_max_m_s
W = math.radians(S.w_max_deg_s)


def keys(*held, slow=False):
    return HeldKeys(frozenset(held), slow)


# --- KeyState -----------------------------------------------------------------------------------------------------

def test_no_update_means_no_keys():
    assert KeyState(150).current(10**12) == NO_KEYS


def test_current_returns_held_while_fresh_and_releases_when_stale():
    ks = KeyState(150)
    ks.update(["x+", "z-"], True, True, 1_000 * MS)
    assert ks.current(1_000 * MS) == keys("x+", "z-", slow=True)
    assert ks.current(1_150 * MS) == keys("x+", "z-", slow=True)      # exactly stale_ms: still fresh
    assert ks.current(1_150 * MS + 1) == NO_KEYS                       # the deadman
    ks.update(["x+"], False, True, 1_200 * MS)                         # a new state revives it
    assert ks.current(1_210 * MS) == keys("x+")


def test_no_focus_means_no_keys():
    ks = KeyState(150)
    ks.update(["x+"], False, False, 0)
    assert ks.current(0) == NO_KEYS


def test_unknown_key_refused():
    with pytest.raises(ValueError, match="unknown motion keys"):
        KeyState(150).update(["jump"], False, True, 0)
    with pytest.raises(ValueError):
        KeyState(0)


def test_changes_only_on_change_and_drained():
    ks = KeyState(150)
    ks.update(["x+"], False, True, 1)
    ks.update(["x+"], False, True, 2)            # a resend: no change
    ks.update(["x+"], True, True, 3)             # slow changed
    ks.update(["x+"], True, False, 4)            # focus changed
    got = ks.changes()
    assert [(t, k, f) for t, k, f in got] == [(1, keys("x+"), True), (3, keys("x+", slow=True), True),
                                              (4, keys("x+", slow=True), False)]
    assert ks.changes() == []
    ks.update(["x+"], True, False, 5)            # same as the last change: still nothing
    assert ks.changes() == []


def test_held_keys_names_in_motion_key_order():
    assert keys("tilt_right", "x-", "x+").names() == ["x+", "x-", "tilt_right"]


# --- twist ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key, axis, sign", [("x+", 0, 1), ("x-", 0, -1), ("y+", 1, 1), ("y-", 1, -1),
                                             ("z+", 2, 1), ("z-", 2, -1)])
def test_translation_keys(key, axis, sign):
    t = twist_from_keys(keys(key), S)
    expected = np.zeros(3)
    expected[axis] = sign * V
    assert np.array_equal(t.v_m_s, expected)
    assert t.w_twist_rad_s == 0.0 and np.array_equal(t.w_tilt_rad_s, np.zeros(2))


def test_opposite_keys_cancel_and_all_keys_is_zero():
    assert twist_from_keys(keys("x+", "x-"), S).is_zero()
    assert twist_from_keys(keys("twist+", "twist-"), S).is_zero()
    assert twist_from_keys(keys("tilt_fwd", "tilt_back", "tilt_left", "tilt_right"), S).is_zero()
    assert twist_from_keys(keys(*MOTION_KEYS), S).is_zero()
    assert twist_from_keys(NO_KEYS, S).is_zero()


def test_slow_scales_everything():
    fast = twist_from_keys(keys("x+", "y-", "z+", "twist+", "tilt_fwd"), S)
    slow = twist_from_keys(keys("x+", "y-", "z+", "twist+", "tilt_fwd", slow=True), S)
    assert np.allclose(np.array(slow.values()), np.array(fast.values()) * S.slow_factor, rtol=0, atol=1e-15)


def test_tilt_axes_and_twist_sign():
    assert np.allclose(twist_from_keys(keys("tilt_left"), S).w_tilt_rad_s, [W, 0.0])
    assert np.allclose(twist_from_keys(keys("tilt_right"), S).w_tilt_rad_s, [-W, 0.0])
    assert np.allclose(twist_from_keys(keys("tilt_fwd"), S).w_tilt_rad_s, [0.0, -W])
    assert np.allclose(twist_from_keys(keys("tilt_back"), S).w_tilt_rad_s, [0.0, W])
    assert twist_from_keys(keys("twist+"), S).w_twist_rad_s == pytest.approx(W)
    assert twist_from_keys(keys("twist-"), S).w_twist_rad_s == pytest.approx(-W)


def test_diagonal_tilt_normalised_to_w_max():
    for combo in (("tilt_fwd", "tilt_left"), ("tilt_back", "tilt_right"), ("tilt_fwd", "tilt_right")):
        t = twist_from_keys(keys(*combo), S)
        assert np.linalg.norm(t.w_tilt_rad_s) == pytest.approx(W, rel=1e-12)
        assert abs(t.w_tilt_rad_s[0]) == pytest.approx(abs(t.w_tilt_rad_s[1]))


@pytest.mark.parametrize("key, direction", [("tilt_fwd", (1, 0)), ("tilt_back", (-1, 0)),
                                            ("tilt_left", (0, 1)), ("tilt_right", (0, -1))])
def test_tilt_by_effect_on_the_tcp_z_axis(key, direction):
    """At home the TCP z axis points down; tilt_fwd moves it toward +x, tilt_left toward +y."""
    T = advance(home_pose(), twist_from_keys(keys(key), S), 1.0)
    z = T[:3, 2]
    moved, still = (0, 1) if direction[0] else (1, 0)
    assert z[moved] * direction[moved] > 0.1
    assert abs(z[still]) < 1e-12
    assert np.allclose(T[:3, 3], home_pose()[:3, 3], rtol=0, atol=1e-15)     # pivot at the TCP point
    assert z_tilt_deg(home_pose()[:3, :3], T[:3, :3]) == pytest.approx(S.w_max_deg_s, rel=1e-9)


def test_twist_plus_turns_the_hand_right_handed_about_its_own_z():
    T0 = home_pose()
    T = advance(T0, twist_from_keys(keys("twist+"), S), 1.0)
    x0, y0 = T0[:3, 0], T0[:3, 1]
    x1 = T[:3, 0]
    assert np.dot(x1, y0) > 0.1                           # x turns toward y: positive about z
    assert np.allclose(T[:3, 2], T0[:3, 2], atol=1e-12)   # the approach axis stays
    assert np.degrees(np.arccos(np.clip(np.dot(x0, x1), -1, 1))) == pytest.approx(S.w_max_deg_s, rel=1e-9)


def test_twist_values_and_validation():
    t = Twist(np.array([1.0, 2.0, 3.0]), 4.0, np.array([5.0, 6.0]))
    assert t.values() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert not t.is_zero() and Twist.zero().is_zero()
    with pytest.raises(ValueError):
        Twist(np.array([np.nan, 0, 0]), 0.0, np.zeros(2))
