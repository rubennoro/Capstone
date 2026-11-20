"""MotionSettings (values, the margin rule), advance (math), WaypointPlanner (times, poses), worst_step."""
from __future__ import annotations

import math

import numpy as np
import pytest
from core_helpers import MS, T0_PC, home_pose, motion_config, motion_settings, rigid_error, tilted_pose, z_tilt_deg

from teleop_collect.teleop.motion import (MotionSettings, Twist, WaypointPlanner, advance, assumed_margins_text,
                                         worst_step)

S = motion_settings()


# --- settings ------------------------------------------------------------------------------------------------------

def test_defaults_load_and_worst_step():
    # wave 3 (no software stops): K 6 (lead 300 ms), state_max_age_ms 250, the PC margins 0.05 m / 10 deg
    assert S.waypoints == 6 and S.spacing_ns == 50 * MS and S.lead_ns == 300 * MS and S.state_max_age_ns == 250 * MS
    assert S.assumed_pc_box_margin_m == 0.05 and S.assumed_pc_cone_margin_deg == 10.0
    step_m, step_deg = worst_step(S)
    assert step_m == pytest.approx(0.0275) and step_deg == pytest.approx(5.5)     # <= 0.8 x 50 mm, 0.8 x 10 deg
    assert step_m <= S.margin_use * S.assumed_pc_box_margin_m
    assert step_deg <= S.margin_use * S.assumed_pc_cone_margin_deg
    assert isinstance(S.spacing_ms, float)


def test_missing_and_unknown_keys_refused():
    cfg = motion_config()
    del cfg["v_max_m_s"]
    with pytest.raises(ValueError, match="missing v_max_m_s"):
        MotionSettings.from_config(cfg)
    with pytest.raises(ValueError, match="unknown extra"):
        MotionSettings.from_config(motion_config(extra=1))


@pytest.mark.parametrize("key, value", [
    ("v_max_m_s", 0.0), ("v_max_m_s", -0.01), ("v_max_m_s", "fast"), ("v_max_m_s", True), ("v_max_m_s", math.nan),
    ("w_max_deg_s", 0.0), ("slow_factor", 0.0), ("slow_factor", 1.5), ("waypoints", 0), ("waypoints", 2.0),
    ("waypoints", 65), ("spacing_ms", 0), ("key_stale_ms", 0), ("state_max_age_ms", -1), ("margin_use", 0.0),
    ("margin_use", 1.2), ("assumed_pc_box_margin_m", 0.0), ("assumed_pc_cone_margin_deg", 0.0),
])
def test_bad_values_refused(key, value):
    with pytest.raises(ValueError, match=key):
        MotionSettings.from_config(motion_config(**{key: value}))


@pytest.mark.parametrize("changes, what", [
    ({"v_max_m_s": 0.08}, "worst step"),                       # 0.08 x 0.55 s = 44 mm > 0.8 x 50 mm
    ({"w_max_deg_s": 15.0}, "worst tilt step"),                # 15 x 0.55 s = 8.25 deg > 0.8 x 10 deg
    ({"waypoints": 12}, "worst step"),                         # lead 600 ms: 0.05 x 0.85 s = 42.5 mm
    ({"state_max_age_ms": 600}, "worst step"),                 # 0.05 x 0.9 s = 45 mm
    ({"assumed_pc_box_margin_m": 0.034}, "worst step"),        # 27.5 mm > 0.8 x 34 mm
    ({"margin_use": 0.5}, "worst step"),
])
def test_margin_rule_refuses(changes, what):
    with pytest.raises(ValueError, match=what):
        motion_settings(**changes)


def test_margin_rule_allows_slower_settings():
    s = motion_settings(v_max_m_s=0.02, w_max_deg_s=5.0, waypoints=4)
    assert worst_step(s)[0] == pytest.approx(0.02 * 0.45)


def test_assumed_margins_text_names_the_pc_margins_and_the_risk():
    """Wave 3: shown at start-up and as teleop_collect check's INFO row (launcher/processes.py, the same text)."""
    from teleop_collect.launcher.processes import ASSUMED_MARGINS
    from teleop_collect.teleop.motion.planner import ASSUMED_MARGINS as PLANNER_TEXT
    assert assumed_margins_text(S) == (
        "teleop assumes the PC's safety.box_margin_m 0.05 and cone_margin_deg 10 (the teleop block of the PC's "
        "station.yaml); with smaller PC margins a press into the floor can fault the PC")
    assert ASSUMED_MARGINS == PLANNER_TEXT


# --- advance -------------------------------------------------------------------------------------------------------

def twist(v=(0.0, 0.0, 0.0), w=0.0, tilt=(0.0, 0.0)):
    return Twist(np.array(v, dtype=float), w, np.array(tilt, dtype=float))


@pytest.mark.parametrize("T", [home_pose(), tilted_pose()], ids=["home", "tilted"])
def test_advance_identity_at_tau_zero(T):
    out = advance(T, twist((0.1, 0.2, 0.3), 0.5, (0.3, -0.2)), 0.0)
    assert np.allclose(out, T, rtol=0, atol=1e-12)
    assert np.array_equal(out[3], [0.0, 0.0, 0.0, 1.0])


def test_advance_translation_in_base_frame():
    T = tilted_pose()
    out = advance(T, twist((0.05, -0.02, 0.01)), 0.1)
    assert np.allclose(out[:3, 3], T[:3, 3] + [0.005, -0.002, 0.001], rtol=0, atol=1e-15)
    assert np.allclose(out[:3, :3], T[:3, :3], rtol=0, atol=1e-12)


@pytest.mark.parametrize("T", [home_pose(), tilted_pose()], ids=["home", "tilted"])
def test_pure_rotation_keeps_the_tcp_position(T):
    out = advance(T, twist(w=0.7, tilt=(0.4, -0.5)), 0.3)
    assert np.array_equal(out[:3, 3], T[:3, 3])
    assert rigid_error(out) < 1e-12


def test_twist_about_own_axis_even_when_tilted():
    T = tilted_pose()
    out = advance(T, twist(w=0.5), 0.4)
    assert np.allclose(out[:3, 2], T[:3, 2], atol=1e-12)                    # own z unchanged
    rel = T[:3, :3].T @ out[:3, :3]                                          # the change in the TCP frame
    assert np.allclose(rel, [[math.cos(0.2), -math.sin(0.2), 0], [math.sin(0.2), math.cos(0.2), 0], [0, 0, 1]],
                       atol=1e-12)


def test_tilt_about_base_axes_when_twisted():
    T = tilted_pose()
    out = advance(T, twist(tilt=(0.0, 0.3)), 0.5)                            # 0.15 rad about base y
    left = out[:3, :3] @ T[:3, :3].T                                         # the change in the base frame
    c, s = math.cos(0.15), math.sin(0.15)
    assert np.allclose(left, [[c, 0, s], [0, 1, 0], [-s, 0, c]], atol=1e-12)
    assert z_tilt_deg(T[:3, :3], out[:3, :3]) <= math.degrees(0.15) + 1e-9


def test_advance_reorthonormalises_a_slightly_off_base():
    T = home_pose()
    T[:3, :3] += 1e-7 * np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert rigid_error(advance(T, twist(w=0.1), 0.02)) < 1e-12


# --- planner -------------------------------------------------------------------------------------------------------

def test_move_times_and_poses():
    p = WaypointPlanner(S)
    tw = twist((0.05, 0.0, 0.0))
    t_base = T0_PC
    now = T0_PC + 20 * MS
    wps = p.move(home_pose(), t_base, tw, now, 0.04)
    assert [w.t_pc_ns for w in wps] == [now + k * 50 * MS for k in range(1, 7)]
    for w in wps:
        tau = (w.t_pc_ns - t_base) / 1e9
        assert np.allclose(w.O_T_TCP, advance(home_pose(), tw, tau), rtol=0, atol=1e-15)
        assert w.width_m == 0.04 and isinstance(w.t_pc_ns, int)
    assert wps[-1].O_T_TCP[0, 3] == pytest.approx(0.45 + 0.05 * 0.32)


def test_release_advances_over_the_state_age_only():
    p = WaypointPlanner(S)
    tw = twist((0.0, 0.05, 0.0))
    now = T0_PC + 30 * MS
    (w,) = p.release(home_pose(), T0_PC, tw, now, 0.08)
    assert w.t_pc_ns == now + 50 * MS
    assert w.O_T_TCP[1, 3] == pytest.approx(0.05 * 0.03)
    (w,) = p.release(home_pose(), T0_PC, tw, now, 0.08, until_pc_ns=T0_PC + 10 * MS)    # the plan ended earlier
    assert w.O_T_TCP[1, 3] == pytest.approx(0.05 * 0.01)
    (w,) = p.release(home_pose(), T0_PC, tw, now, 0.08, until_pc_ns=T0_PC - 10 * MS)    # ended before the state
    assert np.allclose(w.O_T_TCP, home_pose(), atol=1e-15)
    (w,) = p.release(home_pose(), now + 5 * MS, tw, now, 0.08)                          # a state "from the future"
    assert np.allclose(w.O_T_TCP, home_pose(), atol=1e-15)


def test_hold_is_the_base_pose_with_the_width():
    (w,) = WaypointPlanner(S).hold(tilted_pose(), T0_PC, 0.0)
    assert w.t_pc_ns == T0_PC + 50 * MS and w.width_m == 0.0
    assert np.allclose(w.O_T_TCP, tilted_pose(), rtol=0, atol=1e-14)


def test_waypoint_is_frozen_and_checked():
    from teleop_collect.teleop.motion import Waypoint
    w = Waypoint(1, home_pose(), 0.01)
    with pytest.raises(ValueError):
        w.O_T_TCP[0, 0] = 2.0
    with pytest.raises(ValueError):
        Waypoint(1, np.eye(3), 0.0)
    with pytest.raises(ValueError):
        Waypoint(1, home_pose(), -0.01)
