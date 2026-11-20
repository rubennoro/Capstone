"""MotionController: the step rules of STRUCTURE.md 7.4, the stale-state guard, release once, fingers, and the
PLAN 5.3 safety bound for every key combination and state age up to the guard."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from core_helpers import MS, T0_PC, arm_state, home_pose, motion_settings, rigid_error, tilted_pose, z_tilt_deg
from deploy_contracts import SetWaypoints

from teleop_collect.contracts import MOTION_KEYS
from teleop_collect.teleop.motion import (NO_KEYS, BaseState, HeldKeys, KeyState, MotionController, advance,
                                          worst_step)

S = motion_settings()
OPEN = 0.08


def keys(*held, slow=False):
    return HeldKeys(frozenset(held), slow)


def base(t_pc_ns=T0_PC, T=None, mode="holding", width=0.08):
    return BaseState(home_pose() if T is None else T, t_pc_ns, width, mode)


def started(width=0.08):
    c = MotionController(S, OPEN)
    c.begin(width)
    return c


def to_c3(cmd, chunk_id=1):
    """The C3 message E3 builds: proves the waypoints satisfy the strict contract (int64, float64, rigid)."""
    return SetWaypoints(chunk_id=chunk_id, t_ns=np.array([w.t_pc_ns for w in cmd.waypoints], dtype=np.int64),
                        O_T_TCP=np.stack([w.O_T_TCP for w in cmd.waypoints]),
                        gripper_width_m=np.array([w.width_m for w in cmd.waypoints], dtype=np.float64))


def test_not_begun_and_stopped_send_nothing():
    c = MotionController(S, OPEN)
    assert c.step(T0_PC, base(), keys("x+")) is None and not c.moving
    c.begin(0.08)
    assert c.step(T0_PC, base(), keys("x+")).kind == "move"
    c.stop()
    assert c.step(T0_PC, base(), keys("x+")) is None and not c.moving
    assert c.step(T0_PC, base(), NO_KEYS) is None                 # no release after stop either


def test_stale_state_guard():
    c = started()
    assert c.step(None, base(), keys("x+")) is None
    assert c.step(T0_PC, None, keys("x+")) is None
    assert c.step(T0_PC + S.state_max_age_ns + 1, base(), keys("x+")) is None     # 250 ms (wave 3)
    assert c.moving                                                # the key is in effect, nothing could be sent
    assert c.step(T0_PC + S.state_max_age_ns, base(), keys("x+")).kind == "move"


@pytest.mark.parametrize("mode", ["idle", "stopped", "homing", "fault"])
def test_no_command_unless_the_pc_follows(mode):
    assert started().step(T0_PC, base(mode=mode), keys("x+")) is None


def test_move_then_release_once():
    c = started(0.05)
    now = T0_PC + 10 * MS
    move = c.step(now, base(), keys("x+"))
    assert move.kind == "move" and len(move.waypoints) == S.waypoints and not move.slow
    assert all(w.width_m == 0.05 for w in move.waypoints)
    assert np.allclose(move.twist.v_m_s, [S.v_max_m_s, 0, 0])
    to_c3(move)
    now2 = now + 20 * MS
    rel = c.step(now2, base(t_pc_ns=now2 - 15 * MS), NO_KEYS)
    assert rel.kind == "release" and len(rel.waypoints) == 1
    assert rel.waypoints[0].t_pc_ns == now2 + S.spacing_ns
    assert np.allclose(rel.twist.v_m_s, move.twist.v_m_s)
    assert rel.waypoints[0].O_T_TCP[0, 3] == pytest.approx(0.45 + S.v_max_m_s * 0.015)
    to_c3(rel)
    assert c.step(now2 + 20 * MS, base(t_pc_ns=now2), NO_KEYS) is None
    assert not c.moving


def test_release_waits_for_a_fresh_state_and_never_passes_the_plan_end():
    c = started()
    move = c.step(T0_PC, base(), keys("y+"))
    plan_end = move.waypoints[-1].t_pc_ns                          # T0 + 100 ms
    late = T0_PC + 400 * MS
    assert c.step(late, base(), NO_KEYS) is None                   # state 400 ms old: nothing, release pending
    rel = c.step(late, base(t_pc_ns=late - 40 * MS), NO_KEYS)      # fresh again; the plan ended long ago
    assert rel.kind == "release"
    assert late - 40 * MS > plan_end
    assert np.allclose(rel.waypoints[0].O_T_TCP, home_pose(), atol=1e-15)   # the target stands still: no advance


def test_fingers_without_motion():
    c = started(0.08)
    assert c.step(T0_PC, base(T=tilted_pose()), NO_KEYS) is None
    c.gripper("close")
    assert c.commanded_width_m == 0.0
    cmd = c.step(T0_PC, base(T=tilted_pose()), NO_KEYS)
    assert cmd.kind == "fingers" and len(cmd.waypoints) == 1 and cmd.twist.is_zero()
    assert cmd.waypoints[0].width_m == 0.0
    assert np.allclose(cmd.waypoints[0].O_T_TCP, tilted_pose(), atol=1e-14)
    to_c3(cmd)
    assert c.step(T0_PC, base(), NO_KEYS) is None                  # once
    c.gripper("open")
    assert c.step(T0_PC, base(), NO_KEYS).waypoints[0].width_m == OPEN


def test_finger_command_rides_on_a_move():
    c = started()
    c.gripper("close")
    cmd = c.step(T0_PC, base(), keys("z-"))
    assert cmd.kind == "move" and all(w.width_m == 0.0 for w in cmd.waypoints)
    rel = c.step(T0_PC + 20 * MS, base(t_pc_ns=T0_PC + 10 * MS), NO_KEYS)
    assert rel.kind == "release" and rel.waypoints[0].width_m == 0.0
    assert c.step(T0_PC + 40 * MS, base(t_pc_ns=T0_PC + 30 * MS), NO_KEYS) is None   # no extra fingers set


def test_finger_command_waits_through_a_stale_state():
    c = started()
    c.gripper("close")
    late = T0_PC + S.state_max_age_ns + 10 * MS
    assert c.step(late, base(), NO_KEYS) is None
    assert c.step(late, base(t_pc_ns=T0_PC + 50 * MS), NO_KEYS).kind == "fingers"


def test_gripper_rules():
    c = MotionController(S, OPEN)
    c.gripper("close")                                             # ignored before begin
    c.begin(0.03)
    assert c.commanded_width_m == 0.03
    with pytest.raises(ValueError):
        c.gripper("toggle")
    with pytest.raises(ValueError):
        MotionController(S, 0.0)


@pytest.mark.parametrize("measured, commanded, clipped", [
    (0.081, OPEN, True), (0.0805, OPEN, True), (-0.001, 0.0, True), (-1.0, 0.0, True), (float("nan"), OPEN, True),
    (0.08, 0.08, False), (0.0, 0.0, False), (0.04, 0.04, False)])
def test_begin_clips_the_measured_width(measured, commanded, clipped):
    """An open hand reads above gripper_width_max_m: the gate would reject it (a fault). Never raises."""
    c = MotionController(S, OPEN)
    assert c.begin(measured) is clipped and c.width_clipped is clipped
    assert c.commanded_width_m == commanded
    for cmd in (c.step(T0_PC, base(), keys("x+")), c.step(T0_PC, base(), NO_KEYS)):
        assert all(w.width_m == commanded for w in cmd.waypoints)
    c.gripper("open")
    assert c.step(T0_PC, base(), NO_KEYS).waypoints[0].width_m == OPEN


def test_every_emitted_width_is_within_range():
    c = MotionController(S, OPEN)
    for measured in (0.2, 0.081, -0.5, 0.03):
        c.begin(measured)
        for g, held in (("close", ("x+",)), ("open", ()), ("open", ("z-",)), ("close", ())):
            c.gripper(g)
            for k in (keys(*held), NO_KEYS):
                cmd = c.step(T0_PC, base(), k)
                if cmd is not None:
                    assert all(0.0 <= w.width_m <= OPEN for w in cmd.waypoints)


def test_begin_resets_a_pending_release():
    c = started()
    c.step(T0_PC, base(), keys("x+"))
    c.stop()
    c.begin(0.08)
    assert c.step(T0_PC, base(), NO_KEYS) is None


def test_stale_keys_release_the_arm():
    ks = KeyState(S.key_stale_ms)
    c = started()
    ks.update(["x+"], False, True, 0)
    assert c.step(T0_PC, base(), ks.current(100 * MS)).kind == "move"
    assert c.step(T0_PC + 20 * MS, base(t_pc_ns=T0_PC + 10 * MS), ks.current(151 * MS)).kind == "release"


def test_slow_flag_and_scaled_speed():
    cmd = started().step(T0_PC, base(), keys("x+", slow=True))
    assert cmd.slow and cmd.twist.v_m_s[0] == pytest.approx(S.v_max_m_s * S.slow_factor)


def test_base_state_from_arm_state():
    st = arm_state(t_ns=T0_PC + 5, width=0.03, mode="running")
    b = BaseState.from_arm_state(st)
    assert b.t_pc_ns == T0_PC + 5 and b.width_m == 0.03 and b.mode == "running"
    assert np.array_equal(b.T_target, st.O_T_TCP_target)


# --- the PLAN 5.3 bound ---------------------------------------------------------------------------------------------

ALL_COMBOS = [frozenset(c) for r in range(len(MOTION_KEYS) + 1) for c in itertools.combinations(MOTION_KEYS, r)]


def test_every_key_combination_is_counted():
    assert len(ALL_COMBOS) == 2 ** 12


HOME = home_pose()


@pytest.mark.parametrize("T", [HOME, tilted_pose()], ids=["home", "tilted"])
def test_bound_for_every_key_combination_and_state_age(T):
    """Every waypoint of every move (and of the release after it) lies within worst_step of its base pose: per
    translation axis, and in the tilt of the TCP z axis. All 4096 combinations, slow and not, ages 0..guard."""
    step_m, step_deg = worst_step(S)
    ages = (0, S.state_max_age_ns)
    c = MotionController(S, OPEN)
    worst_seen_m = worst_seen_deg = 0.0
    for held, slow, age in itertools.product(ALL_COMBOS, (False, True), ages):
        c.begin(0.08)
        now = T0_PC + age
        b = BaseState(T, T0_PC, 0.08, "running")
        cmds = [c.step(now, b, HeldKeys(held, slow)), c.step(now, b, NO_KEYS)]
        for cmd in cmds:
            if cmd is None:
                continue
            for w in cmd.waypoints:
                dm = float(np.max(np.abs(w.O_T_TCP[:3, 3] - T[:3, 3])))
                ddeg = z_tilt_deg(T[:3, :3], w.O_T_TCP[:3, :3])
                assert dm <= step_m + 1e-12, (sorted(held), slow, age, dm)
                assert ddeg <= step_deg + 1e-9, (sorted(held), slow, age, ddeg)
                assert rigid_error(w.O_T_TCP) < 1e-9
                assert np.array_equal(w.O_T_TCP[3], [0.0, 0.0, 0.0, 1.0])
                worst_seen_m, worst_seen_deg = max(worst_seen_m, dm), max(worst_seen_deg, ddeg)
        assert c.step(now + 1, BaseState(T, now + 1 - S.state_max_age_ns - 1, 0.08, "running"),
                      HeldKeys(held, slow)) is None                # just past the guard: nothing
    assert worst_seen_m == pytest.approx(step_m)                   # the bound is reached, so it is the true worst
    if T is HOME:                                                  # (tilted, a base-axis turn moves z by less)
        assert worst_seen_deg == pytest.approx(step_deg)


def test_advance_matches_the_planner_used_by_the_controller():
    c = started()
    cmd = c.step(T0_PC + 20 * MS, base(), keys("x+", "twist+", "tilt_left"))
    for w in cmd.waypoints:
        assert np.allclose(w.O_T_TCP, advance(home_pose(), cmd.twist, (w.t_pc_ns - T0_PC) / 1e9), atol=1e-15)
