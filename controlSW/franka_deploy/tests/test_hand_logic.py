"""A5 hand_logic (pure; runs in env_realenv and env_bp): WidthMap, HandCommandPolicy hysteresis/deadband/interval/
busy/latest-wins/limits, config parsing.

    cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=. $RE -m pytest -q tests/test_hand_logic.py
    cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=. $BP -m pytest -q tests/test_hand_logic.py
"""
import numpy as np
import pytest

from franka_deploy.gripper.hand_logic import HandCmd, HandCommandPolicy, HandPolicyConfig, WidthMap

IDLE = {"busy": False}
BUSY = {"busy": True}
HOLDING = {"busy": False, "mode": "holding"}


def pol(**kw):
    return HandCommandPolicy(HandPolicyConfig(**kw), WidthMap())


def test_module_is_pure():
    import franka_deploy.gripper.hand_logic as hl

    src = open(hl.__file__).read()
    for mod in ("zmq", "mujoco", "robotmq"):
        assert f"import {mod}" not in src and f"from {mod}" not in src


def test_width_map_round_trip_and_clip():
    wm = WidthMap(a=1.02, b=0.007, gap_min_m=0.0, gap_max_m=0.08)
    for g in np.linspace(0, 0.08, 9):
        assert abs(wm.gap(wm.train(g)) - g) < 1e-12
    assert wm.gap(0.0) == 0.0 and wm.gap(0.5) == 0.08
    assert WidthMap.from_config({"a": 1.0, "b": 0.0431, "gap_min_m": 0.0, "gap_max_m": 0.08, "extra": 1}) == WidthMap()
    assert WidthMap().b == 0.0431                                # = train_closed_m (A5 review)
    with pytest.raises(ValueError):
        WidthMap.from_config({"a": 0.0})


def test_cmd_msg():
    m = HandCmd("grasp", 0.03, 0.1, 30.0, 0.01, 0.08).to_msg(7)
    assert m == {"kind": "grasp", "width_m": 0.03, "speed_m_s": 0.1, "force_n": 30.0, "eps_inner_m": 0.01,
                 "eps_outer_m": 0.08, "seq": 7}


def test_no_state_no_command():
    p = pol()
    assert p.update(0.05, 0.0, None) is None and p.update(float("nan"), 0.0, IDLE) is None


def test_hysteresis_no_chatter():
    c = HandPolicyConfig()
    p = HandCommandPolicy(c, WidthMap())
    t = 0.0
    cmds = []
    # oscillate +-3 mm around open_above while OPEN-at-start: only deadband moves, never a grasp
    for k in range(200):
        t += 0.02
        w = c.open_above_train_m + 0.003 * np.sin(k)
        r = p.update(w, t, IDLE)
        if r:
            cmds.append(r.kind)
    assert "grasp" not in cmds and p.state == "OPEN"
    # cross below close_below -> one grasp; then oscillate +-3 mm around close_below: nothing more
    r = p.update(c.close_below_train_m - 0.004, t + 0.02, IDLE)
    assert r.kind == "grasp" and p.state == "GRASPING"
    assert abs(r.width_m - WidthMap().gap(c.close_below_train_m - 0.004)) < 1e-12
    assert (r.force_n, r.eps_inner_m, r.eps_outer_m) == (c.force_n, c.eps_inner_m, c.eps_outer_m)
    n = 0
    for k in range(200):
        t += 0.02
        if p.update(c.close_below_train_m + 0.003 * np.sin(k), t, BUSY if k % 2 else HOLDING):
            n += 1
    assert n == 0 and p.state == "GRASPING"
    # above open_above -> one release move (also while busy, not rate limited)
    r = p.update(c.open_above_train_m + 0.001, t + 0.001, BUSY)
    assert r.kind == "move" and p.state == "OPEN"


def test_band_is_data_derived():
    """Defaults from firstreal-single (A5 WORKLOG): hold widths <= 0.0618 close, re-open widths >= 0.073 open."""
    c = HandPolicyConfig()
    assert 0.0618 < c.close_below_train_m < c.open_above_train_m < 0.073
    with pytest.raises(ValueError):
        HandPolicyConfig(close_below_train_m=0.07, open_above_train_m=0.07)


def test_deadband_and_min_interval():
    p = pol()
    r = p.update(0.08, 0.0, IDLE)
    wm = WidthMap()
    assert r.kind == "move" and abs(r.width_m - wm.gap(0.08)) < 1e-12
    assert p.update(0.0835, 1.0, IDLE) is None                 # 3.5 mm < deadband 4 mm
    r = p.update(0.0845, 0.1, IDLE)                            # > deadband but only 0.1 s after the last command
    assert r is None
    r = p.update(0.0845, 0.25, IDLE)
    assert r.kind == "move" and abs(r.width_m - wm.gap(0.0845)) < 1e-12


def test_no_move_while_busy_and_latest_wins():
    p = pol()
    p.update(0.08, 0.0, IDLE)
    for k, w in enumerate((0.075, 0.072, 0.070 + 0.0005)):     # all arrive while busy: nothing sent
        assert p.update(w, 1.0 + k, BUSY) is None
    r = p.update(0.0705, 5.0, IDLE)                             # not busy any more: the LATEST width goes out
    assert r.kind == "move" and abs(r.width_m - WidthMap().gap(0.0705)) < 1e-12


def test_grasp_not_rate_limited_and_not_blocked_by_busy():
    p = pol()
    p.update(0.08, 0.0, IDLE)
    r = p.update(0.05, 0.01, BUSY)
    assert r is not None and r.kind == "grasp"


def test_limits_force_speed_and_clips():
    p = pol(force_n=200.0, force_max_n=500.0, speed_m_s=0.5)
    assert p.force_n == 70.0 and p.speed_m_s == 0.1
    p = pol(force_n=50.0, force_max_n=40.0)
    assert p.force_n == 40.0
    p = pol()
    r = p.update(0.2, 0.0, IDLE)                                 # gap 0.192 > 0.08: clipped and counted
    assert r.width_m == 0.08 and p.clips == 1
    r = p.update(0.0, 1.0, IDLE)                                 # gap < 0: grasp at 0, counted
    assert r.kind == "grasp" and r.width_m == 0.0 and p.clips == 2
    with pytest.raises(ValueError):
        pol(force_n=0.0)


def test_send_failed_restores_state():
    p = pol()
    p.update(0.08, 0.0, IDLE)
    r = p.update(0.05, 1.0, IDLE)
    assert r.kind == "grasp" and p.state == "GRASPING"
    p.send_failed()
    assert p.state == "OPEN"
    assert p.update(0.05, 1.02, IDLE).kind == "grasp"           # re-issued on the next period


def test_config_from_gripper_dict():
    cfg = {"type": "franka_hand", "endpoint": "tcp://x:1", "speed_m_s": 0.05, "force_n": 20.0, "force_max_n": 70.0,
           "grasp_epsilon_inner_m": 0.02, "grasp_epsilon_outer_m": 0.07, "calibration": {"width_m": [0, 1]},
           "squeeze_delta_m": 0.005, "current_limit": 300, "policy": {"close_below_train_m": 0.062}}
    c = HandPolicyConfig.from_config(cfg)
    assert (c.speed_m_s, c.force_n, c.eps_inner_m, c.eps_outer_m, c.close_below_train_m) == (0.05, 20.0, 0.02, 0.07, 0.062)
    assert c.open_above_train_m == HandPolicyConfig().open_above_train_m
    with pytest.raises(ValueError, match="unknown key"):
        HandPolicyConfig.from_config({"policy": {"close_below": 0.06}})


def test_regrasp_after_grasp_failed_or_server_restart():
    """Review A5-5: GRASPING while the hand is idle and not holding re-grasps (min interval, at most max_regrasps in a
    row), then waits for the release; holding resets the count."""
    p = pol()
    p.update(0.08, 0.0, IDLE)
    assert p.update(0.05, 1.0, IDLE).kind == "grasp"
    assert p.update(0.05, 1.02, {"busy": True, "mode": "grasping"}) is None
    failed = {"busy": False, "mode": "idle", "last_result": "grasp_failed"}
    assert p.update(0.05, 1.1, failed) is None                   # min_cmd_interval_s since the grasp
    r = p.update(0.05, 1.3, failed)
    assert r.kind == "grasp" and p.regrasps == 1 and p.state == "GRASPING"
    assert p.update(0.05, 1.6, failed).kind == "grasp" and p.regrasps == 2
    assert p.update(0.05, 2.0, failed) is None and p.update(0.05, 5.0, failed) is None    # max_regrasps = 2
    assert p.update(0.075, 5.1, failed).kind == "move" and p.state == "OPEN"             # release still works
    # holding resets the count; a restarted hand_server (idle, no result) mid-hold gets a fresh grasp
    assert p.update(0.05, 6.0, IDLE).kind == "grasp"
    assert p.update(0.05, 6.5, {"busy": False, "mode": "holding", "last_result": "grasp_ok"}) is None
    restarted = {"busy": False, "mode": "idle", "last_result": "", "seq_done": 0}
    assert p.update(0.05, 7.0, restarted).kind == "grasp" and p.regrasps == 3
    p.send_failed()                                              # undelivered re-grasp: not counted
    assert p.regrasps == 2 and p.update(0.05, 7.3, restarted).kind == "grasp"
    with pytest.raises(ValueError):
        HandPolicyConfig.from_config({"policy": {"max_regrasps": 1, "nope": 1}})
    assert HandPolicyConfig.from_config({"policy": {"max_regrasps": 0}}).max_regrasps == 0
