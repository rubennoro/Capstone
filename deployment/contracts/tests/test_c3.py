import numpy as np
import pytest

from contracts import (Ack, ContractLimits, GetState, Heartbeat, Home, SetWaypoints, SetWaypointsReply, Start, Stop,
                       WaypointIssue, decode, encode)
from helpers import contract_error

POSE = np.array([[0.6, 0.8, 0.0, 0.4], [0.8, -0.6, 0.0, 0.1], [0.0, 0.0, -1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])


def waypoints(n=4, **change):
    d = dict(chunk_id=3, t_ns=np.arange(n, dtype=np.int64) * 50_000_000 + 1_000,
             O_T_TCP=np.stack([POSE] * n) if n else np.zeros((0, 4, 4)),
             gripper_width_m=np.full(n, 0.03))
    d.update(change)
    return SetWaypoints(**d)


def test_good_waypoints_round_trip():
    w = waypoints()
    assert decode(encode(w)) == w
    assert decode(encode(waypoints(1))).t_ns.shape == (1,)


def test_count_limits():
    top = ContractLimits.load().max_waypoints
    assert decode(encode(waypoints(top))).O_T_TCP.shape == (top, 4, 4)
    contract_error(lambda: waypoints(top + 1), "t_ns", "value")
    contract_error(lambda: waypoints(0), "t_ns", "value")


def test_lengths_agree():
    contract_error(lambda: waypoints(O_T_TCP=np.stack([POSE] * 3)), "O_T_TCP", "value")
    contract_error(lambda: waypoints(gripper_width_m=np.full(5, 0.03)), "gripper_width_m", "value")


@pytest.mark.parametrize("t", [[1, 2, 2, 3], [1, 3, 2, 4], [4, 3, 2, 1]])
def test_times_strictly_increasing(t):
    contract_error(lambda: waypoints(t_ns=np.array(t, np.int64)), "t_ns", "value")


def test_every_pose_rigid():
    poses = np.stack([POSE] * 4)
    poses[2] = POSE.T
    e = contract_error(lambda: waypoints(O_T_TCP=poses), "O_T_TCP", "value")
    assert "at [2]" in str(e)
    poses = np.stack([POSE] * 4)
    poses[1, 0, 3] = np.nan
    contract_error(lambda: waypoints(O_T_TCP=poses), "O_T_TCP", "value")


def test_widths():
    contract_error(lambda: waypoints(gripper_width_m=np.array([0.01, -0.01, 0.0, 0.0])), "gripper_width_m", "value")
    contract_error(lambda: waypoints(gripper_width_m=np.array([0.01, np.nan, 0.0, 0.0])), "gripper_width_m",
                   "value")
    contract_error(lambda: waypoints(gripper_width_m=np.full(4, 0.03, np.float32)), "gripper_width_m", "dtype")
    waypoints(gripper_width_m=np.full(4, 50.0))  # a wide width is the gate's question, not the contract's


def test_reply():
    r = SetWaypointsReply(chunk_id=3, accepted=4, shortened=4, skipped=2,
                          rejected=[WaypointIssue(index=5, reason="x out of the box")])
    back = decode(encode(r))
    assert back == r and back.rejected[0].reason == "x out of the box" and back.skipped == 2
    contract_error(lambda: SetWaypointsReply(chunk_id=3, accepted=1, shortened=2, skipped=0, rejected=()),
                   "shortened", "value")
    contract_error(lambda: SetWaypointsReply(chunk_id=3, accepted=0, shortened=0, skipped=-1, rejected=()),
                   "skipped", "value")
    contract_error(lambda: SetWaypointsReply(chunk_id=3, accepted=0, shortened=0, skipped=0, rejected=(
        WaypointIssue(index=1, reason="a"), WaypointIssue(index=1, reason="b"))), "rejected[1].index", "value")


@pytest.mark.parametrize("cls", [Heartbeat, Start, Stop, Home, GetState])
def test_commands_without_fields(cls):
    assert decode(encode(cls())) == cls()
    assert list(cls().to_dict()) == ["schema"]


def test_replies_are_ack():
    assert decode(encode(Ack(ok=False, reason="refused: mode is holding")), Ack).ok is False
