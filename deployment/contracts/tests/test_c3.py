# adapted from rnoro/deployment:deployment/contracts/tests/test_c3.py @ 91b178b
# changes: imports renamed; + Hello, HelloReply (every rule of STRUCTURE.md 3.8) and the home stopped: prefix
import numpy as np
import pytest

from deploy_contracts import (HOME_STOPPED_PREFIX, SHARED_SCHEMAS, Ack, ContractLimits, GetState, Heartbeat, Hello,
                              HelloReply, Home, SetWaypoints, SetWaypointsReply, Start, Stop, WaypointIssue, decode,
                              encode, home_was_stopped)
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


@pytest.mark.parametrize("cls", [Heartbeat, Start, Stop, Home, GetState, Hello])
def test_commands_without_fields(cls):
    assert decode(encode(cls())) == cls()
    assert list(cls().to_dict()) == ["schema"]


def test_replies_are_ack():
    assert decode(encode(Ack(ok=False, reason="refused: mode is holding")), Ack).ok is False


def test_home_stopped_prefix():
    assert HOME_STOPPED_PREFIX == "stopped:" and len(HOME_STOPPED_PREFIX) == 8
    assert home_was_stopped(Ack(ok=False, reason="stopped: s key pressed during homing"))
    assert not home_was_stopped(Ack(ok=True, reason="stopped: never with ok"))
    assert not home_was_stopped(Ack(ok=False, reason="refused: mode is holding"))
    assert not home_was_stopped(Ack(ok=False, reason="homing timed out after 30 s"))
    assert not home_was_stopped(Ack(ok=False, reason="stopped:no space"))
    assert not home_was_stopped(Ack(ok=False, reason="Stopped: wrong case"))


# ------------------------------------------------------------------------------------------------ hello

F_T_TCP = np.array([[0.6, 0.8, 0.0, 0.0], [-0.8, 0.6, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15], [0.0, 0.0, 0.0, 1.0]])


def reply(**change):
    d = dict(t_pc_ns=5_000_000_000, commit="0123456789abcdef0123456789abcdef01234567", mock=False,
             schemas=SHARED_SCHEMAS, F_T_TCP=F_T_TCP.copy(), rotation_tol=1.0e-6, gripper_width_max_m=0.08,
             max_waypoints=64, watchdog_limit_s=1.0, homing_timeout_s=30.0, max_lead_s=1.5)
    d.update(change)
    return HelloReply(**d)


def test_hello_has_no_fields():
    assert Hello().to_dict() == {"schema": "c3.hello/v1"}


def test_hello_reply_round_trip():
    r = reply()
    back = decode(encode(r), HelloReply)
    assert back == r and back.schemas == SHARED_SCHEMAS and back.F_T_TCP.dtype == np.float64
    assert decode(encode(reply(mock=True, commit="unknown"))).commit == "unknown"
    assert reply(t_pc_ns=-5).t_pc_ns == -5  # int64: any value


@pytest.mark.parametrize("commit", ["0123456", "0123456789ABCDEF0123456789ABCDEF01234567",
                                    "0123456789abcdef0123456789abcdef012345678", "", "Unknown", "unknown ", "unknown\n",
                                    "g123456789abcdef0123456789abcdef01234567"])
def test_commit_rule(commit):
    contract_error(lambda: reply(commit=commit), "commit", "value")


def test_commit_type():
    contract_error(lambda: reply(commit=None), "commit", "type")


def test_schemas_each_well_formed():
    s = list(SHARED_SCHEMAS)
    s[2] = "c3.Get_State/v1"
    contract_error(lambda: reply(schemas=s), "schemas[2]", "value")
    s = list(SHARED_SCHEMAS)
    s[3] = 7
    contract_error(lambda: reply(schemas=s), "schemas[3]", "type")
    contract_error(lambda: reply(schemas="ack/v1"), "schemas", "type")


def test_schemas_strictly_increasing_by_byte_order():
    s = list(SHARED_SCHEMAS)
    s[3], s[4] = s[4], s[3]
    contract_error(lambda: reply(schemas=s), "schemas", "value")
    contract_error(lambda: reply(schemas=["ack/v1", "ack/v1"]), "schemas", "value")
    # "c3.hello.reply/v1" sorts before "c3.hello/v1" ('.' 0x2e < '/' 0x2f): whole strings, not names
    reply(schemas=["c3.hello.reply/v1", "c3.hello/v1"])
    contract_error(lambda: reply(schemas=["c3.hello/v1", "c3.hello.reply/v1"]), "schemas", "value")
    reply(schemas=[])  # the contract allows any increasing list; compare_hello judges the content


def test_check_order_commit_then_schemas_then_frame_then_floats():
    s = list(SHARED_SCHEMAS)[::-1]
    contract_error(lambda: reply(commit="x", schemas=s, F_T_TCP=F_T_TCP.T.copy(), max_lead_s=0.0), "commit", "value")
    contract_error(lambda: reply(schemas=s, F_T_TCP=F_T_TCP.T.copy(), max_lead_s=0.0), "schemas", "value")
    contract_error(lambda: reply(F_T_TCP=F_T_TCP.T.copy(), max_lead_s=0.0), "F_T_TCP", "value")
    contract_error(lambda: reply(rotation_tol=0.0, max_lead_s=0.0), "rotation_tol", "value")


def test_frame_rigid_with_the_receivers_rotation_tol():
    tol = ContractLimits.load().rotation_tol
    near = F_T_TCP.copy()
    near[0, 0] += tol / 10
    reply(F_T_TCP=near)
    bent = F_T_TCP.copy()
    bent[0, 0] += tol * 10
    contract_error(lambda: reply(F_T_TCP=bent), "F_T_TCP", "value")
    mirror = F_T_TCP.copy()
    mirror[2, 2] = -1.0
    contract_error(lambda: reply(F_T_TCP=mirror), "F_T_TCP", "value")
    nan = F_T_TCP.copy()
    nan[1, 3] = np.nan
    contract_error(lambda: reply(F_T_TCP=nan), "F_T_TCP", "value")
    contract_error(lambda: reply(F_T_TCP=F_T_TCP[:3].copy()), "F_T_TCP", "shape")
    contract_error(lambda: reply(F_T_TCP=F_T_TCP.astype(np.float32)), "F_T_TCP", "dtype")


@pytest.mark.parametrize("name", ["rotation_tol", "gripper_width_max_m", "watchdog_limit_s", "homing_timeout_s",
                                  "max_lead_s"])
@pytest.mark.parametrize("value,problem", [(0.0, "value"), (-1.0, "value"), (float("inf"), "value"),
                                           (float("nan"), "value"), (1, "type")])
def test_positive_floats(name, value, problem):
    contract_error(lambda: reply(**{name: value}), name, problem)


def test_max_waypoints():
    contract_error(lambda: reply(max_waypoints=0), "max_waypoints", "value")
    contract_error(lambda: reply(max_waypoints=-1), "max_waypoints", "value")
    contract_error(lambda: reply(max_waypoints=64.0), "max_waypoints", "type")
    assert reply(max_waypoints=1).max_waypoints == 1


def test_other_types():
    contract_error(lambda: reply(t_pc_ns=5.0), "t_pc_ns", "type")
    contract_error(lambda: reply(mock=1), "mock", "type")
