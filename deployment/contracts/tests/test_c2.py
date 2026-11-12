import numpy as np
import pytest

from contracts import ArmState, GateCounters, decode, encode
from contracts.c2_state import MODES
from helpers import contract_error

POSE = np.array([[0.6, 0.8, 0.0, 0.4], [0.8, -0.6, 0.0, 0.1], [0.0, 0.0, -1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])


def gate(accepted=10, shortened=1, pulled_back=2, rejected=0):
    return GateCounters(accepted=accepted, shortened=shortened, pulled_back=pulled_back, rejected=rejected)


def state(**change):
    d = dict(t_ns=10, seq=1, O_T_TCP=POSE.copy(), O_T_TCP_target=POSE.copy(), q=np.zeros(7), dq=np.zeros(7),
             gripper_width_m=0.04, mode="running", error="", gate=gate(), loop_late=3)
    d.update(change)
    return ArmState(**d)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode(mode):
    assert decode(encode(state(mode=mode))).mode == mode


def test_modes_are_the_six():
    assert set(MODES) == {"idle", "running", "holding", "stopped", "homing", "fault"}
    contract_error(lambda: state(mode="moving"), "mode", "value")


def test_loop_late_and_pulled_back_carried():
    s = decode(encode(state(loop_late=17, gate=gate(pulled_back=4))))
    assert s.loop_late == 17 and s.gate.pulled_back == 4
    contract_error(lambda: state(loop_late=-1), "loop_late", "value")


def test_gate_bounds():
    contract_error(lambda: state(gate=gate(accepted=3, shortened=4)), "gate.shortened", "value")
    contract_error(lambda: state(gate=gate(accepted=3, pulled_back=4)), "gate.pulled_back", "value")
    state(gate=gate(accepted=3, shortened=3, pulled_back=3, rejected=100))  # rejected is not bounded by accepted


def test_transposed_pose_refused():
    e = contract_error(lambda: state(O_T_TCP=POSE.T.copy()), "O_T_TCP", "value")
    assert "bottom row" in str(e)
    contract_error(lambda: state(O_T_TCP_target=np.ascontiguousarray(POSE.T)), "O_T_TCP_target", "value")


def test_rigid_checks():
    bent = POSE.copy()
    bent[0, 1] = 0.81
    contract_error(lambda: state(O_T_TCP=bent), "O_T_TCP", "value")
    mirror = POSE.copy()
    mirror[2, 2] = 1.0  # orthonormal but a reflection
    contract_error(lambda: state(O_T_TCP=mirror), "O_T_TCP", "value")
    nearly = POSE.copy()
    nearly[0, 0] += 1e-9  # within rotation_tol
    state(O_T_TCP=nearly)
    bottom = POSE.copy()
    bottom[3, 3] = 1.0 + 1e-12  # the bottom row must be exact
    contract_error(lambda: state(O_T_TCP=bottom), "O_T_TCP", "value")


@pytest.mark.parametrize("field", ["q", "dq"])
def test_non_finite_refused(field):
    a = np.zeros(7)
    a[3] = np.inf
    contract_error(lambda: state(**{field: a}), field, "value")


def test_width():
    contract_error(lambda: state(gripper_width_m=-0.001), "gripper_width_m", "value")
    contract_error(lambda: state(gripper_width_m=float("inf")), "gripper_width_m", "value")
    contract_error(lambda: state(gripper_width_m=float("nan")), "gripper_width_m", "value")
