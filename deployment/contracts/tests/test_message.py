from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from contracts import (Ack, ArmState, ContractError, FramesReply, GateCounters, Observation, SetWaypoints, decode,
                       encode)
from contracts.examples import observation
from helpers import contract_error


def pose():
    return np.eye(4)


def state(**change):
    d = dict(t_ns=1, seq=2, O_T_TCP=pose(), O_T_TCP_target=pose(), q=np.zeros(7), dq=np.zeros(7),
             gripper_width_m=0.0, mode="idle", error="", gate=GateCounters(accepted=0, shortened=0, pulled_back=0,
                                                                            rejected=0), loop_late=0)
    d.update(change)
    return ArmState(**d)


def test_building_validates():
    contract_error(lambda: state(mode="flying"), "mode", "value")
    contract_error(lambda: state(q=np.zeros(6)), "q", "shape")


def test_to_dict_order_is_field_order_with_schema_first():
    d = state().to_dict()
    assert list(d) == ["schema"] + [f.name for f in dataclasses.fields(ArmState)]
    assert d["schema"] == "c2.state/v1"
    assert isinstance(d["gate"], dict) and list(d["gate"]) == ["accepted", "shortened", "pulled_back", "rejected"]
    assert isinstance(d["q"], np.ndarray)


def test_from_dict_round_trip():
    s = state(seq=5)
    assert ArmState.from_dict(s.to_dict()) == s


def test_from_dict_refuses_missing_and_unknown_keys():
    d = state().to_dict()
    del d["error"]
    contract_error(lambda: ArmState.from_dict(d), "error", "missing")
    d = state().to_dict()
    d["colour"] = "red"
    contract_error(lambda: ArmState.from_dict(d), "colour", "unknown_field")


def test_from_dict_checks_schema():
    d = state().to_dict()
    del d["schema"]
    contract_error(lambda: ArmState.from_dict(d), "", "no_schema")
    contract_error(lambda: ArmState.from_dict({**state().to_dict(), "schema": "c2.state/v3"}), "", "schema_version")
    contract_error(lambda: ArmState.from_dict(Ack(ok=True, reason="").to_dict()), "", "unexpected_message")


def test_frozen():
    a = Ack(ok=True, reason="")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.ok = False  # type: ignore[misc]


def test_float64_array_is_not_cast_to_float32():
    obs = observation()
    with pytest.raises(ContractError) as info:
        dataclasses.replace(obs, eef_pos=obs.eef_pos.astype(np.float64))
    assert (info.value.field, info.value.problem) == ("eef_pos", "dtype")
    with pytest.raises(ContractError):
        Observation(**{**{f.name: getattr(obs, f.name) for f in dataclasses.fields(Observation)},
                       "gripper_width_m": np.array([[0.05], [0.05]])})


def test_equality_by_content():
    assert Ack(ok=True, reason="") == Ack(ok=True, reason="")
    assert Ack(ok=True, reason="") != Ack(ok=False, reason="no")
    assert state() == decode(encode(state()))


def test_first_failure_in_field_order():
    # two problems: mode (field 8) and loop_late (field 11): the first one in table order is reported
    d = state().to_dict()
    d["mode"] = "flying"
    d["loop_late"] = -1
    contract_error(lambda: ArmState.from_dict(d), "mode", "value")


def test_optional_none_message():
    from contracts.examples import _health
    r = FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None, health=_health(main="down"))
    assert decode(encode(r)) == r


def test_waypoint_arrays_not_copied():
    t = np.array([1, 2], dtype=np.int64)
    w = SetWaypoints(chunk_id=1, t_ns=t, O_T_TCP=np.stack([np.eye(4)] * 2), gripper_width_m=np.zeros(2))
    assert w.t_ns is t
