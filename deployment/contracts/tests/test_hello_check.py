"""compare_hello (STRUCTURE.md 3.9)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from deploy_contracts import (SHARED_SCHEMAS, ContractLimits, HelloMismatch, HelloReply, LinkSettings, TcpFrame,
                              compare_hello)


def equal_reply(**change) -> HelloReply:
    """A reply that matches this checkout's contract files exactly."""
    limits, frame, link = ContractLimits.load(), TcpFrame.load(), LinkSettings.load()
    d = dict(t_pc_ns=1, commit="unknown", mock=False, schemas=SHARED_SCHEMAS, F_T_TCP=np.array(frame.F_T_TCP),
             rotation_tol=limits.rotation_tol, gripper_width_max_m=limits.gripper_width_max_m,
             max_waypoints=limits.max_waypoints, watchdog_limit_s=link.watchdog_limit_s, homing_timeout_s=30.0,
             max_lead_s=1.5)
    d.update(change)
    return HelloReply(**d)


def test_equal_reply_gives_no_mismatch():
    assert compare_hello(equal_reply(), mock=False) == []
    assert compare_hello(equal_reply(mock=True), mock=True) == []


def test_fields_the_check_ignores():
    # commit, t_pc_ns, homing_timeout_s, max_lead_s are the laptop's own checks, not equality
    r = equal_reply(t_pc_ns=-7, commit="0123456789abcdef0123456789abcdef01234567", homing_timeout_s=5.0,
                    max_lead_s=0.1)
    assert compare_hello(r, mock=False) == []


def _fields(mismatches: list[HelloMismatch]) -> list[str]:
    return [m.field for m in mismatches]


def test_mock_differs():
    out = compare_hello(equal_reply(mock=True), mock=False)
    assert out == [HelloMismatch("mock", "true", "false (this side's mock flag)")]
    assert _fields(compare_hello(equal_reply(), mock=True)) == ["mock"]


@pytest.mark.parametrize("name,value", [("rotation_tol", 2.0e-6), ("gripper_width_max_m", 0.085),
                                        ("max_waypoints", 63), ("watchdog_limit_s", 0.5)])
def test_each_value_differing_gives_exactly_that_field(name, value):
    out = compare_hello(equal_reply(**{name: value}), mock=False)
    assert _fields(out) == [name]
    assert out[0].pc == (str(value) if name == "max_waypoints" else repr(value))
    source = "link.yaml timing.watchdog_limit_s" if name == "watchdog_limit_s" else f"limits.yaml {name}"
    assert out[0].expected.endswith(f"({source})")


def test_watchdog_expected_text():
    out = compare_hello(equal_reply(watchdog_limit_s=0.5), mock=False)
    assert out[0].expected == "1.0 (link.yaml timing.watchdog_limit_s)"
    assert str(out[0]) == "watchdog_limit_s: PC 0.5, expected 1.0 (link.yaml timing.watchdog_limit_s)"


def test_scalars_relative_tolerance():
    limits = ContractLimits.load()
    near = limits.gripper_width_max_m * (1 + 5e-10)
    assert compare_hello(equal_reply(gripper_width_max_m=near), mock=False) == []
    far = limits.gripper_width_max_m * (1 + 5e-9)
    assert _fields(compare_hello(equal_reply(gripper_width_max_m=far), mock=False)) == ["gripper_width_max_m"]


def test_frame_within_1e9_passes_1e6_off_fails():
    base = np.array(TcpFrame.load().F_T_TCP)
    near = base.copy()
    near[0, 3] += 0.5e-9
    near[1, 0] -= 0.5e-9
    assert compare_hello(equal_reply(F_T_TCP=near), mock=False) == []
    off = base.copy()
    off[2, 3] += 1e-6
    out = compare_hello(equal_reply(F_T_TCP=off), mock=False)
    assert _fields(out) == ["F_T_TCP"]
    assert "frames.yaml F_T_TCP" in out[0].expected


def test_schemas_list_only_the_differing_entries():
    s = list(SHARED_SCHEMAS)
    s[s.index("c3.hello/v1")] = "c3.hello/v2"
    out = compare_hello(equal_reply(schemas=sorted(s)), mock=False)
    assert out == [HelloMismatch("schemas", "c3.hello/v2", "c3.hello/v1 (deploy_contracts SHARED_SCHEMAS)")]
    extra = sorted(list(SHARED_SCHEMAS) + ["c1.frames/v1", "c4.chunk/v1"])
    out = compare_hello(equal_reply(schemas=extra), mock=False)
    assert out == [HelloMismatch("schemas", "c1.frames/v1, c4.chunk/v1", "(none) (deploy_contracts SHARED_SCHEMAS)")]
    fewer = [x for x in SHARED_SCHEMAS if x != "c3.hello.reply/v1"]
    out = compare_hello(equal_reply(schemas=fewer), mock=False)
    assert (out[0].pc, out[0].expected) == ("(none)", "c3.hello.reply/v1 (deploy_contracts SHARED_SCHEMAS)")


def test_every_mismatch_in_table_order():
    base = np.array(TcpFrame.load().F_T_TCP)
    off = base.copy()
    off[0, 3] += 0.01
    r = equal_reply(schemas=["ack/v1"], mock=True, F_T_TCP=off, rotation_tol=1e-5, gripper_width_max_m=0.1,
                    max_waypoints=8, watchdog_limit_s=2.0)
    assert _fields(compare_hello(r, mock=False)) == ["schemas", "mock", "F_T_TCP", "rotation_tol",
                                                     "gripper_width_max_m", "max_waypoints", "watchdog_limit_s"]


def test_explicit_objects_replace_the_files():
    limits = dataclasses.replace(ContractLimits.load(), max_waypoints=8)
    link = dataclasses.replace(LinkSettings.load(), watchdog_limit_s=0.5)
    frame = TcpFrame(F_T_TCP=np.eye(4), settled=True)
    out = compare_hello(equal_reply(), mock=False, limits=limits, frame=frame, link=link)
    assert _fields(out) == ["F_T_TCP", "max_waypoints", "watchdog_limit_s"]
    r = equal_reply(F_T_TCP=np.eye(4), max_waypoints=8, watchdog_limit_s=0.5)
    assert compare_hello(r, mock=False, limits=limits, frame=frame, link=link) == []
