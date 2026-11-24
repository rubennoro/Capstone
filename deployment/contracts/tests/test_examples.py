# adapted from rnoro/deployment:deployment/contracts/tests/test_examples.py @ 91b178b
# changes: imports renamed; only C2/C3 (all stored); + the STRUCTURE.md 5.2 list by name, hello examples, the public
# helpers, encode(decode(b)) == b for every stored good example.
from __future__ import annotations

import pytest

from deploy_contracts import (SHARED_SCHEMAS, ContractError, Heartbeat, broken_example, decode, drop_key, encode,
                              examples_dir, good_example, load_index, raw_example, set_key)
from deploy_contracts.examples import INDEX_FILE, build_examples, resolve_path
from deploy_contracts.make_examples import main, write_examples

ALL = build_examples()
STORED = load_index()


def _check(example):
    if example.valid:
        msg = decode(example.data)
        assert msg.SCHEMA == example.schema
        d = msg.to_dict()
        for path, want in example.values.items():
            assert resolve_path(d, path) == want, path
    else:
        with pytest.raises(ContractError) as info:
            decode(example.data)
        assert (info.value.field, info.value.problem) == (example.field, example.problem), str(info.value)
        if example.schema != "?":
            assert info.value.schema == example.schema


@pytest.mark.parametrize("example", ALL, ids=lambda e: e.name)
def test_built_examples(example):
    _check(example)


@pytest.mark.parametrize("example", STORED, ids=lambda e: e.name)
def test_stored_examples(example):
    _check(example)


@pytest.mark.parametrize("example", [e for e in STORED if e.valid], ids=lambda e: e.name)
def test_stored_good_examples_are_canonical(example):
    assert encode(decode(example.data)) == example.data


def test_every_shared_message_has_a_good_example():
    assert {e.schema for e in ALL if e.valid} == set(SHARED_SCHEMAS)


@pytest.mark.parametrize("contract", ["c2", "c3"])
def test_every_contract_has_the_broken_kinds(contract):
    problems = {e.problem for e in ALL if e.contract == contract and not e.valid}
    assert {"type", "value", "missing", "unknown_field", "schema_version", "shape", "dtype"} <= problems


def test_every_problem_code_is_covered_except_unexpected_message():
    from deploy_contracts import PROBLEMS
    assert {e.problem for e in ALL if not e.valid} == set(PROBLEMS) - {"unexpected_message"}


C2_GOOD = ["state_running", "state_holding", "state_homing", "state_fault", "state_idle_zero", "state_stopped"]
C2_BROKEN = {"bad_pose_transposed": ("O_T_TCP", "value"), "bad_pose_not_orthonormal": ("O_T_TCP", "value"),
             "bad_mode": ("mode", "value"), "missing_loop_late": ("loop_late", "missing"),
             "missing_gate_pulled_back": ("gate.pulled_back", "missing"), "wrong_shape_q": ("q", "shape"),
             "wrong_dtype_q": ("q", "dtype"), "bad_width_negative": ("gripper_width_m", "value"),
             "bad_width_int": ("gripper_width_m", "type"), "bad_gate_shortened": ("gate.shortened", "value"),
             "bad_dq_nan": ("dq", "value"), "bad_type_seq": ("seq", "type"), "bad_seq_negative": ("seq", "value"),
             "bad_array_byte_count": ("O_T_TCP", "shape"), "unknown_field": ("temperature", "unknown_field"),
             "schema_version": ("", "schema_version")}
C3_GOOD = ["set_waypoints_8", "set_waypoints_1", "heartbeat", "start", "stop", "home", "get_state",
           "set_waypoints_reply_ok", "set_waypoints_reply_rejected", "ack_ok", "ack_refused",
           "error_set_waypoints_refused", "error_contract", "hello", "hello_reply", "hello_reply_mock"]
C3_BROKEN = [f"set_waypoints_{n}" for n in (
    "times_not_increasing", "lengths_differ", "pose_not_rigid", "pose_nan", "too_many", "empty", "width_negative",
    "width_nan", "wrong_shape", "wrong_dtype_t", "wrong_dtype_width", "bad_array_map", "missing_width",
    "unknown_field", "type_chunk_id", "chunk_id_negative", "schema_version")] + [
    "heartbeat_unknown_field", "unknown_schema", "no_schema", "not_a_map", "not_msgpack", "trailing_bytes",
    "set_waypoints_reply_shortened_gt_accepted", "set_waypoints_reply_duplicate_index",
    "set_waypoints_reply_missing_skipped", "set_waypoints_reply_missing_reason", "ack_ok_not_bool",
    "ack_missing_reason", "ack_schema_version", "error_missing_field"]
HELLO_BROKEN = {"hello_unknown_field": ("commit", "unknown_field"), "hello_schema_version": ("", "schema_version"),
                "hello_reply_missing_max_lead_s": ("max_lead_s", "missing"),
                "hello_reply_t_pc_ns_float": ("t_pc_ns", "type"), "hello_reply_mock_int": ("mock", "type"),
                "hello_reply_commit_short": ("commit", "value"), "hello_reply_schemas_unsorted": ("schemas", "value"),
                "hello_reply_schemas_item_int": ("schemas[3]", "type"),
                "hello_reply_F_T_TCP_transposed": ("F_T_TCP", "value"),
                "hello_reply_F_T_TCP_shape": ("F_T_TCP", "shape"),
                "hello_reply_watchdog_zero": ("watchdog_limit_s", "value"),
                "hello_reply_max_waypoints_zero": ("max_waypoints", "value"),
                "hello_reply_schema_version": ("", "schema_version")}


def test_the_list_of_structure_5_2_exactly():
    want_good = {f"c2/{n}" for n in C2_GOOD} | {f"c3/{n}" for n in C3_GOOD}
    want_broken = ({f"c2/{n}" for n in C2_BROKEN} | {f"c3/{n}" for n in C3_BROKEN} |
                   {f"c3/{n}" for n in HELLO_BROKEN})
    assert {e.name for e in ALL if e.valid} == want_good
    assert {e.name for e in ALL if not e.valid} == want_broken
    by_name = {e.name: e for e in ALL}
    for name, (field, problem) in C2_BROKEN.items():
        assert (by_name[f"c2/{name}"].field, by_name[f"c2/{name}"].problem) == (field, problem), name
    for name, (field, problem) in HELLO_BROKEN.items():
        assert (by_name[f"c3/{name}"].field, by_name[f"c3/{name}"].problem) == (field, problem), name


def test_hello_reply_values():
    e = next(e for e in STORED if e.name == "c3/hello_reply")
    assert e.values == {"t_pc_ns": 123461789000000, "commit": "0123456789abcdef0123456789abcdef01234567",
                        "mock": False, "rotation_tol": 1.0e-6, "gripper_width_max_m": 0.08, "max_waypoints": 64,
                        "watchdog_limit_s": 1.0, "homing_timeout_s": 30.0, "max_lead_s": 1.5,
                        "F_T_TCP[0][1]": 0.8, "F_T_TCP[2][3]": 0.15}
    msg = decode(e.data)
    assert msg.schemas == SHARED_SCHEMAS
    assert msg.F_T_TCP.tolist() == [[0.6, 0.8, 0.0, 0.0], [-0.8, 0.6, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15],
                                    [0.0, 0.0, 0.0, 1.0]]
    mock = decode(next(e for e in STORED if e.name == "c3/hello_reply_mock").data)
    assert (mock.mock, mock.commit) == (True, "unknown")


def test_named_cases_present():
    homing = next(e for e in ALL if e.name == "c2/state_homing")
    assert homing.values["loop_late"] > 0 and homing.values["gate.pulled_back"] > 0
    stopped = next(e for e in ALL if e.name == "c2/state_stopped")
    assert stopped.values["mode"] == "stopped"


def test_index_lists_exactly_the_files():
    root = examples_dir()
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {INDEX_FILE}
    assert on_disk == {e.file for e in STORED}


def test_everything_built_is_stored_and_only_c2_c3():
    assert {p.name for p in examples_dir().iterdir()} == {"c2", "c3", INDEX_FILE}
    assert all(e.stored for e in ALL)
    for e in STORED:
        assert e.contract in ("c2", "c3")
        assert e.schema.split("/")[0].split(".")[0] in ("c2", "c3", "ack", "error", "?"), e.schema
    assert [e.name for e in STORED] == [e.name for e in ALL]


def test_folder_equals_a_fresh_make_examples(tmp_path):
    write_examples(tmp_path)
    root = examples_dir()

    def files(d):
        return {p.relative_to(d).as_posix(): p.read_bytes() for p in d.rglob("*") if p.is_file()}

    fresh, committed = files(tmp_path), files(root)
    assert sorted(fresh) == sorted(committed), "stale examples: run python -m deploy_contracts.make_examples"
    for name in fresh:
        assert fresh[name] == committed[name], f"{name} is stale: run python -m deploy_contracts.make_examples"


def test_make_examples_removes_stale_files(tmp_path, capsys):
    (tmp_path / "c2").mkdir()
    (tmp_path / "c2" / "old.msgpack").write_bytes(b"x")
    assert main(["--out", str(tmp_path)]) == 0
    assert not (tmp_path / "c2" / "old.msgpack").exists()
    assert "examples" in capsys.readouterr().out
    assert len(load_index(tmp_path)) == len(ALL)


def test_public_helpers_for_in_memory_examples():
    # what policy_deploy uses for its C0/C1/C4 examples (never stored)
    good = good_example("c1/x", Heartbeat(), note="n")
    assert good.valid and good.data == encode(Heartbeat()) and good.contract == "c1" and not good.stored
    bad = broken_example("c1/y", Heartbeat(), set_key("extra", 1), "extra", "unknown_field", "n")
    _check(bad)
    dropped = broken_example("c1/z", Heartbeat(), drop_key("schema"), "", "no_schema", "n", schema="?")
    _check(dropped)
    raw = raw_example("c1/w", b"\xc1", "?", "", "not_msgpack", "n")
    _check(raw)
    assert resolve_path({"a": [{"b": 3}]}, "a[0].b") == 3
