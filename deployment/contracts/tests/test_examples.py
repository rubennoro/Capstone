from __future__ import annotations

import pytest

from contracts import ContractError, decode, examples_dir, load_index
from contracts.examples import INDEX_FILE, build_examples, resolve_path
from contracts.make_examples import main, write_examples
from contracts.registry import registered

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


@pytest.mark.parametrize("example", ALL, ids=lambda e: e.name)
def test_built_examples(example):
    _check(example)


@pytest.mark.parametrize("example", STORED, ids=lambda e: e.name)
def test_stored_examples(example):
    _check(example)


def test_every_message_has_a_good_example():
    good = {e.schema for e in ALL if e.valid}
    assert {c.SCHEMA for c in registered()} <= good


@pytest.mark.parametrize("contract", ["c0", "c1", "c2", "c3", "c4"])
def test_every_contract_has_the_broken_kinds(contract):
    problems = {e.problem for e in ALL if e.contract == contract and not e.valid}
    wanted = {"type", "value", "missing", "unknown_field", "schema_version"}
    if contract != "c0":  # C0 has no arrays
        wanted |= {"shape", "dtype"}
    assert wanted <= problems


def test_named_cases_present():
    names = {e.name for e in ALL}
    for name in ["c1/frames_main_stale", "c1/frames_images_while_down", "c2/state_running", "c2/state_holding",
                 "c2/state_homing", "c2/bad_pose_transposed", "c2/bad_mode", "c2/missing_loop_late",
                 "c2/missing_gate_pulled_back", "c3/set_waypoints_times_not_increasing",
                 "c3/set_waypoints_lengths_differ", "c3/set_waypoints_pose_not_rigid",
                 "c3/set_waypoints_reply_shortened_gt_accepted", "c3/error_set_waypoints_refused", "c3/ack_ok",
                 "c4/chunk_nan", "c4/chunk_rot6d_not_unit", "c4/chunk_rot6d_not_perpendicular",
                 "c4/chunk_wrong_shape", "c4/chunk_width_clipped_missing", "c4/chunk_width_clipped_too_big",
                 "c4/chunk_odd_step0_5cm"]:
        assert name in names
    homing = next(e for e in ALL if e.name == "c2/state_homing")
    assert homing.values["loop_late"] > 0 and homing.values["gate.pulled_back"] > 0


def test_index_lists_exactly_the_files():
    root = examples_dir()
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {INDEX_FILE}
    assert on_disk == {e.file for e in STORED}


def test_only_c2_c3_and_their_replies_are_stored():
    assert {p.name for p in examples_dir().iterdir()} == {"c2", "c3", INDEX_FILE}
    for e in STORED:
        assert e.contract in ("c2", "c3")
        assert e.schema.split("/")[0].split(".")[0] in ("c2", "c3", "ack", "error", "?"), e.schema
    assert {e.name for e in STORED} == {e.name for e in ALL if e.stored}
    assert not any(e.contract in ("c0", "c1", "c4") for e in STORED)


def test_folder_equals_a_fresh_make_examples(tmp_path):
    write_examples(tmp_path)
    root = examples_dir()

    def files(d):
        return {p.relative_to(d).as_posix(): p.read_bytes() for p in d.rglob("*") if p.is_file()}

    fresh, committed = files(tmp_path), files(root)
    assert sorted(fresh) == sorted(committed), "stale examples: run python -m contracts.make_examples"
    for name in fresh:
        assert fresh[name] == committed[name], f"{name} is stale: run python -m contracts.make_examples"


def test_make_examples_removes_stale_files(tmp_path, capsys):
    (tmp_path / "c2").mkdir()
    (tmp_path / "c2" / "old.msgpack").write_bytes(b"x")
    assert main(["--out", str(tmp_path)]) == 0
    assert not (tmp_path / "c2" / "old.msgpack").exists()
    assert "examples" in capsys.readouterr().out
