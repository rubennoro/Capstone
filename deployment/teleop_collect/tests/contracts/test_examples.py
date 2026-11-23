# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/contracts/test_examples.py @ 2ffe641
# changes: the R1/U1/U2 examples (STRUCTURE.md 4.5) instead of C0/C1/C4; the pi_stream and c4 cases dropped (the
# pi_stream wire cases are in test_pi_stream.py); + the broken cases STRUCTURE.md 4.5 asks for. + start pose
# (2026-11-20): no u1.home/v1.
from __future__ import annotations

import pytest
from deploy_contracts import SHARED_SCHEMAS, ContractError, decode, examples_dir, resolve_path
from deploy_contracts.registry import registered

import teleop_collect.contracts  # noqa: F401  (registers R1/U1/U2)
from teleop_collect.contracts.examples import build_examples

ALL = build_examples()
NEW = ("r1", "u1", "u2")


@pytest.mark.parametrize("example", ALL, ids=lambda e: e.name)
def test_built_examples(example):
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


def test_every_new_message_has_a_good_example():
    good = {e.schema for e in ALL if e.valid}
    new = {c.SCHEMA for c in registered()} - set(SHARED_SCHEMAS)
    new = {s for s in new if s.split(".")[0] in NEW}
    assert len(new) == 20       # u1.home/v1 gone (2026-11-20); + label_episode, delete_task, set_data_root (2026-11-22)
    assert new <= good, new - good


def test_every_schema_of_the_structure_is_registered():
    names = {c.SCHEMA for c in registered()}
    assert {"r1.start_recording/v1", "r1.stop_recording/v1", "r1.get_health/v1", "r1.summary/v1", "r1.health/v1",
            "u2.get_preview/v1", "u2.preview/v1",
            "u1.get_status/v1", "u1.keys/v1", "u1.gripper/v1", "u1.select_task/v1", "u1.create_task/v1",
            "u1.start_episode/v1", "u1.stop_episode/v1", "u1.delete_episode/v1", "u1.stop_arm/v1",
            "u1.label_episode/v1", "u1.delete_task/v1", "u1.set_data_root/v1", "u1.status/v1"} <= names
    assert "u1.home/v1" not in names                                    # no laptop home (start pose, 2026-11-20)


@pytest.mark.parametrize("contract", NEW)
def test_every_contract_has_the_broken_kinds(contract):
    problems = {e.problem for e in ALL if e.contract == contract and not e.valid}
    assert {"type", "value", "missing", "unknown_field", "schema_version"} <= problems


def test_the_broken_cases_of_the_structure():
    by_name = {e.name: e for e in ALL}
    for name, field in (("u1/keys_unknown_key", "held[1]"),                         # an unknown key name
                        ("u1/stop_episode_unknown_outcome", "outcome"),            # an unknown outcome
                        ("u1/delete_bad_episode_name", "episode"),                 # a bad episode name
                        ("u2/get_preview_main_after_below_minus_one", "main_after"),
                        ("u1/status_message_ids_not_increasing", "messages[1].id"),
                        ("u1/status_refusal_while_allowed", "start_refusal")):
        e = by_name[name]
        assert not e.valid and (e.field, e.problem) == (field, "value"), name


def test_none_are_stored():
    assert not any(e.stored for e in ALL)
    stored = {p.relative_to(examples_dir()).as_posix() for p in examples_dir().rglob("*.msgpack")}
    assert not any(name.startswith(tuple(f"{c}/" for c in NEW)) for name in stored)
    assert {e.contract for e in ALL} == set(NEW)
