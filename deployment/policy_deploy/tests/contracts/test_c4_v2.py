"""c4.config.reply/v2: `kind` required and non-empty; the old v1 reply is refused (STRUCTURE.md 3.3, 12.2)."""
from __future__ import annotations

import dataclasses

import pytest
from contract_helpers import contract_error
from deploy_contracts import ContractError, decode, encode, pack_obj

from policy_deploy.contracts import OBSERVATION_INPUTS, PolicyConfig


def config(**change):
    d = dict(obs_steps=2, action_steps=16, step_ms=50, image_size=224, inputs=OBSERVATION_INPUTS, uses_prompt=False,
             kind="hold_still")
    d.update(change)
    return PolicyConfig(**d)


def test_schema_is_v2_and_kind_round_trips():
    assert PolicyConfig.SCHEMA == "c4.config.reply/v2"
    c = config(kind="diffusion")
    assert decode(encode(c)).kind == "diffusion"
    assert list(c.to_dict())[-1] == "kind"


def test_kind_required():
    d = config().to_dict()
    del d["kind"]
    with pytest.raises(ContractError) as e:
        decode(pack_obj(d))
    assert (e.value.field, e.value.problem) == ("kind", "missing")


def test_kind_non_empty_and_a_string():
    contract_error(lambda: config(kind=""), "kind", "value")
    contract_error(lambda: config(kind=3), "kind", "type")
    assert dataclasses.replace(config(), kind="anything_else").kind == "anything_else"


def test_v1_reply_refused():
    d = config().to_dict()
    del d["kind"]
    d["schema"] = "c4.config.reply/v1"
    with pytest.raises(ContractError) as e:
        decode(pack_obj(d))
    assert e.value.problem == "schema_version"
