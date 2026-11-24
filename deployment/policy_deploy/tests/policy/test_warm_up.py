# copied from rnoro/deployment:deployment/policy/tests/test_warm_up.py @ 91b178b
# changes: imports; + the warm-up prompt of a PromptablePolicy (run_warm_up_prompt)
# copied from rnoro/final-deployment:deployment/policy_deploy/tests/policy/test_warm_up.py @ f632576
# changes: make_chunk orthonormalizes rot6d (Gram-Schmidt), so non-unit rows are repaired, not fatal
# (sim_deploy, rnoro/main-sim)
from __future__ import annotations

import pytest

from policy_fakes import (BadChunkPolicy, FakeClock, FakePromptablePolicy, NonUnitRot6dPolicy, RaisingPolicy,
                          WrongShapePolicy, _TestPolicy)
from policy_deploy.policy.app import run_warm_up, run_warm_up_prompt
from policy_deploy.policy.errors import PolicyFatal
from policy_deploy.policy.neutral_observation import neutral_observation


def test_calls_infer_n_times_times_each_then_resets():
    policy = _TestPolicy()
    times = run_warm_up(policy, neutral_observation(0.05), 3, clock=FakeClock(step_ns=10_000_000))
    assert policy.infers == 3 and policy.resets == 1
    assert times == pytest.approx([0.01, 0.01, 0.01])


@pytest.mark.parametrize("policy_cls", [BadChunkPolicy, WrongShapePolicy, RaisingPolicy])
def test_every_warm_up_failure_is_fatal(policy_cls):
    policy = policy_cls()
    with pytest.raises(PolicyFatal) as e:
        run_warm_up(policy, neutral_observation(0.05), 2)
    assert e.value.what.startswith("warm-up inference failed")
    assert policy.infers == 1 and policy.resets == 0


def test_non_unit_rot6d_is_orthonormalized_not_fatal():
    policy = NonUnitRot6dPolicy()                         # rot6d rows of length 2: make_chunk's Gram-Schmidt fixes them
    run_warm_up(policy, neutral_observation(0.05), 2)
    assert policy.infers == 2 and policy.resets == 1


def test_promptable_policy_is_prompted_with_its_warm_up_prompt_first():
    policy = FakePromptablePolicy()
    warm = run_warm_up_prompt(policy, clock=FakeClock(step_ns=20_000_000))
    run_warm_up(policy, neutral_observation(0.05), 2)
    assert warm == {"task_name": "longest_task", "prompt_index": 7, "prompt_s": pytest.approx(0.02)}
    assert policy.calls == [("prompt", "longest_task", 7), ("infer",), ("infer",), ("reset",)]


def test_warm_up_prompt_failure_is_fatal():
    policy = FakePromptablePolicy()

    def refuse(task_name, prompt_index):
        raise ValueError("no such demo")
    policy.prompt = refuse
    with pytest.raises(PolicyFatal) as e:
        run_warm_up_prompt(policy)
    assert e.value.what == "warm-up prompt failed: ValueError: no such demo"
    assert "policy.prompt_dataset" in e.value.check
