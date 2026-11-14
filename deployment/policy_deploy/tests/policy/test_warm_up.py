# copied from rnoro/deployment:deployment/policy/tests/test_warm_up.py @ 91b178b
# changes: imports
from __future__ import annotations

import pytest

from policy_fakes import BadChunkPolicy, FakeClock, NonUnitRot6dPolicy, RaisingPolicy, WrongShapePolicy, _TestPolicy
from policy_deploy.policy.app import run_warm_up
from policy_deploy.policy.errors import PolicyFatal
from policy_deploy.policy.neutral_observation import neutral_observation


def test_calls_infer_n_times_times_each_then_resets():
    policy = _TestPolicy()
    times = run_warm_up(policy, neutral_observation(0.05), 3, clock=FakeClock(step_ns=10_000_000))
    assert policy.infers == 3 and policy.resets == 1
    assert times == pytest.approx([0.01, 0.01, 0.01])


@pytest.mark.parametrize("policy_cls", [NonUnitRot6dPolicy, BadChunkPolicy, WrongShapePolicy, RaisingPolicy])
def test_every_warm_up_failure_is_fatal(policy_cls):
    policy = policy_cls()
    with pytest.raises(PolicyFatal) as e:
        run_warm_up(policy, neutral_observation(0.05), 2)
    assert e.value.what.startswith("warm-up inference failed")
    assert policy.infers == 1 and policy.resets == 0
