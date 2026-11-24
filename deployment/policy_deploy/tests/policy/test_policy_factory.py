# adapted from rnoro/deployment:deployment/policy/tests/test_policy_factory.py @ 91b178b
# changes: the new Settings.load(station, mock) and station schema; --mock builds mock_kind; module path
# copied from rnoro/final-deployment:deployment/policy_deploy/tests/policy/test_policy_factory.py @ f632576
# changes: KINDS has replay (ReplayPolicy, no torch) (sim_deploy, rnoro/main-sim)
from __future__ import annotations

import sys

import pytest

from policy_deploy.policy.errors import PolicyLoadError
from policy_deploy.policy.hold_still_policy import HoldStillPolicy
from policy_deploy.policy.policy_factory import KINDS, build_policy
from policy_deploy.policy.settings import Settings


def test_kinds_name_their_classes():
    assert set(KINDS) == {"hold_still", "diffusion", "replay"}
    assert KINDS["hold_still"].needs_torch is False and KINDS["diffusion"].needs_torch is True
    assert KINDS["replay"].needs_torch is False


def test_hold_still_builds_its_class(policy_station, tmp_path):
    policy = build_policy(Settings.load(policy_station({"kind": "hold_still"}), mock=False))
    assert isinstance(policy, HoldStillPolicy) and policy.KIND == "hold_still"


def test_mock_builds_mock_kind_whatever_the_kind(policy_station, tmp_path):
    policy = build_policy(Settings.load(policy_station({"kind": "diffusion"}), mock=True))
    assert isinstance(policy, HoldStillPolicy)                     # mock_kind defaults to hold_still


def test_diffusion_class_kind_matches():
    pytest.importorskip("torch", reason="torch tests run in the policy environment")
    from policy_deploy.policy.diffusion_policy import DiffusionPolicy
    assert DiffusionPolicy.KIND == "diffusion"


def test_diffusion_with_torch_blocked(policy_station, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)            # `import torch` now raises ImportError
    monkeypatch.delitem(sys.modules, "policy_deploy.policy.diffusion_policy", raising=False)
    settings = Settings.load(policy_station({"kind": "diffusion", "checkpoint": "best.ckpt"},
                                            python={"light": sys.executable, "torch": sys.executable}), mock=False)
    with pytest.raises(PolicyLoadError) as e:
        build_policy(settings)
    assert e.value.what == "torch is not importable (kind diffusion)"
    assert e.value.check == "run the policy with python.torch, the torch environment (station.yaml)"
