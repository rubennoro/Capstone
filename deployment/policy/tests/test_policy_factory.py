from __future__ import annotations

import sys

import pytest

from policy.errors import PolicyLoadError
from policy.hold_still_policy import HoldStillPolicy
from policy.policy_factory import KINDS, build_policy
from policy.settings import Settings


def test_kinds_name_their_classes():
    assert set(KINDS) == {"hold_still", "diffusion"}
    assert KINDS["hold_still"].needs_torch is False and KINDS["diffusion"].needs_torch is True


def test_hold_still_builds_its_class(station_file, tmp_path):
    policy = build_policy(Settings.load(station_file({"kind": "hold_still"}), tmp_path, mock=False))
    assert isinstance(policy, HoldStillPolicy) and policy.KIND == "hold_still"


def test_mock_builds_hold_still_whatever_the_kind(station_file, tmp_path):
    policy = build_policy(Settings.load(station_file({"kind": "diffusion"}), tmp_path, mock=True))
    assert isinstance(policy, HoldStillPolicy)


def test_diffusion_class_kind_matches():
    pytest.importorskip("torch", reason="torch tests run in the policy environment")
    from policy.diffusion_policy import DiffusionPolicy
    assert DiffusionPolicy.KIND == "diffusion"


def test_diffusion_with_torch_blocked(station_file, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)            # `import torch` now raises ImportError
    monkeypatch.delitem(sys.modules, "policy.diffusion_policy", raising=False)
    settings = Settings.load(station_file({"kind": "diffusion", "checkpoint": "best.ckpt"}), tmp_path, mock=False)
    with pytest.raises(PolicyLoadError) as e:
        build_policy(settings)
    assert e.value.what == "torch is not importable (kind diffusion)"
    assert e.value.check == "run policy with python.policy, the torch environment"
