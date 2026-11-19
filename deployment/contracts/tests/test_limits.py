# adapted from rnoro/deployment:deployment/contracts/tests/test_limits.py @ 91b178b
# changes: imports renamed; 3 keys exactly (rotation_tol, gripper_width_max_m, max_waypoints); the file is
# deployment/contracts/limits.yaml; errors name the file.
from __future__ import annotations

import pytest
import yaml

from deploy_contracts import ContractLimits, limits_file

KEYS = ["rotation_tol", "gripper_width_max_m", "max_waypoints"]


def test_limits_yaml_loads():
    limits = ContractLimits.load()
    assert limits.rotation_tol == 1.0e-6
    assert limits.gripper_width_max_m == 0.08
    assert limits.max_waypoints == 64
    assert ContractLimits.load() is limits  # cached
    assert ContractLimits.load(limits_file()) == limits
    assert ContractLimits.load(str(limits_file())) == limits


def test_exactly_three_keys():
    assert list(yaml.safe_load(limits_file().read_text())) == KEYS
    assert [f for f in ContractLimits.__dataclass_fields__ if not f.startswith("_")] == KEYS


def _write(tmp_path, data):
    p = tmp_path / "limits.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def _good():
    return yaml.safe_load(limits_file().read_text())


@pytest.mark.parametrize("key", KEYS)
def test_missing_key_refused(tmp_path, key):
    d = _good()
    del d[key]
    p = _write(tmp_path, d)
    with pytest.raises(ValueError, match=key) as info:
        ContractLimits.load(p)
    assert str(p) in str(info.value)


@pytest.mark.parametrize("key", ["anchor_row_tol", "rot6d_norm_tol", "rot6d_dot_max", "speed_limit"])
def test_unknown_key_refused(tmp_path, key):
    # the old C4-only keys moved to policy_deploy: they are unknown here
    d = _good()
    d[key] = 0.1
    with pytest.raises(ValueError, match=key):
        ContractLimits.load(_write(tmp_path, d))


@pytest.mark.parametrize("key,value", [
    ("max_waypoints", 0), ("max_waypoints", -3), ("max_waypoints", 2.5), ("max_waypoints", "64"),
    ("max_waypoints", True),
    ("rotation_tol", -1.0), ("rotation_tol", 0.0), ("rotation_tol", "small"), ("rotation_tol", float("inf")),
    ("rotation_tol", float("nan")),
    ("gripper_width_max_m", True), ("gripper_width_max_m", 0), ("gripper_width_max_m", None),
])
def test_bad_value_refused(tmp_path, key, value):
    d = _good()
    d[key] = value
    with pytest.raises(ValueError, match=key):
        ContractLimits.load(_write(tmp_path, d))


def test_not_a_map_refused(tmp_path):
    p = tmp_path / "limits.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="limits.yaml"):
        ContractLimits.load(p)


def test_yaml_numbers_are_floats():
    # PyYAML reads 1e-6 (no dot) as a string; the file must write 1.0e-6
    raw = yaml.safe_load(limits_file().read_text())
    assert all(isinstance(raw[k], float) for k in raw if k != "max_waypoints")
    assert isinstance(raw["max_waypoints"], int)
