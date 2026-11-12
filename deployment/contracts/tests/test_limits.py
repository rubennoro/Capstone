from __future__ import annotations

import pytest
import yaml

from contracts import ContractLimits
from contracts.limits import LIMITS_FILE


def test_limits_yaml_loads():
    limits = ContractLimits.load()
    assert limits.rotation_tol == 1.0e-6
    assert limits.anchor_row_tol == 1.0e-5
    assert limits.rot6d_norm_tol == 0.1
    assert limits.rot6d_dot_max == 0.1
    assert limits.gripper_width_max_m == 0.08
    assert limits.max_waypoints == 64
    assert ContractLimits.load() is limits  # cached
    assert ContractLimits.load(LIMITS_FILE) == limits


def _write(tmp_path, data):
    p = tmp_path / "limits.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def _good():
    return yaml.safe_load(LIMITS_FILE.read_text())


def test_missing_key_refused(tmp_path):
    d = _good()
    del d["rot6d_dot_max"]
    with pytest.raises(ValueError, match="rot6d_dot_max"):
        ContractLimits.load(_write(tmp_path, d))


def test_unknown_key_refused(tmp_path):
    d = _good()
    d["speed_limit"] = 1.0
    with pytest.raises(ValueError, match="speed_limit"):
        ContractLimits.load(_write(tmp_path, d))


@pytest.mark.parametrize("key,value", [("max_waypoints", 0), ("max_waypoints", 2.5), ("rotation_tol", -1.0),
                                       ("rotation_tol", "small"), ("gripper_width_max_m", True)])
def test_bad_value_refused(tmp_path, key, value):
    d = _good()
    d[key] = value
    with pytest.raises(ValueError, match=key):
        ContractLimits.load(_write(tmp_path, d))


def test_yaml_numbers_are_floats():
    # PyYAML reads 1e-6 (no dot) as a string; the file must write 1.0e-6
    raw = yaml.safe_load(LIMITS_FILE.read_text())
    assert all(isinstance(raw[k], float) for k in raw if k != "max_waypoints")
