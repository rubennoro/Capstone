# adapted from rnoro/deployment:deployment/deploy/tests/test_config_merge.py @ 91b178b
# changes: against policy_deploy.contracts.config_merge.load_config(name, station, path) (a test config file passed as
# path); messages name config/<file>; resolve_path takes dotted keys and returns None for null.
from __future__ import annotations

import pytest
import yaml

from policy_deploy.contracts import Station, StationError, paths
from policy_deploy.contracts.config_merge import load_config, resolve_path

BASE = {"kind": None, "checkpoint": "models/x.ckpt", "device": "cuda", "steps": [1, 2, 3],
        "limits": {"box": {"x": 1.0, "y": 2.0}, "speed": 0.5}}


class _Section:
    """A stand-in station: only section() and path are used by load_config."""

    def __init__(self, section, path):
        self._section = section
        self.path = path

    def section(self, name):
        return self._section


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(BASE))
    return path


def load(config, section):
    return load_config("policy", _Section(section, config.parent / "station.yaml"), config)


def test_empty_section_changes_nothing(config):
    assert load(config, {}) == BASE
    assert load_config("policy", None, config) == BASE


def test_maps_merge_key_by_key(config):
    assert load(config, {"limits": {"box": {"y": 9.0}}})["limits"] == {"box": {"x": 1.0, "y": 9.0}, "speed": 0.5}


def test_lists_and_scalars_are_replaced(config):
    merged = load(config, {"steps": [7], "device": "cpu", "kind": "hold_still"})
    assert merged["steps"] == [7] and (merged["device"], merged["kind"]) == ("cpu", "hold_still")


def test_base_not_mutated_between_loads(config):
    load(config, {"limits": {"speed": 3.0}})
    assert load(config, {})["limits"]["speed"] == 0.5


def test_unknown_keys_refused_with_path(config):
    with pytest.raises(StationError) as info:
        load(config, {"chekpoint": "x", "limits": {"box": {"z": 1}}})
    assert info.value.problems == ["station.yaml: policy.chekpoint: not a key of config/policy.yaml",
                                   "station.yaml: policy.limits.box.z: not a key of config/policy.yaml"]


def test_scalar_over_a_map_refused(config):
    with pytest.raises(StationError, match=r"policy\.limits: expected a map"):
        load(config, {"limits": 3})


def test_missing_config_file(tmp_path):
    with pytest.raises(StationError, match="config/policy.yaml: not found"):
        load_config("policy", None, tmp_path / "nothing.yaml")


def test_contract_keys_in_a_config_file_refused(tmp_path):
    path = tmp_path / "manager.yaml"
    path.write_text(yaml.safe_dump({"timeouts": {"c3_timeout_s": 0.3}}))
    with pytest.raises(StationError) as info:
        load_config("manager", None, path)
    assert info.value.problems == ["config/manager.yaml: timeouts.c3_timeout_s: belongs to "
                                   "deployment/contracts/link.yaml"]


def test_resolve_path_by_where_the_value_came_from(make_station):
    station = Station.load(make_station(policy={"kind": "hold_still", "checkpoint": "ckpts/best.ckpt"}), False)
    config = load_config("policy", station)
    assert resolve_path(station, "policy", config, "checkpoint") == station.path.parent / "ckpts/best.ckpt"
    frames = load_config("frames", station)
    assert resolve_path(station, "frames", frames, "rig_file") == paths.folder() / ".." / "contracts" / "rigs" / "rig_pi_v3_640.yaml"
    assert resolve_path(station, "frames", frames, "mock_episode").is_absolute()          # ~ expanded


def test_resolve_path_dotted_and_unset(make_station):
    station = Station.load(make_station(), False)
    assert resolve_path(station, "policy", load_config("policy", station), "checkpoint") is None
    cfg = {"a": {"b": "x/y"}}
    assert resolve_path(None, "policy", cfg, "a.b") == paths.folder() / "x/y"
