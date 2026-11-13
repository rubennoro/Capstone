from __future__ import annotations

from pathlib import Path

import pytest

from deploy import Station, StationError, load_process_config, resolve_path
from helpers import write_yaml

BASE = {"kind": None, "checkpoint": "models/x.ckpt", "device": "cuda", "steps": [1, 2, 3],
        "limits": {"box": {"x": 1.0, "y": 2.0}, "speed": 0.5}}


@pytest.fixture
def folder(tmp_path) -> Path:
    write_yaml(tmp_path / "policy" / "config.yaml", BASE)
    return tmp_path / "policy"


def test_empty_section_changes_nothing(folder):
    assert load_process_config(folder, {}) == BASE


def test_maps_merge_key_by_key(folder):
    merged = load_process_config(folder, {"limits": {"box": {"y": 9.0}}})
    assert merged["limits"] == {"box": {"x": 1.0, "y": 9.0}, "speed": 0.5}


def test_lists_and_scalars_are_replaced(folder):
    merged = load_process_config(folder, {"steps": [7], "device": "cpu", "kind": "hold_still"})
    assert merged["steps"] == [7]
    assert (merged["device"], merged["kind"]) == ("cpu", "hold_still")


def test_base_not_mutated_between_loads(folder):
    load_process_config(folder, {"limits": {"speed": 3.0}})
    assert load_process_config(folder, {})["limits"]["speed"] == 0.5


def test_unknown_keys_refused_with_path(folder):
    with pytest.raises(StationError) as info:
        load_process_config(folder, {"chekpoint": "x", "limits": {"box": {"z": 1}}})
    assert info.value.problems == ["station.yaml: policy.chekpoint: not a key of policy/config.yaml",
                                   "station.yaml: policy.limits.box.z: not a key of policy/config.yaml"]


def test_scalar_over_a_map_refused(folder):
    with pytest.raises(StationError, match=r"policy\.limits: expected a map"):
        load_process_config(folder, {"limits": 3})


def test_missing_config_yaml(tmp_path):
    with pytest.raises(StationError, match="nothing/config.yaml: not found"):
        load_process_config(tmp_path / "nothing", {})


def test_resolve_path_by_where_the_value_came_from(fake_root, make_station):
    station = Station.load(make_station(policy={"checkpoint": "ckpts/best.ckpt"}), False)
    config = load_process_config(fake_root / "policy", station.section("policy"))
    assert resolve_path(station, "policy", config, "checkpoint") == station.path.parent / "ckpts/best.ckpt"
    frames = load_process_config(fake_root / "process_frames", {})
    assert resolve_path(station, "process_frames", frames, "rig_file") == \
        fake_root / "process_frames" / "rigs" / "rig_pi_v2.yaml"


def test_resolve_path_unset_value(fake_root, make_station):
    station = Station.load(make_station(), False)
    config = load_process_config(fake_root / "policy", {})
    with pytest.raises(StationError, match="policy.checkpoint is not set"):
        resolve_path(station, "policy", config, "checkpoint")
