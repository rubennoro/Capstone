from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from policy.errors import PolicyLoadError
from policy.key_map import ACTION_FIELDS, OBS_FIELDS, KeyMap
from policy.settings import CONFIG_FILE


def _default_map() -> KeyMap:
    return KeyMap.from_config(yaml.safe_load(CONFIG_FILE.read_text())["key_map"])


def _check_fails(key_map: KeyMap, shape_meta, *fragments: str) -> str:
    with pytest.raises(PolicyLoadError) as e:
        key_map.check(shape_meta)
    for fragment in fragments:
        assert fragment in str(e.value)
    return str(e.value)


def test_round_trip_names():
    km = _default_map()
    for field in OBS_FIELDS:
        assert km.contract_field(km.model_key(field)) == field
    for field in ACTION_FIELDS:
        assert km.contract_field(km.model_key(field, action=True), action=True) == field
    assert km.model_key("eef_rot6d") == "gripper_right_eef_rot_axis_angle"
    assert km.model_key("eef_rot6d", action=True) == "right_rot"


def test_action_slices_from_run_001_components(run_001_shape_meta):
    slices = _default_map().action_slices(run_001_shape_meta)
    assert slices == {"eef_pos": slice(0, 3), "eef_rot6d": slice(3, 9), "gripper_width_m": slice(9, 10)}


def test_split_then_join_gives_back_the_vector(run_001_shape_meta):
    km = _default_map()
    slices = km.action_slices(run_001_shape_meta)
    action = np.arange(16 * 10, dtype=np.float32).reshape(16, 10)
    parts = km.split_action(action, slices)
    assert parts["eef_pos"].shape == (16, 3) and parts["eef_rot6d"].shape == (16, 6)
    assert parts["gripper_width_m"].shape == (16, 1)
    assert all(p.flags["C_CONTIGUOUS"] and p.dtype == np.float32 for p in parts.values())
    np.testing.assert_array_equal(km.join_action(parts, slices), action)


def test_check_passes_for_run_001_and_plain(run_001_shape_meta, plain_shape_meta):
    _default_map().check(run_001_shape_meta)
    _default_map().check(plain_shape_meta)


def test_used_fields_leave_out_wrt_start(run_001_shape_meta):
    assert _default_map().used_fields(run_001_shape_meta) == (
        "eef_pos", "eef_rot6d", "gripper_width_m", "main", "ultrawide")


def test_used_model_input_without_contract_field(plain_shape_meta):
    plain_shape_meta["obs"]["camera_left_main_rgb"] = copy.deepcopy(plain_shape_meta["obs"]["camera_right_main_rgb"])
    _check_fails(_default_map(), plain_shape_meta, "shape_meta.obs.camera_left_main_rgb")


def test_mapped_key_missing_from_shape_meta(plain_shape_meta):
    del plain_shape_meta["obs"]["camera_right_ultrawide_rgb"]
    _check_fails(_default_map(), plain_shape_meta, "key_map.obs.ultrawide", "camera_right_ultrawide_rgb")


def test_unknown_contract_field(plain_shape_meta):
    cfg = yaml.safe_load(CONFIG_FILE.read_text())["key_map"]
    cfg["obs"]["wrist"] = "camera_wrist_rgb"
    _check_fails(KeyMap.from_config(cfg), plain_shape_meta, "key_map.obs.wrist: unknown C4 field")


def test_component_size_differs(plain_shape_meta):
    plain_shape_meta["action"]["components"][1]["size"] = 4
    _check_fails(_default_map(), plain_shape_meta, "'right_rot' has size 4")


def test_component_without_contract_field(plain_shape_meta):
    plain_shape_meta["action"]["components"].append({"type": "left_pos", "size": 3})
    _check_fails(_default_map(), plain_shape_meta, "'left_pos' has no C4 field")


def test_not_one_to_one(plain_shape_meta):
    cfg = yaml.safe_load(CONFIG_FILE.read_text())["key_map"]
    cfg["obs"]["ultrawide"] = "camera_right_main_rgb"
    _check_fails(KeyMap.from_config(cfg), plain_shape_meta, "one-to-one")
    cfg = yaml.safe_load(CONFIG_FILE.read_text())["key_map"]
    cfg["action"]["eef_rot6d"] = "right_pos"
    _check_fails(KeyMap.from_config(cfg), plain_shape_meta, "one-to-one")


def test_every_problem_listed(plain_shape_meta):
    del plain_shape_meta["obs"]["camera_right_ultrawide_rgb"]
    plain_shape_meta["action"]["components"][1]["size"] = 4
    text = _check_fails(_default_map(), plain_shape_meta)
    assert "ultrawide" in text and "right_rot" in text
