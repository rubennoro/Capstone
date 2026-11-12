from __future__ import annotations

import copy

import numpy as np
import pytest

from deploy import GeometryError
from deploy.geometry import GripperFile
from helpers import REAL_GRIPPER, write_yaml

GOOD = {"ultrawide": {"position_mm": [55.0, 0.0, 10.0], "up_side": "+x", "tilt_deg": 15.0}}


def problems(tmp_path, data) -> list[str]:
    with pytest.raises(GeometryError) as info:
        GripperFile.load(write_yaml(tmp_path / "gripper.yaml", data))
    return info.value.problems


def with_cam(**changes) -> dict:
    data = copy.deepcopy(GOOD)
    for key, value in changes.items():
        if value is None:
            del data["ultrawide"][key]
        else:
            data["ultrawide"][key] = value
    return data


def test_real_gripper_file_loads():
    gripper = GripperFile.load(REAL_GRIPPER)
    np.testing.assert_allclose(gripper.position_m, [0.055, 0.0, 0.010])
    np.testing.assert_array_equal(gripper.up_axis, [1.0, 0.0, 0.0])
    assert gripper.tilt_deg == 15.0 and gripper.path == REAL_GRIPPER


@pytest.mark.parametrize("side,axis", [("+x", [1, 0, 0]), ("-x", [-1, 0, 0]), ("+y", [0, 1, 0]), ("-y", [0, -1, 0])])
def test_up_sides(tmp_path, side, axis):
    gripper = GripperFile.load(write_yaml(tmp_path / "g.yaml", with_cam(up_side=side)))
    np.testing.assert_array_equal(gripper.up_axis, axis)


@pytest.mark.parametrize("key", ["position_mm", "up_side", "tilt_deg"])
def test_missing_key(tmp_path, key):
    assert problems(tmp_path, with_cam(**{key: None})) == [f"gripper.yaml: ultrawide.{key}: required"]


def test_unknown_key(tmp_path):
    assert problems(tmp_path, with_cam(tilt=15)) == ["gripper.yaml: ultrawide.tilt: unknown key"]
    assert problems(tmp_path, {**GOOD, "main": {}}) == ["gripper.yaml: main: unknown key"]


def test_finger_extra_refused(tmp_path):
    assert problems(tmp_path, with_cam(finger_extra_m=0.0466)) == \
        ["gripper.yaml: ultrawide.finger_extra_m: unknown key"]


@pytest.mark.parametrize("position", [[55.0, 0.0, float("nan")], [55.0, float("inf"), 1.0], [1.0, 2.0], "here",
                                      [1.0, True, 2.0]])
def test_bad_position(tmp_path, position):
    [problem] = problems(tmp_path, with_cam(position_mm=position))
    assert problem.startswith("gripper.yaml: ultrawide.position_mm: expected 3 finite numbers")


@pytest.mark.parametrize("side", ["+z", "-z", "x", "up", 1, ["+x"]])
def test_bad_up_side(tmp_path, side):
    [problem] = problems(tmp_path, with_cam(up_side=side))
    assert problem.startswith("gripper.yaml: ultrawide.up_side: expected one of +x, -x, +y, -y")


@pytest.mark.parametrize("tilt", [-0.1, 90.0, 120.0, "15", float("nan")])
def test_bad_tilt(tmp_path, tilt):
    [problem] = problems(tmp_path, with_cam(tilt_deg=tilt))
    assert problem.startswith("gripper.yaml: ultrawide.tilt_deg: expected a number in [0, 90)")


def test_tilt_zero_allowed(tmp_path):
    assert GripperFile.load(write_yaml(tmp_path / "g.yaml", with_cam(tilt_deg=0))).tilt_deg == 0.0


def test_all_problems_listed_at_once(tmp_path):
    data = with_cam(position_mm=[1, 2], up_side="+z", tilt_deg=None, extra=1)
    assert len(problems(tmp_path, data)) == 4


def test_missing_file_named(tmp_path):
    with pytest.raises(GeometryError, match=r"gripper.yaml: not found: .*gripper.yaml"):
        GripperFile.load(tmp_path / "gripper.yaml")


def test_ultrawide_must_be_a_map(tmp_path):
    assert problems(tmp_path, {"ultrawide": 3})[0].startswith("gripper.yaml: ultrawide: required")
