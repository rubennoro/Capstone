"""T1: exported episodes pass the iPhone-format validator; the validator catches broken episodes.
T1b: the validator accepts the real iPhone exports (ground-truths the validator)."""

import json
import shutil

import pytest

from sensor_rig.export.exporter import export_episode
from sensor_rig.export.iphone_json import dumps_swift
from sensor_rig.export.validate import validate_export_root, validate_iphone_episode
from wpa_helpers import CALIB_NAME, make_raw_episode

REAL_DEMOS = ["49242", "18798", "87784"]
REAL_CALIB = "2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left"


@pytest.fixture(scope="module")
def exported(tmp_path_factory, test_rig):
    root = tmp_path_factory.mktemp("t1")
    calib = make_raw_episode(root / "raw", test_rig, kind="gripper_calibration", name=CALIB_NAME)
    demo = make_raw_episode(root / "raw", test_rig)
    out = root / "out"
    rc = export_episode(calib, out)
    rd = export_episode(demo, out)
    assert rc.status == "exported" and rd.status == "exported", (rc, rd)
    return out, rd.out_dir, rc.out_dir


def test_exported_episodes_valid(exported):
    out, demo, calib = exported
    assert validate_iphone_episode(demo, decode=True) == []
    assert validate_iphone_episode(calib, decode=True) == []
    assert all(v == [] for v in validate_export_root(out).values())
    assert {p.name for p in demo.iterdir()} == {demo.name + s for s in (".json", "_rgb.mp4", "_ultrawidergb.mp4")}
    assert {p.name for p in calib.iterdir()} == {calib.name + s for s in (".json", "_ultrawidergb.mp4")}


def test_gripper_calibration_lookup(exported, tmp_path):
    _, demo, _ = exported
    yml = tmp_path / "gc.yaml"
    yml.write_text("default:\n  iPhone17,1: {}\n")
    assert any("calibration:" in e for e in validate_iphone_episode(demo, yml))
    yml.write_text("default:\n  JetsonRig1,1: {}\n")
    assert validate_iphone_episode(demo, yml) == []


def _broken_copy(src, tmp_path, mutate_json=None, mutate_files=None):
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    jp = dst / f"{src.name}.json"
    if mutate_json:
        j = json.loads(jp.read_text())
        mutate_json(j)
        jp.write_text(dumps_swift(j))
    if mutate_files:
        mutate_files(dst)
    return validate_iphone_episode(dst)


def _drop_uw(j):
    i = next(i for i, s in enumerate(j["ultrawideRGBTimes"]) if s)
    j["ultrawideRGBTimes"][i] = ""


def _identity_pose(j):
    j["poseTransforms"][3] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _shift_uw(j):  # move an ultrawide time string to a neighbouring slot
    uw = j["ultrawideRGBTimes"]
    i = [k for k, s in enumerate(uw) if s][2]
    uw[i], uw[i + 1] = "", j["poseTimes"][i + 1]


@pytest.mark.parametrize("mutate,expect", [
    (lambda j: j.pop("isVoiceHost"), "key set"),
    (lambda j: j.update(frameCount=j["frameCount"] + 1), "counts"),
    (lambda j: j.update(frameCount=str(j["frameCount"])), "type"),
    (_drop_uw, "inv. 2"),
    (_shift_uw, "inv. 4"),
    (_identity_pose, "identity"),
    (lambda j: j.update(type="gripper_calibration"), "type"),
    (lambda j: j.update(rgbTimes=j["rgbTimes"][::-1]), "rgbTimes"),
    (lambda j: j.update(depthTimes=j["poseTimes"]), "depth"),
    (lambda j: j.update(sidesPresent=["right"]), "sidesPresent"),
    (lambda j: j["poseTimes"].__setitem__(0, "2026-10-20 15:02:11.143"), "ISO"),
    (lambda j: j.update(ultrawideCameraIntrinsics=[[259.0, 0, 100.0], [0, 259.0, 239.7], [0, 0, 1]]), "principal"),
])
def test_validator_catches(exported, tmp_path, mutate, expect):
    _, demo, _ = exported
    errors = _broken_copy(demo, tmp_path, mutate_json=mutate)
    assert errors and any(expect in e for e in errors), errors


def test_validator_catches_files(exported, tmp_path):
    _, demo, _ = exported
    errors = _broken_copy(demo, tmp_path, mutate_files=lambda d: (d / f"{d.name}_extra.txt").write_text("x"))
    assert any("unexpected" in e for e in errors)
    errors = _broken_copy(demo, tmp_path / "b", mutate_files=lambda d: (d / f"{d.name}_rgb.mp4").unlink())
    assert any("missing" in e for e in errors)


def test_export_root_unresolved_calibration(exported, tmp_path):
    _, demo, _ = exported
    shutil.copytree(demo, tmp_path / "2026-10-20" / demo.name)
    res = validate_export_root(tmp_path)
    assert any("is not under" in e for e in res[demo.name])


@pytest.mark.data
def test_real_iphone_exports_pass(iphone_data, gripper_calibration_yaml):
    """T1b: three real demos + the calibration they reference pass with 0 errors."""
    folders = [iphone_data / REAL_CALIB]
    for rid in REAL_DEMOS:
        folders += [p for p in iphone_data.iterdir() if f"_{rid}_" in p.name]
    assert len(folders) == 4
    for f in folders:
        assert validate_iphone_episode(f, gripper_calibration_yaml, decode=False) == [], f.name


@pytest.mark.data
def test_real_iphone_export_root(iphone_data, gripper_calibration_yaml):
    res = validate_export_root(iphone_data, gripper_calibration_yaml)
    assert len(res) >= 45
    assert {k: v for k, v in res.items() if v} == {}
