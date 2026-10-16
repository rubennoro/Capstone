"""T7: iPhone JSON: 32 keys, types, Swift-like compact float32 output, calibration empties."""

import json

import numpy as np
import pytest

from sensor_rig.export.iphone_json import KEY_TYPES, KEYS, build_json, dumps_swift, format_float32

K_MAIN = [[1406.3428, 0, 953.2503], [0, 1406.3428, 723.22003], [0, 0, 1]]
K_UW = [[259.01538, 0, 320.2935], [0, 259.01538, 239.67455], [0, 0, 1]]
TIMES = ["2026-10-20T15:02:11.143Z", "2026-10-20T15:02:11.160Z", "2026-10-20T15:02:11.176Z"]


def _demo(**kw):
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, :3, :3] = [[0.8660254, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.8660254]]
    T[:, :3, 3] = [[1.0, -0.2, 5.0], [1.01, -0.2, 5.0], [1.02, -0.2, 5.0]]
    args = dict(kind="demonstration", frame_count=3, pose_times=TIMES, pose_transforms=T,
                uw_times=["", TIMES[1], ""], recording_start_time="2026-10-20T15:02:11.123Z", side="left",
                session="rig-test", gripper_id="default",
                calibration_run="2026-10-20T14-58-40.002Z_77311_rig-test_grippercalibration_left",
                sides_present=["left"], device_name="Jetson Sensor Rig", device_identifier="JetsonRig1,1",
                K_main=K_MAIN, K_uw=K_UW)
    args.update(kw)
    return build_json(**args)


def test_keys_and_types_demo():
    j = json.loads(dumps_swift(_demo()))
    assert len(KEYS) == 32 and set(j) == set(KEYS)
    for k, t in KEY_TYPES.items():
        assert isinstance(j[k], t), k
        if t is int:
            assert not isinstance(j[k], bool)
    assert j["type"] == "demonstration" and j["rgbTimes"] == j["poseTimes"]
    assert j["hasRGB"] and j["hasUltrawideRGB"] and not j["hasDepth"] and j["depthTimes"] == []
    assert j["labelType"] == "None" and j["taskNames"] == []
    assert j["deviceIdentifier"] == "JetsonRig1,1"
    assert np.asarray(j["poseTransforms"]).shape == (3, 4, 4)


def test_calibration_empties():
    j = json.loads(dumps_swift(_demo(kind="gripper_calibration", pose_times=[], pose_transforms=[],
                                     calibration_run="ignored")))
    assert j["type"] == "gripper_calibration"
    for k in ("poseTimes", "poseTransforms", "rgbTimes", "depthTimes", "depthFrameIsRepeat", "taskNames"):
        assert j[k] == []
    assert j["gripperCalibrationRunName"] == "" and not j["hasRGB"] and j["hasUltrawideRGB"]
    assert j["frameCount"] == 3 and len(j["ultrawideRGBTimes"]) == 3


def test_label_task():
    j = _demo(task_label="block_raise")
    assert j["labelType"] == "Predefined" and j["taskNames"] == ["block_raise"]
    assert j["taskStartTimestamps"] == [TIMES[0]] and j["taskEndTimestamps"] == [TIMES[-1]]


def test_compact_float32_format():
    s = dumps_swift(_demo())
    assert not s.endswith("\n") and ": " not in s and ", " not in s
    assert '"mainCameraIntrinsics":[[1406.3428,0,953.2503],[0,1406.3428,723.22003],[0,0,1]]' in s
    assert format_float32(1.0) == "1" and format_float32(-0.0) == "-0"
    assert format_float32(0.99999994) == "0.99999994"
    assert format_float32(-6.782484e-05) == "-6.782484e-05"
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(size=200), rng.normal(size=50) * 1e-6, rng.normal(size=50) * 1e4])
    back = json.loads(dumps_swift(vals.tolist()))
    assert np.array_equal(np.float32(back), np.float32(vals))
    with pytest.raises(ValueError):
        dumps_swift([float("nan")])


def test_consistency_checks():
    with pytest.raises(ValueError):
        _demo(uw_times=["", ""])
    with pytest.raises(ValueError):
        _demo(kind="gripper_calibration")  # calibrations carry no poses
    with pytest.raises(ValueError):
        _demo(kind="demo")
