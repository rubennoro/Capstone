import copy
import math
import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from dataproc.config import (
    ConfigError, TaskConfig, dump_task_config, load_rig, load_task_config, parse_rig,
)

RIG_V1 = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"


@pytest.fixture
def rig_dict():
    return yaml.safe_load(RIG_V1.read_text())


# --------------------------------------------------------------------------------------------- task.yaml

def test_task_roundtrip(tmp_path):
    cfg = TaskConfig(label="block_raise", rig="rig_v1", created="2026-11-04T21:00:00+00:00", notes="first")
    dump_task_config(cfg, tmp_path / "task.yaml")
    assert load_task_config(tmp_path / "task.yaml") == cfg


def test_task_unquoted_timestamp_and_no_notes(tmp_path):
    (tmp_path / "task.yaml").write_text("label: block_raise\nrig: rig_v1\ncreated: 2026-11-04 21:00:00\n")
    cfg = load_task_config(tmp_path / "task.yaml")
    assert cfg.created == datetime.datetime(2026, 11, 4, 21).isoformat() and cfg.notes == ""


@pytest.mark.parametrize("text, where", [
    ("rig: rig_v1\ncreated: x\n", "label: missing"),
    ("label: block raise\nrig: rig_v1\ncreated: x\n", "label: must match"),
    ("label: a\nrig: r\ncreated: x\nextra: 1\n", "extra: unknown key"),
    ("label: a\nrig: r\ncreated: ''\n", "created:"),
    ("label: a\nrig: r\ncreated: x\nnotes: [1]\n", "notes:"),
    ("- a\n", "must be a mapping"),
    ("label: [\n", "invalid YAML"),
])
def test_task_errors(tmp_path, text, where):
    p = tmp_path / "task.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match=where) as e:
        load_task_config(p)
    assert str(e.value).startswith(str(p))


def test_task_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_task_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------------------------- rig yaml

def test_rig_v1_loads():
    rig = load_rig(RIG_V1)
    assert rig.rig_id == "rig_v1" and rig.schema == "dataproc.rig.v1"
    main, uw = rig.cameras["main"], rig.cameras["ultrawide"]
    assert (main.sensor, main.resolution, main.fps, main.model.type) == ("imx519", (1280, 720), 60.0,
                                                                         "pinhole-radtan")
    hfov = 2 * np.degrees(np.arctan(main.resolution[0] / 2 / main.model.intrinsics[0]))
    assert 60 < hfov < 72
    assert (uw.sensor, uw.resolution, uw.fps, uw.model.type) == ("imx219", (820, 616), 30.0, "fisheye-equidistant")
    half_diag = np.hypot(*uw.resolution) / 2
    assert np.degrees(2 * half_diag / uw.model.intrinsics[0]) == pytest.approx(160, abs=3)
    assert rig.virtual_ultrawide.resolution == (640, 480)
    assert rig.virtual_ultrawide.K[0, 0] == pytest.approx(246.6, abs=0.1)
    assert rig.aruco.dict == "DICT_4X4_50" and rig.aruco.marker_size_m == 0.016
    assert (rig.aruco.left_id, rig.aruco.right_id) == (0, 1)
    assert rig.imu.sensor == "lsm6dsox" and 1e-5 < rig.imu.noise.gyroscope_noise_density < 1e-3
    for T in (main.T_imu_cam, uw.T_imu_cam, rig.T_cam_tcp):
        assert T.shape == (4, 4) and np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    assert 0.1 < rig.T_cam_tcp[2, 3] < 0.3                    # TCP in front of the camera
    assert rig.nominal_sections() == ["cameras.main", "cameras.ultrawide", "imu", "tcp"]
    assert main.model.K[0, 2] == main.model.intrinsics[2]


def test_rig_matrix_form_and_calibrated(rig_dict):
    T = np.eye(4)
    T[:3, 3] = [0.01, 0.02, 0.03]
    rig_dict["cameras"]["main"]["T_imu_cam"] = {"matrix": T.tolist()}
    rig_dict["cameras"]["ultrawide"]["T_imu_cam"] = {"rotation": {"matrix": np.eye(3).tolist()},
                                                    "translation_m": [0, 0, 0]}
    for sec in (rig_dict["cameras"]["main"], rig_dict["cameras"]["ultrawide"], rig_dict["imu"], rig_dict["tcp"]):
        sec["provenance"] = "calibrated"
    rig = parse_rig(rig_dict)
    assert np.array_equal(rig.cameras["main"].T_imu_cam, T) and rig.nominal_sections() == []


def _set(d, path, value):
    *head, last = path.split(".")
    for k in head:
        d = d[k]
    if value is _DEL:
        del d[last]
    else:
        d[last] = value


_DEL = object()


@pytest.mark.parametrize("path, value, where", [
    ("schema", "dataproc.rig.v0", "schema"),
    ("rig_id", "rig v1", "rig_id"),
    ("cameras.main", _DEL, "cameras.main: missing"),
    ("cameras.depth", {}, "cameras.depth: unknown key"),
    ("cameras.main.fps", 0, "cameras.main.fps"),
    ("cameras.main.resolution", [1280], "cameras.main.resolution"),
    ("cameras.main.model.type", "kb8", "cameras.main.model.type"),
    ("cameras.main.model.intrinsics", [985, 985, 2000, 359.5], "principal point"),
    ("cameras.main.model.intrinsics", [0, 985, 639.5, 359.5], "fx and fy"),
    ("cameras.main.model.distortion", [0, 0], "cameras.main.model.distortion"),
    ("cameras.ultrawide.model.type", "pinhole-radtan", "must be fisheye-equidistant"),
    ("cameras.main.T_imu_cam", {"matrix": [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}, "rigid"),
    ("cameras.main.T_imu_cam", {"rotation": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]},
                                "translation_m": [0, 0, 0]}, "not a rotation"),
    ("cameras.main.T_imu_cam", {"rotation": {"euler_xyz_deg": [0, 0, 0], "matrix": np.eye(3).tolist()},
                                "translation_m": [0, 0, 0]}, "exactly one"),
    ("cameras.main.T_imu_cam.translation_m", [0, 0], "translation_m"),
    ("cameras.main.time_offset_ns", 1.5, "time_offset_ns"),
    ("cameras.main.provenance", "guessed", "provenance"),
    ("cameras.main.timestamp_ref", _DEL, "cameras.main.timestamp_ref: missing"),
    ("cameras.ultrawide.timestamp_ref", "mid_exposure", "cameras.ultrawide.timestamp_ref"),
    ("virtual_ultrawide.rotation", {"euler_xyz_deg": [0, 0]}, "virtual_ultrawide.rotation.euler_xyz_deg"),
    ("virtual_ultrawide.rotation", {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 2]]}, "not a rotation"),
    ("imu.vio_inflation", {"noise_density": 0, "random_walk": 1}, "imu.vio_inflation.noise_density"),
    ("imu.vio_inflation", {"noise_density": 2}, "imu.vio_inflation.random_walk: missing"),
    ("world_offset", {"translation_m": [0, 0, 0], "yaw_deg": 0}, "world_offset.yaw_deg: must be at least 1 deg"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": 360.0}, "multiple of 360"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": -720}, "multiple of 360"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": 0.5}, "multiple of 360"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": -0.999}, "multiple of 360"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": 359.5}, "multiple of 360"),
    ("world_offset", {"translation_m": [1, 2], "yaw_deg": 30}, "world_offset.translation_m"),
    ("world_offset", {"translation_m": [1, 2, 3], "yaw_deg": 30, "roll": 1}, "world_offset.roll: unknown"),
    ("virtual_ultrawide.intrinsics", [-1, 1, 1, 1], "virtual_ultrawide.intrinsics"),
    ("tcp.T_cam_tcp", _DEL, "tcp.T_cam_tcp: missing"),
    ("imu.noise.gyroscope_noise_density", -1.0, "gyroscope_noise_density"),
    ("imu.noise.extra", 1.0, "imu.noise.extra: unknown key"),
    ("aruco.dict", "DICT_NOPE", "aruco.dict"),
    ("aruco.right_id", 0, "must differ"),
    ("aruco.marker_size_m", True, "aruco.marker_size_m"),
    ("bogus", 1, "bogus: unknown key"),
    ("gripper.max_opening_m", 0, "gripper.max_opening_m: must be > 0"),
    ("gripper.opening_tolerance_m", _DEL, "gripper.opening_tolerance_m: missing"),
    ("gripper.extra", 1.0, "gripper.extra: unknown key"),
])
def test_rig_errors(rig_dict, path, value, where):
    d = copy.deepcopy(rig_dict)
    _set(d, path, value)
    with pytest.raises(ConfigError, match=where):
        parse_rig(d, "rig.yaml")


def test_gripper_section(rig_dict):
    from dataproc.config import GripperConfig

    pi = load_rig(RIG_V1.with_name("rig_pi_v1.yaml"))
    assert pi.gripper == GripperConfig(0.110, 0.010)
    assert load_rig(RIG_V1).gripper == GripperConfig(0.060, 0.010)       # what the simulator's sweep measures
    d = copy.deepcopy(rig_dict)
    del d["gripper"]
    assert parse_rig(d, "rig.yaml").gripper is None                     # optional: older rig files still load


def test_rig_file_name_must_match_id(tmp_path):
    p = tmp_path / "rig_v2.yaml"
    p.write_text(RIG_V1.read_text())
    with pytest.raises(ConfigError, match="does not match the file name"):
        load_rig(p)


# --------------------------------------------------------------------------------------------- fix round (B1-B4)

def test_rig_v1_time_world_virtual_defaults():
    rig = load_rig(RIG_V1)
    assert {c.timestamp_ref for c in rig.cameras.values()} == {"end_of_exposure"}
    assert np.array_equal(rig.virtual_ultrawide.R_phys_virt, np.eye(3))
    assert rig.imu.noise.inflation_noise == 1.0 and rig.imu.noise.inflation_random_walk == 1.0
    T = rig.T_world_offset
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3]) and np.allclose(T[2, :3], [0, 0, 1])     # yaw about +z
    assert np.degrees(np.arctan2(T[1, 0], T[0, 0])) == pytest.approx(30.0)


def test_camera_and_imu_time(rig_dict):
    rig_dict["cameras"]["main"].update(timestamp_ref="start_of_exposure", time_offset_ns=7)
    rig_dict["cameras"]["ultrawide"].update(timestamp_ref="end_of_exposure", time_offset_ns=-3)
    rig_dict["imu"]["time_offset_ns"] = 11
    rig = parse_rig(rig_dict)
    assert rig.camera_time_ns("main", 1_000, 401) == 1_000 + 200 + 7
    assert rig.camera_time_ns("ultrawide", 1_000, 400) == 1_000 - 200 - 3
    t = np.array([10 ** 18, 10 ** 18 + 1], np.int64)                  # stays exact int64
    out = rig.camera_time_ns("ultrawide", t, np.array([4_000_000, 2_000_000]))
    assert out.dtype == np.int64 and out.tolist() == [10 ** 18 - 2_000_003, 10 ** 18 + 1 - 1_000_003]
    assert isinstance(rig.camera_time_ns("main", 5, 2), int)
    assert rig.imu_time_ns(100) == 111 and rig.imu_time_ns(t).tolist() == [10 ** 18 + 11, 10 ** 18 + 12]


def test_vio_inflation_and_for_vio(rig_dict):
    rig_dict["imu"]["vio_inflation"] = {"noise_density": 5.0, "random_walk": 10.0}
    n = parse_rig(rig_dict).imu.noise
    v = n.for_vio()
    assert v.gyroscope_noise_density == pytest.approx(5 * n.gyroscope_noise_density)
    assert v.accelerometer_noise_density == pytest.approx(5 * n.accelerometer_noise_density)
    assert v.gyroscope_random_walk == pytest.approx(10 * n.gyroscope_random_walk)
    assert v.accelerometer_random_walk == pytest.approx(10 * n.accelerometer_random_walk)
    assert (v.inflation_noise, v.inflation_random_walk) == (1.0, 1.0)
    del rig_dict["imu"]["vio_inflation"]
    assert parse_rig(rig_dict).imu.noise.for_vio() == parse_rig(rig_dict).imu.noise


def test_world_offset_default_and_custom(rig_dict):
    del rig_dict["world_offset"]
    T = parse_rig(rig_dict).T_world_offset
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3]) and T[0, 0] == pytest.approx(np.cos(np.radians(30)))
    rig_dict["world_offset"] = {"translation_m": [0, 0, 0], "yaw_deg": -45}
    T = parse_rig(rig_dict).T_world_offset
    assert np.allclose(T[:3, 3], 0) and T[1, 0] == pytest.approx(-np.sin(np.radians(45)))


@pytest.mark.parametrize("yaw", [1.0, -1.0, 361.0, 359.0, 180.0])
def test_world_offset_yaw_boundary_accepted(rig_dict, yaw):
    rig_dict["world_offset"] = {"translation_m": [0, 0, 0], "yaw_deg": yaw}
    T = parse_rig(rig_dict).T_world_offset
    assert np.degrees(np.arctan2(T[1, 0], T[0, 0])) == pytest.approx(math.remainder(yaw, 360.0))


def test_virtual_rotation(rig_dict):
    rig_dict["virtual_ultrawide"]["rotation"] = {"euler_xyz_deg": [10.0, 0.0, 0.0]}
    R = parse_rig(rig_dict).virtual_ultrawide.R_phys_virt
    assert R.shape == (3, 3) and np.degrees(np.arctan2(R[2, 1], R[1, 1])) == pytest.approx(10.0)
    del rig_dict["virtual_ultrawide"]["rotation"]
    assert np.array_equal(parse_rig(rig_dict).virtual_ultrawide.R_phys_virt, np.eye(3))
