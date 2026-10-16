import copy

import numpy as np
import pytest
import yaml

from sensor_rig.camera_model import FisheyeEquidistant, PinholeRadtan, VirtualPinhole
from sensor_rig.rig_config import DEFAULT_RIG_PATH, RigConfigError, load_rig, validate
from sensor_rig.transforms import D_CV_ARKIT, inv_T, make_T, rot_y, rot_z


@pytest.fixture(scope="module")
def rig():
    return load_rig(DEFAULT_RIG_PATH)


@pytest.fixture()
def raw():
    return yaml.safe_load(DEFAULT_RIG_PATH.read_text())


def test_load_default(rig, tmp_path):
    assert rig.schema_version == 1 and rig.rig_name == "jetson-rig-v1-default"
    main, uw = rig.cameras["main"], rig.cameras["ultrawide"]
    assert isinstance(main.model, PinholeRadtan) and isinstance(uw.model, FisheyeEquidistant)
    assert main.resolution == (1640, 1232) and main.model.size == (1640, 1232)
    assert main.sensor_id == 0 and uw.sensor_id == 1 and main.sensor_mode == 3 and main.fps == 30.0
    assert main.argus["wbmode"] == 1 and main.time.total_offset_ns == -30_000_000
    assert isinstance(rig.virtual["main"].reference, VirtualPinhole)
    assert rig.virtual["main"].outputs == {"export": (1920, 1440), "deploy": (960, 720)}
    assert rig.imu.odr_hz == 1000 and rig.imu.gyro_range_dps == 2000.0 and rig.imu.spi_device == "/dev/spidev0.0"
    assert rig.imu.noise.for_vio().gyroscope_noise_density == pytest.approx(5 * 4.9e-5)
    assert rig.imu.noise.for_vio().accelerometer_random_walk == pytest.approx(10 * 1.0e-4)
    assert rig.vio.camera == "ultrawide" and rig.vio.trajectory_frame == "imu"
    assert rig.export.device_identifier == "JetsonRig1,1" and rig.export.gripper_id == "default"
    assert rig.capture["backends"]["imu"] == "icm42688_spi"         # pass-through (RECONCILIATION R3)
    assert rig.placeholder
    warnings = validate(rig)
    assert sum("placeholder" in w for w in warnings) == 3
    assert not any("coverage" in w for w in warnings)
    assert rig.coverage_report() == {"main": 1.0, "ultrawide": 1.0}
    # dump/load round trip: identical canonical content and fingerprint
    p = tmp_path / "rig.yaml"
    rig.dump(p)
    rig2 = load_rig(p)
    assert rig2.fingerprint() == rig.fingerprint()
    assert rig2.to_dict() == rig.to_dict()
    assert p.read_text().startswith("# Sensor rig description")
    # provenance does not change the fingerprint; geometry does
    assert rig.replace(calibration={"source": "x"}).fingerprint() == rig.fingerprint()
    assert len(rig.fingerprint()) == 64
    # bare names resolve inside sensor_rig/configs
    assert load_rig("rig_default.yaml").fingerprint() == rig.fingerprint()


def test_remap_cache_and_sizes(rig):
    t = rig.remap("main", "deploy")
    assert t is rig.remap("main", (960, 720))
    assert t.dst_size == (960, 720) and t.src_size == (1640, 1232) and t.coverage == 1.0
    assert rig.remap("ultrawide", "export").dst_size == (640, 480)
    assert rig.virtual_camera("main", "deploy").fx == pytest.approx(703.1714)
    assert rig.virtual_camera("main", "reference") is rig.virtual["main"].reference
    with pytest.raises(KeyError):
        rig.virtual_camera("main", "nope")
    inv = rig.inverse_remap("ultrawide")
    assert inv.src_size == (640, 480) and inv.dst_size == (1640, 1232) and inv is rig.inverse_remap("ultrawide")


def _err(raw, mutate, match):
    d = copy.deepcopy(raw)
    mutate(d)
    with pytest.raises(RigConfigError, match=match):
        load_rig(d)


def test_validation_errors(raw):
    def setp(path, value):
        def f(d):
            node = d
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
        return f

    _err(raw, setp(["cameras", "main", "bogus"], 1), r"^cameras\.main\.bogus: unknown key")
    _err(raw, setp(["bogus_top"], 1), r"^bogus_top: unknown key")
    _err(raw, setp(["cameras", "main", "T_imu_cam", "rotation"], {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]}),
         r"^cameras\.main\.T_imu_cam\.rotation: not a rotation")
    _err(raw, setp(["cameras", "ultrawide", "model", "distortion"], [0.1, 0.0, 0.0]),
         r"^cameras\.ultrawide\.model\.distortion: expected 4")
    _err(raw, setp(["virtual_cameras", "main", "outputs", "deploy"], [960, 700]),
         r"^virtual_cameras\.main\.outputs\.deploy: 960x700 is not a uniform scale")
    _err(raw, setp(["export", "world_offset"], {"translation_m": [0, 0, 0], "yaw_deg": 0.0}),
         r"^export\.world_offset: must not be the identity")
    _err(raw, setp(["imu", "odr_hz"], 300), r"^imu\.odr_hz: must be an ICM-42688-P ODR")
    _err(raw, setp(["imu", "gyro_range_dps"], 3000), r"^imu\.gyro_range_dps")
    _err(raw, setp(["imu", "accel_range_g"], 3), r"^imu\.accel_range_g")
    _err(raw, setp(["schema_version"], 2), r"^schema_version")
    _err(raw, setp(["cameras", "main", "resolution"], [1640, 0]), r"^cameras\.main\.resolution\[1\]")
    _err(raw, setp(["cameras", "main", "model", "intrinsics"], [1150.0, 1150.0, 3000.0, 615.5]),
         r"^cameras\.main\.model\.intrinsics: principal point")
    _err(raw, setp(["cameras", "main", "model", "type"], "omni"), r"^cameras\.main\.model\.type")
    _err(raw, setp(["virtual_cameras", "ultrawide", "source"], "nope"), r"^virtual_cameras\.ultrawide\.source")
    _err(raw, setp(["vio", "camera"], "nope"), r"^vio\.camera")
    _err(raw, setp(["vio", "trajectory_frame"], "virtual_main"), r"^vio\.trajectory_frame")
    _err(raw, lambda d: d["cameras"]["main"].pop("fps"), r"^cameras\.main\.fps: missing")
    _err(raw, lambda d: d["virtual_cameras"].pop("ultrawide"), r"^virtual_cameras\.ultrawide: missing")
    _err(raw, setp(["cameras", "main", "time", "offset_prior_ns"], 1.5), r"^cameras\.main\.time\.offset_prior_ns")
    _err(raw, setp(["capture"], [1, 2]), r"^capture: must be a mapping")
    # non-strict load: unknown keys become warnings
    d = copy.deepcopy(raw)
    d["cameras"]["main"]["bogus"] = 1
    r = load_rig(d, strict=False)
    assert any("cameras.main.bogus" in w for w in validate(r))
    # 4x4 matrix form and euler form are accepted
    d = copy.deepcopy(raw)
    d["cameras"]["main"]["T_imu_cam"] = {"matrix": make_T(rot_z(90), [0, 0.01, -0.015]).tolist()}
    d["virtual_cameras"]["main"]["rotation"] = {"euler_xyz_deg": [0.0, 1.5, 0.0]}
    r = load_rig(d)
    assert np.allclose(r.virtual["main"].R_phys_virt, rot_y(1.5), atol=1e-15)


def test_derived_extrinsics(raw):
    d = copy.deepcopy(raw)
    d["virtual_cameras"]["main"]["rotation"] = {"euler_xyz_deg": [0.5, 1.5, -2.0]}
    rig = load_rig(d)
    T_im = rig.cameras["main"].T_imu_cam
    T_iu = rig.cameras["ultrawide"].T_imu_cam
    assert np.allclose(rig.T("main", "ultrawide"), inv_T(T_im) @ T_iu, atol=1e-12)
    assert np.allclose(rig.T("main", "ultrawide") @ rig.T("ultrawide", "main"), np.eye(4), atol=1e-12)
    R = rig.virtual["main"].R_phys_virt
    assert np.allclose(rig.T("main", "virtual_main"), make_T(R), atol=1e-12)
    assert np.allclose(rig.T_imu("virtual_main"), T_im @ make_T(R), atol=1e-12)
    assert np.array_equal(rig.T("imu", "imu"), np.eye(4)) and np.array_equal(rig.T_imu("imu0"), np.eye(4))
    assert np.allclose(rig.T_S_virtual_main_arkit(), rig.T("imu", "virtual_main") @ D_CV_ARKIT, atol=1e-12)
    assert np.allclose(rig.T_S_virtual_main_arkit("ultrawide"),
                       rig.T("ultrawide", "virtual_main") @ D_CV_ARKIT, atol=1e-12)
    with pytest.raises(KeyError):
        rig.T("imu", "virtual_nope")
    # world offset: yaw about +y, then translation
    Tw = rig.export.T_world_offset
    assert np.allclose(Tw[:3, :3], rot_y(30.0)) and np.array_equal(Tw[:3, 3], [1.0, -0.2, 5.0])


def test_capture_time(raw):
    d = copy.deepcopy(raw)
    d["cameras"]["main"]["time"] = {"offset_prior_ns": -28_000_000, "kalibr_timeshift_cam_imu_s": 0.0012345678}
    rig = load_rig(d)
    off = -28_000_000 + round(0.0012345678 * 1e9)
    assert rig.cameras["main"].time.total_offset_ns == off == -26_765_432
    assert rig.capture_time_ns("main", 1_000_000_000) == 1_000_000_000 + off
    assert isinstance(rig.capture_time_ns("main", np.int64(5)), int)
    t = np.array([10**18 + 1, 10**18 + 3, 2**62 + 7], dtype=np.int64)
    out = rig.capture_time_ns("main", t)
    assert out.dtype == np.int64 and np.array_equal(out - t, np.full(3, off, np.int64))
    assert rig.capture_time_ns("imu", 123) == 123
    with pytest.raises(TypeError):
        rig.capture_time_ns("main", np.array([1.0e18]))
    with pytest.raises(KeyError):
        rig.capture_time_ns("nope", 1)


def test_virtual_K_json(rig):
    assert rig.virtual_camera("ultrawide", "export").K_float32_rows() == [
        [259.01538, 0, 320.2935], [0, 259.01538, 239.67455], [0, 0, 1]]
    rows = rig.virtual_camera("main", "export").K_float32_rows()
    assert np.array_equal(np.float32(rows), np.float32([[1406.3428, 0, 953.2503], [0, 1406.3428, 723.22003],
                                                        [0, 0, 1]]))


def test_with_virtual_reference(rig):
    K1278 = [[1278.4507, 0, 954.6028], [0, 1278.4507, 723.5324], [0, 0, 1]]
    r2 = rig.with_virtual_reference("main", K1278, (1920, 1440))
    assert r2.virtual["main"].reference.fx == 1278.4507
    assert r2.virtual["main"].outputs == rig.virtual["main"].outputs
    a, b = rig.to_dict(), r2.to_dict()
    assert a["virtual_cameras"]["main"]["reference"] != b["virtual_cameras"]["main"]["reference"]
    a["virtual_cameras"]["main"].pop("reference")
    b["virtual_cameras"]["main"].pop("reference")
    assert a == b
    assert r2.fingerprint() != rig.fingerprint()
    assert rig.virtual["main"].reference.fx == 1406.3428                 # original untouched
    assert r2.remap("main", "deploy") is not rig.remap("main", "deploy")  # fresh cache
