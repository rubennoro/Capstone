"""Raspberry Pi backend, pure logic: settings validation, full-FOV sensor-mode choice, Picamera2 configuration and
controls, camera selection, timestamp conversion; rig.yaml ``picamera2:`` schema and ``rig_pi5.yaml``."""

from __future__ import annotations

import copy
import types

import pytest
import yaml

from sensor_rig.clock import boot_minus_mono_ns
from sensor_rig.drivers.picamera2_logic import (
    KNOWN_SENSOR_MODES,
    PIXEL_FORMAT,
    Picamera2ConfigError,
    SensorMode,
    SensorTimestampConverter,
    build_configuration,
    build_controls,
    check_configured,
    choose_sensor_mode,
    known_modes,
    missing_frames,
    modes_from_picamera2,
    offline_mode_check,
    picamera2_params_from_camera,
    picamera2_params_from_dict,
    select_camera_num,
    unsupported_controls,
)
from sensor_rig.rig_config import DEFAULT_RIG_PATH, RigConfigError, load_rig, validate

PI5_RIG = DEFAULT_RIG_PATH.parent / "rig_pi5.yaml"
IMX708 = KNOWN_SENSOR_MODES["imx708"]
IMX219 = KNOWN_SENSOR_MODES["imx219"]
IMX477 = KNOWN_SENSOR_MODES["imx477"]
IMX296 = KNOWN_SENSOR_MODES["imx296"]


@pytest.fixture(scope="module")
def pi_rig():
    return load_rig(PI5_RIG)


def params(settings=None, res=(1536, 864), fps=30.0, sensor="imx708_wide", sensor_id=0):
    return picamera2_params_from_dict(settings or {}, sensor_id=sensor_id, resolution=res, fps=fps, sensor=sensor)


# ---------------------------------------------------------------------------------------------------- settings
def test_params_from_pi5_rig(pi_rig):
    p = picamera2_params_from_camera(pi_rig.cameras["main"])
    assert (p.camera_num, p.width, p.height, p.fps) == (0, 1014, 760, 30.0)
    assert p.model == "imx477" and p.sensor_size == (2028, 1520) and p.bit_depth == 12
    assert p.exposure_us == 4000 and p.analogue_gain == 4.0 and p.colour_gains == (2.0, 1.8)
    assert p.af == "none" and p.lens_position is None and p.timestamp_clock == "monotonic"
    assert p.frame_duration_us == 33333
    uw = picamera2_params_from_camera(pi_rig.cameras["ultrawide"])
    assert uw.camera_num == 1 and uw.af == "none" and uw.lens_position is None and uw.size == (1640, 1232)


def test_params_defaults_and_unknown_keys():
    p = params(sensor_id=1)
    assert p.camera_num == 1 and p.exposure_us is None and p.colour_gains is None and p.af == "none"
    assert p.buffer_count == 6 and not p.hflip and not p.vflip and p.noise_reduction is None
    with pytest.warns(UserWarning, match="not understood"):
        p = params({"exposure_us": 2000, "analogue_gain": 2.0, "shutter": 5})
    assert p.unknown_keys == ("shutter",)


@pytest.mark.parametrize("settings, match", [
    ({"exposure_us": 2000}, "give both"),
    ({"analogue_gain": 2.0}, "give both"),
    ({"exposure_us": 40000, "analogue_gain": 2.0}, "longer than the frame duration"),
    ({"exposure_us": 2000, "analogue_gain": 0.5}, "analogue_gain"),
    ({"af": "manual"}, "lens_position"),
    ({"af": "continuous"}, "af: must be one of"),
    ({"af": "none", "lens_position": 2.0}, "contradicts"),
    ({"lens_position": -1}, "lens_position"),
    ({"colour_gains": [1.0]}, "colour_gains"),
    ({"buffer_count": 1}, "buffer_count"),
    ({"hflip": 1}, "hflip"),
    ({"sensor_size": [2304]}, "sensor_size"),
    ({"timestamp_clock": "realtime"}, "timestamp_clock"),
    ({"noise_reduction": "max"}, "noise_reduction"),
    ({"camera_num": 1.5}, "camera_num"),
])
def test_params_rejects(settings, match):
    with pytest.raises(Picamera2ConfigError, match=match):
        params(settings)


def test_lens_position_implies_manual_focus():
    p = params({"lens_position": 0.0})
    assert p.af == "manual" and p.lens_position == 0.0


# ---------------------------------------------------------------------------------------------------- sensor modes
def test_choose_mode_imx708_full_fov():
    arr, modes = IMX708
    # 1536x864 output: the 1536x864 SENSOR mode is cropped, so the 2304x1296 binned mode is chosen
    m = choose_sensor_mode(modes, arr, (1536, 864), 30)
    assert m.size == (2304, 1296) and m.full_fov(arr)
    assert choose_sensor_mode(modes, arr, (2304, 1296), 50).size == (2304, 1296)
    assert choose_sensor_mode(modes, arr, (2304, 1296), 10, sensor_size=(4608, 2592)).size == (4608, 2592)


@pytest.mark.parametrize("out, fps, sensor_size, match", [
    ((1536, 864), 30, (1536, 864), "crops the sensor"),
    ((1440, 1080), 30, None, "no full-FOV sensor mode"),          # 4:3 output from a 16:9 sensor -> ISP crop
    ((1440, 1080), 30, (2304, 1296), "the ISP would crop"),
    ((1536, 864), 60, None, "no full-FOV sensor mode"),           # 2304x1296 tops out at 56 fps
    ((1536, 864), 60, (2304, 1296), "exceeds mode"),
    ((4608, 2592), 30, None, "no full-FOV sensor mode"),
    ((3000, 1688), 10, (2304, 1296), "upscaling"),
    ((1536, 864), 30, (2000, 1000), "no sensor mode 2000x1000"),
])
def test_choose_mode_imx708_rejects(out, fps, sensor_size, match):
    arr, modes = IMX708
    with pytest.raises(Picamera2ConfigError, match=match):
        choose_sensor_mode(modes, arr, out, fps, sensor_size=sensor_size)


def test_choose_mode_imx219():
    arr, modes = IMX219
    assert choose_sensor_mode(modes, arr, (1640, 1232), 30).size == (1640, 1232)
    assert choose_sensor_mode(modes, arr, (820, 616), 30).size == (1640, 1232)     # ISP downscale keeps the FOV
    with pytest.raises(Picamera2ConfigError, match="crops"):
        choose_sensor_mode(modes, arr, (1920, 1080), 30, sensor_size=(1920, 1080))
    with pytest.raises(Picamera2ConfigError):
        choose_sensor_mode(modes, arr, (1640, 1232), 45)                          # binned mode max ~41.9 fps
    with pytest.raises(Picamera2ConfigError, match="bit depth|12-bit"):
        choose_sensor_mode(modes, arr, (1640, 1232), 30, sensor_size=(1640, 1232), bit_depth=12)


def test_choose_mode_imx477_and_imx296():
    arr, modes = IMX477
    assert choose_sensor_mode(modes, arr, (1014, 760), 30).size == (2028, 1520)     # exact 1/2 ISP downscale
    assert choose_sensor_mode(modes, arr, (2028, 1520), 40).size == (2028, 1520)
    with pytest.raises(Picamera2ConfigError, match="crops"):
        choose_sensor_mode(modes, arr, (2028, 1080), 30, sensor_size=(2028, 1080))  # 16:9 crop of a 4:3 sensor
    with pytest.raises(Picamera2ConfigError, match="no full-FOV"):
        choose_sensor_mode(modes, arr, (1332, 990), 60)
    arr, modes = IMX296
    assert choose_sensor_mode(modes, arr, (1456, 1088), 60).size == (1456, 1088)
    assert choose_sensor_mode(modes, arr, (728, 544), 30).size == (1456, 1088)


def test_known_modes_and_conversion():
    assert known_modes("imx708_wide_noir") is IMX708 and known_modes("IMX219") is IMX219
    assert known_modes("ov5647") is None and known_modes(None) is None
    raw = [{"format": "SRGGB10_CSI2P", "bit_depth": 10, "size": (2304, 1296), "fps": 56.03,
            "crop_limits": (0, 0, 4608, 2592), "exposure_limits": (9, 77208384)},
           {"format": "YUYV"}]
    assert modes_from_picamera2(raw) == [SensorMode((2304, 1296), 10, (0, 0, 4608, 2592), 56.03, "SRGGB10_CSI2P")]


def test_offline_mode_check(pi_rig):
    assert offline_mode_check(picamera2_params_from_camera(pi_rig.cameras["main"])).size == (2028, 1520)
    assert offline_mode_check(picamera2_params_from_camera(pi_rig.cameras["ultrawide"])).size == (1640, 1232)
    assert offline_mode_check(params(sensor="fakecam")) is None
    with pytest.raises(Picamera2ConfigError, match="crops"):
        offline_mode_check(params({"sensor_size": [1536, 864]}))


# ---------------------------------------------------------------------------------------------------- configuration
def test_controls_locked_and_auto():
    p = params({"exposure_us": 3000, "analogue_gain": 2.5, "colour_gains": [2.1, 1.7], "lens_position": 2.0,
                "noise_reduction": "off"})
    c = build_controls(p, scaler_crop=(0, 0, 4608, 2592))
    assert c == {"FrameDurationLimits": (33333, 33333), "AeEnable": False, "ExposureTime": 3000,
                 "AnalogueGain": 2.5, "AwbEnable": False, "ColourGains": (2.1, 1.7), "AfMode": 0,
                 "LensPosition": 2.0, "NoiseReductionMode": 0, "ScalerCrop": (0, 0, 4608, 2592)}
    auto = build_controls(params(fps=25))
    assert auto == {"FrameDurationLimits": (40000, 40000), "AeEnable": True, "AwbEnable": True}


def test_build_configuration(pi_rig):
    p = picamera2_params_from_camera(pi_rig.cameras["main"])
    mode = offline_mode_check(p)
    kw = build_configuration(p, mode)
    assert kw["main"] == {"size": (1014, 760), "format": "RGB888"} and PIXEL_FORMAT == "RGB888"
    assert kw["sensor"] == {"output_size": (2028, 1520), "bit_depth": 12}
    assert kw["buffer_count"] == 6 and kw["queue"] is False
    assert kw["transform"] == {"hflip": False, "vflip": False}
    assert "AfMode" not in kw["controls"] and kw["controls"]["FrameDurationLimits"] == (33333, 33333)
    assert kw["controls"]["AeEnable"] is False and kw["controls"]["AwbEnable"] is False
    good = {"main": {"size": (1014, 760), "format": "RGB888"}, "sensor": {"output_size": (2028, 1520)}}
    assert check_configured(good, p, mode) == []
    bad = copy.deepcopy(good)
    bad["main"].update(size=(1024, 760), format="BGR888")
    bad["sensor"]["output_size"] = (2028, 1080)
    probs = check_configured(bad, p, mode)
    assert len(probs) == 3 and "calibration resolution" in probs[0] and "BGR byte order" in probs[1]
    assert unsupported_controls({"AfMode": 0, "AeEnable": True}, {"AeEnable": (0, 1, 1)}) == ["AfMode"]


def test_select_camera_num():
    info = [{"Model": "imx708_wide"}, {"Model": "imx219"}]
    assert select_camera_num(info, 0) == 0 and select_camera_num(info, 1, "imx219") == 1
    assert select_camera_num(info, 0, "imx219") == 1        # numbering changed: unique model match wins
    assert select_camera_num(info, 1, "imx708") == 0
    with pytest.raises(Picamera2ConfigError, match="not present"):
        select_camera_num(info, 2)
    with pytest.raises(Picamera2ConfigError, match="no unique match"):
        select_camera_num(info, 0, "imx477")
    with pytest.raises(Picamera2ConfigError, match="no unique match"):
        select_camera_num([{"Model": "imx219"}, {"Model": "imx219"}], 3, "imx219")
    with pytest.raises(Picamera2ConfigError, match="no cameras"):
        select_camera_num([], 0)


def test_timestamps():
    assert SensorTimestampConverter()(1_234_567_890_123) == 1_234_567_890_123
    assert SensorTimestampConverter("monotonic", 5_000)(10_000) == 10_000       # offset ignored for MONOTONIC
    assert SensorTimestampConverter("boottime", 5_000)(10_000) == 5_000
    with pytest.raises(ValueError):
        SensorTimestampConverter("realtime")
    off = boot_minus_mono_ns()
    assert isinstance(off, int) and off >= -1_000_000        # BOOTTIME >= MONOTONIC (suspended time)
    period = 1e9 / 30
    assert missing_frames(None, 5, 30) == 0
    assert missing_frames(0, int(period), 30) == 0
    assert missing_frames(0, int(1.6 * period), 30) == 1
    assert missing_frames(0, int(4 * period), 30) == 3


# ---------------------------------------------------------------------------------------------------- rig.yaml
def test_pi5_rig_loads_validates_round_trips(pi_rig, tmp_path):
    assert pi_rig.capture["backends"] == {"main": "picamera2", "ultrawide": "picamera2", "imu": "icm42688_spi"}
    assert pi_rig.cameras["main"].sensor_mode is None and pi_rig.cameras["main"].resolution == (1014, 760)
    assert pi_rig.coverage_report() == {"main": 1.0, "ultrawide": 1.0}
    w = validate(pi_rig)
    assert sum("placeholder" in x for x in w) == 3 and len(w) == 3, w
    pi_rig.dump(tmp_path / "rig.yaml")
    again = load_rig(tmp_path / "rig.yaml")
    assert again.cameras["main"].picamera2 == pi_rig.cameras["main"].picamera2
    assert again.fingerprint() == pi_rig.fingerprint()
    # cameras without picamera2 settings serialise exactly as before (existing fingerprints unchanged)
    assert "picamera2" not in load_rig(DEFAULT_RIG_PATH).to_dict()["cameras"]["main"]


def _pi_raw():
    return yaml.safe_load(PI5_RIG.read_text())


def test_rig_picamera2_schema_errors_and_warnings():
    raw = _pi_raw()
    raw["cameras"]["main"]["picamera2"]["exposure_us"] = 50000
    with pytest.raises(RigConfigError, match=r"cameras\.main\.picamera2\.exposure_us"):
        load_rig(raw)
    raw = _pi_raw()
    raw["cameras"]["main"]["picamera2"]["iso"] = 100
    rig = load_rig(raw)                                   # strict load still only warns inside picamera2
    assert any("cameras.main.picamera2.iso: unknown key" in x for x in rig.load_warnings)
    raw = _pi_raw()
    raw["cameras"]["main"]["picamera2"]["sensor_size"] = [2028, 1080]   # cropped mode
    del raw["cameras"]["ultrawide"]["picamera2"]["colour_gains"]
    w = validate(load_rig(raw))
    assert any("cameras.main.picamera2" in x and "crops" in x for x in w)
    assert any("ultrawide.picamera2: auto white balance" in x for x in w)


def test_make_camera_picamera2_is_lazy(pi_rig):
    from sensor_rig.drivers.base import BackendError, make_camera

    cam = make_camera(pi_rig, "main")                    # rig.capture.backends.main == picamera2
    assert type(cam).__name__ == "Picamera2Camera" and (cam.width, cam.height, cam.fps) == (1014, 760, 30.0)
    assert cam.info()["backend"] == "picamera2"
    with pytest.raises(BackendError, match="crops"):
        make_camera(pi_rig, "main", settings_overrides={"sensor_size": [2028, 1080]})
    stub = types.SimpleNamespace(cameras={"main": types.SimpleNamespace(
        resolution=(640, 480), fps=30.0, sensor_id=0, sensor="fakecam", argus={})}, capture={})
    assert make_camera(stub, "main", backend="picamera2").params.camera_num == 0
