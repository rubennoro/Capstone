from pathlib import Path

import numpy as np
import pytest
import yaml

from sensor_rig.calib.cli import main as calib_main
from sensor_rig.calib.kalibr_import import auto_cam_map, import_kalibr, read_kalibr_camchain, read_kalibr_imu
from sensor_rig.camera_model import FisheyeEquidistant, PinholeRadtan
from sensor_rig.rig_config import DEFAULT_RIG_PATH, load_rig, validate
from sensor_rig.transforms import inv_T

DATA = Path(__file__).resolve().parent / "data" / "kalibr"
CAMCHAIN = DATA / "camchain-imucam.yaml"
IMU = DATA / "imu.yaml"


@pytest.fixture(scope="module")
def rig():
    return load_rig(DEFAULT_RIG_PATH)


def _write_chain(tmp_path, mutate):
    d = yaml.safe_load(CAMCHAIN.read_text())
    mutate(d)
    p = tmp_path / "camchain-imucam.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_import_fixture(rig, tmp_path):
    raw = read_kalibr_camchain(CAMCHAIN)
    assert list(raw) == ["cam0", "cam1"]
    assert auto_cam_map(raw, list(rig.cameras)) == {"cam0": "ultrawide", "cam1": "main"}
    new, log = import_kalibr(rig, CAMCHAIN, IMU, date="2026-10-15T00:00:00+00:00")
    main, uw = new.cameras["main"], new.cameras["ultrawide"]
    assert isinstance(main.model, PinholeRadtan) and isinstance(uw.model, FisheyeEquidistant)
    assert np.allclose([main.model.fx, main.model.fy, main.model.cx, main.model.cy],
                       [1162.3, 1161.9, 815.2, 619.4], atol=1e-9, rtol=0)
    assert np.allclose(main.model.dist, [-0.2913, 0.0874, 0.00041, -0.00023], atol=1e-12, rtol=0)
    assert np.allclose(uw.model.dist, [0.0213, -0.0071, 0.0016, -0.00028], atol=1e-12, rtol=0)
    for cid, cam in (("cam0", uw), ("cam1", main)):
        assert np.allclose(cam.T_imu_cam, inv_T(np.array(raw[cid]["T_cam_imu"])), atol=1e-9, rtol=0)
        assert not cam.placeholder
    assert main.time.kalibr_timeshift_cam_imu_s == 0.0018 and uw.time.kalibr_timeshift_cam_imu_s == 0.0021
    assert main.time.offset_prior_ns == rig.cameras["main"].time.offset_prior_ns
    assert main.time.total_offset_ns == -30_000_000 + 1_800_000
    # IMU noise imported, inflation kept, placeholders cleared
    n = new.imu.noise
    assert (n.accelerometer_noise_density, n.accelerometer_random_walk) == (0.00071, 0.000123)
    assert (n.gyroscope_noise_density, n.gyroscope_random_walk) == (5.2e-05, 1.4e-06)
    assert (n.inflation_noise, n.inflation_random_walk) == (5.0, 10.0)
    assert read_kalibr_imu(IMU).gyroscope_noise_density == 5.2e-05
    assert not new.placeholder and not any("placeholder" in w for w in validate(new))
    # untouched sections
    assert new.virtual["main"].reference.fx == rig.virtual["main"].reference.fx
    assert new.export == rig.export and new.vio == rig.vio and new.capture == rig.capture
    # chain check passes; provenance block written
    assert any("T_cn_cnm1" in line and "consistent" in line for line in log)
    assert not any(line.startswith("WARNING") for line in log)
    assert new.calibration["source"] == "kalibr" and len(new.calibration["camchain"]["sha256"]) == 64
    assert new.calibration["cam_map"] == {"cam0": "ultrawide", "cam1": "main"}
    # the result dumps to a loadable rig with the same fingerprint
    out = tmp_path / "rig_v1.yaml"
    new.dump(out)
    assert load_rig(out).fingerprint() == new.fingerprint() != rig.fingerprint()
    assert np.allclose(new.T("main", "ultrawide"), np.array(raw["cam1"]["T_cn_cnm1"]), atol=1e-9, rtol=0)


def test_chain_check_fails_on_perturbed_fixture(rig, tmp_path):
    def perturb(d):
        d["cam1"]["T_cn_cnm1"][0][3] += 0.002          # 2 mm
    p = _write_chain(tmp_path, perturb)
    _, log = import_kalibr(rig, p)
    assert any(line.startswith("WARNING") and "T_cn_cnm1" in line for line in log)


def test_explicit_map_and_camera_only_chain(rig, tmp_path):
    def strip(d):
        for c in d.values():
            c.pop("T_cam_imu")
            c.pop("timeshift_cam_imu")
            c["rostopic"] = "/a/b"
        d["cam1"].pop("T_cn_cnm1")
    p = _write_chain(tmp_path, strip)
    with pytest.raises(ValueError, match="--map"):
        import_kalibr(rig, p)
    new, log = import_kalibr(rig, p, cam_map={"cam0": "ultrawide", "cam1": "main"})
    assert new.cameras["main"].placeholder                       # extrinsics not calibrated yet
    assert np.array_equal(new.cameras["main"].T_imu_cam, rig.cameras["main"].T_imu_cam)
    assert new.cameras["main"].model.fx == 1162.3
    assert any("camera-only" in line for line in log)


def test_resolution_mismatch_and_unsupported_model(rig, tmp_path):
    p = _write_chain(tmp_path, lambda d: d["cam1"].__setitem__("resolution", [1280, 720]))
    with pytest.raises(ValueError, match="resolution"):
        import_kalibr(rig, p)
    p = _write_chain(tmp_path, lambda d: d["cam0"].__setitem__("camera_model", "ds"))
    with pytest.raises(NotImplementedError, match="pinhole-equi"):
        import_kalibr(rig, p)
    p = _write_chain(tmp_path, lambda d: d["cam0"].__setitem__("distortion_model", "fov"))
    with pytest.raises(NotImplementedError):
        import_kalibr(rig, p)


def test_cli_import_and_validate(tmp_path, capsys):
    out = tmp_path / "rig_v1.yaml"
    rc = calib_main(["calib-import", "--rig", "rig_default.yaml", "--camchain", str(CAMCHAIN), "--imu", str(IMU),
                     "--out", str(out)])
    assert rc == 0 and load_rig(out).cameras["main"].model.fx == 1162.3
    assert calib_main(["rig-validate", str(out)]) == 0
    text = capsys.readouterr().out
    assert "coverage main      = 1.0000" in text and "warning" not in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(out.read_text().replace("schema_version: 1", "schema_version: 7"))
    assert calib_main(["rig-validate", str(bad)]) == 1
