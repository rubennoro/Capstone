"""umiproc.sensors: the data-root copies of the rig file and SLAM config (copied when missing, never overwritten,
checked against the repo), the optional lens file, and the effective sensor values made from them; plus
lens-calibrate reading the data-root copies."""

import json

import numpy as np
import pytest
import yaml

from umiproc import sensors as S
from umiproc.dp.config import load_rig
from umiproc.lens import calibrate as lens_calibrate
from umiproc.lens.lensfile import dump_lens, lens_document
from umiproc.paths import repo_config, rig_file, rig_lens_file, rig_slam_file
from umiproc_testlib import tree_digest

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731
LENS_K = [580.0, 579.0, 641.0, 478.0]
LENS_D = [0.021, -0.006, 0.0012, -0.0003]


def _edit(path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))


def _write_lens(root, **over):
    doc = lens_document(rig_id=RIG, sensor="imx219", resolution=(1280, 960), intrinsics=LENS_K, distortion=LENS_D,
                        date="2026-11-16T18:00:00Z",
                        source={"pi_name": "pi-2026-11-16_episode_000123", "manifest_sha256": "ab"},
                        board={}, metrics={})
    doc.update(over)
    p = rig_lens_file(root, RIG)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_lens(doc))
    return p


def test_copies_missing_files_from_the_repo(tmp_path):
    lines = []
    files = S.ensure_rig_files(tmp_path, RIG, log=lines.append)
    assert rig_file(tmp_path, RIG).read_bytes() == repo_config("rigs", f"{RIG}.yaml").read_bytes()
    assert rig_slam_file(tmp_path, RIG).read_bytes() == repo_config("slam", f"{RIG}.yaml").read_bytes()
    assert (files.rig.state, files.slam.state, files.lens, files.warnings) == ("same", "same", None, [])
    assert len(lines) == 2 and all("copied" in ln for ln in lines)
    assert sorted(p.name for p in (tmp_path / "rigs").iterdir()) == [f"{RIG}.slam.yaml", f"{RIG}.yaml"]
    lines.clear()
    S.ensure_rig_files(tmp_path, RIG, log=lines.append)                  # nothing to do the second time
    assert lines == []


def test_an_edited_copy_is_kept_and_reported(tmp_path):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d["imu"]["noise"].update(gyroscope_noise_density=1e-4))
    _edit(rig_file(tmp_path, RIG), lambda d: d.update(notes="only a note"))           # notes are not compared
    _edit(rig_slam_file(tmp_path, RIG), lambda d: d["orb"].update(n_features=2000))
    _edit(rig_slam_file(tmp_path, RIG), lambda d: d["mask"].update(source="re-traced"))
    edited = tree_digest(tmp_path / "rigs")
    lines = []
    files = S.ensure_rig_files(tmp_path, RIG, log=lines.append)
    assert tree_digest(tmp_path / "rigs") == edited                       # never overwritten
    assert files.rig.state == files.slam.state == "differs"
    assert files.rig.differences == ["imu.noise.gyroscope_noise_density"]
    assert files.slam.differences == ["orb.n_features"]
    assert len(lines) == 2 and "imu.noise.gyroscope_noise_density" in lines[0] and "orb.n_features" in lines[1]
    assert "the processing uses this copy" in files.rig.describe()


def test_numbers_compare_by_value(tmp_path):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_slam_file(tmp_path, RIG), lambda d: d["mask"].update(
        polygons=[[[float(x), float(y)] for x, y in poly] for poly in d["mask"]["polygons"]]))
    assert S.check_rig_files(tmp_path, RIG).slam.state == "same"          # 176 == 176.0


def test_missing_everywhere(tmp_path):
    with pytest.raises(S.RigFilesError, match="no rig file for rig_nope.*rig_pi_v3"):
        S.ensure_rig_files(tmp_path, "rig_nope", log=quiet)
    with pytest.raises(S.RigFilesError, match="no SLAM config for rig_pi_v1"):
        S.ensure_rig_files(tmp_path, "rig_pi_v1", log=quiet)             # the repo has its rig file only
    with pytest.raises(S.RigFilesError, match="invalid rig id"):
        S.ensure_rig_files(tmp_path, "../x", log=quiet)
    assert list(tmp_path.iterdir()) == []                                 # nothing copied when one is missing


def test_check_is_read_only(tmp_path):
    files = S.check_rig_files(tmp_path / "root", RIG)
    assert files.rig.state == files.slam.state == "missing" and "not there yet" in files.rig.describe()
    S.ensure_rig_files(tmp_path / "root", RIG, log=quiet, copy_missing=False)
    assert not (tmp_path / "root").exists()


def test_rig_copy_rules(tmp_path):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d["cameras"]["ultrawide"].update(
        slam_mask={"polygons": [[[0, 0], [10, 0], [10, 10]]]}))
    with pytest.raises(S.RigFilesError, match="SLAM config's"):
        S.ensure_rig_files(tmp_path, RIG, log=quiet)
    rig_file(tmp_path, RIG).unlink()
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d.pop("gripper"))           # the copied check_rig_copy rule
    with pytest.raises(S.RigFilesError, match="no gripper section"):
        S.ensure_rig_files(tmp_path, RIG, log=quiet)


def test_nominal_values(tmp_path):
    s = S.load_sensors(tmp_path, RIG, log=quiet)
    rig = load_rig(repo_config("rigs", f"{RIG}.yaml"))
    assert s.lens is None and s.lens_provenance == S.NOMINAL == "nominal, uncalibrated"
    assert s.rig.cameras["ultrawide"].model == rig.cameras["ultrawide"].model
    np.testing.assert_array_equal(s.ultrawide.T_imu_cam, rig.cameras["ultrawide"].T_imu_cam)
    assert s.rig_path == rig_file(tmp_path, RIG) and s.slam_path == rig_slam_file(tmp_path, RIG)
    rec = json.loads(json.dumps(s.record()))
    uw, imu = rec["ultrawide"], rec["imu"]
    assert uw["lens"] == {"provenance": "nominal, uncalibrated", "rig_file_provenance": "nominal"}
    assert uw["model"] == {"type": "fisheye-equidistant", "intrinsics": [572.9, 572.9, 639.5, 479.5],
                           "distortion": [0.02, -0.005, 0.001, 0.0]}
    assert (uw["resolution"], uw["fps"], uw["time_offset_ns"], uw["timestamp_ref"]) == ([1280, 960], 30.0, 0,
                                                                                         "end_of_exposure")
    assert uw["T_imu_cam"][1][3] == pytest.approx(-0.02949)
    assert (imu["sensor"], imu["rate_hz"], imu["time_offset_ns"]) == ("lsm6dsox", 104.0, 0)
    assert imu["noise"] == {"gyroscope_noise_density": 6.63e-5, "gyroscope_random_walk": 2.0e-6,
                            "accelerometer_noise_density": 7.36e-4, "accelerometer_random_walk": 1.0e-4}
    assert imu["noise_for_slam"] == imu["noise"]                          # vio_inflation 1.0 / 1.0


def test_calibrated_lens_replaces_only_the_ultrawide_model(tmp_path):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    lens = _write_lens(tmp_path)
    s = S.load_sensors(tmp_path, RIG, log=quiet)
    assert s.lens_provenance == "calibrated" and s.lens_path == lens
    m = s.ultrawide.model
    assert (list(m.intrinsics), list(m.distortion), m.type) == (LENS_K, LENS_D, "fisheye-equidistant")
    nominal = s.nominal_rig
    assert nominal.cameras["ultrawide"].model.intrinsics == (572.9, 572.9, 639.5, 479.5)
    assert s.rig.cameras["main"] is nominal.cameras["main"]
    for attr in ("T_imu_cam", "resolution", "fps", "timestamp_ref", "time_offset_ns", "provenance", "sensor"):
        assert np.array_equal(getattr(s.ultrawide, attr), getattr(nominal.cameras["ultrawide"], attr))
    assert s.rig.imu is nominal.imu and np.array_equal(s.rig.T_cam_tcp, nominal.T_cam_tcp)
    lr = s.record()["ultrawide"]["lens"]
    assert lr["provenance"] == "calibrated" and lr["date"] == "2026-11-16T18:00:00Z"
    assert lr["source"]["pi_name"] == "pi-2026-11-16_episode_000123" and len(lr["file_sha256"]) == 64


@pytest.mark.parametrize("over, msg", [
    ({"rig_id": "rig_pi_v2"}, "it is for rig rig_pi_v2"),
    ({"resolution": [640, 480]}, "measured at 640x480"),
    ({"sensor": "imx519"}, "sensor imx519"),
])
def test_lens_file_for_another_camera_is_refused(tmp_path, over, msg):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    if "resolution" in over:                                              # a valid model at 640x480
        over["model"] = {"type": "fisheye-equidistant", "intrinsics": [290.0, 290.0, 319.5, 239.5],
                         "distortion": LENS_D}
    _write_lens(tmp_path, **over)
    with pytest.raises(S.SensorError, match=msg):
        S.load_sensors(tmp_path, RIG, log=quiet)


def test_design_rules(tmp_path):
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d["imu"].update(time_offset_ns=2_000_000))
    with pytest.raises(S.SensorError, match="time offset of 0"):
        S.load_sensors(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d["imu"].update(time_offset_ns=0))
    _edit(rig_slam_file(tmp_path, RIG), lambda d: d["mask"].update(resolution=[640, 480], polygons=[
        [[0, 0], [10, 0], [10, 10]]]))
    with pytest.raises(S.SensorError, match="mask.resolution"):
        S.load_sensors(tmp_path, RIG, log=quiet)
    rig_slam_file(tmp_path, RIG).write_text("schema: nope\n")
    with pytest.raises(S.SensorError, match="schema"):
        S.load_sensors(tmp_path, RIG, log=quiet)


def test_load_without_ensure_needs_the_copies(tmp_path):
    with pytest.raises(S.RigFilesError, match="does not exist yet"):
        S.load_sensors(tmp_path, RIG, ensure=False)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------------------- lens-calibrate

def test_lens_calibrate_reads_the_data_root_copies(tmp_path):
    rig, slam, rig_path, slam_path = lens_calibrate._load_configs(tmp_path, RIG, log=quiet)
    assert (rig_path, slam_path) == (repo_config("rigs", f"{RIG}.yaml"), repo_config("slam", f"{RIG}.yaml"))
    assert list(tmp_path.iterdir()) == []                                 # copies nothing itself
    S.ensure_rig_files(tmp_path, RIG, log=quiet)
    _edit(rig_file(tmp_path, RIG), lambda d: d["cameras"]["ultrawide"]["model"].update(
        intrinsics=[575.0, 575.0, 639.5, 479.5]))
    lines = []
    rig, slam, rig_path, slam_path = lens_calibrate._load_configs(tmp_path, RIG, log=lines.append)
    assert (rig_path, slam_path) == (rig_file(tmp_path, RIG), rig_slam_file(tmp_path, RIG))
    assert rig.cameras["ultrawide"].model.intrinsics[0] == 575.0 and str(rig.source_path) == str(rig_path)
    assert any("cameras.ultrawide.model.intrinsics" in ln for ln in lines)


def test_recording_mismatches(tmp_path):
    s = S.load_sensors(tmp_path, RIG, log=quiet)
    real = json.loads("""{"cameras": [{"file": "cam0.mp4", "model": "imx219", "size": [1280, 960], "fps": 30},
                                      {"file": "cam1.mp4", "model": "imx519", "size": [1600, 1200], "fps": 30}],
                          "imu": {"odr_hz": 104, "units": "raw int16 counts"}}""")      # as the Pi writes it
    assert S.recording_mismatches(s, real) == []
    bad = json.loads(json.dumps(real))
    bad["cameras"][0].update(size=[1640, 1232], fps=60)
    bad["imu"]["odr_hz"] = 208
    got = S.recording_mismatches(s, bad)
    assert len(got) == 3 and "[1640, 1232]" in got[0] and "60 fps" in got[1] and "208 Hz" in got[2]
    assert S.recording_mismatches(s, {"cameras": [], "imu": {}})[0] == "meta.json has 0 imx219 cameras (expected 1)"
