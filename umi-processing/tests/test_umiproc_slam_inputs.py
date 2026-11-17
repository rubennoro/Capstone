"""umiproc.slam_inputs: the ORB-SLAM3 settings file (every key the fork's Settings parser needs, with the type it
accepts, and the values from the effective sensors), the gripper mask PNG (pixel-identical to the one used in the
2026-11-16 experiments), and the fingerprint.

Types are checked with OpenCV's own FileStorage (``isReal`` / ``isInt`` / ``isString`` / ``mat``: what
``Settings::readParameter`` checks), and, when the fork's source is there, against ``src/Settings.cc`` itself. The
slow test runs the real umiproc_slam on the file and reads back what its parser printed.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from umiproc import slam_inputs as SI
from umiproc.sensors import ensure_rig_files, load_sensors
from umiproc.paths import rig_file, rig_lens_file, rig_slam_file
from umiproc.lens.lensfile import dump_lens, lens_document
from umiproc.slam.runner import default_binary, default_fork_dir, default_vocabulary

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731

#: Every key Settings.cc reads for a monocular-inertial KannalaBrandt8 camera, required ones only (the parser aborts
#: without them), with the type it demands: readCamera1 (KannalaBrandt8 branch), readImageInfo, readIMU, readORB,
#: readViewer (src/Settings.cc of cheng-chi/ORB_SLAM3 b741dca). Plus File.version (System.cc picks this parser).
REQUIRED = {
    "File.version": "string", "Camera.type": "string",
    **{f"Camera1.{k}": "float" for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")},
    "Camera.height": "int", "Camera.width": "int", "Camera.fps": "int", "Camera.RGB": "int",
    **{f"IMU.{k}": "float" for k in ("NoiseGyro", "NoiseAcc", "GyroWalk", "AccWalk", "Frequency")},
    "IMU.T_b_c1": "cv::Mat",
    "ORBextractor.nFeatures": "int", "ORBextractor.scaleFactor": "float", "ORBextractor.nLevels": "int",
    "ORBextractor.iniThFAST": "int", "ORBextractor.minThFAST": "int",
    **{f"Viewer.{k}": "float" for k in ("KeyFrameSize", "KeyFrameLineWidth", "GraphLineWidth", "PointSize",
                                        "CameraSize", "CameraLineWidth", "ViewpointX", "ViewpointY", "ViewpointZ",
                                        "ViewpointF")},
}
#: Optional keys we write on purpose (their parser default differs from what we want, or UMI sets them).
OPTIONAL = {"IMU.InsertKFsWhenLost": "int", "System.thFarPoints": "float", "Viewer.imageViewScale": "float"}

#: The mask used in the experiments (umi_mask_1280x960.png, 255 = masked): sha256 of "960x1280:" + its pixels, and
#: its masked pixel count. The file itself lives in a scratch folder: compared directly when it is still there.
EXPERIMENT_MASK_SHA256 = "85a65194e8255e2b85641227265c1af63d74365dec59ee0cd87248158d4876bf"
EXPERIMENT_MASK_COUNT = 358232
EXPERIMENT_MASK = Path("/tmp/claude-1000/-home-ruben-Capstone-data-work/6b9540ba-9ac3-4e29-baef-c4535f45cc0e/"
                       "scratchpad/umi_mask_1280x960.png")


@pytest.fixture
def sensors(tmp_path):
    return load_sensors(tmp_path / "root", RIG, log=quiet)


def _fs(text: str, tmp_path: Path):
    p = tmp_path / "settings_read.yaml"
    p.write_text(text)
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    return fs


def _keys(text: str) -> list[str]:
    return re.findall(r"^([A-Za-z][\w.]*):", text, re.M)


def _type_of(node) -> str:
    if node.isString():
        return "string"
    if node.isInt():
        return "int"
    if node.isReal():
        return "float"
    if node.isMap():
        return "cv::Mat"
    return f"type {node.type()}"


def test_every_key_with_the_parsers_type(sensors, tmp_path):
    text = SI.orbslam3_settings_text(sensors)
    keys = _keys(text)
    assert len(keys) == len(set(keys))
    assert set(keys) == set(REQUIRED) | set(OPTIONAL)                     # nothing missing, nothing unknown
    fs = _fs(text, tmp_path)
    for k, t in {**REQUIRED, **OPTIONAL}.items():
        assert _type_of(fs.getNode(k)) == t, k
    assert fs.getNode("File.version").string() == "1.0"                  # System.cc: Settings parser only for "1.0"
    assert fs.getNode("Camera.type").string() == "KannalaBrandt8"
    T = fs.getNode("IMU.T_b_c1").mat()
    assert T.dtype == np.float32 and T.shape == (4, 4)                    # Converter::toSophus reads at<float>


def test_types_agree_with_the_forks_source(sensors):
    src = default_fork_dir() / "src" / "Settings.cc"
    if not src.is_file():
        pytest.skip(f"no {src}")
    calls = re.findall(r'readParameter<([\w:]+)>\(fSettings,\s*"([^"]+)",\s*found(\s*,\s*false)?\)',
                       src.read_text())
    parsed = {}
    for typ, key, optional in calls:
        parsed.setdefault(key, set()).add((typ, bool(optional)))
    for key, typ in {**REQUIRED, **OPTIONAL}.items():
        if key == "File.version":
            continue
        assert key in parsed, f"{key} is not read by {src}"
        assert {t for t, _ in parsed[key]} == {typ}, (key, parsed[key])
    for key in REQUIRED:
        if key != "File.version":
            assert any(not opt for _, opt in parsed[key]), f"{key} is optional in {src}"
    for key in OPTIONAL:
        assert all(opt for _, opt in parsed[key]), f"{key} is required in {src}"
    system = (src.parent / "System.cc").read_text()
    assert 'node.string() == "1.0"' in system and 'fsSettings["File.version"]' in system


def test_values_from_the_effective_sensors(sensors, tmp_path):
    fs = _fs(SI.orbslam3_settings_text(sensors), tmp_path)
    cam = sensors.ultrawide
    got = [fs.getNode(f"Camera1.{k}").real() for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")]
    np.testing.assert_allclose(got, list(cam.model.intrinsics) + list(cam.model.distortion), rtol=1e-9, atol=1e-12)
    assert (int(fs.getNode("Camera.width").real()), int(fs.getNode("Camera.height").real())) == (1280, 960)
    assert fs.getNode("Camera.fps").real() == 30 and fs.getNode("Camera.RGB").real() == 0
    assert fs.getNode("Camera.newWidth").empty() and fs.getNode("Camera.newHeight").empty()     # no scaling
    np.testing.assert_allclose(fs.getNode("IMU.T_b_c1").mat(), cam.T_imu_cam, atol=1e-7)
    noise = sensors.rig.imu.noise.for_vio()
    for key, v in (("NoiseGyro", noise.gyroscope_noise_density), ("NoiseAcc", noise.accelerometer_noise_density),
                   ("GyroWalk", noise.gyroscope_random_walk), ("AccWalk", noise.accelerometer_random_walk)):
        assert fs.getNode(f"IMU.{key}").real() == pytest.approx(v, rel=1e-9)
    assert (fs.getNode("IMU.NoiseGyro").real(), fs.getNode("IMU.NoiseAcc").real()) == (6.63e-5, 7.36e-4)
    assert fs.getNode("IMU.Frequency").real() == 104.0
    assert fs.getNode("IMU.InsertKFsWhenLost").real() == 1
    orb = sensors.slam.orb
    assert [fs.getNode(f"ORBextractor.{k}").real() for k in ("nFeatures", "scaleFactor", "nLevels", "iniThFAST",
                                                              "minThFAST")] == [
        orb.n_features, orb.scale_factor, orb.n_levels, orb.ini_th_fast, orb.min_th_fast]
    assert fs.getNode("System.thFarPoints").real() == 20.0


def test_vio_inflation_and_the_lens_file_are_used(tmp_path):
    root = tmp_path / "root"
    ensure_rig_files(root, RIG, log=quiet)
    d = yaml.safe_load(rig_file(root, RIG).read_text())
    d["imu"]["vio_inflation"] = {"noise_density": 10.0, "random_walk": 3.0}
    rig_file(root, RIG).write_text(yaml.safe_dump(d, sort_keys=False))
    rig_lens_file(root, RIG).write_text(dump_lens(lens_document(
        rig_id=RIG, sensor="imx219", resolution=(1280, 960), intrinsics=[580.5, 579.25, 641.0, 478.0],
        distortion=[0.021, -0.006, 0.0012, -0.0003], date="2026-11-16T18:00:00Z", source={}, board={},
        metrics={})))
    text = SI.orbslam3_settings_text(load_sensors(root, RIG, log=quiet))
    fs = _fs(text, tmp_path)
    assert [fs.getNode(f"Camera1.{k}").real() for k in ("fx", "fy", "k4")] == [580.5, 579.25, -0.0003]
    assert fs.getNode("IMU.NoiseGyro").real() == pytest.approx(6.63e-4)
    assert fs.getNode("IMU.AccWalk").real() == pytest.approx(3.0e-4)
    assert "lens: calibrated 2026-11-16T18:00:00Z" in text and str(root) not in text      # no paths in the file


def test_vio_inflation_ab_setting_scales_all_four(tmp_path, monkeypatch):
    """The first mapping session's IMU-noise A/B (docs/DESIGN.md "SLAM"): imu.vio_inflation {10, 10} in the data-root
    rig copy multiplies both densities and both random walks by 10 in the settings file (the default 1.0/1.0
    writes the datasheet values), and gives another fingerprint."""
    root = tmp_path / "root"
    ensure_rig_files(root, RIG, log=quiet)
    binary = fake_build(tmp_path / "fork", monkeypatch)
    keys = ("NoiseGyro", "NoiseAcc", "GyroWalk", "AccWalk")
    datasheet = (6.63e-5, 7.36e-4, 2.0e-6, 1.0e-4)                  # configs/rigs/rig_pi_v3.yaml imu.noise

    def written(name):
        sensors = load_sensors(root, RIG, log=quiet)
        fs = _fs(SI.orbslam3_settings_text(sensors), tmp_path / name)
        return [fs.getNode(f"IMU.{k}").real() for k in keys], _fp(sensors, tmp_path / name, binary)[1]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base, h_base = written("a")
    assert base == pytest.approx(datasheet, rel=1e-9)
    d = yaml.safe_load(rig_file(root, RIG).read_text())
    assert d["imu"]["vio_inflation"] == {"noise_density": 1.0, "random_walk": 1.0}       # the documented default
    d["imu"]["vio_inflation"] = {"noise_density": 10, "random_walk": 10}
    rig_file(root, RIG).write_text(yaml.safe_dump(d, sort_keys=False))
    inflated, h_inflated = written("b")
    assert inflated == pytest.approx([10 * v for v in datasheet], rel=1e-9)
    assert h_inflated != h_base


@pytest.mark.parametrize("v", [1.0, 100, 0, -0.0, 1e-6, 2e-6, 6.63e-5, 1e16, -3.5, 0.1 + 0.2, 123456789.0])
def test_reals_are_reals_for_opencv(v, tmp_path):
    s = SI._real(v)
    fs = _fs(f"%YAML:1.0\n---\nx: {s}\n", tmp_path)
    assert fs.getNode("x").isReal(), s
    assert fs.getNode("x").real() == pytest.approx(float(v), rel=1e-8)


@pytest.mark.parametrize("v", [float("nan"), float("inf")])
def test_non_finite_is_refused(v):
    with pytest.raises(ValueError):
        SI._real(v)


# --------------------------------------------------------------------------------------------- mask

def test_mask_is_the_experiments_mask(sensors, tmp_path):
    m = SI.mask_png(sensors.slam)
    assert m.shape == (960, 1280) and m.dtype == np.uint8 and set(np.unique(m)) == {0, 255}
    assert SI.mask_pixel_sha256(m) == EXPERIMENT_MASK_SHA256 and int((m == 255).sum()) == EXPERIMENT_MASK_COUNT
    p = SI.write_mask_png(sensors.slam, tmp_path / "mask.png")
    back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint8 and back.ndim == 2 and np.array_equal(back, m)          # 8-bit grey PNG
    if EXPERIMENT_MASK.is_file():
        assert np.array_equal(cv2.imread(str(EXPERIMENT_MASK), cv2.IMREAD_UNCHANGED), back)
    # what the front-end does with it: nonzero = masked (im.setTo(0, mask))
    frame = np.full((960, 1280), 200, np.uint8)
    frame[back != 0] = 0
    assert frame[959, 671] == 0 and frame[0, 0] == 200


# --------------------------------------------------------------------------------------------- fingerprint

PATCH_NAMES = ("0001-a.patch", "0002-b.patch")


def fake_build(base: Path, monkeypatch, patches=PATCH_NAMES, binary_bytes=b"binary v1", lib_bytes=b"lib v1",
               library_line=True) -> Path:
    """A native/build.sh-like folder: ``base/umiproc_build/{umiproc_slam,BUILD_INFO}`` and
    ``base/lib/libORB_SLAM3.so``; ``ldd`` is replaced by "the binary loads base/lib/libORB_SLAM3.so". Returns the
    binary."""
    out, lib = base / "umiproc_build", base / "lib" / "libORB_SLAM3.so"
    out.mkdir(parents=True, exist_ok=True)
    lib.parent.mkdir(parents=True, exist_ok=True)
    (out / "umiproc_slam").write_bytes(binary_bytes)
    lib.write_bytes(lib_bytes)
    write_build_info(out / "BUILD_INFO", lib if library_line else None, patches)
    monkeypatch.setattr(SI, "loaded_orbslam3_library", lambda b: lib)
    return out / "umiproc_slam"


def write_build_info(path: Path, library, patches, built="2026-11-16T12:00:00Z"):
    lines = ["fork https://github.com/cheng-chi/ORB_SLAM3.git b741dca39015330ef4bcc3a85f89493503ade04b",
             "pangolin d484494645cb7361374ac0ef6b27e9ee6feffbd7"]
    if library is not None:
        lines.append(f"library {library}")
    lines += [f"patch {hashlib.sha256(n.encode()).hexdigest()[:16]} {n}" for n in patches]
    lines += ["opencv /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5d", "machine x86_64 test", f"built {built}"]
    path.write_text("\n".join(lines) + "\n")


def _fp(sensors, out: Path, binary: Path):
    out.mkdir(parents=True, exist_ok=True)
    settings = SI.write_orbslam3_settings(sensors, out / SI.SETTINGS_NAME)
    mask = SI.write_mask_png(sensors.slam, out / SI.MASK_NAME)
    rec = SI.slam_fingerprint(sensors, settings, mask, binary)
    return rec, SI.fingerprint_hash(rec)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fingerprint_record(sensors, tmp_path, monkeypatch):
    binary = fake_build(tmp_path / "fork", monkeypatch)
    rec, h = _fp(sensors, tmp_path / "a", binary)
    rec2, h2 = _fp(sensors, tmp_path / "b", binary)                      # same inputs elsewhere: same hash
    assert h == h2 and len(h) == 64 and json.loads(json.dumps(rec)) == rec
    assert rec["format"] == "umiproc.slam_fingerprint.v2"
    assert set(rec) == {"format", "sensors", "slam_config", "mask", "settings", "binary", "build_info", "patches",
                        "library"}
    assert rec["sensors"] == sensors.record()
    assert rec["mask"]["pixel_sha256"] == EXPERIMENT_MASK_SHA256 and rec["mask"]["size"] == [1280, 960]
    assert rec["binary"] == {"path": str(binary), "sha256": _sha(binary)}
    info = binary.parent / "BUILD_INFO"
    assert rec["build_info"] == {"path": str(info), "sha256": SI.build_info_sha256(info)}
    assert rec["patches"] == [{"name": n, "sha256_16": hashlib.sha256(n.encode()).hexdigest()[:16]}
                              for n in PATCH_NAMES]                    # from BUILD_INFO, not native/patches/
    lib = tmp_path / "fork" / "lib" / "libORB_SLAM3.so"
    assert rec["library"] == {"path": str(lib), "realpath": str(lib), "sha256": _sha(lib)}
    assert rec["slam_config"]["table_tag"] == {"dictionary": "DICT_4X4_50", "id": 13, "size_m": 0.16}


def test_fingerprint_changes_with_every_input(tmp_path, monkeypatch):
    root = tmp_path / "root"
    binary = fake_build(tmp_path / "fork", monkeypatch)
    info = binary.parent / "BUILD_INFO"
    lib = tmp_path / "fork" / "lib" / "libORB_SLAM3.so"
    base = _fp(load_sensors(root, RIG, log=quiet), tmp_path / "0", binary)[1]
    hashes = {"base": base}

    def again(name):
        hashes[name] = _fp(load_sensors(root, RIG, log=quiet), tmp_path / name, binary)[1]

    def edit(path, fn):
        d = yaml.safe_load(path.read_text())
        fn(d)
        path.write_text(yaml.safe_dump(d, sort_keys=False))

    again("same")
    binary.write_bytes(b"binary v2")
    again("binary")
    lib.write_bytes(b"lib v2")                                           # a patch change lands in the library only
    again("library")
    write_build_info(info, lib, PATCH_NAMES + ("0003-c.patch",))
    again("patches")
    write_build_info(info, lib, PATCH_NAMES + ("0003-c.patch",), built="2026-11-16T13:00:00Z")
    rerun = _fp(load_sensors(root, RIG, log=quiet), tmp_path / "rerun", binary)[1]
    assert rerun == hashes["patches"]                       # a build.sh re-run only rewrites the "built" time
    write_build_info(info, lib, PATCH_NAMES + ("0003-c.patch",), built="2026-11-16T13:00:00Z")
    info.write_text(info.read_text().replace("machine x86_64 test", "machine aarch64 test"))
    again("build_info")
    edit(rig_slam_file(root, RIG), lambda d: d["orb"].update(n_features=1250))
    again("orb")
    edit(rig_slam_file(root, RIG), lambda d: d["mask"]["polygons"][0].__setitem__(0, [180, 960]))
    again("mask")
    edit(rig_slam_file(root, RIG), lambda d: d["table_tag"].update(size_m=0.15))
    again("tag")
    edit(rig_file(root, RIG), lambda d: d["imu"]["noise"].update(accelerometer_random_walk=2e-4))
    again("noise")
    edit(rig_file(root, RIG), lambda d: d["imu"].update(vio_inflation={"noise_density": 10.0, "random_walk": 1.0}))
    again("vio_inflation_density")
    edit(rig_file(root, RIG), lambda d: d["imu"].update(vio_inflation={"noise_density": 10.0, "random_walk": 10.0}))
    again("vio_inflation_walk")
    edit(rig_file(root, RIG), lambda d: d["cameras"]["ultrawide"]["T_imu_cam"]["matrix"][0].__setitem__(3, -0.0147))
    again("extrinsics")
    rig_lens_file(root, RIG).write_text(dump_lens(lens_document(
        rig_id=RIG, sensor="imx219", resolution=(1280, 960), intrinsics=[580.0, 580.0, 640.0, 480.0],
        distortion=[0.02, -0.005, 0.001, 0.0], date="2026-11-16T18:00:00Z", source={}, board={}, metrics={})))
    again("lens")
    assert hashes["same"] == hashes["base"]
    changed = {k: v for k, v in hashes.items() if k not in ("base", "same")}
    assert len(set(changed.values()) | {base}) == len(changed) + 1, hashes


def test_variant_build_folder(sensors, tmp_path, monkeypatch):
    """A PATCHES_UPTO variant: BUILD_INFO next to umiproc_build_NNNN/umiproc_slam names its own fork/lib library."""
    out = tmp_path / "fork" / "umiproc_build_0006"
    lib = out / "fork" / "lib" / "libORB_SLAM3.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"variant lib")
    (out / "umiproc_slam").write_bytes(b"variant binary")
    write_build_info(out / "BUILD_INFO", lib, ("0001-a.patch",))
    monkeypatch.setattr(SI, "loaded_orbslam3_library", lambda b: lib)
    rec, _ = _fp(sensors, tmp_path / "a", out / "umiproc_slam")
    assert rec["build_info"]["path"] == str(out / "BUILD_INFO")
    assert [p["name"] for p in rec["patches"]] == ["0001-a.patch"]
    assert rec["library"]["realpath"] == str(lib) and rec["library"]["sha256"] == _sha(lib)


def test_library_through_a_symlink(sensors, tmp_path, monkeypatch):
    binary = fake_build(tmp_path / "fork", monkeypatch)
    lib = tmp_path / "fork" / "lib" / "libORB_SLAM3.so"
    link = tmp_path / "linked" / "libORB_SLAM3.so"
    link.parent.mkdir()
    link.symlink_to(lib)
    monkeypatch.setattr(SI, "loaded_orbslam3_library", lambda b: link)    # e.g. a variant's shared-lib symlink
    rec, _ = _fp(sensors, tmp_path / "a", binary)
    assert rec["library"] == {"path": str(link), "realpath": str(lib), "sha256": _sha(lib)}


@pytest.mark.parametrize("breakage, match", [
    ("no_binary", r"no umiproc_slam binary .*native/build\.sh"),
    ("no_build_info", r"no BUILD_INFO at .*native/build\.sh"),
    ("no_library_line", r"records no library.*native/build\.sh"),
    ("no_patch_line", r"records no patch.*native/build\.sh"),
    ("bad_patch_line", r"unreadable patch line.*native/build\.sh"),
    ("library_missing", r"not a file.*native/build\.sh"),
    ("other_library", r"loads libORB_SLAM3\.so from .*, but .*BUILD_INFO records .*native/build\.sh"),
])
def test_missing_or_inconsistent_build_is_an_error(sensors, tmp_path, monkeypatch, breakage, match):
    binary = fake_build(tmp_path / "fork", monkeypatch)
    info = binary.parent / "BUILD_INFO"
    lib = tmp_path / "fork" / "lib" / "libORB_SLAM3.so"
    if breakage == "no_binary":
        binary.unlink()
    elif breakage == "no_build_info":
        info.unlink()
    elif breakage == "no_library_line":
        write_build_info(info, None, PATCH_NAMES)
    elif breakage == "no_patch_line":
        write_build_info(info, lib, ())
    elif breakage == "bad_patch_line":
        info.write_text(info.read_text() + "patch zz 0003-c.patch\n")
    elif breakage == "library_missing":
        lib.unlink()
    elif breakage == "other_library":
        other = tmp_path / "elsewhere" / "libORB_SLAM3.so"
        other.parent.mkdir()
        other.write_bytes(b"lib v1")
        monkeypatch.setattr(SI, "loaded_orbslam3_library", lambda b: other)
    with pytest.raises(SI.SlamBuildError, match=match):
        _fp(sensors, tmp_path / "a", binary)
    with pytest.raises(SI.SlamBuildError, match=match):                 # and nothing is written
        SI.write_slam_inputs(sensors, tmp_path / "inputs", binary=binary)
    assert not (tmp_path / "inputs").exists()


class _Ldd:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


@pytest.mark.parametrize("stdout, rc, expect", [
    ("\tlinux-vdso.so.1 (0x00007ffd)\n\tlibORB_SLAM3.so => /x/lib/libORB_SLAM3.so (0x000077dc19800000)\n"
     "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000)\n", 0, "/x/lib/libORB_SLAM3.so"),
    ("\tlibORB_SLAM3.so => not found\n", 0, r"cannot find libORB_SLAM3\.so.*native/build\.sh"),
    ("\tnot a dynamic executable\n", 1, r"ldd shows no libORB_SLAM3\.so.*native/build\.sh"),
    ("\tlibc.so.6 => /lib/libc.so.6 (0x0000)\n", 0, r"ldd shows no libORB_SLAM3\.so"),
])
def test_loaded_library_from_ldd(monkeypatch, stdout, rc, expect):
    monkeypatch.setattr(SI.subprocess, "run", lambda cmd, **kw: _Ldd(stdout, rc))
    if expect.startswith("/"):
        assert SI.loaded_orbslam3_library(Path("/b/umiproc_slam")) == Path(expect)
    else:
        with pytest.raises(SI.SlamBuildError, match=expect):
            SI.loaded_orbslam3_library(Path("/b/umiproc_slam"))


@pytest.mark.skipif(not (default_binary().is_file() and (default_binary().parent / "BUILD_INFO").is_file()),
                    reason=f"needs {default_binary()} and its BUILD_INFO (native/build.sh)")
def test_real_build_record():
    """The real build: ldd resolves the library BUILD_INFO records, and the patches are the repo's (all of them, or
    the first ones for a PATCHES_UPTO variant), with the repo files' hashes."""
    rec = SI.slam_build_record(default_binary())
    info = SI.read_build_info(default_binary().parent / "BUILD_INFO")
    assert Path(rec["library"]["realpath"]) == Path(os.path.realpath(info["library"]))
    names = [p["name"] for p in rec["patches"]]
    assert names and names == SI.patch_series()[:len(names)]
    for p in rec["patches"]:
        assert p["sha256_16"] == _sha(SI.PATCH_DIR / p["name"])[:16]


def test_write_slam_inputs(sensors, tmp_path, monkeypatch):
    binary = fake_build(tmp_path / "fork", monkeypatch)
    inp = SI.write_slam_inputs(sensors, tmp_path / "inputs", binary=binary)
    assert sorted(p.name for p in (tmp_path / "inputs").iterdir()) == [SI.SETTINGS_NAME, SI.MASK_NAME]
    assert inp.settings.read_text() == SI.orbslam3_settings_text(sensors)
    assert inp.fingerprint_hash == SI.fingerprint_hash(inp.fingerprint)


# --------------------------------------------------------------------------------------------- the real parser

@pytest.mark.slow
@pytest.mark.skipif(not (default_binary().is_file() and default_vocabulary().is_file()),
                    reason=f"needs {default_binary()} and {default_vocabulary()} (native/build.sh)")
def test_the_forks_parser_reads_it(sensors, tmp_path):
    """umiproc_slam on a 15-frame sequence with our settings: the fork's Settings parser loads every section and
    prints back the values we wrote (it aborts on a missing or mistyped key)."""
    inp = SI.write_slam_inputs(sensors, tmp_path / "inputs")
    seq = tmp_path / "seq"
    (seq / "mav0" / "cam0" / "data").mkdir(parents=True)
    (seq / "mav0" / "imu0").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = (rng.random((960, 1280)) * 255).astype(np.uint8)
    t = [1_000_000_000_000 + i * 33_333_333 for i in range(15)]
    for i, ti in enumerate(t):
        cv2.imwrite(str(seq / "mav0" / "cam0" / "data" / f"{ti}.png"), np.roll(img, 2 * i, axis=1))
    (seq / "times.txt").write_text("".join(f"{x}\n" for x in t))
    imu = range(t[0] - 20_000_000, t[-1] + 40_000_000, 9_615_385)
    (seq / "mav0" / "imu0" / "data.csv").write_text("#t\n" + "".join(f"{x},0,0,0,0,0,9.81\n" for x in imu))
    p = subprocess.run([str(default_binary()), "--vocabulary", str(default_vocabulary()), "--setting",
                        str(inp.settings), "--sequence", str(seq), "--output_trajectory_csv",
                        str(tmp_path / "traj.csv"), "--mask_img", str(inp.mask), "--no_realtime"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-3000:]
    assert "aborting" not in log and "required parameter" not in log
    for section in ("camera 1", "image info", "IMU calibration", "ORB settings", "viewer settings",
                    "Atlas settings", "misc parameters"):
        assert f"-Loaded {section}" in log, section
    assert "-Camera 1 parameters (Kannala-Brandt): [ 572.9 572.9 639.5 479.5 0.02 -0.005 0.001 0 ]" in log
    assert "-Original image size: [ 1280 , 960 ]" in log and "-Current image size: [ 1280 , 960 ]" in log
    for line in ("-Sequence FPS: 30", "-Gyro noise: 6.63e-05", "-Accelerometer noise: 0.000736",
                 "-Gyro walk: 2e-06", "-Accelerometer walk: 0.0001", "-IMU frequency: 104",
                 "-Features per image: 1500", "-ORB scale factor: 1.2", "-ORB number of scales: 8",
                 "-Initial FAST threshold: 20", "-Min FAST threshold: 7",
                 "Discard points further than 20 m from current camera", "Camera 0 is fisheye"):
        assert line in log, line


def test_build_info_hash_ignores_only_the_built_time(tmp_path):
    from umiproc.slam_inputs import build_info_sha256

    body = "fork x b741dca\npatch 0123456789abcdef 0001-a.patch\nlibrary /x/libORB_SLAM3.so\n"
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_text(body + "built 2026-11-16T01:00:00Z\n")
    b.write_text(body + "built 2026-11-17T09:30:00Z\n")
    c.write_text(body.replace("0123456789abcdef", "fedcba9876543210") + "built 2026-11-16T01:00:00Z\n")
    assert build_info_sha256(a) == build_info_sha256(b)
    assert build_info_sha256(a) != build_info_sha256(c)
