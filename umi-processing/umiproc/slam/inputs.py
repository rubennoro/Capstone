"""What every SLAM run is given besides the frames: the ORB-SLAM3 settings file, the gripper mask PNG, and the
**fingerprint** of both plus the sensor values and the umiproc_slam binary (docs/DESIGN.md "SLAM").

::

    sensors = load_sensors(root, "rig_pi_v3")                       # umiproc.sensors
    inputs = write_slam_inputs(sensors, out_dir)                    # orbslam3_settings.yaml + slam_mask.png
    run_umi_slam(seq, inputs.settings, work, mask_img=inputs.mask, ...)
    inputs.fingerprint, inputs.fingerprint_hash                     # stored with the map / episode

**Settings** (:func:`orbslam3_settings_text`): the ``File.version "1.0"`` format, which UMI's fork
(cheng-chi/ORB_SLAM3 ``b741dca``) parses with its ``Settings`` class (``src/System.cc`` picks it when
``File.version`` is the string ``"1.0"``; ``src/Settings.cc`` reads it; ``Tracking::newParameterLoader`` uses it).
That parser aborts on a missing required key and on a wrong type: ``readParameter<float>`` needs a YAML real
(written with a ``.`` or an ``e``), ``readParameter<int>`` an integer, strings are quoted, and ``IMU.T_b_c1`` must
be ``dt: f`` (``Converter::toSophus`` reads it with ``at<float>``). Keys, in the parser's order:

=============================  =====  ==========================================================================
key                            type   value
=============================  =====  ==========================================================================
File.version                   str    "1.0"
Camera.type                    str    "KannalaBrandt8" (the rig's fisheye-equidistant = Kannala-Brandt k1..k4)
Camera1.fx fy cx cy            real   the effective ultrawide model (lens file, else the rig's nominal), native px
Camera1.k1 k2 k3 k4            real   its Kannala-Brandt coefficients
Camera.width, Camera.height    int    1280, 960: the native size, no scaling (no Camera.newWidth/newHeight)
Camera.fps                     int    30 (the ultrawide's rate; patches 0001/0005 derive periods from it)
Camera.RGB                     int    0 (frames are grey PNGs read as grey: the value is not used)
IMU.NoiseGyro NoiseAcc         real   rig imu.noise densities x imu.vio_inflation.noise_density (datasheet)
IMU.GyroWalk AccWalk           real   rig imu.noise random walks x imu.vio_inflation.random_walk (assumptions)
IMU.Frequency                  real   104.0 (the LSM6DSOX rate the Pi sets; umiproc.sensors.IMU_RATE_HZ)
IMU.T_b_c1                     mat f  the ultrawide's CAD T_imu_cam (camera -> IMU body)
IMU.InsertKFsWhenLost          int    1 = the parser's default, which UMI's settings use (they omit the key)
ORBextractor.nFeatures         int    SLAM config orb.n_features
ORBextractor.scaleFactor       real   orb.scale_factor
ORBextractor.nLevels           int    orb.n_levels
ORBextractor.iniThFAST         int    orb.ini_th_fast
ORBextractor.minThFAST         int    orb.min_th_fast
System.thFarPoints             real   20.0 (UMI's: points further than 20 m are not used)
Viewer.*                       real   required by the parser although no viewer runs; UMI's values
=============================  =====  ==========================================================================

No file paths go into the settings file, so its hash does not depend on where the data root is.

**Mask** (:func:`mask_png`): the SLAM config's polygons drawn by :func:`umiproc.slam.config.mask_keep` at the
mask's resolution (1280x960), written as an 8-bit grey PNG with **255 = masked**, 0 = kept (the front-end sets the
nonzero pixels to 0 in every frame it feeds, ``im.setTo(0, mask)``, as UMI's ``gopro_slam.cc``).

**Fingerprint** (:func:`slam_fingerprint`): the effective sensor values, the SLAM config, the mask's pixel hash,
the settings file's hash, the umiproc_slam binary's path and sha256, and the build it belongs to: the sha256 of the
build's ``BUILD_INFO`` (written by ``native/build.sh`` next to the binary), the patch list parsed from that
``BUILD_INFO`` (names and content hashes: the patches the build applied, not the repo's ``native/patches/``), and the
path and sha256 of the ``libORB_SLAM3.so`` the binary actually loads (every patch lives in that library, which the
binary links dynamically). The library is resolved as ``build.sh``'s own check does, with ``ldd`` on the binary,
and must be the ``library`` that ``BUILD_INFO`` records. A missing binary, ``BUILD_INFO`` or library is an error
naming ``native/build.sh``. The hash is sha256 of the canonical JSON. A map or episode made with another hash is out
of date (changing any of these re-runs the map and then every episode).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umiproc.paths import REPO

__all__ = ["SETTINGS_NAME", "MASK_NAME", "FINGERPRINT_FORMAT", "THFARPOINTS_M", "INSERT_KFS_WHEN_LOST",
           "PATCH_DIR", "orbslam3_settings_text", "write_orbslam3_settings", "mask_png", "write_mask_png",
           "mask_pixel_sha256", "patch_series", "BUILD_INFO_NAME", "SlamBuildError", "read_build_info",
           "loaded_orbslam3_library", "slam_build_record", "slam_fingerprint", "fingerprint_hash", "SlamInputs",
           "write_slam_inputs"]

SETTINGS_NAME = "orbslam3_settings.yaml"
MASK_NAME = "slam_mask.png"
FINGERPRINT_FORMAT = "umiproc.slam_fingerprint.v2"
#: The file ``native/build.sh`` writes next to every umiproc_slam it builds.
BUILD_INFO_NAME = "BUILD_INFO"
#: UMI's ``System.thFarPoints`` (fork Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml).
THFARPOINTS_M = 20.0
#: ``IMU.InsertKFsWhenLost``: 1 = Settings::readIMU's default when the key is absent, as in all of UMI's settings.
INSERT_KFS_WHEN_LOST = 1
PATCH_DIR = REPO / "native" / "patches"
#: UMI's viewer values (the parser requires them; no viewer runs).
_VIEWER = (("KeyFrameSize", 0.05), ("KeyFrameLineWidth", 1.0), ("GraphLineWidth", 0.9), ("PointSize", 2.0),
           ("CameraSize", 0.08), ("CameraLineWidth", 3.0), ("ViewpointX", 0.0), ("ViewpointY", -0.7),
           ("ViewpointZ", -3.5), ("ViewpointF", 500.0), ("imageViewScale", 1.0))


# --------------------------------------------------------------------------------------------- settings

def _real(v) -> str:
    """A YAML real OpenCV's parser types as REAL: 9 significant digits, always with a ``.`` or an ``e``."""
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"not a finite number: {v}")
    s = format(v, ".9g")
    return s if ("." in s or "e" in s) else s + ".0"


def _int(v, what: str) -> str:
    f = float(v)
    if not f.is_integer():
        raise ValueError(f"{what} must be a whole number for ORB-SLAM3 (an int setting), got {v}")
    return str(int(f))


def _matrix_f(name: str, M) -> list[str]:
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"{name}: expected a 4x4 matrix, got {M.shape}")
    data = ", ".join(f"{v:.9f}" for v in M.reshape(-1))
    return [f"{name}: !!opencv-matrix", "   rows: 4", "   cols: 4", "   dt: f", f"   data: [{data}]"]


def orbslam3_settings_text(sensors) -> str:
    """The settings file for :class:`umiproc.sensors.SensorValues` (keys and reasons: the module docstring)."""
    cam = sensors.ultrawide
    if cam.model.type != "fisheye-equidistant":
        raise ValueError(f"the ultrawide model must be fisheye-equidistant, got {cam.model.type}")
    fx, fy, cx, cy = cam.model.intrinsics
    k1, k2, k3, k4 = cam.model.distortion
    w, h = (int(v) for v in cam.resolution)
    noise = sensors.rig.imu.noise.for_vio()
    orb = sensors.slam.orb
    lens = sensors.lens_record()
    lens_line = (f"calibrated {lens['date']}" if lens["provenance"] == "calibrated" else lens["provenance"])
    lines = [
        "%YAML:1.0",
        "---",
        "# ORB-SLAM3 settings for UMI's fork (cheng-chi/ORB_SLAM3 b741dca + umi-processing/native/patches), written by",
        # the old module name is kept on purpose: this text is hashed into the SLAM fingerprint (settings.sha256),
        # and changing it makes every earlier map "out of date"
        "# umiproc.slam_inputs from the data-root rig file, SLAM config and lens file. Do not edit: change those.",
        f"# rig {sensors.rig_id}; ultrawide {cam.sensor} {w}x{h} @ {_real(cam.fps)} fps; lens: {lens_line}",
        'File.version: "1.0"',
        "",
        'Camera.type: "KannalaBrandt8"',
        f"Camera1.fx: {_real(fx)}",
        f"Camera1.fy: {_real(fy)}",
        f"Camera1.cx: {_real(cx)}",
        f"Camera1.cy: {_real(cy)}",
        f"Camera1.k1: {_real(k1)}",
        f"Camera1.k2: {_real(k2)}",
        f"Camera1.k3: {_real(k3)}",
        f"Camera1.k4: {_real(k4)}",
        f"Camera.width: {w}",
        f"Camera.height: {h}",
        f"Camera.fps: {_int(cam.fps, 'cameras.ultrawide.fps')}",
        "Camera.RGB: 0",
        "",
        "# camera -> IMU body: the ultrawide's T_imu_cam (CAD)",
        *_matrix_f("IMU.T_b_c1", cam.T_imu_cam),
        f"IMU.InsertKFsWhenLost: {INSERT_KFS_WHEN_LOST}",
        f"IMU.NoiseGyro: {_real(noise.gyroscope_noise_density)}",
        f"IMU.NoiseAcc: {_real(noise.accelerometer_noise_density)}",
        f"IMU.GyroWalk: {_real(noise.gyroscope_random_walk)}",
        f"IMU.AccWalk: {_real(noise.accelerometer_random_walk)}",
        f"IMU.Frequency: {_real(sensors.imu_rate_hz)}",
        "",
        f"ORBextractor.nFeatures: {_int(orb.n_features, 'orb.n_features')}",
        f"ORBextractor.scaleFactor: {_real(orb.scale_factor)}",
        f"ORBextractor.nLevels: {_int(orb.n_levels, 'orb.n_levels')}",
        f"ORBextractor.iniThFAST: {_int(orb.ini_th_fast, 'orb.ini_th_fast')}",
        f"ORBextractor.minThFAST: {_int(orb.min_th_fast, 'orb.min_th_fast')}",
        "",
        f"System.thFarPoints: {_real(THFARPOINTS_M)}",
        "",
        "# required by the parser; no viewer runs",
        *(f"Viewer.{k}: {_real(v)}" for k, v in _VIEWER),
    ]
    return "\n".join(lines) + "\n"


def write_orbslam3_settings(sensors, path) -> Path:
    path = Path(path)
    path.write_text(orbslam3_settings_text(sensors), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------- mask

def mask_png(slam_cfg) -> np.ndarray:
    """The front-end's mask image: uint8 ``(H, W)`` at the mask's resolution, **255 = masked**, 0 = kept."""
    from umiproc.slam.config import mask_keep

    keep = mask_keep(slam_cfg)
    return np.where(keep == 0, 255, 0).astype(np.uint8)


def mask_pixel_sha256(mask: np.ndarray) -> str:
    """sha256 of the mask's shape and pixels (independent of the PNG encoder)."""
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    return hashlib.sha256(f"{m.shape[0]}x{m.shape[1]}:".encode() + m.tobytes()).hexdigest()


def write_mask_png(slam_cfg, path) -> Path:
    """Write :func:`mask_png` as an 8-bit grey PNG and check it reads back identical."""
    import cv2

    path = Path(path)
    m = mask_png(slam_cfg)
    if not cv2.imwrite(str(path), m):
        raise OSError(f"cannot write {path}")
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if back is None or back.shape != m.shape or not np.array_equal(back, m):
        raise OSError(f"{path} does not read back as the mask that was written")
    return path


# --------------------------------------------------------------------------------------------- fingerprint

def patch_series(patch_dir: Path = PATCH_DIR) -> list[str]:
    """The names of the fork patches in the repo (``native/patches/*.patch``), in the order they are applied. (The
    fingerprint lists the patches a build applied, from its ``BUILD_INFO``, not these.)"""
    d = Path(patch_dir)
    return sorted(p.name for p in d.glob("*.patch")) if d.is_dir() else []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slam_config_record(slam) -> dict:
    return {"rig": slam.rig,
            "mask": {"resolution": [int(v) for v in slam.mask.resolution],
                     "polygons": [[[float(x), float(y)] for x, y in poly] for poly in slam.mask.polygons]},
            "table_tag": {"dictionary": slam.table_tag.dictionary, "id": int(slam.table_tag.id),
                          "size_m": float(slam.table_tag.size_m)},
            "orb": {"n_features": slam.orb.n_features, "scale_factor": float(slam.orb.scale_factor),
                    "n_levels": slam.orb.n_levels, "ini_th_fast": slam.orb.ini_th_fast,
                    "min_th_fast": slam.orb.min_th_fast}}


class SlamBuildError(RuntimeError):
    """The umiproc_slam build cannot be fingerprinted (binary, ``BUILD_INFO`` or ``libORB_SLAM3.so`` missing or
    inconsistent)."""


_BUILD_HINT = "build it with umi-processing/native/build.sh (native/README.md)"
_PATCH_LINE = re.compile(r"^patch ([0-9a-f]{16}) (\S+\.patch)$")
_LDD_LINE = re.compile(r"^\s*libORB_SLAM3\.so\S*\s+=>\s+(.+?)(?:\s+\(0x[0-9a-f]+\))?\s*$")


def read_build_info(path) -> dict:
    """Parse a ``BUILD_INFO`` written by ``native/build.sh``: ``{"library": Path, "patches": [{"name", "sha256_16"}]}``
    (``sha256_16`` = the first 16 hex digits of the patch file's sha256, as build.sh writes them)."""
    path = Path(path)
    if not path.is_file():
        raise SlamBuildError(f"no {BUILD_INFO_NAME} at {path}: not a native/build.sh build; {_BUILD_HINT}")
    library, patches = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("library "):
            library = Path(line[len("library "):].strip())
        elif line.startswith("patch "):
            m = _PATCH_LINE.match(line.strip())
            if not m:
                raise SlamBuildError(f"{path}: unreadable patch line {line!r}; {_BUILD_HINT}")
            patches.append({"name": m.group(2), "sha256_16": m.group(1)})
    if library is None or not patches:
        raise SlamBuildError(f"{path} records no {'library' if library is None else 'patch'}; rebuild: "
                             f"{_BUILD_HINT}")
    return {"library": library, "patches": patches}


def build_info_sha256(path) -> str:
    """sha256 of a ``BUILD_INFO`` without its ``built <UTC time>`` line: build.sh rewrites that line on every run,
    even when nothing was rebuilt, and a re-run that changes nothing must not make every map out of date."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if not ln.startswith("built ")]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def loaded_orbslam3_library(binary) -> Path:
    """The ``libORB_SLAM3.so`` that ``binary`` loads, as ``ldd`` resolves it in this process's environment (the
    environment umiproc runs the binary in): the same resolution as ``native/build.sh``'s check."""
    try:
        p = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise SlamBuildError(f"cannot run ldd on {binary} ({e}); {_BUILD_HINT}") from None
    for line in p.stdout.splitlines():
        m = _LDD_LINE.match(line)
        if m:
            where = m.group(1).strip()
            if where == "not found" or not where.startswith("/"):
                raise SlamBuildError(f"{binary} cannot find libORB_SLAM3.so (ldd: {line.strip()}); "
                                     f"{_BUILD_HINT}, and do not set LD_LIBRARY_PATH")
            return Path(where)
    raise SlamBuildError(f"ldd shows no libORB_SLAM3.so for {binary} (exit {p.returncode}: "
                         f"{(p.stdout + p.stderr).strip()[:300]}); {_BUILD_HINT}")


def slam_build_record(binary) -> dict:
    """The build part of the fingerprint: ``binary`` (path, sha256), ``build_info`` (path, sha256), ``patches``
    (parsed from ``BUILD_INFO``) and ``library`` (the ``libORB_SLAM3.so`` the binary loads: path as ldd gives it,
    realpath, sha256). :class:`SlamBuildError` when any of it is missing or the loaded library is not the one
    ``BUILD_INFO`` records."""
    binary = Path(binary)
    if not binary.is_file():
        raise SlamBuildError(f"no umiproc_slam binary at {binary}; {_BUILD_HINT}")
    info_path = binary.parent / BUILD_INFO_NAME
    info = read_build_info(info_path)
    lib = loaded_orbslam3_library(binary)
    real = Path(os.path.realpath(lib))
    if not real.is_file():
        raise SlamBuildError(f"{binary} loads {lib}, which is not a file; {_BUILD_HINT}")
    if real != Path(os.path.realpath(info["library"])):
        raise SlamBuildError(f"{binary} loads libORB_SLAM3.so from {lib}, but {info_path} records "
                             f"{info['library']} (check LD_LIBRARY_PATH, or rebuild: {_BUILD_HINT})")
    return {"binary": {"path": str(binary), "sha256": _sha256_file(binary)},
            "build_info": {"path": str(info_path), "sha256": build_info_sha256(info_path)},
            "patches": info["patches"],
            "library": {"path": str(lib), "realpath": str(real), "sha256": _sha256_file(real)}}


def slam_fingerprint(sensors, settings_path, mask, binary=None) -> dict:
    """The record a map or episode stores: the effective sensor values (:meth:`SensorValues.record`), the SLAM
    config, the mask's pixel hash (``mask``: the array or a PNG path), the settings file's sha256, and the build of
    the umiproc_slam binary (``binary``, default :func:`umiproc.slam.runner.default_binary`;
    :func:`slam_build_record`). :class:`SlamBuildError` when the build is missing or incomplete."""
    if binary is None:
        from umiproc.slam.runner import default_binary

        binary = default_binary()
    build = slam_build_record(binary)
    if isinstance(mask, (str, Path)):
        import cv2

        arr = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise OSError(f"cannot read the mask {mask}")
    else:
        arr = np.asarray(mask)
    return {
        "format": FINGERPRINT_FORMAT,
        "sensors": sensors.record(),
        "slam_config": _slam_config_record(sensors.slam),
        "mask": {"pixel_sha256": mask_pixel_sha256(arr), "size": [int(arr.shape[1]), int(arr.shape[0])],
                 "masked_fraction": round(float(np.mean(arr != 0)), 6)},
        "settings": {"sha256": _sha256_file(Path(settings_path))},
        **build,
    }


def fingerprint_hash(record: dict) -> str:
    """sha256 of the record's canonical JSON (sorted keys, no whitespace)."""
    from umiproc.dp.files import hash_of

    return hash_of(record)


@dataclass(frozen=True)
class SlamInputs:
    settings: Path
    mask: Path
    fingerprint: dict
    fingerprint_hash: str


def write_slam_inputs(sensors, out_dir, binary=None) -> SlamInputs:
    """Write ``orbslam3_settings.yaml`` and ``slam_mask.png`` into ``out_dir`` (created) and return them with the
    fingerprint and its hash (:class:`SlamBuildError`, before anything is written, when the build is missing)."""
    if binary is None:
        from umiproc.slam.runner import default_binary

        binary = default_binary()
    slam_build_record(binary)                      # fail before writing anything
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = write_orbslam3_settings(sensors, out / SETTINGS_NAME)
    mask = write_mask_png(sensors.slam, out / MASK_NAME)
    record = slam_fingerprint(sensors, settings, mask, binary)
    return SlamInputs(settings, mask, record, fingerprint_hash(record))
