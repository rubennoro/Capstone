"""Typed, validated loaders for ``task.yaml`` and ``rigs/<rig_id>.yaml``.

Conventions (normative for the package):

* ``T_a_b`` is a 4x4 float64 rigid transform mapping points in frame ``b`` to frame ``a`` (the pose of ``b`` in
  ``a``), metres. Camera frames are OpenCV optical (x right, y down, z forward).
* ``T_imu_cam`` per camera; ``T_cam_tcp`` is the TCP pose in the **physical** ultrawide optical frame (the SLAM
  camera), so ``T_W_tcp = T_W_cam @ T_cam_tcp``. The virtual ultrawide is rotated from it by
  ``virtual_ultrawide.rotation`` (``R_phys_virt``: ``d_phys = R_phys_virt @ d_virt``, default identity); ArUco tag
  ``tvec`` and ``aruco.nominal_z_m`` are in the virtual frame.
* The training world is gravity-aligned **z-up** (sensor_rig ``gravity_align``). ``world_offset`` (yaw about +z,
  then translation) is applied on the left: ``T_W'_tcp = T_world_offset @ T_W_tcp``. Its yaw must be >= 1 deg away
  from a multiple of 360 so that poses near the SLAM origin practically never have rotation vector 0 (iPhUMI's
  ``pose_6d_to_4x4`` is NaN there); the pose stage flags any such row as invalid.
* Time. Raw ``t_ns`` is the libcamera SensorTimestamp verbatim (docs/CONTRACT.md); the Jetson owns its meaning
  through ``cameras.<role>.timestamp_ref``:

      mid-exposure camera time  t_cam = t_ns + s * (exposure_ns // 2) + time_offset_ns
                                s = +1 for start_of_exposure, -1 for end_of_exposure
      IMU time                  t_imu = t_ns + imu.time_offset_ns

  Kalibr: ``time_offset_ns = round(timeshift_cam_imu_s * 1e9)``, with Kalibr run on these mid-exposure stamps; a
  constant rolling-shutter half-readout is absorbed in ``time_offset_ns``. See :meth:`RigConfig.camera_time_ns`.
* ``provenance`` is ``nominal`` (CAD / datasheet guesses) or ``calibrated`` (measured), per section.

Transforms in YAML are either ``{matrix: 4x4}`` or ``{rotation: {matrix: 3x3} | {euler_xyz_deg: [rx, ry, rz]},
translation_m: [x, y, z]}`` (extrinsic x-y-z Euler angles, ``Rz @ Ry @ Rx``). Unknown keys are errors.

Errors raise :class:`ConfigError` whose message starts with the file and YAML path of the bad field.
"""

# The mapping reader, scalar/transform parsers and ImuNoise.for_vio are adapted from
# sensor_rig/sensor_rig/rig_config.py and sensor_rig/sensor_rig/transforms.py (commit 4be9982).

from __future__ import annotations

import datetime as _dt
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dataproc.paths import NAME_RE

__all__ = [
    "RIG_SCHEMA", "CAMERA_MODEL_TYPES", "PROVENANCE", "TIMESTAMP_REFS", "DEFAULT_WORLD_OFFSET", "ConfigError",
    "TaskConfig", "CameraModel", "CameraConfig", "VirtualCamera", "ImuNoise", "ImuConfig", "ArucoConfig",
    "GripperConfig", "RigConfig",
    "load_task_config", "dump_task_config", "load_rig", "parse_rig",
]

RIG_SCHEMA = "dataproc.rig.v1"
CAMERA_MODEL_TYPES = ("pinhole-radtan", "fisheye-equidistant")
PROVENANCE = ("nominal", "calibrated")
TIMESTAMP_REFS = ("start_of_exposure", "end_of_exposure")
#: Used when the rig yaml has no ``world_offset``.
DEFAULT_WORLD_OFFSET = {"translation_m": (0.1, 0.2, 0.3), "yaw_deg": 30.0}
CAMERA_ROLES = ("main", "ultrawide")


class ConfigError(ValueError):
    """Invalid task.yaml or rig yaml. The message starts with ``<file>: <yaml.path>:``."""


# --------------------------------------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class TaskConfig:
    """``tasks/<folder>/task.yaml``. ``label`` is the task name stored in the zarr (``block_raise`` also for the
    folder ``block_raise_2``); ``rig`` is the rig id of its episodes; ``created`` is an ISO 8601 string."""

    label: str
    rig: str
    created: str
    notes: str = ""


@dataclass(frozen=True)
class CameraModel:
    """Physical camera intrinsics. ``distortion`` is ``[k1, k2, p1, p2]`` for ``pinhole-radtan`` and
    Kannala-Brandt ``[k1, k2, k3, k4]`` for ``fisheye-equidistant``."""

    type: str
    intrinsics: tuple[float, float, float, float]    # fx, fy, cx, cy (pixels)
    distortion: tuple[float, float, float, float]

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True, eq=False)
class CameraConfig:
    role: str                            # main | ultrawide
    sensor: str
    resolution: tuple[int, int]          # (width, height) of the recorded video
    fps: float
    model: CameraModel
    T_imu_cam: np.ndarray                # 4x4
    timestamp_ref: str                   # start_of_exposure | end_of_exposure: what the raw t_ns marks
    time_offset_ns: int
    provenance: str


@dataclass(frozen=True, eq=False)
class VirtualCamera:
    """Distortion-free pinhole the ultrawide is remapped to (tag detection and the ultrawide images)."""

    resolution: tuple[int, int]
    intrinsics: tuple[float, float, float, float]
    R_phys_virt: np.ndarray              # 3x3, d_phys = R_phys_virt @ d_virt

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class ImuNoise:
    """Continuous-time noise (Kalibr units), plus the VIO inflation factors (``imu.vio_inflation``)."""

    gyroscope_noise_density: float       # rad/s/sqrt(Hz)
    gyroscope_random_walk: float         # rad/s^2/sqrt(Hz)
    accelerometer_noise_density: float   # m/s^2/sqrt(Hz)
    accelerometer_random_walk: float     # m/s^3/sqrt(Hz)
    inflation_noise: float = 1.0
    inflation_random_walk: float = 1.0

    def for_vio(self) -> "ImuNoise":
        """Inflated values for VIO configs (densities x inflation_noise, walks x inflation_random_walk)."""
        return ImuNoise(self.gyroscope_noise_density * self.inflation_noise,
                        self.gyroscope_random_walk * self.inflation_random_walk,
                        self.accelerometer_noise_density * self.inflation_noise,
                        self.accelerometer_random_walk * self.inflation_random_walk)


@dataclass(frozen=True)
class ImuConfig:
    sensor: str
    noise: ImuNoise
    time_offset_ns: int
    provenance: str


@dataclass(frozen=True)
class ArucoConfig:
    dict: str                            # OpenCV predefined dictionary name, e.g. DICT_4X4_50
    marker_size_m: float
    left_id: int
    right_id: int
    nominal_z_m: float                   # rough tag depth along the virtual ultrawide z axis (outlier filter)


@dataclass(frozen=True)
class GripperConfig:
    """What the calibration fit must measure: ``max_width - min_width`` (the finger-tag separation range, metres,
    exactly what the width stage exports as the full opening) within ``opening_tolerance_m`` of ``max_opening_m``."""

    max_opening_m: float
    opening_tolerance_m: float


@dataclass(frozen=True, eq=False)
class RigConfig:
    schema: str
    rig_id: str
    cameras: dict[str, CameraConfig]     # keyed by role: main, ultrawide
    virtual_ultrawide: VirtualCamera
    T_cam_tcp: np.ndarray                # 4x4, TCP in the physical ultrawide optical frame
    tcp_provenance: str
    imu: ImuConfig
    aruco: ArucoConfig
    T_world_offset: np.ndarray           # 4x4, applied on the left in the gravity-aligned z-up world
    gripper: GripperConfig | None = None  # None: the rig yaml has no gripper section (opening check skipped)
    notes: str = ""
    source_path: str | None = None

    def camera_time_ns(self, role: str, t_ns, exposure_ns):
        """Mid-exposure time on the IMU time base: ``t_ns + s * (exposure_ns // 2) + time_offset_ns`` with
        ``s = +1`` (start_of_exposure) or ``-1`` (end_of_exposure). Scalars give an int, arrays an int64 array."""
        cam = self.cameras[role]
        s = 1 if cam.timestamp_ref == "start_of_exposure" else -1
        out = np.asarray(t_ns, np.int64) + s * (np.asarray(exposure_ns, np.int64) // 2) + np.int64(cam.time_offset_ns)
        return out if out.ndim else int(out)

    def imu_time_ns(self, t_ns):
        """``t_ns + imu.time_offset_ns``. Scalars give an int, arrays an int64 array."""
        out = np.asarray(t_ns, np.int64) + np.int64(self.imu.time_offset_ns)
        return out if out.ndim else int(out)

    def nominal_sections(self) -> list[str]:
        """Sections still marked ``provenance: nominal`` (empty when fully calibrated)."""
        out = [f"cameras.{r}" for r, c in self.cameras.items() if c.provenance == "nominal"]
        if self.imu.provenance == "nominal":
            out.append("imu")
        if self.tcp_provenance == "nominal":
            out.append("tcp")
        return out


# --------------------------------------------------------------------------------------------- parsing helpers

class _Map:
    """Mapping reader that tracks used keys and reports errors with their YAML path."""

    def __init__(self, d, path: str, src: str):
        if not isinstance(d, dict):
            raise ConfigError(f"{src}: {path or '<root>'}: must be a mapping, got {type(d).__name__}")
        self.d, self.path, self.src, self.used = d, path, src, set()

    def p(self, key: str) -> str:
        return f"{self.path}.{key}" if self.path else key

    def err(self, key: str, msg: str) -> ConfigError:
        return ConfigError(f"{self.src}: {self.p(key)}: {msg}")

    def has(self, key: str) -> bool:
        return key in self.d

    def req(self, key: str):
        if key not in self.d:
            raise self.err(key, "missing required key")
        self.used.add(key)
        return self.d[key]

    def opt(self, key: str, default=None):
        self.used.add(key)
        return self.d.get(key, default)

    def sub(self, key: str) -> "_Map":
        return _Map(self.req(key), self.p(key), self.src)

    def finish(self) -> None:
        for k in self.d:
            if k not in self.used:
                raise self.err(str(k), "unknown key")

    # typed getters
    def int(self, key: str, min_value: int | None = None) -> int:
        v = self.req(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise self.err(key, f"expected an integer, got {v!r}")
        if min_value is not None and v < min_value:
            raise self.err(key, f"must be >= {min_value}, got {v}")
        return v

    def float(self, key: str, positive: bool = False) -> float:
        return _float(self.req(key), self, key, positive)

    def str(self, key: str, choices=None, pattern=None) -> str:
        v = self.req(key)
        if not isinstance(v, str) or not v:
            raise self.err(key, f"expected a non-empty string, got {v!r}")
        if choices is not None and v not in choices:
            raise self.err(key, f"must be one of {list(choices)}, got {v!r}")
        if pattern is not None and not pattern.fullmatch(v):
            raise self.err(key, f"must match {pattern.pattern}, got {v!r}")
        return v

    def vec(self, key: str, n: int, positive: bool = False) -> tuple[float, ...]:
        v = self.req(key)
        if not isinstance(v, (list, tuple)) or len(v) != n:
            raise self.err(key, f"expected a list of {n} numbers, got {v!r}")
        return tuple(_float(x, self, f"{key}[{i}]", positive) for i, x in enumerate(v))

    def size(self, key: str) -> tuple[int, int]:
        v = self.req(key)
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v)):
            raise self.err(key, f"expected [width, height] positive integers, got {v!r}")
        return (v[0], v[1])


def _float(v, m: _Map, key: str, positive: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise m.err(key, f"expected a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise m.err(key, f"must be finite, got {v}")
    if positive and v <= 0:
        raise m.err(key, f"must be > 0, got {v}")
    return v


def _is_rigid(T: np.ndarray, tol: float = 1e-6) -> bool:
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) <= tol
                and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol))


def _matrix(m: _Map, key: str, n: int) -> np.ndarray:
    raw = m.req(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        raise m.err(key, f"expected a {n}x{n} matrix")
    rows = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise m.err(f"{key}[{i}]", f"expected a list of {n} numbers, got {row!r}")
        rows.append([_float(x, m, f"{key}[{i}][{j}]") for j, x in enumerate(row)])
    return np.array(rows, dtype=np.float64)


def _rotation(r: _Map) -> np.ndarray:
    keys = [k for k in ("matrix", "euler_xyz_deg") if r.has(k)]
    if len(keys) != 1:
        raise ConfigError(f"{r.src}: {r.path}: give exactly one of 'matrix' or 'euler_xyz_deg'")
    if keys[0] == "matrix":
        R = _matrix(r, "matrix", 3)
    else:
        R = Rotation.from_euler("xyz", r.vec("euler_xyz_deg", 3), degrees=True).as_matrix()
        R[np.abs(R) < 1e-15] = 0.0
    r.finish()
    T = np.eye(4)
    T[:3, :3] = R
    if not _is_rigid(T):
        raise ConfigError(f"{r.src}: {r.path}: not a rotation (orthonormal to 1e-6 with det +1)")
    return R


def _transform(m: _Map) -> np.ndarray:
    if m.has("matrix"):
        T = _matrix(m, "matrix", 4)
        m.finish()
        if not _is_rigid(T):
            raise m.err("matrix", "not a rigid transform (orthonormal rotation with det +1, last row [0, 0, 0, 1])")
        return T
    T = np.eye(4)
    T[:3, :3] = _rotation(m.sub("rotation"))
    T[:3, 3] = m.vec("translation_m", 3)
    m.finish()
    return T


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _world_offset(m: _Map) -> np.ndarray:
    raw = m.opt("world_offset")
    wm = _Map(dict(DEFAULT_WORLD_OFFSET) if raw is None else raw, m.p("world_offset"), m.src)
    t = wm.vec("translation_m", 3)
    yaw = wm.float("yaw_deg")
    wm.finish()
    if abs(math.remainder(yaw, 360.0)) < 1.0:
        raise wm.err("yaw_deg", f"must be at least 1 deg away from a multiple of 360 (a near-identity offset leaves "
                     f"poses near the SLAM origin at rotation vector ~0), got {yaw}")
    T = np.eye(4)
    T[:3, :3] = _rot_z(yaw)
    T[:3, 3] = t
    return T


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: cannot read ({e})") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML ({e})") from None


# --------------------------------------------------------------------------------------------- task.yaml

def load_task_config(path: str | os.PathLike) -> TaskConfig:
    path = Path(path)
    m = _Map(_load_yaml(path), "", str(path))
    created = m.req("created")
    if isinstance(created, (_dt.datetime, _dt.date)):       # YAML parses unquoted timestamps
        created = created.isoformat()
    if not isinstance(created, str) or not created:
        raise m.err("created", f"expected an ISO 8601 timestamp, got {created!r}")
    notes = m.opt("notes", "") or ""
    if not isinstance(notes, str):
        raise m.err("notes", f"expected a string, got {notes!r}")
    cfg = TaskConfig(label=m.str("label", pattern=NAME_RE), rig=m.str("rig", pattern=NAME_RE), created=created,
                     notes=notes)
    m.finish()
    return cfg


def dump_task_config(cfg: TaskConfig, path: str | os.PathLike) -> None:
    """Write ``task.yaml`` (round-trips through :func:`load_task_config`)."""
    d = {"label": cfg.label, "rig": cfg.rig, "created": cfg.created, "notes": cfg.notes}
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------------------------- rig yaml

def _camera(role: str, m: _Map) -> CameraConfig:
    res = m.size("resolution")
    mm = m.sub("model")
    typ = mm.str("type", choices=CAMERA_MODEL_TYPES)
    fx, fy, cx, cy = mm.vec("intrinsics", 4)
    if fx <= 0 or fy <= 0:
        raise mm.err("intrinsics", f"fx and fy must be > 0, got {fx}, {fy}")
    if not (-0.5 <= cx <= res[0] - 0.5 and -0.5 <= cy <= res[1] - 0.5):
        raise mm.err("intrinsics", f"principal point ({cx}, {cy}) is outside the {res[0]}x{res[1]} image")
    model = CameraModel(typ, (fx, fy, cx, cy), mm.vec("distortion", 4))
    mm.finish()
    cam = CameraConfig(role=role, sensor=m.str("sensor"), resolution=res, fps=m.float("fps", positive=True),
                       model=model, T_imu_cam=_transform(m.sub("T_imu_cam")),
                       timestamp_ref=m.str("timestamp_ref", choices=TIMESTAMP_REFS),
                       time_offset_ns=m.int("time_offset_ns"), provenance=m.str("provenance", choices=PROVENANCE))
    m.finish()
    return cam


def parse_rig(d: Any, src: str = "<rig>") -> RigConfig:
    """Validate a decoded rig YAML document (``src`` names it in error messages)."""
    m = _Map(d, "", src)
    schema = m.str("schema")
    if schema != RIG_SCHEMA:
        raise m.err("schema", f"expected {RIG_SCHEMA!r}, got {schema!r}")
    rig_id = m.str("rig_id", pattern=NAME_RE)
    notes = m.opt("notes", "") or ""
    if not isinstance(notes, str):
        raise m.err("notes", f"expected a string, got {notes!r}")

    cm = m.sub("cameras")
    cameras = {role: _camera(role, cm.sub(role)) for role in CAMERA_ROLES}
    cm.finish()
    if cameras["ultrawide"].model.type != "fisheye-equidistant":
        raise cm.err("ultrawide.model.type", "the ultrawide must be fisheye-equidistant")

    vm = m.sub("virtual_ultrawide")
    vres = vm.size("resolution")
    vint = vm.vec("intrinsics", 4)
    if vint[0] <= 0 or vint[1] <= 0:
        raise vm.err("intrinsics", "fx and fy must be > 0")
    R_phys_virt = _rotation(vm.sub("rotation")) if vm.has("rotation") else np.eye(3)
    virtual = VirtualCamera(vres, vint, R_phys_virt)
    vm.finish()

    tm = m.sub("tcp")
    T_cam_tcp = _transform(tm.sub("T_cam_tcp"))
    tcp_prov = tm.str("provenance", choices=PROVENANCE)
    tm.finish()

    im = m.sub("imu")
    nm = im.sub("noise")
    noise_keys = ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
                  "accelerometer_random_walk")
    vals = {k: nm.float(k, positive=True) for k in noise_keys}
    nm.finish()
    if im.has("vio_inflation"):
        vi = im.sub("vio_inflation")
        vals.update(inflation_noise=vi.float("noise_density", positive=True),
                    inflation_random_walk=vi.float("random_walk", positive=True))
        vi.finish()
    noise = ImuNoise(**vals)
    imu = ImuConfig(im.str("sensor"), noise, im.int("time_offset_ns"), im.str("provenance", choices=PROVENANCE))
    im.finish()

    am = m.sub("aruco")
    aruco = ArucoConfig(am.str("dict"), am.float("marker_size_m", positive=True), am.int("left_id", 0),
                        am.int("right_id", 0), am.float("nominal_z_m", positive=True))
    am.finish()
    import cv2

    if not aruco.dict.startswith("DICT_") or not isinstance(getattr(cv2.aruco, aruco.dict, None), int):
        raise am.err("dict", f"not an OpenCV predefined ArUco dictionary: {aruco.dict!r}")
    if aruco.left_id == aruco.right_id:
        raise am.err("right_id", "left_id and right_id must differ")

    gripper = None
    if m.has("gripper"):
        gm = m.sub("gripper")
        gripper = GripperConfig(gm.float("max_opening_m", positive=True),
                                gm.float("opening_tolerance_m", positive=True))
        gm.finish()

    T_world_offset = _world_offset(m)
    m.finish()
    return RigConfig(schema=schema, rig_id=rig_id, cameras=cameras, virtual_ultrawide=virtual, T_cam_tcp=T_cam_tcp,
                     tcp_provenance=tcp_prov, imu=imu, aruco=aruco, T_world_offset=T_world_offset, gripper=gripper,
                     notes=notes.strip(), source_path=src)


def load_rig(path: str | os.PathLike) -> RigConfig:
    """Load and validate ``rigs/<rig_id>.yaml``. The file stem must equal ``rig_id``."""
    path = Path(path)
    rig = parse_rig(_load_yaml(path), str(path))
    if path.stem != rig.rig_id:
        raise ConfigError(f"{path}: rig_id: {rig.rig_id!r} does not match the file name {path.stem!r}")
    return rig
