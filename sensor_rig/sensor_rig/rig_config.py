"""rig.yaml: typed, validated description of the sensor rig (physical cameras, virtual iPhone cameras, IMU, VIO,
export settings) plus derived geometry. Single source of truth for extrinsics, time offsets and remaps.

Conventions: ``T_a_b`` maps points in frame b to frame a (4x4, metres); camera frames are OpenCV optical; times are
CLOCK_MONOTONIC integer ns and ``t_capture = t_stamp + offset``. Frames for :meth:`RigConfig.T`: ``imu``, each
physical camera name (``main``, ``ultrawide``) and ``virtual_<name>`` for each virtual camera.

Usage::

    rig = load_rig("rig.yaml")            # errors raise RigConfigError("<yaml.path>: ...")
    warnings = validate(rig)              # placeholders, coverage below min_coverage
    img_v = rig.remap("main", "export").apply(img_phys)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sensor_rig.camera_model import (
    FisheyeEquidistant, PinholeRadtan, RemapTable, VirtualPinhole, build_inverse_remap, build_remap, coverage,
    model_from_dict, model_to_dict,
)
from sensor_rig.transforms import D_CV_ARKIT, R_from_euler_xyz_deg, inv_T, is_rigid, make_T, rot_y

__all__ = [
    "RigConfigError", "TimeOffsets", "PhysicalCamera", "VirtualCameraSpec", "ImuNoise", "ImuConfig", "VioConfig",
    "ExportConfig", "RigConfig", "load_rig", "validate", "DEFAULT_RIG_PATH", "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1
DEFAULT_RIG_PATH = Path(__file__).resolve().parent / "configs" / "rig_default.yaml"
VIRTUAL_NAMES = ("main", "ultrawide")

# ICM-42688-P settings (datasheet DS-000347): accel FS, gyro FS, UI ODRs (integer Hz), UI filter bandwidth options.
ICM42688_ACCEL_RANGES_G = (2, 4, 8, 16)
ICM42688_GYRO_RANGES_DPS = (15.625, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0)
ICM42688_ODRS_HZ = (25, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000)
ICM42688_UI_FILTER_BW = ("odr_div_2", "odr_div_4", "odr_div_5", "odr_div_8", "odr_div_10", "odr_div_16",
                         "odr_div_20", "odr_div_40", "low_latency_0", "low_latency_1")
ICM42688_SPI_MAX_HZ = 24_000_000

_HEADER = """\
# Sensor rig description (sensor_rig rig.yaml, schema_version 1).
# Conventions: T_a_b maps points in frame b to frame a (pose of b in a), metres; camera frames are OpenCV optical
# (x right, y down, z forward); times are CLOCK_MONOTONIC integer ns, t_capture = t_stamp + offset.
# Sections marked placeholder: true are guesses; replace them with `python -m sensor_rig calib-import`
# (sensor_rig/docs/calibration.md).
"""


class RigConfigError(ValueError):
    """Invalid rig.yaml. The message starts with the yaml path of the bad field, e.g. ``cameras.main.fps: ...``."""


# --------------------------------------------------------------------------------------------- dataclasses

@dataclass(frozen=True)
class TimeOffsets:
    offset_prior_ns: int                  # known correction raw stamp -> mid-exposure (e.g. -30 ms for appsink stamps)
    kalibr_timeshift_cam_imu_s: float     # Kalibr's t_imu = t_cam + shift, estimated on bags stamped with the prior

    @property
    def total_offset_ns(self) -> int:
        return int(self.offset_prior_ns) + int(round(self.kalibr_timeshift_cam_imu_s * 1e9))


@dataclass(frozen=True, eq=False)
class PhysicalCamera:
    name: str
    sensor: str
    sensor_id: int
    sensor_mode: int | None
    resolution: tuple[int, int]
    fps: float
    model: PinholeRadtan | FisheyeEquidistant
    T_imu_cam: np.ndarray
    time: TimeOffsets
    argus: dict[str, Any]
    placeholder: bool


@dataclass(frozen=True, eq=False)
class VirtualCameraSpec:
    name: str                             # "main" | "ultrawide"
    source: str                           # physical camera name
    reference: VirtualPinhole             # iPhone K at the reference resolution
    R_phys_virt: np.ndarray               # 3x3, d_phys = R_phys_virt @ d_virt
    outputs: dict[str, tuple[int, int]]   # purpose -> (w, h), e.g. {"export": (1920, 1440), "deploy": (960, 720)}
    min_coverage: float

    def size(self, purpose_or_size) -> tuple[int, int]:
        if isinstance(purpose_or_size, str):
            if purpose_or_size == "reference":
                return self.reference.size
            if purpose_or_size not in self.outputs:
                raise KeyError(f"virtual camera {self.name!r} has no output {purpose_or_size!r} "
                               f"(known: reference, {', '.join(self.outputs)})")
            return self.outputs[purpose_or_size]
        w, h = purpose_or_size
        return (int(w), int(h))

    def camera(self, purpose_or_size) -> VirtualPinhole:
        """The virtual camera at an output purpose (``"export"``, ``"deploy"``, ``"reference"``) or a ``(w, h)``."""
        w, h = self.size(purpose_or_size)
        return self.reference if (w, h) == self.reference.size else self.reference.scaled(w, h)


@dataclass(frozen=True)
class ImuNoise:
    """Continuous-time IMU noise (Kalibr / allan_variance_ros units)."""

    accelerometer_noise_density: float    # m/s^2/sqrt(Hz)
    accelerometer_random_walk: float      # m/s^3/sqrt(Hz)
    gyroscope_noise_density: float        # rad/s/sqrt(Hz)
    gyroscope_random_walk: float          # rad/s^2/sqrt(Hz)
    inflation_noise: float = 1.0
    inflation_random_walk: float = 1.0

    def for_vio(self) -> "ImuNoise":
        """Inflated values for VIO configs (densities x inflation_noise, walks x inflation_random_walk)."""
        return ImuNoise(self.accelerometer_noise_density * self.inflation_noise,
                        self.accelerometer_random_walk * self.inflation_random_walk,
                        self.gyroscope_noise_density * self.inflation_noise,
                        self.gyroscope_random_walk * self.inflation_random_walk, 1.0, 1.0)

    def kalibr_dict(self) -> dict[str, float]:
        """The four values under their Kalibr ``imu.yaml`` names."""
        return {k: getattr(self, k) for k in ("accelerometer_noise_density", "accelerometer_random_walk",
                                              "gyroscope_noise_density", "gyroscope_random_walk")}


@dataclass(frozen=True)
class ImuConfig:
    type: str
    spi_device: str
    spi_speed_hz: int
    spi_mode: int
    odr_hz: int
    output_rate_hz: int
    accel_range_g: int
    gyro_range_dps: float
    accel_ui_filter_bw: str
    gyro_ui_filter_bw: str
    fifo: dict[str, Any]
    noise: ImuNoise
    time_offset_ns: int
    placeholder: bool


@dataclass(frozen=True)
class VioConfig:
    camera: str                           # physical camera the VIO uses
    trajectory_frame: str                 # frame S whose pose the TUM trajectory holds: imu | <camera name>


@dataclass(frozen=True)
class ExportConfig:
    device_identifier: str
    device_name: str
    gripper_id: str
    world_offset_translation_m: tuple[float, float, float]
    world_offset_yaw_deg: float

    @property
    def T_world_offset(self) -> np.ndarray:
        """4x4 applied on the left in the y-up world: rotation about +y by yaw, then translation."""
        return make_T(rot_y(self.world_offset_yaw_deg), self.world_offset_translation_m)


@dataclass(frozen=True, eq=False)
class RigConfig:
    schema_version: int
    rig_name: str
    cameras: dict[str, PhysicalCamera]
    virtual: dict[str, VirtualCameraSpec]
    imu: ImuConfig
    vio: VioConfig
    export: ExportConfig
    calibration: dict[str, Any]
    source_path: str | None = None
    capture: dict[str, Any] = field(default_factory=dict)   # recorder-only settings, passed through (R3)
    notes: str | None = None
    load_warnings: tuple[str, ...] = ()
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ---- derived geometry
    @property
    def frames(self) -> list[str]:
        return ["imu", *self.cameras, *(f"virtual_{n}" for n in self.virtual)]

    def T_imu(self, frame: str) -> np.ndarray:
        """Pose of ``frame`` in the IMU frame (OpenCV axes for cameras)."""
        if frame in ("imu", "imu0"):
            return np.eye(4)
        if frame in self.cameras:
            return self.cameras[frame].T_imu_cam.copy()
        if frame.startswith("virtual_") and frame[8:] in self.virtual:
            v = self.virtual[frame[8:]]
            return self.cameras[v.source].T_imu_cam @ make_T(v.R_phys_virt)
        raise KeyError(f"unknown frame {frame!r}; known: {self.frames}")

    def T(self, a: str, b: str) -> np.ndarray:
        """``T_a_b`` = inv(T_imu(a)) @ T_imu(b)."""
        return inv_T(self.T_imu(a)) @ self.T_imu(b)

    def T_S_virtual_main_arkit(self, S: str | None = None) -> np.ndarray:
        """Virtual main camera with ARKit axes in frame S (default ``vio.trajectory_frame``):
        export uses ``T_W_main_arkit = T_W_S @ this``."""
        return self.T(S or self.vio.trajectory_frame, "virtual_main") @ D_CV_ARKIT

    # ---- cameras / remaps
    def virtual_camera(self, name: str, purpose_or_size) -> VirtualPinhole:
        return self.virtual[name].camera(purpose_or_size)

    def remap(self, name: str, purpose_or_size) -> RemapTable:
        """Forward remap physical -> virtual camera ``name`` at a purpose or ``(w, h)``; cached per instance."""
        v = self.virtual[name]
        size = v.size(purpose_or_size)
        key = ("fwd", name, size)
        if key not in self._cache:
            self._cache[key] = build_remap(self.cameras[v.source].model, v.camera(size), v.R_phys_virt)
        return self._cache[key]

    def inverse_remap(self, name: str, virt: VirtualPinhole | None = None) -> RemapTable:
        """Virtual -> physical remap for the simulator (``virt`` defaults to the reference camera); cached."""
        v = self.virtual[name]
        virt = virt or v.reference
        key = ("inv", name, virt.size, virt.fx, virt.fy, virt.cx, virt.cy)
        if key not in self._cache:
            self._cache[key] = build_inverse_remap(virt, self.cameras[v.source].model, v.R_phys_virt)
        return self._cache[key]

    def coverage_report(self) -> dict[str, float]:
        """Coverage of each virtual camera at its reference resolution."""
        key = ("coverage",)
        if key not in self._cache:
            self._cache[key] = {n: coverage(self.cameras[v.source].model, v.reference, v.R_phys_virt)
                                for n, v in self.virtual.items()}
        return dict(self._cache[key])

    def capture_time_ns(self, cam: str, t_stamp_ns):
        """``t_stamp + total offset`` (cameras) or ``+ imu.time_offset_ns`` (``imu``). Exact on int64 arrays."""
        if cam in ("imu", "imu0"):
            off = int(self.imu.time_offset_ns)
        elif cam in self.cameras:
            off = self.cameras[cam].time.total_offset_ns
        else:
            raise KeyError(f"unknown sensor {cam!r}; known: imu, {', '.join(self.cameras)}")
        if isinstance(t_stamp_ns, (int, np.integer)) and not isinstance(t_stamp_ns, bool):
            return int(t_stamp_ns) + off
        a = np.asarray(t_stamp_ns)
        if a.dtype.kind not in "iu":
            raise TypeError(f"timestamps must be integer ns, got dtype {a.dtype}")
        return a.astype(np.int64) + np.int64(off)

    # ---- misc
    @property
    def placeholder(self) -> bool:
        return self.imu.placeholder or any(c.placeholder for c in self.cameras.values())

    def to_dict(self) -> dict:
        """Canonical rig.yaml content (plain Python types)."""
        d: dict[str, Any] = {"schema_version": self.schema_version, "rig_name": self.rig_name}
        if self.notes:
            d["notes"] = self.notes
        d["cameras"] = {n: _camera_to_dict(c) for n, c in self.cameras.items()}
        d["virtual_cameras"] = {n: _virtual_to_dict(v) for n, v in self.virtual.items()}
        d["imu"] = _imu_to_dict(self.imu)
        d["vio"] = {"camera": self.vio.camera, "trajectory_frame": self.vio.trajectory_frame}
        e = self.export
        d["export"] = {"device_identifier": e.device_identifier, "device_name": e.device_name,
                       "gripper_id": e.gripper_id,
                       "world_offset": {"translation_m": [float(x) for x in e.world_offset_translation_m],
                                        "yaw_deg": float(e.world_offset_yaw_deg)}}
        if self.capture:
            d["capture"] = _plain(self.capture)
        d["calibration"] = _plain(self.calibration)
        return d

    def fingerprint(self) -> str:
        """sha256 of the canonical content, excluding provenance (``calibration``) and ``notes``."""
        d = self.to_dict()
        d.pop("calibration", None)
        d.pop("notes", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def dump(self, path: str | os.PathLike) -> None:
        """Write canonical rig.yaml (header comment, stable key order). ``load_rig`` reads it back identically."""
        text = _HEADER + yaml.dump(self.to_dict(), Dumper=_Dumper, sort_keys=False, width=120, allow_unicode=True)
        Path(path).write_text(text)

    def dumps(self) -> str:
        return _HEADER + yaml.dump(self.to_dict(), Dumper=_Dumper, sort_keys=False, width=120, allow_unicode=True)

    def with_virtual_reference(self, name: str, K, size) -> "RigConfig":
        """Copy with ``virtual[name].reference`` replaced (sim: the iPhone episode's own K, e.g. the 1278 variant)."""
        v = self.virtual[name]
        ref = VirtualPinhole.from_K(K, size)
        for purpose, (w, h) in v.outputs.items():
            if w * ref.height != h * ref.width:
                raise RigConfigError(f"virtual_cameras.{name}.outputs.{purpose}: {w}x{h} is not a uniform scale "
                                     f"of {ref.width}x{ref.height}")
        return self.replace(virtual={**self.virtual, name: dataclasses.replace(v, reference=ref)})

    def replace(self, **changes) -> "RigConfig":
        return dataclasses.replace(self, **changes)


# --------------------------------------------------------------------------------------------- serialisation helpers

def _plain(o):
    """Convert numpy scalars/arrays and tuples to plain Python for yaml/json."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def _rotation_to_dict(R: np.ndarray) -> dict:
    if np.array_equal(R, np.eye(3)):
        return {"euler_xyz_deg": [0.0, 0.0, 0.0]}
    return {"matrix": [[float(x) for x in row] for row in R]}


def _camera_to_dict(c: PhysicalCamera) -> dict:
    return {
        "placeholder": bool(c.placeholder), "sensor": c.sensor, "sensor_id": int(c.sensor_id),
        "sensor_mode": None if c.sensor_mode is None else int(c.sensor_mode),
        "resolution": [int(c.resolution[0]), int(c.resolution[1])], "fps": float(c.fps),
        "argus": _plain(c.argus), "model": model_to_dict(c.model),
        "T_imu_cam": {"rotation": _rotation_to_dict(c.T_imu_cam[:3, :3]),
                      "translation_m": [float(x) for x in c.T_imu_cam[:3, 3]]},
        "time": {"offset_prior_ns": int(c.time.offset_prior_ns),
                 "kalibr_timeshift_cam_imu_s": float(c.time.kalibr_timeshift_cam_imu_s)},
    }


def _virtual_to_dict(v: VirtualCameraSpec) -> dict:
    r = v.reference
    return {"source": v.source,
            "reference": {"resolution": [r.width, r.height], "intrinsics": [r.fx, r.fy, r.cx, r.cy]},
            "rotation": _rotation_to_dict(v.R_phys_virt),
            "outputs": {k: [int(w), int(h)] for k, (w, h) in v.outputs.items()},
            "min_coverage": float(v.min_coverage)}


def _imu_to_dict(m: ImuConfig) -> dict:
    n = m.noise
    return {
        "placeholder": bool(m.placeholder), "type": m.type,
        "spi": {"device": m.spi_device, "speed_hz": int(m.spi_speed_hz), "mode": int(m.spi_mode)},
        "odr_hz": int(m.odr_hz), "output_rate_hz": int(m.output_rate_hz),
        "accel_range_g": int(m.accel_range_g), "gyro_range_dps": float(m.gyro_range_dps),
        "accel_ui_filter_bw": m.accel_ui_filter_bw, "gyro_ui_filter_bw": m.gyro_ui_filter_bw,
        "fifo": _plain(m.fifo),
        "noise": {**{k: float(v) for k, v in n.kalibr_dict().items()},
                  "vio_inflation": {"noise_density": float(n.inflation_noise),
                                    "random_walk": float(n.inflation_random_walk)}},
        "time": {"offset_ns": int(m.time_offset_ns)},
    }


class _Dumper(yaml.SafeDumper):
    """Block mappings; flow style for lists of scalars and matrices; no aliases."""

    def ignore_aliases(self, data):
        return True


def _is_scalar(x) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _represent_list(dumper, data):
    flow = all(_is_scalar(x) or (isinstance(x, list) and all(_is_scalar(y) for y in x)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


# --------------------------------------------------------------------------------------------- parsing

class _Map:
    """Mapping reader that records which keys were used and reports errors with their yaml path."""

    def __init__(self, d, path: str):
        if not isinstance(d, dict):
            raise RigConfigError(f"{path or '<root>'}: must be a mapping, got {type(d).__name__}")
        self.d, self.path, self.used = d, path, set()

    def p(self, key: str) -> str:
        return f"{self.path}.{key}" if self.path else key

    def has(self, key: str) -> bool:
        return key in self.d

    def req(self, key: str):
        if key not in self.d:
            raise RigConfigError(f"{self.p(key)}: missing required key")
        self.used.add(key)
        return self.d[key]

    def opt(self, key: str, default=None):
        self.used.add(key)
        return self.d.get(key, default)

    def sub(self, key: str, required: bool = True) -> "_Map":
        v = self.req(key) if required else self.opt(key, {})
        return _Map({} if v is None else v, self.p(key))

    def finish(self, strict: bool, warnings: list[str]) -> None:
        for k in self.d:
            if k not in self.used:
                msg = f"{self.p(str(k))}: unknown key"
                if strict:
                    raise RigConfigError(msg)
                warnings.append(msg + " (ignored)")


def _int(v, path, *, min_value=None, allow_none=False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        else:
            raise RigConfigError(f"{path}: expected an integer, got {v!r}")
    v = int(v)
    if min_value is not None and v < min_value:
        raise RigConfigError(f"{path}: must be >= {min_value}, got {v}")
    return v


def _float(v, path, *, positive=False, min_value=None, max_value=None):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise RigConfigError(f"{path}: expected a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise RigConfigError(f"{path}: must be finite, got {v}")
    if positive and v <= 0:
        raise RigConfigError(f"{path}: must be > 0, got {v}")
    if min_value is not None and v < min_value:
        raise RigConfigError(f"{path}: must be >= {min_value}, got {v}")
    if max_value is not None and v > max_value:
        raise RigConfigError(f"{path}: must be <= {max_value}, got {v}")
    return v


def _str(v, path, *, choices=None):
    if not isinstance(v, str) or not v:
        raise RigConfigError(f"{path}: expected a non-empty string, got {v!r}")
    if choices is not None and v not in choices:
        raise RigConfigError(f"{path}: must be one of {list(choices)}, got {v!r}")
    return v


def _bool(v, path):
    if not isinstance(v, bool):
        raise RigConfigError(f"{path}: expected true/false, got {v!r}")
    return v


def _vec(v, n, path):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise RigConfigError(f"{path}: expected a list of {n} numbers, got {v!r}")
    return [_float(x, f"{path}[{i}]") for i, x in enumerate(v)]


def _size(v, path):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise RigConfigError(f"{path}: expected [width, height], got {v!r}")
    return tuple(_int(x, f"{path}[{i}]", min_value=1) for i, x in enumerate(v))


def _mapping(v, path):
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise RigConfigError(f"{path}: must be a mapping, got {type(v).__name__}")
    return dict(v)


def _rotation(m: _Map, strict, warnings) -> np.ndarray:
    keys = [k for k in ("matrix", "euler_xyz_deg") if m.has(k)]
    if len(keys) != 1:
        raise RigConfigError(f"{m.path}: give exactly one of 'matrix' or 'euler_xyz_deg'")
    if keys[0] == "matrix":
        raw = m.req("matrix")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RigConfigError(f"{m.p('matrix')}: expected a 3x3 matrix")
        R = np.array([_vec(row, 3, f"{m.p('matrix')}[{i}]") for i, row in enumerate(raw)])
    else:
        R = R_from_euler_xyz_deg(*_vec(m.req("euler_xyz_deg"), 3, m.p("euler_xyz_deg")))
    m.finish(strict, warnings)
    if not is_rigid(make_T(R)):
        raise RigConfigError(f"{m.path}: not a rotation (must be orthonormal to 1e-6 with det = +1, "
                             f"det = {np.linalg.det(R):.6f})")
    return R


def _transform(m: _Map, strict, warnings) -> np.ndarray:
    if m.has("matrix"):
        raw = m.req("matrix")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise RigConfigError(f"{m.p('matrix')}: expected a 4x4 matrix")
        T = np.array([_vec(row, 4, f"{m.p('matrix')}[{i}]") for i, row in enumerate(raw)])
        m.finish(strict, warnings)
        if not is_rigid(T):
            raise RigConfigError(f"{m.p('matrix')}: not a rigid transform (orthonormal rotation, det +1, "
                                 "last row [0,0,0,1])")
        return T
    R = _rotation(m.sub("rotation"), strict, warnings)
    t = _vec(m.req("translation_m"), 3, m.p("translation_m"))
    m.finish(strict, warnings)
    return make_T(R, t)


def _model(m_raw, path, size):
    try:
        model = model_from_dict(m_raw, size)
    except ValueError as e:
        msg = str(e)
        raise RigConfigError(f"{path}.{msg}" if not msg.startswith(":") else f"{path}{msg}") from None
    if model.kind == "pinhole":
        raise RigConfigError(f"{path}.type: physical cameras must be pinhole-radtan or fisheye-equidistant")
    _check_pp(model, f"{path}.intrinsics")
    return model


def _check_pp(model, path):
    if not (-0.5 <= model.cx <= model.width - 0.5 and -0.5 <= model.cy <= model.height - 0.5):
        raise RigConfigError(f"{path}: principal point ({model.cx}, {model.cy}) is outside the "
                             f"{model.width}x{model.height} image")


def _parse_camera(name, m: _Map, strict, w) -> PhysicalCamera:
    res = _size(m.req("resolution"), m.p("resolution"))
    model = _model(m.req("model"), m.p("model"), res)
    tm = m.sub("time")
    time = TimeOffsets(_int(tm.req("offset_prior_ns"), tm.p("offset_prior_ns")),
                       _float(tm.opt("kalibr_timeshift_cam_imu_s", 0.0), tm.p("kalibr_timeshift_cam_imu_s")))
    tm.finish(strict, w)
    cam = PhysicalCamera(
        name=name,
        sensor=_str(m.req("sensor"), m.p("sensor")),
        sensor_id=_int(m.req("sensor_id"), m.p("sensor_id"), min_value=0),
        sensor_mode=_int(m.opt("sensor_mode"), m.p("sensor_mode"), min_value=0, allow_none=True),
        resolution=res,
        fps=_float(m.req("fps"), m.p("fps"), positive=True),
        model=model,
        T_imu_cam=_transform(m.sub("T_imu_cam"), strict, w),
        time=time,
        argus=_mapping(m.opt("argus", {}), m.p("argus")),
        placeholder=_bool(m.opt("placeholder", False), m.p("placeholder")),
    )
    m.finish(strict, w)
    return cam


def _parse_virtual(name, m: _Map, cameras, strict, w) -> VirtualCameraSpec:
    if name not in VIRTUAL_NAMES:
        raise RigConfigError(f"{m.path}: virtual camera name must be one of {list(VIRTUAL_NAMES)}")
    source = _str(m.req("source"), m.p("source"))
    if source not in cameras:
        raise RigConfigError(f"{m.p('source')}: no physical camera named {source!r} (have {list(cameras)})")
    rm = m.sub("reference")
    ref_size = _size(rm.req("resolution"), rm.p("resolution"))
    fx, fy, cx, cy = _vec(rm.req("intrinsics"), 4, rm.p("intrinsics"))
    if fx <= 0 or fy <= 0:
        raise RigConfigError(f"{rm.p('intrinsics')}: focal lengths must be > 0")
    rm.finish(strict, w)
    ref = VirtualPinhole(ref_size[0], ref_size[1], fx, fy, cx, cy)
    _check_pp(ref, rm.p("intrinsics"))
    R = _rotation(m.sub("rotation"), strict, w) if m.has("rotation") else np.eye(3)
    om = m.sub("outputs")
    outputs = {}
    for purpose in list(om.d):
        size = _size(om.req(purpose), om.p(str(purpose)))
        if size[0] * ref.height != size[1] * ref.width:
            raise RigConfigError(f"{om.p(str(purpose))}: {size[0]}x{size[1]} is not a uniform scale of the "
                                 f"reference {ref.width}x{ref.height}")
        outputs[str(purpose)] = size
    spec = VirtualCameraSpec(name=name, source=source, reference=ref, R_phys_virt=R, outputs=outputs,
                             min_coverage=_float(m.opt("min_coverage", 0.99), m.p("min_coverage"),
                                                 min_value=0.0, max_value=1.0))
    m.finish(strict, w)
    return spec


def _parse_imu(m: _Map, strict, w) -> ImuConfig:
    sm = m.sub("spi")
    spi = (_str(sm.req("device"), sm.p("device")),
           _int(sm.req("speed_hz"), sm.p("speed_hz"), min_value=1),
           _int(sm.opt("mode", 0), sm.p("mode"), min_value=0))
    if spi[1] > ICM42688_SPI_MAX_HZ:
        raise RigConfigError(f"{sm.p('speed_hz')}: ICM-42688-P SPI clock is at most {ICM42688_SPI_MAX_HZ} Hz")
    if spi[2] not in (0, 3):
        raise RigConfigError(f"{sm.p('mode')}: ICM-42688-P supports SPI mode 0 or 3, got {spi[2]}")
    sm.finish(strict, w)
    odr = _int(m.req("odr_hz"), m.p("odr_hz"))
    if odr not in ICM42688_ODRS_HZ:
        raise RigConfigError(f"{m.p('odr_hz')}: must be an ICM-42688-P ODR {list(ICM42688_ODRS_HZ)}, got {odr}")
    out_rate = _int(m.opt("output_rate_hz", odr), m.p("output_rate_hz"), min_value=1)
    if out_rate > odr:
        raise RigConfigError(f"{m.p('output_rate_hz')}: must be <= odr_hz ({odr}), got {out_rate}")
    acc = _int(m.req("accel_range_g"), m.p("accel_range_g"))
    if acc not in ICM42688_ACCEL_RANGES_G:
        raise RigConfigError(f"{m.p('accel_range_g')}: must be one of {list(ICM42688_ACCEL_RANGES_G)}, got {acc}")
    gyr = _float(m.req("gyro_range_dps"), m.p("gyro_range_dps"))
    if gyr not in ICM42688_GYRO_RANGES_DPS:
        raise RigConfigError(f"{m.p('gyro_range_dps')}: must be one of {list(ICM42688_GYRO_RANGES_DPS)}, got {gyr}")
    nm = m.sub("noise")
    vals = {k: _float(nm.req(k), nm.p(k), positive=True)
            for k in ("accelerometer_noise_density", "accelerometer_random_walk",
                      "gyroscope_noise_density", "gyroscope_random_walk")}
    im = nm.sub("vio_inflation", required=False)
    noise = ImuNoise(**vals, inflation_noise=_float(im.opt("noise_density", 1.0), im.p("noise_density"),
                                                    positive=True),
                     inflation_random_walk=_float(im.opt("random_walk", 1.0), im.p("random_walk"), positive=True))
    im.finish(strict, w)
    nm.finish(strict, w)
    tm = m.sub("time", required=False)
    t_off = _int(tm.opt("offset_ns", 0), tm.p("offset_ns"))
    tm.finish(strict, w)
    cfg = ImuConfig(
        type=_str(m.req("type"), m.p("type"), choices=("icm42688p",)),
        spi_device=spi[0], spi_speed_hz=spi[1], spi_mode=spi[2], odr_hz=odr, output_rate_hz=out_rate,
        accel_range_g=acc, gyro_range_dps=gyr,
        accel_ui_filter_bw=_str(m.opt("accel_ui_filter_bw", "odr_div_4"), m.p("accel_ui_filter_bw"),
                                choices=ICM42688_UI_FILTER_BW),
        gyro_ui_filter_bw=_str(m.opt("gyro_ui_filter_bw", "odr_div_4"), m.p("gyro_ui_filter_bw"),
                               choices=ICM42688_UI_FILTER_BW),
        fifo=_mapping(m.opt("fifo", {}), m.p("fifo")), noise=noise, time_offset_ns=t_off,
        placeholder=_bool(m.opt("placeholder", False), m.p("placeholder")),
    )
    m.finish(strict, w)
    return cfg


def _parse(d: dict, source_path: str | None, strict: bool) -> RigConfig:
    w: list[str] = []
    root = _Map(d, "")
    ver = _int(root.req("schema_version"), "schema_version")
    if ver != SCHEMA_VERSION:
        raise RigConfigError(f"schema_version: unsupported version {ver} (this code reads {SCHEMA_VERSION})")
    rig_name = _str(root.req("rig_name"), "rig_name")
    notes = root.opt("notes")
    if notes is not None and not isinstance(notes, str):
        raise RigConfigError("notes: must be a string")
    notes = notes.strip() if notes else None

    cm = root.sub("cameras")
    if not cm.d:
        raise RigConfigError("cameras: at least one physical camera is required")
    cameras = {}
    for name in list(cm.d):
        cameras[str(name)] = _parse_camera(str(name), cm.sub(name), strict, w)
    if "imu" in cameras or any(str(n).startswith("virtual_") for n in cameras):
        raise RigConfigError("cameras: names 'imu' and 'virtual_*' are reserved frame names")

    vm = root.sub("virtual_cameras")
    for name in VIRTUAL_NAMES:
        if name not in vm.d:
            raise RigConfigError(f"virtual_cameras.{name}: missing required key")
    virtual = {str(n): _parse_virtual(str(n), vm.sub(n), cameras, strict, w) for n in list(vm.d)}

    imu = _parse_imu(root.sub("imu"), strict, w)

    vio_m = root.sub("vio")
    vio = VioConfig(camera=_str(vio_m.req("camera"), vio_m.p("camera"), choices=list(cameras)),
                    trajectory_frame=_str(vio_m.opt("trajectory_frame", "imu"), vio_m.p("trajectory_frame"),
                                          choices=["imu", *cameras]))
    vio_m.finish(strict, w)

    em = root.sub("export")
    wo = em.sub("world_offset")
    export = ExportConfig(
        device_identifier=_str(em.req("device_identifier"), em.p("device_identifier")),
        device_name=_str(em.req("device_name"), em.p("device_name")),
        gripper_id=_str(em.opt("gripper_id", "default"), em.p("gripper_id")),
        world_offset_translation_m=tuple(_vec(wo.req("translation_m"), 3, wo.p("translation_m"))),
        world_offset_yaw_deg=_float(wo.req("yaw_deg"), wo.p("yaw_deg")),
    )
    wo.finish(strict, w)
    em.finish(strict, w)
    if np.allclose(export.T_world_offset, np.eye(4), atol=1e-9):
        raise RigConfigError("export.world_offset: must not be the identity (an identity pose becomes NaN in "
                             "iphumi's pose_6d_to_4x4; brief D4)")

    capture = _mapping(root.opt("capture", {}), "capture")
    calibration = _mapping(root.opt("calibration", {}), "calibration")
    root.finish(strict, w)
    return RigConfig(schema_version=ver, rig_name=rig_name, cameras=cameras, virtual=virtual, imu=imu, vio=vio,
                     export=export, calibration=calibration, source_path=source_path, capture=capture,
                     notes=notes, load_warnings=tuple(w))


def load_rig(src: str | os.PathLike | dict, *, strict: bool = True) -> RigConfig:
    """Load and validate a rig from a yaml path or an already-parsed dict.

    A bare name that does not exist relative to the cwd (e.g. ``"rig_default.yaml"``) is looked up in
    ``sensor_rig/configs/``. ``strict=False`` turns unknown keys into load warnings instead of errors.
    """
    if isinstance(src, dict):
        return _parse(src, None, strict)
    p = Path(src)
    if not p.exists() and not p.is_absolute() and len(p.parts) == 1:
        alt = DEFAULT_RIG_PATH.parent / p
        if alt.exists():
            p = alt
    try:
        d = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise RigConfigError(f"<root>: {p} is not valid yaml: {e}") from None
    return _parse(d, str(p.resolve()), strict)


def validate(cfg: RigConfig) -> list[str]:
    """Non-fatal warnings: unknown keys (non-strict load), placeholders, coverage below ``min_coverage``."""
    out = list(cfg.load_warnings)
    for n, c in cfg.cameras.items():
        if c.placeholder:
            out.append(f"cameras.{n}: placeholder calibration (replace with calib-import)")
    if cfg.imu.placeholder:
        out.append("imu: placeholder noise/settings (replace with calib-import --imu)")
    for n, cov in cfg.coverage_report().items():
        if cov < cfg.virtual[n].min_coverage:
            out.append(f"virtual_cameras.{n}: coverage {cov:.4f} < min_coverage {cfg.virtual[n].min_coverage} "
                       "(the physical lens does not see the whole virtual image)")
    return out
