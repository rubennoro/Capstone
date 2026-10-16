"""Pure logic for the Argus/GStreamer camera backend (unit-tested on a laptop; no GStreamer imports here).

* ``build_argus_pipeline``: the ``nvarguscamerasrc ... ! appsink`` launch string (R4 §1.5).
* ``argus_params_from_camera``: maps a P2 ``PhysicalCamera`` (``sensor_id``, ``sensor_mode``, ``resolution``, ``fps``,
  ``argus`` dict) to ``build_argus_pipeline`` keyword arguments.
* ``pts_to_monotonic``: buffer PTS -> CLOCK_MONOTONIC (the pipeline clock is forced to the MONOTONIC system clock, so
  ``clock_time = base_time + running_time`` and, with a zero-start segment, ``running_time == pts``).
* ``LatencyTracker``: running min/median of ``t_arrival - t_ns``; flags median drift > 5 ms.
* ``detect_sensor_gaps`` / ``GapDetector``: frames lost before userspace (dt > 1.5 / fps, or backend seq gaps).

Timestamp caveat (R4 §1.5, §10): the nvarguscamerasrc PTS is the time the source *received* the buffer from Argus,
not the sensor start-of-frame. The constant part of that latency is a rig.yaml ``time.offset_prior_ns`` (P2 §4.3)
refined by Kalibr; its jitter is visible in ``frames_meta.csv`` (``t_arrival_ns - t_ns``). The upgrade path is a small
C++ libargus helper emitting ``(frame_no, ICaptureMetadata::getSensorTimestamp())`` (CLOCK_MONOTONIC_RAW, converted
with ``clock.RawToMono``) behind the same ``CameraBackend`` interface.
"""

from __future__ import annotations

import collections
import warnings
from fractions import Fraction
from typing import Any

import numpy as np

# Keys of P2's cameras.<c>.argus mapping that this backend understands.
ARGUS_KEYS = {
    "exposure_time_range_ns",   # [lo, hi] ns -> exposuretimerange="lo hi"
    "gain_range",               # [lo, hi]    -> gainrange="lo hi"
    "isp_digital_gain_range",   # [lo, hi]    -> ispdigitalgainrange="lo hi"
    "aelock", "awblock",        # bool
    "wbmode",                   # int (0 off, 1 auto, ...)
    "exposure_compensation",    # float EV -> exposurecompensation
    "tnr_mode", "ee_mode",      # int: temporal noise reduction / edge enhancement (0 = off)
    "saturation",               # float
    # P1-style fixed values (converted to a [v, v] range)
    "exposure_us", "gain",
}


def _fmt_num(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def _range_prop(name: str, rng) -> str:
    lo, hi = rng
    return f'{name}="{_fmt_num(lo)} {_fmt_num(hi)}"'


def _framerate(fps: float) -> str:
    fr = Fraction(float(fps)).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"


def build_argus_pipeline(sensor_id: int, sensor_mode: int | None, width: int, height: int, fps: float,
                         exposure_time_range_ns=None, gain_range=None, isp_digital_gain_range=(1, 1),
                         wbmode: int | None = 1, aelock: bool = False, awblock: bool = False,
                         exposure_compensation: float | None = None, tnr_mode: int | None = None,
                         ee_mode: int | None = None, saturation: float | None = None,
                         max_buffers: int = 8, sink_name: str = "sink") -> str:
    """GStreamer launch string: Argus ISP (NV12 in NVMM) -> nvvidconv -> BGRx in system memory -> appsink.

    ``sensor_mode=None`` lets Argus pick the mode from width/height/fps (not recommended: request the exact mode).
    The appsink keeps up to ``max_buffers`` buffers and does not drop (drops are counted in Python instead).
    """
    props = [f"sensor-id={int(sensor_id)}"]
    if sensor_mode is not None and int(sensor_mode) >= 0:
        props.append(f"sensor-mode={int(sensor_mode)}")
    if exposure_time_range_ns is not None:
        props.append(_range_prop("exposuretimerange", exposure_time_range_ns))
    if gain_range is not None:
        props.append(_range_prop("gainrange", gain_range))
    if isp_digital_gain_range is not None:
        props.append(_range_prop("ispdigitalgainrange", isp_digital_gain_range))
    if exposure_compensation is not None:
        props.append(f"exposurecompensation={_fmt_num(exposure_compensation)}")
    if tnr_mode is not None:
        props.append(f"tnr-mode={int(tnr_mode)}")
    if ee_mode is not None:
        props.append(f"ee-mode={int(ee_mode)}")
    if saturation is not None:
        props.append(f"saturation={_fmt_num(saturation)}")
    props.append(f"aelock={'true' if aelock else 'false'}")
    props.append(f"awblock={'true' if awblock else 'false'}")
    if wbmode is not None:
        props.append(f"wbmode={int(wbmode)}")
    src = "nvarguscamerasrc " + " ".join(props)
    caps = f"video/x-raw(memory:NVMM),width={int(width)},height={int(height)},framerate={_framerate(fps)},format=NV12"
    sink = (f"appsink name={sink_name} emit-signals=false max-buffers={int(max_buffers)} drop=false sync=false")
    return f"{src} ! {caps} ! nvvidconv ! video/x-raw,format=BGRx ! {sink}"


def argus_params_from_dict(argus: dict | None) -> dict[str, Any]:
    """Translate a P2 ``argus`` mapping into ``build_argus_pipeline`` keyword arguments (unknown keys warn)."""
    argus = dict(argus or {})
    out: dict[str, Any] = {}
    unknown = sorted(set(argus) - ARGUS_KEYS)
    if unknown:
        warnings.warn(f"argus settings not understood by ArgusCamera (ignored): {unknown}", stacklevel=2)
    if "exposure_time_range_ns" in argus:
        out["exposure_time_range_ns"] = tuple(argus["exposure_time_range_ns"])
    elif argus.get("exposure_us") is not None:
        e = int(round(float(argus["exposure_us"]) * 1000))
        out["exposure_time_range_ns"] = (e, e)
    if "gain_range" in argus:
        out["gain_range"] = tuple(argus["gain_range"])
    elif argus.get("gain") is not None:
        out["gain_range"] = (float(argus["gain"]), float(argus["gain"]))
    if "isp_digital_gain_range" in argus:
        out["isp_digital_gain_range"] = tuple(argus["isp_digital_gain_range"])
    for k in ("aelock", "awblock"):
        if k in argus:
            out[k] = bool(argus[k])
    for k in ("wbmode", "tnr_mode", "ee_mode"):
        if argus.get(k) is not None:
            out[k] = int(argus[k])
    for k in ("exposure_compensation", "saturation"):
        if argus.get(k) is not None:
            out[k] = float(argus[k])
    return out


def argus_params_from_camera(cam) -> dict[str, Any]:
    """``cam``: P2 ``PhysicalCamera`` (attributes sensor_id, sensor_mode, resolution, fps, argus)."""
    w, h = (int(v) for v in cam.resolution)
    params = {"sensor_id": int(cam.sensor_id), "sensor_mode": cam.sensor_mode, "width": w, "height": h,
              "fps": float(cam.fps)}
    params.update(argus_params_from_dict(getattr(cam, "argus", None)))
    return params


def pts_to_monotonic(base_time_ns: int, pts_ns: int) -> int:
    """CLOCK_MONOTONIC capture estimate of a buffer: ``base_time + pts`` (pipeline clock forced to MONOTONIC)."""
    return int(base_time_ns) + int(pts_ns)


class LatencyTracker:
    """Running min / median of ``t_arrival - t_ns`` over the last ``window`` frames.

    ``drifted`` becomes True when the median moves more than ``drift_ns`` away from the median of the first
    ``baseline_n`` frames (a sign the stamp/clock mapping changed, e.g. pipeline clock not MONOTONIC).
    """

    def __init__(self, window: int = 300, baseline_n: int = 60, drift_ns: int = 5_000_000):
        self._buf: collections.deque[int] = collections.deque(maxlen=window)
        self.baseline_n = baseline_n
        self.drift_ns = drift_ns
        self.baseline_ns: float | None = None
        self.min_ns: int | None = None
        self.n = 0
        self.negative = 0       # arrival before capture: impossible -> stamp domain is wrong

    def update(self, latency_ns: int) -> None:
        latency_ns = int(latency_ns)
        self._buf.append(latency_ns)
        self.n += 1
        if latency_ns < 0:
            self.negative += 1
        self.min_ns = latency_ns if self.min_ns is None else min(self.min_ns, latency_ns)
        if self.baseline_ns is None and self.n >= self.baseline_n:
            self.baseline_ns = float(np.median(self._buf))

    @property
    def median_ns(self) -> float | None:
        return float(np.median(self._buf)) if self._buf else None

    @property
    def drifted(self) -> bool:
        if self.baseline_ns is None or not self._buf:
            return False
        return abs(self.median_ns - self.baseline_ns) > self.drift_ns

    def summary(self) -> dict:
        med = self.median_ns
        return {"latency_ms_median": None if med is None else med / 1e6,
                "latency_ms_min": None if self.min_ns is None else self.min_ns / 1e6,
                "latency_drifted": self.drifted, "latency_negative": self.negative}


def detect_sensor_gaps(t_ns: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """``(index, missing_count)`` for every frame ``index`` whose interval to the previous frame exceeds 1.5 / fps.

    ``missing_count = max(1, round(dt * fps) - 1)``.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    if len(t) < 2:
        return []
    dt = np.diff(t)
    period = 1e9 / float(fps)
    idx = np.nonzero(dt > 1.5 * period)[0]
    return [(int(i + 1), max(1, int(round(dt[i] / period)) - 1)) for i in idx]


class GapDetector:
    """Streaming version used by the recorder: ``update(seq, t_ns) -> missing frames before this one``.

    Uses the larger of the backend seq gap and the time gap (seq gaps catch drops inside the backend's pipeline;
    time gaps catch drops the backend didn't count, e.g. in the sensor/driver).
    """

    def __init__(self, fps: float):
        self.period = 1e9 / float(fps)
        self.last_seq: int | None = None
        self.last_t: int | None = None
        self.total_missing = 0

    def update(self, seq: int | None, t_ns: int) -> int:
        missing = 0
        if self.last_t is not None:
            dt = int(t_ns) - self.last_t
            if dt > 1.5 * self.period:
                missing = max(1, int(round(dt / self.period)) - 1)
            if seq is not None and self.last_seq is not None:
                missing = max(missing, int(seq) - self.last_seq - 1)
        self.last_t = int(t_ns)
        self.last_seq = None if seq is None else int(seq)
        self.total_missing += missing
        return missing
