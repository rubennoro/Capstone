"""Pure logic for the Raspberry Pi (libcamera / Picamera2) camera backend. Unit-tested on a laptop; no picamera2 or
libcamera imports here. Sources for every claim below: ``sensor_rig/docs/raspberry_pi.md``.

* ``picamera2_params_from_dict`` / ``picamera2_params_from_camera``: rig.yaml ``cameras.<c>`` (``sensor_id``,
  ``resolution``, ``fps``, ``sensor`` and the optional ``picamera2:`` settings dict) -> validated ``Picamera2Params``.
* ``SensorMode``, ``KNOWN_SENSOR_MODES``, ``modes_from_picamera2``, ``choose_sensor_mode``: pick the raw sensor mode
  and refuse anything that would crop the field of view (cropped sensor mode, or an output aspect ratio different from
  the mode's, which makes the ISP crop) or cannot reach the fps.
* ``build_configuration`` / ``build_controls``: keyword arguments for ``Picamera2.create_video_configuration`` and the
  control dict (fixed frame duration, locked exposure / gain / white balance, manual focus, full-FOV ScalerCrop).
* ``select_camera_num``: map rig.yaml to a Picamera2 camera index, checked against ``global_camera_info()``.
* ``SensorTimestampConverter``: request metadata ``SensorTimestamp`` -> CLOCK_MONOTONIC ns.

Timestamps: on a Pi 5, ``SensorTimestamp`` is the CSI-2 receiver's (rp1-cfe) V4L2 buffer timestamp, taken with
``ktime_get_ns()`` (CLOCK_MONOTONIC) in the frame-start interrupt, i.e. when the first row is read out of the sensor
(rolling shutter: the END of the first row's exposure). libcamera documents the control as CLOCK_BOOTTIME; the two
clocks differ only by time spent in suspend, which a rig never does. Default: ``t_ns = SensorTimestamp``.
``timestamp_clock: boottime`` subtracts the BOOTTIME - MONOTONIC offset in case a future kernel switches clocks.
The constant start-of-frame -> mid-exposure offset (``+readout/2 - exposure/2``, positive) belongs in rig.yaml
``cameras.<c>.time.offset_prior_ns``, never in the backend.

Pixels: Picamera2 ``"RGB888"`` arrays are laid out ``[B, G, R]`` per pixel, which is OpenCV's BGR, so frames need no
channel swap (``"BGR888"`` would be ``[R, G, B]``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

# Picamera2 "RGB888" == bytes [B, G, R] per pixel == OpenCV BGR (Picamera2 manual, "Image formats" appendix).
PIXEL_FORMAT = "RGB888"

# Keys of rig.yaml cameras.<c>.picamera2 that this backend understands.
PICAMERA2_KEYS = {
    "camera_num",        # int: Picamera2(camera_num); default: cameras.<c>.sensor_id
    "model",             # str: expected global_camera_info() "Model" prefix, e.g. imx708_wide (checked at start)
    "sensor_size",       # [w, h]: raw sensor mode size (Picamera2 sensor={"output_size": ...}); default: auto, full FOV
    "bit_depth",         # int: raw sensor mode bit depth (default: from the chosen mode)
    "exposure_us",       # int: manual exposure (needs analogue_gain); omitted -> auto exposure
    "analogue_gain",     # float >= 1: manual sensor gain (needs exposure_us)
    "colour_gains",      # [red, blue]: manual white balance; omitted -> auto white balance
    "af",                # "manual" (autofocus lens, needs lens_position) | "none" (fixed-focus lens)
    "lens_position",     # float dioptres (1 / focus distance in m; 0 = infinity); implies af: manual
    "buffer_count",      # int: libcamera buffer sets (default 6, like Picamera2's video configuration)
    "hflip", "vflip",    # bool: ISP transform (applied before calibration, so calibrate with the same flips)
    "noise_reduction",   # "off" | "fast" | "high_quality" | "minimal" (default: libcamera's default)
    "timestamp_clock",   # "monotonic" (default, Pi 5 rp1-cfe) | "boottime"
}

AF_MODES = ("manual", "none")
NOISE_REDUCTION_MODES = {"off": 0, "fast": 1, "high_quality": 2, "minimal": 3}   # libcamera NoiseReductionMode
TIMESTAMP_CLOCKS = ("monotonic", "boottime")
AF_MODE_MANUAL = 0          # libcamera controls::AfModeManual
DEFAULT_BUFFER_COUNT = 6
ASPECT_TOL = 0.005          # relative aspect-ratio difference that still counts as "the same" (rounding of sizes)


class Picamera2ConfigError(ValueError):
    """Invalid picamera2 settings or a mode choice that would crop the field of view / miss the frame rate."""


# --------------------------------------------------------------------------------------------------- sensor modes
@dataclass(frozen=True)
class SensorMode:
    size: tuple[int, int]                 # raw output size of the sensor (w, h)
    bit_depth: int
    crop: tuple[int, int, int, int]       # field of view inside the pixel array (x, y, w, h), native pixels
    max_fps: float
    format: str | None = None

    def full_fov(self, pixel_array: tuple[int, int]) -> bool:
        """True if the mode reads the whole pixel array (binned/scaled, not cropped)."""
        x, y, w, h = self.crop
        return x == 0 and y == 0 and (w, h) == tuple(pixel_array)

    @property
    def aspect(self) -> float:
        return self.size[0] / self.size[1]


def _mode(w, h, crop, fps, bits=10, fmt=None) -> SensorMode:
    return SensorMode((w, h), bits, crop, fps, fmt)


# Offline copies of the kernel mode tables (raspberrypi/linux rpi-6.6.y drivers/media/i2c/imx708.c, imx219.c) and the
# max frame rates libcamera reports for them. Used for validation without hardware; at runtime the camera's own
# ``Picamera2.sensor_modes`` are authoritative. Crops are relative to the active pixel array.
KNOWN_SENSOR_MODES: dict[str, tuple[tuple[int, int], tuple[SensorMode, ...]]] = {
    # Camera Module 3 / 3 Wide / NoIR (Model "imx708", "imx708_wide", "imx708_noir", "imx708_wide_noir")
    "imx708": ((4608, 2592), (
        _mode(1536, 864, (768, 432, 3072, 1728), 120.13),     # 2x2 binned AND cropped: NOT full FOV
        _mode(2304, 1296, (0, 0, 4608, 2592), 56.03),          # 2x2 binned, full FOV
        _mode(4608, 2592, (0, 0, 4608, 2592), 14.35),          # full resolution
    )),
    # Camera Module 2 and IMX219 M12 fisheye boards (e.g. 160 deg lenses)
    "imx219": ((3280, 2464), (
        _mode(640, 480, (1000, 752, 1280, 960), 206.65),       # binned + cropped
        _mode(1640, 1232, (0, 0, 3280, 2464), 41.85),          # 2x2 binned, full FOV
        _mode(1920, 1080, (680, 692, 1920, 1080), 47.57),      # cropped
        _mode(3280, 2464, (0, 0, 3280, 2464), 21.19),          # full resolution
    )),
    # HQ Camera (C/CS mount and M12 variants); crops/fps as Picamera2 reports them (Picamera2 manual, sensor_modes)
    "imx477": ((4056, 3040), (
        _mode(1332, 990, (696, 528, 2664, 1980), 120.05),      # binned + cropped
        _mode(2028, 1080, (0, 440, 4056, 2160), 50.03, 12),    # binned, cropped to 16:9
        _mode(2028, 1520, (0, 0, 4056, 3040), 40.01, 12),      # 2x2 binned, full FOV
        _mode(4056, 3040, (0, 0, 4056, 3040), 10.0, 12),       # full resolution
    )),
    # Global Shutter Camera (single full-array mode)
    "imx296": ((1456, 1088), (
        _mode(1456, 1088, (0, 0, 1456, 1088), 60.38),
    )),
}


def known_modes(sensor: str | None) -> tuple[tuple[int, int], tuple[SensorMode, ...]] | None:
    """Offline mode table for a rig.yaml ``sensor`` / libcamera Model name (prefix match: imx708_wide -> imx708)."""
    s = (sensor or "").lower()
    for key in sorted(KNOWN_SENSOR_MODES, key=len, reverse=True):
        if s.startswith(key):
            return KNOWN_SENSOR_MODES[key]
    return None


def modes_from_picamera2(sensor_modes: Iterable[dict]) -> list[SensorMode]:
    """``Picamera2.sensor_modes`` (dicts with size, bit_depth, crop_limits, fps, format) -> ``SensorMode`` list."""
    out = []
    for m in sensor_modes:
        if "size" not in m or "crop_limits" not in m:
            continue                       # non-raw sensors report only a format
        w, h = (int(v) for v in m["size"])
        crop = tuple(int(v) for v in m["crop_limits"])
        out.append(SensorMode((w, h), int(m.get("bit_depth", 0)), crop, float(m.get("fps", 0.0)),
                              None if m.get("format") is None else str(m["format"])))
    return out


def _same_aspect(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs((a[0] / a[1]) / (b[0] / b[1]) - 1.0) <= ASPECT_TOL


def choose_sensor_mode(modes: Iterable[SensorMode], pixel_array: tuple[int, int], output_size: tuple[int, int],
                       fps: float, sensor_size: tuple[int, int] | None = None,
                       bit_depth: int | None = None) -> SensorMode:
    """The raw sensor mode for a full-FOV ``output_size`` stream at ``fps``.

    With ``sensor_size`` (and optionally ``bit_depth``) that exact mode is required; otherwise the smallest full-FOV
    mode with the output's aspect ratio that is at least as large as the output and reaches ``fps`` (ties: higher
    bit depth). Raises ``Picamera2ConfigError`` if the mode is cropped, the output's aspect ratio differs from the
    mode's (the ISP would crop to match), the output is larger than the mode (upscaling), or ``fps`` is too high.
    """
    modes = list(modes)
    pixel_array = (int(pixel_array[0]), int(pixel_array[1]))
    ow, oh = (int(v) for v in output_size)
    if not modes:
        raise Picamera2ConfigError("the camera reports no raw sensor modes")

    def problems(m: SensorMode) -> list[str]:
        p = []
        if not m.full_fov(pixel_array):
            p.append(f"mode {m.size[0]}x{m.size[1]} crops the sensor to {m.crop} of {pixel_array[0]}x{pixel_array[1]}")
        if not _same_aspect(m.size, (ow, oh)):
            p.append(f"output {ow}x{oh} (aspect {ow / oh:.4f}) differs from mode {m.size[0]}x{m.size[1]} "
                     f"(aspect {m.aspect:.4f}): the ISP would crop the field of view")
        if ow > m.size[0] or oh > m.size[1]:
            p.append(f"output {ow}x{oh} is larger than mode {m.size[0]}x{m.size[1]} (upscaling)")
        if m.max_fps > 0 and fps > m.max_fps * (1 + 1e-3):
            p.append(f"{fps:g} fps exceeds mode {m.size[0]}x{m.size[1]} maximum {m.max_fps:g} fps")
        return p

    if sensor_size is not None:
        want = (int(sensor_size[0]), int(sensor_size[1]))
        cands = [m for m in modes if m.size == want and (bit_depth is None or m.bit_depth == int(bit_depth))]
        if not cands:
            avail = ", ".join(f"{m.size[0]}x{m.size[1]}/{m.bit_depth}bit" for m in modes)
            raise Picamera2ConfigError(f"no sensor mode {want[0]}x{want[1]}"
                                       f"{'' if bit_depth is None else f' {bit_depth}-bit'} (available: {avail})")
        cands.sort(key=lambda m: -m.bit_depth)
        p = problems(cands[0])
        if p:
            raise Picamera2ConfigError("; ".join(p))
        return cands[0]
    ok = [m for m in modes if not problems(m) and (bit_depth is None or m.bit_depth == int(bit_depth))]
    if not ok:
        detail = " | ".join(f"{m.size[0]}x{m.size[1]}: {'; '.join(problems(m)) or 'bit depth'}" for m in modes)
        raise Picamera2ConfigError(f"no full-FOV sensor mode delivers {ow}x{oh} at {fps:g} fps ({detail})")
    ok.sort(key=lambda m: (m.size[0] * m.size[1], -m.bit_depth))
    return ok[0]


# --------------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Picamera2Params:
    camera_num: int
    width: int
    height: int
    fps: float
    model: str | None = None
    sensor: str | None = None                       # rig.yaml cameras.<c>.sensor (for the offline mode table)
    sensor_size: tuple[int, int] | None = None
    bit_depth: int | None = None
    exposure_us: int | None = None
    analogue_gain: float | None = None
    colour_gains: tuple[float, float] | None = None
    af: str = "none"
    lens_position: float | None = None
    buffer_count: int = DEFAULT_BUFFER_COUNT
    hflip: bool = False
    vflip: bool = False
    noise_reduction: str | None = None
    timestamp_clock: str = "monotonic"
    unknown_keys: tuple[str, ...] = field(default=(), compare=False)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def frame_duration_us(self) -> int:
        return frame_duration_us(self.fps)


def frame_duration_us(fps: float) -> int:
    """Frame duration in integer microseconds (libcamera FrameDurationLimits unit)."""
    return int(round(1e6 / float(fps)))


def _int(v, key, lo=None, hi=None) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise Picamera2ConfigError(f"{key}: expected an integer, got {v!r}")
    v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise Picamera2ConfigError(f"{key}: must be in [{lo}, {hi}], got {v}")
    return v


def _num(v, key, lo=None, hi=None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise Picamera2ConfigError(f"{key}: expected a finite number, got {v!r}")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise Picamera2ConfigError(f"{key}: must be in [{lo}, {hi}], got {v}")
    return v


def _bool(v, key) -> bool:
    if not isinstance(v, bool):
        raise Picamera2ConfigError(f"{key}: expected true/false, got {v!r}")
    return v


def _pair(v, key, conv) -> tuple:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise Picamera2ConfigError(f"{key}: expected a list of 2 values, got {v!r}")
    return tuple(conv(x, f"{key}[{i}]") for i, x in enumerate(v))


def _choice(v, key, choices) -> str:
    if not isinstance(v, str) or v not in choices:
        raise Picamera2ConfigError(f"{key}: must be one of {list(choices)}, got {v!r}")
    return v


def picamera2_params_from_dict(settings: dict | None, *, sensor_id: int, resolution, fps: float,
                               sensor: str | None = None, warn: bool = True) -> Picamera2Params:
    """Validate a rig.yaml ``picamera2`` mapping. Unknown keys are kept in ``unknown_keys`` (and warned about when
    ``warn``); bad values raise ``Picamera2ConfigError`` whose message starts with the key."""
    s = dict(settings or {})
    unknown = tuple(sorted(str(k) for k in set(s) - PICAMERA2_KEYS))
    if unknown and warn:
        warnings.warn(f"picamera2 settings not understood by Picamera2Camera (ignored): {list(unknown)}",
                      stacklevel=2)
    w, h = (int(v) for v in resolution)
    fps = float(fps)
    if not (fps > 0 and math.isfinite(fps)):
        raise Picamera2ConfigError(f"fps: must be > 0, got {fps}")
    duration = frame_duration_us(fps)

    exposure = None if s.get("exposure_us") is None else _int(s["exposure_us"], "exposure_us", lo=1)
    gain = None if s.get("analogue_gain") is None else _num(s["analogue_gain"], "analogue_gain", lo=1.0)
    if (exposure is None) != (gain is None):
        raise Picamera2ConfigError("exposure_us / analogue_gain: give both (manual exposure) or neither (auto "
                                   "exposure); with only one, libcamera's AE would still move the other")
    if exposure is not None and exposure > duration:
        raise Picamera2ConfigError(f"exposure_us: {exposure} us is longer than the frame duration {duration} us "
                                   f"at {fps:g} fps")
    colour = None if s.get("colour_gains") is None else _pair(s["colour_gains"], "colour_gains",
                                                               lambda x, k: _num(x, k, lo=0.0, hi=32.0))
    lens = None if s.get("lens_position") is None else _num(s["lens_position"], "lens_position", lo=0.0, hi=32.0)
    af = _choice(s.get("af", "manual" if lens is not None else "none"), "af", AF_MODES)
    if af == "manual" and lens is None:
        raise Picamera2ConfigError("lens_position: af: manual needs a lens position in dioptres (1 / distance in m; "
                                   "0 = infinity), otherwise the lens sits wherever libcamera left it")
    if af == "none" and lens is not None:
        raise Picamera2ConfigError("af: 'none' (fixed-focus lens) contradicts lens_position")
    sensor_size = None if s.get("sensor_size") is None else _pair(s["sensor_size"], "sensor_size",
                                                                   lambda x, k: _int(x, k, lo=1))
    bit_depth = None if s.get("bit_depth") is None else _int(s["bit_depth"], "bit_depth", lo=8, hi=16)
    nr = None if s.get("noise_reduction") is None else _choice(s["noise_reduction"], "noise_reduction",
                                                                NOISE_REDUCTION_MODES)
    model = s.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise Picamera2ConfigError(f"model: expected a non-empty string, got {model!r}")
    return Picamera2Params(
        camera_num=_int(s.get("camera_num", sensor_id), "camera_num", lo=0),
        width=w, height=h, fps=fps, model=model, sensor=sensor, sensor_size=sensor_size, bit_depth=bit_depth,
        exposure_us=exposure, analogue_gain=gain, colour_gains=colour, af=af, lens_position=lens,
        buffer_count=_int(s.get("buffer_count", DEFAULT_BUFFER_COUNT), "buffer_count", lo=2, hi=32),
        hflip=_bool(s.get("hflip", False), "hflip"), vflip=_bool(s.get("vflip", False), "vflip"),
        noise_reduction=nr, timestamp_clock=_choice(s.get("timestamp_clock", "monotonic"), "timestamp_clock",
                                                    TIMESTAMP_CLOCKS),
        unknown_keys=unknown,
    )


def picamera2_params_from_camera(cam, warn: bool = True) -> Picamera2Params:
    """``cam``: rig ``PhysicalCamera`` (sensor_id, resolution, fps, sensor, optional ``picamera2`` dict)."""
    return picamera2_params_from_dict(getattr(cam, "picamera2", None), sensor_id=int(cam.sensor_id),
                                      resolution=cam.resolution, fps=float(cam.fps),
                                      sensor=getattr(cam, "sensor", None), warn=warn)


def offline_mode_check(params: Picamera2Params) -> SensorMode | None:
    """Validate the mode choice against ``KNOWN_SENSOR_MODES`` (None if the sensor is not in the table)."""
    known = known_modes(params.model or params.sensor)
    if known is None:
        return None
    pixel_array, modes = known
    return choose_sensor_mode(modes, pixel_array, params.size, params.fps, params.sensor_size, params.bit_depth)


# --------------------------------------------------------------------------------------------------- configuration
def build_controls(params: Picamera2Params, scaler_crop: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """libcamera controls for a fixed frame rate, locked exposure/gain/white balance (when configured), manual focus
    and (when given) the full-FOV ScalerCrop."""
    d = params.frame_duration_us
    c: dict[str, Any] = {"FrameDurationLimits": (d, d)}
    if params.exposure_us is not None:
        c.update({"AeEnable": False, "ExposureTime": int(params.exposure_us),
                  "AnalogueGain": float(params.analogue_gain)})
    else:
        c["AeEnable"] = True
    if params.colour_gains is not None:
        c.update({"AwbEnable": False, "ColourGains": tuple(float(g) for g in params.colour_gains)})
    else:
        c["AwbEnable"] = True
    if params.af == "manual":
        c.update({"AfMode": AF_MODE_MANUAL, "LensPosition": float(params.lens_position)})
    if params.noise_reduction is not None:
        c["NoiseReductionMode"] = NOISE_REDUCTION_MODES[params.noise_reduction]
    if scaler_crop is not None:
        c["ScalerCrop"] = tuple(int(v) for v in scaler_crop)
    return c


def build_configuration(params: Picamera2Params, mode: SensorMode) -> dict[str, Any]:
    """Keyword arguments for ``Picamera2.create_video_configuration`` (the caller turns ``transform`` into a
    ``libcamera.Transform``). No ``align_configuration``: the main stream must be exactly the calibration size."""
    return {
        "main": {"size": params.size, "format": PIXEL_FORMAT},
        "sensor": {"output_size": tuple(mode.size), "bit_depth": int(mode.bit_depth)},
        "buffer_count": int(params.buffer_count),
        "queue": False,                       # do not hold back a frame for capture_* calls
        "controls": build_controls(params),
        "transform": {"hflip": bool(params.hflip), "vflip": bool(params.vflip)},
    }


def check_configured(camera_config: dict, params: Picamera2Params, mode: SensorMode) -> list[str]:
    """Problems with what libcamera actually configured (it may adjust sizes/formats silently)."""
    out = []
    main = camera_config.get("main") or {}
    size = tuple(int(v) for v in main.get("size", ()))
    if size != params.size:
        out.append(f"main stream is {size}, expected the calibration resolution {params.size}")
    if main.get("format") != PIXEL_FORMAT:
        out.append(f"main stream format is {main.get('format')!r}, expected {PIXEL_FORMAT!r} (BGR byte order)")
    sensor = camera_config.get("sensor") or {}
    if sensor.get("output_size") is not None and tuple(int(v) for v in sensor["output_size"]) != tuple(mode.size):
        out.append(f"sensor mode is {tuple(sensor['output_size'])}, expected {tuple(mode.size)}")
    return out


def unsupported_controls(controls: dict, camera_controls: dict) -> list[str]:
    """Controls we set that this camera does not offer (e.g. AfMode on a fixed-focus IMX219)."""
    return sorted(k for k in controls if k not in camera_controls)


def select_camera_num(camera_info: list[dict], camera_num: int, model: str | None = None) -> int:
    """Check ``camera_num`` against ``Picamera2.global_camera_info()``. With ``model``, the camera's "Model" must
    start with it; if ``camera_num`` points at another model but exactly one camera matches, that one is used
    (camera numbering can change with what is plugged in)."""
    if not camera_info:
        raise Picamera2ConfigError("no cameras detected by libcamera (check the ribbon cables and "
                                   "`rpicam-hello --list-cameras`)")
    models = [str(c.get("Model", "")) for c in camera_info]
    if model is None:
        if camera_num >= len(camera_info):
            raise Picamera2ConfigError(f"camera_num {camera_num} not present (detected: {models})")
        return camera_num
    m = model.lower()
    if camera_num < len(camera_info) and models[camera_num].lower().startswith(m):
        return camera_num
    matches = [i for i, name in enumerate(models) if name.lower().startswith(m)]
    if len(matches) == 1:
        return matches[0]
    raise Picamera2ConfigError(f"camera_num {camera_num} / model {model!r}: no unique match among the detected "
                               f"cameras {models}")


# --------------------------------------------------------------------------------------------------- timestamps
class SensorTimestampConverter:
    """``SensorTimestamp`` (ns) -> CLOCK_MONOTONIC ns.

    ``clock="monotonic"``: identity (Pi 5 rp1-cfe stamps with ``ktime_get_ns()``). ``clock="boottime"``: subtract
    ``boot_minus_mono_ns`` (CLOCK_BOOTTIME - CLOCK_MONOTONIC, sampled by the backend; both run at the same rate and
    differ only by suspended time).
    """

    def __init__(self, clock: str = "monotonic", boot_minus_mono_ns: int = 0):
        if clock not in TIMESTAMP_CLOCKS:
            raise ValueError(f"timestamp clock must be one of {TIMESTAMP_CLOCKS}, got {clock!r}")
        self.clock = clock
        self.offset_ns = int(boot_minus_mono_ns) if clock == "boottime" else 0

    def __call__(self, sensor_ts_ns) -> int:
        return int(sensor_ts_ns) - self.offset_ns


def missing_frames(prev_t_ns: int | None, t_ns: int, fps: float) -> int:
    """Frames lost between two consecutive stamps (dt > 1.5 periods -> round(dt / period) - 1, at least 1)."""
    if prev_t_ns is None:
        return 0
    period = 1e9 / float(fps)
    dt = int(t_ns) - int(prev_t_ns)
    if dt <= 1.5 * period:
        return 0
    return max(1, int(round(dt / period)) - 1)
