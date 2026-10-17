"""Picamera2Camera: CSI camera on a Raspberry Pi 5 via libcamera / Picamera2 (Pi only).

``picamera2`` and ``libcamera`` are imported lazily in ``_open`` so this module imports on any machine. Configuration,
mode choice, controls and timestamp conversion are in ``picamera2_logic`` (unit-tested); the backend itself is tested
with a fake ``picamera2`` module (``tests/fake_picamera2.py``).

Data flow:
1. ``_open``: pick the camera (``global_camera_info``), choose a full-FOV sensor mode from the camera's own
   ``sensor_modes`` (refuses cropped modes / wrong aspect / too-high fps), configure a ``"RGB888"`` main stream at
   exactly the calibration resolution, lock frame duration / exposure / gain / white balance / focus, set ScalerCrop
   to the full mode, install ``post_callback`` and start.
2. ``post_callback`` (Picamera2's camera thread, once per completed request): stamp ``t_arrival``, copy the main
   array and the metadata into a bounded hand-off deque. It does nothing else, so libcamera never waits on Python.
3. Reader thread (``_run``): converts ``SensorTimestamp`` to CLOCK_MONOTONIC, numbers frames (``seq`` jumps over
   frames missing from the timestamp sequence, so the recorder's gap logic sees them), tracks latency and pushes
   ``Frame`` objects into the ``CameraBackend`` queue / ring.

Frames are HxWx3 uint8 BGR ("RGB888" is [B, G, R] in memory), stamped at the sensor's start of frame (first row read
out). The start-of-frame -> mid-exposure offset is a rig.yaml ``time.offset_prior_ns``, never applied here.

NOT RUN ON HARDWARE YET. See ``sensor_rig/record/HARDWARE_CHECKLIST.md`` (Raspberry Pi section).
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from sensor_rig.clock import boot_minus_mono_ns
from sensor_rig.drivers.argus_logic import LatencyTracker
from sensor_rig.drivers.base import BackendError, CameraBackend, Frame
from sensor_rig.drivers.picamera2_logic import (
    Picamera2ConfigError,
    Picamera2Params,
    SensorTimestampConverter,
    build_configuration,
    build_controls,
    check_configured,
    choose_sensor_mode,
    missing_frames,
    modes_from_picamera2,
    offline_mode_check,
    picamera2_params_from_camera,
    select_camera_num,
    unsupported_controls,
)


class Picamera2Camera(CameraBackend):
    def __init__(self, name: str, params: Picamera2Params, clock=None, queue_size: int = 8, ring_size: int = 16,
                 handoff_size: int | None = None, start_timeout_s: float = 5.0, frame_timeout_s: float = 2.0):
        super().__init__(name, params.width, params.height, params.fps, queue_size=queue_size, ring_size=ring_size,
                         clock=clock)
        self.params = params
        self.start_timeout_s = float(start_timeout_s)
        self.frame_timeout_s = float(frame_timeout_s)
        self.latency = LatencyTracker()
        self._handoff: collections.deque = collections.deque()
        self._handoff_max = int(handoff_size or max(4, params.buffer_count))
        self._handoff_cond = threading.Condition()
        self._picam2 = None
        self._converter = SensorTimestampConverter(params.timestamp_clock)
        self._description: dict = {}
        self.stats.update({"bad_stamps": 0, "sensor_gaps": 0, "sensor_missing": 0, "handoff_drops": 0,
                           "nonmonotonic_ts": 0})

    @classmethod
    def from_rig(cls, rig, name: str, **opts) -> "Picamera2Camera":
        """Build from ``rig.cameras[name]``; ``settings_overrides`` (dict) replaces individual ``picamera2`` keys.
        The mode choice is checked offline against the known sensor tables (fails fast on a cropping config)."""
        cam = rig.cameras[name]
        overrides = opts.pop("settings_overrides", None) or {}
        if overrides:
            import dataclasses

            cam = dataclasses.replace(cam, picamera2={**(getattr(cam, "picamera2", None) or {}), **overrides})
        try:
            params = picamera2_params_from_camera(cam)
            offline_mode_check(params)
        except Picamera2ConfigError as e:
            raise BackendError(f"camera {name}: cameras.{name}.picamera2: {e}") from None
        return cls(name, params, **opts)

    # ---- libcamera plumbing (hardware, or the fake module in tests) ----
    def _open(self) -> None:
        try:
            from picamera2 import Picamera2
        except Exception as e:  # pragma: no cover - needs a Pi
            raise BackendError("the picamera2 backend needs Picamera2 (Raspberry Pi OS: sudo apt install "
                               "python3-picamera2; use a venv with --system-site-packages)") from e
        p = self.params
        try:
            num = select_camera_num(Picamera2.global_camera_info(), p.camera_num, p.model)
        except Picamera2ConfigError as e:
            raise BackendError(f"camera {self.name}: {e}") from None
        picam2 = Picamera2(num)
        self._picam2 = picam2
        try:
            self._configure(picam2, num)
        except Exception:
            self._close()
            raise
        if p.timestamp_clock == "boottime":
            self._converter = SensorTimestampConverter("boottime", boot_minus_mono_ns())
        with self._handoff_cond:
            self._handoff.clear()
        picam2.post_callback = self._on_request
        picam2.start()

    def _configure(self, picam2, num: int) -> None:
        p = self.params
        props = dict(getattr(picam2, "camera_properties", {}) or {})
        pixel_array = _pixel_array(props)
        try:
            mode = choose_sensor_mode(modes_from_picamera2(picam2.sensor_modes), pixel_array, p.size, p.fps,
                                      p.sensor_size, p.bit_depth)
        except Picamera2ConfigError as e:
            raise BackendError(f"camera {self.name} ({props.get('Model', '?')}): {e}") from None
        kw = build_configuration(p, mode)
        flips = kw.pop("transform")
        if flips["hflip"] or flips["vflip"]:
            from libcamera import Transform

            kw["transform"] = Transform(hflip=int(flips["hflip"]), vflip=int(flips["vflip"]))
        config = picam2.create_video_configuration(**kw)
        picam2.configure(config)
        problems = check_configured(dict(picam2.camera_config), p, mode)
        if problems:
            raise BackendError(f"camera {self.name}: " + "; ".join(problems))
        camera_controls = dict(picam2.camera_controls)
        scaler_crop = None
        if "ScalerCrop" in camera_controls:
            scaler_crop = tuple(int(v) for v in camera_controls["ScalerCrop"][1])   # (min, max, default): max = full mode
        controls = build_controls(p, scaler_crop)
        missing = unsupported_controls(controls, camera_controls)
        if missing:
            raise BackendError(f"camera {self.name} ({props.get('Model', '?')}) does not support {missing}; "
                               "remove the corresponding picamera2 settings (e.g. lens_position on a fixed-focus "
                               "lens)")
        if "AfMode" in camera_controls and p.af != "manual":
            raise BackendError(f"camera {self.name} has an autofocus lens: set cameras.{self.name}.picamera2."
                               "lens_position (dioptres) so the focus is fixed (no hunting, calibration stays valid)")
        picam2.set_controls(controls)
        self._description = {
            "camera_num": num, "model": props.get("Model"), "sensor_mode": f"{mode.size[0]}x{mode.size[1]}",
            "sensor_bit_depth": mode.bit_depth, "sensor_crop": list(mode.crop), "pixel_array": list(pixel_array),
            "scaler_crop": None if scaler_crop is None else list(scaler_crop), "buffer_count": p.buffer_count,
            "exposure_us": p.exposure_us, "analogue_gain": p.analogue_gain,
            "colour_gains": None if p.colour_gains is None else list(p.colour_gains),
            "lens_position": p.lens_position, "hflip": p.hflip, "vflip": p.vflip,
            "timestamp_clock": p.timestamp_clock,
        }

    def _close(self) -> None:
        picam2, self._picam2 = self._picam2, None
        if picam2 is None:
            return
        try:
            picam2.post_callback = None
            picam2.stop()
        finally:
            picam2.close()

    def _on_request(self, request) -> None:
        """Picamera2 camera thread: copy out and return immediately (the request is released by Picamera2)."""
        if self._stopping.is_set():
            return
        t_arrival = self.clock.now_ns()
        md = dict(request.get_metadata())
        img = request.make_array("main")          # a copy; [B, G, R] per pixel for "RGB888"
        with self._handoff_cond:
            if len(self._handoff) >= self._handoff_max:
                self._handoff.popleft()
                self.stats["handoff_drops"] += 1
                self.stats["internal_drops"] += 1
            self._handoff.append((img, md, t_arrival))
            self._handoff_cond.notify_all()

    def _next(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        with self._handoff_cond:
            while not self._handoff:
                if self._stopping.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._handoff_cond.wait(min(remaining, 0.05))
            return self._handoff.popleft()

    def _run(self) -> None:
        seq = 0
        last_t: int | None = None
        got_first = False
        t_wait = time.monotonic()
        while not self._stopping.is_set():
            item = self._next(0.1)
            if item is None:
                waited = time.monotonic() - t_wait
                if not got_first and waited > self.start_timeout_s:
                    raise BackendError(f"camera {self.name}: no frame within {self.start_timeout_s} s")
                if got_first and waited > self.frame_timeout_s:
                    raise BackendError(f"camera {self.name}: no frame for {self.frame_timeout_s} s")
                continue
            got_first = True
            t_wait = time.monotonic()
            img, md, t_arrival = item
            ts = md.get("SensorTimestamp")
            if ts is None:
                self.stats["bad_stamps"] += 1
                continue
            t_ns = self._converter(ts)
            if last_t is not None and t_ns <= last_t:
                self.stats["nonmonotonic_ts"] += 1
                continue
            gap = missing_frames(last_t, t_ns, self.fps)
            if gap:
                self.stats["sensor_gaps"] += 1
                self.stats["sensor_missing"] += gap
                seq += gap
            last_t = t_ns
            lat = t_arrival - t_ns
            self.latency.update(lat)
            if lat < 0 or lat > 1_000_000_000:
                self.stats["bad_stamps"] += 1
            img = _as_bgr(img, self.width, self.height)
            meta = {"sensor_ts_raw_ns": int(ts)}
            if md.get("ExposureTime") is not None:
                meta["exposure_ns"] = int(md["ExposureTime"]) * 1000
            if md.get("AnalogueGain") is not None:
                meta["gain"] = float(md["AnalogueGain"])
            if md.get("LensPosition") is not None:
                meta["lens_position"] = float(md["LensPosition"])
            self._push(Frame(image=img, t_ns=t_ns, seq=seq, t_arrival_ns=t_arrival, meta=meta))
            seq += 1

    def info(self) -> dict:
        d = {"backend": "picamera2", "width": self.width, "height": self.height, "fps_nominal": self.fps}
        d.update(self._description)
        d.update(self.latency.summary())
        d.update({k: self.stats[k] for k in ("sensor_gaps", "sensor_missing", "handoff_drops", "nonmonotonic_ts",
                                             "bad_stamps")})
        return d


def _pixel_array(props: dict) -> tuple[int, int]:
    """Active pixel array size from ``camera_properties`` (crop coordinates are relative to it)."""
    areas = props.get("PixelArrayActiveAreas")
    if areas:
        a = areas[0]
        return (int(a[2]), int(a[3]))
    if props.get("PixelArraySize"):
        w, h = props["PixelArraySize"]
        return (int(w), int(h))
    raise BackendError("camera_properties has neither PixelArrayActiveAreas nor PixelArraySize")


def _as_bgr(img, width: int, height: int) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise BackendError(f"unexpected main stream array {a.shape} {a.dtype} (expected HxWx3 uint8 RGB888)")
    a = a[:height, :width]                 # stride padding, if any survived make_array
    if a.shape[:2] != (height, width):
        raise BackendError(f"main stream array is {a.shape[1]}x{a.shape[0]}, expected {width}x{height}")
    return np.ascontiguousarray(a)
