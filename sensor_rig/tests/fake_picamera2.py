"""A fake ``picamera2`` (+ ``libcamera.Transform``) for laptop tests of ``Picamera2Camera``.

It implements only the API surface the backend uses (``global_camera_info``, ``sensor_modes``, ``camera_properties``,
``camera_controls``, ``create_video_configuration``, ``configure``, ``camera_config``, ``set_controls``,
``post_callback``, ``start / stop / close``; requests with ``get_metadata`` / ``make_array``) and behaves like the
real thing where it matters to the backend:

* frames arrive from a separate "camera thread" through ``post_callback``;
* ``SensorTimestamp`` is the start-of-frame time on the injected clock (``SimClock`` / ``RealClock``) plus
  ``ts_offset_ns`` (e.g. to imitate CLOCK_BOOTTIME), delivered ``latency_ns`` later;
* "RGB888" arrays are ``[B, G, R]`` per pixel: the scene is a red square (R=230) on a blue-ish background;
* ``drop_frames`` removes frame indices before userspace (a sensor / libcamera drop: a gap in the timestamps);
* the frame period follows ``FrameDurationLimits`` (falls back to the mode's max fps).

Usage::

    state = install(monkeypatch, [FakeCameraSpec(model="fakecam", ...)], clock=SimClock(speed=4))
    cam = Picamera2Camera.from_rig(rig, "main", clock=state.clock)
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass, field

import numpy as np

RED_BGR = (40, 40, 230)          # [B, G, R] of the square
BACKGROUND_BGR = (200, 120, 20)


@dataclass
class FakeCameraSpec:
    model: str = "fakecam"
    pixel_array: tuple[int, int] = (1280, 960)
    # (w, h, crop (x, y, w, h), max_fps, bit_depth)
    modes: list = field(default_factory=lambda: [(640, 480, (0, 0, 1280, 960), 60.0, 10),
                                                 (1280, 960, (0, 0, 1280, 960), 30.0, 10)])
    autofocus: bool = False
    latency_ns: int = 8_000_000
    ts_offset_ns: int = 0
    exposure_us: int = 4000
    drop_frames: set = field(default_factory=set)
    stride_pad: int = 0               # extra columns in make_array (stride padding that survived)
    adjust_main_size: tuple[int, int] | None = None   # imitate libcamera silently changing the main stream size


class FakeState:
    def __init__(self, cameras: list[FakeCameraSpec], clock):
        self.cameras = cameras
        self.clock = clock
        self.instances: list = []


class Transform:
    def __init__(self, hflip=0, vflip=0, transpose=0):
        self.hflip, self.vflip, self.transpose = int(hflip), int(vflip), int(transpose)


class CompletedRequest:
    def __init__(self, arr: np.ndarray, md: dict):
        self._arr = arr
        self._md = md

    def get_metadata(self) -> dict:
        return dict(self._md)

    def make_array(self, name: str) -> np.ndarray:
        assert name == "main"
        return self._arr.copy()


def _make_module(state: FakeState):
    class Picamera2:
        def __init__(self, camera_num: int = 0, **_):
            if camera_num >= len(state.cameras):
                raise IndexError(f"camera {camera_num} not present")
            self.camera_num = camera_num
            self.spec = state.cameras[camera_num]
            w, h = self.spec.pixel_array
            self.camera_properties = {"Model": self.spec.model, "PixelArraySize": (w, h),
                                      "PixelArrayActiveAreas": [(0, 0, w, h)]}
            self.camera_config = None
            self.post_callback = None
            self.controls_set: dict = {}
            self.mode = None
            self.started = False
            self.closed = False
            self.frames_emitted = 0
            self._stop = threading.Event()
            self._thread = None
            state.instances.append(self)

        @staticmethod
        def global_camera_info():
            return [{"Model": c.model, "Location": 2, "Rotation": 0, "Num": i,
                     "Id": f"/base/axi/pcie@120000/rp1/i2c@{88000 if i == 0 else 80000}/{c.model}@1a"}
                    for i, c in enumerate(state.cameras)]

        @property
        def sensor_modes(self):
            return [{"format": f"SRGGB{b}_CSI2P", "unpacked": f"SRGGB{b}", "bit_depth": b, "size": (w, h),
                     "fps": fps, "crop_limits": crop, "exposure_limits": (26, 1000000)}
                    for (w, h, crop, fps, b) in self.spec.modes]

        @property
        def camera_controls(self):
            crop = self.mode[2] if self.mode is not None else (0, 0, *self.spec.pixel_array)
            c = {"FrameDurationLimits": (8000, 1000000, 33333), "ExposureTime": (26, 1000000, 20000),
                 "AnalogueGain": (1.0, 16.0, 1.0), "AeEnable": (False, True, True), "AwbEnable": (False, True, True),
                 "ColourGains": (0.0, 32.0, None), "NoiseReductionMode": (0, 4, 0),
                 "ScalerCrop": ((0, 0, 64, 64), crop, crop)}
            if self.spec.autofocus:
                c.update({"AfMode": (0, 2, 0), "LensPosition": (0.0, 15.0, 1.0)})
            return c

        def create_video_configuration(self, main=None, lores=None, raw=None, transform=None, colour_space=None,
                                       buffer_count=6, controls=None, display=None, encode=None, queue=True,
                                       sensor=None, **_):
            return {"use_case": "video", "main": dict(main or {}), "sensor": dict(sensor or {}),
                    "buffer_count": buffer_count, "queue": queue, "controls": dict(controls or {}),
                    "transform": transform}

        def configure(self, config):
            cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
            want = tuple(cfg["sensor"].get("output_size") or ())
            match = [m for m in self.spec.modes if (m[0], m[1]) == want]
            self.mode = match[0] if match else self.spec.modes[-1]
            cfg["sensor"]["output_size"] = (self.mode[0], self.mode[1])
            if self.spec.adjust_main_size is not None:
                cfg["main"]["size"] = self.spec.adjust_main_size
            self.camera_config = cfg
            self.controls_set.update(cfg["controls"])

        def set_controls(self, controls):
            self.controls_set.update(controls)

        def start(self):
            assert self.camera_config is not None, "configure first"
            self.started = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._emit, name=f"fake-picam-{self.camera_num}", daemon=True)
            self._thread.start()

        def stop(self):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                self._thread = None
            self.started = False

        def close(self):
            self.stop()
            self.closed = True

        # ---- frame generation
        def _period_ns(self) -> float:
            fdl = self.controls_set.get("FrameDurationLimits")
            if fdl:
                return float(fdl[0]) * 1000.0
            return 1e9 / self.mode[3]

        def _image(self, k: int) -> np.ndarray:
            w, h = self.camera_config["main"]["size"]
            img = np.empty((h, w + self.spec.stride_pad, 3), np.uint8)
            img[:] = BACKGROUND_BGR
            s = max(4, h // 4)
            x = (k * 3) % max(1, w - s)
            img[h // 3:h // 3 + s, x:x + s] = RED_BGR
            return img

        def _emit(self):
            clock = state.clock
            period = self._period_ns()
            t0 = clock.now_ns() + int(period)
            k = 0
            while not self._stop.is_set():
                t_sof = t0 + int(round(k * period))
                clock.sleep_until(t_sof + self.spec.latency_ns)
                if self._stop.is_set():
                    break
                if k in self.spec.drop_frames:
                    k += 1
                    continue
                md = {"SensorTimestamp": t_sof + self.spec.ts_offset_ns,
                      "ExposureTime": int(self.controls_set.get("ExposureTime", self.spec.exposure_us)),
                      "AnalogueGain": float(self.controls_set.get("AnalogueGain", 2.0)),
                      "FrameDuration": int(period / 1000)}
                if self.spec.autofocus:
                    md["LensPosition"] = float(self.controls_set.get("LensPosition", 1.0))
                cb = self.post_callback
                if cb is not None:
                    cb(CompletedRequest(self._image(k), md))
                self.frames_emitted += 1
                k += 1

    mod = types.ModuleType("picamera2")
    mod.Picamera2 = Picamera2
    mod.__fake__ = True
    lib = types.ModuleType("libcamera")
    lib.Transform = Transform
    lib.__fake__ = True
    return mod, lib


def install(monkeypatch, cameras: list[FakeCameraSpec], clock) -> FakeState:
    """Put fake ``picamera2`` and ``libcamera`` modules into ``sys.modules`` for the duration of a test."""
    state = FakeState(cameras, clock)
    mod, lib = _make_module(state)
    monkeypatch.setitem(sys.modules, "picamera2", mod)
    monkeypatch.setitem(sys.modules, "libcamera", lib)
    return state
