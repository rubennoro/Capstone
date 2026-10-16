"""ArgusCamera: CSI camera via GStreamer ``nvarguscamerasrc`` -> appsink (Jetson only; P1 §2.2, R4 §1.5).

``gi`` / ``Gst`` are imported lazily in ``_open`` so this module imports on any machine. All string building and
timestamp logic is in ``argus_logic`` (unit-tested).

Timestamps: the pipeline clock is forced to ``Gst.SystemClock`` with ``clock-type=MONOTONIC``, so a buffer's
``base_time + pts`` is CLOCK_MONOTONIC (the rig clock, D5). That is the time nvarguscamerasrc got the buffer from
Argus, i.e. later than mid-exposure by a roughly constant latency (P2 §4.3 ``offset_prior_ns``, refined by Kalibr).
``t_arrival_ns`` (when Python got the frame) is recorded too; ``LatencyTracker`` reports the median and drift.

NOT RUN ON HARDWARE YET. See ``sensor_rig/record/HARDWARE_CHECKLIST.md``.
"""

from __future__ import annotations

import numpy as np

from sensor_rig.drivers.argus_logic import (
    LatencyTracker,
    argus_params_from_camera,
    build_argus_pipeline,
    pts_to_monotonic,
)
from sensor_rig.drivers.base import BackendError, CameraBackend, Frame


class ArgusCamera(CameraBackend):
    def __init__(self, name: str, pipeline: str, width: int, height: int, fps: float, clock=None,
                 queue_size: int = 8, ring_size: int = 16, pull_timeout_s: float = 0.5,
                 start_timeout_s: float = 5.0, description: dict | None = None):
        if clock is not None and not hasattr(clock, "sample"):
            raise TypeError("clock must be a RealClock-like object")
        super().__init__(name, width, height, fps, queue_size=queue_size, ring_size=ring_size, clock=clock)
        self.pipeline_str = pipeline
        self.pull_timeout_s = pull_timeout_s
        self.start_timeout_s = start_timeout_s
        self.latency = LatencyTracker()
        self._description = description or {}
        self._Gst = None
        self._pipeline = None
        self._sink = None
        self.stats.update({"bad_stamps": 0, "pull_timeouts": 0})

    @classmethod
    def from_rig(cls, rig, name: str, **opts) -> "ArgusCamera":
        cam = rig.cameras[name]
        params = argus_params_from_camera(cam)
        params.update(opts.pop("pipeline_overrides", {}) or {})
        pipeline = build_argus_pipeline(**params)
        desc = {"sensor_id": params["sensor_id"], "sensor_mode": params["sensor_mode"],
                "exposure_time_range_ns": params.get("exposure_time_range_ns"),
                "gain_range": params.get("gain_range"), "pipeline": pipeline}
        return cls(name, pipeline, params["width"], params["height"], params["fps"], description=desc, **opts)

    # ---- GStreamer plumbing (hardware only) ----
    def _open(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as e:  # pragma: no cover - needs a Jetson
            raise BackendError("GStreamer Python bindings (python3-gi, gir1.2-gstreamer-1.0) are required for the "
                               "argus backend") from e
        Gst.init(None)
        self._Gst = Gst
        pipeline = Gst.parse_launch(self.pipeline_str)
        clk = Gst.SystemClock.obtain()
        clk.set_property("clock-type", Gst.ClockType.MONOTONIC)
        pipeline.use_clock(clk)
        self._pipeline = pipeline
        self._sink = pipeline.get_by_name("sink")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise BackendError(f"camera {self.name}: pipeline failed to start: {self.pipeline_str}")

    def _close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._Gst.State.NULL)
            self._pipeline = None
            self._sink = None

    def _sample_to_bgr(self, sample) -> np.ndarray:
        Gst = self._Gst
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise BackendError("cannot map GstBuffer")
        try:
            stride = info.size // h if info.size >= w * h * 4 else w * 4
            arr = np.ndarray((h, stride // 4, 4), dtype=np.uint8, buffer=info.data)
            return np.ascontiguousarray(arr[:, :w, :3])   # BGRx -> BGR copy (buffer is reused after unmap)
        finally:
            buf.unmap(info)

    def _run(self) -> None:  # pragma: no cover - needs a Jetson
        Gst = self._Gst
        seq = 0
        timeout_ns = int(self.pull_timeout_s * 1e9)
        first_deadline = self.clock.now_ns() + int(self.start_timeout_s * 1e9)
        got_first = False
        while not self._stopping.is_set():
            sample = self._sink.emit("try-pull-sample", timeout_ns)
            if sample is None:
                if self._sink.get_property("eos"):
                    raise BackendError(f"camera {self.name}: end of stream")
                self.stats["pull_timeouts"] += 1
                if not got_first and self.clock.now_ns() > first_deadline:
                    raise BackendError(f"camera {self.name}: no frame within {self.start_timeout_s} s")
                continue
            got_first = True
            t_arrival = self.clock.now_ns()
            buf = sample.get_buffer()
            pts = buf.pts
            if pts == Gst.CLOCK_TIME_NONE:
                self.stats["bad_stamps"] += 1
                continue
            t_ns = pts_to_monotonic(self._pipeline.get_base_time(), pts)
            lat = t_arrival - t_ns
            self.latency.update(lat)
            if lat < 0 or lat > 1_000_000_000:
                self.stats["bad_stamps"] += 1
            img = self._sample_to_bgr(sample)
            self._push(Frame(image=img, t_ns=t_ns, seq=seq, t_arrival_ns=t_arrival, meta={"pts_ns": int(pts)}))
            seq += 1

    def info(self) -> dict:
        d = {"backend": "argus", "width": self.width, "height": self.height, "fps_nominal": self.fps}
        d.update(self._description)
        d.update(self.latency.summary())
        return d
