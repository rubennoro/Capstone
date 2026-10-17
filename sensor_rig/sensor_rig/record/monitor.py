"""Live status + abort rules while recording (P1 §3.5, mirroring the app: VC:1821 fps rule, VC:1240 start guard).

``evaluate_live_abort(snapshot, cfg)`` is pure (unit-tested); ``Monitor`` is the 1 Hz thread that feeds it the
recorder's ``live_snapshot()`` and aborts the episode when a rule fires.

Rules (each returns a reason string):
* ``main_fps_low``: after ``after_s``, main frames in the last 1 s (sensor time) < ``min_fps_ratio * fps_nominal``
  (0.52 x 60 = 31.2, the app's rule; 15.6 for a 30 fps main). A camera that delivered nothing for ``stale_s`` counts
  as 0 fps.
* ``queue_full``: queue_full drops > ``max_queue_full_frac`` (2 %) of a camera's frames in the last 2 s.
* ``imu_rate_low``: after 2 s, IMU samples in the last 2 s < ``imu_min_ratio`` (0.9) x ODR.
* ``disk_low``: free space on the output disk < ``min_disk_free_gb``.
* ``<cam>_fps_low`` (opt-in, ``all_cameras_fps: true``): the main-camera fps rule applied to every camera, e.g. a
  stalled ultrawide (the VIO camera).
* ``frame_gap`` (opt-in, ``max_frame_gap_s``): after ``after_s``, any camera with an interval between consecutive
  frames (or since its last frame) longer than this, within the last ``gap_window_s``.
The two opt-in rules are off by default, so existing rig.yaml files behave as before; ``collect`` and
``configs/rig_default.yaml`` turn them on.
There is no live pose; the app's 7 m/s rule is applied at export.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, fields
from typing import Callable

import numpy as np


@dataclass
class LiveAbortConfig:
    min_fps_ratio: float = 0.52
    after_s: float = 1.0
    min_disk_free_gb: float = 2.0
    max_queue_full_frac: float = 0.02
    queue_window_s: float = 2.0
    imu_min_ratio: float = 0.9
    imu_window_s: float = 2.0
    stale_s: float = 0.5
    warmup_min_fps_ratio: float = 0.85   # open(): refuse if a warm-up fps is below this x nominal
    all_cameras_fps: bool = False        # opt-in: min_fps_ratio rule for every camera, not only main
    max_frame_gap_s: float | None = None  # opt-in: abort when a camera has a longer gap between frames
    gap_window_s: float = 2.0            # how far back the gap rule looks (>= monitor period)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LiveAbortConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown capture.live_abort keys: {sorted(unknown)}")
        return cls(**d)


def _count_window(times: np.ndarray, t_end: int, window_ns: int) -> int:
    """Number of stamps in (t_end - window, t_end]; ``times`` sorted."""
    lo = np.searchsorted(times, t_end - window_ns, side="right")
    hi = np.searchsorted(times, t_end, side="right")
    return int(hi - lo)


def evaluate_live_abort(snap: dict, cfg: LiveAbortConfig) -> str | None:
    """``snap``::

        {"elapsed_s": float, "now_ns": int, "disk_free_gb": float | None,
         "cameras": {name: {"fps_nominal": float, "is_main": bool,
                            "frame_times": sorted int64 array (all frames in the episode, incl. queue_full ones),
                            "queue_full_times": sorted int64 array}},
         "imu": {"rate_nominal": float, "sample_times": sorted int64 array} | None}
    """
    elapsed = float(snap["elapsed_s"])
    now = int(snap["now_ns"])
    stale_ns = int(cfg.stale_s * 1e9)
    free = snap.get("disk_free_gb")
    if free is not None and free < cfg.min_disk_free_gb:
        return f"disk_low: {free:.2f} GB free < {cfg.min_disk_free_gb} GB"
    for name, c in (snap.get("cameras") or {}).items():
        ft = np.asarray(c.get("frame_times", []), dtype=np.int64)
        qf = np.asarray(c.get("queue_full_times", []), dtype=np.int64)
        if (c.get("is_main") or cfg.all_cameras_fps) and elapsed >= cfg.after_s:
            if len(ft) == 0 or now - int(ft[-1]) > stale_ns:
                n = 0
            else:
                n = _count_window(ft, int(ft[-1]), 1_000_000_000)
            thresh = cfg.min_fps_ratio * float(c["fps_nominal"])
            if n < thresh:
                rule = "main_fps_low" if c.get("is_main") else f"{name}_fps_low"
                return f"{rule}: {name} {n} frames in the last 1 s < {thresh:.1f}"
        if cfg.max_frame_gap_s is not None and elapsed >= cfg.after_s and len(ft):
            gap_ns = int(cfg.max_frame_gap_s * 1e9)
            since_last = now - int(ft[-1])
            recent = ft[ft >= now - int(cfg.gap_window_s * 1e9)]
            worst = int(np.max(np.diff(recent))) if len(recent) > 1 else 0
            worst = max(worst, since_last)
            if worst > gap_ns:
                return f"frame_gap: {name} has a {worst / 1e6:.0f} ms gap between frames > {cfg.max_frame_gap_s * 1e3:.0f} ms"
        if len(ft) and len(qf):
            w = int(cfg.queue_window_s * 1e9)
            t_end = int(ft[-1])
            n_all = _count_window(ft, t_end, w)
            n_qf = _count_window(qf, t_end, w)
            if n_all > 0 and n_qf / n_all > cfg.max_queue_full_frac:
                return (f"queue_full: {name} dropped {n_qf}/{n_all} frames in the last {cfg.queue_window_s:g} s "
                        f"(> {cfg.max_queue_full_frac:.0%})")
    imu = snap.get("imu")
    if imu is not None and elapsed >= cfg.imu_window_s:
        st = np.asarray(imu.get("sample_times", []), dtype=np.int64)
        w = int(cfg.imu_window_s * 1e9)
        if len(st) == 0 or now - int(st[-1]) > stale_ns:
            n = 0
        else:
            n = _count_window(st, int(st[-1]), w)
        need = cfg.imu_min_ratio * float(imu["rate_nominal"]) * cfg.imu_window_s
        if n < need:
            return f"imu_rate_low: {n} samples in the last {cfg.imu_window_s:g} s < {need:.0f}"
    return None


def disk_free_gb(path) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


class Monitor:
    """1 Hz (clock time) status line + live abort rules. Calls ``on_abort(reason)`` once per episode."""

    def __init__(self, recorder, cfg: LiveAbortConfig, period_s: float = 1.0,
                 status: Callable[[str], None] | None = None):
        self.recorder = recorder
        self.cfg = cfg
        self.period_s = period_s
        self.status = status
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        clock = self.recorder.clock
        period_ns = int(self.period_s * 1e9)
        next_t = clock.now_ns() + period_ns
        while not self._stop.is_set():
            # sleep in small real-time steps so stop() is responsive
            while not self._stop.is_set() and clock.now_ns() < next_t:
                clock.sleep(min(0.05, self.period_s))
            if self._stop.is_set():
                return
            next_t += period_ns
            try:
                self.tick()
            except Exception as e:  # never let the monitor kill the recorder
                if self.status:
                    self.status(f"monitor error: {e!r}")

    def tick(self) -> str | None:
        snap = self.recorder.live_snapshot()
        if snap is None:
            return None
        if self.status:
            self.status(self.recorder.status_line())
        reason = evaluate_live_abort(snap, self.cfg)
        if reason:
            self.recorder.abort_episode(reason, from_monitor=True)
        return reason
