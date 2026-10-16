"""Per-episode IMU writer (P1 §3.4): writes samples with ``start - preroll <= t <= stop + postroll``.

The recorder's IMU pump keeps a pre-roll ring of recent batches; at episode start it hands that ring to the new
``ImuStreamWriter``, so the episode's IMU log begins ``preroll_s`` before the first frame (VIO initialisation needs
IMU before the first image).
"""

from __future__ import annotations

import threading
import time

import numpy as np

from sensor_rig.drivers.base import ImuBatch


class ImuStreamWriter:
    def __init__(self, writer, start_ns: int, preroll_s: float = 1.0, postroll_s: float = 1.0,
                 preroll_batches: list[ImuBatch] | None = None, flush_period_s: float = 0.5):
        self.writer = writer
        self.imu = writer.imu_writer()
        self.start_ns = int(start_ns)
        self.preroll_ns = int(preroll_s * 1e9)
        self.postroll_ns = int(postroll_s * 1e9)
        self.stop_ns: int | None = None
        self.done = False
        self.last_t_ns: int | None = None
        self.sample_times: list[np.ndarray] = []   # for the live rate check
        self.flush_period_s = flush_period_s
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()
        self._finished = False
        for b in preroll_batches or []:
            self.submit(b)

    @property
    def window_start_ns(self) -> int:
        return self.start_ns - self.preroll_ns

    @property
    def window_end_ns(self) -> int | None:
        return None if self.stop_ns is None else self.stop_ns + self.postroll_ns

    def submit(self, batch: ImuBatch) -> None:
        with self._lock:
            if self._finished or not len(batch):
                return
            t = batch.t_ns
            mask = t >= self.window_start_ns
            if self.last_t_ns is not None:
                mask &= t > self.last_t_ns
            end = self.window_end_ns
            if end is not None:
                if t[-1] > end:
                    self.done = True
                mask &= t <= end
            if not mask.any():
                return
            b = batch.select(mask) if not mask.all() else batch
            self.imu.add_batch(b.t_ns, b.gyro, b.accel, b.aux or None)
            self.last_t_ns = int(b.t_ns[-1])
            self.sample_times.append(b.t_ns.copy())
            if time.monotonic() - self._last_flush > self.flush_period_s:
                self.imu.flush()
                self._last_flush = time.monotonic()

    def request_stop(self, stop_ns: int) -> None:
        with self._lock:
            self.stop_ns = int(stop_ns)
            if self.last_t_ns is not None and self.last_t_ns > self.window_end_ns:
                self.done = True

    def finish(self) -> dict:
        with self._lock:
            self._finished = True
            self.imu.flush()
        t = np.concatenate(self.sample_times) if self.sample_times else np.zeros(0, np.int64)
        rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e9) if len(t) > 1 else None
        return {"rate_measured": rate}
