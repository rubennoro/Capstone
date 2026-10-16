"""Backend interfaces shared by the recorder and deployment (P1 §2.2, RECONCILIATION R4).

* ``Frame``: one camera image + its capture stamp (CLOCK_MONOTONIC ns, *before* any rig.yaml offsets). ``Frame.t_ns``
  is exactly the value the recorder writes to ``<cam>/data.csv``.
* ``CameraBackend``: ``start / read / latest / read_recent / stop``. ``read`` returns frames in capture order from a
  bounded queue (oldest dropped on overflow, counted in ``stats["internal_drops"]``); ``latest`` and ``read_recent``
  look at a ring of the newest frames (for deployment pairing) without consuming the queue.
* ``ImuBatch`` / ``ImuBackend``: batches of IMU samples (rad/s, m/s^2, IMU frame, CLOCK_MONOTONIC ns).
* ``make_camera`` / ``make_imu``: build a backend from a ``RigConfig`` (P2 §4.1 camera/imu sections + the recorder-only
  ``capture:`` mapping, R3). Hardware modules are imported lazily.
* ``RealClock`` / ``SimClock``: the clock objects mocks and the recorder use (``now_ns``, ``sleep_until``, ``sample``).

Hardware backends run a producer thread that calls ``_push``; mocks use the same machinery, so the queue/ring code the
recorder relies on is exercised by the laptop tests.
"""

from __future__ import annotations

import abc
import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sensor_rig.clock import ClockSample, now_ns, sample_clocks

DEFAULT_QUEUE_SIZE = 8
DEFAULT_RING_SIZE = 16


# --------------------------------------------------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------------------------------------------------
class RealClock:
    """CLOCK_MONOTONIC (the rig clock, D5)."""

    speed = 1.0

    def now_ns(self) -> int:
        return now_ns()

    def sleep_until(self, t_ns: int) -> None:
        dt = (int(t_ns) - now_ns()) / 1e9
        if dt > 0:
            time.sleep(dt)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def sample(self) -> ClockSample:
        return sample_clocks()


class SimClock:
    """A shared, faster-than-real-time clock for tests.

    Sim time runs ``speed`` times faster than the host's monotonic clock, starting at ``start_ns``. Every thread sees
    the same time (the recorder is multi-threaded, so a per-thread "jump to the next event" clock would let streams
    drift apart). ``sleep_until`` sleeps the corresponding real interval. ``sample()`` returns a ``ClockSample`` whose
    realtime is ``realtime_offset_ns + now`` so episode names/wall times are deterministic.
    """

    def __init__(self, speed: float = 10.0, start_ns: int = 1_000_000_000_000,
                 realtime_offset_ns: int = 1_792_508_400_000_000_000 - 1_000_000_000_000):
        if not (speed > 0 and np.isfinite(speed)):
            raise ValueError("SimClock speed must be a positive finite number")
        self.speed = float(speed)
        self.start_ns = int(start_ns)
        self.realtime_offset_ns = int(realtime_offset_ns)  # default: wall clock 2026-10-20T15:00:00Z at start_ns
        self._real0 = time.monotonic_ns()

    def now_ns(self) -> int:
        return self.start_ns + int((time.monotonic_ns() - self._real0) * self.speed)

    def sleep_until(self, t_ns: int) -> None:
        dt_real = (int(t_ns) - self.now_ns()) / self.speed / 1e9
        if dt_real > 0:
            time.sleep(dt_real)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds / self.speed)

    def sample(self) -> ClockSample:
        t = self.now_ns()
        return ClockSample(monotonic_ns=t, monotonic_raw_ns=t, realtime_ns=t + self.realtime_offset_ns,
                           uncertainty_ns=0)


def clock_or_real(clock) -> Any:
    return clock if clock is not None else RealClock()


# --------------------------------------------------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    image: np.ndarray            # HxWx3 uint8 BGR, physical camera resolution (unrectified)
    t_ns: int                    # capture time, CLOCK_MONOTONIC ns, BEFORE rig.yaml offsets
    seq: int                     # backend counter; gaps mean frames lost before userspace
    t_arrival_ns: int            # CLOCK_MONOTONIC when the frame reached Python
    meta: dict = field(default_factory=dict)   # optional: exposure_ns, gain, sensor_ts_raw_ns, ...


class BackendError(RuntimeError):
    pass


class CameraBackend(abc.ABC):
    """Camera interface (one interface for the recorder AND deployment, R4).

    Subclasses implement ``_run()`` (the producer loop, executed in a thread; it must return when ``self._stopping``
    is set) and may override ``_open()`` / ``_close()``. The producer calls ``self._push(frame)``.
    """

    name: str
    width: int
    height: int
    fps: float

    def __init__(self, name: str, width: int, height: int, fps: float, queue_size: int = DEFAULT_QUEUE_SIZE,
                 ring_size: int = DEFAULT_RING_SIZE, clock=None):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.clock = clock_or_real(clock)
        self.queue_size = max(1, int(queue_size))
        self._queue: collections.deque[Frame] = collections.deque()
        self._ring: collections.deque[Frame] = collections.deque(maxlen=max(8, int(ring_size)))
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._finished = False   # producer ran out of frames (finite mock sources)
        self.stats: dict[str, Any] = {"frames": 0, "internal_drops": 0, "errors": 0}

    # ---- producer side ----
    def _push(self, frame: Frame, block: bool = False) -> None:
        with self._cond:
            if len(self._queue) >= self.queue_size:
                if block:
                    while len(self._queue) >= self.queue_size and not self._stopping.is_set():
                        self._cond.wait(0.05)
                    if self._stopping.is_set():
                        return
                else:
                    self._queue.popleft()
                    self.stats["internal_drops"] += 1
            self._queue.append(frame)
            self._ring.append(frame)
            self.stats["frames"] += 1
            self._cond.notify_all()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as e:  # surfaced to the consumer by read()
            self._error = e
            self.stats["errors"] += 1
        finally:
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    @abc.abstractmethod
    def _run(self) -> None:
        ...

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    # ---- consumer side ----
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"camera {self.name} already started")
        self._stopping.clear()
        self._open()
        self._thread = threading.Thread(target=self._run_guarded, name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        """Next frame in capture order (blocking). None on timeout or when a finite source is exhausted.

        Raises ``BackendError`` if the producer thread died with an exception and the queue is empty.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while not self._queue:
                if self._error is not None:
                    raise BackendError(f"camera {self.name} failed: {self._error!r}") from self._error
                if self._finished or self._stopping.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            f = self._queue.popleft()
            self._cond.notify_all()
            return f

    def latest(self) -> Frame | None:
        """Newest frame (non-blocking, does not consume the read() queue)."""
        with self._cond:
            return self._ring[-1] if self._ring else None

    def read_recent(self, n: int) -> list[Frame]:
        """Up to ``n`` newest frames, newest last (ring of >= 8 frames; for deployment 'nearest' pairing)."""
        with self._cond:
            if n <= 0:
                return []
            return list(self._ring)[-n:]

    @property
    def exhausted(self) -> bool:
        """True when a finite source has delivered everything and the queue is empty."""
        with self._cond:
            return self._finished and not self._queue

    def stop(self) -> None:
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._close()

    def info(self) -> dict:
        """Backend description for meta.json ``sensors.<cam>``."""
        return {"backend": type(self).__name__, "width": self.width, "height": self.height, "fps_nominal": self.fps}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


# --------------------------------------------------------------------------------------------------------------------
# IMU
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class ImuBatch:
    t_ns: np.ndarray          # (N,) int64, CLOCK_MONOTONIC, strictly increasing
    gyro: np.ndarray          # (N,3) rad/s, IMU frame
    accel: np.ndarray         # (N,3) m/s^2 (specific force), IMU frame
    aux: dict = field(default_factory=dict)   # optional per-sample arrays: chip_us, temp_c, batch, host_read_ns

    def __len__(self) -> int:
        return int(len(self.t_ns))

    @staticmethod
    def concat(batches: list["ImuBatch"]) -> "ImuBatch":
        batches = [b for b in batches if len(b)]
        if not batches:
            return ImuBatch(np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)), {})
        keys = set(batches[0].aux)
        for b in batches[1:]:
            keys &= set(b.aux)
        return ImuBatch(
            np.concatenate([b.t_ns for b in batches]).astype(np.int64),
            np.concatenate([b.gyro for b in batches]),
            np.concatenate([b.accel for b in batches]),
            {k: np.concatenate([np.asarray(b.aux[k]) for b in batches]) for k in keys},
        )

    def select(self, mask: np.ndarray) -> "ImuBatch":
        return ImuBatch(self.t_ns[mask], self.gyro[mask], self.accel[mask],
                        {k: np.asarray(v)[mask] for k, v in self.aux.items()})


class ImuBackend(abc.ABC):
    """IMU interface. Subclasses implement ``_run()`` and call ``_push(batch)``; ``read_batch`` returns every pending
    sample concatenated into one batch (None on timeout)."""

    name: str = "imu0"
    rate_hz: float

    def __init__(self, rate_hz: float, clock=None, max_pending_s: float = 4.0):
        self.rate_hz = float(rate_hz)
        self.clock = clock_or_real(clock)
        self._pending: collections.deque[ImuBatch] = collections.deque()
        self._pending_samples = 0
        self._max_pending = max(1, int(max_pending_s * self.rate_hz))
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._finished = False
        self.stats: dict[str, Any] = {"samples": 0, "internal_drops": 0, "fifo_overflows": 0, "invalid_samples": 0,
                                      "errors": 0}

    def _push(self, batch: ImuBatch) -> None:
        if not len(batch):
            return
        with self._cond:
            self._pending.append(batch)
            self._pending_samples += len(batch)
            while self._pending_samples > self._max_pending and len(self._pending) > 1:
                old = self._pending.popleft()
                self._pending_samples -= len(old)
                self.stats["internal_drops"] += len(old)
            self.stats["samples"] += len(batch)
            self._cond.notify_all()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as e:
            self._error = e
            self.stats["errors"] += 1
        finally:
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    @abc.abstractmethod
    def _run(self) -> None:
        ...

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("IMU already started")
        self._stopping.clear()
        self._open()
        self._thread = threading.Thread(target=self._run_guarded, name="imu", daemon=True)
        self._thread.start()

    def read_batch(self, timeout_s: float = 0.05) -> ImuBatch | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while not self._pending:
                if self._error is not None:
                    raise BackendError(f"IMU failed: {self._error!r}") from self._error
                if self._finished or self._stopping.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            batches = list(self._pending)
            self._pending.clear()
            self._pending_samples = 0
        return ImuBatch.concat(batches)

    def stop(self) -> None:
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._close()

    def info(self) -> dict:
        """Backend description for meta.json ``sensors.imu0``."""
        return {"backend": type(self).__name__, "odr_hz": self.rate_hz}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


# --------------------------------------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------------------------------------
CAMERA_BACKENDS = ("mock", "argus")
IMU_BACKENDS = ("mock", "icm42688_spi")


def _capture(rig) -> dict:
    cap = getattr(rig, "capture", None)
    return cap if isinstance(cap, dict) else {}


def camera_backend_name(rig, name: str) -> str:
    """Backend for camera ``name`` from ``rig.capture.backends`` (R3); default ``argus``."""
    return str((_capture(rig).get("backends") or {}).get(name, "argus"))


def imu_backend_name(rig) -> str:
    return str((_capture(rig).get("backends") or {}).get("imu", "icm42688_spi"))


def make_camera(rig, name: str, backend: str | None = None, **opts) -> CameraBackend:
    """Build the camera backend for physical camera ``name``.

    ``backend``: ``"mock"`` or ``"argus"`` (default: ``rig.capture.backends.<name>``, else ``argus``).
    Mock options: ``episode_dir`` (replay a raw episode), ``realtime`` (default True), ``loop``,
    ``preserve_timestamps`` (``restamp=True`` is accepted as its negation), ``origin_ns`` / ``start_now_ns`` (shared
    replay origin / live start, see ``MockCamera.replay``), ``pattern``, ``clock``, ``jitter_ns``, ``drop_seqs``,
    ``latency_ns``. Replay-only options are ignored by synthetic mocks. Argus options are forwarded to ``ArgusCamera.from_rig``.
    """
    backend = backend or camera_backend_name(rig, name)
    cam_cfg = rig.cameras[name]
    w, h = (int(v) for v in cam_cfg.resolution)
    fps = float(cam_cfg.fps)
    if backend == "mock":
        from sensor_rig.drivers.mock_camera import MockCamera

        opts = dict(opts)
        episode_dir = opts.pop("episode_dir", None) or opts.pop("replay", None)
        opts.pop("synthetic", None)
        if episode_dir is not None:
            # replay kwargs (realtime, loop, preserve_timestamps / restamp, origin_ns, start_now_ns, clock) pass through
            return MockCamera.replay(episode_dir, name, **opts)
        # synthetic ignores replay-only kwargs (debug log)
        mock_cfg = _capture(rig).get("mock") or {}
        opts.setdefault("pattern", mock_cfg.get("pattern", "moving_checker"))
        return MockCamera.synthetic(name, w, h, fps, **opts)
    if backend == "argus":
        from sensor_rig.drivers.argus_camera import ArgusCamera

        return ArgusCamera.from_rig(rig, name, **opts)
    raise ValueError(f"unknown camera backend {backend!r} (expected one of {CAMERA_BACKENDS})")


def make_imu(rig, backend: str | None = None, **opts) -> ImuBackend:
    """Build the IMU backend: ``"mock"`` (static, gravity + noise) or ``"icm42688_spi"``."""
    backend = backend or imu_backend_name(rig)
    imu_cfg = getattr(rig, "imu", None)
    if backend == "mock":
        from sensor_rig.drivers.mock_imu import MockImu

        rate = opts.pop("rate_hz", None) or getattr(imu_cfg, "output_rate_hz", None) or getattr(imu_cfg, "odr_hz", 1000)
        return MockImu.static(rate_hz=float(rate), **opts)
    if backend == "icm42688_spi":
        from sensor_rig.drivers.icm42688_spi import Icm42688Spi

        return Icm42688Spi.from_rig(rig, **opts)
    raise ValueError(f"unknown IMU backend {backend!r} (expected one of {IMU_BACKENDS})")
