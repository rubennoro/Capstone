"""MockCamera: synthetic, replayed or array-backed frames behind the ``CameraBackend`` interface (P1 §2.2).

Sources:

* ``MockCamera.synthetic(name, width, height, fps, pattern="moving_checker", clock=None)``: frame k is a checkerboard
  shifted by k pixels with the frame number drawn in; ``t_ns = t0 + round(k * 1e9 / fps)`` (+ optional jitter), where
  ``t0`` is the clock time at ``start()``. Frames are delivered at ``t_ns + latency_ns`` of the clock.
* ``MockCamera.replay(raw_episode_dir, cam, realtime=True, loop=False, preserve_timestamps=False)``: **frozen API**
  (deployment test D10d uses it). Replays a raw episode's JPEGs. With ``preserve_timestamps=False`` the stamps are
  re-based: ``t_ns = start_now + (t_rec - t_rec0)`` (plus one episode span per loop).
* ``MockCamera.from_arrays(images, t_ns)``: explicit images and stamps.

``realtime=False`` (replay / arrays) delivers frames as fast as the consumer reads them (the producer blocks when the
queue is full instead of dropping), which is what offline tests want.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from sensor_rig.drivers.base import CameraBackend, Frame, clock_or_real

PATTERNS = ("moving_checker", "noise", "flat")
REPLAY_ONLY_KWARGS = {"realtime", "loop", "preserve_timestamps", "restamp", "origin_ns", "start_now_ns",
                      "episode_dir", "replay", "synthetic"}
log = logging.getLogger(__name__)


def _checker_tile(width: int, height: int, square: int) -> np.ndarray:
    """A checkerboard 'square' pixels wider than the image, so shifting by k mod 2*square is a crop."""
    period = 2 * square
    ys, xs = np.mgrid[0:height, 0:width + period]
    board = (((xs // square) + (ys // square)) % 2).astype(np.uint8)
    img = np.empty((height, width + period, 3), np.uint8)
    img[..., 0] = 40 + 170 * board
    img[..., 1] = 60 + 150 * board
    img[..., 2] = 200 - 150 * board
    return img


class MockCamera(CameraBackend):
    def __init__(self, name: str, width: int, height: int, fps: float,
                 frame_fn: Callable[[int], np.ndarray], times_fn: Callable[[int, int], int | None],
                 realtime: bool = True, clock=None, latency_ns: int = 0, drop_seqs: Iterable[int] = (),
                 queue_size: int = 8, ring_size: int = 16, description: dict | None = None):
        """Generic constructor; prefer the classmethods.

        ``frame_fn(k)`` returns the BGR image of source frame k. ``times_fn(k, t_start_ns)`` returns its stamp or None
        when the source is exhausted. ``drop_seqs``: source indices that are skipped (seq still advances), to simulate
        frames lost before userspace.
        """
        super().__init__(name, width, height, fps, queue_size=queue_size, ring_size=ring_size, clock=clock)
        self._frame_fn = frame_fn
        self._times_fn = times_fn
        self.realtime = bool(realtime)
        self.latency_ns = int(latency_ns)
        self.drop_seqs = set(int(s) for s in drop_seqs)
        self._description = description or {}
        self.t_start_ns: int | None = None
        # When stamps are not on the live clock (preserved recorded stamps), pace delivery by their offset from this.
        self._pace_t0: int | None = None

    def _run(self) -> None:
        clock = self.clock
        t_start = clock.now_ns()
        self.t_start_ns = t_start
        k = 0
        while not self._stopping.is_set():
            t = self._times_fn(k, t_start)
            if t is None:
                return
            if self.realtime:
                due = t if self._pace_t0 is None else t_start + (t - self._pace_t0)
                clock.sleep_until(due + self.latency_ns)
                if self._stopping.is_set():
                    return
            if k in self.drop_seqs:
                k += 1
                continue
            img = self._frame_fn(k)
            self._push(Frame(image=img, t_ns=int(t), seq=k, t_arrival_ns=clock.now_ns(), meta={}),
                       block=not self.realtime)
            k += 1

    def info(self) -> dict:
        d = {"backend": "mock", "width": self.width, "height": self.height, "fps_nominal": self.fps}
        d.update(self._description)
        return d

    # ---------------------------------------------------------------------------------------------------------------
    @classmethod
    def synthetic(cls, name: str, width: int, height: int, fps: float, pattern: str = "moving_checker", clock=None,
                  jitter_ns: int = 0, seed: int = 0, n_frames: int | None = None, square: int = 16,
                  draw_number: bool = True, **kw) -> "MockCamera":
        ignored = sorted(set(kw) & REPLAY_ONLY_KWARGS)
        if ignored:
            log.debug("MockCamera.synthetic(%s): ignoring replay-only options %s", name, ignored)
            kw = {k: v for k, v in kw.items() if k not in REPLAY_ONLY_KWARGS}
        if pattern not in PATTERNS:
            raise ValueError(f"unknown mock pattern {pattern!r} (expected one of {PATTERNS})")
        width, height = int(width), int(height)
        period_ns = 1e9 / float(fps)
        rng = np.random.default_rng(seed)
        jitter = int(jitter_ns)
        tile = _checker_tile(width, height, square) if pattern == "moving_checker" else None
        noise_rng = np.random.default_rng(seed + 1)

        def frame_fn(k: int) -> np.ndarray:
            if pattern == "moving_checker":
                s = k % (2 * square)
                img = np.ascontiguousarray(tile[:, s:s + width])
            elif pattern == "noise":
                img = noise_rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            else:
                img = np.full((height, width, 3), 128, np.uint8)
            if draw_number:
                import cv2

                scale = max(0.4, height / 240.0)
                cv2.putText(img, str(k), (4, int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                            max(1, int(2 * scale)), cv2.LINE_AA)
            return img

        def times_fn(k: int, t_start: int) -> int | None:
            if n_frames is not None and k >= n_frames:
                return None
            j = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            return t_start + int(round(k * period_ns)) + j

        desc = {"pattern": pattern}
        return cls(name, width, height, fps, frame_fn, times_fn, realtime=True, clock=clock, description=desc, **kw)

    @classmethod
    def from_arrays(cls, images: Sequence[np.ndarray], t_ns: Sequence[int], name: str = "mock", fps: float | None = None,
                    realtime: bool = False, preserve_timestamps: bool = True, clock=None, **kw) -> "MockCamera":
        """Frames with explicit stamps. ``realtime=True`` paces delivery by the stamp intervals (re-based to start
        unless ``preserve_timestamps``)."""
        t = np.asarray(t_ns, dtype=np.int64)
        if len(images) != len(t) or len(t) == 0:
            raise ValueError("images and t_ns must be non-empty and the same length")
        h, w = images[0].shape[:2]
        if fps is None:
            fps = 1e9 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
        t0 = int(t[0])

        def times_fn(k: int, t_start: int) -> int | None:
            if k >= len(t):
                return None
            return int(t[k]) if preserve_timestamps else t_start + int(t[k]) - t0

        cam = cls(name, w, h, fps, lambda k: images[k], times_fn, realtime=realtime, clock=clock,
                  description={"source": "arrays"}, **kw)
        if preserve_timestamps:
            cam._pace_t0 = t0
        return cam

    @classmethod
    def replay(cls, raw_episode_dir, cam: str, realtime: bool = True, loop: bool = False,
               preserve_timestamps: bool = False, clock=None, origin_ns: int | None = None,
               start_now_ns: int | None = None, restamp: bool | None = None, **kw) -> "MockCamera":
        """Replay camera ``cam`` of a raw rig episode (FROZEN API, used by deployment test D10d; new kwargs optional).

        Stamps: ``preserve_timestamps=True`` keeps the recorded stamps (not allowed with ``loop``); otherwise
        ``t_ns = start_now + (t_rec - origin) + loop_index * span``.

        * ``origin_ns``: episode-wide origin; default = the earliest stamp of ANY camera in the episode, so cameras
          replayed from the same episode keep their relative (main <-> ultrawide) timing.
        * ``start_now_ns``: the live time that ``origin`` maps to; default = clock time at ``start()`` of this camera.
          Pass the same value to every camera (see ``replay_set``) to keep the offsets exact in live replays.
        * ``span`` (loop period) = episode-wide (latest stamp of any camera - origin + the largest median frame
          interval), identical for all cameras.
        * ``restamp`` is accepted as the negation of ``preserve_timestamps``.
        ``realtime=True`` paces delivery by the (re-based) stamps on ``clock``.
        """
        import cv2

        from sensor_rig.raw_episode import CAM_DIRS, RawEpisode

        if restamp is not None:
            preserve_timestamps = not restamp
        ep = RawEpisode(Path(raw_episode_dir))
        t_rec = ep.camera_times(cam)
        files = ep.camera_files(cam)
        if len(t_rec) == 0:
            raise ValueError(f"{raw_episode_dir}: camera {cam} has no frames")
        if loop and preserve_timestamps:
            raise ValueError("loop=True requires preserve_timestamps=False (stamps must keep increasing)")
        n = len(t_rec)
        dt_med = int(np.median(np.diff(t_rec))) if n > 1 else int(1e9 / 30)
        fps = 1e9 / dt_med
        # episode-wide origin and loop span
        firsts, lasts, dts = [], [], []
        for c in CAM_DIRS:
            tc = t_rec if c == cam else ep.camera_times(c)
            if len(tc):
                firsts.append(int(tc[0]))
                lasts.append(int(tc[-1]))
                dts.append(int(np.median(np.diff(tc))) if len(tc) > 1 else dt_med)
        origin = int(origin_ns) if origin_ns is not None else min(firsts)
        span = max(lasts) - origin + max(dts)
        first = cv2.imdecode(np.fromfile(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise IOError(f"cannot decode {files[0]}")
        h, w = first.shape[:2]
        t0 = int(t_rec[0])
        clk = clock_or_real(clock)

        def frame_fn(k: int) -> np.ndarray:
            i = k % n
            img = cv2.imdecode(np.fromfile(files[i], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"cannot decode {files[i]}")
            return img

        def times_fn(k: int, t_start: int) -> int | None:
            if k >= n and not loop:
                return None
            i, lap = k % n, k // n
            if preserve_timestamps:
                return int(t_rec[i])
            base = int(start_now_ns) if start_now_ns is not None else t_start
            return base + int(t_rec[i]) - origin + lap * span

        desc = {"source": "replay", "episode": str(raw_episode_dir), "camera": cam, "origin_ns": origin}
        mc = cls(cam, w, h, fps, frame_fn, times_fn, realtime=realtime, clock=clk, description=desc, **kw)
        if preserve_timestamps:
            mc._pace_t0 = origin if origin_ns is not None else t0
        return mc

    @classmethod
    def replay_set(cls, raw_episode_dir, cams=("main", "ultrawide"), clock=None, start_delay_s: float = 0.05,
                   start_now_ns: int | None = None, **kw) -> dict[str, "MockCamera"]:
        """Replay several cameras of one episode with a shared origin and a shared live start time
        (default: clock now + ``start_delay_s``), so their relative timing is preserved exactly."""
        from sensor_rig.raw_episode import RawEpisode

        clk = clock_or_real(clock)
        ep = RawEpisode(Path(raw_episode_dir))
        origin = min(int(ep.camera_times(c)[0]) for c in cams if len(ep.camera_times(c)))
        if start_now_ns is None:
            start_now_ns = clk.now_ns() + int(start_delay_s * 1e9)
        return {c: cls.replay(raw_episode_dir, c, clock=clk, origin_ns=origin, start_now_ns=start_now_ns, **kw)
                for c in cams}
