"""Replay a list of raw episodes through the camera/IMU interfaces, one source per recorded episode (T3).

Used by ``collect --mock-replay``. All backends share one ``ReplayPlaylist``:

* Between episodes ("idle") the current source loops, so the recorder's warm-up and rate checks see live sensors.
* ``advance(kind)`` picks the next source of that kind (calibration or demonstration, round robin) and restarts it
  ``lead_s`` from now. Every frame / IMU sample of that pass is stamped ``t_live = t_src + offset`` with ONE constant
  ``offset = base - origin`` (``origin`` = the source's first camera frame; IMU samples that would land before the
  last delivered sample are skipped). ``collect`` stores it in the
  episode's ``meta.json`` (``sim.replay_offset_ns``), and ``pose-import`` adds it to the source's trajectory, so a
  simulated trajectory lines up exactly with the re-recorded frames.
* ``finished`` turns True once the pass has delivered its last camera frame; ``collect`` then stops the episode, before the
  source loops again (a loop seam would be a jump in the data).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensor_rig.drivers.base import CameraBackend, Frame, ImuBackend, ImuBatch, clock_or_real

CAMS = ("main", "ultrawide")


@dataclass
class _Source:
    path: Path
    name: str
    kind: str
    cam_t: dict          # cam -> int64 stamps (as recorded)
    cam_files: dict      # cam -> list[Path]
    imu_t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    origin: int
    end: int             # latest stamp of any sensor
    span: int            # loop period
    cam_end: int         # latest camera frame


class ReplayPlaylist:
    def __init__(self, episode_dirs, clock=None, lead_s: float = 0.05):
        from sensor_rig.raw_episode import RawEpisode, find_raw_episodes

        eps = []
        for d in episode_dirs:
            eps.extend(find_raw_episodes(Path(d)))
        if not eps:
            raise ValueError(f"no raw episodes in {list(map(str, episode_dirs))}")
        self.sources: list[_Source] = []
        for ep in eps:
            ct = {c: np.asarray(ep.camera_times(c), np.int64) for c in CAMS}
            if any(len(v) < 2 for v in ct.values()):
                raise ValueError(f"{ep.path}: needs >= 2 frames per camera")
            it, g, a = ep.imu()
            # origin = first camera frame (the IMU usually starts earlier: its pre-roll maps to just before the
            # restart and is clipped so the live IMU stream stays strictly increasing)
            firsts = [int(v[0]) for v in ct.values()]
            lasts = [int(v[-1]) for v in ct.values()] + ([int(it[-1])] if len(it) else [])
            dts = [int(np.median(np.diff(v))) for v in ct.values()]
            origin, end = min(firsts), max(lasts)
            self.sources.append(_Source(ep.path, ep.name, ep.kind, ct, {c: ep.camera_files(c) for c in CAMS},
                                        it, g, a, origin, end, end - origin + max(dts),
                                        max(int(v[-1]) for v in ct.values())))
        self.first_episode = eps[0]
        self.clock = clock_or_real(clock)
        self.lead_ns = int(lead_s * 1e9)
        self._lock = threading.Lock()
        self._rr: dict[str, int] = {}
        self.cur = 0
        self.generation = 0
        self.base = None          # live time that the current source's origin maps to
        self.once = False         # True while a recorded pass is playing
        self.history: list[dict] = []

    # ---- control (collect) ----
    def start_idle(self) -> None:
        with self._lock:
            if self.base is None:
                self.base = self.clock.now_ns() + self.lead_ns

    def advance(self, kind: str) -> dict:
        """Restart with the next source of ``kind`` (falls back to any source). Returns the meta.sim record."""
        with self._lock:
            idx = [i for i, s in enumerate(self.sources) if s.kind == kind] or list(range(len(self.sources)))
            k = self._rr.get(kind, 0)
            self._rr[kind] = k + 1
            self.cur = idx[k % len(idx)]
            src = self.sources[self.cur]
            self.base = self.clock.now_ns() + self.lead_ns
            self.once = True
            self.generation += 1
            rec = {"replay_source": src.name, "replay_source_dir": str(src.path), "replay_offset_ns": self.base -
                   src.origin, "replay_kind": src.kind}
            self.history.append(rec)
            return rec

    @property
    def finished(self) -> bool:
        with self._lock:
            if not self.once:
                return False
            src = self.sources[self.cur]
            # the last camera frame is out (+ one frame interval): stop before the cameras see a loop seam
            return self.clock.now_ns() > self.base + (src.cam_end - src.origin) + 60_000_000

    def end_pass(self) -> None:
        with self._lock:
            self.once = False

    def snapshot(self):
        with self._lock:
            return self.generation, self.sources[self.cur], self.base


class PlaylistCamera(CameraBackend):
    def __init__(self, playlist: ReplayPlaylist, cam: str, queue_size: int = 8):
        import cv2

        src0 = playlist.sources[0]
        img = cv2.imdecode(np.fromfile(src0.cam_files[cam][0], dtype=np.uint8), cv2.IMREAD_COLOR)
        fps = 1e9 / float(np.median(np.diff(src0.cam_t[cam])))
        super().__init__(cam, img.shape[1], img.shape[0], fps, queue_size=queue_size, clock=playlist.clock)
        self.playlist = playlist
        self.cam = cam

    def _run(self) -> None:
        import cv2

        pl, clock = self.playlist, self.clock
        pl.start_idle()
        seq = 0
        while not self._stopping.is_set():
            gen, src, base = pl.snapshot()
            t_src, files = src.cam_t[self.cam], src.cam_files[self.cam]
            lap = 0
            restart = False
            while not self._stopping.is_set() and not restart:
                for i in range(len(t_src)):
                    t = base + lap * src.span + int(t_src[i]) - src.origin
                    while not self._stopping.is_set():
                        now = clock.now_ns()
                        if pl.generation != gen:
                            restart = True
                            break
                        if now >= t:
                            break
                        clock.sleep(min(0.005, (t - now) / 1e9))
                    if restart or self._stopping.is_set():
                        break
                    img = cv2.imdecode(np.fromfile(files[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                    self._push(Frame(img, t, seq, clock.now_ns(), {"replay_index": i, "lap": lap}))
                    seq += 1
                lap += 1

    def info(self) -> dict:
        return {"backend": "mock", "source": "replay_playlist", "width": self.width, "height": self.height,
                "fps_nominal": self.fps}


class PlaylistImu(ImuBackend):
    def __init__(self, playlist: ReplayPlaylist, batch_period_s: float = 0.005):
        src0 = playlist.sources[0]
        if len(src0.imu_t) < 2:
            raise ValueError(f"{src0.path}: no IMU samples to replay")
        super().__init__(1e9 / float(np.median(np.diff(src0.imu_t))), clock=playlist.clock)
        self.playlist = playlist
        self.batch_ns = int(batch_period_s * 1e9)

    def _run(self) -> None:
        pl, clock = self.playlist, self.clock
        pl.start_idle()
        last = -1
        while not self._stopping.is_set():
            gen, src, base = pl.snapshot()
            n = len(src.imu_t)
            if n == 0:
                clock.sleep(0.01)
                continue
            lap, k = 0, 0
            while not self._stopping.is_set() and pl.generation == gen:
                clock.sleep(self.batch_ns / 1e9)
                now = clock.now_ns()
                out_t, idx = [], []
                while True:
                    t = base + lap * src.span + int(src.imu_t[k]) - src.origin
                    if t > now:
                        break
                    if t > last:
                        out_t.append(t)
                        idx.append(k)
                        last = t
                    k += 1
                    if k >= n:
                        k, lap = 0, lap + 1
                if out_t:
                    ii = np.asarray(idx)
                    self._push(ImuBatch(np.asarray(out_t, np.int64), src.gyro[ii], src.accel[ii], {}))

    def info(self) -> dict:
        return {"backend": "mock", "source": "replay_playlist", "odr_hz": self.rate_hz}
