"""FakePiSender: the stand-in Pi for unit tests and --mock (STRUCTURE.md §5.14). Never used on the real arm.

It plays back a short recorded clip only to produce realistic C0 traffic, and speaks exactly C0: thread
"fake-pi-frames" binds a PUB socket and sends every frame when it is due; thread "fake-pi-time" binds a REP socket and
answers time_sync with ``pi_clock_ns(now)``. The clip loops with ``seq`` and the clock still counting up.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable

from contracts import Publisher, RpcServer, TimeSyncReply, TimeSyncRequest

from process_frames.pi_recording import PiRecording

log = logging.getLogger("process_frames.fake_pi_sender")

#: How often each thread looks at its stop flag while idle.
_POLL_S = 0.02
CAMERAS = ("main", "ultrawide")


class FakePiSender:
    def __init__(self, recording: PiRecording, frames_endpoint: str, time_sync_endpoint: str, *,
                 pi_clock_offset_ns: int, drift_ppm: float, send_hwm: int,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.recording = recording
        self.frames_endpoint = frames_endpoint          # replaced by the bound address in start()
        self.time_sync_endpoint = time_sync_endpoint
        self._offset_ns = int(pi_clock_offset_ns)
        self._drift = float(drift_ppm) * 1e-6
        self._send_hwm = send_hwm
        self._clock = clock
        self._start_ns: int | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._bound = {"frames": threading.Event(), "time": threading.Event()}
        self._bind_error: BaseException | None = None
        self._lock = threading.Lock()
        self._paused = {cam: False for cam in CAMERAS}
        self._drop = {cam: 0 for cam in CAMERAS}
        self.sent = {cam: 0 for cam in CAMERAS}

    # ---------------------------------------------------------------------------------------------- control

    def pi_clock_ns(self, pc_ns: int) -> int:
        """``pc_ns + offset + drift_ppm * 1e-6 * (pc_ns - start)``."""
        start = self._start_ns if self._start_ns is not None else pc_ns
        return int(pc_ns) + self._offset_ns + int(round(self._drift * (pc_ns - start)))

    def start(self) -> None:
        """Start both threads; returns once both sockets are bound (a bind error is raised here)."""
        self._start_ns = self._clock()
        self._threads = [threading.Thread(target=self._run_frames, name="fake-pi-frames", daemon=True),
                         threading.Thread(target=self._run_time, name="fake-pi-time", daemon=True)]
        for t in self._threads:
            t.start()
        for ev in self._bound.values():
            ev.wait()
        if self._bind_error is not None:
            self.stop()
            raise self._bind_error

    def stop(self) -> None:
        """Stop both threads and close both sockets: the Pi is "lost"."""
        self._stop.set()
        for t in self._threads:
            t.join()
        self._threads = []

    def pause(self, camera: str | None = None) -> None:
        """Stop sending (None: both). Frames due meanwhile are lost: their seq numbers are skipped."""
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = True

    def resume(self, camera: str | None = None) -> None:
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = False

    def drop_next(self, camera: str, n: int = 1) -> None:
        """Skip the next n frames of that camera (a seq gap)."""
        with self._lock:
            self._drop[camera] += int(n)

    @staticmethod
    def _cams(camera: str | None) -> tuple[str, ...]:
        if camera is None:
            return CAMERAS
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}")
        return (camera,)

    # ---------------------------------------------------------------------------------------------- threads

    def _bind(self, kind: str, attr: str, make):
        """Create this thread's socket; record its bound address in ``attr``; tell start() either way."""
        try:
            sock = make()
            setattr(self, attr, sock.endpoint)
            return sock
        except BaseException as e:            # reported by start(), in the caller's thread
            self._bind_error = self._bind_error or e
            return None
        finally:
            self._bound[kind].set()

    def _run_frames(self) -> None:
        pub = self._bind("frames", "frames_endpoint", lambda: Publisher(self.frames_endpoint, self._send_hwm))
        if pub is None:
            return
        rec = self.recording
        t0 = rec.start_ns
        # per camera: the index of the next due frame, counting over all loops (= its seq)
        nxt = {cam: 0 for cam in CAMERAS}
        n = {cam: len(rec.cameras[cam].frames) for cam in CAMERAS}

        def due_ns(cam: str) -> int:
            k, i = divmod(nxt[cam], n[cam])
            return self._start_ns + k * rec.duration_ns + (rec.cameras[cam].frames[i].t_pi_ns - t0)

        try:
            while not self._stop.is_set():
                cam = min(CAMERAS, key=due_ns)
                wait_ns = due_ns(cam) - self._clock()
                if wait_ns > 0:
                    self._stop.wait(min(wait_ns / 1e9, _POLL_S))
                    continue
                seq = nxt[cam]
                nxt[cam] += 1
                with self._lock:
                    skip = self._paused[cam] or self._drop[cam] > 0
                    if not self._paused[cam] and self._drop[cam] > 0:
                        self._drop[cam] -= 1
                if skip:
                    continue
                recorded = rec.cameras[cam].frames[seq % n[cam]]
                frame = dataclasses.replace(recorded, seq=seq, t_pi_ns=self.pi_clock_ns(self._clock()))
                pub.publish(frame)
                self.sent[cam] += 1
        finally:
            pub.close()

    def _run_time(self) -> None:
        def answer(req: TimeSyncRequest) -> TimeSyncReply:
            return TimeSyncReply(t_pi_ns=self.pi_clock_ns(self._clock()))

        server = self._bind("time", "time_sync_endpoint",
                            lambda: RpcServer(self.time_sync_endpoint, {TimeSyncRequest: answer}))
        if server is None:
            return
        try:
            while not self._stop.is_set():
                server.serve_once(_POLL_S)
        finally:
            server.close()
