"""FakePiSender: the stand-in Pi for unit tests and --mock (STRUCTURE.md §5.16). Never used on the real arm.

It plays back a short recorded clip only to produce realistic C0 traffic, and speaks exactly C0 (PI_PAYLOAD.md):
thread "fake-pi-frames" binds a PUB socket and sends one multipart packet per main frame, paired by PacketPairer, at
the recording's rhythm (30 FPS); thread "fake-pi-time" binds a REP socket and answers time_sync with
``pi_clock_ns(now)``. The clip loops with every counter and the clock still counting up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from contracts import EncodedFrame, Packet, PacketPublisher, RpcServer, TimeSyncReply, TimeSyncRequest

from process_frames.packet_pairer import PacketPairer
from process_frames.pi_recording import PiRecording, with_seq_and_time

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
        self._drop_frames = {cam: 0 for cam in CAMERAS}
        self._drop_packets = 0
        self.sent = {"packets": 0, "ultrawide": 0}       # packets sent, ultrawide frames sent in them

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
        """The Pi stops encoding that camera's frames (None: both); their frame_seq numbers are skipped. Main
        paused: no packets at all (packet_seq still counts). Ultrawide paused: packets without ultrawide frames."""
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = True

    def resume(self, camera: str | None = None) -> None:
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = False

    def drop_frames(self, camera: str, n: int = 1) -> None:
        """The Pi drops that camera's next n frames before encoding them (a frame_seq gap; main: no packet for
        them, a packet_seq gap)."""
        with self._lock:
            self._drop_frames[self._cams(camera)[0]] += int(n)

    def drop_packets(self, n: int = 1) -> None:
        """The next n packets are lost on the way, as when ZeroMQ's queue is full: packet_seq and both cameras'
        frame_seq show gaps."""
        with self._lock:
            self._drop_packets += int(n)

    @staticmethod
    def _cams(camera: str | None) -> tuple[str, ...]:
        if camera is None:
            return CAMERAS
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}")
        return (camera,)

    def _skip_frame(self, camera: str) -> bool:
        with self._lock:
            if self._paused[camera]:
                return True
            if self._drop_frames[camera] > 0:
                self._drop_frames[camera] -= 1
                return True
            return False

    def _lose_packet(self) -> bool:
        with self._lock:
            if self._drop_packets > 0:
                self._drop_packets -= 1
                return True
            return False

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
        pub = self._bind("frames", "frames_endpoint", lambda: PacketPublisher(self.frames_endpoint, self._send_hwm))
        if pub is None:
            return
        t0 = self.recording.start_ns
        pairer: PacketPairer[EncodedFrame] = PacketPairer()
        try:
            for ev in self.recording.events():
                while not self._stop.is_set():                    # wait until the Pi would handle it
                    wait_ns = self._start_ns + (ev.due_ns - t0) - self._clock()
                    if wait_ns <= 0:
                        break
                    self._stop.wait(min(wait_ns / 1e9, _POLL_S))
                if self._stop.is_set():
                    return
                if self._skip_frame(ev.camera):
                    continue
                frame = with_seq_and_time(ev.frame, ev.frame_seq,
                                          self.pi_clock_ns(self._start_ns + (ev.capture_ns - t0)))
                if ev.camera == "ultrawide":
                    pairer.add_ultrawide(frame.info.frame_seq, frame.info.t_pi_ns, frame)
                    continue
                uw, pair = pairer.take(frame.info.t_pi_ns)
                packet = Packet.build(ev.frame_seq, frame, uw, pair)   # packet_seq counts main frames captured
                if self._lose_packet():
                    continue
                pub.publish(packet)
                self.sent["packets"] += 1
                self.sent["ultrawide"] += len(uw)
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
