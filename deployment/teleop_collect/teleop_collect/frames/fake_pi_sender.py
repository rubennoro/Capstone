# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/fake_pi_sender.py @ 2ffe641
# changes: imports only
# adapted from rnoro/deployment:deployment/process_frames/process_frames/fake_pi_sender.py @ 91b178b (the pi_v1 path,
# lines 197-240, and the control methods)
# changes: pi_v1 only (no c0 packets, no time_sync thread); pairs through the nearest-frame PacketPairer, a main
# frame's packet sent half an ultrawide period after its capture (PiRecording.events), so the nearest frame is known;
# uw_reused and skew_ns follow the pair; adds repeat_ultrawide(seconds)
# (STRUCTURE.md 4.3 fix 4); pause("ultrawide") now means the same as a repeat (the pairer keeps the last frame). Fix
# wave: the live Pi's numbers (PI_PAYLOAD.md 0.6): skew_ns with the live sign (ultrawide.t_mid - main.t_mid), and each
# packet sent SEND_DELAY_NS (45.8 ms) after its main frame's t_sensor_ns.
"""FakePiSender: the stand-in Pi for unit tests and --mock (STRUCTURE.md 4.2). Never used on the real arm.

It plays back a short recorded clip only to produce realistic Pi traffic in the Pi's own format, pi_v1
(PI_PAYLOAD.md section 0): thread "fake-pi-frames" binds a PUB socket and sends one 3-part message per main frame (JSON
header, main JPEG, ultrawide JPEG). The ultrawide frame is the main frame's pair: the frame captured nearest it
(PacketPairer); when the pair was already sent, it is sent again with ``uw_reused`` true. No ultrawide frame yet: no
message (the packet number still counts). No time_sync socket: the laptop's clock map uses ``t_send_ns``.

Like the live Pi (PI_PAYLOAD.md 0.6): a packet leaves SEND_DELAY_NS after its main frame's ``t_sensor_ns``, and
``skew_ns`` is ``ultrawide.t_mid_ns - main.t_mid_ns`` (the live sign; nothing on the laptop reads it).

At the recording's rhythm (30 FPS). The clip loops with every counter and the clock still counting up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import zmq

from teleop_collect.contracts import EncodedFrame, PiFrameStamp, PiStreamHeader, PiStreamPacket

from teleop_collect.frames.packet_pairer import PacketPairer
from teleop_collect.frames.pi_recording import PiRecording, with_seq_and_time

log = logging.getLogger("frames.fake_pi_sender")

#: How often the thread looks at its stop flag while idle.
_POLL_S = 0.02
#: A packet leaves this long after its main frame's t_sensor_ns (the live Pi: 45.8 ms, PI_PAYLOAD.md 0.6). Longer than
#: the pairing lookahead (half an ultrawide period), so the nearest ultrawide frame is always known by then.
SEND_DELAY_NS = 45_800_000
CAMERAS = ("main", "ultrawide")


class FakePiSender:
    def __init__(self, recording: PiRecording, frames_endpoint: str, *, pi_clock_offset_ns: int, drift_ppm: float,
                 send_hwm: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.recording = recording
        self.frames_endpoint = frames_endpoint          # replaced by the bound address in start()
        self._offset_ns = int(pi_clock_offset_ns)
        self._drift = float(drift_ppm) * 1e-6
        self._send_hwm = send_hwm
        self._clock = clock
        self._start_ns: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bound = threading.Event()
        self._bind_error: BaseException | None = None
        self._lock = threading.Lock()
        self._paused = {cam: False for cam in CAMERAS}
        self._drop_frames = {cam: 0 for cam in CAMERAS}
        self._drop_packets = 0
        self._repeat_until_ns: int | None = None        # laptop time until which no new ultrawide frame is added
        self.sent = {"packets": 0, "ultrawide": 0, "reused": 0}   # packets; new ultrawide frames in them; repeats
        self.wall_offset_ns = time.time_ns() - self._clock()   # clock_offset_ns base (sent, never used)

    # ---------------------------------------------------------------------------------------------- control

    def pi_clock_ns(self, local_ns: int) -> int:
        """``local_ns + offset + drift_ppm * 1e-6 * (local_ns - start)``: the fake Pi clock."""
        start = self._start_ns if self._start_ns is not None else local_ns
        return int(local_ns) + self._offset_ns + int(round(self._drift * (local_ns - start)))

    def start(self) -> None:
        """Start the thread; returns once its socket is bound (a bind error is raised here)."""
        self._start_ns = self._clock()
        self._thread = threading.Thread(target=self._run_frames, name="fake-pi-frames", daemon=True)
        self._thread.start()
        self._bound.wait()
        if self._bind_error is not None:
            self.stop()
            raise self._bind_error

    def stop(self) -> None:
        """Stop the thread and close its socket: the Pi is "lost"."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def pause(self, camera: str | None = None) -> None:
        """The Pi stops encoding that camera's frames (None: both); their frame_seq numbers are skipped. Main
        paused: no packets at all (packet_seq still counts). Ultrawide paused: packets keep the last ultrawide frame
        as their pair (uw_reused), as repeat_ultrawide does."""
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = True

    def resume(self, camera: str | None = None) -> None:
        with self._lock:
            for cam in self._cams(camera):
                self._paused[cam] = False

    def repeat_ultrawide(self, seconds: float) -> None:
        """Fix 4: for that long the Pi adds no new ultrawide frame, so the nearest-frame pairer keeps returning the
        last one (same frame_no, uw_reused true), as a stalled IMX219 would on the live Pi. Its frame numbers are
        skipped, as a stalled camera's are."""
        with self._lock:
            self._repeat_until_ns = self._clock() + int(round(seconds * 1e9))

    def drop_frames(self, camera: str, n: int = 1) -> None:
        """The Pi drops that camera's next n frames before encoding them (a frame_seq gap; main: no packet for
        them, a packet_seq gap)."""
        with self._lock:
            self._drop_frames[self._cams(camera)[0]] += int(n)

    def drop_packets(self, n: int = 1) -> None:
        """The next n packets are lost on the way, as when ZeroMQ's queue is full: packet_no and both cameras'
        frame_no show gaps."""
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
            if camera == "ultrawide" and self._repeat_until_ns is not None:
                if self._clock() < self._repeat_until_ns:
                    return True
                self._repeat_until_ns = None
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

    # ---------------------------------------------------------------------------------------------- thread

    @staticmethod
    def stamp(frame: EncodedFrame) -> PiFrameStamp:
        """The header's frame object for a frame (t_mid = t_sensor - exposure / 2, the Pi's own rule)."""
        exposure_us = frame.info.exposure_ns // 1000
        return PiFrameStamp(frame_no=frame.info.frame_seq, t_sensor_ns=frame.info.t_pi_ns, exposure_us=exposure_us,
                            t_mid_ns=frame.info.t_pi_ns - exposure_us * 1000 // 2)

    def _run_frames(self) -> None:
        try:
            pub = _RawPublisher(self.frames_endpoint, self._send_hwm)
            self.frames_endpoint = pub.endpoint
        except BaseException as e:            # reported by start(), in the caller's thread
            self._bind_error = e
            return
        finally:
            self._bound.set()
        t0 = self.recording.start_ns
        pairer: PacketPairer[EncodedFrame] = PacketPairer()
        recent: dict[int, EncodedFrame] = {}             # ultrawide frames by frame_seq, the last ~2 s
        last_sent_uw: int | None = None
        try:
            for ev in self.recording.events():
                due = max(ev.due_ns, ev.capture_ns + SEND_DELAY_NS) if ev.camera == "main" else ev.due_ns
                while not self._stop.is_set():
                    wait_ns = self._start_ns + (due - t0) - self._clock()
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
                    recent[frame.info.frame_seq] = frame
                    for old in [k for k in recent if k < frame.info.frame_seq - 60]:
                        del recent[old]
                    continue
                _, pair = pairer.take(frame.info.t_pi_ns)
                if pair < 0 or pair not in recent or self._lose_packet():
                    continue                              # nothing to pair with yet / lost: a packet_no gap
                uw = recent[pair]
                main_stamp, uw_stamp = self.stamp(frame), self.stamp(uw)
                reused = pair == last_sent_uw
                t_send = self.pi_clock_ns(self._clock())
                header = PiStreamHeader(version=1, packet_no=ev.frame_seq, t_send_ns=t_send,
                                        clock_offset_ns=self.wall_offset_ns - self._offset_ns,
                                        main=main_stamp, ultrawide=uw_stamp,
                                        skew_ns=uw_stamp.t_mid_ns - main_stamp.t_mid_ns, uw_reused=reused)
                pub.publish(PiStreamPacket(header, frame.data, uw.data).to_parts())
                last_sent_uw = pair
                self.sent["packets"] += 1
                self.sent["reused" if reused else "ultrawide"] += 1
        finally:
            pub.close()


class _RawPublisher:
    """PUB socket for pi_v1 messages (multipart, never blocks: dropped whole at send_hwm, as on the Pi)."""

    def __init__(self, endpoint: str, send_hwm: int) -> None:
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, send_hwm)
        self._socket.bind(endpoint)
        self.endpoint: str = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def publish(self, parts: list[bytes]) -> None:
        try:
            self._socket.send_multipart(parts, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
