"""Receiver: the receiving side, thread "receiver" (STRUCTURE.md §5.11, §6.1). One C0 message at a time."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

from contracts import ContractError, Frame

from process_frames._dataproc.config import RigConfig
from process_frames.camera_health import CameraHealth
from process_frames.clock_map import ClockMap
from process_frames.errors import DecodeError, FatalError
from process_frames.frame_buffer import BufferedFrame, FrameBuffer
from process_frames.frame_decoder import FrameDecoder
from process_frames.frame_source import FrameSource

log = logging.getLogger("process_frames.receiver")

CAMERAS = ("main", "ultrawide")


@dataclass
class ReceiverCounters:
    """Per camera, since start (latency: since the last take_latency())."""
    received: int = 0
    buffered: int = 0
    dropped: int = 0          # frames lost to seq gaps
    unusable: int = 0         # decoded while waiting for a keyframe: never buffered
    unsynced: int = 0         # no fresh clock: never buffered
    out_of_order: int = 0
    ignored: int = 0          # frames of a camera latched down by a stream restart
    latency_ms: list[float] = field(default_factory=list)    # arrival - t_pc of buffered frames

    def take_latency(self) -> tuple[float, float] | None:
        """(mean, max) of the latency since the last call, or None; then starts a new period."""
        values, self.latency_ms = self.latency_ms, []
        if not values:
            return None
        return sum(values) / len(values), max(values)


@dataclass
class _Last:
    seq: int
    t_pi_ns: int


class Receiver:
    def __init__(self, source: FrameSource, decoders: Mapping[str, FrameDecoder], rig: RigConfig,
                 clock_map: ClockMap, buffers: Mapping[str, FrameBuffer], health: Mapping[str, CameraHealth],
                 default_exposure_ns: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._source = source
        self._decoders = decoders
        self._rig = rig
        self._clock_map = clock_map
        self._buffers = buffers
        self._health = health
        self._clock = clock
        self._last: dict[str, _Last] = {}
        self._restarted: set[str] = set()
        self._default_exposure = {}
        for cam in CAMERAS:
            period = round(1e9 / rig.cameras[cam].fps)
            self._default_exposure[cam] = min(int(default_exposure_ns), period // 2)
        self.counters: Mapping[str, ReceiverCounters] = {cam: ReceiverCounters() for cam in CAMERAS}

    def default_exposure_ns(self, camera: str) -> int:
        """The exposure used when C0's exposure_ns is 0: min(default_exposure_ns, period // 2)."""
        return self._default_exposure[camera]

    def receive_one(self, timeout_s: float) -> bool:
        """Handle at most one C0 message; False if none came. FatalError for this process's own problems."""
        try:
            frame = self._source.next_frame(timeout_s)
        except ContractError as e:
            raise FatalError(f"C0 frame breaks its contract: {e}", "check the Pi sender version") from e
        if frame is None:
            return False
        arrival_ns = self._clock()
        cam = frame.camera
        c = self.counters[cam]
        c.received += 1
        if cam in self._restarted:                                    # 2. latched down
            c.ignored += 1
            return True
        self._check_stream(frame)                                     # 3. rig file and config
        if not self._check_seq(frame, c):                             # 4. seq
            return True
        decoder = self._decoders[cam]                                 # 5. decode
        try:
            rgb = decoder.decode(frame)
        except DecodeError as e:
            raise FatalError(f"{cam} {decoder.ENCODING.upper()} decode failed at seq {frame.seq} ({e.detail})",
                             "check the Pi encoder settings (no B-frames, SPS/PPS before every keyframe)") from e
        if rgb is None:
            c.unusable += 1
            return True
        if rgb.shape != (frame.height, frame.width, 3):
            raise FatalError(f"{cam} decoded picture is {rgb.shape[1]}x{rgb.shape[0]}, the C0 message says "
                             f"{frame.width}x{frame.height}", "check the Pi encoder settings")
        if not self._clock_map.is_fresh(arrival_ns):                  # 6. time
            c.unsynced += 1
            return True
        exposure = int(frame.exposure_ns) or self._default_exposure[cam]
        t_pi_mid = int(self._rig.camera_time_ns(cam, int(frame.t_pi_ns), exposure))
        t_pc = self._clock_map.to_pc_ns(t_pi_mid, at_pc_ns=arrival_ns)
        buffer = self._buffers[cam]
        newest = buffer.newest()                                      # 7. order
        if newest is not None and t_pc <= newest.t_pc_ns:
            c.out_of_order += 1
            log.warning("%s: frame seq %d at %d is not after the newest buffered frame (%d): dropped", cam,
                        frame.seq, t_pc, newest.t_pc_ns)
            return True
        rgb.flags.writeable = False                                   # 8. buffer
        buffer.push(BufferedFrame(camera=cam, seq=int(frame.seq), t_pc_ns=t_pc, t_pi_mid_ns=t_pi_mid,
                                  arrival_ns=arrival_ns, rgb=rgb))
        self._health[cam].record_frame(t_pc)
        c.buffered += 1
        c.latency_ms.append((arrival_ns - t_pc) / 1e6)
        return True

    def _check_stream(self, frame: Frame) -> None:
        cam = frame.camera
        rc = self._rig.cameras[cam]
        if frame.sensor != rc.sensor:
            raise FatalError(f"{cam} stream sensor is {frame.sensor}, rig file says {rc.sensor}",
                             "check which camera is on which Pi port, or process_frames.rig_file")
        w, h = rc.resolution
        if (frame.width, frame.height) != (w, h):
            raise FatalError(f"{cam} frame size {frame.width}x{frame.height}, rig file says {w}x{h}",
                             "check the Pi stream settings or process_frames.rig_file")
        expected = self._decoders[cam].ENCODING
        if frame.encoding != expected:
            raise FatalError(f"{cam} stream is {frame.encoding}, process_frames.decoder is {expected}",
                             "set process_frames.decoder, or fix the Pi sender")

    def _check_seq(self, frame: Frame, c: ReceiverCounters) -> bool:
        """False if the camera was just latched down by a stream restart."""
        cam = frame.camera
        last = self._last.get(cam)
        if last is not None:
            if frame.seq <= last.seq or frame.t_pi_ns <= last.t_pi_ns:
                self._restarted.add(cam)
                self._health[cam].mark_down(f"Pi stream restarted: seq {frame.seq} after {last.seq}")
                c.ignored += 1
                return False
            gap = int(frame.seq) - last.seq - 1
            if gap > 0:
                c.dropped += gap
                self._health[cam].record_dropped(gap)
                self._decoders[cam].notify_gap()
        self._last[cam] = _Last(int(frame.seq), int(frame.t_pi_ns))
        return True

    def run(self, stop: threading.Event, timeout_s: float) -> None:
        """receive_one until ``stop``; closes the source at the end (it is owned by this thread)."""
        try:
            while not stop.is_set():
                self.receive_one(timeout_s)
        finally:
            self._source.close()
