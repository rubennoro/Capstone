"""Receiver: the receiving side, thread "receiver" (STRUCTURE.md §5.12, §6.1). One C0 packet at a time."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from contracts import ContractError, EncodedFrame, FrameInfo, Packet

from process_frames._dataproc.config import RigConfig
from process_frames.camera_health import CameraHealth
from process_frames.clock_map import ClockMap
from process_frames.errors import DecodeError, FatalError
from process_frames.frame_decoder import FrameDecoder
from process_frames.packet_buffer import BufferedPacket, PacketBuffer
from process_frames.packet_source import PacketSource
from process_frames.ultrawide_store import DecodedUltrawide, UltrawideStore

log = logging.getLogger("process_frames.receiver")

CAMERAS = ("main", "ultrawide")


@dataclass
class CameraCounters:
    """Per camera, since start."""
    received: int = 0         # frames received (and decoded)
    usable: int = 0           # buffered packets with a usable image of this camera
    dropped: int = 0          # frames lost to frame_seq gaps
    unusable: int = 0         # decoded while waiting for a keyframe: never served
    unpaired: int = 0         # ultrawide only: packets with pair_seq -1
    pair_missing: int = 0     # ultrawide only: a pair_seq not among the decoded frames kept (lost, or too old)


@dataclass
class PacketCounters:
    """Packets since start (latency and pair gaps: since the last take_stats())."""
    received: int = 0
    buffered: int = 0
    lost: int = 0             # packet_seq gaps
    unsynced: int = 0         # no fresh clock: never buffered
    out_of_order: int = 0
    ignored: int = 0          # after a Pi stream restart (both cameras latched down)
    latency_ms: list[float] = field(default_factory=list)     # arrival - packet time, buffered packets
    pair_gap_ms: list[float] = field(default_factory=list)    # main capture - pair capture, buffered pairs

    def take_stats(self) -> dict[str, tuple[float, float] | None]:
        """(mean, max) of the latency and the pair gap since the last call, or None; then starts a new period."""
        out = {}
        for name in ("latency_ms", "pair_gap_ms"):
            values = getattr(self, name)
            setattr(self, name, [])
            out[name] = (sum(values) / len(values), max(values)) if values else None
        return out


@dataclass
class _Last:
    frame_seq: int
    t_pi_ns: int


class Receiver:
    def __init__(self, source: PacketSource, decoders: Mapping[str, FrameDecoder], rig: RigConfig,
                 clock_map: ClockMap, buffer: PacketBuffer, health: Mapping[str, CameraHealth],
                 default_exposure_ns: int, keep_s: float, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._source = source
        self._decoders = decoders
        self._rig = rig
        self._clock_map = clock_map
        self._buffer = buffer
        self._health = health
        self._clock = clock
        self._store = UltrawideStore(keep_s)
        self._last_packet_seq: int | None = None
        self._last: dict[str, _Last] = {}
        self._restarted = False
        period = round(1e9 / rig.cameras["main"].fps)
        self._default_exposure = min(int(default_exposure_ns), period // 2)
        self.counters: Mapping[str, CameraCounters] = {cam: CameraCounters() for cam in CAMERAS}
        self.packets = PacketCounters()

    def default_exposure_ns(self) -> int:
        """The main exposure used when C0's exposure_ns is 0: min(default_exposure_ns, period // 2)."""
        return self._default_exposure

    def receive_one(self, timeout_s: float) -> bool:
        """Handle at most one C0 packet; False if none came. FatalError for this process's own problems."""
        try:
            packet = self._source.next_packet(timeout_s)
        except ContractError as e:
            raise FatalError(f"C0 packet breaks its contract: {e}", "check the Pi sender version") from e
        if packet is None:
            return False
        arrival_ns = self._clock()
        pc = self.packets
        pc.received += 1
        if self._restarted:                                            # 2. latched down
            pc.ignored += 1
            return True
        self._check_stream(packet)                                     # 3. rig file and config
        restart = self._restart_reason(packet)                         # 4. counters going back
        if restart:
            self._restarted = True
            for cam in CAMERAS:
                self._health[cam].mark_down(f"Pi stream restarted: {restart}")
            pc.ignored += 1
            return True
        self._count_packet_gap(packet.header.packet_seq)
        main_rgb = self._take_frame("main", packet.main)               # 5. decode every frame, in order
        for f in packet.ultrawide:
            rgb = self._take_frame("ultrawide", f)
            self._store.add(DecodedUltrawide(int(f.info.frame_seq), int(f.info.t_pi_ns), rgb))
        pair_seq = int(packet.header.pair_seq)                         # 6. the pair
        pair = self._find_pair(pair_seq)
        if not self._clock_map.is_fresh(arrival_ns):                   # 7. packet time
            pc.unsynced += 1
            return True
        main = packet.header.main
        exposure = int(main.exposure_ns) or self._default_exposure
        t_pi_mid = int(self._rig.camera_time_ns("main", int(main.t_pi_ns), exposure))
        t_pc = self._clock_map.to_pc_ns(t_pi_mid, at_pc_ns=arrival_ns)
        newest = self._buffer.newest()                                 # 8. order
        if newest is not None and t_pc <= newest.t_pc_ns:
            pc.out_of_order += 1
            log.warning("packet_seq %d at %d is not after the newest buffered packet (%d): dropped",
                        packet.header.packet_seq, t_pc, newest.t_pc_ns)
            return True
        uw_rgb = pair.rgb if pair is not None else None                # 9. buffer, health
        bp = BufferedPacket(packet_seq=int(packet.header.packet_seq), t_pc_ns=t_pc, t_pi_mid_ns=t_pi_mid,
                            arrival_ns=arrival_ns, main_seq=int(main.frame_seq), main_t_pi_ns=int(main.t_pi_ns),
                            main_rgb=main_rgb, pair_seq=pair_seq,
                            pair_t_pi_ns=pair.t_pi_ns if pair is not None else None, ultrawide_rgb=uw_rgb)
        self._buffer.push(bp)
        pc.buffered += 1
        pc.latency_ms.append((arrival_ns - t_pc) / 1e6)
        if bp.pair_gap_ns is not None:
            pc.pair_gap_ms.append(bp.pair_gap_ns / 1e6)
        for cam, rgb in (("main", main_rgb), ("ultrawide", uw_rgb)):
            if rgb is not None:
                self._health[cam].record_frame(t_pc)
                self.counters[cam].usable += 1
        return True

    # ---------------------------------------------------------------------------------------------- steps

    def _check_stream(self, packet: Packet) -> None:
        """Every frame map against the rig file and the configured decoder; FATAL naming the camera."""
        for cam, info in (("main", packet.header.main), *(("ultrawide", f) for f in packet.header.ultrawide)):
            self._check_frame(cam, info)

    def _check_frame(self, cam: str, info: FrameInfo) -> None:
        rc = self._rig.cameras[cam]
        if info.sensor != rc.sensor:
            raise FatalError(f"{cam} stream sensor is {info.sensor}, rig file says {rc.sensor}",
                             "check which camera is on which Pi port, or process_frames.rig_file")
        w, h = rc.resolution
        if (info.width, info.height) != (w, h):
            raise FatalError(f"{cam} frame size {info.width}x{info.height}, rig file says {w}x{h}",
                             "check the Pi stream settings or process_frames.rig_file")
        expected = self._decoders[cam].ENCODING
        if info.encoding != expected:
            raise FatalError(f"{cam} stream is {info.encoding}, process_frames.decoder is {expected}",
                             "set process_frames.decoder, or fix the Pi sender")

    def _restart_reason(self, packet: Packet) -> str:
        """Why this packet means the Pi sender restarted (a counter or a time went back), or ""."""
        last_p = self._last_packet_seq
        if last_p is not None and packet.header.packet_seq <= last_p:
            return f"packet_seq {packet.header.packet_seq} after {last_p}"
        frames = [("main", packet.header.main), *(("ultrawide", f) for f in packet.header.ultrawide)]
        last = dict(self._last)
        for cam, info in frames:
            prev = last.get(cam)
            if prev is not None and info.frame_seq <= prev.frame_seq:
                return f"{cam} frame_seq {info.frame_seq} after {prev.frame_seq}"
            if prev is not None and info.t_pi_ns <= prev.t_pi_ns:
                return f"{cam} t_pi_ns {info.t_pi_ns} after {prev.t_pi_ns}"
            last[cam] = _Last(int(info.frame_seq), int(info.t_pi_ns))
        return ""

    def _count_packet_gap(self, packet_seq: int) -> None:
        if self._last_packet_seq is not None:
            self.packets.lost += int(packet_seq) - self._last_packet_seq - 1
        self._last_packet_seq = int(packet_seq)

    def _take_frame(self, cam: str, frame: EncodedFrame) -> np.ndarray | None:
        """frame_seq gap check, then decode (always, even unsynced, so the H.264 chain stays intact). None:
        unusable. The image comes back read-only."""
        info, c = frame.info, self.counters[cam]
        prev = self._last.get(cam)
        if prev is not None:
            gap = int(info.frame_seq) - prev.frame_seq - 1
            if gap > 0:
                c.dropped += gap
                self._health[cam].record_dropped(gap)
                self._decoders[cam].notify_gap()
        self._last[cam] = _Last(int(info.frame_seq), int(info.t_pi_ns))
        c.received += 1
        decoder = self._decoders[cam]
        try:
            rgb = decoder.decode(frame)
        except DecodeError as e:
            raise FatalError(f"{cam} {decoder.ENCODING.upper()} decode failed at frame_seq {info.frame_seq} "
                             f"({e.detail})",
                             "check the Pi encoder settings (no B-frames, SPS/PPS before every keyframe)") from e
        if rgb is None:
            c.unusable += 1
            return None
        if rgb.shape != (info.height, info.width, 3):
            raise FatalError(f"{cam} decoded picture is {rgb.shape[1]}x{rgb.shape[0]}, the C0 header says "
                             f"{info.width}x{info.height}", "check the Pi encoder settings")
        rgb.flags.writeable = False
        return rgb

    def _find_pair(self, pair_seq: int) -> DecodedUltrawide | None:
        c = self.counters["ultrawide"]
        if pair_seq < 0:
            c.unpaired += 1
            return None
        pair = self._store.get(pair_seq)
        if pair is None:
            c.pair_missing += 1
        return pair

    def run(self, stop: threading.Event, timeout_s: float) -> None:
        """receive_one until ``stop``; closes the source at the end (it is owned by this thread)."""
        try:
            while not stop.is_set():
                self.receive_one(timeout_s)
        finally:
            self._source.close()
