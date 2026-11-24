# adapted from rnoro/deployment:deployment/process_frames/process_frames/receiver.py @ 91b178b
# changes: steps 6-11 of STRUCTURE.md 4.3: fix 3 (the ultrawide store evicts by time on every packet), fix 2
# (ultrawide health from new frames at their own mapped capture time), fix 1 (PairRule: the pair window around the
# main frame and the reuse limit); counters add uw_reused, pair_too_old, pair_too_new, pair_reused; times on the
# laptop clock (t_ns); the clock map is a OneWayClockMap (any policy_deploy.contracts.ClockMap works). Fix wave: the
# ultrawide health counts a frame only once it is served as an accepted pair (as the old receiver did), so a pair that
# is always refused makes the ultrawide stale, then down; the reuse run counts packet_seq.
"""Receiver: the receiving side, thread "receiver" (STRUCTURE.md 4.1, 4.3). One C0 packet at a time."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from deploy_contracts import ContractError

from policy_deploy.contracts import ClockMap, EncodedFrame, FrameInfo, Packet
from policy_deploy.frames._dataproc.config import RigConfig
from policy_deploy.frames.camera_health import CameraHealth
from policy_deploy.frames.errors import DecodeError, FatalError
from policy_deploy.frames.frame_decoder import FrameDecoder
from policy_deploy.frames.packet_buffer import BufferedPacket, PacketBuffer
from policy_deploy.frames.packet_source import PacketSource
from policy_deploy.frames.pair_rule import OK, REUSED, TOO_NEW, TOO_OLD, PairRule
from policy_deploy.frames.ultrawide_store import DecodedUltrawide, UltrawideStore

log = logging.getLogger("frames.receiver")

CAMERAS = ("main", "ultrawide")


@dataclass
class CameraCounters:
    """Per camera, since start."""
    received: int = 0         # frames received (and decoded)
    usable: int = 0           # main: buffered packets with a usable main image; ultrawide: with an accepted pair
    dropped: int = 0          # frames lost to frame_seq gaps
    unusable: int = 0         # decoded but unusable (a decoder returning None): never served
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
    uw_reused: int = 0        # packets carrying no new ultrawide frame, their pair an earlier one (the Pi's uw_reused)
    pair_too_old: int = 0     # pair refused: more than pair_max_age_ms older than the main frame
    pair_too_new: int = 0     # pair refused: more than pair_max_lead_ms newer than the main frame
    pair_reused: int = 0      # pair refused: the same ultrawide frame for more than pair_max_reuse packets in a row
    latency_ms: list[float] = field(default_factory=list)     # arrival - packet time, buffered packets
    pair_gap_ms: list[float] = field(default_factory=list)    # main mid - pair mid (Pi clock), accepted pairs

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
                 default_exposure_ns: int, keep_s: float, clock: Callable[[], int] = time.monotonic_ns,
                 pair_rule: PairRule | None = None) -> None:
        """pair_rule: default PairRule(60 ms, 25 ms, 2), the config defaults."""
        self._source = source
        self._decoders = decoders
        self._rig = rig
        self._clock_map = clock_map
        self._buffer = buffer
        self._health = health
        self._clock = clock
        self._keep_ns = int(keep_s * 1e9)
        self._store = UltrawideStore(keep_s)
        self.pair_rule = pair_rule if pair_rule is not None else PairRule(60.0, 25.0, 2)
        self._last_packet_seq: int | None = None
        self._uw_recorded_seq = -1                      # the newest ultrawide frame_seq counted into its health
        self._last: dict[str, _Last] = {}
        self._restarted = False
        self._default_exposure = {cam: min(int(default_exposure_ns), round(1e9 / rig.cameras[cam].fps) // 2)
                                  for cam in CAMERAS}
        self.counters: Mapping[str, CameraCounters] = {cam: CameraCounters() for cam in CAMERAS}
        self.packets = PacketCounters()

    def default_exposure_ns(self, camera: str = "main") -> int:
        """The exposure used when C0's exposure_ns is 0: min(default_exposure_ns, that camera's period // 2)."""
        return self._default_exposure[camera]

    def _mid_pi(self, camera: str, t_pi_ns: int, exposure_ns: int) -> int:
        """Mid-exposure on the Pi clock (the rig file's time model)."""
        return int(self._rig.camera_time_ns(camera, int(t_pi_ns), int(exposure_ns) or self._default_exposure[camera]))

    def receive_one(self, timeout_s: float) -> bool:
        """Handle at most one C0 packet; False if none came. FatalError for this process's own problems."""
        try:                                                           # 1. receive (the source fed the clock)
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
        self._check_stream(packet)                                     # 3. rig file
        restart = self._restart_reason(packet)                         # 4. counters going back
        if restart:
            self._restarted = True
            self._uw_recorded_seq = -1
            self.pair_rule.reset()
            for cam in CAMERAS:
                self._health[cam].mark_down(f"Pi stream restarted: {restart}")
            pc.ignored += 1
            return True
        self._count_packet_gap(packet.header.packet_seq)
        main_rgb = self._take_frame("main", packet.main)               # 5. decode, in order
        for f in packet.ultrawide:
            rgb = self._take_frame("ultrawide", f)
            self._store.add(DecodedUltrawide(int(f.info.frame_seq), int(f.info.t_pi_ns), int(f.info.exposure_ns),
                                             rgb))
        main = packet.header.main
        pair_seq = int(packet.header.pair_seq)
        if not packet.ultrawide and pair_seq >= 0:
            pc.uw_reused += 1
        self._store.evict_before(int(main.t_pi_ns) - self._keep_ns)    # 6. fix 3: evict by time on every packet
        if not self._clock_map.is_fresh(arrival_ns):                   # 7. clock
            pc.unsynced += 1
            return True
        t_pi_mid = self._mid_pi("main", main.t_pi_ns, main.exposure_ns)
        t_ns = self._clock_map.to_local_ns(t_pi_mid, at_local_ns=arrival_ns)   # 8. packet time (laptop)
        pair, pair_mid = self._judge_pair(pair_seq, int(packet.header.packet_seq), t_pi_mid)   # 10. fix 1
        newest = self._buffer.newest()                                 # 11. order, buffer, main health
        if newest is not None and t_ns <= newest.t_ns:
            pc.out_of_order += 1
            log.warning("packet_seq %d at %d is not after the newest buffered packet (%d): dropped",
                        packet.header.packet_seq, t_ns, newest.t_ns)
            return True
        uw_rgb = pair.rgb if pair is not None else None
        bp = BufferedPacket(packet_seq=int(packet.header.packet_seq), t_ns=t_ns, t_pi_mid_ns=t_pi_mid,
                            arrival_ns=arrival_ns, main_seq=int(main.frame_seq), main_t_pi_ns=int(main.t_pi_ns),
                            main_rgb=main_rgb, pair_seq=pair_seq,
                            pair_t_pi_ns=pair.t_pi_ns if uw_rgb is not None else None,
                            pair_t_pi_mid_ns=pair_mid if uw_rgb is not None else None, ultrawide_rgb=uw_rgb)
        self._buffer.push(bp)
        pc.buffered += 1
        pc.latency_ms.append((arrival_ns - t_ns) / 1e6)
        if bp.pair_gap_ns is not None:
            pc.pair_gap_ms.append(bp.pair_gap_ns / 1e6)
        if main_rgb is not None:
            self._health["main"].record_frame(t_ns)
            self.counters["main"].usable += 1
        if uw_rgb is not None:
            self.counters["ultrawide"].usable += 1
            if pair.frame_seq > self._uw_recorded_seq:                # 9. fix 2: once per frame, at its own time
                self._uw_recorded_seq = pair.frame_seq
                self._health["ultrawide"].record_frame(self._clock_map.to_local_ns(pair_mid, at_local_ns=arrival_ns))
        return True

    # ---------------------------------------------------------------------------------------------- steps

    def _judge_pair(self, pair_seq: int, packet_seq: int,
                    main_mid_pi: int) -> tuple[DecodedUltrawide | None, int | None]:
        """The packet's pair if found and accepted by the PairRule, with its mid-exposure (Pi clock); else None."""
        c, pc = self.counters["ultrawide"], self.packets
        if pair_seq < 0:
            c.unpaired += 1
            return None, None
        pair = self._store.get(pair_seq)
        pair_mid = self._mid_pi("ultrawide", pair.t_pi_ns, pair.exposure_ns) if pair is not None else None
        verdict = self.pair_rule.check(pair_seq, packet_seq, main_mid_pi, pair_mid)
        if verdict == REUSED:
            pc.pair_reused += 1
            return None, None
        if pair is None:
            c.pair_missing += 1
            return None, None
        if verdict == TOO_NEW:
            pc.pair_too_new += 1
            return None, None
        if verdict == TOO_OLD:
            pc.pair_too_old += 1
            return None, None
        assert verdict == OK
        return pair, pair_mid

    def _check_stream(self, packet: Packet) -> None:
        """Every frame map against the rig file and the decoder; FATAL naming the camera."""
        for cam, info in (("main", packet.header.main), *(("ultrawide", f) for f in packet.header.ultrawide)):
            self._check_frame(cam, info)

    def _check_frame(self, cam: str, info: FrameInfo) -> None:
        rc = self._rig.cameras[cam]
        if info.sensor != rc.sensor:
            raise FatalError(f"{cam} stream sensor is {info.sensor}, rig file says {rc.sensor}",
                             "check which camera is on which Pi port, or frames.rig_file")
        w, h = rc.resolution
        if (info.width, info.height) != (w, h):
            raise FatalError(f"{cam} frame size {info.width}x{info.height}, rig file says {w}x{h}",
                             "check the Pi stream settings or frames.rig_file")
        expected = self._decoders[cam].ENCODING
        if info.encoding != expected:
            raise FatalError(f"{cam} stream is {info.encoding}, the frames decoder reads {expected}",
                             "fix the Pi sender (it must send JPEG, PI_PAYLOAD.md 0.2)")

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
        """frame_seq gap check, then decode (always, even unsynced). None: unusable. The image comes back
        read-only."""
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
                             f"({e.detail})", "check the Pi encoder settings") from e
        if rgb is None:
            c.unusable += 1
            return None
        if rgb.shape != (info.height, info.width, 3):
            raise FatalError(f"{cam} decoded picture is {rgb.shape[1]}x{rgb.shape[0]}, the C0 header says "
                             f"{info.width}x{info.height}", "check the Pi encoder settings")
        rgb.flags.writeable = False
        return rgb

    def run(self, stop: threading.Event, timeout_s: float) -> None:
        """receive_one until ``stop``; closes the source at the end (it is owned by this thread)."""
        try:
            while not stop.is_set():
                self.receive_one(timeout_s)
        finally:
            self._source.close()
