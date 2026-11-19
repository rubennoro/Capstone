# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/receiver.py @ 2ffe641
# changes: images are never decoded (no FrameDecoder, no PacketBuffer, STRUCTURE.md 5.1): _take_frame keeps the
# frame-gap counting and checks the JPEG header's size against the frame map (jpeg_size) instead of the decoded
# picture; the encoding check names JPEG; UltrawideStore holds EncodedFrames; the out-of-order check compares with the
# last accepted packet time; each accepted packet becomes one RecordItem handed to the sinks (preview, recorder) with
# the source's raw one-way sample (t_send_ns, arrival); arrival_ns is the source's last_arrival_ns (the very time
# its sample fed the clock map; the receiver's own clock only for a source without one), and it is what the
# freshness check, the laptop times and the latency use (pd: the receiver's clock read after next_packet); counters: "buffered" is "accepted", "unusable" dropped (a JPEG
# is always usable); pair_gap kept for refused pairs too (RecordItem.pair_gap_ms); imports. The checks, gap counters,
# restart detection, PairRule, OneWayClockMap times and health semantics are unchanged. The source's own header
# follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/receiver.py @ 91b178b
# changes: steps 6-11 of STRUCTURE.md 4.3: fix 3 (the ultrawide store evicts by time on every packet), fix 2
# (ultrawide health from new frames at their own mapped capture time), fix 1 (PairRule: the pair window around the
# main frame and the reuse limit); counters add uw_reused, pair_too_old, pair_too_new, pair_reused; times on the
# laptop clock (t_ns); the clock map is a OneWayClockMap (any policy_deploy.contracts.ClockMap works). Fix wave: the
# ultrawide health counts a frame only once it is served as an accepted pair (as the old receiver did), so a pair that
# is always refused makes the ultrawide stale, then down; the reuse run counts packet_seq.
"""Receiver: the receiving side, thread "receiver" (STRUCTURE.md 5.1). One C0 packet at a time.

Per packet: the rig-file checks (sensor, size, JPEG), restart detection (a counter or a time going back latches both
cameras down), packet and frame gap counting, the JPEG header's size, the ultrawide store, the Pi clock (no fresh
clock: counted as unsynced, nothing else), the mid-exposure times on the laptop clock, the pair rule, the order check,
then health (main: every accepted packet; ultrawide: a frame once, when it is first an accepted pair, at its own
time) and one RecordItem handed to every sink (the preview, and the recorder, which ignores it unless recording).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from deploy_contracts import ContractError

from teleop_collect.contracts import ClockMap, EncodedFrame, FrameInfo, Packet, jpeg_size
from teleop_collect.frames._dataproc.config import RigConfig
from teleop_collect.frames.camera_health import CameraHealth
from teleop_collect.frames.errors import FatalError
from teleop_collect.frames.packet_source import PacketSource
from teleop_collect.frames.pair_rule import OK, REUSED, TOO_NEW, TOO_OLD, PairRule
from teleop_collect.frames.recorder import RecordItem
from teleop_collect.frames.ultrawide_store import UltrawideStore

log = logging.getLogger("frames.receiver")

CAMERAS = ("main", "ultrawide")
ENCODING = "jpeg"


@dataclass
class CameraCounters:
    """Per camera, since start."""
    received: int = 0         # frames received
    usable: int = 0           # main: accepted packets; ultrawide: accepted packets with an accepted pair
    dropped: int = 0          # frames lost to frame_seq gaps
    unpaired: int = 0         # ultrawide only: packets with pair_seq -1
    pair_missing: int = 0     # ultrawide only: a pair_seq not among the frames kept (lost, or too old)


@dataclass
class PacketCounters:
    """Packets since start (latency and pair gaps: since the last take_stats())."""
    received: int = 0
    accepted: int = 0         # handed to the sinks (preview, recorder)
    lost: int = 0             # packet_seq gaps
    unsynced: int = 0         # no fresh clock: never accepted
    out_of_order: int = 0
    ignored: int = 0          # after a Pi stream restart (both cameras latched down)
    uw_reused: int = 0        # packets carrying no new ultrawide frame, their pair an earlier one (the Pi's uw_reused)
    pair_too_old: int = 0     # pair refused: more than pair_max_age_ms older than the main frame
    pair_too_new: int = 0     # pair refused: more than pair_max_lead_ms newer than the main frame
    pair_reused: int = 0      # pair refused: the same ultrawide frame for more than pair_max_reuse packets in a row
    latency_ms: list[float] = field(default_factory=list)     # arrival - packet time, accepted packets
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


@dataclass(frozen=True)
class _Pair:
    frame: EncodedFrame | None     # the pair, when found and accepted by the PairRule
    mid_pi_ns: int | None          # its mid-exposure (Pi clock), when found (accepted or not)


class Receiver:
    def __init__(self, source: PacketSource, rig: RigConfig, clock_map: ClockMap,
                 health: Mapping[str, CameraHealth], default_exposure_ns: int, keep_s: float,
                 clock: Callable[[], int] = time.monotonic_ns, pair_rule: PairRule | None = None,
                 sinks: Sequence[Callable[[RecordItem], None]] = ()) -> None:
        """pair_rule: default PairRule(60 ms, 25 ms, 2), the config defaults. sinks: called with each RecordItem,
        in this thread, in order (they must not block)."""
        self._source = source
        self._rig = rig
        self._clock_map = clock_map
        self._health = health
        self._clock = clock
        self._keep_ns = int(keep_s * 1e9)
        self._store = UltrawideStore(keep_s)
        self.pair_rule = pair_rule if pair_rule is not None else PairRule(60.0, 25.0, 2)
        self.sinks: list[Callable[[RecordItem], None]] = list(sinks)
        self._last_packet_seq: int | None = None
        self._last_accepted_t_ns: int | None = None
        self._uw_recorded_seq = -1                      # the newest ultrawide frame_seq counted into its health
        self._last: dict[str, _Last] = {}
        self._restarted = False
        self._default_exposure = {cam: min(int(default_exposure_ns), round(1e9 / rig.cameras[cam].fps) // 2)
                                  for cam in CAMERAS}
        self.counters: Mapping[str, CameraCounters] = {cam: CameraCounters() for cam in CAMERAS}
        self.packets = PacketCounters()
        self.last_item: RecordItem | None = None

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
        arrival_ns = self._source.last_arrival_ns
        if arrival_ns is None:                                         # a source without its own stamp (tests)
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
        main_frame = packet.main
        self._take_frame("main", main_frame)                           # 5. gaps and the JPEG header, in order
        new_uw = packet.ultrawide
        for f in new_uw:
            self._take_frame("ultrawide", f)
            self._store.add(f)
        main = packet.header.main
        pair_seq = int(packet.header.pair_seq)
        uw_reused = not new_uw and pair_seq >= 0
        if uw_reused:
            pc.uw_reused += 1
        self._store.evict_before(int(main.t_pi_ns) - self._keep_ns)    # 6. fix 3: evict by time on every packet
        if not self._clock_map.is_fresh(arrival_ns):                   # 7. clock
            pc.unsynced += 1
            return True
        t_pi_mid = self._mid_pi("main", main.t_pi_ns, main.exposure_ns)
        t_ns = self._clock_map.to_local_ns(t_pi_mid, at_local_ns=arrival_ns)   # 8. packet time (laptop)
        pair = self._judge_pair(pair_seq, int(packet.header.packet_seq), t_pi_mid)   # 10. fix 1
        last = self._last_accepted_t_ns                                # 11. order, main health
        if last is not None and t_ns <= last:
            pc.out_of_order += 1
            log.warning("packet_seq %d at %d is not after the last accepted packet (%d): dropped",
                        packet.header.packet_seq, t_ns, last)
            return True
        self._last_accepted_t_ns = t_ns
        pc.accepted += 1
        pc.latency_ms.append((arrival_ns - t_ns) / 1e6)
        gap_ms = (t_pi_mid - pair.mid_pi_ns) / 1e6 if pair.mid_pi_ns is not None else None
        if pair.frame is not None and gap_ms is not None:
            pc.pair_gap_ms.append(gap_ms)
        self._health["main"].record_frame(t_ns)
        self.counters["main"].usable += 1
        if pair.frame is not None:
            self.counters["ultrawide"].usable += 1
            if pair.frame.info.frame_seq > self._uw_recorded_seq:      # 9. fix 2: once per frame, at its own time
                self._uw_recorded_seq = int(pair.frame.info.frame_seq)
                self._health["ultrawide"].record_frame(
                    self._clock_map.to_local_ns(pair.mid_pi_ns, at_local_ns=arrival_ns))
        uw = new_uw[-1] if new_uw else None                            # 12. the record (pi_stream: at most one)
        uw_mid_pi = self._mid_pi("ultrawide", uw.info.t_pi_ns, uw.info.exposure_ns) if uw is not None else None
        t_send = self._source.last_t_send_ns
        item = RecordItem(
            packet_no=int(packet.header.packet_seq),
            t_send_pi_ns=int(t_send) if t_send is not None else int(main.t_pi_ns),
            arrival_ns=int(arrival_ns), main=main_frame, main_t_mid_pi_ns=t_pi_mid, main_t_mid_ns=int(t_ns),
            ultrawide=uw, ultrawide_t_mid_pi_ns=uw_mid_pi,
            ultrawide_t_mid_ns=(int(self._clock_map.to_local_ns(uw_mid_pi, at_local_ns=arrival_ns))
                                if uw_mid_pi is not None else None),
            ultrawide_frame_no=pair_seq, uw_reused=uw_reused, pair_used=pair.frame is not None, pair_gap_ms=gap_ms)
        self.last_item = item
        for sink in self.sinks:
            sink(item)
        return True

    # ---------------------------------------------------------------------------------------------- steps

    def _judge_pair(self, pair_seq: int, packet_seq: int, main_mid_pi: int) -> _Pair:
        """The packet's pair if found and accepted by the PairRule (with its mid-exposure, Pi clock)."""
        c, pc = self.counters["ultrawide"], self.packets
        if pair_seq < 0:
            c.unpaired += 1
            return _Pair(None, None)
        pair = self._store.get(pair_seq)
        pair_mid = self._mid_pi("ultrawide", pair.info.t_pi_ns, pair.info.exposure_ns) if pair is not None else None
        verdict = self.pair_rule.check(pair_seq, packet_seq, main_mid_pi, pair_mid)
        if verdict == REUSED:
            pc.pair_reused += 1
            return _Pair(None, pair_mid)
        if pair is None:
            c.pair_missing += 1
            return _Pair(None, None)
        if verdict == TOO_NEW:
            pc.pair_too_new += 1
            return _Pair(None, pair_mid)
        if verdict == TOO_OLD:
            pc.pair_too_old += 1
            return _Pair(None, pair_mid)
        assert verdict == OK
        return _Pair(pair, pair_mid)

    def _check_stream(self, packet: Packet) -> None:
        """Every frame map against the rig file and JPEG; FATAL naming the camera."""
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
        if info.encoding != ENCODING:
            raise FatalError(f"{cam} stream is {info.encoding}, frames records {ENCODING}",
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

    def _take_frame(self, cam: str, frame: EncodedFrame) -> None:
        """frame_seq gap check, then the JPEG header's size against the frame map (always, even unsynced). The
        bytes are never decoded."""
        info, c = frame.info, self.counters[cam]
        prev = self._last.get(cam)
        if prev is not None:
            gap = int(info.frame_seq) - prev.frame_seq - 1
            if gap > 0:
                c.dropped += gap
                self._health[cam].record_dropped(gap)
        self._last[cam] = _Last(int(info.frame_seq), int(info.t_pi_ns))
        c.received += 1
        try:
            w, h = jpeg_size(frame.data)
        except ContractError as e:
            raise FatalError(f"{cam} frame_seq {info.frame_seq} is not a readable JPEG ({e})",
                             "check the Pi encoder settings") from e
        if (w, h) != (info.width, info.height):
            raise FatalError(f"{cam} JPEG is {w}x{h} at frame_seq {info.frame_seq}, the C0 header says "
                             f"{info.width}x{info.height}", "check the Pi encoder settings")

    def run(self, stop: threading.Event, timeout_s: float) -> None:
        """receive_one until ``stop``; closes the source at the end (it is owned by this thread)."""
        try:
            while not stop.is_set():
                self.receive_one(timeout_s)
        finally:
            self._source.close()
