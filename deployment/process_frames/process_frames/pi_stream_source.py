"""PiStreamPacketSource: the Pi's pi_stream/v1 messages as C0 packets (STRUCTURE.md §5.4a, PI_PAYLOAD.md §0).

A SUB socket connected to the Pi's PUB (receive only: nothing goes back to the Pi). Each 3-part message (JSON header,
main JPEG, ultrawide JPEG) is checked (contracts.pi_stream) and converted to the internal C0 form the receiver
already handles, so decoding, pairing, buffering, picking and the C1 images are unchanged:

* ``packet_seq`` = ``packet_no``; each frame map: ``frame_seq`` = ``frame_no``, ``t_pi_ns`` = ``t_sensor_ns``,
  ``exposure_ns`` = ``exposure_us`` x 1000, width and height from the JPEG header, ``encoding`` jpeg, keyframe true.
* The ultrawide frame goes into the packet only when it is new, i.e. its ``frame_no`` differs from the last one
  passed on. A repeat (the Pi's ``uw_reused``) is not decoded again: the packet carries no ultrawide frame and its
  ``pair_seq`` names the earlier one, which the receiver still holds. ``pair_seq`` is always the ultrawide
  ``frame_no``.
* The clock: every message's ``t_send_ns`` against its arrival time feeds the OneWayClockMap (no time_sync).
"""

from __future__ import annotations

import math
import time
from typing import Callable

import zmq

from contracts import EncodedFrame, FrameInfo, Packet, PiStreamPacket, jpeg_size
from contracts.pi_stream import MAIN_SENSOR, ULTRAWIDE_SENSOR, PiFrameStamp

from process_frames.clock_map import OneWayClockMap
from process_frames.packet_source import PacketSource


def frame_info(sensor: str, stamp: PiFrameStamp, jpeg: bytes) -> FrameInfo:
    w, h = jpeg_size(jpeg)
    return FrameInfo(sensor=sensor, frame_seq=stamp.frame_no, t_pi_ns=stamp.t_sensor_ns,
                     exposure_ns=stamp.exposure_us * 1000, width=w, height=h, encoding="jpeg", keyframe=True)


class PiStreamConverter:
    """pi_stream/v1 -> C0 packets, remembering the last ultrawide frame passed on (pure, no socket)."""

    def __init__(self) -> None:
        self.last_ultrawide: int | None = None
        self.repeats = 0                      # ultrawide frames not passed on again

    def convert(self, msg: PiStreamPacket) -> Packet:
        h = msg.header
        main = EncodedFrame(frame_info(MAIN_SENSOR, h.main, msg.main_jpeg), msg.main_jpeg)
        uw_no = h.ultrawide.frame_no
        if uw_no == self.last_ultrawide:
            ultrawide: tuple[EncodedFrame, ...] = ()
            self.repeats += 1
        else:
            ultrawide = (EncodedFrame(frame_info(ULTRAWIDE_SENSOR, h.ultrawide, msg.ultrawide_jpeg),
                                      msg.ultrawide_jpeg),)
            self.last_ultrawide = uw_no
        return Packet.build(h.packet_no, main, ultrawide, pair_seq=uw_no)


class PiStreamPacketSource(PacketSource):
    def __init__(self, endpoint: str, recv_hwm: int, clock_map: OneWayClockMap,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.endpoint = endpoint
        self._recv_hwm = recv_hwm
        self._clock_map = clock_map
        self._clock = clock
        self.converter = PiStreamConverter()
        self._socket: zmq.Socket | None = None          # opened on first use, in the thread that uses it

    def _open(self) -> zmq.Socket:
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, self._recv_hwm)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(self.endpoint)
        return s

    def next_packet(self, timeout_s: float) -> Packet | None:
        """The next packet as C0, or None if none came within ``timeout_s``. ContractError if it breaks the format."""
        if self._socket is None:
            self._socket = self._open()
        if not self._socket.poll(max(0, int(math.ceil(timeout_s * 1000.0))), zmq.POLLIN):
            return None
        parts = self._socket.recv_multipart()
        arrival_ns = self._clock()
        msg = PiStreamPacket.from_parts(parts)
        packet = self.converter.convert(msg)            # may raise ContractError (JPEG header, C0 checks)
        self._clock_map.add_one_way(msg.header.t_send_ns, arrival_ns)
        return packet

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
