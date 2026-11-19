# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/c0_packet.py @ 2ffe641
# changes: none (module path teleop_collect.contracts.c0_packet); the source's own header follows
# adapted from rnoro/deployment:deployment/contracts/contracts/c0_frames.py @ 91b178b
# changes: renamed c0_packet; the internal packet form only (TimeSyncRequest/TimeSyncReply dropped: the Pi has no
# time sync); imports from deploy_contracts; c0.packet/v1 is a laptop-only schema.
"""C0, the internal packet form (c0.packet/v1, laptop only): what the frames receiver works on.

Each Pi message (pi_stream/v1, pi_stream.py) is converted into one Packet (PI_PAYLOAD.md appendix sections 3-4,
STRUCTURE.md 3.1). It never crosses a socket now; to_parts/from_parts stay for the tests. Part 0 is the header
(PacketHeader, msgpack), part 1 the main frame, parts 2 ... 1+N the N ultrawide frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from deploy_contracts import ContractError, Message, Record, decode, encode, register
from deploy_contracts.fields import list_of, nested, scalar

MAIN_SENSOR = "imx519"
ULTRAWIDE_SENSOR = "imx219"
NO_PAIR = -1


@dataclass(frozen=True, eq=False)
class FrameInfo(Record):
    """The frame map: one encoded frame of one camera, as listed in the packet header (PI_PAYLOAD.md section 4)."""

    sensor: str = scalar("str", choices=(MAIN_SENSOR, ULTRAWIDE_SENSOR))
    frame_seq: int = scalar("uint64")                 # per camera, counts captured frames from 0
    t_pi_ns: int = scalar("int64")                    # libcamera SensorTimestamp, Pi CLOCK_BOOTTIME
    exposure_ns: int = scalar("int64", min=0)         # ExposureTime (us) x 1000; 0 if unknown
    width: int = scalar("uint32")
    height: int = scalar("uint32")
    encoding: str = scalar("str", choices=("h264", "jpeg"))
    keyframe: bool = scalar("bool")                   # JPEG: always true

    def check_values(self) -> None:
        if self.encoding == "jpeg" and not self.keyframe:
            self.fail("keyframe", "a jpeg frame must be a keyframe")


@register
@dataclass(frozen=True, eq=False)
class PacketHeader(Message):
    """Part 0 of a packet: one packet per main frame (PI_PAYLOAD.md section 4)."""

    SCHEMA = "c0.packet/v1"
    packet_seq: int = scalar("uint64")                # counts main frames captured since the sender started
    main: FrameInfo = nested(FrameInfo)               # the frame of part 1
    ultrawide: tuple[FrameInfo, ...] = list_of(FrameInfo)    # the frames of parts 2 ... 1+N, in capture order
    pair_seq: int = scalar("int64", min=NO_PAIR)      # frame_seq of the paired ultrawide frame, or -1

    def check_values(self) -> None:
        if self.main.sensor != MAIN_SENSOR:
            self.fail("main.sensor", f"the main frame's sensor must be {MAIN_SENSOR}, got {self.main.sensor}")
        for i, f in enumerate(self.ultrawide):
            if f.sensor != ULTRAWIDE_SENSOR:
                self.fail(f"ultrawide[{i}].sensor",
                          f"an ultrawide frame's sensor must be {ULTRAWIDE_SENSOR}, got {f.sensor}")
            if i > 0 and f.frame_seq <= self.ultrawide[i - 1].frame_seq:
                self.fail(f"ultrawide[{i}].frame_seq",
                          f"ultrawide frames must be in capture order: frame_seq {f.frame_seq} after "
                          f"{self.ultrawide[i - 1].frame_seq}")
            if f.encoding != self.ultrawide[0].encoding:
                self.fail(f"ultrawide[{i}].encoding",
                          f"every frame of a camera has the same encoding: {f.encoding} after "
                          f"{self.ultrawide[0].encoding}")
        if self.ultrawide and self.pair_seq > self.ultrawide[-1].frame_seq:
            self.fail("pair_seq", f"pair_seq {self.pair_seq} is newer than this packet's newest ultrawide frame "
                                  f"({self.ultrawide[-1].frame_seq})")


@dataclass(frozen=True)
class EncodedFrame:
    """One frame map plus its encoded bytes (one H.264 access unit in Annex-B form, or one JPEG)."""

    info: FrameInfo
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.info, FrameInfo):
            raise TypeError(f"EncodedFrame.info must be a FrameInfo, got {type(self.info).__name__}")


@dataclass(frozen=True)
class Packet:
    """One C0 packet: the header plus the raw frame parts. A Packet object that exists is valid."""

    header: PacketHeader
    main_data: bytes
    ultrawide_data: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.header, PacketHeader):
            raise TypeError(f"Packet.header must be a PacketHeader, got {type(self.header).__name__}")
        object.__setattr__(self, "ultrawide_data", tuple(self.ultrawide_data))
        want = 2 + len(self.header.ultrawide)
        got = 2 + len(self.ultrawide_data)
        if got != want:
            raise _part_error("parts", "value", f"{got} parts, the header lists {len(self.header.ultrawide)} "
                                                f"ultrawide frames: expected {want}")
        for i, data in enumerate((self.main_data, *self.ultrawide_data), start=1):
            if not isinstance(data, bytes):
                raise _part_error(f"parts[{i}]", "type", f"expected bytes, got {type(data).__name__}")
            if not data:
                raise _part_error(f"parts[{i}]", "value", "empty frame data")

    @classmethod
    def build(cls, packet_seq: int, main: EncodedFrame, ultrawide: Sequence[EncodedFrame], pair_seq: int) -> Packet:
        """A packet from its frames (the sender's side)."""
        header = PacketHeader(packet_seq=packet_seq, main=main.info, ultrawide=tuple(f.info for f in ultrawide),
                              pair_seq=pair_seq)
        return cls(header, main.data, tuple(f.data for f in ultrawide))

    @property
    def main(self) -> EncodedFrame:
        return EncodedFrame(self.header.main, self.main_data)

    @property
    def ultrawide(self) -> tuple[EncodedFrame, ...]:
        return tuple(EncodedFrame(info, data) for info, data in zip(self.header.ultrawide, self.ultrawide_data))

    def to_parts(self) -> list[bytes]:
        """The multipart message: [header msgpack, main frame, ultrawide frames ...]."""
        return [encode(self.header), self.main_data, *self.ultrawide_data]

    @classmethod
    def from_parts(cls, parts: Sequence[bytes]) -> Packet:
        """Parse and check a received multipart message. ContractError (schema c0.packet/v1) if it breaks C0."""
        parts = list(parts)
        if not parts:
            raise _part_error("parts", "value", "no parts")
        header = decode(parts[0], PacketHeader)
        if not isinstance(header, PacketHeader):      # decode lets ErrorReply through; a packet is never one
            raise ContractError(header.SCHEMA, "", "unexpected_message",
                                f"expected {PacketHeader.SCHEMA}, got {header.SCHEMA}")
        if len(parts) < 2:
            raise _part_error("parts", "value", f"{len(parts)} part, the header lists {len(header.ultrawide)} "
                                                f"ultrawide frames: expected {2 + len(header.ultrawide)}")
        return cls(header, parts[1], tuple(parts[2:]))


def _part_error(field: str, problem: str, detail: str) -> ContractError:
    return ContractError(PacketHeader.SCHEMA, field, problem, detail)
