"""contracts: the only code shared between the deployment processes (PLAN rule 3).

Message definitions C0-C4 with their validators (C0 also as the Pi's pi_stream/v1 wire format), the msgpack wire
format, schema versions, example messages and three small ZeroMQ helpers. See contracts/STRUCTURE.md.
"""
from __future__ import annotations

from contracts.c0_frames import EncodedFrame, FrameInfo, Packet, PacketHeader, TimeSyncReply, TimeSyncRequest
from contracts.c1_frames import (CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest, HealthReply,
                                 HealthReport)
from contracts.c2_state import ArmState, GateCounters
from contracts.c3_commands import (GetState, Heartbeat, Home, SetWaypoints, SetWaypointsReply, Start, Stop,
                                   WaypointIssue)
from contracts.c4_policy import ActionChunk, Observation, PolicyConfig, PolicyConfigRequest, Prompt, Reset
from contracts.codec import decode, encode, pack_array, pack_obj, unpack_array, unpack_obj
from contracts.common import Ack, ErrorReply
from contracts.errors import ContractError, RemoteError, TransportCancelled, TransportTimeout
from contracts.examples import Example, build_examples, examples_dir, load_index
from contracts.limits import ContractLimits
from contracts.message import Message, Record
from contracts.pi_stream import PiFrameStamp, PiStreamHeader, PiStreamPacket, jpeg_size
from contracts.pubsub import PacketPublisher, PacketSubscriber, Publisher, Subscriber
from contracts.registry import register
from contracts.rpc import RpcClient, RpcServer
from contracts.schema import SchemaName
from contracts.sizes import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS

__all__ = [
    "ACTION_STEPS", "IMAGE_SIZE", "OBS_STEPS", "STEP_MS",
    "Ack", "ErrorReply",
    "PacketHeader", "FrameInfo", "Packet", "EncodedFrame", "TimeSyncRequest", "TimeSyncReply",
    "PiStreamPacket", "PiStreamHeader", "PiFrameStamp", "jpeg_size",
    "GetFramesRequest", "FramesReply", "GetHealthRequest", "HealthReply", "HealthReport", "CameraHealthReport",
    "ArmState", "GateCounters",
    "SetWaypoints", "SetWaypointsReply", "WaypointIssue", "Heartbeat", "Start", "Stop", "Home", "GetState",
    "PolicyConfigRequest", "PolicyConfig", "Reset", "Observation", "ActionChunk", "Prompt",
    "encode", "decode", "pack_obj", "unpack_obj", "pack_array", "unpack_array",
    "ContractError", "TransportTimeout", "TransportCancelled", "RemoteError",
    "ContractLimits", "Message", "Record", "SchemaName", "register",
    "RpcClient", "RpcServer", "Publisher", "Subscriber", "PacketPublisher", "PacketSubscriber",
    "Example", "build_examples", "examples_dir", "load_index",
]
