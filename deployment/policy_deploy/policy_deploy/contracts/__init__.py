"""policy_deploy.contracts: the laptop-only contracts (C0, C1, C4) and the laptop conventions the processes share
(paths, station, config merge, common process arguments, the clock fit). STRUCTURE.md section 3.

Built on deploy_contracts. Importing this package registers c0.packet/v1 and the C1 and C4 schemas into the shared
registry (deploy_contracts.SHARED_SCHEMAS does not change). It imports no other policy_deploy subpackage.
"""
from __future__ import annotations

from policy_deploy.contracts import paths
from policy_deploy.contracts.c0_packet import NO_PAIR, EncodedFrame, FrameInfo, Packet, PacketHeader
from policy_deploy.contracts.c1_frames import (CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest,
                                               HealthReply, HealthReport)
from policy_deploy.contracts.c4_limits import C4Limits
from policy_deploy.contracts.c4_policy import (IDENTITY_ROT6D, OBSERVATION_INPUTS, ActionChunk, Observation,
                                               PolicyConfig, PolicyConfigRequest, Prompt, Reset)
from policy_deploy.contracts.clock_fit import ClockFit, ClockMap, ClockSample, ClockStats, NotSynced, OneWayClockMap
from policy_deploy.contracts.config_merge import load_config, resolve_path
from policy_deploy.contracts.pi_stream import PiFrameStamp, PiStreamHeader, PiStreamPacket, jpeg_size
from policy_deploy.contracts.process_args import ProcessArgs, fatal_line, print_fatal
from policy_deploy.contracts.sizes import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS
from policy_deploy.contracts.station import Station, StationError

__all__ = [
    "paths", "Station", "StationError", "load_config", "resolve_path", "ProcessArgs", "fatal_line", "print_fatal",
    "ACTION_STEPS", "IMAGE_SIZE", "OBS_STEPS", "STEP_MS",
    "PacketHeader", "FrameInfo", "Packet", "EncodedFrame", "NO_PAIR",
    "PiStreamPacket", "PiStreamHeader", "PiFrameStamp", "jpeg_size",
    "GetFramesRequest", "FramesReply", "GetHealthRequest", "HealthReply", "HealthReport", "CameraHealthReport",
    "PolicyConfigRequest", "PolicyConfig", "Reset", "Observation", "ActionChunk", "Prompt", "OBSERVATION_INPUTS",
    "IDENTITY_ROT6D", "C4Limits",
    "ClockSample", "ClockStats", "ClockFit", "ClockMap", "OneWayClockMap", "NotSynced",
]
