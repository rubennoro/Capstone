# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/__init__.py @ 2ffe641
# changes: teleop_collect's contracts: C0 (pi_stream, c0_packet), the clock fit, station, config merge, process args
# and paths as before; the health Records (health.py) instead of C1; R1, U1, U2 and the shared names (keys.py)
# instead of C4; no sizes; + setup_logging.
"""teleop_collect.contracts: the laptop-only contracts (C0 from the Pi, R1, U1, U2) and the laptop conventions the
processes share (paths, station, config merge, common process arguments, the clock fit). STRUCTURE.md section 4.

Built on deploy_contracts. Importing this package registers c0.packet/v1 and every R1, U1 and U2 schema into the
shared registry (deploy_contracts.SHARED_SCHEMAS does not change). It imports no other teleop_collect subpackage.
"""
from __future__ import annotations

from teleop_collect.contracts import paths
from teleop_collect.contracts.c0_packet import NO_PAIR, EncodedFrame, FrameInfo, Packet, PacketHeader
from teleop_collect.contracts.clock_fit import (ClockFit, ClockMap, ClockSample, ClockStats, NotSynced,
                                                OneWayClockMap)
from teleop_collect.contracts.config_merge import load_config, resolve_path
from teleop_collect.contracts.health import CAMERAS, CameraHealthReport, HealthReport
from teleop_collect.contracts.keys import (EPISODE_NAME_RE, GRIPPER_COMMANDS, MOTION_KEYS, OUTCOMES, SAVED_OUTCOMES,
                                           TASK_NAME_RE, is_episode_name, is_task_name)
from teleop_collect.contracts.pi_stream import PiFrameStamp, PiStreamHeader, PiStreamPacket, jpeg_size
from teleop_collect.contracts.process_args import ProcessArgs, fatal_line, print_fatal, setup_logging
from teleop_collect.contracts.r1_recorder import (CameraSummary, FramesHealth, GetFramesHealth, RecordingSummary,
                                                  StartRecording, StopRecording)
from teleop_collect.contracts.station import Station, StationError
from teleop_collect.contracts.u1_teleop import (MESSAGE_LEVELS, PHASES, STATUS_MAX_MESSAGES, CreateTask,
                                                DeleteEpisode, EpisodeInfo, GetStatus, GripperCommand, HomeRequest,
                                                KeysUpdate, SelectTask, StartEpisode, StopArm, StopEpisode,
                                                TeleopStatus, UiMessage)
from teleop_collect.contracts.u2_preview import NO_FRAME, GetPreview, Preview

__all__ = [
    "paths", "Station", "StationError", "load_config", "resolve_path",
    "ProcessArgs", "fatal_line", "print_fatal", "setup_logging",
    "PacketHeader", "FrameInfo", "Packet", "EncodedFrame", "NO_PAIR",
    "PiStreamPacket", "PiStreamHeader", "PiFrameStamp", "jpeg_size",
    "ClockSample", "ClockStats", "ClockFit", "ClockMap", "OneWayClockMap", "NotSynced",
    "CAMERAS", "CameraHealthReport", "HealthReport",
    "MOTION_KEYS", "GRIPPER_COMMANDS", "OUTCOMES", "SAVED_OUTCOMES", "TASK_NAME_RE", "EPISODE_NAME_RE",
    "is_task_name", "is_episode_name",
    "StartRecording", "StopRecording", "GetFramesHealth", "CameraSummary", "RecordingSummary", "FramesHealth",
    "GetPreview", "Preview", "NO_FRAME",
    "GetStatus", "KeysUpdate", "GripperCommand", "SelectTask", "CreateTask", "HomeRequest", "StartEpisode",
    "StopEpisode", "DeleteEpisode", "StopArm", "EpisodeInfo", "UiMessage", "TeleopStatus",
    "PHASES", "MESSAGE_LEVELS", "STATUS_MAX_MESSAGES",
]
