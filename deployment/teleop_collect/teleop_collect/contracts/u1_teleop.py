"""U1: ui -> teleop, the page's keys, buttons and status (STRUCTURE.md 4.4). REP on station.u1_endpoint().

| request                 | reply                                                   |
|-------------------------|---------------------------------------------------------|
| u1.get_status/v1        | u1.status/v1                                            |
| u1.keys/v1              | ack/v1 (always ok; ignored outside an episode)          |
| u1.gripper/v1           | ack/v1                                                  |
| u1.select_task/v1       | ack/v1                                                  |
| u1.create_task/v1       | ack/v1 (also selects it)                                |
| u1.start_episode/v1     | ack/v1                                                  |
| u1.stop_episode/v1      | ack/v1                                                  |
| u1.delete_episode/v1    | ack/v1                                                  |
| u1.stop_arm/v1          | ack/v1                                                  |

Refusals are ack(ok=false, reason) in plain words. Task names are checked by teleop (an ack refusal the operator can
read), not by the contract; only delete_episode's name has a contract pattern.

There is no home request: homing (h) and the start pose (p) are the PC's keyboard keys. Start is allowed only at the
start pose (start_pose_known, at_start_pose in the status).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from deploy_contracts import MODES, Message, Record, register
from deploy_contracts.fields import array, list_of, nested, optional, scalar, str_list

from teleop_collect.contracts.health import HealthReport
from teleop_collect.contracts.keys import (EPISODE_NAME_RE, GRIPPER_COMMANDS, MOTION_KEYS, OUTCOMES, SAVED_OUTCOMES,
                                           is_episode_name)

PHASES = ("starting", "idle", "recording", "stopping", "done")
MESSAGE_LEVELS = ("info", "warn", "error")
#: TeleopStatus.messages holds at most this many (those after the request's after_message, oldest first).
STATUS_MAX_MESSAGES = 50


# ------------------------------------------------------------------------------------------------------- requests

@register
@dataclass(frozen=True, eq=False)
class GetStatus(Message):
    """after_message: the newest UiMessage id the ui already has (0: none)."""

    SCHEMA = "u1.get_status/v1"
    after_message: int = scalar("uint64")


@register
@dataclass(frozen=True, eq=False)
class KeysUpdate(Message):
    """The page's key state, sent on every change and at least every keys_resend_ms while it has focus."""

    SCHEMA = "u1.keys/v1"
    held: tuple[str, ...] = str_list(MOTION_KEYS)
    slow: bool = scalar("bool")
    focus: bool = scalar("bool")

    def check_values(self) -> None:
        if len(set(self.held)) != len(self.held):
            self.fail("held", f"a key listed twice: {list(self.held)}")


@register
@dataclass(frozen=True, eq=False)
class GripperCommand(Message):
    """N: close (a grasp with force); M: open fully."""

    SCHEMA = "u1.gripper/v1"
    command: str = scalar("str", choices=GRIPPER_COMMANDS)


@register
@dataclass(frozen=True, eq=False)
class SelectTask(Message):
    SCHEMA = "u1.select_task/v1"
    name: str = scalar("str")


@register
@dataclass(frozen=True, eq=False)
class CreateTask(Message):
    """name: lowercase letters, digits and `_` (checked by teleop); description: the task in one line."""

    SCHEMA = "u1.create_task/v1"
    name: str = scalar("str")
    description: str = scalar("str")


@register
@dataclass(frozen=True, eq=False)
class StartEpisode(Message):
    SCHEMA = "u1.start_episode/v1"


@register
@dataclass(frozen=True, eq=False)
class StopEpisode(Message):
    """outcome: success or fail (saved), or discard (deleted)."""

    SCHEMA = "u1.stop_episode/v1"
    outcome: str = scalar("str", choices=OUTCOMES)


@register
@dataclass(frozen=True, eq=False)
class DeleteEpisode(Message):
    SCHEMA = "u1.delete_episode/v1"
    episode: str = scalar("str")

    def check_values(self) -> None:
        if not is_episode_name(self.episode):
            self.fail("episode", f"{self.episode!r} does not match {EPISODE_NAME_RE}")


@register
@dataclass(frozen=True, eq=False)
class StopArm(Message):
    """Stop arm / Esc: C3 stop; an open episode is aborted (deleted)."""

    SCHEMA = "u1.stop_arm/v1"


# -------------------------------------------------------------------------------------------------------- records

@dataclass(frozen=True, eq=False)
class EpisodeInfo(Record):
    """One complete episode of the selected task."""

    name: str = scalar("str")
    outcome: str = scalar("str", choices=SAVED_OUTCOMES)
    duration_s: float = scalar("float64", min=0.0)
    frames_main: int = scalar("uint64")

    def check_values(self) -> None:
        if not is_episode_name(self.name):
            self.fail("name", f"{self.name!r} does not match {EPISODE_NAME_RE}")


@dataclass(frozen=True, eq=False)
class UiMessage(Record):
    """One line for the page (a hint, a problem, an aborted episode's reason)."""

    id: int = scalar("uint64")
    level: str = scalar("str", choices=MESSAGE_LEVELS)
    text: str = scalar("str")


# ---------------------------------------------------------------------------------------------------------- reply

@register
@dataclass(frozen=True, eq=False)
class TeleopStatus(Message):
    """Everything the page shows. Times: episode_s since Start, laptop clock."""

    SCHEMA = "u1.status/v1"
    phase: str = scalar("str", choices=PHASES)
    ready: bool = scalar("bool")                               # start-up finished
    task: str = scalar("str")                                  # "" when none selected
    task_description: str = scalar("str")
    tasks: tuple[str, ...] = str_list()                        # every task under data_root
    episodes: tuple[EpisodeInfo, ...] = list_of(EpisodeInfo)   # the selected task's complete episodes, oldest first
    recording_episode: str = scalar("str")                     # "" unless recording
    episode_s: float = scalar("float64", min=0.0)              # 0 when not recording
    arm_mode: str = scalar("str")                              # newest C2 mode, "" before any
    tcp_m: np.ndarray | None = optional(array("float64", (3,)))    # newest C2 O_T_TCP position
    gripper_width_m: float | None = optional(scalar("float64"))    # newest C2 width
    commanded_width_m: float = scalar("float64")
    start_pose_known: bool = scalar("bool")                    # learned at start-up or after the PC's h/p
    start_pose_confirmed: bool = scalar("bool")                # the PC accepted a start there: fixed for the session
    at_start_pose: bool = scalar("bool")                       # known, and the newest C2 within its tolerances
    start_allowed: bool = scalar("bool")
    start_refusal: str = scalar("str")                         # why not, "" when allowed
    at_limit: bool = scalar("bool")                            # gate pulled_back rose within at_limit_hold_ms
    moving: bool = scalar("bool")                              # a motion key is in effect
    slow: bool = scalar("bool")
    cameras: HealthReport | None = optional(nested(HealthReport))  # the newest R1 health
    c2_hz: float = scalar("float64", min=0.0)
    rtt_ms: float | None = optional(scalar("float64", min=0.0))
    clock_ok: bool = scalar("bool")
    pc_mock: bool = scalar("bool")
    messages: tuple[UiMessage, ...] = list_of(UiMessage)       # ids > after_message, oldest first
    final_line: str = scalar("str")                            # the one-line reason when phase is done

    def check_values(self) -> None:
        if self.arm_mode not in ("", *MODES):
            self.fail("arm_mode", f"{self.arm_mode!r} is not one of '', {', '.join(MODES)}")
        if self.start_pose_confirmed and not self.start_pose_known:
            self.fail("start_pose_confirmed", "true while start_pose_known is false")
        if self.at_start_pose and not self.start_pose_known:
            self.fail("at_start_pose", "true while start_pose_known is false")
        if self.start_allowed and self.start_refusal:
            self.fail("start_refusal", f"{self.start_refusal!r} while start_allowed")
        if not self.start_allowed and not self.start_refusal:
            self.fail("start_refusal", "empty while start is not allowed (say why)")
        if len(self.messages) > STATUS_MAX_MESSAGES:
            self.fail("messages", f"{len(self.messages)} messages, at most {STATUS_MAX_MESSAGES}")
        ids = [m.id for m in self.messages]
        for i in range(1, len(ids)):
            if ids[i] <= ids[i - 1]:
                self.fail(f"messages[{i}].id", f"ids not strictly increasing: {ids}")
