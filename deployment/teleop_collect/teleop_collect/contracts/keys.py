"""The names the ui and teleop share (STRUCTURE.md 4.1): motion keys, gripper commands, outcomes, task and episode
names.

The keyboard map (W/S -> x+/x-, ...) belongs to the ui alone; teleop only ever sees these names.
"""
from __future__ import annotations

import re

#: Motion keys, as U1 `keys.held` names them (PLAN 5.4).
MOTION_KEYS = ("x+", "x-", "y+", "y-", "z+", "z-", "twist+", "twist-",
               "tilt_fwd", "tilt_back", "tilt_left", "tilt_right")
#: U1 `gripper.command`: N closes (a grasp with force), M opens fully.
GRIPPER_COMMANDS = ("close", "open")
#: U1 `stop_episode.outcome`; a discarded episode is deleted, the others are saved.
OUTCOMES = ("success", "fail", "discard")
#: The outcome of a saved episode (EpisodeInfo.outcome).
SAVED_OUTCOMES = ("success", "fail")
TASK_NAME_RE = r"^[a-z0-9_]{1,64}$"
EPISODE_NAME_RE = r"^episode_[0-9]{3,}$"

_TASK_NAME = re.compile(TASK_NAME_RE)
_EPISODE_NAME = re.compile(EPISODE_NAME_RE)


def is_task_name(name: object) -> bool:
    """Lowercase letters, digits and `_`, 1-64 characters."""
    return isinstance(name, str) and _TASK_NAME.fullmatch(name) is not None


def is_episode_name(name: object) -> bool:
    """`episode_` and at least three digits."""
    return isinstance(name, str) and _EPISODE_NAME.fullmatch(name) is not None
