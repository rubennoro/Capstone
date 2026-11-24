"""SHARED_SCHEMAS: every schema that crosses between the laptop and the PC (STRUCTURE.md 3.4).

A fixed tuple, sorted by byte order. franka_control's codec knows exactly these, and its hello reply lists them; the
laptop compares the two lists at start-up (compare_hello). Laptop-only schemas registered later by policy_deploy
never join it. test_shared.py checks it against the classes this package registers.
"""
from __future__ import annotations

SHARED_SCHEMAS: tuple[str, ...] = (
    "ack/v1",
    "c2.state/v1",
    "c3.get_state/v1",
    "c3.heartbeat/v1",
    "c3.hello.reply/v1",
    "c3.hello/v1",
    "c3.home/v1",
    "c3.set_waypoints.reply/v1",
    "c3.set_waypoints/v1",
    "c3.start/v1",
    "c3.stop/v1",
    "error/v1",
)
