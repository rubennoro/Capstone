"""StatusLine: one line, at most 120 characters, rewritten in place (STRUCTURE.md 4.9)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contracts import HealthReport

if TYPE_CHECKING:
    from message_comms.episode_loop import Phase

MAX_LEN = 120
HOMING_LINE = "[homing] keys inactive; Ctrl-C quits; immediate stop: the robot's external stop button"
KEYS = {                              # by Phase value (Phase lives in episode_loop.py, which imports this file)
    "starting": "keys: q quit",
    "idle": "keys: c start  h home  q quit",
    "resetting": "keys: s cancel  q quit",
    "running": "keys: y n s q",
    "done": "",
}


@dataclass(frozen=True)
class StatusSnapshot:
    phase: Phase
    episode_id: int
    episode_s: float
    steps_sent: int
    cycle_ms: float | None
    policy_ms: float | None
    image_age_ms: float | None
    margin_ms: float | None          # last waypoint time of the previous set - send time of the new one
    waiting_policy_ms: float | None  # worker running for (set only when > 2 x typical)
    skips: Mapping[str, int]
    late: int
    anchor_fallbacks: int
    odd_chunks: int
    skipping_s: float | None         # set only after skip_warn_s of skipping without a break
    skipping_reason: str
    shortened: int                   # C2 gate.shortened since the episode started (session total in IDLE)
    pulled_back: int
    loop_late: int
    width_clips: int
    cameras: HealthReport | None
    arm_mode: str | None
    anchor_dev_mm: float | None


class StatusLine:
    def format(self, snap: StatusSnapshot) -> str:
        if snap.phase.value == "homing":
            return HOMING_LINE[:MAX_LEN]
        keys = KEYS[snap.phase.value]
        arm = f"arm {snap.arm_mode or '?'}"
        if snap.phase.value == "running":
            head = f"[ep {snap.episode_id} running {snap.episode_s:.1f} s]"
            if snap.waiting_policy_ms is not None:
                return _fit(head, [f"waiting for policy {snap.waiting_policy_ms / 1000:.1f} s", arm], [], keys)
            if snap.skipping_s is not None:
                return _fit(head, [f"SKIPPING {snap.skipping_s:.1f} s: {snap.skipping_reason}", arm], [], keys)
            parts = [f"step {snap.steps_sent}"]
            if snap.cycle_ms is not None:
                policy = f" (policy {snap.policy_ms:.0f})" if snap.policy_ms is not None else ""
                parts.append(f"cycle {snap.cycle_ms:.0f} ms{policy}")
            if snap.image_age_ms is not None:
                parts.append(f"img {snap.image_age_ms:.0f} ms")
            if snap.margin_ms is not None:
                parts.append(f"margin {snap.margin_ms:+.0f} ms")
            counters = [f"short {snap.shortened}", f"pulled {snap.pulled_back}", f"clip {snap.width_clips}",
                        f"odd {snap.odd_chunks}", f"late loops {snap.loop_late}"]
            if snap.late:
                counters.append(f"late {snap.late}")
            if snap.anchor_fallbacks:
                counters.append(f"fallback {snap.anchor_fallbacks}")
            skipped = sum(snap.skips.values())
            if skipped:
                counters.append(f"skips {skipped}")
            return _fit_running(head, parts, counters, keys)
        if snap.phase.value == "resetting":
            return _fit(f"[ep {snap.episode_id + 1} starting]", ["resetting the policy", arm], [], keys)
        return _fit(f"[{snap.phase.value}]", [arm, _cameras(snap.cameras), f"late loops {snap.loop_late}"], [], keys)


def _cameras(health: HealthReport | None) -> str:
    if health is None:
        return "cams ?"
    return (f"cams {health.main.status} {health.main.fps:.0f}/"
            f"{health.ultrawide.status} {health.ultrawide.fps:.0f} fps")


def _join(head: str, middle: list[str], keys: str) -> str:
    pieces = [head + (" " + middle[0] if middle else "")] + middle[1:] + ([keys] if keys else [])
    return " | ".join(pieces)


def _fit(head: str, parts: list[str], extras: list[str], keys: str) -> str:
    """head parts[0] | parts... | keys, cut to MAX_LEN."""
    return _join(head, parts + extras, keys)[:MAX_LEN]


def _fit_running(head: str, parts: list[str], counters: list[str], keys: str) -> str:
    """The running line. Too long: drop the counters at 0, then put the counters in one segment, then drop the
    image age, then drop the least important counters."""
    line = _join(head, parts + counters, keys)
    if len(line) <= MAX_LEN:
        return line
    counters = [c for c in counters if not c.endswith(" 0")]
    while True:
        line = _join(head, parts + ([" ".join(counters)] if counters else []), keys)
        if len(line) <= MAX_LEN or not counters:
            break
        img = [p for p in parts if p.startswith("img ")]
        if img:
            parts = [p for p in parts if p not in img]
        else:
            counters.remove(min(counters, key=_keep_rank))
    return line[:MAX_LEN]


_KEEP_ORDER = ("odd", "short", "pulled", "late ", "clip", "skips", "fallback", "late loops")   # most important first


def _keep_rank(counter: str) -> int:
    ranks = [i for i, prefix in enumerate(_KEEP_ORDER) if counter.startswith(prefix)]
    return -max(ranks)
