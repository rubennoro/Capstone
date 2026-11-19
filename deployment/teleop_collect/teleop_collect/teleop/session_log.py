# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/session_log.py @ 2ffe641
# changes: teleop_collect STRUCTURE.md 6.2: session.yaml, events.jsonl, clock.jsonl and link.jsonl only (no
# episodes.jsonl, steps.jsonl, tap or policy config: an episode's start, stop, abort and reason are events; its data is
# under data_root); write_session takes the settings, the hello reply, the laptop commit and the clock fit;
# + hello_dict() (the hello reply as plain values, also for the tests); _plain also turns paths into strings; imports.
# The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/session_log.py @ 91b178b
# changes: + clock.jsonl, link.jsonl; session.yaml adds the hello reply, both commits and the clock fit; StepRecord
# adds t_anchor_pc_ns and c3_rtt_ms (margin_ms is PC clock); the tap adds the PC times; imports.
"""SessionLog: <session>/teleop/ session.yaml, events.jsonl, clock.jsonl, link.jsonl (STRUCTURE.md 6.2).

Never written to data_root: an aborted episode's reason lives here and on the page only (PLAN 7).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from deploy_contracts import HelloReply

if TYPE_CHECKING:
    from teleop_collect.teleop.settings import Settings

NS_PER_S = 1_000_000_000
FILES = ("events", "clock", "link")


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def hello_dict(hello: HelloReply | None) -> dict[str, Any] | None:
    """The hello reply as plain values (F_T_TCP a 4x4 list), without its schema."""
    if hello is None:
        return None
    return _plain({k: v for k, v in hello.to_dict().items() if k != "schema"})


class SessionLog:
    def __init__(self, folder: Path, wall_clock: Callable[[], datetime], clock: Callable[[], int]) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._wall = wall_clock
        self._clock = clock
        self._files = {name: open(self._folder / f"{name}.jsonl", "a", encoding="utf-8") for name in FILES}
        self._closed = False

    @property
    def folder(self) -> Path:
        return self._folder

    def write_session(self, settings: Settings, hello: HelloReply | None = None, laptop_commit: str = "unknown",
                      clock_fit: Mapping[str, Any] | None = None) -> None:
        """Written once at start-up: the merged settings (incl. the link.yaml and frames.yaml values used), the PC's
        hello reply, both commits, the clock estimator and its fit at ready."""
        doc = {"settings": settings.to_dict(), "hello": hello_dict(hello), "laptop_commit": laptop_commit,
               "pc_commit": hello.commit if hello is not None else None, "clock": _plain(dict(clock_fit or {}))}
        (self._folder / "session.yaml").write_text(yaml.safe_dump(_plain(doc), sort_keys=False))

    def log_clock(self, row: Mapping[str, Any]) -> None:
        """clock.jsonl: estimator, offset (PC - laptop), drift, samples, shortest RTT, newest age; and fit jumps."""
        self._write("clock", {"t_ns": self._clock(), "wall": self._wall_iso(), **row})

    def log_link(self, row: Mapping[str, Any]) -> None:
        """link.jsonl: C3 RTT p50/p95/max, C2 rate, seq gaps, newest C2 age."""
        self._write("link", {"t_ns": self._clock(), "wall": self._wall_iso(), **row})

    def log_event(self, kind: str, detail: str) -> None:
        self._write("events", {"t_ns": self._clock(), "wall": self._wall_iso(), "kind": kind, "detail": detail})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for f in self._files.values():
            f.close()

    def _wall_iso(self) -> str:
        return self._wall().isoformat(timespec="milliseconds")

    def _write(self, name: str, row: Mapping[str, Any]) -> None:
        if self._closed:
            return
        f = self._files[name]
        f.write(json.dumps(_plain(row)) + "\n")
        f.flush()
