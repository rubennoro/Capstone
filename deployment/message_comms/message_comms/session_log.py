"""SessionLog: session.yaml, episodes.jsonl, steps.jsonl, events.jsonl and the optional tap (STRUCTURE.md 4.10).

main-sim eval/ records were the reference only.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from contracts import PolicyConfig, SetWaypointsReply

if TYPE_CHECKING:
    from message_comms.policy_step import StepDone
    from message_comms.settings import Settings

NS_PER_S = 1_000_000_000
# the int fields of EpisodeRecord that end_episode reads from `counts`; skips come as "skips.<reason>" keys
COUNT_FIELDS = ("steps_sent", "late", "anchor_fallbacks", "odd_chunks", "shortened", "pulled_back", "loop_late",
                "width_clips")
SKIP_PREFIX = "skips."


class Outcome(Enum):
    SUCCESS = "success"
    FAIL = "fail"
    STOPPED = "stopped"
    ABORTED = "aborted"


@dataclass(frozen=True)
class StepRecord:
    episode_id: int
    chunk_id: int
    t_anchor_ns: int
    anchor_source: str
    anchor_dev_mm: float
    image_age_ms: float
    frames_ms: float
    policy_ms: float
    cycle_ms: float
    sent: bool
    skip_reason: str            # "" when sent
    accepted: int
    skipped: int
    shortened: int
    rejected: int
    margin_ms: float | None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    outcome: Outcome
    reason: str
    wall_start: str             # ISO 8601 local time
    wall_end: str
    duration_s: float
    steps_sent: int
    skips: Mapping[str, int]
    late: int
    anchor_fallbacks: int
    odd_chunks: int
    shortened: int              # C2 deltas over the episode
    pulled_back: int
    loop_late: int
    width_clips: int            # sum of width_clipped over the episode's chunks
    start_pose: list[list[float]]   # 4x4


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class SessionLog:
    def __init__(self, folder: Path, wall_clock: Callable[[], datetime], clock: Callable[[], int]) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._wall = wall_clock
        self._clock = clock
        self._files = {name: open(self._folder / f"{name}.jsonl", "a", encoding="utf-8")
                       for name in ("episodes", "steps", "events")}
        self._log_steps = True
        self._save_tap = False
        self._tap_images_every = 1
        self._tap_count = 0
        self._episode: tuple[int, np.ndarray, str, int] | None = None   # id, start pose, wall start, t start
        self._closed = False

    @property
    def folder(self) -> Path:
        return self._folder

    def write_session(self, settings: Settings, policy_config: PolicyConfig) -> None:
        """Written once at startup. Also takes log_steps, save_tap and tap_images_every from the settings."""
        self._log_steps = settings.log_steps
        self._save_tap = settings.save_tap
        self._tap_images_every = settings.tap_images_every
        doc = {"settings": settings.to_dict(), "policy_config": _plain(policy_config.to_dict())}
        (self._folder / "session.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    def episode_open(self) -> bool:
        return self._episode is not None

    def start_episode(self, episode_id: int, start_pose: np.ndarray) -> None:
        self._episode = (episode_id, np.array(start_pose, dtype=np.float64), self._wall_iso(), self._clock())
        self._tap_count = 0

    def log_step(self, record: StepRecord) -> None:
        if self._log_steps:
            self._write("steps", dataclasses.asdict(record))

    def end_episode(self, outcome: Outcome, reason: str, counts: Mapping[str, int]) -> EpisodeRecord:
        if self._episode is None:
            raise RuntimeError("end_episode without an open episode")
        episode_id, start_pose, wall_start, t_start = self._episode
        self._episode = None
        record = EpisodeRecord(
            episode_id=episode_id, outcome=outcome, reason=reason, wall_start=wall_start,
            wall_end=self._wall_iso(), duration_s=(self._clock() - t_start) / NS_PER_S,
            skips={k[len(SKIP_PREFIX):]: int(v) for k, v in counts.items() if k.startswith(SKIP_PREFIX)},
            start_pose=start_pose.tolist(),
            **{name: int(counts.get(name, 0)) for name in COUNT_FIELDS},
        )
        self._write("episodes", dataclasses.asdict(record))
        return record

    def log_event(self, kind: str, detail: str) -> None:
        self._write("events", {"t_ns": self._clock(), "wall": self._wall_iso(), "kind": kind, "detail": detail})

    def save_tap(self, step: StepDone, reply: SetWaypointsReply) -> None:
        """One uncompressed .npz per sent step; the four images only every tap_images_every steps."""
        if not self._save_tap:
            return
        obs, chunk, wp, anchor = step.observation, step.chunk, step.waypoints, step.anchor
        arrays = {
            "eef_pos": obs.eef_pos, "eef_rot6d": obs.eef_rot6d, "eef_rot6d_wrt_start": obs.eef_rot6d_wrt_start,
            "gripper_width_m": obs.gripper_width_m,
            "main_t_ns": step.main_t_ns, "ultrawide_t_ns": step.ultrawide_t_ns,
            "anchor_T": anchor.T, "anchor_t_ns": np.int64(anchor.t_ns), "anchor_source": np.array(anchor.source),
            "start_pose": step.job.start_pose,
            "chunk_eef_pos": chunk.eef_pos, "chunk_eef_rot6d": chunk.eef_rot6d,
            "chunk_gripper_width_m": chunk.gripper_width_m, "chunk_width_clipped": np.uint64(chunk.width_clipped),
            "waypoints_t_ns": wp.t_ns, "waypoints_O_T_TCP": wp.O_T_TCP,
            "waypoints_gripper_width_m": wp.gripper_width_m,
            "reply_accepted": np.uint64(reply.accepted), "reply_shortened": np.uint64(reply.shortened),
            "reply_skipped": np.uint64(reply.skipped), "reply_rejected": np.uint64(len(reply.rejected)),
        }
        if self._tap_count % self._tap_images_every == 0:
            arrays["main"] = obs.main
            arrays["ultrawide"] = obs.ultrawide
        self._tap_count += 1
        folder = self._folder / "tap" / f"ep_{step.job.episode_id:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"chunk_{step.job.chunk_id}.npz", **arrays)

    def close(self) -> None:
        """An episode still open is closed as aborted (EpisodeLoop normally closes it itself first)."""
        if self._closed:
            return
        if self._episode is not None:
            self.end_episode(Outcome.ABORTED, "message_comms exited with the episode open", {})
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
