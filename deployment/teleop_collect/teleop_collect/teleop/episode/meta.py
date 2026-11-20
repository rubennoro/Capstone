"""meta.yaml of a saved episode (PLAN 6.3, 13.9, 13.12; STRUCTURE.md 8.3).

Only what helps training and the offline alignment: the task and its one-line description, the episode, the outcome,
the times on the laptop and the PC, the counts, frames' summary, the full `hello` reply (with `F_T_TCP`), the laptop's
commit, the rig file (its path and its parsed content, `rig: null` plus `rig_note` if unreadable), the teleop
settings and the final Pi and PC clock fits, and `warnings`: the data problems seen while recording (a camera down or
stale), each once: {code, what, t_s} (t_s: seconds into the episode); empty when there were none. No operator names,
no notes.
`Task.episodes` reads `outcome`, `duration_s` and `counts.frames_main` back.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import yaml
from deploy_contracts import HelloReply

from teleop_collect.contracts import SAVED_OUTCOMES, RecordingSummary
from teleop_collect.teleop.episode.store import episode_number, to_plain
from teleop_collect.teleop.motion.planner import MotionSettings, worst_step

if TYPE_CHECKING:
    from teleop_collect.teleop.episode.store import EpisodeDir, Task

META_VERSION = 1
TIME_KEYS = ("start_ns", "stop_ns", "start_pc_ns", "stop_pc_ns")
WARNING_KEYS = ("code", "what", "t_s")


def _int_or_none(name: str, value: Any, required: bool) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"times.{name}: {value!r} is not an integer ns time")
    return int(value)


def read_rig(rig_file: Path) -> tuple[Any, str]:
    """(the parsed rig yaml, "") or (None, why it could not be read): meta.yaml fails soft on the rig."""
    try:
        rig = yaml.safe_load(Path(rig_file).read_text(encoding="utf-8"))
        return to_plain(rig), ""
    except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
        return None, f"{rig_file}: {type(e).__name__}: {e}".splitlines()[0]


def build_meta(*, task: Task, episode: EpisodeDir, outcome: str, times: Mapping[str, int], frames: RecordingSummary,
               counts: Mapping[str, int], hello: HelloReply, laptop_commit: str, laptop_dirty: bool,
               rig_file: Path, motion: MotionSettings, pi_fit: Mapping, pc_fit: Mapping,
               warnings: Sequence[Mapping[str, Any]] = ()) -> dict:
    """The meta.yaml content as plain YAML-able types. ValueError for an outcome that is not saved or bad times.

    times: start_ns, stop_ns (laptop, required), start_pc_ns, stop_pc_ns (mapped PC times, None if unknown).
    counts: EpisodeRecording.counts() plus anything the caller adds; frames' counts are added from `frames`.
    warnings: EpisodeFlow's {code, what, t_s} per data problem seen while recording (ValueError for other keys).
    """
    if outcome not in SAVED_OUTCOMES:
        raise ValueError(f"outcome {outcome!r}: only {', '.join(SAVED_OUTCOMES)} are saved")
    unknown = sorted(set(times) - set(TIME_KEYS))
    if unknown:
        raise ValueError(f"times: unknown {', '.join(unknown)}")
    t = {k: _int_or_none(k, times.get(k), k in ("start_ns", "stop_ns")) for k in TIME_KEYS}
    if t["stop_ns"] < t["start_ns"]:
        raise ValueError(f"times: stop_ns {t['stop_ns']} before start_ns {t['start_ns']}")
    for w in warnings:
        if set(w) != set(WARNING_KEYS):
            raise ValueError(f"warnings: {dict(w)!r} must have exactly the keys {', '.join(WARNING_KEYS)}")
    frame_counts = {
        "frames_main": frames.main.frames,
        "frames_ultrawide": frames.ultrawide.frames,
        "frame_no_gaps_main": frames.main.frame_no_gaps,
        "frame_no_gaps_ultrawide": frames.ultrawide.frame_no_gaps,
        "packets": frames.packets,
        "packet_no_gaps": frames.packet_no_gaps,
        "uw_reused": frames.uw_reused,
        "pair_refused": frames.pair_refused,
    }
    clash = sorted(set(frame_counts) & set(counts))
    if clash:
        raise ValueError(f"counts: {', '.join(clash)} come from the frames summary")
    step_m, step_deg = worst_step(motion)
    rig, rig_note = read_rig(rig_file)
    hello_d = to_plain(hello)
    hello_d.pop("schema", None)
    frames_d = to_plain(frames)
    frames_d.pop("schema", None)
    return {
        "meta_version": META_VERSION,
        "task": task.name,
        "task_description": task.description,
        "episode": episode.name,
        "episode_number": episode_number(episode.name),
        "outcome": outcome,
        "times": t,
        "duration_s": (t["stop_ns"] - t["start_ns"]) / 1e9,
        "counts": {**frame_counts, **{str(k): to_plain(v) for k, v in counts.items()}},
        "frames": frames_d,
        "pc_hello": hello_d,
        "laptop": {"commit": str(laptop_commit), "dirty": bool(laptop_dirty)},
        "rig_file": str(rig_file),
        "rig": rig,                                  # the rig file's content (PLAN 6.3); null if unreadable
        "rig_note": rig_note,                        # "" or why the rig could not be read
        "teleop": {**motion.to_dict(), "lead_ms": motion.waypoints * motion.spacing_ms,
                   "worst_step_m": step_m, "worst_step_deg": step_deg},
        "clock": {"pi": to_plain(pi_fit), "pc": to_plain(pc_fit)},
        "warnings": [{k: to_plain(w[k]) for k in WARNING_KEYS} for w in warnings],
    }
