"""Read one raw teleop_collect episode into arrays (PLAN.md §5.1) with the file-level checks of §5.3. Owner: E1.

Reads only: COMPLETE, meta.yaml, robot_states.csv, main_frames.csv, ultrawide_frames.csv, and the packet count of
main.mkv / ultrawide.mkv (demux only, no decoding). Never writes. Formats: teleop_collect PLAN §6.2-6.3
(``deployment/teleop_collect/teleop_collect/teleop/episode/{recorders,meta}.py``, ``frames/recorder.py``).

CSV values as teleop_collect writes them: integers as written, floats in ``repr`` form, booleans 1 / 0, a missing
value (here: an unknown mapped time ``t_ns`` / ``t_mid_ns``) as an empty field.

The signatures below are the contract with the other modules.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import yaml

META_VERSION = 1
OUTCOMES = ("success", "fail")
#: pose source -> the robot_states.csv column prefix (16 columns <prefix>_00 .. <prefix>_33, row-major)
POSE_COLUMNS = {"target": "O_T_TCP_target", "measured": "O_T_TCP"}
RIGID_TOL = 1e-6                 # |R^T R - I| and |det R - 1|, and the bottom row [0, 0, 0, 1]

COMPLETE_FILE = "COMPLETE"
META_FILE = "meta.yaml"
STATES_FILE = "robot_states.csv"
CAMERAS = ("main", "ultrawide")
FRAMES_FILES = {cam: f"{cam}_frames.csv" for cam in CAMERAS}
VIDEO_FILES = {cam: f"{cam}.mkv" for cam in CAMERAS}


class EpisodeError(ValueError):
    """The episode cannot be used; the message is the one-line reason stored in status.json."""


@dataclass(eq=False)
class RawEpisode:
    episode_dir: Path
    name: str                    # episode_NNN
    task: str                    # meta.task
    outcome: str                 # "success" | "fail"
    meta: dict[str, Any]         # the whole meta.yaml
    meta_sha256: str             # of the meta.yaml bytes
    rig: dict[str, Any]          # meta.rig (the rig file's content); never None here
    F_T_TCP: np.ndarray          # (4, 4) float64, meta.pc_hello.F_T_TCP (nested 4x4 list in the yaml)
    pose_source: str             # "target" | "measured"
    # arm states, rows with an empty t_ns dropped (counted in dropped_states), strictly increasing t_ns
    t_state_ns: np.ndarray       # (S,) int64, laptop clock
    seq: np.ndarray              # (S,) int64
    T_state: np.ndarray          # (S, 4, 4) float64, the chosen pose source, each a rigid transform
    width_m: np.ndarray          # (S,) float64, gripper_width_m
    dropped_states: int
    # frames: video_index == row number in the CSV (checked); rows with an empty t_mid_ns dropped (counted),
    # so main_video_index keeps the link from each remaining time to its frame in the .mkv
    main_t_ns: np.ndarray        # (F,) int64, t_mid_ns, strictly increasing
    main_video_index: np.ndarray # (F,) int64
    uw_t_ns: np.ndarray          # (U,) int64
    uw_video_index: np.ndarray   # (U,) int64
    dropped_frames: dict[str, int]   # {"main": n, "ultrawide": n}
    main_video: Path
    uw_video: Path


def pose_columns(pose_source: str) -> list[str]:
    """The 16 robot_states.csv columns of a pose source, row-major. ValueError for an unknown pose source."""
    if pose_source not in POSE_COLUMNS:
        raise ValueError(f"pose_source {pose_source!r}: use one of {', '.join(POSE_COLUMNS)}")
    prefix = POSE_COLUMNS[pose_source]
    return [f"{prefix}_{r}{c}" for r in range(4) for c in range(4)]


def rigid_error(T: np.ndarray) -> np.ndarray:
    """Per transform of a (..., 4, 4) array: the largest of |R^T R - I|, |det R - 1| and |bottom row - [0,0,0,1]|
    (inf where any entry is not finite)."""
    T = np.asarray(T, dtype=np.float64)
    finite = np.isfinite(T).all(axis=(-1, -2))
    T = np.where(finite[..., None, None], T, np.eye(4))       # no NaN warnings; those are inf below
    R = T[..., :3, :3]
    eye = np.eye(3)
    ortho = np.abs(np.swapaxes(R, -1, -2) @ R - eye).max(axis=(-1, -2))
    det = np.abs(np.linalg.det(R) - 1.0)
    bottom = np.abs(T[..., 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max(axis=-1)
    err = np.maximum(np.maximum(ortho, det), bottom)
    return np.where(finite, err, np.inf)


def count_video_frames(video: Path) -> int:
    """Number of packets of the single video stream of an .mkv (demux only). EpisodeError if unreadable."""
    video = Path(video)
    if not video.is_file():
        raise EpisodeError(f"{video.name}: missing; check the episode folder")
    try:
        with av.open(str(video)) as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise EpisodeError(f"{video.name}: {len(streams)} video streams, expected 1; check the recording")
            # the demuxer ends with an empty flush packet (size 0, no dts): not a frame
            return sum(1 for packet in container.demux(streams[0]) if packet.size > 0)
    except EpisodeError:
        raise
    except (av.error.FFmpegError, OSError, ValueError) as e:
        why = str(e).splitlines()[0] if str(e) else ""
        raise EpisodeError(f"{video.name}: unreadable ({type(e).__name__}: {why}); check the recording") from None


# ---- CSV ------------------------------------------------------------------------------------------------------

def _read_csv(path: Path, required: list[str]) -> tuple[dict[str, int], list[list[str]]]:
    """(column -> index, the data rows). Every required column present, every row with as many fields as the
    header."""
    if not path.is_file():
        raise EpisodeError(f"{path.name}: missing; check the episode folder")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise EpisodeError(f"{path.name}: unreadable ({type(e).__name__}); check the file") from None
    if not rows:
        raise EpisodeError(f"{path.name}: empty (no header); check the recording")
    header = rows[0]
    cols = {name: i for i, name in enumerate(header)}
    missing = [c for c in required if c not in cols]
    if missing:
        more = f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""
        raise EpisodeError(f"{path.name}: column {missing[0]} missing{more}; check the teleop_collect version")
    data = rows[1:]
    for i, row in enumerate(data):
        if len(row) != len(header):
            raise EpisodeError(f"{path.name} line {i + 2}: {len(row)} fields for {len(header)} columns; "
                               f"check the file")
    return cols, data


def _int(path: Path, line: int, column: str, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise EpisodeError(f"{path.name} line {line}: {column} {value!r} is not an integer; check the file") from None


def _float(path: Path, line: int, column: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise EpisodeError(f"{path.name} line {line}: {column} {value!r} is not a number; check the file") from None


def _check_increasing(path: Path, column: str, t: np.ndarray, lines: np.ndarray) -> None:
    if len(t) > 1:
        bad = np.flatnonzero(np.diff(t) <= 0)
        if bad.size:
            k = int(bad[0]) + 1
            raise EpisodeError(f"{path.name} line {int(lines[k])}: {column} {int(t[k])} not after {int(t[k - 1])} "
                               f"(must strictly increase); check the clock mapping")


def _read_states(path: Path, pose_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pcols = pose_columns(pose_source)
    cols, data = _read_csv(path, ["t_ns", "seq", *pcols, "gripper_width_m"])
    i_t, i_seq, i_w = cols["t_ns"], cols["seq"], cols["gripper_width_m"]
    i_pose = [cols[c] for c in pcols]
    t, seq, width, poses, lines = [], [], [], [], []
    dropped = 0
    for k, row in enumerate(data):
        line = k + 2
        if row[i_t] == "":
            dropped += 1
            continue
        t.append(_int(path, line, "t_ns", row[i_t]))
        seq.append(_int(path, line, "seq", row[i_seq]))
        w = _float(path, line, "gripper_width_m", row[i_w])
        if not math.isfinite(w):
            raise EpisodeError(f"{path.name} line {line}: gripper_width_m {row[i_w]!r} is not finite; check the file")
        width.append(w)
        poses.append([_float(path, line, c, row[i]) for c, i in zip(pcols, i_pose)])
        lines.append(line)
    if len(t) < 2:
        raise EpisodeError(f"{path.name}: {len(t)} rows with a mapped t_ns ({dropped} without), need at least 2; "
                           f"check the recording")
    t_arr = np.asarray(t, dtype=np.int64)
    line_arr = np.asarray(lines, dtype=np.int64)
    _check_increasing(path, "t_ns", t_arr, line_arr)
    T = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    err = rigid_error(T)
    bad = np.flatnonzero(~(err <= RIGID_TOL))
    if bad.size:
        k = int(bad[0])
        raise EpisodeError(f"{path.name} line {int(line_arr[k])}: {POSE_COLUMNS[pose_source]} is not a rigid "
                           f"transform (error {float(err[k]):.3g} > {RIGID_TOL:g}); check the arm states")
    return t_arr, np.asarray(seq, dtype=np.int64), T, np.asarray(width, dtype=np.float64), dropped


def _read_frames(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(t_mid_ns, video_index) of the rows with a mapped time, the number dropped, and the total row count."""
    cols, data = _read_csv(path, ["video_index", "t_mid_ns"])
    i_v, i_t = cols["video_index"], cols["t_mid_ns"]
    # video_index == row number on every row, before any row is dropped
    for k, row in enumerate(data):
        if _int(path, k + 2, "video_index", row[i_v]) != k:
            raise EpisodeError(f"{path.name} line {k + 2}: video_index {row[i_v]} != row number {k}; "
                               f"check the recording")
    t, index, lines = [], [], []
    dropped = 0
    for k, row in enumerate(data):
        if row[i_t] == "":
            dropped += 1
            continue
        t.append(_int(path, k + 2, "t_mid_ns", row[i_t]))
        index.append(k)
        lines.append(k + 2)
    if len(t) < 2:
        raise EpisodeError(f"{path.name}: {len(t)} rows with a mapped t_mid_ns ({dropped} without), need at least 2; "
                           f"check the recording")
    t_arr = np.asarray(t, dtype=np.int64)
    _check_increasing(path, "t_mid_ns", t_arr, np.asarray(lines, dtype=np.int64))
    return t_arr, np.asarray(index, dtype=np.int64), dropped, len(data)


# ---- meta.yaml --------------------------------------------------------------------------------------------------

def _matrix4(value: Any) -> np.ndarray | None:
    """A nested 4x4 list of real numbers as a float64 array, else None."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
            return None
    return np.asarray(value, dtype=np.float64)


def _read_meta(episode_dir: Path) -> tuple[dict[str, Any], str, dict[str, Any], np.ndarray]:
    path = episode_dir / META_FILE
    if not path.is_file():
        raise EpisodeError(f"{META_FILE}: missing; check the episode folder")
    try:
        raw = path.read_bytes()
        meta = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise EpisodeError(f"{META_FILE}: unreadable ({type(e).__name__}); check the file") from None
    if not isinstance(meta, dict):
        raise EpisodeError(f"{META_FILE}: not a mapping; check the file")
    version = meta.get("meta_version")
    if isinstance(version, bool) or version != META_VERSION:
        raise EpisodeError(f"{META_FILE}: meta_version {version!r}, this reader knows {META_VERSION}; "
                           f"check the teleop_collect version")
    outcome = meta.get("outcome")
    if outcome not in OUTCOMES:
        raise EpisodeError(f"{META_FILE}: outcome {outcome!r} not one of {', '.join(OUTCOMES)}; check the file")
    task = meta.get("task")
    if not isinstance(task, str) or task != episode_dir.parent.name:
        raise EpisodeError(f"{META_FILE}: task {task!r} != task folder {episode_dir.parent.name!r}; "
                           f"check that the episode was not moved")
    if meta.get("episode") != episode_dir.name:
        raise EpisodeError(f"{META_FILE}: episode {meta.get('episode')!r} != folder {episode_dir.name!r}; "
                           f"check that the episode was not renamed")
    rig = meta.get("rig")
    if rig is None:
        note_raw = meta.get("rig_note")
        note = (str(note_raw).splitlines() or [""])[0] if note_raw else "no rig_note"
        raise EpisodeError(f"{META_FILE}: rig is null ({note}); check the rig file on the laptop when recording")
    if not isinstance(rig, dict):
        raise EpisodeError(f"{META_FILE}: rig is not a mapping; check the rig file")
    hello = meta.get("pc_hello")
    F = _matrix4(hello.get("F_T_TCP") if isinstance(hello, dict) else None)
    if F is None:
        raise EpisodeError(f"{META_FILE}: pc_hello.F_T_TCP missing or not a 4x4 list of numbers; check the PC hello")
    err = float(rigid_error(F))
    if not err <= RIGID_TOL:
        raise EpisodeError(f"{META_FILE}: pc_hello.F_T_TCP is not a rigid transform (error {err:.3g} > "
                           f"{RIGID_TOL:g}); check the PC's frames.yaml")
    return meta, hashlib.sha256(raw).hexdigest(), rig, F


# ---- the episode ------------------------------------------------------------------------------------------------

def read_episode(episode_dir: Path | str, pose_source: str = "target") -> RawEpisode:
    """One raw episode. EpisodeError (one line) for:

    - no COMPLETE; meta.yaml missing / unreadable / meta_version != 1 / outcome not in OUTCOMES;
    - meta.task != the task folder's name or meta.episode != the episode folder's name;
    - meta.rig null (meta.rig_note says why) or F_T_TCP not a 4x4 rigid transform;
    - a missing file or column; fewer than 2 usable rows in robot_states.csv or either frames CSV;
    - t_ns / t_mid_ns not strictly increasing (after dropping empty ones); video_index != row number;
    - a .mkv packet count != its CSV row count;
    - a pose that is not a rigid transform (RIGID_TOL); a non-finite width.
    ValueError for an unknown pose_source.
    """
    pose_columns(pose_source)                       # ValueError first
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise EpisodeError(f"{episode_dir}: not a folder; check the episode path")
    if not (episode_dir / COMPLETE_FILE).is_file():
        raise EpisodeError(f"{COMPLETE_FILE}: missing (the recording did not finish); check the episode")
    meta, meta_sha, rig, F = _read_meta(episode_dir)

    t_state, seq, T, width, dropped_states = _read_states(episode_dir / STATES_FILE, pose_source)

    frames: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for cam in CAMERAS:
        csv_path = episode_dir / FRAMES_FILES[cam]
        t, index, dropped, rows = _read_frames(csv_path)
        packets = count_video_frames(episode_dir / VIDEO_FILES[cam])
        if packets != rows:
            raise EpisodeError(f"{VIDEO_FILES[cam]}: {packets} frames but {FRAMES_FILES[cam]} has {rows} rows; "
                               f"check the recording")
        frames[cam] = (t, index, dropped)

    return RawEpisode(
        episode_dir=episode_dir,
        name=episode_dir.name,
        task=meta["task"],
        outcome=meta["outcome"],
        meta=meta,
        meta_sha256=meta_sha,
        rig=rig,
        F_T_TCP=F,
        pose_source=pose_source,
        t_state_ns=t_state,
        seq=seq,
        T_state=T,
        width_m=width,
        dropped_states=dropped_states,
        main_t_ns=frames["main"][0],
        main_video_index=frames["main"][1],
        uw_t_ns=frames["ultrawide"][0],
        uw_video_index=frames["ultrawide"][1],
        dropped_frames={"main": frames["main"][2], "ultrawide": frames["ultrawide"][2]},
        main_video=episode_dir / VIDEO_FILES["main"],
        uw_video=episode_dir / VIDEO_FILES["ultrawide"],
    )
