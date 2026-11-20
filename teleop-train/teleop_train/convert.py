"""``teleop_train convert TASK``: one episode end to end, the §5.3 checks, status.json, the up-to-date rule
(PLAN.md §4.1, §5). Owner: E4 (wave 2).

Per episode (``convert_episode``): read the raw episode (episode_reader), the common window, the 60 Hz grid, the
pose / width interpolation, motion detection and trimming, the gap checks on the kept span (trajectory), the images
on the kept grid (images), then ``EpisodeArrays`` -> ``processed/episode.zarr`` and, last, ``processed/status.json``.
Any check that fails leaves the episode out (``excluded``) with a one-line reason in status.json; a stale
``episode.zarr`` of an earlier conversion is then removed so it cannot be packaged. Nothing is written under the raw
root (paths.task_paths refuses an output folder that overlaps it).

Up to date: an episode whose status.json has this module's schema, ``CONVERT_VERSION``, ``SETTINGS``, the same
pose source and the sha256 of the raw ``meta.yaml`` as now is not converted again (unless ``force``). An episode
without ``COMPLETE`` (or without a readable ``meta.yaml``) is never up to date: it is checked again every time
(cheap, the reader refuses it first), so an episode whose recording finishes later is picked up.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from teleop_train import __version__
from teleop_train import images as _images
from teleop_train import paths
from teleop_train import trajectory as _traj
from teleop_train._copied.zarr_writer import EpisodeArrays, write_episode_zarr
from teleop_train.episode_reader import (COMPLETE_FILE, META_FILE, OUTCOMES, RIGID_TOL, EpisodeError, RawEpisode,
                                         read_episode)
from teleop_train.images import ImageError, prepare_images
from teleop_train.paths import PathsError, TaskPaths
from teleop_train.trajectory import (TrajectoryError, check_gaps, common_window, interpolate, make_grid,
                                     moving_mask, trim_span)

CONVERT_VERSION = 1
STATUS_SCHEMA = "teleop_train.episode.v1"

#: every number of PLAN §5.2 / §5.3, from the modules that use them (recorded in status.json; part of up to date)
SETTINGS: dict[str, Any] = {
    "grid_hz": _traj.GRID_HZ,
    "trans_speed_mps": _traj.TRANS_SPEED_MPS,
    "rot_speed_deg_s": _traj.ROT_SPEED_DEG_S,
    "width_rate_mps": _traj.WIDTH_RATE_MPS,
    "smooth_s": _traj.SMOOTH_S,
    "margin_s": _traj.MARGIN_S,
    "max_state_gap_ns": _traj.MAX_STATE_GAP_NS,
    "max_main_gap_ns": _traj.MAX_MAIN_GAP_NS,
    "max_uw_gap_ns": _traj.MAX_UW_GAP_NS,
    "min_kept_s": _traj.MIN_KEPT_S,
    "main_max_dt_ns": _images.MAIN_MAX_DT_NS,
    "uw_rate_hz": _images.UW_RATE_HZ,
    "rigid_tol": RIGID_TOL,
}

STATES = ("converted", "excluded", "up to date")
_TELEOP_TRAIN_DIR = Path(__file__).resolve().parent.parent       # teleop-train/
_PACKAGE_DIR = Path(__file__).resolve().parent                   # teleop-train/teleop_train/
#: the source files that shape a converted episode; their hash is part of the up-to-date rule, so a code change
#: converts again even without a CONVERT_VERSION bump (review finding: stale zarrs after a fix)
CODE_FILES = ("episode_reader.py", "trajectory.py", "images.py", "convert.py", "_copied")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", metavar="TASK", help="the task folder under the raw root")
    p.add_argument("--episodes", default=None, metavar="LIST",
                   help="only these episode numbers, e.g. 3,7-9 (default: all)")
    p.add_argument("--force", action="store_true", help="convert again even when up to date")


@dataclass
class EpisodeResult:
    episode: str                 # episode_NNN
    state: str                   # "converted" | "excluded" | "up to date"
    reason: str                  # "" unless excluded (for "up to date": the stored reason, "" when converted)
    outcome: str | None          # "success" | "fail" | None (meta unreadable)
    raw_s: float | None          # common window length, s
    kept_s: float | None
    n_steps: int | None
    status: dict                 # the status.json content written (or found, when up to date)

    @property
    def usable(self) -> bool:
        """True when the episode has a converted episode.zarr (converted now, or up to date and converted)."""
        return self.status.get("state") == "converted"


# ---- small helpers ------------------------------------------------------------------------------------------------

def parse_episode_list(text: str) -> set[int]:
    """``"3,7-9"`` -> ``{3, 7, 8, 9}``. ValueError (one line) for empty or malformed text or a reversed range."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"--episodes {text!r}: give episode numbers like 3,7-9")
    out: set[int] = set()
    for item in text.split(","):
        part = item.strip()
        lo_s, sep, hi_s = part.partition("-")
        lo_s, hi_s = lo_s.strip(), hi_s.strip()
        if not lo_s.isdigit() or (sep and not hi_s.isdigit()):
            raise ValueError(f"--episodes {text!r}: {part!r} is not a number or a range N-M; give e.g. 3,7-9")
        lo = int(lo_s)
        hi = int(hi_s) if sep else lo
        if hi < lo:
            raise ValueError(f"--episodes {text!r}: range {part!r} runs backwards; give e.g. 7-9")
        out.update(range(lo, hi + 1))
    return out


def read_status(path: Path | str) -> dict | None:
    """The status.json content, or None if it is missing, unreadable or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_up_to_date(status: dict | None, meta_sha256: str | None, pose_source: str,
                  episode_zarr: Path | str | None = None) -> bool:
    """True iff ``status`` has schema STATUS_SCHEMA, ``convert_version`` == CONVERT_VERSION, ``settings`` ==
    SETTINGS, ``code_sha256`` == code_sha256(), the same ``pose_source`` and ``meta_sha256`` (False when ``meta_sha256`` is None: nothing to compare)
    and a state of converted / excluded. When the state is converted and ``episode_zarr`` is given, that folder must
    also exist."""
    if not isinstance(status, dict) or meta_sha256 is None:
        return False
    if (status.get("schema") != STATUS_SCHEMA or status.get("convert_version") != CONVERT_VERSION
            or status.get("settings") != SETTINGS or status.get("code_sha256") != code_sha256()
            or status.get("pose_source") != pose_source
            or status.get("meta_sha256") != meta_sha256):
        return False
    state = status.get("state")
    if state == "converted":
        return episode_zarr is None or Path(episode_zarr).is_dir()
    return state == "excluded"


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


@functools.lru_cache(maxsize=1)
def code_sha256() -> str:
    """sha256 over the relative path and bytes of every CODE_FILES source (.py files, sorted)."""
    files: list[Path] = []
    for name in CODE_FILES:
        p = _PACKAGE_DIR / name
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(_PACKAGE_DIR)).encode("utf-8") + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def rig_sha256(rig: Any) -> str:
    """sha256 of the rig content as compact sorted JSON (non-JSON scalars such as YAML dates as their str)."""
    text = json.dumps(rig, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _outcome_from_meta(raw_dir: Path) -> str | None:
    """meta.yaml's outcome when readable and valid, else None (for an episode the reader refused)."""
    try:
        meta = yaml.safe_load((raw_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    outcome = meta.get("outcome") if isinstance(meta, dict) else None
    return outcome if outcome in OUTCOMES else None


@functools.lru_cache(maxsize=1)
def teleop_train_info() -> dict[str, Any]:
    """{version, commit, dirty} of this teleop-train folder from git; commit "unknown" and dirty None if git fails."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(_TELEOP_TRAIN_DIR), *a], capture_output=True, text=True,
                              check=True, timeout=20).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty: bool | None = bool(git("status", "--porcelain", "--", "."))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    return {"version": __version__, "commit": commit or "unknown", "dirty": dirty}


def _one_line(e: BaseException) -> str:
    return " ".join(str(e).split()) or type(e).__name__


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _remove_zarr(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


# ---- one episode --------------------------------------------------------------------------------------------------

@dataclass
class _Progress:
    """What is known so far about one episode (the status.json fields that may stay null)."""
    raw: RawEpisode | None = None
    outcome: str | None = None
    window: tuple[int, int] | None = None
    kept: tuple[int, int] | None = None
    n_steps: int | None = None
    n_uw: int | None = None


def _status(tp: TaskPaths, episode: str, pose_source: str, meta_sha: str | None, state: str, reason: str,
            pr: _Progress) -> dict[str, Any]:
    raw = pr.raw
    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "reason": reason,
        "episode": episode,
        "outcome": pr.outcome,
        "raw_episode_dir": str(tp.raw_episode_dir(episode)),
        "meta_sha256": meta_sha,
        "window_ns": [int(pr.window[0]), int(pr.window[1])] if pr.window else None,
        "kept_ns": [int(pr.kept[0]), int(pr.kept[1])] if pr.kept else None,
        "raw_s": (pr.window[1] - pr.window[0]) / 1e9 if pr.window else None,
        "kept_s": (pr.kept[1] - pr.kept[0]) / 1e9 if pr.kept else None,
        "n_steps": pr.n_steps,
        "n_uw": pr.n_uw,
        "pose_source": pose_source,
        "F_T_TCP": raw.F_T_TCP.tolist() if raw is not None else None,
        "rig_id": raw.rig.get("rig_id") if raw is not None else None,
        "rig_sha256": rig_sha256(raw.rig) if raw is not None else None,
        "settings": dict(SETTINGS),
        "code_sha256": code_sha256(),
        "convert_version": CONVERT_VERSION,
        "teleop_train": dict(teleop_train_info()),
        "dropped": ({"states": int(raw.dropped_states), "main": int(raw.dropped_frames["main"]),
                     "ultrawide": int(raw.dropped_frames["ultrawide"])} if raw is not None else None),
        "created": _now_utc(),
    }


def _result(episode: str, state: str, status: dict[str, Any]) -> EpisodeResult:
    return EpisodeResult(episode=episode, state=state, reason=status.get("reason") or "",
                         outcome=status.get("outcome"), raw_s=status.get("raw_s"), kept_s=status.get("kept_s"),
                         n_steps=status.get("n_steps"), status=status)


def _write_status(tp: TaskPaths, episode: str, status: dict[str, Any]) -> None:
    data = (json.dumps(status, indent=2, allow_nan=False) + "\n").encode("utf-8")
    paths._write_atomic(tp.status_json(episode), data)


def _build(tp: TaskPaths, episode: str, pose_source: str, pr: _Progress) -> EpisodeArrays:
    """Steps 1-8 of PLAN §5.2 on the raw episode; fills ``pr`` as it goes. Raises the modules' errors."""
    raw = read_episode(tp.raw_episode_dir(episode), pose_source)
    pr.raw, pr.outcome = raw, raw.outcome
    lo, hi = common_window(raw.t_state_ns, raw.main_t_ns, raw.uw_t_ns)
    pr.window = (lo, hi)
    t_grid = make_grid(lo, hi)
    poses = interpolate(raw.t_state_ns, raw.T_state, raw.width_m, t_grid)
    i0, i1 = trim_span(t_grid, moving_mask(poses))
    t_kept = t_grid[i0:i1]
    pr.kept = (int(t_kept[0]), int(t_kept[-1]))
    check_gaps(raw.t_state_ns, raw.main_t_ns, raw.uw_t_ns, int(t_kept[0]), int(t_kept[-1]))
    imgs = prepare_images(t_kept, raw.main_t_ns, raw.main_video_index, raw.uw_t_ns, raw.uw_video_index,
                          raw.main_video, raw.uw_video, raw.rig)
    return EpisodeArrays(name=f"{tp.task}_{episode}", task_label=tp.task, main_rgb=imgs.main_rgb,
                         uw_rgb=imgs.uw_rgb, uw_upsample_local=imgs.uw_upsample_local,
                         eef_pos=poses.pos[i0:i1], eef_rot_axis_angle=poses.rotvec[i0:i1],
                         gripper_width=poses.width[i0:i1])


def convert_episode(tp: TaskPaths, episode: str, pose_source: str, force: bool = False) -> EpisodeResult:
    """Convert one raw episode into ``processed/episode.zarr`` + ``status.json`` (PLAN §5).

    Never raises for a problem with the episode: that gives state "excluded" with a one-line reason. Raises
    PathsError for a bad episode name, ValueError for an unknown pose source, and anything else only for bugs or
    an output folder that cannot be written."""
    if pose_source not in paths.POSE_SOURCES:
        raise ValueError(f"pose_source {pose_source!r}: use one of {', '.join(paths.POSE_SOURCES)}")
    raw_dir = tp.raw_episode_dir(episode)
    meta_sha = _sha256_file(raw_dir / META_FILE)
    zarr_path = tp.episode_zarr(episode)
    status_path = tp.status_json(episode)

    if not force and (raw_dir / COMPLETE_FILE).is_file():
        found = read_status(status_path)
        if is_up_to_date(found, meta_sha, pose_source, zarr_path):
            assert found is not None
            return _result(episode, "up to date", found)

    processed = tp.processed_dir(episode)
    processed.mkdir(parents=True, exist_ok=True)
    try:                                          # a crash from here on leaves no status.json describing old data
        status_path.unlink()
    except FileNotFoundError:
        pass

    pr = _Progress()
    try:
        arrays = _build(tp, episode, pose_source, pr)
    except (EpisodeError, TrajectoryError, ImageError, ValueError) as e:
        # ValueError: EpisodeArrays' validation (EpisodeError / TrajectoryError / ImageError are ValueErrors too)
        if pr.outcome is None:
            pr.outcome = _outcome_from_meta(raw_dir)
        _remove_zarr(zarr_path)
        status = _status(tp, episode, pose_source, meta_sha, "excluded", _one_line(e), pr)
        _write_status(tp, episode, status)
        return _result(episode, "excluded", status)

    write_episode_zarr(arrays, zarr_path)
    pr.n_steps, pr.n_uw = int(arrays.n_steps), int(arrays.n_uw)
    status = _status(tp, episode, pose_source, meta_sha, "converted", "", pr)
    _write_status(tp, episode, status)
    return _result(episode, "converted", status)


def convert_task(tp: TaskPaths, pose_source: str, episodes: set[int] | None = None, force: bool = False,
                 progress: Callable[[EpisodeResult], None] | None = None) -> list[EpisodeResult]:
    """``paths.prepare_out_task(tp)``, then ``convert_episode`` for every raw episode (only the numbers in
    ``episodes`` when given), in episode order. ``progress`` is called with each result as it is done."""
    paths.prepare_out_task(tp)
    names = paths.raw_episodes(tp)
    if episodes is not None:
        names = [n for n in names if paths.episode_number(n) in episodes]
    results = []
    for name in names:
        r = convert_episode(tp, name, pose_source, force)
        if progress is not None:
            progress(r)
        results.append(r)
    return results


# ---- the command ------------------------------------------------------------------------------------------------

_ROW = "{:>7}  {:<7}  {:>7}  {:>7}  {:>6}  {:<10}  {}"


def _fmt_s(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def format_row(r: EpisodeResult) -> str:
    """One output row: number, outcome, raw s, kept s, steps, state, reason."""
    reason = r.reason
    if r.state == "up to date" and r.status.get("state") == "excluded":
        reason = f"excluded: {reason}"
    return _ROW.format(paths.episode_number(r.episode), r.outcome or "-", _fmt_s(r.raw_s), _fmt_s(r.kept_s),
                       "-" if r.n_steps is None else r.n_steps, r.state, reason).rstrip()


def summary_line(results: list[EpisodeResult]) -> str:
    n = {s: sum(1 for r in results if r.state == s) for s in STATES}
    usable = sum(1 for r in results if r.usable)
    return (f"{n['converted']} converted, {n['up to date']} up to date, {n['excluded']} excluded; "
            f"{usable} of {len(results)} usable")


def run(args: argparse.Namespace) -> int:
    """``teleop_train convert``: 0 if at least one episode is usable (converted now, or up to date and converted);
    1 if none is, or on a refusal (one line on stderr)."""
    try:
        wanted = parse_episode_list(args.episodes) if args.episodes is not None else None
        tp = paths.task_paths(args.task, args.raw_root, args.out_root)
    except (PathsError, ValueError) as e:
        print(f"teleop_train convert: {_one_line(e)}", file=sys.stderr)
        return 1

    print(_ROW.format("episode", "outcome", "raw s", "kept s", "steps", "state", "reason").rstrip(), flush=True)
    try:
        results = convert_task(tp, args.pose_source, wanted, bool(args.force),
                               progress=lambda r: print(format_row(r), flush=True))
    except PathsError as e:
        print(f"teleop_train convert: {_one_line(e)}", file=sys.stderr)
        return 1

    if wanted is not None:
        missing = sorted(wanted - {paths.episode_number(r.episode) for r in results})
        if missing:
            shown = ", ".join(str(m) for m in missing[:10]) + (" ..." if len(missing) > 10 else "")
            print(f"teleop_train convert: no raw episode folder for number(s) {shown}", file=sys.stderr)
    if not results:
        print(f"teleop_train convert: no episodes to convert in {tp.raw_task_dir}", file=sys.stderr)
        return 1
    print(summary_line(results), flush=True)
    return 0 if any(r.usable for r in results) else 1
