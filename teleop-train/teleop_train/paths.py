"""Roots, names and folder layout (PLAN.md §4, §4.1). Owner: E1.

Layout::

    <raw_root>/<task>/task.yaml, episode_NNN/...                   teleop_collect's data_root, read only
    <out_root>/<task>/teleop_source.yaml                            marker: made by teleop_train, from which raw folder
    <out_root>/<task>/task.yaml                                     copied from the raw task folder
    <out_root>/<task>/episode_NNN/processed/episode.zarr            one per converted episode
    <out_root>/<task>/episode_NNN/processed/status.json
    <out_root>/<task>/package/<dataset>/<dataset>.zarr.zip, package.json

The constants and signatures below are the contract with the other modules.

Nothing here writes under the raw root: ``task_paths`` refuses an output task folder that overlaps the raw root, and
``prepare_out_task`` only reads the raw ``task.yaml``.
"""
from __future__ import annotations

import datetime
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RAW_ROOT = Path("~/CapstoneTeleop").expanduser()
DEFAULT_OUT_ROOT = Path("~/Capstone/data").expanduser()
POSE_SOURCES = ("target", "measured")          # O_T_TCP_target, O_T_TCP (PLAN §8)
DEFAULT_POSE_SOURCE = "target"

TASK_NAME_RE = r"^[a-z0-9_]{1,64}$"            # teleop_collect contracts/keys.py
EPISODE_NAME_RE = r"^episode_[0-9]{3,}$"       # teleop_collect contracts/keys.py
DATASET_NAME_RE = r"^[A-Za-z0-9_-]{1,64}$"     # as umiproc / dataproc package
MARKER_FILE = "teleop_source.yaml"
TASK_FILE = "task.yaml"
PROCESSED_DIR = "processed"
EPISODE_ZARR = "episode.zarr"
STATUS_FILE = "status.json"
PACKAGE_DIR = "package"
PACKAGE_JSON = "package.json"

#: the marker's ``source`` value
MARKER_SOURCE = "teleop_collect"

_TASK_NAME = re.compile(TASK_NAME_RE)
_EPISODE_NAME = re.compile(EPISODE_NAME_RE)
_DATASET_NAME = re.compile(DATASET_NAME_RE)
_EPISODE_PREFIX = "episode_"


class PathsError(ValueError):
    """A refusal about names or folders; the message is one line saying what to check."""


def _check_episode(episode: str) -> str:
    if not is_episode_name(episode):
        raise PathsError(f"{episode!r} is not an episode name (episode_NNN); check the episode folder name")
    return episode


@dataclass(frozen=True)
class TaskPaths:
    task: str
    raw_task_dir: Path          # <raw_root>/<task>
    out_task_dir: Path          # <out_root>/<task>

    def raw_episode_dir(self, episode: str) -> Path:
        return self.raw_task_dir / _check_episode(episode)

    def processed_dir(self, episode: str) -> Path:
        """<out_task_dir>/<episode>/processed"""
        return self.out_task_dir / _check_episode(episode) / PROCESSED_DIR

    def episode_zarr(self, episode: str) -> Path:
        return self.processed_dir(episode) / EPISODE_ZARR

    def status_json(self, episode: str) -> Path:
        return self.processed_dir(episode) / STATUS_FILE

    def package_dir(self, dataset: str) -> Path:
        """<out_task_dir>/package/<dataset>"""
        check_dataset_name(dataset)
        return self.out_task_dir / PACKAGE_DIR / dataset

    def zip_path(self, dataset: str) -> Path:
        """<package_dir>/<dataset>.zarr.zip"""
        return self.package_dir(dataset) / f"{dataset}.zarr.zip"

    def package_json(self, dataset: str) -> Path:
        return self.package_dir(dataset) / PACKAGE_JSON

    @property
    def marker(self) -> Path:
        """<out_task_dir>/teleop_source.yaml"""
        return self.out_task_dir / MARKER_FILE


def is_task_name(name: object) -> bool:
    return isinstance(name, str) and _TASK_NAME.fullmatch(name) is not None


def is_episode_name(name: object) -> bool:
    return isinstance(name, str) and _EPISODE_NAME.fullmatch(name) is not None


def episode_number(name: str) -> int:
    """``episode_007`` -> 7. ValueError for a name that is not an episode name."""
    if not is_episode_name(name):
        raise ValueError(f"{name!r} is not an episode name (episode_NNN)")
    return int(name[len(_EPISODE_PREFIX):])


def check_dataset_name(name: str) -> None:
    """PathsError unless the name matches DATASET_NAME_RE."""
    if not isinstance(name, str) or _DATASET_NAME.fullmatch(name) is None:
        raise PathsError(f"dataset name {name!r}: use 1-64 letters, digits, '_' or '-'")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def task_paths(task: str, raw_root: Path | str, out_root: Path | str) -> TaskPaths:
    """Roots expanded (``~``) and resolved. PathsError: bad task name; ``<raw_root>/<task>/task.yaml`` missing;
    an output task folder that overlaps the raw root (it would write under it)."""
    if not is_task_name(task):
        raise PathsError(f"task name {task!r}: use 1-64 lowercase letters, digits or '_' (teleop_collect's names)")
    raw = Path(raw_root).expanduser().resolve()
    out = Path(out_root).expanduser().resolve()
    tp = TaskPaths(task=task, raw_task_dir=raw / task, out_task_dir=out / task)
    task_file = tp.raw_task_dir / TASK_FILE
    if not task_file.is_file():
        raise PathsError(f"{task_file}: missing; check the task name and --raw-root (teleop_collect's data_root)")
    if _is_within(tp.out_task_dir, raw) or _is_within(raw, tp.out_task_dir):
        raise PathsError(f"output folder {tp.out_task_dir} overlaps the raw root {raw} (read only); "
                         f"choose another --out-root")
    return tp


def raw_episodes(tp: TaskPaths) -> list[str]:
    """Names of the raw episode folders (real directories, not symlinks, matching EPISODE_NAME_RE), by number.
    Complete or not: the caller checks COMPLETE."""
    try:
        entries = list(tp.raw_task_dir.iterdir())
    except OSError as e:
        raise PathsError(f"{tp.raw_task_dir}: cannot list ({type(e).__name__}); check --raw-root") from None
    names = [p.name for p in entries if is_episode_name(p.name) and not p.is_symlink() and p.is_dir()]
    return sorted(names, key=episode_number)


def _write_atomic(path: Path, data: bytes) -> None:
    """Replace path with data: a temp file in the same folder, then renamed over it."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_marker(tp: TaskPaths) -> dict[str, Any]:
    """The marker of ``<out_task_dir>`` as a dict. PathsError when it is missing, unreadable, not made by
    teleop_train or names another raw task folder than ``tp.raw_task_dir``."""
    marker = tp.marker
    if not marker.is_file():
        raise PathsError(f"{tp.out_task_dir} exists without {MARKER_FILE} (a handheld folder of the same name?); "
                         f"choose another task name or --out-root")
    try:
        data = yaml.safe_load(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        raise PathsError(f"{marker}: unreadable ({type(e).__name__}); check or remove the folder") from None
    if not isinstance(data, dict) or data.get("source") != MARKER_SOURCE or not isinstance(
            data.get("raw_task_dir"), str):
        raise PathsError(f"{marker}: not a teleop_train marker (source: {MARKER_SOURCE}, raw_task_dir); "
                         f"check the folder")
    named = Path(data["raw_task_dir"]).expanduser().resolve()
    if named != tp.raw_task_dir.resolve():
        raise PathsError(f"{tp.out_task_dir} was made from {data['raw_task_dir']}, not {tp.raw_task_dir}; "
                         f"check --raw-root or choose another --out-root")
    return data


def prepare_out_task(tp: TaskPaths) -> None:
    """Make ``<out_task_dir>`` with MARKER_FILE (``source: teleop_collect``, ``raw_task_dir: <abs path>``,
    ``created``) and a copy of ``task.yaml``; idempotent. PathsError when the folder exists without the marker (a
    handheld folder of the same name) or the marker names another raw folder. Never writes under the raw root.

    The marker is written once and kept; the ``task.yaml`` copy is refreshed whenever the raw one's bytes differ
    (teleop_collect advances ``next_episode`` in it at every recording).
    """
    raw = tp.raw_task_dir.parent.resolve()
    out_dir = tp.out_task_dir
    if _is_within(out_dir.resolve(), raw) or _is_within(raw, out_dir.resolve()):
        raise PathsError(f"output folder {out_dir} overlaps the raw root {raw} (read only); choose another --out-root")
    try:
        task_bytes = (tp.raw_task_dir / TASK_FILE).read_bytes()
    except OSError as e:
        raise PathsError(f"{tp.raw_task_dir / TASK_FILE}: unreadable ({type(e).__name__}); "
                         f"check the task and --raw-root") from None
    created_dir = False
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        raise PathsError(f"{out_dir}: not a real folder; check --out-root")
    if out_dir.exists():
        read_marker(tp)
    else:
        try:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir()
        except OSError as e:
            raise PathsError(f"{out_dir}: cannot create ({type(e).__name__}: {e}); check --out-root") from None
        created_dir = True
        marker = {
            "source": MARKER_SOURCE,
            "raw_task_dir": str(tp.raw_task_dir),
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            _write_atomic(tp.marker, yaml.safe_dump(marker, sort_keys=False, allow_unicode=True).encode("utf-8"))
        except BaseException as e:
            _remove_new_dir(out_dir)
            if isinstance(e, OSError):
                raise PathsError(f"{tp.marker}: cannot write ({type(e).__name__}: {e}); check --out-root") from None
            raise
    copy = out_dir / TASK_FILE
    try:
        if not copy.is_file() or copy.read_bytes() != task_bytes:
            _write_atomic(copy, task_bytes)
    except BaseException as e:
        if created_dir:
            _remove_new_dir(out_dir)
        if isinstance(e, OSError):
            raise PathsError(f"{copy}: cannot write ({type(e).__name__}: {e}); check --out-root") from None
        raise


def _remove_new_dir(out_dir: Path) -> None:
    """Undo a half-made output folder (only the files this module writes), so a retry is not refused."""
    for name in (MARKER_FILE, TASK_FILE):
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass
    try:
        out_dir.rmdir()
    except OSError:
        pass
