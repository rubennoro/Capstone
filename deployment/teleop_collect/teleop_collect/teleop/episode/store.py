"""The task and episode store on disk (PLAN 6.1, 13.4, 13.5, 13.12; STRUCTURE.md 8.1).

```
<data_root>/<task>/task.yaml            name, description, created, next_episode, settings
<data_root>/<task>/episode_NNN/         ... meta.yaml, COMPLETE (written last)
```

- Task names match TASK_NAME_RE; episode names are `episode_%03d` (more digits beyond 999).
- Episode numbers only count up: `next_episode` in task.yaml is advanced (atomically: temp file, then rename) before
  the folder is made, so a deleted or discarded number is never reused, not even the highest.
- An episode is complete (saved) once `COMPLETE` exists; `finish` writes meta.yaml, then COMPLETE, each fsynced.
- A folder without COMPLETE is an unsaved episode: it ended without the operator's choice (the PC stopped the arm,
  Stop arm, a fatal problem, Ctrl-C; `keep_unsaved` writes its meta.yaml with outcome `unsaved`), or a crash left it.
  Nothing deletes it automatically: `Task.label` saves it (outcome success or fail, then COMPLETE), `Task.delete`
  removes it. Training reads only folders with COMPLETE.
- Every removal goes through `_remove_episode_dir`: only a real (not symlinked) `episode_*` folder directly inside its
  own task folder, after resolving symlinks. `TaskStore.delete_task` removes a whole task the same way: only a real
  folder with a valid task.yaml directly inside data_root.
- `probe_writable`: data_root is made if missing and a file can be written in it (start-up step 9, the page's storage
  folder change).
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from teleop_collect.contracts import (LISTED_OUTCOMES, SAVED_OUTCOMES, TASK_NAME_RE, UNSAVED, is_episode_name,
                                      is_task_name)

log = logging.getLogger(__name__)

#: libyaml's safe loader (about 10x faster) when PyYAML was built with it; the pure-Python one otherwise.
_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

TASK_FILE = "task.yaml"
WRITE_PROBE = ".teleop_collect_write_test"
META_FILE = "meta.yaml"
COMPLETE_FILE = "COMPLETE"
EPISODE_PREFIX = "episode_"


class TaskError(ValueError):
    """A task or episode request that cannot be done; the message is one line for the page."""


@dataclass(frozen=True)
class EpisodeInfo:
    """One episode, read from its meta.yaml (E3 turns it into the U1 EpisodeInfo record). outcome is `unsaved` for
    a folder without COMPLETE (duration_s and frames_main 0 when its meta.yaml is missing or broken)."""

    name: str
    outcome: str
    duration_s: float
    frames_main: int


def episode_name(number: int) -> str:
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ValueError(f"episode number {number!r} must be an int >= 0")
    return f"{EPISODE_PREFIX}{number:03d}"


def episode_number(name: str) -> int:
    if not is_episode_name(name):
        raise ValueError(f"{name!r} is not an episode name")
    return int(name[len(EPISODE_PREFIX):])


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_file_synced(path: Path, text: str) -> None:
    """Write text to a new file (refused if it exists), flush and fsync it and its folder."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(path.parent)


def write_yaml_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Replace path with data: a temp file in the same folder, fsynced, then renamed over it."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.fchmod(fd, 0o666 & ~umask)                    # mkstemp makes 0600: the usual mode, as open() would
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(path.parent)


def probe_writable(root: Path) -> None:
    """Make root (and its parents) if missing, then write, fsync and remove a small file in it. OSError when not
    (NotADirectoryError "not a folder" when root is a file)."""
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    probe = root / WRITE_PROBE
    root.mkdir(parents=True, exist_ok=True)
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok\n")
        f.flush()
        os.fsync(f.fileno())
    probe.unlink()


def _remove_episode_dir(task_dir: Path, path: Path) -> None:
    """Delete path (and everything in it) only if it is a real episode_* folder directly inside task_dir."""
    task_real = task_dir.resolve()
    if path.is_symlink():
        raise TaskError(f"{path}: a symlink, refused (only real episode folders are deleted)")
    real = path.resolve()
    if real.parent != task_real or not is_episode_name(real.name):
        raise TaskError(f"{path}: not an episode folder of {task_real}, refused")
    if not real.exists():
        return
    if not real.is_dir():
        raise TaskError(f"{real}: not a folder, refused")
    shutil.rmtree(real)


def _read_task_yaml(task_dir: Path) -> dict | None:
    """task.yaml as a dict if it is valid for this folder, else None."""
    try:
        data = yaml.load((task_dir / TASK_FILE).read_text(encoding="utf-8"), Loader=_SAFE_LOADER)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict) or data.get("name") != task_dir.name:
        return None
    nxt = data.get("next_episode")
    if isinstance(nxt, bool) or not isinstance(nxt, int) or nxt < 0:
        return None
    if not isinstance(data.get("description", ""), str):
        return None
    return data


class EpisodeDir:
    """One episode folder: made by Task.new_episode, then finished (saved) or discarded."""

    def __init__(self, task_dir: Path, name: str) -> None:
        if not is_episode_name(name):
            raise TaskError(f"{name!r} is not an episode name")
        self.task_dir = Path(task_dir)
        self.name = name
        self.path = self.task_dir / name

    @property
    def number(self) -> int:
        return episode_number(self.name)

    @property
    def complete(self) -> bool:
        return (self.path / COMPLETE_FILE).exists()

    def finish(self, meta: Mapping[str, Any]) -> None:
        """Write meta.yaml, then COMPLETE (each fsynced): the episode is saved."""
        if not self.path.is_dir():
            raise TaskError(f"{self.path}: missing, cannot finish")
        if self.complete:
            raise TaskError(f"{self.name}: already complete")
        write_file_synced(self.path / META_FILE,
                          yaml.safe_dump(dict(meta), sort_keys=False, allow_unicode=True))
        write_file_synced(self.path / COMPLETE_FILE, "")

    def keep_unsaved(self, meta: Mapping[str, Any]) -> None:
        """Write meta.yaml (outcome unsaved) without COMPLETE: the episode stays until it is labeled or deleted."""
        if not self.path.is_dir():
            raise TaskError(f"{self.path}: missing, cannot keep")
        if self.complete:
            raise TaskError(f"{self.name}: already complete")
        write_yaml_atomic(self.path / META_FILE, meta)

    def discard(self) -> None:
        """Delete the folder (nothing outside its task folder is ever touched). No-op if it is already gone."""
        _remove_episode_dir(self.task_dir, self.path)

    def __repr__(self) -> str:
        return f"EpisodeDir({str(self.path)!r})"


class Task:
    """One task folder with a valid task.yaml."""

    def __init__(self, task_dir: Path, description: str) -> None:
        self.dir = Path(task_dir)
        self.name = self.dir.name
        self.description = description
        self._info_cache: dict[str, tuple[tuple, EpisodeInfo | None]] = {}   # folder name -> (stat key, info)

    def _episode_dirs(self) -> list[Path]:
        try:
            entries = list(self.dir.iterdir())
        except FileNotFoundError:
            return []
        return [p for p in entries if is_episode_name(p.name) and not p.is_symlink() and p.is_dir()]

    def episodes(self) -> list[EpisodeInfo]:
        """Every episode, by number: the complete ones (one with an unreadable meta.yaml is skipped and logged once)
        and the unsaved ones (no COMPLETE: listed as `unsaved` whatever their meta.yaml says).

        Cheap enough for the 20 ms loop: each meta.yaml is parsed once and its EpisodeInfo cached by folder name,
        keyed on the stat of COMPLETE and meta.yaml; a later call only lists the folder and stats two files each.
        """
        out = []
        seen = set()
        for path in sorted(self._episode_dirs(), key=lambda p: episode_number(p.name)):
            try:
                complete_ns: int | None = os.stat(path / COMPLETE_FILE).st_mtime_ns
            except OSError:
                complete_ns = None
            try:
                m = os.stat(path / META_FILE)
                key: tuple = (complete_ns, m.st_mtime_ns, m.st_size, m.st_ino)
            except OSError:
                key = (complete_ns, None)
            seen.add(path.name)
            cached = self._info_cache.get(path.name)
            if cached is not None and cached[0] == key:
                info = cached[1]
            else:
                info = read_episode_info(path)
                if complete_ns is None:
                    info = (EpisodeInfo(path.name, UNSAVED, 0.0, 0) if info is None
                            else dataclasses.replace(info, outcome=UNSAVED))
                elif info is not None and info.outcome not in SAVED_OUTCOMES:
                    info = None
                self._info_cache[path.name] = (key, info)
                if info is None:
                    log.warning("%s: COMPLETE but meta.yaml unreadable; not listed", path)
            if info is not None:
                out.append(info)
        for name in set(self._info_cache) - seen:                 # deleted folders
            del self._info_cache[name]
        return out

    def new_episode(self) -> EpisodeDir:
        """The next episode folder: next_episode from task.yaml (never below any folder already there), advanced on
        disk before the folder is made, so its number is never handed out again."""
        data = _read_task_yaml(self.dir)
        if data is None:
            raise TaskError(f"{self.dir / TASK_FILE}: missing or invalid")
        number = data["next_episode"]
        existing = [episode_number(p.name) for p in self._episode_dirs()]
        if existing:
            number = max(number, max(existing) + 1)
        data["next_episode"] = number + 1
        write_yaml_atomic(self.dir / TASK_FILE, data)
        episode = EpisodeDir(self.dir, episode_name(number))
        episode.path.mkdir()
        _fsync_dir(self.dir)
        return episode

    def delete(self, episode: str) -> None:
        """Delete a saved or an unsaved episode (the caller never passes the one recording). TaskError when the name
        is bad or it is missing."""
        path = self._episode_path(episode)
        _remove_episode_dir(self.dir, path)

    def label(self, episode: str, outcome: str) -> EpisodeInfo:
        """Save an unsaved episode: its meta.yaml outcome set to success or fail (atomically), then COMPLETE. TaskError
        when it is missing, already saved, or has no readable meta.yaml (a crash: only delete)."""
        if outcome not in SAVED_OUTCOMES:
            raise TaskError(f"outcome {outcome!r}: only {', '.join(SAVED_OUTCOMES)}")
        path = self._episode_path(episode)
        if (path / COMPLETE_FILE).exists():
            raise TaskError(f"{self.name}/{episode}: already saved")
        try:
            meta = yaml.load((path / META_FILE).read_text(encoding="utf-8"), Loader=_SAFE_LOADER)
        except (OSError, yaml.YAMLError) as exc:
            raise TaskError(f"{self.name}/{episode}: no readable meta.yaml (the run crashed): it can only be "
                            f"deleted ({type(exc).__name__})") from None
        if not isinstance(meta, dict) or meta.get("outcome") not in LISTED_OUTCOMES:
            raise TaskError(f"{self.name}/{episode}: meta.yaml is broken: it can only be deleted")
        meta["outcome"] = outcome
        write_yaml_atomic(path / META_FILE, meta)
        write_file_synced(path / COMPLETE_FILE, "")
        info = read_episode_info(path)
        if info is None:
            raise TaskError(f"{self.name}/{episode}: saved, but meta.yaml does not read back")
        return info

    def _episode_path(self, episode: str) -> Path:
        if not is_episode_name(episode):
            raise TaskError(f"{episode!r} is not an episode name")
        path = self.dir / episode
        if not path.is_dir() or path.is_symlink():
            raise TaskError(f"{self.name}/{episode}: no such episode")
        return path

    def __repr__(self) -> str:
        return f"Task({self.name!r})"


def read_episode_info(path: Path) -> EpisodeInfo | None:
    """EpisodeInfo from <path>/meta.yaml, or None when it is missing or broken."""
    try:
        meta = yaml.load((path / META_FILE).read_text(encoding="utf-8"), Loader=_SAFE_LOADER)
        outcome = meta["outcome"]
        duration = meta["duration_s"]
        frames_main = meta["counts"]["frames_main"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return None
    if outcome not in LISTED_OUTCOMES or isinstance(duration, bool) or not isinstance(duration, (int, float)) \
            or isinstance(frames_main, bool) or not isinstance(frames_main, int) or frames_main < 0 \
            or not duration >= 0:
        return None
    return EpisodeInfo(path.name, outcome, float(duration), frames_main)


class TaskStore:
    """Every task under data_root."""

    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root).expanduser()

    def tasks(self) -> list[str]:
        """Sorted names of the folders with a valid task.yaml ([] when root is missing or not a folder: start-up step 9
        then reports it)."""
        try:
            entries = list(self.root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return []
        return sorted(p.name for p in entries
                      if is_task_name(p.name) and p.is_dir() and _read_task_yaml(p) is not None)

    def create(self, name: str, description: str, settings: Mapping[str, Any]) -> Task:
        """A new task folder with its task.yaml. TaskError: bad name, bad description, exists."""
        if not is_task_name(name):
            raise TaskError(f"{name!r}: a task name is 1-64 lowercase letters, digits or _ ({TASK_NAME_RE})")
        if not isinstance(description, str) or re.search(r"[\r\n]", description):
            raise TaskError("the description is one line")
        description = description.strip()
        task_dir = self.root / name
        if task_dir.exists() or task_dir.is_symlink():
            raise TaskError(f"task {name} already exists")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            task_dir.mkdir()
        except FileExistsError:
            raise TaskError(f"task {name} already exists") from None
        data = {
            "name": name,
            "description": description,
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "next_episode": 0,
            "settings": to_plain(settings),
        }
        write_yaml_atomic(task_dir / TASK_FILE, data)
        return Task(task_dir, description)

    def open(self, name: str) -> Task:
        """TaskError when the name is bad or there is no valid task of that name."""
        if not is_task_name(name):
            raise TaskError(f"{name!r} is not a task name")
        task_dir = self.root / name
        data = _read_task_yaml(task_dir) if task_dir.is_dir() else None
        if data is None:
            raise TaskError(f"no task {name} under {self.root}")
        return Task(task_dir, data.get("description", "") or "")

    def delete_task(self, name: str) -> int:
        """Delete a task folder and every episode in it; returns how many episodes it held. TaskError when the name is
        bad, there is no valid task of that name, or it is a symlink (only a real folder directly inside root)."""
        if not is_task_name(name):
            raise TaskError(f"{name!r} is not a task name")
        task_dir = self.root / name
        if task_dir.is_symlink():
            raise TaskError(f"{task_dir}: a symlink, refused (only real task folders are deleted)")
        task = self.open(name)
        real = task_dir.resolve()
        if real.parent != self.root.resolve() or real.name != name:
            raise TaskError(f"{task_dir}: not a task folder of {self.root}, refused")
        count = len(task._episode_dirs())
        shutil.rmtree(real)
        return count

    def unsaved(self) -> list[Path]:
        """Every episode_* folder without COMPLETE in every task (kept: never deleted automatically)."""
        found = []
        for name in self.tasks():
            task = Task(self.root / name, "")
            for path in sorted(task._episode_dirs(), key=lambda p: episode_number(p.name)):
                if not (path / COMPLETE_FILE).exists():
                    found.append(path)
        return found


def to_plain(value: Any) -> Any:
    """YAML-safe plain types: dicts, lists, str, int, float, bool, None (numpy, Paths, contract messages and
    dataclasses converted; a message's `schema` key is kept)."""
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_plain(dataclasses.asdict(value))
    raise TypeError(f"not storable in YAML: {type(value).__name__}")
