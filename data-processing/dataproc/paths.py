"""Where things live on the Jetson: the data root, task folders and ``episode_N`` directories.

Layout under the root (``$DATAPROC_ROOT``, default ``~/data``)::

    incoming/<pi_episode>/            raw episodes as the Pi delivers them (docs/CONTRACT.md)
    tasks/<folder>/                   task.yaml, episodes.csv, calibration/, episode_N/{raw,processed}, output/
    rigs/<rig_id>.yaml                rig descriptions (dataproc.config.load_rig)
    quarantine/<pi_episode>/          episodes intake rejected, with reason.json

Everything here is pure path logic: nothing in this module creates, moves or deletes files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_ROOT", "DEFAULT_ROOT", "NAME_RE", "EPISODE_DIR_RE", "DataRoot", "TaskRef", "data_root", "validate_name",
    "validate_episode_dir_name", "episode_index", "episode_dirs", "next_episode_index", "resolve_task",
    "resolve_intake_target",
]

ENV_ROOT = "DATAPROC_ROOT"
DEFAULT_ROOT = Path.home() / "data"             # /home/<user>/data

#: Allowed task, folder and rig names.
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
#: Allowed Pi episode directory names (incoming/, quarantine/, calibration/): no leading dot, so never . or ..
EPISODE_DIR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_EPISODE_RE = re.compile(r"episode_(0|[1-9][0-9]*)")   # no leading zeros: one name per N
_SPEC_RE = re.compile(r"(?P<name>[A-Za-z0-9_-]+)(?:/episode_(?P<ep>0|[1-9][0-9]*))?/?")
_LEADING_ZERO_RE = re.compile(r"episode_0[0-9]+")


@dataclass(frozen=True)
class DataRoot:
    """The data root and its four top-level folders."""

    path: Path

    @property
    def incoming(self) -> Path:
        return self.path / "incoming"

    @property
    def tasks(self) -> Path:
        return self.path / "tasks"

    @property
    def rigs(self) -> Path:
        return self.path / "rigs"

    @property
    def quarantine(self) -> Path:
        return self.path / "quarantine"

    def rig_file(self, rig_id: str) -> Path:
        """``rigs/<rig_id>.yaml`` (``rig_id`` is validated like a task name)."""
        return self.rigs / f"{validate_name(rig_id, 'rig id')}.yaml"


@dataclass(frozen=True)
class TaskRef:
    """A resolved ``--task``: the folder name, its directory, and the episode if one was named."""

    folder: str
    task_dir: Path
    episode: int | None = None

    @property
    def episode_dir(self) -> Path | None:
        return None if self.episode is None else self.task_dir / f"episode_{self.episode}"


def data_root(path: str | os.PathLike | None = None) -> DataRoot:
    """The data root: ``path`` if given, else ``$DATAPROC_ROOT`` if set and non-empty, else ``~/data``."""
    if path is None:
        path = os.environ.get(ENV_ROOT) or DEFAULT_ROOT
    return DataRoot(Path(path).expanduser().absolute())


def validate_name(name: str, what: str = "task name") -> str:
    """Return ``name`` if it matches ``[A-Za-z0-9_-]+``, else raise ``ValueError``."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid {what} {name!r}: use only letters, digits, '_' and '-'")
    return name


def validate_episode_dir_name(name: str) -> str:
    """Return ``name`` if it is a valid Pi episode directory name (``[A-Za-z0-9][A-Za-z0-9._-]*``), else raise
    ``ValueError``."""
    if not isinstance(name, str) or not EPISODE_DIR_RE.fullmatch(name):
        raise ValueError(f"invalid episode directory name {name!r}: start with a letter or digit, then use only "
                         "letters, digits, '.', '_' and '-'")
    return name


def episode_index(path: str | os.PathLike) -> int | None:
    """N for a path whose last component is ``episode_N``, else None."""
    m = _EPISODE_RE.fullmatch(Path(path).name)
    return int(m.group(1)) if m else None


def _episode_entries(task_dir: Path) -> list[tuple[int, Path]]:
    """``(N, path)`` for every entry named ``episode_N`` (dirs, files, symlinks, dangling or not)."""
    if not task_dir.is_dir():
        return []
    found = [(episode_index(p), p) for p in task_dir.iterdir()]
    return sorted((i, p) for i, p in found if i is not None)


def episode_dirs(task_dir: str | os.PathLike) -> list[Path]:
    """The ``episode_N`` directories of a task folder, sorted by N (``episode_2`` before ``episode_10``).

    Returns ``[]`` if ``task_dir`` does not exist. Other entries (``calibration/``, ``output/``, files, dangling
    symlinks) are ignored.
    """
    return [p for _, p in _episode_entries(Path(task_dir)) if p.is_dir()]


def next_episode_index(task_dir: str | os.PathLike) -> int:
    """0 if the task has no ``episode_N`` entry yet, else the largest N + 1. Every entry named ``episode_N`` counts
    as taken (also a file or a dangling symlink), so the result never collides; gaps are not refilled."""
    entries = _episode_entries(Path(task_dir))
    return entries[-1][0] + 1 if entries else 0


def resolve_task(spec: str, root: DataRoot) -> TaskRef:
    """Resolve a ``--task`` value.

    * A bare name (``block_raise``) or ``name/episode_N`` resolves under ``<root>/tasks/``.
    * Anything else is a filesystem path (absolute, ``./...``, ``~/...``) and must be
      ``<root>/tasks/<folder>`` or ``<root>/tasks/<folder>/episode_N``. ``..`` is normalised lexically and the
      parent folder is compared by real path, so a symlinked root works.

    The returned ``task_dir`` is always ``root.tasks / folder``. Raises ``ValueError`` with a user-facing message.
    Does not check that the folder exists.
    """
    if _LEADING_ZERO_RE.fullmatch(Path(spec).name):
        raise ValueError(f"--task {spec!r}: episode numbers have no leading zeros (episode_1, not episode_01)")
    m = _SPEC_RE.fullmatch(spec)
    if m:
        ep = m.group("ep")
        return TaskRef(m.group("name"), root.tasks / m.group("name"), int(ep) if ep is not None else None)
    p = Path(os.path.normpath(Path(spec).expanduser().absolute()))
    episode = episode_index(p)
    task_dir = p.parent if episode is not None else p
    if os.path.realpath(task_dir.parent) != os.path.realpath(root.tasks):
        raise ValueError(f"--task {spec!r}: expected a task name or a path inside {root.tasks} "
                         f"(set ${ENV_ROOT} to use another root)")
    folder = validate_name(task_dir.name)
    return TaskRef(folder, root.tasks / folder, episode)


def resolve_intake_target(tasks_dir: str | os.PathLike, name: str, new: bool) -> str:
    """The folder name intake should use for task ``name``.

    * nothing named ``tasks_dir/<name>`` -> ``name`` (it will be created);
    * it exists and ``new`` is False -> ``name`` (add to it); ``ValueError`` if it is not a directory;
    * it exists and ``new`` is True -> the first of ``name_2``, ``name_3``, ... that is not taken (the first
      folder is ``name`` itself, so the second session of a task is ``name_2``).

    Any entry counts as taken (``os.path.lexists``: also files and dangling symlinks). Creates nothing.
    """
    validate_name(name)
    tasks_dir = Path(tasks_dir)
    target = tasks_dir / name
    if not os.path.lexists(target):
        return name
    if not new:
        if not target.is_dir():
            raise ValueError(f"{target} exists but is not a directory")
        return name
    k = 2
    while os.path.lexists(tasks_dir / f"{name}_{k}"):
        k += 1
    return f"{name}_{k}"
