"""Recordings taken from ``/data/incoming`` into the data root, by kind (docs/DESIGN.md "Where it runs and the data
root"): a session is one task folder with one map recording, one gripper calibration and the demonstrations.

Every recording a command takes is converted to the copied raw format (:mod:`umiproc.dp.pi_adapter`), moved into
its task folder by the copied intake (:mod:`umiproc.dp.intake`), and recorded in ``<root>/pi_sources.csv`` with its
kind, so it is never taken twice and ``run``'s range skips the map and calibration recordings by itself:

==============  ==================================  ===================================================
kind            slot in ``tasks/<folder>/``         taken by
==============  ==================================  ===================================================
map             ``map/<pi_name>/raw``               ``umiproc map``: creates the folder (TASK, TASK_2, ...)
calibration     ``calibration/<pi_name>/raw``       ``umiproc calibrate`` (into the map's folder)
demonstration   ``.staging/<pi_name>/raw``          ``umiproc run`` (then ``episode_N`` or ``failed/<pi_name>``)
==============  ==================================  ===================================================

``pi_name`` is the converted name ``pi-YYYY-MM-DD_episode_NNNNNN`` (:func:`umiproc.dp.pi_adapter.converted_name`).
Each folder also has its own ``episodes.csv`` (the copied intake's ledger, one row per slot, with the same kinds).

Taking is two steps, so a command can check a recording before it lands anywhere (a failed map attaches nothing, a
failed calibration is not recorded, and both can be retried)::

    with session_lock(root):                                  # <root>/.run.lock: one command at a time
        src = inspect_recording(root, Path("/data/incoming"), "episode_000123")   # read only; RecordingError
        staged = stage(root, src, "rig_pi_v3")                # converted copy in <root>/incoming/<pi_name>/
        try:
            ...                                               # build and check the map from staged.path
        except ...:
            discard(staged)                                   # nothing recorded: it can be retried
        taken = take(root, staged, MAP, task="block_raise")   # tasks/block_raise (or block_raise_2, ...)

:func:`take_recording` does all of it in one call (no check in between). The source directory is only ever read.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from umiproc.dp.paths import DataRoot, validate_episode_dir_name, validate_name

__all__ = ["MAP", "CALIBRATION", "DEMONSTRATION", "KINDS", "MAP_DIR", "CALIBRATION_DIR", "SOURCES_FILE",
           "RecordingError", "Staged", "Taken", "session_lock", "ledger_rows", "taken_row", "kind_of",
           "count_by_kind", "not_a_demonstration", "inspect_recording", "stage", "discard", "take", "take_recording",
           "task_folders", "map_slots"]

MAP, CALIBRATION, DEMONSTRATION = "map", "calibration", "demonstration"
KINDS = (MAP, CALIBRATION, DEMONSTRATION)
MAP_DIR, CALIBRATION_DIR = "map", "calibration"
SOURCES_FILE = "pi_sources.csv"
_NOUN = {MAP: "the map recording", CALIBRATION: "the gripper calibration recording",
         DEMONSTRATION: "a demonstration"}


class RecordingError(Exception):
    """The recording cannot be taken (``code`` says why, e.g. ``still_syncing``, ``already_taken``) or the
    command cannot run (``code`` ``usage`` / ``busy`` / ``folder``). Nothing was changed."""

    def __init__(self, message: str, code: str = "usage", details: dict | None = None):
        super().__init__(message)
        self.code, self.details = code, dict(details or {})


@dataclass(frozen=True)
class Staged:
    """A recording converted into ``<root>/incoming/<pi_name>/`` (:func:`stage`), ready for :func:`take`."""

    source: object                 # umiproc.dp.pi_adapter.PiSource (the directory in /data/incoming)
    path: Path                     # <root>/incoming/<pi_name>
    rig_id: str

    @property
    def pi_name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class Taken:
    """Where :func:`take` put a recording. ``status``: ``ingested`` (in ``slot``) or ``quarantined`` (the copied
    intake's validation or duplicate check refused it: ``<root>/quarantine/<pi_name>/reason.json``; ``reasons``).
    Both are recorded in ``pi_sources.csv``."""

    kind: str
    folder: str
    task_dir: Path
    pi_name: str
    status: str
    slot: str | None = None        # e.g. "map/pi-2026-11-16_episode_000123"
    reasons: tuple[str, ...] = ()
    quarantine_dir: Path | None = None
    row: dict | None = None        # the pi_sources.csv row

    @property
    def slot_dir(self) -> Path | None:
        return self.task_dir / self.slot if self.slot else None


def _root(root) -> DataRoot:
    return root if isinstance(root, DataRoot) else DataRoot(Path(root))


@contextmanager
def session_lock(root):
    """Hold ``<root>/.run.lock`` (the copied ``RUN_LOCK``): one command that takes recordings at a time. Raises
    :class:`RecordingError` (code ``busy``) when another one holds it. Creates the root if missing."""
    from umiproc.dp.fsutil import FileLock
    from umiproc.dp.pi_intake import RUN_LOCK

    r = _root(root)
    r.path.mkdir(parents=True, exist_ok=True)
    lock = FileLock(r.path / RUN_LOCK)
    try:
        lock.acquire()
    except BlockingIOError:
        raise RecordingError(f"another umiproc command is working in {r.path} ({lock.path} is locked); wait for it",
                             "busy") from None
    try:
        yield r
    finally:
        lock.release()


# --------------------------------------------------------------------------------------------- the ledger

def ledger_rows(root) -> list[dict]:
    """The rows of ``<root>/pi_sources.csv`` (columns ``umiproc.dp.pi_adapter.SOURCES_COLUMNS``; ``[]`` if none)."""
    from umiproc.dp.pi_adapter import read_sources

    return read_sources(_root(root).path / SOURCES_FILE)


def taken_row(root, manifest_sha256: str) -> dict | None:
    """The ``pi_sources.csv`` row of the recording whose ``MANIFEST.sha256`` hashes to ``manifest_sha256``."""
    for r in ledger_rows(root):
        if r.get("source_manifest_sha256") == manifest_sha256:
            return r
    return None


def kind_of(root, manifest_sha256: str) -> str | None:
    """``map`` | ``calibration`` | ``demonstration`` for a taken recording, None when it was never taken."""
    r = taken_row(root, manifest_sha256)
    return r.get("kind") if r else None


def count_by_kind(root) -> dict[str, int]:
    """Taken recordings per kind (every kind present, also at 0; an unknown kind is counted under its own name)."""
    out = {k: 0 for k in KINDS}
    for r in ledger_rows(root):
        k = r.get("kind") or "?"
        out[k] = out.get(k, 0) + 1
    return out


def not_a_demonstration(row: dict | None) -> str | None:
    """Why ``run`` skips a recording with this ``pi_sources.csv`` row because of its kind (``"the map recording of
    task folder block_raise"``), or None when it is a demonstration or was never taken."""
    if row is None:
        return None
    kind = row.get("kind")
    if kind in (MAP, CALIBRATION):
        return f"{_NOUN[kind]} of task folder {row.get('task')}"
    if kind != DEMONSTRATION:
        return f"recorded as {kind!r} in {SOURCES_FILE}, not a demonstration"
    return None


# --------------------------------------------------------------------------------------------- taking

def inspect_recording(root, source: Path, name: str, *, verify_hashes: bool = True):
    """The Pi recording ``source/name``, checked read-only (``COMPLETE``, every file against ``MANIFEST.sha256``,
    ``meta.json`` valid) and not taken yet. Returns its ``PiSource``; raises :class:`RecordingError` with code
    ``usage`` (bad name / ``source`` not a directory / ``source`` is the root's own ``incoming/``), one of the
    copied calibrate's source codes (``source_missing``, ``still_syncing``, ``manifest_failed``, ``pi_invalid``,
    ``meta_unreadable``) or ``already_taken`` (``details["row"]``: its ``pi_sources.csv`` row)."""
    from umiproc.dp.calibrate.task import source_problem_code
    from umiproc.dp.pi_adapter import inspect_source

    r = _root(root)
    try:
        validate_episode_dir_name(name)
    except ValueError as e:
        raise RecordingError(str(e)) from None
    source = Path(source)
    if not source.is_dir():
        raise RecordingError(f"{source} is not a directory")
    if os.path.realpath(r.incoming) == os.path.realpath(source):
        raise RecordingError(f"{source} is umiproc's own staging folder ({r.incoming}), not where the Pi delivers")
    src = inspect_source(source / name, verify_hashes=verify_hashes)
    if not src.ok:
        raise RecordingError(f"{source / name}: {src.problem}", source_problem_code(src.problem))
    row = taken_row(r, src.manifest_sha256)
    if row is not None:
        raise RecordingError(f"{source / name} was already taken: {not_a_demonstration(row) or 'a demonstration'} "
                             f"({row.get('converted_name')}, {row.get('status')})", "already_taken", {"row": row})
    return src


def stage(root, src, rig_id: str) -> Staged:
    """Convert ``src`` (from :func:`inspect_recording`) into ``<root>/incoming/<pi_name>/`` for ``rig_id``. Call it
    under :func:`session_lock`. ``incoming/`` must be empty (a leftover there is an error naming it). Raises
    :class:`RecordingError` code ``cannot_convert`` (nothing left behind) when the Pi files cannot be converted."""
    from umiproc.dp.pi_intake import PiIntakeError, adapt_sources, clear_staging

    r = _root(root)
    try:
        validate_name(rig_id, "rig id")
        clear_staging(r)
    except (ValueError, PiIntakeError) as e:
        raise RecordingError(str(e)) from None
    adapted, problems = adapt_sources(r, [src], rig_id)
    if problems:
        raise RecordingError(f"{src.path}: {problems[src.name]}", "cannot_convert")
    return Staged(src, r.incoming / next(iter(adapted)), rig_id)


def discard(staged: Staged) -> None:
    """Remove a staged copy that will not be taken (the recording stays untaken and can be retried)."""
    shutil.rmtree(staged.path, ignore_errors=True)


def _existing_folder(r: DataRoot, folder: str | None) -> str:
    if not folder:
        raise RecordingError("a task folder is needed (the one `umiproc map` created)", "folder")
    try:
        validate_name(folder, "task folder")
    except ValueError as e:
        raise RecordingError(str(e), "folder") from None
    if not (r.tasks / folder / "task.yaml").is_file():
        raise RecordingError(f"task folder {r.tasks / folder} does not exist (or has no task.yaml): run `umiproc map` "
                             "first, it creates the folder", "folder")
    return folder


def take(root, staged: Staged, kind: str, *, task: str | None = None, folder: str | None = None,
         log: Callable = print) -> Taken:
    """Move the staged recording into its task folder by ``kind`` (call under :func:`session_lock`):

    * ``map``: ``task`` = the session's label; creates the first free folder ``task``, ``task_2``, ... (its
      ``task.yaml`` label is ``task``, its rig ``staged.rig_id``) with the recording in ``map/<pi_name>/raw``;
    * ``calibration``: into ``calibration/<pi_name>/raw`` of the existing ``folder``;
    * ``demonstration``: into ``.staging/<pi_name>/raw`` of the existing ``folder``.

    Writes the folder's ``episodes.csv`` row and the ``pi_sources.csv`` row (``kind``, folder, ``ingested`` or
    ``quarantined``). Raises :class:`RecordingError` for a bad kind / missing folder (nothing changed) and the copied
    intake's errors (``umiproc.dp.intake.IntakeError``, ``umiproc.dp.pi_intake.PiIntakeError``) otherwise."""
    from umiproc.dp.pi_intake import intake_adapted

    r = _root(root)
    if kind not in KINDS:
        raise RecordingError(f"unknown kind {kind!r} (expected one of {', '.join(KINDS)})")
    if kind == MAP:
        if folder is not None or not task:
            raise RecordingError("a map recording takes the task label (task=), never an existing folder: it creates "
                                 "the session's folder")
        try:
            validate_name(task)
        except ValueError as e:
            raise RecordingError(str(e)) from None
        res = intake_adapted(r, task, {staged.pi_name: staged.source}, MAP, new=True, map_slot=True, log=log)
    else:
        if task is not None:
            raise RecordingError(f"a {kind} recording goes into an existing task folder (folder=), not a new one")
        name = _existing_folder(r, folder)
        res = intake_adapted(r, name, {staged.pi_name: staged.source}, kind, calibration=kind == CALIBRATION,
                             staging=kind == DEMONSTRATION, log=log)
    o = res.outcomes.get(staged.source.name)
    if o is None or o.status not in ("ingested", "quarantined"):
        why = o.describe() if o is not None else "not taken by intake"
        raise RecordingError(f"intake did not take {staged.pi_name} ({why})", "intake")
    row = res.rows[0] if res.rows else None
    return Taken(kind=kind, folder=res.folder, task_dir=r.tasks / res.folder, pi_name=o.pi_name, status=o.status,
                 slot=o.slot, reasons=tuple(o.reasons), quarantine_dir=o.quarantine_dir, row=row)


def take_recording(root, source: Path, name: str, kind: str, *, rig_id: str, task: str | None = None,
                   folder: str | None = None, log: Callable = print) -> Taken:
    """:func:`inspect_recording` + :func:`stage` + :func:`take` under :func:`session_lock`, for a recording that
    needs no check before it lands (a staged copy is removed again if taking it fails)."""
    with session_lock(root) as r:
        src = inspect_recording(r, source, name)
        staged = stage(r, src, rig_id)
        try:
            return take(r, staged, kind, task=task, folder=folder, log=log)
        finally:
            discard(staged)                 # gone already when intake moved it


# --------------------------------------------------------------------------------------------- reading folders

def task_folders(root) -> list[Path]:
    """Every ``tasks/<folder>/`` (a real directory, not a dot-entry), sorted by name."""
    t = _root(root).tasks
    if not t.is_dir():
        return []
    return sorted(p for p in t.iterdir() if not p.name.startswith(".") and p.is_dir() and not p.is_symlink())


def map_slots(task_dir: Path) -> list[Path]:
    """The ``map/<pi_name>/`` entries of a folder that hold a ``raw/`` (one, normally)."""
    d = Path(task_dir) / MAP_DIR
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if not p.name.startswith(".") and (p / "raw").exists())

