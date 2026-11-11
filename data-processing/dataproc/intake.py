"""``dataproc intake``: move verified raw episodes from ``incoming/`` into a task folder (docs/DESIGN.md "intake",
decisions in docs/INTAKE.md).

::

    result = run_intake(data_root(), "block_raise", new=False, calibration=False)
    print(format_summary(result))

One run is :meth:`Intake.run`, which reads as the list of steps::

    with self.lock():                         # <root>/.intake.lock (flock, non-blocking; held -> LockHeldError)
        self.preflight_root()                 # incoming/, tasks/, quarantine/ must be writable
        self.reconcile()                      # rebuild ledger rows / reason.json lost in a crash
        candidates = self.scan_incoming()     # dirs with .verified and no other dot-entry; symlinks too
        self.open_target()                    # resolve + check the task folder, load the duplicate index
        for candidate in self.in_order(candidates):
            self.handle(candidate)            # validate -> re-check transfer -> duplicate -> rig -> move

and :meth:`Intake.handle` reads as the per-episode steps. Collaborators: :class:`Ledger` (one ``episodes.csv``),
:class:`Quarantine` (``quarantine/<name>/raw`` + ``reason.json``), :class:`TaskTarget` (the target folder, made
lazily and rolled back after a failed first ingest) and :class:`DuplicateIndex` (names and session/episode pairs
already taken). Generic filesystem plumbing lives in :mod:`dataproc.fsutil`.

**Staging target** (``staging=True``, used by ``dataproc run``): the folder must already exist with its
``task.yaml`` (it is never created), and each episode goes to ``tasks/<folder>/.staging/<pi_name>/raw`` instead of
``episode_N``. ``dataproc run`` later moves it on with :func:`move_slot` (to ``episode_N`` when it passed every
stage, to ``failed/<pi_name>`` otherwise), which appends the new slot's ledger row. ``episodes.csv`` stays
append-only: the newest row of a recording is where it is now; :meth:`Intake.reconcile` knows all four slot kinds.

Nothing is ever deleted or copied; raw files are never opened for writing.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from dataproc import __version__, fsutil
from dataproc.config import ConfigError, TaskConfig, dump_task_config, load_task_config
from dataproc.paths import (FAILED_DIR, STAGING_DIR, DataRoot, episode_dirs, next_episode_index,
                            resolve_intake_target, validate_name)
from dataproc.raw_episode import MANIFEST, VERIFIED, EpisodeMeta, Problem, RawEpisode, RawEpisodeError, \
    ValidationReport

__all__ = [
    "LOCK_NAME", "LEDGER_NAME", "LEDGER_COLUMNS", "TASK_YAML", "PENDING_NAME", "REASON_NAME", "IntakeError",
    "LockHeldError", "EpisodeOutcome", "IntakeResult", "run_intake", "read_ledger", "format_summary", "cli_intake",
    "reconcile", "move_slot",
]

LOCK_NAME = ".intake.lock"
LEDGER_NAME = "episodes.csv"
LEDGER_COLUMNS = ("slot", "pi_name", "session_id", "episode_number", "kind", "rig_id", "start_unix_ns",
                  "intake_unix_ns", "manifest_sha256")
TASK_YAML = "task.yaml"
#: Written (atomically) in the task folder right before each move: {"slot", "pi_name", "kind"}. Lets reconcile
#: recover the Pi dir name of a move whose ledger row was never written (the name is lost once it is ``raw``).
PENDING_NAME = ".intake_pending.json"
REASON_NAME = "reason.json"

DEMONSTRATION, CALIBRATION = "demonstration", "calibration"


class IntakeError(Exception):
    """Intake cannot run (target folder problem, EXDEV, unreadable ledger, ...). The CLI exits 1."""


class LockHeldError(IntakeError):
    """Another intake holds ``<root>/.intake.lock``."""


@dataclass
class EpisodeOutcome:
    """What happened to one ``incoming/`` entry.

    ``status``: ``ingested`` (``slot`` set), ``quarantined`` (``reasons``, ``quarantine_dir`` set), ``skipped``
    (left in incoming: ``note`` says why) or ``waiting`` (a valid candidate left for later by ``--calibration``).
    """

    pi_name: str
    status: str
    slot: str | None = None
    reasons: tuple[str, ...] = ()
    quarantine_dir: Path | None = None
    note: str = ""

    def describe(self) -> str:
        if self.status == "ingested":
            return self.slot or ""
        if self.status == "quarantined":
            return f"quarantined: {', '.join(self.reasons)}"
        return f"{self.status}: {self.note}" if self.note else self.status


@dataclass
class IntakeResult:
    folder: str | None                      # the target folder name (None if nothing was waiting)
    task_dir: Path | None
    outcomes: list[EpisodeOutcome] = field(default_factory=list)
    reconciled: list[tuple[str, str]] = field(default_factory=list)   # (folder, slot)
    created_folder: bool = False
    incoming_missing: bool = False

    @property
    def nothing_waiting(self) -> bool:
        return not any(o.status in ("ingested", "quarantined", "waiting", "error") for o in self.outcomes)

    def by_status(self, status: str) -> list[EpisodeOutcome]:
        return [o for o in self.outcomes if o.status == status]


# --------------------------------------------------------------------------------------------- small helpers

def _now_ns() -> int:
    return time.time_ns()


def _iso_utc(ns: int) -> str:
    return _dt.datetime.fromtimestamp(ns / 1e9, tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _move(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst`` with ``os.rename`` only; ``EXDEV`` becomes an :class:`IntakeError` (never copies)."""
    try:
        fsutil.rename_no_copy(src, dst)
    except fsutil.CrossDeviceError:
        raise IntakeError(f"cannot move {src} to {dst}: different filesystems (EXDEV). incoming/, tasks/ and "
                          "quarantine/ must be on one filesystem; nothing is ever copied") from None


def _require_writable(checks: list[tuple[Path, str]]) -> None:
    """Raise :class:`IntakeError` for the first ``(dir, description)`` this process cannot write to."""
    for d, what in checks:
        why = fsutil.not_writable(d)
        if why:
            raise IntakeError(f"permission denied: {d} ({what}) is not writable by {fsutil.checked_uid()} ({why}); "
                              "nothing was moved or created")


def _manifest_sha(ep_dir: Path) -> str:
    m = ep_dir / MANIFEST
    return fsutil.sha256_file(m) if fsutil.is_regular_file(m) else ""


def _meta_of(ep_dir: Path) -> EpisodeMeta | None:
    try:
        return RawEpisode.open(ep_dir).meta
    except (OSError, RawEpisodeError):
        return None


def _safe_meta(ep: RawEpisode | None) -> EpisodeMeta | None:
    if ep is None:
        return None
    try:
        return ep.meta
    except RawEpisodeError:
        return None


def _dot_entries(d: Path) -> list[str]:
    """Dot-entries other than ``.verified`` in ``d`` (a transfer in progress), sorted; ``[]`` if unreadable."""
    try:
        return sorted(e for e in os.listdir(d) if e.startswith(".") and e != VERIFIED)
    except OSError:
        return []


def _error_codes(report: ValidationReport) -> list[str]:
    """The distinct error codes of a report, in report order (warnings are not reasons)."""
    out = []
    for p in report.errors:
        if p.code not in out:
            out.append(p.code)
    return out


def _task_folders(root: DataRoot) -> list[Path]:
    if not root.tasks.is_dir():
        return []
    return sorted(p for p in root.tasks.iterdir()
                  if not p.name.startswith(".") and p.is_dir() and not p.is_symlink())


def _slots_on_disk(task_dir: Path) -> list[tuple[str, str, Path]]:
    """``(slot, kind, raw_dir)`` for every ``episode_N/raw``, ``calibration/<x>/raw``, ``.staging/<x>/raw`` and
    ``failed/<x>/raw`` present (the last two are demonstrations ``dataproc run`` has not finished / that failed)."""
    out = []
    for d in episode_dirs(task_dir):
        if os.path.lexists(d / "raw"):
            out.append((d.name, DEMONSTRATION, d / "raw"))
    for sub, kind in (("calibration", CALIBRATION), (STAGING_DIR, DEMONSTRATION), (FAILED_DIR, DEMONSTRATION)):
        parent = task_dir / sub
        if parent.is_dir() and not parent.is_symlink():
            for d in sorted(parent.iterdir()):
                if not d.name.startswith(".") and os.path.lexists(d / "raw"):
                    out.append((f"{sub}/{d.name}", kind, d / "raw"))
    return out


def _named_slot(slot: str) -> bool:
    """True for ``calibration/<x>``, ``.staging/<x>`` and ``failed/<x>``: slots whose dir name is the Pi name."""
    return slot.split("/", 1)[0] in ("calibration", STAGING_DIR, FAILED_DIR) and "/" in slot


def _read_pending(task_dir: Path) -> dict:
    try:
        d = json.loads((task_dir / PENDING_NAME).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------------------------- the ledger

def read_ledger(path: str | os.PathLike) -> list[dict]:
    """The rows of an ``episodes.csv`` (``[]`` if it does not exist). Raises :class:`IntakeError` on a bad header."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        if tuple(header) != LEDGER_COLUMNS:
            raise IntakeError(f"{path}: unexpected header {header!r} (expected {','.join(LEDGER_COLUMNS)})")
        rows = []
        for i, r in enumerate(reader, start=2):
            if not r:
                continue
            if len(r) != len(LEDGER_COLUMNS):
                raise IntakeError(f"{path} line {i}: expected {len(LEDGER_COLUMNS)} fields, got {len(r)}")
            rows.append(dict(zip(LEDGER_COLUMNS, r)))
        return rows


class Ledger:
    """One task folder's append-only ``episodes.csv``."""

    def __init__(self, task_dir: Path):
        self.path = task_dir / LEDGER_NAME

    def rows(self) -> list[dict]:
        return read_ledger(self.path)

    def slots(self) -> set[str]:
        return {r["slot"] for r in self.rows()}

    @staticmethod
    def make_row(slot: str, pi_name: str, meta: EpisodeMeta | None, kind: str, intake_ns: int,
                 manifest_sha256: str) -> dict:
        return {
            "slot": slot, "pi_name": pi_name,
            "session_id": meta.session_id if meta else "",
            "episode_number": str(meta.episode_number) if meta else "",
            "kind": kind,
            "rig_id": meta.rig_id if meta else "",
            "start_unix_ns": str(meta.start_unix_ns) if meta else "",
            "intake_unix_ns": str(intake_ns),
            "manifest_sha256": manifest_sha256,
        }

    def append(self, row: dict) -> None:
        """Append one full line (with the header first if the file is new or empty), then flush + fsync."""
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerow([row.get(c, "") for c in LEDGER_COLUMNS])
            data = buf.getvalue()
            if os.fstat(fd).st_size == 0:
                data = ",".join(LEDGER_COLUMNS) + "\n" + data
            view = data.encode("utf-8")
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        fsutil.fsync_dir(self.path.parent)


# --------------------------------------------------------------------------------------------- collaborators

@dataclass
class Candidate:
    """One ``incoming/`` entry that will be validated (a verified dir, or any symlink)."""

    name: str
    path: Path
    ep: RawEpisode | None            # None for a symlink that is not a directory
    meta: EpisodeMeta | None

    @property
    def sort_key(self):
        m = self.meta
        return (m.start_unix_ns, m.session_id, m.episode_number, self.name)


class DuplicateIndex:
    """Every Pi dir name and (session_id, episode_number) already taken in by any task (plus this run)."""

    def __init__(self, root: DataRoot):
        self.names: dict[str, str] = {}
        self.pairs: dict[tuple[str, str], str] = {}
        for task_dir in _task_folders(root):
            for r in Ledger(task_dir).rows():
                self.add(r["pi_name"], r["session_id"], r["episode_number"], f"{task_dir.name}/{r['slot']}")

    def add(self, name: str, session_id: str, episode_number: str, where: str) -> None:
        if name:
            self.names.setdefault(name, where)
        if session_id != "" and episode_number != "":
            self.pairs.setdefault((session_id, str(episode_number)), where)

    def check(self, name: str, meta: EpisodeMeta | None) -> dict | None:
        """The ``details["duplicate"]`` entry for this episode, or None if it is new."""
        d = {}
        if name in self.names:
            d["pi_name"] = {"value": name, "taken_by": self.names[name]}
        if meta is not None:
            key = (meta.session_id, str(meta.episode_number))
            if key in self.pairs:
                d["session_episode"] = {"value": list(key), "taken_by": self.pairs[key]}
        return d or None


class Quarantine:
    """``<root>/quarantine/<name>[__k]/``: the episode moved in as ``raw/`` plus a ``reason.json``."""

    def __init__(self, root: DataRoot):
        self.dir = root.quarantine

    def put(self, c: Candidate, reasons: list[str], report: ValidationReport, details: dict, target_task: str,
            mode: str) -> EpisodeOutcome:
        """Create the wrapper dir, move the episode in as ``raw/`` (on failure remove the still-empty wrapper),
        then write ``reason.json`` atomically."""
        self.dir.mkdir(parents=True, exist_ok=True)
        qdir = fsutil.free_name(self.dir, c.name)
        os.mkdir(qdir)
        now = _now_ns()
        reason = {
            "pi_name": c.name,
            "reasons": reasons,
            "details": details,
            "report": report.to_dict(),
            "quarantined_at": _iso_utc(now),
            "quarantined_unix_ns": now,
            "dataproc_version": __version__,
            "target_task": target_task,
            "mode": mode,
            "episode": "raw/",
        }
        try:
            _move(c.path, qdir / "raw")
        except BaseException:
            fsutil.rmdir_if_empty(qdir)             # no reason-only orphan folder
            raise
        fsutil.write_atomic(qdir / REASON_NAME, json.dumps(reason, indent=2) + "\n")
        return EpisodeOutcome(c.name, "quarantined", reasons=tuple(reasons), quarantine_dir=qdir)

    def repair_missing_reasons(self, log) -> None:
        """Give every ``quarantine/<x>/raw`` without a ``reason.json`` (a crash between the move and the write) one
        with reason ``interrupted``."""
        if not self.dir.is_dir():
            return
        for q in sorted(self.dir.iterdir()):
            if (not q.name.startswith(".") and q.is_dir() and not q.is_symlink() and os.path.lexists(q / "raw")
                    and not os.path.lexists(q / REASON_NAME)):
                now = _now_ns()
                fsutil.write_atomic(q / REASON_NAME, json.dumps({
                    "pi_name": q.name, "reasons": ["interrupted"],
                    "details": {"interrupted": "intake stopped between the move and writing reason.json; the "
                                               "original reasons are unknown (re-run validation on raw/)"},
                    "report": None, "quarantined_at": _iso_utc(now), "quarantined_unix_ns": now,
                    "dataproc_version": __version__, "target_task": None, "mode": None, "episode": "raw/",
                }, indent=2) + "\n")
                log(f"reconciled: quarantine/{q.name} had no {REASON_NAME} (interrupted intake); wrote one with "
                    "reason 'interrupted'")


class TaskTarget:
    """The task folder, created lazily on the first episode that is ingested."""

    def __init__(self, root: DataRoot, label: str, folder: str):
        self.root, self.label, self.folder = root, label, folder
        self.dir = root.tasks / folder
        self.ledger = Ledger(self.dir)
        self.created = False
        self.rig: str | None = None
        self._ready = False
        self._made: list[Path] = []          # files/dirs inside the folder that this run created (for rollback)
        self._made_tasks_dir = False
        if self.dir.is_dir():
            ty = self.dir / TASK_YAML
            if os.path.lexists(ty):
                try:
                    self.rig = load_task_config(ty).rig
                except ConfigError as e:
                    raise IntakeError(f"target folder {self.dir}: {e}") from None
                self._ready = True
            elif self._has_episodes():
                raise IntakeError(f"target folder {self.dir} has episodes but no {TASK_YAML}; fix it by hand")
        elif os.path.lexists(self.dir):
            raise IntakeError(f"{self.dir} exists but is not a directory")

    def _has_episodes(self) -> bool:
        return bool(_slots_on_disk(self.dir)) or next_episode_index(self.dir) > 0

    def ensure(self, rig_id: str) -> None:
        """Create the folder and ``task.yaml`` (atomically) if they do not exist yet."""
        if self._ready:
            return
        if not self.root.tasks.is_dir():
            self.root.tasks.mkdir(parents=True)
            self._made_tasks_dir = True
        if not self.dir.is_dir():
            os.mkdir(self.dir)
            self.created = True
            fsutil.fsync_dir(self.root.tasks)
        cfg = TaskConfig(label=self.label, rig=rig_id,
                         created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), notes="")
        tmp = fsutil.tmp_path(self.dir / TASK_YAML)
        dump_task_config(cfg, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp, self.dir / TASK_YAML)
        self._made.append(self.dir / TASK_YAML)
        fsutil.fsync_dir(self.dir)
        self.rig = rig_id
        self._ready = True

    def next_slot(self, pi_name: str, calibration: bool, staging: bool = False) -> tuple[Path, str, str]:
        """``(slot_dir, slot, kind)`` for the next episode: ``episode_N`` (N = next free number), with
        ``calibration`` ``calibration/<pi_name>[__k]`` (creating ``calibration/`` if missing), or with ``staging``
        ``.staging/<pi_name>`` (creating ``.staging/`` if missing; an existing one is an :class:`IntakeError`).
        Not created yet."""
        if staging:
            st = self.dir / STAGING_DIR
            if not os.path.lexists(st):
                os.mkdir(st)
                self.note_made(st)
            slot_dir = st / pi_name
            if os.path.lexists(slot_dir):
                raise IntakeError(f"{slot_dir} already exists (an unfinished dataproc run of the same recording)")
            return slot_dir, f"{STAGING_DIR}/{pi_name}", DEMONSTRATION
        if calibration:
            cal = self.dir / "calibration"
            if not os.path.lexists(cal):
                os.mkdir(cal)
                self.note_made(cal)
            slot_dir = fsutil.free_name(cal, pi_name)
            return slot_dir, f"calibration/{slot_dir.name}", CALIBRATION
        slot_dir = self.dir / f"episode_{next_episode_index(self.dir)}"
        return slot_dir, slot_dir.name, DEMONSTRATION

    def write_pending(self, slot: str, pi_name: str, kind: str) -> None:
        """Atomically record the move about to happen in ``.intake_pending.json`` (read by reconcile)."""
        pending = self.dir / PENDING_NAME
        fresh_pending = not os.path.lexists(pending)
        fsutil.write_atomic(pending, json.dumps({"slot": slot, "pi_name": pi_name, "kind": kind}) + "\n")
        if fresh_pending:
            self.note_made(pending)

    def note_made(self, path: Path) -> None:
        self._made.append(path)

    def rollback_if_unused(self) -> None:
        """After a failed first ingest: undo what this run created in the folder (``task.yaml``, the pending
        record, an empty ``calibration/``, the folder itself if this run made it and it is then empty). Only when
        this run wrote ``task.yaml`` and the folder still holds no episode; never touches episode data."""
        if not any(p.name == TASK_YAML for p in self._made) or self._has_episodes():
            return
        for p in reversed(self._made):
            try:
                if p.is_dir() and not p.is_symlink():
                    os.rmdir(p)                  # only if empty
                else:
                    os.unlink(p)
            except OSError:
                pass
        self._made.clear()
        self.rig, self._ready = None, False
        if self.created:
            try:
                os.rmdir(self.dir)
                self.created = False
            except OSError:
                pass
        if self._made_tasks_dir:
            fsutil.rmdir_if_empty(self.root.tasks)
            self._made_tasks_dir = os.path.lexists(self.root.tasks)


# --------------------------------------------------------------------------------------------- one run

class Intake:
    """One ``dataproc intake`` run. :meth:`run` is the whole sequence; each step is one method below it."""

    def __init__(self, root: DataRoot, task: str, new: bool = False, calibration: bool = False, log=print,
                 staging: bool = False):
        validate_name(task)
        if staging and (new or calibration):
            raise IntakeError("the staging target takes demonstrations into an existing folder (no --new, no "
                              "--calibration)")
        self.root, self.task, self.new, self.calibration, self.log = root, task, new, calibration, log
        self.staging = staging
        self.result = IntakeResult(None, None)
        self.quarantine_area = Quarantine(root)
        self.target: TaskTarget | None = None
        self.duplicates: DuplicateIndex | None = None
        self.calibration_taken = False       # --calibration takes one episode; later candidates wait
        self.current: Candidate | None = None

    def run(self) -> IntakeResult:
        if not self.root.path.is_dir():
            return IntakeResult(None, None, incoming_missing=True)
        with self.lock():
            self.preflight_root()
            self.reconcile()
            if not self.root.incoming.is_dir():
                self.result.incoming_missing = True
                return self.result
            candidates = self.scan_incoming()
            if not candidates:
                return self.result
            self.open_target()
            with self.stop_on_error():
                for candidate in self.in_order(candidates):
                    self.handle(candidate)
            self.result.created_folder = self.target.created
            return self.result

    # ----------------------------------------------------------------------------------------- run steps

    @contextmanager
    def lock(self):
        """Hold ``<root>/.intake.lock`` (non-blocking exclusive ``flock``) for the whole run.

        Creates only the lock file (never removed). Raises :class:`LockHeldError` if another intake holds it.
        """
        lock = fsutil.FileLock(self.root.path / LOCK_NAME)
        try:
            lock.acquire()
        except BlockingIOError:
            raise LockHeldError(f"another intake is running ({lock.path} is locked)") from None
        try:
            yield
        finally:
            lock.release()

    def preflight_root(self) -> None:
        """Check that the root-level dirs intake writes to are writable: ``incoming/`` (only if it exists),
        ``tasks/`` and ``quarantine/`` (or the nearest existing parent of a missing one). Writes nothing; raises
        :class:`IntakeError` ("permission denied") before anything is moved or created."""
        checks = [(fsutil.nearest_existing(self.root.tasks), "tasks/"),
                  (fsutil.nearest_existing(self.root.quarantine), "quarantine/")]
        if self.root.incoming.is_dir():
            checks.insert(0, (self.root.incoming, "incoming/"))
        _require_writable(checks)

    def reconcile(self) -> None:
        """Repair what a crashed run left behind, in every task folder (not only the target) and in quarantine.

        Reads each ``episodes.csv``, ``.intake_pending.json`` and the slots on disk; appends a rebuilt row for every
        ``episode_N/raw`` or ``calibration/<x>/raw`` without one (recorded in ``result.reconciled``), notes empty
        ``episode_N/`` dirs, and writes an ``interrupted`` ``reason.json`` where one is missing. Nothing to do on a
        clean root. A ``PermissionError`` becomes an :class:`IntakeError`.
        """
        try:
            for task_dir in _task_folders(self.root):
                self.reconcile_task_folder(task_dir)
            self.quarantine_area.repair_missing_reasons(self.log)
        except PermissionError as e:
            raise IntakeError(f"permission denied while reconciling an interrupted intake: {e}") from None

    def reconcile_task_folder(self, task_dir: Path) -> None:
        """One task folder of :meth:`reconcile`. The Pi dir name comes from the pending record when its slot
        matches, else ``<x>`` for ``calibration/<x>``, ``.staging/<x>`` and ``failed/<x>``, else empty."""
        ledger = Ledger(task_dir)
        known = ledger.slots()
        pending = None
        for slot, kind, raw in _slots_on_disk(task_dir):
            if slot in known:
                continue
            if pending is None:
                pending = _read_pending(task_dir)
            if pending.get("slot") == slot and isinstance(pending.get("pi_name"), str):
                pi_name = pending["pi_name"]
            elif _named_slot(slot):
                pi_name = slot.split("/", 1)[1]
            else:
                pi_name = ""
            meta = _meta_of(raw)
            ledger.append(Ledger.make_row(slot, pi_name, meta, kind, _now_ns(), _manifest_sha(raw)))
            self.result.reconciled.append((task_dir.name, slot))
            self.log(f"reconciled: {task_dir.name}/{slot} had no {LEDGER_NAME} row (interrupted intake); row "
                     f"rebuilt from meta.json + manifest" + ("" if pi_name else ", Pi dir name unknown (left empty)"))
        for d in episode_dirs(task_dir):
            if not os.path.lexists(d / "raw") and not any(d.iterdir()):
                self.log(f"note: {task_dir.name}/{d.name} is empty (interrupted intake); its number stays taken")

    def scan_incoming(self) -> list[Candidate]:
        """List ``incoming/`` (sorted by name) and return the candidates: any symlink, and every directory with
        ``.verified``, no other dot-entry, and write permission on itself. Dot-prefixed names are ignored; other
        entries get a ``skipped`` outcome (not a directory / transfer in progress / not verified yet / permission
        denied). Writes nothing."""
        cands = []
        for name in sorted(os.listdir(self.root.incoming)):
            if name.startswith("."):
                continue                         # rsync temp dirs, hidden files: never candidates
            p = self.root.incoming / name
            if p.is_symlink():
                ep = RawEpisode.open(p) if p.is_dir() else None
                cands.append(Candidate(name, p, ep, _safe_meta(ep)))
                continue
            if not p.is_dir():
                self.skip(name, "not a directory")
                continue
            if _dot_entries(p):
                self.skip(name, "transfer in progress")
                continue
            if not fsutil.is_regular_file(p / VERIFIED):
                self.skip(name, "not verified yet (no .verified)")
                continue
            why = fsutil.not_writable(p)         # moving a dir to another parent rewrites its '..': needs write on it
            if why:
                self.skip(name, f"permission denied ({why})")
                continue
            ep = RawEpisode.open(p)
            cands.append(Candidate(name, p, ep, _safe_meta(ep)))
        return cands

    def open_target(self) -> None:
        """Resolve the target folder (``resolve_intake_target``), load it as a :class:`TaskTarget` (its
        ``task.yaml`` rig, if any), check ``incoming/`` / ``tasks/`` / ``quarantine/`` share one filesystem, check
        the target (and ``calibration/``) is writable, and load the :class:`DuplicateIndex` from every ledger.
        Sets ``result.folder`` / ``result.task_dir``; creates nothing (the folder is made on the first ingest). The
        staging target requires ``tasks/<task>/task.yaml`` to exist already."""
        if self.staging:
            folder = self.task
            ty = self.root.tasks / folder / TASK_YAML
            if not (self.root.tasks / folder).is_dir() or not fsutil.is_regular_file(ty):
                raise IntakeError(f"task folder {self.root.tasks / folder} does not exist or has no {TASK_YAML} "
                                  "(the staging target never creates one)")
        else:
            try:
                folder = resolve_intake_target(self.root.tasks, self.task, self.new)
            except ValueError as e:
                raise IntakeError(str(e)) from None
        self.target = TaskTarget(self.root, self.task, folder)
        self.result.folder, self.result.task_dir = folder, self.target.dir
        self.check_same_filesystem()
        self.preflight_target()
        self.duplicates = DuplicateIndex(self.root)

    def check_same_filesystem(self) -> None:
        """``st_dev`` pre-check: ``tasks/`` and ``quarantine/`` (or the root, where missing) are on the same
        filesystem as ``incoming/``, since intake only moves with ``os.rename``. Reads only."""
        dev = os.stat(self.root.incoming).st_dev
        for d in (self.root.tasks, self.root.quarantine):
            probe = d if d.exists() else self.root.path
            if os.stat(probe).st_dev != dev:
                raise IntakeError(f"{self.root.incoming} and {probe} are on different filesystems; intake only moves "
                                  "with os.rename and never copies")

    def preflight_target(self) -> None:
        """Before the first move: the existing target folder (and, with ``--calibration``, its ``calibration/`` or
        the nearest existing parent) must be writable. Skipped for a folder that does not exist yet."""
        checks = []
        if self.target.dir.is_dir():
            checks.append((self.target.dir, "the target task folder"))
            if self.calibration:
                checks.append((fsutil.nearest_existing(self.target.dir / "calibration"),
                               "the target calibration/ folder"))
        _require_writable(checks)

    def in_order(self, candidates: list[Candidate]) -> list[Candidate]:
        """Candidates with unreadable ``meta.json`` first (in name order), then the rest oldest first by
        (``start_unix_ns``, ``session_id``, ``episode_number``, name)."""
        bad = [c for c in candidates if c.meta is None]
        good = sorted((c for c in candidates if c.meta is not None), key=lambda c: c.sort_key)
        return bad + good

    @contextmanager
    def stop_on_error(self):
        """Around the candidate loop: on any exception, give the candidate being handled an ``error`` outcome
        (left in incoming, or already moved), attach the partial result to the exception (``intake_result``,
        ``intake_stopped_at``) for :func:`cli_intake`, and re-raise."""
        try:
            yield
        except Exception as e:
            c = self.current
            if c is not None:
                where = ("left in incoming" if os.path.lexists(c.path) else
                         "already moved out of incoming; the next intake reconciles its ledger row")
                self.result.outcomes.append(EpisodeOutcome(c.name, "error",
                                                           note=f"{str(e) or type(e).__name__} ({where})"))
            e.intake_result = self.result
            e.intake_stopped_at = c.name if c is not None else None
            self.result.created_folder = self.target.created
            raise

    # ----------------------------------------------------------------------------------------- per episode

    def handle(self, c: Candidate) -> None:
        """One candidate: wait (once --calibration took its episode) -> validate -> transfer re-check ->
        duplicate -> rig -> quarantine (any reason) or ingest."""
        self.current = c
        if self.calibration_taken:
            self.record(EpisodeOutcome(c.name, "waiting", note="left in incoming (--calibration takes one episode)"))
            return
        report = self.validate(c)
        if self.transfer_restarted(c):
            self.skip(c.name, "transfer in progress")
            return
        details = {}
        dup = self.check_duplicate(c)
        if dup:
            details["duplicate"] = dup
        rig = self.check_rig(c)
        if rig:
            details["rig_mismatch"] = rig
        reasons = _error_codes(report) + list(details)
        if reasons:
            self.quarantine(c, reasons, report, details)
        else:
            self.ingest(c)

    def validate(self, c: Candidate) -> ValidationReport:
        """Full validation (videos decoded, manifest hashed). Reads the episode only. A symlink that is not a
        directory gets a report with the single error ``episode_is_symlink``."""
        if c.ep is None:
            return ValidationReport(c.path, (Problem.make("episode_is_symlink", f"{c.path} is a symlink, not a "
                                                          "directory", None),))
        return c.ep.validate(decode_videos=True, check_manifest=True)

    def transfer_restarted(self, c: Candidate) -> bool:
        """True if a dot-entry (other than ``.verified``) appeared during validation: a re-send started, so the
        episode is left in incoming. Always False for symlinks. Reads the directory listing only."""
        return not c.path.is_symlink() and bool(_dot_entries(c.path))

    def check_duplicate(self, c: Candidate) -> dict | None:
        """The ``duplicate`` detail if the Pi dir name or (session_id, episode_number) is already in any ledger or
        was ingested earlier in this run; None otherwise. Invalid meta: checked by name only."""
        return self.duplicates.check(c.name, c.meta)

    def check_rig(self, c: Candidate) -> dict | None:
        """The ``rig_mismatch`` detail if ``meta.rig_id`` differs from the target's rig (its ``task.yaml``, or the
        first episode ingested into a folder made this run). Skipped while there is no reference or meta is
        invalid."""
        rig_ref = self.target.rig
        if c.meta is not None and rig_ref is not None and c.meta.rig_id != rig_ref:
            return {"episode_rig": c.meta.rig_id, "task_rig": rig_ref}
        return None

    def quarantine(self, c: Candidate, reasons: list[str], report: ValidationReport, details: dict) -> None:
        """Move the episode to ``quarantine/<name>[__k]/raw`` and write its ``reason.json`` (see
        :meth:`Quarantine.put`)."""
        mode = CALIBRATION if self.calibration else DEMONSTRATION
        self.record(self.quarantine_area.put(c, reasons, report, details, self.target.folder, mode))

    def ingest(self, c: Candidate) -> None:
        """Make the task folder + ``task.yaml`` if needed, pick the slot, hash the manifest, write the pending
        record, ``mkdir`` the slot and rename the episode in as ``raw/``, then append its ledger row. If anything
        before the ledger append fails: the still-empty slot dir is removed (the number is not burned) and a
        failed first ingest is rolled back, then the error is re-raised."""
        meta, target = c.meta, self.target
        slot_dir = None
        try:
            target.ensure(meta.rig_id)
            slot_dir, slot, kind = target.next_slot(c.name, self.calibration, self.staging)
            manifest_sha = _manifest_sha(c.path)
            target.write_pending(slot, c.name, kind)
            os.mkdir(slot_dir)
            _move(c.path, slot_dir / "raw")
        except BaseException:
            if slot_dir is not None:
                fsutil.rmdir_if_empty(slot_dir)   # do not burn the number on a failed move (fails if raw/ got in)
            target.rollback_if_unused()           # a failed first ingest leaves no task folder behind
            raise
        target.ledger.append(Ledger.make_row(slot, c.name, meta, kind, _now_ns(), manifest_sha))
        self.record(EpisodeOutcome(c.name, "ingested", slot=slot))
        self.duplicates.add(c.name, meta.session_id, str(meta.episode_number), f"{target.folder}/{slot}")
        self.calibration_taken = self.calibration

    # ----------------------------------------------------------------------------------------- outcomes

    def record(self, outcome: EpisodeOutcome) -> None:
        self.result.outcomes.append(outcome)

    def skip(self, name: str, note: str) -> None:
        self.record(EpisodeOutcome(name, "skipped", note=note))


# --------------------------------------------------------------------------------------------- public entry points

def run_intake(root: DataRoot, task: str, new: bool = False, calibration: bool = False, log=print,
               staging: bool = False) -> IntakeResult:
    """Run one intake for task name ``task`` (``[A-Za-z0-9_-]+``) into ``resolve_intake_target(...)``, or with
    ``staging`` into ``tasks/<task>/.staging/<pi_name>/`` of the existing folder ``task``.

    ``log`` receives progress lines (reconcile notes). Raises :class:`IntakeError` (incl. :class:`LockHeldError`);
    quarantined episodes are not errors. See :class:`Intake` and docs/INTAKE.md.
    """
    return Intake(root, task, new=new, calibration=calibration, log=log, staging=staging).run()


def reconcile(root: DataRoot, log=print) -> list[tuple[str, str]]:
    """Only the reconcile step of an intake (under ``.intake.lock``): rebuild the ledger rows a crashed intake or
    :func:`move_slot` left out, in every task folder. Returns the ``(folder, slot)`` rows rebuilt."""
    it = Intake(root, "reconcile", log=log)
    if not root.path.is_dir():
        return []
    with it.lock():
        it.reconcile()
    return it.result.reconciled


def move_slot(task_dir: Path, src_slot: str, dst_slot: str, pi_name: str) -> None:
    """Rename the demonstration ``task_dir/<src_slot>`` (with its ``raw/`` and ``processed/``) to
    ``task_dir/<dst_slot>`` (``.staging/<x>``, ``failed/<x>`` or ``episode_N``; its parent is created if needed),
    then append the ledger row of ``dst_slot``: a copy of ``src_slot``'s newest row (rebuilt from meta.json when it
    has none) with the new slot and time. The pending record is written first, so :meth:`Intake.reconcile`
    rebuilds the row (with ``pi_name``) if the process dies between the rename and the append. Raises
    :class:`IntakeError` if ``dst_slot`` exists or the rename would cross filesystems."""
    task_dir = Path(task_dir)
    src, dst = task_dir / src_slot, task_dir / dst_slot
    if os.path.lexists(dst):
        raise IntakeError(f"cannot move {src} to {dst}: it already exists")
    ledger = Ledger(task_dir)
    rows = [r for r in ledger.rows() if r["slot"] == src_slot]
    row = dict(rows[-1]) if rows else Ledger.make_row(src_slot, pi_name, _meta_of(src / "raw"), DEMONSTRATION, 0,
                                                     _manifest_sha(src / "raw"))
    fsutil.write_atomic(task_dir / PENDING_NAME,
                        json.dumps({"slot": dst_slot, "pi_name": pi_name, "kind": DEMONSTRATION}) + "\n")
    dst.parent.mkdir(exist_ok=True)
    _move(src, dst)
    row.update(slot=dst_slot, pi_name=pi_name, kind=DEMONSTRATION, intake_unix_ns=str(_now_ns()))
    ledger.append(row)


def format_summary(result: IntakeResult, partial_run: bool = False) -> str:
    """The table ``cli_intake`` prints: one line per incoming entry, then totals (``partial_run``: the run stopped
    on an error, so the totals cover only what was done before it)."""
    lines = []
    if result.incoming_missing:
        lines.append("nothing to intake: incoming/ does not exist")
        return "\n".join(lines)
    if result.outcomes:
        w = max(len(o.pi_name) for o in result.outcomes)
        lines += [f"  {o.pi_name:<{w}}  ->  {o.describe()}" for o in result.outcomes]
    if result.nothing_waiting and not partial_run:
        lines.append("nothing waiting in incoming/ (no verified episode); no task folder created")
        return "\n".join(lines)
    n_in, n_q = len(result.by_status("ingested")), len(result.by_status("quarantined"))
    n_s = len(result.by_status("skipped")) + len(result.by_status("waiting"))
    where = f"tasks/{result.folder}" + (" (created)" if result.created_folder else "")
    prefix = "before the error: " if partial_run else ""
    n_e = len(result.by_status("error"))
    tail = f", {n_e} failed" if n_e else ""
    lines.append(f"{prefix}{n_in} ingested into {where}, {n_q} quarantined, {n_s} left in incoming{tail}")
    return "\n".join(lines)


def cli_intake(root: DataRoot, task: str, new: bool = False, calibration: bool = False) -> int:
    """``dataproc intake``: run, print the summary, return the exit code (0 also when episodes were quarantined)."""
    try:
        result = run_intake(root, task, new=new, calibration=calibration)
    except Exception as e:                       # noqa: BLE001 - every failure is reported, exit 1
        if not isinstance(e, IntakeError):
            traceback.print_exc()
        partial = getattr(e, "intake_result", None)
        if partial is not None and partial.outcomes:
            print(format_summary(partial, partial_run=True))
        kind = "" if isinstance(e, IntakeError) else "unexpected error: "
        print(f"dataproc intake: {kind}{e}", file=sys.stderr)
        stopped = getattr(e, "intake_stopped_at", None)
        if stopped is not None:
            print(f"dataproc intake: stopped at {stopped}; nothing after it was processed", file=sys.stderr)
        return 1
    print(format_summary(result))
    return 0
