"""``dataproc intake``: move verified raw episodes from ``incoming/`` into a task folder (docs/DESIGN.md "intake",
decisions in docs/INTAKE.md).

::

    result = run_intake(data_root(), "block_raise", new=False, calibration=False)
    print(format_summary(result))

One run:

1. take ``<root>/.intake.lock`` (``flock``, non-blocking; held -> :class:`LockHeldError`);
2. reconcile every task folder: an ``episode_N/raw`` or ``calibration/<x>/raw`` without an ``episodes.csv`` row (a
   crash between the move and the ledger append) gets its row rebuilt;
3. scan ``incoming/``: dirs with ``.verified`` and no other dot-entry are candidates; dot-entries mean "transfer in
   progress" (left alone); symlinks are quarantined;
4. order: invalid ``meta.json`` first, then oldest ``start_unix_ns`` first (ties: session_id, episode_number, name);
5. per candidate: full validation, duplicate check, rig check; any failure -> ``quarantine/<name>[__k]/`` (the
   episode in ``raw/`` + ``reason.json``), else ``os.rename`` to ``episode_N/raw`` (or ``calibration/<name>/raw``)
   and one fsync'd ``episodes.csv`` row.

Nothing is ever deleted or copied; raw files are never opened for writing.
"""

from __future__ import annotations

import csv
import datetime as _dt
import errno
import fcntl
import hashlib
import io
import json
import os
import stat
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from dataproc import __version__
from dataproc.config import ConfigError, TaskConfig, dump_task_config, load_task_config
from dataproc.paths import DataRoot, episode_dirs, next_episode_index, resolve_intake_target, validate_name
from dataproc.raw_episode import MANIFEST, VERIFIED, EpisodeMeta, Problem, RawEpisode, RawEpisodeError, \
    ValidationReport

__all__ = [
    "LOCK_NAME", "LEDGER_NAME", "LEDGER_COLUMNS", "TASK_YAML", "PENDING_NAME", "REASON_NAME", "IntakeError",
    "LockHeldError", "EpisodeOutcome", "IntakeResult", "run_intake", "read_ledger", "format_summary", "cli_intake",
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _fsync_dir(path: Path) -> None:
    """Best effort: make a rename/creation in ``path`` durable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a dot temp file in the same directory + fsync + rename."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)
    _fsync_dir(path.parent)


def _rmdir_if_empty(d: Path) -> None:
    """Remove a directory this run just created, only if it is still empty (never removes episode data)."""
    try:
        os.rmdir(d)
    except OSError:
        pass


def _rename(src: Path, dst: Path) -> None:
    """``os.rename`` that never falls back to copying."""
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            raise IntakeError(f"cannot move {src} to {dst}: different filesystems (EXDEV). incoming/, tasks/ and "
                              "quarantine/ must be on one filesystem; nothing is ever copied") from None
        raise
    _fsync_dir(dst.parent)
    _fsync_dir(src.parent)


# --------------------------------------------------------------------------------------------- the ledger

def _row_line(row: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow([row.get(c, "") for c in LEDGER_COLUMNS])
    return buf.getvalue()


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


def _append_ledger(path: Path, row: dict) -> None:
    """Append one full line (with the header first if the file is new or empty), then flush + fsync."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        data = _row_line(row)
        if os.fstat(fd).st_size == 0:
            data = ",".join(LEDGER_COLUMNS) + "\n" + data
        view = data.encode("utf-8")
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _ledger_row(slot: str, pi_name: str, meta: EpisodeMeta | None, kind: str, intake_ns: int,
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


def _manifest_sha(ep_dir: Path) -> str:
    m = ep_dir / MANIFEST
    return _sha256_file(m) if _is_regular(m) else ""


def _meta_of(ep_dir: Path) -> EpisodeMeta | None:
    try:
        return RawEpisode.open(ep_dir).meta
    except (OSError, RawEpisodeError):
        return None


# --------------------------------------------------------------------------------------------- reconcile

def _task_folders(root: DataRoot) -> list[Path]:
    if not root.tasks.is_dir():
        return []
    return sorted(p for p in root.tasks.iterdir()
                  if not p.name.startswith(".") and p.is_dir() and not p.is_symlink())


def _slots_on_disk(task_dir: Path) -> list[tuple[str, str, Path]]:
    """``(slot, kind, raw_dir)`` for every ``episode_N/raw`` and ``calibration/<x>/raw`` present."""
    out = []
    for d in episode_dirs(task_dir):
        if os.path.lexists(d / "raw"):
            out.append((d.name, DEMONSTRATION, d / "raw"))
    cal = task_dir / "calibration"
    if cal.is_dir() and not cal.is_symlink():
        for d in sorted(cal.iterdir()):
            if not d.name.startswith(".") and os.path.lexists(d / "raw"):
                out.append((f"calibration/{d.name}", CALIBRATION, d / "raw"))
    return out


def _read_pending(task_dir: Path) -> dict:
    try:
        d = json.loads((task_dir / PENDING_NAME).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _reconcile(root: DataRoot, log) -> list[tuple[str, str]]:
    """Rebuild missing ledger rows in every task folder. Returns ``(folder, slot)`` for each row added."""
    added = []
    for task_dir in _task_folders(root):
        ledger = task_dir / LEDGER_NAME
        known = {r["slot"] for r in read_ledger(ledger)}
        pending = None
        for slot, kind, raw in _slots_on_disk(task_dir):
            if slot in known:
                continue
            if pending is None:
                pending = _read_pending(task_dir)
            if pending.get("slot") == slot and isinstance(pending.get("pi_name"), str):
                pi_name = pending["pi_name"]
            elif kind == CALIBRATION:
                pi_name = slot.split("/", 1)[1]
            else:
                pi_name = ""
            meta = _meta_of(raw)
            _append_ledger(ledger, _ledger_row(slot, pi_name, meta, kind, _now_ns(), _manifest_sha(raw)))
            added.append((task_dir.name, slot))
            log(f"reconciled: {task_dir.name}/{slot} had no {LEDGER_NAME} row (interrupted intake); row rebuilt "
                f"from meta.json + manifest" + ("" if pi_name else ", Pi dir name unknown (left empty)"))
        for d in episode_dirs(task_dir):
            if not os.path.lexists(d / "raw") and not any(d.iterdir()):
                log(f"note: {task_dir.name}/{d.name} is empty (interrupted intake); its number stays taken")
    if root.quarantine.is_dir():
        for q in sorted(root.quarantine.iterdir()):
            if (not q.name.startswith(".") and q.is_dir() and not q.is_symlink() and os.path.lexists(q / "raw")
                    and not os.path.lexists(q / REASON_NAME)):
                now = _now_ns()
                _write_atomic(q / REASON_NAME, json.dumps({
                    "pi_name": q.name, "reasons": ["interrupted"],
                    "details": {"interrupted": "intake stopped between the move and writing reason.json; the "
                                               "original reasons are unknown (re-run validation on raw/)"},
                    "report": None, "quarantined_at": _iso_utc(now), "quarantined_unix_ns": now,
                    "dataproc_version": __version__, "target_task": None, "mode": None, "episode": "raw/",
                }, indent=2) + "\n")
                log(f"reconciled: quarantine/{q.name} had no {REASON_NAME} (interrupted intake); wrote one with "
                    "reason 'interrupted'")
    return added


# --------------------------------------------------------------------------------------------- candidates

@dataclass
class _Candidate:
    name: str
    path: Path
    ep: RawEpisode | None            # None for a symlink that is not a directory
    meta: EpisodeMeta | None

    @property
    def sort_key(self):
        m = self.meta
        return (m.start_unix_ns, m.session_id, m.episode_number, self.name)


def _dot_entries(d: Path) -> list[str]:
    try:
        return sorted(e for e in os.listdir(d) if e.startswith(".") and e != VERIFIED)
    except OSError:
        return []


def _not_writable(d: Path) -> str | None:
    """None if this process can create/rename entries in ``d`` (W_OK | X_OK), else a description of the dir."""
    if os.access(d, os.W_OK | os.X_OK, effective_ids=_EFFECTIVE):
        return None
    try:
        st = os.stat(d)
        return f"owner uid {st.st_uid}, mode {stat.filemode(st.st_mode)}"
    except OSError as e:
        return str(e)


#: os.access checks the effective ids where the platform supports it (Linux does), else the real ones.
_EFFECTIVE = os.access in os.supports_effective_ids


def _checked_uid() -> str:
    return f"effective uid {os.geteuid()}" if _EFFECTIVE else f"real uid {os.getuid()}"


def _existing(d: Path) -> Path:
    """``d``, or its nearest existing ancestor (where ``d`` would be created)."""
    while not d.exists() and d.parent != d:
        d = d.parent
    return d


def _require_writable(checks: list[tuple[Path, str]]) -> None:
    for d, what in checks:
        why = _not_writable(d)
        if why:
            raise IntakeError(f"permission denied: {d} ({what}) is not writable by {_checked_uid()} ({why}); "
                              "nothing was moved or created")


def _preflight_root(root: DataRoot) -> None:
    """Before reconcile: the root-level directories intake writes to (or the nearest existing parent of a missing
    one) must be writable."""
    checks = [(_existing(root.tasks), "tasks/"), (_existing(root.quarantine), "quarantine/")]
    if root.incoming.is_dir():
        checks.insert(0, (root.incoming, "incoming/"))
    _require_writable(checks)


def _preflight_target(target_dir: Path, calibration: bool) -> None:
    """Before the first move: the target folder (and ``calibration/`` with --calibration), where they exist."""
    checks = []
    if target_dir.is_dir():
        checks.append((target_dir, "the target task folder"))
        if calibration:
            checks.append((_existing(target_dir / "calibration"), "the target calibration/ folder"))
    _require_writable(checks)


def _scan_incoming(root: DataRoot, outcomes: list[EpisodeOutcome]) -> list[_Candidate]:
    cands = []
    for name in sorted(os.listdir(root.incoming)):
        if name.startswith("."):
            continue                         # rsync temp dirs, hidden files: never candidates
        p = root.incoming / name
        if p.is_symlink():
            ep = RawEpisode.open(p) if p.is_dir() else None
            cands.append(_Candidate(name, p, ep, _safe_meta(ep)))
            continue
        if not p.is_dir():
            outcomes.append(EpisodeOutcome(name, "skipped", note="not a directory"))
            continue
        if _dot_entries(p):
            outcomes.append(EpisodeOutcome(name, "skipped", note="transfer in progress"))
            continue
        if not _is_regular(p / VERIFIED):
            outcomes.append(EpisodeOutcome(name, "skipped", note="not verified yet (no .verified)"))
            continue
        why = _not_writable(p)            # moving a dir to another parent rewrites its '..': needs write on it
        if why:
            outcomes.append(EpisodeOutcome(name, "skipped", note=f"permission denied ({why})"))
            continue
        ep = RawEpisode.open(p)
        cands.append(_Candidate(name, p, ep, _safe_meta(ep)))
    return cands


def _safe_meta(ep: RawEpisode | None) -> EpisodeMeta | None:
    if ep is None:
        return None
    try:
        return ep.meta
    except RawEpisodeError:
        return None


def _order(cands: list[_Candidate]) -> list[_Candidate]:
    bad = [c for c in cands if c.meta is None]
    good = sorted((c for c in cands if c.meta is not None), key=lambda c: c.sort_key)
    return bad + good


# --------------------------------------------------------------------------------------------- checks

class _Seen:
    """Every Pi dir name and (session_id, episode_number) already taken in by any task (plus this run)."""

    def __init__(self, root: DataRoot):
        self.names: dict[str, str] = {}
        self.pairs: dict[tuple[str, str], str] = {}
        for task_dir in _task_folders(root):
            for r in read_ledger(task_dir / LEDGER_NAME):
                self.add(r["pi_name"], r["session_id"], r["episode_number"], f"{task_dir.name}/{r['slot']}")

    def add(self, name: str, session_id: str, episode_number: str, where: str) -> None:
        if name:
            self.names.setdefault(name, where)
        if session_id != "" and episode_number != "":
            self.pairs.setdefault((session_id, str(episode_number)), where)

    def check(self, name: str, meta: EpisodeMeta | None) -> dict | None:
        d = {}
        if name in self.names:
            d["pi_name"] = {"value": name, "taken_by": self.names[name]}
        if meta is not None:
            key = (meta.session_id, str(meta.episode_number))
            if key in self.pairs:
                d["session_episode"] = {"value": list(key), "taken_by": self.pairs[key]}
        return d or None


def _validate(c: _Candidate) -> ValidationReport:
    if c.ep is None:
        return ValidationReport(c.path, (Problem.make("episode_is_symlink", f"{c.path} is a symlink, not a "
                                                      "directory", None),))
    return c.ep.validate(decode_videos=True, check_manifest=True)


def _reasons(report: ValidationReport, extra: list[str]) -> list[str]:
    out = []
    for p in report.errors:
        if p.code not in out:
            out.append(p.code)
    return out + extra


# --------------------------------------------------------------------------------------------- target folder

class _Target:
    """The task folder, created lazily on the first episode that is ingested."""

    def __init__(self, root: DataRoot, label: str, folder: str):
        self.root, self.label, self.folder = root, label, folder
        self.dir = root.tasks / folder
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
            _fsync_dir(self.root.tasks)
        cfg = TaskConfig(label=self.label, rig=rig_id,
                         created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), notes="")
        tmp = self.dir / f".{TASK_YAML}.tmp-{os.getpid()}"
        dump_task_config(cfg, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.rename(tmp, self.dir / TASK_YAML)
        self._made.append(self.dir / TASK_YAML)
        _fsync_dir(self.dir)
        self.rig = rig_id
        self._ready = True

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
            _rmdir_if_empty(self.root.tasks)
            self._made_tasks_dir = os.path.lexists(self.root.tasks)


def _free_name(parent: Path, name: str) -> Path:
    """``parent/name``, or ``name__2``, ``name__3``, ... if taken (any entry counts)."""
    p, k = parent / name, 2
    while os.path.lexists(p):
        p, k = parent / f"{name}__{k}", k + 1
    return p


def _same_fs_check(root: DataRoot) -> None:
    dev = os.stat(root.incoming).st_dev
    for d in (root.tasks, root.quarantine):
        probe = d if d.exists() else root.path
        if os.stat(probe).st_dev != dev:
            raise IntakeError(f"{root.incoming} and {probe} are on different filesystems; intake only moves with "
                              "os.rename and never copies")


# --------------------------------------------------------------------------------------------- actions

def _quarantine(root: DataRoot, c: _Candidate, reasons: list[str], report: ValidationReport, details: dict,
                folder: str, calibration: bool) -> EpisodeOutcome:
    root.quarantine.mkdir(parents=True, exist_ok=True)
    qdir = _free_name(root.quarantine, c.name)
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
        "target_task": folder,
        "mode": CALIBRATION if calibration else DEMONSTRATION,
        "episode": "raw/",
    }
    try:
        _rename(c.path, qdir / "raw")
    except BaseException:
        _rmdir_if_empty(qdir)             # no reason-only orphan folder
        raise
    _write_atomic(qdir / REASON_NAME, json.dumps(reason, indent=2) + "\n")
    return EpisodeOutcome(c.name, "quarantined", reasons=tuple(reasons), quarantine_dir=qdir)


def _ingest(target: _Target, c: _Candidate, calibration: bool) -> EpisodeOutcome:
    meta = c.meta
    slot_dir = None
    try:
        target.ensure(meta.rig_id)
        if calibration:
            cal = target.dir / "calibration"
            if not os.path.lexists(cal):
                os.mkdir(cal)
                target.note_made(cal)
            slot_dir = _free_name(cal, c.name)
            slot, kind = f"calibration/{slot_dir.name}", CALIBRATION
        else:
            n = next_episode_index(target.dir)
            slot_dir = target.dir / f"episode_{n}"
            slot, kind = slot_dir.name, DEMONSTRATION
        manifest_sha = _manifest_sha(c.path)
        pending = target.dir / PENDING_NAME
        fresh_pending = not os.path.lexists(pending)
        _write_atomic(pending, json.dumps({"slot": slot, "pi_name": c.name, "kind": kind}) + "\n")
        if fresh_pending:
            target.note_made(pending)
        os.mkdir(slot_dir)
        _rename(c.path, slot_dir / "raw")
    except BaseException:
        if slot_dir is not None:
            _rmdir_if_empty(slot_dir)     # do not burn the number on a failed move (fails if raw/ got in)
        target.rollback_if_unused()       # a failed first ingest leaves no task folder behind
        raise
    _append_ledger(target.dir / LEDGER_NAME, _ledger_row(slot, c.name, meta, kind, _now_ns(), manifest_sha))
    return EpisodeOutcome(c.name, "ingested", slot=slot)


# --------------------------------------------------------------------------------------------- run

class _Lock:
    def __init__(self, path: Path):
        self.path, self.fd = path, None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise LockHeldError(f"another intake is running ({self.path} is locked)") from None
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def run_intake(root: DataRoot, task: str, new: bool = False, calibration: bool = False, log=print) -> IntakeResult:
    """Run one intake for task name ``task`` (``[A-Za-z0-9_-]+``) into ``resolve_intake_target(...)``.

    ``log`` receives progress lines (reconcile notes). Raises :class:`IntakeError` (incl. :class:`LockHeldError`);
    quarantined episodes are not errors. See the module docstring and docs/INTAKE.md.
    """
    validate_name(task)
    if not root.path.is_dir():
        return IntakeResult(None, None, incoming_missing=True)
    with _Lock(root.path / LOCK_NAME):
        result = IntakeResult(None, None)
        _preflight_root(root)
        try:
            result.reconciled = _reconcile(root, log)
        except PermissionError as e:
            raise IntakeError(f"permission denied while reconciling an interrupted intake: {e}") from None
        if not root.incoming.is_dir():
            result.incoming_missing = True
            return result
        cands = _scan_incoming(root, result.outcomes)
        if not cands:
            return result
        try:
            folder = resolve_intake_target(root.tasks, task, new)
        except ValueError as e:
            raise IntakeError(str(e)) from None
        target = _Target(root, task, folder)
        result.folder, result.task_dir = folder, target.dir
        _same_fs_check(root)
        _preflight_target(target.dir, calibration)
        seen = _Seen(root)
        done, c = False, None
        try:
            for c in _order(cands):
                done = _one(root, target, folder, seen, c, calibration, done, result)
        except Exception as e:
            if c is not None:
                where = ("left in incoming" if os.path.lexists(c.path) else
                         "already moved out of incoming; the next intake reconciles its ledger row")
                result.outcomes.append(EpisodeOutcome(c.name, "error",
                                                      note=f"{str(e) or type(e).__name__} ({where})"))
            e.intake_result = result
            e.intake_stopped_at = c.name if c is not None else None
            result.created_folder = target.created
            raise
        result.created_folder = target.created
        return result


def _one(root: DataRoot, target: _Target, folder: str, seen: _Seen, c: _Candidate, calibration: bool, done: bool,
         result: IntakeResult) -> bool:
    """Handle one candidate; returns the new ``done`` (True once --calibration has taken its episode)."""
    if done:
        result.outcomes.append(EpisodeOutcome(c.name, "waiting",
                                              note="left in incoming (--calibration takes one episode)"))
        return done
    report = _validate(c)
    if not c.path.is_symlink() and _dot_entries(c.path):      # a re-send started during validation
        result.outcomes.append(EpisodeOutcome(c.name, "skipped", note="transfer in progress"))
        return done
    extra, details = [], {}
    dup = seen.check(c.name, c.meta)
    if dup:
        extra.append("duplicate")
        details["duplicate"] = dup
    rig_ref = target.rig
    if c.meta is not None and rig_ref is not None and c.meta.rig_id != rig_ref:
        extra.append("rig_mismatch")
        details["rig_mismatch"] = {"episode_rig": c.meta.rig_id, "task_rig": rig_ref}
    reasons = _reasons(report, extra)
    if reasons:
        result.outcomes.append(_quarantine(root, c, reasons, report, details, folder, calibration))
        return done
    out = _ingest(target, c, calibration)
    result.outcomes.append(out)
    seen.add(c.name, c.meta.session_id, str(c.meta.episode_number), f"{folder}/{out.slot}")
    return calibration


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
