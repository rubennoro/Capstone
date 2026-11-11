"""The moves of ``dataproc run`` inside one task folder, and what a crashed run left behind.

::

    tasks/<folder>/.staging/<pi_name>/   intake put it here; the stages run here (or it waits for SLAM)
        -> promote()   -> episode_<N>/                passed every stage (N = next free number, never reused)
        -> fail()      -> failed/<pi_name>/ + reason.json   failed or rejected at some stage
    failed/<pi_name>/ -> unfail()  -> .staging/<pi_name>/   --retry-failed

Every move is one same-filesystem rename (:func:`dataproc.intake.move_slot`, which also appends the ledger row of
the new slot), done under ``FileLock($ROOT/.run.lock)`` by the caller. ``reason.json`` is written right after the
move into ``failed/`` and can always be rebuilt from ``processed/status.json``, so a crash between the two loses
nothing: :func:`finish_interrupted` (start of every run) writes the missing ones, removes a stale one left in
``.staging/`` by an interrupted ``unfail``, rebuilds missing ledger rows (:func:`dataproc.intake.reconcile`) and
brings ``pi_sources.csv`` in line with where each recording is (:func:`sync_sources`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from dataproc.files import iso_utc, load_status, read_json, write_json
from dataproc.paths import FAILED_DIR, STAGING_DIR, DataRoot, next_episode_index
from dataproc.report import STAGE_NAMES

__all__ = ["REASON_FILE", "staging_names", "failed_names", "read_reason", "format_reason", "first_failure",
           "waiting_for_slam", "promote", "fail", "unfail", "sync_sources", "finish_interrupted"]

REASON_FILE = "reason.json"
#: ``dataproc run`` numbers its episodes from 1 (``episode_1``, ``episode_2``, ...).
FIRST_EPISODE = 1


def _subdir_names(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def staging_names(task_dir: Path) -> list[str]:
    """The ``.staging/<pi_name>`` names (recordings in progress or waiting for SLAM)."""
    return _subdir_names(Path(task_dir) / STAGING_DIR)


def failed_names(task_dir: Path) -> list[str]:
    """The ``failed/<pi_name>`` names."""
    return _subdir_names(Path(task_dir) / FAILED_DIR)


def read_reason(task_dir: Path, pi_name: str) -> dict:
    """``failed/<pi_name>/reason.json`` (``{}`` if missing or unreadable)."""
    rec = read_json(Path(task_dir) / FAILED_DIR / pi_name / REASON_FILE, {})
    return rec if isinstance(rec, dict) else {}


def format_reason(rec: dict) -> str:
    """``"<stage>: <reason>; <reason>"`` of a ``reason.json`` (the form ``dataproc package`` lists too)."""
    reasons = "; ".join(str(r) for r in rec.get("reasons") or []) or rec.get("message") or "no reason.json"
    return f"{rec['stage']}: {reasons}" if rec.get("stage") else reasons


def first_failure(status: dict) -> tuple[str | None, dict]:
    """``(stage, record)`` of the first ``failed`` / ``rejected`` stage in ``status.json`` (``(None, {})``)."""
    for s in STAGE_NAMES:
        rec = status.get(s) or {}
        if rec.get("status") in ("failed", "rejected"):
            return s, rec
    return None, {}


def waiting_for_slam(status: dict) -> bool:
    """True when the episode stopped at slam because no SLAM backend is installed (``slam_unavailable``)."""
    rec = status.get("slam") or {}
    return rec.get("status") == "failed" and "slam_unavailable" in (rec.get("reasons") or [])


def _reason_record(ep_dir: Path, pi_name: str, source_name: str | None, rebuilt: bool = False) -> dict:
    """``reason.json`` from the dir's ``processed/status.json``: ``stage``, ``status``, ``reasons``, ``message``,
    ``time``."""
    stage, rec = first_failure(load_status(ep_dir / "processed"))
    out = {"pi_name": pi_name, "source_name": source_name, "stage": stage, "status": rec.get("status"),
           "reasons": list(rec.get("reasons") or []), "message": rec.get("reason") or "",
           "time": iso_utc(time.time())}
    if stage is None:
        out.update(stage="incomplete", reasons=["incomplete"], message="the stages did not finish")
    if rebuilt:
        out["note"] = "rebuilt from processed/status.json after an interrupted dataproc run"
    return out


# --------------------------------------------------------------------------------------------- the moves

def promote(task_dir: Path, pi_name: str) -> str:
    """``.staging/<pi_name>`` -> the next free ``episode_N`` (N >= 1). Returns the slot."""
    from dataproc.intake import move_slot

    slot = f"episode_{max(FIRST_EPISODE, next_episode_index(task_dir))}"
    move_slot(task_dir, f"{STAGING_DIR}/{pi_name}", slot, pi_name)
    return slot


def fail(task_dir: Path, src_slot: str, pi_name: str, source_name: str | None = None) -> dict:
    """``<src_slot>`` (``.staging/<pi_name>``, or an ``episode_N`` re-run with ``--force``) -> ``failed/<pi_name>``,
    then its ``reason.json`` (from ``processed/status.json``). Returns the reason record."""
    from dataproc.intake import move_slot

    move_slot(task_dir, src_slot, f"{FAILED_DIR}/{pi_name}", pi_name)
    d = Path(task_dir) / FAILED_DIR / pi_name
    rec = _reason_record(d, pi_name, source_name)
    write_json(d / REASON_FILE, rec)
    return rec


def unfail(task_dir: Path, pi_name: str) -> None:
    """``failed/<pi_name>`` -> ``.staging/<pi_name>`` (``--retry-failed``); its ``reason.json`` is removed."""
    from dataproc.intake import move_slot

    move_slot(task_dir, f"{FAILED_DIR}/{pi_name}", f"{STAGING_DIR}/{pi_name}", pi_name)
    (Path(task_dir) / STAGING_DIR / pi_name / REASON_FILE).unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------- consistency

def sync_sources(root: DataRoot, task_dir: Path) -> None:
    """Set the ``pi_sources.csv`` status of this folder's demonstrations from where they are: ``failed`` for a
    ``failed/<x>``, ``ingested`` for a ``.staging/<x>`` or an ``episode_N`` (rows matched by ``converted_name``)."""
    from dataproc.pi_adapter import SOURCES_FILE, read_sources
    from dataproc.pi_intake import set_source_status

    task_dir = Path(task_dir)
    failed, staged = set(failed_names(task_dir)), set(staging_names(task_dir))
    for r in read_sources(root.path / SOURCES_FILE):
        if r.get("task") != task_dir.name or r.get("kind") != "demonstration" or r.get("status") == "quarantined":
            continue
        name = r.get("converted_name")
        want = "failed" if name in failed else "ingested" if name in staged else None
        if want is None and r.get("status") == "failed" and not (task_dir / FAILED_DIR / name).exists():
            want = "ingested"                       # promoted or moved back after a failure
        if want is not None and r.get("status") != want:
            set_source_status(root, r["source_manifest_sha256"], want)


def finish_interrupted(root: DataRoot, task_dir: Path, log: Callable = print) -> None:
    """Repair what a crashed ``dataproc run`` left in ``task_dir`` (see the module docstring). Leftovers in
    ``.staging/`` themselves stay: the run resumes them."""
    from dataproc.intake import reconcile

    task_dir = Path(task_dir)
    reconcile(root, log)
    for name in failed_names(task_dir):
        d = task_dir / FAILED_DIR / name
        if not (d / REASON_FILE).is_file():
            write_json(d / REASON_FILE, _reason_record(d, name, None, rebuilt=True))
            log(f"reconciled: {FAILED_DIR}/{name} had no {REASON_FILE} (interrupted run); rebuilt it")
    for name in staging_names(task_dir):
        stale = task_dir / STAGING_DIR / name / REASON_FILE
        if stale.is_file():
            stale.unlink()                          # an interrupted --retry-failed: it is retried now
    sync_sources(root, task_dir)
