"""What ``dataproc run TASK_FOLDER START_DIR END_DIR`` will do, decided before anything changes (and printed as is by
``--dry-run``).

1. ``tasks/<folder>`` must exist with a ``task.yaml`` and a calibration whose recorded result is ``done``
   (``dataproc calibrate`` made it).
2. ``START_DIR`` and ``END_DIR`` must be directories in ``--source`` with START <= END in natural order
   (``episode_9`` < ``episode_10``). The range is every non-dot directory between them, inclusive.
3. Each recording in the range gets an action:

   * ``take``: new (complete, manifest verified, not marked invalid, not in ``pi_sources.csv``);
   * ``resume``: already in this folder's ``.staging/`` (interrupted, or waiting for SLAM);
   * ``retry``: in this folder's ``failed/`` and ``--retry-failed``;
   * ``reprocess``: already an ``episode_N`` of this folder and ``--force``;
   * ``skip`` with why: still syncing, manifest failed, marked invalid by the Pi, already an episode of this folder
     (``quiet``), taken by another task folder, a calibration, quarantined at intake, failed earlier at
     ``<stage>: <reason>``.

   Every ``.staging/`` leftover outside the range is resumed too (its source is not needed any more). Everything
   is processed in natural order of the source names, so episode numbers follow the recording order.

Read-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from dataproc.files import load_status
from dataproc.paths import DataRoot, episode_dirs, validate_episode_dir_name, validate_name
from dataproc.pi_intake import PiIntakeError, taken_sources
from dataproc.run.staging import failed_names, format_reason, read_reason, staging_names

__all__ = ["TAKE", "RESUME", "RETRY", "REPROCESS", "SKIP", "RunError", "Recording", "RunPlan", "natural_key",
           "check_folder", "select_range", "plan_run", "format_plan"]

TAKE, RESUME, RETRY, REPROCESS, SKIP = "take", "resume", "retry", "reprocess", "skip"


class RunError(PiIntakeError):
    """``dataproc run`` cannot run (bad arguments, missing or uncalibrated folder, ``incoming/`` not empty, ...):
    exit 1. Raised before anything is changed, or between recordings with what was done so far reported."""


@dataclass
class Recording:
    """One recording and what the run does with it. ``name`` is its directory in ``--source`` (or, for a
    ``.staging/`` leftover outside the range whose source is unknown, its ``pi_name``); ``pi_name`` is the converted episode name used in
    the folder (``.staging/<pi_name>``, ``failed/<pi_name>``, the ledger and ``meta/episode_names``)."""

    name: str
    action: str
    why: str = ""
    quiet: bool = False                 # a skip not worth a line of its own (already an episode of this folder)
    pi_name: str | None = None
    source: object = None               # PiSource (None for a leftover outside the range)
    taken: dict | None = None           # its pi_sources.csv row
    slot: str | None = None             # the episode_N of a reprocess
    in_range: bool = True

    @property
    def source_name(self) -> str | None:
        """The directory name in ``--source``, when known."""
        if self.source is not None:
            return self.source.name
        return (self.taken or {}).get("source_name") or None

    def describe(self) -> str:
        if self.action == SKIP:
            return f"skip ({self.why})"
        return f"{self.action}" + (f" ({self.why})" if self.why else "")


@dataclass
class RunPlan:
    folder: str
    task_dir: Path
    label: str
    rig_id: str
    source_dir: Path
    start: str
    end: str
    recordings: list[Recording] = field(default_factory=list)

    def to_do(self) -> list[Recording]:
        """The recordings that will be processed, in natural order of their names."""
        return [r for r in self.recordings if r.action != SKIP]

    def skipped(self) -> list[Recording]:
        return [r for r in self.recordings if r.action == SKIP]


# --------------------------------------------------------------------------------------------- folder and range

def natural_key(name: str) -> list:
    """Sort key with digit runs compared as numbers: ``episode_9`` < ``episode_10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)] + [name]


def check_folder(root: DataRoot, folder: str):
    """``(task_dir, TaskConfig)`` of a folder with a successful calibration, else :class:`RunError`."""
    from dataproc.config import ConfigError, load_task_config
    from dataproc.intake import IntakeError, read_ledger
    from dataproc.stages.calibration import newest_calibration

    try:
        validate_name(folder, "task folder")
    except ValueError as e:
        raise RunError(str(e)) from None
    task_dir = root.tasks / folder
    hint = (f"task folder {folder} does not exist (or has no successful calibration): run "
            "`dataproc calibrate <task> <episode_dir>` first")
    if not task_dir.is_dir() or not (task_dir / "task.yaml").is_file():
        raise RunError(hint)
    try:
        task = load_task_config(task_dir / "task.yaml")
        cal = newest_calibration(task_dir, read_ledger(task_dir / "episodes.csv"))
    except (ConfigError, IntakeError, ValueError) as e:
        raise RunError(f"task folder {folder}: {e}") from None
    rec = load_status(cal / "processed").get("gripper_range") if cal is not None else None
    if not rec or rec.get("status") != "done":
        raise RunError(hint)
    return task_dir, task


def select_range(source_dir: Path, start: str, end: str) -> list[str]:
    """The non-dot directory names of ``source_dir`` from ``start`` to ``end`` inclusive, in natural order."""
    for what, name in (("START_DIR", start), ("END_DIR", end)):
        try:
            validate_episode_dir_name(name)
        except ValueError as e:
            raise RunError(f"{what}: {e}") from None
        if not (source_dir / name).is_dir():
            raise RunError(f"{what} {name} is not a directory in {source_dir}")
    k0, k1 = natural_key(start), natural_key(end)
    if k0 > k1:
        raise RunError(f"START_DIR {start} comes after END_DIR {end} (natural order); swap them")
    names = [p.name for p in source_dir.iterdir() if not p.name.startswith(".") and p.is_dir()]
    return sorted((n for n in names if k0 <= natural_key(n) <= k1), key=natural_key)


# --------------------------------------------------------------------------------------------- the plan

def _source_problem(problem: str) -> str:
    if problem.startswith("no COMPLETE"):
        return "still syncing (no COMPLETE yet; run the same range again later)"
    return problem


def _episode_slots(task_dir: Path) -> dict[str, str]:
    """``{pi_name: episode_N}`` of the folder's ``episode_N`` dirs (newest ledger row of each slot)."""
    from dataproc.intake import read_ledger

    by_slot = {r["slot"]: r.get("pi_name") or "" for r in read_ledger(task_dir / "episodes.csv")}
    return {by_slot[d.name]: d.name for d in episode_dirs(task_dir)
            if (d / "raw").is_dir() and by_slot.get(d.name)}


def plan_run(root: DataRoot, folder: str, start: str, end: str, source: Path, retry_failed: bool = False,
             force: bool = False) -> RunPlan:
    """The :class:`RunPlan` (see the module docstring). Raises :class:`RunError`; reads only."""
    from dataproc.pi_adapter import converted_name, inspect_source

    task_dir, task = check_folder(root, folder)
    source = Path(source)
    if not source.is_dir():
        raise RunError(f"--source {source} is not a directory")
    if root.incoming.resolve() == source.resolve():
        raise RunError(f"--source {source} is dataproc's own incoming/ (staging); point DATAPROC_ROOT elsewhere")
    names = select_range(source, start, end)
    plan = RunPlan(folder, task_dir, task.label, task.rig, source, start, end)
    taken = taken_sources(root)
    staged, failed, episodes = set(staging_names(task_dir)), set(failed_names(task_dir)), _episode_slots(task_dir)

    in_range = []
    for name in names:
        rec = Recording(name, SKIP)
        quick = inspect_source(source / name, verify_hashes=False)
        if not quick.ok:
            rec.why = _source_problem(quick.problem)
        elif quick.manifest_sha256 in taken:
            rec.taken = taken[quick.manifest_sha256]
            rec.source = quick
            _classify_taken(rec, folder, task_dir, staged, failed, episodes, retry_failed, force)
        elif converted_name(quick) in staged:
            rec.action, rec.pi_name, rec.source, rec.why = RESUME, converted_name(quick), quick, "in .staging"
        else:
            full = inspect_source(source / name)                     # verifies every file against the manifest
            if not full.ok:
                rec.why = _source_problem(full.problem)
            else:
                rec.action, rec.pi_name, rec.source = TAKE, converted_name(full), full
        in_range.append(rec)
    covered = {r.pi_name for r in in_range if r.action == RESUME}
    rows_of = {r.get("converted_name"): r for r in taken.values()
               if r.get("task") == folder and r.get("kind") == "demonstration"}
    leftovers = [Recording((rows_of.get(n) or {}).get("source_name") or n, RESUME, "left in .staging by an earlier "
                           "run", pi_name=n, taken=rows_of.get(n), in_range=False) for n in sorted(staged - covered)]
    plan.recordings = sorted(leftovers + in_range, key=lambda r: natural_key(r.name))
    return plan


def _classify_taken(rec: Recording, folder: str, task_dir: Path, staged: set, failed: set, episodes: dict,
                    retry_failed: bool, force: bool) -> None:
    """The action of a recording already in ``pi_sources.csv``."""
    row = rec.taken
    pi = rec.pi_name = row.get("converted_name") or None
    if row.get("kind") == "calibration":
        rec.why = f"is a calibration (of task folder {row.get('task')})"
    elif row.get("task") != folder:
        rec.why = f"taken by another task folder ({row.get('task')}: {row.get('kind')}, {row.get('status')})"
    elif pi in staged:
        rec.action, rec.why = RESUME, "in .staging"
    elif pi in failed:
        if retry_failed:
            rec.action, rec.why = RETRY, f"failed earlier at {format_reason(read_reason(task_dir, pi))}"
        else:
            rec.why = (f"failed earlier at {format_reason(read_reason(task_dir, pi))}; --retry-failed to try "
                       "again")
    elif pi in episodes:
        if force:
            rec.action, rec.slot, rec.why = REPROCESS, episodes[pi], f"{episodes[pi]}, --force"
        else:
            rec.why, rec.quiet = f"already {episodes[pi]} of this folder", True
    elif row.get("status") == "quarantined":
        rec.why = f"quarantined at intake (see {Path('quarantine') / (pi or '?')}*/reason.json)"
    else:
        rec.why = f"recorded in pi_sources.csv as {row.get('status')} for this folder, but {pi} is not in it"


def format_plan(plan: RunPlan) -> str:
    """One line per recording (the ``--dry-run`` output); quiet skips are counted on one line."""
    lines = [f"run: tasks/{plan.folder} (label {plan.label}, rig {plan.rig_id}), {plan.start}..{plan.end} in "
             f"{plan.source_dir}"]
    quiet = [r for r in plan.recordings if r.quiet]
    for r in plan.recordings:
        if not r.quiet:
            lines.append(f"  {r.name}: {r.describe()}")
    if quiet:
        lines.append(f"  {len(quiet)} already episodes of this folder: {', '.join(r.name for r in quiet)}")
    lines.append(f"{len(plan.to_do())} to process, {len(plan.skipped())} skipped")
    return "\n".join(lines)
