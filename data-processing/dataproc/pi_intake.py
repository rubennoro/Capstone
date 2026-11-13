"""Pi recordings -> intake, the part ``dataproc calibrate`` and ``dataproc run`` share (docs/PI_ADAPTER.md).

::

    with FileLock(root.path / RUN_LOCK):
        ensure_rig(root, rig_id, log)                 # rigs/<rig_id>.yaml from configs/rigs/ if missing
        clear_staging(root)                           # incoming/ must hold nothing else
        adapted, problems = adapt_sources(root, sources, rig_id)       # -> incoming/<converted_name>/
        taken = intake_adapted(root, task, adapted, "demonstration")   # intake + pi_sources.csv rows

Every source that reached intake (ingested or quarantined) is recorded in ``$DATAPROC_ROOT/pi_sources.csv``, so it
is never taken twice. A source that failed to convert is not recorded (it can be retried).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc.paths import DataRoot

__all__ = ["RUN_LOCK", "REPO_RIGS", "PiIntakeError", "TakeResult", "ensure_rig", "check_rig_copy", "clear_staging",
           "adapt_sources", "intake_adapted", "taken_sources", "set_source_status"]

#: Held (``fsutil.FileLock``) by every command that adapts Pi recordings into ``incoming/`` and takes them in.
RUN_LOCK = ".run.lock"
REPO_RIGS = Path(__file__).resolve().parents[1] / "configs" / "rigs"


class PiIntakeError(Exception):
    """Stops the command before anything is changed (or between steps, with what was done so far reported)."""


@dataclass
class TakeResult:
    """What :func:`intake_adapted` did: the task folder used, the intake outcome per source name, and the
    ``pi_sources.csv`` rows it appended."""

    folder: str | None
    outcomes: dict = field(default_factory=dict)       # source name -> EpisodeOutcome
    rows: list[dict] = field(default_factory=list)


def ensure_rig(root: DataRoot, rig_id: str, log: Callable = print) -> None:
    """Copy ``configs/rigs/<rig_id>.yaml`` to ``rigs/<rig_id>.yaml`` if it is missing there (never overwrites).
    An existing copy is checked by :func:`check_rig_copy`."""
    dst = root.rig_file(rig_id)
    src = REPO_RIGS / f"{rig_id}.yaml"
    if dst.is_file():
        check_rig_copy(dst, src)
        return
    if not src.is_file():
        raise PiIntakeError(f"no rig yaml for {rig_id!r}: neither {dst} nor {src} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log(f"rig: copied {src.name} to {dst}")


def check_rig_copy(dst: Path, src: Path) -> None:
    """Raise :class:`PiIntakeError` when the data root's rig yaml ``dst`` has no ``gripper`` section but the repo's
    ``src`` (same rig id) has one: ``dst`` is a copy from before the section existed, and with it the calibration
    opening and closed-width warnings would be skipped without anyone noticing. Nothing to check when ``src`` does not exist or either
    file cannot be parsed (loading the rig reports that)."""
    import yaml

    if not src.is_file():
        return
    try:
        mine, repo = yaml.safe_load(dst.read_text(encoding="utf-8")), yaml.safe_load(src.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(mine, dict) or not isinstance(repo, dict) or "gripper" in mine or "gripper" not in repo:
        return
    section = yaml.safe_dump({"gripper": repo["gripper"]}, sort_keys=False).rstrip()
    raise PiIntakeError(f"{dst} has no gripper section, but the repo's {src} has one: the copy predates it, and "
                        "without it the calibration opening warnings are skipped. Add the section to "
                        f"{dst}:\n{section}\nor delete {dst} (the repo copy is then used).")


def clear_staging(root: DataRoot) -> None:
    """``incoming/`` must hold nothing but our own leftovers, so intake can only take what this command adapted.
    Removes ``incoming/.adapt-tmp``; raises :class:`PiIntakeError` if anything else is there."""
    from dataproc.pi_adapter import ADAPT_TMP

    inc = root.incoming
    if not inc.is_dir():
        return
    shutil.rmtree(inc / ADAPT_TMP, ignore_errors=True)
    stray = sorted(p.name for p in inc.iterdir() if not p.name.startswith("."))
    if stray:
        raise PiIntakeError(f"{inc} is not empty ({', '.join(stray[:5])}{' ...' if len(stray) > 5 else ''}); "
                            "move them away first")


def taken_sources(root: DataRoot) -> dict[str, dict]:
    """``pi_sources.csv`` rows keyed by ``source_manifest_sha256``."""
    from dataproc.pi_adapter import SOURCES_FILE, read_sources

    return {r["source_manifest_sha256"]: r for r in read_sources(root.path / SOURCES_FILE)}


def adapt_sources(root: DataRoot, sources: list, rig_id: str) -> tuple[dict, dict[str, str]]:
    """Convert each :class:`~dataproc.pi_adapter.PiSource` into ``incoming/<converted_name>/``. Returns
    ``({converted_name: source}, {source name: why it could not be converted})``."""
    from dataproc.pi_adapter import PiEpisodeError, adapt_into

    root.incoming.mkdir(parents=True, exist_ok=True)
    adapted, problems = {}, {}
    for s in sources:
        try:
            adapted[adapt_into(s, root.incoming, rig_id).name] = s
        except PiEpisodeError as e:
            problems[s.name] = f"cannot convert: {e}"
    return adapted, problems


def set_source_status(root: DataRoot, manifest_sha256: str, status: str) -> bool:
    """Change the ``status`` of a recording in ``pi_sources.csv`` (see
    :func:`~dataproc.pi_adapter.update_source_status`)."""
    from dataproc.pi_adapter import SOURCES_FILE, update_source_status

    return update_source_status(root.path / SOURCES_FILE, manifest_sha256, status)


def intake_adapted(root: DataRoot, task: str, adapted: dict, kind: str, *, calibration: bool = False,
                   new: bool = False, staging: bool = False, log: Callable = print) -> TakeResult:
    """Run intake on the episodes :func:`adapt_sources` put in ``incoming/`` (``task``, ``new``, ``calibration``
    and ``staging`` as for :func:`~dataproc.intake.run_intake`), then append a ``pi_sources.csv`` row (``kind``,
    the folder, ingested / quarantined) for each source that reached intake. Raises :class:`PiIntakeError` if
    intake left anything in ``incoming/``."""
    from dataproc.intake import run_intake
    from dataproc.pi_adapter import SOURCES_FILE, append_sources

    res = TakeResult(None)
    if not adapted:
        return res
    result = run_intake(root, task, new=new, calibration=calibration, log=log, staging=staging)
    res.folder = result.folder
    for o in result.outcomes:
        s = adapted.get(o.pi_name)
        if s is None:
            continue
        res.outcomes[s.name] = o
        log(f"  {s.name} -> {o.pi_name} -> {o.describe()}")
        if o.status in ("ingested", "quarantined"):
            res.rows.append({"source_name": s.name, "source_manifest_sha256": s.manifest_sha256,
                             "converted_name": o.pi_name, "task": result.folder or task, "kind": kind,
                             "status": o.status, "adapted_unix_ns": time.time_ns()})
    append_sources(root.path / SOURCES_FILE, res.rows)
    left = sorted(p.name for p in root.incoming.iterdir() if not p.name.startswith("."))
    if left:
        raise PiIntakeError(f"intake left {left} in {root.incoming}; see the lines above")
    return res
