"""``calibrate_task``: one Pi calibration recording -> a new, calibrated task folder, or nothing at all.

::

    res = calibrate_task(root, "block_raise", "episode_000001", source=Path("/data/incoming"), rig_id="rig_pi_v1")
    res.ok, res.folder, res.fit.opening_m, res.reasons

Steps (everything after the inspect step runs under ``FileLock($ROOT/.run.lock)``):

1. **Inspect** ``source/EPISODE_DIR`` (read only): it must exist, have ``COMPLETE``, match its manifest, not be
   marked invalid by the Pi, and not be in ``pi_sources.csv`` yet.
2. **Stage**: adapt it into ``incoming/<converted_name>/`` (built under ``incoming/.adapt-tmp/``).
3. **Fit and check** (:func:`~dataproc.stages.calibration.fit_calibration`): contract + decode, rig id and cameras
   vs ``--rig``, ArUco tags, gripper range, opening within the rig's ``gripper`` tolerance.
4. **On failure**: delete the staged copy. No folder is created and the recording is not recorded, so it can be
   retried once the cause is fixed.
5. **On success**: intake the staged episode as the calibration of the first free folder ``TASK``, ``TASK_2``, ...
   (``task.yaml`` label ``TASK``), write the fit where the process step reuses it without fitting again
   (:func:`~dataproc.stages.calibration.write_calibration_result`), and record the recording in
   ``pi_sources.csv`` (kind ``calibration``).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc.fsutil import FileLock
from dataproc.paths import DataRoot, validate_episode_dir_name, validate_name
from dataproc.pi_intake import (RUN_LOCK, PiIntakeError, adapt_sources, clear_staging, ensure_rig, intake_adapted,
                                taken_sources)

__all__ = ["CalibrateError", "CalibrateResult", "calibrate_task", "source_problem_code"]


class CalibrateError(PiIntakeError):
    """The command cannot run at all (bad arguments, ``--source`` missing, rig yaml invalid, ``incoming/`` not
    empty). Nothing is created."""


@dataclass
class CalibrateResult:
    """``ok``: the folder ``folder`` (``task_dir``) was created with this calibration. Otherwise ``reasons`` (codes,
    see :mod:`dataproc.calibrate.diagnostics`) and ``message`` say why; ``fit`` is set once the fit ran."""

    ok: bool
    task: str
    source_name: str
    source_dir: Path
    rig_id: str
    folder: str | None = None
    task_dir: Path | None = None
    slot: str | None = None                       # calibration/<name> inside the folder
    fit: object = None                            # CalibrationFit
    reasons: list[str] = field(default_factory=list)
    message: str = ""
    details: dict = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        return list(self.fit.warnings) if self.fit is not None else []


def source_problem_code(problem: str) -> str:
    """A reason code for a :attr:`~dataproc.pi_adapter.PiSource.problem` message."""
    if problem.startswith("not a directory"):
        return "source_missing"
    if problem.startswith("no COMPLETE"):
        return "still_syncing"
    if "manifest" in problem.lower() or problem.startswith("no MANIFEST"):
        return "manifest_failed"
    if problem.startswith("the Pi"):
        return "pi_invalid"
    return "meta_unreadable"


def calibrate_task(root: DataRoot, task: str, episode_dir: str, source: Path = Path("/data/incoming"),
                   rig_id: str = "rig_pi_v1", log: Callable = print) -> CalibrateResult:
    """``dataproc calibrate`` (see the module docstring). Returns the result for every recording problem; raises
    :class:`CalibrateError` (or :class:`~dataproc.pi_intake.PiIntakeError`) only when the command cannot run."""
    from dataproc.pi_adapter import inspect_source

    try:
        validate_name(task)
        validate_name(rig_id, "rig id")
        validate_episode_dir_name(episode_dir)
    except ValueError as e:
        raise CalibrateError(str(e)) from None
    source = Path(source)
    if not source.is_dir():
        raise CalibrateError(f"--source {source} is not a directory")
    if root.incoming.resolve() == source.resolve():
        raise CalibrateError(f"--source {source} is dataproc's own incoming/ (staging); point DATAPROC_ROOT "
                             "elsewhere")
    res = CalibrateResult(False, task, episode_dir, source / episode_dir, rig_id)

    # 1. inspect (read only)
    log(f"calibrate: inspecting {res.source_dir}")
    src = inspect_source(res.source_dir)
    if not src.ok:
        return _failed(res, [source_problem_code(src.problem)], src.problem)

    root.path.mkdir(parents=True, exist_ok=True)
    with FileLock(root.path / RUN_LOCK):
        prev = taken_sources(root).get(src.manifest_sha256)
        if prev is not None:
            res.details["taken"] = prev
            return _failed(res, ["already_taken"], f"already taken ({prev['kind']} of task folder {prev['task']}, "
                                                   f"{prev['status']})")
        rig = _load_rig(root, rig_id, log)
        clear_staging(root)

        # 2. stage
        adapted, problems = adapt_sources(root, [src], rig_id)
        if problems:
            return _failed(res, ["cannot_convert"], problems[src.name])
        staged = root.incoming / next(iter(adapted))
        try:
            # 3. fit and check
            log(f"calibrate: fitting the gripper range on {staged.name} (rig {rig_id})")
            from dataproc.imageprep import ImagePrep
            from dataproc.stages.calibration import fit_calibration

            res.fit = fit = fit_calibration(staged, rig, ImagePrep(rig), rig_id)
            if not fit.ok:
                return _failed(res, fit.reasons, fit.reason or "calibration fit failed")
            # 5. success: create the folder through intake, save the fit, record the source
            return _take(root, res, adapted, rig, log)
        finally:
            shutil.rmtree(staged, ignore_errors=True)       # 4. gone already when intake took it


def _load_rig(root: DataRoot, rig_id: str, log: Callable):
    from dataproc.config import ConfigError, load_rig

    ensure_rig(root, rig_id, log)
    try:
        return load_rig(root.rig_file(rig_id))
    except ConfigError as e:
        raise CalibrateError(str(e)) from None


def _take(root: DataRoot, res: CalibrateResult, adapted: dict, rig, log: Callable) -> CalibrateResult:
    """Intake the staged episode as the calibration of a new folder, then write the fit into it."""
    from dataproc.stages.calibration import write_calibration_result

    taken = intake_adapted(root, res.task, adapted, "calibration", calibration=True, new=True, log=log)
    outcome = taken.outcomes.get(res.source_name)
    if outcome is None or outcome.status != "ingested":
        why = outcome.describe() if outcome is not None else "not taken by intake"
        res.details["intake"] = why
        return _failed(res, ["intake"], f"intake did not take it in ({why})")
    res.folder, res.slot = taken.folder, outcome.slot
    res.task_dir = root.tasks / taken.folder
    write_calibration_result(res.task_dir / outcome.slot, res.task_dir, res.fit, rig, res.rig_id)
    res.ok = True
    return res


def _failed(res: CalibrateResult, reasons: list[str], message: str) -> CalibrateResult:
    res.ok, res.reasons, res.message = False, list(reasons), message
    return res
