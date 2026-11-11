"""``run_range``: the whole ``dataproc run TASK_FOLDER START_DIR END_DIR`` (docs/PI_ADAPTER.md).

::

    res = run_range(root, "block_raise", "episode_000002", "episode_000010", source=Path("/data/incoming"),
                    slam="auto")
    res.exit_code, [o.describe() for o in res.outcomes], res.report

1. **Plan** (:func:`~dataproc.run.plan.plan_run`): the folder must be calibrated; every recording in the range
   gets an action or a skip reason. ``dry_run`` stops here (nothing is changed, no lock is taken).
2. Under ``FileLock($ROOT/.run.lock)``: the rig yaml (:func:`~dataproc.pi_intake.ensure_rig`), an empty
   ``incoming/``, then :func:`~dataproc.run.staging.finish_interrupted` and the plan again (now under the lock).
3. **The folder context**, once: task.yaml, rig, SLAM backend, and the calibration (reused as ``dataproc
   calibrate`` saved it, never fitted again unless its inputs changed).
4. **Each recording**, one at a time (:class:`~dataproc.run.episode.RecordingProcessor`): ``episode_N``,
   ``failed/``, or waiting for SLAM in ``.staging/``.
5. ``tasks/<folder>/report.json``: the outcome of every recording of this run.

Exit codes: 0 = done (also when some recordings failed: they are listed), 3 = some wait for SLAM, 1 = the run could
not run or stopped (the recordings done so far keep their outcome).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc import __version__
from dataproc.files import iso_utc, write_json
from dataproc.fsutil import FileLock
from dataproc.paths import DataRoot
from dataproc.pi_intake import RUN_LOCK, clear_staging, ensure_rig
from dataproc.run.episode import EPISODE, FAILED, SKIPPED, WAITING, Outcome, RecordingProcessor
from dataproc.run.plan import SKIP, RunError, RunPlan, plan_run

__all__ = ["EXIT_OK", "EXIT_FAILED", "EXIT_WAITING_FOR_SLAM", "REPORT_FILE", "RunResult", "run_range"]

EXIT_OK, EXIT_FAILED, EXIT_WAITING_FOR_SLAM = 0, 1, 3
REPORT_FILE = "report.json"


@dataclass
class RunResult:
    """``outcomes``: one per recording of the plan (processed or skipped), in plan order. ``error``: why the run
    stopped (exit 1); ``report``: what was written to ``tasks/<folder>/report.json`` (None for a dry run)."""

    exit_code: int
    plan: RunPlan | None = None
    slam: str | None = None
    outcomes: list[Outcome] = field(default_factory=list)
    calibration: dict | None = None
    report: dict | None = None
    error: str | None = None

    def by_state(self, state: str) -> list[Outcome]:
        return [o for o in self.outcomes if o.state == state]


def run_range(root: DataRoot, folder: str, start_dir: str, end_dir: str, source: Path = Path("/data/incoming"),
              slam: str = "auto", retry_failed: bool = False, force: bool = False, dry_run: bool = False,
              backend=None, log: Callable = print) -> RunResult:
    """``dataproc run`` (see the module docstring). Raises :class:`~dataproc.run.plan.RunError` when the run cannot
    start; a problem while processing stops it and is returned as ``error`` (exit 1). ``backend``: a SLAM backend
    instance (tests)."""
    from dataproc.slam import BACKENDS, resolve_backend_name

    slam_name = resolve_backend_name(slam)
    if slam_name not in BACKENDS:
        raise RunError(f"unknown --slam {slam!r} (expected auto or one of {sorted(BACKENDS)})")
    if force and slam_name == "none":
        raise RunError("--force re-runs slam, which needs a SLAM backend (--slam none / auto without ORB-SLAM3)")
    source = Path(source)
    if dry_run:
        plan = plan_run(root, folder, start_dir, end_dir, source, retry_failed, force)
        return RunResult(EXIT_OK, plan, slam_name, [_skipped(r) for r in plan.skipped()])
    lock = FileLock(root.path / RUN_LOCK)
    try:
        lock.acquire()
    except BlockingIOError:
        raise RunError(f"another dataproc calibrate/run holds {lock.path}; wait for it") from None
    try:
        return _run_locked(root, folder, start_dir, end_dir, source, slam_name, retry_failed, force, backend, log)
    finally:
        lock.release()


def _run_locked(root, folder, start_dir, end_dir, source, slam_name, retry_failed, force, backend, log) -> RunResult:
    from dataproc.process import ProcessError, load_folder_context, process_calibration
    from dataproc.run.staging import finish_interrupted

    t0 = time.time()
    plan = plan_run(root, folder, start_dir, end_dir, source, retry_failed, force)    # checks the folder first
    ensure_rig(root, plan.rig_id, log)
    clear_staging(root)
    finish_interrupted(root, plan.task_dir, log)
    plan = plan_run(root, folder, start_dir, end_dir, source, retry_failed, force)    # after the repairs
    res = RunResult(EXIT_FAILED, plan, slam_name)
    try:
        ctx = load_folder_context(root, folder, slam=slam_name, backend=backend, log=log)
    except ProcessError as e:
        raise RunError(str(e)) from None
    cal = process_calibration(ctx)
    res.calibration = cal.to_report()
    if cal.state != "done":
        raise RunError(f"the calibration of {folder} is {cal.state}" + (f": {cal.reason}" if cal.reason else "")
                       + " (the rig yaml changed since dataproc calibrate?); calibrate a new folder")
    todo = plan.to_do()
    log(f"run: tasks/{folder}, {len(todo)} recording(s) to process, {len(plan.skipped())} skipped, slam {slam_name}")
    rp = RecordingProcessor(root, ctx, log)
    done: dict[int, Outcome] = {}
    for i, rec in enumerate(todo, start=1):
        log(f"[{i}/{len(todo)}] {rec.name}: {rec.describe()}")
        try:
            o = rp.run(rec, force=force and rec.in_range)
        except Exception as e:  # noqa: BLE001 - stop the run, keep what was done
            res.error = f"{rec.name}: {type(e).__name__}: {e}"
            log(traceback.format_exc().rstrip())
            break
        done[id(rec)] = o
        log(f"[{i}/{len(todo)}] {rec.name} -> {o.describe()}  [ran: {','.join(o.ran) or 'nothing'}; "
            f"{o.seconds:.1f} s]")
    res.outcomes = [done[id(r)] if id(r) in done else _skipped(r) if r.action == SKIP else
                    Outcome(r.name, r.pi_name, SKIPPED, message="not processed (the run stopped)", action=r.action)
                    for r in plan.recordings]
    if res.error is None:
        res.exit_code = EXIT_WAITING_FOR_SLAM if res.by_state(WAITING) else EXIT_OK
    res.report = _write_report(res, ctx, source, t0)
    return res


def _skipped(rec) -> Outcome:
    return Outcome(rec.name, rec.pi_name, SKIPPED, message=rec.why, action=rec.action, quiet=rec.quiet)


def _write_report(res: RunResult, ctx, source: Path, t0: float) -> dict:
    """``tasks/<folder>/report.json``: this run's outcome per recording, totals, calibration, versions."""
    plan = res.plan
    totals = {s: len(res.by_state(s)) for s in (EPISODE, FAILED, WAITING, SKIPPED)}
    report = {"folder": plan.folder, "label": plan.label, "rig_id": plan.rig_id, "source": str(source),
              "range": [plan.start, plan.end], "slam_backend": res.slam, "exit_code": res.exit_code,
              "error": res.error, "calibration": res.calibration, "recordings": [o.to_dict() for o in res.outcomes],
              "totals": totals, "rig_nominal_sections": ctx.rig.nominal_sections(),
              "dataproc_version": __version__, "created": iso_utc(time.time()), "duration_s": time.time() - t0}
    write_json(plan.task_dir / REPORT_FILE, report)
    return report
