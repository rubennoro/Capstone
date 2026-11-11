"""``dataproc run TASK_FOLDER START_DIR END_DIR``: progress, the per-recording summary, the folder status and the
next command; exit 0 (done, also with failed recordings), 3 (waiting for SLAM) or 1 (the run could not run)."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from dataproc.paths import DataRoot
from dataproc.pi_intake import PiIntakeError
from dataproc.run.episode import EPISODE, FAILED, SKIPPED, WAITING
from dataproc.run.plan import format_plan
from dataproc.run.range import EXIT_FAILED, EXIT_OK, EXIT_WAITING_FOR_SLAM, RunResult, run_range

__all__ = ["cli_run", "format_summary"]


def format_summary(root: DataRoot, res: RunResult) -> str:
    """Per recording (quiet skips counted on one line), then ``dataproc status``, then what to do next."""
    from dataproc.status import format_status, task_status

    plan = res.plan
    lines = ["", f"run summary: tasks/{plan.folder}, {plan.start}..{plan.end}"]
    quiet = [o for o in res.outcomes if o.quiet]
    w = max([len(o.name) for o in res.outcomes if not o.quiet] or [1])
    lines += [f"  {o.name:<{w}}  ->  {o.describe()}" for o in res.outcomes if not o.quiet]
    if quiet:
        lines.append(f"  ({len(quiet)} already episodes of this folder: {', '.join(o.name for o in quiet)})")
    n = {s: len(res.by_state(s)) for s in (EPISODE, FAILED, WAITING, SKIPPED)}
    lines.append(f"  {n[EPISODE]} episode(s), {n[FAILED]} failed, {n[WAITING]} waiting for SLAM, {n[SKIPPED]} skipped")
    if res.error:
        lines.append(f"  stopped: {res.error}")
    lines.append("")
    try:
        lines.append(format_status(task_status(root, plan.folder)))
    except Exception as e:  # noqa: BLE001 - the summary must never hide the run's own outcome
        lines.append(f"(status unavailable: {e})")
    if res.exit_code == EXIT_WAITING_FOR_SLAM:
        lines += ["", "Waiting for SLAM: no backend is installed, so these recordings stop at the slam stage and wait "
                  "in .staging/. Build ORB-SLAM3 (docs/ORBSLAM3.md), then run the same command again (any range "
                  "resumes them); nothing before slam is redone."]
    if res.exit_code != EXIT_FAILED:
        lines += ["", "Next: package the episodes for training (pick a dataset name):",
                  f"  dataproc package {plan.label}-{time.strftime('%m%d')} {plan.folder}"]
    return "\n".join(lines)


def cli_run(root: DataRoot, folder: str, start_dir: str, end_dir: str, source="/data/incoming", slam: str = "auto",
            retry_failed: bool = False, force: bool = False, dry_run: bool = False) -> int:
    """``dataproc run``: returns the exit code. Errors go to stderr, progress and the summary to stdout."""
    try:
        res = run_range(root, folder, start_dir, end_dir, source=Path(source), slam=slam, retry_failed=retry_failed,
                        force=force, dry_run=dry_run, log=print)
    except (PiIntakeError, ValueError) as e:
        print(f"dataproc run: {e}", file=sys.stderr)
        return EXIT_FAILED
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    if dry_run:
        print(format_plan(res.plan))
        print(f"slam: {res.slam}" + (" (auto)" if slam == "auto" else "") + "; --dry-run: nothing was changed")
        return EXIT_OK
    print(format_summary(root, res))
    if res.error:
        print(f"dataproc run: stopped at {res.error}", file=sys.stderr)
    return res.exit_code
