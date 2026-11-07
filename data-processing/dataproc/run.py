"""``dataproc run``: Pi episodes -> adapt -> intake -> process for one task, in one command (docs/PI_ADAPTER.md).

::

    dataproc run --task block_raise --calibration-episode episode_000001     # first time
    dataproc run --task block_raise                                         # later: new episodes, or resume

1. **Plan.** Scan ``--source`` (read only). The calibration episode is the named directory; every other complete,
   valid source not yet recorded in ``$DATAPROC_ROOT/pi_sources.csv`` is a demonstration of the task.
2. **Calibration.** Adapt it into ``incoming/`` and intake it with ``--calibration``.
3. **Demonstrations.** Adapt them all into ``incoming/`` and intake them.
4. **Record** every taken source in ``pi_sources.csv`` (ingested or quarantined), so it is never taken twice.
5. **Process** the task (``--slam auto``: ORB-SLAM3 if built, else ``none`` = stop at slam until it is).
6. **Summary**: per-episode status, the SLAM state, the output zip and the cluster hand-off commands.

Exit codes: 0 = output zip built, 3 = stopped at SLAM (no backend installed yet), 1 = anything else failed.
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc.fsutil import FileLock
from dataproc.paths import DataRoot, validate_name

__all__ = ["RunError", "RunResult", "run_pipeline", "cli_run", "EXIT_OK", "EXIT_FAILED", "EXIT_WAITING_FOR_SLAM"]

EXIT_OK, EXIT_FAILED, EXIT_WAITING_FOR_SLAM = 0, 1, 3
RUN_LOCK = ".run.lock"
REPO_RIGS = Path(__file__).resolve().parents[1] / "configs" / "rigs"


class RunError(Exception):
    """Stops the run before anything is changed (or between steps, with what was done so far reported)."""


@dataclass
class RunResult:
    exit_code: int
    slam: str | None = None
    calibration: str | None = None                       # source name taken as calibration this run
    demos: list[str] = field(default_factory=list)       # source names taken as demonstrations this run
    skipped: dict[str, str] = field(default_factory=dict)  # source name -> why it was not taken
    outcomes: dict[str, str] = field(default_factory=dict)  # source name -> intake outcome
    process: object | None = None                         # ProcessResult
    zip_path: Path | None = None


def _task_has_calibration(task_dir: Path) -> bool:
    cdir = task_dir / "calibration"
    return cdir.is_dir() and any((d / "raw").is_dir() for d in cdir.iterdir() if not d.name.startswith("."))


def _ensure_rig(root: DataRoot, rig_id: str, log) -> None:
    dst = root.rig_file(rig_id)
    if dst.is_file():
        return
    src = REPO_RIGS / f"{rig_id}.yaml"
    if not src.is_file():
        raise RunError(f"no rig yaml for {rig_id!r}: neither {dst} nor {src} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log(f"rig: copied {src.name} to {dst}")


def _clear_staging(root: DataRoot) -> None:
    """``incoming/`` must hold nothing but our own leftovers, so intake can only take what this run adapted."""
    from dataproc.pi_adapter import ADAPT_TMP

    inc = root.incoming
    if not inc.is_dir():
        return
    shutil.rmtree(inc / ADAPT_TMP, ignore_errors=True)
    stray = sorted(p.name for p in inc.iterdir() if not p.name.startswith("."))
    if stray:
        raise RunError(f"{inc} is not empty ({', '.join(stray[:5])}{' ...' if len(stray) > 5 else ''}); "
                       "run `dataproc intake` for them or move them away first")


def run_pipeline(root: DataRoot, task: str, calibration_episode: str | None = None,
                 source: Path = Path("/data/incoming"), rig_id: str = "rig_pi_v1", slam: str = "auto",
                 dry_run: bool = False, no_calibration: bool = False, episodes: list[str] | None = None,
                 backend=None,
                 log: Callable = print) -> RunResult:
    """The whole ``dataproc run`` (see module docstring). ``no_calibration``: allow a task without a calibration
    episode (everything is taken in and processed; the width stage then fails until one is added, so no zip is
    built). ``episodes``: take only these source directories as demonstrations (default: every new one).
    ``backend``: a SLAM backend instance (tests)."""
    from dataproc.intake import run_intake
    from dataproc.pi_adapter import (SOURCES_FILE, PiEpisodeError, adapt_into, append_sources, converted_name,
                                     read_sources, scan_sources)
    from dataproc.process import run_process
    from dataproc.slam import resolve_backend_name

    validate_name(task)
    validate_name(rig_id, "rig id")
    source = Path(source)
    if not source.is_dir():
        raise RunError(f"--source {source} is not a directory")
    if root.incoming.resolve() == source.resolve():
        raise RunError(f"--source {source} is dataproc's own incoming/ (staging); point DATAPROC_ROOT elsewhere")
    task_dir = root.tasks / task
    res = RunResult(EXIT_FAILED)

    # 1. plan
    taken = {r["source_manifest_sha256"]: r for r in read_sources(root.path / SOURCES_FILE)}
    cal_src, demos = None, []
    wanted = None if episodes is None else list(dict.fromkeys(episodes))
    if wanted is not None:
        missing = [n for n in wanted if not (source / n).is_dir() or n.startswith(".")]
        if missing:
            raise RunError(f"--episodes: not a directory in {source}: {', '.join(missing)}")
        if calibration_episode in wanted:
            raise RunError(f"{calibration_episode} is both --calibration-episode and in --episodes")
    for s in scan_sources(source):
        if wanted is not None and s.name not in wanted and s.name != calibration_episode:
            continue                                     # not selected: left for a later run
        if wanted is not None and s.name in wanted and (s.manifest_sha256 in taken or not s.ok):
            why = (f"already taken ({taken[s.manifest_sha256]['kind']} of task {taken[s.manifest_sha256]['task']})"
                   if s.manifest_sha256 in taken else s.problem)
            raise RunError(f"--episodes {s.name} cannot be used: {why}")
        if s.manifest_sha256 in taken:
            r = taken[s.manifest_sha256]
            res.skipped[s.name] = f"already taken ({r['kind']} of task {r['task']}, {r['status']})"
            if s.name == calibration_episode:
                raise RunError(f"--calibration-episode {s.name} was already taken in ({r['kind']} of task "
                               f"{r['task']}); record a new calibration")
        elif not s.ok:
            res.skipped[s.name] = s.problem
            if s.name == calibration_episode:
                raise RunError(f"--calibration-episode {s.name} cannot be used: {s.problem}")
        elif s.name == calibration_episode:
            cal_src = s
        else:
            demos.append(s)
    if calibration_episode is not None and cal_src is None:
        raise RunError(f"--calibration-episode {calibration_episode} is not a directory in {source}")
    if calibration_episode is not None and no_calibration:
        raise RunError("--calibration-episode and --no-calibration contradict each other")
    if cal_src is None and not _task_has_calibration(task_dir):
        if not no_calibration:
            raise RunError(f"task {task} has no calibration yet: pass --calibration-episode <dir in {source}> "
                           "(or --no-calibration to process without one; no training zip until one is added)")
        log(f"note: task {task} has no calibration (--no-calibration): episodes stop at the width stage and no "
            "zip is built until you add one with --calibration-episode")
    res.calibration = cal_src.name if cal_src else None
    res.demos = [d.name for d in demos]
    res.slam = resolve_backend_name(slam)
    import os

    env_root = os.environ.get("DATAPROC_ROOT")
    log(f"data root: {root.path}" + (" (from $DATAPROC_ROOT; `unset DATAPROC_ROOT` to use ~/data)"
                                     if env_root and Path(env_root).expanduser() == root.path else ""))
    cal_label = (res.calibration if res.calibration else "(task already has one)" if _task_has_calibration(task_dir)
                 else "(none: --no-calibration)")
    log(f"source {source}: calibration {cal_label}, "
        f"{len(demos)} new demonstration(s): {', '.join(res.demos) or '-'}")
    for name, why in res.skipped.items():
        log(f"  skip {name}: {why}")
    log(f"slam: {res.slam}" + (" (auto)" if slam == "auto" else ""))
    if dry_run:
        res.exit_code = EXIT_OK
        return res

    root.path.mkdir(parents=True, exist_ok=True)
    with FileLock(root.path / RUN_LOCK):
        _ensure_rig(root, rig_id, log)
        _clear_staging(root)
        root.incoming.mkdir(parents=True, exist_ok=True)

        # 2.-4. adapt + intake, recording every source that reached intake
        def take(srcs, kind: str, calibration: bool) -> None:
            names = {}
            for s in srcs:
                try:
                    names[adapt_into(s, root.incoming, rig_id).name] = s
                except PiEpisodeError as e:
                    res.skipped[s.name] = f"cannot convert: {e}"
                    log(f"  skip {s.name}: cannot convert: {e}")
            if not names:
                return
            result = run_intake(root, task, calibration=calibration, log=log)
            rows = []
            for o in result.outcomes:
                s = names.get(o.pi_name)
                if s is None:
                    continue
                res.outcomes[s.name] = o.describe()
                log(f"  {s.name} -> {o.pi_name} -> {o.describe()}")
                if o.status in ("ingested", "quarantined"):
                    rows.append({"source_name": s.name, "source_manifest_sha256": s.manifest_sha256,
                                 "converted_name": o.pi_name, "task": task, "kind": kind, "status": o.status,
                                 "adapted_unix_ns": time.time_ns()})
            append_sources(root.path / SOURCES_FILE, rows)
            left = sorted(p.name for p in root.incoming.iterdir() if not p.name.startswith("."))
            if left:
                raise RunError(f"intake left {left} in {root.incoming}; see the lines above")

        if cal_src is not None:
            take([cal_src], "calibration", calibration=True)
            if not _task_has_calibration(task_dir):
                raise RunError(f"calibration episode {cal_src.name} was not taken in "
                               f"({res.outcomes.get(cal_src.name, res.skipped.get(cal_src.name))})")
        take(demos, "demonstration", calibration=False)

        # 5. process
        if not task_dir.is_dir():
            raise RunError(f"nothing was taken into task {task}")
        log(f"process: task {task}, slam {res.slam}")
        pr = res.process = run_process(root, task, slam=res.slam, backend=backend, log=log)
        res.zip_path = pr.zip_path
        if pr.exit_code == 0:
            res.exit_code = EXIT_OK
        elif res.slam == "none" and pr.report is not None:
            res.exit_code = EXIT_WAITING_FOR_SLAM
        else:
            res.exit_code = EXIT_FAILED
            if pr.error:
                log(f"process: {pr.error}")
    return res


def _summary(root: DataRoot, task: str, res: RunResult) -> str:
    from dataproc.status import format_status, task_status

    lines = [""]
    try:
        lines.append(format_status(task_status(root, task)))
    except Exception as e:  # noqa: BLE001 - the summary must never hide the run's own outcome
        lines.append(f"(status unavailable: {e})")
    if res.exit_code == EXIT_WAITING_FOR_SLAM:
        lines += ["", "Waiting for SLAM: no backend is installed, so every episode stops at the slam stage",
                  "(slam_unavailable). Build ORB-SLAM3 (docs/ORBSLAM3.md), then re-run:",
                  f"  dataproc run --task {task}", "Everything before slam is kept and not redone."]
    elif res.exit_code == EXIT_OK and res.zip_path:
        lines += ["", f"Output: {res.zip_path}  (trains umi_policy_dunetp and umi_policy_dunet unchanged)",
                  "To train on the cluster (pick a dataset name, e.g. "
                  f"{task}-{time.strftime('%m%d')}):",
                  f"  ssh <you>@xfer.discovery.neu.edu mkdir -p CapstoneProject/datasets/{task}/<dataset>",
                  f"  rsync -av {res.zip_path} <you>@xfer.discovery.neu.edu:CapstoneProject/datasets/{task}/"
                  "<dataset>/<dataset>.zarr.zip",
                  f"  scripts/run_pipeline.sh train --task {task} --dataset <dataset> --note \"...\"   (on the laptop)"]
    rep = getattr(res.process, "report", None) or {}
    if res.exit_code == EXIT_OK and rep.get("totals", {}).get("exported", 0) < 2:
        lines.append("warning: fewer than 2 exported demonstrations; behavior prompting (dunetp) needs >= 2 per task")
    return "\n".join(lines)


def cli_run(root: DataRoot, task: str, calibration_episode: str | None = None, source="/data/incoming",
            rig_id: str = "rig_pi_v1", slam: str = "auto", dry_run: bool = False,
            no_calibration: bool = False, episodes: list[str] | None = None) -> int:
    """``dataproc run``: prints progress and the summary; returns the exit code."""
    try:
        res = run_pipeline(root, task, calibration_episode=calibration_episode, source=Path(source), rig_id=rig_id,
                           slam=slam, dry_run=dry_run, no_calibration=no_calibration,
                           episodes=episodes, log=print)
    except (RunError, ValueError) as e:
        print(f"dataproc run: {e}", file=sys.stderr)
        return EXIT_FAILED
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    if not dry_run:
        print(_summary(root, task, res))
    return res.exit_code
