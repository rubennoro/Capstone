"""``umiproc run TASK_FOLDER START_DIR END_DIR``: every demonstration in the range, one at a time, localized against
the folder's map (docs/DESIGN.md "The session", "Checks -> run", "Episodes"; DIRECTIONS.md sections 3 and 5).

Steps, all under ``<root>/.run.lock``:

1. **The folder and its map**: ``tasks/<folder>/task.yaml``; a map that is complete and passed (``report.json``),
   with ``map_atlas.osa`` and ``tx_slam_tag.json``; the effective sensor values (:func:`umiproc.sensors.load_sensors`)
   and the SLAM inputs written from them now. **Refused** when the fingerprint they give differs from the map's
   ``slam_fingerprint.json`` (the map is out of date: same rig file, SLAM config, lens file, mask, settings and SLAM
   build required; the differing keys are named).
2. The copied plan (``umiproc.dp.run.plan``, read only, without requiring a calibration): ``take`` (new),
   ``resume`` (in ``.staging/``, also outside the range), ``retry`` (``--retry-failed``) or ``skip`` with the reason
   (the map and calibration recordings, by their ``pi_sources.csv`` kind; another folder's; still syncing; failed
   earlier). The folder context (effective rig). **Refused** when the gripper calibration would be fitted again
   (its inputs changed) while the folder has episodes (:func:`check_calibration_unchanged`), or when ``incoming/``
   is not empty. Everything up to here only reads, so a refusal changes nothing.
3. Repairs after an interrupted run (the copied ``clear_staging`` and ``finish_interrupted``, then
   :func:`umiproc.run.recording.finish_interrupted_pieces`); the plan is made again when they changed what it reads.
4. **The gripper calibration**: the copied calibration step on the folder (reuses what ``umiproc calibrate`` saved)
   plus UMI's 80 % finger-tag rule on its recorded detections. Without a usable one every demo is localized and
   checked, then waits in ``.staging/``.
5. Each recording (:class:`umiproc.run.recording.RecordingProcessor`): ``episode_N`` (one per kept piece),
   ``failed/<pi_name>/``, or waiting for the gripper calibration.
6. ``tasks/<folder>/report.json``: this run's outcome per recording.

Exit codes: 0 every recording has its outcome (episodes or failed: both listed); 3 some wait for the gripper
calibration; 2 the command could not run (folder, map missing or out of date, SLAM build, range, gripper
calibration changed after episodes were made, incoming/ not empty, another command running: nothing was changed);
1 an unexpected error stopped the run (the recordings done so far keep their outcome).
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["EXIT_OK", "EXIT_FAILED", "EXIT_INPUT", "EXIT_WAITING", "REPORT_NAME", "REPORT_FORMAT", "WORK_SUBDIR",
           "RunInputError", "RunResult", "MapRef", "the_map", "check_map_fingerprint", "calibration_state",
           "check_calibration_unchanged", "run_range"]

EXIT_OK, EXIT_FAILED, EXIT_INPUT, EXIT_WAITING = 0, 1, 2, 3
REPORT_NAME = "report.json"
REPORT_FORMAT = "umiproc.run_report.v1"
WORK_SUBDIR = "run"                    # <root>/.work/run/: the SLAM inputs and each recording's EuRoC frames


class RunInputError(Exception):
    """The command cannot run (exit 2). Nothing was changed. ``code``: a short reason."""

    def __init__(self, message: str, code: str = "usage"):
        super().__init__(message)
        self.code = code


@dataclass
class MapRef:
    """The folder's map: its slot, atlas, table-tag pose and the fingerprint it was made with."""

    pi_name: str
    dir: Path
    atlas: Path
    tx_slam_tag: np.ndarray
    fingerprint_hash: str
    fingerprint_record: dict | None


@dataclass
class RunResult:
    exit_code: int
    plan: object = None
    outcomes: list = field(default_factory=list)
    calibration: dict | None = None
    map: MapRef | None = None
    fingerprint_hash: str | None = None
    report: dict | None = None
    error: str | None = None

    def by_state(self, state: str) -> list:
        return [o for o in self.outcomes if o.state == state]


def the_map(task_dir: Path) -> MapRef:
    """The folder's map, or :class:`RunInputError` (code ``map``) when there is none, it is incomplete or failed, or a
    file ``run`` needs is missing."""
    from umiproc.dp.files import read_json
    from umiproc.map.build import ATLAS_NAME, FINGERPRINT_NAME, TAG_POSE_NAME
    from umiproc.map.stored import read_map

    maps = read_map(task_dir)
    if not maps:
        raise RunInputError(f"task folder {task_dir.name} has no map: `umiproc map` makes the folder with its map",
                            "map")
    if len(maps) > 1:
        names = ", ".join(m["pi_name"] for m in maps)
        raise RunInputError(f"task folder {task_dir.name} has {len(maps)} maps ({names}): one folder is one session "
                            "with one map", "map")
    m = maps[0]
    if not m["complete"]:
        raise RunInputError(f"the map of {task_dir.name} (map/{m['pi_name']}) is INCOMPLETE (an interrupted `umiproc "
                            "map`, see `umiproc status`): record a new mapping clip, it makes a new session folder",
                            "map")
    if m["passed"] is not True:
        raise RunInputError(f"the map of {task_dir.name} (map/{m['pi_name']}) did not pass its checks", "map")
    d = m["dir"]
    if not (d / TAG_POSE_NAME).is_file():
        raise RunInputError(f"the map of {task_dir.name} (map/{m['pi_name']}) has no table-tag pose ({TAG_POSE_NAME}: "
                            "M5 saw tag 13 in no keyframe), so episodes cannot be put in the table's frame: map again "
                            "with tag 13 lying flat in view (a new mapping clip makes a new session folder)", "map")
    for f in (ATLAS_NAME, TAG_POSE_NAME, FINGERPRINT_NAME):
        if not (d / f).is_file():
            raise RunInputError(f"the map of {task_dir.name} has no {f} ({d}): it cannot be used", "map")
    tag = read_json(d / TAG_POSE_NAME, {}) or {}
    T = np.asarray(tag.get("tx_slam_tag"), dtype=np.float64) if tag.get("tx_slam_tag") is not None else None
    if T is None or T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise RunInputError(f"{d / TAG_POSE_NAME} has no valid 4x4 tx_slam_tag", "map")
    fp = read_json(d / FINGERPRINT_NAME, {}) or {}
    return MapRef(m["pi_name"], d, d / ATLAS_NAME, T, str(fp.get("hash") or m.get("fingerprint_hash")),
                  fp.get("record"))


def check_map_fingerprint(the: MapRef, current_hash: str, current_record: dict) -> None:
    """:class:`RunInputError` (code ``out_of_date``) when the current fingerprint is not the map's."""
    from umiproc.map.stored import fingerprint_differences

    if current_hash == the.fingerprint_hash:
        return
    diff = fingerprint_differences(the.fingerprint_record, current_record)
    raise RunInputError(
        f"the map (map/{the.pi_name}) is OUT OF DATE: it was made with fingerprint {the.fingerprint_hash[:12]}, the "
        f"current rig files and SLAM build give {current_hash[:12]} (they differ in: {', '.join(diff[:8])}). Demos "
        "must be localized with exactly the map's settings: put back what changed (rigs/ in the data root, the lens "
        "file, $UMIPROC_SLAM_BINARY / the build), or record a new mapping clip with the new settings (a new session "
        "folder) and process the demos there. `umiproc status <folder>` shows the same comparison.", "out_of_date")


def calibration_state(ctx) -> tuple[bool, str]:
    """``(usable, why)`` of the folder's gripper calibration after the copied calibration step (``ctx.calibration``):
    usable when the step is ``done`` and the calibration recording passes UMI's 80 % finger-tag rule (on the
    detections recorded with it)."""
    from umiproc.calibrate.gripper import finger_tag_rule
    from umiproc.dp.files import load_status

    cal = ctx.calibration
    if cal is None or cal.state == "no_calibration":
        return False, "no gripper calibration yet (`umiproc calibrate`)"
    if cal.state != "done":
        return False, f"the gripper calibration {cal.name} is {cal.state}" + (f": {cal.reason}" if cal.reason else "")
    rec = load_status(ctx.task_dir / "calibration" / cal.name / "processed").get("gripper_range") or {}
    rule = finger_tag_rule(rec.get("diagnostics") or {})
    if not rule["ok"]:
        return False, f"the gripper calibration {cal.name} fails UMI's finger-tag rule: {rule['message']}"
    return True, f"gripper calibration {cal.name}"


def check_calibration_unchanged(ctx) -> None:
    """:class:`RunInputError` (code ``calibration_changed``) when the folder's gripper calibration would be fitted
    again (the inputs it was fitted with, the copied ``calibration_inputs_hash``, differ from the current ones: the
    rig file's ``aruco`` / ``gripper`` / ultrawide sections, the lens file, the copied calibration version) while the
    folder already has episodes made with it: one calibration per session. Without episodes the copied calibration
    step simply fits it again, as before."""
    from umiproc.dp.files import load_status
    from umiproc.dp.paths import episode_dirs
    from umiproc.dp.stages.calibration import calibration_inputs_hash, newest_calibration

    cal_dir = newest_calibration(ctx.task_dir, ctx.ledger)
    if cal_dir is None:
        return
    prev = load_status(cal_dir / "processed").get("gripper_range") or {}
    if not prev.get("inputs_hash"):
        return
    if prev["inputs_hash"] == calibration_inputs_hash(cal_dir / "raw", ctx.rig, ctx.task.rig):
        return
    eps = [d.name for d in episode_dirs(ctx.task_dir) if os.path.lexists(d / "raw")]
    if not eps:
        return
    raise RunInputError(
        f"the gripper calibration calibration/{cal_dir.name} was fitted with other settings than the current ones "
        "(the rig file's aruco, gripper or ultrawide camera section, the lens file, or the calibration step's "
        f"version changed), and {ctx.task_dir.name} already has {len(eps)} episode(s) made with it "
        f"({', '.join(eps[:5])}{' ...' if len(eps) > 5 else ''}). Fitting it again now would give this session's "
        "episodes two different gripper ranges: one calibration per session. Put back the settings it was made "
        "with, or record a new session (mapping clip, gripper calibration, demos) with the new ones.",
        "calibration_changed")


def _check_incoming(r) -> None:
    """The copied ``clear_staging``'s refusal, read only: the data root's ``incoming/`` must hold nothing but
    umiproc's own temporary leftovers (dot-entries)."""
    inc = r.incoming
    if not inc.is_dir():
        return
    stray = sorted(p.name for p in inc.iterdir() if not p.name.startswith("."))
    if stray:
        raise RunInputError(f"{inc} is not empty ({', '.join(stray[:5])}{' ...' if len(stray) > 5 else ''}): an "
                            "interrupted command left it; move them away first", "incoming")


def _repair_state(task_dir: Path) -> tuple:
    """What the plan reads of the folder: the ledger, the slots on disk and the failed recordings' reasons. The
    repairs at the start of a run changed something the plan sees when this differs before and after them."""
    from umiproc.dp.paths import episode_dirs
    from umiproc.dp.run.staging import REASON_FILE, failed_names, staging_names

    ledger = task_dir / "episodes.csv"
    return (ledger.read_bytes() if ledger.is_file() else b"", tuple(staging_names(task_dir)),
            tuple((n, (task_dir / "failed" / n / REASON_FILE).is_file()) for n in failed_names(task_dir)),
            tuple((d.name, os.path.lexists(d / "raw")) for d in episode_dirs(task_dir)))


def _check_folder(r, folder: str):
    from umiproc.dp.run.plan import RunError, check_folder

    try:
        return check_folder(r, folder, require_calibration=False)
    except RunError as e:
        msg = str(e).replace("(or has no successful calibration): run `dataproc calibrate <task> <episode_dir>` "
                             "first", "(or no task.yaml): `umiproc map` makes it")
        raise RunInputError(msg, "folder") from None


def _plan(r, folder, start, end, source, retry_failed):
    from umiproc.dp.run.plan import RunError, plan_run

    try:
        return plan_run(r, folder, start, end, Path(source), retry_failed, False, require_calibration=False)
    except RunError as e:
        raise RunInputError(str(e).replace("dataproc", "umiproc"), "range") from None


def run_range(root, folder: str, start: str, end: str, *, source: Path = Path("/data/incoming"),
              retry_failed: bool = False, dry_run: bool = False, log: Callable = print, runner=None, binary=None,
              vocabulary=None, timeout_s: float | None = None) -> RunResult:
    """``umiproc run`` (module docstring). Raises :class:`RunInputError` (exit 2) when it cannot run; returns the
    :class:`RunResult` otherwise. ``runner``/``binary``/``vocabulary``/``timeout_s`` go to
    :func:`umiproc.slam.runner.run_umi_slam` (tests inject a fake runner)."""
    from umiproc.dp.paths import DataRoot, validate_name
    from umiproc.map.stored import current_fingerprint
    from umiproc.recordings import RecordingError, session_lock
    from umiproc.slam.runner import default_binary, default_vocabulary
    from umiproc.slam_inputs import SlamBuildError

    try:
        validate_name(folder, "task folder")
    except ValueError as e:
        raise RunInputError(str(e)) from None
    r = root if isinstance(root, DataRoot) else DataRoot(Path(root))
    binary = Path(binary).expanduser() if binary is not None else default_binary()
    vocabulary = Path(vocabulary).expanduser() if vocabulary is not None else default_vocabulary()
    if dry_run:
        task_dir, task = _check_folder(r, folder)
        the = the_map(task_dir)
        try:
            h, rec = current_fingerprint(r.path, task.rig, binary)
        except SlamBuildError as e:
            raise RunInputError(str(e), "slam_build") from None
        except ValueError as e:
            raise RunInputError(str(e), "rig_files") from None
        check_map_fingerprint(the, h, rec)
        plan = _plan(r, folder, start, end, source, retry_failed)
        from umiproc.dp.process import ProcessError, load_folder_context

        try:
            check_calibration_unchanged(load_folder_context(r, folder, slam="none", log=log))
        except ProcessError as e:
            raise RunInputError(str(e), "folder") from None
        _check_incoming(r)
        return RunResult(EXIT_OK, plan, [_skipped(x) for x in plan.skipped()], map=the, fingerprint_hash=h)
    try:
        lock = session_lock(r)
        lock.__enter__()
    except RecordingError as e:
        raise RunInputError(str(e), e.code) from None
    work = r.path / ".work" / WORK_SUBDIR
    try:
        return _run_locked(r, folder, start, end, Path(source), retry_failed, log, runner, binary, vocabulary,
                           timeout_s, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            work.parent.rmdir()
        except OSError:
            pass
        lock.__exit__(None, None, None)


def _run_locked(r, folder, start, end, source, retry_failed, log, runner, binary, vocabulary, timeout_s,
                work) -> RunResult:
    from umiproc.dp.pi_intake import clear_staging
    from umiproc.dp.process import ProcessError, load_folder_context, process_calibration
    from umiproc.dp.run.staging import finish_interrupted
    from umiproc.run.recording import RecordingProcessor, Session, finish_interrupted_pieces
    from umiproc.sensors import RigFilesError, SensorError, load_sensors
    from umiproc.slam_inputs import SlamBuildError, write_slam_inputs

    t0 = time.time()
    task_dir, task = _check_folder(r, folder)
    the = the_map(task_dir)
    try:
        sensors = load_sensors(r.path, task.rig, log)
    except (RigFilesError, SensorError) as e:
        raise RunInputError(str(e), "rig_files") from None
    if not vocabulary.is_file():
        raise RunInputError(f"ORB vocabulary not found: {vocabulary} (build with umi-processing/native/build.sh)",
                            "slam_build")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        inputs = write_slam_inputs(sensors, work / "inputs", binary)
    except SlamBuildError as e:
        raise RunInputError(str(e), "slam_build") from None
    check_map_fingerprint(the, inputs.fingerprint_hash, inputs.fingerprint)
    # everything that can refuse the run comes before the first change (the repairs below)
    plan = _plan(r, folder, start, end, source, retry_failed)
    try:
        ctx = load_folder_context(r, folder, slam="none", rig=sensors.rig, log=log)
    except ProcessError as e:
        raise RunInputError(str(e), "folder") from None
    check_calibration_unchanged(ctx)
    _check_incoming(r)
    before = _repair_state(task_dir)
    clear_staging(r)                                 # only removes umiproc's own temporary leftovers now
    finish_interrupted(r, task_dir, log)
    for slot in finish_interrupted_pieces(task_dir):
        log(f"reconciled: {slot} made from a piece an interrupted run left behind")
    if _repair_state(task_dir) != before:            # a repair moved or recorded something: read it again
        plan = _plan(r, folder, start, end, source, retry_failed)
        ctx = load_folder_context(r, folder, slam="none", rig=sensors.rig, log=log)
    cal = process_calibration(ctx)
    usable, cal_why = calibration_state(ctx)
    log(f"run: tasks/{folder} (label {task.label}), map/{the.pi_name}, fingerprint {inputs.fingerprint_hash[:12]}, "
        f"lens {sensors.lens_provenance}; " + (cal_why if usable else f"{cal_why}: demos will wait for it"))
    session = Session(root=r, folder=folder, task_dir=task_dir, label=task.label, rig_id=task.rig, sensors=sensors,
                      inputs=inputs, map_pi_name=the.pi_name, atlas=the.atlas, tx_slam_tag=the.tx_slam_tag, ctx=ctx,
                      calibrated=usable, work=work, log=log, runner=runner, binary=binary, vocabulary=vocabulary,
                      timeout_s=timeout_s)
    res = RunResult(EXIT_FAILED, plan, map=the, fingerprint_hash=inputs.fingerprint_hash,
                    calibration={**cal.to_report(), "usable": usable, "why": cal_why})
    todo = plan.to_do()
    log(f"run: {len(todo)} recording(s) to process, {len(plan.skipped())} skipped")
    rp = RecordingProcessor(session)
    done: dict[int, object] = {}
    for i, rec in enumerate(todo, start=1):
        log(f"[{i}/{len(todo)}] {rec.name}: {rec.describe()}")
        try:
            o = rp.run(rec)
        except Exception as e:  # noqa: BLE001 - stop the run, keep what was done
            res.error = f"{rec.name}: {type(e).__name__}: {e}"
            log(traceback.format_exc().rstrip())
            break
        done[id(rec)] = o
        log(f"[{i}/{len(todo)}] {rec.name} -> {o.describe()}  [{o.seconds:.1f} s]")
    from umiproc.dp.run.plan import SKIP
    from umiproc.run.recording import SKIPPED, WAITING, Outcome

    res.outcomes = [done[id(x)] if id(x) in done else _skipped(x) if x.action == SKIP else
                    Outcome(x.name, x.pi_name, SKIPPED, message="not processed (the run stopped)", action=x.action)
                    for x in plan.recordings]
    if res.error is None:
        res.exit_code = EXIT_WAITING if res.by_state(WAITING) else EXIT_OK
    res.report = _write_report(res, task_dir, task, source, sensors, t0)
    return res


def _skipped(rec):
    from umiproc.run.recording import SKIPPED, Outcome

    return Outcome(rec.name, rec.pi_name, SKIPPED, message=rec.why.replace("dataproc", "umiproc"), action=rec.action,
                   quiet=rec.quiet)


def _write_report(res: RunResult, task_dir: Path, task, source: Path, sensors, t0: float) -> dict:
    from umiproc.dp.files import iso_utc, write_json
    from umiproc.run.recording import EPISODE, FAILED, SKIPPED, WAITING
    from umiproc.run.rules import LIMITS

    plan = res.plan
    totals = {s: len(res.by_state(s)) for s in (EPISODE, FAILED, WAITING, SKIPPED)}
    totals["episode_folders"] = sum(len(o.slots) for o in res.by_state(EPISODE))
    report = {"format": REPORT_FORMAT, "folder": plan.folder, "label": task.label, "rig_id": task.rig,
              "source": str(source), "range": [plan.start, plan.end], "map": res.map.pi_name,
              "fingerprint_hash": res.fingerprint_hash, "lens": sensors.lens_record(), "exit_code": res.exit_code,
              "error": res.error, "calibration": res.calibration, "rules": LIMITS,
              "recordings": [o.to_dict() for o in res.outcomes], "totals": totals,
              "created_utc": iso_utc(time.time()), "duration_s": round(time.time() - t0, 1)}
    write_json(task_dir / REPORT_NAME, report)
    return report
