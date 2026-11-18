"""``umiproc map``: one mapping recording -> the session's task folder with a checked map, or nothing at all.

UMI's recipe (``umi_repo/scripts_slam_pipeline``):

* ``02_create_map.py:71-89``: the mapping run, SLAM with the gripper mask and ``--save_map`` (the fork's
  ``gopro_slam`` has the table-tag initialisation on by default, id 13, 0.16 m: ``gopro_slam.cc:140-144``); here
  ``umiproc_slam --save_map map_atlas.osa --mask_img slam_mask.png --init_tag_id 13 --init_tag_size 0.16`` (the
  SLAM config's table tag), frames fed in real time.
* ``03_batch_slam.py:50, 113-123``: the mapping clip (``map*``) is localized again, like every demo, against the
  saved map: ``--load_map --mask_img --max_lost_frames 60`` (60 frames at 60 fps = 1 s); here 30 frames at 30 fps.
* ``05_run_calibrations.py:26-46`` + ``scripts/calibrate_slam_tag.py``: the table tag's pose in the map from that
  re-localized trajectory's keyframes (:mod:`umiproc.map.tag`), written as ``tx_slam_tag.json``.

Then the checks M1-M5 (:mod:`umiproc.map.checks`; UMI has none). Only M1 and M5's sightings **block**
(:data:`umiproc.map.checks.BLOCKING`); M2, M3, M4 and M5's spread and tilt are **warnings**: printed, listed in the
report's ``warnings``, and the folder is created anyway. The test demos (DIRECTIONS.md section 3) are the real gate.

Steps, all under ``<root>/.run.lock``:

1. Rig file + SLAM config (copied into ``<root>/rigs/`` when missing) and the **effective** sensor values
   (:func:`umiproc.sensors.load_sensors`: the lens file's model when there is one). Every step below uses
   ``SensorValues.rig`` (settings, EuRoC conversion, tag poses).
2. The recording: ``inspect_recording`` (read only; not taken yet), ``recording_mismatches`` (refused on a
   mismatch), the SLAM inputs (settings, mask, fingerprint: fails before anything is staged when the build is
   missing), ``stage`` (converted into ``<root>/incoming/<pi_name>/``). The copied intake's own refusals (contract
   validation, duplicate) are checked before SLAM: such a recording is taken straight away as ``quarantined``, as
   the copied intake does, and no SLAM runs.
3. Work folder ``<root>/.work/map/<pi_name>/``: the EuRoC folder (copied ``write_euroc``, native 1280x960, no mask
   in the frames), the table-tag detections in every frame (same decode), the mapping run, the re-localization run,
   the checks.
4. **Pass** (every blocking check passed; warnings allowed): ``report.json`` (with the folder and slot) is written into the work folder, then
   ``take(MAP, task=TASK)`` creates ``tasks/TASK`` (``TASK_2``, ... when taken) with the recording in
   ``map/<pi_name>/raw``; the map is moved next to it (:data:`MAP_FILES`), ``report.json`` last (``os.replace``). If
   intake raises or quarantines it, the report, logs and atlas go to ``maps_failed/`` (exit 2).
5. **Fail** (a blocking check failed): the staged copy is removed and nothing is recorded anywhere (no task folder, no ``pi_sources.csv`` row,
   no ``episodes.csv`` row), so the same recording can be run again (another build, another IMU setting, or after
   fixing a cause outside the recording). The report and logs are kept in
   ``<root>/maps_failed/<pi_name>/<UTC time>/`` (without the atlas and the EuRoC frames).
6. The work folder is removed (a crash leaves it; the next ``map`` clears it). A hard kill (power, SIGKILL, a closed
   ssh session without tmux) after ``take()`` leaves the map files and ``report.json`` there until the next ``map``
   clears it (manual recovery: DIRECTIONS.md section 3, "An interrupted map"); Ctrl-C runs the clean-up.

**First map piece** (``--first-map``, the user's decision 2026-11-18). When the whole clip's mapping run ends with more
than one map with keyframes (SLAM lost tracking and started a new map), its per-frame states give the map pieces
(:func:`umiproc.map.checks.map_pieces`); the first one that tracked more than a map dropped right after initialising
(:data:`umiproc.map.checks.FIRST_MAP_MIN_TRACKED` frames) ends at its last tracked frame. The clip is cut there (the
same EuRoC frames, times.txt shortened) and the mapping run, the re-localization and every check run again on the cut
clip; the whole clip's run is kept in ``mapping_full/``. The cut is in ``report.json`` (``first_map``), not in the
SLAM fingerprint (that is the SLAM inputs, which a demo must share with the map).

**Longest map piece: the default** (2026-11-18, the user's decision). Without ``--first-map``, a clip whose whole run
ends with several maps is cut to the piece that tracked the most frames (:func:`umiproc.map.checks.longest_map_piece`)
by itself, no flag needed: the fed frames from right after the previous piece's last frame (where SLAM began
initialising this map; 0 for the first piece) to the piece's last tracked frame, mapped and checked again like the
``--first-map`` cut; the cut is in ``report.json`` (``longest_map``). ``--longest-map`` only names this default. M1
still blocks when even the cut clip gives no single map (its hint says to record a new mapping clip).

**Map points.** The mapping run(s) (``mapping/``, and ``mapping_full/`` with ``--first-map`` / ``--longest-map``) pass
``--save_points <run>/map_points.csv`` when the build has it (:func:`umiproc.slam.runner.supports_save_points`, its
``--help`` asked once; an older build: no points, logged), so the map's points, keyframes and meta are stored with the
run for ``umiproc slam-3d`` (the EuRoC folder is deleted). ``map`` draws nothing itself (2026-11-18): figures, videos
and the 3-D page come from ``umiproc slam-video FOLDER`` and ``umiproc slam-3d FOLDER`` into ``tasks/<folder>/analysis/``.

Exit codes (:data:`EXIT_OK`, :data:`EXIT_FAILED`, :data:`EXIT_INPUT`): 0 the map was attached (possibly with
warnings); 1 a blocking check failed (or an unexpected error); 2 the command could not run on this input (bad name, recording missing / still syncing / already
taken / not this rig's settings / refused by intake, rig files, SLAM build missing, another command running).
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from umiproc.map import checks as C

__all__ = ["EXIT_OK", "EXIT_FAILED", "EXIT_INPUT", "REPORT_FORMAT", "REPORT_NAME", "ATLAS_NAME", "TAG_POSE_NAME",
           "TAG_DETECTIONS_NAME", "FINGERPRINT_NAME", "MAPPING_DIR", "RELOCALIZE_DIR", "MAPPING_FULL_DIR", "POINTS_NAME",
           "FAILED_DIR", "WORK_DIR",
           "MAP_FILES", "MapInputError", "MapOutcome", "build_map", "failed_map_dir", "patches_label"]

EXIT_OK, EXIT_FAILED, EXIT_INPUT = 0, 1, 2
REPORT_FORMAT = "umiproc.map_report.v1"
REPORT_NAME = "report.json"
ATLAS_NAME = "map_atlas.osa"
TAG_POSE_NAME = "tx_slam_tag.json"
TAG_DETECTIONS_NAME = "tag_detections.json"
FINGERPRINT_NAME = "slam_fingerprint.json"
MAPPING_DIR, RELOCALIZE_DIR = "mapping", "relocalize"
#: ``--first-map``: the whole clip's mapping run, kept when the clip was cut to its first map piece
MAPPING_FULL_DIR = "mapping_full"
#: the cut clip's EuRoC folder (a times.txt and a link to the whole clip's frames and IMU)
CUT_EUROC_DIR = "euroc_cut"
#: The map export (``--save_points``) stored in the mapping run folder(s), for :mod:`umiproc.viz.slam_3d`
POINTS_NAME = "map_points.csv"
FAILED_DIR = "maps_failed"
WORK_DIR = ".work"
#: What a passed map leaves next to ``raw/`` in ``map/<pi_name>/`` (a failed one: the same without the atlas).
MAP_FILES = (ATLAS_NAME, "orbslam3_settings.yaml", "slam_mask.png", FINGERPRINT_NAME, MAPPING_DIR, RELOCALIZE_DIR,
             MAPPING_FULL_DIR, TAG_DETECTIONS_NAME, TAG_POSE_NAME, REPORT_NAME)


class MapInputError(Exception):
    """The command cannot run on this input (exit 2). Nothing was created or recorded. ``code``: a short reason."""

    def __init__(self, message: str, code: str = "usage"):
        super().__init__(message)
        self.code = code


@dataclass
class MapOutcome:
    """The result of :func:`build_map`. ``where``: the folder holding the map and report (the task folder's
    ``map/<pi_name>/`` on a pass, ``maps_failed/<pi_name>/<UTC>/`` on a failure)."""

    passed: bool
    exit_code: int
    pi_name: str | None = None
    folder: str | None = None
    task_dir: Path | None = None
    where: Path | None = None
    checks: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    message: str = ""


def patches_label(patches: list) -> str:
    """``"0001-0009 (9 patches)"`` from the fingerprint's patch list (names start with their number)."""
    nums = [str(p.get("name", ""))[:4] for p in patches]
    if not nums:
        return "none"
    return f"{nums[0]}-{nums[-1]} ({len(nums)} patches)" if len(nums) > 1 else f"{nums[0]} (1 patch)"


def failed_map_dir(root: Path, pi_name: str, now: float | None = None) -> Path:
    """``<root>/maps_failed/<pi_name>/<UTC time>/`` (``-2``, ``-3``, ... appended when that second is taken)."""
    stamp = _dt.datetime.fromtimestamp(time.time() if now is None else now, _dt.timezone.utc)
    base = Path(root) / FAILED_DIR / pi_name / stamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    d, k = base, 1
    while d.exists():
        k += 1
        d = base.with_name(f"{base.name}-{k}")
    return d


# --------------------------------------------------------------------------------------------- the command

def build_map(root, task: str, map_dir: str, *, source: Path = Path("/data/incoming"), rig_id: str = "rig_pi_v3",
              log: Callable = print, runner=None, binary=None, vocabulary=None, timeout_s: float | None = None,
              first_map: bool = False, longest_map: bool = False) -> MapOutcome:
    """``umiproc map TASK MAP_DIR`` (module docstring). Raises :class:`MapInputError` (exit 2) when it cannot run;
    returns the :class:`MapOutcome` otherwise. ``runner``/``binary``/``vocabulary``/``timeout_s`` go to
    :func:`umiproc.slam.runner.run_umi_slam` (tests inject a fake runner). ``first_map``: ``--first-map`` (module
    docstring, "First map piece"); ``longest_map``: ``--longest-map`` (the same, with the longest piece)."""
    from umiproc.dp.paths import DataRoot, validate_episode_dir_name, validate_name
    from umiproc.recordings import RecordingError, discard, inspect_recording, session_lock, stage
    from umiproc.sensors import RigFilesError, SensorError, load_sensors, recording_mismatches
    from umiproc.slam.runner import default_binary, default_vocabulary
    from umiproc.slam_inputs import SlamBuildError, write_slam_inputs

    try:
        validate_name(task)
        validate_name(rig_id, "rig id")
        validate_episode_dir_name(map_dir)
        if first_map and longest_map:
            raise ValueError("--first-map and --longest-map exclude each other: pick one")
    except ValueError as e:
        raise MapInputError(str(e)) from None
    r = root if isinstance(root, DataRoot) else DataRoot(Path(root))
    binary = Path(binary).expanduser() if binary is not None else default_binary()
    vocabulary = Path(vocabulary).expanduser() if vocabulary is not None else default_vocabulary()
    try:
        lock = session_lock(r)
        lock.__enter__()
    except RecordingError as e:
        raise MapInputError(str(e), e.code) from None
    try:
        try:
            sensors = load_sensors(r.path, rig_id, log)
        except (RigFilesError, SensorError) as e:
            raise MapInputError(str(e), "rig_files") from None
        log(f"map: inspecting {Path(source) / map_dir}")
        try:
            src = inspect_recording(r, Path(source), map_dir)
        except RecordingError as e:
            raise MapInputError(str(e), e.code) from None
        bad = recording_mismatches(sensors, src.meta or {})
        if bad:
            raise MapInputError(f"{src.path} does not fit rig {rig_id}'s settings: " + "; ".join(bad) +
                                f" (is it a recording of this rig? pass --rig, or fix {sensors.rig_path})",
                                "rig_mismatch")
        if not vocabulary.is_file():
            raise MapInputError(f"ORB vocabulary not found: {vocabulary} (build with umi-processing/native/build.sh)",
                                "slam_build")
        from umiproc.dp.pi_adapter import converted_name

        pi_name = converted_name(src)
        work = r.path / WORK_DIR / "map" / pi_name
        shutil.rmtree(r.path / WORK_DIR / "map", ignore_errors=True)       # a crashed run's leftovers
        work.mkdir(parents=True)
        try:
            try:
                inputs = write_slam_inputs(sensors, work, binary)
            except SlamBuildError as e:
                raise MapInputError(str(e), "slam_build") from None
            try:
                staged = stage(r, src, rig_id)
            except RecordingError as e:
                raise MapInputError(str(e), e.code) from None
            try:
                return _process(r, task, src, staged, sensors, inputs, work, source=Path(source), log=log,
                                runner=runner, binary=binary, vocabulary=vocabulary, timeout_s=timeout_s,
                                first_map=first_map, longest_map=longest_map)
            finally:
                discard(staged)                              # gone already when intake took it
        finally:
            shutil.rmtree(r.path / WORK_DIR / "map", ignore_errors=True)
            _rmdir_quiet(r.path / WORK_DIR)
    finally:
        lock.__exit__(None, None, None)


def _rmdir_quiet(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


def _intake_refusals(r, staged) -> tuple[list[str], dict]:
    """What the copied intake would quarantine this recording for (its validation errors, a duplicate), checked
    before SLAM so a recording intake refuses does not cost two SLAM runs. ``([], {})`` when it would take it."""
    from umiproc.dp.intake import DuplicateIndex
    from umiproc.dp.raw_episode import RawEpisode

    ep = RawEpisode.open(staged.path)
    report = ep.validate(decode_videos=True, check_manifest=True)
    codes = sorted({p.code for p in report.errors})
    details = {}
    try:
        meta = ep.meta
    except Exception:  # noqa: BLE001 - invalid meta is one of the validation errors above
        meta = None
    dup = DuplicateIndex(r).check(staged.pi_name, meta)
    if dup:
        codes.append("duplicate")
        details["duplicate"] = dup
    return codes, details


def _process(r, task, src, staged, sensors, inputs, work: Path, *, source: Path, log, runner, binary, vocabulary,
             timeout_s, first_map: bool = False, longest_map: bool = False) -> MapOutcome:
    from umiproc.dp.files import write_json
    from umiproc.recordings import MAP, RecordingError, take

    started = time.time()
    pi_name = staged.pi_name
    refused, details = _intake_refusals(r, staged)
    if refused:
        log(f"map: the copied intake refuses {pi_name} ({', '.join(refused)}): it is quarantined, no SLAM runs")
        taken = take(r, staged, MAP, task=task, log=log)
        return MapOutcome(False, EXIT_INPUT, pi_name, message=(
            f"{src.path} was refused by intake ({', '.join(refused)}) and moved to "
            f"{taken.quarantine_dir or r.quarantine}; see its reason.json"),
            report={"refused": refused, "details": details})

    try:
        report, results = _run_and_check(r, task, src, staged, sensors, inputs, work, log=log, runner=runner,
                                         binary=binary, vocabulary=vocabulary, timeout_s=timeout_s,
                                         first_map=first_map, longest_map=longest_map)
    except Exception:  # noqa: BLE001 - keep what there is, then fail (exit 1); the recording stays untaken
        tb = traceback.format_exc()
        where = failed_map_dir(r.path, pi_name)
        report = {"format": REPORT_FORMAT, "passed": False, "task": task, "pi_name": pi_name,
                  "source": str(src.path), "error": tb, "created_utc": _utc(time.time())}
        _keep_failed(work, where, report)
        return MapOutcome(False, EXIT_FAILED, pi_name, where=where, report=report,
                          message=f"unexpected error (traceback in {where / REPORT_NAME}):\n{tb}")
    report["duration_s"] = round(time.time() - started, 1)
    checks = results["checks"]
    passed = not any(c.failed for c in checks)          # only blocking checks stop it; the others are warnings
    report["passed"] = passed
    report["warnings"] = [c.id for c in checks if c.warned]
    if results.get("tag_pose") is not None:
        write_json(work / TAG_POSE_NAME, results["tag_pose"])

    if not passed:
        where = failed_map_dir(r.path, pi_name)
        report["kept_in"] = str(where)
        _keep_failed(work, where, report)
        return MapOutcome(False, EXIT_FAILED, pi_name, where=where, checks=checks, report=report,
                          message=f"the map of {src.path} failed: nothing was attached and {src.name} was not "
                                  "taken (it can be run again)")
    # Every blocking check passed. report.json goes into the work folder first (with the folder and slot intake will use), so
    # a pass interrupted after take() leaves it in .work/map/<pi_name>/ with the map (DIRECTIONS.md section 3, "An
    # interrupted map"); then take(), the map files, and report.json last (os.replace: the slot is complete exactly
    # when report.json appears in it).
    from umiproc.dp.intake import IntakeError
    from umiproc.dp.paths import resolve_intake_target
    from umiproc.dp.pi_intake import PiIntakeError

    report.update(folder=resolve_intake_target(r.tasks, task, new=True), slot=f"map/{pi_name}")
    write_json(work / REPORT_NAME, report)
    try:
        taken = take(r, staged, MAP, task=task, log=log)
    except (RecordingError, IntakeError, PiIntakeError) as e:
        where = failed_map_dir(r.path, pi_name)
        report.update(passed=False, folder=None, slot=None, intake_error=f"{type(e).__name__}: {e}")
        _keep_failed(work, where, report, keep_atlas=True)
        return MapOutcome(False, EXIT_INPUT, pi_name, where=where, checks=checks, report=report,
                          message=f"the map passed, but intake did not take {pi_name}: {e} (the map and report "
                                  f"are kept in {where})")
    if taken.status != "ingested":
        where = failed_map_dir(r.path, pi_name)
        report.update(passed=False, folder=None, slot=None, intake_refused=list(taken.reasons))
        _keep_failed(work, where, report, keep_atlas=True)
        return MapOutcome(False, EXIT_INPUT, pi_name, where=where, checks=checks, report=report,
                          message=f"the map passed, but intake quarantined {pi_name} ({', '.join(taken.reasons)})"
                                  f": {taken.quarantine_dir}")
    dest = taken.slot_dir
    report.update(folder=taken.folder, slot=taken.slot)
    write_json(work / REPORT_NAME, report)
    for name in MAP_FILES:
        if name != REPORT_NAME and (work / name).exists():
            os.replace(work / name, dest / name)
    os.replace(work / REPORT_NAME, dest / REPORT_NAME)    # last: a map folder with report.json is complete
    return MapOutcome(True, EXIT_OK, pi_name, folder=taken.folder, task_dir=taken.task_dir, where=dest,
                      checks=checks, report=report,
                      message=f"map OK: tasks/{taken.folder} created (label {task}) with the map in {taken.slot}"
                              + (f" (warnings: {', '.join(report['warnings'])})" if report["warnings"] else ""))


def _keep_failed(work: Path, where: Path, report: dict, keep_atlas: bool = False) -> None:
    """Move the logs, CSVs, inputs and report of a failed map to ``where`` (no EuRoC frames; the atlas only with
    ``keep_atlas``: a map that passed every check but that intake did not take)."""
    from umiproc.dp.files import write_json

    where.mkdir(parents=True, exist_ok=False)
    for name in MAP_FILES:
        if name == REPORT_NAME or (name == ATLAS_NAME and not keep_atlas) or not (work / name).exists():
            continue
        os.replace(work / name, where / name)
    report["kept_in"] = str(where)
    write_json(where / REPORT_NAME, report)


def _utc(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------- SLAM and checks

def _observed_frames(episode, detector):
    """Every ultrawide frame, decoded once: the tag detector sees it, then the copied ``write_euroc`` writes it."""
    from umiproc.dp.slam.base import ultrawide_camera

    for i, rgb in episode.read_video_frames(ultrawide_camera(episode)):
        detector.observe(i, rgb)
        yield i, rgb


def _run_info(res) -> dict:
    """What the report keeps of one umiproc_slam run."""
    s, lg = res.summary, res.log
    out = {"command": list(res.command), "exit_code": res.returncode, "timed_out": res.timed_out,
           "duration_s": round(res.duration_s, 1), "log": res.log_path.name, "summary_error": res.summary_error}
    if s is not None:
        out.update(frames_total=s.frames_total, frames_fed=s.frames_fed, lost_frames=s.lost_frames,
                   stopped_early=s.stopped_early, stop_reason=s.stop_reason,
                   state_counts={str(k): v for k, v in s.state_counts.items()},
                   maps=[vars(m) for m in (s.maps or ())],
                   pacing=None if s.pacing is None else vars(s.pacing))
    out.update(map_resets=[e.detail.get("reason") for e in lg.resets],
               initializations=[e.detail.get("method") for e in lg.of_kind("map_initialized")],
               relocalization_times_s=lg.relocalization_times_s, viba_completed=list(lg.viba_completed),
               relocalization_failures=lg.relocalization_failures)
    return out


def _run_and_check(r, task, src, staged, sensors, inputs, work: Path, *, log, runner, binary, vocabulary,
                   timeout_s, first_map: bool = False, longest_map: bool = False) -> tuple[dict, dict]:
    import cv2

    from umiproc.dp.raw_episode import RawEpisode
    from umiproc.dp.slam.base import SlamError, finalize_trajectory, imu_samples, ultrawide_times_ns
    from umiproc.dp.slam.orbslam3 import write_euroc
    from umiproc.map.tag import TableTagDetector, aggregate_sightings, collect_sightings
    from umiproc.slam import runner as slam_runner
    from umiproc.slam.runner import run_umi_slam
    from umiproc.slam.trajectory_csv import TrajectoryCsvError, read_times

    episode = RawEpisode.open(staged.path)
    rig = sensors.rig                                     # the EFFECTIVE rig everywhere below
    tag = sensors.slam.table_tag
    fp = inputs.fingerprint
    build = {"binary": fp["binary"]["path"], "patches": [p["name"] for p in fp["patches"]],
             "patches_label": patches_label(fp["patches"]), "library": fp["library"]["realpath"],
             "build_info": fp["build_info"]["path"]}
    vio = fp["sensors"]["imu"]["vio_inflation"]
    log(f"map: {staged.pi_name}: SLAM build {build['binary']} (patches {build['patches_label']}); IMU vio_inflation "
        f"noise_density {vio['noise_density']:g}, random_walk {vio['random_walk']:g}; lens "
        f"{sensors.lens_provenance}; fingerprint {inputs.fingerprint_hash[:12]}")

    # 1. EuRoC folder + table-tag detections (one decode)
    detector = TableTagDetector(sensors)
    euroc = work / "euroc"
    log(f"map: converting the ultrawide frames ({rig.cameras['ultrawide'].resolution[0]}x"
        f"{rig.cameras['ultrawide'].resolution[1]}, no scaling) and looking for tag {tag.id}")
    prep = write_euroc(episode, rig, euroc, frames=_observed_frames(episode, detector))
    times = read_times(euroc / "times.txt")
    t_uw = ultrawide_times_ns(episode, rig)
    frame_of_t = {int(t): i for i, t in enumerate(t_uw)}
    frame_of_row = np.array([frame_of_t[int(t)] for t in times], dtype=np.int64)
    (work / TAG_DETECTIONS_NAME).write_text(_dumps(detector.to_json()))
    n_tag = sum(1 for i in frame_of_row if int(i) in detector.detections)
    log(f"map: {len(times)} frames, tag {tag.id} detected in {n_tag} of them")
    slam_kw = dict(mask_img=inputs.mask, realtime=True, binary=binary, vocabulary=vocabulary, timeout_s=timeout_s,
                   runner=runner)

    # 2. mapping run (02_create_map.py)
    atlas = work / ATLAS_NAME
    with_points = slam_runner.supports_save_points(binary)
    if not with_points:
        log("map: this SLAM build has no --save_points (built before it): no map points are stored, so "
            "`umiproc slam-3d` will show the camera path only (rebuild with native/build.sh for the points)")

    def mapping_run(seq: Path, n: int):
        log(f"map: mapping run (frames fed in real time: about {n / 30:.0f} s or more)")
        res = run_umi_slam(seq, inputs.settings, work / MAPPING_DIR, save_map=atlas, init_tag_id=tag.id,
                           init_tag_size=tag.size_m, aruco_dict_id=int(getattr(cv2.aruco, tag.dictionary)),
                           save_points=work / MAPPING_DIR / POINTS_NAME if with_points else None, **slam_kw)
        states = None                                     # per-frame states: how long each dropped map tracked
        if res.summary is not None:
            try:
                states = res.trajectory().state
            except Exception:  # noqa: BLE001 - only refines a hint; the log alone is used then
                states = None
        return res, states, C.init_resets(res.log, states)

    m, m_states, early = mapping_run(euroc, len(times))
    runs: dict = {}
    first = _first_piece_info(m, m_states, times)         # None: one map (or none), nothing to cut
    # Several maps: pick one automatically (the longest piece; --first-map: the first), no flag needed
    mode = "first" if first_map else "longest"
    flag = "--first-map" if first_map else ("--longest-map" if longest_map else "longest map (automatic)")
    key = "piece" if mode == "first" else "longest"
    cut = None
    if first is not None and first.get(key) is not None:
        # keep the whole clip's run, then map and check only the chosen piece's part of the clip
        piece = first[key]
        n_start = 0 if mode == "first" else int(first["longest_start"])
        n_end = piece["last_tracked"] + 1
        n_cut = n_end - n_start
        os.replace(work / MAPPING_DIR, work / MAPPING_FULL_DIR)
        atlas.unlink(missing_ok=True)
        runs["mapping_full"] = _run_info(m)
        runs["mapping_full"]["init_resets"] = early
        t_full_s = (np.asarray(times, dtype=np.int64) - int(times[0])) / 1e9
        cut = {**first, "applied": True, "mode": mode, "piece": piece, "start_frame": n_start, "frames": n_cut,
               "frames_full": int(len(times)), "start_s": round(float(t_full_s[n_start]), 3),
               "end_s": piece["last_tracked_s"],
               "full_run": {"maps_with_keyframes": first["maps_with_keyframes"], "log": f"{MAPPING_FULL_DIR}/"
                            f"{m.log_path.name}"}}
        what = "first" if mode == "first" else "longest"
        log(f"map: {flag}: the whole clip gave {first['maps_with_keyframes']} maps with keyframes; the {what} map "
            f"piece tracked {piece['first_tracked_s']:.2f}-{piece['last_tracked_s']:.2f} s ({piece['tracked']} "
            f"frames), so the clip is cut to fed frames {n_start}-{n_end - 1} ({cut['start_s']:.2f}-{cut['end_s']:.2f}"
            f" s, {n_cut} of {len(times)} frames) and mapped again")
        euroc = _cut_sequence(euroc, work / CUT_EUROC_DIR, n_end, start=n_start)
        times, frame_of_row = times[n_start:n_end], frame_of_row[n_start:n_end]
        n_tag = sum(1 for i in frame_of_row if int(i) in detector.detections)
        log(f"map: tag {tag.id} detected in {n_tag} of the {n_cut} frames kept")
        m, m_states, early = mapping_run(euroc, n_cut)
    elif first is not None or first_map or longest_map:     # several maps but no usable piece, or an explicit flag
        cut = {**(first or {}), "applied": False, "mode": mode,
               "reason": ("the whole clip gave one map (or none): nothing to cut" if first is None else
                          f"no map piece tracked >= {C.FIRST_MAP_MIN_TRACKED} frames")}
        log(f"map: {flag}: not cut ({cut['reason']})")
    checks = [C.check_m1(m.returncode, m.timed_out, m.summary, m.log, m.map_saved, early),
              C.check_m2(m.summary, early)]
    if cut and cut.get("applied") and not checks[0].ok:
        p_ = cut["piece"]
        checks[0].hints.append(
            f"even the {'first' if mode == 'first' else 'longest'} map piece alone ({p_['first_tracked_s']:.2f}-"
            f"{p_['last_tracked_s']:.2f} s of the whole clip) did not give one map: record a new mapping clip"
            + (" (or run it again without that flag: the longest piece is then picked)" if mode == "first" else ""))
    runs["mapping"] = _run_info(m)
    runs["mapping"]["init_resets"] = early
    diag: dict = {}

    # 3. re-localize the mapping clip against the saved map (03_batch_slam.py, map*)
    rel = traj = fin = None
    fin_error = None
    if m.map_saved and m.summary is not None and m.summary.maps_with_keyframes:
        log("map: re-localizing the mapping clip against the saved map")
        rel = run_umi_slam(euroc, inputs.settings, work / RELOCALIZE_DIR, load_map=atlas,
                           max_lost_frames=C.RELOCALIZE_MAX_LOST_FRAMES, **slam_kw)
        runs["relocalize"] = _run_info(rel)
        traj_error = None
        if rel.summary is not None:
            try:
                traj = rel.trajectory()
            except TrajectoryCsvError as e:
                traj_error = str(e)
        runs["relocalize"]["trajectory_error"] = traj_error
        checks.append(C.check_m3(rel.returncode, rel.timed_out, rel.summary, rel.log, traj, times))
        if traj is not None:
            imu_traj = C.imu_trajectory(traj, rig.cameras["ultrawide"].T_imu_cam)
            try:
                fin = finalize_trajectory(imu_traj, episode, rig, t_uw, align=True, align_min_deg=0.0)
            except (SlamError, ValueError) as e:
                fin_error = str(e)
            imu_t, _gyro, acc = imu_samples(episode, rig)
            orient = C.orientation_consistency(imu_traj, imu_t, acc, t_ref_ns=int(times[0]))
            checks.append(C.check_m4(fin, fin_error, C.max_speed(traj), orient))
            stats = collect_sightings(traj, frame_of_row, detector.detections, rig.cameras["ultrawide"].resolution)
            agg = aggregate_sightings(np.stack([s.T_world_tag for s in stats.sightings])) if stats.sightings else None
            up = C.up_direction(fin)
            checks.append(C.check_m5(stats, agg, up, n_tag, len(times)))
            if agg is not None:
                results_tag = {"tx_slam_tag": agg["T_world_tag"].tolist(), **_tag_record(stats, agg, up, tag,
                                                                                         sensors)}
            else:
                results_tag = None
            if fin is not None:
                d = fin.diagnostics
                diag["time_offset_check"] = d.get("time_offset_check")
                diag["motion_check"] = d.get("motion_check")
                diag["gravity_alignment"] = d.get("gravity_alignment")
        else:
            why = "the re-localized trajectory could not be read" + (f" ({traj_error})" if traj_error else "")
            checks += [C.not_evaluated("M4", why), C.not_evaluated("M5", why)]
            results_tag = None
    else:
        why = "the mapping run left no saved map with keyframes (M1)"
        checks += [C.not_evaluated(k, why) for k in ("M3", "M4", "M5")]
        results_tag = None

    # 4. diagnostics (reported, never judged)
    the_map = C.the_map(m.summary)
    warnings = C.pacing_warnings("mapping run", m.summary)
    if rel is not None:
        warnings += C.pacing_warnings("re-localization run", rel.summary)
    toff = diag.get("time_offset_check")
    if toff is not None and toff.get("traj_delay_ms") is not None and not toff.get("ok"):
        warnings.append(f"camera-IMU time offset estimate {toff['traj_delay_ms']:.1f} ms (correlation "
                        f"{toff['corr']:.2f}); umiproc assumes 0: a diagnostic only")
    diag.update(
        keyframes=None if the_map is None else the_map.keyframes,
        lens=sensors.lens_record(), vio_inflation=vio, build=build,
        pacing={k: v.get("pacing") for k, v in runs.items()},
        tag_frames=n_tag, frames=int(len(times)),
        euroc={k: v for k, v in prep.items() if k != "t_cam_ns"},
        warnings=warnings)
    report = {"format": REPORT_FORMAT, "task": task, "pi_name": staged.pi_name, "source": str(src.path),
              "source_manifest_sha256": src.manifest_sha256, "rig_id": sensors.rig_id,
              "created_utc": _utc(time.time()), "fingerprint_hash": inputs.fingerprint_hash,
              "checks": [c.to_json() for c in checks], "diagnostics": diag, "runs": runs,
              "files": {"atlas": ATLAS_NAME, "table_tag": TAG_POSE_NAME, "fingerprint": FINGERPRINT_NAME,
                        "mapping_log": f"{MAPPING_DIR}/{m.log_path.name}",
                        "relocalize_log": f"{RELOCALIZE_DIR}/umiproc_slam.log" if rel is not None else None,
                        "mapping_full_log": cut["full_run"]["log"] if cut and cut.get("applied") else None},
              "first_map": cut if first_map else None, "longest_map": None if first_map else cut}
    (work / FINGERPRINT_NAME).write_text(_dumps({"hash": inputs.fingerprint_hash, "record": fp}))
    shutil.rmtree(work / CUT_EUROC_DIR, ignore_errors=True)   # only a times.txt and a link
    shutil.rmtree(work / "euroc", ignore_errors=True)      # ~400 KB per grey PNG, ~1.4 GB for 2 min; a re-run rebuilds
    return report, {"checks": checks, "tag_pose": results_tag}


def _first_piece_info(m, states, times) -> dict | None:
    """When the mapping run ended with more than one map with keyframes: its map pieces
    (:func:`umiproc.map.checks.map_pieces`) and the first one that tracked (``piece``, None when none did), with
    times in s since the first fed frame. None otherwise (one map or none, a crashed run, no readable states)."""
    if m.summary is None or states is None or len(m.summary.maps_with_keyframes) <= 1:
        return None
    t_s = (np.asarray(times, dtype=np.int64) - int(times[0])) / 1e9
    pieces = C.map_pieces(states)
    piece = C.first_map_piece(states)
    longest = C.longest_map_piece(states)
    return {"maps_with_keyframes": len(m.summary.maps_with_keyframes), "min_tracked": C.FIRST_MAP_MIN_TRACKED,
            "pieces": [p.to_json(t_s) for p in pieces], "piece": None if piece is None else piece.to_json(t_s),
            "longest": None if longest is None else longest[0].to_json(t_s),
            "longest_start": None if longest is None else int(longest[1])}


def _cut_sequence(euroc: Path, out: Path, n: int, start: int = 0) -> Path:
    """A EuRoC folder with only lines ``start`` to ``n - 1`` of ``euroc``'s times.txt (umiproc_slam feeds exactly
    those, its IMU from one sample before the first); frames and IMU are ``euroc``'s (``mav0`` is a link)."""
    out.mkdir()
    lines = [ln for ln in (euroc / "times.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    (out / "times.txt").write_text("\n".join(lines[start:n]) + "\n")
    (out / "mav0").symlink_to((euroc / "mav0").resolve(), target_is_directory=True)
    return out


def _tag_record(stats, agg, up, tag, sensors) -> dict:
    """The rest of ``tx_slam_tag.json`` (UMI reads only ``tx_slam_tag``)."""
    return {"world": "the SLAM map's world: the camera frame of the map's first keyframe (as every trajectory CSV "
                     "localized against this map)",
            "tag": {"dictionary": tag.dictionary, "id": int(tag.id), "size_m": float(tag.size_m)},
            "method": "UMI scripts/calibrate_slam_tag.py --keyframe_only on the mapping clip re-localized against "
                      "the map (umiproc.map.tag)",
            "sightings": len(stats.sightings), "kept_after_outliers": int(np.sum(agg["kept"])),
            "std_m": np.asarray(agg["std_m"]).tolist(), "mean_m": np.asarray(agg["mean"]).tolist(),
            "normal_median_world": np.asarray(agg["normal_median"]).tolist(),
            "frame_index": int(stats.sightings[agg["index"]].frame_index),
            "up_world": None if up is None else np.asarray(up).tolist(),
            "lens": sensors.lens_provenance}


def _dumps(obj) -> str:
    import json

    from umiproc.dp.files import jsonable

    return json.dumps(jsonable(obj), indent=2) + "\n"
