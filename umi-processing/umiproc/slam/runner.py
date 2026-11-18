"""Run ``umiproc_slam`` (native/umi_slam) on one EuRoC folder: build the command, run it in a work folder with a
timeout, and return the exit code, the output paths and the parsed summary/log.

The binary and vocabulary come from native/build.sh: ``${ORB_SLAM3_UMI_DIR:-~/ORB_SLAM3_umi}/umiproc_build/
umiproc_slam`` and ``.../Vocabulary/ORBvoc.txt``. ``$UMIPROC_SLAM_BINARY`` (or the ``binary=`` argument) selects
another build, e.g. the patches-0001-0006 variant ``~/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam`` for A/B runs
(``PATCHES_UPTO=0006 native/build.sh``); the vocabulary stays the main one. The process runs with cwd = the work folder (anything the fork
writes to its current directory lands there), stdout and stderr go to one log file in it, and every path on its
command line is absolute.

``save_points=`` passes ``--save_points PATH`` (the saved map's points, keyframes and meta JSON for ``umiproc slam-3d``);
builds older than that option do not know it: :func:`supports_save_points` asks the binary's ``--help`` once.

Outputs are cleared before the run (trajectory CSV, its ``.t0_ns``, the log, the ``save_map`` file and the
``save_points`` files), so a crash cannot leave an older result looking current; after a run that did not exit 0 (including a timeout or a crash) the
``save_map`` file is removed, as umiproc_slam itself does on its own non-zero exits.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from umiproc.slam.log import SlamLog, SlamLogError, SlamSummary, parse_events, read_log
from umiproc.slam.trajectory_csv import ForkTrajectory, read_fork_csv, read_times

FORK_DIR_ENV = "ORB_SLAM3_UMI_DIR"
BINARY_ENV = "UMIPROC_SLAM_BINARY"           # another umiproc_slam build (A/B variants)
TRAJECTORY_NAME = "trajectory.csv"
LOG_NAME = "umiproc_slam.log"
# default timeout: the run feeds frames in real time, then shuts down (final BA, atlas save); generous on the Jetson
TIMEOUT_PER_SECOND_OF_VIDEO = 3.0
TIMEOUT_EXTRA_S = 300.0

# (command, cwd, log path, timeout in s) -> exit code; raises subprocess.TimeoutExpired on a timeout
Runner = Callable[[Sequence[str], Path, Path, float], int]


class SlamRunError(RuntimeError):
    """The run could not be started (missing binary, vocabulary or input)."""


def default_fork_dir() -> Path:
    return Path(os.environ.get(FORK_DIR_ENV) or Path.home() / "ORB_SLAM3_umi").expanduser()


def default_binary() -> Path:
    if os.environ.get(BINARY_ENV):
        return Path(os.environ[BINARY_ENV]).expanduser()
    return default_fork_dir() / "umiproc_build" / "umiproc_slam"


def default_vocabulary() -> Path:
    return default_fork_dir() / "Vocabulary" / "ORBvoc.txt"


POINTS_SIBLINGS = ("_keyframes.csv", "_meta.json")     # written next to the --save_points CSV
_SAVE_POINTS_SUPPORT: dict = {}


def points_files(save_points: str | Path) -> tuple[Path, ...]:
    """The ``--save_points`` CSV and the two files the front-end writes next to it."""
    p = Path(save_points)
    stem = p.name[:-4] if p.name.endswith(".csv") and len(p.name) > 4 else p.name
    return (p,) + tuple(p.with_name(stem + s) for s in POINTS_SIBLINGS)


def supports_save_points(binary: str | Path, timeout_s: float = 20.0) -> bool:
    """Whether this umiproc_slam build knows ``--save_points`` (its ``--help`` names it). Asked once per binary file
    (path, size, mtime); False when the binary cannot be run."""
    b = Path(binary)
    try:
        st = b.stat()
    except OSError:
        return False
    key = (str(b.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SAVE_POINTS_SUPPORT:
        try:
            r = subprocess.run([str(b), "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout_s, check=False)
            _SAVE_POINTS_SUPPORT[key] = b"--save_points" in r.stdout
        except (OSError, subprocess.SubprocessError):
            _SAVE_POINTS_SUPPORT[key] = False
    return _SAVE_POINTS_SUPPORT[key]


def subprocess_runner(cmd: Sequence[str], cwd: Path, log_path: Path, timeout_s: float) -> int:
    with open(log_path, "wb") as f:
        return subprocess.run(list(cmd), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s,
                              check=False).returncode


@dataclass(frozen=True)
class SlamRunResult:
    returncode: int | None       # None when the run timed out (it was killed)
    timed_out: bool
    command: tuple[str, ...]
    work_dir: Path
    log_path: Path
    trajectory_csv: Path
    t0_path: Path
    load_map: Path | None
    save_map: Path | None
    summary: SlamSummary | None  # None when the run died before printing it (or printed a malformed one)
    log: SlamLog
    duration_s: float
    summary_error: str | None = None   # why a summary line that is there could not be read

    @property
    def ok(self) -> bool:
        """Exit 0 with a summary that agrees (3 = stopped at --max_lost_frames is not ok)."""
        return self.returncode == 0 and self.summary is not None and self.summary.exit_code == 0

    @property
    def map_saved(self) -> bool:
        return self.save_map is not None and self.save_map.is_file()

    def trajectory(self, times: str | Path | None = None) -> ForkTrajectory:
        """The trajectory CSV read against the fed frames (default: the sequence's times.txt from the command)."""
        if self.summary is None:
            raise SlamRunError(f"no summary in {self.log_path}: the run did not finish")
        if times is None:
            times = Path(self.command[self.command.index("--sequence") + 1]) / "times.txt"
        return read_fork_csv(self.trajectory_csv, times, frames_fed=self.summary.frames_fed)


def build_command(binary: Path, vocabulary: Path, settings: Path, sequence_dir: Path, trajectory_csv: Path, *,
                  load_map: Path | None = None, save_map: Path | None = None, mask_img: Path | None = None,
                  max_lost_frames: int | None = None, init_tag_id: int | None = None,
                  init_tag_size: float | None = None, aruco_dict_id: int | None = None,
                  realtime: bool = True, save_points: Path | None = None) -> list[str]:
    cmd = [str(binary), "--vocabulary", str(vocabulary), "--setting", str(settings), "--sequence", str(sequence_dir),
           "--output_trajectory_csv", str(trajectory_csv)]
    if load_map is not None:
        cmd += ["--load_map", str(load_map)]
    if save_map is not None:
        cmd += ["--save_map", str(save_map)]
    if mask_img is not None:
        cmd += ["--mask_img", str(mask_img)]
    if max_lost_frames is not None:
        cmd += ["--max_lost_frames", str(int(max_lost_frames))]
    if aruco_dict_id is not None:
        cmd += ["--aruco_dict_id", str(int(aruco_dict_id))]
    if init_tag_id is not None:
        cmd += ["--init_tag_id", str(int(init_tag_id))]
    if init_tag_size is not None:
        cmd += ["--init_tag_size", repr(float(init_tag_size))]
    if save_points is not None:
        cmd += ["--save_points", str(save_points)]
    if not realtime:
        cmd.append("--no_realtime")
    return cmd


def _require_file(path: Path, what: str, hint: str = "") -> Path:
    if not path.is_file():
        raise SlamRunError(f"{what} not found: {path}{hint}")
    return path


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_umi_slam(sequence_dir: str | Path, settings: str | Path, work_dir: str | Path, *,
                 load_map: str | Path | None = None, save_map: str | Path | None = None,
                 mask_img: str | Path | None = None, max_lost_frames: int | None = None,
                 init_tag_id: int | None = None, init_tag_size: float | None = None,
                 aruco_dict_id: int | None = None, realtime: bool = True, trajectory_name: str = TRAJECTORY_NAME,
                 log_name: str = LOG_NAME, binary: str | Path | None = None, vocabulary: str | Path | None = None,
                 timeout_s: float | None = None, runner: Runner | None = None,
                 save_points: str | Path | None = None) -> SlamRunResult:
    """Run umiproc_slam once. Mapping: ``save_map=``, ``init_tag_id=13``; localization: ``load_map=``.
    ``save_points``: ``--save_points`` (only for builds that have it: :func:`supports_save_points`).

    ``timeout_s`` default: 3 s per second of video + 300 s. ``runner`` replaces the subprocess call (tests).
    """
    hint = " (build it with umi-processing/native/build.sh)"
    binary = Path(binary).expanduser().resolve() if binary is not None else default_binary()
    vocabulary = Path(vocabulary).expanduser().resolve() if vocabulary is not None else default_vocabulary()
    _require_file(binary, "umiproc_slam binary", hint)
    if not os.access(binary, os.X_OK):
        raise SlamRunError(f"umiproc_slam binary is not executable: {binary}")
    _require_file(vocabulary, "ORB vocabulary", hint)
    sequence_dir = Path(sequence_dir).expanduser().resolve()
    times_path = _require_file(sequence_dir / "times.txt", "times.txt")
    _require_file(sequence_dir / "mav0" / "imu0" / "data.csv", "IMU file")
    if not (sequence_dir / "mav0" / "cam0" / "data").is_dir():
        raise SlamRunError(f"frame folder not found: {sequence_dir / 'mav0' / 'cam0' / 'data'}")
    settings = _require_file(Path(settings).expanduser().resolve(), "ORB-SLAM3 settings")
    mask = _require_file(Path(mask_img).expanduser().resolve(), "mask image") if mask_img is not None else None
    load = _require_file(Path(load_map).expanduser().resolve(), "map to load") if load_map is not None else None
    save = Path(save_map).expanduser().resolve() if save_map is not None else None
    if save is not None and load is not None and save == load:
        raise SlamRunError(f"save_map and load_map are the same file: {save}")
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if save is not None and not save.parent.is_dir():
        raise SlamRunError(f"folder for the map does not exist: {save.parent}")
    points = Path(save_points).expanduser().resolve() if save_points is not None else None
    if points is not None and not points.parent.is_dir():
        raise SlamRunError(f"folder for the map points does not exist: {points.parent}")

    trajectory_csv = work_dir / trajectory_name
    t0_path = Path(str(trajectory_csv) + ".t0_ns")
    log_path = work_dir / log_name
    for p in (trajectory_csv, t0_path, log_path) + ((save,) if save is not None else ()) + \
            (points_files(points) if points is not None else ()):
        _unlink(p)

    if timeout_s is None:
        t = read_times(times_path)
        video_s = (int(t[-1]) - int(t[0])) / 1e9 if len(t) else 0.0
        timeout_s = TIMEOUT_PER_SECOND_OF_VIDEO * video_s + TIMEOUT_EXTRA_S

    cmd = build_command(binary, vocabulary, settings, sequence_dir, trajectory_csv, load_map=load, save_map=save,
                        mask_img=mask, max_lost_frames=max_lost_frames, init_tag_id=init_tag_id,
                        init_tag_size=init_tag_size, aruco_dict_id=aruco_dict_id, realtime=realtime,
                        save_points=points)
    run = runner or subprocess_runner
    start = time.monotonic()
    timed_out = False
    try:
        rc: int | None = int(run(cmd, work_dir, log_path, float(timeout_s)))
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True
    duration = time.monotonic() - start
    if rc != 0 and save is not None:
        _unlink(save)                     # never leave a map from a run that did not finish cleanly
    summary_error = None
    try:
        log = read_log(log_path) if log_path.is_file() else _empty_log()
    except SlamLogError as e:            # e.g. a crash in the middle of the summary line: keep the events
        events, n_failed = parse_events(log_path.read_text(errors="replace"))
        log = SlamLog(summary=None, events=events, relocalization_failures=n_failed)
        summary_error = str(e)
    return SlamRunResult(returncode=rc, timed_out=timed_out, command=tuple(cmd), work_dir=work_dir,
                         log_path=log_path, trajectory_csv=trajectory_csv, t0_path=t0_path, load_map=load,
                         save_map=save, summary=log.summary, log=log, duration_s=duration,
                         summary_error=summary_error)


def _empty_log() -> SlamLog:
    """The log of a run that wrote no log file at all."""
    return SlamLog(summary=None, events=(), relocalization_failures=0)
