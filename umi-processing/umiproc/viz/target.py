"""What ``umiproc slam-video`` and ``umiproc slam-3d`` look at, and where they write.

The argument is a **task folder** (its name under ``<root>/tasks/``, as ``umiproc run``/``status`` take it): the
folder's map (``map/<pi_name>/``), its run ``--run`` (``mapping/``, ``relocalize/``, ``mapping_full/``), output into
``<root>/tasks/<folder>/analysis/``. Or an **existing folder**: a run folder (with ``trajectory.csv``, e.g. a failed
map's ``maps_failed/<pi_name>/<UTC>/mapping``; output into ``<RUN_DIR>/../analysis/``), or a map folder (with
``report.json``: its run ``--run``; output into ``<MAP_DIR>/analysis/``).

The recording a run was made from (for the frames and the exact frame times): the map's ``raw/`` copy (converted
format: ``meta.json`` ``cameras.<cam>.role``, ``<cam>.mp4`` + ``<cam>_frames.csv``), else the Pi folder named by the
report's ``source`` while it is still there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from umiproc.viz.slam_run import PlotRunError

__all__ = ["ANALYSIS_DIR", "RUNS", "Target", "resolve", "resolve_runs", "report_of", "find_recording",
           "ultrawide_video"]

ANALYSIS_DIR = "analysis"
RUNS = ("mapping", "relocalize", "mapping_full")


@dataclass(frozen=True)
class Target:
    run_dir: Path        # the run folder (trajectory.csv, umiproc_slam.log)
    map_dir: Path        # the folder holding the map (report.json, raw/, slam_mask.png), normally run_dir.parent
    analysis: Path       # where to write
    folder: str | None   # the task folder's name, when given by name

    @property
    def run(self) -> str:
        return self.run_dir.name


def _map_dir_of_folder(name: str, root) -> tuple[Path, Path]:
    from umiproc.dp.paths import DataRoot
    from umiproc.paths import data_root
    from umiproc.recordings import map_slots

    try:
        r = DataRoot(data_root(root))
    except ValueError as e:
        raise PlotRunError(str(e)) from None
    task_dir = r.tasks / name
    if "/" in name or name.startswith(".") or not task_dir.is_dir():
        have = sorted(p.name for p in r.tasks.iterdir() if p.is_dir() and not p.name.startswith(".")) \
            if r.tasks.is_dir() else []
        raise PlotRunError(f"no task folder {name!r} in {r.tasks} (folders: {', '.join(have) or 'none'}; or pass a "
                           "run folder, e.g. maps_failed/<pi_name>/<UTC>/mapping)")
    slots = map_slots(task_dir)
    if not slots:
        raise PlotRunError(f"{task_dir} has no map (map/<pi_name>/)")
    return task_dir, slots[-1]


def resolve_runs(arg: str, run: str, *, root=None) -> list[Target]:
    """The targets for ``arg`` (module docstring); ``run`` is one of :data:`RUNS` or ``all`` (the runs there)."""
    p = Path(arg).expanduser()
    names = RUNS if run == "all" else (run,)
    if p.is_dir() and (p / "trajectory.csv").is_file():
        return [Target(run_dir=p, map_dir=p.parent, analysis=p.parent / ANALYSIS_DIR, folder=None)]
    if p.is_dir() and (p / "report.json").is_file():
        map_dir, analysis, folder = p, p / ANALYSIS_DIR, None
    else:
        task_dir, map_dir = _map_dir_of_folder(arg, root)
        analysis, folder = task_dir / ANALYSIS_DIR, task_dir.name
    out = [Target(run_dir=map_dir / n, map_dir=map_dir, analysis=analysis, folder=folder) for n in names
           if (map_dir / n / "trajectory.csv").is_file()]
    if not out:
        have = [n for n in RUNS if (map_dir / n / "trajectory.csv").is_file()]
        raise PlotRunError(f"{map_dir} has no {run} run (runs there: {', '.join(have) or 'none'})")
    return out


def resolve(arg: str, run: str, *, root=None) -> Target:
    return resolve_runs(arg, run, root=root)[0]


def report_of(t: Target) -> dict:
    try:
        rep = json.loads((t.map_dir / "report.json").read_text())
    except (OSError, ValueError):
        return {}
    return rep if isinstance(rep, dict) else {}


def find_recording(t: Target, recording: str | Path | None = None) -> Path:
    """``recording``, else the map's ``raw/``, else the report's ``source`` (module docstring)."""
    if recording is not None:
        return Path(recording).expanduser()
    if (t.map_dir / "raw" / "meta.json").is_file():
        return t.map_dir / "raw"
    src = report_of(t).get("source")
    if not src:
        raise PlotRunError(f"no raw/ and no report.json with a source next to {t.run_dir}: pass --recording (the "
                           "recording folder)")
    if not Path(src).is_dir():
        raise PlotRunError(f"the map's recording {src} is not there any more: pass --recording")
    return Path(src)


def ultrawide_video(recording: Path) -> tuple[Path, Path]:
    """``(mp4, frames csv)`` of the IMX219 (the ultrawide) in a recording: the converted format (``raw/``) or the
    Pi's, from its ``meta.json``."""
    try:
        meta = json.loads((recording / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise PlotRunError(f"cannot find the ultrawide camera in {recording / 'meta.json'}: {e}") from e
    cams = meta.get("cameras") if isinstance(meta, dict) else None
    if isinstance(cams, dict):                                 # converted: {"cam1": {"role": "ultrawide", ...}}
        cam = next((k for k, v in cams.items() if isinstance(v, dict) and v.get("role") == "ultrawide"), None)
        if cam is None:
            raise PlotRunError(f"no ultrawide camera in {recording / 'meta.json'}")
        video = recording / f"{cam}.mp4"
    else:
        from umiproc.dp.pi_adapter import PiEpisodeError, _camera_roles

        try:
            video = recording / _camera_roles(meta)["ultrawide"]["file"]
        except (KeyError, TypeError, PiEpisodeError) as e:
            raise PlotRunError(f"cannot find the ultrawide camera in {recording / 'meta.json'}: {e}") from e
    frames = recording / f"{video.stem}_frames.csv"
    for p in (video, frames):
        if not p.is_file():
            raise PlotRunError(f"{p} is missing")
    return video, frames


def run_times(t: Target, n: int, t0: int, recording: str | Path | None = None):
    """The run's fed frame times from the recording's frames CSV, or None when there is no recording to read."""
    from umiproc.viz.slam_run import times_from_frames_csv

    try:
        _video, frames = ultrawide_video(find_recording(t, recording))
        return times_from_frames_csv(frames, t0, n)[0]
    except PlotRunError:
        return None
