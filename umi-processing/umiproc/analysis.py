"""The analysis folder: a copy of what to look at after ``map`` and ``lens-calibrate``, and where ``slam-video`` writes
by default. ``~/data/analysis`` unless ``$UMIPROC_ANALYSIS`` says otherwise (the laptop: ``~/Downloads/data/analysis``).

    <analysis>/mapping/<episode_dir>/<UTC>_<passed|failed>[_first_map]/
        report.json, tx_slam_tag.json, map_output.txt (what map printed),
        mapping/ relocalize/ mapping_full/ (trajectory CSV, log, slam_run.png; slam_run.mp4 from slam-video)
    <analysis>/lens_calibration/<episode_dir>/   report.json, coverage.png, lens.yaml

Only copies: the data root keeps its own files, and a failed copy is printed, never fails the command.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["ENV_ANALYSIS", "DEFAULT_ANALYSIS", "analysis_root", "map_analysis_dir", "export_map", "export_lens"]

ENV_ANALYSIS = "UMIPROC_ANALYSIS"
DEFAULT_ANALYSIS = Path.home() / "data" / "analysis"
RUN_DIRS = ("mapping_full", "mapping", "relocalize")
MAP_FILES = ("report.json", "tx_slam_tag.json")


def analysis_root() -> Path:
    env = os.environ.get(ENV_ANALYSIS)
    return Path(env).expanduser() if env else DEFAULT_ANALYSIS


def map_analysis_dir(report: dict, root: Path | None = None) -> Path:
    """The folder of one map attempt: episode from the report's ``source``, time from its ``created_utc``."""
    episode = Path(str(report.get("source") or "unknown")).name
    stamp = str(report.get("created_utc") or "unknown")[:19].replace(":", "-") + "Z"
    label = "passed" if report.get("passed") else "failed"
    if (report.get("first_map") or {}).get("applied"):
        label += "_first_map"
    return (root or analysis_root()) / "mapping" / episode / f"{stamp}_{label}"


def export_map(where: Path, report: dict, output_text: str, root: Path | None = None) -> Path:
    """Copy a map attempt's report, tag pose and run folders from ``where`` (its folder in the data root) and write
    ``map_output.txt``. Returns the folder."""
    dest = map_analysis_dir(report, root)
    dest.mkdir(parents=True, exist_ok=True)
    for name in MAP_FILES:
        if (where / name).is_file():
            shutil.copy2(where / name, dest / name)
    for name in RUN_DIRS:
        if (where / name).is_dir():
            shutil.copytree(where / name, dest / name, dirs_exist_ok=True)
    (dest / "map_output.txt").write_text(output_text.rstrip("\n") + "\n")
    return dest


def export_lens(out_dir: Path, episode_dir: str, root: Path | None = None) -> Path:
    """Copy one lens calibration's results (``<data root>/lens/<pi_name>/``) to ``lens_calibration/<episode_dir>/``."""
    dest = (root or analysis_root()) / "lens_calibration" / Path(episode_dir).name
    dest.mkdir(parents=True, exist_ok=True)
    for p in Path(out_dir).iterdir():
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    return dest
