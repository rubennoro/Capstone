"""The analysis folder for ``lens-calibrate``: a copy of its results. ``~/data/analysis`` unless
``$UMIPROC_ANALYSIS`` says otherwise (the laptop: ``~/Downloads/data/analysis``).

    <analysis>/lens_calibration/<episode_dir>/   report.json, coverage.png, lens.yaml

Only copies: the data root keeps its own files, and a failed copy is printed, never fails the command. (A map's
figures, video and 3-D page go into its task folder's ``analysis/``: ``umiproc slam-video`` / ``umiproc slam-3d``.)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["ENV_ANALYSIS", "DEFAULT_ANALYSIS", "analysis_root", "export_lens"]

ENV_ANALYSIS = "UMIPROC_ANALYSIS"
DEFAULT_ANALYSIS = Path.home() / "data" / "analysis"


def analysis_root() -> Path:
    env = os.environ.get(ENV_ANALYSIS)
    return Path(env).expanduser() if env else DEFAULT_ANALYSIS


def export_lens(out_dir: Path, episode_dir: str, root: Path | None = None) -> Path:
    """Copy one lens calibration's results (``<data root>/lens/<pi_name>/``) to ``lens_calibration/<episode_dir>/``."""
    dest = (root or analysis_root()) / "lens_calibration" / Path(episode_dir).name
    dest.mkdir(parents=True, exist_ok=True)
    for p in Path(out_dir).iterdir():
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    return dest
