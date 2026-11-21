"""``python -m policy_deploy.frames.trial_video [FOLDER ...] [--fps N] [--force]``: trial recordings -> videos.

Turns the image sets that ``probe --live --save`` wrote (``NNNNNN_{main,ultrawide}_{0,1}.png``) into one video per
folder, ``<folder>/<folder name>.mp4``. Each frame is a 2x2 grid like the live window: main on top, ultrawide below,
the two observation steps as columns, with the set number and its time below. The PNGs are kept.

With no FOLDER it does every ``~/Capstone/data/*/policy/*/deploy/*_probe_live`` folder that has no video yet
(``--force`` rebuilds them). The video is MPEG-4 (``mp4v``): VLC and most players open it, a browser may not.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

DEFAULT_GLOB = "Capstone/data/*/policy/*/deploy/*_probe_live"
SCALE = 2                       # the 224x224 images shown at 448x448, as in probe --live
_SET = re.compile(r"(\d{6})_(main|ultrawide)_([01])\.png")


def _sets(folder: Path) -> list[tuple[int, dict[tuple[str, int], Path]]]:
    """The folder's complete image sets (all four images), in order."""
    sets: dict[int, dict[tuple[str, int], Path]] = defaultdict(dict)
    for f in folder.iterdir():
        m = _SET.fullmatch(f.name)
        if m:
            sets[int(m[1])][(m[2], int(m[3]))] = f
    return [(k, v) for k, v in sorted(sets.items()) if len(v) == 4]


def _frame(n: int, images: dict[tuple[str, int], Path], fps: float) -> np.ndarray:
    rows = []
    for cam in ("main", "ultrawide"):
        tiles = []
        for i in (0, 1):
            tile = cv2.resize(cv2.imread(str(images[(cam, i)])), None, fx=SCALE, fy=SCALE,
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f"{cam}_{i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    frame = np.vstack(rows)
    cv2.putText(frame, f"set {n}  t {n / fps:5.1f} s", (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    return frame


def make_video(folder: Path, fps: float) -> tuple[Path, int]:
    """Write <folder>/<folder name>.mp4 from the folder's image sets; returns the path and the frame count."""
    out = folder / f"{folder.name}.mp4"
    writer = None
    count = 0
    try:
        for n, images in _sets(folder):
            frame = _frame(n, images, fps)
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot write {out}")
            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return out, count


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m policy_deploy.frames.trial_video", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", type=Path, nargs="*", help=f"trial folders (default: ~/{DEFAULT_GLOB} without a video)")
    p.add_argument("--fps", type=float, default=10.0, help="frames per second; 10 = the rate probe --live 10 saved at")
    p.add_argument("--force", action="store_true", help="rebuild videos that already exist")
    args = p.parse_args(argv)
    if not args.fps > 0:
        print(f"trial_video: --fps must be > 0, got {args.fps}", file=sys.stderr)
        return 1
    folders = args.folders or sorted(Path.home().glob(DEFAULT_GLOB))
    status = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"trial_video: {folder}: not a folder", file=sys.stderr)
            status = 1
            continue
        if not args.force and (folder / f"{folder.name}.mp4").exists():
            if args.folders:
                print(f"{folder}: video exists (--force to rebuild)")
            continue
        out, count = make_video(folder, args.fps)
        if count == 0:
            print(f"trial_video: {folder}: no image sets", file=sys.stderr)
            status = 1
            continue
        print(f"{out}: {count} frames, {count / args.fps:.1f} s")
    return status


if __name__ == "__main__":
    sys.exit(main())
