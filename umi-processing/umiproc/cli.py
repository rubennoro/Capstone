"""umiproc: UMI's map -> calibrate -> run SLAM recipe for the Pi 5 rig's recordings (DIRECTIONS.md, docs/DESIGN.md).

Each command lives in its own module with ``add_arguments(parser)`` and ``run(args) -> int`` (the exit code); a module
that does not exist yet makes its command print "not implemented yet".
"""

from __future__ import annotations

import argparse
import importlib
import sys

# (command, module, one-line help): the optional lens calibration, then a session in UMI's order (docs/DESIGN.md
# "The session"), then status
COMMANDS = [
    ("lens-board", "umiproc.lens.board", "write the printable ChArUco board PDF for the lens calibration"),
    ("lens-calibrate", "umiproc.lens.calibrate", "fit the IMX219 fisheye lens model from a board recording"),
    ("map", "umiproc.map.cli", "build and check the session map from a mapping recording (creates the task folder)"),
    ("calibrate", "umiproc.calibrate.cli", "fit the gripper range from a gripper calibration recording"),
    ("run", "umiproc.run.cli", "localize and check a range of demo recordings against the folder's map"),
    ("status", "umiproc.status", "show the data root (rig files, SLAM program, recordings, task folders) or one "
                                 "task folder (map, gripper calibration, episodes, waiting, failed)"),
    ("plot-run", "umiproc.viz.slam_run", "plot one SLAM run (trajectory CSV + log): tracking timeline, path, frames"),
    ("slam-video", "umiproc.viz.slam_video", "a video of one SLAM run: the fed frames with their tracking state, the "
                                             "path from above, the timeline"),
    ("slam-3d", "umiproc.viz.slam_3d", "a 3-D view (HTML) of one SLAM run: the map's points, its keyframes and the "
                                       "camera path"),
]


def _load(module: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name and module.startswith(e.name):      # the command's own module is missing, not a dependency
            return None
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umiproc", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module, help_text in COMMANDS:
        p = sub.add_parser(name, help=help_text, description=help_text)
        mod = _load(module)
        if mod is None:
            p.set_defaults(_run=lambda args, _n=name: _not_implemented(_n))
        else:
            mod.add_arguments(p)
            p.set_defaults(_run=mod.run)
    return parser


def _not_implemented(name: str) -> int:
    print(f"umiproc {name}: not implemented yet", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args._run(args) or 0)
