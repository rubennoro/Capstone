"""``dataproc`` command line: ``calibrate`` -> ``run`` -> ``package``, plus ``status`` and ``sim`` (docs/DESIGN.md).

The data root is ``$DATAPROC_ROOT`` (default ``~/data``). This module only parses the arguments and calls each
command's ``cli_*`` function (``dataproc/calibrate``, ``dataproc/run``, ``dataproc/package``, ``dataproc/status.py``,
``dataproc/sim``), which prints and returns the exit code.
"""

from __future__ import annotations

import argparse
import sys

from dataproc import __version__
from dataproc.paths import DataRoot, data_root

__all__ = ["build_parser", "main"]

_SLAM_CHOICES = ("auto", "orbslam3", "oracle", "none")      # = dataproc.slam.BACKEND_CHOICES (not imported here)
_SOURCE_HELP = "where the Pi delivers its recordings; read only (default /data/incoming)"
_FOLDER_HELP = "task folder under $DATAPROC_ROOT/tasks, as printed by calibrate (e.g. block_raise or block_raise_2)"


def _non_negative_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {s!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _positive_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {s!r}") from None
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {s}")
    return v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dataproc", description="Pi recordings -> training data: calibrate -> run -> "
                                 "package, plus status and sim (docs/DESIGN.md).")
    ap.add_argument("--version", action="version", version=f"dataproc {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{calibrate,run,package,status,sim}")

    p = sub.add_parser("calibrate", help="fit the gripper calibration on one Pi recording; on success create a new "
                       "task folder (TASK, else TASK_2, TASK_3, ...)",
                       description="Fit and check the gripper calibration on the Pi recording EPISODE_DIR. On failure "
                       "print why and create nothing; on success create the first free task folder TASK, TASK_2, "
                       "TASK_3, ... (one folder = one calibration = one session).")
    p.add_argument("task", metavar="TASK", help="task name: the label written to the zarr (e.g. block_raise)")
    p.add_argument("episode_dir", metavar="EPISODE_DIR",
                   help="the calibration recording: a directory name in --source (e.g. episode_000001)")
    p.add_argument("--source", default="/data/incoming", metavar="DIR", help=_SOURCE_HELP)
    p.add_argument("--rig", default="rig_pi_v2", metavar="RIG_ID",
                   help="rig id; selects rigs/<id>.yaml and is recorded in task.yaml (default rig_pi_v2: main "
                   "1600x1200 @ 30; rig_pi_v1 = the old zoomed-in main 1600x900 @ 60)")

    p = sub.add_parser("run", help="process the Pi recordings START_DIR..END_DIR into episode_N of a calibrated "
                       "task folder (failures go to failed/)",
                       description="Process every recording in --source from START_DIR to END_DIR (inclusive, "
                       "natural sort), one at a time. Only recordings that pass every stage become episode_N; the "
                       "others go to failed/<name>/ with reason.json. Exit 0 = done (also with failures), 3 = "
                       "waiting for SLAM, 1 = error.")
    p.add_argument("folder", metavar="TASK_FOLDER", help=_FOLDER_HELP)
    p.add_argument("start_dir", metavar="START_DIR", help="first recording: a directory name in --source")
    p.add_argument("end_dir", metavar="END_DIR", help="last recording (inclusive): a directory name in --source")
    p.add_argument("--source", default="/data/incoming", metavar="DIR", help=_SOURCE_HELP)
    p.add_argument("--slam", choices=_SLAM_CHOICES, default="auto",
                   help="SLAM backend (default auto = orbslam3 if built, else none: stop at slam and wait, exit 3; "
                   "oracle = ground truth of dataproc sim recordings)")
    p.add_argument("--retry-failed", action="store_true", help="also re-run recordings of the range in failed/")
    p.add_argument("--force", action="store_true", help="re-run every stage for the selected range")
    p.add_argument("--dry-run", action="store_true", help="only list what would happen, change nothing")

    p = sub.add_parser("package", help="combine the finished episodes of task folders into one training .zarr.zip",
                       description="Combine every exported episode_N of the FOLDERs into "
                       "$DATAPROC_ROOT/datasets/<label>/DATASET/DATASET.zarr.zip (+ package.json); several labels "
                       "go to datasets/multi/. failed/ and .staging/ are never included.")
    p.add_argument("dataset", metavar="DATASET", help="dataset name (e.g. block_raise-1110)")
    p.add_argument("folders", metavar="FOLDER", nargs="+", help="task folders to include")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset of that name")

    p = sub.add_parser("status", help="show the episodes, failed/ and .staging/ recordings and calibration of a "
                       "task folder (read only)")
    p.add_argument("folder", metavar="TASK_FOLDER", help=_FOLDER_HELP)

    p = sub.add_parser("sim", help="write synthetic recordings (with ground truth) for testing; Pi format by default",
                       description="Write synthetic recordings with ground truth (truth.npz, for --slam oracle). "
                       "Default: the Pi's format as episode_NNNNNN, numbered after the largest number already in "
                       "--out (so calibration and demonstration runs can share one --out). --contract: dataproc's "
                       "own raw format as <session>_episode_NNNNNN.")
    p.add_argument("--out", required=True, help="output folder (a scratch --source; never the real /data/incoming)")
    p.add_argument("--episodes", type=_non_negative_int, default=1, metavar="N", help="number of recordings")
    p.add_argument("--calibration", action="store_true", help="write gripper-calibration recordings")
    p.add_argument("--seed", type=_non_negative_int, default=0, metavar="S",
                   help="base seed (recording n uses seed S*1000003 + n; default 0)")
    p.add_argument("--duration", type=_positive_float, default=None, metavar="SEC",
                   help="recording length incl. pre-roll (2.5 s, --fast: 2 s) and 0.5 s post-roll (default 15, --fast: 5)")
    p.add_argument("--rig", default=None, metavar="PATH", help="rig yaml (default configs/rigs/rig_v1.yaml)")
    p.add_argument("--fast", action="store_true",
                   help="low-resolution rig (<rig_id>_fast: main x0.25, ultrawide x0.5) and short recordings")
    p.add_argument("--imu-gap", type=_positive_float, default=None, metavar="SEC",
                   help="drop SEC of IMU samples mid-recording (an IMU dropout: run rejects it at validate, imu_gap)")
    p.add_argument("--contract", action="store_true",
                   help="write dataproc's own raw format (docs/CONTRACT.md) instead of the Pi's")
    p.add_argument("--session", default=None, metavar="ID",
                   help="session id (default sim-<UTC time, microseconds>); --contract directory names are "
                   "<session>_episode_NNNNNN (Pi format: only recorded in truth.npz)")
    p.add_argument("--rigs-dir", default=None, metavar="DIR",
                   help="where the rig yaml used is copied if missing (default $DATAPROC_ROOT/rigs)")
    p.add_argument("--unverified", action="store_true", help="with --contract: do not write .verified")
    for parser in sub.choices.values():
        parser.set_defaults(_parser=parser)        # usage errors print the subcommand's usage line
    return ap


def cmd_calibrate(args, root: DataRoot) -> int:
    from dataproc.calibrate.cli import cli_calibrate

    return cli_calibrate(root, args.task, args.episode_dir, source=args.source, rig_id=args.rig)


def cmd_run(args, root: DataRoot) -> int:
    from dataproc.run.cli import cli_run

    return cli_run(root, args.folder, args.start_dir, args.end_dir, source=args.source, slam=args.slam,
                   retry_failed=args.retry_failed, force=args.force, dry_run=args.dry_run)


def cmd_package(args, root: DataRoot) -> int:
    from dataproc.package.cli import cli_package

    return cli_package(root, args.dataset, args.folders, overwrite=args.overwrite)


def cmd_status(args, root: DataRoot) -> int:
    from dataproc.status import cli_status

    return cli_status(root, args.folder)


def cmd_sim(args, root: DataRoot) -> int:
    from dataproc.sim.episode import run_sim

    if args.unverified and not args.contract:
        args._parser.error("--unverified needs --contract (the Pi format has no .verified)")
    try:
        run_sim(out=args.out, episodes=args.episodes, calibration=args.calibration, seed=args.seed,
                duration_s=args.duration, rig_path=args.rig, fast=args.fast, session_id=args.session,
                rigs_dir=args.rigs_dir if args.rigs_dir is not None else root.rigs, verified=not args.unverified,
                pi_format=not args.contract, imu_gap_s=args.imu_gap)
    except (ValueError, OSError) as e:
        print(f"dataproc sim: {e}", file=sys.stderr)
        return 1
    return 0


COMMANDS = {"calibrate": cmd_calibrate, "run": cmd_run, "package": cmd_package, "status": cmd_status,
            "sim": cmd_sim}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args, data_root())
