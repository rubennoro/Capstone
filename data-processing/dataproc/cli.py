"""``dataproc`` command line: ``intake``, ``process``, ``status`` and ``sim`` (behaviour: docs/DESIGN.md).

The data root is ``$DATAPROC_ROOT`` (default ``/data/rig``). ``--task`` is required for intake, process and
status; it is a task folder name, ``name/episode_N``, or a path inside ``<root>/tasks/``.
"""

from __future__ import annotations

import argparse
import sys

from dataproc import __version__
from dataproc.paths import DataRoot, TaskRef, data_root, resolve_task

__all__ = ["build_parser", "main"]

_TASK_HELP = "task folder name (e.g. block_raise), name/episode_N, or a path inside $DATAPROC_ROOT/tasks (required)"


def _non_negative_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {s!r}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dataproc", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"dataproc {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{intake,process,status,sim}")

    p = sub.add_parser("intake", help="move verified episodes from incoming/ into tasks/<folder>/episode_N/raw/")
    p.add_argument("--task", required=True, help=_TASK_HELP)
    p.add_argument("--new", action="store_true", help="if the task folder exists, start <task>_1, <task>_2, ...")
    p.add_argument("--calibration", action="store_true",
                   help="send the next waiting episode to calibration/<name>/raw/ instead of episode_N")

    p = sub.add_parser("process", help="process episodes and rebuild output/<folder>.zarr.zip")
    p.add_argument("--task", required=True, help=_TASK_HELP)
    p.add_argument("--episode", type=_non_negative_int, metavar="N", help="only process episode_N")
    p.add_argument("--force", action="store_true", help="rerun stages even if their inputs are unchanged")
    p.add_argument("--slam", choices=("orbslam3", "oracle"), default="orbslam3",
                   help="SLAM backend (default orbslam3; oracle = synthetic ground truth, needs raw/truth.npz)")

    p = sub.add_parser("status", help="show per-episode processing status of a task")
    p.add_argument("--task", required=True, help=_TASK_HELP)

    p = sub.add_parser("sim", help="write synthetic episodes (with ground truth) for testing")
    p.add_argument("--out", required=True, help="output folder, usually $DATAPROC_ROOT/incoming")
    p.add_argument("--episodes", type=_non_negative_int, default=1, metavar="N", help="number of episodes")
    p.add_argument("--calibration", action="store_true", help="write gripper-calibration episodes")
    p.add_argument("--seed", type=_non_negative_int, default=0, metavar="S",
                   help="base seed (episode n uses seed S*1000003 + n; default 0)")
    p.add_argument("--duration", type=float, default=None, metavar="SEC",
                   help="recording length incl. 2.5 s pre-roll and 0.5 s post-roll (default 15, --fast: 5)")
    p.add_argument("--rig", default=None, metavar="PATH", help="rig yaml (default configs/rigs/rig_v1.yaml)")
    p.add_argument("--fast", action="store_true",
                   help="low-resolution rig (<rig_id>_fast: main x0.25, ultrawide x0.5) and short episodes")
    p.add_argument("--session", default=None, metavar="ID",
                   help="session id (default sim-<UTC time, microseconds>); episode dirs are <session>_episode_NNNNNN")
    p.add_argument("--rigs-dir", default=None, metavar="DIR",
                   help="where the rig yaml used is copied if missing (default $DATAPROC_ROOT/rigs)")
    p.add_argument("--unverified", action="store_true", help="do not write .verified")
    for parser in sub.choices.values():
        parser.set_defaults(_parser=parser)        # usage errors print the subcommand's usage line
    return ap


def _task(args, root: DataRoot) -> TaskRef:
    ap = args._parser
    try:
        ref = resolve_task(args.task, root)
    except ValueError as e:
        ap.error(str(e))
    if args.command in ("intake", "status") and ref.episode is not None:
        ap.error(f"{args.command} --task takes a task, not an episode ({args.task!r})")
    if args.command == "process" and ref.episode is not None:
        if args.episode is not None and args.episode != ref.episode:
            ap.error(f"--task names episode_{ref.episode} but --episode is {args.episode}")
        args.episode = ref.episode
    return ref


def cmd_intake(args, root: DataRoot, task: TaskRef) -> int:
    from dataproc.intake import cli_intake

    return cli_intake(root, task.folder, new=args.new, calibration=args.calibration)


def cmd_process(args, root: DataRoot, task: TaskRef) -> int:
    from dataproc.process import cli_process

    return cli_process(root, task.folder, episode=args.episode, force=args.force, slam=args.slam)


def cmd_status(args, root: DataRoot, task: TaskRef) -> int:
    from dataproc.status import cli_status

    return cli_status(root, task.folder)


def cmd_sim(args, root: DataRoot) -> int:
    from dataproc.sim.episode import run_sim

    if args.duration is not None and not args.duration > 0:
        args._parser.error(f"--duration must be > 0, got {args.duration}")
    try:
        run_sim(out=args.out, episodes=args.episodes, calibration=args.calibration, seed=args.seed,
                duration_s=args.duration, rig_path=args.rig, fast=args.fast, session_id=args.session,
                rigs_dir=args.rigs_dir if args.rigs_dir is not None else root.rigs, verified=not args.unverified)
    except (ValueError, OSError) as e:
        print(f"dataproc sim: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    root = data_root()
    if args.command == "sim":
        return cmd_sim(args, root)
    task = _task(args, root)
    return {"intake": cmd_intake, "process": cmd_process, "status": cmd_status}[args.command](args, root, task)
