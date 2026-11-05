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

    p = sub.add_parser("status", help="show per-episode processing status of a task")
    p.add_argument("--task", required=True, help=_TASK_HELP)

    p = sub.add_parser("sim", help="write synthetic episodes (with ground truth) for testing")
    p.add_argument("--out", required=True, help="output folder, usually $DATAPROC_ROOT/incoming")
    p.add_argument("--episodes", type=_non_negative_int, default=1, metavar="N", help="number of episodes")
    p.add_argument("--calibration", action="store_true", help="write gripper-calibration episodes")
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


def _not_yet(what: str, wave: str):
    raise NotImplementedError(f"dataproc {what} is implemented in {wave}")


def cmd_intake(args, root: DataRoot, task: TaskRef) -> int:
    _not_yet("intake", "wave 1 (E1)")


def cmd_process(args, root: DataRoot, task: TaskRef) -> int:
    _not_yet("process", "wave 2 (E5)")


def cmd_status(args, root: DataRoot, task: TaskRef) -> int:
    _not_yet("status", "wave 2 (E5)")


def cmd_sim(args, root: DataRoot) -> int:
    _not_yet("sim", "wave 1 (E2)")


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    root = data_root()
    try:
        if args.command == "sim":
            return cmd_sim(args, root)
        task = _task(args, root)
        return {"intake": cmd_intake, "process": cmd_process, "status": cmd_status}[args.command](args, root, task)
    except NotImplementedError as e:
        print(f"dataproc: {e}", file=sys.stderr)
        return 1
