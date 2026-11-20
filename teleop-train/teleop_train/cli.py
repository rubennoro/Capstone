"""The command line: ``teleop_train convert`` and ``teleop_train package`` (PLAN.md §4.1).

Each command lives in its own module, which provides ``add_arguments(parser)`` (its own options) and
``run(args) -> int`` (the exit code: 0 done, 1 nothing usable or a refusal). The options every command shares are
added here: ``--raw-root``, ``--out-root``, ``--pose-source``.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from teleop_train import __version__
from teleop_train.paths import DEFAULT_OUT_ROOT, DEFAULT_RAW_ROOT, POSE_SOURCES, DEFAULT_POSE_SOURCE

#: (command, module, one-line help)
COMMANDS = (
    ("convert", "teleop_train.convert", "convert the saved episodes of a task into per-episode training zarrs"),
    ("package", "teleop_train.package", "build a training .zarr.zip + package.json from a task's converted episodes"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="teleop_train", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"teleop_train {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name, module, help_text in COMMANDS:
        sp = sub.add_parser(name, help=help_text, description=help_text)
        sp.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), metavar="DIR",
                        help=f"teleop_collect's data_root (read only; default {DEFAULT_RAW_ROOT})")
        sp.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), metavar="DIR",
                        help=f"rnoro/training's data folder (default {DEFAULT_OUT_ROOT})")
        sp.add_argument("--pose-source", choices=POSE_SOURCES, default=DEFAULT_POSE_SOURCE,
                        help=f"which arm pose is recorded: O_T_TCP_target or O_T_TCP (default {DEFAULT_POSE_SOURCE})")
        importlib.import_module(module).add_arguments(sp)
        sp.set_defaults(_module=module)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(importlib.import_module(args._module).run(args))


if __name__ == "__main__":
    sys.exit(main())
