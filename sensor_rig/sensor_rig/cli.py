"""``python -m sensor_rig <command>``: one argparse CLI; each command group registers itself lazily.

A group whose module fails to import (missing optional dependency, or not built yet) is skipped with a note in
``--help``, so the other commands keep working. Exit codes: 0 success, 1 failed/aborted episode, 2 usage error.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# (module, description). Every module exposes register(subparsers).
GROUPS = [
    ("sensor_rig.record.cli", "record, bench-encode"),
    ("sensor_rig.pose.cli_pose", "pose-prep, pose-import"),
    ("sensor_rig.export.cli_export", "export, validate"),
    ("sensor_rig.sim.cli", "sim"),
    ("sensor_rig.calib.cli", "calib-import, calib-check, calib-align, calib-tcp, rig-validate, bench-remap"),
]


def build_parser() -> tuple[argparse.ArgumentParser, list[str]]:
    parser = argparse.ArgumentParser(
        prog="python -m sensor_rig",
        description="Jetson sensor rig: record, pose (VIO adapters), export to the iPhUMI iPhone format, sim, calib.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    unavailable = []
    for module, cmds in GROUPS:
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            unavailable.append(f"{cmds} ({module}: {e})")
            continue
        mod.register(sub)
    if unavailable:
        parser.epilog = "unavailable commands: " + "; ".join(unavailable)
    return parser, unavailable


def main(argv: list[str] | None = None) -> int:
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    rc = args.func(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
