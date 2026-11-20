"""``teleop_train convert TASK``: one episode end to end, the §5.3 checks, status.json, the up-to-date rule
(PLAN.md §4.1, §5). Owner: E4 (wave 2).

The options below are fixed by the plan; E4 implements the rest.
"""
from __future__ import annotations

import argparse

CONVERT_VERSION = 1


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", metavar="TASK", help="the task folder under the raw root")
    p.add_argument("--episodes", default=None, metavar="LIST",
                   help="only these episode numbers, e.g. 3,7-9 (default: all)")
    p.add_argument("--force", action="store_true", help="convert again even when up to date")


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError
