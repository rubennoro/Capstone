"""``teleop_train package TASK DATASET``: selection, consistency, the zip, package.json (PLAN.md §4.1, §6).
Owner: E5 (wave 2).

The options below are fixed by the plan; E5 implements the rest.
"""
from __future__ import annotations

import argparse


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", metavar="TASK", help="the task folder under the raw root")
    p.add_argument("dataset", metavar="DATASET", help="a new dataset name, e.g. pick_v1 (letters, digits, '_', '-')")
    p.add_argument("--episodes", type=int, default=None, metavar="N",
                   help="only the first N usable episodes, in episode order (default: all)")
    p.add_argument("--include-fail", action="store_true", help="also package episodes saved as fail")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset of that name")


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError
