"""Subcommand registry for ``python -m franka_sim <cmd>``.

Why: other tracks add commands (e.g. ``eval``) without editing this file's callers; every module imported by a
command is loaded lazily so ``--help`` never needs mujoco.
Usage: ``cli.register("name", add_parser_fn, main_fn)``; ``add_parser_fn(parser)`` adds arguments,
``main_fn(args) -> int`` runs the command.
"""

from __future__ import annotations

import argparse
from typing import Callable

_REGISTRY: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], int]]] = {}


def register(name: str, add_parser_fn: Callable[[argparse.ArgumentParser], None],
             main_fn: Callable[[argparse.Namespace], int], help: str = "") -> None:
    """Register a subcommand (re-registering a name replaces it)."""
    _REGISTRY[name] = (help, add_parser_fn, main_fn)


def registered() -> list[str]:
    return sorted(_REGISTRY)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="franka_sim", description="Franka MuJoCo simulation tools")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    for name in sorted(_REGISTRY):
        help_, add_parser_fn, main_fn = _REGISTRY[name]
        p = sub.add_parser(name, help=help_ or None, description=help_ or None)
        add_parser_fn(p)
        p.set_defaults(_main=main_fn)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "_main", None) is None:
        ap.print_help()
        return 2
    return int(args._main(args) or 0)
