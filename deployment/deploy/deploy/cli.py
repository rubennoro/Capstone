"""The `deploy` command: `deploy check | run [--station FILE] [--mock]` (STRUCTURE.md section 3.7)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import deploy
from deploy.errors import StationError
from deploy.launcher import Launcher
from deploy.station import Station


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--station", type=Path, default=None, metavar="FILE",
                        help="the station file (default: deployment/station.yaml)")
    common.add_argument("--mock", action="store_true", help="use every process's stand-in")
    parser = argparse.ArgumentParser(prog="deploy", description="Check and run the deployment.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{check,run}")
    commands.add_parser("check", parents=[common], help="is everything ready? one table of every check")
    commands.add_parser("run", parents=[common], help="check, start the four processes, wait, stop them all")
    return parser.parse_args(argv)


def shadowed_message(module: str) -> str | None:
    """Why it can't run, when the folder deployment/deploy/ was imported as an empty namespace package instead of the
    installed package: `python -m <module>` run from inside deployment/ (Python puts the working folder first on the
    path). The `deploy` command itself works from any folder."""
    if getattr(deploy, "__file__", None) is not None:
        return None
    return (f"{module}: `python -m {module}` run from inside deployment/ finds the folder deployment/deploy/ instead "
            f"of the installed package: run it from another folder" +
            (", or use the `deploy` command" if module == "deploy" else ""))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    message = shadowed_message("deploy")
    if message is not None:
        print(message, file=sys.stderr)
        return 2
    try:
        station = Station.load(args.station, args.mock)
    except StationError as exc:
        print("deploy: the station file has problems:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 2
    launcher = Launcher(station, args.mock)
    return launcher.check() if args.command == "check" else launcher.run()
