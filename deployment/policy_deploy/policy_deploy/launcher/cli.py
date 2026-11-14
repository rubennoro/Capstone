# adapted from rnoro/deployment:deployment/deploy/deploy/cli.py @ 91b178b
# changes: command name policy_deploy; the default station is deployment/policy_deploy/station.yaml; a station that
# does not load is the failing `station.yaml` row of the table (one row per problem), then exit 2; the namespace-
# shadowing message is gone (the package name policy_deploy is not a folder of deployment/).
"""The `policy_deploy` command: `policy_deploy check | run [--station FILE] [--mock]` (STRUCTURE.md 9.2, 9.3)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from policy_deploy.contracts.station import Station, StationError
from policy_deploy.launcher.check_result import CheckResult, print_table
from policy_deploy.launcher.launcher import Launcher


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--station", type=Path, default=None, metavar="FILE",
                        help="the station file (default: deployment/policy_deploy/station.yaml)")
    common.add_argument("--mock", action="store_true", help="the laptop alone: fake Pi, fake PC on 127.0.0.1")
    parser = argparse.ArgumentParser(prog="policy_deploy", description="Check and run the laptop's processes.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{check,run}")
    commands.add_parser("check", parents=[common], help="is everything ready? one table of every check")
    commands.add_parser("run", parents=[common], help="check, start frames, policy and manager, wait, stop them")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        station = Station.load(args.station, args.mock)
        launcher = Launcher(station, args.mock)
    except StationError as exc:
        rows = [CheckResult("station", "station.yaml", "fail", problem) for problem in exc.problems]
        print_table(rows, sys.stdout)
        if args.command == "run":
            print(f"policy_deploy: not started: {len(rows)} check(s) failed (see above)")
        else:
            print(f"policy_deploy check: {len(rows)} check(s) failed")
        return 2
    return launcher.check() if args.command == "check" else launcher.run()
