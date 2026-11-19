# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/cli.py @ 2ffe641
# changes: command name teleop_collect; the default station is deployment/teleop_collect/station.yaml; + run
# --no-browser (passed to the Launcher as open_browser=False); imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/deploy/cli.py @ 91b178b
# changes: command name policy_deploy; the default station is deployment/policy_deploy/station.yaml; a station that
# does not load is the failing `station.yaml` row of the table (one row per problem), then exit 2; the namespace-
# shadowing message is gone (the package name policy_deploy is not a folder of deployment/).
"""The `teleop_collect` command: `teleop_collect check | run [--mock] [--station FILE] [--no-browser]`
(STRUCTURE.md 10.1, PLAN.md rule 3)."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from teleop_collect.contracts.station import Station, StationError
from teleop_collect.launcher.check_result import CheckResult, print_table
from teleop_collect.launcher.launcher import Launcher


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--station", type=Path, default=None, metavar="FILE",
                        help="the station file (default: deployment/teleop_collect/station.yaml)")
    common.add_argument("--mock", action="store_true", help="the laptop alone: fake Pi, fake PC on 127.0.0.1")
    parser = argparse.ArgumentParser(prog="teleop_collect", description="Check and run the laptop's processes.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{check,run}")
    commands.add_parser("check", parents=[common], help="is everything ready? one table of every check")
    run = commands.add_parser("run", parents=[common],
                              help="check, start fake_pc (--mock), frames, teleop and ui, open the page, stop them")
    run.add_argument("--no-browser", action="store_true", help="print the page's address, do not open a browser")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        station = Station.load(args.station, args.mock)
        if args.command == "run":
            launcher = Launcher(station, args.mock, open_browser=not args.no_browser)
        else:
            launcher = Launcher(station, args.mock)
    except StationError as exc:
        rows = [CheckResult("station", "station.yaml", "fail", problem) for problem in exc.problems]
        print_table(rows, sys.stdout)
        if args.command == "run":
            print(f"teleop_collect: not started: {len(rows)} check(s) failed (see above)")
        else:
            print(f"teleop_collect check: {len(rows)} check(s) failed")
        return 2
    return launcher.check() if args.command == "check" else launcher.run()
