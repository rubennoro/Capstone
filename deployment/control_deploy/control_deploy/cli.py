"""control_deploy check | run | console | firewall [--mock] [--local] [--station FILE] (control_deploy/STRUCTURE.md 6.1).

  check     the PC's checks (a table; exit 2 on any fail). Never moves the arm, never connects to FCI.
  run       check, then franka_control on this terminal (keys h home arm, p start pose, g home hand, s stop, q quit).
  console   the bring-up C3 client (a second terminal, while run shows idle or stopped; never with the laptop).
  firewall  prints the nftables table for contracts/link.yaml (install it once with sudo).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from control_deploy import paths


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mock", action="store_true", help="FakeArm and FakeGripper; skip the arm and real-time rows")
    p.add_argument("--local", action="store_true", help="bind / connect on 127.0.0.1 with the link.yaml ports")
    p.add_argument("--station", default=None, help="this PC's station file (default: control_deploy/station.yaml)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="control_deploy", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="command", required=True)
    _common(sub.add_parser("check", help="the PC's checks"))
    _common(sub.add_parser("run", help="check, then franka_control on this terminal"))
    c = sub.add_parser("console", help="the bring-up C3 client")
    _common(c)
    c.add_argument("--only", nargs="+", metavar="STEP", help="a subset of: hold move rotate gripper watchdog out_of_box")
    c.add_argument("--yes", action="store_true", help="no Enter before each moving step (refused without --mock)")
    sub.add_parser("firewall", help="print the nftables table for contracts/link.yaml")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "firewall":
        from deploy_contracts import LinkSettings

        from control_deploy.firewall import rule_text
        sys.stdout.write(rule_text(LinkSettings.load(paths.contracts_dir() / "link.yaml")))
        return 0
    if args.command == "check":
        from control_deploy.check_result import exit_code, print_table
        from control_deploy.checks import PcChecks
        results = PcChecks(args.station, args.mock, args.local).run()
        print_table(results, sys.stdout)
        return exit_code(results)
    if args.command == "run":
        from control_deploy.runner import Runner
        return Runner(args.station, args.mock, args.local).run()
    from control_deploy.console.console import Console
    try:
        return Console(args.station, args.mock, args.local, only=args.only, yes=args.yes).run()
    except ValueError as exc:   # the station or a contract file
        sys.stdout.write(f"console: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
