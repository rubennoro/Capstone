# adapted from rnoro/deployment:deployment/deploy/deploy/cli.py @ 91b178b
# changes: command name policy_deploy; the default station is deployment/policy_deploy/station.yaml; a station that
# does not load is the failing `station.yaml` row of the table (one row per problem), then exit 2; the namespace-
# shadowing message is gone (the package name policy_deploy is not a folder of deployment/).
"""The `policy_deploy` command: `policy_deploy check | run [--station FILE] [--mock]` (STRUCTURE.md 9.2, 9.3), and
`policy_deploy use RUN_FOLDER | hold_still` (use_policy.py)."""
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
    use = commands.add_parser("use", help="point station.yaml at a training run folder (or hold_still)",
                              description="Rewrite the policy: and manager: sections of station.yaml from a fetched "
                                          "training run (data/<folder>/policy/<run>/). Nothing else changes.")
    use.add_argument("target", metavar="RUN_FOLDER|hold_still", help="a run folder, or hold_still")
    use.add_argument("--station", type=Path, default=None, metavar="FILE",
                     help="the station file (default: deployment/policy_deploy/station.yaml)")
    use.add_argument("--prompt-index", type=int, default=None, metavar="N", help="which demo to show it (default 0)")
    use.add_argument("--task", default=None, metavar="NAME", help="the task in the dataset (default: its only task)")
    use.add_argument("--prompt-dataset", type=Path, default=None, metavar="ZARR_ZIP",
                     help="another .zarr.zip of demos (default: the one the run trained on, from data/<folder>/package/)")
    use.add_argument("--prompt-split", choices=("train", "val", "all"), default=None,
                     help="which demos of the dataset (default train; all for a dataset it never trained on)")
    use.add_argument("--precision", choices=("fp32", "bf16"), default=None, help="default: keep what is set")
    use.add_argument("--diffusion-steps", type=int, default=None, metavar="N", help="default: keep what is set")
    use.add_argument("--exec-h", type=int, default=None, metavar="N", help="default: keep what is set")
    return parser.parse_args(argv)


def _use(args: argparse.Namespace) -> int:
    from policy_deploy.launcher import use_policy as up
    try:
        old, new, choice = up.use(args.station, args.target, task=args.task, prompt_index=args.prompt_index,
                                  prompt_dataset=args.prompt_dataset, prompt_split=args.prompt_split,
                                  precision=args.precision, diffusion_steps=args.diffusion_steps,
                                  exec_h=args.exec_h)
    except up.UseError as e:
        print(f"policy_deploy use: FAIL {e.what}" + (f"\n  hint: {e.hint}" if e.hint else ""))
        print("policy_deploy use: station.yaml not changed")
        return 2
    except StationError as e:
        print("policy_deploy use: FAIL the new station.yaml would not load:")
        for p in e.problems:
            print(f"  {p}")
        print("policy_deploy use: station.yaml not changed")
        return 2
    print(up.diff(old, new, "station.yaml") or "(no change)")
    for note in choice.notes:
        print(f"note: {note}")
    if choice.label != up.HOLD_STILL and up.frames_settled() is False:
        print("note: contracts/frames.yaml says settled: false, so check/run refuse a trained policy on the real arm "
              "(--mock works)")
    print(f"policy_deploy use: station.yaml now runs {choice.label}. Next: policy_deploy check")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "use":
        return _use(args)
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
