"""``python -m franka_sim eval``: registers the subcommand through franka_sim.cli (I-CLI). Lazy imports only, so
``--help`` needs no mujoco."""

from __future__ import annotations

import argparse


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--deploy-config", default=None, help="deploy.yaml (default: controlSW/franka_deploy/configs/deploy.yaml)")
    p.add_argument("--profile", required=True, help="deploy.yaml station profile, e.g. sim-iphone-fake, sim-rig-fake")
    p.add_argument("--stage", default="all", choices=["1", "2a", "2b", "all"], help="all = 1, 2a, 2b in turn")
    p.add_argument("--policy", default=None, help="run/policy id (default: the profile's run.policy)")
    p.add_argument("--episodes", type=int, default=None, help="N demos: Stage 2a = the first N of the checkpoint's held-out split (default all of it); Stage 1 / 2b = demos 0..N-1 (2b default 1)")
    p.add_argument("--demos", default=None, help="explicit zarr demo indices, e.g. 0,3 (wins over --episodes)")
    p.add_argument("--set", action="append", default=[], metavar="group.path=value", help="station override (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="Stage 1: coarse 3x3x4 placement grid, one fine placement")
    p.add_argument("--jobs", type=int, default=4, help="Stage 1 worker processes")
    p.add_argument("--stage1", default=None, help="Stage 2a: an existing Stage 1 run dir (stage1.json) to take placements from")
    p.add_argument("--port-offset", type=int, default=None, help="Stage 2b: shift every launcher port (default $FD_TEST_PORT_OFFSET)")
    p.add_argument("--out", required=True, help="output directory (outside the repo)")


def run(args: argparse.Namespace) -> int:
    from franka_sim.eval import runner

    return runner.run(args)


def register() -> None:
    from franka_sim import cli

    cli.register("eval", add_arguments, run, help="evaluate a policy on a sensor/station: stages 1, 2a, 2b (criteria-gated reports)")
