"""``python -m franka_sim eval``: registers the subcommand through franka_sim.cli (I-CLI). Lazy imports only, so
``--help`` needs no mujoco."""

from __future__ import annotations

import argparse


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--deploy-config", default=None, help="deploy.yaml (default: controlSW/franka_deploy/configs/deploy.yaml)")
    p.add_argument("--profile", required=True, help="deploy.yaml station profile, e.g. sim-rig")
    p.add_argument("--stage", default="3", choices=["3"],
                   help="3 = the closed loop (judge, between-episode reset); the only stage on this branch")
    p.add_argument("--policy", default=None, help="run/policy id (default: the profile's run.policy)")
    p.add_argument("--set", action="append", default=[], metavar="group.path=value", help="station override (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port-offset", type=int, default=None, help="Stage 3: shift every launcher port (default $FD_TEST_PORT_OFFSET)")
    p.add_argument("--sim-stub", default=None, metavar="SCENARIO",
                   help="Stage 3: use the in-process stub simulator (lift_success, no_lift, lift_drop, knock_over, collision, reflex) instead of MuJoCo")
    p.add_argument("--episodes-per-pose", type=int, default=None, help="Stage 3: episodes per eraser pose (default sim.stage3.n_episodes_per_pose)")
    p.add_argument("--poses", type=int, default=None, help="Stage 3: use the first N eraser poses (default all of sim.stage3.poses)")
    p.add_argument("--out", required=True, help="output directory (outside the repo)")


def run(args: argparse.Namespace) -> int:
    from franka_sim.eval import runner

    return runner.run(args)


def register() -> None:
    from franka_sim import cli

    cli.register("eval", add_arguments, run, help="evaluate a policy on a sensor/station: Stage 3, the closed loop (criteria-gated report; exit 0 PASS, 1 INCOMPLETE, 2 usage/config error, 3 FAIL)")
