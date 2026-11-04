"""``python -m franka_sim eval``: registers the subcommand through franka_sim.cli (I-CLI). Lazy imports only, so
``--help`` needs no mujoco."""

from __future__ import annotations

import argparse


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--deploy-config", default=None, help="deploy.yaml (default: controlSW/franka_deploy/configs/deploy.yaml)")
    p.add_argument("--profile", required=True, help="deploy.yaml station profile, e.g. sim-iphone-fake, sim-rig-fake")
    p.add_argument("--stage", default="all", choices=["1", "1b", "2a", "2b", "3", "all"],
                   help="all = 1, 1b, 2a, 2b in turn (1b = the full-stack demo-replay campaign, A8: ~1 h for the 45 demos; pass --stage1 / --demos to shorten it); 3 = the closed loop (judge, between-episode reset), asked for explicitly")
    p.add_argument("--policy", default=None, help="run/policy id (default: the profile's run.policy)")
    p.add_argument("--episodes", type=int, default=None, help="N demos: Stage 2a = the first N of the checkpoint's held-out split (default all of it); Stage 1 / 2b = demos 0..N-1 (2b default 1)")
    p.add_argument("--demos", default=None, help="explicit zarr demo indices, e.g. 0,3 (wins over --episodes)")
    p.add_argument("--set", action="append", default=[], metavar="group.path=value", help="station override (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="Stage 1: coarse 3x3x4 placement grid, one fine placement")
    p.add_argument("--jobs", type=int, default=4, help="Stage 1 worker processes / Stage 1b campaign jobs")
    p.add_argument("--stage1", default=None, help="Stage 1b / 2a / 3 (center stage1): an existing Stage 1 run dir (stage1.json) to take placements from")
    p.add_argument("--speeds", default=None, help="Stage 1b: replay speeds, e.g. 0.5,1.0 (default sim.stage1b.speeds; the first one is gated)")
    p.add_argument("--port-offset", type=int, default=None, help="Stage 2b: shift every launcher port (default $FD_TEST_PORT_OFFSET)")
    p.add_argument("--sweep", default=None, metavar="ID|PATH", help="run a sweep spec (run/sweeps/<id>.yaml or a path): the cartesian grid of its axes, with resume")
    p.add_argument("--max-runs", type=int, default=None, help="--sweep: at most N new runs (overrides the spec's max_runs)")
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

    cli.register("eval", add_arguments, run, help="evaluate a policy on a sensor/station: stages 1, 2a, 2b, 3 and sweeps (criteria-gated reports; exit 0 PASS, 1 INCOMPLETE (a sweep with unrun points too), 2 usage/config error, 3 FAIL)")
