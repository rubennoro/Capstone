"""``python -m franka_sim <cmd>``: registers the built-in commands (lazily imported) and dispatches.

Usage: ``python -m franka_sim --help`` | ``python -m franka_sim check-assets``
       | ``python -m franka_sim model --profile sim-iphone [--robot fr3] [--clearance]``
       | ``python -m franka_sim stage1 --zarr Z --ids robot=panda,... --out DIR``
       | ``python -m franka_sim camera-check ...`` (RE) | ``python -m franka_sim camera-gap ...`` (BP)
       | ``python -m franka_sim campaign --stage1 S --profile sim-iphone --demos all --speed 0.5 --out DIR`` (RE, A8).
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["stage1"]:
    # Before numpy loads BLAS: one thread per stage1 worker process (stage1_replay.limit_threads; A3 review 8).
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

if __name__ == "__main__" and sys.argv[1:2] in (["camera-check"], ["campaign"]):
    os.environ.setdefault("MUJOCO_GL", "egl")           # headless renders (A7)

from franka_sim import cli  # noqa: E402


def _register_builtins() -> None:
    def add_check_assets(p):
        from franka_sim.tools import check_assets

        check_assets.add_arguments(p)

    def run_check_assets(args):
        from franka_sim.tools import check_assets

        return check_assets.run(args)

    cli.register("check-assets", add_check_assets, run_check_assets,
                 help="verify the vendored menagerie assets against SOURCE.md")

    def add_model(p):
        from franka_sim import model

        model.add_arguments(p)

    def run_model(args):
        from franka_sim import model

        return model.run(args)

    cli.register("model", add_model, run_model,
                 help="build the station model; print frames (F_T_TCP, cameras, tags), payload and clearance")

    def add_stage1(p):
        from franka_sim import stage1_replay

        stage1_replay.add_arguments(p)

    def run_stage1(args):
        from franka_sim import stage1_replay

        return stage1_replay.run(args)

    cli.register("stage1", add_stage1, run_stage1,
                 help="offline kinematic feasibility of the recorded demos (placement, home, finger extension, camera side)")

    def add_characterize(p):
        from franka_sim import characterize

        characterize.add_arguments(p)

    def run_characterize(args):
        from franka_sim import characterize

        return characterize.run(args)

    cli.register("characterize", add_characterize, run_characterize,
                 help="sim impedance tracking table (gonogo steps profile, stiffness x payload error) -> JSON")

    def add_camera_check(p):
        from franka_sim import camera_check

        camera_check.add_arguments(p)

    def run_camera_check(args):
        from franka_sim import camera_check

        return camera_check.run(args)

    cli.register("camera-check", add_camera_check, run_camera_check,
                 help="A7: renders vs the real training frames (tags, hand mask, MAE/SSIM at matched poses) (RE)")

    def add_camera_gap(p):
        from franka_sim import camera_gap

        camera_gap.add_arguments(p)

    def run_camera_gap(args):
        from franka_sim import camera_gap

        return camera_gap.run(args)

    cli.register("camera-gap", add_camera_gap, run_camera_gap,
                 help="A7: policy action gap, rendered vs real images, same seed (BP; no mujoco)")

    def add_campaign(p):
        from franka_sim import campaign

        campaign.add_arguments(p)

    def run_campaign(args):
        from franka_sim import campaign

        return campaign.run(args)

    cli.register("campaign", add_campaign, run_campaign,
                 help="A8 Stage 1b: replay every demo through the full sim stack (launcher, MuJoCo, hand, scene, render) "
                      "and measure tracking, reflexes, pad-table contacts, faults -> campaign.json (RE)")


def main(argv: list[str] | None = None) -> int:
    _register_builtins()
    from franka_sim.eval import cli as _eval_cli  # B8 (Track B): registers `eval`; lazy, no mujoco at import

    _eval_cli.register()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
