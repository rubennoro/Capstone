"""``python -m franka_sim <cmd>``: registers the built-in commands (lazily imported) and dispatches.

Usage: ``python -m franka_sim --help`` | ``python -m franka_sim check-assets``
       | ``python -m franka_sim model --profile sim-rig [--clearance]``
       | ``python -m franka_sim eval --profile sim-rig --stage 3 --out DIR``.
"""

from __future__ import annotations

import sys

from franka_sim import cli


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


def main(argv: list[str] | None = None) -> int:
    _register_builtins()
    from franka_sim.eval import cli as _eval_cli  # B8 (Track B): registers `eval`; lazy, no mujoco at import

    _eval_cli.register()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
