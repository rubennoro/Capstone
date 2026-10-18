"""``python -m franka_sim <cmd>``: registers the built-in commands (lazily imported) and dispatches.

Usage: ``python -m franka_sim --help`` | ``python -m franka_sim check-assets``.
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


def main(argv: list[str] | None = None) -> int:
    _register_builtins()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
