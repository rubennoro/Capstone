# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/__main__.py @ 2ffe641
# changes: teleop_collect; the launcher's cli is imported inside main() (lazily), so `python -m teleop_collect` can be
# imported before the launcher exists.
"""python -m teleop_collect check | run [--mock] [--station FILE] [--no-browser]
(= teleop_collect.launcher.cli.main)."""
from __future__ import annotations

import sys


def main() -> int:
    from teleop_collect.launcher.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
