"""python -m policy_deploy check | run [--station FILE] [--mock]  (= policy_deploy.launcher.cli.main)."""
from __future__ import annotations

import sys

from policy_deploy.launcher.cli import main

if __name__ == "__main__":
    sys.exit(main())
