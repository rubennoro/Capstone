# adapted from rnoro/deployment:deployment/policy/policy/__main__.py @ 91b178b
# changes: the common arguments come from policy_deploy.contracts.process_args (--station optional, default
# deployment/policy_deploy/station.yaml); logging set up here
"""`python -m policy_deploy.policy [--station FILE] [--session DIR] [--ready-file FILE] [--mock]` (STRUCTURE.md 2.6, 5)."""

from __future__ import annotations

import sys
from typing import Sequence

from policy_deploy.contracts.process_args import ProcessArgs, setup_logging


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse("policy", argv, description="the policy process (C4 server)")
    setup_logging()
    from policy_deploy.policy.app import PolicyApp            # after logging is set up
    return PolicyApp(args).run()


if __name__ == "__main__":
    sys.exit(main())
