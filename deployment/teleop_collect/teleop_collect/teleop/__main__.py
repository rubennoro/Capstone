# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/__main__.py @ 2ffe641 (main)
# changes: process name teleop; the body moved to app.run (imported inside main, so the FATAL path covers an import
# error); imports.
# The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/__main__.py @ 91b178b
# changes: the common arguments of policy_deploy.contracts.ProcessArgs; settings from Settings.load (station + link.yaml
# + frames.yaml); builds the PC side (PcClient, PcClock, HeartbeatSender, PcListener, LinkStats, StartupChecks); the
# FATAL line through print_fatal ("FATAL manager: ...").
"""python -m teleop_collect.teleop [--station FILE] [--session DIR] [--ready-file FILE] [--mock] (STRUCTURE.md 6).

--mock: the PC is on 127.0.0.1 (the fake PC, or franka_control --mock --local in the cross test), with the link.yaml
ports, and hello must say mock.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from teleop_collect.contracts import ProcessArgs, print_fatal, setup_logging

NAME = "teleop"
log = logging.getLogger(NAME)


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="teleop: the teleop loop, episodes, C2 in, C3 out, U1")
    root = logging.getLogger()
    if not root.handlers:
        setup_logging()
    try:
        from teleop_collect.teleop.app import run
        return run(args)
    except Exception as exc:                 # anything unexpected (after run's cleanup): traceback, then FATAL
        log.exception("unexpected error: %s", exc)
        return print_fatal(NAME, f"internal error: {type(exc).__name__}: {exc}", "see the traceback in the teleop log")


if __name__ == "__main__":
    sys.exit(main())
