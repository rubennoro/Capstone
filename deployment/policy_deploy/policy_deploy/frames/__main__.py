# adapted from rnoro/deployment:deployment/process_frames/process_frames/__main__.py @ 91b178b
# changes: the common arguments, session folder and FATAL line come from policy_deploy.contracts.process_args
# (--station defaults to deployment/policy_deploy/station.yaml); the process is named "frames".
"""``python -m policy_deploy.frames [--station FILE] [--session DIR] [--ready-file FILE] [--mock]`` (STRUCTURE.md 4).

A failure ends with the line ``FATAL frames: <what>: <check>`` and exit code 1; SIGINT / SIGTERM is a clean stop,
exit 0. Outputs: ``<session>/frames/stats.jsonl``.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Sequence

from policy_deploy.contracts.process_args import ProcessArgs, print_fatal, setup_logging

from policy_deploy.frames import NAME
from policy_deploy.frames.errors import FatalError, SettingsError


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="Receives the Pi's frames (C0), serves C1.")
    setup_logging()
    from policy_deploy.frames.app import FramesApp            # after logging is set up
    from policy_deploy.frames.settings import Settings
    try:
        settings = Settings.load(args.station, args.mock)
    except SettingsError as e:
        return print_fatal(NAME, f"invalid settings: {'; '.join(e.problems)}",
                           "check station.yaml frames: and config/frames_process.yaml")
    out_dir = args.out_dir(settings.session_root)
    app = FramesApp(settings, args.ready_file, out_dir)

    def on_signal(signum, frame) -> None:
        logging.getLogger(NAME).info("%s: stopping", signal.Signals(signum).name)
        app.request_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        app.run()
    except FatalError as e:
        return print_fatal(NAME, e.what, e.check)
    except Exception as e:
        logging.getLogger(NAME).exception("unexpected error")
        return print_fatal(NAME, f"internal error: {type(e).__name__}: {e}", "see the traceback above")
    logging.getLogger(NAME).info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
