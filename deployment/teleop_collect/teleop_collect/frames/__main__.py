# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/__main__.py @ 2ffe641
# changes: imports; python -m teleop_collect.frames; the description (R1, U2 instead of C1); the ready file by
# ProcessArgs.write_ready (FramesApp on_ready). The source's own
# header follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/__main__.py @ 91b178b
# changes: the common arguments, session folder and FATAL line come from teleop_collect.contracts.process_args
# (--station defaults to deployment/policy_deploy/station.yaml); the process is named "frames".
"""``python -m teleop_collect.frames [--station FILE] [--session DIR] [--ready-file FILE] [--mock]`` (STRUCTURE.md 5.4).

A failure ends with the line ``FATAL frames: <what>: <check>`` and exit code 1; SIGINT / SIGTERM is a clean stop,
exit 0. Outputs: ``<session>/frames/stats.jsonl``.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Sequence

from teleop_collect.contracts.process_args import ProcessArgs, print_fatal, setup_logging

from teleop_collect.frames import NAME
from teleop_collect.frames.errors import FatalError, SettingsError


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="Receives the Pi's frames (C0), records episodes (R1), serves previews (U2).")
    setup_logging()
    from teleop_collect.frames.app import FramesApp            # after logging is set up
    from teleop_collect.frames.settings import Settings
    try:
        settings = Settings.load(args.station, args.mock)
    except SettingsError as e:
        return print_fatal(NAME, f"invalid settings: {'; '.join(e.problems)}",
                           "check station.yaml frames: and config/frames_process.yaml")
    out_dir = args.out_dir(settings.session_root)
    app = FramesApp(settings, args.ready_file, out_dir, on_ready=args.write_ready)

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
