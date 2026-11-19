# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/__main__.py @ 2ffe641
# changes: the ui process instead of frames: UiApp (from teleop_collect.ui.server) instead of FramesApp, taking only
# the settings and the ready file (no out_dir: the ui writes no outputs; the session folder is still created);
# Settings imported at the top from teleop_collect.ui.settings; FatalError from teleop_collect.ui.bridge and
# SettingsError from teleop_collect.ui.settings (the ui has no errors.py); the argparse description; the settings
# FATAL hint names station.yaml ui: and config/ui.yaml; the docstring (the page on 127.0.0.1 only, the ready rule,
# never ends because teleop or frames is slow or gone; no stats.jsonl output); imports teleop_collect.contracts and
# teleop_collect.ui. The source's own header follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/__main__.py @ 91b178b
# changes: the common arguments, session folder and FATAL line come from policy_deploy.contracts.process_args
# (--station defaults to deployment/policy_deploy/station.yaml); the process is named "frames".
"""``python -m teleop_collect.ui [--station FILE] [--session DIR] [--ready-file FILE] [--mock]`` (STRUCTURE.md 9).

Serves the operator's page on http://127.0.0.1:<ports.http> only. Ready (the ready file) once the HTTP socket is
bound and teleop answered one get_status. A failure ends with ``FATAL ui: <what>: <check>`` and exit code 1;
SIGINT / SIGTERM is a clean stop, exit 0. teleop or frames being slow or gone never ends it.
"""
from __future__ import annotations

import logging
import signal
import sys
from typing import Sequence

from teleop_collect.contracts import ProcessArgs, print_fatal, setup_logging

from teleop_collect.ui import NAME
from teleop_collect.ui.bridge import FatalError
from teleop_collect.ui.settings import Settings, SettingsError


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="Serves the operator's page (127.0.0.1 only).")
    setup_logging()
    from teleop_collect.ui.server import UiApp                # after logging is set up
    try:
        settings = Settings.load(args.station, args.mock)
    except SettingsError as e:
        return print_fatal(NAME, f"invalid settings: {'; '.join(e.problems)}",
                           "check station.yaml ui: and config/ui.yaml")
    args.out_dir(settings.session_root)
    app = UiApp(settings, ready_file=args.ready_file)

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
