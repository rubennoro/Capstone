"""``python -m process_frames --station FILE [--session DIR] [--ready-file FILE] [--mock]`` (STRUCTURE.md §12).

The common arguments of deploy/STRUCTURE.md §4.2. A failure ends with the line
``FATAL process_frames: <what>: <check>`` and exit code 1; SIGINT / SIGTERM is a clean stop, exit 0.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import signal
import sys
from pathlib import Path
from typing import Sequence

from process_frames import NAME
from process_frames.errors import FatalError, SettingsError

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"python -m {NAME}", description="Receives the Pi's frames, serves C1.")
    p.add_argument("--station", type=Path, required=True, help="the station file")
    p.add_argument("--session", type=Path, default=None, help="the session folder (outputs go to DIR/process_frames/)")
    p.add_argument("--ready-file", type=Path, default=None, help="create this file once ready")
    p.add_argument("--mock", action="store_true", help="use the fake Pi (FakePiSender) inside this process")
    return p.parse_args(argv)


def out_dir_for(session: Path | None, session_root: Path) -> Path:
    """``DIR/process_frames/``; alone: ``<session_root>/<YYYY-MM-DD_HH-MM-SS>_process_frames_alone/process_frames/``."""
    if session is None:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session = session_root / f"{stamp}_{NAME}_alone"
    return Path(session) / NAME


def _fatal(e: FatalError) -> int:
    logging.getLogger(NAME).error("%s", e)
    sys.stderr.flush()
    print(f"FATAL {NAME}: {e}", file=sys.stderr, flush=True)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
    from process_frames.app import FramesApp            # after logging is set up
    from process_frames.settings import Settings
    try:
        settings = Settings.load(args.station, args.mock)
    except SettingsError as e:
        return _fatal(FatalError(f"invalid settings: {'; '.join(e.problems)}",
                                 "check station.yaml's process_frames section"))
    out_dir = out_dir_for(args.session, settings.session_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = FramesApp(settings, args.ready_file, out_dir)

    def on_signal(signum, frame) -> None:
        logging.getLogger(NAME).info("%s: stopping", signal.Signals(signum).name)
        app.request_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        app.run()
    except FatalError as e:
        return _fatal(e)
    except Exception as e:
        logging.getLogger(NAME).exception("unexpected error")
        return _fatal(FatalError(f"internal error: {type(e).__name__}: {e}", "see the traceback above"))
    logging.getLogger(NAME).info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
