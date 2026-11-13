"""``python -m process_frames.fake_pi --station FILE``: run the fake Pi alone (STRUCTURE.md §5.16).

Binds the station's ``endpoints.mock_pi_*`` and plays back ``process_frames.mock_episode`` until Ctrl-C / SIGTERM,
in ``process_frames.stream`` (pi_v1: as the real Pi, frames only) at the rig file's frame sizes.
A ``python -m process_frames`` (no --mock) whose station points ``endpoints.pi_*`` at those addresses can then be
tested against it in a second terminal.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from process_frames.__main__ import LOG_FORMAT
from process_frames.errors import SettingsError
from process_frames.fake_pi_sender import FakePiSender
from process_frames._dataproc.config import ConfigError
from process_frames.pi_recording import PiRecording, PiRecordingError
from process_frames.rig_file import camera_sizes, load_live_rig
from process_frames.settings import Settings

log = logging.getLogger("process_frames.fake_pi")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m process_frames.fake_pi", description=__doc__)
    p.add_argument("--station", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
    try:
        s = Settings.load(args.station, mock=True)
        m = s.mock_settings
        rec = PiRecording.load(s.mock_episode, encoding=s.decoder, max_seconds=m.max_seconds,
                               keyframe_interval=m.keyframe_interval, jpeg_quality=m.jpeg_quality,
                               sizes=camera_sizes(load_live_rig(s.rig_file)))
    except (SettingsError, PiRecordingError, ConfigError) as e:
        print(f"fake_pi: {e}", file=sys.stderr)
        return 1
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sender = FakePiSender(rec, s.pi_frames, s.pi_time_sync, pi_clock_offset_ns=int(round(m.pi_clock_offset_s * 1e9)),
                          drift_ppm=m.drift_ppm, send_hwm=m.send_hwm, stream=s.stream)
    sender.start()
    log.info("fake Pi: stream %s, frames on %s, time_sync on %s, playing %s (%s); Ctrl-C to stop", s.stream,
             sender.frames_endpoint, sender.time_sync_endpoint, s.mock_episode, s.decoder)
    while not stop.wait(0.2):
        pass
    sender.stop()
    log.info("fake Pi stopped; sent %s", dict(sender.sent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
