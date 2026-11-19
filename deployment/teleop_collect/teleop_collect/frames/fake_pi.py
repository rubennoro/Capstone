# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/fake_pi.py @ 2ffe641
# changes: imports; the module path python -m teleop_collect.frames.fake_pi (docstring, prog). The source's own
# header follows.
# copied from rnoro/deployment:deployment/process_frames/process_frames/fake_pi.py @ 91b178b
# changes: the station schema (binds ports.mock_pi on 127.0.0.1, plays frames.mock_episode); pi_v1 only (no
# time_sync); --station optional (default deployment/policy_deploy/station.yaml).
"""``python -m teleop_collect.frames.fake_pi [--station FILE]``: run the fake Pi alone.

Binds ``tcp://127.0.0.1:<ports.mock_pi>`` and plays back ``frames.mock_episode`` until Ctrl-C / SIGTERM, as the
real Pi sends it (pi_v1) at the rig file's frame sizes. A ``python -m teleop_collect.frames`` in a second terminal,
whose station's ``pi.address``/``pi.port`` point at that address, can then be tested against it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from teleop_collect.contracts.process_args import setup_logging

from teleop_collect.frames._dataproc.config import ConfigError
from teleop_collect.frames.errors import SettingsError
from teleop_collect.frames.fake_pi_sender import FakePiSender
from teleop_collect.frames.pi_recording import PiRecording, PiRecordingError
from teleop_collect.frames.rig_file import camera_sizes, load_live_rig
from teleop_collect.frames.settings import Settings

log = logging.getLogger("frames.fake_pi")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m teleop_collect.frames.fake_pi", description=__doc__)
    p.add_argument("--station", type=Path, default=None)
    args = p.parse_args(argv)
    setup_logging()
    try:
        s = Settings.load(args.station, mock=True)
        m = s.mock_settings
        rec = PiRecording.load(s.mock_episode, max_seconds=m.max_seconds, jpeg_quality=m.jpeg_quality,
                               sizes=camera_sizes(load_live_rig(s.rig_file)))
    except (SettingsError, PiRecordingError, ConfigError) as e:
        print(f"fake_pi: {e}", file=sys.stderr)
        return 1
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sender = FakePiSender(rec, s.mock_pi_frames, pi_clock_offset_ns=int(round(m.pi_clock_offset_s * 1e9)),
                          drift_ppm=m.drift_ppm, send_hwm=m.send_hwm)
    sender.start()
    log.info("fake Pi: frames on %s, playing %s (JPEG); Ctrl-C to stop", sender.frames_endpoint, s.mock_episode)
    while not stop.wait(0.2):
        pass
    sender.stop()
    log.info("fake Pi stopped; sent %s", dict(sender.sent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
