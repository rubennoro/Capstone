"""python -m policy_deploy.fake_pc [--station FILE] [--session DIR] [--ready-file FILE] --mock (STRUCTURE.md 10).

Binds C2 PUB and C3 REP on 127.0.0.1 with the link.yaml ports. Ready (ready file) once both are bound and the first C2
is published. A failure ends with `FATAL fake_pc: ...`; SIGINT/SIGTERM stop it (exit 0).
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
from typing import Sequence

import zmq

from policy_deploy.contracts import StationError
from policy_deploy.contracts.process_args import ProcessArgs, print_fatal, setup_logging

NAME = "fake_pc"
log = logging.getLogger(NAME)


def main(argv: Sequence[str] | None = None) -> int:
    args, _ = ProcessArgs.parse(NAME, argv, description="the fake PC (franka_control's messages) for --mock")
    setup_logging()
    from policy_deploy.fake_pc.fake_pc import FakePc
    from policy_deploy.fake_pc.settings import FakePcSettings
    try:
        settings = FakePcSettings.load(args.station, mock=True)
    except StationError as e:
        return print_fatal(NAME, f"config: {e}", "check station.yaml fake_pc: and config/fake_pc.yaml")
    out = args.out_dir(settings.session_root)
    fake = FakePc(settings, session_dir=out)
    try:
        fake.start()
    except zmq.ZMQError as e:
        return print_fatal(NAME, f"cannot bind {settings.state_endpoint} / {settings.commands_endpoint}: {e}",
                           "is franka_control --mock --local or another fake PC running? (policy_deploy check)")
    stop = threading.Event()

    def on_signal(signum, frame) -> None:
        log.info("%s: stopping", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if not fake.ready.wait(10.0):
        fake.close()
        return print_fatal(NAME, "no C2 published within 10 s", "see the fake_pc log")
    log.info("fake PC up: C2 %s, C3 %s, clock +%.0f s %+.0f ppm, mode %s", fake.state_endpoint,
             fake.commands_endpoint, settings.clock_offset_s, settings.clock_drift_ppm, fake.mode)
    args.write_ready()
    while not stop.wait(0.2):
        pass
    fake.close()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
