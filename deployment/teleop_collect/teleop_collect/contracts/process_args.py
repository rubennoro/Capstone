# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/process_args.py @ 2ffe641
# changes: PROCESS_NAMES frames, teleop, ui, fake_pc; prog python -m teleop_collect.<name>; --station defaults to
# deployment/teleop_collect/station.yaml; imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/__main__.py @ 91b178b (parse_args,
# out_dir_for, _fatal) and the same code of policy/ and message_comms/
# changes: written once for every laptop process (frames, policy, manager, fake_pc); --station defaults to
# deployment/policy_deploy/station.yaml; fatal_line() builds the FATAL line of STRUCTURE.md 2.6.
"""The common arguments of every process, its session folder, its ready file and its FATAL line (STRUCTURE.md 1).

`python -m teleop_collect.<process> [--station FILE] [--session DIR] [--ready-file FILE] [--mock]`. Alone (no
--session) a process writes into `<session_root>/<YYYY-MM-DD_HH-MM-SS>_<name>_alone/<name>/`, the same layout as a
run. A failure ends with `FATAL <process>: <what failed>: <what to check>` as the last stderr line and exit 1; a
normal stop (q, SIGINT, SIGTERM) is exit 0 with no FATAL line.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from teleop_collect.contracts import paths

PROCESS_NAMES = ("frames", "teleop", "ui", "fake_pc")
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


@dataclass(frozen=True)
class ProcessArgs:
    name: str                   # frames | teleop | ui | fake_pc
    station: Path
    session: Path | None
    ready_file: Path | None
    mock: bool

    @classmethod
    def parse(cls, name: str, argv: Sequence[str] | None = None, description: str = "",
              extra: Callable[[argparse.ArgumentParser], None] | None = None) -> tuple[ProcessArgs, argparse.Namespace]:
        """Parse the common arguments (plus any `extra` adds); returns them and the full namespace."""
        if name not in PROCESS_NAMES:
            raise ValueError(f"unknown process {name!r}; known: {', '.join(PROCESS_NAMES)}")
        p = argparse.ArgumentParser(prog=f"python -m teleop_collect.{name}", description=description)
        p.add_argument("--station", type=Path, default=None,
                       help="the station file (default: deployment/teleop_collect/station.yaml)")
        p.add_argument("--session", type=Path, default=None, help=f"the session folder (outputs go to DIR/{name}/)")
        p.add_argument("--ready-file", type=Path, default=None, help="create this file once ready")
        p.add_argument("--mock", action="store_true", help="the laptop alone (fake Pi, fake PC on 127.0.0.1)")
        if extra is not None:
            extra(p)
        ns = p.parse_args(argv)
        station = Path(ns.station).expanduser().absolute() if ns.station is not None else paths.default_station()
        return cls(name=name, station=station, session=ns.session, ready_file=ns.ready_file, mock=ns.mock), ns

    def out_dir(self, session_root: Path) -> Path:
        """`<session>/<name>/`, or alone `<session_root>/<stamp>_<name>_alone/<name>/`; created."""
        session = self.session
        if session is None:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session = Path(session_root) / f"{stamp}_{self.name}_alone"
        out = Path(session) / self.name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def write_ready(self) -> None:
        """Create the ready file (if one was asked for)."""
        if self.ready_file is not None:
            path = Path(self.ready_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready\n", encoding="utf-8")


def fatal_line(process: str, what: str, check: str = "") -> str:
    """`FATAL <process>: <what failed>: <what to check>`, on one line."""
    text = " ".join(f"{what}: {check}".split()) if check else " ".join(what.split())
    return f"FATAL {process}: {text}"


def print_fatal(process: str, what: str, check: str = "") -> int:
    """Log and print the FATAL line as the last stderr line; returns 1 (the exit code)."""
    line = fatal_line(process, what, check)
    logging.getLogger(process).error("%s", line[len("FATAL ") + len(process) + 2:])
    sys.stderr.flush()
    print(line, file=sys.stderr, flush=True)
    return 1


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
