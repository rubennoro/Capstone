"""`python -m policy --station FILE [--session DIR] [--ready-file FILE] [--mock]` (STRUCTURE.md 9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from policy.app import PolicyApp


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m policy", description="the policy process (C4 server)")
    ap.add_argument("--station", type=Path, required=True, help="the station file")
    ap.add_argument("--session", type=Path, default=None, help="the session folder; outputs go to DIR/policy/")
    ap.add_argument("--ready-file", type=Path, default=None, help="created (empty) once ready")
    ap.add_argument("--mock", action="store_true", help="use the hold_still stand-in (no torch)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    return PolicyApp(a.station, a.session, a.ready_file, a.mock).run()


if __name__ == "__main__":
    sys.exit(main())
