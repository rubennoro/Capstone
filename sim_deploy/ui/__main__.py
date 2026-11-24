"""``python -m sim_deploy.ui [--port 8770] [--stub [PARTS]] [--config FILE]`` (env ~/envs/sim_deploy).

Serves the page on http://127.0.0.1:<port> only (ui/DESIGN.md). `--stub` uses fake runner, live data and catalog
(ui/_stubs.py), `--stub runner,live` only some of them. Ctrl-C (or SIGTERM) stops a running eval cleanly first (a
second Ctrl-C kills it), then the server; exit 0. Bad settings or a missing part: ``FATAL sim_deploy.ui: ...``, exit 1.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from sim_deploy.ui import NAME

PARTS = ("runner", "live", "catalog")


def _stub_parts(value: str) -> tuple[str, ...]:
    if value in ("all", ""):
        return PARTS
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown part(s) {', '.join(bad)} (parts: {', '.join(PARTS)}, or all)")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim_deploy.ui",
                                description="Pick a policy and run sim rollouts on it (127.0.0.1 only).")
    p.add_argument("--port", type=int, help="the page's port (default: config.yaml http_port, 8770)")
    p.add_argument("--stub", nargs="?", const="all", default=None, type=_stub_parts, metavar="PARTS",
                   help="fake parts (ui/_stubs.py): all (no value), or a comma list of runner, live, catalog")
    p.add_argument("--config", help="settings file (default: sim_deploy/ui/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    from sim_deploy.ui.server import PartMissing, run
    from sim_deploy.ui.settings import Settings, SettingsError
    stub = args.stub or ()
    try:
        settings = Settings.load(args.config, http_port=args.port, stub=bool(stub))
    except SettingsError as e:
        print(f"FATAL {NAME}: invalid settings: {'; '.join(e.problems)}", file=sys.stderr)
        return 1
    try:
        return run(settings, stub)
    except PartMissing as e:
        print(f"FATAL {NAME}: {e}", file=sys.stderr)
        return 1
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
