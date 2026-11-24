"""``python -m sim_deploy.sim_pi [--station FILE] [--model scene.xml] [--state ENDPOINT] [--bind ENDPOINT] ...``

The live sim Pi (README.md). SIGINT/SIGTERM stop it; a model/state mismatch exits 1 with one FAILED line.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

import sim_deploy.sim_pi.render  # noqa: F401  (selects EGL + the NVIDIA vendor before mujoco is imported)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.sim_pi", description=__doc__.splitlines()[0])
    ap.add_argument("--station", type=Path, help="station file (its sim_pi: block; model = build/<id>/scene.xml)")
    ap.add_argument("--model", type=Path, help="scene.xml (model.json beside it)")
    ap.add_argument("--state", help="sim.state/v1 endpoint (default tcp://127.0.0.1:5650)")
    ap.add_argument("--bind", help="pi_stream/v1 PUB endpoint (default tcp://127.0.0.1:5555)")
    ap.add_argument("--rig-file", type=Path, dest="rig_file", help="deployed rig file (default the copy's)")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--save-dir", type=Path, dest="save_dir", help="save raw 640x480 frames here")
    ap.add_argument("--save-every", type=int, dest="save_every", help="every n-th packet (default 30)")
    ap.add_argument("--stats-json", type=Path, dest="stats_json", help="write running totals (JSON) here")
    ap.add_argument("--no-shadows", dest="shadows", action="store_const", const=False)
    ap.add_argument("--send-delay-ms", type=float, dest="send_delay_ms")
    ap.add_argument("--camera-look", type=Path, dest="camera_look",
                    help="opt-in camera model YAML (sim_deploy/look/camera_look.py) applied before JPEG")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    from sim_deploy.sim_pi import settings
    from sim_deploy.sim_pi.sim_pi import SimPi
    over = {k: getattr(a, k) for k in ("model", "state", "bind", "rig_file", "seconds", "save_dir", "save_every",
                                       "stats_json", "shadows", "send_delay_ms", "camera_look")}
    try:
        s = settings.load(a.station, over)
        pi = SimPi(s)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"sim_pi: FAILED start: {e}", flush=True)
        return 1
    print(pi.describe(), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: pi.stop())
    signal.signal(signal.SIGINT, lambda *_: pi.stop())
    try:
        summary = pi.run()
    except RuntimeError as e:
        print(f"sim_pi: FAILED {e}", flush=True)
        return 1
    print("sim_pi: stopped " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
