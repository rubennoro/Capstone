"""CLI group for the wireless path: ``sensor-rig stream-send`` (``register(subparsers)``, wired by ``sensor_rig.cli``).

    python -m sensor_rig stream-send --rig rig.yaml --host 192.168.1.20 [--port 5555] [--fps 30] [--jpeg-quality 90]
           [--backend argus|mock] [--episode RAW_EPISODE_DIR] [--loop] [--duration S] [--max-queue 4]
           [--drop-policy drop_oldest|drop_newest] [--connect-timeout S] [--clock sender_monotonic|sender_realtime]

Streams the rig's iPhone-equivalent frames (960x720, true capture stamps) to the policy host's receiver as v2 packets.
``--backend mock --episode DIR`` replays a raw rig episode instead of the cameras (laptop test; the sim profiles use
franka_deploy.stream.rig_replay_sender, which runs this same sender code behind a Wi-Fi emulator). Needs the argus
backend and the Jetson cameras on hardware (untested off the rig: hardware day).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RIG = Path(__file__).resolve().parents[1] / "configs" / "rig_default.yaml"


def register(subparsers) -> None:
    p = subparsers.add_parser("stream-send", help="stream the rig's frames to the policy host (Wi-Fi, v2 protocol)",
                              description=__doc__.splitlines()[0])
    p.add_argument("--rig", default=str(DEFAULT_RIG), help="rig.yaml (default: configs/rig_default.yaml)")
    p.add_argument("--host", required=True, help="policy host (receiver) address")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--fps", type=float, default=30.0, help="send rate (frames faster than this are decimated)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--backend", choices=["argus", "mock"], default="argus")
    p.add_argument("--episode", default=None, help="raw rig episode to replay (with --backend mock)")
    p.add_argument("--loop", action="store_true", help="loop the replayed episode")
    p.add_argument("--duration", type=float, default=None, help="stop after S seconds (default: until Ctrl-C)")
    p.add_argument("--max-queue", type=int, default=4)
    p.add_argument("--drop-policy", choices=["drop_oldest", "drop_newest"], default="drop_oldest")
    p.add_argument("--connect-timeout", type=float, default=15.0, help="seconds to wait for the receiver at start")
    p.add_argument("--clock", choices=["sender_monotonic", "sender_realtime"], default="sender_monotonic")
    p.set_defaults(func=_cmd_stream_send)


def _cmd_stream_send(args: argparse.Namespace) -> int:
    from sensor_rig.rig_config import load_rig_arg
    from sensor_rig.stream.sender import RigStreamSender

    rig = load_rig_arg(args.rig)
    if args.episode and args.backend != "mock":
        print("sensor-rig stream-send: --episode needs --backend mock", file=sys.stderr)
        return 2
    opts = {"episode_dir": args.episode, "loop": args.loop, "realtime": True} if args.episode else {}
    sender = RigStreamSender(rig, args.host, args.port, fps_hz=args.fps, jpeg_quality=args.jpeg_quality,
                             backend=args.backend, backend_opts=opts, max_queue=args.max_queue,
                             drop_policy=args.drop_policy, clock=args.clock,
                             connect_timeout_s=args.connect_timeout)
    try:
        stats = sender.run(args.duration)
    except KeyboardInterrupt:
        sender.stop()
        stats = sender.stats()
    except ConnectionError as e:
        print(f"sensor-rig stream-send: {e}", file=sys.stderr)
        return 1
    finally:
        sender.close()
    print(json.dumps({k: v for k, v in stats.items() if k != "source"}, default=str))
    return 1 if stats.get("capture_error") else 0
