"""CLI group for recording: ``record`` and ``bench-encode`` (``register(subparsers)``, wired by ``sensor_rig.cli``).

    python -m sensor_rig record --rig rig.yaml --out /mnt/nvme/raw --session rig-pilot --side left
           [--kind demonstration|gripper_calibration] [--gripper-id default] [--calibration-run auto|NAME|none]
           [--allow-no-calibration] [--task-label TEXT] [--note TEXT] [--duration S] [--count N] [--pause S]
           [--keep-aborted] [--mock] [--warmup S]
    python -m sensor_rig bench-encode [--width 1640] [--height 1232] [--quality 90] [--n 300] [--workers 1,2,3]

``--mock`` replaces every backend with a mock (synthetic cameras at the rig.yaml resolution/fps, static IMU), so the
full recorder runs on a laptop. Without ``--duration`` the recorder is interactive: Enter starts/stops an episode,
``a`` + Enter aborts it, ``q`` + Enter quits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RIG = Path(__file__).resolve().parents[1] / "configs" / "rig_default.yaml"


def register(subparsers) -> None:
    p = subparsers.add_parser("record", help="record raw rig episodes (Jetson / Raspberry Pi 5; --mock on a laptop)",
                              description=__doc__.splitlines()[0])
    p.add_argument("--rig", default=str(DEFAULT_RIG), help="rig.yaml (default: configs/rig_default.yaml)")
    p.add_argument("--out", required=True, help="raw root: episodes go to <out>/<YYYY-MM-DD>/<NAME>/")
    p.add_argument("--session", required=True)
    p.add_argument("--side", required=True, choices=["left", "right", "head"])
    p.add_argument("--kind", default="demonstration", choices=["demonstration", "gripper_calibration"])
    p.add_argument("--gripper-id", default="default")
    p.add_argument("--calibration-run", default="auto", help="auto (last calibration of this session/side), "
                                                             "an episode NAME, or none")
    p.add_argument("--allow-no-calibration", action="store_true")
    p.add_argument("--task-label", default=None)
    p.add_argument("--note", default="")
    p.add_argument("--duration", type=float, default=None, help="record N episodes of S seconds, non-interactive")
    p.add_argument("--count", type=int, default=1, help="number of episodes with --duration")
    p.add_argument("--pause", type=float, default=1.0, help="seconds between episodes with --duration")
    p.add_argument("--keep-aborted", action="store_true", help="keep aborted episodes as hidden .<NAME>.aborted")
    p.add_argument("--mock", action="store_true", help="mock cameras + IMU (laptop)")
    p.add_argument("--warmup", type=float, default=2.0, help="sensor warm-up before the first episode (s)")
    p.add_argument("--quiet", action="store_true", help="no 1 Hz status line")
    p.set_defaults(func=cmd_record)

    b = subparsers.add_parser("bench-encode", help="JPEG encode cost / throughput per worker count")
    b.add_argument("--width", type=int, default=1640)
    b.add_argument("--height", type=int, default=1232)
    b.add_argument("--quality", type=int, default=90)
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--workers", default="1,2,3", help="comma-separated worker counts")
    b.add_argument("--encoder", default="opencv", choices=["opencv", "turbojpeg"])
    b.set_defaults(func=cmd_bench)


def cmd_bench(args) -> int:
    from sensor_rig.record.bench import bench_encode

    try:
        workers = tuple(int(w) for w in str(args.workers).split(",") if w.strip())
    except ValueError:
        print(f"bad --workers {args.workers!r}", file=sys.stderr)
        return 2
    bench_encode(args.width, args.height, args.quality, args.n, workers, args.encoder)
    return 0


def _make_recorder(args):
    from sensor_rig.drivers.base import make_camera, make_imu
    from sensor_rig.record.recorder import Recorder
    from sensor_rig.rig_config import load_rig_arg

    rig = load_rig_arg(args.rig)
    cameras = imu = None
    if args.mock:
        cameras = {c: make_camera(rig, c, backend="mock") for c in ("main", "ultrawide")}
        imu = make_imu(rig, backend="mock")
    return Recorder(rig, args.out, cameras=cameras, imu=imu, warmup_s=args.warmup, keep_aborted=args.keep_aborted,
                    log=print, status_lines=not args.quiet)


def _episode_kwargs(args) -> dict:
    return dict(kind=args.kind, session=args.session, side=args.side, gripper_id=args.gripper_id,
                calibration_run=args.calibration_run, task_label=args.task_label, note=args.note,
                allow_no_calibration=args.allow_no_calibration)


def cmd_record(args) -> int:
    from sensor_rig.record.recorder import EpisodeAborted, RecorderError

    if args.duration is not None and (args.duration <= 0 or args.count < 1):
        print("--duration must be > 0 and --count >= 1", file=sys.stderr)
        return 2
    from sensor_rig.rig_config import load_rig_arg

    load_rig_arg(args.rig)          # missing / invalid rig file -> RigFileError -> exit 2 (before any sensor)
    from sensor_rig.naming import session_name_problems
    from sensor_rig.timefmt import sanitize_session

    try:
        probs = session_name_problems(sanitize_session(args.session))
    except ValueError as e:
        probs = [str(e)]
    if probs:
        print("record: " + "; ".join(probs), file=sys.stderr)
        return 2
    try:
        rec = _make_recorder(args)
        rec.open()
    except (RecorderError, OSError, ValueError) as e:
        print(f"cannot start recorder: {e}", file=sys.stderr)
        return 1
    failed = 0
    try:
        if args.duration is not None:
            for i in range(args.count):
                if i:
                    rec.clock.sleep(args.pause)
                try:
                    rec.start_episode(**_episode_kwargs(args))
                except RecorderError as e:
                    print(f"cannot start episode: {e}", file=sys.stderr)
                    return 1
                t_end = rec.clock.now_ns() + int(args.duration * 1e9)
                while rec.clock.now_ns() < t_end and rec.recording:
                    rec.clock.sleep(0.05)
                try:
                    rec.stop_episode()
                except EpisodeAborted as e:
                    print(str(e), file=sys.stderr)
                    failed += 1
        else:
            failed = _interactive(rec, args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        failed += 1
    finally:
        rec.close()
    return 1 if failed else 0


def _interactive(rec, args) -> int:
    from sensor_rig.record.recorder import EpisodeAborted, RecorderError

    failed = 0
    print("Enter = start/stop episode, a = abort, q = quit")
    while True:
        try:
            line = input().strip().lower()
        except EOFError:
            line = "q"
        if line == "q":
            if rec.recording:
                rec.abort_episode("quit while recording")
                failed += 1
            return failed
        if line == "a":
            if rec.recording:
                rec.abort_episode("aborted by user")
                failed += 1
            continue
        if rec.recording:
            try:
                rec.stop_episode()
            except EpisodeAborted as e:
                print(str(e))
                failed += 1
        else:
            try:
                name = rec.start_episode(**_episode_kwargs(args))
                print(f"recording {name} (Enter to stop)")
            except RecorderError as e:
                print(f"cannot start episode: {e}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sensor_rig.record.cli")
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
