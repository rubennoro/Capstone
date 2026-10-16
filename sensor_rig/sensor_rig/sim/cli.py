"""CLI: ``sim`` (simulated raw rig episodes from real iPhone exports, D10a)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers) -> None:
    p = subparsers.add_parser("sim", help="simulate raw rig episodes (+ VIO trajectory, IMU) from iPhone exports")
    p.add_argument("--iphone", required=True, nargs="+",
                   help="iPhone episode folder(s), or folder(s) containing episode folders")
    p.add_argument("--out", required=True, help="raw root (episodes go to <out>/<YYYY-MM-DD>/<NAME>/)")
    p.add_argument("--rig", default=None, help="rig yaml (default: sensor_rig/sim/sim_rig.yaml)")
    p.add_argument("--filter", default=None, help="only episodes whose folder name contains this substring")
    p.add_argument("--main-fps", type=float, default=None, help="override the rig's main fps (60 or 30)")
    p.add_argument("--uw-mode", choices=["native", "30fps_hold"], default="native")
    p.add_argument("--traj-rate", choices=["main", "uw"], default="main")
    p.add_argument("--traj-frame", choices=["imu", "imu0", "main", "ultrawide"], default=None)
    p.add_argument("--vio-init-delay", type=float, default=0.0, help="seconds without poses at the start")
    p.add_argument("--vio-gap", type=float, nargs=2, default=None, metavar=("START_S", "DUR_S"))
    p.add_argument("--ts-jitter-ns", type=int, default=0)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--imu-rate", type=float, default=400.0)
    p.add_argument("--imu-noise", choices=["nominal", "none"], default="nominal")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--session", default=None, help="replace the session name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-import", action="store_true",
                   help="only write <ep>/sim/vio_trajectory.tum (run pose-import yourself)")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_cmd_sim)


def _one(args_tuple):
    src, rig_path, out, opts = args_tuple
    from sensor_rig.sim.simulator import simulate_raw_from_iphone

    return str(simulate_raw_from_iphone(src, rig_path, out, opts))


def _cmd_sim(args: argparse.Namespace) -> int:
    from sensor_rig.sim.simulator import SIM_RIG_PATH, SimOptions, find_iphone_episodes, kind_order

    srcs = sorted(find_iphone_episodes(args.iphone), key=kind_order)
    if args.filter:
        srcs = [s for s in srcs if args.filter in s.name]
    if not srcs:
        print(f"no iPhone episodes found under {args.iphone}", file=sys.stderr)
        return 1
    opts = SimOptions(
        main_fps=args.main_fps, uw_mode=args.uw_mode, traj_rate=args.traj_rate, traj_frame=args.traj_frame,
        vio_init_delay_s=args.vio_init_delay, vio_gap=tuple(args.vio_gap) if args.vio_gap else None,
        ts_jitter_ns=args.ts_jitter_ns, drop_rate=args.drop_rate, imu_rate_hz=args.imu_rate,
        imu_noise=args.imu_noise, max_frames=args.max_frames, session=args.session, seed=args.seed,
        import_trajectory=not args.no_import,
    )
    rig_path = Path(args.rig) if args.rig else SIM_RIG_PATH
    jobs = [(s, rig_path, Path(args.out), opts) for s in srcs]
    failed = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.workers) as ex:
            futs = [(j[0], ex.submit(_one, j)) for j in jobs]
            results = []
            for s, f in futs:
                try:
                    results.append((s, f.result(), None))
                except Exception as e:  # noqa: BLE001 - report and continue
                    results.append((s, None, e))
    else:
        results = []
        for j in jobs:
            try:
                results.append((j[0], _one(j), None))
            except Exception as e:  # noqa: BLE001
                results.append((j[0], None, e))
    for s, out, err in results:
        if err is None:
            print(f"simulated {s.name} -> {out}")
        else:
            failed += 1
            print(f"FAILED    {s.name}: {type(err).__name__}: {err}", file=sys.stderr)
    return 1 if failed else 0
