"""franka_deploy command line.

    python -m franka_deploy run --config deploy.yaml [--mock] [--script "t:key,..."]
        [--replay-episode E --replay-speed 0.5]     (open-loop demo replay instead of the policy)
    python -m franka_deploy check-config --config deploy.yaml [--profile jetson|mock]
    python -m franka_deploy note --session <session_dir> "text"      # non-blocking operator note
    python -m franka_deploy validate <episode_dir | session_dir> [--mode auto|real|sim]
    python -m franka_deploy annotate <episode_dir> [--success yes|no --category CAT ...]
    python -m franka_deploy eval <sessions...> --out DIR                 (env_bp: zarr, normalizer)
    python -m franka_deploy reinfer <session_dir> --work-dir DIR          (env_bp)
    python -m franka_deploy offline-eval --ckpt C --work-dir W --dataset-path Z [--per-demo]   (env_bp)
    python -m franka_deploy fdcal handeye|pivot|width ...                 (calibration tools)
    python -m franka_deploy gonogo run|evaluate ...                       (Jetson RT go/no-go load test)
    python -m franka_deploy policy-contract --endpoint E [--expect run/policy/<id>.yaml] [--out contract.json]
    python -m franka_deploy replay-server --zarr Z --episode NAME ...     (single-arm action replay)
    python -m franka_deploy wireless-latency measure|analyze|show|synth ...   (QR-clock latency of the wireless camera)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--mock", action="store_true", help="profile 'mock': FakeBackend + FakeGripper + camera mock + real inference")
    r.add_argument("--profile", default=None)
    r.add_argument("--set", action="append", default=[], metavar="GROUP.PATH=VALUE",
                   help="station override (repeatable), e.g. host.work_root=/scratch/runs")
    r.add_argument("--script", default=None, help="timed keys/injections for tests, e.g. '3:c,20:y'")
    r.add_argument("--no-tty", action="store_true")
    r.add_argument("--replay-episode", default=None,
                   help="open-loop action replay: serve this prompt-zarr episode (name or index) instead of the policy")
    r.add_argument("--replay-speed", type=float, default=1.0, help="replay speed (0.5 or 1.0)")
    c = sub.add_parser("check-config")
    c.add_argument("--config", required=True)
    c.add_argument("--profile", default="jetson")
    c.add_argument("--set", action="append", default=[], metavar="GROUP.PATH=VALUE", help="station override (repeatable)")
    n = sub.add_parser("note")
    n.add_argument("--session", required=True)
    n.add_argument("text")
    v = sub.add_parser("validate")
    v.add_argument("path")
    v.add_argument("--mode", choices=("auto", "real", "sim"), default="auto",
                   help="auto: run_mode of the episode's config/deploy_resolved.yaml (sim accepts no video / no checkpoint)")
    passthrough = {"annotate": "franka_deploy.eval.annotate", "eval": "franka_deploy.eval.report",
                   "reinfer": "franka_deploy.tools.reinfer", "offline-eval": "franka_deploy.tools.offline_eval",
                   "fdcal": "franka_deploy.fdcal.__main__", "gonogo": "franka_deploy.tools.gonogo",
                   "replay-server": "franka_deploy.tools.replay_server",
                   "policy-contract": "franka_deploy.tools.policy_contract",
                   "wireless-latency": "franka_deploy.tools.wireless_latency"}
    for name in passthrough:
        sub.add_parser(name, add_help=False)
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in passthrough:
        mod = passthrough[argv[0]]
        import importlib

        return importlib.import_module(mod).main(argv[1:]) or 0
    a = ap.parse_args(argv)
    if a.cmd in ("run", "check-config"):
        from franka_deploy.launch.launcher import ConfigUsageError

        try:
            return _dispatch(a)
        except ConfigUsageError as e:
            print(f"usage error: {e}", file=sys.stderr)
            return 2
    return _dispatch(a)


def _dispatch(a) -> int:
    if a.cmd == "run":
        from franka_deploy.launch.launcher import Launcher, load_config

        cfg = load_config(a.config, a.profile or ("mock" if a.mock else "jetson"), station_set=a.set)
        if a.replay_episode is not None:
            cfg["inference"]["replay"] = {"episode": a.replay_episode, "speed": a.replay_speed}
        return Launcher(cfg, interactive=False if a.no_tty else None, script=a.script).run()
    if a.cmd == "check-config":
        from franka_deploy.launch.launcher import check_config, load_config

        return check_config(load_config(a.config, a.profile, station_set=a.set))
    if a.cmd == "note":
        with open(Path(a.session) / "commands.txt", "a") as f:
            f.write("note:" + a.text.replace("\n", " ") + "\n")
        return 0
    if a.cmd == "validate":
        from franka_deploy.tools.validate_episode import main as vmain

        return vmain([a.path, "--mode", a.mode])
    return 1


if __name__ == "__main__":
    sys.exit(main())
