"""``python -m sensor_rig <command>``: one argparse CLI; each command group registers itself lazily.

A group whose module fails to import gets a stub command that explains the missing dependency, so the other commands
keep working; a command that hits a missing module at run time (deps are imported lazily) prints the apt package to
install instead of a traceback. Exit codes: 0 success, 1 failed/aborted episode, 2 usage error.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# (module, description). Every module exposes register(subparsers).
GROUPS = [
    ("sensor_rig.doctor", "doctor"),
    ("sensor_rig.record.collect", "collect"),
    ("sensor_rig.record.cli", "record, bench-encode"),
    ("sensor_rig.pose.cli_pose", "pose-prep, pose-import"),
    ("sensor_rig.export.cli_export", "export, validate"),
    ("sensor_rig.process.session", "process-session"),
    ("sensor_rig.sim.cli", "sim"),
    ("sensor_rig.calib.stamp_offset", "stamp-offset"),
    ("sensor_rig.calib.cli", "calib-import, calib-check, calib-align, calib-tcp, rig-validate, bench-remap"),
]


def build_parser() -> tuple[argparse.ArgumentParser, list[str]]:
    parser = argparse.ArgumentParser(
        prog="python -m sensor_rig",
        description="Jetson sensor rig: doctor, collect, record, pose (VIO adapters), export to the iPhUMI iPhone format, sim, calib.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    unavailable = []
    for module, cmds in GROUPS:
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            unavailable.append(f"{cmds} ({module}: {e})")
            for cmd in cmds.split(", "):
                _register_unavailable(sub, cmd, module, e)
            continue
        mod.register(sub)
    if unavailable:
        parser.epilog = "unavailable commands: " + "; ".join(unavailable)
    return parser, unavailable


# apt package that provides a missing module on JetPack (sensor_rig/scripts/jetson_setup.sh installs all of them)
_FIX = {"scipy": "python3-scipy", "yaml": "python3-yaml", "numpy": "python3-numpy", "cv2": "python3-opencv",
        "av": "PyAV (laptop only: pip install av)", "spidev": "python3-spidev", "gi": "python3-gi"}


def missing_module_message(cmd: str, err: ImportError) -> str:
    missing = (getattr(err, "name", None) or "").split(".")[0]
    hint = _FIX.get(missing)
    msg = f"sensor-rig {cmd}: missing Python module {missing or err!s}."
    if hint:
        msg += f" Install {hint}" + (" (sudo apt install ...)" if hint.startswith("python3-") else "") + "."
    return msg + " On the Jetson: rerun sensor_rig/scripts/jetson_setup.sh, then `sensor-rig doctor`."


def _register_unavailable(sub, cmd: str, module: str, err: ImportError) -> None:
    """A stub so ``sensor-rig <cmd>`` explains the missing dependency instead of 'invalid choice'."""
    msg = missing_module_message(cmd, err) + f" ({module}: {err})"
    p = sub.add_parser(cmd, help=f"(unavailable: {err})", add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=lambda args, _m=msg: (print(_m, file=sys.stderr), 1)[1])


def main(argv: list[str] | None = None) -> int:
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    from sensor_rig.rig_config import RigFileError

    try:
        rc = args.func(args)
    except RigFileError as e:            # a missing / invalid --rig file: one message, exit 2 (every subcommand)
        print(f"sensor-rig {args.command}: {e}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as e:   # optional/hardware deps are imported lazily by the commands
        print(missing_module_message(args.command, e), file=sys.stderr)
        return 1
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
