# adapted from rnoro/deployment:deployment/deploy/deploy/check_helpers.py @ 91b178b
# changes: parse_endpoint lives here (the new station builds endpoints from ports); adds json_check_lines (runs a
# --json self-check subprocess and turns its lines into rows, from old policy/launch.py _self_check),
# rig_file_check (runs `python -m policy_deploy.frames.rig_file FILE --json`, from old process_frames/launch.py),
# address_is_local; hints name policy_deploy.
"""Small reusable checks for the launcher (STRUCTURE.md 9.2). Each returns one CheckResult (or a list of them)."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

from policy_deploy.launcher.check_result import STATUSES, CheckResult

_ENDPOINT_RE = re.compile(r"^tcp://([^:/\s]+):(\d+)$")

# Run by the interpreter under test: tries every module, prints one line per failure, exits 1 if any failed.
_IMPORT_SCRIPT = """
import importlib, sys
bad = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(f"{name} ({type(exc).__name__}: {exc})")
print("; ".join(bad))
sys.exit(1 if bad else 0)
"""


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """`tcp://host:port` -> (host, port). Raises ValueError for anything else."""
    match = _ENDPOINT_RE.match(endpoint) if isinstance(endpoint, str) else None
    if match is None or not 0 < int(match.group(2)) < 65536:
        raise ValueError(f'expected "tcp://host:port", got {endpoint!r}')
    return match.group(1), int(match.group(2))


def file_exists(process: str, label: str, path: Path) -> CheckResult:
    path = Path(path)
    if path.exists():
        return CheckResult(process, label, "pass", f"found: {path}")
    return CheckResult(process, label, "fail", f"not found: {path}")


def executable(process: str, label: str, path: Path) -> CheckResult:
    path = Path(path)
    if not path.exists():
        return CheckResult(process, label, "fail", f"not found: {path}")
    if not path.is_file() or not os.access(path, os.X_OK):
        return CheckResult(process, label, "fail", f"not an executable file: {path}")
    return CheckResult(process, label, "pass", f"found: {path}")


def port_free(process: str, endpoint: str, label: str | None = None) -> CheckResult:
    """The endpoint's port can be bound now (nothing else is listening on it)."""
    name = label or f"port {endpoint}"
    try:
        host, port = parse_endpoint(endpoint)
    except ValueError as exc:
        return CheckResult(process, name, "fail", str(exc))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # a port in TIME_WAIT after a run is free
        sock.bind(("" if host == "*" else host, port))
    except OSError as exc:
        return CheckResult(process, name, "fail",
                           f"cannot bind {endpoint} ({exc.strerror or exc}): is another policy_deploy run, or this "
                           f"process started alone, still running?")
    finally:
        sock.close()
    return CheckResult(process, name, "pass", f"{endpoint} is free")


def python_imports(process: str, python: str, modules: Sequence[str], timeout_s: float) -> CheckResult:
    """`python` (run as a subprocess, it may be another environment) can import every module."""
    try:
        # cwd "/": `python -c` puts its working folder first on the path; a folder named like a package there
        # would be imported as an empty namespace package instead of the installed one.
        done = subprocess.run([python, "-c", _IMPORT_SCRIPT, *modules], capture_output=True, text=True,
                              timeout=timeout_s, stdin=subprocess.DEVNULL, cwd="/")
    except FileNotFoundError:
        return CheckResult(process, "imports", "fail", f"python not found: {python}")
    except PermissionError:
        return CheckResult(process, "imports", "fail", f"python not executable: {python}")
    except subprocess.TimeoutExpired:
        return CheckResult(process, "imports", "fail", f"{python}: importing {', '.join(modules)} took over "
                                                         f"{timeout_s:g} s")
    if done.returncode != 0:
        why = done.stdout.strip() or (done.stderr.strip().splitlines() or ["(no output)"])[-1]
        return CheckResult(process, "imports", "fail", f"{python} cannot import: {why}")
    return CheckResult(process, "imports", "pass", f"{python} imports {', '.join(modules)}")


def host_reachable(process: str, host: str, port: int, timeout_s: float) -> CheckResult:
    """A TCP connection to host:port opens within timeout_s (nothing is sent)."""
    name = f"reach {host}:{port}"
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as exc:
        return CheckResult(process, name, "fail",
                           f"no TCP connection to {host}:{port} within {timeout_s:g} s ({exc.strerror or exc})")
    return CheckResult(process, name, "pass", f"TCP connection to {host}:{port} opened")


def address_is_local(address: str) -> bool:
    """True if this machine has the IPv4 address on one of its interfaces (it can bind it)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((address, 0))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _tail(stderr: str) -> str:
    return " | ".join(stderr.strip().splitlines()[-3:]) if stderr else ""


def json_check_lines(process: str, cmd: Sequence[str], timeout_s: float, cwd: Path | None = None,
                     label: str = "self_check", hint: str = "") -> list[CheckResult]:
    """Run a `--json` self-check subprocess (stdin closed) and turn each printed line {"name", "status", "detail"}
    into a row of `process`. A crash, a timeout, a non-zero exit, a line that is not a check or no line at all is one
    failing row named `label`."""
    what = " ".join(cmd[1:4]) if len(cmd) > 1 else str(cmd[0])

    def failed(why: str, stderr: str) -> list[CheckResult]:
        detail = f"{what} {why}"
        if _tail(stderr):
            detail += f": {_tail(stderr)}"
        if hint:
            detail += f" ({hint})"
        return [CheckResult(process, label, "fail", detail)]

    try:
        proc = subprocess.run(list(cmd), cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return failed(f"timed out after {timeout_s:g} s", _text(e.stderr))
    except OSError as e:
        return failed(f"could not run {cmd[0]}: {e}", "")
    if proc.returncode != 0:
        return failed(f"exited with code {proc.returncode}", proc.stderr)
    results = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            name, status, detail = item["name"], item["status"], item["detail"]
            if status not in STATUSES or not isinstance(name, str) or not isinstance(detail, str):
                raise ValueError(line)
        except (ValueError, KeyError, TypeError):
            return failed(f"printed a line that is not a check: {line[:200]!r}", proc.stderr)
        results.append(CheckResult(process, name, status, detail))
    if not results:
        return failed("printed no checks", proc.stderr)
    return results


def rig_file_check(process: str, python: str, rig: Path, timeout_s: float, cwd: Path | None = None) -> CheckResult:
    """`python -m policy_deploy.frames.rig_file <rig> --json` prints one JSON line {"ok": true, "rig_id", "cameras"}
    or {"ok": false, "error"}; one row "rig file"."""
    exists = file_exists(process, "rig file", rig)
    if exists.status == "fail":
        return exists
    module = "policy_deploy.frames.rig_file"
    try:
        done = subprocess.run([python, "-m", module, str(rig), "--json"], capture_output=True, text=True,
                              timeout=timeout_s, stdin=subprocess.DEVNULL, cwd=cwd)
        lines = done.stdout.strip().splitlines()
        result = json.loads(lines[-1]) if lines else None
    except subprocess.TimeoutExpired:
        return CheckResult(process, "rig file", "fail", f"python -m {module} took over {timeout_s:g} s")
    except (OSError, json.JSONDecodeError) as e:
        return CheckResult(process, "rig file", "fail", f"cannot run python -m {module}: {e}")
    if not isinstance(result, dict):
        why = (done.stderr.strip().splitlines() or ["no output"])[-1]
        return CheckResult(process, "rig file", "fail", f"python -m {module} failed: {why}")
    if not result.get("ok"):
        return CheckResult(process, "rig file", "fail", str(result.get("error")))
    try:
        cams = ", ".join(f"{role} {c['sensor']} {c['width']}x{c['height']}@{c['fps']:g}"
                         for role, c in result["cameras"].items())
        return CheckResult(process, "rig file", "pass", f"{result['rig_id']}: {cams}")
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return CheckResult(process, "rig file", "fail", f"python -m {module} printed an unexpected summary: {e}")


def info(process: str, name: str, detail: str) -> CheckResult:
    return CheckResult(process, name, "info", detail)


def warn(process: str, name: str, detail: str) -> CheckResult:
    return CheckResult(process, name, "warn", detail)
