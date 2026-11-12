"""Small reusable checks for launch.py files (STRUCTURE.md section 4.3). Each returns one CheckResult."""
from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

from deploy.check_result import CheckResult
from deploy.station import parse_endpoint

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


def port_free(process: str, endpoint: str) -> CheckResult:
    """The endpoint's port can be bound now (nothing else is listening on it)."""
    name = f"port {endpoint}"
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
                           f"cannot bind {endpoint} ({exc.strerror or exc}): is another deploy run, or this process "
                           f"started alone, still running?")
    finally:
        sock.close()
    return CheckResult(process, name, "pass", f"{endpoint} is free")


def python_imports(process: str, python: str, modules: Sequence[str], timeout_s: float) -> CheckResult:
    """`python` (run as a subprocess, it may be another environment) can import every module."""
    try:
        # cwd "/": `python -c` puts its working folder first on the path, and from deployment/ the folder
        # deployment/contracts/ would be imported as an empty namespace package instead of the installed package.
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


def info(process: str, name: str, detail: str) -> CheckResult:
    return CheckResult(process, name, "info", detail)


def warn(process: str, name: str, detail: str) -> CheckResult:
    return CheckResult(process, name, "warn", detail)
