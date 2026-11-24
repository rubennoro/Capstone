# from rnoro/deployment:deployment/deploy/deploy/check_helpers.py @ 91b178b
# changes: minus python_imports (control_deploy/STRUCTURE.md 2); parse_endpoint moved here (it was deploy.station's);
# port_free's hint names control_deploy run.
"""Small reusable checks for `control_deploy check` (control_deploy/STRUCTURE.md 6.3). Each returns one CheckResult."""
from __future__ import annotations

import os
import socket
from pathlib import Path

from control_deploy.check_result import CheckResult


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """("host", port) from "tcp://host:port"; ValueError for anything else."""
    if not endpoint.startswith("tcp://") or ":" not in endpoint[6:]:
        raise ValueError(f"expected tcp://host:port, got {endpoint!r}")
    host, _, port = endpoint[6:].rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"expected tcp://host:port, got {endpoint!r}")
    return host, int(port)


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
                           f"{endpoint} in use ({exc.strerror or exc}): is control_deploy run already running? "
                           f"(ss -ltnp | grep 56)")
    finally:
        sock.close()
    return CheckResult(process, name, "pass", f"{endpoint} is free")


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
